"""
test_eta_app.py -- pytest suite for eta_app.py (Part A2).
Run from the project root (where data/ lives):   pytest partA -v
"""
import os
import re
import uuid

import pytest

if "ETA_MODEL" not in os.environ and not os.path.exists("data/eta_v1.joblib"):
    os.environ["ETA_MODEL"] = os.path.join(os.path.dirname(__file__), "..", "data", "eta_v1.joblib")

from fastapi.testclient import TestClient  # noqa: E402

import eta_app  # noqa: E402

URL = "/v1/models/eta:predict"
GOOD = {"distance_km": 12.5, "n_stops": 4, "hour": 10, "weekday": 2,
        "traffic_index": 0.3, "rain": 0, "weight_kg": 5.0}


@pytest.fixture
def client():
    # raise_server_exceptions=False: we want to see the HTTP response a real client would get
    return TestClient(eta_app.app, raise_server_exceptions=False)


def row(**changes):
    return {**GOOD, **changes}


# ---------------------------------------------------------------- valid requests
def test_valid_request(client):
    r = client.post(URL, json={"instances": [GOOD]})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"predictions", "model_version", "request_id"}
    assert body["model_version"] == "v1"
    assert len(body["predictions"]) == 1
    p = body["predictions"][0]
    assert isinstance(p, float) and p == round(p, 1) and p > 0


def test_batch_order_preserved(client):
    rows = [row(distance_km=d) for d in (2.0, 80.0, 30.0, 150.0, 0.5)]
    forward = client.post(URL, json={"instances": rows}).json()["predictions"]
    backward = client.post(URL, json={"instances": rows[::-1]}).json()["predictions"]
    assert backward == forward[::-1]
    # and each value matches the model on that row alone
    singles = [client.post(URL, json={"instances": [x]}).json()["predictions"][0] for x in rows]
    assert forward == singles


# ---------------------------------------------------------------- invalid input
@pytest.mark.parametrize("bad, field", [
    (row(distance_km=0.0), "distance_km"),           # below range
    (row(hour=24), "hour"),                          # above range
    (row(rain=2), "rain"),                           # not 0/1
    (row(weight_kg=0), "weight_kg"),                 # must be > 0
    (row(n_stops="three"), "n_stops"),               # wrong type
    ({k: v for k, v in GOOD.items() if k != "weekday"}, "weekday"),  # missing field
])
def test_invalid_rows_rejected(client, bad, field):
    r = client.post(URL, json={"instances": [bad]})
    assert r.status_code == 422
    fields = [d["field"] for d in r.json()["details"]]
    assert any(f.endswith(field) for f in fields), fields


def test_malformed_json_rejected(client):
    r = client.post(URL, content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert r.json()["error"] == "validation_failed"


def test_validation_error_body_format(client):
    r = client.post(URL, json={"instances": [row(traffic_index=1.5)]})
    body = r.json()
    assert r.status_code == 422
    assert set(body) == {"error", "details", "request_id"}
    assert body["error"] == "validation_failed"
    assert body["details"] and all(set(d) == {"field", "msg"} for d in body["details"])


# ---------------------------------------------------------------- cross-field rule
def test_infeasible_rejected(client):
    r = client.post(URL, json={"instances": [row(distance_km=150.1, n_stops=21)]})
    assert r.status_code == 422
    assert "infeasible" in r.text


def test_infeasible_boundary_accepted(client):
    # exactly 150 km or exactly 20 stops is still feasible ("> 150 and > 20" is the rule)
    ok = [row(distance_km=150.0, n_stops=21), row(distance_km=150.1, n_stops=20)]
    assert client.post(URL, json={"instances": ok}).status_code == 200


# ---------------------------------------------------------------- batch-size limits
@pytest.mark.parametrize("n, status", [(0, 422), (1, 200), (500, 200), (501, 422)])
def test_batch_size_limits(client, n, status):
    assert client.post(URL, json={"instances": [GOOD] * n}).status_code == status


# ---------------------------------------------------------------- request id
def test_request_id_echoed(client):
    r = client.post(URL, json={"instances": [GOOD]}, headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    assert r.json()["request_id"] == "abc-123"


def test_request_id_generated(client):
    r = client.post(URL, json={"instances": [GOOD]})
    rid = r.headers["X-Request-ID"]
    assert str(uuid.UUID(rid)) == rid
    assert r.json()["request_id"] == rid
    assert client.post(URL, json={"instances": [GOOD]}).headers["X-Request-ID"] != rid


def test_request_id_on_error(client):
    echoed = client.post(URL, json={"instances": []}, headers={"X-Request-ID": "err-1"})
    assert echoed.status_code == 422
    assert echoed.headers["X-Request-ID"] == "err-1" and echoed.json()["request_id"] == "err-1"
    generated = client.post(URL, json={"instances": []})
    assert generated.headers["X-Request-ID"] == generated.json()["request_id"]


# ---------------------------------------------------------------- health + metrics
def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "v1"}


def counter(text, path, status):
    m = re.search(rf'^eta_requests_total{{path="{re.escape(path)}",status="{status}"}} (\d+)$', text, re.M)
    return int(m.group(1)) if m else 0


def test_metrics_format_and_counting(client):
    before = client.get("/metrics").text
    client.post(URL, json={"instances": [GOOD]})
    client.post(URL, json={"instances": []})
    text = client.get("/metrics").text
    assert counter(text, URL, 200) == counter(before, URL, 200) + 1
    assert counter(text, URL, 422) == counter(before, URL, 422) + 1
    assert "# TYPE eta_request_latency_seconds histogram" in text
    buckets = re.findall(r'^eta_request_latency_seconds_bucket{le="([^"]+)"} (\d+)$', text, re.M)
    assert len(buckets) >= 5 and buckets[-1][0] == "+Inf"
    counts = [int(c) for _, c in buckets]
    assert counts == sorted(counts)                       # cumulative
    total = int(re.search(r"^eta_request_latency_seconds_count (\d+)$", text, re.M).group(1))
    assert counts[-1] == total
    assert re.search(r"^eta_request_latency_seconds_sum [0-9.]+$", text, re.M)


def test_metrics_not_counted(client):
    client.get("/metrics")
    client.get("/metrics")
    assert 'path="/metrics"' not in client.get("/metrics").text


# ---------------------------------------------------------------- generic 500
def test_generic_500(client, monkeypatch):
    class Broken:
        def predict(self, X):
            raise RuntimeError("secret internal detail /home/model.joblib")

    monkeypatch.setattr(eta_app, "model", Broken())
    r = client.post(URL, json={"instances": [GOOD]}, headers={"X-Request-ID": "boom"})
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error", "request_id": "boom"}
    assert "secret" not in r.text and "Traceback" not in r.text
    assert r.headers["X-Request-ID"] == "boom"


def test_unknown_field_ignored(client):
    # same behaviour as the lecture's rest_app.py: an extra column is dropped, not an error
    r = client.post(URL, json={"instances": [row(eta_minutes=30.0)]})
    assert r.status_code == 200
    assert r.json()["predictions"] == client.post(URL, json={"instances": [GOOD]}).json()["predictions"]
