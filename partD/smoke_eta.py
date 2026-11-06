"""
smoke_eta.py -- Part D1: release gate run against a live ETA server before it takes real traffic.

    python partD/smoke_eta.py --url http://localhost:8000 --expect-version v1 [--max-p95-ms 80]

--url may be the server root (http://host:port) or the full predict URL (.../v1/models/eta:predict).
Works against the Part A FastAPI service and the Part C Ray Serve router (both use the same route).
Exit code 0 = every check passed, 1 = at least one failed (so CI / a deploy script can block on it).
"""
import argparse
import math
import sys
import time

import requests

PREDICT = "/v1/models/eta:predict"
BASE_ROW = {"distance_km": 20.0, "n_stops": 5, "hour": 11, "weekday": 2,
            "traffic_index": 0.4, "rain": 0, "weight_kg": 8.0}


def row(**changes):
    return {**BASE_ROW, **changes}


class Smoke:
    def __init__(self, url, expect_version, max_p95_ms, timeout=5):
        url = url.rstrip("/")
        self.predict_url = url if url.endswith(":predict") else url + PREDICT
        self.base = self.predict_url[: -len(PREDICT)]
        self.expect_version, self.max_p95_ms, self.timeout = expect_version, max_p95_ms, timeout
        self.s = requests.Session()
        self.results = []

    def post(self, instances, **kw):
        return self.s.post(self.predict_url, json={"instances": instances}, timeout=self.timeout, **kw)

    def preds(self, instances):
        r = self.post(instances)
        r.raise_for_status()
        return r.json()["predictions"]

    def check(self, name, fn):
        try:
            ok, info = fn()
        except Exception as e:  # a crash or timeout inside a check is a failure, not a smoke-test bug
            ok, info = False, f"{type(e).__name__}: {e}"[:160]
        self.results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"       {info}")

    # ------------------------------------------------------------ the checks
    def health(self):
        r = self.s.get(self.base + "/healthz", timeout=self.timeout)
        ok = r.status_code == 200 and r.json().get("status") == "ok"
        return ok, f"GET /healthz -> HTTP {r.status_code}, body {r.text[:80]}"

    def version(self):
        got = self.post([BASE_ROW]).json().get("model_version")
        return got == self.expect_version, f"expected model {self.expect_version}, server answered with {got}"

    def valid_predictions(self):
        rows = [row(distance_km=d) for d in (1, 15, 60)]
        p = self.preds(rows)
        ok = len(p) == 3 and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 < x < 2000 for x in p)
        mins = ", ".join(f"{d} km -> {x} min" for d, x in zip((1, 15, 60), p))
        return ok, f"{mins}  (each must be a number between 0 and 2000)"

    def rejects_invalid(self):
        bad = {"negative distance": [row(distance_km=-5)], "hour 25": [row(hour=25)],
               "infeasible": [row(distance_km=200, n_stops=30)], "empty batch": []}
        codes = {k: self.post(v).status_code for k, v in bad.items()}
        return all(c == 422 for c in codes.values()), ", ".join(f"{k}: {c}" for k, c in codes.items())

    def monotone_distance(self):
        ds = [1, 5, 10, 20, 40, 80, 120]
        p = self.preds([row(distance_km=d) for d in ds])
        bad = [(ds[i], ds[i + 1]) for i in range(len(p) - 1) if p[i + 1] < p[i]]
        trend = ", ".join(f"{d} km: {x}" for d, x in zip(ds, p))
        return not bad, f"ETA (min) by distance -> {trend}" + (f"  DROPS at {bad}" if bad else "")

    def monotone_traffic(self):
        ts = [0.0, 0.25, 0.5, 0.75, 1.0]
        p = self.preds([row(traffic_index=t) for t in ts])
        bad = [(ts[i], ts[i + 1]) for i in range(len(p) - 1) if p[i + 1] < p[i]]
        trend = ", ".join(f"{t}: {x}" for t, x in zip(ts, p))
        return not bad, f"ETA (min) by traffic index -> {trend}" + (f"  DROPS at {bad}" if bad else "")

    def deterministic_and_ordered(self):
        rows = [row(distance_km=d) for d in (90, 3, 45)]
        a, b = self.preds(rows), self.preds(rows[::-1])
        return a == b[::-1], f"in order {a}, reversed batch put back in order {b[::-1]}"

    def latency(self):
        for _ in range(5):
            self.post([BASE_ROW])
        lat = []
        for _ in range(50):
            t0 = time.perf_counter()
            self.post([BASE_ROW]).raise_for_status()
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]   # same percentile rule as load_test.py
        return p95 <= self.max_p95_ms, f"p95 = {p95:.1f} ms over 50 single-row calls (limit {self.max_p95_ms:.0f} ms)"

    def run(self):
        print(f"Smoke test for {self.predict_url}")
        print(f"Expected model: {self.expect_version}    latency limit: p95 <= {self.max_p95_ms:.0f} ms\n")
        self.check("1. Server is up (/healthz)", self.health)
        self.check("2. Server runs the expected model version", self.version)
        self.check("3. A valid batch gets sensible ETAs", self.valid_predictions)
        self.check("4. Bad input is rejected with HTTP 422", self.rejects_invalid)
        self.check("5. Sanity: a longer distance never gives a shorter ETA", self.monotone_distance)
        self.check("6. Sanity: more traffic never gives a shorter ETA", self.monotone_traffic)
        self.check("7. Same input gives the same output, in the same order", self.deterministic_and_ordered)
        self.check("8. Latency is within the limit", self.latency)
        passed = sum(self.results)
        go = passed == len(self.results)
        verdict = "GO" if go else "NO-GO"
        print(f"\nResult: {passed} of {len(self.results)} checks passed -> {verdict}")
        return 0 if passed == len(self.results) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    a = ap.parse_args()
    sys.exit(Smoke(a.url, a.expect_version, a.max_p95_ms).run())