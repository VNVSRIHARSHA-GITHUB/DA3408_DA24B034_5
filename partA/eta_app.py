"""
eta_app.py -- SwiftParcel ETA service (Module 4 assignment, Part A).

Run (from the project root, where data/ lives):
    ETA_MODEL=data/eta_v1.joblib uvicorn eta_app:app --app-dir partA --port 8000

Endpoints
    POST /v1/models/eta:predict   {"instances": [{7 features}, ...]}  -> predicted minutes
    GET  /healthz                 liveness / readiness probe
    GET  /metrics                 Prometheus text format (requests to /metrics are not counted)

Ideas reused from the Lecture 4a notebook (rest_app.py): loading the bundle once at import time,
pydantic Field ranges as the input contract, a plain `def` endpoint so sklearn runs in the thread pool.
"""
import os
import threading
import time
import uuid
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_PATH = os.getenv("ETA_MODEL", "data/eta_v1.joblib")
bundle = joblib.load(MODEL_PATH)  # loaded once per process, never per request
model = bundle["model"]
FEATURES = bundle["feature_columns"]
VERSION = bundle["version"]

PREDICT_PATH = "/v1/models/eta:predict"


# ---------------------------------------------------------------- input contract
class Delivery(BaseModel):
    model_config = ConfigDict(extra="ignore")  # unknown fields are dropped, like rest_app.py in Lecture 4a

    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 together with n_stops > 20")
        return self


class PredictRequest(BaseModel):
    instances: List[Delivery] = Field(min_length=1, max_length=500)


# ---------------------------------------------------------------- metrics (Prometheus text format)
BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
_lock = threading.Lock()
_requests = {}                      # (path, status) -> count
_bucket_counts = [0] * len(BUCKETS)  # cumulative counts are built when /metrics is read
_lat_sum = 0.0
_lat_count = 0


def observe(path: str, status: int, seconds: float) -> None:
    global _lat_sum, _lat_count
    with _lock:
        _requests[(path, status)] = _requests.get((path, status), 0) + 1
        for i, upper in enumerate(BUCKETS):
            if seconds <= upper:
                _bucket_counts[i] += 1
                break
        _lat_sum += seconds
        _lat_count += 1


def metrics_text() -> str:
    with _lock:
        lines = ["# HELP eta_requests_total Requests handled, by route and HTTP status.",
                 "# TYPE eta_requests_total counter"]
        for (path, status), n in sorted(_requests.items()):
            lines.append(f'eta_requests_total{{path="{path}",status="{status}"}} {n}')
        lines += ["# HELP eta_request_latency_seconds Time from request received to response ready.",
                  "# TYPE eta_request_latency_seconds histogram"]
        cumulative = 0
        for upper, n in zip(BUCKETS, _bucket_counts):
            cumulative += n
            lines.append(f'eta_request_latency_seconds_bucket{{le="{upper}"}} {cumulative}')
        lines.append(f'eta_request_latency_seconds_bucket{{le="+Inf"}} {_lat_count}')
        lines.append(f"eta_request_latency_seconds_sum {_lat_sum:.6f}")
        lines.append(f"eta_request_latency_seconds_count {_lat_count}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- app
app = FastAPI(title="SwiftParcel ETA service")


def error_500(request_id: str) -> JSONResponse:
    # generic on purpose: no exception text, no stack trace, no file paths
    return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": request_id})


@app.middleware("http")
async def request_id_and_metrics(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:                      # anything not handled below still becomes a generic 500
        response = error_500(request_id)
    response.headers["X-Request-ID"] = request_id
    if request.url.path != "/metrics":
        route = request.scope.get("route")  # route template keeps label values bounded
        path = route.path if route is not None else "unmatched"
        observe(path, response.status_code, time.perf_counter() - t0)
    return response


@app.exception_handler(RequestValidationError)
async def validation_failed(request: Request, exc: RequestValidationError):
    details = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", ()) if p != "body"]
        details.append({"field": ".".join(loc) or "body", "msg": err.get("msg", "invalid")})
    return JSONResponse(status_code=422, content={"error": "validation_failed", "details": details,
                                                  "request_id": request.state.request_id})


@app.post(PREDICT_PATH)
def predict(req: PredictRequest, request: Request):
    request_id = request.state.request_id
    try:
        X = pd.DataFrame([d.model_dump() for d in req.instances])[FEATURES]
        minutes = model.predict(X)
        predictions = [round(float(m), 1) for m in minutes]
    except Exception:
        return error_500(request_id)
    return {"predictions": predictions, "model_version": VERSION, "request_id": request_id}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": VERSION}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(metrics_text(), media_type="text/plain; version=0.0.4")
