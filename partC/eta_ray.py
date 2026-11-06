"""
eta_ray.py -- Part C: Ray Serve app with a STICKY canary (per customer, not per request).

    client --HTTP--> EtaRouter (FastAPI ingress)  --> EtaProd      x 2 replicas   (v1)
                                                  +-> EtaCandidate x 1..3 replicas (v2, autoscaled)

Routing for POST /v1/models/eta:predict  {"customer_id": "...", "instances": [...]}
  1. header X-Beta: 1                      -> candidate (testers, support staff)
  2. stable_bucket(customer_id) < weight   -> candidate      (same customer -> same arm, always)
  3. otherwise / no customer_id            -> prod

CPU budget (fits a 2-CPU Ray): router 0.1 + prod 2 x 0.25 + candidate up to 3 x 0.25 = 1.35 CPU.

Ideas reused from the Lecture 4c notebook (ray_serve_app.py): deployment class loading the bundle once per
replica, a FastAPI ingress router composed with DeploymentHandles, fractional num_cpus, extra_work_ms.
The canary there was random per request; here it is a deterministic hash per customer.
"""
import hashlib
import os
import sys
import time
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, ConfigDict, Field, model_validator
import ray.cloudpickle
from ray import serve
from ray.serve.handle import DeploymentHandle

# Ship this module's classes to Ray workers by value, so replicas start even when the worker
# processes cannot import `eta_ray` themselves (e.g. it is driven from another folder or a test).
if __name__ != "__main__":
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])

FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_CPUS = 0.25
ROUTER_CPUS = 0.1
PROD_REPLICAS = 2
CANDIDATE_AUTOSCALING = {
    "min_replicas": 1,
    "max_replicas": 3,
    "target_ongoing_requests": 2,   # each replica aims at ~2 requests in flight
    "upscale_delay_s": 5,           # add replicas quickly when the candidate gets busy
    "downscale_delay_s": 60,        # remove them slowly, so a short lull does not cause flapping
    "metrics_interval_s": 2,
    "look_back_period_s": 10,
}


def stable_bucket(customer_id: str) -> float:
    """Map an id to [0, 1) the same way in every process, on every machine, after every restart.
    SHA-256 is unsalted, unlike Python's built-in hash() for str (PYTHONHASHSEED randomises it)."""
    digest = hashlib.sha256(customer_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def choose_arm(customer_id: Optional[str], weight: float, beta: Optional[str]) -> str:
    if beta == "1":
        return "v2"
    if customer_id and stable_bucket(customer_id) < weight:
        return "v2"
    return "v1"


@serve.deployment(ray_actor_options={"num_cpus": MODEL_CPUS}, max_ongoing_requests=8,
                  health_check_period_s=5, logging_config={"log_level": "WARNING"})
class EtaModel:
    def __init__(self, path: str, extra_work_ms: float = 0.0):
        b = joblib.load(path)                       # once per replica, not per request
        self.model, self.cols, self.version = b["model"], b["feature_columns"], b["version"]
        self.extra = extra_work_ms / 1000.0         # simulates a heavier model; blocking on purpose
        self.replica = serve.get_replica_context().replica_id.unique_id

    async def __call__(self, rows: List[dict]) -> dict:
        X = pd.DataFrame(rows)[self.cols]
        minutes = [round(float(m), 1) for m in self.model.predict(X)]
        if self.extra:
            time.sleep(self.extra)
        return {"predictions": minutes, "model_version": self.version, "served_by": self.replica}

    async def check_health(self):
        if self.model is None:
            raise RuntimeError("model not loaded")


class Delivery(BaseModel):
    model_config = ConfigDict(extra="ignore")
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def feasible(self):            # same contract as the Part A service
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 together with n_stops > 20")
        return self


class RouteRequest(BaseModel):
    customer_id: Optional[str] = Field(default=None, min_length=1, max_length=200)
    instances: List[Delivery] = Field(min_length=1, max_length=500)


api = FastAPI(title="SwiftParcel ETA - sticky canary")


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": ROUTER_CPUS},
                  logging_config={"log_level": "WARNING"})
@serve.ingress(api)
class EtaRouter:
    def __init__(self, prod: DeploymentHandle, candidate: DeploymentHandle, weight: float):
        self.handles = {"v1": prod, "v2": candidate}
        self.weight = weight

    @api.get("/healthz")
    def healthz(self):
        return {"status": "ok", "canary_weight": self.weight}

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: RouteRequest, x_beta: Optional[str] = Header(default=None)):
        arm = choose_arm(req.customer_id, self.weight, x_beta)
        out = await self.handles[arm].remote([d.model_dump() for d in req.instances])
        return {**out, "arm": arm, "customer_id": req.customer_id}


def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    """The Serve application: router -> prod (2 replicas) + candidate (autoscaled 1..3)."""
    v1 = os.path.join(ROOT, os.getenv("ETA_V1", "data/eta_v1.joblib"))
    v2 = os.path.join(ROOT, os.getenv("ETA_V2", "data/eta_v2.joblib"))
    prod = EtaModel.options(name="EtaProd", num_replicas=PROD_REPLICAS).bind(v1, extra_work_ms)
    cand = EtaModel.options(name="EtaCandidate", num_replicas=None,
                            autoscaling_config=CANDIDATE_AUTOSCALING).bind(v2, extra_work_ms)
    return EtaRouter.bind(prod, cand, weight)


app = build(float(os.getenv("CANARY_WEIGHT", "0.1")))   # for `serve run partC.eta_ray:app`
