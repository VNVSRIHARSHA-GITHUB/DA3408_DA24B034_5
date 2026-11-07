"""
eta_mlflow.py -- Part B: MLflow pyfunc, registry aliases and a multi-criteria promotion gate.

Run from the project root (where eta_data.py and data/ live):
    export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    python partB/eta_mlflow.py register                 #  log + register v1, v2; aliases prod -> v1, candidate -> v2
    python partB/eta_mlflow.py gate                     #  gate with default thresholds
    python partB/eta_mlflow.py gate --min-rel-gain 0.60 #  gate with a stricter gain threshold

Ideas reused from the Lecture 4b notebook (mlflow_register.py, promotion_gate): wrapping a joblib bundle
in a PythonModel, a float64 signature that tolerates ints from JSON, aliases set with MlflowClient.
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # eta_data.py lives in the project root

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = "eta-model"
EXPERIMENT = "eta-serving"
RUSH_HOURS = (7, 8, 9, 16, 17, 18, 19)
EPS = 1e-9  # thresholds are inclusive; EPS absorbs floating-point error exactly at a boundary


# the pure decision function
def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    """Promote the candidate only if ALL criteria hold. Pure: no I/O, no globals, no randomness.

    prod / cand: dicts with keys mae, mae_rush, mae_rain, p95_ms.
    - overall MAE gain relative to prod:      (prod.mae - cand.mae) / prod.mae >= min_rel_gain
    - candidate single-row latency:           cand.p95_ms <= max_p95_ms
    - no slice regresses by more than max_slice_regression (relative to prod's slice MAE):
                                              cand.mae_rush <= prod.mae_rush * (1 + max_slice_regression)
                                              cand.mae_rain <= prod.mae_rain * (1 + max_slice_regression)
    """
    gain = (prod["mae"] - cand["mae"]) / prod["mae"]
    rush_change = (cand["mae_rush"] - prod["mae_rush"]) / prod["mae_rush"]
    rain_change = (cand["mae_rain"] - prod["mae_rain"]) / prod["mae_rain"]
    criteria = {
        "mae_gain": gain >= min_rel_gain - EPS,
        "latency": cand["p95_ms"] <= max_p95_ms + EPS,
        "rush_slice": rush_change <= max_slice_regression + EPS,
        "rain_slice": rain_change <= max_slice_regression + EPS,
    }
    details = {
        "rel_mae_gain": round(gain, 4), "min_rel_gain": min_rel_gain,
        "cand_p95_ms": cand["p95_ms"], "max_p95_ms": max_p95_ms,
        "rush_rel_change": round(rush_change, 4), "rain_rel_change": round(rain_change, 4),
        "max_slice_regression": max_slice_regression,
    }
    return {"criteria": criteria, "promote": all(criteria.values()), "details": details}


#  pyfunc wrapper + registration
def _pyfunc_class():
    import mlflow.pyfunc

    class EtaPyfunc(mlflow.pyfunc.PythonModel):
        """Wraps an ETA joblib bundle; returns eta_minutes (1 decimal) and model_version per row."""

        def load_context(self, context):
            b = joblib.load(context.artifacts["bundle"])
            self.model, self.cols, self.version = b["model"], b["feature_columns"], b["version"]

        def predict(self, context, model_input, params=None):
            X = pd.DataFrame(model_input)[self.cols]
            minutes = np.round(self.model.predict(X).astype(float), 1)
            return pd.DataFrame({"eta_minutes": minutes, "model_version": [self.version] * len(X)})

    return EtaPyfunc


def register():
    import joblib as jl
    import mlflow
    import sklearn
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
    from sklearn.metrics import mean_absolute_error

    import eta_data

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    EtaPyfunc = _pyfunc_class()

    train = pd.read_csv(os.path.join("data", "eta_train.csv"))
    X = train[eta_data.FEATURES].astype("float64")  # float schema: JSON ints and floats both accepted
    pins = [f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}", f"joblib=={jl.__version__}", f"mlflow=={mlflow.__version__}"]
    out_example = pd.DataFrame({"eta_minutes": [42.0], "model_version": ["v1"]})
    signature = infer_signature(X.head(5), out_example)

    registered = {}
    for ver in ("v1", "v2"):
        bundle_path = os.path.join("data", f"eta_{ver}.joblib")
        b = jl.load(bundle_path)
        mae = mean_absolute_error(train.eta_minutes, b["model"].predict(train[b["feature_columns"]]))
        with mlflow.start_run(run_name=f"eta-{ver}"):
            mlflow.log_params({"algorithm": type(b["model"]).__name__, "bundle_version": ver})
            mlflow.log_metric("train_mae", mae)
            info = mlflow.pyfunc.log_model(
                name="model", python_model=EtaPyfunc(), artifacts={"bundle": bundle_path},
                signature=signature, input_example=X.head(2), pip_requirements=pins,
                registered_model_name=MODEL_NAME)
        registered[ver] = str(info.registered_model_version)
        client.set_model_version_tag(MODEL_NAME, registered[ver], "bundle_version", ver)

    client.set_registered_model_alias(MODEL_NAME, "prod", registered["v1"])
    client.set_registered_model_alias(MODEL_NAME, "candidate", registered["v2"])

    aliases = client.get_registered_model(MODEL_NAME).aliases
    print(f"\nRegistry: {MODEL_NAME}   (tracking URI {MLFLOW_TRACKING_URI})\n")
    print("| version | bundle | aliases |\n|---:|---|---|")
    for mv in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version)):
        names = [a for a, v in aliases.items() if str(v) == str(mv.version)]
        print(f"| {mv.version} | {mv.tags.get('bundle_version', '?')} | {', '.join(names) or '-'} |")
    print("\npinned pip_requirements:", pins)
    return registered


#  evaluation + gate
def evaluate(model, fresh, n_latency=200):
    """Metrics the gate needs, for one loaded pyfunc model, on fresh (never trained on) data."""
    import eta_data

    X = fresh[eta_data.FEATURES].astype("float64")
    y = fresh["eta_minutes"].to_numpy()
    pred = model.predict(X)["eta_minutes"].to_numpy()
    err = np.abs(pred - y)
    rush = fresh["hour"].isin(RUSH_HOURS).to_numpy()
    rain = (fresh["rain"] == 1).to_numpy()

    for i in range(10):  # warm-up: first calls pay one-off costs (imports, caches)
        model.predict(X.iloc[[i]])
    times = []
    for i in range(n_latency):
        row = X.iloc[[i]]
        t0 = time.perf_counter()
        model.predict(row)
        times.append((time.perf_counter() - t0) * 1000)
    return {"mae": round(float(err.mean()), 4), "mae_rush": round(float(err[rush].mean()), 4),
            "mae_rain": round(float(err[rain].mean()), 4),
            "p95_ms": round(float(np.percentile(times, 95)), 3),
            "n_rows": int(len(y)), "n_rush": int(rush.sum()), "n_rain": int(rain.sum())}


def promotion_gate(**thresholds):
    import mlflow

    import eta_data

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    fresh = eta_data.make_data(2000, seed=99)
    metrics = {}
    for alias in ("prod", "candidate"):
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
        metrics[alias] = evaluate(model, fresh)
    result = decide(metrics["prod"], metrics["candidate"], **thresholds)
    return {"thresholds": thresholds or "defaults", "prod": metrics["prod"],
            "candidate": metrics["candidate"], **result}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["register", "gate"])
    ap.add_argument("--min-rel-gain", type=float)
    ap.add_argument("--max-p95-ms", type=float)
    ap.add_argument("--max-slice-regression", type=float)
    a = ap.parse_args()
    if a.command == "register":
        register()
    else:
        th = {k: v for k, v in {"min_rel_gain": a.min_rel_gain, "max_p95_ms": a.max_p95_ms,
                                "max_slice_regression": a.max_slice_regression}.items() if v is not None}
        print(json.dumps(promotion_gate(**th), indent=2))
