"""
shadow.py -- Part B3: send the same 200 single-row requests to prod (:5101) and candidate (:5102).

Start the two servers first (each in its own terminal, from the project root):
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve -m "models:/eta-model@prod"      -p 5101 --env-manager local
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local
Then:
    python partB/shadow.py | tee partB/shadow_output.txt

Clients would only ever see prod's answer; the candidate is observed, not trusted.
The order of the two calls alternates per row so neither server always gets the "second, warmer" slot.
"""
import argparse
import time

import numpy as np
import pandas as pd
import requests

FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]


def call(sess, port, rec):
    t0 = time.perf_counter()
    r = sess.post(f"http://localhost:{port}/invocations", json={"dataframe_records": [rec]}, timeout=10)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    out = r.json()["predictions"][0]
    return out["eta_minutes"], out["model_version"], ms


def main(n, data, prod_port, cand_port):
    recs = pd.read_csv(data)[FEATURES].head(n).to_dict(orient="records")
    sess = requests.Session()
    for port in (prod_port, cand_port):  # warm-up, not measured
        for rec in recs[:5]:
            call(sess, port, rec)

    res = {prod_port: ([], [], set()), cand_port: ([], [], set())}
    for i, rec in enumerate(recs):
        order = (prod_port, cand_port) if i % 2 == 0 else (cand_port, prod_port)
        for port in order:
            eta, ver, ms = call(sess, port, rec)
            res[port][0].append(eta); res[port][1].append(ms); res[port][2].add(ver)

    p, c = np.array(res[prod_port][0]), np.array(res[cand_port][0])
    diff = np.abs(p - c)
    print(f"Shadow comparison on {len(recs)} identical single-row requests from {data}")
    print(f"  prod      :{prod_port} answered as {sorted(res[prod_port][2])}")
    print(f"  candidate :{cand_port} answered as {sorted(res[cand_port][2])}\n")
    print("| metric | value |\n|---|---:|")
    print(f"| mean absolute difference (min) | {diff.mean():.2f} |")
    print(f"| median absolute difference (min) | {np.median(diff):.2f} |")
    print(f"| max absolute difference (min) | {diff.max():.2f} |")
    print(f"| requests differing by > 10 min | {int((diff > 10).sum())} / {len(diff)} ({100 * (diff > 10).mean():.1f} %) |")
    print(f"| mean prediction prod / candidate (min) | {p.mean():.1f} / {c.mean():.1f} |")
    for name, port in (("prod", prod_port), ("candidate", cand_port)):
        lat = np.array(res[port][1])
        print(f"| {name} latency p50 / p95 (ms) | {np.percentile(lat, 50):.1f} / {np.percentile(lat, 95):.1f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="data/eta_fresh.csv")
    ap.add_argument("--prod-port", type=int, default=5101)
    ap.add_argument("--cand-port", type=int, default=5102)
    a = ap.parse_args()
    main(a.n, a.data, a.prod_port, a.cand_port)
