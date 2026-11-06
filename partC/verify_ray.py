"""
verify_ray.py -- Part C2: deploy eta_ray on a 2-CPU Ray and prove the sticky canary works.

    python partC/verify_ray.py | tee partC/verify_output.txt

Checks (each is an assert; the script exits non-zero on the first failure):
  (a) 1000 distinct customers at weight 0.2 -> candidate share within +-4 percentage points of 20 %
  (b) 200 customers each get the same arm on a second call
  (c) the same holds after deleting and redeploying the identical app (new replicas, new router)
  (d) raising the weight 0.2 -> 0.5: no customer who was on the candidate moves back to prod
  (e) X-Beta: 1 forces the candidate for 50 requests from customers normally on prod
"""
import collections
import os
import sys
import time

import ray
import requests
from ray import serve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eta_ray  # noqa: E402

PORT = 8100
URL = f"http://127.0.0.1:{PORT}/v1/models/eta:predict"
APP = "eta"
ROW = {"distance_km": 12.5, "n_stops": 4, "hour": 10, "weekday": 2,
       "traffic_index": 0.3, "rain": 0, "weight_kg": 5.0}
IDS = [f"cust-{i:04d}" for i in range(1000)]
sess = requests.Session()


def deploy(weight):
    serve.run(eta_ray.build(weight=weight), name=APP, route_prefix="/")
    for _ in range(60):                       # wait until the router answers
        try:
            if sess.get(f"http://127.0.0.1:{PORT}/healthz", timeout=2).json()["canary_weight"] == weight:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("app did not become ready")


def arm_of(cid, beta=False):
    headers = {"X-Beta": "1"} if beta else {}
    r = sess.post(URL, json={"customer_id": cid, "instances": [ROW]}, headers=headers, timeout=10)
    assert r.status_code == 200, (r.status_code, r.text[:200])
    j = r.json()
    assert j["model_version"] == j["arm"], j          # the arm really is the model that answered
    return j["arm"]


def status_lines():
    st = serve.status().applications[APP]
    for name, d in st.deployments.items():
        print(f"    {name:14s} {str(d.status).split('.')[-1]:8s} replicas {dict(d.replica_states)}")


def main():
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True,
             logging_level="ERROR", log_to_driver=False)
    serve.start(http_options={"host": "127.0.0.1", "port": PORT}, logging_config={"log_level": "WARNING"})
    total = ray.cluster_resources().get("CPU")
    print(f"Ray CPUs: {total}")
    print(f"CPU reservations: router {eta_ray.ROUTER_CPUS}, prod {eta_ray.PROD_REPLICAS} x {eta_ray.MODEL_CPUS}, "
          f"candidate up to {eta_ray.CANDIDATE_AUTOSCALING['max_replicas']} x {eta_ray.MODEL_CPUS} = "
          f"{eta_ray.ROUTER_CPUS + (eta_ray.PROD_REPLICAS + eta_ray.CANDIDATE_AUTOSCALING['max_replicas']) * eta_ray.MODEL_CPUS:.2f} "
          f"of {total} CPUs")
    print(f"Candidate autoscaling_config: {eta_ray.CANDIDATE_AUTOSCALING}\n")

    deploy(0.2)
    used = total - ray.available_resources().get("CPU", 0)
    print(f"Deployed at weight 0.2 (CPU reserved right now: {used:.2f}):")
    status_lines()

    # (a) share of customers on the candidate
    first = {cid: arm_of(cid) for cid in IDS}
    share = sum(a == "v2" for a in first.values()) / len(IDS)
    print(f"\n(a) {len(IDS)} customers, weight 0.2 -> candidate share {share:.3f} "
          f"({collections.Counter(first.values())})")
    assert abs(share - 0.2) <= 0.04, share
    print("    PASS: within +-0.04 of 0.2")

    # (b) stickiness on a second call
    sample = IDS[:200]
    second = {cid: arm_of(cid) for cid in sample}
    changed = [c for c in sample if second[c] != first[c]]
    print(f"\n(b) second call for 200 customers: {len(changed)} changed arm "
          f"({sum(second[c] == 'v2' for c in sample)} of them on v2)")
    assert not changed, changed[:5]
    print("    PASS: every customer got the same arm again")

    # (c) stickiness after a full redeploy of the identical app
    serve.delete(APP)
    time.sleep(2)
    deploy(0.2)
    print("\n(c) app deleted and redeployed (new router and replicas):")
    status_lines()
    third = {cid: arm_of(cid) for cid in sample}
    changed = [c for c in sample if third[c] != first[c]]
    print(f"    {len(changed)} of 200 customers changed arm after the redeploy")
    assert not changed, changed[:5]
    print("    PASS: assignment survives a restart (it depends only on the id)")

    # (d) raising the weight only moves customers prod -> candidate
    deploy(0.5)
    after = {cid: arm_of(cid) for cid in IDS}
    back = [c for c in IDS if first[c] == "v2" and after[c] == "v1"]
    moved = [c for c in IDS if first[c] == "v1" and after[c] == "v2"]
    share5 = sum(a == "v2" for a in after.values()) / len(IDS)
    print(f"\n(d) weight 0.2 -> 0.5: candidate share {share:.3f} -> {share5:.3f}; "
          f"{len(moved)} moved prod->candidate, {len(back)} moved candidate->prod")
    assert not back, back[:5]
    print("    PASS: nobody who was on the candidate went back to prod")

    # (e) header override, using customers that the hash puts on prod at weight 0.5
    prod_ids = [c for c in IDS if after[c] == "v1"][:50]
    forced = [arm_of(c, beta=True) for c in prod_ids]
    print(f"\n(e) X-Beta: 1 on 50 requests from prod customers -> {collections.Counter(forced)}")
    assert len(forced) == 50 and all(a == "v2" for a in forced)
    assert all(arm_of(c) == "v1" for c in prod_ids[:10])   # and without the header they stay on prod
    print("    PASS: header forces the candidate; without it the same customers stay on prod")

    print("\nALL CHECKS PASSED")
    serve.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    main()
