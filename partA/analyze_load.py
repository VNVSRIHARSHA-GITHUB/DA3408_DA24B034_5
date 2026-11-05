"""
analyze_load.py -- turn load_test.py JSON output into the Part A3 report tables.

    python partA/analyze_load.py partA/loadtest_results.json                      # sweep table + SLO
    python partA/analyze_load.py partA/loadtest_batch1.json partA/loadtest_batch100.json --batch 1 100
    python partA/analyze_load.py partA/loadtest_results.json partA/loadtest_workers2.json --compare

Little's Law for a closed loop: concurrency L = throughput X * latency W, so W = L / X.
Comparing that implied mean latency with the measured p50 checks that the numbers are consistent.
"""
import argparse
import json

SLO_P95_MS = 80.0


def load(path):
    rows = json.load(open(path))
    for r in rows:
        if r["errors"]:
            raise SystemExit(f"{path}: concurrency {r['concurrency']} had {r['errors']} errors -- "
                             "fix the server and re-run; numbers with errors are not valid")
    return rows


def sweep(path):
    rows = load(path)
    print(f"\n### Load sweep ({path})\n")
    print("| clients | requests | req/s | p50 ms | p95 ms | p99 ms | errors | Little's Law W = L/X (ms) | p95 <= 80 ms |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for r in rows:
        w = 1000 * r["concurrency"] / r["rps"]
        ok = "yes" if r["p95_ms"] <= SLO_P95_MS else "no"
        print(f"| {r['concurrency']} | {r['requests']} | {r['rps']} | {r['p50_ms']} | {r['p95_ms']} | "
              f"{r['p99_ms']} | {r['errors']} | {w:.1f} | {ok} |")
    best = max(r["rps"] for r in rows)
    knee = next(r for r in rows if r["rps"] >= 0.9 * best)
    within = [r for r in rows if r["p95_ms"] <= SLO_P95_MS]
    print(f"\n- Peak throughput: {best} req/s.")
    print(f"- Saturation (first level reaching >= 90% of the peak): {knee['concurrency']} clients "
          f"at {knee['rps']} req/s; beyond it extra clients mostly add queueing (W grows ~ L).")
    if within:
        cap = max(within, key=lambda r: r["rps"])
        print(f"- Capacity under p95 <= {SLO_P95_MS:.0f} ms: {cap['rps']} req/s "
              f"(at {cap['concurrency']} clients, p95 {cap['p95_ms']} ms).")
    else:
        print(f"- No tested level met p95 <= {SLO_P95_MS:.0f} ms.")
    for r in rows:
        nxt = next((s for s in rows if s["concurrency"] > r["concurrency"]), None)
        if r["p95_ms"] <= SLO_P95_MS and nxt and nxt["p95_ms"] > SLO_P95_MS:
            print(f"- The SLO breaks between {r['concurrency']} and {nxt['concurrency']} clients "
                  f"(p95 {r['p95_ms']} -> {nxt['p95_ms']} ms).")


def batches(paths, sizes):
    print("\n### Cost per row, concurrency 1\n")
    print("| batch | req/s | p50 ms per request | rows/s | ms per row (1000 / rows/s) |")
    print("|---:|---:|---:|---:|---:|")
    for path, b in zip(paths, sizes):
        r = load(path)[0]
        rows_s = r["rps"] * b
        print(f"| {b} | {r['rps']} | {r['p50_ms']} | {rows_s:.0f} | {1000 / rows_s:.3f} |")


def compare(base_path, new_path):
    base = {r["concurrency"]: r for r in load(base_path)}
    new = {r["concurrency"]: r for r in load(new_path)}
    print(f"\n### Before / after ({base_path} vs {new_path})\n")
    print("| clients | req/s before | req/s after | speed-up | p95 before | p95 after |")
    print("|---:|---:|---:|---:|---:|---:|")
    for c in sorted(set(base) & set(new)):
        b, n = base[c], new[c]
        print(f"| {c} | {b['rps']} | {n['rps']} | {n['rps'] / b['rps']:.2f}x | {b['p95_ms']} | {n['p95_ms']} |")

    def capacity(rows):
        ok = [r["rps"] for r in rows.values() if r["p95_ms"] <= SLO_P95_MS]
        return max(ok) if ok else 0.0
    cb, cn = capacity(base), capacity(new)
    ratio = f"{cn / cb:.2f}x" if cb else "n/a"
    print(f"\n- Capacity under the SLO: {cb} -> {cn} req/s ({ratio}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--batch", type=int, nargs="+", help="batch size of each file (cost-per-row table)")
    ap.add_argument("--compare", action="store_true", help="first file = before, second = after")
    a = ap.parse_args()
    if a.batch:
        batches(a.files, a.batch)
    elif a.compare:
        compare(a.files[0], a.files[1])
    else:
        for f in a.files:
            sweep(f)
