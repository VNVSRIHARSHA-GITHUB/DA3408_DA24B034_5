# AI Usage Disclosure

## AI Tool Used

- Claude (Anthropic)

## How It Was Used and Impact

1. **Conceptual Clarification**
   - Used AI to understand the assignment and the ideas behind each part: FastAPI validation and middleware, Prometheus metrics, load testing with Little's Law, why uvicorn workers help with Python's GIL, MLflow pyfunc models, registry aliases and why a running server does not pick up an alias change, shadow testing, Ray Serve deployments and autoscaling, sticky canaries with a stable hash (and why Python's `hash()` is not stable), and release smoke tests.
   - **Impact:** Helped me understand how the four parts connect (build the service → prove v2 is better → release it slowly → block a bad release) and how to read my own results.

2. **Code Generation**
   - some parts of code in `partA/` to `partD/` was written with AI help, following the patterns from the Lecture 4a–4d notebooks, which are cited in each file's docstring. The given starter files (`eta_data.py`, `load_test.py`) were not changed.
   - I made design decisions with it after checking the lecture material, for example ignoring unknown input fields the same way the lecture's `rest_app.py` does (MLflow deck, slide 15).
   - **Impact:** Sped up the implementation, so more time went into running the stages, measuring, collecting evidence and checking the results.

3. **Debugging and Troubleshooting**
   - Used AI to work through problems found while running the stages, for example: the throughput drop after 2 clients in the load test (I tested whether uvicorn's access log caused it by repeating the run with `--no-access-log`; it did not), redoing the 1-worker baseline so the 2-worker comparison ran under the same conditions, a port 8000 conflict with my other project's Docker containers, and reading MLflow and Ray warnings.
   - **Impact:** Helped identify the causes and keep the measurements fair and comparable.

4. **Report, Command and Documentation Assistance**
   - Used AI to understand the terminal commands (uvicorn, MLflow, Ray, git), to draft `REPORT.pdf` from my own output files(The written answers are my own; AI only helped structure tables and correct the grammar.), and to structure this `README.md`. I reviewed and edited both.
   - **Impact:** Helped me run every stage correctly and document the work clearly within the four-page limit.

## Responsibility

- I reviewed, ran and understood the code, commands and configuration in the final submission, and I can explain how each part works.
- Everything was actually run on my laptop (Intel Core i5-13420H, Ubuntu 26.04, Python 3.14.4). All screenshots, output files (`.json`, `.txt`) and measured numbers (throughput, latency, MAE, canary shares, smoke-test results) come from real runs, not generated or fabricated by AI.
- I checked that every number in `REPORT.pdf` matches the output files in `partA/` to `partD/`.
