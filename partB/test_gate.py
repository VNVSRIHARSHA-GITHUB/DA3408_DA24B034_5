"""
test_gate.py -- tests for the pure decide() function (Part-B2), synthetic numbers only.
Run from the project root:   pytest partB -v
"""
import pytest

from eta_mlflow import decide

# prod: MAE 10.0 overall, 12.0 in rush hours, 11.0 in rain
PROD = {"mae": 10.0, "mae_rush": 12.0, "mae_rain": 11.0, "p95_ms": 3.0}
# a candidate that passes every default criterion comfortably
GOOD = {"mae": 9.0, "mae_rush": 11.0, "mae_rain": 10.0, "p95_ms": 4.0}


def cand(**changes):
    return {**GOOD, **changes}


def test_clear_win_promotes():
    r = decide(PROD, GOOD)
    assert r["promote"] is True
    assert set(r["criteria"]) == {"mae_gain", "latency", "rush_slice", "rain_slice"}
    assert all(r["criteria"].values())


def test_identical_models_do_not_promote():
    r = decide(PROD, dict(PROD))
    assert r["promote"] is False
    assert r["criteria"]["mae_gain"] is False


@pytest.mark.parametrize("mae, promote", [
    (9.70, True),    # exactly 3.0 % better: boundary is inclusive
    (9.69, True),    # 3.1 % better: just inside
    (9.71, False),   # 2.9 % better: just outside
])
def test_mae_gain_boundary(mae, promote):
    r = decide(PROD, cand(mae=mae))
    assert r["criteria"]["mae_gain"] is promote
    assert r["promote"] is promote


@pytest.mark.parametrize("p95, promote", [(10.0, True), (9.99, True), (10.01, False)])
def test_latency_boundary(p95, promote):
    r = decide(PROD, cand(p95_ms=p95))
    assert r["criteria"]["latency"] is promote
    assert r["promote"] is promote


@pytest.mark.parametrize("rush, promote", [
    (12.60, True),   # exactly +5 % on the rush slice: allowed
    (12.59, True),   # just inside
    (12.61, False),  # just outside
])
def test_rush_slice_boundary(rush, promote):
    r = decide(PROD, cand(mae_rush=rush))
    assert r["criteria"]["rush_slice"] is promote
    assert r["promote"] is promote


@pytest.mark.parametrize("rain, promote", [(11.55, True), (11.54, True), (11.56, False)])
def test_rain_slice_boundary(rain, promote):
    r = decide(PROD, cand(mae_rain=rain))
    assert r["criteria"]["rain_slice"] is promote
    assert r["promote"] is promote


def test_better_overall_but_worse_in_rain_is_blocked():
    # the reason slices exist: a big average gain hides a regression for rainy deliveries
    r = decide(PROD, cand(mae=7.0, mae_rain=13.0))
    assert r["criteria"]["mae_gain"] is True
    assert r["criteria"]["rain_slice"] is False
    assert r["promote"] is False


def test_fast_enough_is_not_enough():
    r = decide(PROD, cand(mae=9.9, p95_ms=0.5))     # 1 % gain only
    assert r["criteria"]["latency"] is True
    assert r["promote"] is False


def test_thresholds_are_parameters():
    assert decide(PROD, GOOD)["promote"] is True                          # 10 % gain
    assert decide(PROD, GOOD, min_rel_gain=0.60)["promote"] is False       # needs 60 %
    assert decide(PROD, cand(p95_ms=25.0), max_p95_ms=30.0)["promote"] is True
    assert decide(PROD, cand(mae_rush=13.0), max_slice_regression=0.10)["promote"] is True


def test_decide_is_pure():
    prod, c = dict(PROD), dict(GOOD)
    first = decide(prod, c)
    assert decide(prod, c) == first                 # same inputs -> same output
    assert prod == PROD and c == GOOD               # inputs not modified
