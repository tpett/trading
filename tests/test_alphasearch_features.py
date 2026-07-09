"""compute_rolling_features: hand-constructed fixtures plus an independent
per-window OLS cross-check (evaluate.ols is the repo's own hand-rolled
reference implementation, so the two paths share no rolling machinery)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trading.alphasearch.evaluate import ols
from trading.alphasearch.panel import (
    BETA_MIN_OBS,
    BETA_WINDOW,
    IVOL_MIN_OBS,
    IVOL_WINDOW,
    ROLLING_FEATURES,
    PanelData,
    compute_rolling_features,
)


def _factors(periods: int, seed: int = 11, start: str = "2019-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0004, 0.01, size=periods),
            "SMB": rng.normal(0.0, 0.005, size=periods),
            "HML": rng.normal(0.0, 0.005, size=periods),
            "RF": 0.0001,
            "Mom": rng.normal(0.0, 0.006, size=periods),
        },
        index=idx,
    )


def _closes_from_excess(excess: np.ndarray, factors: pd.DataFrame) -> pd.Series:
    """A close series whose pct_change() minus RF reproduces `excess` on the
    factor calendar exactly (one leading seed bar absorbs the NaN first
    return)."""
    rets = excess + factors["RF"].to_numpy()
    seed_day = factors.index[0] - pd.Timedelta(1, unit="D")
    idx = pd.DatetimeIndex([seed_day]).append(factors.index)
    return pd.Series(100.0 * np.cumprod([1.0, *(1 + rets)]), index=idx)


def test_pure_ff3_combination_has_zero_ivol_and_exact_beta():
    factors = _factors(300)
    mkt = factors["Mkt-RF"].to_numpy()
    excess = 0.0005 + 0.7 * mkt  # exactly linear in the design -> zero residuals
    closes = {"AAA": _closes_from_excess(excess, factors)}
    feats = compute_rolling_features(closes, factors)["AAA"]
    assert list(feats.columns) == ROLLING_FEATURES
    assert feats["ivol"].iloc[-1] < 1e-6
    assert math.isclose(feats["beta"].iloc[-1], 0.7, rel_tol=1e-6)


def test_ivol_matches_an_independent_per_window_ols():
    factors = _factors(120)
    rng = np.random.default_rng(5)
    excess = rng.normal(0.0002, 0.01, size=120)
    closes = {"AAA": _closes_from_excess(excess, factors)}
    ivol = compute_rolling_features(closes, factors)["AAA"]["ivol"]
    x3 = factors[["Mkt-RF", "SMB", "HML"]].to_numpy()
    for t in range(120):
        lo = max(0, t + 1 - IVOL_WINDOW)
        yy, xx = excess[lo:t + 1], x3[lo:t + 1]
        got = ivol.iloc[t]
        if len(yy) < IVOL_MIN_OBS:
            assert math.isnan(got), t
            continue
        design = np.column_stack([np.ones(len(yy)), xx])
        beta, _se, _t, _r2, n = ols(design, yy)
        resid = yy - design @ beta
        want = math.sqrt(float(resid @ resid) / (n - 4)) * math.sqrt(252)
        assert math.isclose(got, want, rel_tol=1e-8), t


def test_beta_matches_cov_over_var_and_respects_the_min_obs_floor():
    factors = _factors(300)
    rng = np.random.default_rng(9)
    mkt = factors["Mkt-RF"].to_numpy()
    excess = 1.3 * mkt + rng.normal(0.0, 0.004, size=300)
    closes = {"AAA": _closes_from_excess(excess, factors)}
    beta = compute_rolling_features(closes, factors)["AAA"]["beta"]
    assert beta.iloc[: BETA_MIN_OBS - 1].isna().all()   # <=125 obs -> NaN
    assert not math.isnan(beta.iloc[BETA_MIN_OBS - 1])  # 126th obs -> value
    for t in (BETA_MIN_OBS - 1, 200, 299):
        lo = max(0, t + 1 - BETA_WINDOW)
        yy, xx = excess[lo:t + 1], mkt[lo:t + 1]
        xc = xx - xx.mean()
        want = float(xc @ (yy - yy.mean()) / (xc @ xc))
        assert math.isclose(beta.iloc[t], want, rel_tol=1e-8), t


def test_ivol_min_obs_boundary():
    factors = _factors(40)
    rng = np.random.default_rng(3)
    excess = rng.normal(0.0, 0.01, size=40)
    closes = {"AAA": _closes_from_excess(excess, factors)}
    ivol = compute_rolling_features(closes, factors)["AAA"]["ivol"]
    assert ivol.iloc[: IVOL_MIN_OBS - 1].isna().all()   # 14 obs -> NaN
    assert not math.isnan(ivol.iloc[IVOL_MIN_OBS - 1])  # 15 obs -> value


def test_empty_factors_yield_no_features():
    idx = pd.date_range("2020-01-02", periods=2, freq="B", tz="UTC")
    closes = {"AAA": pd.Series([100.0, 101.0], index=idx)}
    assert compute_rolling_features(closes, pd.DataFrame()) == {}


def test_feature_gather_is_as_of_and_nan_when_absent():
    idx = pd.date_range("2020-01-06", periods=3, freq="B", tz="UTC")
    feats = {"AAA": pd.DataFrame(
        {"ivol": [0.1, 0.2, 0.3], "beta": [1.0, 1.1, 1.2]}, index=idx
    )}
    panel = PanelData(closes={}, features=feats, symbols=("AAA",))
    before = pd.Timestamp("2020-01-05", tz="UTC")
    assert math.isnan(panel.view(before).feature("AAA", "ivol"))
    mid = pd.Timestamp("2020-01-07", tz="UTC")
    assert panel.view(mid).feature("AAA", "ivol") == 0.2
    assert panel.view(mid).feature("AAA", "beta") == 1.1
    assert math.isnan(panel.view(mid).feature("NOPE", "ivol"))
