"""Offline unit tests for the factor-regression tool (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from factor_regression import (  # noqa: E402
    load_returns,
    ols,
    parse_ff_csv,
    run_regression,
)


def _utc_index(n, start="2020-01-02"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


# --------------------------------------------------------------------------- #
# OLS core
# --------------------------------------------------------------------------- #
def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(size=n) * 1e-4
    y = 0.5 + 2.0 * x1 - 1.0 * x2 + noise
    design = np.column_stack([np.ones(n), x1, x2])
    beta, se, tstat, r2, obs = ols(design, y)
    assert obs == n
    np.testing.assert_allclose(beta, [0.5, 2.0, -1.0], atol=1e-4)
    assert r2 > 0.999
    # Tiny noise -> a hugely significant intercept.
    assert abs(tstat[0]) > 50


def test_ols_pure_noise_has_insignificant_alpha():
    rng = np.random.default_rng(7)
    n = 1000
    x1 = rng.normal(size=n)
    y = rng.normal(size=n)  # unrelated to x1, zero mean
    design = np.column_stack([np.ones(n), x1])
    _beta, _se, tstat, _r2, _obs = ols(design, y)
    assert abs(tstat[0]) < 2  # intercept not distinguishable from zero


# --------------------------------------------------------------------------- #
# Fama-French parser
# --------------------------------------------------------------------------- #
FF_SAMPLE = """This file was created by CMPT_ME_BEME ... (preamble line 1)
Copyright ... (preamble line 2)

,Mkt-RF,SMB,HML,RF
20200102,  1.00, -0.50,  0.25, 0.010
20200103, -2.00,  0.30, -99.99, 0.010
20200106,  0.50,  0.10,  0.05, 0.010

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
2019,  22.00,  5.00, -3.00, 2.00
"""


def test_parse_ff_keeps_only_daily_rows_and_scales():
    frame = parse_ff_csv(FF_SAMPLE)
    # Three 8-digit daily rows only; the 4-digit annual row is excluded.
    assert list(frame.index.strftime("%Y%m%d")) == ["20200102", "20200103", "20200106"]
    assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    # Values divided by 100 (percent -> decimal).
    assert frame.loc["2020-01-02", "Mkt-RF"] == 1.00 / 100
    assert frame.loc["2020-01-06", "SMB"] == 0.10 / 100
    # UTC-aware index.
    assert frame.index.tz is not None
    # -99.99 sentinel became NaN.
    assert np.isnan(frame.loc["2020-01-03", "HML"])
    # 22.00 (the annual Mkt-RF) must NOT appear anywhere.
    assert (frame["Mkt-RF"] != 0.22).all()


def test_parse_ff_momentum_shape_with_padded_header():
    text = ",   Mom   \n20200102,  1.50\n20200103, -0.75\n"
    frame = parse_ff_csv(text)
    assert list(frame.columns) == ["Mom"]  # whitespace stripped
    assert frame.loc["2020-01-02", "Mom"] == 1.50 / 100


# --------------------------------------------------------------------------- #
# Alignment + excess return
# --------------------------------------------------------------------------- #
def test_regression_recovers_construction_beta_and_zero_alpha():
    rng = np.random.default_rng(3)
    n = 750
    idx = _utc_index(n)
    mkt = rng.normal(0.0005, 0.01, size=n)  # daily market excess return
    rf = np.full(n, 0.0001)
    # Strategy built as 1.5x the market factor, with RF added back (so its
    # EXCESS return is 1.5 * Mkt-RF and its true alpha is zero). A whisper of
    # noise keeps the fit imperfect so standard errors are well-defined.
    strat = rf + 1.5 * mkt + rng.normal(0.0, 1e-5, size=n)
    factors = pd.DataFrame({"Mkt-RF": mkt, "SMB": 0.0, "HML": 0.0, "Mom": 0.0, "RF": rf}, index=idx)
    returns = pd.DataFrame({"strategy": strat}, index=idx)

    result = run_regression(returns, factors, ["Mkt-RF"])
    assert result.n == n
    # beta[0] is alpha, beta[1] is the market loading.
    assert abs(result.beta[1] - 1.5) < 1e-4
    assert abs(result.alpha_daily) < 1e-5
    assert result.r2 > 0.999
    assert abs(result.alpha_tstat) < 2  # alpha not distinguishable from zero


def test_regression_inner_joins_on_dates():
    # Factors cover a wider window than the returns; only the overlap is used.
    fidx = _utc_index(20)
    factors = pd.DataFrame(
        {
            "Mkt-RF": np.linspace(-0.01, 0.01, 20),
            "SMB": 0.0,
            "HML": 0.0,
            "Mom": 0.0,
            "RF": 0.0001,
        },
        index=fidx,
    )
    ridx = fidx[5:15]  # 10 overlapping days
    returns = pd.DataFrame({"strategy": np.linspace(0, 0.02, 10)}, index=ridx)
    result = run_regression(returns, factors, ["Mkt-RF"])
    assert result.n == 10


def test_load_returns_parses_date_to_utc(tmp_path):
    path = tmp_path / "rets.csv"
    path.write_text("date,strategy,benchmark\n2020-01-02,0.01,0.005\n2020-01-03,-0.02,0.001\n")
    frame = load_returns(path)
    assert list(frame.columns) == ["strategy", "benchmark"]
    assert frame.index.tz is not None
    assert frame.loc["2020-01-02", "strategy"] == 0.01
