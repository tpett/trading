"""Tests for the promoted factor-regression core + AlphaResult (no network)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from trading.alphasearch.evaluate import (
    ALL_FACTORS,
    annualized_sharpe,
    evaluate_alpha,
    run_regression,
)


def _utc_index(n, start="2020-01-02"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _factors(n, mkt, rf=0.0001):
    idx = _utc_index(n)
    # SMB/HML/Mom carry no signal here, but they must NOT be exact-zero columns:
    # a constant-zero regressor makes the four-factor design matrix singular
    # (rank-deficient), which raises LinAlgError before evaluate_alpha's
    # RF-handling assertions ever run. Tiny independent noise keeps the design
    # well-conditioned while staying far below the 1e-5 alpha_daily tolerance.
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "Mkt-RF": mkt,
            "SMB": rng.normal(0.0, 1e-8, size=n),
            "HML": rng.normal(0.0, 1e-8, size=n),
            "Mom": rng.normal(0.0, 1e-8, size=n),
            "RF": rf,
        },
        index=idx,
    )


def test_run_regression_subtract_rf_false_uses_raw_series():
    # A self-financing spread built as exactly alpha + 0.5 * Mkt-RF (plus a
    # whisper of noise so standard errors are defined). With subtract_rf=False
    # the intercept recovers alpha; the default (True) would shift it by -RF.
    rng = np.random.default_rng(11)
    n = 750
    mkt = rng.normal(0.0005, 0.01, size=n)
    alpha = 0.0003
    spread = alpha + 0.5 * mkt + rng.normal(0.0, 1e-5, size=n)
    factors = _factors(n, mkt)
    returns = pd.DataFrame({"strategy": spread}, index=factors.index)

    raw = run_regression(returns, factors, ["Mkt-RF"], subtract_rf=False)
    assert abs(raw.alpha_daily - alpha) < 1e-5
    assert abs(raw.beta[1] - 0.5) < 1e-3

    excess = run_regression(returns, factors, ["Mkt-RF"])  # default True
    assert abs(excess.alpha_daily - (alpha - 0.0001)) < 1e-5


def test_evaluate_alpha_ls_vs_lo_rf_handling():
    rng = np.random.default_rng(5)
    n = 600
    mkt = rng.normal(0.0004, 0.01, size=n)
    factors = _factors(n, mkt)
    series = pd.Series(0.0002 + 1.0 * mkt + rng.normal(0, 1e-5, size=n),
                       index=factors.index)

    ls = evaluate_alpha(series, factors, self_financing=True)
    lo = evaluate_alpha(series, factors, self_financing=False)
    # Same series: the long-only treatment subtracts RF=0.0001 from the alpha.
    assert abs(ls.four_factor.alpha_daily - 0.0002) < 1e-5
    assert abs(lo.four_factor.alpha_daily - 0.0001) < 1e-5
    # Both models present, with CAPM for contrast, and n populated.
    assert ls.four_factor.names == ["alpha", *ALL_FACTORS]
    assert ls.capm.names == ["alpha", "Mkt-RF"]
    assert ls.n == n
    assert not math.isnan(ls.sharpe_annual)


def test_annualized_sharpe_known_value():
    # Alternating +1%/-0.5% forever: mean 0.0025, std(ddof=1) known.
    r = pd.Series([0.01, -0.005] * 50)
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert math.isclose(annualized_sharpe(r), expected, rel_tol=1e-12)
    assert math.isnan(annualized_sharpe(pd.Series([0.01])))  # too short


def test_script_reexports_are_the_package_objects():
    # The promotion contract: the script is a thin wrapper, not a fork.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import factor_regression as script

    from trading.alphasearch import evaluate as package

    assert script.ols is package.ols
    assert script.parse_ff_csv is package.parse_ff_csv
    assert script.run_regression is package.run_regression
    assert script.load_factors is package.load_factors
    assert script.RegressionResult is package.RegressionResult
