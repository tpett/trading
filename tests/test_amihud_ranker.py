"""Offline tests for the amihud_v1 ranker (spec: docs/experiments.md section
11 -- adapts the battery-passed, HOLDOUT-ELIGIBLE alphasearch `amihud` signal
to the walk-forward backtester's ranker contract).

Fixtures use a CONSTANT-DOLLAR-VOLUME construction (volume = D / close, a
geometric close), the same closed-form trick
`tests/test_alphasearch_tier1.py::test_amihud_constant_dollar_volume_closed_form_and_floor`
uses to pin `amihud_lambda` exactly: every daily |return| is the geometric
rate `r` and every day's dollar volume is the constant `D`, so
`amihud_lambda == r / D` to machine precision -- letting every composite here
be a hand-computable percentile, not an approximation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading.config import SignalConfig
from trading.signals.engine import compute_features
from trading.signals.illiquidity import AMIHUD_V1_COLUMNS, amihud_v1
from trading.signals.registry import get_ranker

CONFIG = SignalConfig(
    momentum_windows=(5, 10, 20),
    calendar_days=False,
    vol_window=5,
    volume_week=5,
    volume_baseline=20,
    breakout_windows=(10, 20),
    rsi_window=14,
    mean_window=20,
    raw_return_days=30,
    ranker="amihud_v1",
)

# Four names spanning a wide, exactly-orderable lambda range: D=1e5 (most
# illiquid, lambda 1e-7) down to D=1e8 (least illiquid, lambda 1e-10). Every
# name shares the same 1% daily geometric drift so only D drives the ordering.
RATE = 0.01
SPECS = [
    ("ILLIQ1", 1e5),   # lambda = 1e-7 (most illiquid -> composite 1.0)
    ("ILLIQ2", 1e6),   # lambda = 1e-8
    ("ILLIQ3", 1e7),   # lambda = 1e-9
    ("ILLIQ4", 1e8),   # lambda = 1e-10 (least illiquid -> composite 0.25)
]
N_BARS = 260  # >= 127 needed for the 126-valid-term amihud floor


def _geom_bars(rate: float, dollar_volume: float, n: int = N_BARS) -> pd.DataFrame:
    idx = pd.date_range("2025-01-02", periods=n, freq="B", tz="UTC")
    close = pd.Series([100.0 * (1.0 + rate) ** i for i in range(n)], index=idx)
    volume = dollar_volume / close
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _universe() -> dict[str, pd.DataFrame]:
    return {name: _geom_bars(RATE, dv) for name, dv in SPECS}


AS_OF = _geom_bars(RATE, 1e6).index[-1]


def test_registered_requires_neither_fundamentals_nor_skew():
    spec = get_ranker("amihud_v1")
    assert spec.fn is amihud_v1
    assert spec.requires_fundamentals is False
    assert spec.requires_skew is False


def test_amihud_v1_lambda_closed_form_and_composite_ordering():
    out = amihud_v1(_universe(), AS_OF, CONFIG)
    assert list(out.columns) == AMIHUD_V1_COLUMNS
    for name, dv in SPECS:
        assert out.loc[name, "amihud_lambda"] == pytest.approx(RATE / dv, rel=1e-9)
    # composite = percentile of lambda over 4 names: most illiquid (ILLIQ1)
    # tops out at 1.0, least illiquid (ILLIQ4) bottoms out at 1/4.
    assert out.loc["ILLIQ1", "composite"] == pytest.approx(1.0)
    assert out.loc["ILLIQ2", "composite"] == pytest.approx(0.75)
    assert out.loc["ILLIQ3", "composite"] == pytest.approx(0.5)
    assert out.loc["ILLIQ4", "composite"] == pytest.approx(0.25)
    composites = [out.loc[name, "composite"] for name, _ in SPECS]
    assert composites == sorted(composites, reverse=True)
    assert "raw_return_30d" in out.columns


def test_amihud_v1_excludes_nan_lambda_names_never_neutral():
    # ZEROVOL clears the momentum price-history gate (260 bars, same as the
    # dense names) but every bar has zero volume -> zero dollar volume on
    # every term -> amihud_lambda is NaN. It must be DROPPED from the output
    # entirely, never neutralized to a fallback percentile (unlike the skew
    # rankers' fail-open policy -- see the module docstring for why).
    universe = _universe()
    zero_vol = _geom_bars(RATE, 1e6).copy()
    zero_vol["volume"] = 0.0
    bars = {**universe, "ZEROVOL": zero_vol}
    out = amihud_v1(bars, AS_OF, CONFIG)
    assert "ZEROVOL" not in out.index
    # The dense names are unaffected -- same composites as the plain universe.
    assert out.loc["ILLIQ1", "composite"] == pytest.approx(1.0)
    assert out.loc["ILLIQ4", "composite"] == pytest.approx(0.25)


def test_amihud_v1_omits_symbols_without_price_history():
    short = {"THIN": _geom_bars(RATE, 1e6, n=5)}
    out = amihud_v1(short, short["THIN"].index[-1], CONFIG)
    assert out.empty


def test_amihud_v1_empty_bars_returns_empty_with_columns():
    out = amihud_v1({}, AS_OF, CONFIG)
    assert out.empty
    assert list(out.columns) == AMIHUD_V1_COLUMNS


def test_amihud_v1_raw_return_30d_matches_momentum_base():
    bars = _universe()
    out = amihud_v1(bars, AS_OF, CONFIG)
    base = compute_features(bars, AS_OF, CONFIG)
    for name in out.index:
        assert out.loc[name, "raw_return_30d"] == pytest.approx(
            base.loc[name, "raw_return_30d"], nan_ok=True
        )


def test_amihud_v1_pit_ignores_bars_strictly_after_as_of():
    full = _universe()
    # An as_of well before the end of history but still far past the amihud
    # floor (index 199 -> 200 bars visible, 199 valid terms >= 126).
    as_of = full["ILLIQ1"].index[199]
    truncated = {name: frame.loc[:as_of] for name, frame in full.items()}
    out_full = amihud_v1(full, as_of, CONFIG)
    out_truncated = amihud_v1(truncated, as_of, CONFIG)
    pd.testing.assert_frame_equal(out_full, out_truncated)

    # Explicitly mutate every row strictly after as_of into nonsense (a huge
    # price/volume dislocation) and confirm the score is bit-for-bit
    # unchanged -- the no-lookahead guarantee, not just "same shape data".
    mutated = {name: frame.copy() for name, frame in full.items()}
    future = mutated["ILLIQ1"].index[200:]
    mutated["ILLIQ1"].loc[future, "close"] *= 50.0
    mutated["ILLIQ1"].loc[future, "volume"] *= 0.001
    out_mutated = amihud_v1(mutated, as_of, CONFIG)
    pd.testing.assert_frame_equal(out_full, out_mutated)


def test_amihud_v1_as_of_row_itself_is_included():
    # side="right" truncation: a bar dated EXACTLY as_of is visible (known by
    # the decision), matching PanelView.bars / FeaturePanel.gather. A flat
    # constant-dollar-volume series can't distinguish "n vs n-1 identical
    # terms" (the mean barely moves), so this fixture puts a single OUTLIER
    # return exactly on the as_of bar: including it must shift the mean by
    # a hand-computable amount; a frame missing that last row must not.
    n = 200
    d = 1e6
    idx = pd.date_range("2025-01-02", periods=n, freq="B", tz="UTC")
    closes = [100.0 * (1.01**i) for i in range(n - 1)]
    closes.append(closes[-1] * 1.5)  # the as_of bar: a 50% jump, not 1%
    close = pd.Series(closes, index=idx)
    volume = d / close  # constant dollar volume d on every bar, incl. the jump
    frame = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": volume},
        index=idx,
    )
    as_of = idx[-1]
    bars = {"ILLIQ1": frame}
    out_with_asof = amihud_v1(bars, as_of, CONFIG)
    out_without_asof = amihud_v1({"ILLIQ1": frame.iloc[:-1]}, as_of, CONFIG)
    # 198 terms at 0.01 + 1 outlier term at 0.5, all over dollar volume d.
    expected_with = (198 * 0.01 + 0.5) / 199 / d
    expected_without = 0.01 / d  # 198 identical 0.01 terms, no outlier visible
    assert out_with_asof.loc["ILLIQ1", "amihud_lambda"] == pytest.approx(expected_with)
    assert out_without_asof.loc["ILLIQ1", "amihud_lambda"] == pytest.approx(expected_without)
    assert out_with_asof.loc["ILLIQ1", "amihud_lambda"] != pytest.approx(
        out_without_asof.loc["ILLIQ1", "amihud_lambda"]
    )
