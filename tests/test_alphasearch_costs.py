"""Corwin-Schultz spread estimator + spread-based rebalance charges + the SPY
benchmark (R1 gate amendment spec section 3): hand-computable fixtures, a
real-AAPL sanity check, and the cost-charging/benchmark arithmetic."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from trading.alphasearch.costs import (
    DEFAULT_SPY_CACHE_DIR,
    SPREAD_CAP,
    SPREAD_FLOOR,
    SPY_SYMBOL,
    apply_rebalance_charges,
    cost_charged_lo,
    effective_spread,
    load_spy_closes,
    spread_rebalance_charges,
    spy_benchmark,
    trailing_effective_spread,
)
from trading.alphasearch.panel import PanelData

REPO_ROOT = Path(__file__).resolve().parent.parent
AAPL_CACHE = REPO_ROOT / "data" / "equities-tiingo" / "AAPL.parquet"


def _bars(highs: list[float], lows: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=len(highs), freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": highs, "high": highs, "low": lows, "close": highs,
         "volume": 1.0, "div_cash": 0.0, "split_factor": 1.0, "close_raw": highs},
        index=idx,
    )


# --------------------------------------------------------------------------- #
# Hand-computed fixtures (spec section 3 formula, verbatim)
# --------------------------------------------------------------------------- #
def test_trailing_effective_spread_hand_computed_two_and_three_days():
    # H/L: day0=(101,99) day1=(102,98) day2=(103,97). Row 0 has no predecessor
    # (NaN beta/gamma); row 1 uses ONLY the (day0,day1) pair (the trailing
    # mean's sole non-NaN observation); row 2 averages BOTH pairs first
    # (spec's month-level convention -- see costs.py docstring), then runs
    # the alpha/spread transform once.
    bars = _bars([101.0, 102.0, 103.0], [99.0, 98.0, 97.0])
    k = 3 - 2 * math.sqrt(2)

    def spread_from(beta_bar: float, gamma_bar: float) -> float:
        alpha = (math.sqrt(2 * beta_bar) - math.sqrt(beta_bar)) / k - math.sqrt(
            gamma_bar / k
        )
        return 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))

    log_hl_sq = [math.log(h / low) ** 2 for h, low in zip([101.0, 102.0, 103.0],
                                                           [99.0, 98.0, 97.0], strict=True)]
    beta1 = log_hl_sq[1] + log_hl_sq[0]
    gamma1 = math.log(102.0 / 98.0) ** 2
    beta2 = log_hl_sq[2] + log_hl_sq[1]
    gamma2 = math.log(103.0 / 97.0) ** 2

    trailing = trailing_effective_spread(bars, window=21)
    assert math.isnan(trailing.iloc[0])
    assert trailing.iloc[1] == pytest.approx(spread_from(beta1, gamma1), rel=1e-9)
    # Row 2's trailing window averages BOTH (beta1,gamma1) and (beta2,gamma2)
    # before the transform.
    expected2 = spread_from((beta1 + beta2) / 2, (gamma1 + gamma2) / 2)
    assert trailing.iloc[2] == pytest.approx(expected2, rel=1e-9)
    assert expected2 == pytest.approx(0.02175069711742099, rel=1e-6)


def test_trailing_effective_spread_floors_negative_estimates_at_zero():
    # A tight range around 100 followed by a tight range around 105 (same
    # width, shifted): the two-day high/low span the FULL 5-point gap while
    # each day's own beta term stays tiny, so gamma >> beta and the raw
    # alpha/spread comes out strongly negative -- floored at 0 per spec.
    bars = _bars([100.1, 105.1], [99.9, 104.9])
    trailing = trailing_effective_spread(bars, window=21)
    assert math.isnan(trailing.iloc[0])
    assert trailing.iloc[1] == 0.0


def test_effective_spread_applies_the_floor_and_cap():
    # Two ultra-tight-range days: the raw estimate floors below 2bps ->
    # effective_spread clamps up to SPREAD_FLOOR.
    bars = _bars([100.001, 100.001], [99.999, 99.999])
    spread = effective_spread(bars)
    assert spread == pytest.approx(SPREAD_FLOOR)

    # Two days with the SAME wide H/L band (125/100, a real ~22% two-day
    # spread signal by the hand-derived beta=2x/gamma=x identity): the raw
    # estimate lands far above 5% -- effective_spread clamps down to
    # SPREAD_CAP.
    bars_wide = _bars([125.0, 125.0], [100.0, 100.0])
    spread_wide = effective_spread(bars_wide)
    assert spread_wide == pytest.approx(SPREAD_CAP)


def test_effective_spread_nan_below_two_bars():
    assert math.isnan(effective_spread(_bars([100.0], [99.0])))
    assert math.isnan(effective_spread(_bars([], [])))


# --------------------------------------------------------------------------- #
# Real-AAPL sanity check (skipped when the gitignored cache isn't present)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not AAPL_CACHE.exists(), reason="no local AAPL bar cache")
def test_aapl_effective_spread_is_single_digit_bps():
    bars = pd.read_parquet(AAPL_CACHE)
    spread = effective_spread(bars)
    assert not math.isnan(spread)
    bps = spread * 1e4
    assert 2.0 <= bps < 10.0, f"AAPL effective spread {bps:.2f}bps not single-digit"


# --------------------------------------------------------------------------- #
# apply_rebalance_charges (generic; moved here from robustness.py)
# --------------------------------------------------------------------------- #
def test_apply_rebalance_charges_lands_on_the_first_following_day():
    idx = pd.date_range("2020-01-02", periods=5, freq="B", tz="UTC")
    series = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=idx)
    charged = apply_rebalance_charges(series, [(idx[1], 0.02), (idx[4], 0.10)])
    expected = series.copy()
    expected.iloc[2] -= 0.02   # first day strictly after idx[1]
    # idx[4] is the LAST day: no day to land on, dropped.
    pd.testing.assert_series_equal(charged, expected)


# --------------------------------------------------------------------------- #
# spread_rebalance_charges / cost_charged_lo
# --------------------------------------------------------------------------- #
def _panel_with_bars(symbols: list[str], n: int = 10) -> PanelData:
    idx = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    bars: dict[str, pd.DataFrame] = {}
    closes: dict[str, pd.Series] = {}
    for i, sym in enumerate(symbols):
        base = 100.0 + i
        high = pd.Series(base * 1.002, index=idx)
        low = pd.Series(base * 0.998, index=idx)
        close = pd.Series(base, index=idx)
        bars[sym] = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close,
             "volume": 1e5, "div_cash": 0.0, "split_factor": 1.0, "close_raw": close},
            index=idx,
        )
        closes[sym] = close
    return PanelData(closes=closes, bars=bars, symbols=tuple(symbols))


def test_spread_rebalance_charges_entries_and_exits_hand_computed():
    panel = _panel_with_bars(["A", "B", "C"])
    idx = panel.closes["A"].index
    # Formation: top=(A,B). Rebalance: top=(A,C) -- B exits, C enters.
    rebalances = (
        (idx[3], ("A", "B"), ()),
        (idx[6], ("A", "C"), ()),
    )
    charges, skipped = spread_rebalance_charges(panel, rebalances)
    assert skipped == 0
    assert len(charges) == 2
    date0, charge0 = charges[0]
    date1, charge1 = charges[1]
    assert date0 == idx[3] and date1 == idx[6]
    # Formation: A and B both ENTER at the current n=2 weight.
    spread_a = effective_spread(panel.view(idx[3]).bars("A"))
    spread_b = effective_spread(panel.view(idx[3]).bars("B"))
    assert charge0 == pytest.approx((spread_a / 2 + spread_b / 2) / 2)
    # Rebalance 2: C enters at the NEW n=2 weight; B exits at the OLD n=2.
    spread_c = effective_spread(panel.view(idx[6]).bars("C"))
    spread_b2 = effective_spread(panel.view(idx[6]).bars("B"))
    assert charge1 == pytest.approx(spread_c / 2 / 2 + spread_b2 / 2 / 2)


def test_spread_rebalance_charges_counts_missing_spread_names():
    panel = _panel_with_bars(["A", "B"], n=1)  # 1 bar: effective_spread is NaN
    idx = panel.closes["A"].index
    rebalances = ((idx[0], ("A", "B"), ()),)
    charges, skipped = spread_rebalance_charges(panel, rebalances)
    assert skipped == 2
    assert charges == [(idx[0], 0.0)]


def test_cost_charged_lo_deducts_on_the_first_return_day_after_rebalance():
    panel = _panel_with_bars(["A", "B"])
    idx = panel.closes["A"].index
    rebalances = ((idx[3], ("A", "B"), ()),)
    lo = pd.Series(0.001, index=idx[4:])
    charged, skipped = cost_charged_lo(panel, lo, rebalances)
    assert skipped == 0
    charges, _ = spread_rebalance_charges(panel, rebalances)
    expected = apply_rebalance_charges(lo, charges)
    pd.testing.assert_series_equal(charged, expected)
    assert charged.iloc[0] < lo.iloc[0]   # a real charge landed


# --------------------------------------------------------------------------- #
# SPY loading + benchmark
# --------------------------------------------------------------------------- #
def test_load_spy_closes_returns_none_when_cache_has_no_spy(tmp_path):
    assert load_spy_closes(tmp_path) is None


def test_load_spy_closes_reads_the_parquet(tmp_path):
    idx = pd.date_range("2020-01-02", periods=5, freq="B", tz="UTC")
    close = pd.Series([300.0, 301.0, 302.0, 303.0, 304.0], index=idx)
    pd.DataFrame({"close": close}).to_parquet(tmp_path / f"{SPY_SYMBOL}.parquet")
    loaded = load_spy_closes(tmp_path)
    assert loaded is not None
    pd.testing.assert_series_equal(loaded, close, check_names=False, check_freq=False)


def test_spy_benchmark_hand_computed():
    idx = pd.date_range("2020-01-02", periods=4, freq="B", tz="UTC")
    closes = pd.Series([100.0, 110.0, 99.0, 108.9], index=idx)
    result = spy_benchmark(closes, idx[0], idx[-1])
    assert result.total_return == pytest.approx(108.9 / 100.0 - 1.0)
    assert result.n_obs == 3
    rets = closes.pct_change().dropna()
    expected_sharpe = float(rets.mean()) / float(rets.std(ddof=1)) * math.sqrt(252)
    assert result.sharpe_annual == pytest.approx(expected_sharpe)


def test_spy_benchmark_restricts_to_the_window():
    idx = pd.date_range("2020-01-02", periods=6, freq="B", tz="UTC")
    closes = pd.Series([100.0, 101.0, 102.0, 200.0, 201.0, 202.0], index=idx)
    result = spy_benchmark(closes, idx[0], idx[2])
    assert result.total_return == pytest.approx(102.0 / 100.0 - 1.0)
    assert result.n_obs == 2


def test_default_spy_cache_dir_is_the_largecap_tiingo_cache():
    assert DEFAULT_SPY_CACHE_DIR == Path("data") / "equities-tiingo"
