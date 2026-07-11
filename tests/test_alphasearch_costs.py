"""Corwin-Schultz spread estimator + spread-based rebalance charges + the SPY
benchmark (R1 gate amendment spec section 3): hand-computable fixtures, a
real-AAPL sanity check, and the cost-charging/benchmark arithmetic.

Also the market-neutral gate's short-borrow model + both-legs cost charging
(R6 Stage 1 amendment, docs/superpowers/specs/2026-07-11-market-neutral-
gate-amendment.md sections 2 and 6)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from trading.alphasearch.costs import (
    BORROW_CAP_BPS,
    DEFAULT_SPY_CACHE_DIR,
    GC_FLOOR_BPS,
    SPREAD_CAP,
    SPREAD_FLOOR,
    SPY_SYMBOL,
    apply_rebalance_charges,
    cost_charged_lo,
    cost_charged_market_neutral,
    effective_spread,
    load_spy_closes,
    short_borrow_bps,
    short_borrow_daily_drag,
    spread_rebalance_charges,
    spy_benchmark,
    trailing_effective_spread,
)
from trading.alphasearch.panel import PanelData
from trading.alphasearch.sort import SortResult

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


# --------------------------------------------------------------------------- #
# spread_rebalance_charges(leg="bottom"): the short leg reuses the IDENTICAL
# entry/exit half-spread construction as the long leg (spec section 2).
# --------------------------------------------------------------------------- #
def test_spread_rebalance_charges_leg_bottom_mirrors_leg_top():
    # Same fixture as test_spread_rebalance_charges_entries_and_exits_hand_
    # computed, but the SAME (A,B)->(A,C) turnover happens on the SHORT leg
    # (the `bottom` slot of the membership tuple) instead of `top`: leg=
    # "bottom" must produce byte-identical charges to what leg="top" would
    # produce on the equivalent top-shaped rebalances -- proving it is the
    # SAME machinery, not a parallel reimplementation.
    panel = _panel_with_bars(["A", "B", "C"])
    idx = panel.closes["A"].index
    bottom_rebalances = (
        (idx[3], (), ("A", "B")),
        (idx[6], (), ("A", "C")),
    )
    top_rebalances = (
        (idx[3], ("A", "B"), ()),
        (idx[6], ("A", "C"), ()),
    )
    bottom_charges, bottom_skipped = spread_rebalance_charges(
        panel, bottom_rebalances, leg="bottom"
    )
    top_charges, top_skipped = spread_rebalance_charges(panel, top_rebalances, leg="top")
    assert bottom_skipped == top_skipped == 0
    assert bottom_charges == top_charges


def test_spread_rebalance_charges_rejects_unknown_leg():
    panel = _panel_with_bars(["A", "B"])
    idx = panel.closes["A"].index
    with pytest.raises(ValueError, match="leg must be"):
        spread_rebalance_charges(panel, ((idx[0], ("A",), ("B",)),), leg="sideways")


# --------------------------------------------------------------------------- #
# short_borrow_bps: the frozen borrow model (spec section 2)
# --------------------------------------------------------------------------- #
def test_short_borrow_bps_floor_median_and_monotonic():
    assert short_borrow_bps(0.0) == pytest.approx(GC_FLOOR_BPS)
    # k is fixed so the MEDIAN shorted name (pctile ~= 0.5, since a rank
    # percentile is uniform on [0, 1]) pays ~1%/yr = 100bps.
    assert short_borrow_bps(0.5) == pytest.approx(100.0)
    lo, mid, hi = short_borrow_bps(0.1), short_borrow_bps(0.5), short_borrow_bps(0.9)
    assert lo < mid < hi   # monotonic in illiquidity


def test_short_borrow_bps_clamps_at_the_cap_for_out_of_domain_input():
    # A well-formed [0, 1] percentile never reaches BORROW_CAP_BPS under this
    # k (see costs.py docstring); the cap is a safety ceiling for an
    # out-of-domain input (a percentile computed against a corrupted
    # cross-section) -- exercised here directly since normal operation never
    # binds it.
    assert short_borrow_bps(100.0) == pytest.approx(BORROW_CAP_BPS)


def test_short_borrow_bps_nan_in_nan_out():
    assert math.isnan(short_borrow_bps(math.nan))


# --------------------------------------------------------------------------- #
# short_borrow_daily_drag / cost_charged_market_neutral: closed-form Amihud
# fixture (the same constant-geometric-return / constant-dollar-volume trick
# tests/test_amihud_ranker.py and tests/test_alphasearch_tier1.py use), so
# every percentile below is EXACT, not approximated.
# --------------------------------------------------------------------------- #
RATE = 0.01
N_BARS = 260  # >= 127 needed for the 126-valid-term amihud floor

# Four names spanning a wide, exactly-orderable amihud_lambda range, so their
# cross-sectional percentile (pandas .rank(pct=True) over 4 names) is exactly
# 1.0 / 0.75 / 0.5 / 0.25 -- see test_amihud_ranker.py's identical trick.
_GEOM_SPECS = [
    ("AAA", 1e5),   # lambda = 1e-7 (most illiquid) -> percentile 1.0
    ("BBB", 1e6),   # lambda = 1e-8 -> percentile 0.75
    ("CCC", 1e7),   # lambda = 1e-9 -> percentile 0.5 (the MEDIAN short)
    ("DDD", 1e8),   # lambda = 1e-10 (least illiquid) -> percentile 0.25
]


def _geom_bars(rate: float, dollar_volume: float, n: int = N_BARS,
               start: str = "2020-01-02") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    close = pd.Series([100.0 * (1.0 + rate) ** i for i in range(n)], index=idx)
    volume = dollar_volume / close
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": volume, "div_cash": 0.0,
         "split_factor": 1.0, "close_raw": close},
        index=idx,
    )


def _geom_panel(specs=_GEOM_SPECS, n: int = N_BARS) -> PanelData:
    bars = {name: _geom_bars(RATE, dv, n=n) for name, dv in specs}
    closes = {name: frame["close"] for name, frame in bars.items()}
    return PanelData(closes=closes, bars=bars, symbols=tuple(name for name, _ in specs))


def test_short_borrow_daily_drag_hand_computed_median_short():
    # Short leg = CCC alone (percentile exactly 0.5 by the closed-form
    # construction) -> short_borrow_bps(0.5) = 100bps/yr -> a per-day rate of
    # (100/1e4)/252 applied to every return day in the holding period.
    panel = _geom_panel()
    idx = panel.closes["AAA"].index
    formation = idx[150]  # well past the 126-valid-term amihud floor
    rebalances = ((formation, ("AAA",), ("CCC",)),)
    ret_index = idx[(idx > formation)]
    drag, skipped = short_borrow_daily_drag(panel, rebalances, ret_index)
    assert skipped == 0
    expected_daily = (short_borrow_bps(0.5) / 1e4) / 252.0
    assert drag.to_numpy() == pytest.approx(expected_daily)


def test_short_borrow_daily_drag_two_names_equal_weighted():
    # Short leg = {AAA (pctile 1.0), DDD (pctile 0.25)} -> equal-weighted
    # average of their two borrow rates, at 1/2 weight each.
    panel = _geom_panel()
    idx = panel.closes["AAA"].index
    formation = idx[150]
    rebalances = ((formation, ("BBB",), ("AAA", "DDD")),)
    ret_index = idx[(idx > formation)]
    drag, skipped = short_borrow_daily_drag(panel, rebalances, ret_index)
    assert skipped == 0
    expected = (
        (short_borrow_bps(1.0) / 1e4) / 252.0 / 2
        + (short_borrow_bps(0.25) / 1e4) / 252.0 / 2
    )
    assert drag.to_numpy() == pytest.approx(expected)


def test_short_borrow_daily_drag_skips_and_counts_unmeasurable_illiquidity():
    # A short name with too little history (< 126 valid amihud terms) has no
    # computable percentile -- skipped and counted, $0 borrow for that name,
    # never a fabricated rate.
    panel = _geom_panel(specs=[("AAA", 1e5), ("SHORTHIST", 1e6)], n=N_BARS)
    # Shrink SHORTHIST's own bars to under the 126-term floor.
    thin = panel.bars["SHORTHIST"].iloc[:100]
    panel = PanelData(
        closes={**panel.closes, "SHORTHIST": thin["close"]},
        bars={**panel.bars, "SHORTHIST": thin},
        symbols=panel.symbols,
    )
    idx = panel.closes["AAA"].index
    formation = idx[150]
    rebalances = ((formation, ("AAA",), ("SHORTHIST",)),)
    ret_index = idx[(idx > formation)]
    drag, skipped = short_borrow_daily_drag(panel, rebalances, ret_index)
    assert skipped == 1
    assert (drag == 0.0).all()


def test_short_borrow_daily_drag_is_pit_no_future_leakage():
    # A future corporate action / future illiquidity dated AFTER the
    # formation decision date must not change that holding period's borrow
    # rate -- PanelView.bars truncates strictly at the decision date.
    panel = _geom_panel()
    idx = panel.closes["AAA"].index
    formation = idx[150]
    rebalances = ((formation, ("AAA",), ("CCC",)),)
    ret_index = idx[(idx > formation) & (idx <= idx[180])]
    baseline, _ = short_borrow_daily_drag(panel, rebalances, ret_index)

    # Mutate CCC's volume (hence its amihud_lambda, hence its percentile)
    # only on bars STRICTLY AFTER the formation date -- a "future illiquidity
    # change" that must be invisible to a borrow rate fixed at formation.
    mutated_ccc = panel.bars["CCC"].copy()
    future_mask = mutated_ccc.index > formation
    mutated_ccc.loc[future_mask, "volume"] *= 1000.0  # much more liquid, later
    mutated_panel = PanelData(
        closes={**panel.closes, "CCC": mutated_ccc["close"]},
        bars={**panel.bars, "CCC": mutated_ccc},
        symbols=panel.symbols,
    )
    mutated, _ = short_borrow_daily_drag(mutated_panel, rebalances, ret_index)
    pd.testing.assert_series_equal(baseline, mutated)

    # Sanity: the SAME mutation applied BEFORE formation (visible at the PIT
    # cut) DOES change the rate -- proving this test is not vacuously
    # insensitive to the mutation.
    mutated_ccc_past = panel.bars["CCC"].copy()
    past_mask = mutated_ccc_past.index <= formation
    mutated_ccc_past.loc[past_mask, "volume"] *= 1000.0
    mutated_past_panel = PanelData(
        closes={**panel.closes, "CCC": mutated_ccc_past["close"]},
        bars={**panel.bars, "CCC": mutated_ccc_past},
        symbols=panel.symbols,
    )
    mutated_past, _ = short_borrow_daily_drag(mutated_past_panel, rebalances, ret_index)
    assert not mutated_past.equals(baseline)


def test_cost_charged_market_neutral_hand_computed():
    # top=AAA (pctile 1.0, irrelevant to borrow), bottom=CCC (pctile 0.5 ->
    # borrow 100bps/yr): net = gross - (long spread + short spread charges,
    # via the SAME spread_rebalance_charges machinery cost_charged_lo uses)
    # - (daily short-borrow accrual).
    panel = _geom_panel()
    idx = panel.closes["AAA"].index
    formation = idx[150]
    rebalances = ((formation, ("AAA",), ("CCC",)),)
    ret_index = idx[(idx > formation) & (idx <= idx[170])]
    # A deliberately non-degenerate gross series (not all-zero) so gross/net
    # Sharpe are both finite and the decomposition is a real check, not a
    # 0 - 0 = 0 tautology.
    gross = pd.Series(
        [0.001 if i % 2 == 0 else -0.0006 for i in range(len(ret_index))],
        index=ret_index,
    )
    sort_result = SortResult(
        ls=gross, lo=gross, turnover_monthly=0.0, skipped_dates=(),
        n_dates=1, n_names_median=2.0, rebalances=rebalances,
    )
    charged, diagnostics = cost_charged_market_neutral(panel, sort_result)

    long_charges, skipped_long = spread_rebalance_charges(panel, rebalances, leg="top")
    short_charges, skipped_short = spread_rebalance_charges(panel, rebalances, leg="bottom")
    borrow_drag, skipped_borrow = short_borrow_daily_drag(panel, rebalances, gross.index)
    expected = apply_rebalance_charges(gross, long_charges)
    expected = apply_rebalance_charges(expected, short_charges)
    expected = expected - borrow_drag
    pd.testing.assert_series_equal(charged, expected)

    assert skipped_long == skipped_short == skipped_borrow == 0
    assert diagnostics["skipped_no_spread_long"] == 0
    assert diagnostics["skipped_no_spread_short"] == 0
    assert diagnostics["skipped_no_illiquidity"] == 0
    # The charged series is strictly worse than gross wherever a charge or
    # borrow accrual landed -- costs never invent a tailwind.
    assert (charged <= gross + 1e-15).all()
    assert charged.iloc[0] < gross.iloc[0]     # the rebalance charge landed here
    # Borrow rate hand-check: CCC's exact closed-form percentile is 0.5 (four
    # names, lambda ordering AAA>BBB>CCC>DDD by construction), and every day
    # in the window carries the SAME rate, so the annualized diagnostic
    # exactly recovers short_borrow_bps(0.5) = 100bps/yr.
    assert diagnostics["borrow_drag_bps"] == pytest.approx(short_borrow_bps(0.5), rel=1e-9)
    assert diagnostics["gross_total_return"] == pytest.approx(
        float((1.0 + gross).prod() - 1.0)
    )
    assert diagnostics["net_total_return"] == pytest.approx(
        float((1.0 + charged).prod() - 1.0)
    )
    assert diagnostics["net_sharpe"] != diagnostics["gross_sharpe"]
