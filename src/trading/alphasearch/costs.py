"""Spread-based transaction costs for the long-only gate (R1 amendment spec
section 3): the Corwin-Schultz (2012) high-low effective-spread estimator,
per-name rebalance charges built from it, and the SPY buy-and-hold benchmark
the amended promotion rule compares against.

Also the market-neutral (long/short) gate's both-legs cost charging (R6
Stage 1 amendment, docs/superpowers/specs/2026-07-11-market-neutral-gate-
amendment.md section 2): ``cost_charged_market_neutral`` REUSES this exact
Corwin-Schultz + rebalance-charge machinery for BOTH legs (via
``spread_rebalance_charges``'s ``leg`` parameter) and adds a frozen
short-borrow accrual (``short_borrow_bps``) on the short leg. Additive,
default-off: every existing call site (``cost_charged_lo``) keeps its
original positional signature and default ``leg="top"`` behavior, bit-
identical.

Leaf module by design: depends only on panel.py/sort.py/spec.py types and
evaluate.py (for annualized_sharpe/total_return/TRADING_DAYS), never on
sweep.py or robustness.py, so both of those may import from here without a
circular import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading.alphasearch.evaluate import TRADING_DAYS, annualized_sharpe, total_return
from trading.alphasearch.panel import PanelData, load_closes
from trading.alphasearch.sort import Membership, SortResult
from trading.alphasearch.spec import amihud_lambda

CS_TRAILING_WINDOW = 21          # sessions (spec section 3)
SPREAD_FLOOR = 0.0002            # 2 bps -- large-cap reality
SPREAD_CAP = 0.05                # 5% -- data-sanity ceiling
_CS_K = 3 - 2 * math.sqrt(2)     # the CS(2012) denominator constant

SPY_SYMBOL = "SPY"
DEFAULT_SPY_CACHE_DIR = Path("data") / "equities-tiingo"

# --------------------------------------------------------------------------- #
# Market-neutral gate frozen short-borrow model (spec section 2)
# --------------------------------------------------------------------------- #
GC_FLOOR_BPS = 50.0     # 0.5%/yr general-collateral floor, ALL shorts pay this
BORROW_CAP_BPS = 1500.0  # 15%/yr cap -- a safety ceiling, see short_borrow_bps
# illiquidity_pctile is a [0, 1] cross-sectional rank percentile (pandas
# .rank(pct=True), uniform on [0, 1] by construction), so the MEDIAN shorted
# name sees pctile ~= 0.5. k=100 makes that median name pay GC_FLOOR_BPS +
# 100*0.5 = 100bps/yr -- the frozen spec's "median shorted name pays ~1%/yr"
# target. See short_borrow_bps's docstring for why BORROW_CAP_BPS is not
# meant to bind under this k.
_BORROW_K = 100.0


# --------------------------------------------------------------------------- #
# Corwin-Schultz (2012) high-low effective-spread estimator
# --------------------------------------------------------------------------- #
def _beta_gamma(bars: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """The CS(2012) two-day components, indexed like `bars`.

    beta_t  = ln(H_t/L_t)^2 + ln(H_{t-1}/L_{t-1})^2   (sum over the pair)
    gamma_t = ln(max(H_t,H_{t-1}) / min(L_t,L_{t-1}))^2

    Row 0 (no predecessor) is NaN in both.
    """
    high, low = bars["high"], bars["low"]
    log_hl_sq = np.log(high / low) ** 2
    beta = log_hl_sq + log_hl_sq.shift(1)
    two_day_high = high.rolling(2).max()
    two_day_low = low.rolling(2).min()
    gamma = np.log(two_day_high / two_day_low) ** 2
    return beta, gamma


def trailing_effective_spread(
    bars: pd.DataFrame, window: int = CS_TRAILING_WINDOW
) -> pd.Series:
    """The CS(2012) effective spread (fraction of price), indexed like `bars`.

    Averages the beta/gamma TWO-DAY COMPONENTS over the trailing `window`
    sessions FIRST, then runs the (alpha -> spread) transform ONCE per row on
    those averaged components. This is a documented BIAS CORRECTION and an
    orchestrator-RATIFIED deviation (2026-07-10) from the amendment spec
    section 3's literal per-day text -- NOT the CS 2012 paper's own
    convention, whose baseline computes alpha/spread per two-day pair and
    averages those spreads (with negative-handling variants).

    Why the deviation: at daily granularity the per-pair route's
    negatives-floored-at-0 average is a one-sided truncation of noise that
    inflates the mean by an order of magnitude. A Monte Carlo check (GBM
    tick price with a KNOWN zero spread, one-minute ticks, 4000 simulated
    days) quantifies it: the raw (unfloored) per-day alpha averages to ~0 as
    expected, but flooring each day's negative estimate at 0 BEFORE
    averaging drags the mean to ~75bps out of a true 0bps spread.
    Component-averaging first (this function) cuts the same zero-spread mean
    bias to ~12bps -- a residual, CONSERVATIVE (cost-overstating)
    window-level truncation bias, mostly absorbed by the 2bps floor for
    liquid names -- and it alone satisfies the spec's own AAPL acceptance
    criterion (single-digit bps, test_alphasearch_costs.py);
    per-pair-then-average lands AAPL at 30-90bps against its real few-bps
    effective spread.

    Caveat: no overnight-gap adjustment (CS 2012's open-vs-prior-close
    correction is omitted) -- mildly ANTI-conservative for gappy names,
    immaterial under the 2bps floor at this universe's liquidity.

    Negative results (beta_bar/gamma_bar dominated by market-wide vol, not
    spread) are floored at 0; the caller applies the final [SPREAD_FLOOR,
    SPREAD_CAP] clamp.
    """
    beta, gamma = _beta_gamma(bars)
    beta_bar = beta.rolling(window, min_periods=1).mean()
    gamma_bar = gamma.rolling(window, min_periods=1).mean()
    alpha = (
        (np.sqrt(2 * beta_bar) - np.sqrt(beta_bar)) / _CS_K
        - np.sqrt(gamma_bar / _CS_K)
    )
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0.0)


def effective_spread(bars: pd.DataFrame) -> float:
    """The trailing effective spread AS OF the last row of `bars` (callers
    pass an as-of-truncated frame, e.g. PanelView.bars(symbol)), floored at
    SPREAD_FLOOR and capped at SPREAD_CAP (spec section 3). NaN when fewer
    than 2 bars are available -- no two-day pair to estimate from."""
    if len(bars) < 2:
        return math.nan
    trailing = trailing_effective_spread(bars)
    value = float(trailing.iloc[-1])
    if math.isnan(value):
        return math.nan
    return min(max(value, SPREAD_FLOOR), SPREAD_CAP)


# --------------------------------------------------------------------------- #
# Rebalance charges: generic application + the spread-based cost model
# --------------------------------------------------------------------------- #
def apply_rebalance_charges(
    series: pd.Series, charges: list[tuple[pd.Timestamp, float]]
) -> pd.Series:
    """Deduct each charge from the first daily return strictly after its
    rebalance date (the first day the traded book exists). A charge dated at
    or after the last return day has no day to land on and is dropped (a
    final-day rebalance is never held). Returns a copy.

    Generic shape shared by the L/S cost/capacity tables (robustness.py) and
    the long-only spread charging below -- moved here (a leaf module) so both
    robustness.py and sweep.py can call it without a circular import.
    """
    charged = series.copy()
    for date, charge in charges:
        pos = int(charged.index.searchsorted(date, side="right"))
        if pos < len(charged):
            charged.iloc[pos] -= charge
    return charged


def spread_rebalance_charges(
    panel: PanelData, rebalances: tuple[Membership, ...], *, leg: str = "top"
) -> tuple[list[tuple[pd.Timestamp, float]], int]:
    """Per-rebalance half-spread charge on one leg (spec section 3): each
    name ENTERING the leg is charged half its effective spread at its new
    1/n weight; each name EXITING is charged the analogue at the OLD leg
    size (the position actually liquidated) -- mirrors
    robustness.capacity_curve's entry/exit shape with a per-name spread
    instead of an Amihud-lambda x book impact. NaN-spread names (fewer than
    2 bars as of the decision date) are skipped and counted, never
    fabricated. Returns (charges, skipped_no_spread).

    leg="top" (default): the LONG leg -- bit-identical to this function's
    pre-market-neutral behavior (R1's cost_charged_lo). leg="bottom": the
    IDENTICAL construction applied to the SHORT leg instead -- the market-
    neutral gate's both-legs cost charging (spec section 2,
    cost_charged_market_neutral) reuses this exact machinery rather than
    reimplementing it, so covering the short leg's spread is a leg=
    "bottom" call, never a second Corwin-Schultz implementation.
    """
    if leg not in ("top", "bottom"):
        raise ValueError(f"leg must be 'top' or 'bottom', got {leg!r}")
    charges: list[tuple[pd.Timestamp, float]] = []
    skipped = 0
    prev_members: tuple[str, ...] = ()
    for date, top, bottom in rebalances:
        members = top if leg == "top" else bottom
        view = panel.view(date)
        charge = 0.0
        cur_set, prev_set = set(members), set(prev_members)
        for sym in sorted(cur_set - prev_set):
            spread = effective_spread(view.bars(sym))
            if math.isnan(spread):
                skipped += 1
                continue
            charge += (spread / 2.0) / len(members)
        for sym in sorted(prev_set - cur_set):
            spread = effective_spread(view.bars(sym))
            if math.isnan(spread):
                skipped += 1
                continue
            charge += (spread / 2.0) / len(prev_members)
        charges.append((date, charge))
        prev_members = members
    return charges, skipped


def cost_charged_lo(
    panel: PanelData, lo: pd.Series, rebalances: tuple[Membership, ...]
) -> tuple[pd.Series, int]:
    """The long-only series with spread-based costs charged at each rebalance
    (spec section 3): reuses apply_rebalance_charges' shape with per-name
    spreads instead of flat bps. Returns (charged series, skipped_no_spread
    count)."""
    charges, skipped = spread_rebalance_charges(panel, rebalances)
    return apply_rebalance_charges(lo, charges), skipped


# --------------------------------------------------------------------------- #
# Market-neutral gate: short-borrow model + both-legs cost charging
# (R6 Stage 1 amendment, docs/superpowers/specs/2026-07-11-market-neutral-
# gate-amendment.md sections 2 and 6)
# --------------------------------------------------------------------------- #
def short_borrow_bps(illiquidity_pctile: float) -> float:
    """The frozen short-borrow model (spec section 2): a general-collateral
    floor of GC_FLOOR_BPS (50bps/yr = 0.5%/yr) that EVERY short pays, scaled
    up by the shorted name's illiquidity percentile -- a [0, 1] cross-
    sectional rank (e.g. pandas ``.rank(pct=True)`` over amihud_lambda, the
    repo-wide percentile convention -- see trading.signals.illiquidity.
    amihud_v1) -- clamped to [GC_FLOOR_BPS, BORROW_CAP_BPS]. Returns
    annualized bps.

    k = _BORROW_K = 100: illiquidity_pctile is uniform on [0, 1] by
    construction, so the MEDIAN shorted name sees pctile ~= 0.5 -> borrow =
    50 + 100*0.5 = 100bps/yr, the frozen spec's "median shorted name pays
    ~1%/yr" target. Monotonic in the percentile by construction (a straight
    line): the most illiquid name in a well-formed [0, 1] cross-section
    (pctile=1.0) pays only 150bps/yr under this k -- BORROW_CAP_BPS
    (1500bps = 15%/yr) is a safety ceiling for OUT-OF-DOMAIN inputs (a
    percentile computed against a corrupted or non-[0, 1] cross-section),
    not something this linear model reaches for a genuine percentile.

    NaN in -> NaN out: an unmeasurable illiquidity (a shorted name with no
    computable amihud_lambda) charges no fabricated rate -- the caller
    (short_borrow_daily_drag) skips and counts it instead, mirroring
    effective_spread's NaN convention.
    """
    if math.isnan(illiquidity_pctile):
        return math.nan
    raw = GC_FLOOR_BPS + _BORROW_K * illiquidity_pctile
    return min(max(raw, GC_FLOOR_BPS), BORROW_CAP_BPS)


def _amihud_percentiles(panel: PanelData, date: pd.Timestamp) -> pd.Series:
    """Cross-sectional percentile (pandas ``.rank(pct=True)``) of
    amihud_lambda across every symbol visible in the panel as of `date`,
    computed PIT from bars truncated at `date` (PanelView.bars -- no
    look-ahead: nothing dated after `date` is ever visible to amihud_lambda).
    Symbols with no computable lambda (fewer than 126 valid trailing terms,
    see spec.amihud_lambda) are excluded from the ranked cross-section
    entirely, never fabricated as a neutral 0.5 -- short_borrow_bps then
    sees a real NaN for those names and the caller counts the skip. The
    percentile is over the FULL as-of universe (panel.view(date).symbols),
    not just the currently-shorted names or a battery symbol_subset draw:
    a real borrow desk prices hard-to-borrow against float-wide liquidity,
    not a research draw's subset.
    """
    view = panel.view(date)
    lam = {
        sym: value
        for sym in view.symbols
        if not math.isnan(value := amihud_lambda(view.bars(sym)))
    }
    return pd.Series(lam, dtype="float64").rank(pct=True)


def short_borrow_daily_drag(
    panel: PanelData, rebalances: tuple[Membership, ...], index: pd.Index
) -> tuple[pd.Series, int]:
    """Daily short-borrow return drag (spec section 2), aligned to `index`
    (the cost-charged series' own calendar): a positive fraction, subtracted
    from that day's return, equal to the equal-weighted sum over the
    currently-held short leg of each name's short_borrow_bps annualized rate
    divided by TRADING_DAYS.

    Unlike the one-time spread charge (spread_rebalance_charges), borrow is
    a CONTINUOUS carrying cost over the whole holding period: every day in a
    holding period gets the SAME per-day rate, fixed by the illiquidity
    percentile AS OF the decision date that FORMED the holding (PIT --
    _amihud_percentiles truncates at that date) and held constant until the
    next rebalance -- mirrors the sort's own hold-to-next-rebalance
    convention for WHICH names are held (sort.py module docstring). A
    shorted name whose illiquidity percentile can't be computed at formation
    is skipped and counted, contributing $0 borrow for that name -- never a
    fabricated rate. Returns (drag series indexed like `index`, skipped
    count).
    """
    drag = pd.Series(0.0, index=index)
    skipped = 0
    if len(index) == 0:
        return drag, skipped
    for i, (date, _top, bottom) in enumerate(rebalances):
        if not bottom:
            continue
        hold_end = rebalances[i + 1][0] if i + 1 < len(rebalances) else index[-1]
        mask = (index > date) & (index <= hold_end)
        if not mask.any():
            continue
        pctiles = _amihud_percentiles(panel, date)
        n = len(bottom)
        daily_rate = 0.0
        for sym in bottom:
            bps = short_borrow_bps(float(pctiles.get(sym, math.nan)))
            if math.isnan(bps):
                skipped += 1
                continue
            daily_rate += (bps / 1e4) / TRADING_DAYS / n
        drag.loc[mask] += daily_rate
    return drag, skipped


def cost_charged_market_neutral(
    panel: PanelData, sort_result: SortResult
) -> tuple[pd.Series, dict]:
    """The market-neutral (ls) daily series with BOTH legs' spread costs
    (spec section 2: the R1 half-spread construction -- spread_rebalance_
    charges -- applied identically to the long AND the short leg via its
    `leg` parameter, never reimplemented) PLUS a daily short-borrow accrual
    on the short leg (short_borrow_daily_drag). Spread charges land on the
    first return day strictly after each decision date, the identical
    apply_rebalance_charges convention cost_charged_lo uses; borrow accrues
    every day of the holding period. Returns (net daily series, diagnostics
    dict: gross vs net Sharpe/total return, spread drag per leg, borrow
    drag, and the honest skip counts).
    """
    long_charges, skipped_long = spread_rebalance_charges(
        panel, sort_result.rebalances, leg="top"
    )
    short_charges, skipped_short = spread_rebalance_charges(
        panel, sort_result.rebalances, leg="bottom"
    )
    charged = apply_rebalance_charges(sort_result.ls, long_charges)
    charged = apply_rebalance_charges(charged, short_charges)
    borrow_drag, skipped_illiquidity = short_borrow_daily_drag(
        panel, sort_result.rebalances, charged.index
    )
    charged = charged - borrow_drag

    n_years = len(charged) / TRADING_DAYS if len(charged) else math.nan

    def _annualized_bps(total_fraction: float) -> float:
        return (total_fraction / n_years * 1e4) if n_years and n_years > 0 else math.nan

    long_spread_bps = _annualized_bps(sum(c for _d, c in long_charges))
    short_spread_bps = _annualized_bps(sum(c for _d, c in short_charges))
    borrow_bps = _annualized_bps(float(borrow_drag.sum()))
    spread_bps_total = (
        math.nan
        if math.isnan(long_spread_bps) or math.isnan(short_spread_bps)
        else long_spread_bps + short_spread_bps
    )
    diagnostics = {
        "gross_sharpe": annualized_sharpe(sort_result.ls),
        "net_sharpe": annualized_sharpe(charged),
        "gross_total_return": total_return(sort_result.ls),
        "net_total_return": total_return(charged),
        "long_spread_drag_bps": long_spread_bps,
        "short_spread_drag_bps": short_spread_bps,
        "spread_drag_bps_total": spread_bps_total,
        "borrow_drag_bps": borrow_bps,
        "skipped_no_spread_long": skipped_long,
        "skipped_no_spread_short": skipped_short,
        "skipped_no_illiquidity": skipped_illiquidity,
    }
    return charged, diagnostics


# --------------------------------------------------------------------------- #
# SPY buy-and-hold benchmark (spec section 2's promotion comparator)
# --------------------------------------------------------------------------- #
def load_spy_closes(cache_dir: Path) -> pd.Series | None:
    """SPY's adjusted close series from `cache_dir`, or None when the cache
    holds no SPY parquet -- callers must refuse loudly (spec: "do NOT
    silently substitute") rather than fall back to another benchmark."""
    closes = load_closes(cache_dir, [SPY_SYMBOL])
    return closes.get(SPY_SYMBOL)


@dataclass(frozen=True)
class BenchmarkResult:
    sharpe_annual: float
    total_return: float
    n_obs: int


def spy_benchmark(
    spy_closes: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> BenchmarkResult:
    """SPY buy-and-hold Sharpe + total return over [start, end] (spec section
    2's promotion comparator). NaN/0 fields when fewer than 2 closes fall in
    the window."""
    window = spy_closes.loc[(spy_closes.index >= start) & (spy_closes.index <= end)]
    rets = window.pct_change().dropna()
    total = (
        float(window.iloc[-1] / window.iloc[0] - 1.0) if len(window) >= 2 else math.nan
    )
    return BenchmarkResult(
        sharpe_annual=annualized_sharpe(rets), total_return=total, n_obs=len(rets)
    )
