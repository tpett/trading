"""Spread-based transaction costs for the long-only gate (R1 amendment spec
section 3): the Corwin-Schultz (2012) high-low effective-spread estimator,
per-name rebalance charges built from it, and the SPY buy-and-hold benchmark
the amended promotion rule compares against.

Leaf module by design: depends only on panel.py/sort.py types and evaluate.py
(for annualized_sharpe), never on sweep.py or robustness.py, so both of those
may import from here without a circular import.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading.alphasearch.evaluate import annualized_sharpe
from trading.alphasearch.panel import PanelData, load_closes
from trading.alphasearch.sort import Membership

CS_TRAILING_WINDOW = 21          # sessions (spec section 3)
SPREAD_FLOOR = 0.0002            # 2 bps -- large-cap reality
SPREAD_CAP = 0.05                # 5% -- data-sanity ceiling
_CS_K = 3 - 2 * math.sqrt(2)     # the CS(2012) denominator constant

SPY_SYMBOL = "SPY"
DEFAULT_SPY_CACHE_DIR = Path("data") / "equities-tiingo"


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
    panel: PanelData, rebalances: tuple[Membership, ...]
) -> tuple[list[tuple[pd.Timestamp, float]], int]:
    """Per-rebalance half-spread charge on the TOP (long-only) leg (spec
    section 3): each name ENTERING the leg is charged half its effective
    spread at its new 1/n weight; each name EXITING is charged the analogue
    at the OLD leg size (the position actually liquidated) -- mirrors
    robustness.capacity_curve's entry/exit shape with a per-name spread
    instead of an Amihud-lambda x book impact. NaN-spread names (fewer than
    2 bars as of the decision date) are skipped and counted, never
    fabricated. Returns (charges, skipped_no_spread)."""
    charges: list[tuple[pd.Timestamp, float]] = []
    skipped = 0
    prev_top: tuple[str, ...] = ()
    for date, top, _bottom in rebalances:
        view = panel.view(date)
        charge = 0.0
        cur_set, prev_set = set(top), set(prev_top)
        for sym in sorted(cur_set - prev_set):
            spread = effective_spread(view.bars(sym))
            if math.isnan(spread):
                skipped += 1
                continue
            charge += (spread / 2.0) / len(top)
        for sym in sorted(prev_set - cur_set):
            spread = effective_spread(view.bars(sym))
            if math.isnan(spread):
                skipped += 1
                continue
            charge += (spread / 2.0) / len(prev_top)
        charges.append((date, charge))
        prev_top = top
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
