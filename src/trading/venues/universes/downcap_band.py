"""Raw-price PIT market cap + the three-screen down-cap band selector (R3 spec
section 2). Dynamic per decision date D: a candidate is IN the band iff its
raw-price cap is in [$50M, $2B] AND its trailing-63 Corwin-Schultz spread is
<= 2% AND its trailing-63 median dollar-volume is >= $50k/day.

Correctness (frozen): the cap uses close_raw (the actual share price), NEVER
the split/dividend-adjusted close -- an adjusted close misstates the absolute
cap (the div_yield look-ahead class). The tradeability screens reuse the
existing adjusted OHLCV (split-consistent by construction), matching the R1
cost machinery and the spec.py liquidity signals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from trading.alphasearch.costs import SPREAD_CAP, SPREAD_FLOOR, trailing_effective_spread

BAND_LO = 50_000_000.0
SMALL_LO = 300_000_000.0
BAND_HI = 2_000_000_000.0
SPREAD_CAP_PCT = 0.02          # spec: 2.0% CS effective-spread screen
DV_FLOOR = 50_000.0            # spec: $50k/day median dollar-volume floor
DOWNCAP_TRAILING_WINDOW = 63   # spec: trailing-63-session for BOTH tradeability screens


def market_cap_raw(shares: float, close_raw: float) -> float:
    """shares_outstanding x RAW close (spec section 2, item 1)."""
    return shares * close_raw


def band_of(market_cap: float) -> str | None:
    """The cap bucket: "micro" ($50M-$300M), "small" ($300M-$2B), or None
    (outside [$50M, $2B]). Boundaries: lower inclusive, $300M is the
    micro/small split (>= $300M is small), $2B inclusive upper."""
    if math.isnan(market_cap) or market_cap < BAND_LO or market_cap > BAND_HI:
        return None
    return "small" if market_cap >= SMALL_LO else "micro"


def downcap_effective_spread(bars: pd.DataFrame) -> float:
    """The trailing-63-session CS effective spread as-of the last row, floored
    at SPREAD_FLOOR and capped at SPREAD_CAP -- reuses the R1 estimator at the
    63-session window the R3 spec names (R1's own `effective_spread` helper is
    21-session). NaN when fewer than 2 bars (no two-day pair)."""
    if len(bars) < 2:
        return math.nan
    trailing = trailing_effective_spread(bars, window=DOWNCAP_TRAILING_WINDOW)
    value = float(trailing.iloc[-1])
    if math.isnan(value):
        return math.nan
    return min(max(value, SPREAD_FLOOR), SPREAD_CAP)


def median_dollar_volume(bars: pd.DataFrame, window: int = DOWNCAP_TRAILING_WINDOW) -> float:
    """Trailing-`window` median of close x volume (adjusted, split-consistent;
    matches spec.py's liquidity signals). NaN on an empty frame."""
    if bars.empty:
        return math.nan
    dollar = (bars["close"] * bars["volume"]).tail(window)
    return float(dollar.median())


@dataclass(frozen=True)
class BandEval:
    """One candidate's as-of-D band decision plus the audit fields the Phase-A
    gate consumes. `band` is non-None ONLY when has_shares AND tradeable AND
    the raw-price cap falls in [$50M, $2B]."""
    band: str | None
    has_shares: bool
    tradeable: bool
    market_cap: float
    spread: float
    dollar_volume: float


def evaluate_band(
    bars: pd.DataFrame,
    shares: float,
    *,
    spread_cap: float = SPREAD_CAP_PCT,
    dv_floor: float = DV_FLOOR,
) -> BandEval:
    """Evaluate a candidate at the as-of bar (last row of `bars`, already
    truncated to <= D by the caller). Fail-closed on missing shares: no cap,
    no band -- but the tradeability screens are still computed so the Phase-A
    shares-coverage denominator (tradeable candidate-months) is honest."""
    spread = downcap_effective_spread(bars)
    dollar_volume = median_dollar_volume(bars)
    tradeable = (
        not math.isnan(spread) and spread <= spread_cap
        and not math.isnan(dollar_volume) and dollar_volume >= dv_floor
    )
    has_shares = not math.isnan(shares) and shares > 0
    if not has_shares:
        return BandEval(None, False, tradeable, math.nan, spread, dollar_volume)
    if bars.empty:
        market_cap = math.nan
    else:
        close_raw = float(bars["close_raw"].iloc[-1])
        market_cap = market_cap_raw(shares, close_raw) if close_raw > 0 else math.nan
    cap_band = band_of(market_cap)
    band = cap_band if (tradeable and cap_band is not None) else None
    return BandEval(band, True, tradeable, market_cap, spread, dollar_volume)
