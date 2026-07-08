"""SignalSpec + the seed SIGNALS registry (spec section 3.1).

A signal is a pure function (PanelView, as_of) -> per-symbol float score,
HIGHER = more attractive to be long. The sign convention is part of the
wrapper and is documented at each registration -- e.g. rev5 registers the
NEGATED trailing 5-day return because short-term reversal makes recent
losers attractive. NaN means "this symbol lacks the signal's inputs on this
date": the sort drops it from that date's cross-section (spec section 5.5),
never imputes.

PIT contract: fn receives a PanelView (never the raw PanelData), whose
accessors truncate at as_of -- forward data is structurally unreachable.
Price-metric semantics (trail / rvol / disthigh) are copied from
scripts/signal_scan.py so alphasearch scores agree with the scanner's panel.

Task 4 registers the price family; Task 5 completes the registry with the
options family (vrp, hedge, excite, atm_iv, smile, atm_spread) and the
fundamentals family (gross_profitability, earnings_yield, book_to_market).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from trading.alphasearch.panel import PanelView

SignalFn = Callable[[PanelView, pd.Timestamp], pd.Series]


@dataclass(frozen=True)
class SignalSpec:
    name: str
    fn: SignalFn
    requires_options: bool = False
    requires_fundamentals: bool = False


SIGNALS: dict[str, SignalSpec] = {}


def _register(
    name: str,
    fn: SignalFn,
    *,
    requires_options: bool = False,
    requires_fundamentals: bool = False,
) -> None:
    SIGNALS[name] = SignalSpec(name, fn, requires_options, requires_fundamentals)


# --------------------------------------------------------------------------- #
# Price metrics -- semantics identical to scripts/signal_scan.py, but applied
# to an already-truncated close series (PanelView.closes), so the "position
# at or before the decision date" is simply the last element.
# --------------------------------------------------------------------------- #
def _trail(closes: pd.Series, window: int) -> float:
    """Trailing `window`-bar return ending at the last close."""
    p = len(closes) - 1
    if p - window < 0:
        return math.nan
    return float(closes.iloc[p] / closes.iloc[p - window] - 1)


def _rvol(closes: pd.Series, window: int = 21) -> float:
    """Annualized realized vol of the `window` bars BEFORE the last close."""
    p = len(closes) - 1
    if p < window:
        return math.nan
    return float(closes.iloc[p - window : p].pct_change().std() * math.sqrt(252))


def _disthigh(closes: pd.Series, window: int = 252) -> float:
    """Distance from the trailing `window`-bar high (<= 0; 0 = at the high)."""
    p = len(closes) - 1
    if p < 0:
        return math.nan
    return float(closes.iloc[p] / closes.iloc[max(0, p - window) : p + 1].max() - 1)


def _price_signal(metric: Callable[[pd.Series], float]) -> SignalFn:
    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores = {symbol: metric(view.closes(symbol)) for symbol in view.symbols}
        return pd.Series(scores, dtype="float64")

    return fn


def _mom(window: int) -> SignalFn:
    def metric(closes: pd.Series, _window: int = window) -> float:
        return _trail(closes, _window)

    return _price_signal(metric)


# Price family. Direction rationale recorded per registration (spec 3.1).
_register("mom21", _mom(21))  # momentum: recent winners attractive
_register("mom63", _mom(63))
_register("mom126", _mom(126))
_register("mom252", _mom(252))
# Short-term reversal: recent LOSERS attractive -> negate the trailing return.
_register("rev5", _price_signal(lambda c: -_trail(c, 5)))
# Low-vol anomaly: quiet names attractive -> negate realized vol.
_register("rvol21", _price_signal(lambda c: -_rvol(c)))
# Proximity to the 52-week high is momentum-like; disthigh is <= 0 with 0 at
# the high, so raw sign already puts near-high names on top.
_register("disthigh", _price_signal(_disthigh))


# --------------------------------------------------------------------------- #
# Options family (requires_options). Column values come straight from
# panel.cell_metrics; each registration bakes in the higher-=-attractive sign.
# --------------------------------------------------------------------------- #
def _option_signal(column: str, sign: float) -> SignalFn:
    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol in view.symbols:
            row = view.option_row(symbol)
            scores[symbol] = math.nan if row is None else sign * float(row[column])
        return pd.Series(scores, dtype="float64")

    return fn


def _vrp(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    """Volatility risk premium: ATM implied vol minus trailing realized vol
    (exactly signal_scan's vrp = atm_iv - rvol21)."""
    scores: dict[str, float] = {}
    for symbol in view.symbols:
        row = view.option_row(symbol)
        if row is None:
            scores[symbol] = math.nan
            continue
        scores[symbol] = float(row["atm_iv"]) - _rvol(view.closes(symbol))
    return pd.Series(scores, dtype="float64")


# Rich vol premium = overpaid insurance on the name -> attractive to own.
_register("vrp", _vrp, requires_options=True)
# Spec 3.1's example: the skew signal registers NEGATED (-skew_put_atm), so
# LESS downside-hedged names score higher -- the risk-veto intuition.
_register("hedge", _option_signal("hedge", -1.0), requires_options=True)
# cell excite is already -skew_put_call (call-side richness): keep raw sign.
_register("excite", _option_signal("excite", +1.0), requires_options=True)
# Lottery/vol-premium effect: high-IV names underperform -> negate.
_register("atm_iv", _option_signal("atm_iv", -1.0), requires_options=True)
# Convex smile = tail fear priced in -> negate.
_register("smile", _option_signal("smile", -1.0), requires_options=True)
# Option-illiquidity premium: WIDER spread predicted higher returns in the
# mid-cap scan (docs/experiments.md section 9) -> keep raw sign.
_register("atm_spread", _option_signal("atm_spread", +1.0), requires_options=True)


# --------------------------------------------------------------------------- #
# Fundamentals family (requires_fundamentals). Ratios computed at scoring time
# from price-free stored primitives x the as-of close, exactly as
# trading.signals.quality / trading.signals.value do. NaN (dropped) when no
# filing is visible -- the sort's missing-data rule handles it; there is no
# 0.5-neutral here because a cross-sectional SORT needs a real value.
# --------------------------------------------------------------------------- #
def _fundamental_field(view: PanelView, symbol: str, key: str) -> float:
    row = view.fundamentals_row(symbol)
    if row is None or key not in row.index:
        return math.nan
    return float(row[key])


def _gross_profitability(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    scores = {s: _fundamental_field(view, s, "gross_profitability") for s in view.symbols}
    return pd.Series(scores, dtype="float64")


def _value_ratio(numerator: str) -> SignalFn:
    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol in view.symbols:
            shares = _fundamental_field(view, symbol, "shares_outstanding")
            value = _fundamental_field(view, symbol, numerator)
            close = view.last_close(symbol)
            score = math.nan
            if not math.isnan(shares) and shares > 0 and not math.isnan(close):
                market_cap = shares * close
                if market_cap > 0 and not math.isnan(value):
                    # Negative income/equity is a real (bottom-ranked) value,
                    # not missing data -- same policy as trading.signals.value.
                    score = value / market_cap
            scores[symbol] = score
        return pd.Series(scores, dtype="float64")

    return fn


_register("gross_profitability", _gross_profitability, requires_fundamentals=True)
_register("earnings_yield", _value_ratio("ttm_net_income"), requires_fundamentals=True)
_register("book_to_market", _value_ratio("book_equity"), requires_fundamentals=True)
