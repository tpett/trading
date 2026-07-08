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
