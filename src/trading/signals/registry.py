"""Ranker registry (spec: pluggable ranking strategies), contract v2.

A registered ranker is a RankerSpec whose fn matches:

    fn(bars: dict[str, pd.DataFrame], as_of: pd.Timestamp,
       config: SignalConfig, fundamentals: dict[str, pd.DataFrame] | None
       ) -> pd.DataFrame

Input contract (what a ranker receives):

- `bars` maps symbol -> OHLCV DataFrame with columns exactly
  [open, high, low, close, volume], indexed by a sorted tz-aware UTC
  DatetimeIndex normalized to the bar's date (the shape enforced by
  trading.venues.base.validate_ohlcv). Frames may extend PAST as_of: the
  caller does not pre-cut them.
- `as_of` is a tz-aware UTC pd.Timestamp; momentum_v1 rejects a naive one.
- `fundamentals` maps symbol -> per-symbol fundamentals frame indexed by
  tz-aware UTC FILING dates (trading.fundamentals.store schema: price-free
  primitives such as "gross_profitability", "ttm_net_income", "book_equity",
  "shares_outstanding" -- a ranker reads only the columns it needs).
  Price-dependent ratios are computed INSIDE the ranker from these
  primitives plus bars. None (or a missing symbol) means "no fundamentals
  known" -- a ranker must treat that as neutral, never crash. Like bars,
  frames may extend past as_of; cutting to rows FILED at or before as_of is
  the RANKER's responsibility (same structural no-lookahead rule as bars).

Contract a registered ranker MUST guarantee (identical to the signal
engine's existing guarantees -- see trading.signals.engine):

- Purity: no I/O, no wall-clock reads; as_of is the only time input.
- Truncation to as_of is the ranker's job for BOTH bars and fundamentals.
- Column contract: the returned DataFrame is indexed by symbol and contains
  the ranker's feature-percentile columns plus "composite" and
  "raw_return_30d" (momentum_v1: trading.signals.engine.OUTPUT_COLUMNS;
  quality_momentum_v1 adds "quality"; value_momentum_v1 adds
  "earnings_yield" and "book_to_market"). Symbols without enough PRICE
  history are omitted; missing FUNDAMENTALS never drop a symbol (neutral
  instead).
- NaN semantics: an individual NaN feature yields a NaN composite for that
  symbol, which sorts last in rank().

requires_fundamentals tells the I/O layers what to do BEFORE the ranker
runs: pipeline/backtest load the fundamentals store and the live runner
refreshes it only when the configured ranker sets this flag -- momentum_v1
venues never touch fundamentals I/O at all.

The shared rank() sort is NOT part of a ranker's job: every ranker's output
feeds through trading.signals.engine.rank() afterward.

To add a new ranker: implement a RankerFn and register a RankerSpec here
under a new key. Select it per-venue via the `ranker` key in a venue's
[signals] TOML section; trading.config validates the name at load time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from trading.config import SignalConfig
from trading.signals.engine import FeaturePanel, compute_features
from trading.signals.quality import quality_momentum_v1
from trading.signals.skew import skew_change_v1, skew_v1
from trading.signals.value import value_momentum_v1

RankerFn = Callable[..., pd.DataFrame]
"""fn(bars, as_of, config, fundamentals=None, *, skew=None, panel=None) -> DataFrame.

`panel` is an optional precomputed FeaturePanel (trading.signals.engine): the
backtester builds one for the whole run so per-session ranking is a gather,
not a rolling recompute. When None, the ranker builds features from `bars`
directly (the live path). Passing it never changes results -- a panel gathered
at as_of equals a from-scratch compute at as_of (no lookahead).

`skew` is the optional per-session IV-skew side channel (trading.signals.skew):
a dict of per-symbol skew history truncated to <= as_of, exactly the shape
`fundamentals` takes, loaded by the I/O layers only when the spec sets
requires_skew. None (the momentum/quality/value default) means no skew -- those
rankers ignore the kwarg."""


@dataclass(frozen=True)
class RankerSpec:
    fn: RankerFn
    requires_fundamentals: bool
    # Parallel to requires_fundamentals: tells the I/O layers (pipeline /
    # backtest prepare) to load the IV-skew store and build an IVSkewPanel
    # before ranking, and gather the as-of skew per session. Defaulted False so
    # existing specs and their construction sites stay unchanged.
    requires_skew: bool = False


def _momentum_v1(
    bars: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: SignalConfig,
    fundamentals: dict[str, pd.DataFrame] | None = None,
    *,
    skew: dict[str, pd.DataFrame] | None = None,
    panel: FeaturePanel | None = None,
) -> pd.DataFrame:
    """v2-contract adapter: momentum_v1 has no fundamentals or skew input by design."""
    return compute_features(bars, as_of, config, panel=panel)


RANKERS: dict[str, RankerSpec] = {
    "momentum_v1": RankerSpec(_momentum_v1, requires_fundamentals=False),
    "quality_momentum_v1": RankerSpec(quality_momentum_v1, requires_fundamentals=True),
    "value_momentum_v1": RankerSpec(value_momentum_v1, requires_fundamentals=True),
    "skew_v1": RankerSpec(skew_v1, requires_fundamentals=False, requires_skew=True),
    "skew_change_v1": RankerSpec(skew_change_v1, requires_fundamentals=False, requires_skew=True),
}


def get_ranker(name: str) -> RankerSpec:
    try:
        return RANKERS[name]
    except KeyError:
        known = ", ".join(sorted(RANKERS))
        raise ValueError(f"unknown ranker {name!r}; known rankers: {known}") from None
