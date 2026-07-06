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
from trading.signals.engine import compute_features

RankerFn = Callable[
    [dict[str, pd.DataFrame], pd.Timestamp, SignalConfig, dict[str, pd.DataFrame] | None],
    pd.DataFrame,
]


@dataclass(frozen=True)
class RankerSpec:
    fn: RankerFn
    requires_fundamentals: bool


def _momentum_v1(
    bars: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: SignalConfig,
    fundamentals: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """v2-contract adapter: momentum_v1 has no fundamentals input by design;
    compute_features keeps its original 3-arg signature untouched."""
    return compute_features(bars, as_of, config)


RANKERS: dict[str, RankerSpec] = {
    "momentum_v1": RankerSpec(_momentum_v1, requires_fundamentals=False),
}


def get_ranker(name: str) -> RankerSpec:
    try:
        return RANKERS[name]
    except KeyError:
        known = ", ".join(sorted(RANKERS))
        raise ValueError(f"unknown ranker {name!r}; known rankers: {known}") from None
