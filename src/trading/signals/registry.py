"""Ranker registry (spec: pluggable ranking strategies).

A "ranker" is any callable matching compute_features' exact contract:

    ranker(bars: dict[str, pd.DataFrame], as_of: pd.Timestamp,
           config: SignalConfig) -> pd.DataFrame

Input contract (what a ranker receives):

- `bars` maps symbol -> OHLCV DataFrame with columns exactly
  [open, high, low, close, volume], indexed by a sorted tz-aware UTC
  DatetimeIndex normalized to the bar's date (the shape enforced by
  trading.venues.base.validate_ohlcv, which every adapter's frames pass
  through). Frames may extend PAST as_of: the caller does not pre-cut them.
- `as_of` is a tz-aware UTC pd.Timestamp; momentum_v1 rejects a naive one.

Contract a registered ranker MUST guarantee (identical to the signal engine's
existing guarantees -- see trading.signals.engine):

- Purity: no I/O, no wall-clock reads. as_of is always an explicit parameter;
  the only time input a ranker may act on is the one it is given.
- Truncation to as_of is the RANKER's responsibility, not the caller's: it
  must cut each symbol's frame to rows at or before as_of itself.
  momentum_v1 (compute_features) performs that truncation as the first step
  of its per-symbol loop, via `window = df.loc[:as_of]` -- the "structural
  no-lookahead cut" in trading.signals.engine. The registry and its caller
  (assemble_rankings) do not truncate on the ranker's behalf.
- Column contract: the returned DataFrame is indexed by symbol and has
  exactly the feature-percentile columns plus "composite" and
  "raw_return_30d" (see trading.signals.engine.OUTPUT_COLUMNS). Symbols
  without enough history to compute all features are simply omitted from
  the index (not included with NaN rows).
- NaN semantics: a symbol may still appear with a NaN in an individual
  feature column (and therefore a NaN composite) when that one feature is
  undefined for it (e.g. zero-volume breaking volume_surge) -- see
  test_partial_nan_feature_gives_nan_composite_and_clean_cross_section in
  tests/test_engine.py. NaN composites sort last in rank().

The shared rank() sort (stable, composite descending, alphabetical tie-break,
NaNs last) is NOT part of a ranker's job: every ranker's output is fed
through the same trading.signals.engine.rank() afterward, so ranker authors
only need to produce the feature/composite table above.

To add a new ranker: implement a callable with the signature above and
register it here under a new key in RANKERS. Select it per-venue via the
`ranker` key in a venue's [signals] TOML section; trading.config validates
the name against this registry at config-load time.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from trading.config import SignalConfig
from trading.signals.engine import compute_features

Ranker = Callable[[dict[str, pd.DataFrame], pd.Timestamp, SignalConfig], pd.DataFrame]

RANKERS: dict[str, Ranker] = {
    "momentum_v1": compute_features,
}


def get_ranker(name: str) -> Ranker:
    try:
        return RANKERS[name]
    except KeyError:
        known = ", ".join(sorted(RANKERS))
        raise ValueError(f"unknown ranker {name!r}; known rankers: {known}") from None
