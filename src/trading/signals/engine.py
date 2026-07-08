"""Cross-sectional signal engine (spec: Signal Engine).

Pure: no I/O, no clock — as_of is always a parameter. Features are
normalized to cross-sectional percentiles; the composite is their
equal-weight blend (weights fixed equal in v1 by design).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from trading.config import SignalConfig
from trading.signals.features import (
    breakout_proximity_series,
    overextension_series,
    raw_return_series,
    vol_adjusted_return_series,
    volume_surge_series,
)

FEATURE_COLUMNS = [
    "mom_short",
    "mom_med",
    "mom_long",
    "volume_surge",
    "breakout",
    "overextension",
]
OUTPUT_COLUMNS = [*FEATURE_COLUMNS, "composite", "raw_return_30d"]
# The pre-percentile feature columns a panel stores per symbol, in order.
RAW_COLUMNS = [*FEATURE_COLUMNS, "raw_return_30d"]


def min_history_rows(config: SignalConfig) -> int:
    return max(
        config.momentum_windows[-1] + 1,
        config.vol_window + 1,
        config.volume_baseline,
        config.breakout_windows[-1],
        config.mean_window,
        config.rsi_window + 1,
    )


def _feature_frame(df: pd.DataFrame, config: SignalConfig) -> pd.DataFrame:
    """Every raw (pre-percentile) feature for one symbol, as a time-series
    aligned to the symbol's bars. One vectorized pass per feature; row i is the
    feature evaluated on data <= bar i (no lookahead)."""
    close, high, volume = df["close"], df["high"], df["volume"]
    short, med, long_ = config.momentum_windows
    vw, cal = config.vol_window, config.calendar_days
    data = {
        "mom_short": vol_adjusted_return_series(close, short, vw, cal),
        "mom_med": vol_adjusted_return_series(close, med, vw, cal),
        "mom_long": vol_adjusted_return_series(close, long_, vw, cal),
        "volume_surge": volume_surge_series(
            close, volume, config.volume_week, config.volume_baseline
        ),
        "breakout": breakout_proximity_series(close, high, config.breakout_windows),
        "overextension": overextension_series(close, config.rsi_window, config.mean_window),
        "raw_return_30d": raw_return_series(close, config.raw_return_days),
    }
    return pd.DataFrame(data, index=df.index, columns=RAW_COLUMNS)


class FeaturePanel:
    """Precomputed raw feature time-series for a set of symbols.

    Built ONCE from full-span bars; a per-session gather then reads the row at
    as_of for whatever symbol subset that session ranks -- turning the
    walk-forward's dominant per-session rolling recompute into a cheap lookup.
    Because every stored feature looks only backward (rolling / shift / asof),
    the value gathered at as_of is identical whether or not bars after as_of
    are present, so a full-span panel and a per-session recompute agree.
    """

    def __init__(self, frames: dict[str, pd.DataFrame], required: int) -> None:
        self._required = required
        # Cache numpy views for gather: an int64 datetime index and the raw
        # feature matrix, so per-session lookup is a searchsorted + row slice
        # with no per-symbol Series allocation.
        self._index: dict[str, np.ndarray] = {}
        self._values: dict[str, np.ndarray] = {}
        for symbol, frame in frames.items():
            # int64 nanoseconds (UTC): unambiguous, tz-safe, fast to searchsorted.
            self._index[symbol] = frame.index.asi8
            self._values[symbol] = frame.to_numpy()

    @classmethod
    def from_bars(cls, bars: dict[str, pd.DataFrame], config: SignalConfig) -> FeaturePanel:
        frames = {symbol: _feature_frame(df, config) for symbol, df in bars.items()}
        return cls(frames, min_history_rows(config))

    def gather(self, symbols: Iterable[str], as_of: pd.Timestamp) -> pd.DataFrame:
        """Raw features at as_of for `symbols` with sufficient history.

        A symbol whose bars up to and including as_of number fewer than
        min_history_rows is omitted entirely -- exactly the scalar path's
        `len(window) < required` skip, not merely a NaN row."""
        as_of_ns = as_of.value  # int64 ns, UTC
        rows: dict[str, np.ndarray] = {}
        for symbol in symbols:
            index = self._index.get(symbol)
            if index is None:
                continue
            # position of the last bar at or before as_of; count = pos + 1.
            pos = int(np.searchsorted(index, as_of_ns, side="right")) - 1
            if pos + 1 < self._required:
                continue
            rows[symbol] = self._values[symbol][pos]
        if not rows:
            return pd.DataFrame(columns=RAW_COLUMNS, dtype="float64")
        return pd.DataFrame.from_dict(rows, orient="index", columns=RAW_COLUMNS)


def compute_features(
    bars: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: SignalConfig,
    *,
    panel: FeaturePanel | None = None,
) -> pd.DataFrame:
    """Cross-sectional feature table at as_of.

    Raw per-symbol features come from `panel` when supplied (the backtester
    precomputes one panel for the whole run and passes it to every session);
    otherwise a panel is built from `bars` on the spot (the live path, one
    as_of per process). Either way the cross-sectional percentile + composite
    below is per-session and cheap.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must be tz-aware UTC")
    if panel is None:
        panel = FeaturePanel.from_bars(bars, config)
    raw = panel.gather(bars.keys(), as_of)
    if raw.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS, dtype="float64")

    pct = raw[FEATURE_COLUMNS].rank(pct=True)
    pct["composite"] = (
        pct["mom_short"]
        + pct["mom_med"]
        + pct["mom_long"]
        + pct["volume_surge"]
        + pct["breakout"]
        + (1.0 - pct["overextension"])  # negative guard
    ) / 6.0
    pct["raw_return_30d"] = raw["raw_return_30d"]
    return pct[OUTPUT_COLUMNS]


def rank(features: pd.DataFrame) -> pd.DataFrame:
    # Sort the symbol index first, then stable-sort by composite so ties (common
    # with percentile-derived composites) resolve alphabetically — deterministic
    # ranking is required for reproducible backtests.
    return features.sort_index().sort_values(
        "composite", ascending=False, na_position="last", kind="mergesort"
    )
