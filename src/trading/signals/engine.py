"""Cross-sectional signal engine (spec: Signal Engine).

Pure: no I/O, no clock — as_of is always a parameter. Features are
normalized to cross-sectional percentiles; the composite is their
equal-weight blend (weights fixed equal in v1 by design).
"""

from __future__ import annotations

import pandas as pd

from trading.config import SignalConfig
from trading.signals.features import (
    breakout_proximity,
    overextension,
    raw_return,
    vol_adjusted_return,
    volume_surge,
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


def min_history_rows(config: SignalConfig) -> int:
    return max(
        config.momentum_windows[-1] + 1,
        config.vol_window + 1,
        config.volume_baseline,
        config.breakout_windows[-1],
        config.mean_window,
        config.rsi_window + 1,
    )


def compute_features(
    bars: dict[str, pd.DataFrame], as_of: pd.Timestamp, config: SignalConfig
) -> pd.DataFrame:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be tz-aware UTC")
    required = min_history_rows(config)
    short, med, long_ = config.momentum_windows

    raw_rows: dict[str, dict[str, float]] = {}
    for symbol, df in bars.items():
        window = df.loc[:as_of]  # structural no-lookahead cut
        if len(window) < required:
            continue
        close, high, volume = window["close"], window["high"], window["volume"]
        raw_rows[symbol] = {
            "mom_short": vol_adjusted_return(close, short, config.vol_window, config.calendar_days),
            "mom_med": vol_adjusted_return(close, med, config.vol_window, config.calendar_days),
            "mom_long": vol_adjusted_return(close, long_, config.vol_window, config.calendar_days),
            "volume_surge": volume_surge(close, volume, config.volume_week, config.volume_baseline),
            "breakout": breakout_proximity(close, high, config.breakout_windows),
            "overextension": overextension(close, config.rsi_window, config.mean_window),
            "raw_return_30d": raw_return(close, config.raw_return_days),
        }

    if not raw_rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS, dtype="float64")

    raw = pd.DataFrame.from_dict(raw_rows, orient="index")
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
