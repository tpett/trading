"""quality_momentum_v1: the six momentum_v1 features + a 7th cross-sectional
gross-profitability percentile (spec: M4 fundamentals overlay).

Quality value per symbol = the LAST fundamentals row FILED at or before
as_of (a forward-filled step function on FILING dates -- never fiscal-period
dates, never interpolated). A NaN latest value or a missing symbol
contributes the NEUTRAL 0.5 percentile: ~half of filers (financials) have no
COGS concept, and punishing missing data would silently tilt the universe.
NaN never reaches back to an older non-NaN value -- the latest filing IS the
point-in-time state of knowledge.

Composite = equal weight over 7 = (6 * momentum_v1 composite + quality) / 7.
No new tunable parameters: the walk-forward surface stays exactly
entry_score_threshold x stop_atr_multiple.

Pure: no I/O, no clock. Fundamentals frames may extend past as_of; the cut
to <= as_of happens here (same structural no-lookahead rule as bars).
"""

from __future__ import annotations

import math

import pandas as pd

from trading.config import SignalConfig
from trading.signals.engine import FEATURE_COLUMNS, FeaturePanel, compute_features

QUALITY_NEUTRAL = 0.5
OUTPUT_COLUMNS = [*FEATURE_COLUMNS, "quality", "composite", "raw_return_30d"]


def latest_filed_row(frame: pd.DataFrame | None, as_of: pd.Timestamp) -> pd.Series | None:
    """The LAST fundamentals row FILED at or before as_of (the step function
    on FILING dates -- the structural no-lookahead cut for fundamentals), or
    None when nothing is visible yet. Shared by the fundamentals rankers
    (quality here, value in trading.signals.value)."""
    if frame is None or frame.empty:
        return None
    window = frame.loc[:as_of]
    return None if window.empty else window.iloc[-1]


def quality_momentum_v1(
    bars: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: SignalConfig,
    fundamentals: dict[str, pd.DataFrame] | None,
    *,
    skew: dict[str, pd.DataFrame] | None = None,
    panel: FeaturePanel | None = None,
) -> pd.DataFrame:
    base = compute_features(bars, as_of, config, panel=panel)
    if base.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS, dtype="float64")
    known = fundamentals or {}
    raw: dict[str, float] = {}
    for symbol in base.index:
        row = latest_filed_row(known.get(symbol), as_of)
        raw[symbol] = math.nan if row is None else float(row["gross_profitability"])
    quality = pd.Series(raw, dtype="float64").rank(pct=True).fillna(QUALITY_NEUTRAL)
    out = base.copy()
    out["quality"] = quality
    # Recompose: base composite already equal-weights the six price features
    # (with the overextension guard inverted inside compute_features).
    out["composite"] = (base["composite"] * len(FEATURE_COLUMNS) + quality) / (
        len(FEATURE_COLUMNS) + 1
    )
    return out[OUTPUT_COLUMNS]
