"""value_momentum_v1: the six momentum_v1 features + earnings-yield and
book-to-market cross-sectional percentiles (spec: M4 fundamentals overlay,
value extension). Composite = equal weight over 8.

Ratios are computed AT RANKING TIME from price-free stored primitives:

    market_cap     = shares_outstanding (latest row FILED <= as_of) x last close <= as_of
    earnings_yield = ttm_net_income / market_cap
    book_to_market = book_equity     / market_cap

The store carries NO prices by design, so fundamentals parquets never
rewrite when price history refreshes (adjusted closes rewrite on corporate
actions; filings do not). A missing component (no filing yet, NaN primitive)
or a non-positive market cap contributes the NEUTRAL 0.5 percentile -- the
same policy as quality. A NEGATIVE ttm_net_income is a real (low) earnings
yield, not missing data: loss-makers rank at the bottom, deliberately. The
same rule holds for NEGATIVE book_equity (distressed or buyback-heavy
filers): present-but-adverse is signal and ranks lowest; only missing is
neutral.

No new tunable parameters: the walk-forward surface stays exactly
entry_score_threshold x stop_atr_multiple.

Pure: no I/O, no clock. Bars and fundamentals frames may extend past as_of;
both cuts happen here (bars via compute_features / the close lookup,
fundamentals via latest_filed_row).
"""

from __future__ import annotations

import math

import pandas as pd

from trading.config import SignalConfig
from trading.signals.engine import FEATURE_COLUMNS, FeaturePanel, compute_features
from trading.signals.quality import QUALITY_NEUTRAL, latest_filed_row

VALUE_NEUTRAL = QUALITY_NEUTRAL  # one neutral policy across fundamentals rankers
OUTPUT_COLUMNS = [
    *FEATURE_COLUMNS,
    "earnings_yield",
    "book_to_market",
    "composite",
    "raw_return_30d",
]


def _component(row: pd.Series | None, key: str) -> float:
    if row is None or key not in row.index:
        return math.nan
    return float(row[key])


def value_momentum_v1(
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
    ey_raw: dict[str, float] = {}
    bm_raw: dict[str, float] = {}
    for symbol in base.index:
        row = latest_filed_row(known.get(symbol), as_of)
        shares = _component(row, "shares_outstanding")
        # base.index symbols passed compute_features' history gate, so a
        # last close <= as_of always exists.
        close = float(bars[symbol].loc[:as_of, "close"].iloc[-1])
        ey = bm = math.nan
        if not math.isnan(shares) and shares > 0:
            market_cap = shares * close
            if market_cap > 0:
                net_income = _component(row, "ttm_net_income")
                equity = _component(row, "book_equity")
                if not math.isnan(net_income):
                    ey = net_income / market_cap
                if not math.isnan(equity):
                    bm = equity / market_cap
        ey_raw[symbol] = ey
        bm_raw[symbol] = bm
    ey_pct = pd.Series(ey_raw, dtype="float64").rank(pct=True).fillna(VALUE_NEUTRAL)
    bm_pct = pd.Series(bm_raw, dtype="float64").rank(pct=True).fillna(VALUE_NEUTRAL)
    out = base.copy()
    out["earnings_yield"] = ey_pct
    out["book_to_market"] = bm_pct
    out["composite"] = (base["composite"] * len(FEATURE_COLUMNS) + ey_pct + bm_pct) / (
        len(FEATURE_COLUMNS) + 2
    )
    return out[OUTPUT_COLUMNS]
