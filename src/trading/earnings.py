"""Earnings-date sourcing for the equities entry blackout (spec: Venue Model).

Kill switch: config [portfolio] earnings_blackout_enabled. A fetch failure
must never crash or block a run — failed symbols are omitted (= unfiltered)
and the degraded flag is journaled upstream as a warning. If the source
proves unreliable in practice, set the switch to false in TOML and document
the drop in the README (spec: dropped entirely, both modes).
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable


def _yf_earnings_dates(symbol: str) -> list[datetime.date]:
    """Network touchpoint, isolated for monkeypatching."""
    import yfinance as yf

    df = yf.Ticker(symbol).get_earnings_dates(limit=8)
    if df is None or df.empty:
        return []
    return sorted({ts.date() for ts in df.index})


def fetch_earnings_dates(symbols: Iterable[str]) -> tuple[dict[str, tuple[str, ...]], bool]:
    dates: dict[str, tuple[str, ...]] = {}
    degraded = False
    for symbol in symbols:
        try:
            dates[symbol] = tuple(d.isoformat() for d in _yf_earnings_dates(symbol))
        except Exception:
            degraded = True  # degrade to no-filter for this symbol, never crash
    return dates, degraded
