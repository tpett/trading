"""Venue adapter contract shared by all venues and all milestones.

Every adapter returns bars as: tz-aware UTC DatetimeIndex, columns exactly
[open, high, low, close, volume], sorted ascending. Equities prices are
corporate-action adjusted.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Optional corporate-action columns that may ride ALONGSIDE the canonical
# OHLCV in an equities frame (populated by the yfinance/tiingo adapters, cached
# for future dividend/split/event experiments). They are EXTENDED, never
# required: consumers that only want adjusted OHLCV keep selecting OHLCV_COLUMNS
# and ignore these. Kept as an explicit allow-list (not "tolerate any extra
# column") so a genuinely-unexpected column -- a mapping bug, a stray raw
# yfinance field like "Adj Close" -- is still rejected loudly rather than
# silently persisted into the cache.
#   div_cash      cash dividend paid that day; 0.0 when none.
#   split_factor  split ratio that day; 1.0 when none (NOT 0.0).
#   close_raw     unadjusted close.
EXTENDED_OHLCV_COLUMNS = ("div_cash", "split_factor", "close_raw")
_ALLOWED_COLUMNS = frozenset(OHLCV_COLUMNS) | frozenset(EXTENDED_OHLCV_COLUMNS)

SymbolStatus = Literal["tradable", "sell_only", "untradable"]


class DataFetchError(RuntimeError):
    """A symbol's bars could not be fetched or were empty."""


class RateLimitError(DataFetchError):
    """The bar source rejected the request for rate-limiting reasons (e.g.
    an hourly request cap). Distinct from a genuine "no such ticker" so a
    backfill can WAIT and retry rather than record the symbol as missing --
    dropping a rate-limited symbol would silently bias coverage. Subclasses
    DataFetchError so the cache's cold/warm handling treats it identically."""


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    status: SymbolStatus


@dataclass(frozen=True)
class VenueConstraints:
    taker_fee_bps: float
    maker_fee_bps: float
    slippage_bps: float
    settlement_days: int
    trades_24_7: bool


class VenueAdapter(Protocol):
    def universe(self, as_of: datetime.date) -> list[SymbolInfo]: ...

    def constraints(self) -> VenueConstraints: ...

    def fetch_ohlcv(
        self, symbol: str, start: datetime.date, end: datetime.date
    ) -> pd.DataFrame: ...


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError unless df matches the adapter OHLCV contract.

    The five canonical columns are REQUIRED (checked as a subset, so a
    consumer that selects OHLCV_COLUMNS always finds them). The extended
    corporate-action columns (EXTENDED_OHLCV_COLUMNS) may ride along and are
    left untouched. Any OTHER column is rejected: extending the schema is a
    deliberate act, not an accident of an adapter leaking a raw field.
    """
    columns = list(df.columns)
    missing = [c for c in OHLCV_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"OHLCV columns must include {OHLCV_COLUMNS}, missing {missing}")
    unknown = [c for c in columns if c not in _ALLOWED_COLUMNS]
    if unknown:
        raise ValueError(
            f"OHLCV frame has unexpected columns {unknown}; "
            f"allowed extended columns are {list(EXTENDED_OHLCV_COLUMNS)}"
        )
    index = df.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or str(index.tz) != "UTC":
        raise ValueError("OHLCV index must be a tz-aware UTC DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV index must be sorted ascending")
    return df
