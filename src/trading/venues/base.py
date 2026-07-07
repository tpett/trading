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
    """Raise ValueError unless df matches the adapter OHLCV contract."""
    if list(df.columns) != OHLCV_COLUMNS:
        raise ValueError(f"OHLCV columns must be {OHLCV_COLUMNS}, got {list(df.columns)}")
    index = df.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or str(index.tz) != "UTC":
        raise ValueError("OHLCV index must be a tz-aware UTC DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV index must be sorted ascending")
    return df
