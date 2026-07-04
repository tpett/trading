"""Crypto venue: ccxt/Kraken daily UTC bars, Robinhood-listed universe CSV.

The universe CSV is the maintained Robinhood listing snapshot with per-symbol
status (tradable / sell_only / untradable). Kraken is the M1 data source;
Bitstamp (Robinhood's routing venue) is a config-free swap later since both
sit behind ccxt's fetch_ohlcv.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import get_args

import pandas as pd

from trading.config import VenueConfig
from trading.venues.base import (
    OHLCV_COLUMNS,
    DataFetchError,
    SymbolInfo,
    SymbolStatus,
    VenueConstraints,
    validate_ohlcv,
)

DEFAULT_UNIVERSE_CSV = Path(__file__).parent / "universes" / "crypto.csv"

_VALID_STATUSES = set(get_args(SymbolStatus))
_KRAKEN_DAILY_LIMIT = 720  # Kraken returns at most 720 daily candles per call


def _kraken_fetch(pair: str, since_ms: int) -> list[list[float]]:
    """Network touchpoint, isolated for monkeypatching."""
    import ccxt

    exchange = ccxt.kraken({"enableRateLimit": True})
    return exchange.fetch_ohlcv(pair, timeframe="1d", since=since_ms, limit=_KRAKEN_DAILY_LIMIT)


class CryptoAdapter:
    def __init__(self, config: VenueConfig, universe_csv: Path | None = None):
        self._config = config
        self._universe_csv = universe_csv or DEFAULT_UNIVERSE_CSV

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        # as_of is part of the locked protocol; the static M1 snapshot ignores it.
        df = pd.read_csv(self._universe_csv)
        infos: list[SymbolInfo] = []
        for row in df.itertuples(index=False):
            if row.status not in _VALID_STATUSES:
                raise ValueError(f"unknown status {row.status!r} for {row.symbol}")
            infos.append(SymbolInfo(symbol=row.symbol, status=row.status))
        return infos

    def constraints(self) -> VenueConstraints:
        c = self._config.costs
        return VenueConstraints(
            taker_fee_bps=c.taker_fee_bps,
            maker_fee_bps=c.maker_fee_bps,
            slippage_bps=c.slippage_bps,
            settlement_days=c.settlement_days,
            trades_24_7=c.trades_24_7,
        )

    def fetch_ohlcv(self, symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        pair = f"{symbol}/USD"
        since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        rows = _kraken_fetch(pair, since_ms)
        if not rows:
            raise DataFetchError(f"no crypto data for {pair}")
        df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
        df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
        df.index.name = None
        df = df.astype("float64").sort_index()
        df = df.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]
        return validate_ohlcv(df)
