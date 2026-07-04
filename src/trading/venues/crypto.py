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
# Per-page size. Note Kraken's public OHLC endpoint also serves at most the 720
# MOST RECENT daily candles regardless of `since` (verified live 2026-07-04), so
# daily history depth is capped at ~2 years; config history_days=500 fits within
# it. The pagination loop in fetch_ohlcv still stitches multiple pages correctly
# on any ccxt exchange that pages forward from `since` (e.g. a Bitstamp swap).
_KRAKEN_DAILY_LIMIT = 720


def _kraken_fetch(pair: str, since_ms: int) -> list[list[float]]:
    """Fetch one page of daily candles starting at `since_ms`.

    Network touchpoint, isolated for monkeypatching. Kraken caps each response
    at _KRAKEN_DAILY_LIMIT candles; fetch_ohlcv paginates to cover longer ranges.
    """
    import ccxt

    exchange = ccxt.kraken({"enableRateLimit": True})
    return exchange.fetch_ohlcv(pair, timeframe="1d", since=since_ms, limit=_KRAKEN_DAILY_LIMIT)


class CryptoAdapter:
    def __init__(self, config: VenueConfig, universe_csv: Path | None = None):
        self._config = config
        self._universe_csv = universe_csv or DEFAULT_UNIVERSE_CSV

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        # as_of is part of the locked protocol; the static M1 snapshot ignores it.
        df = pd.read_csv(self._universe_csv, comment="#")
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
        import ccxt

        pair = f"{symbol}/USD"
        since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
        rows: list[list[float]] = []
        while True:
            try:
                page = _kraken_fetch(pair, since_ms)
            except ccxt.BaseError as e:
                raise DataFetchError(f"kraken fetch failed for {pair}: {e}") from e
            # Keep only rows past what we already have: dedupes page overlap and
            # guarantees progress (a page that adds nothing new ends the loop).
            new = [r for r in page if not rows or r[0] > rows[-1][0]]
            if not new:
                break
            rows.extend(new)
            if new[-1][0] >= end_ms:
                break
            since_ms = int(new[-1][0]) + 1
        if not rows:
            raise DataFetchError(f"no crypto data for {pair}")
        df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
        df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
        df.index.name = None
        df = df.astype("float64").sort_index()
        df = df.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]
        return validate_ohlcv(df)
