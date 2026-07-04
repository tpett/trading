"""Equities venue: yfinance daily bars, static S&P500+NDX100 universe CSV.

M1 uses a committed static membership snapshot (see universes/equities.csv,
built by scripts/build_equities_universe.py). Point-in-time membership
history is Milestone 3. Prices are corporate-action adjusted
(auto_adjust=True) so signals, stops and fills share one price basis.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

from trading.config import VenueConfig
from trading.venues.base import (
    OHLCV_COLUMNS,
    DataFetchError,
    SymbolInfo,
    VenueConstraints,
    validate_ohlcv,
)

DEFAULT_UNIVERSE_CSV = Path(__file__).parent / "universes" / "equities.csv"


def _yf_download(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Network touchpoint, isolated for monkeypatching. yfinance `end` is exclusive."""
    import yfinance as yf

    return yf.download(
        symbol,
        start=start.isoformat(),
        end=(end + datetime.timedelta(days=1)).isoformat(),
        auto_adjust=True,
        actions=False,
        progress=False,
    )


class EquitiesAdapter:
    def __init__(self, config: VenueConfig, universe_csv: Path | None = None):
        self._config = config
        self._universe_csv = universe_csv or DEFAULT_UNIVERSE_CSV

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        # as_of is part of the locked protocol; the static M1 snapshot ignores it.
        df = pd.read_csv(self._universe_csv)
        return [SymbolInfo(symbol=s, status="tradable") for s in df["symbol"]]

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
        raw = _yf_download(symbol, start, end)
        if raw is None or raw.empty:
            raise DataFetchError(f"no equities data for {symbol}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[OHLCV_COLUMNS].astype("float64")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index().loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]
        return validate_ohlcv(df)
