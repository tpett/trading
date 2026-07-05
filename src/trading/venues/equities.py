"""Equities venue: yfinance daily bars, point-in-time S&P500+NDX100 universe.

universe(as_of) reads a committed membership-intervals CSV (see
universes/equities_membership.csv, built by scripts/build_pit_membership.py
from the snapshotted fja05680/sp500 dataset plus Wikipedia's NDX change
history; see universes/sources/PROVENANCE.md). Backtesting today's members
over the past is prohibited (spec). Prices are corporate-action adjusted
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

DEFAULT_MEMBERSHIP_CSV = Path(__file__).parent / "universes" / "equities_membership.csv"


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
    def __init__(self, config: VenueConfig, membership_csv: Path | None = None):
        self._config = config
        self._membership_csv = membership_csv or DEFAULT_MEMBERSHIP_CSV
        self._membership: pd.DataFrame | None = None

    def _load_membership(self) -> pd.DataFrame:
        # Cached in memory: the backtester calls universe() once per session
        # (~2100 times per prepared span); re-reading the CSV each call is waste.
        if self._membership is None:
            df = pd.read_csv(self._membership_csv, comment="#", dtype=str).fillna("")
            self._membership = df
        return self._membership

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        """Point-in-time S&P 500 + NDX membership as-of the given date (spec:
        backtesting today's members over the past is prohibited)."""
        df = self._load_membership()
        iso = as_of.isoformat()
        active = df[(df["start"] <= iso) & ((df["end"] == "") | (iso < df["end"]))]
        return [SymbolInfo(symbol=s, status="tradable") for s in sorted(set(active["symbol"]))]

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
