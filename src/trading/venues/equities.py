"""Equities venue: yfinance daily bars, point-in-time S&P500+NDX100(+sp400) universe.

universe(as_of) reads a committed membership-intervals CSV (see
universes/equities_membership.csv, built by scripts/build_pit_membership.py
from the snapshotted fja05680/sp500 dataset plus Wikipedia's NDX and sp400
change histories; see universes/sources/PROVENANCE.md). Backtesting today's
members over the past is prohibited (spec). Prices are corporate-action
adjusted (auto_adjust=True) so signals, stops and fills share one price
basis.

The CSV carries three index values (sp500, ndx, sp400); universe() only
counts config.universe.indices, which defaults to ("sp500", "ndx") -- so
live/paper is unaffected by sp400's addition unless a config opts in.
membership_intervals(symbol) exposes the raw per-symbol intervals (all
indices, not filtered) for the backtest engine's ticker-recycling guard
(see prepare() in trading.backtest.engine).
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


TIINGO_CONFIG_PATH = Path.home() / ".config" / "trading" / "config.toml"
_TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{symbol}/prices"
# Tiingo's adjusted fields are split+dividend adjusted -- the same basis as
# yfinance auto_adjust=True, so signals/stops/fills keep one price basis
# regardless of source.
_TIINGO_COLUMNS = {
    "adjOpen": "open",
    "adjHigh": "high",
    "adjLow": "low",
    "adjClose": "close",
    "adjVolume": "volume",
}


def tiingo_token() -> str:
    """$TIINGO_API_KEY wins (tests/CI); otherwise ~/.config/trading/config.toml."""
    import os
    import tomllib

    if token := os.environ.get("TIINGO_API_KEY", ""):
        return token
    if TIINGO_CONFIG_PATH.exists():
        with TIINGO_CONFIG_PATH.open("rb") as f:
            if token := tomllib.load(f).get("tiingo_api_key", ""):
                return token
    raise DataFetchError(
        f"bar_source=tiingo needs an API key: set tiingo_api_key in {TIINGO_CONFIG_PATH} "
        "or export TIINGO_API_KEY"
    )


def _tiingo_get(url: str, params: dict[str, str]) -> tuple[int, bytes]:
    """Network touchpoint, isolated for monkeypatching. The token rides an
    Authorization header, never the URL, so it cannot leak into error
    messages, logs, or proxies' access logs."""
    import urllib.error
    import urllib.parse
    import urllib.request

    req = urllib.request.Request(
        url + "?" + urllib.parse.urlencode(params),
        headers={"Authorization": f"Token {tiingo_token()}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _tiingo_download(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Daily bars from Tiingo, adjusted OHLCV renamed to the venue schema.
    A 404 means Tiingo has no such ticker; an empty array means no bars in
    range (e.g. a delisted name after its endDate) -- both surface as an
    empty frame, matching _yf_download's shape so the cache layer's
    gap-tolerance semantics apply identically."""
    import json

    status, body = _tiingo_get(
        _TIINGO_URL.format(symbol=symbol),
        {"startDate": start.isoformat(), "endDate": end.isoformat(), "format": "json"},
    )
    if status == 404:
        return pd.DataFrame()
    if status != 200:
        raise DataFetchError(f"tiingo {symbol}: HTTP {status}: {body[:200]!r}")
    rows = json.loads(body)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df.index.name = "Date"
    return df[list(_TIINGO_COLUMNS)].rename(columns=_TIINGO_COLUMNS)


# Dispatch by source NAME resolved at call time (not a dict of function
# refs captured at import): tests monkeypatch _yf_download/_tiingo_download
# on the module, and a frozen dict would keep pointing at the originals.
_DOWNLOADER_NAMES = {"yfinance": "_yf_download", "tiingo": "_tiingo_download"}


def _download(source: str, symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    import sys

    fn = getattr(sys.modules[__name__], _DOWNLOADER_NAMES[source])
    return fn(symbol, start, end)


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
        """Point-in-time membership as-of the given date, restricted to
        config.universe.indices (spec: backtesting today's members over the
        past is prohibited). Default indices = ("sp500", "ndx"): sp400 rows
        exist in the CSV but are excluded unless a config opts in, so live's
        universe(as_of=today()) is unchanged by sp400's addition."""
        df = self._load_membership()
        iso = as_of.isoformat()
        active = df[
            df["index"].isin(self._config.universe.indices)
            & (df["start"] <= iso)
            & ((df["end"] == "") | (iso < df["end"]))
        ]
        return [SymbolInfo(symbol=s, status="tradable") for s in sorted(set(active["symbol"]))]

    def membership_intervals(self, symbol: str) -> list[tuple[str, str]]:
        """All committed (start, end) intervals for `symbol`, across every
        index -- not filtered by config.universe.indices. Used by the
        backtest engine's ticker-recycling guard, which needs to know
        whether a symbol has EVER left the universe, independent of which
        index a given backtest run is opting into. end == "" means open-ended
        (still a current member of at least one index)."""
        df = self._load_membership()
        rows = df[df["symbol"] == symbol]
        return list(zip(rows["start"], rows["end"], strict=True))

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
        raw = _download(self._config.data.bar_source, symbol, start, end)
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
