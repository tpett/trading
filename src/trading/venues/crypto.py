"""Crypto venue: ccxt/Kraken daily UTC bars, Robinhood-listed universe CSV.

The universe CSV is the maintained Robinhood listing snapshot with per-symbol
status (tradable / sell_only / untradable). Kraken is the M1 data source;
Bitstamp (Robinhood's routing venue) is a config-free swap later since both
sit behind ccxt's fetch_ohlcv.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
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


def _backfill_fetch(exchange_id: str, pair: str, since_ms: int, limit: int) -> list[list[float]]:
    """One page of daily candles from the deep-history exchange (spec Open
    Item: Kraken caps daily history at ~720 candles). Network touchpoint,
    isolated for monkeypatching.

    Winner verified live 2026-07-04 per candidate order coinbase ->
    coinbaseexchange -> bitstamp: coinbase served BTC/USD daily candles back
    to 2018-01-01 without API keys, honored `since`, and paginated forward
    deterministically (300-candle pages regardless of requested limit).
    """
    import ccxt

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    try:
        return exchange.fetch_ohlcv(pair, timeframe="1d", since=since_ms, limit=limit)
    except ccxt.BaseError as e:
        raise DataFetchError(f"{exchange_id} fetch failed for {pair}: {e}") from e


def _paginate(
    fetch_page: Callable[[int], list[list[float]]],
    start: datetime.date,
    end: datetime.date,
) -> list[list[float]]:
    """Stitch forward-paged OHLCV rows over [start, end]. Progress is
    guaranteed: a page adding nothing new ends the loop."""
    since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows: list[list[float]] = []
    while True:
        page = fetch_page(since_ms)
        new = [r for r in page if not rows or r[0] > rows[-1][0]]
        if not new:
            break
        rows.extend(new)
        if new[-1][0] >= end_ms:
            break
        since_ms = int(new[-1][0]) + 1
    return rows


def _rows_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
    df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
    df.index.name = None
    return df.astype("float64").sort_index()


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
        """Daily bars over [start, end], spliced from Kraken + the backfill exchange.

        Kraken's public OHLC retention is anchored to TODAY (~720 most recent
        candles), not to the requested window, so coverage is judged from what
        Kraken actually RETURNED -- never from clock/config arithmetic. Kraken
        is fetched first; the backfill exchange fills only the head Kraken
        could not serve; Kraken wins any overlap.
        """
        import ccxt

        pair = f"{symbol}/USD"
        cfg = self._config.data
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")

        try:
            kraken_rows = _paginate(lambda since_ms: _kraken_fetch(pair, since_ms), start, end)
        except ccxt.BaseError as e:
            raise DataFetchError(f"kraken fetch failed for {pair}: {e}") from e
        # A fully-historical request can come back empty-in-range (Kraken only
        # returns rows from today-720d onward, all AFTER the requested end) or
        # starting mid-window. Only in-range rows count as coverage.
        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)
        kraken_in_range = [r for r in kraken_rows if start_ms <= r[0] <= end_ms]
        kraken_first: pd.Timestamp | None = None
        if kraken_in_range:
            kraken_first = pd.Timestamp(kraken_in_range[0][0], unit="ms", tz="UTC")

        # Head covered by Kraken alone (the live path: end ~ today, start
        # inside the retention window)? Then the backfill exchange is never
        # contacted. seam_max_gap_days doubles as the tolerance here: the same
        # small hole we would accept at a splice seam is acceptable at the head.
        head_covered = (
            kraken_first is not None and (kraken_first - start_ts).days <= cfg.seam_max_gap_days
        )

        frames: list[pd.DataFrame] = []
        backfill_last: pd.Timestamp | None = None
        if cfg.backfill_exchange and not head_covered:
            # Backfill the head Kraken could not serve: up to Kraken's first
            # in-range row (inclusive, so the sources overlap and Kraken wins
            # the shared day), or the whole range if Kraken had nothing in it.
            backfill_end = min(kraken_first.date(), end) if kraken_first is not None else end
            try:
                deep_rows = _paginate(
                    lambda since_ms: _backfill_fetch(
                        cfg.backfill_exchange, pair, since_ms, cfg.backfill_page_limit
                    ),
                    start,
                    backfill_end,
                )
                if deep_rows:
                    deep_frame = _rows_to_frame(deep_rows)
                    frames.append(deep_frame)
                    backfill_last = deep_frame.index[-1]
            except DataFetchError as e:
                # A pair the backfill exchange does not list is not an error --
                # it simply has Kraken-depth history only. Anything else
                # (network, rate limit) must propagate: silently truncating a
                # deep request would corrupt the backtest.
                if not isinstance(e.__cause__, ccxt.BadSymbol):
                    raise
        if kraken_in_range:
            frames.append(_rows_to_frame(kraken_in_range))

        if backfill_last is not None and kraken_first is not None:
            # Seam guard, only when BOTH sources contributed: a date hole
            # between the backfill's last row and Kraken's first row must fail
            # loudly here rather than reach signal computation silently.
            # Full-backfill and full-Kraken results have no seam to check.
            if (kraken_first - backfill_last).days > cfg.seam_max_gap_days:
                raise DataFetchError(
                    f"deep history seam gap for {pair}: backfill ends "
                    f"{backfill_last.date()}, kraken starts {kraken_first.date()} "
                    f"(seam_max_gap_days={cfg.seam_max_gap_days})"
                )

        if not frames:
            raise DataFetchError(f"no crypto data for {pair}")
        df = pd.concat(frames)
        # Kraken appended last: keep="last" makes Kraken win any overlap (the
        # documented splice precedence; both sources are spot USD prices).
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.loc[start_ts:end_ts]
        return validate_ohlcv(df)
