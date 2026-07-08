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
import logging
from pathlib import Path

import pandas as pd

from trading.config import VenueConfig
from trading.symbols import load_symbol_allowlist, resolve_current
from trading.venues.base import (
    EXTENDED_OHLCV_COLUMNS,
    OHLCV_COLUMNS,
    DataFetchError,
    RateLimitError,
    SymbolInfo,
    VenueConstraints,
    validate_ohlcv,
)

DEFAULT_MEMBERSHIP_CSV = Path(__file__).parent / "universes" / "equities_membership.csv"

logger = logging.getLogger(__name__)


def _yf_download(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Network touchpoint, isolated for monkeypatching. yfinance `end` is exclusive.

    Approach (b): the FIRST call is byte-for-byte the frozen live call
    (auto_adjust=True, actions=False) -- its five columns are the canonical
    adjusted basis the golden fixtures and live system depend on, and they are
    returned untouched. A SECOND call (auto_adjust=False, actions=True) supplies
    ONLY the extended corporate-action fields (raw close, dividends, splits),
    merged by date. Reconstructing the adjusted basis from the raw call (the
    Adj-Close/Close ratio, approach a) would risk drifting those five values by
    a rounding ULP; keeping the original call guarantees they cannot move. The
    extra request per symbol is acceptable for a one-off backfill.

    Extended columns are returned pre-named (close_raw/div_cash/split_factor) so
    the str.lower() rename in fetch_ohlcv leaves them unchanged.
    """
    import yfinance as yf

    start_s = start.isoformat()
    end_s = (end + datetime.timedelta(days=1)).isoformat()
    adjusted = yf.download(
        symbol, start=start_s, end=end_s, auto_adjust=True, actions=False, progress=False
    )
    if adjusted is None or adjusted.empty:
        return adjusted
    # The second (actions) call only supplies the extended fields. It now runs
    # on EVERY live nightly fetch, so its failure must never discard the
    # canonical bars the first call already produced: on any error the extras
    # fall back to neutral defaults (see _merge_yf_extended).
    try:
        raw = yf.download(
            symbol, start=start_s, end=end_s, auto_adjust=False, actions=True, progress=False
        )
    except Exception:  # noqa: BLE001 - a flaky actions call must not lose the bars
        raw = None
    return _merge_yf_extended(adjusted, raw)


def _merge_yf_extended(adjusted: pd.DataFrame, raw: pd.DataFrame | None) -> pd.DataFrame:
    """Attach close_raw/div_cash/split_factor from the raw+actions frame onto
    the canonical adjusted frame. The five canonical columns come SOLELY from
    `adjusted` and are never touched. If `raw` is missing/empty/malformed, the
    extras default (close_raw<-adjusted close, div_cash 0.0, split_factor 1.0)
    so a survivorship-free backtest still gets its bars; and any label
    misalignment (a date in adjusted absent from raw) is filled with those same
    neutral defaults rather than left NaN (validate_ohlcv has no NaN check)."""
    if isinstance(adjusted.columns, pd.MultiIndex):
        adjusted.columns = adjusted.columns.get_level_values(0)
    out = adjusted[["Open", "High", "Low", "Close", "Volume"]].copy()
    if raw is not None and not raw.empty and isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if raw is None or raw.empty or "Close" not in raw.columns:
        out["close_raw"] = out["Close"]
        out["div_cash"] = 0.0
        out["split_factor"] = 1.0
        return out
    out["close_raw"] = raw["Close"].reindex(out.index).fillna(out["Close"])
    dividends = raw["Dividends"] if "Dividends" in raw.columns else None
    out["div_cash"] = dividends.reindex(out.index).fillna(0.0) if dividends is not None else 0.0
    # yfinance encodes "no split" as 0.0; normalize to our 1.0-when-none convention.
    splits = raw["Stock Splits"] if "Stock Splits" in raw.columns else None
    if splits is not None:
        splits = splits.reindex(out.index)
        out["split_factor"] = splits.where(splits.notna() & (splits != 0.0), 1.0)
    else:
        out["split_factor"] = 1.0
    return out


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
# Extended corporate-action fields carried alongside the adjusted OHLCV. Tiingo
# already ships them on every row: divCash (0.0 when none), splitFactor (1.0
# when none -- Tiingo's own convention matches ours), and the RAW (unadjusted)
# close. Cached now so future dividend/split experiments need no re-pull.
_TIINGO_EXTENDED_COLUMNS = {
    "divCash": "div_cash",
    "splitFactor": "split_factor",
    "close": "close_raw",
}


def tiingo_token() -> str:
    """$TIINGO_API_KEY wins (tests/CI); otherwise ~/.config/trading/config.toml."""
    import os
    import tomllib

    if token := os.environ.get("TIINGO_API_KEY", ""):
        return token
    if TIINGO_CONFIG_PATH.exists():
        try:
            with TIINGO_CONFIG_PATH.open("rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise DataFetchError(f"{TIINGO_CONFIG_PATH} is not valid TOML: {exc}") from exc
        if token := data.get("tiingo_api_key", ""):
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


# Transient conditions worth retrying during an unattended, thousands-of-
# symbols backfill: without this a rate-limit burst or a blip drops symbols
# and silently biases coverage exactly where it's used for the decision run.
_TIINGO_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_TIINGO_MAX_ATTEMPTS = 5
_TIINGO_BACKOFF_BASE_S = 1.0


def _tiingo_sleep(seconds: float) -> None:
    """Sleep touchpoint, isolated so tests don't actually wait."""
    import time

    time.sleep(seconds)


def _tiingo_get_retrying(url: str, params: dict[str, str]) -> tuple[int, bytes]:
    """_tiingo_get with bounded exponential backoff on transient failures
    (429/5xx and network errors). A URLError/timeout raises inside _tiingo_get
    and is retried here; a 404 or other 4xx returns immediately (not
    transient). After the last attempt the final status/body is returned so
    the caller raises a normal DataFetchError."""
    import urllib.error

    for attempt in range(_TIINGO_MAX_ATTEMPTS):
        try:
            status, body = _tiingo_get(url, params)
        except (urllib.error.URLError, TimeoutError):
            if attempt == _TIINGO_MAX_ATTEMPTS - 1:
                raise
            _tiingo_sleep(_TIINGO_BACKOFF_BASE_S * 2**attempt)
            continue
        if status in _TIINGO_RETRY_STATUSES and attempt < _TIINGO_MAX_ATTEMPTS - 1:
            _tiingo_sleep(_TIINGO_BACKOFF_BASE_S * 2**attempt)
            continue
        return status, body
    return status, body  # exhausted retries on a retryable status


def _tiingo_download(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Daily bars from Tiingo, adjusted OHLCV renamed to the venue schema.
    A 404 means Tiingo has no such ticker; an empty array means no bars in
    range (e.g. a delisted name after its endDate) -- both surface as an
    empty frame, matching _yf_download's shape so the cache layer's
    gap-tolerance semantics apply identically."""
    import json

    status, body = _tiingo_get_retrying(
        _TIINGO_URL.format(symbol=symbol),
        {"startDate": start.isoformat(), "endDate": end.isoformat(), "format": "json"},
    )
    if status == 404:
        return pd.DataFrame()
    if status == 429:
        raise RateLimitError(f"tiingo {symbol}: HTTP 429 (hourly cap): {body[:200]!r}")
    if status != 200:
        raise DataFetchError(f"tiingo {symbol}: HTTP {status}: {body[:200]!r}")
    rows = json.loads(body)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df.index.name = "Date"
    mapping = {**_TIINGO_COLUMNS, **_TIINGO_EXTENDED_COLUMNS}
    return df[list(mapping)].rename(columns=mapping)


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
        # Loaded lazily on first universe() call; -1 is the "not yet loaded"
        # sentinel (None is a legitimate "no allowlist configured" value).
        self._allowlist: frozenset[str] | None | int = -1

    def _load_membership(self) -> pd.DataFrame:
        # Cached in memory: the backtester calls universe() once per session
        # (~2100 times per prepared span); re-reading the CSV each call is waste.
        if self._membership is None:
            df = pd.read_csv(self._membership_csv, comment="#", dtype=str).fillna("")
            self._membership = df
        return self._membership

    def _allowlist_symbols(self) -> frozenset[str] | None:
        """The configured symbols allowlist (cached), or None when unset.

        Cached in memory: universe() runs ~once per backtest session (thousands
        of calls per span); re-reading the file each call is waste."""
        if self._allowlist == -1:
            path = self._config.universe.symbols_allowlist_path
            self._allowlist = load_symbol_allowlist(path) if path else None
        return self._allowlist  # type: ignore[return-value]

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        """Point-in-time membership as-of the given date, restricted to
        config.universe.indices (spec: backtesting today's members over the
        past is prohibited). Default indices = ("sp500", "ndx"): sp400 rows
        exist in the CSV but are excluded unless a config opts in, so live's
        universe(as_of=today()) is unchanged by sp400's addition.

        When config.universe.symbols_allowlist_path is set (the options-skew
        experiment), the result is further intersected with that allowlist so a
        run trades only the gathered names -- still PIT-safe: a listed name
        absent from the allowlist is dropped, and an allowlisted name not yet a
        member on `as_of` is still correctly excluded by the membership filter."""
        df = self._load_membership()
        iso = as_of.isoformat()
        active = df[
            df["index"].isin(self._config.universe.indices)
            & (df["start"] <= iso)
            & ((df["end"] == "") | (iso < df["end"]))
        ]
        symbols = sorted(set(active["symbol"]))
        allow = self._allowlist_symbols()
        if allow is not None:
            symbols = [s for s in symbols if s in allow]
        return [SymbolInfo(symbol=s, status="tradable") for s in symbols]

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
        source = self._config.data.bar_source
        # Resolve a renamed/namespace-collided ticker to the successor Tiingo
        # serves the continuous history under, BEFORE the network call. Two
        # invariants keep this safe:
        #   * The CACHE keys by the ORIGINAL `symbol` (the point-in-time
        #     membership ticker) -- resolution only changes the STRING sent to
        #     the vendor. The returned frame is date-indexed and carries no
        #     ticker label, so the caller/cache is unaffected: the parquet for
        #     ABC holds the continuous history fetched via COR.
        #   * PIT-safe, NO lookahead: a bar's adjusted price on any past date
        #     is identical whether the entity was labeled ABC or COR -- only
        #     the label differs, and the recycling guard (engine.prepare's
        #     _truncate_for_membership_exit) still truncates the ORIGINAL
        #     symbol's frame at its membership-interval end, so no post-rename
        #     bars reach the simulator.
        # Gated to tiingo: the successor-keyed continuous serving is a verified
        # Tiingo behaviour, and NAMESPACE_OVERRIDES (MMC->MRSH) is a Tiingo
        # ticker string that would be WRONG on yfinance (where bare MMC is the
        # correct US listing). yfinance current members already use current
        # tickers, so resolution there is at best a no-op and at worst breaks
        # the override case -- so it is not applied.
        fetch_symbol = resolve_current(symbol) if source == "tiingo" else symbol
        if fetch_symbol != symbol:
            logger.info(
                "resolved %s -> %s for %s fetch (rename/namespace)", symbol, fetch_symbol, source
            )
        raw = _download(source, fetch_symbol, start, end)
        if raw is None or raw.empty:
            raise DataFetchError(f"no equities data for {symbol}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        renamed = raw.rename(columns=str.lower)
        # Canonical OHLCV is required; the extended corporate-action columns
        # ride along when the source provides them (both do now) and are
        # tolerated-absent so nothing breaks if a source ever omits them.
        extras = [c for c in EXTENDED_OHLCV_COLUMNS if c in renamed.columns]
        df = renamed[OHLCV_COLUMNS + extras].astype("float64")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index().loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]
        return validate_ohlcv(df)
