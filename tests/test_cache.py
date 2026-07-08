import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import datetime

import pandas as pd
import pytest

from trading.data.cache import OhlcvCache


def _frame(start: datetime.date, end: datetime.date, value: float) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": value, "high": value, "low": value, "close": value, "volume": value},
        index=idx,
    )


def _wide_frame(start: datetime.date, end: datetime.date, value: float) -> pd.DataFrame:
    """OHLCV plus the extended corporate-action columns."""
    df = _frame(start, end, value)
    df["div_cash"] = 0.0
    df["split_factor"] = 1.0
    df["close_raw"] = value * 2
    return df


class RecordingFetcher:
    def __init__(self, value: float):
        self.value = value
        self.calls: list[tuple[str, datetime.date, datetime.date]] = []

    def __call__(self, symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        return _frame(start, end, self.value)


START = datetime.date(2026, 1, 1)
END = datetime.date(2026, 3, 1)


def test_cold_cache_fetches_full_range_and_writes_parquet(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    fetcher = RecordingFetcher(1.0)
    df = cache.fetch("AAPL", START, END, fetcher)
    assert fetcher.calls == [("AAPL", START, END)]
    assert cache.path_for("AAPL").exists()
    assert df.index.min() == pd.Timestamp(START, tz="UTC")
    assert df.index.max() == pd.Timestamp(END, tz="UTC")


def test_warm_cache_refetches_only_trailing_window(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    cache.fetch("AAPL", START, END, RecordingFetcher(1.0))

    second = RecordingFetcher(2.0)
    df = cache.fetch("AAPL", START, END, second)

    cutoff = END - datetime.timedelta(days=30)  # 2026-01-30
    assert second.calls == [("AAPL", cutoff, END)]
    # Rows before the cutoff come from the cache (old value)...
    assert df.loc[pd.Timestamp("2026-01-15", tz="UTC"), "close"] == 1.0
    # ...rows in the trailing window come from the fresh fetch (new value).
    assert df.loc[pd.Timestamp("2026-02-15", tz="UTC"), "close"] == 2.0
    assert not df.index.duplicated().any()


def test_result_is_sliced_to_requested_range(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    cache.fetch("AAPL", START, END, RecordingFetcher(1.0))
    df = cache.fetch("AAPL", datetime.date(2026, 2, 1), END, RecordingFetcher(2.0))
    assert df.index.min() == pd.Timestamp("2026-02-01", tz="UTC")
    assert df.index.max() == pd.Timestamp(END, tz="UTC")


def test_cache_missing_early_history_triggers_full_refetch(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    cache.fetch("AAPL", datetime.date(2026, 2, 1), END, RecordingFetcher(1.0))

    second = RecordingFetcher(2.0)
    cache.fetch("AAPL", START, END, second)  # asks for more history than cached
    assert second.calls == [("AAPL", START, END)]


def test_narrow_request_does_not_truncate_cache_file(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    cache.fetch("AAPL", START, END, RecordingFetcher(1.0))

    # Narrow request: start after the cutoff, end before the cached max.
    narrow_start = datetime.date(2026, 2, 5)
    narrow_end = datetime.date(2026, 2, 20)
    df = cache.fetch("AAPL", narrow_start, narrow_end, RecordingFetcher(2.0))

    # Returned frame is the narrow slice.
    assert df.index.min() == pd.Timestamp(narrow_start, tz="UTC")
    assert df.index.max() == pd.Timestamp(narrow_end, tz="UTC")

    # On-disk file still spans the full original range.
    on_disk = pd.read_parquet(cache.path_for("AAPL"))
    assert on_disk.index.min() == pd.Timestamp(START, tz="UTC")
    assert on_disk.index.max() == pd.Timestamp(END, tz="UTC")
    # Rows before the cutoff (2026-01-21) still come from the original cache.
    assert on_disk.loc[pd.Timestamp("2026-01-15", tz="UTC"), "close"] == 1.0
    # Rows in [cutoff, narrow_start) were refetched, not deleted.
    assert on_disk.loc[pd.Timestamp("2026-01-25", tz="UTC"), "close"] == 2.0
    # Rows in the refetch window carry the fresh values.
    assert on_disk.loc[pd.Timestamp("2026-02-10", tz="UTC"), "close"] == 2.0
    # Rows after the narrow end are preserved.
    assert on_disk.loc[pd.Timestamp("2026-02-25", tz="UTC"), "close"] == 1.0
    assert not on_disk.index.duplicated().any()


def _empty_fetcher(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": [], "high": [], "low": [], "close": [], "volume": []},
        index=pd.DatetimeIndex([], tz="UTC"),
    )


def test_empty_fetch_result_does_not_shrink_cache_file(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    cache.fetch("AAPL", START, END, RecordingFetcher(1.0))

    df = cache.fetch("AAPL", START, END, _empty_fetcher)

    # A data-source gap must never shrink the file: all original dates remain.
    on_disk = pd.read_parquet(cache.path_for("AAPL"))
    expected_idx = pd.date_range(START, END, freq="D", tz="UTC")
    assert on_disk.index.equals(expected_idx)
    assert (on_disk["close"] == 1.0).all()
    # The returned frame is served from the untouched cache.
    assert df.index.equals(expected_idx)
    assert (df["close"] == 1.0).all()


def test_full_refetch_preserves_cached_rows_after_requested_end(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    cache.fetch("AAPL", datetime.date(2026, 2, 1), END, RecordingFetcher(1.0))

    # Requesting more history than cached triggers a full refetch, but with an
    # earlier end than the cached max; the cached tail must survive.
    second = RecordingFetcher(2.0)
    cache.fetch("AAPL", START, datetime.date(2026, 2, 15), second)
    assert second.calls == [("AAPL", START, datetime.date(2026, 2, 15))]

    on_disk = pd.read_parquet(cache.path_for("AAPL"))
    assert on_disk.index.min() == pd.Timestamp(START, tz="UTC")
    assert on_disk.index.max() == pd.Timestamp(END, tz="UTC")
    # Refetched range carries fresh values...
    assert on_disk.loc[pd.Timestamp("2026-02-10", tz="UTC"), "close"] == 2.0
    # ...while cached rows after the requested end are preserved.
    assert on_disk.loc[pd.Timestamp("2026-02-20", tz="UTC"), "close"] == 1.0
    assert not on_disk.index.duplicated().any()


def test_path_for_sanitizes_pair_symbols(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    assert cache.path_for("BTC/USD").name == "BTC-USD.parquet"


def test_full_refetch_gappy_fresh_is_authoritative_in_range(tmp_path):
    """Full refetch (request needs more history than cached): fresh data is
    authoritative for the fetched range — cached in-range days missing from
    fresh are dropped, while cached rows after the requested end survive."""
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    cache.fetch("AAPL", datetime.date(2026, 2, 1), END, RecordingFetcher(1.0))

    gap = pd.date_range("2026-02-10", "2026-02-12", freq="D", tz="UTC")

    def gappy(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        return _frame(start, end, 2.0).drop(gap)

    # start=START < cached min (2026-02-01) => full refetch path.
    cache.fetch("AAPL", START, datetime.date(2026, 2, 15), gappy)

    on_disk = pd.read_parquet(cache.path_for("AAPL"))
    # Fresh wins the refetched range: the gap days are gone, not backfilled.
    assert pd.Timestamp("2026-02-11", tz="UTC") not in on_disk.index
    assert on_disk.loc[pd.Timestamp("2026-02-05", tz="UTC"), "close"] == 2.0
    # Cached tail past the requested end is preserved.
    assert on_disk.loc[pd.Timestamp("2026-02-20", tz="UTC"), "close"] == 1.0
    assert not on_disk.index.duplicated().any()


def test_source_marker_written_on_first_use(tmp_path):
    OhlcvCache(tmp_path / "c", refetch_days=5, source="tiingo")
    assert (tmp_path / "c" / ".source").read_text() == "tiingo"


def test_source_mismatch_raises(tmp_path):
    from trading.data.cache import CacheSourceError

    OhlcvCache(tmp_path / "c", refetch_days=5, source="yfinance")
    with pytest.raises(CacheSourceError, match="fresh directory"):
        OhlcvCache(tmp_path / "c", refetch_days=5, source="tiingo")


def test_legacy_unmarked_dir_adopted_only_by_yfinance(tmp_path):
    from trading.data.cache import CacheSourceError

    d = tmp_path / "legacy"
    d.mkdir()
    (d / "AAPL.parquet").write_bytes(b"")  # a pre-marker cache file
    # A non-default source must not inherit unmarked (yfinance) parquets.
    with pytest.raises(CacheSourceError, match="legacy yfinance"):
        OhlcvCache(d, refetch_days=5, source="tiingo")
    # yfinance adopts it and stamps the marker.
    OhlcvCache(d, refetch_days=5, source="yfinance")
    assert (d / ".source").read_text() == "yfinance"


def test_same_source_reopen_is_fine(tmp_path):
    OhlcvCache(tmp_path / "c", refetch_days=5, source="tiingo")
    OhlcvCache(tmp_path / "c", refetch_days=5, source="tiingo")  # no raise


def test_datafetcherror_on_warm_cache_preserves_and_serves_history(tmp_path):
    # C1 regression: a delisted name's trailing-refetch window is empty years
    # after delisting, and the real adapter signals that by RAISING
    # DataFetchError (not returning empty). On a warm cache that must be
    # treated as a gap -- serve the cached history, never drop the symbol
    # (which would silently reintroduce survivorship bias on a re-run).
    from trading.venues.base import DataFetchError

    cache = OhlcvCache(tmp_path / "c", refetch_days=30)
    seed = RecordingFetcher(100.0)
    first = cache.fetch("XLNX", START, END, seed)
    assert not first.empty  # cold run caches the delisted name's history

    def raiser(symbol, start, end):
        raise DataFetchError(f"no equities data for {symbol}")

    served = cache.fetch("XLNX", START, END, raiser)
    assert served.equals(first)  # cached history preserved and served
    # And the persisted file was not shrunk.
    assert not cache.fetch("XLNX", START, END, raiser).empty


def test_datafetcherror_on_cold_miss_propagates(tmp_path):
    # No cache to fall back on: a genuine fetch failure must surface, not be
    # silently swallowed into an empty frame.
    from trading.venues.base import DataFetchError

    cache = OhlcvCache(tmp_path / "c", refetch_days=30)

    def raiser(symbol, start, end):
        raise DataFetchError("boom")

    with pytest.raises(DataFetchError):
        cache.fetch("NEVER", START, END, raiser)


def test_extended_columns_survive_cache_round_trip(tmp_path):
    cache = OhlcvCache(tmp_path / "c", refetch_days=30)

    class WideFetcher:
        def __init__(self, value):
            self.value = value

        def __call__(self, symbol, start, end):
            return _wide_frame(start, end, self.value)

    df = cache.fetch("AAPL", START, END, WideFetcher(3.0))
    assert list(df.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "div_cash",
        "split_factor",
        "close_raw",
    ]
    assert (df["close_raw"] == 6.0).all()

    # Reopen and read straight from the parquet: the extra columns persisted.
    on_disk = pd.read_parquet(cache.path_for("AAPL"))
    assert list(on_disk.columns) == list(df.columns)
    assert (on_disk["split_factor"] == 1.0).all()

    # A warm refetch keeps the wide schema through the concat/dedup merge.
    warm = cache.fetch("AAPL", START, END, WideFetcher(4.0))
    assert list(warm.columns) == list(df.columns)
    assert warm.loc[pd.Timestamp("2026-02-15", tz="UTC"), "close_raw"] == 8.0


def test_narrow_cache_migrates_to_wide_on_next_fetch(tmp_path):
    # M2: a legacy narrow (pre-corporate-action) cache must UPGRADE seamlessly
    # when the widened code returns wide frames -- the live yfinance cache dir
    # holds real narrow parquets and its nightly run must not break.
    cache = OhlcvCache(tmp_path / "c", refetch_days=30)
    cache.fetch("AAPL", START, END, RecordingFetcher(1.0))  # narrow parquet on disk

    def wide(symbol, start, end):
        return _wide_frame(start, end, 2.0)

    merged = cache.fetch("AAPL", START, END, wide)
    assert set(merged.columns) == {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "div_cash",
        "split_factor",
        "close_raw",
    }
    # Old rows (pre-migration) get neutral defaults; close_raw falls back to close.
    old = merged.loc[: pd.Timestamp("2026-01-20", tz="UTC")]
    assert (old["div_cash"] == 0.0).all()
    assert (old["split_factor"] == 1.0).all()
    assert (old["close_raw"] == old["close"]).all()


def test_real_schema_corruption_still_raises(tmp_path):
    from trading.data.cache import CacheSchemaError

    cache = OhlcvCache(tmp_path / "c", refetch_days=30)
    cache.fetch("AAPL", START, END, lambda s, a, b: _wide_frame(a, b, 1.0))  # wide on disk

    def narrower(symbol, start, end):  # fetch DROPS a canonical column -> real corruption
        return _frame(start, end, 2.0).drop(columns=["volume"])

    with pytest.raises(CacheSchemaError, match="rebuild this cache dir"):
        cache.fetch("AAPL", START, END, narrower)


def test_offline_serves_covered_range_without_fetching(tmp_path):
    # Seed a full cache online, then reopen offline and confirm no fetch fires.
    OhlcvCache(tmp_path / "c", refetch_days=30).fetch("AAPL", START, END, RecordingFetcher(1.0))

    offline = OhlcvCache(tmp_path / "c", refetch_days=30, offline=True)

    def boom(symbol, start, end):
        raise AssertionError("offline mode must never call fetch_fn")

    df = offline.fetch("AAPL", datetime.date(2026, 1, 15), datetime.date(2026, 2, 15), boom)
    assert df.index.min() == pd.Timestamp("2026-01-15", tz="UTC")
    assert df.index.max() == pd.Timestamp("2026-02-15", tz="UTC")
    assert (df["close"] == 1.0).all()


def test_offline_missing_file_raises(tmp_path):
    from trading.data.cache import OfflineCacheError

    offline = OhlcvCache(tmp_path / "c", refetch_days=30, offline=True)
    with pytest.raises(OfflineCacheError, match="no parquet"):
        offline.fetch("AAPL", START, END, RecordingFetcher(1.0))


def test_offline_uncovered_start_raises(tmp_path):
    from trading.data.cache import OfflineCacheError

    # A request STARTING well before the cached span is a real miss.
    OhlcvCache(tmp_path / "c", refetch_days=30).fetch("AAPL", START, END, RecordingFetcher(1.0))
    offline = OhlcvCache(tmp_path / "c", refetch_days=30, offline=True)
    with pytest.raises(OfflineCacheError, match="after requested start"):
        offline.fetch("AAPL", datetime.date(2025, 6, 1), END, RecordingFetcher(1.0))


def test_offline_serves_delisted_trailing_gap_without_raising(tmp_path):
    # C1 regression: a delisted name's bars stop at its delisting date, years
    # before the uniform backtest `end`. prepare() still requests [start, end]
    # for every symbol. Offline mode MUST serve the cached history (matching the
    # online path) rather than raise -- raising would drop every delisted name
    # and silently reintroduce the survivorship bias the frozen cache removes.
    delisting = datetime.date(2026, 2, 1)
    OhlcvCache(tmp_path / "c", refetch_days=30).fetch(
        "XLNX", START, delisting, RecordingFetcher(1.0)
    )
    offline = OhlcvCache(tmp_path / "c", refetch_days=30, offline=True)

    def boom(symbol, start, end):
        raise AssertionError("offline mode must never call fetch_fn")

    # Ask for the full window even though bars end at delisting -- must be served.
    df = offline.fetch("XLNX", START, datetime.date(2026, 6, 1), boom)
    assert not df.empty
    assert df.index.max().date() == delisting


def test_offline_gross_interior_gap_raises(tmp_path):
    from trading.data.cache import OfflineCacheError

    # A backfill that died mid-window leaves a months-long interior hole. That is
    # corruption, not a delisting, and offline mode must fail loudly.
    idx = pd.DatetimeIndex(
        list(pd.date_range("2026-01-01", "2026-01-31", freq="D", tz="UTC"))
        + list(pd.date_range("2026-05-01", "2026-05-31", freq="D", tz="UTC"))  # ~3-month hole
    )
    holed = pd.DataFrame({c: 1.0 for c in ["open", "high", "low", "close", "volume"]}, index=idx)
    path = tmp_path / "c"
    path.mkdir()
    (path / ".source").write_text("yfinance")
    holed.to_parquet(path / "HOLE.parquet")
    offline = OhlcvCache(path, refetch_days=30, offline=True)
    with pytest.raises(OfflineCacheError, match="interior gap"):
        offline.fetch("HOLE", datetime.date(2026, 1, 1), datetime.date(2026, 5, 31), None)


def test_offline_tolerates_weekend_gap_at_edges(tmp_path):
    # Cached span starts a few days after the requested start (weekend/holiday);
    # within _START_TOLERANCE this must be served, not rejected.
    OhlcvCache(tmp_path / "c", refetch_days=30).fetch(
        "AAPL", datetime.date(2026, 1, 3), END, RecordingFetcher(1.0)
    )
    offline = OhlcvCache(tmp_path / "c", refetch_days=30, offline=True)
    df = offline.fetch("AAPL", START, END, RecordingFetcher(1.0))  # asks from Jan 1
    assert not df.empty


def test_backfill_waits_on_ratelimit_then_succeeds(tmp_path, monkeypatch):
    import scripts.backfill_bars as bf

    from trading.venues.base import RateLimitError

    waits = []
    monkeypatch.setattr(bf, "_sleep", lambda s: waits.append(s))
    cache = OhlcvCache(tmp_path / "c", refetch_days=30)
    calls = {"n": 0}

    class Adapter:
        def fetch_ohlcv(self, symbol, start, end):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RateLimitError("429")
            return _frame(START, END, 50.0)

    df = bf._fetch_waiting_on_rate_limit(cache, Adapter(), "X", START, END, wait_s=42)
    assert not df.empty
    assert calls["n"] == 3  # waited through two 429s, then got bars
    assert waits == [42, 42]
