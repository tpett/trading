import datetime

import pandas as pd

from trading.data.cache import OhlcvCache


def _frame(start: datetime.date, end: datetime.date, value: float) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": value, "high": value, "low": value, "close": value, "volume": value},
        index=idx,
    )


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


def test_path_for_sanitizes_pair_symbols(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    assert cache.path_for("BTC/USD").name == "BTC-USD.parquet"
