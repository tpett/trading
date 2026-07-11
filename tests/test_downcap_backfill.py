import datetime

import pandas as pd
import pytest

from trading.data.cache import CacheSourceError, OhlcvCache
from trading.venues.base import DataFetchError, RateLimitError
from trading.venues.universes import downcap_backfill as bf


def _roster():
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "GONE"],
            "exchange": ["NYSE", "NASDAQ", "NYSE"],
            "assetType": ["Stock", "Stock", "Stock"],
            "priceCurrency": ["USD", "USD", "USD"],
            "startDate": ["2019-01-01", "2019-01-01", "2019-01-01"],
            "endDate": ["", "", ""],
        }
    )


class _Adapter:
    """AAA/BBB serve bars; GONE 404s (adapter raises DataFetchError)."""

    def __init__(self):
        self.calls: dict[str, int] = {}

    def fetch_ohlcv(self, symbol, start, end):
        self.calls[symbol] = self.calls.get(symbol, 0) + 1
        if symbol == "GONE":
            raise DataFetchError(f"no equities data for {symbol}")
        idx = pd.date_range(start, periods=3, freq="B", tz="UTC")
        return pd.DataFrame({"close": [1.0, 1.0, 1.0]}, index=idx)


def test_roster_symbols_are_window_candidates():
    syms = bf.roster_symbols(_roster(), datetime.date(2019, 1, 1), datetime.date(2019, 6, 1))
    assert syms == ["AAA", "BBB", "GONE"]


def test_run_backfill_records_gaps_not_guesses(tmp_path):
    cache = OhlcvCache(tmp_path / "equities-downcap-tiingo", refetch_days=5, source="tiingo")
    adapter = _Adapter()
    report = bf.run_backfill(
        ["AAA", "BBB", "GONE"], cache, adapter,
        datetime.date(2019, 1, 1), datetime.date(2019, 1, 10),
    )
    assert report.fetched == 2
    assert report.missing == ["GONE"]        # recorded, never imputed
    assert report.total == 3
    assert report.coverage == pytest.approx(2 / 3)
    assert adapter.calls["GONE"] == 1        # genuine 404 is not retried forever


def test_run_backfill_waits_on_ratelimit_then_succeeds(tmp_path, monkeypatch):
    """The rate-limit retry path is the safety-critical guarantee that a 429
    never punches a coverage hole: fetch_ohlcv raises RateLimitError twice,
    then succeeds on the third call to the SAME symbol."""
    waits = []
    monkeypatch.setattr(bf, "_sleep", lambda s: waits.append(s))
    cache = OhlcvCache(tmp_path / "equities-downcap-tiingo", refetch_days=5, source="tiingo")
    calls = {"n": 0}

    class FlakyAdapter:
        def fetch_ohlcv(self, symbol, start, end):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RateLimitError("429")
            idx = pd.date_range(start, periods=3, freq="B", tz="UTC")
            return pd.DataFrame({"close": [1.0, 1.0, 1.0]}, index=idx)

    report = bf.run_backfill(
        ["AAA"], cache, FlakyAdapter(),
        datetime.date(2019, 1, 1), datetime.date(2019, 1, 10),
        rate_limit_wait_s=42,
    )
    assert calls["n"] == 3                   # waited through two 429s, then got bars
    assert waits == [42, 42]
    assert report.fetched == 1
    assert report.missing == []              # never dropped into a coverage hole


def test_cache_dir_gets_tiingo_source_marker(tmp_path):
    cache_dir = tmp_path / "equities-downcap-tiingo"
    OhlcvCache(cache_dir, refetch_days=5, source="tiingo")
    assert (cache_dir / ".source").read_text().strip() == "tiingo"


def test_refuses_source_mismatch(tmp_path):
    cache_dir = tmp_path / "equities-downcap-tiingo"
    OhlcvCache(cache_dir, refetch_days=5, source="tiingo")   # writes .source=tiingo
    with pytest.raises(CacheSourceError):
        OhlcvCache(cache_dir, refetch_days=5, source="yfinance")
