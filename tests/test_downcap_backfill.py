import datetime

import pandas as pd
import pytest

from trading.data.cache import CacheSourceError, OhlcvCache
from trading.venues.base import DataFetchError
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

    def fetch_ohlcv(self, symbol, start, end):
        if symbol == "GONE":
            raise DataFetchError(f"no equities data for {symbol}")
        idx = pd.date_range(start, periods=3, freq="B", tz="UTC")
        return pd.DataFrame({"close": [1.0, 1.0, 1.0]}, index=idx)


def test_roster_symbols_are_window_candidates():
    syms = bf.roster_symbols(_roster(), datetime.date(2019, 1, 1), datetime.date(2019, 6, 1))
    assert syms == ["AAA", "BBB", "GONE"]


def test_run_backfill_records_gaps_not_guesses(tmp_path):
    cache = OhlcvCache(tmp_path / "equities-downcap-tiingo", refetch_days=5, source="tiingo")
    report = bf.run_backfill(
        ["AAA", "BBB", "GONE"], cache, _Adapter(),
        datetime.date(2019, 1, 1), datetime.date(2019, 1, 10),
    )
    assert report.fetched == 2
    assert report.missing == ["GONE"]        # recorded, never imputed
    assert report.total == 3
    assert report.coverage == pytest.approx(2 / 3)


def test_cache_dir_gets_tiingo_source_marker(tmp_path):
    cache_dir = tmp_path / "equities-downcap-tiingo"
    OhlcvCache(cache_dir, refetch_days=5, source="tiingo")
    assert (cache_dir / ".source").read_text().strip() == "tiingo"


def test_refuses_source_mismatch(tmp_path):
    cache_dir = tmp_path / "equities-downcap-tiingo"
    OhlcvCache(cache_dir, refetch_days=5, source="tiingo")   # writes .source=tiingo
    with pytest.raises(CacheSourceError):
        OhlcvCache(cache_dir, refetch_days=5, source="yfinance")
