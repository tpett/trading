import datetime

import pandas as pd
import pytest

from trading.fundamentals.metrics import PROVENANCE_COLUMNS, SERIES_COLUMNS
from trading.fundamentals.store import FundamentalsStore


def _rows(dated: dict[str, float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dated], name="filed")
    frame = pd.DataFrame({"gross_profitability": list(dated.values())}, index=idx)
    for col in SERIES_COLUMNS:
        if col not in frame.columns:
            frame[col] = "" if col in PROVENANCE_COLUMNS else 0.0
    return frame[SERIES_COLUMNS]


def test_read_missing_symbol_is_empty_with_utc_index(tmp_path):
    store = FundamentalsStore(tmp_path)
    frame = store.read("AAPL")
    assert frame.empty
    assert list(frame.columns) == SERIES_COLUMNS
    assert frame.index.tz is not None


def test_append_then_read_round_trips(tmp_path):
    store = FundamentalsStore(tmp_path)
    added = store.append("AAPL", _rows({"2023-02-03": 0.48, "2023-05-05": 0.47}))
    assert added == 2
    frame = store.read("AAPL")
    assert list(frame["gross_profitability"]) == [0.48, 0.47]
    assert not store.path_for("AAPL").with_suffix(".parquet.tmp").exists()  # atomic write


def test_history_is_immutable_existing_filed_dates_never_rewrite(tmp_path):
    store = FundamentalsStore(tmp_path)
    store.append("AAPL", _rows({"2023-02-03": 0.48}))
    # Re-appending the same filed date with a DIFFERENT value must be a no-op:
    # fundamentals history is append-only by design (no OhlcvCache-style
    # trailing refetch).
    added = store.append("AAPL", _rows({"2023-02-03": 0.99, "2023-05-05": 0.47}))
    assert added == 1
    frame = store.read("AAPL")
    assert frame.loc[pd.Timestamp("2023-02-03", tz="UTC"), "gross_profitability"] == 0.48


def test_append_rejects_naive_index(tmp_path):
    store = FundamentalsStore(tmp_path)
    naive = _rows({"2023-02-03": 0.48})
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        store.append("AAPL", naive)


def test_append_empty_is_noop(tmp_path):
    store = FundamentalsStore(tmp_path)
    assert store.append("AAPL", _rows({})) == 0
    assert not store.path_for("AAPL").exists()


def test_load_returns_only_symbols_with_rows(tmp_path):
    store = FundamentalsStore(tmp_path)
    store.append("AAPL", _rows({"2023-02-03": 0.48}))
    loaded = store.load(["AAPL", "MSFT"])
    assert set(loaded) == {"AAPL"}


def test_slash_symbol_maps_to_dash_and_round_trips(tmp_path):
    store = FundamentalsStore(tmp_path)
    added = store.append("BRK/B", _rows({"2023-02-03": 0.48}))
    assert added == 1
    path = store.path_for("BRK/B")
    assert path.name == "BRK-B.parquet"
    assert path.exists()
    frame = store.read("BRK/B")
    assert list(frame["gross_profitability"]) == [0.48]


def test_out_of_order_appends_keep_index_sorted(tmp_path):
    store = FundamentalsStore(tmp_path)
    store.append("AAPL", _rows({"2023-05-05": 0.47}))
    store.append("AAPL", _rows({"2023-02-03": 0.48}))
    frame = store.read("AAPL")
    assert frame.index.is_monotonic_increasing
    assert list(frame["gross_profitability"]) == [0.48, 0.47]


def test_refresh_marker_round_trips(tmp_path):
    store = FundamentalsStore(tmp_path)
    assert store.last_refresh() is None
    store.mark_refreshed(datetime.date(2026, 7, 6))
    assert store.last_refresh() == datetime.date(2026, 7, 6)
