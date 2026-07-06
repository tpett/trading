import datetime

import pandas as pd

from fundamentals_helpers import num_line, sub_line, write_quarter_zip
from trading.fundamentals.backfill import backfill_quarters, last_complete_quarter, quarter_range
from trading.fundamentals.store import FundamentalsStore

# CIK 1326801 renamed FB -> NEWCO at 2023-08-01 in this fixture map: one CIK's
# series must split across the two symbols by FILED date.
CIK_MAP = pd.DataFrame(
    {
        "symbol": ["FB", "NEWCO", "OTHER"],
        "cik": [1326801, 1326801, 555],
        "start": ["2017-01-01", "2023-08-01", "2017-01-01"],
        "end": ["2023-08-01", "", ""],
    }
)


def _quarter_zip(tmp_path, name, adsh, fy, fp, period, filed, rev, cogs, assets, form="10-Q"):
    subs = [sub_line(adsh, 1326801, form, period, fy, fp, filed)]
    qtrs = 4 if form == "10-K" else 1
    nums = [
        num_line(adsh, "Revenues", period, qtrs, rev),
        num_line(adsh, "CostOfRevenue", period, qtrs, cogs),
        num_line(adsh, "Assets", period, 0, assets),
    ]
    return write_quarter_zip(tmp_path / name, subs, nums)


def _fixture_zips(tmp_path):
    return [
        _quarter_zip(
            tmp_path, "2023q2.zip", "f-01", "2023", "Q1", "20230331", "20230510", 10.0, 4.0, 90.0
        ),
        _quarter_zip(
            tmp_path, "2023q3.zip", "f-02", "2023", "Q2", "20230630", "20230809", 12.0, 5.0, 95.0
        ),
        _quarter_zip(
            tmp_path, "2023q4.zip", "f-03", "2023", "Q3", "20230930", "20231108", 11.0, 5.0, 98.0
        ),
        _quarter_zip(
            tmp_path,
            "2024q1.zip",
            "f-04",
            "2023",
            "FY",
            "20231231",
            "20240220",
            48.0,
            20.0,
            100.0,
            form="10-K",
        ),
    ]


def test_backfill_splits_one_cik_series_across_rename_symbols(tmp_path):
    store = FundamentalsStore(tmp_path / "store")
    stats = backfill_quarters(_fixture_zips(tmp_path), CIK_MAP, store)
    assert stats == {"filers": 1, "symbols": 2, "rows": 4}
    fb = store.read("FB")
    newco = store.read("NEWCO")
    # Filed 2023-05-10 lands on FB (pre-rename); the rest on NEWCO.
    assert list(fb.index) == [pd.Timestamp("2023-05-10", tz="UTC")]
    assert len(newco) == 3
    # TTM continuity survives the split: the 10-K row completes 4 quarters.
    at_10k = newco.loc[pd.Timestamp("2024-02-20", tz="UTC")]
    assert at_10k["gross_profitability"] == (48.0 - 20.0) / 100.0
    assert store.read("OTHER").empty


def test_backfill_rerun_is_idempotent(tmp_path):
    store = FundamentalsStore(tmp_path / "store")
    zips = _fixture_zips(tmp_path)
    backfill_quarters(zips, CIK_MAP, store)
    stats = backfill_quarters(zips, CIK_MAP, store)
    assert stats["rows"] == 0  # append-only store: reprocessing adds nothing


def test_quarter_range_and_last_complete_quarter():
    assert quarter_range("2018q1", "2018q4") == ["2018q1", "2018q2", "2018q3", "2018q4"]
    assert quarter_range("2018q3", "2019q2") == ["2018q3", "2018q4", "2019q1", "2019q2"]
    assert last_complete_quarter(datetime.date(2026, 7, 6)) == "2026q2"
    assert last_complete_quarter(datetime.date(2026, 2, 1)) == "2025q4"
