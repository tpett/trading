import datetime

import pandas as pd

from trading.fundamentals.cik_map import cik_for, interval_slice, load_cik_map

MAP = load_cik_map()  # the committed artifact


def test_committed_map_shape():
    assert list(MAP.columns) == ["symbol", "cik", "start", "end"]
    assert MAP["cik"].dtype.kind == "i"
    assert len(MAP) > 400  # roughly the current sp500+ndx membership, plus history


def test_fb_meta_one_cik_across_the_rename():
    fb = MAP[MAP["symbol"] == "FB"].iloc[0]
    meta = MAP[MAP["symbol"] == "META"].iloc[0]
    assert fb["cik"] == meta["cik"] == 1326801
    assert fb["end"] == "2022-06-09"
    assert meta["start"] == "2022-06-09"
    # PIT lookup: before the rename FB resolves and META does not; after, vice versa.
    assert cik_for(MAP, "FB", datetime.date(2022, 6, 8)) == 1326801
    assert cik_for(MAP, "META", datetime.date(2022, 6, 8)) is None
    assert cik_for(MAP, "META", datetime.date(2022, 6, 9)) == 1326801
    assert cik_for(MAP, "FB", datetime.date(2022, 6, 9)) is None


def test_abc_cor_one_cik_across_the_rename():
    abc = MAP[MAP["symbol"] == "ABC"].iloc[0]
    cor = MAP[MAP["symbol"] == "COR"].iloc[0]
    assert abc["cik"] == cor["cik"] == 1140859


def test_unknown_symbol_resolves_to_none():
    assert cik_for(MAP, "NOSUCHTICKER", datetime.date(2024, 1, 1)) is None


def test_interval_slice_is_start_inclusive_end_exclusive():
    idx = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="UTC") for d in ("2022-06-08", "2022-06-09", "2022-07-01")]
    )
    frame = pd.DataFrame({"gross_profitability": [1.0, 2.0, 3.0]}, index=idx)
    before = interval_slice(frame, "2017-01-01", "2022-06-09")
    after = interval_slice(frame, "2022-06-09", "")
    assert list(before["gross_profitability"]) == [1.0]
    assert list(after["gross_profitability"]) == [2.0, 3.0]
