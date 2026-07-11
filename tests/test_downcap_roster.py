import datetime
import zipfile
from pathlib import Path

import pandas as pd

from trading.venues.universes import downcap_roster as dr


def _zip_with(csv_text: str, dest: Path) -> Path:
    with zipfile.ZipFile(dest, "w") as zf:
        zf.writestr("supported_tickers.csv", csv_text)
    return dest


# Live Tiingo semantics: endDate is the LAST-DATA-DATE, populated for ACTIVE
# names too. MSFT carries a RECENT endDate (2024-01-01, the fixture's
# reference/max) -- the live-data case that flags 100% of the roster as
# delisted if mishandled. AAPL keeps an EMPTY endDate -- an anomaly under
# live data (which always populates it) -- to exercise NaN normalization and
# the "empty = not delisted" decision. OLDCO carries an OLD endDate: a
# genuinely delisted name.
CSV = (
    "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
    "AAPL,NASDAQ,Stock,USD,1990-01-01,\n"                 # keep: live, empty (anomaly) endDate
    "MSFT,NASDAQ,Stock,USD,1990-01-01,2024-01-01\n"       # keep: live, recent (== max) endDate
    "OLDCO,NYSE,Stock,USD,2001-01-01,2015-06-30\n"        # keep: delisted, old endDate
    "ARCAX,NYSE ARCA,Stock,USD,2010-01-01,2024-01-01\n"   # keep: NYSE ARCA, live
    "MKTCO,NYSE MKT,Stock,USD,2012-01-01,2024-01-01\n"    # keep: NYSE MKT, live
    "SPY,NYSE ARCA,ETF,USD,1993-01-01,2024-01-01\n"       # drop: assetType
    "FOREIGN,NASDAQ,Stock,CAD,2000-01-01,2024-01-01\n"    # drop: currency
    "PINKY,OTC,Stock,USD,2005-01-01,2024-01-01\n"         # drop: exchange
)


def test_parse_reads_all_columns_as_strings(tmp_path):
    zp = _zip_with(CSV, tmp_path / "st.zip")
    df = dr.parse_supported_tickers(zp)
    assert list(df.columns) == [
        "ticker", "exchange", "assetType", "priceCurrency", "startDate", "endDate"
    ]
    assert df.loc[df["ticker"] == "AAPL", "endDate"].iloc[0] == ""  # NaN normalized to ""


def test_structural_roster_applies_frozen_filters(tmp_path):
    zp = _zip_with(CSV, tmp_path / "st.zip")
    raw = dr.parse_supported_tickers(zp)
    roster, report = dr.structural_roster(raw)
    assert set(roster["ticker"]) == {"AAPL", "MSFT", "OLDCO", "ARCAX", "MKTCO"}
    assert report["dropped"] == {"OTC": 1}  # only the venue-dropped row lands here
    assert report["kept"]["NYSE ARCA"] == 1


def test_candidates_at_uses_listing_interval(tmp_path):
    zp = _zip_with(CSV, tmp_path / "st.zip")
    roster, _ = dr.structural_roster(dr.parse_supported_tickers(zp))
    # OLDCO delisted 2015-06-30: a candidate before, not after.
    assert "OLDCO" in dr.candidates_at(roster, datetime.date(2015, 1, 1))
    assert "OLDCO" not in dr.candidates_at(roster, datetime.date(2016, 1, 1))
    assert "AAPL" in dr.candidates_at(roster, datetime.date(2020, 1, 1))
    # Live-data case the old fixtures missed: an ACTIVE name with a RECENT
    # (non-empty) endDate must still be a candidate for a 2019-2023 decision
    # date -- candidates_at's [startDate, endDate] interval is correct as-is
    # for this case and must not be "fixed" alongside delisted_symbols.
    assert "MSFT" in dr.candidates_at(roster, datetime.date(2020, 1, 1))
    assert "MSFT" in dr.candidates_at(roster, datetime.date(2023, 12, 31))


def test_delisted_symbols_flags_old_enddate_not_mere_nonemptiness(tmp_path):
    zp = _zip_with(CSV, tmp_path / "st.zip")
    roster, _ = dr.structural_roster(dr.parse_supported_tickers(zp))
    # Only OLDCO (old endDate) is delisted. MSFT's endDate is non-empty but
    # RECENT (== the roster's max) -- live-data active, not delisted. AAPL's
    # empty endDate is the anomaly case -- also not delisted.
    assert dr.delisted_symbols(roster) == {"OLDCO"}


def test_delisted_excludes_active_names_with_recent_enddate():
    # The confirmed live bug: Tiingo's endDate is the LAST-DATA-DATE, set for
    # ACTIVE names too (AAPL/MSFT/NVDA all carry the file's fetch date, not
    # an empty string) -- so "endDate non-empty" alone flags ~100% of the
    # roster as delisted. A name is only delisted when its endDate trails
    # the roster's own inferred data-as-of date (the max endDate present) by
    # more than DELISTED_BUFFER_DAYS.
    roster = pd.DataFrame(
        {
            "ticker": ["ACTIVE", "DELISTED"],
            "endDate": ["2026-07-10", "2022-02-14"],  # ACTIVE == the roster's max
        }
    )
    delisted = dr.delisted_symbols(roster)
    assert "ACTIVE" not in delisted
    assert "DELISTED" in delisted


def test_delisted_reference_is_not_poisoned_by_future_enddate():
    # A single corrupt FUTURE-dated endDate row (a 2099 typo, plausible in a
    # 15k-row live file) must NOT poison the data-as-of reference. Without the
    # clamp, end.max() = 2099 pushes the cutoff decades forward and drags the
    # genuinely ACTIVE name (endDate == as_of) into "delisted" -- re-breaking
    # the survivorship metric. This test FAILS under the unclamped end.max().
    as_of = datetime.date(2026, 7, 10)
    roster = pd.DataFrame(
        {
            "ticker": ["ACTIVE", "OLDCO", "POISON"],
            "endDate": ["2026-07-10", "2015-06-30", "2099-01-01"],
        }
    )
    delisted = dr.delisted_symbols(roster, as_of=as_of)
    assert "ACTIVE" not in delisted   # endDate == as_of -> still active
    assert "OLDCO" in delisted        # old endDate -> genuinely delisted
    # POISON is future-of-everything; it may land either way, but it must not
    # have dragged ACTIVE in with it.


def test_fetch_supported_tickers_uses_download_seam(tmp_path, monkeypatch):
    def fake_download(url, dest):
        _zip_with(CSV, dest)

    monkeypatch.setattr(dr, "_download_zip", fake_download)
    out = dr.fetch_supported_tickers(tmp_path / "st.zip")
    assert out.exists()
    df = dr.parse_supported_tickers(out)
    assert "AAPL" in set(df["ticker"])
