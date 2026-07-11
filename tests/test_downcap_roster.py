import datetime
import zipfile
from pathlib import Path

from trading.venues.universes import downcap_roster as dr


def _zip_with(csv_text: str, dest: Path) -> Path:
    with zipfile.ZipFile(dest, "w") as zf:
        zf.writestr("supported_tickers.csv", csv_text)
    return dest


CSV = (
    "ticker,exchange,assetType,priceCurrency,startDate,endDate\n"
    "AAPL,NASDAQ,Stock,USD,1990-01-01,\n"                 # keep: live
    "OLDCO,NYSE,Stock,USD,2001-01-01,2015-06-30\n"        # keep: delisted
    "ARCAX,NYSE ARCA,Stock,USD,2010-01-01,\n"             # keep: NYSE ARCA
    "MKTCO,NYSE MKT,Stock,USD,2012-01-01,\n"              # keep: NYSE MKT
    "SPY,NYSE ARCA,ETF,USD,1993-01-01,\n"                 # drop: assetType
    "FOREIGN,NASDAQ,Stock,CAD,2000-01-01,\n"              # drop: currency
    "PINKY,OTC,Stock,USD,2005-01-01,\n"                   # drop: exchange
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
    assert set(roster["ticker"]) == {"AAPL", "OLDCO", "ARCAX", "MKTCO"}
    assert report["dropped"] == {"OTC": 1}  # only the venue-dropped row lands here
    assert report["kept"]["NYSE ARCA"] == 1


def test_candidates_at_uses_listing_interval(tmp_path):
    zp = _zip_with(CSV, tmp_path / "st.zip")
    roster, _ = dr.structural_roster(dr.parse_supported_tickers(zp))
    # OLDCO delisted 2015-06-30: a candidate before, not after.
    assert "OLDCO" in dr.candidates_at(roster, datetime.date(2015, 1, 1))
    assert "OLDCO" not in dr.candidates_at(roster, datetime.date(2016, 1, 1))
    assert "AAPL" in dr.candidates_at(roster, datetime.date(2020, 1, 1))


def test_delisted_symbols_flags_nonempty_enddate(tmp_path):
    zp = _zip_with(CSV, tmp_path / "st.zip")
    roster, _ = dr.structural_roster(dr.parse_supported_tickers(zp))
    assert dr.delisted_symbols(roster) == {"OLDCO"}


def test_fetch_supported_tickers_uses_download_seam(tmp_path, monkeypatch):
    def fake_download(url, dest):
        _zip_with(CSV, dest)

    monkeypatch.setattr(dr, "_download_zip", fake_download)
    out = dr.fetch_supported_tickers(tmp_path / "st.zip")
    assert out.exists()
    df = dr.parse_supported_tickers(out)
    assert "AAPL" in set(df["ticker"])
