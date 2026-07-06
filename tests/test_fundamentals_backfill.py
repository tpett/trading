import datetime
import sys
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from fundamentals_helpers import num_line, sub_line, write_quarter_zip
from trading.fundamentals.backfill import (
    backfill_from_companyfacts,
    backfill_quarters,
    last_complete_quarter,
    quarter_range,
)
from trading.fundamentals.companyfacts import COMPANYFACTS_URL
from trading.fundamentals.store import FundamentalsStore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import backfill_fundamentals as backfill_script  # noqa: E402

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


def _quarter_zip(
    tmp_path, name, adsh, fy, fp, period, filed, rev, cogs, assets, form="10-Q", cik=1326801
):
    subs = [sub_line(adsh, cik, form, period, fy, fp, filed)]
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
    assert stats == {"filers": 1, "symbols": 2, "rows": 4, "dropped": 0}
    fb = store.read("FB")
    newco = store.read("NEWCO")
    # Filed 2023-05-10 lands on FB (pre-rename); the rest on NEWCO.
    assert list(fb.index) == [pd.Timestamp("2023-05-10", tz="UTC")]
    assert len(newco) == 3
    # TTM continuity survives the split: the 10-K row completes 4 quarters.
    at_10k = newco.loc[pd.Timestamp("2024-02-20", tz="UTC")]
    assert at_10k["gross_profitability"] == (48.0 - 20.0) / 100.0
    assert store.read("OTHER").empty


def test_backfill_boundary_filing_lands_on_successor_symbol(tmp_path):
    # Filed exactly ON the rename date: start is inclusive, end exclusive, so
    # the row belongs to the successor (NEWCO) and never the predecessor (FB).
    zips = [
        _quarter_zip(
            tmp_path, "2023q3.zip", "f-01", "2023", "Q2", "20230630", "20230801", 12.0, 5.0, 95.0
        )
    ]
    store = FundamentalsStore(tmp_path / "store")
    stats = backfill_quarters(zips, CIK_MAP, store)
    assert store.read("FB").empty
    assert list(store.read("NEWCO").index) == [pd.Timestamp("2023-08-01", tz="UTC")]
    assert stats == {"filers": 1, "symbols": 1, "rows": 1, "dropped": 0}


def test_backfill_rerun_is_idempotent(tmp_path):
    store = FundamentalsStore(tmp_path / "store")
    zips = _fixture_zips(tmp_path)
    backfill_quarters(zips, CIK_MAP, store)
    first_run = {symbol: store.read(symbol) for symbol in ("FB", "NEWCO")}
    stats = backfill_quarters(zips, CIK_MAP, store)
    assert stats["rows"] == 0  # append-only store: reprocessing adds nothing
    # Stored VALUES are byte-identical too, not just the appended-row counter.
    for symbol, frame in first_run.items():
        pd.testing.assert_frame_equal(store.read(symbol), frame)


def test_backfill_merges_recycled_ticker_across_ciks(tmp_path):
    # Ticker reuse: two DIFFERENT companies (CIKs) hold the same symbol over
    # disjoint intervals -> one chronologically merged store, each row from
    # the CIK that owned the symbol at its FILED date.
    cik_map = pd.DataFrame(
        {
            "symbol": ["RECYCLE", "RECYCLE"],
            "cik": [111, 222],
            "start": ["2017-01-01", "2023-08-01"],
            "end": ["2023-08-01", ""],
        }
    )
    zips = [
        _quarter_zip(
            tmp_path,
            "2023q2.zip",
            "a-01",
            "2023",
            "Q1",
            "20230331",
            "20230510",
            10.0,
            4.0,
            90.0,
            cik=111,
        ),
        _quarter_zip(
            tmp_path,
            "2023q4.zip",
            "b-01",
            "2023",
            "Q3",
            "20230930",
            "20231108",
            11.0,
            5.0,
            98.0,
            cik=222,
        ),
    ]
    store = FundamentalsStore(tmp_path / "store")
    stats = backfill_quarters(zips, cik_map, store)
    assert stats == {"filers": 2, "symbols": 1, "rows": 2, "dropped": 0}
    frame = store.read("RECYCLE")
    assert list(frame.index) == [
        pd.Timestamp("2023-05-10", tz="UTC"),
        pd.Timestamp("2023-11-08", tz="UTC"),
    ]
    assert list(frame["adsh"]) == ["a-01", "b-01"]


def _cf_payload(
    rev,
    cogs,
    assets,
    adsh="a-01",
    period="2023-03-31",
    start="2023-01-01",
    fy=2023,
    fp="Q1",
    form="10-Q",
    filed="2023-05-10",
):
    def entry(val, **kw):
        e = {
            "end": period,
            "val": val,
            "accn": adsh,
            "fy": fy,
            "fp": fp,
            "form": form,
            "filed": filed,
        }
        e.update(kw)
        return e

    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [entry(rev, start=start)]}},
                "CostOfRevenue": {"units": {"USD": [entry(cogs, start=start)]}},
                "Assets": {"units": {"USD": [entry(assets)]}},
            }
        }
    }


def test_backfill_from_companyfacts_fetches_once_per_cik_and_splits_by_symbol(tmp_path):
    # CIK 1326801: FB (pre-rename) / NEWCO (post-rename) share one fetch;
    # CIK 555 (OTHER) is a second, independent fetch.
    payloads = {
        1326801: _cf_payload(10.0, 4.0, 90.0, adsh="a-01", filed="2023-05-10"),
        555: _cf_payload(20.0, 8.0, 190.0, adsh="b-01", filed="2023-06-01"),
    }
    calls: list[str] = []

    def fetch(url: str) -> dict:
        calls.append(url)
        cik = int(url.rsplit("CIK", 1)[1].split(".")[0])
        return payloads[cik]

    store = FundamentalsStore(tmp_path / "store")
    progress_calls: list[tuple[int, int]] = []
    stats = backfill_from_companyfacts(
        CIK_MAP, store, fetch_json=fetch, on_progress=lambda i, n: progress_calls.append((i, n))
    )
    assert stats == {"filers": 2, "symbols": 2, "rows": 2, "dropped": 0, "failed": 0}
    # One fetch per unique cik, sorted ascending, not per symbol/row.
    assert calls == [COMPANYFACTS_URL.format(cik=555), COMPANYFACTS_URL.format(cik=1326801)]
    assert progress_calls == [(1, 2), (2, 2)]
    assert list(store.read("FB").index) == [pd.Timestamp("2023-05-10", tz="UTC")]
    assert store.read("NEWCO").empty  # NEWCO's interval starts after this filing
    assert list(store.read("OTHER").index) == [pd.Timestamp("2023-06-01", tz="UTC")]


def test_backfill_from_companyfacts_is_fail_open_per_cik(tmp_path):
    payloads = {1326801: _cf_payload(10.0, 4.0, 90.0, filed="2023-05-10")}

    def fetch(url: str) -> dict:
        if "0000000555" in url:
            raise OSError("edgar down")
        return payloads[1326801]

    store = FundamentalsStore(tmp_path / "store")
    stats = backfill_from_companyfacts(CIK_MAP, store, fetch_json=fetch)
    assert stats["failed"] == 1
    assert stats["filers"] == 1
    # FB still gets its data despite OTHER's cik failing.
    assert list(store.read("FB").index) == [pd.Timestamp("2023-05-10", tz="UTC")]
    assert store.read("OTHER").empty


def test_ensure_empty_for_rebuild_aborts_when_store_has_files(tmp_path):
    (tmp_path / "AAPL.parquet").write_bytes(b"not-really-parquet")
    with pytest.raises(SystemExit, match="EMPTY store"):
        backfill_script._ensure_empty_for_rebuild(tmp_path)


def test_ensure_empty_for_rebuild_passes_when_store_is_empty(tmp_path):
    backfill_script._ensure_empty_for_rebuild(tmp_path)  # no raise
    (tmp_path / "not-a-parquet.txt").write_text("x")
    backfill_script._ensure_empty_for_rebuild(tmp_path)  # still no raise: only *.parquet counts


def test_ensure_zips_allowed_refuses_when_marker_says_companyfacts(tmp_path):
    (tmp_path / backfill_script.SOURCE_MARKER).write_text("companyfacts")
    with pytest.raises(SystemExit, match="companyfacts"):
        backfill_script._ensure_zips_allowed(tmp_path)


def test_ensure_zips_allowed_proceeds_with_no_marker_or_its_own_marker(tmp_path):
    backfill_script._ensure_zips_allowed(tmp_path)  # no marker yet: no raise
    (tmp_path / backfill_script.SOURCE_MARKER).write_text("zips")
    backfill_script._ensure_zips_allowed(tmp_path)  # its own marker: still no raise


def test_companyfacts_backfill_writes_source_marker(tmp_path):
    # Same shape as scripts/backfill_fundamentals.py's companyfacts branch:
    # empty-store guard, THEN the marker write, THEN the backfill itself --
    # the marker precedes the per-CIK loop (see the interrupt-simulation
    # test below for why) -- all against tmp store fixtures, no network.
    store_root = tmp_path / "store"
    backfill_script._ensure_empty_for_rebuild(store_root)
    store = FundamentalsStore(store_root)  # creates store_root, like main() does
    backfill_script._write_source_marker(store_root, "companyfacts")
    payloads = {
        1326801: _cf_payload(10.0, 4.0, 90.0, adsh="a-01", filed="2023-05-10"),
        555: _cf_payload(20.0, 8.0, 190.0, adsh="b-01", filed="2023-06-01"),
    }

    def fetch(url: str) -> dict:
        cik = int(url.rsplit("CIK", 1)[1].split(".")[0])
        return payloads[cik]

    backfill_from_companyfacts(CIK_MAP, store, fetch_json=fetch)
    assert (store_root / backfill_script.SOURCE_MARKER).read_text() == "companyfacts"
    # A stale marker from a prior source is overwritten, not left in place.
    backfill_script._write_source_marker(store_root, "companyfacts")
    assert (store_root / backfill_script.SOURCE_MARKER).read_text() == "companyfacts"


def test_interrupted_companyfacts_backfill_still_leaves_marker_guarding_zips(tmp_path):
    # Regression for the marker-write timing bug: it used to be written only
    # AFTER backfill_from_companyfacts returned, so an interrupt mid per-CIK
    # loop left a non-empty, marker-less store that a subsequent
    # --source zips run would treat as unguarded (silently mixing regimes).
    # Simulated here with a fetch_json that raises KeyboardInterrupt (a
    # BaseException, NOT caught by backfill_from_companyfacts's per-cik
    # `except Exception` fail-open) partway through the loop -- exactly like
    # a real Ctrl-C landing mid-fetch.
    store_root = tmp_path / "store"
    backfill_script._ensure_empty_for_rebuild(store_root)
    store = FundamentalsStore(store_root)  # creates store_root, like main() does
    backfill_script._write_source_marker(store_root, "companyfacts")  # written BEFORE the loop

    def fetch(url: str) -> dict:
        raise KeyboardInterrupt("simulated interrupt mid per-cik loop")

    with pytest.raises(KeyboardInterrupt):
        backfill_from_companyfacts(CIK_MAP, store, fetch_json=fetch)

    # The store is guarded even though the backfill never finished (and
    # never wrote a single row).
    assert (store_root / backfill_script.SOURCE_MARKER).read_text() == "companyfacts"
    with pytest.raises(SystemExit, match="companyfacts"):
        backfill_script._ensure_zips_allowed(store_root)


def test_write_source_marker_overwrites_stale_marker(tmp_path):
    (tmp_path / backfill_script.SOURCE_MARKER).write_text("zips")
    backfill_script._write_source_marker(tmp_path, "companyfacts")
    assert (tmp_path / backfill_script.SOURCE_MARKER).read_text() == "companyfacts"


def test_quarter_range_and_last_complete_quarter():
    assert quarter_range("2018q1", "2018q4") == ["2018q1", "2018q2", "2018q3", "2018q4"]
    assert quarter_range("2018q3", "2019q2") == ["2018q3", "2018q4", "2019q1", "2019q2"]
    assert last_complete_quarter(datetime.date(2026, 7, 6)) == "2026q2"
    assert last_complete_quarter(datetime.date(2026, 2, 1)) == "2025q4"


def _raise_http(code):
    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, code, "err", None, None)

    return fake_urlopen


def test_download_range_404_on_newest_quarter_skips(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(backfill_script, "RAW_DIR", tmp_path)
    monkeypatch.setattr(backfill_script.urllib.request, "urlopen", _raise_http(404))
    # The NEWEST quarter is the only one SEC may not have published yet:
    # warn + skip, no abort.
    assert backfill_script.download_range(["2026q2"]) == []
    assert "not published yet" in capsys.readouterr().out


def test_download_range_404_on_older_quarter_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill_script, "RAW_DIR", tmp_path)
    monkeypatch.setattr(backfill_script.urllib.request, "urlopen", _raise_http(404))
    # 2018q1 is definitely published: its 404 means a broken URL, never a
    # silent skip that would produce an incomplete backfill with exit 0.
    with pytest.raises(SystemExit) as excinfo:
        backfill_script.download_range(["2018q1", "2018q2"])
    assert backfill_script.ZIP_URL.format(quarter="2018q1") in str(excinfo.value)


def test_download_range_non_404_error_aborts_even_on_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill_script, "RAW_DIR", tmp_path)
    monkeypatch.setattr(backfill_script.urllib.request, "urlopen", _raise_http(500))
    with pytest.raises(SystemExit) as excinfo:
        backfill_script.download_range(["2026q2"])
    assert backfill_script.ZIP_URL.format(quarter="2026q2") in str(excinfo.value)
