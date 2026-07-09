"""scripts/build_insider_store.py: offline unit tests on synthetic
transactions + cik_map fixtures (the build_cik_map_historical test pattern:
pure functions, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_insider_store as bis  # noqa: E402

from trading.fundamentals.form345 import INSIDER_COLUMNS  # noqa: E402


def _tx(rows: list[tuple]) -> pd.DataFrame:
    """(accession, issuer_cik, filed_iso, code, shares, price, owner_cik,
    is_officer) -> a TRANSACTION_COLUMNS-shaped frame."""
    out = pd.DataFrame(
        rows,
        columns=["accession", "issuer_cik", "filed", "code", "shares", "price",
                 "owner_cik", "is_officer"],
    )
    out["filed"] = pd.to_datetime(out["filed"]).dt.tz_localize("UTC")
    out["trans_date"] = out["filed"] - pd.Timedelta(2, unit="D")
    out["value"] = out["shares"] * out["price"]
    out["is_director"] = False
    out["is_ten_pct"] = False
    return out


def _cik_map(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["symbol", "cik", "start", "end"]).astype(str)
    df["cik"] = df["cik"].astype(int)
    return df


def test_map_to_symbols_uses_filed_date_intervals():
    # FB/META shape: one cik, two symbol intervals; the FILED date selects.
    tx = _tx([
        ("a1", 1326801, "2021-06-01", "P", 100.0, 10.0, 9001, True),
        ("a2", 1326801, "2022-06-01", "P", 200.0, 10.0, 9002, False),
    ])
    cmap = _cik_map([
        ("FB", 1326801, "2017-01-01", "2022-06-09"),
        ("META", 1326801, "2022-06-09", ""),
    ])
    frames, unmapped_rows, unmapped_ciks = bis.map_to_symbols(tx, cmap)
    assert set(frames) == {"FB"}             # both filings pre-rename
    assert len(frames["FB"]) == 2
    assert unmapped_rows == 0 and unmapped_ciks == 0
    assert list(frames["FB"].columns) == INSIDER_COLUMNS
    assert frames["FB"].index.name == "filed"
    assert str(frames["FB"].index.tz) == "UTC"


def test_map_to_symbols_counts_unmapped_never_guesses():
    tx = _tx([
        ("a1", 999999, "2021-06-01", "P", 100.0, 10.0, 9001, True),   # unknown cik
        ("a2", 55, "2010-06-01", "S", 50.0, 10.0, 9002, False),       # pre-interval
        ("a3", 55, "2021-06-01", "S", 50.0, 10.0, 9002, False),       # mapped
    ])
    cmap = _cik_map([("XCO", 55, "2017-01-01", "")])
    frames, unmapped_rows, unmapped_ciks = bis.map_to_symbols(tx, cmap)
    assert set(frames) == {"XCO"}
    assert len(frames["XCO"]) == 1
    assert unmapped_rows == 2
    assert unmapped_ciks == 2                # 999999 entirely + 55 partially


def test_map_to_symbols_shared_cik_maps_to_both_symbols():
    # GOOG/GOOGL: two concurrent intervals share one cik -- both symbols get
    # the row (the fundamentals backfill rule); nothing is unmapped.
    tx = _tx([("a1", 1652044, "2021-06-01", "P", 10.0, 100.0, 9001, True)])
    cmap = _cik_map([
        ("GOOG", 1652044, "2017-01-01", ""),
        ("GOOGL", 1652044, "2017-01-01", ""),
    ])
    frames, unmapped_rows, _ = bis.map_to_symbols(tx, cmap)
    assert set(frames) == {"GOOG", "GOOGL"}
    assert unmapped_rows == 0


def test_write_store_atomic_layout_and_marker(tmp_path):
    tx = _tx([
        ("a1", 55, "2021-06-01", "P", 100.0, 10.0, 9001, True),
        ("a2", 55, "2021-03-01", "S", 40.0, 10.0, 9002, False),
    ])
    cmap = _cik_map([("BRK/B", 55, "2017-01-01", "")])
    frames, _, _ = bis.map_to_symbols(tx, cmap)
    store = tmp_path / "insider" / "equities"
    bis.write_store(frames, store)
    path = store / "BRK-B.parquet"           # '/' sanitized like FundamentalsStore
    assert path.exists()
    got = pd.read_parquet(path)
    assert list(got.columns) == INSIDER_COLUMNS
    assert got.index.is_monotonic_increasing  # sorted by filed
    assert (store / ".source").read_text() == "form345"
    assert not list(store.glob("*.tmp"))     # atomic: no torn files left


def test_ensure_empty_refuses_a_populated_store(tmp_path):
    store = tmp_path / "equities"
    store.mkdir(parents=True)
    (store / "AAA.parquet").write_bytes(b"")
    with pytest.raises(SystemExit, match="Delete"):
        bis.ensure_empty(store)
    (store / "AAA.parquet").unlink()
    bis.ensure_empty(store)                  # empty dir: fine
    bis.ensure_empty(tmp_path / "missing")   # absent dir: fine


def test_gap_recording_shape():
    # main() records failed quarters and exits 1 -- the pure helper just
    # formats; pin the message carries quarter + reason (loud, greppable).
    line = bis.gap_line("2020q2", ValueError("boom"))
    assert "2020q2" in line and "ValueError" in line and "boom" in line
