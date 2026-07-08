"""Offline tests for the raw-underlying backfill script.

The Tiingo fetch is injected, so nothing touches the network. Covers the row
parser, incremental skip / --force, and fail-open handling of empty / error
responses.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from backfill_options_underlying_raw import parse_raw_close, run_backfill  # noqa: E402


def _rows(*pairs: tuple[str, float]) -> list[dict]:
    return [{"date": f"{d}T00:00:00.000Z", "close": c} for d, c in pairs]


def test_parse_raw_close_shapes_the_frame():
    df = parse_raw_close(_rows(("2019-02-01", 166.52), ("2019-02-04", 171.25)))
    assert list(df.columns) == ["close_raw"]
    assert df.index.name == "Date"
    assert str(df.index.tz) == "UTC"
    assert df["close_raw"].iloc[0] == 166.52  # RAW, not adjusted


def test_run_backfill_writes_then_skips_then_forces(tmp_path: Path):
    calls: list[str] = []

    def fetch(url, params):
        calls.append(url)
        return 200, json.dumps(_rows(("2019-02-01", 166.52))).encode()

    s1 = run_backfill(["AAPL"], tmp_path, date(2019, 1, 1), date(2019, 3, 1), fetch=fetch)
    assert s1["pulled"] == 1
    assert (tmp_path / "AAPL.parquet").exists()

    # Already cached -> skipped, not re-fetched.
    s2 = run_backfill(["AAPL"], tmp_path, date(2019, 1, 1), date(2019, 3, 1), fetch=fetch)
    assert s2 == {"symbols": 1, "pulled": 0, "skipped": 1, "empty": 0, "errors": 0}
    assert len(calls) == 1

    # --force re-pulls.
    s3 = run_backfill(
        ["AAPL"], tmp_path, date(2019, 1, 1), date(2019, 3, 1), force=True, fetch=fetch
    )
    assert s3["pulled"] == 1
    assert len(calls) == 2


def test_run_backfill_fail_open_on_empty_and_error(tmp_path: Path):
    def fetch(url, params):
        if "EMPTY" in url:
            return 200, b"[]"  # 200 but no rows
        return 404, b"not found"  # non-200

    summary = run_backfill(
        ["EMPTY", "BAD"], tmp_path, date(2019, 1, 1), date(2019, 3, 1), fetch=fetch
    )
    assert summary["empty"] == 1
    assert summary["errors"] == 1
    assert summary["pulled"] == 0
    assert not (tmp_path / "EMPTY.parquet").exists()
    assert not (tmp_path / "BAD.parquet").exists()
