"""scripts/build_sic_map.py: offline unit tests on the mocked fetch seam,
plus committed-artifact anchors (same pattern as tests/test_fundamentals_cik_map.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_sic_map  # noqa: E402

SIC_MAP_CSV = (
    Path(__file__).resolve().parent.parent
    / "src" / "trading" / "venues" / "universes" / "sic_map.csv"
)


def _cik_map(rows: list[tuple[str, int, str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["symbol", "cik", "start", "end"])
    return frame.astype({"cik": str})


def test_window_pairs_selects_intervals_overlapping_the_discovery_window():
    frame = _cik_map(
        [
            ("OLD", 111, "2017-01-01", "2018-06-01"),  # ends pre-window: skipped
            ("FB", 222, "2017-01-01", "2022-06-09"),   # overlaps: chosen
            ("META", 222, "2022-06-09", ""),           # overlaps (open end): chosen
            ("NEWCO", 333, "2024-05-01", ""),          # starts post-window: skipped
        ]
    )
    assert build_sic_map.window_pairs(frame) == [("FB", 222), ("META", 222)]


def test_window_pairs_first_overlapping_interval_wins_deterministically():
    # Dict-in-insertion-order dedupe (Piece 1 lesson: never rely on sort
    # stability): the FIRST overlapping interval in file order is the one used.
    frame = _cik_map(
        [
            ("DUAL", 111, "2017-01-01", "2020-01-01"),
            ("DUAL", 999, "2020-01-01", ""),
        ]
    )
    assert build_sic_map.window_pairs(frame) == [("DUAL", 111)]


def test_build_rows_parses_sic_and_fetches_each_cik_once():
    calls: list[str] = []

    def fetch(url: str) -> dict:
        calls.append(url)
        return {"sic": "2836", "sicDescription": "Biological Products, (No Diagnostic)"}

    rows, unmapped = build_sic_map.build_rows(
        [("GOOG", 1652044), ("GOOGL", 1652044)], fetch, fetched_at="2026-07-08"
    )
    assert unmapped == []
    assert rows == [
        ("GOOG", 1652044, 2836, "Biological Products, (No Diagnostic)", "2026-07-08"),
        ("GOOGL", 1652044, 2836, "Biological Products, (No Diagnostic)", "2026-07-08"),
    ]
    # GOOG/GOOGL share one CIK: exactly one request.
    assert calls == [build_sic_map.SUBMISSIONS_URL.format(cik=1652044)]


def test_fetch_sic_retries_once_then_succeeds():
    attempts = {"n": 0}

    def flaky(url: str) -> dict:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("edgar hiccup")
        return {"sic": "6022", "sicDescription": "State Commercial Banks"}

    assert build_sic_map.fetch_sic(1, flaky) == (6022, "State Commercial Banks")
    assert attempts["n"] == 2


def test_fetch_sic_two_failures_is_unmapped():
    def boom(url: str) -> dict:
        raise OSError("edgar down")

    assert build_sic_map.fetch_sic(1, boom) is None


def test_filer_without_sic_is_recorded_unmapped_never_guessed():
    rows, unmapped = build_sic_map.build_rows(
        [("XX", 1)], lambda url: {"sic": "", "sicDescription": ""}, fetched_at="2026-07-08"
    )
    assert rows == []
    assert unmapped == ["XX"]


def test_default_fetch_seam_is_the_throttled_companyfacts_one():
    # SEC policy: 0.11 s process-global throttle + mandatory User-Agent live
    # in ONE place; this script must ride that seam, not roll its own.
    from trading.fundamentals import companyfacts

    assert build_sic_map.http_get_json is companyfacts.http_get_json


def test_validate_exits_nonzero_below_90_percent_coverage():
    rows = [("A", 1, 2836, "Biological Products", "2026-07-08")]
    members = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"}  # 1/11 mapped
    with pytest.raises(SystemExit) as excinfo:
        build_sic_map.validate(rows, members)
    assert "90%" in str(excinfo.value)


def test_validate_passes_at_full_coverage(capsys):
    rows = [("A", 1, 2836, "Biological Products", "2026-07-08")]
    build_sic_map.validate(rows, {"A"})
    assert "coverage OK" in capsys.readouterr().out


def test_write_csv_round_trips_comma_bearing_descriptions(tmp_path):
    # SEC sicDescription values contain commas; string-joined CSV would tear.
    out = tmp_path / "sic_map.csv"
    rows = [
        ("XX", 1, 7370, "Services-Computer Programming, Data Processing, Etc.", "2026-07-08")
    ]
    build_sic_map.write_csv(rows, out)
    df = pd.read_csv(out, comment="#", dtype=str)
    assert list(df.columns) == ["symbol", "cik", "sic", "sic_description", "fetched_at"]
    assert df.iloc[0]["sic_description"] == (
        "Services-Computer Programming, Data Processing, Etc."
    )


# --- committed artifact (exists only after the real generation run) ----------


def test_committed_sic_map_shape_and_anchors():
    df = pd.read_csv(SIC_MAP_CSV, comment="#", dtype=str)
    assert list(df.columns) == ["symbol", "cik", "sic", "sic_description", "fetched_at"]
    assert len(df) > 900  # ~1130 window symbols minus the unmapped tail
    by_symbol = dict(zip(df["symbol"], df["sic"].astype(int), strict=True))
    assert by_symbol["AAPL"] == 3571          # Electronic Computers
    assert by_symbol["AMGN"] == 2836          # Biological Products: biotech anchor
    assert 6020 <= by_symbol["JPM"] <= 6039   # commercial bank: banks anchor
    assert df["symbol"].is_unique             # one row per symbol
