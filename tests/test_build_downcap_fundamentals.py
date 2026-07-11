"""Unit tests for the down-cap fundamentals orchestrator's roster->guarded-
function adaptation, the merge, and the shares-store isolation. The FSDS
download/parse/verify network flow is validated by RUN-TIME spot checks
(--validate), NOT here -- these tests never touch SEC."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_downcap_fundamentals as m  # noqa: E402


def _roster() -> pd.DataFrame:
    # AAA: listed across the whole window (open endDate).
    # BBB: mid-window listing that delists inside the window.
    # CCC: delisted BEFORE the window -> not a candidate.
    # DDD: listed AFTER the window -> not a candidate.
    # EEE: still-active name -> endDate is Tiingo's post-window file BUILD date
    #      (the NORMAL live case, never empty); end must clamp to window.
    # FFF: empty endDate -> the degenerate-anomaly fallback branch.
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "exchange": ["NYSE"] * 6,
            "assetType": ["Stock"] * 6,
            "priceCurrency": ["USD"] * 6,
            "startDate": [
                "2018-01-01",
                "2020-06-01",
                "2010-01-01",
                "2025-01-01",
                "2019-01-01",
                "2018-01-01",
            ],
            "endDate": ["2024-06-01", "2021-03-15", "2015-06-30", "2026-01-01", "2099-01-01", ""],
        }
    )


def test_downcap_fundamentals_roster_candidates_window_scoped():
    candidates = m.roster_candidates(_roster())
    assert candidates == ["AAA", "BBB", "EEE", "FFF"]  # sorted; CCC/DDD out of window


def test_downcap_fundamentals_tenure_shape_matches_resolve_target():
    roster = _roster()
    tenure = m.roster_tenure(roster, m.roster_candidates(roster))
    # Same (lo, hi) 2-tuple-of-str shape membership_tenure yields.
    for value in tenure.values():
        assert isinstance(value, tuple) and len(value) == 2
        assert all(isinstance(x, str) for x in value)
    # AAA: active name, post-window build-date endDate -> clamped to window end
    assert tenure["AAA"] == ("2018-01-01", m.DISCOVERY_END)
    assert tenure["BBB"] == ("2020-06-01", "2021-03-15")  # in-window delist kept
    assert tenure["EEE"] == ("2019-01-01", m.DISCOVERY_END)  # far-future endDate clamped
    assert tenure["FFF"] == ("2018-01-01", m.DISCOVERY_END)  # empty-endDate anomaly fallback


def test_downcap_fundamentals_tenure_usable_by_qualifying_candidates():
    # The tenure tuple must plug straight into the guarded resolver's filter.
    from build_cik_map_historical import qualifying_candidates

    tenure = m.roster_tenure(_roster(), ["BBB"])["BBB"]
    cands = {
        111: {"name": "B Co", "filed": ["2020-11-01"]},
        222: {"name": "X", "filed": ["2018-01-01"]},
    }
    got = qualifying_candidates(cands, tenure)
    assert got == [(111, "B Co")]  # only the filing inside BBB's tenure qualifies


def test_downcap_fundamentals_historical_targets_excludes_mapped_and_recycled():
    excluded = next(iter(m.EXCLUSIONS))  # a real recycled ticker (e.g. APC)
    candidates = ["AAA", "BBB", "CCC", excluded]
    current_map = pd.DataFrame(
        {"symbol": ["AAA"], "cik": [1], "start": ["2017-01-01"], "end": [""]}
    )
    assert m.historical_targets(candidates, current_map) == ["BBB", "CCC"]


def test_downcap_fundamentals_merge_produces_valid_cik_map(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "CURRENT_MAP", tmp_path / "cik_map_current.csv")
    monkeypatch.setattr(m, "HISTORICAL_MAP", tmp_path / "cik_map_historical.csv")
    monkeypatch.setattr(m, "CIK_MAP", tmp_path / "cik_map.csv")
    excluded = next(iter(m.EXCLUSIONS))

    m._write_map(
        m.CURRENT_MAP,
        [("AAA", 111, "2017-01-01", ""), ("BBB", 222, "2017-01-01", "2021-03-15")],
        ["# current"],
    )
    m.HISTORICAL_MAP.write_text(
        "symbol,cik,start,end,name\n"
        "ZZZ,333,2017-01-01,2020-01-01,\"Zed Co\"\n"  # kept
        f"{excluded},999,2017-01-01,2020-01-01,\"Recycled\"\n"  # skipped (EXCLUSIONS)
        "AAA,444,2018-01-01,2019-01-01,\"Overlap\"\n"  # skipped (overlaps current AAA)
    )

    m.stage_merge()

    merged = m.load_cik_map(m.CIK_MAP)
    assert list(merged.columns) == ["symbol", "cik", "start", "end"]
    by_symbol = {r.symbol: (int(r.cik), r.start, r.end) for r in merged.itertuples()}
    assert "ZZZ" in by_symbol and by_symbol["ZZZ"][0] == 333
    assert excluded not in by_symbol  # recycled ticker never leaks in
    assert by_symbol["AAA"] == (111, "2017-01-01", "")  # current wins over overlap
    # intervals sane: start < end (or open end)
    for _cik, start, end in by_symbol.values():
        assert end == "" or start < end
    # deterministic sort by (symbol, start)
    keys = [(r.symbol, r.start) for r in merged.itertuples()]
    assert keys == sorted(keys)


def test_downcap_fundamentals_store_is_isolated_from_index_store():
    assert m.STORE_DIR.resolve() != m.INDEX_STORE_DIR.resolve()
    assert m.STORE_DIR.name == "equities-downcap"
    assert m.INDEX_STORE_DIR.name == "equities"


def test_downcap_fundamentals_shares_writes_own_marker_and_leaves_index_store(
    tmp_path, monkeypatch
):
    store_dir = tmp_path / "equities-downcap"
    index_dir = tmp_path / "equities"
    index_dir.mkdir()
    (index_dir / m.SOURCE_MARKER).write_text("companyfacts")  # untouched sentinel
    cik_map_path = tmp_path / "cik_map.csv"
    m._write_map(cik_map_path, [("AAA", 111, "2017-01-01", "")], ["# map"])

    monkeypatch.setattr(m, "STORE_DIR", store_dir)
    monkeypatch.setattr(m, "INDEX_STORE_DIR", index_dir)
    monkeypatch.setattr(m, "CIK_MAP", cik_map_path)

    calls: dict[str, object] = {}

    def fake_backfill(cik_map, store, on_progress=None):
        calls["root"] = store._root
        calls["ciks"] = set(cik_map["cik"])
        return {"filers": 1, "symbols": 1, "rows": 1, "dropped": 0, "failed": 0}

    monkeypatch.setattr(m, "backfill_from_companyfacts", fake_backfill)
    m.stage_shares()

    assert calls["root"] == store_dir  # backfill ran against the down-cap store only
    assert calls["ciks"] == {111}
    assert (store_dir / m.SOURCE_MARKER).read_text() == "companyfacts"
    # index store's own marker untouched, no down-cap files written into it
    assert (index_dir / m.SOURCE_MARKER).read_text() == "companyfacts"
    assert not list(index_dir.glob("*.parquet"))


def test_downcap_fundamentals_shares_refuses_foreign_source_marker(tmp_path, monkeypatch):
    store_dir = tmp_path / "equities-downcap"
    store_dir.mkdir()
    (store_dir / m.SOURCE_MARKER).write_text("zips")  # a different regime owns it
    cik_map_path = tmp_path / "cik_map.csv"
    m._write_map(cik_map_path, [("AAA", 111, "2017-01-01", "")], ["# map"])
    monkeypatch.setattr(m, "STORE_DIR", store_dir)
    monkeypatch.setattr(m, "CIK_MAP", cik_map_path)
    monkeypatch.setattr(m, "backfill_from_companyfacts", lambda *a, **k: pytest.fail("ran"))
    with pytest.raises(SystemExit):
        m.stage_shares()


def test_downcap_fundamentals_shares_refuses_nonempty_store(tmp_path, monkeypatch):
    # Mirrors the index companyfacts guard: append-only store means a rebuild
    # on top of existing rows would keep stale (possibly wrong-CIK) shares.
    store_dir = tmp_path / "equities-downcap"
    store_dir.mkdir()
    (store_dir / "AAA.parquet").write_bytes(b"stale")
    cik_map_path = tmp_path / "cik_map.csv"
    m._write_map(cik_map_path, [("AAA", 111, "2017-01-01", "")], ["# map"])
    monkeypatch.setattr(m, "STORE_DIR", store_dir)
    monkeypatch.setattr(m, "CIK_MAP", cik_map_path)
    monkeypatch.setattr(m, "backfill_from_companyfacts", lambda *a, **k: pytest.fail("ran"))
    with pytest.raises(SystemExit, match="EMPTY store"):
        m.stage_shares()


def test_downcap_fundamentals_shares_refuses_index_store_collision(tmp_path, monkeypatch):
    cik_map_path = tmp_path / "cik_map.csv"
    m._write_map(cik_map_path, [("AAA", 111, "2017-01-01", "")], ["# map"])
    monkeypatch.setattr(m, "STORE_DIR", tmp_path / "same")
    monkeypatch.setattr(m, "INDEX_STORE_DIR", tmp_path / "same")
    monkeypatch.setattr(m, "CIK_MAP", cik_map_path)
    monkeypatch.setattr(m, "backfill_from_companyfacts", lambda *a, **k: pytest.fail("ran"))
    with pytest.raises(SystemExit):
        m.stage_shares()


def test_downcap_fundamentals_validate_checks_known_ciks(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "CIK_MAP", tmp_path / "cik_map.csv")
    rows = [
        ("AAPL", 320193, "2017-01-01", ""),
        ("XLNX", 743988, "2017-01-01", "2022-02-14"),
        ("ATVI", 718877, "2017-01-01", "2023-10-13"),
    ]
    m._write_map(m.CIK_MAP, rows, ["# map"])
    assert m.run_validate() == 0  # all known CIKs match

    m._write_map(m.CIK_MAP, [("AAPL", 999, "2017-01-01", "")], ["# map"])
    assert m.run_validate() == 1  # mismatch -> nonzero
