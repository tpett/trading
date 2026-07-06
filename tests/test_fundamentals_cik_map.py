import datetime
import sys
from pathlib import Path

import pandas as pd

from trading.fundamentals.cik_map import cik_for, interval_slice, load_cik_map

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_cik_map  # noqa: E402

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


# --- build_rows chain logic (offline unit tests on synthetic fixtures) -------
# build_rows reads the module-global RENAMES table, so each test swaps in a
# synthetic table via monkeypatch. No network: current_tickers is a dict.


def test_build_rows_two_hop_chain(monkeypatch):
    monkeypatch.setattr(build_cik_map, "RENAMES", [("AAA", "BBB", "2020-01-01")])
    rows, unmapped = build_cik_map.build_rows({"AAA", "BBB"}, {"BBB": 111})
    assert unmapped == []
    assert rows == [
        ("AAA", 111, "2017-01-01", "2020-01-01"),
        ("BBB", 111, "2020-01-01", ""),
    ]


def test_build_rows_three_hop_chain_boundaries_touch(monkeypatch):
    # Mirrors HCP->PEAK->DOC: each interval's end is exactly the next's start,
    # all three symbols share the terminal ticker's CIK.
    monkeypatch.setattr(
        build_cik_map,
        "RENAMES",
        [("AAA", "BBB", "2019-01-01"), ("BBB", "CCC", "2021-01-01")],
    )
    rows, unmapped = build_cik_map.build_rows({"AAA", "BBB", "CCC"}, {"CCC": 222})
    assert unmapped == []
    assert rows == [
        ("AAA", 222, "2017-01-01", "2019-01-01"),
        ("BBB", 222, "2019-01-01", "2021-01-01"),
        ("CCC", 222, "2021-01-01", ""),
    ]
    by_symbol = {s: (start, end) for s, _, start, end in rows}
    assert by_symbol["AAA"][1] == by_symbol["BBB"][0]  # end == next start
    assert by_symbol["BBB"][1] == by_symbol["CCC"][0]


def test_build_rows_cycle_terminates_as_unmapped(monkeypatch):
    # A->B->A with neither ticker live must not loop forever or emit rows:
    # the seen-set guard deterministically classifies both as unmapped.
    monkeypatch.setattr(
        build_cik_map,
        "RENAMES",
        [("AAA", "BBB", "2020-01-01"), ("BBB", "AAA", "2021-01-01")],
    )
    rows, unmapped = build_cik_map.build_rows({"AAA", "BBB"}, {"ZZZ": 999})
    assert rows == []
    assert unmapped == ["AAA", "BBB"]


def test_build_rows_rename_target_start_overrides_since(monkeypatch):
    # A rename target's interval starts at the rename date, not SINCE; a
    # symbol with no rename involvement starts at SINCE.
    monkeypatch.setattr(build_cik_map, "RENAMES", [("OLD", "NEW", "2022-05-01")])
    rows, unmapped = build_cik_map.build_rows({"NEW", "PLAIN"}, {"NEW": 333, "PLAIN": 444})
    assert unmapped == []
    assert rows == [
        ("NEW", 333, "2022-05-01", ""),
        ("PLAIN", 444, "2017-01-01", ""),
    ]


def test_build_rows_dead_end_symbol_is_unmapped(monkeypatch):
    # Not a live ticker and not in RENAMES -> unmapped (fail-open), no row.
    monkeypatch.setattr(build_cik_map, "RENAMES", [])
    rows, unmapped = build_cik_map.build_rows({"GONE", "LIVE"}, {"LIVE": 555})
    assert rows == [("LIVE", 555, "2017-01-01", "")]
    assert unmapped == ["GONE"]


def test_build_rows_excluded_symbol_is_unmapped_even_when_currently_live(monkeypatch):
    # A recycled ticker DOES resolve in current_tickers (a squatter now holds
    # it) -- EXCLUSIONS must short-circuit before that lookup so it lands
    # unmapped (fail-open) instead of silently attaching the squatter's CIK.
    monkeypatch.setattr(build_cik_map, "RENAMES", [])
    monkeypatch.setattr(build_cik_map, "EXCLUSIONS", {"APC": "ticker recycled: see test"})
    rows, unmapped = build_cik_map.build_rows({"APC", "LIVE"}, {"APC": 2080921, "LIVE": 555})
    assert rows == [("LIVE", 555, "2017-01-01", "")]
    assert unmapped == ["APC"]


def test_committed_map_excludes_confirmed_ticker_recycling_squatters():
    # APC/BID/CONE (Task-4-review-confirmed recycled-ticker mismaps; see
    # PROVENANCE.md) must never appear in the committed map.
    assert set(MAP["symbol"]) & set(build_cik_map.EXCLUSIONS) == set()


# --- identity-mismatch audit (recycling defenses, part b) --------------------


def test_identity_audit_flags_a_disagreeing_direct_listing():
    # NEW resolved via chain to cik 111 (a rename target), but company_
    # tickers.json's OWN "NEW" entry today independently belongs to cik 999
    # -- two SEC-recognized identities disagree, worth a human look even
    # though the zero-filing audit (verify_fundamentals.py) might not catch
    # it (the wrong cik could easily have real filings).
    rows = [("NEW", 111, "2020-01-01", "")]
    current_tickers_raw = {"NEW": {"cik": 999, "title": "Some Unrelated Squatter Inc"}}
    calls: list[str] = []

    def fetch(url: str) -> dict:
        calls.append(url)
        return {"entityName": "The Real Renamed Company"}

    audit = build_cik_map.check_identity_mismatches(rows, current_tickers_raw, fetch_json=fetch)
    assert audit == [
        {
            "symbol": "NEW",
            "mapped_cik": 111,
            "mapped_entity_name": "The Real Renamed Company",
            "symbol_direct_cik": 999,
            "symbol_direct_title": "Some Unrelated Squatter Inc",
        }
    ]
    assert calls == [build_cik_map.COMPANYFACTS_URL.format(cik=111)]


def test_identity_audit_skips_agreeing_or_unlisted_symbols():
    rows = [
        ("AGREES", 111, "2017-01-01", ""),  # direct listing matches the assigned cik
        ("DELISTED", 222, "2017-01-01", "2020-01-01"),  # no independent current listing at all
    ]
    current_tickers_raw = {"AGREES": {"cik": 111, "title": "Agrees Co"}}
    audit = build_cik_map.check_identity_mismatches(
        rows, current_tickers_raw, fetch_json=lambda url: {}
    )
    assert audit == []


def test_identity_audit_is_fail_open_on_a_companyfacts_fetch_failure():
    rows = [("NEW", 111, "2020-01-01", "")]
    current_tickers_raw = {"NEW": {"cik": 999, "title": "Squatter"}}

    def boom(url: str) -> dict:
        raise OSError("edgar down")

    audit = build_cik_map.check_identity_mismatches(rows, current_tickers_raw, fetch_json=boom)
    assert audit[0]["mapped_entity_name"] == "(companyfacts fetch failed)"
