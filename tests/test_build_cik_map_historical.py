"""scripts/build_cik_map_historical.py: offline unit tests on synthetic FSDS
fixtures and a mocked submissions seam, plus committed-artifact anchors
(same pattern as tests/test_fundamentals_cik_map.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_cik_map  # noqa: E402
import build_cik_map_historical as hist  # noqa: E402

FUND = Path(__file__).resolve().parent.parent / "src" / "trading" / "fundamentals"
CIK_MAP_CSV = FUND / "cik_map.csv"
HISTORICAL_CSV = FUND / "cik_map_historical.csv"


def _frame(rows: list[tuple], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns).astype(str)


# --- instance-prefix ticker extraction ---------------------------------------


def test_ticker_from_instance_extracts_the_filer_chosen_prefix():
    assert hist.ticker_from_instance("xlnx-20201226.xml") == "XLNX"
    assert hist.ticker_from_instance("k-20221231.htm") == "K"
    assert hist.ticker_from_instance("brka-20191231.xml") == "BRKA"


def test_ticker_from_instance_rejects_non_ticker_shapes():
    assert hist.ticker_from_instance("0001564590-20-004475.xml") is None  # accession-style
    assert hist.ticker_from_instance("d123456d10q.htm") is None  # no dash-delimited prefix
    assert hist.ticker_from_instance("") is None
    assert hist.ticker_from_instance("toolongprefix-2020.xml") is None  # > 6 chars


# --- target selection ---------------------------------------------------------


def test_target_symbols_wants_window_members_without_window_overlapping_intervals():
    cik_map = _frame(
        [
            ("LIVE", "1", "2017-01-01", ""),  # covered: not a target
            ("DOC", "2", "2024-03-04", ""),  # row exists but post-window: target
        ],
        ["symbol", "cik", "start", "end"],
    )
    membership = _frame(
        [
            ("LIVE", "sp500", "2017-01-01", ""),
            ("DOC", "sp400", "2020-05-18", "2024-03-01"),
            ("DEAD", "sp500", "2017-01-01", "2021-06-01"),
            ("PREWIN", "sp500", "2017-01-01", "2018-06-01"),  # pre-window: not a target
            ("APC", "sp500", "2017-01-01", "2019-08-09"),  # EXCLUSIONS: never a target
        ],
        ["symbol", "index", "start", "end"],
    )
    assert hist.target_symbols(cik_map, membership) == ["DEAD", "DOC"]


# --- candidate building + resolution ------------------------------------------


def _sub(rows: list[tuple[int, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["cik", "name", "filed", "ticker"])


def _no_fetch(url: str) -> dict:
    raise AssertionError("network must not be touched")


def _committed(rows: list[tuple] = ()) -> pd.DataFrame:
    return _frame(list(rows), ["symbol", "cik", "start", "end"])


def test_resolve_target_unique_in_tenure_candidate_wins():
    cands = hist.candidates_for(
        _sub([(743988, "XILINX INC", "2020-05-08", "XLNX")]), {"XLNX"}
    )
    got = hist.resolve_target(
        "XLNX", cands["XLNX"], ("2017-01-01", "2022-02-14"), _committed(), _no_fetch
    )
    assert got == (743988, "XILINX INC", "unique")


def test_resolve_target_filters_out_a_post_tenure_recycler():
    # The real company filed in-tenure; a recycler of the same ticker filed
    # long after the membership tenure ended -- it must not block resolution.
    cands = hist.candidates_for(
        _sub(
            [
                (111, "REAL HISTORICAL CO", "2020-05-08", "GONE"),
                (999, "SQUATTER CORP", "2023-11-01", "GONE"),
            ]
        ),
        {"GONE"},
    )
    got = hist.resolve_target(
        "GONE", cands["GONE"], ("2017-01-01", "2021-06-01"), _committed(), _no_fetch
    )
    assert got == (111, "REAL HISTORICAL CO", "unique")


def test_resolve_target_grace_tail_admits_a_straggling_final_report():
    # SCG shape: SCANA's final 10-K was FILED 57 days after index removal;
    # within the 90-day grace it still counts, and a later filer does not.
    cands = hist.candidates_for(
        _sub(
            [
                (754737, "SCANA CORP", "2019-02-28", "SCG"),
                (91882, "DOMINION ENERGY SOUTH CAROLINA", "2019-05-09", "SCG"),
            ]
        ),
        {"SCG"},
    )
    got = hist.resolve_target(
        "SCG", cands["SCG"], ("2017-01-01", "2019-01-02"), _committed(), _no_fetch
    )
    assert got == (754737, "SCANA CORP", "unique-grace")


def test_resolve_target_strict_unique_beats_grace_induced_ambiguity():
    # ARNC shape: Arconic Inc filed strictly in tenure; the spun-off Arconic
    # Corp's first 10-Q landed within the grace tail. The strict-tenure
    # unique answer must win -- grace may widen, never break, resolution.
    cands = hist.candidates_for(
        _sub(
            [
                (4281, "ARCONIC INC", "2020-02-20", "ARNC"),
                (1790982, "ARCONIC CORP", "2020-05-15", "ARNC"),
            ]
        ),
        {"ARNC"},
    )
    got = hist.resolve_target(
        "ARNC", cands["ARNC"], ("2017-01-01", "2020-04-06"), _committed(), _no_fetch
    )
    assert got == (4281, "ARCONIC INC", "unique")


def test_resolve_target_renames_successor_beats_co_filer_prefix_collisions(monkeypatch):
    # CTL shape: Lumen's subsidiaries (Qwest, Level 3) file under the ctl-
    # prefix too; the reviewed RENAMES successor's committed CIK decides.
    monkeypatch.setattr(hist, "RENAMES", [("CTL", "LUMN", "2020-09-18")])
    cands = hist.candidates_for(
        _sub(
            [
                (18926, "LUMEN TECHNOLOGIES, INC.", "2019-03-11", "CTL"),
                (68622, "QWEST CORP", "2019-03-22", "CTL"),
            ]
        ),
        {"CTL"},
    )
    committed = _committed([("LUMN", "18926", "2017-01-01", "")])
    got = hist.resolve_target(
        "CTL", cands["CTL"], ("2017-01-01", "2020-09-18"), committed, _no_fetch
    )
    assert got == (18926, "LUMEN TECHNOLOGIES, INC.", "renames-successor")


def test_resolve_target_sec_ticker_attribution_breaks_a_two_filer_tie():
    # K shape: Kellanova and WK Kellogg both filed under a k- prefix in
    # tenure; SEC's submissions JSON attributes ticker K to exactly one.
    cands = hist.candidates_for(
        _sub(
            [
                (55067, "KELLANOVA", "2020-02-25", "K"),
                (1959348, "WK KELLOGG CO", "2023-11-08", "K"),
            ]
        ),
        {"K"},
    )
    payloads = {
        hist.SUBMISSIONS_URL.format(cik=55067): {"tickers": ["K"]},
        hist.SUBMISSIONS_URL.format(cik=1959348): {"tickers": ["KLG"]},
    }
    got = hist.resolve_target(
        "K", cands["K"], ("2017-01-01", "2025-12-11"), _committed(), payloads.__getitem__
    )
    assert got == (55067, "KELLANOVA", "sec-tickers")


def test_resolve_target_stays_unmapped_when_no_tiebreak_discriminates():
    cands = hist.candidates_for(
        _sub(
            [
                (111, "COMPANY A", "2020-05-08", "DUAL"),
                (222, "COMPANY B", "2020-06-08", "DUAL"),
            ]
        ),
        {"DUAL"},
    )
    both = {"tickers": ["DUAL"]}
    got = hist.resolve_target(
        "DUAL", cands["DUAL"], ("2017-01-01", "2021-06-01"), _committed(), lambda u: both
    )
    assert got is None  # both claim the ticker: never guess


def test_resolve_target_tiebreak_fails_closed_on_a_dead_fetch():
    cands = hist.candidates_for(
        _sub(
            [
                (111, "COMPANY A", "2020-05-08", "DUAL"),
                (222, "COMPANY B", "2020-06-08", "DUAL"),
            ]
        ),
        {"DUAL"},
    )

    def boom(url: str) -> dict:
        raise OSError("edgar down")

    got = hist.resolve_target(
        "DUAL", cands["DUAL"], ("2017-01-01", "2021-06-01"), _committed(), boom
    )
    assert got is None


def test_candidates_keep_the_latest_filing_name():
    cands = hist.candidates_for(
        _sub(
            [
                (111, "OLD NAME INC", "2019-05-08", "REN"),
                (111, "NEW NAME INC", "2021-05-08", "REN"),
            ]
        ),
        {"REN"},
    )
    assert cands["REN"][111]["name"] == "NEW NAME INC"
    assert cands["REN"][111]["filed"] == ["2019-05-08", "2021-05-08"]


# --- interval dating ----------------------------------------------------------


def test_choose_interval_renamed_away_symbol_ends_at_the_rename_date(monkeypatch):
    monkeypatch.setattr(hist, "RENAMES", [("OLD", "NEW", "2022-02-02")])
    existing = _frame([], ["symbol", "cik", "start", "end"])
    assert hist.choose_interval("OLD", existing, "2022-02-02") == ("2017-01-01", "2022-02-02")


def test_choose_interval_hands_off_to_an_existing_successor_row(monkeypatch):
    # DOC shape: the committed map already has the POST-window filer's row
    # starting 2024-03-04; the historical interval must end exactly there.
    monkeypatch.setattr(hist, "RENAMES", [])
    existing = _frame([("DOC", "765880", "2024-03-04", "")], ["symbol", "cik", "start", "end"])
    assert hist.choose_interval("DOC", existing, "9999-12-31") == ("2017-01-01", "2024-03-04")


def test_choose_interval_plain_dead_symbol_ends_at_membership_end(monkeypatch):
    monkeypatch.setattr(hist, "RENAMES", [])
    existing = _frame([], ["symbol", "cik", "start", "end"])
    assert hist.choose_interval("XLNX", existing, "2022-02-14") == ("2017-01-01", "2022-02-14")


def test_choose_interval_refuses_open_or_degenerate_intervals(monkeypatch):
    monkeypatch.setattr(hist, "RENAMES", [("BAD", "NEW", "2016-01-01")])
    existing = _frame([], ["symbol", "cik", "start", "end"])
    # open-ended tenure with no successor row: a live ticker should have been
    # mapped by build_cik_map.py -- never emit an open historical interval.
    assert hist.choose_interval("GHOST", existing, hist.FAR_FUTURE) is None
    # end before SINCE -> degenerate
    assert hist.choose_interval("BAD", existing, "2016-01-01") is None


# --- submissions-JSON verification --------------------------------------------


def test_verify_resolution_accepts_a_ticker_match():
    def fetch(url: str) -> dict:
        assert url == hist.SUBMISSIONS_URL.format(cik=743988)
        return {"tickers": ["XLNX"], "name": "Xilinx Inc"}

    assert hist.verify_resolution("XLNX", 743988, "XILINX INC", fetch) is True


def test_verify_resolution_accepts_a_former_name_match_when_ticker_moved_on():
    # Discovery-into-WBD shape: the CIK's tickers array now shows the merged
    # entity's ticker, but formerNames still carries the window-era identity.
    payload = {
        "tickers": ["WBD"],
        "name": "Warner Bros. Discovery, Inc.",
        "formerNames": [{"name": "Discovery, Inc."}],
    }
    assert hist.verify_resolution("DISCA", 1437107, "DISCOVERY INC", lambda u: payload) is True


def test_verify_resolution_rejects_a_wrong_company():
    payload = {"tickers": ["ZZZ"], "name": "Unrelated Corp", "formerNames": []}
    assert hist.verify_resolution("XLNX", 999, "XILINX INC", lambda u: payload) is False


def test_verify_resolution_retries_once_then_fails_closed():
    attempts = {"n": 0}

    def flaky(url: str) -> dict:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("edgar hiccup")
        return {"tickers": ["GONE"], "name": "Gone Inc"}

    assert hist.verify_resolution("GONE", 1, "GONE INC", flaky) is True
    assert attempts["n"] == 2

    def boom(url: str) -> dict:
        raise OSError("edgar down")

    assert hist.verify_resolution("GONE", 1, "GONE INC", boom) is False


# --- rename cross-consistency --------------------------------------------------


def test_check_rename_consistency_flags_a_boundary_disagreement(monkeypatch):
    monkeypatch.setattr(hist, "RENAMES", [("OLD", "NEW", "2022-01-01")])
    committed = _frame([("NEW", "222", "2022-01-01", "")], ["symbol", "cik", "start", "end"])
    bad = [("OLD", 111, "2017-01-01", "2022-01-01", "OLD CO")]
    good = [("OLD", 222, "2017-01-01", "2022-01-01", "OLD CO")]
    assert hist.check_rename_consistency(bad, committed) == ["OLD->NEW@2022-01-01: 111 != 222"]
    assert hist.check_rename_consistency(good, committed) == []


def test_check_rename_consistency_allows_a_pre_handoff_different_company(monkeypatch):
    # DOC shape: the rename target's ticker was a DIFFERENT company before
    # the handoff date -- intervals meet at the boundary, no conflict.
    monkeypatch.setattr(hist, "RENAMES", [("PEAK", "DOC", "2024-03-04")])
    committed = _frame(
        [
            ("PEAK", "765880", "2019-11-05", "2024-03-04"),
            ("DOC", "765880", "2024-03-04", ""),
        ],
        ["symbol", "cik", "start", "end"],
    )
    rows = [("DOC", 1574540, "2017-01-01", "2024-03-04", "PHYSICIANS REALTY TRUST")]
    assert hist.check_rename_consistency(rows, committed) == []


# --- output writing ------------------------------------------------------------


def test_append_to_cik_map_preserves_existing_content_byte_for_byte(tmp_path):
    path = tmp_path / "cik_map.csv"
    existing = "# header comment\nsymbol,cik,start,end\nAAPL,320193,2017-01-01,\n"
    path.write_text(existing)
    hist.append_to_cik_map(
        [("XLNX", 743988, "2017-01-01", "2022-02-14", "XILINX INC")], path
    )
    text = path.read_text()
    assert text.startswith(existing)  # nothing rewritten or reordered
    df = pd.read_csv(path, comment="#", dtype=str).fillna("")
    assert list(df.iloc[-1]) == ["XLNX", "743988", "2017-01-01", "2022-02-14"]


def test_append_to_cik_map_refuses_a_double_append(tmp_path):
    path = tmp_path / "cik_map.csv"
    path.write_text("symbol,cik,start,end\nAAPL,320193,2017-01-01,\n")
    rows = [("XLNX", 743988, "2017-01-01", "2022-02-14", "XILINX INC")]
    hist.append_to_cik_map(rows, path)
    with pytest.raises(SystemExit) as excinfo:
        hist.append_to_cik_map(rows, path)
    assert "already appended" in str(excinfo.value)


def test_write_historical_csv_round_trips_comma_bearing_names(tmp_path):
    out = tmp_path / "cik_map_historical.csv"
    hist.write_historical_csv(
        [("CELG", 816284, "2017-01-01", "2019-11-21", "CELGENE CORP /DE/, THE")], out
    )
    df = pd.read_csv(out, comment="#", dtype=str)
    assert list(df.columns) == ["symbol", "cik", "start", "end", "name"]
    assert df.iloc[0]["name"] == "CELGENE CORP /DE/, THE"


# --- build_cik_map.merge_historical (regeneration keeps historical rows) ------


def test_merge_historical_appends_non_overlapping_and_sorts():
    rows = [("DOC", 765880, "2024-03-04", ""), ("ZZZ", 1, "2017-01-01", "")]
    historical = _frame(
        [
            ("DOC", "1574540", "2017-01-01", "2024-03-04"),
            ("XLNX", "743988", "2017-01-01", "2022-02-14"),
        ],
        ["symbol", "cik", "start", "end"],
    )
    merged, skipped = build_cik_map.merge_historical(rows, historical)
    assert skipped == []
    assert merged == [
        ("DOC", 1574540, "2017-01-01", "2024-03-04"),
        ("DOC", 765880, "2024-03-04", ""),
        ("XLNX", 743988, "2017-01-01", "2022-02-14"),
        ("ZZZ", 1, "2017-01-01", ""),
    ]


def test_merge_historical_skips_overlaps_and_exclusions():
    rows = [("LIVE", 5, "2017-01-01", "")]
    historical = _frame(
        [
            ("LIVE", "6", "2017-01-01", "2020-01-01"),  # overlaps the live row: skipped
            ("APC", "773910", "2017-01-01", "2019-08-09"),  # EXCLUSIONS: skipped
        ],
        ["symbol", "cik", "start", "end"],
    )
    merged, skipped = build_cik_map.merge_historical(rows, historical)
    assert merged == rows
    assert sorted(skipped) == ["APC", "LIVE"]


# --- committed artifacts (exist only after the real generation run) -----------


def test_committed_historical_csv_shape_and_anchors():
    df = pd.read_csv(HISTORICAL_CSV, comment="#", dtype=str).fillna("")
    assert list(df.columns) == ["symbol", "cik", "start", "end", "name"]
    assert df["symbol"].is_unique
    by_symbol = dict(zip(df["symbol"], df["cik"].astype(int), strict=True))
    assert by_symbol["XLNX"] == 743988  # Xilinx (acquired by AMD 2022)
    assert by_symbol["TWTR"] == 1418091  # Twitter (taken private 2022)
    assert by_symbol["CELG"] == 816284  # Celgene (acquired by BMY 2019)
    assert by_symbol["CTL"] == 18926  # CenturyLink == committed LUMN
    # FRC (First Republic) and SBNY (Signature Bank) are FDIC filers with no
    # SEC 10-K/10-Q at all (the OZK pattern) -- structurally unresolvable
    # from EDGAR; they must stay unmapped rather than get a guessed CIK.
    assert "FRC" not in by_symbol
    assert "SBNY" not in by_symbol
    assert (df["end"] != "").all()  # every historical interval is closed


def test_committed_cik_map_gains_the_historical_rows_consistently():
    df = pd.read_csv(CIK_MAP_CSV, comment="#", dtype=str).fillna("")
    by_symbol: dict[str, list] = {}
    for row in df.itertuples():
        by_symbol.setdefault(row.symbol, []).append(row)
    # RENAMES pairs resolved from two independent sources agree on the CIK
    assert by_symbol["CTL"][0].cik == by_symbol["LUMN"][0].cik  # CenturyLink/Lumen
    assert by_symbol["MMC"][0].cik == by_symbol["MRSH"][0].cik  # Marsh & McLennan
    assert by_symbol["GPS"][0].cik == by_symbol["GAP"][0].cik  # Gap Inc
    # DOC: window-era filer hands off to the committed successor row exactly
    doc = sorted(by_symbol["DOC"], key=lambda r: r.start)
    assert len(doc) == 2 and doc[0].end == doc[1].start == "2024-03-04"
    # recycled tickers stay unmapped
    assert not (set(by_symbol) & set(build_cik_map.EXCLUSIONS))
    # every historical row landed in the merged map with the same interval
    hist_df = pd.read_csv(HISTORICAL_CSV, comment="#", dtype=str).fillna("")
    committed = {(r.symbol, r.cik, r.start, r.end) for r in df.itertuples()}
    for row in hist_df.itertuples():
        assert (row.symbol, row.cik, row.start, row.end) in committed
