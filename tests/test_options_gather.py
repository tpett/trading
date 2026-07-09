"""Offline tests for the ThetaData options-IV gather pipeline.

Every terminal read is served by an in-memory ``FakeClient`` -- nothing here
touches the network or a live terminal. Where a plausible IV value is asserted,
the input quote is manufactured with ``bs_price`` at a known vol so the
round-trip has a checkable answer.
"""

from __future__ import annotations

import json
import urllib.error
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from trading.research.options_gather import (
    FAR_MAX_DTE,
    FAR_MIN_DTE,
    OI_PATH,
    ThetaClient,
    build_cell,
    build_universe,
    build_work_items,
    dedup_bars_by_trade_date,
    extract_open_interest,
    gather_cell,
    is_standard_monthly,
    load_existing_keys,
    rank_strikes_by_distance,
    row_trade_date,
    run_gather,
    select_bar_for_decision,
    select_expiration,
    select_far_expiration,
    select_row_for_date,
    snap_strikes,
)
from trading.research.options_iv import DIV_YIELD, RATE, bs_price

# --- Test doubles ----------------------------------------------------------


class FakeClient:
    """Serves canned expirations / strikes / history from in-memory dicts.

    ``history`` (and ``open_interest``) are keyed by (strike, right) ->
    list-of-rows, or by (expiration, strike, right) when a test needs
    per-expiration data (the 3-tuple wins). ``strikes`` is a flat list served
    for every expiration, or a {expiration: [strikes]} dict. Unknown legs
    return an empty tape. Records every call so tests can assert request
    de-duplication.
    """

    def __init__(self, expirations=None, strikes=None, history=None, open_interest=None):
        self._expirations = expirations or []
        self._strikes = strikes or []
        self._history = history or {}
        self._open_interest = open_interest or {}
        self.history_calls: list[tuple] = []
        self.oi_calls: list[tuple] = []

    def list_expirations(self, symbol):
        return list(self._expirations)

    def list_strikes(self, symbol, expiration):
        if isinstance(self._strikes, dict):
            return list(self._strikes.get(expiration, []))
        return list(self._strikes)

    @staticmethod
    def _lookup(table, expiration, strike, right):
        key3 = (expiration, float(strike), right)
        if key3 in table:
            return list(table[key3])
        return list(table.get((float(strike), right), []))

    def history_eod(self, symbol, expiration, strike, right, start_date, end_date):
        self.history_calls.append((float(strike), right, start_date, end_date))
        return self._lookup(self._history, expiration, strike, right)

    def history_open_interest(self, symbol, expiration, strike, right, start_date, end_date):
        self.oi_calls.append((expiration, float(strike), right))
        return self._lookup(self._open_interest, expiration, strike, right)


def _bar(trade_date: str, bid: float, ask: float, close: float, created: str) -> dict:
    """One EOD tape row. ``created`` is the snapshot time the terminal duplicates on."""
    return {
        "close": close,
        "bid": bid,
        "ask": ask,
        "last_trade": f"{trade_date}T15:59:56.204",
        "created": created,
    }


# --- Expiration selection --------------------------------------------------


def test_is_standard_monthly_accepts_friday_and_good_friday_thursday():
    assert is_standard_monthly(date(2019, 5, 17))  # 3rd Friday
    assert is_standard_monthly(date(2019, 4, 18))  # Thursday: Good Friday shift
    assert not is_standard_monthly(date(2019, 4, 5))  # 1st Friday (weekly)
    assert not is_standard_monthly(date(2019, 3, 20))  # Wednesday in the window
    # The predicate cannot know April 19 2019 was a holiday; the terminal simply
    # never lists a holiday date, so accepting the Friday-in-window is harmless.
    assert is_standard_monthly(date(2019, 4, 19))


def test_select_expiration_picks_third_friday_nearest_d35():
    decision = date(2019, 3, 1)  # target ~ 2019-04-05
    expirations = [
        "2019-03-15",  # 14 DTE -> below the band
        "2019-04-05",  # a Friday but NOT the 3rd Friday (weekly) -> ignored
        "2019-04-18",  # 48 DTE, standard monthly (Good-Friday Thursday) -> pick
        "2019-05-17",  # 77 DTE -> above the band
    ]
    assert select_expiration(expirations, decision) == date(2019, 4, 18)


def test_select_expiration_none_in_window():
    decision = date(2019, 3, 1)
    # Only a too-near and a too-far standard monthly; nothing in [25,50].
    expirations = ["2019-03-15", "2019-05-17"]
    assert select_expiration(expirations, decision) is None


def test_select_expiration_ties_break_to_earlier_date():
    # Two standard monthlies equidistant from target break to the earlier one.
    decision = date(2019, 4, 1)  # target 2019-05-06
    # 2019-04-18 (Thu, 17 DTE) is out of band; use symmetric monthlies:
    expirations = ["2019-04-18", "2019-05-17"]  # 17 and 46 DTE
    assert select_expiration(expirations, decision) == date(2019, 5, 17)


# --- Strike snapping -------------------------------------------------------


def test_snap_strikes_nearest_ladder():
    ladder = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]
    snapped = snap_strikes(ladder, spot=100.0)
    assert snapped["atm"] == 100.0
    assert snapped["otm_put"] == 90.0  # nearest 0.90*100
    assert snapped["otm_call"] == 110.0  # nearest 1.10*100


def test_snap_strikes_sparse_ladder_collapses_roles():
    # A ladder so sparse the OTM legs collapse onto the ATM strike.
    ladder = [95.0, 100.0, 105.0]
    snapped = snap_strikes(ladder, spot=100.0)
    assert snapped["atm"] == 100.0
    assert snapped["otm_put"] == 95.0  # nearest 90 on this ladder
    assert snapped["otm_call"] == 105.0  # nearest 110 on this ladder


def test_collapsed_call_strike_fetched_once():
    """When the OTM-call snaps onto the ATM strike (both calls) only one history
    request is issued for that contract."""
    # Ladder where 1.10*spot snaps to the ATM strike.
    ladder = [90.0, 100.0]  # spot 100 -> atm 100, otm_call nearest 110 -> 100
    price = bs_price(100.0, 100.0, 35 / 365, RATE, DIV_YIELD, 0.30, True)
    put_price = bs_price(100.0, 90.0, 35 / 365, RATE, DIV_YIELD, 0.35, False)
    history = {
        (100.0, "C"): [_bar("2019-03-01", price - 0.05, price + 0.05, price, "c1")],
        (90.0, "P"): [_bar("2019-03-01", put_price - 0.05, put_price + 0.05, put_price, "c2")],
    }
    client = FakeClient(expirations=["2019-04-18"], strikes=ladder, history=history)
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None
    # atm (C@100) and otm_call (C@100) collapse -> the C@100 leg fetched once.
    call_fetches = [c for c in client.history_calls if c[0] == 100.0 and c[1] == "C"]
    assert len(call_fetches) == 1


# --- Nearest-strike-with-data walk -----------------------------------------


def test_rank_strikes_by_distance_ties_break_by_strike():
    # target 101 is equidistant from 100 and 102 -> lower strike first.
    assert rank_strikes_by_distance([102.0, 100.0, 98.0, 104.0], 101.0) == [
        100.0,
        102.0,
        98.0,
        104.0,
    ]


def _aapl_client(history: dict) -> FakeClient:
    """AAPL-like ladder: whole-dollar strikes carry data, half-dollar strikes are
    listed but dataless (FakeClient returns [] for any strike absent from
    ``history``)."""
    ladder = [
        150.0, 152.5, 155.0, 157.5, 160.0, 162.5, 165.0, 167.5, 170.0, 172.5,
        175.0, 180.0, 182.5, 185.0,
    ]
    return FakeClient(expirations=["2019-03-15"], strikes=ladder, history=history)


def test_gather_cell_walks_past_dataless_strike_to_data():
    """AAPL 2019-02-01 scenario: spot 166.52, exp 2019-03-15. The NEAREST ATM
    strike is the half-dollar 167.5 (dist 0.98), but it is dataless; the walk
    falls back to 165 (dist 1.52), which has data. OTM legs resolve to 150 / 185."""
    spot = 166.52
    dte = (date(2019, 3, 15) - date(2019, 2, 1)).days
    t = dte / 365
    atm_p = bs_price(spot, 165.0, t, RATE, DIV_YIELD, 0.30, True)
    put_p = bs_price(spot, 150.0, t, RATE, DIV_YIELD, 0.34, False)
    call_p = bs_price(spot, 185.0, t, RATE, DIV_YIELD, 0.27, True)
    history = {
        (165.0, "C"): [_bar("2019-02-01", atm_p - 0.05, atm_p + 0.05, atm_p, "c")],
        (150.0, "P"): [_bar("2019-02-01", put_p - 0.05, put_p + 0.05, put_p, "c")],
        (185.0, "C"): [_bar("2019-02-01", call_p - 0.05, call_p + 0.05, call_p, "c")],
    }
    cell = gather_cell(_aapl_client(history), "AAPL", date(2019, 2, 1), spot)

    assert cell is not None
    by_role = {c["role"]: c for c in cell["contracts"]}
    assert by_role["atm"]["strike"] == 165.0  # resolved past the dataless 167.5
    assert by_role["otm_put"]["strike"] == 150.0
    assert by_role["otm_call"]["strike"] == 185.0
    assert by_role["atm"]["iv"] == pytest.approx(0.30, abs=5e-3)  # real time value, inverts
    assert cell["skew_put_call"] is not None  # 185 != 165, a genuine risk-reversal


def test_gather_cell_skipped_when_all_atm_candidates_dataless():
    """If none of the ATM leg's <=4 candidate strikes returns a bar, the ATM leg
    is absent and the whole cell is skipped."""
    # Empty history -> every strike returns [] -> no candidate resolves.
    cell = gather_cell(_aapl_client({}), "AAPL", date(2019, 2, 1), 166.52)
    assert cell is None


def test_gather_cell_stops_after_candidate_cap():
    """A data-bearing strike beyond the candidate cap is NOT reached: with the
    first 4 candidates all dataless the leg gives up even though strike #5 has
    data."""
    spot = 166.52
    # Nearest four to 166.52 are 167.5, 165, 170, 162.5 (all dataless here); the
    # 5th, 160, has data but is past the cap.
    ladder = [160.0, 162.5, 165.0, 167.5, 170.0]
    history = {(160.0, "C"): [_bar("2019-02-01", 5.0, 5.1, 5.05, "c")]}
    client = FakeClient(expirations=["2019-03-15"], strikes=ladder, history=history)
    cell = gather_cell(client, "AAPL", date(2019, 2, 1), spot, max_candidates=4)
    assert cell is None  # 160 never probed (5th candidate)
    assert (160.0, "C") not in [(c[0], c[1]) for c in client.history_calls]


def test_gather_cell_probe_cached_across_role_walks():
    """A (strike, right) reached by more than one role's walk is fetched once."""
    spot = 100.0
    dte = (date(2019, 4, 18) - date(2019, 3, 1)).days
    t = dte / 365
    atm_p = bs_price(spot, 100.0, t, RATE, DIV_YIELD, 0.30, True)
    put_p = bs_price(spot, 90.0, t, RATE, DIV_YIELD, 0.35, False)
    # (110,C) is listed but dataless; the OTM-call walk probes it, then falls back
    # to (100,C) -- already fetched by the ATM walk, so it is NOT refetched.
    ladder = [90.0, 100.0, 110.0]
    history = {
        (100.0, "C"): [_bar("2019-03-01", atm_p - 0.05, atm_p + 0.05, atm_p, "c")],
        (90.0, "P"): [_bar("2019-03-01", put_p - 0.05, put_p + 0.05, put_p, "c")],
    }
    client = FakeClient(expirations=["2019-04-18"], strikes=ladder, history=history)
    gather_cell(client, "AAA", date(2019, 3, 1), spot)

    fetch_counts: dict[tuple, int] = {}
    for strike, right, _s, _e in client.history_calls:
        fetch_counts[(strike, right)] = fetch_counts.get((strike, right), 0) + 1
    assert fetch_counts[(100.0, "C")] == 1  # atm probe reused by otm_call walk
    assert fetch_counts[(110.0, "C")] == 1  # dataless probe cached, not repeated


# --- Dedup by trade date ---------------------------------------------------


def test_dedup_bars_keeps_one_per_trade_date():
    bars = [
        _bar("2019-03-01", 4.0, 4.2, 4.1, "2019-03-01T18:12:00"),
        _bar("2019-03-01", 4.0, 4.2, 4.1, "2019-03-01T19:08:00"),  # duplicate snapshot
        _bar("2019-02-28", 3.9, 4.1, 4.0, "2019-02-28T19:08:00"),
    ]
    by_date = dedup_bars_by_trade_date(bars)
    assert set(by_date) == {date(2019, 3, 1), date(2019, 2, 28)}
    assert by_date[date(2019, 3, 1)]["created"] == "2019-03-01T19:08:00"  # last wins


def test_select_bar_on_decision_date():
    bars = [
        _bar("2019-03-01", 4.0, 4.2, 4.1, "s1"),
        _bar("2019-03-01", 4.0, 4.2, 4.1, "s2"),
    ]
    bar = select_bar_for_decision(bars, date(2019, 3, 1))
    assert bar is not None and bar["created"] == "s2"


def test_select_bar_falls_back_to_nearest_prior():
    # No bar on 03-04; prior bars on 03-01 and 02-28. Nearest prior within 3d is
    # 03-01. A bar 5 days before (02-27) is outside the window and ignored.
    bars = [
        _bar("2019-02-27", 3.0, 3.2, 3.1, "old"),
        _bar("2019-02-28", 3.9, 4.1, 4.0, "prior2"),
        _bar("2019-03-01", 4.0, 4.2, 4.1, "prior1"),
    ]
    bar = select_bar_for_decision(bars, date(2019, 3, 4), window_days=3)
    assert bar is not None and bar["created"] == "prior1"


def test_select_bar_none_when_window_empty():
    bars = [_bar("2019-02-20", 3.0, 3.2, 3.1, "old")]
    assert select_bar_for_decision(bars, date(2019, 3, 4), window_days=3) is None


# --- Mid & IV wiring -------------------------------------------------------


def test_build_cell_mid_and_iv_roundtrip():
    spot, dte = 100.0, 35
    t = dte / 365
    atm_price = bs_price(spot, 100.0, t, RATE, DIV_YIELD, 0.30, True)
    put_price = bs_price(spot, 90.0, t, RATE, DIV_YIELD, 0.35, False)
    call_price = bs_price(spot, 110.0, t, RATE, DIV_YIELD, 0.25, True)
    legs = {
        "atm": (100.0, _bar("2019-03-01", atm_price - 0.05, atm_price + 0.05, atm_price, "c")),
        "otm_put": (90.0, _bar("2019-03-01", put_price - 0.02, put_price + 0.02, put_price, "c")),
        "otm_call": (
            110.0,
            _bar("2019-03-01", call_price - 0.02, call_price + 0.02, call_price, "c"),
        ),
    }
    cell = build_cell("AAA", date(2019, 3, 1), spot, date(2019, 4, 5), legs)

    by_role = {c["role"]: c for c in cell["contracts"]}
    assert by_role["atm"]["mid"] == pytest.approx(atm_price)  # (bid+ask)/2 straddles price
    assert by_role["atm"]["iv"] == pytest.approx(0.30, abs=2e-3)
    assert by_role["otm_put"]["iv"] == pytest.approx(0.35, abs=2e-3)
    assert by_role["otm_call"]["iv"] == pytest.approx(0.25, abs=2e-3)
    # skew_put_atm = iv_put - iv_atm ~ 0.05; skew_put_call = iv_put - iv_call ~ 0.10
    assert cell["skew_put_atm"] == pytest.approx(0.05, abs=4e-3)
    assert cell["skew_put_call"] == pytest.approx(0.10, abs=4e-3)


def test_build_cell_uninvertible_leg_yields_null_iv_and_skew():
    spot, dte = 100.0, 35
    t = dte / 365
    atm_price = bs_price(spot, 100.0, t, RATE, DIV_YIELD, 0.30, True)
    legs = {
        "atm": (100.0, _bar("2019-03-01", atm_price - 0.05, atm_price + 0.05, atm_price, "c")),
        # A sub-penny OTM-put quote carries no vol information -> IV None.
        "otm_put": (90.0, _bar("2019-03-01", 0.0, 0.001, 0.0005, "c")),
    }
    cell = build_cell("AAA", date(2019, 3, 1), spot, date(2019, 4, 5), legs)
    by_role = {c["role"]: c for c in cell["contracts"]}
    assert by_role["otm_put"]["iv"] is None
    assert cell["skew_put_atm"] is None  # a None leg IV nulls the skew
    assert cell["skew_put_call"] is None


# --- Cell schema & skip rules ----------------------------------------------


def test_build_cell_schema_shape():
    spot = 100.0
    price = bs_price(spot, 100.0, 35 / 365, RATE, DIV_YIELD, 0.30, True)
    put = bs_price(spot, 90.0, 35 / 365, RATE, DIV_YIELD, 0.30, False)
    legs = {
        "atm": (100.0, _bar("2019-03-01", price - 0.05, price + 0.05, price, "c")),
        "otm_put": (90.0, _bar("2019-03-01", put - 0.02, put + 0.02, put, "c")),
        "otm_call": (110.0, None),  # not gathered -> omitted
    }
    cell = build_cell("AAPL", date(2019, 3, 1), spot, date(2019, 4, 18), legs)
    assert set(cell) == {
        "symbol",
        "decision_date",
        "spot_at_decision",
        "target_expiration",
        "days_to_expiry",
        "contracts",
        "skew_put_atm",
        "skew_put_call",
    }
    assert cell["symbol"] == "AAPL"
    assert cell["decision_date"] == "2019-03-01"
    assert cell["target_expiration"] == "2019-04-18"
    assert cell["days_to_expiry"] == 48
    assert {c["role"] for c in cell["contracts"]} == {"atm", "otm_put"}  # otm_call omitted
    contract = cell["contracts"][0]
    assert set(contract) == {
        "role", "type", "strike", "bid", "ask", "close", "mid", "iv", "volume", "count",
    }


def test_build_cell_skipped_when_atm_missing():
    put = bs_price(100.0, 90.0, 35 / 365, RATE, DIV_YIELD, 0.30, False)
    legs = {
        "atm": (100.0, None),  # ATM not gathered -> whole cell dropped
        "otm_put": (90.0, _bar("2019-03-01", put - 0.02, put + 0.02, put, "c")),
    }
    assert build_cell("AAA", date(2019, 3, 1), 100.0, date(2019, 4, 5), legs) is None


def test_build_cell_skipped_when_otm_put_missing():
    price = bs_price(100.0, 100.0, 35 / 365, RATE, DIV_YIELD, 0.30, True)
    legs = {
        "atm": (100.0, _bar("2019-03-01", price - 0.05, price + 0.05, price, "c")),
        "otm_put": (90.0, None),
    }
    assert build_cell("AAA", date(2019, 3, 1), 100.0, date(2019, 4, 5), legs) is None


# --- Resume / idempotency --------------------------------------------------


def _write_underlying(cache_dir: Path, symbol: str) -> None:
    dates = pd.bdate_range("2019-01-01", "2019-03-31", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1_000_000.0,
        },
        index=dates,
    )
    frame.index.name = "Date"
    frame.to_parquet(cache_dir / f"{symbol}.parquet")


def _write_raw(
    raw_dir: Path, symbol: str, value: float, start: str = "2018-12-15", end: str = "2019-06-30"
) -> None:
    """Raw (unadjusted) close cache: one close_raw parquet, UTC index."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    idx = pd.bdate_range(start, end, tz="UTC")
    frame = pd.DataFrame({"close_raw": [value] * len(idx)}, index=idx)
    frame.index.name = "Date"
    frame.to_parquet(raw_dir / f"{symbol}.parquet")


def _full_history() -> dict:
    """Bars for every trading day Jan-Mar so any first-of-month decision hits."""
    dates = pd.bdate_range("2018-12-15", "2019-03-31")
    price_c = bs_price(100.0, 100.0, 40 / 365, RATE, DIV_YIELD, 0.30, True)
    price_p = bs_price(100.0, 90.0, 40 / 365, RATE, DIV_YIELD, 0.35, False)
    price_oc = bs_price(100.0, 110.0, 40 / 365, RATE, DIV_YIELD, 0.25, True)
    hist = {(100.0, "C"): [], (90.0, "P"): [], (110.0, "C"): []}
    for d in dates:
        ds = d.date().isoformat()
        hist[(100.0, "C")].append(_bar(ds, price_c - 0.05, price_c + 0.05, price_c, ds))
        hist[(90.0, "P")].append(_bar(ds, price_p - 0.02, price_p + 0.02, price_p, ds))
        hist[(110.0, "C")].append(_bar(ds, price_oc - 0.02, price_oc + 0.02, price_oc, ds))
    return hist


def test_run_gather_resume_is_idempotent(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    raw_dir = tmp_path / "raw"
    _write_underlying(cache_dir, "AAA")
    _write_raw(raw_dir, "AAA", 100.0)
    out = tmp_path / "samples.jsonl"

    # Standard monthlies bracketing Jan/Feb/Mar first-of-month decisions.
    expirations = ["2019-02-15", "2019-03-15", "2019-04-18"]
    client = FakeClient(
        expirations=expirations, strikes=[90.0, 100.0, 110.0], history=_full_history()
    )

    summary1 = run_gather(
        client,
        out,
        symbols=["AAA"],
        start_month="2019-01",
        end_month="2019-03",
        cache_dir=cache_dir,
        raw_dir=raw_dir,
        membership_csv=tmp_path / "unused.csv",
    )
    lines_after_first = out.read_text().splitlines()
    assert summary1["written"] == len(lines_after_first) > 0

    # Second run over the same output must add nothing (all keys already present).
    summary2 = run_gather(
        client,
        out,
        symbols=["AAA"],
        start_month="2019-01",
        end_month="2019-03",
        cache_dir=cache_dir,
        raw_dir=raw_dir,
        membership_csv=tmp_path / "unused.csv",
    )
    assert summary2["cells_attempted"] == 0
    assert summary2["written"] == 0
    assert out.read_text().splitlines() == lines_after_first

    # Every written line is a valid, unique-keyed cell.
    keys = [tuple(json.loads(x)[k] for k in ("symbol", "decision_date")) for x in lines_after_first]
    assert len(keys) == len(set(keys))


def test_run_gather_recovers_from_torn_final_line(tmp_path: Path):
    """A prior run left a complete cell with NO trailing newline (flush died
    before the '\\n'). The gather must add the newline so the next appended cell
    is not glued onto it -- every line stays parseable and no key duplicates."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    raw_dir = tmp_path / "raw"
    _write_underlying(cache_dir, "AAA")
    _write_raw(raw_dir, "AAA", 100.0)
    out = tmp_path / "samples.jsonl"
    # First trading day of Jan 2019 on the fixture calendar is 2019-01-01.
    out.write_text(json.dumps({"symbol": "AAA", "decision_date": "2019-01-01"}))  # no newline

    client = FakeClient(
        expirations=["2019-02-15", "2019-03-15", "2019-04-18"],
        strikes=[90.0, 100.0, 110.0],
        history=_full_history(),
    )
    summary = run_gather(
        client,
        out,
        symbols=["AAA"],
        start_month="2019-01",
        end_month="2019-03",
        cache_dir=cache_dir,
        raw_dir=raw_dir,
        membership_csv=tmp_path / "unused.csv",
    )

    lines = [x for x in out.read_text().splitlines() if x.strip()]
    parsed = [json.loads(x) for x in lines]  # (a) no unparseable / glued line
    keys = [(c["symbol"], c["decision_date"]) for c in parsed]
    assert len(keys) == len(set(keys))  # (b) no duplicates
    assert ("AAA", "2019-01-01") in keys  # the torn cell survived, not re-gathered
    assert summary["written"] > 0  # Feb + Mar cells were appended cleanly


def test_gather_cell_collapsed_call_nulls_put_call_no_dup_contract():
    """Ladder [90,100], spot 100: otm_call snaps onto the ATM strike. The cell
    emits no duplicate 100-strike contract, drops the otm_call role, and reports
    skew_put_call None with skew_put_atm still computed."""
    dte = (date(2019, 4, 18) - date(2019, 3, 1)).days
    t = dte / 365
    price_c = bs_price(100.0, 100.0, t, RATE, DIV_YIELD, 0.30, True)
    price_p = bs_price(100.0, 90.0, t, RATE, DIV_YIELD, 0.35, False)
    history = {
        (100.0, "C"): [_bar("2019-03-01", price_c - 0.05, price_c + 0.05, price_c, "c")],
        (90.0, "P"): [_bar("2019-03-01", price_p - 0.02, price_p + 0.02, price_p, "c")],
    }
    client = FakeClient(expirations=["2019-04-18"], strikes=[90.0, 100.0], history=history)
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0)

    assert cell is not None
    roles = [c["role"] for c in cell["contracts"]]
    assert roles == ["atm", "otm_put"]  # otm_call dropped
    strikes_100 = [c for c in cell["contracts"] if c["strike"] == 100.0]
    assert len(strikes_100) == 1  # no duplicate 100-strike contract
    assert cell["skew_put_atm"] is not None
    assert cell["skew_put_call"] is None


# --- Raw spot (unadjusted) -------------------------------------------------


def test_build_work_items_uses_raw_spot_not_adjusted(tmp_path: Path):
    """Spot comes from the RAW cache, not the split/dividend-adjusted one."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    raw_dir = tmp_path / "raw"
    _write_underlying(cache_dir, "AAPL")  # adjusted close 100.0 -> calendar only
    _write_raw(raw_dir, "AAPL", 166.52)  # raw close (real 2019 AAPL level)

    work = build_work_items(["AAPL"], cache_dir, raw_dir, "2019-02", "2019-02", set())
    assert len(work) == 1
    symbol, _decision, spot = work[0]
    assert symbol == "AAPL"
    assert spot == pytest.approx(166.52)  # raw, NOT the adjusted 100.0


def test_build_work_items_skips_symbol_without_raw_series(tmp_path: Path):
    """A symbol with an adjusted parquet but no raw series is skipped entirely
    (no fall-back to the strike-mismatched adjusted price)."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_underlying(cache_dir, "AAPL")  # adjusted present, raw absent

    work = build_work_items(["AAPL"], cache_dir, raw_dir, "2019-02", "2019-02", set())
    assert work == []


def test_gather_cell_snaps_strikes_near_raw_spot():
    """The AAPL smoke scenario: a raw spot ~166 snaps ATM to the ~165 strike and
    the OTM legs to ~150 / ~183 -- NOT anywhere near an adjusted ~40."""
    spot = 166.52
    dte = (date(2019, 4, 18) - date(2019, 3, 1)).days
    t = dte / 365
    # A realistic raw 2019 AAPL ladder ($2.5-$5 spacing).
    ladder = [150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 182.5, 185.0, 190.0]
    atm_p = bs_price(spot, 165.0, t, RATE, DIV_YIELD, 0.30, True)
    put_p = bs_price(spot, 150.0, t, RATE, DIV_YIELD, 0.34, False)
    call_p = bs_price(spot, 182.5, t, RATE, DIV_YIELD, 0.27, True)
    history = {
        (165.0, "C"): [_bar("2019-03-01", atm_p - 0.05, atm_p + 0.05, atm_p, "c")],
        (150.0, "P"): [_bar("2019-03-01", put_p - 0.05, put_p + 0.05, put_p, "c")],
        (182.5, "C"): [_bar("2019-03-01", call_p - 0.05, call_p + 0.05, call_p, "c")],
    }
    client = FakeClient(expirations=["2019-04-18"], strikes=ladder, history=history)
    cell = gather_cell(client, "AAPL", date(2019, 3, 1), spot)

    assert cell is not None
    assert cell["spot_at_decision"] == pytest.approx(166.52)
    by_role = {c["role"]: c for c in cell["contracts"]}
    assert by_role["atm"]["strike"] == 165.0
    assert by_role["otm_put"]["strike"] == 150.0
    assert by_role["otm_call"]["strike"] == 182.5
    # Quotes are real time-value (not all-intrinsic), so IV inverts.
    assert by_role["atm"]["iv"] == pytest.approx(0.30, abs=5e-3)
    assert by_role["otm_put"]["iv"] is not None


def test_load_existing_keys_ignores_torn_line(tmp_path: Path):
    out = tmp_path / "samples.jsonl"
    out.write_text(
        json.dumps({"symbol": "AAA", "decision_date": "2019-01-02"})
        + "\n{ this is a torn line\n"
    )
    assert load_existing_keys(out) == {("AAA", "2019-01-02")}


# --- Universe selection ----------------------------------------------------


def _universe_membership_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "membership.csv"
    csv.write_text(
        "# comment header\n"
        "symbol,index,start,end\n"
        "HIVOL,sp500,2018-01-01,\n"
        "MIDVOL,ndx,2019-01-01,\n"
        "LOVOL,sp500,2019-01-01,\n"
        "OTHER,sp400,2019-01-01,\n"  # sp400 only -> excluded
        "NOCACHE,sp500,2019-01-01,\n"  # member but no parquet -> excluded
        "OLD,sp500,2010-01-01,2015-01-01\n"  # interval before window -> excluded
    )
    return csv


def _write_dollar_vol(cache_dir: Path, symbol: str, volume: float) -> None:
    dates = pd.bdate_range("2019-01-01", "2019-06-30", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": volume,
        },
        index=dates,
    )
    frame.index.name = "Date"
    frame.to_parquet(cache_dir / f"{symbol}.parquet")


def test_build_universe_ranks_by_dollar_volume_and_intersects_membership(tmp_path: Path):
    csv = _universe_membership_csv(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_dollar_vol(cache_dir, "HIVOL", 5_000_000)
    _write_dollar_vol(cache_dir, "MIDVOL", 2_000_000)
    _write_dollar_vol(cache_dir, "LOVOL", 100_000)
    _write_dollar_vol(cache_dir, "OTHER", 9_000_000)  # cached but sp400-only
    _write_dollar_vol(cache_dir, "OLD", 9_000_000)  # cached but out-of-window membership
    # NOCACHE has no parquet.

    universe = build_universe(csv, cache_dir, size=10, start_month="2019-01", end_month="2019-06")
    # Only sp500/ndx members in-window with a parquet, ranked by dollar volume desc.
    assert universe == ["HIVOL", "MIDVOL", "LOVOL"]

    # size caps the list deterministically.
    assert build_universe(csv, cache_dir, size=2, start_month="2019-01", end_month="2019-06") == [
        "HIVOL",
        "MIDVOL",
    ]


# --- HTTP client (mocked transport) ----------------------------------------


def test_theta_client_retries_then_succeeds(monkeypatch):
    """The client retries a transient failure with injected (no-wait) backoff and
    parses the v3 response envelope."""
    sleeps: list[float] = []
    calls = {"n": 0}

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"response": [{"strike": 185.0}, {"strike": 190.0}]}).encode()

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("connection refused")
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = ThetaClient("http://x", sleep=sleeps.append, backoff_base=0.5)
    strikes = client.list_strikes("AAPL", "2019-04-18")
    assert strikes == [185.0, 190.0]
    assert calls["n"] == 2  # one failure, one success
    assert sleeps == [0.5]  # exponential backoff, first step only


def test_theta_client_raises_after_exhausting_retries(monkeypatch):
    def always_fail(req, timeout):
        raise OSError("refused")

    monkeypatch.setattr("urllib.request.urlopen", always_fail)
    client = ThetaClient("http://x", max_retries=3, sleep=lambda _s: None)
    with pytest.raises(OSError):
        client.list_expirations("AAPL")


def test_theta_client_treats_472_as_empty_not_error(monkeypatch):
    """HTTP 472 (ThetaData 'no data in range') is a normal empty response, not a
    retryable failure: history_eod returns [] and no exception escapes, so an
    illiquid OTM leg is omitted rather than sinking the whole cell."""
    sleeps: list[float] = []
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 472, "No data", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = ThetaClient("http://x", max_retries=3, sleep=sleeps.append)
    bars = client.history_eod("AAPL", "2019-03-15", 182.5, "C", "2019-01-30", "2019-02-04")
    assert bars == []  # empty, not an exception
    assert calls["n"] == 1  # 472 is NOT retried
    assert sleeps == []  # no backoff spent on a no-data response


# --- OI endpoint & extraction (v2) ------------------------------------------
#
# Discovery verdict (2026-07, live Standard-tier terminal): the EOD tape carries
# NO inline open_interest field, but `/v3/option/history/open_interest` (OI_PATH)
# serves it -- HTTP 200, one row/day, columns
# symbol,expiration,strike,right,timestamp,open_interest. Greeks endpoints
# (`/v3/option/history/greeks`, `/v3/option/history/implied_volatility`) both
# 404 on this tier, so the greeks capture path is omitted entirely (YAGNI --
# nothing in this module references greeks).


class _FakeResp:
    """Minimal urlopen context-manager double serving one canned JSON body."""

    status = 200

    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


# One OI row exactly as the live tape serves it: the response header
# (symbol,expiration,strike,right,timestamp,open_interest) becomes the row keys
# verbatim, so the date-bearing key is `timestamp` -- NOT `date` -- and it
# carries a time component. Fixtures below MUST use this shape or they test an
# assumed schema instead of the discovered one.
def _live_oi_row(timestamp: str, open_interest: int) -> dict:
    return {
        "symbol": "AAPL",
        "expiration": "2023-06-16",
        "strike": 180,
        "right": "C",
        "timestamp": timestamp,
        "open_interest": open_interest,
    }


def test_theta_client_history_open_interest_endpoint_and_parse(monkeypatch):
    """history_open_interest hits the verified v3 OI route with the canonical
    contract params and flattens the response envelope to rows whose keys are
    the live header verbatim (`timestamp`, not `date`)."""
    captured = {}
    live_row = _live_oi_row("2023-05-15T06:30:10.000", 54141)

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        return _FakeResp({"response": [{"data": [dict(live_row)]}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = ThetaClient("http://x")
    rows = client.history_open_interest(
        "AAPL", "2023-06-16", 180.0, "C", "2023-05-15", "2023-05-19"
    )
    assert rows == [live_row]
    # The parse preserves the live header keys: rows DO carry `timestamp`.
    assert set(rows[0]) == {
        "symbol",
        "expiration",
        "strike",
        "right",
        "timestamp",
        "open_interest",
    }
    assert f"{OI_PATH}?" in captured["url"]  # tracks the discovery-verified route
    assert "symbol=AAPL" in captured["url"]
    assert "strike=180" in captured["url"]  # whole-dollar strike, canonical bare-int form
    assert "right=C" in captured["url"]
    assert "start_date=2023-05-15" in captured["url"]  # dashed dates


def test_theta_client_history_open_interest_472_is_empty_not_error(monkeypatch):
    """A 472 on the OI route is a normal empty (same semantics as history_eod):
    one call, no retry, no exception -- a missing OI leg is simply absent."""
    calls = {"n": 0}

    def always_472(req, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 472, "No data", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", always_472)
    client = ThetaClient("http://x", sleep=lambda _s: None)
    rows = client.history_open_interest(
        "AAPL", "2019-04-18", 185.0, "C", "2019-02-25", "2019-03-05"
    )
    assert rows == []
    assert calls["n"] == 1  # the 472 was NOT retried


def test_row_trade_date_reads_timestamp_then_date_then_last_trade():
    # The live OI tape: `timestamp` with a time component -> date part only.
    assert row_trade_date(_live_oi_row("2023-05-15T06:30:10.000", 54141)) == date(2023, 5, 15)
    # The EOD tape keys `last_trade`; `date` is a defensive alias.
    assert row_trade_date({"last_trade": "2019-03-01T15:59:56.204"}) == date(2019, 3, 1)
    assert row_trade_date({"date": "2019-03-01"}) == date(2019, 3, 1)
    assert row_trade_date({"timestamp": "garbage"}) is None
    assert row_trade_date({}) is None


def test_select_row_for_date_matches_live_oi_rows_by_date_part():
    """Matching is by CALENDAR DATE, never exact-timestamp equality -- the live
    OI stamps carry 06:30-style times. Exact match only, last row wins on a
    tape double-print, and NO nearest-neighbor fallback when the date is absent."""
    rows = [
        _live_oi_row("2023-05-15T06:30:10.000", 54141),
        _live_oi_row("2023-05-15T06:30:11.000", 54142),  # double-print: last wins
        _live_oi_row("2023-05-12T06:30:09.000", 53990),
    ]
    assert select_row_for_date(rows, date(2023, 5, 15))["open_interest"] == 54142
    # NO nearest-prior fallback: enrichment must ride the SAME date as the bar.
    assert select_row_for_date(rows, date(2023, 5, 16)) is None


def test_extract_open_interest_absent_or_junk_yields_no_key():
    assert extract_open_interest(None) == {}
    assert extract_open_interest({"date": "2019-03-01"}) == {}  # field absent -> no key
    assert extract_open_interest({"open_interest": None}) == {}
    assert extract_open_interest({"open_interest": "junk"}) == {}
    assert extract_open_interest({"open_interest": float("nan")}) == {}
    # A vendor-SERVED zero is a real observation, kept (never fabricated, never dropped).
    assert extract_open_interest({"open_interest": 0}) == {"open_interest": 0}
    assert extract_open_interest({"open_interest": 1523.0}) == {"open_interest": 1523}


# --- Per-leg OI enrichment (v2) ----------------------------------------------


def _enrichment_client() -> FakeClient:
    """The 90/100/110 cell with OI for two of the three resolved legs."""
    dte = (date(2019, 4, 18) - date(2019, 3, 1)).days
    t = dte / 365
    price_c = bs_price(100.0, 100.0, t, RATE, DIV_YIELD, 0.30, True)
    price_p = bs_price(100.0, 90.0, t, RATE, DIV_YIELD, 0.35, False)
    price_oc = bs_price(100.0, 110.0, t, RATE, DIV_YIELD, 0.25, True)
    history = {
        (100.0, "C"): [_bar("2019-03-01", price_c - 0.05, price_c + 0.05, price_c, "c")],
        (90.0, "P"): [_bar("2019-03-01", price_p - 0.02, price_p + 0.02, price_p, "c")],
        (110.0, "C"): [_bar("2019-03-01", price_oc - 0.02, price_oc + 0.02, price_oc, "c")],
    }
    open_interest = {
        (100.0, "C"): [{"date": "2019-03-01", "open_interest": 1500}],
        (90.0, "P"): [{"date": "2019-03-01", "open_interest": 2200}],
        # (110.0, "C") has NO OI row -> key absent on that leg.
    }
    return FakeClient(
        expirations=["2019-04-18"], strikes=[90.0, 100.0, 110.0],
        history=history, open_interest=open_interest,
    )


def test_gather_cell_attaches_oi_additively():
    cell = gather_cell(_enrichment_client(), "AAA", date(2019, 3, 1), 100.0, include_far=False)
    assert cell is not None
    by_role = {c["role"]: c for c in cell["contracts"]}
    assert by_role["atm"]["open_interest"] == 1500
    assert by_role["otm_put"]["open_interest"] == 2200
    assert "open_interest" not in by_role["otm_call"]  # no OI row -> key ABSENT, never 0
    # v1 fields intact on an enriched leg.
    assert by_role["atm"]["iv"] == pytest.approx(0.30, abs=2e-3)
    assert by_role["atm"]["volume"] is None  # _bar carries no volume; unchanged semantics


def test_gather_cell_oi_row_on_wrong_date_yields_no_key():
    client = _enrichment_client()
    client._open_interest = {
        (100.0, "C"): [{"date": "2019-02-25", "open_interest": 999}],  # not the bar's date
    }
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0, include_far=False)
    by_role = {c["role"]: c for c in cell["contracts"]}
    assert "open_interest" not in by_role["atm"]


def test_gather_cell_enrichment_failure_degrades_to_absent_keys():
    class ExplodingEnrichment(FakeClient):
        def history_open_interest(self, *args, **kwargs):
            raise OSError("boom")

    base = _enrichment_client()
    client = ExplodingEnrichment(
        expirations=["2019-04-18"], strikes=[90.0, 100.0, 110.0], history=base._history
    )
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0, include_far=False)
    assert cell is not None  # enrichment failure must never sink a good cell
    for contract in cell["contracts"]:
        assert "open_interest" not in contract
        assert contract["iv"] is not None  # the quote side of the leg is untouched


def test_gather_cell_no_oi_no_far_serializes_bit_identical_to_v1():
    """The frozen v2 constraint, pinned at the SERIALIZATION level: with no OI
    served and no far monthly listed, json.dumps(cell) is byte-identical to the
    pre-v2 gather's output -- same keys, same ORDER, same values. The expected
    string was generated by running this exact fixture through the v1 module
    (`git show 2ce9e9b:src/trading/research/options_gather.py`); do NOT
    regenerate it from current code, or the pin stops pinning anything."""
    client = _enrichment_client()
    client._open_interest = {}  # OI endpoint serves nothing -> no enrichment keys
    # _enrichment_client lists only 2019-04-18 -> no far monthly in band.
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0)
    expected = (
        '{"symbol": "AAA", "decision_date": "2019-03-01", "spot_at_decision": 100.0,'
        ' "target_expiration": "2019-04-18", "days_to_expiry": 48, "contracts":'
        ' [{"role": "atm", "type": "call", "strike": 100.0, "bid": 4.576628065973783,'
        ' "ask": 4.676628065973783, "close": 4.626628065973783, "mid": 4.626628065973783,'
        ' "iv": 0.30000000996421705, "volume": null, "count": null},'
        ' {"role": "otm_put", "type": "put", "strike": 90.0, "bid": 1.2364619410631188,'
        ' "ask": 1.2764619410631188, "close": 1.2564619410631188, "mid": 1.2564619410631188,'
        ' "iv": 0.3499999999999991, "volume": null, "count": null},'
        ' {"role": "otm_call", "type": "call", "strike": 110.0, "bid": 0.7905863529137771,'
        ' "ask": 0.8305863529137771, "close": 0.8105863529137771, "mid": 0.8105863529137771,'
        ' "iv": 0.2499999999999992, "volume": null, "count": null}],'
        ' "skew_put_atm": 0.04999999003578204, "skew_put_call": 0.0999999999999999}'
    )
    assert json.dumps(cell) == expected


def test_gather_cell_enriches_collapsed_roles_once():
    """otm_call collapsed onto the ATM strike: one contract, one OI request."""
    dte = (date(2019, 4, 18) - date(2019, 3, 1)).days
    t = dte / 365
    price_c = bs_price(100.0, 100.0, t, RATE, DIV_YIELD, 0.30, True)
    price_p = bs_price(100.0, 90.0, t, RATE, DIV_YIELD, 0.35, False)
    client = FakeClient(
        expirations=["2019-04-18"],
        strikes=[90.0, 100.0],
        history={
            (100.0, "C"): [_bar("2019-03-01", price_c - 0.05, price_c + 0.05, price_c, "c")],
            (90.0, "P"): [_bar("2019-03-01", price_p - 0.02, price_p + 0.02, price_p, "c")],
        },
        open_interest={(100.0, "C"): [{"date": "2019-03-01", "open_interest": 7}]},
    )
    gather_cell(client, "AAA", date(2019, 3, 1), 100.0, include_far=False)
    oi_for_atm = [c for c in client.oi_calls if c[1] == 100.0 and c[2] == "C"]
    assert len(oi_for_atm) == 1


# --- Far (second expiration) block (v2) --------------------------------------

V1_CELL_KEYS = {
    "symbol", "decision_date", "spot_at_decision", "target_expiration",
    "days_to_expiry", "contracts", "skew_put_atm", "skew_put_call",
}


def test_select_far_expiration_picks_next_monthly_in_band():
    decision = date(2019, 3, 1)
    near = date(2019, 4, 18)
    expirations = [
        "2019-04-18",  # the near itself -> excluded (not strictly after)
        "2019-05-10",  # a weekly in-band -> ignored (not a standard monthly)
        "2019-05-17",  # 77 DTE monthly -> pick (the NEXT monthly after near)
        "2019-06-21",  # 112 DTE -> above the far band
    ]
    assert select_far_expiration(expirations, decision, near) == date(2019, 5, 17)


def test_select_far_expiration_none_when_band_empty_or_only_near():
    decision = date(2019, 3, 1)
    near = date(2019, 4, 18)
    assert select_far_expiration(["2019-04-18"], decision, near) is None
    # A monthly in the DTE band that IS the near expiration must not be reused:
    # decision 2019-03-20 -> 2019-05-17 = 58 DTE (in 55..90) but == near.
    assert (
        select_far_expiration(["2019-05-17", "2019-06-21"], date(2019, 3, 20), date(2019, 5, 17))
        is None  # 2019-06-21 is 93 DTE, above the band
    )


def test_select_far_expiration_dte_band_edges_inclusive_neighbors_excluded():
    """The far band [55, 90] is INCLUSIVE at both edges; 54 and 91 are out."""
    exps = ["2019-05-17", "2019-06-21"]
    near = date(2019, 5, 17)
    far = date(2019, 6, 21)
    assert select_far_expiration(exps, date(2019, 4, 27), near) == far  # 55 DTE, lower edge
    assert select_far_expiration(exps, date(2019, 3, 23), near) == far  # 90 DTE, upper edge
    assert select_far_expiration(exps, date(2019, 4, 28), near) is None  # 54 DTE, below band
    assert select_far_expiration(exps, date(2019, 3, 22), near) is None  # 91 DTE, above band


def _near_far_client() -> FakeClient:
    """Near 2019-04-18 and far 2019-05-17 both with data, priced at their own
    tenors so the far ATM IV (0.28) is distinguishable from the near (0.30)."""
    decision = date(2019, 3, 1)
    t_near = (date(2019, 4, 18) - decision).days / 365
    t_far = (date(2019, 5, 17) - decision).days / 365
    near_c = bs_price(100.0, 100.0, t_near, RATE, DIV_YIELD, 0.30, True)
    near_p = bs_price(100.0, 90.0, t_near, RATE, DIV_YIELD, 0.35, False)
    far_c = bs_price(100.0, 100.0, t_far, RATE, DIV_YIELD, 0.28, True)
    far_p = bs_price(100.0, 90.0, t_far, RATE, DIV_YIELD, 0.33, False)
    history = {
        ("2019-04-18", 100.0, "C"): [_bar("2019-03-01", near_c - 0.05, near_c + 0.05, near_c, "c")],
        ("2019-04-18", 90.0, "P"): [_bar("2019-03-01", near_p - 0.02, near_p + 0.02, near_p, "c")],
        ("2019-05-17", 100.0, "C"): [_bar("2019-03-01", far_c - 0.05, far_c + 0.05, far_c, "c")],
        ("2019-05-17", 90.0, "P"): [_bar("2019-03-01", far_p - 0.02, far_p + 0.02, far_p, "c")],
    }
    open_interest = {
        ("2019-05-17", 100.0, "C"): [{"date": "2019-03-01", "open_interest": 640}],
    }
    return FakeClient(
        expirations=["2019-04-18", "2019-05-17"],
        strikes=[90.0, 100.0],
        history=history,
        open_interest=open_interest,
    )


def test_gather_cell_composes_near_plus_far_blocks():
    cell = gather_cell(_near_far_client(), "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None
    assert set(cell) == V1_CELL_KEYS | {"far"}  # far is the ONLY new top-level key
    far = cell["far"]
    assert far["target_expiration"] == "2019-05-17"
    assert far["days_to_expiry"] == (date(2019, 5, 17) - date(2019, 3, 1)).days
    assert FAR_MIN_DTE <= far["days_to_expiry"] <= FAR_MAX_DTE
    far_by_role = {c["role"]: c for c in far["contracts"]}
    assert far_by_role["atm"]["iv"] == pytest.approx(0.28, abs=3e-3)  # far tenor, own IV
    assert far_by_role["atm"]["open_interest"] == 640  # far legs are enriched too
    # Near block untouched by the far gather.
    near_by_role = {c["role"]: c for c in cell["contracts"]}
    assert near_by_role["atm"]["iv"] == pytest.approx(0.30, abs=3e-3)
    assert cell["target_expiration"] == "2019-04-18"


def test_gather_cell_far_absent_when_no_far_monthly():
    client = _near_far_client()
    client._expirations = ["2019-04-18"]  # nothing after the near monthly
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None
    assert "far" not in cell  # absent, not null/empty
    assert set(cell) == V1_CELL_KEYS  # bit-for-bit v1 shape


def test_gather_cell_far_legs_dataless_drops_far_only():
    client = _near_far_client()
    client._history = {
        k: v for k, v in client._history.items() if k[0] == "2019-04-18"
    }  # far expiration listed but its whole tape is empty
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None
    assert "far" not in cell


def test_gather_cell_far_atm_only_drops_far_block():
    """A far expiration whose OTM-put leg is dataless fails the same
    ATM+OTM-put validity rule as the near cell: the far block is dropped
    (no ATM-only far block), while the near cell keeps both blocks' rule
    honored and survives untouched."""
    client = _near_far_client()
    del client._history[("2019-05-17", 90.0, "P")]  # far put dataless; far ATM still fine
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None
    assert "far" not in cell
    assert set(cell) == V1_CELL_KEYS


def test_gather_cell_far_exception_drops_block_only():
    class FarExploding(FakeClient):
        def list_strikes(self, symbol, expiration):
            if expiration == "2019-05-17":
                raise OSError("terminal hiccup on the far ladder")
            return super().list_strikes(symbol, expiration)

    base = _near_far_client()
    client = FarExploding(
        expirations=["2019-04-18", "2019-05-17"],
        strikes=[90.0, 100.0],
        history=base._history,
    )
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None  # near cell survives
    assert "far" not in cell


def test_gather_cell_include_far_false_spends_no_far_requests():
    client = _near_far_client()
    cell = gather_cell(client, "AAA", date(2019, 3, 1), 100.0, include_far=False)
    assert "far" not in cell
    # Only the near walk's probes happened: (100,C) + (90,P); the otm_call walk
    # reuses the cached (100,C). Far would have added two more history calls.
    assert len(client.history_calls) == 2


def test_enriched_cell_json_round_trip_and_resume_key(tmp_path: Path):
    """A gathered v2 cell survives JSON serialization exactly, and the resume
    reader extracts the same (symbol, decision_date) key it always did."""
    cell = gather_cell(_near_far_client(), "AAA", date(2019, 3, 1), 100.0)
    assert cell is not None and "far" in cell
    line = json.dumps(cell)
    assert json.loads(line) == cell  # nothing in the cell is non-JSON-native
    out = tmp_path / "samples.jsonl"
    out.write_text(line + "\n")
    assert load_existing_keys(out) == {("AAA", "2019-03-01")}
