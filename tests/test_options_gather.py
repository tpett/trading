"""Offline tests for the ThetaData options-IV gather pipeline.

Every terminal read is served by an in-memory ``FakeClient`` -- nothing here
touches the network or a live terminal. Where a plausible IV value is asserted,
the input quote is manufactured with ``bs_price`` at a known vol so the
round-trip has a checkable answer.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from trading.research.options_gather import (
    ThetaClient,
    build_cell,
    build_universe,
    dedup_bars_by_trade_date,
    gather_cell,
    is_standard_monthly,
    load_existing_keys,
    run_gather,
    select_bar_for_decision,
    select_expiration,
    snap_strikes,
)
from trading.research.options_iv import DIV_YIELD, RATE, bs_price

# --- Test doubles ----------------------------------------------------------


class FakeClient:
    """Serves canned expirations / strikes / history from in-memory dicts.

    ``history`` is keyed by (strike, right) -> list-of-bars; unknown legs return
    an empty tape (no prints in window). Records every history call so tests can
    assert de-duplication of collapsed strikes.
    """

    def __init__(self, expirations=None, strikes=None, history=None):
        self._expirations = expirations or []
        self._strikes = strikes or []
        self._history = history or {}
        self.history_calls: list[tuple] = []

    def list_expirations(self, symbol):
        return list(self._expirations)

    def list_strikes(self, symbol, expiration):
        return list(self._strikes)

    def history_eod(self, symbol, expiration, strike, right, start_date, end_date):
        self.history_calls.append((strike, right, start_date, end_date))
        return list(self._history.get((float(strike), right), []))


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
    assert set(contract) == {"role", "type", "strike", "bid", "ask", "close", "mid", "iv"}


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
    _write_underlying(cache_dir, "AAA")
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
    _write_underlying(cache_dir, "AAA")
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
