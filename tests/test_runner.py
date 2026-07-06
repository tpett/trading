import dataclasses
import datetime
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sim_helpers import CR, EQ
from trading.config import load_venue_config
from trading.data.cache import OhlcvCache
from trading.journal import Journal
from trading.runner import (
    RunLock,
    intraday_partial_bar_reason,
    load_state,
    lock_path,
    restore_from_journal,
    run_venue,
    save_state,
    state_path,
)
from trading.simulator.state import StateError
from trading.venues.base import DataFetchError, SymbolInfo, VenueConstraints

NOW = datetime.datetime(2026, 7, 1, 22, 30, tzinfo=datetime.UTC)  # Wed evening UTC
SYMBOLS = ["UPUP", "FLAT", "MEH1", "MEH2"]


@pytest.fixture(autouse=True)
def _no_earnings_network(monkeypatch):
    # Belt-and-suspenders: the filter is disabled in config/equities.toml,
    # but keep tests offline in case a test flips it back on.
    monkeypatch.setattr("trading.runner.fetch_earnings_dates", lambda symbols: ({}, False))


def _bars(drift: float, end: datetime.date) -> pd.DataFrame:
    idx = pd.date_range(end=end, periods=320, freq="B", tz="UTC")
    jitter = np.where(np.arange(320) % 2 == 0, 0.002, -0.002)
    close = 100 * np.cumprod(1 + drift + jitter)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(320, 1e6),
        },
        index=idx,
    )


class FakeAdapter:
    """Serves deterministic frames through any end date; UPUP leads every feature."""

    def __init__(self, fail: frozenset[str] = frozenset()):
        self.fail = fail
        self.drifts = {"UPUP": 0.01, "FLAT": 0.0, "MEH1": -0.002, "MEH2": -0.004, "SPY": 0.001}

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        return [SymbolInfo(s, "tradable") for s in SYMBOLS]

    def constraints(self) -> VenueConstraints:
        return VenueConstraints(0.0, 0.0, 5.0, 1, False)

    def fetch_ohlcv(self, symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        if symbol in self.fail:
            raise DataFetchError(symbol)
        df = _bars(self.drifts[symbol], end)
        return df.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]


def _run(tmp_path, now=NOW, fail=frozenset(), notes=None):
    notes = notes if notes is not None else []
    adapter = FakeAdapter(fail=fail)
    cache = OhlcvCache(tmp_path / "cache", EQ.data.refetch_days)
    outcome = run_venue(
        EQ,
        adapter,
        cache,
        now=now,
        state_root=tmp_path / "state",
        journal_root=tmp_path / "journal",
        notify=lambda title, message: notes.append((title, message)),
        digest_root=tmp_path / "digest",
    )
    return outcome, notes


def test_first_run_bootstraps_journals_and_orders_top_candidate(tmp_path):
    outcome, notes = _run(tmp_path)
    assert outcome.status == "ok"
    assert outcome.run_key == "equities:2026-07-01T00:00:00+00:00"
    assert notes == []

    state = load_state(state_path(tmp_path / "state", "equities"))
    assert state.cash == 1000.0  # nothing filled yet
    assert [(o.symbol, o.side) for o in state.pending_orders] == [("UPUP", "buy")]
    assert state.benchmark_start_price > 0

    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    assert [e["event"] for e in events] == ["bootstrap", "run"]
    run_event = events[1]
    assert run_event["run_key"] == outcome.run_key
    assert run_event["starting_balance"] == 1000.0
    assert len(run_event["ranking"]) == 4
    assert run_event["state_after"]["positions"] == {}
    assert not (tmp_path / "state" / "equities" / ".lock").exists()  # released


def test_same_decision_bar_is_a_noop(tmp_path):
    _run(tmp_path)
    outcome, _ = _run(tmp_path, now=NOW + datetime.timedelta(hours=1))
    assert outcome.status == "noop"
    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    assert len(events) == 2  # nothing new journaled; never trades twice


def test_next_day_run_fills_pending_order(tmp_path):
    _run(tmp_path)
    outcome, _ = _run(tmp_path, now=NOW + datetime.timedelta(days=1))
    assert outcome.status == "ok"
    state = load_state(state_path(tmp_path / "state", "equities"))
    assert "UPUP" in state.positions
    run_event = Journal(tmp_path / "journal" / "equities.jsonl").last_event(
        types=frozenset({"run"})
    )
    assert [f["symbol"] for f in run_event["fills"]] == ["UPUP"]


def test_stale_run_still_fills_and_exits_but_skips_entries(tmp_path):
    _run(tmp_path)
    # Friday bars (decision 2026-07-03) processed Sunday: way past 1d + 2h.
    late = datetime.datetime(2026, 7, 5, 18, 0, tzinfo=datetime.UTC)
    outcome, _ = _run(tmp_path, now=late)
    assert outcome.status == "ok"
    run_event = Journal(tmp_path / "journal" / "equities.jsonl").last_event(
        types=frozenset({"run"})
    )
    assert [f["symbol"] for f in run_event["fills"]] == ["UPUP"]  # fill still happened
    assert any(s["reason"] == "stale_run_entries_skipped" for s in run_event["skips"])
    state = load_state(state_path(tmp_path / "state", "equities"))
    assert all(o.side != "buy" for o in state.pending_orders)


def test_live_lock_skips_run_and_notifies(tmp_path):
    lock = lock_path(tmp_path / "state", "equities")
    lock.parent.mkdir(parents=True)
    import os

    lock.write_text(str(os.getpid()))  # a live process holds the lock
    outcome, notes = _run(tmp_path)
    assert outcome.status == "skipped"
    assert notes and "lock" in notes[0][1]
    assert lock.exists()  # never steal a live lock


def test_stale_lock_from_dead_process_is_broken(tmp_path):
    dead = subprocess.Popen(["true"])
    dead.wait()
    lock = lock_path(tmp_path / "state", "equities")
    lock.parent.mkdir(parents=True)
    lock.write_text(str(dead.pid))
    outcome, _ = _run(tmp_path)
    assert outcome.status == "ok"


def test_corrupt_state_refuses_to_run_and_notifies(tmp_path):
    _run(tmp_path)
    state_file = state_path(tmp_path / "state", "equities")
    state_file.write_text("{ not json")
    outcome, notes = _run(tmp_path, now=NOW + datetime.timedelta(days=1))
    assert outcome.status == "failed"
    assert "restore-from-journal" in outcome.message
    assert notes and "corrupt" in notes[0][1].lower()
    assert state_file.read_text() == "{ not json"  # never silently regenerated


def test_coverage_failure_journals_notifies_and_touches_nothing(tmp_path):
    outcome, notes = _run(tmp_path, fail=frozenset({"MEH1", "MEH2"}))  # 50% < 90%
    assert outcome.status == "failed"
    assert notes
    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    assert [e["event"] for e in events] == ["run_failed"]
    assert not state_path(tmp_path / "state", "equities").exists()


def test_restore_from_journal_rebuilds_state(tmp_path):
    _run(tmp_path)
    state_file = state_path(tmp_path / "state", "equities")
    good = state_file.read_text()
    state_file.write_text("garbage")
    message = restore_from_journal("equities", tmp_path / "state", tmp_path / "journal")
    assert "equities:2026-07-01" in message
    assert json.loads(state_file.read_text()) == json.loads(good)


def test_save_state_is_atomic_and_load_round_trips(tmp_path):
    from trading.simulator.state import initial_state

    path = state_path(tmp_path / "state", "equities")
    state = initial_state("equities", 1000.0, 620.0, "2026-07-01T00:00:00+00:00")
    save_state(path, state)
    assert load_state(path) == state
    assert not path.with_suffix(".json.tmp").exists()
    assert load_state(state_path(tmp_path / "state", "crypto")) is None
    path.write_text("{}")
    with pytest.raises(StateError):
        load_state(path)


def test_run_lock_is_reentrant_safe(tmp_path):
    lock = RunLock(tmp_path / ".lock")
    assert lock.acquire() is True
    assert RunLock(tmp_path / ".lock").acquire() is False  # our own pid is alive
    lock.release()
    assert RunLock(tmp_path / ".lock").acquire() is True


# --- crash-safe persistence ordering (review findings) ---


def test_state_write_failure_after_journal_append_is_recoverable(tmp_path, monkeypatch):
    _run(tmp_path)
    state_file = state_path(tmp_path / "state", "equities")
    before = state_file.read_text()

    def boom(path, state):
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr("trading.runner.save_state", boom)
        outcome, notes = _run(tmp_path, now=NOW + datetime.timedelta(days=1))

    assert outcome.status == "failed"
    assert "restore-from-journal" in outcome.message
    assert notes  # operator is paged

    # Journal is AHEAD of state: the run event landed, state is one run behind.
    run_event = Journal(tmp_path / "journal" / "equities.jsonl").last_event(
        types=frozenset({"run"})
    )
    assert run_event["run_key"] == "equities:2026-07-02T00:00:00+00:00"
    assert state_file.read_text() == before  # untouched by the failed write

    # The anticipated recovery: replay the journal's last snapshot.
    restore_from_journal("equities", tmp_path / "state", tmp_path / "journal")
    restored = load_state(state_file)
    assert restored.last_run_key == run_event["run_key"]
    assert json.loads(state_file.read_text()) == {**run_event["state_after"]}


def test_journal_append_failure_leaves_state_untouched(tmp_path, monkeypatch):
    _run(tmp_path)
    state_file = state_path(tmp_path / "state", "equities")
    before = state_file.read_text()

    def boom(self, event):
        raise OSError("disk full")

    monkeypatch.setattr(Journal, "append", boom)
    outcome, notes = _run(tmp_path, now=NOW + datetime.timedelta(days=1))
    assert outcome.status == "failed"
    assert notes
    assert state_file.read_text() == before  # byte-identical: journal-first ordering


def test_first_run_state_write_failure_refuses_until_restored(tmp_path, monkeypatch):
    def boom(path, state):
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr("trading.runner.save_state", boom)
        outcome, notes = _run(tmp_path)
    assert outcome.status == "failed"
    assert notes

    # Journal has bootstrap + run 1 but no state file: the next run must
    # refuse (not re-bootstrap and silently drop run 1's pending orders).
    outcome, notes = _run(tmp_path, now=NOW + datetime.timedelta(days=1))
    assert outcome.status == "failed"
    assert "behind journal" in outcome.message
    assert len(notes) == 1
    assert not state_path(tmp_path / "state", "equities").exists()

    restore_from_journal("equities", tmp_path / "state", tmp_path / "journal")
    outcome, _ = _run(tmp_path, now=NOW + datetime.timedelta(days=1))
    assert outcome.status == "ok"
    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    assert [e["event"] for e in events].count("bootstrap") == 1


def test_state_behind_journal_refuses_until_restored(tmp_path):
    _run(tmp_path)
    state_file = state_path(tmp_path / "state", "equities")
    stale = state_file.read_text()  # state as of run 1
    _run(tmp_path, now=NOW + datetime.timedelta(days=1))  # run 2 persists normally
    state_file.write_text(stale)  # rewind: journal is now one run ahead

    outcome, notes = _run(tmp_path, now=NOW + datetime.timedelta(days=2))
    assert outcome.status == "failed"
    assert "behind journal" in outcome.message
    assert "restore-from-journal" in outcome.message
    assert len(notes) == 1  # notified exactly once
    assert state_file.read_text() == stale  # untouched

    restore_from_journal("equities", tmp_path / "state", tmp_path / "journal")
    outcome, notes = _run(tmp_path, now=NOW + datetime.timedelta(days=2))
    assert outcome.status == "ok"  # reconciled: the same run now proceeds
    assert notes == []


def test_bootstrap_event_is_reused_after_crash_before_first_run_event(tmp_path, monkeypatch):
    real_append = Journal.append

    def fail_run_events(self, event):
        if event["event"] == "run":
            raise OSError("disk full")
        return real_append(self, event)

    with monkeypatch.context() as m:
        m.setattr(Journal, "append", fail_run_events)
        outcome, notes = _run(tmp_path)
    assert outcome.status == "failed"
    assert notes
    assert not state_path(tmp_path / "state", "equities").exists()

    # Journal has only the bootstrap event: not behind, so the next run
    # proceeds and reuses it rather than appending a second one.
    outcome, _ = _run(tmp_path, now=NOW + datetime.timedelta(days=1))
    assert outcome.status == "ok"
    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    assert [e["event"] for e in events].count("bootstrap") == 1
    # The reused bootstrap keeps the original benchmark baseline and birthdate.
    boot = events[0]
    state = load_state(state_path(tmp_path / "state", "equities"))
    assert state.benchmark_start_price == boot["benchmark_start_price"]
    assert state.created_at == boot["ts"]


def test_restore_from_journal_corrupt_journal_raises_runner_error(tmp_path):
    from trading.runner import RunnerError

    _run(tmp_path)
    journal_file = tmp_path / "journal" / "equities.jsonl"
    journal_file.write_text("{ corrupt\n" + journal_file.read_text())
    state_file = state_path(tmp_path / "state", "equities")
    before = state_file.read_text()
    with pytest.raises(RunnerError, match="journal corrupt"):
        restore_from_journal("equities", tmp_path / "state", tmp_path / "journal")
    assert state_file.read_text() == before  # refused before touching state


def test_restore_from_journal_refuses_when_lock_held(tmp_path):
    import os

    from trading.runner import RunnerError

    _run(tmp_path)
    state_file = state_path(tmp_path / "state", "equities")
    before = state_file.read_text()
    lock = lock_path(tmp_path / "state", "equities")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))  # a live process holds the run lock

    with pytest.raises(RunnerError, match="another run is in progress"):
        restore_from_journal("equities", tmp_path / "state", tmp_path / "journal")
    assert state_file.read_text() == before  # refused before touching state
    assert lock.exists()  # never steal a live lock


def test_restore_from_journal_proceeds_once_lock_is_free(tmp_path):
    _run(tmp_path)
    state_file = state_path(tmp_path / "state", "equities")
    good = state_file.read_text()
    state_file.write_text("garbage")
    message = restore_from_journal("equities", tmp_path / "state", tmp_path / "journal")
    assert "equities:2026-07-01" in message
    assert json.loads(state_file.read_text()) == json.loads(good)
    assert not lock_path(tmp_path / "state", "equities").exists()  # released


def test_stale_lock_reclaim_race_loser_retries_via_atomic_create(tmp_path, monkeypatch):
    import os

    dead = subprocess.Popen(["true"])
    dead.wait()
    path = tmp_path / ".lock"
    path.write_text(str(dead.pid))

    rename_calls = []

    def racing_rename(src, dst):
        # Simulate losing the reclaim race: another process renamed the stale
        # lock away between our read and our rename.
        rename_calls.append((src, dst))
        os.unlink(src)
        raise FileNotFoundError(src)

    monkeypatch.setattr("trading.runner.os.rename", racing_rename)
    assert RunLock(path).acquire() is True  # loser retried and won via O_EXCL
    assert rename_calls  # the stale lock was claimed via atomic rename, not unlink
    assert path.read_text() == str(os.getpid())  # single lockfile, winner's pid
    assert list(tmp_path.iterdir()) == [path]  # no leftover .reclaim files


def test_orphaned_reclaim_files_from_dead_processes_are_swept(tmp_path):
    import os

    dead = subprocess.Popen(["true"])
    dead.wait()
    orphan = tmp_path / f".lock.reclaim.{dead.pid}"
    orphan.write_text(str(dead.pid))  # crashed between rename and unlink
    live = tmp_path / f".lock.reclaim.{os.getpid()}"
    live.write_text(str(os.getpid()))  # a live reclaim in flight

    assert RunLock(tmp_path / ".lock").acquire() is True
    assert not orphan.exists()  # dead-pid orphan swept on acquire
    assert live.exists()  # live process's reclaim left alone


def test_run_writes_daily_digest(tmp_path):
    _run(tmp_path)
    digest_file = tmp_path / "digest" / "2026-07-01.md"
    assert digest_file.exists()
    text = digest_file.read_text()
    assert "## equities" in text
    assert "Top 5 ranking" in text


# --- earnings input journaled behind the blackout flag ---


def test_run_event_omits_earnings_when_blackout_disabled(tmp_path):
    # config/equities.toml ships earnings_blackout_enabled = false.
    _run(tmp_path)
    run_event = Journal(tmp_path / "journal" / "equities.jsonl").last_event(
        types=frozenset({"run"})
    )
    assert "earnings" not in run_event


def test_run_event_carries_earnings_map_when_enabled_and_fetched(tmp_path, monkeypatch):
    import dataclasses

    enabled = dataclasses.replace(
        EQ, portfolio=dataclasses.replace(EQ.portfolio, earnings_blackout_enabled=True)
    )
    fetched = {"UPUP": ("2026-07-02",), "FLAT": ()}
    monkeypatch.setattr(
        "trading.runner.fetch_earnings_dates", lambda symbols: (dict(fetched), False)
    )
    adapter = FakeAdapter()
    cache = OhlcvCache(tmp_path / "cache", enabled.data.refetch_days)
    run_venue(
        enabled,
        adapter,
        cache,
        now=NOW,
        state_root=tmp_path / "state",
        journal_root=tmp_path / "journal",
        notify=lambda title, message: None,
        digest_root=tmp_path / "digest",
    )
    run_event = Journal(tmp_path / "journal" / "equities.jsonl").last_event(
        types=frozenset({"run"})
    )
    # JSON round-trips tuples as lists.
    assert run_event["earnings"] == {
        "dates": {k: list(v) for k, v in fetched.items()},
        "degraded": False,
    }


# --- intraday partial-bar guard (session venues only) ---


def test_intraday_run_during_market_session_aborts_before_any_journal_write(tmp_path):
    # A launchd-coalesced run lands 14:00 ET on the decision bar's own date --
    # yfinance is still serving that day's bar IN PROGRESS. Must abort clean:
    # no journal event (bootstrap or otherwise), no state file, one notify.
    now = datetime.datetime(2026, 7, 6, 18, 0, tzinfo=datetime.UTC)  # 14:00 ET Monday
    outcome, notes = _run(tmp_path, now=now)
    assert outcome.status == "failed"
    assert "market session" in outcome.message
    assert len(notes) == 1
    assert not (tmp_path / "journal" / "equities.jsonl").exists()
    assert not state_path(tmp_path / "state", "equities").exists()


def test_run_after_session_close_buffer_proceeds_normally(tmp_path):
    # Same calendar date, but past 16:00 ET close + the 90min buffer: the
    # bar is final at yfinance by now, so the run must proceed as usual.
    now = datetime.datetime(2026, 7, 6, 22, 30, tzinfo=datetime.UTC)  # 18:30 ET Monday
    outcome, notes = _run(tmp_path, now=now)
    assert outcome.status == "ok"
    assert notes == []


def test_guard_only_applies_when_decision_bar_is_todays_date():
    # A stale run processing an OLD (already-final) bar during market hours
    # must not be blocked -- only a same-day partial bar is unsafe.
    friday_bar = pd.Timestamp("2026-07-03", tz="UTC")
    monday_market_hours = datetime.datetime(2026, 7, 6, 14, 0, tzinfo=datetime.UTC)  # 10:00 ET
    assert intraday_partial_bar_reason(EQ, friday_bar, monday_market_hours) is None


def test_crypto_never_gated_by_session_guard():
    # trades_24_7 venues have no session to be mid-way through.
    decision_ts = pd.Timestamp("2026-07-06", tz="UTC")
    for hour in (0, 9, 14, 16, 20, 23):
        now = datetime.datetime(2026, 7, 6, hour, 0, tzinfo=datetime.UTC)
        assert intraday_partial_bar_reason(CR, decision_ts, now) is None


def test_digest_write_failure_never_blocks_state_save(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("trading.runner.write_digest", boom)
    outcome, notes = _run(tmp_path)

    # The digest is pure reporting: state and journal must be durable and
    # the run must still succeed — a missing digest file is a reporting gap.
    assert outcome.status == "ok"
    state = load_state(state_path(tmp_path / "state", "equities"))
    assert state is not None and state.last_run_key == outcome.run_key
    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    assert [e["event"] for e in events] == ["bootstrap", "run"]
    assert not (tmp_path / "digest" / "2026-07-01.md").exists()
    digest_notes = [n for n in notes if "digest" in n[0]]
    assert len(digest_notes) == 1  # notified exactly once, then carried on


def _quality_eq(tmp_path):
    return dataclasses.replace(
        EQ,
        signals=dataclasses.replace(EQ.signals, ranker="quality_momentum_v1"),
        data=dataclasses.replace(EQ.data, fundamentals_dir=str(tmp_path / "fund")),
    )


def _run_quality(tmp_path, monkeypatch, refresh):
    monkeypatch.setattr("trading.runner.refresh_fundamentals", refresh)
    notes: list[tuple[str, str]] = []
    outcome = run_venue(
        _quality_eq(tmp_path),
        FakeAdapter(),
        OhlcvCache(tmp_path / "cache", EQ.data.refetch_days),
        now=NOW,
        state_root=tmp_path / "state",
        journal_root=tmp_path / "journal",
        notify=lambda title, message: notes.append((title, message)),
    )
    return outcome, notes


def _last_run_event(tmp_path):
    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    return [e for e in events if e["event"] == "run"][-1]


def test_momentum_v1_run_never_refreshes_fundamentals(tmp_path, monkeypatch):
    # Assert on OBSERVABLES, not a raising tripwire: the refresh block is
    # fail-open, so a raised AssertionError would be swallowed into a warning
    # and this test would pass vacuously. Record calls instead.
    calls: list[list[str]] = []

    def recorder(store, cik_map, symbols, as_of, **kwargs):
        calls.append(sorted(symbols))
        return 0, False

    monkeypatch.setattr("trading.runner.refresh_fundamentals", recorder)
    outcome, _ = _run(tmp_path)  # default EQ config: momentum_v1
    assert outcome.status == "ok"
    assert calls == []  # the refresh hook never fired
    warnings = _last_run_event(tmp_path)["warnings"]
    assert not any(w.startswith("fundamentals refresh") for w in warnings)
    assert not (tmp_path / "fund").exists()  # no store dir, no marker file

    # Sensitivity check: the identical recorder DOES fire under a
    # fundamentals-requiring ranker, so the empty list above is load-bearing.
    _run_quality(tmp_path / "quality", monkeypatch, recorder)
    assert calls == [sorted(SYMBOLS)]


def test_quality_run_refreshes_once_then_respects_the_weekly_gate(tmp_path, monkeypatch):
    calls: list[tuple] = []

    def refresh(store, cik_map, symbols, as_of, **kwargs):
        calls.append((sorted(symbols), as_of))
        return 3, False

    outcome, _ = _run_quality(tmp_path, monkeypatch, refresh)
    assert outcome.status == "ok"
    assert calls == [(sorted(SYMBOLS), NOW.date())]

    from trading.fundamentals.store import FundamentalsStore

    store = FundamentalsStore(tmp_path / "fund")
    assert store.last_refresh() == NOW.date()

    # Second run inside the 7-day window: the gate must skip the refresh.
    _run_quality(tmp_path, monkeypatch, refresh)
    assert len(calls) == 1


def test_degraded_refresh_journals_a_warning_and_run_proceeds(tmp_path, monkeypatch):
    outcome, _ = _run_quality(tmp_path, monkeypatch, lambda *a, **k: (0, True))
    assert outcome.status == "ok"
    warnings = _last_run_event(tmp_path)["warnings"]
    assert any(w.startswith("fundamentals refresh degraded") for w in warnings)

    from trading.fundamentals.store import FundamentalsStore

    # The load-bearing distinction from total failure: a PARTIAL degradation
    # still advances the marker -- the weekly cadence stands.
    assert FundamentalsStore(tmp_path / "fund").last_refresh() == NOW.date()


def test_failed_refresh_is_fail_open_with_journaled_warning(tmp_path, monkeypatch):
    def refresh(*args, **kwargs):
        raise OSError("edgar unreachable")

    outcome, notes = _run_quality(tmp_path, monkeypatch, refresh)
    assert outcome.status == "ok"  # rankings ran on the (empty) stored fundamentals
    warnings = _last_run_event(tmp_path)["warnings"]
    assert any(w.startswith("fundamentals refresh failed") for w in warnings)

    from trading.fundamentals.store import FundamentalsStore

    # Marker NOT written on total failure: the next run retries immediately.
    assert FundamentalsStore(tmp_path / "fund").last_refresh() is None


def test_corrupt_marker_file_fails_open_with_journaled_warning(tmp_path, monkeypatch):
    # A garbage .last_refresh makes last_refresh() raise ValueError BEFORE
    # the staleness decision; that too must degrade, never escape run_venue.
    marker = tmp_path / "fund" / ".last_refresh"
    marker.parent.mkdir(parents=True)
    marker.write_text("not-a-date")

    calls: list[list[str]] = []

    def recorder(store, cik_map, symbols, as_of, **kwargs):
        calls.append(sorted(symbols))
        return 0, False

    outcome, _ = _run_quality(tmp_path, monkeypatch, recorder)
    assert outcome.status == "ok"
    assert calls == []  # never got as far as the refresh
    warnings = _last_run_event(tmp_path)["warnings"]
    assert any(w.startswith("fundamentals refresh failed") for w in warnings)
    assert marker.read_text() == "not-a-date"  # untouched: next run retries


def test_store_construction_failure_fails_open_with_journaled_warning(tmp_path, monkeypatch):
    # Store construction (mkdir) sits INSIDE the fail-open block: an
    # unwritable fundamentals dir degrades with a warning, never a traceback.
    def boom(root):
        raise PermissionError(f"mkdir denied: {root}")

    monkeypatch.setattr("trading.runner.FundamentalsStore", boom)
    outcome, _ = _run_quality(tmp_path, monkeypatch, lambda *a, **k: (0, False))
    assert outcome.status == "ok"
    warnings = _last_run_event(tmp_path)["warnings"]
    assert any(w.startswith("fundamentals refresh failed") for w in warnings)


def test_stale_marker_on_processed_bar_refreshes_once_and_noops_cleanly(tmp_path, monkeypatch):
    # Bounded-cost tradeoff, pinned: the refresh gate runs before the has_run
    # noop check (which needs build_rankings for decision_ts), so a rerun on
    # an already-processed bar with a stale marker refreshes exactly once
    # more, advances the marker, and the run still noops cleanly.
    calls: list[datetime.date] = []

    def refresh(store, cik_map, symbols, as_of, **kwargs):
        calls.append(as_of)
        return 0, False

    outcome, _ = _run_quality(tmp_path, monkeypatch, refresh)
    assert outcome.status == "ok"
    assert len(calls) == 1

    from trading.fundamentals.store import FundamentalsStore

    store = FundamentalsStore(tmp_path / "fund")
    store.mark_refreshed(NOW.date() - datetime.timedelta(days=8))  # expire the gate

    outcome, _ = _run_quality(tmp_path, monkeypatch, refresh)
    assert outcome.status == "noop"  # decision bar already processed
    assert len(calls) == 2  # the one extra refresh the ordering costs
    assert store.last_refresh() == NOW.date()  # marker advanced: gates the week


def test_session_guard_is_dst_aware():
    eq = load_venue_config("equities", Path("config"))
    # July (EDT, UTC-4): 21:45 UTC = 17:45 local -> past the 17:30 deadline: run allowed.
    summer_bar = pd.Timestamp("2026-07-06", tz="UTC")
    summer_now = datetime.datetime(2026, 7, 6, 21, 45, tzinfo=datetime.UTC)
    assert intraday_partial_bar_reason(eq, summer_bar, summer_now) is None
    # November (EST, UTC-5): 21:45 UTC = 16:45 local -> BEFORE the deadline: refuse.
    winter_bar = pd.Timestamp("2026-11-02", tz="UTC")
    winter_now = datetime.datetime(2026, 11, 2, 21, 45, tzinfo=datetime.UTC)
    assert intraday_partial_bar_reason(eq, winter_bar, winter_now) is not None
