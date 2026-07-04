import datetime
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from sim_helpers import EQ
from trading.data.cache import OhlcvCache
from trading.journal import Journal
from trading.runner import (
    RunLock,
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
    # config/equities.toml enables the earnings filter; keep tests offline.
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


def test_first_run_state_write_failure_does_not_double_bootstrap(tmp_path, monkeypatch):
    def boom(path, state):
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr("trading.runner.save_state", boom)
        outcome, notes = _run(tmp_path)
    assert outcome.status == "failed"
    assert notes

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
