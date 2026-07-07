"""Tests for the earnings-calendar journaling script: event shape, same-day
idempotency, and partial-progress-on-failure semantics."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dump_earnings_calendar import WINDOWS_DAYS, dump, run_key  # noqa: E402

from trading.journal import Journal
from trading.rhmcp import RhMcpError

TODAY = datetime.date(2026, 7, 6)

REPORT = {
    "symbol": "PEP",
    "eps": {"estimate": "2.21", "actual": None},
    "report": {"date": "2026-07-09", "timing": "am", "verified": True},
}


class FakeClient:
    def __init__(self, fail_on_days: set[int] = frozenset()):
        self.calls = []
        self.fail_on_days = fail_on_days

    def call_tool(self, name, arguments):
        assert name == "get_earnings_calendar"
        self.calls.append(arguments)
        if arguments["days"] in self.fail_on_days:
            raise RhMcpError("HTTP 500")
        return {"data": {"results": [REPORT]}, "guide": "ignored"}


def test_dump_journals_both_windows_point_in_time(tmp_path):
    journal = Journal(tmp_path / "earnings.jsonl")
    client = FakeClient()
    messages = dump(client, journal, TODAY)
    assert [c["days"] for c in client.calls] == list(WINDOWS_DAYS)
    assert all(c["start_date"] == "2026-07-06" for c in client.calls)
    events = list(journal.events())
    assert [e["run_key"] for e in events] == ["earnings:2026-07-06:+14", "earnings:2026-07-06:-7"]
    for event in events:
        assert event["event"] == "earnings_calendar"
        assert event["results"] == [REPORT]
        assert event["fetched_at"].startswith("2026-")  # PIT: when we SAW it
    assert len(messages) == 2


def test_same_day_rerun_is_a_no_op(tmp_path):
    journal = Journal(tmp_path / "earnings.jsonl")
    first = FakeClient()
    dump(first, journal, TODAY)
    second = FakeClient()
    messages = dump(second, journal, TODAY)
    assert second.calls == []  # no network on the catch-up run
    assert len(list(journal.events())) == 2
    assert all("skipping" in m for m in messages)


def test_next_day_appends_new_events(tmp_path):
    journal = Journal(tmp_path / "earnings.jsonl")
    dump(FakeClient(), journal, TODAY)
    dump(FakeClient(), journal, TODAY + datetime.timedelta(days=1))
    assert len(list(journal.events())) == 4


def test_failed_window_keeps_earlier_windows_journaled(tmp_path):
    journal = Journal(tmp_path / "earnings.jsonl")
    with pytest.raises(RhMcpError):
        dump(FakeClient(fail_on_days={-7}), journal, TODAY)
    events = list(journal.events())
    assert [e["run_key"] for e in events] == ["earnings:2026-07-06:+14"]
    # The retry after the failure fills only the missing window.
    retry = FakeClient()
    dump(retry, journal, TODAY)
    assert [c["days"] for c in retry.calls] == [-7]
    assert len(list(journal.events())) == 2


def test_run_key_signs_negative_windows_distinctly():
    assert run_key(TODAY, 14) != run_key(TODAY, -14)


def test_main_notifies_and_exits_nonzero_on_failure(tmp_path, monkeypatch):
    import dump_earnings_calendar as script

    notifications = []
    monkeypatch.setattr(script, "McpClient", lambda path: FakeClient(fail_on_days={14, -7}))
    monkeypatch.setattr(script, "notify", lambda title, msg: notifications.append((title, msg)))
    monkeypatch.setattr(
        "sys.argv",
        [
            "x",
            "--journal",
            str(tmp_path / "e.jsonl"),
            "--token-path",
            str(tmp_path / "t.json"),
            "--state-dir",
            str(tmp_path / "state"),
        ],
    )
    rc = script.main()
    assert rc == 1
    assert notifications and "earnings dump failed" in notifications[0][0]


def test_main_refuses_to_race_a_held_lock(tmp_path, monkeypatch):
    import os

    import dump_earnings_calendar as script

    lock_path = tmp_path / "state" / "earnings" / ".lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(str(os.getpid()))  # a live pid holds the lock
    fetched = []
    monkeypatch.setattr(script, "McpClient", lambda path: fetched.append(path) or FakeClient())
    monkeypatch.setattr(
        "sys.argv",
        [
            "x",
            "--journal",
            str(tmp_path / "e.jsonl"),
            "--token-path",
            str(tmp_path / "t.json"),
            "--state-dir",
            str(tmp_path / "state"),
        ],
    )
    assert script.main() == 1
    assert fetched == []  # never touched the network
    assert lock_path.exists()  # the holder's lock is left alone
