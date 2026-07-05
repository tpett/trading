import dataclasses
from pathlib import Path

import pytest

from trading.config import load_venue_config
from trading.journal import Journal, JournalError, config_hash


def _journal(tmp_path) -> Journal:
    return Journal(tmp_path / "journal" / "equities.jsonl")


def test_append_and_read_round_trip(tmp_path):
    journal = _journal(tmp_path)
    journal.append({"event": "bootstrap", "venue": "equities"})
    journal.append({"event": "run", "run_key": "equities:2026-07-01T00:00:00+00:00"})
    events = list(journal.events())
    assert [e["event"] for e in events] == ["bootstrap", "run"]


def test_missing_file_yields_no_events(tmp_path):
    assert list(_journal(tmp_path).events()) == []
    assert _journal(tmp_path).has_run("equities:2026-07-01T00:00:00+00:00") is False


def test_has_run_finds_run_key(tmp_path):
    journal = _journal(tmp_path)
    journal.append({"event": "run", "run_key": "equities:2026-07-01T00:00:00+00:00"})
    assert journal.has_run("equities:2026-07-01T00:00:00+00:00") is True
    assert journal.has_run("equities:2026-07-02T00:00:00+00:00") is False


def test_last_event_with_type_filter(tmp_path):
    journal = _journal(tmp_path)
    journal.append({"event": "bootstrap"})
    journal.append({"event": "run", "n": 1})
    journal.append({"event": "run_failed"})
    journal.append({"event": "run", "n": 2})
    assert journal.last_event()["event"] == "run"
    assert journal.last_event(types=frozenset({"run"}))["n"] == 2
    assert journal.last_event(types=frozenset({"bootstrap"}))["event"] == "bootstrap"
    assert journal.last_event(types=frozenset({"nope"})) is None


def test_torn_final_line_is_skipped(tmp_path):
    journal = _journal(tmp_path)
    journal.append({"event": "run", "n": 1})
    with (tmp_path / "journal" / "equities.jsonl").open("a") as f:
        f.write('{"event": "run", "n"')  # crash mid-append
    assert [e["n"] for e in journal.events()] == [1]


def test_append_after_torn_final_line_preserves_new_event(tmp_path):
    journal = _journal(tmp_path)
    journal.append({"event": "run", "n": 1})
    with (tmp_path / "journal" / "equities.jsonl").open("a") as f:
        f.write('{"event": "run", "n"')  # crash mid-append
    journal.append({"event": "run", "n": 2})
    assert [e["n"] for e in journal.events()] == [1, 2]


def test_append_after_torn_final_line_has_run_sees_new_key(tmp_path):
    journal = _journal(tmp_path)
    journal.append({"event": "run", "run_key": "equities:2026-07-01T00:00:00+00:00"})
    with (tmp_path / "journal" / "equities.jsonl").open("a") as f:
        f.write('{"event": "run", "run_key": "equities:2026-07-02T00:00')  # torn
    journal.append({"event": "run", "run_key": "equities:2026-07-03T00:00:00+00:00"})
    assert journal.has_run("equities:2026-07-03T00:00:00+00:00") is True
    # The torn record was never durable; it must not count as run.
    assert journal.has_run("equities:2026-07-02T00:00:00+00:00") is False


def test_append_to_empty_file_works(tmp_path):
    path = tmp_path / "journal" / "equities.jsonl"
    journal = Journal(path)
    path.touch()
    journal.append({"event": "run", "n": 1})
    assert [e["n"] for e in journal.events()] == [1]


def test_corruption_mid_file_raises(tmp_path):
    path = tmp_path / "journal" / "equities.jsonl"
    journal = Journal(path)
    journal.append({"event": "run", "n": 1})
    with path.open("a") as f:
        f.write("not json\n")
    journal.append({"event": "run", "n": 2})
    with pytest.raises(JournalError, match="line 2"):
        list(journal.events())


def test_config_hash_is_stable_and_sensitive(tmp_path):
    config = load_venue_config("equities", Path("config"))
    assert config_hash(config) == config_hash(config)
    assert len(config_hash(config)) == 12
    changed = dataclasses.replace(
        config, portfolio=dataclasses.replace(config.portfolio, max_positions=4)
    )
    assert config_hash(changed) != config_hash(config)


def test_append_to_fully_torn_single_line_file(tmp_path):
    path = tmp_path / "torn.jsonl"
    path.write_bytes(b'{"event": "run", "torn')  # no newline anywhere in the file
    journal = Journal(path)
    assert list(journal.events()) == []

    journal.append({"event": "after_repair"})
    events = list(journal.events())
    assert [e["event"] for e in events] == ["after_repair"]
    # The torn fragment is gone entirely, not merged into the new line.
    assert path.read_text().count("\n") == 1
    assert "torn" not in path.read_text()
