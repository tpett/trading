import datetime
import json
from pathlib import Path

import pytest

import trading.cli as cli
from backtest_helpers import FakeBacktestAdapter, noisy_frame, small_config
from trading.backtest.experiments import experiments_journal
from trading.cli import main


@pytest.fixture()
def backtest_env(tmp_path, monkeypatch):
    """A crypto-venue config dir + fake adapter wired through the real CLI."""
    frames = {
        "AAA": noisy_frame(seed=1, drift=0.008, periods=430, start="2025-01-01"),
        "BBB": noisy_frame(seed=2, drift=0.001, periods=430, start="2025-01-01"),
        "CCC": noisy_frame(seed=3, drift=-0.002, periods=430, start="2025-01-01"),
        "BTC": noisy_frame(seed=9, drift=0.002, periods=430, start="2025-01-01"),
    }
    config = small_config(
        train_months=6,
        test_months=2,
        entry_score_threshold_grid=(0.5, 0.7),
        stop_atr_multiple_grid=(1.0, 2.0),
        stress_segments=((datetime.date(2025, 8, 1), datetime.date(2025, 9, 30)),),
        start=datetime.date(2025, 2, 1),
        holdout_start=datetime.date(2025, 12, 1),
    )
    # Write the small config as a real TOML the CLI can load.
    _write_venue_toml(
        tmp_path / "config" / "crypto.toml", config, cache_dir=str(tmp_path / "cache")
    )
    adapter = FakeBacktestAdapter(frames, "BTC")
    monkeypatch.setattr(cli, "make_adapter", lambda config: adapter)
    monkeypatch.setattr(
        cli, "_utcnow", lambda: datetime.datetime(2026, 1, 15, 3, 0, tzinfo=datetime.UTC)
    )
    return tmp_path


def _write_venue_toml(path: Path, config, cache_dir: str) -> None:
    """Serialize a VenueConfig back to TOML for CLI-level tests."""
    from dataclasses import asdict

    raw = asdict(config)
    raw["data"]["cache_dir"] = cache_dir
    lines = ["[venue]", f'name = "{config.name}"', 'benchmark = "BTC"']
    for section in ("costs", "universe", "signals", "regime", "portfolio", "data", "backtest"):
        lines.append(f"[{section}]")
        for key, value in raw[section].items():
            lines.append(f"{key} = {_toml_value(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _toml_value(value) -> str:
    # Two container shapes exist in VenueConfig: flat number lists (grids,
    # windows) and date-pair lists (stress_segments, rendered as string pairs).
    import datetime as dt

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, list | tuple):
        if value and isinstance(value[0], list | tuple):
            inner = ", ".join(f'["{a.isoformat()}", "{b.isoformat()}"]' for a, b in value)
        else:
            inner = ", ".join(_toml_value(v) for v in value)
        return f"[{inner}]"
    return str(value)


def test_backtest_json_reports_metrics_gate_and_experiment_count(backtest_env, capsys):
    rc = main(
        [
            "backtest",
            "--venue",
            "crypto",
            "--json",
            "--from",
            "2025-03-01",
            "--to",
            "2025-06-30",
            "--config-dir",
            str(backtest_env / "config"),
            "--journal-dir",
            str(backtest_env / "journal"),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["venue"] == "crypto"
    assert "sharpe" in payload["metrics"] and "fee_drag" in payload["metrics"]
    assert payload["gate_passed"] in (True, False)
    assert payload["experiment_count"] == 1
    assert 0.0 < payload["survivorship_ratio"] <= 1.0
    # The coverage-gate denominator is surfaced so a shrinking historical
    # universe (today-snapshot venues) is visible in machine output.
    assert payload["eligible_members"]["min"] >= 1
    assert payload["eligible_members"]["mean"] >= payload["eligible_members"]["min"]
    journal = experiments_journal(backtest_env / "journal", "crypto")
    event = next(journal.events())
    assert event["kind"] == "backtest"
    assert event["grid_point"]["entry_score_threshold"] == 0.55  # TOML value, journaled


def test_backtest_clamps_to_before_holdout(backtest_env, capsys):
    rc = main(
        [
            "backtest",
            "--venue",
            "crypto",
            "--json",
            "--from",
            "2025-03-01",
            "--to",
            "2026-01-10",  # inside the holdout
            "--config-dir",
            str(backtest_env / "config"),
            "--journal-dir",
            str(backtest_env / "journal"),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["to"] == "2025-11-30"  # holdout_start - 1 day


def test_backtest_from_inside_holdout_is_an_error(backtest_env, capsys):
    rc = main(
        [
            "backtest",
            "--venue",
            "crypto",
            "--from",
            "2025-12-05",
            "--config-dir",
            str(backtest_env / "config"),
            "--journal-dir",
            str(backtest_env / "journal"),
        ]
    )
    assert rc == 1
    assert "holdout" in capsys.readouterr().err.lower()


def test_walk_forward_journals_every_window_plus_summary(backtest_env, capsys):
    rc = main(
        [
            "backtest",
            "--venue",
            "crypto",
            "--walk-forward",
            "--json",
            "--from",
            "2025-02-01",
            "--to",
            "2025-11-30",
            "--config-dir",
            str(backtest_env / "config"),
            "--journal-dir",
            str(backtest_env / "journal"),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["windows"], "expected at least one OOS window"
    assert payload["stress_segments_covered"]
    journal = experiments_journal(backtest_env / "journal", "crypto")
    kinds = [e["kind"] for e in journal.events()]
    assert kinds.count("walk_forward_window") == len(payload["windows"])
    assert kinds.count("walk_forward") == 1
    assert payload["experiment_count"] == len(kinds)


def test_holdout_runs_once_then_requires_typed_confirmation(backtest_env, capsys, monkeypatch):
    args = [
        "backtest",
        "--venue",
        "crypto",
        "--holdout",
        "--json",
        "--to",
        "2026-01-14",
        "--config-dir",
        str(backtest_env / "config"),
        "--journal-dir",
        str(backtest_env / "journal"),
    ]
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["from"] == "2025-12-01"  # holdout_start
    journal = experiments_journal(backtest_env / "journal", "crypto")
    assert [e["kind"] for e in journal.events()] == ["holdout"]

    # Second invocation refuses without the typed phrase. Even under --json the
    # refusal must keep stdout empty: stdout is the machine payload channel.
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    assert main(args) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "holdout already evaluated" in captured.err.lower()
    assert [e["kind"] for e in journal.events()] == ["holdout"]

    # And proceeds with it: the JSON payload alone on stdout, notices on stderr.
    monkeypatch.setattr("builtins.input", lambda prompt="": "RERUN HOLDOUT")
    assert main(args) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["kind"] == "holdout"
    assert "holdout already evaluated" in captured.err.lower()
    events = list(journal.events())
    assert [e["kind"] for e in events] == ["holdout", "holdout"]
    # The override is journaled as an explicit rerun pointing at what it spent.
    assert "rerun" not in events[0]
    assert events[1]["rerun"] is True
    assert events[1]["prior_ts"] == events[0]["ts"]


def test_walk_forward_and_holdout_together_is_an_error(backtest_env, capsys):
    rc = main(
        [
            "backtest",
            "--venue",
            "crypto",
            "--walk-forward",
            "--holdout",
            "--config-dir",
            str(backtest_env / "config"),
            "--journal-dir",
            str(backtest_env / "journal"),
        ]
    )
    assert rc == 1
    assert "holdout" in capsys.readouterr().err.lower()
