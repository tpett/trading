import dataclasses
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading.cli import main
from trading.config import load_venue_config
from trading.venues import make_adapter
from trading.venues.base import DataFetchError
from trading.venues.crypto import CryptoAdapter
from trading.venues.equities import EquitiesAdapter

AS_OF = "2026-07-01"


# --- scaffold behavior (from Task 1) ---


def test_no_command_exits_with_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_help_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_unknown_venue_rejected():
    with pytest.raises(SystemExit) as excinfo:
        main(["rankings", "--venue", "bonds"])
    assert excinfo.value.code == 2


# --- adapter factory ---


def test_make_adapter_dispatches_by_venue_name():
    assert isinstance(make_adapter(load_venue_config("equities", Path("config"))), EquitiesAdapter)
    assert isinstance(make_adapter(load_venue_config("crypto", Path("config"))), CryptoAdapter)


def test_make_adapter_unknown_venue_raises():
    config = dataclasses.replace(load_venue_config("equities", Path("config")), name="bonds")
    with pytest.raises(ValueError, match="bonds"):
        make_adapter(config)


# --- end-to-end fixtures (network fully monkeypatched) ---


def _write_config(tmp_path: Path, venue: str) -> Path:
    """Copy the real venue TOML, pointing its cache at the test tmp dir."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    text = Path("config", f"{venue}.toml").read_text()
    text = text.replace(f'cache_dir = "data/{venue}"', f'cache_dir = "{tmp_path}/cache/{venue}"')
    (cfg_dir / f"{venue}.toml").write_text(text)
    return cfg_dir


def _fake_history(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Deterministic yfinance-shaped frame (naive index, capitalized columns)."""
    rng = np.random.default_rng(sum(map(ord, symbol)))
    idx = pd.date_range(start, end, freq="B")
    rets = rng.normal(0.0005, 0.015, len(idx))
    close = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.uniform(1e5, 1e6, len(idx)),
        },
        index=idx,
    )


def _fake_kraken(pair: str, since_ms: int) -> list[list[float]]:
    """Deterministic ccxt-shaped daily rows ending on AS_OF."""
    rng = np.random.default_rng(sum(map(ord, pair)))
    n = 520
    end = pd.Timestamp(AS_OF, tz="UTC")
    start = end - pd.Timedelta(n - 1, unit="D")
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    rows = []
    for i in range(n):
        ts = int((start + pd.Timedelta(i, unit="D")).timestamp() * 1000)
        c = float(close[i])
        rows.append([ts, c, c * 1.01, c * 0.99, c, float(rng.uniform(1e5, 1e6))])
    return rows


def _setup_equities(tmp_path, monkeypatch) -> Path:
    cfg_dir = _write_config(tmp_path, "equities")
    universe = tmp_path / "equities_universe.csv"
    universe.write_text("symbol\nAAA\nBBB\nCCC\nDDD\n")
    monkeypatch.setattr("trading.venues.equities.DEFAULT_UNIVERSE_CSV", universe)
    monkeypatch.setattr("trading.venues.equities._yf_download", _fake_history)
    monkeypatch.setattr("trading.runner.fetch_earnings_dates", lambda symbols: ({}, False))
    return cfg_dir


def _setup_crypto(tmp_path, monkeypatch) -> Path:
    cfg_dir = _write_config(tmp_path, "crypto")
    universe = tmp_path / "crypto_universe.csv"
    universe.write_text("symbol,status\nBTC,tradable\nETH,tradable\nSOL,sell_only\n")
    monkeypatch.setattr("trading.venues.crypto.DEFAULT_UNIVERSE_CSV", universe)
    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", _fake_kraken)
    return cfg_dir


# --- end-to-end tests ---


def test_rankings_json_equities_end_to_end(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    rc = main(
        [
            "rankings",
            "--venue",
            "equities",
            "--as-of",
            AS_OF,
            "--json",
            "--config-dir",
            str(cfg_dir),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["venue"] == "equities"
    assert payload["as_of"] == AS_OF
    assert payload["regime"]["state"] in {"risk_on", "neutral", "risk_off"}
    assert payload["coverage"] == {"requested": 4, "fetched": 4, "ratio": 1.0}
    assert {row["symbol"] for row in payload["rankings"]} == {"AAA", "BBB", "CCC", "DDD"}
    composites = [row["composite"] for row in payload["rankings"]]
    assert composites == sorted(composites, reverse=True)
    assert all("raw_return_30d" in row for row in payload["rankings"])


def test_rankings_json_crypto_end_to_end(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_crypto(tmp_path, monkeypatch)
    rc = main(
        ["rankings", "--venue", "crypto", "--as-of", AS_OF, "--json", "--config-dir", str(cfg_dir)]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["venue"] == "crypto"
    by_symbol = {row["symbol"]: row for row in payload["rankings"]}
    assert set(by_symbol) == {"BTC", "ETH", "SOL"}
    assert by_symbol["SOL"]["status"] == "sell_only"  # sell_only stays rankable


def test_rankings_table_renders_human_output(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    rc = main(["rankings", "--venue", "equities", "--as-of", AS_OF, "--config-dir", str(cfg_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "regime" in out
    assert "AAA" in out


def test_coverage_failure_warns_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)

    def flaky(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        if symbol in {"BBB", "CCC"}:
            raise DataFetchError(f"boom {symbol}")
        return _fake_history(symbol, start, end)

    monkeypatch.setattr("trading.venues.equities._yf_download", flaky)
    rc = main(
        [
            "rankings",
            "--venue",
            "equities",
            "--as-of",
            AS_OF,
            "--json",
            "--config-dir",
            str(cfg_dir),
        ]
    )
    assert rc == 1
    assert "WARNING" in capsys.readouterr().err


# --- trading run (Task 10) ---


def _run_args(tmp_path, cfg_dir, extra=()):
    return [
        "run",
        "--venue",
        "equities",
        "--config-dir",
        str(cfg_dir),
        "--state-dir",
        str(tmp_path / "state"),
        "--journal-dir",
        str(tmp_path / "journal"),
        "--digest-dir",
        str(tmp_path / "digest"),
        *extra,
    ]


def _freeze_now(monkeypatch, iso: str):
    frozen = datetime.datetime.fromisoformat(iso)
    monkeypatch.setattr("trading.cli._utcnow", lambda: frozen)


def test_run_bootstraps_then_noops_same_bar(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    assert main(_run_args(tmp_path, cfg_dir)) == 0
    assert (tmp_path / "state" / "equities" / "portfolio.json").exists()
    assert (tmp_path / "journal" / "equities.jsonl").exists()
    capsys.readouterr()

    assert main(_run_args(tmp_path, cfg_dir)) == 0
    assert "noop" in capsys.readouterr().out


def test_run_next_day_fills_and_reports_json(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()
    _freeze_now(monkeypatch, "2026-07-02T22:30:00+00:00")
    rc = main(_run_args(tmp_path, cfg_dir, extra=["--json"]))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["run_key"] == "equities:2026-07-02T00:00:00+00:00"


def test_run_coverage_failure_exits_nonzero_and_notifies(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")

    def flaky(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        if symbol in {"BBB", "CCC"}:
            raise DataFetchError(f"boom {symbol}")
        return _fake_history(symbol, start, end)

    monkeypatch.setattr("trading.venues.equities._yf_download", flaky)
    notes = []
    monkeypatch.setattr("trading.cli.notify", lambda t, m: notes.append((t, m)))
    assert main(_run_args(tmp_path, cfg_dir)) == 1
    assert notes  # every failed run fires a notification


def test_run_restore_from_journal_requires_typed_confirmation(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    state_file = tmp_path / "state" / "equities" / "portfolio.json"
    good = state_file.read_text()
    state_file.write_text("garbage")
    capsys.readouterr()

    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    assert main(_run_args(tmp_path, cfg_dir, extra=["--restore-from-journal"])) == 1
    assert state_file.read_text() == "garbage"

    monkeypatch.setattr("builtins.input", lambda prompt="": "RESTORE")
    assert main(_run_args(tmp_path, cfg_dir, extra=["--restore-from-journal"])) == 0
    assert json.loads(state_file.read_text()) == json.loads(good)


def test_run_restore_with_corrupt_journal_fails_cleanly(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    journal_file = tmp_path / "journal" / "equities.jsonl"
    journal_file.write_text("{ corrupt\n" + journal_file.read_text())
    state_file = tmp_path / "state" / "equities" / "portfolio.json"
    state_file.write_text("garbage")
    capsys.readouterr()

    monkeypatch.setattr("builtins.input", lambda prompt="": "RESTORE")
    assert main(_run_args(tmp_path, cfg_dir, extra=["--restore-from-journal"])) == 1
    assert "ERROR" in capsys.readouterr().err  # clean message, no traceback
    assert state_file.read_text() == "garbage"  # state untouched


def test_run_restore_from_journal_refuses_when_lock_held(tmp_path, monkeypatch, capsys):
    import os

    from trading.runner import lock_path

    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    state_file = tmp_path / "state" / "equities" / "portfolio.json"
    state_file.write_text("garbage")
    lock = lock_path(tmp_path / "state", "equities")
    lock.write_text(str(os.getpid()))  # a live process holds the run lock
    capsys.readouterr()

    monkeypatch.setattr("builtins.input", lambda prompt="": "RESTORE")
    assert main(_run_args(tmp_path, cfg_dir, extra=["--restore-from-journal"])) == 1
    assert "another run is in progress" in capsys.readouterr().err
    assert state_file.read_text() == "garbage"  # untouched
    assert lock.exists()  # never steal a live lock


def test_digest_command_prints_latest_and_specific(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    digest_args = ["--digest-dir", str(tmp_path / "digest")]
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()

    assert main(["digest", *digest_args]) == 0
    assert "Trading digest — 2026-07-01" in capsys.readouterr().out

    assert main(["digest", "--date", "2026-07-01", *digest_args]) == 0
    assert "equities" in capsys.readouterr().out

    assert main(["digest", "--date", "1999-01-01", *digest_args]) == 1


# --- trading status / reset-breaker (Task 12) ---


def _store_args(tmp_path):
    return ["--state-dir", str(tmp_path / "state"), "--journal-dir", str(tmp_path / "journal")]


def test_status_reports_both_venues(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()
    _freeze_now(monkeypatch, "2026-07-02T01:30:00+00:00")  # 3h after the run

    assert main(["status", "--json", *_store_args(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    by_venue = {v["venue"]: v for v in payload["venues"]}
    equities = by_venue["equities"]
    assert equities["state"] == "ok"
    assert equities["value"] == pytest.approx(1000.0)
    assert equities["pnl_pct"] == pytest.approx(0.0)
    assert "benchmark_pnl_pct" in equities
    assert equities["breaker_tripped"] is False
    assert equities["hours_since_last_success"] == pytest.approx(3.0)
    assert by_venue["crypto"]["state"] == "not bootstrapped"


def test_status_flags_corrupt_state(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    (tmp_path / "state" / "equities" / "portfolio.json").write_text("garbage")
    capsys.readouterr()
    assert main(["status", "--json", *_store_args(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    by_venue = {v["venue"]: v for v in payload["venues"]}
    assert by_venue["equities"]["state"] == "corrupt"


def test_status_human_output_renders(tmp_path, monkeypatch, capsys):
    _freeze_now(monkeypatch, "2026-07-02T01:30:00+00:00")
    assert main(["status", *_store_args(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "equities" in out and "crypto" in out


def test_reset_breaker_requires_typed_confirmation(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()

    from trading.runner import load_state, save_state, state_path

    path = state_path(tmp_path / "state", "equities")
    state = load_state(path)
    state.breaker_tripped = True
    state.breaker_tripped_at = "2026-07-01T00:00:00+00:00"
    state.high_water_mark = 5000.0
    save_state(path, state)

    monkeypatch.setattr("builtins.input", lambda prompt="": "nope")
    assert main(["reset-breaker", "--venue", "equities", *_store_args(tmp_path)]) == 1
    assert load_state(path).breaker_tripped is True

    monkeypatch.setattr("builtins.input", lambda prompt="": "RESET")
    assert main(["reset-breaker", "--venue", "equities", *_store_args(tmp_path)]) == 0
    restored = load_state(path)
    assert restored.breaker_tripped is False
    assert restored.breaker_tripped_at is None
    # HWM rebased to the last journaled snapshot value, not the stale 5000.
    assert restored.high_water_mark == pytest.approx(1000.0)

    from trading.journal import Journal

    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    assert events[-1]["event"] == "breaker_reset"


def test_reset_breaker_when_not_tripped_is_a_noop(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()
    assert main(["reset-breaker", "--venue", "equities", *_store_args(tmp_path)]) == 0
    assert "not tripped" in capsys.readouterr().out


def test_reset_breaker_without_state_errors(tmp_path, capsys):
    assert main(["reset-breaker", "--venue", "equities", *_store_args(tmp_path)]) == 1


def _trip_breaker(tmp_path, hwm=5000.0):
    from trading.runner import load_state, save_state, state_path

    path = state_path(tmp_path / "state", "equities")
    state = load_state(path)
    state.breaker_tripped = True
    state.breaker_tripped_at = "2026-07-01T00:00:00+00:00"
    state.high_water_mark = hwm
    save_state(path, state)
    return path


def test_reset_breaker_journals_first_and_restore_replays_reset(tmp_path, monkeypatch, capsys):
    """Crash between the two writes must leave state BEHIND the journal (the
    recoverable direction), and restore must re-apply the journaled reset."""
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()

    from trading.journal import Journal
    from trading.runner import load_state, save_state

    path = _trip_breaker(tmp_path)
    before = path.read_text()

    monkeypatch.setattr("builtins.input", lambda prompt="": "RESET")

    def torn_write(path, state):
        raise OSError("disk full")

    monkeypatch.setattr("trading.cli.save_state", torn_write)
    assert main(["reset-breaker", "--venue", "equities", *_store_args(tmp_path)]) == 1
    assert path.read_text() == before  # state untouched by the failed reset

    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    reset_event = events[-1]
    assert reset_event["event"] == "breaker_reset"
    assert reset_event["high_water_mark"] == pytest.approx(1000.0)
    assert reset_event["last_run_key"] == "equities:2026-07-01T00:00:00+00:00"

    # restore_from_journal re-applies the journaled reset on top of the run
    # snapshot: it can never silently undo an operator's reset.
    monkeypatch.setattr("trading.cli.save_state", save_state)
    monkeypatch.setattr("builtins.input", lambda prompt="": "RESTORE")
    assert main(_run_args(tmp_path, cfg_dir, extra=["--restore-from-journal"])) == 0
    restored = load_state(path)
    assert restored.breaker_tripped is False
    assert restored.breaker_tripped_at is None
    assert restored.high_water_mark == pytest.approx(1000.0)


def test_reset_breaker_refuses_when_state_behind_journal(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()

    from trading.runner import load_state, save_state, state_path

    path = state_path(tmp_path / "state", "equities")
    state = load_state(path)
    state.breaker_tripped = True
    state.last_run_key = "equities:1999-01-01T00:00:00+00:00"  # stale state file
    save_state(path, state)

    monkeypatch.setattr("builtins.input", lambda prompt="": "RESET")
    assert main(["reset-breaker", "--venue", "equities", *_store_args(tmp_path)]) == 1
    assert "behind journal" in capsys.readouterr().err
    assert load_state(path).breaker_tripped is True  # untouched


def test_reset_breaker_refuses_when_lock_held(tmp_path, monkeypatch, capsys):
    import os

    from trading.runner import lock_path

    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()

    from trading.runner import load_state, state_path

    _trip_breaker(tmp_path)
    lock = lock_path(tmp_path / "state", "equities")
    lock.write_text(str(os.getpid()))  # a live process holds the run lock

    monkeypatch.setattr("builtins.input", lambda prompt="": "RESET")
    assert main(["reset-breaker", "--venue", "equities", *_store_args(tmp_path)]) == 1
    assert "another run is in progress" in capsys.readouterr().err
    assert load_state(state_path(tmp_path / "state", "equities")).breaker_tripped is True
    assert lock.exists()  # never steal a live lock


def test_reset_breaker_eof_at_prompt_aborts_cleanly(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()

    from trading.runner import load_state

    path = _trip_breaker(tmp_path)

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert main(["reset-breaker", "--venue", "equities", *_store_args(tmp_path)]) == 1
    assert "aborted" in capsys.readouterr().out
    assert load_state(path).breaker_tripped is True


def test_schedule_status_cli_json(tmp_path, monkeypatch, capsys):
    import subprocess as sp

    monkeypatch.setattr(
        "trading.schedule._launchctl",
        lambda *a: sp.CompletedProcess(args=a, returncode=113, stdout="", stderr=""),
    )
    rc = main(["schedule", "status", "--agents-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "equities": {"installed": False, "loaded": False},
        "crypto": {"installed": False, "loaded": False},
    }
