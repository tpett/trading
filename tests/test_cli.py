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
