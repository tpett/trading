"""CLI wiring for `trading alphasearch` (journal-only paths; no data dirs)."""

from __future__ import annotations

import json

import pandas as pd

from trading import cli
from trading.alphasearch.sweep import (
    DISCOVERY_WINDOW,
    log_trial,
    trial_config,
    trials_journal,
)


def _seed_journal(journal_dir, *, with_holdout: bool = False) -> None:
    journal = trials_journal(journal_dir)
    leg = {"alpha_annual_pct": 8.0, "alpha_t": 4.2, "p": 1e-4,
           "capm_alpha_annual_pct": 9.0, "capm_alpha_t": 4.4,
           "loadings": {"Mkt-RF": 0.1, "SMB": 0.0, "HML": 0.0, "Mom": 0.0},
           "loadings_t": {"Mkt-RF": 1.0, "SMB": 0.0, "HML": 0.0, "Mom": 0.0},
           "r2": 0.1, "n_obs": 1200, "sharpe": 1.1, "sharpe_daily": 0.07,
           "skew": -0.2, "kurt": 4.0}
    result = {"n_dates": 60, "n_names_median": 97.0, "ls": leg, "lo": dict(leg),
              "turnover_monthly": 0.35, "skipped_dates": []}
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", DISCOVERY_WINDOW),
              ts="t1", result=result)
    if with_holdout:
        log_trial(journal, kind="holdout",
                  config=trial_config("mom21", "largecap", "2024-01-01..2026-07-07"),
                  ts="t2", result=result)


def test_leaderboard_json_from_journal(tmp_path, capsys):
    _seed_journal(tmp_path)
    rc = cli.main(["alphasearch", "leaderboard", "--journal-dir", str(tmp_path),
                   "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["trials"] == 1
    assert payload["rows"][0]["signal"] == "mom21"
    assert payload["rows"][0]["bh_pass"] is True


def test_leaderboard_table_renders(tmp_path, capsys):
    _seed_journal(tmp_path)
    rc = cli.main(["alphasearch", "leaderboard", "--journal-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mom21" in out
    assert "honest trial count" in out
    # Classical-SE caveat printed under the table (spec §5 gate is unchanged;
    # this is a read-time warning, not a different pass rule).
    assert "classical OLS SEs" in out
    assert "HAC" in out


def test_leaderboard_empty_journal_is_fine(tmp_path, capsys):
    rc = cli.main(["alphasearch", "leaderboard", "--journal-dir", str(tmp_path),
                   "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["trials"] == 0 and payload["rows"] == []


def test_sweep_unknown_signal_rejected_before_any_io(tmp_path, capsys):
    rc = cli.main(["alphasearch", "sweep", "--signals", "nope,mom21",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "unknown signals: nope" in capsys.readouterr().err
    assert not (tmp_path / "alphasearch-trials.jsonl").exists()


def test_holdout_requires_trial_id(tmp_path, capsys):
    rc = cli.main(["alphasearch", "holdout", "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "signal:universe" in capsys.readouterr().err


def test_holdout_unknown_universe_rejected(tmp_path, capsys):
    rc = cli.main(["alphasearch", "holdout", "mom21:smallcap",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "unknown universe" in capsys.readouterr().err


def test_holdout_double_touch_refused_via_prompt(tmp_path, capsys, monkeypatch):
    _seed_journal(tmp_path, with_holdout=True)
    # Factors are irrelevant to the refusal path; keep the test offline.
    monkeypatch.setattr("trading.alphasearch.evaluate.load_factors",
                        lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    rc = cli.main(["alphasearch", "holdout", "mom21:largecap",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "already evaluated" in capsys.readouterr().err
