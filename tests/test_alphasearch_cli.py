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


# --------------------------------------------------------------------------- #
# --segments (Piece 2)
# --------------------------------------------------------------------------- #


def _stub_segments(monkeypatch, tmp_path):
    from trading.alphasearch.sweep import UniverseSpec

    seg = UniverseSpec("largecap:banks", tmp_path, None, None, symbols=("A", "B"))
    monkeypatch.setattr(
        "trading.alphasearch.segments.segment_universes",
        lambda root, sic_map_path=None, **kwargs: (
            {"largecap:banks": seg},
            [{"segment": "construction", "cap": "opt-largecap", "count": 3,
              "reason": "below-min"}],
        ),
    )
    monkeypatch.setattr(
        "trading.alphasearch.evaluate.load_factors", lambda *a, **k: pd.DataFrame()
    )
    return seg


def test_sweep_segments_merges_universes_and_prints_exclusions(
    tmp_path, monkeypatch, capsys
):
    _stub_segments(monkeypatch, tmp_path)
    captured = {}

    def fake_run_sweep(universes, journal, factors, ts, **kwargs):
        captured["names"] = set(universes)
        return [], len(universes)

    monkeypatch.setattr("trading.alphasearch.sweep.run_sweep", fake_run_sweep)
    rc = cli.main(["alphasearch", "sweep", "--segments",
                   "--journal-dir", str(tmp_path), "--json"])
    assert rc == 0
    assert captured["names"] == {"largecap", "midcap", "largecap:banks"}
    err = capsys.readouterr().err
    assert "opt-largecap:construction" in err   # the exclusions report, on stderr
    assert "3 names" in err and "below-min" in err


def test_sweep_universe_flag_selects_a_single_segment(tmp_path, monkeypatch, capsys):
    _stub_segments(monkeypatch, tmp_path)
    captured = {}

    def fake_run_sweep(universes, journal, factors, ts, **kwargs):
        captured["names"] = set(universes)
        return [], len(universes)

    monkeypatch.setattr("trading.alphasearch.sweep.run_sweep", fake_run_sweep)
    rc = cli.main(["alphasearch", "sweep", "--segments",
                   "--universe", "largecap:banks",
                   "--journal-dir", str(tmp_path), "--json"])
    assert rc == 0
    assert captured["names"] == {"largecap:banks"}


def test_sweep_unknown_universe_lists_known_names(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "trading.alphasearch.evaluate.load_factors", lambda *a, **k: pd.DataFrame()
    )
    rc = cli.main(["alphasearch", "sweep", "--universe", "nope",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown universe" in err and "largecap" in err


def test_sweep_segments_missing_sic_map_is_an_actionable_error(
    tmp_path, monkeypatch, capsys
):
    from trading.alphasearch.segments import SegmentError

    monkeypatch.setattr(
        "trading.alphasearch.evaluate.load_factors", lambda *a, **k: pd.DataFrame()
    )

    def boom(root, sic_map_path=None, **kwargs):
        raise SegmentError("SIC map not found; build it with "
                           "`uv run python scripts/build_sic_map.py`")

    monkeypatch.setattr("trading.alphasearch.segments.segment_universes", boom)
    rc = cli.main(["alphasearch", "sweep", "--segments", "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "build_sic_map.py" in capsys.readouterr().err


def test_holdout_resolves_segment_universe_names_without_a_flag(
    tmp_path, monkeypatch, capsys
):
    from trading.alphasearch import sweep as engine

    _stub_segments(monkeypatch, tmp_path)
    captured = {}

    def fake_run_holdout(uspec, journal, factors, ts, signal_name, **kwargs):
        captured["universe"] = uspec.name
        raise engine.SweepError("stub refusal after resolution")

    monkeypatch.setattr("trading.alphasearch.sweep.run_holdout", fake_run_holdout)
    # partition(":") splits at the FIRST colon: mom21 / largecap:banks.
    rc = cli.main(["alphasearch", "holdout", "mom21:largecap:banks",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert captured["universe"] == "largecap:banks"
    assert "stub refusal" in capsys.readouterr().err
