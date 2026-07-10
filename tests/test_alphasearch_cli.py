"""CLI wiring for `trading alphasearch` (journal-only paths; no data dirs)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from trading import cli
from trading.alphasearch.sweep import (
    DISCOVERY_WINDOW,
    log_trial,
    trial_config,
    trials_journal,
)


def _stub_spy(monkeypatch) -> None:
    """Keep robustness/leaderboard CLI tests off the real (gitignored)
    data/equities-tiingo/SPY cache -- a synthetic series covering any
    fixture window used here."""
    idx = pd.date_range("2015-01-02", periods=3000, freq="B", tz="UTC")
    spy = pd.Series(100.0 * (1.0003 ** np.arange(len(idx))), index=idx)
    monkeypatch.setattr("trading.alphasearch.costs.load_spy_closes",
                        lambda *a, **k: spy)


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
    # Piece 3: the holdout now requires a battery-passed verdict.
    log_trial(journal, kind="battery",
              config=trial_config("mom21", "largecap", DISCOVERY_WINDOW),
              ts="t1b", result={"eligible": True})
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


def test_sweep_segments_refusal_prints_signal_family_hint(tmp_path, monkeypatch, capsys):
    from trading.alphasearch import sweep as engine

    _stub_segments(monkeypatch, tmp_path)

    def fake_run_sweep(universes, journal, factors, ts, **kwargs):
        raise engine.SweepError("mismatches: largecap:banks wants options, has none")

    monkeypatch.setattr("trading.alphasearch.sweep.run_sweep", fake_run_sweep)
    rc = cli.main(["alphasearch", "sweep", "--segments", "--journal-dir", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "hint:" in err
    assert "mom21,mom63,mom126,mom252,rev5,rvol21,disthigh" in err
    assert "segment-safe signals" in err
    assert "ind_rel_rev" in err  # within-sector demeaning still varies -> safe
    # ind_mom is structurally degenerate on single-sector segments (fix 2):
    # excluded from the suggestion so the hint never steers an operator into
    # a predictable SortError/skip.
    assert "ind_mom" not in err
    # requires_insider signals are excluded from the segment-safe list for
    # the same reason as fundamentals: their store may not be synced, and a
    # hint containing them would hand the operator a still-failing command.
    assert "npr_90" not in err
    assert "cluster_buys_90" not in err
    assert "officer_buy_90" not in err
    assert "opt-largecap:<segment>" in err


def test_sweep_refusal_without_segments_has_no_hint(tmp_path, monkeypatch, capsys):
    from trading.alphasearch import sweep as engine

    monkeypatch.setattr(
        "trading.alphasearch.evaluate.load_factors", lambda *a, **k: pd.DataFrame()
    )

    def fake_run_sweep(universes, journal, factors, ts, **kwargs):
        raise engine.SweepError("boom")

    monkeypatch.setattr("trading.alphasearch.sweep.run_sweep", fake_run_sweep)
    rc = cli.main(["alphasearch", "sweep", "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "hint:" not in capsys.readouterr().err


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


# --------------------------------------------------------------------------- #
# robustness (Piece 3)
# --------------------------------------------------------------------------- #


def test_robustness_requires_trial_id(tmp_path, capsys):
    rc = cli.main(["alphasearch", "robustness", "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "signal:universe" in capsys.readouterr().err


def test_robustness_unknown_universe_rejected(tmp_path, capsys):
    rc = cli.main(["alphasearch", "robustness", "mom21:smallcap",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "unknown universe" in capsys.readouterr().err


def test_robustness_refusal_prints_error(tmp_path, capsys, monkeypatch):
    # Empty journal: run_battery refuses (no discovery trial) before any IO
    # beyond factors, which the stub keeps offline.
    monkeypatch.setattr("trading.alphasearch.evaluate.load_factors",
                        lambda *a, **k: pd.DataFrame())
    _stub_spy(monkeypatch)
    rc = cli.main(["alphasearch", "robustness", "mom21:largecap",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "no discovery trial" in capsys.readouterr().err


def _fake_battery_outcome(*, eligible=False, flagged=True):
    from trading.alphasearch.robustness import BatteryOutcome, CheckResult

    checks = (
        CheckResult(1, "sub_period_halves", True, {"halves": [
            {"window": "2019-01-01..2021-06-30", "alpha_annual_pct": 30.0,
             "alpha_t": 4.0, "error": None, "passed": True},
            {"window": "2021-07-01..2023-12-31", "alpha_annual_pct": 25.0,
             "alpha_t": 3.0, "error": None, "passed": True}]}),
        CheckResult(2, "universe_subsets", True,
                    {"draws": [], "n_pass": 5}),
        CheckResult(3, "parameter_jitter", True, {"trials": []}),
        CheckResult(4, "decision_offset", True,
                    {"offset_sessions": 1, "alpha_annual_pct": 28.0,
                     "alpha_t": 3.5, "retention": 0.9, "error": None}),
        CheckResult(5, "name_concentration", False,
                    {"excluded": ["AAA", "BBB", "CCC"],
                     "alpha_annual_pct": 5.0, "retention": 0.16, "error": None}),
        CheckResult(6, "month_concentration", True, {"top3_share": 0.41}),
    )
    return BatteryOutcome(
        signal="amihud", universe="midcap", window="2019-01-01..2023-12-31",
        checks=checks,
        factor_proxy={"flagged": flagged, "offenders": {"SMB": 13.1},
                      "alpha_t": 3.0, "r2": 0.6},
        cost_table=[{"cost_bps": 10, "alpha_annual_pct": 55.0, "alpha_t": 7.1},
                    {"cost_bps": 30, "alpha_annual_pct": 48.0, "alpha_t": 6.0},
                    {"cost_bps": 50, "alpha_annual_pct": 41.0, "alpha_t": 4.9}],
        capacity_curve=[
            {"book_usd": 1e4, "alpha_annual_pct": 60.0, "alpha_t": 8.0,
             "total_impact_charge": 0.01, "skipped_no_lambda": 0},
            {"book_usd": 1e5, "alpha_annual_pct": 52.0, "alpha_t": 6.8,
             "total_impact_charge": 0.1, "skipped_no_lambda": 0},
            {"book_usd": 1e6, "alpha_annual_pct": 20.0, "alpha_t": 2.1,
             "total_impact_charge": 1.0, "skipped_no_lambda": 0}],
        long_only_gate={"lo_sharpe": 0.9, "lo_total_return": 0.35,
                        "spy_sharpe": 0.6, "spy_total_return": 0.25,
                        "skipped_no_spread": 0, "passed": eligible},
        eligible=eligible,
        event={"event": "trial", "kind": "battery", "signal": "amihud",
               "universe": "midcap", "window": "2019-01-01..2023-12-31",
               "config_hash": "4f3d0819382a", "ts": "t1", "error": None,
               "eligible": eligible},
    )


def test_robustness_report_card_renders_with_red_proxy_warning(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr("trading.alphasearch.evaluate.load_factors",
                        lambda *a, **k: pd.DataFrame())
    _stub_spy(monkeypatch)
    monkeypatch.setattr("trading.alphasearch.robustness.run_battery",
                        lambda *a, **k: _fake_battery_outcome())
    rc = cli.main(["alphasearch", "robustness", "amihud:midcap",
                   "--journal-dir", str(tmp_path)])
    assert rc == 0                                   # completed battery = 0
    out = capsys.readouterr().out
    assert "name_concentration" in out and "FAIL" in out
    assert "sub_period_halves" in out and "PASS" in out
    assert "FACTOR-PROXY WARNING" in out and "SMB" in out
    assert "30" in out and "capacity" in out.lower()
    assert "holdout-eligible: NO" in out


def test_robustness_report_card_surfaces_errored_subset_draws(
    tmp_path, capsys, monkeypatch
):
    """Fix (final-review): an ERRORED draw (e.g. the half-universe fell below
    MIN_NAMES) is a different failure mode than a merely sign-mismatched
    draw, and the report card must say so instead of collapsing both into
    the same "sign-matched" count."""
    from trading.alphasearch.robustness import CheckResult

    outcome = _fake_battery_outcome()
    errored_subsets = CheckResult(
        2, "universe_subsets", False,
        {"n_pass": 0, "draws": [
            {"seed": 42 + i, "n_symbols": 5, "alpha_annual_pct": None,
             "error": "SortError: cross-section below minimum", "passed": False}
            for i in range(5)
        ]},
    )
    checks = tuple(
        errored_subsets if c.name == "universe_subsets" else c
        for c in outcome.checks
    )
    outcome = replace(outcome, checks=checks)
    monkeypatch.setattr("trading.alphasearch.evaluate.load_factors",
                        lambda *a, **k: pd.DataFrame())
    _stub_spy(monkeypatch)
    monkeypatch.setattr("trading.alphasearch.robustness.run_battery",
                        lambda *a, **k: outcome)
    rc = cli.main(["alphasearch", "robustness", "amihud:midcap",
                   "--journal-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "5 errored: SortError: cross-section below minimum" in out
    assert "0/5 draws sign-matched" in out


def test_robustness_json_dumps_the_verdict_event(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("trading.alphasearch.evaluate.load_factors",
                        lambda *a, **k: pd.DataFrame())
    _stub_spy(monkeypatch)
    monkeypatch.setattr("trading.alphasearch.robustness.run_battery",
                        lambda *a, **k: _fake_battery_outcome(eligible=True))
    rc = cli.main(["alphasearch", "robustness", "amihud:midcap",
                   "--journal-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "battery"
    assert payload["eligible"] is True
    assert payload["signal"] == "amihud"
