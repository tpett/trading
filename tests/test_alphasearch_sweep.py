"""Sweep runner + leaderboard: journaling order, honesty, refusal, recompute."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphasearch_helpers import make_factors, make_panel
from trading.alphasearch.spec import SIGNALS
from trading.alphasearch.sweep import (
    RERUN_CONFIRMATION,
    SweepError,
    UniverseSpec,
    build_leaderboard,
    discovery_trials,
    holdout_passes,
    prior_holdout_trial,
    run_holdout,
    run_sweep,
    trials_journal,
)

WINDOW = "2020-01-01..2020-06-30"


def _universe(tmp_path) -> dict[str, UniverseSpec]:
    # Paths are unused: tests inject panel_factory instead of touching disk.
    dummy = UniverseSpec("largecap", tmp_path, tmp_path / "s.jsonl", None)
    return {"largecap": dummy}


def _subset(*names: str) -> dict:
    return {n: SIGNALS[n] for n in names}


def test_sweep_journals_every_trial_and_ranks_by_abs_t(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    rows, n_trials = run_sweep(
        _universe(tmp_path), journal, make_factors(), ts="t1",
        signals=_subset("mom21", "rev5", "rvol21"), window=WINDOW,
        panel_factory=lambda _u: panel,
    )
    assert n_trials == 3
    assert len(discovery_trials(journal)) == 3      # journaled, not just returned
    assert len(rows) == 3
    ts = [abs(r.alpha_t) for r in rows if r.alpha_t is not None]
    assert ts == sorted(ts, reverse=True)           # sorted by |4F L/S t|
    # The engineered momentum spread must be strongly significant.
    mom = next(r for r in rows if r.signal == "mom21")
    assert abs(mom.alpha_t) > 5
    assert mom.bh_pass
    assert mom.dsr is not None and 0.0 <= mom.dsr <= 1.0  # DSR shown for survivors
    assert set(mom.loadings) == {"Mkt-RF", "SMB", "HML", "Mom"}
    assert mom.turnover_monthly is not None


def test_sweep_rerun_is_idempotent_for_trial_count(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    kwargs = dict(signals=_subset("mom21"), window=WINDOW,
                  panel_factory=lambda _u: panel)
    _, first = run_sweep(_universe(tmp_path), journal, make_factors(), "t1", **kwargs)
    _, second = run_sweep(_universe(tmp_path), journal, make_factors(), "t2", **kwargs)
    assert first == second == 1                     # identical config: ONE trial
    assert len(list(journal.events())) == 2         # ...but both runs appended


def test_error_trial_is_journaled_flagged_and_counted(tmp_path):
    # mom252 needs 253 bars; the fixture has 130 -> every date skips -> SortError.
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    rows, n_trials = run_sweep(
        _universe(tmp_path), journal, make_factors(), ts="t1",
        signals=_subset("mom21", "mom252"), window=WINDOW,
        panel_factory=lambda _u: panel,
    )
    assert n_trials == 2                            # the failure still counts
    failed = next(r for r in rows if r.signal == "mom252")
    assert failed.error is not None and "SortError" in failed.error
    assert failed.alpha_t is None
    assert not failed.bh_pass
    event = next(e for e in discovery_trials(journal) if e["signal"] == "mom252")
    assert event["error"] is not None and "ls" not in event


def test_options_signal_without_cells_refused_before_any_trial(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(with_options=False)
    with pytest.raises(SweepError):
        run_sweep(
            _universe(tmp_path), journal, make_factors(), ts="t1",
            signals=_subset("mom21", "hedge"), window=WINDOW,
            panel_factory=lambda _u: panel,
        )
    assert list(journal.events()) == []             # refused at assembly: no trials


def test_fundamentals_signal_without_store_refused(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(with_fundamentals=False)
    with pytest.raises(SweepError):
        run_sweep(
            _universe(tmp_path), journal, make_factors(), ts="t1",
            signals=_subset("earnings_yield"), window=WINDOW,
            panel_factory=lambda _u: panel,
        )


def test_leaderboard_recomputes_from_journal_alone(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    rows, n_trials = run_sweep(
        _universe(tmp_path), journal, make_factors(), ts="t1",
        signals=_subset("mom21", "rev5"), window=WINDOW,
        panel_factory=lambda _u: panel,
    )
    again, n_again = build_leaderboard(journal)     # no panels, no factors
    assert n_again == n_trials
    assert [(r.signal, r.alpha_t, r.bh_pass) for r in again] == [
        (r.signal, r.alpha_t, r.bh_pass) for r in rows
    ]


def test_default_universes_point_at_gathered_pools():
    from trading.alphasearch.sweep import default_universes

    got = default_universes(Path("."))
    assert set(got) == {"largecap", "midcap"}
    assert got["largecap"].samples.name == "samples.jsonl"
    assert got["midcap"].samples.name == "samples-midcap.jsonl"
    assert got["midcap"].cache_dir.name == "equities-midcap-tiingo"
    assert got["largecap"].fundamentals_dir is not None


@pytest.mark.parametrize("bad_name,good_name", [("aaa", "zzz"), ("zzz", "aaa")])
def test_incompatible_universe_refuses_the_whole_sweep_either_order(
    tmp_path, bad_name, good_name
):
    # Validation must cover ALL universes BEFORE any trial runs: whether the
    # compatible universe's trials get journaled must not depend on whether
    # the incompatible one sorts first or last.
    journal = trials_journal(tmp_path / "journal")
    panels = {bad_name: make_panel(with_options=False), good_name: make_panel()}
    universes = {
        n: UniverseSpec(n, tmp_path, tmp_path / "s.jsonl", None)
        for n in (bad_name, good_name)
    }
    with pytest.raises(SweepError) as excinfo:
        run_sweep(
            universes, journal, make_factors(), ts="t1",
            signals=_subset("mom21", "hedge"), window=WINDOW,
            panel_factory=lambda u: panels[u.name],
        )
    assert bad_name in str(excinfo.value)           # names the offending pair
    assert list(journal.events()) == []             # zero trials, deterministically


def test_empty_signal_selection_is_refused(tmp_path):
    # An explicitly-empty selection must NOT silently expand to the full
    # registry (`signals or SIGNALS` truthiness gotcha): sweeping nothing is
    # a caller bug and must be loud.
    journal = trials_journal(tmp_path / "journal")
    with pytest.raises(SweepError):
        run_sweep(
            _universe(tmp_path), journal, make_factors(), ts="t1",
            signals={}, window=WINDOW, panel_factory=lambda _u: make_panel(),
        )
    assert list(journal.events()) == []


def test_signals_none_runs_the_full_registry(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    _, n_trials = run_sweep(
        _universe(tmp_path), journal, make_factors(), ts="t1",
        signals=None, window=WINDOW, panel_factory=lambda _u: panel,
    )
    assert n_trials == len(SIGNALS)


def test_min_names_change_is_a_new_trial(tmp_path):
    # tercile_below/min_names must enter the hashed config: re-running with a
    # different sort parameter is a NEW trial, never deduped against the old.
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    kwargs = dict(signals=_subset("mom21"), window=WINDOW,
                  panel_factory=lambda _u: panel)
    _, first = run_sweep(_universe(tmp_path), journal, make_factors(), "t1", **kwargs)
    _, second = run_sweep(_universe(tmp_path), journal, make_factors(), "t2",
                          min_names=10, **kwargs)
    assert (first, second) == (1, 2)


def test_tercile_below_change_is_a_new_trial(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    kwargs = dict(signals=_subset("mom21"), window=WINDOW,
                  panel_factory=lambda _u: panel)
    _, first = run_sweep(_universe(tmp_path), journal, make_factors(), "t1", **kwargs)
    _, second = run_sweep(_universe(tmp_path), journal, make_factors(), "t2",
                          tercile_below=40, **kwargs)
    assert (first, second) == (1, 2)


def test_bh_gate_spans_the_whole_journal_not_one_sweep(tmp_path):
    # Trials from an EARLIER sweep (different window -> different hashes)
    # must raise n for the BH gate of a later sweep.
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    run_sweep(_universe(tmp_path), journal, make_factors(), "t1",
              signals=_subset("mom21"), window=WINDOW,
              panel_factory=lambda _u: panel)
    _, n_trials = run_sweep(_universe(tmp_path), journal, make_factors(), "t2",
                            signals=_subset("rev5"), window=WINDOW,
                            panel_factory=lambda _u: panel)
    assert n_trials == 2  # the gate sees ALL journaled discovery trials


# --------------------------------------------------------------------------- #
# Holdout (Task 9)
# --------------------------------------------------------------------------- #

DISCOVERY = "2020-01-01..2020-03-31"
HOLDOUT_FROM = "2020-04-01"


def _sweep_then_holdout_setup(tmp_path):
    """Discovery on Q1 2020; the fixture's remaining bars are the holdout."""
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=DISCOVERY,
              panel_factory=lambda _u: panel)
    return journal, panel, factors


def test_holdout_pass_rule_is_signed_ratio():
    assert holdout_passes(10.0, 6.0) is True      # kept 60% of the effect
    assert holdout_passes(10.0, 4.9) is False     # faded below half
    assert holdout_passes(10.0, -6.0) is False    # flipped sign
    assert holdout_passes(-10.0, -6.0) is True    # negative alphas: same rule
    assert holdout_passes(-10.0, -4.0) is False
    assert holdout_passes(-10.0, 6.0) is False
    assert holdout_passes(0.0, 1.0) is False      # degenerate discovery
    assert holdout_passes(float("nan"), 1.0) is False


def test_holdout_runs_once_and_journals_with_end_date(tmp_path):
    journal, panel, factors = _sweep_then_holdout_setup(tmp_path)
    outcome = run_holdout(
        _universe(tmp_path)["largecap"], journal, factors, "t2", "mom21",
        holdout_start=HOLDOUT_FROM, discovery_window=DISCOVERY,
        panel_factory=lambda _u: panel,
    )
    latest = max(s.index.max() for s in panel.closes.values())
    assert outcome.window == f"{HOLDOUT_FROM}..{latest.date().isoformat()}"
    prior = prior_holdout_trial(journal, "mom21", "largecap")
    assert prior is not None and prior["kind"] == "holdout"
    assert prior["window"] == outcome.window       # reproducible end date
    assert outcome.passed in (True, False)
    assert outcome.holdout_alpha is not None


def test_holdout_double_touch_refused_without_literal_confirmation(tmp_path):
    journal, panel, factors = _sweep_then_holdout_setup(tmp_path)
    kwargs = dict(holdout_start=HOLDOUT_FROM, discovery_window=DISCOVERY,
                  panel_factory=lambda _u: panel)
    run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t2",
                "mom21", **kwargs)
    events_before = len(list(journal.events()))
    # Default confirm refuses; a wrong phrase refuses; nothing is journaled.
    with pytest.raises(SweepError):
        run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t3",
                    "mom21", **kwargs)
    with pytest.raises(SweepError):
        run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t3",
                    "mom21", confirm=lambda: "yes please", **kwargs)
    assert len(list(journal.events())) == events_before
    # The literal phrase re-runs (and appends a fresh holdout event).
    run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t4",
                "mom21", confirm=lambda: RERUN_CONFIRMATION, **kwargs)
    assert len(list(journal.events())) == events_before + 1


def test_holdout_refused_without_discovery_trial(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    with pytest.raises(SweepError):
        run_holdout(_universe(tmp_path)["largecap"], journal, make_factors(),
                    "t1", "mom21", holdout_start=HOLDOUT_FROM,
                    discovery_window=DISCOVERY, panel_factory=lambda _u: panel)
    assert list(journal.events()) == []


def test_holdout_refused_for_non_bh_survivor(tmp_path):
    # Fabricate a deterministic non-survivor: a clean discovery event whose
    # p-value can never clear BH, alongside mom21's real (surviving) trial.
    from trading.alphasearch.sweep import log_trial, trial_config

    journal, panel, factors = _sweep_then_holdout_setup(tmp_path)
    dull = _result_like(alpha_annual_pct=0.3, alpha_t=0.1, p=0.92)
    log_trial(journal, kind="discovery",
              config=trial_config("rvol21", "largecap", DISCOVERY),
              ts="t1b", result=dull)
    rows, _ = build_leaderboard(journal)
    rvol = next(r for r in rows if r.signal == "rvol21")
    assert not rvol.bh_pass  # precondition for the refusal below
    with pytest.raises(SweepError):
        run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t2",
                    "rvol21", holdout_start=HOLDOUT_FROM,
                    discovery_window=DISCOVERY, panel_factory=lambda _u: panel)


def _result_like(*, alpha_annual_pct: float, alpha_t: float, p: float) -> dict:
    """Minimal spec-section-4 result payload for fabricated journal events."""
    leg = {
        "alpha_annual_pct": alpha_annual_pct, "alpha_t": alpha_t, "p": p,
        "capm_alpha_annual_pct": alpha_annual_pct, "capm_alpha_t": alpha_t,
        "loadings": {}, "loadings_t": {}, "r2": 0.0, "n_obs": 120,
        "sharpe": 0.1, "sharpe_daily": 0.006, "skew": 0.0, "kurt": 3.0,
    }
    return {"n_dates": 3, "n_names_median": 16.0, "ls": leg, "lo": dict(leg),
            "turnover_monthly": 0.3, "skipped_dates": []}


def test_unknown_signal_refused(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    with pytest.raises(SweepError):
        run_holdout(_universe(tmp_path)["largecap"], journal, make_factors(),
                    "t1", "no_such_signal")


def test_holdout_survivor_gate_binds_to_the_exact_discovery_trial(tmp_path):
    # Two discovery trials for the SAME (signal, universe): the canonical
    # window FAILS BH while an alternate exploratory window PASSES. The gate
    # must bind to the specific trial being re-proven -- a surviving row for
    # some OTHER config must not qualify a holdout whose baseline alpha comes
    # from the failed canonical trial.
    from trading.alphasearch.sweep import log_trial, trial_config

    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", DISCOVERY), ts="t1",
              result=_result_like(alpha_annual_pct=0.3, alpha_t=0.1, p=0.92))
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", "2019-01-01..2019-12-31"),
              ts="t1", result=_result_like(alpha_annual_pct=12.0, alpha_t=8.0, p=1e-8))
    rows, _ = build_leaderboard(journal)
    assert any(r.signal == "mom21" and r.bh_pass for r in rows)  # the decoy survives
    with pytest.raises(SweepError):
        run_holdout(_universe(tmp_path)["largecap"], journal, make_factors(),
                    "t2", "mom21", holdout_start=HOLDOUT_FROM,
                    discovery_window=DISCOVERY, panel_factory=lambda _u: panel)
    assert all(e.get("kind") != "holdout" for e in journal.events())


def test_holdout_journals_the_actual_params(tmp_path):
    # A caller-overridden sort parameter must land in the journaled holdout
    # config (and its hash) -- recording what evaluate_trial truly ran -- and
    # the discovery lookup must use those same params.
    from trading.alphasearch.sweep import trial_config, trial_config_hash

    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=DISCOVERY, quantiles=4,
              panel_factory=lambda _u: panel)
    outcome = run_holdout(
        _universe(tmp_path)["largecap"], journal, factors, "t2", "mom21",
        holdout_start=HOLDOUT_FROM, discovery_window=DISCOVERY, quantiles=4,
        panel_factory=lambda _u: panel,
    )
    assert outcome.event["params"]["quantiles"] == 4
    default_hash = trial_config_hash(
        trial_config("mom21", "largecap", outcome.window)
    )
    assert outcome.event["config_hash"] != default_hash


def test_holdout_refused_when_discovery_alpha_missing(tmp_path):
    # A discovery trial whose L/S alpha journaled as null (NaN -> None) has no
    # usable baseline: refuse in the PRE-checks, BEFORE the once-only touch is
    # spent -- crashing after journaling would burn the holdout for nothing.
    from trading.alphasearch.sweep import log_trial, trial_config

    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    broken = _result_like(alpha_annual_pct=float("nan"), alpha_t=8.0, p=1e-8)
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", DISCOVERY),
              ts="t1", result=broken)
    with pytest.raises(SweepError):
        run_holdout(_universe(tmp_path)["largecap"], journal, make_factors(),
                    "t2", "mom21", holdout_start=HOLDOUT_FROM,
                    discovery_window=DISCOVERY, panel_factory=lambda _u: panel)
    assert all(e.get("kind") != "holdout" for e in journal.events())
