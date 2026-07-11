"""Sweep runner + leaderboard: journaling order, honesty, refusal, recompute."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphasearch_helpers import make_factors, make_panel, make_spy_closes
from trading.alphasearch.panel import PanelError
from trading.alphasearch.spec import SIGNALS, SignalSpec
from trading.alphasearch.sweep import (
    DEFAULT_PARAMS,
    RERUN_CONFIRMATION,
    MarketNeutralRow,
    SweepError,
    UniverseSpec,
    _hashed_params,
    build_leaderboard,
    build_long_only_leaderboard,
    build_market_neutral_leaderboard,
    default_universes,
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
        panel_factory=lambda _u, _f: panel,
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


def test_evaluate_trial_threads_subset_and_offset():
    from trading.alphasearch.sweep import evaluate_trial

    panel = make_panel()
    factors = make_factors()
    subset = tuple(panel.symbols[:8])
    got = evaluate_trial(panel, SIGNALS["mom21"], WINDOW, factors,
                         min_names=5, symbol_subset=subset)
    assert got["n_names_median"] == 8.0        # subset reached the sort
    offset = evaluate_trial(panel, SIGNALS["mom21"], WINDOW, factors,
                            calendar_offset=1)
    base = evaluate_trial(panel, SIGNALS["mom21"], WINDOW, factors)
    # Same months, different sessions: the offset run rebalances on the 2nd
    # session, so its decision-date count matches and its series differs.
    assert offset["n_dates"] == base["n_dates"]
    assert offset["ls"]["alpha_annual_pct"] != base["ls"]["alpha_annual_pct"]


def test_sweep_rerun_is_idempotent_for_trial_count(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    kwargs = dict(signals=_subset("mom21"), window=WINDOW,
                  panel_factory=lambda _u, _f: panel)
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
        panel_factory=lambda _u, _f: panel,
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
    with pytest.raises(SweepError) as excinfo:
        run_sweep(
            _universe(tmp_path), journal, make_factors(), ts="t1",
            signals=_subset("mom21", "hedge"), window=WINDOW,
            panel_factory=lambda _u, _f: panel,
        )
    assert list(journal.events()) == []             # refused at assembly: no trials
    # Actionable: how to gather it, and the zero-setup workaround.
    message = str(excinfo.value)
    assert "scripts/gather_options_iv.py" in message
    assert "--signals" in message


def test_fundamentals_signal_without_store_refused(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(with_fundamentals=False)
    with pytest.raises(SweepError) as excinfo:
        run_sweep(
            _universe(tmp_path), journal, make_factors(), ts="t1",
            signals=_subset("earnings_yield"), window=WINDOW,
            panel_factory=lambda _u, _f: panel,
        )
    # Actionable: the expected store path, how to populate it, and the
    # zero-setup workaround -- not just "has none".
    message = str(excinfo.value)
    assert "data/fundamentals/equities" in message
    assert "scripts/backfill_fundamentals.py" in message
    assert "--signals" in message


def test_insider_signal_without_store_refused(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(with_insider=False)
    with pytest.raises(SweepError) as excinfo:
        run_sweep(
            _universe(tmp_path), journal, make_factors(), ts="t1",
            signals=_subset("npr_90"), window=WINDOW,
            panel_factory=lambda _u, _f: panel,
        )
    assert list(journal.events()) == []             # refused at assembly: no trials
    message = str(excinfo.value)
    assert "data/insider/equities" in message
    assert "scripts/build_insider_store.py" in message
    assert "--signals" in message


def test_discovery_trial_with_stale_factors_is_journaled_as_error(tmp_path):
    # Factors ending well before the window end must refuse loudly (naming the
    # factor end date, the window end, and the fix) rather than let
    # run_regression's inner join silently truncate to whatever overlaps.
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    stale_factors = make_factors(periods=60)  # ends 2020-02-21, WINDOW ends 06-30
    rows, n_trials = run_sweep(
        _universe(tmp_path), journal, stale_factors, ts="t1",
        signals=_subset("mom21"), window=WINDOW,
        panel_factory=lambda _u, _f: panel,
    )
    assert n_trials == 1                             # the failure still counts
    row = rows[0]
    assert row.error is not None and "factor cache" in row.error
    assert "--refresh-factors" in row.error
    assert not row.bh_pass
    assert row.alpha_t is None


def test_leaderboard_recomputes_from_journal_alone(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    rows, n_trials = run_sweep(
        _universe(tmp_path), journal, make_factors(), ts="t1",
        signals=_subset("mom21", "rev5"), window=WINDOW,
        panel_factory=lambda _u, _f: panel,
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
            panel_factory=lambda u, _f: panels[u.name],
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
            signals={}, window=WINDOW, panel_factory=lambda _u, _f: make_panel(),
        )
    assert list(journal.events()) == []


def test_signals_none_runs_the_full_registry(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    _, n_trials = run_sweep(
        _universe(tmp_path), journal, make_factors(), ts="t1",
        signals=None, window=WINDOW, panel_factory=lambda _u, _f: panel,
    )
    assert n_trials == len(SIGNALS)


def test_min_names_change_is_a_new_trial(tmp_path):
    # tercile_below/min_names must enter the hashed config: re-running with a
    # different sort parameter is a NEW trial, never deduped against the old.
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    kwargs = dict(signals=_subset("mom21"), window=WINDOW,
                  panel_factory=lambda _u, _f: panel)
    _, first = run_sweep(_universe(tmp_path), journal, make_factors(), "t1", **kwargs)
    _, second = run_sweep(_universe(tmp_path), journal, make_factors(), "t2",
                          min_names=10, **kwargs)
    assert (first, second) == (1, 2)


def test_tercile_below_change_is_a_new_trial(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    kwargs = dict(signals=_subset("mom21"), window=WINDOW,
                  panel_factory=lambda _u, _f: panel)
    _, first = run_sweep(_universe(tmp_path), journal, make_factors(), "t1", **kwargs)
    _, second = run_sweep(_universe(tmp_path), journal, make_factors(), "t2",
                          tercile_below=40, **kwargs)
    assert (first, second) == (1, 2)


# --------------------------------------------------------------------------- #
# Concentration axis (2026-07-11 amendment): top_n threaded through identity,
# the sweep evaluation, and --long-only re-derivation.
# --------------------------------------------------------------------------- #
def test_hashed_params_top_n_none_is_bit_identical_to_default_params():
    from trading.alphasearch.sort import MIN_NAMES, QUANTILES, TERCILE_BELOW
    from trading.alphasearch.sweep import DEFAULT_PARAMS, _hashed_params

    got = _hashed_params(QUANTILES, TERCILE_BELOW, MIN_NAMES, top_n=None)
    assert got == DEFAULT_PARAMS
    assert "top_n" not in got


def test_hashed_params_top_n_set_adds_the_key():
    from trading.alphasearch.sort import MIN_NAMES, QUANTILES, TERCILE_BELOW
    from trading.alphasearch.sweep import _hashed_params

    got = _hashed_params(QUANTILES, TERCILE_BELOW, MIN_NAMES, top_n=10)
    assert got["top_n"] == 10


def test_top_n_hash_distinctness():
    from trading.alphasearch.sweep import trial_config, trial_config_hash

    quintile = trial_config("mom21", "largecap", WINDOW)          # top_n=None
    ten = trial_config("mom21", "largecap", WINDOW,
                       params={**trial_config("mom21", "largecap", WINDOW)["params"],
                               "top_n": 10})
    twenty = trial_config("mom21", "largecap", WINDOW,
                          params={**trial_config("mom21", "largecap", WINDOW)["params"],
                                  "top_n": 20})
    hashes = {trial_config_hash(quintile), trial_config_hash(ten),
              trial_config_hash(twenty)}
    assert len(hashes) == 3


def test_run_sweep_top_n_journals_a_distinct_trial_with_top_n_in_params(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    kwargs = dict(signals=_subset("mom21"), window=WINDOW,
                  panel_factory=lambda _u, _f: panel)
    _, quintile_n = run_sweep(_universe(tmp_path), journal, make_factors(), "t1",
                              **kwargs)
    _, top_n_n = run_sweep(_universe(tmp_path), journal, make_factors(), "t2",
                           top_n=5, **kwargs)
    assert (quintile_n, top_n_n) == (1, 2)     # distinct trials, not deduped
    trials = discovery_trials(journal)
    top_trial = next(t for t in trials if t["params"].get("top_n") == 5)
    assert top_trial["params"]["top_n"] == 5
    quintile_trial = next(t for t in trials if "top_n" not in t["params"])
    assert quintile_trial["config_hash"] != top_trial["config_hash"]
    # The top-N trial actually ran the fixed-count construction: 5 names,
    # not a quintile of the 16-symbol fixture (which would be 3-4 names).
    assert top_trial["n_names_median"] == 16.0   # full cross-section reached


def test_long_only_leaderboard_rederives_top_n_trial_as_top_n_not_quintile(
    tmp_path,
):
    # The --long-only re-derivation must read a top-N trial's journaled
    # params and rebuild it via the fixed-count construction; if it silently
    # hardcoded quintile, its lo series would differ from a direct top_n
    # portfolio_sort call on the same panel/window.
    from trading.alphasearch.sort import portfolio_sort
    from trading.alphasearch.spec import SIGNALS

    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=WINDOW, top_n=5,
              panel_factory=lambda _u, _f: panel)
    rows = build_long_only_leaderboard(
        journal, _universe(tmp_path), factors, make_spy_closes(),
        panel_factory=lambda _u, _f: panel,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.error is None
    start, end = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-06-30", tz="UTC")
    dates = panel.decision_dates(start, end)
    direct = portfolio_sort(panel, SIGNALS["mom21"], dates, end, top_n=5)
    from trading.alphasearch.costs import cost_charged_lo
    from trading.alphasearch.evaluate import annualized_sharpe

    charged, _skipped = cost_charged_lo(panel, direct.lo, direct.rebalances)
    assert row.lo_sharpe == pytest.approx(annualized_sharpe(charged))


def test_bh_gate_spans_the_whole_journal_not_one_sweep(tmp_path):
    # Trials from an EARLIER sweep (different window -> different hashes)
    # must raise n for the BH gate of a later sweep.
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    run_sweep(_universe(tmp_path), journal, make_factors(), "t1",
              signals=_subset("mom21"), window=WINDOW,
              panel_factory=lambda _u, _f: panel)
    _, n_trials = run_sweep(_universe(tmp_path), journal, make_factors(), "t2",
                            signals=_subset("rev5"), window=WINDOW,
                            panel_factory=lambda _u, _f: panel)
    assert n_trials == 2  # the gate sees ALL journaled discovery trials


# --------------------------------------------------------------------------- #
# Holdout (Task 9)
# --------------------------------------------------------------------------- #

DISCOVERY = "2020-01-01..2020-03-31"
HOLDOUT_FROM = "2020-04-01"
# The 130-bar fixture leaves only ~3 months after HOLDOUT_FROM -- under the
# production HOLDOUT_MIN_FACTOR_SPAN_DAYS floor -- so tests that actually RUN
# the holdout shrink the floor to fit the fixture.
MIN_SPAN = 30


def _log_battery_verdict(journal, signal, universe, window, *, eligible,
                         params=None):
    """Fabricate a Piece 3 battery verdict for the (default-params) config."""
    from trading.alphasearch.sweep import log_trial, trial_config

    log_trial(journal, kind="battery",
              config=trial_config(signal, universe, window, params=params),
              ts="tb", result={"eligible": eligible})


def _sweep_then_holdout_setup(tmp_path, with_battery: bool = True):
    """Discovery on Q1 2020; the fixture's remaining bars are the holdout.
    with_battery fabricates the Piece 3 battery-passed verdict the holdout
    now requires (its own tests set False to exercise the refusal)."""
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=DISCOVERY,
              panel_factory=lambda _u, _f: panel)
    if with_battery:
        _log_battery_verdict(journal, "mom21", "largecap", DISCOVERY,
                             eligible=True)
    return journal, panel, factors


def test_holdout_refused_for_battery_less_survivor(tmp_path):
    # Prospective amendment to Piece 1 spec 3.6 (Piece 3 design spec): no
    # holdout may be spent on a survivor that has not passed its battery.
    journal, panel, factors = _sweep_then_holdout_setup(tmp_path,
                                                        with_battery=False)
    with pytest.raises(SweepError) as excinfo:
        run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t2",
                    "mom21", holdout_start=HOLDOUT_FROM,
                    discovery_window=DISCOVERY, min_factor_span_days=MIN_SPAN,
                    panel_factory=lambda _u, _f: panel)
    # The refusal names the command that fixes it, and journals no touch.
    assert "trading alphasearch robustness mom21:largecap" in str(excinfo.value)
    assert all(e.get("kind") != "holdout" for e in journal.events())


def test_holdout_refused_for_battery_failed_survivor(tmp_path):
    journal, panel, factors = _sweep_then_holdout_setup(tmp_path,
                                                        with_battery=False)
    _log_battery_verdict(journal, "mom21", "largecap", DISCOVERY,
                         eligible=False)
    with pytest.raises(SweepError, match="did not pass"):
        run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t2",
                    "mom21", holdout_start=HOLDOUT_FROM,
                    discovery_window=DISCOVERY, min_factor_span_days=MIN_SPAN,
                    panel_factory=lambda _u, _f: panel)
    assert all(e.get("kind") != "holdout" for e in journal.events())


def test_holdout_battery_gate_binds_to_the_exact_config(tmp_path):
    # A battery verdict for DIFFERENT params must not qualify the default-
    # params holdout (hash-keyed, like the BH survivor gate).
    journal, panel, factors = _sweep_then_holdout_setup(tmp_path,
                                                        with_battery=False)
    other_params = {"quantiles": 4, "weighting": "equal", "cadence": "monthly",
                    "tercile_below": 50, "min_names": 15}
    _log_battery_verdict(journal, "mom21", "largecap", DISCOVERY,
                         eligible=True, params=other_params)
    with pytest.raises(SweepError, match="robustness"):
        run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t2",
                    "mom21", holdout_start=HOLDOUT_FROM,
                    discovery_window=DISCOVERY, min_factor_span_days=MIN_SPAN,
                    panel_factory=lambda _u, _f: panel)


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
        min_factor_span_days=MIN_SPAN, panel_factory=lambda _u, _f: panel,
    )
    # Fixture factors outlast the bars, so the clamped end IS the latest bar.
    latest = max(s.index.max() for s in panel.closes.values())
    assert factors.index.max() > latest            # precondition for the line above
    assert outcome.window == f"{HOLDOUT_FROM}..{latest.date().isoformat()}"
    prior = prior_holdout_trial(journal, "mom21", "largecap")
    assert prior is not None and prior["kind"] == "holdout"
    assert prior["window"] == outcome.window       # reproducible end date
    assert outcome.passed in (True, False)
    assert outcome.holdout_alpha is not None


def test_holdout_clamps_window_end_to_factor_coverage(tmp_path):
    # The FF publication lag means the factor cache routinely ends before the
    # latest bar even freshly refreshed. The holdout must RUN (not refuse) and
    # journal a window ending at the FACTOR end -- the window evaluate_trial
    # actually saw -- never at the latest bar the regression couldn't reach.
    journal, panel, _ = _sweep_then_holdout_setup(tmp_path)
    lagged = make_factors(periods=145)             # ends 2020-06-19
    latest = max(s.index.max() for s in panel.closes.values())
    assert lagged.index.max() < latest             # the clamp genuinely binds
    outcome = run_holdout(
        _universe(tmp_path)["largecap"], journal, lagged, "t2", "mom21",
        holdout_start=HOLDOUT_FROM, discovery_window=DISCOVERY,
        min_factor_span_days=MIN_SPAN, panel_factory=lambda _u, _f: panel,
    )
    factors_end = lagged.index.max().date().isoformat()
    assert outcome.window == f"{HOLDOUT_FROM}..{factors_end}"
    prior = prior_holdout_trial(journal, "mom21", "largecap")
    assert prior["window"] == outcome.window       # journaled == evaluated
    assert prior["error"] is None                  # a clean run, not an error trial
    assert outcome.holdout_alpha is not None


def test_holdout_double_touch_refused_without_literal_confirmation(tmp_path):
    journal, panel, factors = _sweep_then_holdout_setup(tmp_path)
    kwargs = dict(holdout_start=HOLDOUT_FROM, discovery_window=DISCOVERY,
                  min_factor_span_days=MIN_SPAN, panel_factory=lambda _u, _f: panel)
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
                    discovery_window=DISCOVERY, panel_factory=lambda _u, _f: panel)
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
                    discovery_window=DISCOVERY, panel_factory=lambda _u, _f: panel)


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
                    discovery_window=DISCOVERY, panel_factory=lambda _u, _f: panel)
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
              panel_factory=lambda _u, _f: panel)
    _log_battery_verdict(
        journal, "mom21", "largecap", DISCOVERY, eligible=True,
        params={"quantiles": 4, "weighting": "equal", "cadence": "monthly",
                "tercile_below": 50, "min_names": 15},
    )
    outcome = run_holdout(
        _universe(tmp_path)["largecap"], journal, factors, "t2", "mom21",
        holdout_start=HOLDOUT_FROM, discovery_window=DISCOVERY, quantiles=4,
        min_factor_span_days=MIN_SPAN, panel_factory=lambda _u, _f: panel,
    )
    assert outcome.event["params"]["quantiles"] == 4
    default_hash = trial_config_hash(
        trial_config("mom21", "largecap", outcome.window)
    )
    assert outcome.event["config_hash"] != default_hash


def test_holdout_refused_before_touch_when_factors_stale(tmp_path):
    # TRULY stale factors -- ending before the holdout even accumulates the
    # minimum span -- must refuse in the PRE-checks, BEFORE the once-only
    # touch is journaled: journaling it as a spent-but-errored holdout would
    # burn the reserved touch on a fixable data problem (a --refresh-factors
    # away). This fixture ends BEFORE holdout_start (2020-02-21 < 04-01), so
    # it is outside the clamp zone (which the clamp test above covers) under
    # ANY minimum span; the production default floor applies here.
    journal, panel, factors = _sweep_then_holdout_setup(tmp_path)
    stale_factors = make_factors(periods=60)  # ends 2020-02-21, pre-holdout
    with pytest.raises(SweepError, match="factor cache") as excinfo:
        run_holdout(
            _universe(tmp_path)["largecap"], journal, stale_factors, "t2", "mom21",
            holdout_start=HOLDOUT_FROM, discovery_window=DISCOVERY,
            panel_factory=lambda _u, _f: panel,
        )
    assert "--refresh-factors" in str(excinfo.value)  # names the fix
    assert all(e.get("kind") != "holdout" for e in journal.events())  # untouched


def test_holdout_journals_error_event_before_reraising_unexpected_exception(
    tmp_path, monkeypatch
):
    # Only (SortError, ValueError, LinAlgError) were ever journaled on
    # failure; any OTHER exception (e.g. ArithmeticError from stats._betacf)
    # used to escape AFTER holdout data was read but BEFORE log_trial,
    # spending the once-only touch with no journal record. Any exception must
    # now journal an error-kind holdout event before propagating.
    import trading.alphasearch.sweep as sweep_mod

    journal, panel, factors = _sweep_then_holdout_setup(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(sweep_mod, "evaluate_trial", _boom)
    with pytest.raises(RuntimeError, match="kaboom"):
        run_holdout(
            _universe(tmp_path)["largecap"], journal, factors, "t2", "mom21",
            holdout_start=HOLDOUT_FROM, discovery_window=DISCOVERY,
            min_factor_span_days=MIN_SPAN, panel_factory=lambda _u, _f: panel,
        )
    holdout_events = [e for e in journal.events() if e.get("kind") == "holdout"]
    assert len(holdout_events) == 1                  # journaled despite the raise
    assert holdout_events[0]["error"] is not None
    assert "RuntimeError" in holdout_events[0]["error"]
    assert "kaboom" in holdout_events[0]["error"]


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
                    discovery_window=DISCOVERY, panel_factory=lambda _u, _f: panel)
    assert all(e.get("kind") != "holdout" for e in journal.events())


# --------------------------------------------------------------------------- #
# Explicit-symbols universes (Piece 2): the real files path, no factory.
# --------------------------------------------------------------------------- #


def _write_deep_universe(tmp_path) -> UniverseSpec:
    """make_panel()'s closes as real parquets + an explicit symbols tuple:
    exactly the shape segment_universes emits for a deep pool."""
    panel = make_panel()
    cache = tmp_path / "cache"
    cache.mkdir()
    for sym in panel.symbols:
        closes = panel.closes[sym]
        pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes,
             "volume": 1000.0},
            index=closes.index,
        ).to_parquet(cache / f"{sym}.parquet")
    return UniverseSpec("largecap:test", cache, None, None, symbols=panel.symbols)


def test_deep_universe_runs_price_signals_end_to_end(tmp_path):
    # No panel_factory injection: build_universe_panel/build_panel must
    # assemble a closes-only panel from the explicit symbols tuple.
    journal = trials_journal(tmp_path / "journal")
    uspec = _write_deep_universe(tmp_path)
    rows, n_trials = run_sweep({uspec.name: uspec}, journal, make_factors(),
                               ts="t1", signals=_subset("mom21"), window=WINDOW)
    assert n_trials == 1
    assert rows[0].universe == "largecap:test"
    assert rows[0].error is None
    assert abs(rows[0].alpha_t) > 5           # the engineered momentum spread


def test_deep_universe_refuses_options_signals_end_to_end(tmp_path):
    # samples=None -> panel.options == {} -> the EXISTING requires_options
    # assembly-time refusal fires; zero trials journaled (all-or-nothing).
    journal = trials_journal(tmp_path / "journal")
    uspec = _write_deep_universe(tmp_path)
    with pytest.raises(SweepError, match="requires options"):
        run_sweep({uspec.name: uspec}, journal, make_factors(), ts="t1",
                  signals=_subset("mom21", "hedge"), window=WINDOW)
    assert list(journal.events()) == []


def test_deep_universe_refuses_fundamentals_signals_end_to_end(tmp_path):
    # fundamentals_dir=None -> panel.fundamentals == {} -> existing refusal.
    journal = trials_journal(tmp_path / "journal")
    uspec = _write_deep_universe(tmp_path)
    with pytest.raises(SweepError, match="requires fundamentals"):
        run_sweep({uspec.name: uspec}, journal, make_factors(), ts="t1",
                  signals=_subset("earnings_yield"), window=WINDOW)
    assert list(journal.events()) == []


def test_deep_universe_refuses_insider_signals_end_to_end(tmp_path):
    # insider_dir=None -> panel.insider == {} -> the requires_insider refusal.
    journal = trials_journal(tmp_path / "journal")
    uspec = _write_deep_universe(tmp_path)
    with pytest.raises(SweepError, match="requires insider"):
        run_sweep({uspec.name: uspec}, journal, make_factors(), ts="t1",
                  signals=_subset("cluster_buys_90"), window=WINDOW)
    assert list(journal.events()) == []


def test_empty_symbols_tuple_refused_at_assembly_no_trials(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    cache = tmp_path / "cache"
    cache.mkdir()
    uspec = UniverseSpec("largecap:empty", cache, None, None, symbols=())
    with pytest.raises(PanelError, match="empty"):
        run_sweep({uspec.name: uspec}, journal, make_factors(), ts="t1",
                  signals=_subset("mom21"), window=WINDOW)
    assert list(journal.events()) == []


def test_universespec_symbols_defaults_none_for_piece1_call_sites():
    got = default_universes(Path("."))
    assert got["largecap"].symbols is None
    assert got["midcap"].symbols is None


def test_run_sweep_passes_factors_to_the_panel_factory(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    seen: list = []

    def factory(_u, f):
        seen.append(f)
        return panel

    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=WINDOW, panel_factory=factory)
    assert len(seen) == 1 and seen[0] is factors


def test_option_volume_signal_refused_on_a_volume_less_universe(tmp_path):
    # requires_option_volume mirrors requires_options: an assembly-time
    # refusal, never silent fake-log(1/1)=0 trials (spec section 2, options
    # family constraint).
    journal = trials_journal(tmp_path / "journal")
    fake = SignalSpec("fake_vol", SIGNALS["hedge"].fn,
                      requires_options=True, requires_option_volume=True)
    largecap_like = make_panel(with_option_volume=False)
    with pytest.raises(SweepError) as excinfo:
        run_sweep(_universe(tmp_path), journal, make_factors(), ts="t1",
                  signals={"fake_vol": fake, "mom21": SIGNALS["mom21"]},
                  window=WINDOW, panel_factory=lambda _u, _f: largecap_like)
    message = str(excinfo.value)
    assert "option volume" in message
    assert "--signals" in message                    # actionable workaround
    assert list(journal.events()) == []              # all-or-nothing: no trials
    # And the same signal RUNS where cells carry leg volume.
    midcap_like = make_panel()
    rows, n = run_sweep(_universe(tmp_path), journal, make_factors(), ts="t2",
                        signals={"fake_vol": fake}, window=WINDOW,
                        panel_factory=lambda _u, _f: midcap_like)
    assert n == 1 and rows[0].error is None


def test_build_universe_panel_derives_sectors_from_the_sic_map(tmp_path):
    from trading.alphasearch.sweep import build_universe_panel

    idx = pd.date_range("2020-01-02", periods=5, freq="B", tz="UTC")
    cache = tmp_path / "cache"
    cache.mkdir()
    for sym in ("AAA", "BBB", "CCC"):
        pd.DataFrame(
            {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
            index=idx,
        ).to_parquet(cache / f"{sym}.parquet")
    sic = tmp_path / "sic.csv"
    sic.write_text(
        "symbol,cik,sic,sic_description,fetched_at\n"
        "AAA,1,2836,biotech,2026-07-09\n"   # pharma-chemicals sector (+biotech industry)
        "BBB,2,6022,bank,2026-07-09\n"      # finance sector (+banks industry)
        "CCC,3,700,farm,2026-07-09\n"       # covered by no segment
    )
    uspec = UniverseSpec("u", cache, None, None, symbols=("AAA", "BBB", "CCC"),
                         sic_map_path=sic)
    panel = build_universe_panel(uspec, make_factors())
    # Sectors only (the 10-way partition); industries never masquerade as one.
    assert panel.sectors == {"AAA": "pharma-chemicals", "BBB": "finance"}


# --------------------------------------------------------------------------- #
# --long-only leaderboard (R1 gate amendment spec section 4 deliverable 2)
# --------------------------------------------------------------------------- #
def test_long_only_leaderboard_rederives_a_real_trial(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=WINDOW,
              panel_factory=lambda _u, _f: panel)
    rows = build_long_only_leaderboard(
        journal, _universe(tmp_path), factors, make_spy_closes(),
        panel_factory=lambda _u, _f: panel,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.signal == "mom21" and row.universe == "largecap"
    assert row.window == WINDOW
    assert row.error is None
    assert row.lo_sharpe is not None
    assert row.spy_sharpe is not None
    assert isinstance(row.beats_spy, bool)
    assert row.skipped_no_spread == 0   # make_panel's bars always carry high/low


def test_long_only_leaderboard_beats_spy_direction_is_pinned(tmp_path):
    # R1 spec section 2 comparator direction: the SAME re-derived trial must
    # flag beats_spy True against a weak benchmark and False against one no
    # fixture leg outruns -- pinning that the comparison runs the right way.
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=WINDOW,
              panel_factory=lambda _u, _f: panel)
    kwargs = dict(panel_factory=lambda _u, _f: panel)
    # Weak = declining drift at normal vol (zero-vol would mean a
    # near-infinite benchmark Sharpe: constant returns, sd -> 0).
    weak = build_long_only_leaderboard(
        journal, _universe(tmp_path), factors,
        make_spy_closes(drift=-0.001), **kwargs)
    strong = build_long_only_leaderboard(
        journal, _universe(tmp_path), factors,
        make_spy_closes(drift=0.01, vol=0.0), **kwargs)
    assert weak[0].beats_spy is True
    assert weak[0].lo_total_return > weak[0].spy_total_return
    assert strong[0].beats_spy is False
    assert strong[0].lo_total_return < strong[0].spy_total_return
    # NaN side (benchmark never overlaps the window): honest False, not a
    # free pass -- and the row still ranks (error is None; data DID re-derive).
    disjoint = build_long_only_leaderboard(
        journal, _universe(tmp_path), factors,
        make_spy_closes(start="2015-01-02", periods=100, vol=0.0), **kwargs)
    assert disjoint[0].beats_spy is False
    assert disjoint[0].spy_sharpe is None and disjoint[0].error is None


def test_long_only_leaderboard_reports_na_for_unresolvable_universe(tmp_path):
    # A trial journaled under a universe name no longer in the resolved set
    # (e.g. a segment whose SIC map went missing) shows honestly as n/a,
    # never silently dropped.
    from trading.alphasearch.sweep import log_trial, trial_config

    journal = trials_journal(tmp_path / "journal")
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap:gone", WINDOW),
              ts="t1", result=_result_like(alpha_annual_pct=8.0, alpha_t=3.0, p=0.01))
    rows = build_long_only_leaderboard(
        journal, _universe(tmp_path), make_factors(), make_spy_closes(),
    )
    assert len(rows) == 1
    assert rows[0].error is not None and "unknown universe" in rows[0].error
    assert rows[0].lo_sharpe is None and rows[0].beats_spy is None


def test_long_only_leaderboard_reports_na_for_unknown_signal(tmp_path):
    # A signal since removed from the registry: honest n/a, not a crash.
    from trading.alphasearch.sweep import log_trial, trial_config

    journal = trials_journal(tmp_path / "journal")
    log_trial(journal, kind="discovery",
              config=trial_config("no_longer_exists", "largecap", WINDOW),
              ts="t1", result=_result_like(alpha_annual_pct=8.0, alpha_t=3.0, p=0.01))
    rows = build_long_only_leaderboard(
        journal, _universe(tmp_path), make_factors(), make_spy_closes(),
        panel_factory=lambda _u, _f: make_panel(),
    )
    assert len(rows) == 1
    assert rows[0].error is not None and "unknown signal" in rows[0].error


def test_long_only_leaderboard_sorts_by_sharpe_with_na_last(tmp_path):
    from trading.alphasearch.sweep import log_trial, trial_config

    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21", "rev5"), window=WINDOW,
              panel_factory=lambda _u, _f: panel)
    log_trial(journal, kind="discovery",
              config=trial_config("gone", "largecap", WINDOW),
              ts="t1", result=_result_like(alpha_annual_pct=1.0, alpha_t=1.0, p=0.5))
    rows = build_long_only_leaderboard(
        journal, _universe(tmp_path), factors, make_spy_closes(),
        panel_factory=lambda _u, _f: panel,
    )
    assert len(rows) == 3
    ranked = [r for r in rows if r.error is None]
    assert len(ranked) == 2
    assert ranked[0].lo_sharpe >= ranked[1].lo_sharpe
    assert rows[-1].error is not None   # the n/a row sorts last


def test_long_only_leaderboard_aligns_spy_window_to_the_actual_lo_start(
    tmp_path, monkeypatch
):
    """Fix (final review): same alignment fix as run_battery's long_only_gate
    (test_run_battery_gate_aligns_spy_window_to_the_actual_lo_start in
    test_alphasearch_robustness.py), applied to _rederive_long_only_row so
    the --long-only leaderboard stays consistent with the gate. A leading
    skipped decision date (below MIN_NAMES) must not leave the SPY
    comparator spanning the full nominal window while the re-derived
    charged_lo starts later -- that would compound different horizons."""
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(n_symbols=40)
    factors = make_factors()
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2020-06-30", tz="UTC")
    dates = panel.decision_dates(start, end)
    cutoff = dates[1]                              # skip ONLY the first date

    def fn(view, as_of):
        if as_of < cutoff:
            return pd.Series({"S00": 1.0, "S01": 2.0})   # 2 < MIN_NAMES: skip
        return pd.Series(
            {s: float(i) for i, s in enumerate(sorted(view.symbols))},
            dtype="float64",
        )

    spec = SignalSpec("leadskip_rank", fn)
    monkeypatch.setitem(SIGNALS, "leadskip_rank", spec)

    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals={"leadskip_rank": spec}, window=WINDOW,
              panel_factory=lambda _u, _f: panel)

    # SPY falls sharply during the skipped lead-in, then a quiet random walk:
    # the full-nominal-window total return and the lo-aligned total return
    # are then provably different numbers.
    idx = panel.closes[panel.symbols[0]].index
    n_lead = int((idx < cutoff).sum())
    rng = np.random.default_rng(0)
    tail_rets = rng.normal(0.0001, 0.001, size=len(idx) - n_lead)
    spy_values = np.concatenate([
        np.linspace(100.0, 70.0, n_lead), 70.0 * np.cumprod(1 + tail_rets),
    ])
    spy = pd.Series(spy_values, index=idx)

    rows = build_long_only_leaderboard(
        journal, _universe(tmp_path), factors, spy,
        panel_factory=lambda _u, _f: panel,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.error is None

    from trading.alphasearch.costs import cost_charged_lo, spy_benchmark
    from trading.alphasearch.sort import portfolio_sort

    sort = portfolio_sort(panel, spec, dates, end)
    assert sort.rebalances[0][0] == cutoff          # confirms the lead skip
    charged, _skipped = cost_charged_lo(panel, sort.lo, sort.rebalances)
    # The correct anchor is the first ACTUAL DECISION date, not charged's
    # first REALIZED RETURN day: portfolio_sort builds hold segments strictly
    # after the decision date (sort.py), so charged.index[0] is already one
    # trading day later and compounds a shorter horizon than SPY needs to
    # match.
    aligned = spy_benchmark(spy, sort.rebalances[0][0], end)
    stale = spy_benchmark(spy, start, end)          # the pre-FIX2 (buggy) window
    off_by_one = spy_benchmark(spy, charged.index[0], end)  # the FIX2 (still off-by-one) window
    assert aligned.total_return != pytest.approx(stale.total_return)
    assert aligned.total_return != pytest.approx(off_by_one.total_return)
    assert row.spy_total_return == pytest.approx(aligned.total_return)
    assert row.spy_sharpe == pytest.approx(aligned.sharpe_annual)
    # Independent invariant catching either a one-day-early OR one-day-late
    # anchor: the SPY window must start exactly at the first decision date,
    # and its daily-return observation count must equal the number of
    # trading sessions from that decision date through `end` -- exactly one
    # MORE obs than the charged.index[0] (FIX2) anchor yields, since that
    # anchor starts one session later. Both the pre-FIX2 `start` anchor and
    # the FIX2 `charged.index[0]` anchor fail this.
    spy_window = spy.loc[(spy.index >= sort.rebalances[0][0]) & (spy.index <= end)]
    assert spy_window.index[0] == sort.rebalances[0][0]
    assert aligned.n_obs == len(spy_window) - 1
    assert aligned.n_obs == off_by_one.n_obs + 1


# --------------------------------------------------------------------------- #
# Default-off bit-identity (R6 Stage 1 market-neutral gate amendment, spec
# section 6 -- PARAMOUNT): the market-neutral leaderboard is purely additive
# read-side eval. No sweep param changed, so _hashed_params/DEFAULT_PARAMS
# and the --long-only leaderboard must be untouched.
# --------------------------------------------------------------------------- #
def test_hashed_params_unchanged_by_market_neutral_addition():
    # The exact frozen dict shape from before this amendment: no new key was
    # added for the market-neutral read path (it hashes NO new sweep params
    # -- it re-derives from the EXISTING journaled top_n/quantiles/etc).
    assert DEFAULT_PARAMS == {
        "quantiles": 5, "weighting": "equal", "cadence": "monthly",
        "tercile_below": 50, "min_names": 15,
    }
    assert _hashed_params(5, 50, 15) == DEFAULT_PARAMS


def test_long_only_leaderboard_output_pinned_after_market_neutral_addition(
    tmp_path,
):
    # A byte-identical pin (spec section 6 PARAMOUNT): the exact same fixture
    # test_long_only_leaderboard_rederives_a_real_trial exercises, pinned to
    # the precise float values produced BEFORE any market-neutral code was
    # added -- proving the long-only path is untouched by this amendment.
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=WINDOW,
              panel_factory=lambda _u, _f: panel)
    rows = build_long_only_leaderboard(
        journal, _universe(tmp_path), factors, make_spy_closes(),
        panel_factory=lambda _u, _f: panel,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.config_hash == "90b16087cebf"
    assert row.lo_sharpe == pytest.approx(8.354236629560836)
    assert row.lo_total_return == pytest.approx(0.07561957204016201)
    assert row.spy_sharpe == pytest.approx(0.04298099280636386)
    assert row.spy_total_return == pytest.approx(-0.0022820952608864076)
    assert row.beats_spy is True
    assert row.skipped_no_spread == 0


# --------------------------------------------------------------------------- #
# --market-neutral leaderboard (R6 Stage 1 market-neutral gate amendment,
# docs/superpowers/specs/2026-07-11-market-neutral-gate-amendment.md)
# --------------------------------------------------------------------------- #
def test_market_neutral_leaderboard_rederives_a_real_trial(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=WINDOW,
              panel_factory=lambda _u, _f: panel)
    rows = build_market_neutral_leaderboard(
        journal, _universe(tmp_path), factors,
        panel_factory=lambda _u, _f: panel,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.signal == "mom21" and row.universe == "largecap"
    assert row.window == WINDOW
    assert row.error is None
    assert row.top_n is None                # quintile trial, not top-N
    assert row.mn_sharpe is not None
    assert row.mn_sharpe_ci_lo is not None and row.mn_sharpe_ci_hi is not None
    assert row.mn_sharpe_ci_lo <= row.mn_sharpe <= row.mn_sharpe_ci_hi
    assert row.borrow_drag_bps is not None and row.borrow_drag_bps >= 0
    assert row.spread_drag_bps is not None and row.spread_drag_bps >= 0
    assert isinstance(row.passes, bool)


def test_market_neutral_leaderboard_rederives_top_n_trial_as_top_n_not_quintile(
    tmp_path,
):
    # Mirrors test_long_only_leaderboard_rederives_top_n_trial_as_top_n_not_
    # quintile: the --market-neutral re-derivation must read a top-N trial's
    # journaled params and rebuild it via the fixed-count construction.
    from trading.alphasearch.costs import cost_charged_market_neutral
    from trading.alphasearch.sort import portfolio_sort
    from trading.alphasearch.stats import sharpe_ci

    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=WINDOW, top_n=5,
              panel_factory=lambda _u, _f: panel)
    rows = build_market_neutral_leaderboard(
        journal, _universe(tmp_path), factors,
        panel_factory=lambda _u, _f: panel,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.error is None
    assert row.top_n == 5
    start, end = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-06-30", tz="UTC")
    dates = panel.decision_dates(start, end)
    direct = portfolio_sort(panel, SIGNALS["mom21"], dates, end, top_n=5)
    charged, _diag = cost_charged_market_neutral(panel, direct)
    expected_sharpe, _lo, _hi = sharpe_ci(charged, seed=0)
    assert row.mn_sharpe == pytest.approx(expected_sharpe)


def test_market_neutral_leaderboard_pass_requires_ci_lo_positive_on_full_and_both_halves(
    tmp_path, monkeypatch,
):
    # spec sections 3-4: passes iff the full-window CI lower bound AND both
    # discovery-half CI lower bounds are > 0. Plant a signal so obviously
    # strong and low-variance that ALL three CI-lo checks clear.
    panel = make_panel()
    factors = make_factors()

    def fn(view, as_of):
        # A rank-based score perfectly correlated with a persistent per-
        # symbol drift the fixture's OWN closes carry (make_panel drifts
        # symbol i at (i - n/2)*2bp/day) -- top bucket outperforms bottom by
        # construction, every month, both halves.
        return pd.Series({s: float(i) for i, s in enumerate(sorted(view.symbols))})

    spec = SignalSpec("strong_mn", fn)
    monkeypatch.setitem(SIGNALS, "strong_mn", spec)
    journal = trials_journal(tmp_path / "journal")
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals={"strong_mn": spec}, window=WINDOW,
              panel_factory=lambda _u, _f: panel)
    rows = build_market_neutral_leaderboard(
        journal, _universe(tmp_path), factors,
        panel_factory=lambda _u, _f: panel,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.error is None
    # The engineered spread is strong and stable enough that all three CI-lo
    # checks clear -- a real behavioral pass, not just internal consistency.
    assert row.mn_sharpe_ci_lo is not None and row.mn_sharpe_ci_lo > 0
    assert row.half1_ci_lo is not None and row.half1_ci_lo > 0
    assert row.half2_ci_lo is not None and row.half2_ci_lo > 0
    assert row.passes is True
    # And the pass flag is EXACTLY that three-way conjunction (spec sections
    # 3-4), not some other derivation.
    assert row.passes == (
        row.mn_sharpe_ci_lo is not None and row.mn_sharpe_ci_lo > 0
        and row.half1_ci_lo is not None and row.half1_ci_lo > 0
        and row.half2_ci_lo is not None and row.half2_ci_lo > 0
    )


def test_market_neutral_leaderboard_reports_na_for_unresolvable_universe(tmp_path):
    from trading.alphasearch.sweep import log_trial, trial_config

    journal = trials_journal(tmp_path / "journal")
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap:gone", WINDOW),
              ts="t1", result=_result_like(alpha_annual_pct=8.0, alpha_t=3.0, p=0.01))
    rows = build_market_neutral_leaderboard(
        journal, _universe(tmp_path), make_factors(),
    )
    assert len(rows) == 1
    assert rows[0].error is not None and "unknown universe" in rows[0].error
    assert rows[0].mn_sharpe is None and rows[0].passes is False


def test_market_neutral_leaderboard_reports_na_for_unknown_signal(tmp_path):
    from trading.alphasearch.sweep import log_trial, trial_config

    journal = trials_journal(tmp_path / "journal")
    log_trial(journal, kind="discovery",
              config=trial_config("no_longer_exists", "largecap", WINDOW),
              ts="t1", result=_result_like(alpha_annual_pct=8.0, alpha_t=3.0, p=0.01))
    rows = build_market_neutral_leaderboard(
        journal, _universe(tmp_path), make_factors(),
        panel_factory=lambda _u, _f: make_panel(),
    )
    assert len(rows) == 1
    assert rows[0].error is not None and "unknown signal" in rows[0].error


def test_market_neutral_leaderboard_sorts_by_ci_lo_with_na_last(tmp_path):
    from trading.alphasearch.sweep import log_trial, trial_config

    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21", "rev5"), window=WINDOW,
              panel_factory=lambda _u, _f: panel)
    log_trial(journal, kind="discovery",
              config=trial_config("gone", "largecap", WINDOW),
              ts="t1", result=_result_like(alpha_annual_pct=1.0, alpha_t=1.0, p=0.5))
    rows = build_market_neutral_leaderboard(
        journal, _universe(tmp_path), factors,
        panel_factory=lambda _u, _f: panel,
    )
    assert len(rows) == 3
    ranked = [r for r in rows if r.error is None]
    assert len(ranked) == 2
    assert ranked[0].mn_sharpe_ci_lo >= ranked[1].mn_sharpe_ci_lo
    assert rows[-1].error is not None   # the n/a row sorts last


def test_market_neutral_leaderboard_is_deterministic_across_calls(tmp_path):
    # Same seed default (seed=0): re-running the leaderboard (a display, no
    # re-journaling) must reproduce identical CI bounds.
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=WINDOW,
              panel_factory=lambda _u, _f: panel)
    kwargs = dict(panel_factory=lambda _u, _f: panel)
    first = build_market_neutral_leaderboard(journal, _universe(tmp_path), factors, **kwargs)
    second = build_market_neutral_leaderboard(journal, _universe(tmp_path), factors, **kwargs)
    assert first == second


def test_market_neutral_row_is_a_frozen_dataclass_with_the_spec_fields():
    import dataclasses

    row = MarketNeutralRow(
        signal="s", universe="u", window="w", top_n=None, config_hash="h",
        mn_sharpe=None, mn_sharpe_ci_lo=None, mn_sharpe_ci_hi=None,
        mn_total_return=None, borrow_drag_bps=None, spread_drag_bps=None,
        half1_ci_lo=None, half2_ci_lo=None, passes=False, error="boom",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.signal = "other"    # frozen
