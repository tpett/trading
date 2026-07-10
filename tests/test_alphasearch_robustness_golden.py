"""Golden end-to-end battery (spec section 7): real files -> build_panel ->
sweep -> run_battery -> verdict, deterministic fixture, no factory injection."""

from __future__ import annotations

import pandas as pd
import pytest

from alphasearch_helpers import make_factors, make_panel, make_spy_closes
from trading.alphasearch.robustness import run_battery
from trading.alphasearch.spec import SIGNALS
from trading.alphasearch.sweep import (
    SweepError,
    UniverseSpec,
    battery_verdict,
    discovery_trials,
    run_holdout,
    run_sweep,
    trials_journal,
)

WINDOW = "2020-01-01..2020-06-30"


def _write_universe(tmp_path) -> UniverseSpec:
    """make_panel(40)'s closes as real parquet caches + an explicit symbols
    tuple (the deep-pool universe shape): 40 names so the half-universe
    draws (20) clear MIN_NAMES 15 and check 2 genuinely runs."""
    panel = make_panel(n_symbols=40)
    cache = tmp_path / "cache"
    cache.mkdir()
    for sym in panel.symbols:
        closes = panel.closes[sym]
        pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes,
             "volume": 1000.0},
            index=closes.index,
        ).to_parquet(cache / f"{sym}.parquet")
    return UniverseSpec("largecap:golden", cache, None, None,
                        symbols=panel.symbols)


def test_golden_battery_end_to_end(tmp_path):
    uspec = _write_universe(tmp_path)
    journal = trials_journal(tmp_path / "journal")
    factors = make_factors()
    run_sweep({uspec.name: uspec}, journal, factors, ts="t0",
              signals={"mom21": SIGNALS["mom21"]}, window=WINDOW)
    assert len(discovery_trials(journal)) == 1

    outcome = run_battery(uspec, journal, factors, "t1", "mom21",
                          discovery_window=WINDOW, spy_closes=make_spy_closes())

    # 12 battery-tagged, BH-counted discovery trials on top of the sweep's 1.
    trials = discovery_trials(journal)
    assert len(trials) == 13
    assert sum(1 for t in trials
               if t.get("battery") == "mom21:largecap:golden") == 12
    # Every check computed with numbers; the report is fully populated.
    assert [c.number for c in outcome.checks] == [1, 2, 3, 4, 5, 6]
    assert all(isinstance(c.passed, bool) for c in outcome.checks)
    subsets = next(c for c in outcome.checks if c.name == "universe_subsets")
    assert subsets.passed                     # 20-name draws genuinely ran
    concentration = next(c for c in outcome.checks
                         if c.name == "name_concentration")
    assert concentration.passed               # linear drift spread is broad
    # Costs monotonically eat alpha; capacity is populated (volume=1000
    # yields real lambdas on this fixture).
    alphas = [r["alpha_annual_pct"] for r in outcome.cost_table]
    assert alphas[0] > alphas[1] > alphas[2]
    assert all(r["alpha_t"] is not None for r in outcome.capacity_curve)
    # Verdict journaled under the discovery config's hash.
    verdict = battery_verdict(journal, outcome.event["config_hash"])
    assert verdict is not None and verdict["eligible"] == outcome.eligible

    # Bit-identical, count-stable re-run: dedupe by config hash everywhere.
    again = run_battery(uspec, journal, factors, "t2", "mom21",
                        discovery_window=WINDOW, spy_closes=make_spy_closes())
    assert len(discovery_trials(journal)) == 13
    assert again.eligible == outcome.eligible
    assert [c.passed for c in again.checks] == [c.passed for c in outcome.checks]
    verdict2 = battery_verdict(journal, outcome.event["config_hash"])
    assert verdict2["ts"] == "t2"             # replaced in place


def test_golden_battery_verdict_gates_a_real_holdout(tmp_path):
    """Chain test (controller-approved Task 5 carry-over): a REAL run_battery
    call's journaled verdict must be what a REAL run_holdout call reads to
    decide whether the once-only touch may be spent. No fabricated verdict
    (contrast _log_battery_verdict in test_alphasearch_sweep.py): if the
    verdict payload were ever reshuffled -- e.g. the "eligible" key silently
    moved or a battery-passed candidate started reading as failed -- either
    this run_battery outcome or the run_holdout gate's reaction to it would
    change, and one of the assertions below would break.

    On this deterministic 40-name fixture the battery is NOT eligible under
    the R1-re-anchored checks (the §2 long-only gate itself PASSES -- the
    planted drift spread crushes the synthetic SPY -- but check 6 fails: the
    active series' top-3 months carry ~91% of the cumulative log return over
    this 6-month window, far over the 60% frozen ceiling; check 1's first
    half also falls below the active-t floor against the noisy synthetic
    benchmark) -- so the holdout must refuse, pre-touch, naming the
    robustness command."""
    uspec = _write_universe(tmp_path)
    journal = trials_journal(tmp_path / "journal")
    factors = make_factors()
    run_sweep({uspec.name: uspec}, journal, factors, ts="t0",
              signals={"mom21": SIGNALS["mom21"]}, window=WINDOW)
    outcome = run_battery(uspec, journal, factors, "t1", "mom21",
                          discovery_window=WINDOW, spy_closes=make_spy_closes())

    # Pin the fixture's real outcome: the §2 comparator passes, yet the
    # re-anchored checks (6, and 1's noisy first half) still block -- the
    # battery is more than the comparator, by design.
    assert outcome.long_only_gate["passed"] is True
    assert outcome.eligible is False
    month = next(c for c in outcome.checks if c.name == "month_concentration")
    assert not month.passed
    verdict = battery_verdict(journal, outcome.event["config_hash"])
    assert verdict["eligible"] is False

    events_before = len(list(journal.events()))

    def refuse() -> str:
        # Would confirm a re-run if ever reached; the battery gate must fire
        # first, so this stub is never called.
        return "should never be reached"

    with pytest.raises(SweepError) as excinfo:
        run_holdout(uspec, journal, factors, "t2", "mom21",
                    discovery_window=WINDOW, confirm=refuse)
    assert "did not pass" in str(excinfo.value)
    assert "trading alphasearch robustness mom21:largecap:golden" in str(
        excinfo.value
    )
    # Refusal is pre-touch: no holdout event, nothing else journaled either.
    assert len(list(journal.events())) == events_before
    assert all(e.get("kind") != "holdout" for e in journal.events())


def test_run_battery_refuses_on_cache_drift_before_any_journaling(tmp_path):
    """Fix (final-review): data caches are gitignored/mutable, unlike the
    journal. If one cached parquet changes after the discovery sweep --
    append a bar, edit a close -- the rebuilt panel can produce a full-window
    alpha that no longer matches the journaled discovery baseline every
    retention check divides by, WITHOUT portfolio_sort ever raising SortError
    (the cross-section is still plenty large). run_battery must catch that
    drift itself and refuse before checks 1-4 journal a single battery
    trial -- otherwise a hash-replace re-run becomes a free re-roll channel
    for a marginal candidate."""
    uspec = _write_universe(tmp_path)
    journal = trials_journal(tmp_path / "journal")
    factors = make_factors()
    run_sweep({uspec.name: uspec}, journal, factors, ts="t0",
              signals={"mom21": SIGNALS["mom21"]}, window=WINDOW)
    events_before = len(list(journal.events()))

    # Mutate one symbol's cached bars: bump a single mid-window close. A
    # uniform rescale of the whole series would cancel out of pct_change()
    # entirely, so this edits one bar in place instead -- a real single-day
    # data error, not a scale change. The cross-section stays well above
    # MIN_NAMES (39 other names untouched), so portfolio_sort succeeds --
    # the drift must be caught by comparing alphas, not by a SortError.
    victim = uspec.cache_dir / f"{uspec.symbols[0]}.parquet"
    frame = pd.read_parquet(victim)
    mid = len(frame) // 2
    frame.iloc[mid, frame.columns.get_loc("close")] *= 1.2
    frame.to_parquet(victim)

    with pytest.raises(SweepError, match="caches drifted"):
        run_battery(uspec, journal, factors, "t2", "mom21",
                    discovery_window=WINDOW, spy_closes=make_spy_closes())
    assert len(list(journal.events())) == events_before  # journaled NOTHING


def test_run_battery_refuses_stale_factors_before_any_journaling(tmp_path):
    """Regression test (controller-approved Task 5 carry-over): factors
    ending mid-window must refuse in run_battery's own pre-check (spec
    section 4), BEFORE any of the 12 re-evaluations journal -- exactly the
    run_sweep-level guarantee test_discovery_trial_with_stale_factors_is_
    journaled_as_error pins for discovery trials, but for run_battery a stale
    cache must refuse loudly rather than journal 12 predictable error trials
    for one fixable data problem."""
    uspec = _write_universe(tmp_path)
    journal = trials_journal(tmp_path / "journal")
    factors = make_factors()
    run_sweep({uspec.name: uspec}, journal, factors, ts="t0",
              signals={"mom21": SIGNALS["mom21"]}, window=WINDOW)
    events_before = len(list(journal.events()))

    stale_factors = make_factors(periods=60)  # ends 2020-02-21, WINDOW ends 06-30
    with pytest.raises(SweepError, match="factor cache"):
        run_battery(uspec, journal, stale_factors, "t2", "mom21",
                    discovery_window=WINDOW, spy_closes=make_spy_closes())
    assert len(list(journal.events())) == events_before  # journaled NOTHING


def test_run_battery_refuses_when_spy_cache_is_absent_before_any_journaling(
    tmp_path, monkeypatch
):
    """R1 amendment (spec section 2): SPY is the frozen promotion comparator.
    An absent cache must refuse loudly, pre-touch -- not silently substitute
    another benchmark and not journal 12 predictable re-evaluations for one
    fixable data problem. Monkeypatches the default loader (rather than
    relying on a real missing cache dir) so the test stays independent of
    whatever data happens to be on disk."""
    uspec = _write_universe(tmp_path)
    journal = trials_journal(tmp_path / "journal")
    factors = make_factors()
    run_sweep({uspec.name: uspec}, journal, factors, ts="t0",
              signals={"mom21": SIGNALS["mom21"]}, window=WINDOW)
    events_before = len(list(journal.events()))

    monkeypatch.setattr("trading.alphasearch.robustness.load_spy_closes",
                        lambda *_a, **_k: None)
    with pytest.raises(SweepError, match="no SPY cache"):
        run_battery(uspec, journal, factors, "t2", "mom21", discovery_window=WINDOW)
    assert len(list(journal.events())) == events_before  # journaled NOTHING
