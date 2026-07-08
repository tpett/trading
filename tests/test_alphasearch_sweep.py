"""Sweep runner + leaderboard: journaling order, honesty, refusal, recompute."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphasearch_helpers import make_factors, make_panel
from trading.alphasearch.spec import SIGNALS
from trading.alphasearch.sweep import (
    SweepError,
    UniverseSpec,
    build_leaderboard,
    discovery_trials,
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
