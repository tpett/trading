"""Piece 3 robustness battery: frozen thresholds, per-check behavior,
cost/capacity arithmetic, verdict + gate honesty."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alphasearch_helpers import make_factors, make_panel
from trading.alphasearch.panel import PanelData
from trading.alphasearch.robustness import (
    BatteryContext,
    check_jitter,
    check_offset,
    check_subperiods,
    check_subsets,
    require_survivor,
    signed_retention,
    subperiod_windows,
    subset_draw,
)
from trading.alphasearch.spec import SIGNALS, SignalSpec
from trading.alphasearch.sweep import (
    DISCOVERY_WINDOW,
    SweepError,
    evaluate_trial,
    log_trial,
    trial_config,
    trials_journal,
)

WINDOW = "2020-01-01..2020-06-30"


def _ctx(journal, panel, factors, *, spec=None, full_alpha=None,
         window=WINDOW, tercile_below=50, min_names=15, quantiles=5):
    spec = spec if spec is not None else SIGNALS["mom21"]
    if full_alpha is None:
        full = evaluate_trial(panel, spec, window, factors,
                              quantiles=quantiles, tercile_below=tercile_below,
                              min_names=min_names)
        full_alpha = full["ls"]["alpha_annual_pct"]
    return BatteryContext(
        journal=journal, panel=panel, spec=spec, factors=factors, ts="t1",
        universe="largecap", window=window, full_alpha=float(full_alpha),
        quantiles=quantiles, tercile_below=tercile_below, min_names=min_names,
        tag=f"{spec.name}:largecap",
    )


def test_subperiod_windows_pin_the_frozen_discovery_split():
    # Spec section 3 check 1, verbatim: the production discovery window
    # splits into EXACTLY these two halves. FROZEN.
    assert subperiod_windows(DISCOVERY_WINDOW) == (
        "2019-01-01..2021-06-30", "2021-07-01..2023-12-31",
    )


def test_subperiod_windows_fixture_split():
    assert subperiod_windows(WINDOW) == (
        "2020-01-01..2020-03-30", "2020-03-31..2020-06-30",
    )


def test_signed_retention_rules():
    assert signed_retention(10.0, 6.0) == pytest.approx(0.6)
    assert signed_retention(10.0, -6.0) == pytest.approx(-0.6)   # sign flip
    assert signed_retention(-10.0, -6.0) == pytest.approx(0.6)   # symmetric
    assert math.isnan(signed_retention(0.0, 1.0))
    assert math.isnan(signed_retention(None, 1.0))
    assert math.isnan(signed_retention(10.0, None))
    assert math.isnan(signed_retention(float("nan"), 1.0))


def test_subset_draw_is_deterministic_sorted_and_half_sized():
    symbols = tuple(f"S{i:02d}" for i in range(40))
    scrambled = tuple(reversed(symbols))
    first = subset_draw(symbols, 0)
    assert first == subset_draw(scrambled, 0)      # input-order-proof
    assert first == subset_draw(symbols, 0)        # deterministic
    assert list(first) == sorted(first)
    assert len(first) == 20
    assert set(first) <= set(symbols)
    assert first != subset_draw(symbols, 1)        # seeds 42 vs 43 differ


def test_check_subperiods_passes_on_a_stable_fixture(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(n_symbols=40)
    ctx = _ctx(journal, panel, make_factors())
    got = check_subperiods(ctx)
    assert got.passed
    assert got.number == 1 and got.name == "sub_period_halves"
    assert len(got.detail["halves"]) == 2
    events = list(journal.events())
    assert len(events) == 2                        # both halves journaled
    assert all(e["battery"] == "mom21:largecap" for e in events)
    assert all(e["kind"] == "discovery" for e in events)


def _two_regime_panel(n=40, periods=130):
    """Planted two-regime cross-section (spec section 7): the drift spread
    REVERSES at the midpoint, so a fixed-rank signal's L/S alpha flips sign
    in the second half -- a deterministic check-1 failure via the sign rule
    (a zero-drift second half would leave |t| to noise, ~32% flaky)."""
    idx = pd.date_range("2020-01-02", periods=periods, freq="B", tz="UTC")
    names = [f"S{i:02d}" for i in range(n)]
    half = periods // 2
    rng = np.random.default_rng(11)
    closes = {}
    for i, sym in enumerate(names):
        drift = (i - n / 2) * 4e-4
        rets = np.concatenate(
            [np.full(half, drift), np.full(periods - half, -drift)]
        ) + rng.normal(0.0, 1e-4, size=periods)
        closes[sym] = pd.Series(100.0 * np.cumprod(1 + rets), index=idx)
    return PanelData(closes=closes, symbols=tuple(names))


def _planted_rank_spec() -> SignalSpec:
    """History-free fixed ranking: symbol S<i> scores i. Unlike momentum it
    cannot re-learn a reversed regime, so the two-regime failure is exact."""

    def fn(view, as_of):
        return pd.Series(
            {s: float(i) for i, s in enumerate(sorted(view.symbols))},
            dtype="float64",
        )

    return SignalSpec("planted_rank", fn)


def test_check_subperiods_fails_a_two_regime_signal(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = _two_regime_panel()
    # The discovery baseline is the first-half sign (positive): the battery
    # compares halves against the JOURNALED full-window alpha, which the
    # test supplies directly.
    ctx = _ctx(journal, panel, make_factors(), spec=_planted_rank_spec(),
               full_alpha=10.0)
    got = check_subperiods(ctx)
    assert not got.passed
    first, second = got.detail["halves"]
    assert first["passed"] is True                 # regime 1: strong + right sign
    assert second["passed"] is False               # regime 2: sign flipped
    assert second["alpha_annual_pct"] < 0


def test_check_subsets_passes_on_a_wide_fixture(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(n_symbols=40)
    ctx = _ctx(journal, panel, make_factors())
    got = check_subsets(ctx)
    assert got.passed and got.detail["n_pass"] == 5
    assert len(list(journal.events())) == 5        # one tagged trial per draw
    # Draw subsets entered the hashed configs (sorted lists).
    for event in journal.events():
        assert event["params"]["symbol_subset"] == sorted(
            event["params"]["symbol_subset"]
        )
        assert len(event["params"]["symbol_subset"]) == 20


def test_check_subsets_error_draws_journal_and_fail(tmp_path):
    # 16-symbol panel: half-draws of 8 < min_names 15 -> every draw journals
    # an honest SortError trial AND fails (spec section 6: an uncomputable
    # perturbation is not a pass). The check itself fails 0/5.
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()                            # 16 symbols
    ctx = _ctx(journal, panel, make_factors(), full_alpha=10.0)
    got = check_subsets(ctx)
    assert not got.passed and got.detail["n_pass"] == 0
    events = list(journal.events())
    assert len(events) == 5
    assert all(e["error"] is not None and "SortError" in e["error"]
               for e in events)
    assert all(e["battery"] == "mom21:largecap" for e in events)


def test_check_jitter_runs_the_frozen_grid(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(n_symbols=40)
    # tercile_below=10 so the jittered quantile counts actually bind on a
    # 40-name fixture (40 >= 10 -> quantiles apply, not the tercile fallback).
    ctx = _ctx(journal, panel, make_factors(), tercile_below=10)
    got = check_jitter(ctx)
    assert got.passed
    grid = {(t["quantiles"], t["min_names"]) for t in got.detail["trials"]}
    assert grid == {(4, 10), (4, 20), (6, 10), (6, 20)}   # FROZEN
    assert len(list(journal.events())) == 4
    hashes = {e["config_hash"] for e in journal.events()}
    assert len(hashes) == 4                        # each jitter is a NEW trial


def test_check_offset_passes_on_a_stable_fixture(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(n_symbols=40)
    ctx = _ctx(journal, panel, make_factors())
    got = check_offset(ctx)
    assert got.passed
    assert got.detail["retention"] >= 0.5
    (event,) = list(journal.events())
    assert event["params"]["calendar_offset"] == 1  # hashed perturbation


def _result_like(*, alpha_annual_pct: float, alpha_t: float, p: float) -> dict:
    leg = {
        "alpha_annual_pct": alpha_annual_pct, "alpha_t": alpha_t, "p": p,
        "capm_alpha_annual_pct": alpha_annual_pct, "capm_alpha_t": alpha_t,
        "loadings": {}, "loadings_t": {}, "r2": 0.0, "n_obs": 120,
        "sharpe": 0.1, "sharpe_daily": 0.006, "skew": 0.0, "kurt": 3.0,
    }
    return {"n_dates": 3, "n_names_median": 16.0, "ls": leg, "lo": dict(leg),
            "turnover_monthly": 0.3, "skipped_dates": []}


def test_require_survivor_refuses_nonsurvivor_listing_survivors(tmp_path):
    from trading.alphasearch.sweep import DEFAULT_PARAMS

    journal = trials_journal(tmp_path / "journal")
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", WINDOW), ts="t1",
              result=_result_like(alpha_annual_pct=12.0, alpha_t=8.0, p=1e-8))
    log_trial(journal, kind="discovery",
              config=trial_config("rvol21", "largecap", WINDOW), ts="t1",
              result=_result_like(alpha_annual_pct=0.3, alpha_t=0.1, p=0.92))
    events_before = len(list(journal.events()))
    with pytest.raises(SweepError) as excinfo:
        require_survivor(journal, "rvol21", "largecap", WINDOW,
                         dict(DEFAULT_PARAMS))
    assert "not a current BH survivor" in str(excinfo.value)
    assert "mom21:largecap" in str(excinfo.value)   # lists current survivors
    assert len(list(journal.events())) == events_before  # journaled NOTHING
    # The survivor itself is admitted and returned.
    got = require_survivor(journal, "mom21", "largecap", WINDOW,
                           dict(DEFAULT_PARAMS))
    assert got["signal"] == "mom21"


def test_require_survivor_refuses_unknown_missing_and_errored(tmp_path):
    from trading.alphasearch.sweep import DEFAULT_PARAMS

    journal = trials_journal(tmp_path / "journal")
    with pytest.raises(SweepError, match="unknown signal"):
        require_survivor(journal, "no_such", "largecap", WINDOW,
                         dict(DEFAULT_PARAMS))
    with pytest.raises(SweepError, match="no discovery trial"):
        require_survivor(journal, "mom21", "largecap", WINDOW,
                         dict(DEFAULT_PARAMS))
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", WINDOW), ts="t1",
              error="SortError: boom")
    with pytest.raises(SweepError, match="errored"):
        require_survivor(journal, "mom21", "largecap", WINDOW,
                         dict(DEFAULT_PARAMS))
    assert len(list(journal.events())) == 1          # refusals journal nothing
