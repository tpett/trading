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


def _hand_closes(values: dict[str, list[float]], start="2020-01-06") -> dict:
    idx = pd.date_range(start, periods=len(next(iter(values.values()))),
                        freq="B", tz="UTC")
    return {sym: pd.Series(v, index=idx) for sym, v in values.items()}


def test_ls_series_replays_portfolio_sort_exactly(tmp_path):
    from trading.alphasearch.robustness import ls_series
    from trading.alphasearch.sort import portfolio_sort

    panel = make_panel(n_symbols=40)
    idx = panel.closes[panel.symbols[0]].index
    dates = panel.decision_dates(idx[0], idx[-1])
    sort = portfolio_sort(panel, SIGNALS["mom21"], dates, idx[-1])
    replayed = ls_series(panel.closes, sort.rebalances, idx[-1])
    pd.testing.assert_series_equal(replayed, sort.ls)


def test_ls_series_excludes_names_from_both_legs():
    from trading.alphasearch.robustness import ls_series

    # 4 names, constant daily growth: A 4%, B 3%, C 1%, D 0%. One rebalance:
    # top=(A,B), bottom=(C,D). Excluding A leaves top=B alone.
    closes = _hand_closes({
        "A": [100.0, 104.0], "B": [100.0, 103.0],
        "C": [100.0, 101.0], "D": [100.0, 100.0],
    })
    date, end = closes["A"].index[0], closes["A"].index[-1]
    rebalances = ((date, ("A", "B"), ("C", "D")),)
    full = ls_series(closes, rebalances, end)
    assert full.iloc[0] == pytest.approx((0.04 + 0.03) / 2 - (0.01 + 0.0) / 2)
    reduced = ls_series(closes, rebalances, end, excluded=frozenset({"A"}))
    assert reduced.iloc[0] == pytest.approx(0.03 - 0.005)
    # Excluding a bottom name symmetrically:
    reduced2 = ls_series(closes, rebalances, end, excluded=frozenset({"D"}))
    assert reduced2.iloc[0] == pytest.approx(0.035 - 0.01)


def test_ls_series_emptied_leg_contributes_nothing():
    from trading.alphasearch.robustness import ls_series
    from trading.alphasearch.sort import SortError

    closes = _hand_closes({"A": [100.0, 104.0], "B": [100.0, 101.0]})
    date, end = closes["A"].index[0], closes["A"].index[-1]
    rebalances = ((date, ("A",), ("B",)),)
    with pytest.raises(SortError):
        ls_series(closes, rebalances, end, excluded=frozenset({"A"}))


def test_top_leg_contributions_hand_computed():
    from trading.alphasearch.robustness import top_leg_contributions

    # top=(A,B) held two days; equal weight 1/2. A returns 4% then ~1.923%,
    # B returns 1% then ~0.990%: contributions are the summed ret/2.
    closes = _hand_closes({
        "A": [100.0, 104.0, 106.0],
        "B": [100.0, 101.0, 102.0],
        "C": [100.0, 100.0, 100.0],
    })
    date, end = closes["A"].index[0], closes["A"].index[-1]
    rebalances = ((date, ("A", "B"), ("C",)),)
    got = top_leg_contributions(closes, rebalances, end)
    assert got.index[0] == "A"                       # ranked descending
    assert got["A"] == pytest.approx((0.04 + 2.0 / 104.0) / 2)
    assert got["B"] == pytest.approx((0.01 + 1.0 / 101.0) / 2)
    assert "C" not in got.index                      # bottom leg never counted


def _six_panel(periods=65, truncate: dict[str, int] | None = None) -> PanelData:
    """Constant-growth six-name panel (the sort tests' SIX rates); `truncate`
    cuts a symbol's series after N bars -- a mid-window delisting."""
    rates = {"S1": -0.02, "S2": -0.01, "S3": 0.0,
             "S4": 0.01, "S5": 0.02, "S6": 0.03}
    idx = pd.date_range("2020-01-02", periods=periods, freq="B", tz="UTC")
    closes = {
        sym: pd.Series([100.0 * (1 + r) ** i for i in range(periods)], index=idx)
        for sym, r in rates.items()
    }
    for sym, n in (truncate or {}).items():
        closes[sym] = closes[sym].iloc[:n]
    return PanelData(closes=closes, symbols=tuple(sorted(closes)))


def test_ls_series_replay_holds_through_a_mid_sequence_skip():
    # A thin cross-section SANDWICHED between rebalances: portfolio_sort
    # holds the prior portfolio through the skipped date (two sub-interval
    # parts), while the replay spans the whole holding as ONE segment --
    # the series must still be bit-identical.
    from trading.alphasearch.robustness import ls_series
    from trading.alphasearch.sort import portfolio_sort

    panel = _six_panel()
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[0], idx[-1])[:3]
    assert len(dates) == 3
    calls = {"n": 0}

    def fn(view, as_of):
        calls["n"] += 1
        if calls["n"] == 2:
            return pd.Series({"S1": 1.0, "S2": 2.0})  # thin -> skip, hold
        base = {"S1": 1.0, "S2": 2.0, "S3": 3.0, "S4": 4.0, "S5": 5.0, "S6": 6.0}
        if calls["n"] == 3:
            base["S6"] = 0.0                          # rotate on the 3rd date
        return pd.Series(base, dtype="float64")

    sort = portfolio_sort(panel, SignalSpec("skipper", fn), dates, idx[-1],
                          quantiles=3, tercile_below=0, min_names=3)
    assert sort.skipped_dates == (dates[1].date().isoformat(),)
    replayed = ls_series(panel.closes, sort.rebalances, idx[-1])
    pd.testing.assert_series_equal(replayed, sort.ls)


def test_ls_series_replay_matches_through_a_mid_holding_delisting():
    # S6 (a top-leg name) delists mid-holding: the leg mean's skipna
    # denominator shrinks, and the replay must track it bit-identically.
    from trading.alphasearch.robustness import ls_series
    from trading.alphasearch.sort import portfolio_sort

    panel = _six_panel(truncate={"S6": 45})
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[30], idx[-1])[:1]  # one rebalance to end
    sort = portfolio_sort(panel, SIGNALS["mom21"], dates, idx[-1], min_names=3)
    assert sort.rebalances[0][1] == ("S5", "S6")        # S6 was in the top leg
    replayed = ls_series(panel.closes, sort.rebalances, idx[-1])
    pd.testing.assert_series_equal(replayed, sort.ls)


def test_top_leg_contributions_reconcile_to_lo_through_a_delisting():
    # Contribution weighting must match the REALIZED leg: mean(skipna) puts
    # weight 1/n_live on each surviving member, so after S6 delists S5
    # carries the whole leg. By construction the contributions then sum to
    # the lo series exactly -- a stale 1/n_top denominator understates the
    # survivor and can rank check 5's exclusion candidates wrongly.
    from trading.alphasearch.robustness import top_leg_contributions
    from trading.alphasearch.sort import portfolio_sort

    panel = _six_panel(truncate={"S6": 45})
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[30], idx[-1])[:1]
    sort = portfolio_sort(panel, SIGNALS["mom21"], dates, idx[-1], min_names=3)
    got = top_leg_contributions(panel.closes, sort.rebalances, idx[-1])
    assert got.sum() == pytest.approx(sort.lo.sum(), rel=1e-12)


def test_check_name_concentration_fails_a_three_name_alpha(tmp_path):
    # Spec section 7's three-name fixture: 3 monsters carry the whole top
    # leg; excluding them collapses the alpha below half -> FAIL.
    from trading.alphasearch.robustness import check_name_concentration
    from trading.alphasearch.sort import portfolio_sort

    idx = pd.date_range("2020-01-02", periods=130, freq="B", tz="UTC")
    names = [f"S{i:02d}" for i in range(40)]
    rng = np.random.default_rng(5)
    closes = {}
    for i, sym in enumerate(names):
        drift = 0.02 if i >= 37 else 0.0             # 3 monsters, 37 duds
        rets = drift + rng.normal(0.0, 1e-4, size=130)
        closes[sym] = pd.Series(100.0 * np.cumprod(1 + rets), index=idx)
    panel = PanelData(closes=closes, symbols=tuple(names))
    journal = trials_journal(tmp_path / "journal")
    factors = make_factors()
    ctx = _ctx(journal, panel, factors, spec=_planted_rank_spec())
    # The sort is built over the ctx WINDOW bounds, exactly as run_battery
    # does (check 5 replays memberships to the window end, not the last bar).
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2020-06-30", tz="UTC")
    sort = portfolio_sort(panel, _planted_rank_spec(),
                          panel.decision_dates(start, end), end)
    got = check_name_concentration(ctx, sort)
    assert not got.passed
    assert set(got.detail["excluded"]) == {"S37", "S38", "S39"}
    assert got.detail["retention"] < 0.5
    assert len(list(journal.events())) == 0          # arithmetic: NO new trials


def test_check_name_concentration_passes_a_broad_alpha(tmp_path):
    from trading.alphasearch.robustness import check_name_concentration
    from trading.alphasearch.sort import portfolio_sort

    panel = make_panel(n_symbols=40)                 # linear drift spread
    journal = trials_journal(tmp_path / "journal")
    ctx = _ctx(journal, panel, make_factors())
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2020-06-30", tz="UTC")
    sort = portfolio_sort(panel, SIGNALS["mom21"],
                          panel.decision_dates(start, end), end)
    got = check_name_concentration(ctx, sort)
    assert got.passed
    assert got.number == 5 and got.name == "name_concentration"


def test_month_share_hand_computed():
    from trading.alphasearch.robustness import month_share

    # One trading day per month with known LOG returns .01/.02/.03/.04:
    # top-3 share = .09/.10. expm1 round-trips through log1p.
    idx = pd.DatetimeIndex(
        ["2020-01-15", "2020-02-14", "2020-03-16", "2020-04-15"], tz="UTC"
    )
    ls = pd.Series(np.expm1([0.01, 0.02, 0.03, 0.04]), index=idx)
    assert month_share(ls) == pytest.approx(0.09 / 0.10)
    # Non-positive cumulative log return: concentration is undefined -> NaN
    # (the caller FAILS the check; spec section 6).
    flat = pd.Series(np.expm1([-0.02, 0.01]), index=idx[:2])
    assert math.isnan(month_share(flat))


def test_check_month_concentration_fails_a_single_month_spike():
    from trading.alphasearch.robustness import check_month_concentration

    idx = pd.date_range("2020-01-02", periods=105, freq="B", tz="UTC")
    values = np.full(105, 0.0001)
    march = (idx.month == 3)
    values[march] = 0.01                             # the spike month
    got = check_month_concentration(pd.Series(values, index=idx))
    assert not got.passed
    assert got.number == 6 and got.name == "month_concentration"
    assert got.detail["top3_share"] > 0.60


def test_check_month_concentration_passes_an_even_series():
    from trading.alphasearch.robustness import check_month_concentration

    idx = pd.date_range("2020-01-02", periods=130, freq="B", tz="UTC")
    got = check_month_concentration(pd.Series(0.001, index=idx))
    assert got.passed                                # ~6 even months: 3/6 = 50%


def test_factor_proxy_flag_is_the_smb_costume_detector():
    from trading.alphasearch.robustness import factor_proxy_flag

    costume = {"alpha_t": 3.0, "r2": 0.6,
               "loadings_t": {"Mkt-RF": 1.0, "SMB": 9.0, "HML": 0.5, "Mom": 0.2}}
    got = factor_proxy_flag(costume)
    assert got["flagged"] is True
    assert got["offenders"] == {"SMB": 9.0}
    # R^2 below the floor: high loading t alone does not flag.
    low_r2 = dict(costume, r2=0.4)
    assert factor_proxy_flag(low_r2)["flagged"] is False
    # Loading below 2x |alpha t|: no flag.
    mild = dict(costume, loadings_t={"SMB": 5.9})
    assert factor_proxy_flag(mild)["flagged"] is False
    # Missing stats (errored trial): never flags, never crashes.
    assert factor_proxy_flag({})["flagged"] is False


def test_apply_rebalance_charges_lands_on_the_first_following_day():
    from trading.alphasearch.robustness import apply_rebalance_charges

    idx = pd.date_range("2020-01-06", periods=10, freq="B", tz="UTC")
    ls = pd.Series(0.001, index=idx)
    charged = apply_rebalance_charges(
        ls, [(idx[0], 0.003), (idx[5], 0.003), (idx[-1], 0.5)]
    )
    assert charged.iloc[1] == pytest.approx(0.001 - 0.003)
    assert charged.iloc[6] == pytest.approx(0.001 - 0.003)
    # A charge dated on the last day has no following return day: dropped.
    # Untouched entries are copies, bit-identical to the input.
    assert charged.drop(charged.index[[1, 6]]).eq(0.001).all()
    assert ls.eq(0.001).all()                        # input never mutated


def test_cost_adjusted_table_hand_arithmetic_and_monotone(tmp_path):
    from trading.alphasearch.robustness import apply_rebalance_charges, cost_adjusted_table
    from trading.alphasearch.sort import portfolio_sort

    panel = make_panel(n_symbols=40)
    factors = make_factors()
    idx = panel.closes[panel.symbols[0]].index
    dates = panel.decision_dates(idx[0], idx[-1])
    sort = portfolio_sort(panel, SIGNALS["mom21"], dates, idx[-1])
    rows = cost_adjusted_table(sort.ls, sort.rebalances,
                               sort.turnover_monthly, factors)
    assert [r["cost_bps"] for r in rows] == [10, 30, 50]        # FROZEN
    # Hand check the 30 bps charge: 2 legs x turnover x 0.003 per rebalance.
    per_rebalance = 2.0 * sort.turnover_monthly * 30 / 1e4
    charged = apply_rebalance_charges(
        sort.ls, [(d, per_rebalance) for d, _t, _b in sort.rebalances]
    )
    from trading.alphasearch.evaluate import evaluate_alpha
    expected = evaluate_alpha(charged, factors, self_financing=True)
    row30 = rows[1]
    assert row30["alpha_annual_pct"] == pytest.approx(expected.alpha_annual_pct)
    assert row30["alpha_t"] == pytest.approx(expected.alpha_tstat)
    # Costs only ever hurt: alpha decreases with c.
    alphas = [r["alpha_annual_pct"] for r in rows]
    assert alphas[0] > alphas[1] > alphas[2]


def test_cost_adjusted_table_without_turnover_reports_error():
    from trading.alphasearch.robustness import cost_adjusted_table

    idx = pd.date_range("2020-01-06", periods=10, freq="B", tz="UTC")
    ls = pd.Series(0.001, index=idx)
    rows = cost_adjusted_table(ls, ((idx[0], ("A",), ("B",)),),
                               float("nan"), make_factors())
    assert all(r["alpha_t"] is None for r in rows)
    assert all("turnover" in r["error"] for r in rows)


def _lambda_bars(n: int, daily_ret: float, dollar: float) -> pd.DataFrame:
    """Bars with constant pct return and constant dollar volume, so
    amihud_lambda == daily_ret / dollar exactly (the tier1 closed form)."""
    idx = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    close = pd.Series(100.0 * (1 + daily_ret) ** np.arange(n), index=idx)
    return pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
        "volume": dollar / close, "div_cash": 0.0, "split_factor": 1.0,
        "close_raw": close,
    })


def test_capacity_curve_two_name_hand_computed():
    from trading.alphasearch.robustness import capacity_curve

    # lambda_A = .01/1e6 = 1e-8, lambda_B = .01/5e5 = 2e-8 (known lambdas).
    # 200 bars so the formation date sees >= 126 valid return terms (the
    # amihud_lambda floor) -- at idx[130] each name has 130 valid terms.
    bars = {"A": _lambda_bars(200, 0.01, 1e6), "B": _lambda_bars(200, 0.01, 5e5)}
    closes = {s: f["close"] for s, f in bars.items()}
    panel = PanelData(closes=closes, bars=bars, symbols=("A", "B"))
    idx = closes["A"].index
    factors = make_factors(periods=300)
    # One formation: both names enter 1-name legs; 69 held days follow.
    date = idx[130]
    rebalances = ((date, ("A",), ("B",)),)
    ls = pd.Series(0.001, index=idx[131:])
    rows = capacity_curve(panel, rebalances, ls, factors)
    assert [r["book_usd"] for r in rows] == [1e4, 1e5, 1e6]     # FROZEN
    # Entry drag per $1 of book: lambda_A/1^2 + lambda_B/1^2 = 3e-8.
    assert rows[2]["total_impact_charge"] == pytest.approx(3e-8 * 1e6, rel=1e-6)
    assert rows[0]["total_impact_charge"] == pytest.approx(3e-8 * 1e4, rel=1e-6)
    assert all(r["skipped_no_lambda"] == 0 for r in rows)
    # More book, more impact, less alpha.
    alphas = [r["alpha_annual_pct"] for r in rows]
    assert alphas[0] > alphas[1] > alphas[2]


def test_capacity_curve_charges_exits_at_the_old_leg_size():
    from trading.alphasearch.robustness import capacity_curve

    # 200 bars: both rebalance dates clear the 126-valid-term lambda floor.
    bars = {"A": _lambda_bars(200, 0.01, 1e6), "B": _lambda_bars(200, 0.01, 1e6),
            "C": _lambda_bars(200, 0.01, 1e6), "D": _lambda_bars(200, 0.01, 1e6)}
    closes = {s: f["close"] for s, f in bars.items()}
    panel = PanelData(closes=closes, bars=bars, symbols=("A", "B", "C", "D"))
    idx = closes["A"].index
    lam = 1e-8
    # Formation: top=(A,B) bottom=(C,) -- then A,B exit for (D,) top.
    rebalances = (
        (idx[130], ("A", "B"), ("C",)),
        (idx[140], ("D",), ("C",)),
    )
    ls = pd.Series(0.001, index=idx[131:])
    rows = capacity_curve(panel, rebalances, ls, make_factors(periods=300))
    # Formation: A,B enter n=2 legs (2 x lam/4), C enters n=1 (lam).
    # Rebalance 2: D enters n=1 (lam); A,B exit at OLD n=2 (2 x lam/4).
    per_dollar = (2 * lam / 4 + lam) + (lam + 2 * lam / 4)
    assert rows[2]["total_impact_charge"] == pytest.approx(per_dollar * 1e6,
                                                           rel=1e-6)


def test_capacity_curve_counts_missing_lambda_names():
    from trading.alphasearch.robustness import capacity_curve

    bars = {"A": _lambda_bars(130, 0.01, 1e6),
            "B": _lambda_bars(60, 0.01, 1e6)}      # < 126 valid terms -> NaN
    closes = {s: f["close"] for s, f in bars.items()}
    panel = PanelData(closes=closes, bars=bars, symbols=("A", "B"))
    idx = closes["A"].index
    rebalances = ((idx[128], ("A",), ("B",)),)
    ls = pd.Series(0.001, index=idx[129:])
    rows = capacity_curve(panel, rebalances, ls, make_factors(periods=200))
    assert all(r["skipped_no_lambda"] == 1 for r in rows)
    assert rows[2]["total_impact_charge"] == pytest.approx(1e-8 * 1e6, rel=1e-6)


# --------------------------------------------------------------------------- #
# run_battery composition + verdict event
# --------------------------------------------------------------------------- #
def _swept_survivor(tmp_path, n_symbols=40):
    """A real BH survivor: mom21 swept on the wide fixture panel."""
    from trading.alphasearch.sweep import UniverseSpec, run_sweep

    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(n_symbols=n_symbols)
    factors = make_factors()
    uspec = UniverseSpec("largecap", tmp_path, tmp_path / "s.jsonl", None)
    run_sweep({"largecap": uspec}, journal, factors, ts="t0",
              signals={"mom21": SIGNALS["mom21"]}, window=WINDOW,
              panel_factory=lambda _u, _f: panel)
    return journal, panel, factors, uspec


def test_run_battery_refuses_nonsurvivor_before_journaling(tmp_path):
    from trading.alphasearch.robustness import run_battery
    from trading.alphasearch.sweep import UniverseSpec

    journal = trials_journal(tmp_path / "journal")
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", WINDOW), ts="t1",
              result=_result_like(alpha_annual_pct=12.0, alpha_t=8.0, p=1e-8))
    log_trial(journal, kind="discovery",
              config=trial_config("rvol21", "largecap", WINDOW), ts="t1",
              result=_result_like(alpha_annual_pct=0.3, alpha_t=0.1, p=0.92))
    uspec = UniverseSpec("largecap", tmp_path, tmp_path / "s.jsonl", None)
    before = len(list(journal.events()))
    with pytest.raises(SweepError, match="not a current BH survivor"):
        run_battery(uspec, journal, make_factors(), "t2", "rvol21",
                    discovery_window=WINDOW,
                    panel_factory=lambda _u, _f: make_panel())
    assert len(list(journal.events())) == before      # journaled NOTHING


def test_run_battery_journals_tagged_trials_and_a_verdict(tmp_path):
    from trading.alphasearch.robustness import run_battery
    from trading.alphasearch.sweep import discovery_trials

    journal, panel, factors, uspec = _swept_survivor(tmp_path)
    outcome = run_battery(uspec, journal, factors, "t1", "mom21",
                          discovery_window=WINDOW,
                          panel_factory=lambda _u, _f: panel)
    # 2 halves + 5 draws + 4 jitter + 1 offset = 12 tagged re-evaluations,
    # all BH-counted discovery trials on top of the 1 swept trial.
    trials = discovery_trials(journal)
    assert len(trials) == 13
    tagged = [t for t in trials if t.get("battery") == "mom21:largecap"]
    assert len(tagged) == 12
    # Checks 1-6 in spec order; 7 is the warning dict.
    assert [c.number for c in outcome.checks] == [1, 2, 3, 4, 5, 6]
    assert [c.name for c in outcome.checks] == [
        "sub_period_halves", "universe_subsets", "parameter_jitter",
        "decision_offset", "name_concentration", "month_concentration",
    ]
    assert isinstance(outcome.eligible, bool)
    assert set(outcome.factor_proxy) == {"flagged", "offenders", "alpha_t", "r2"}
    assert [r["cost_bps"] for r in outcome.cost_table] == [10, 30, 50]
    assert [r["book_usd"] for r in outcome.capacity_curve] == [1e4, 1e5, 1e6]
    # The verdict event: kind="battery", SAME config hash as the discovery
    # trial (default params), full payload, json-safe.
    event = outcome.event
    assert event["kind"] == "battery"
    discovery = next(t for t in trials if t.get("battery") is None
                     and t["signal"] == "mom21" and t["window"] == WINDOW)
    assert event["config_hash"] == discovery["config_hash"]
    assert event["eligible"] == outcome.eligible
    assert set(event["checks"]) == {c.name for c in outcome.checks}


def test_run_battery_rerun_replaces_the_verdict_and_double_counts_nothing(tmp_path):
    from trading.alphasearch.robustness import run_battery
    from trading.alphasearch.sweep import battery_verdict, discovery_trials

    journal, panel, factors, uspec = _swept_survivor(tmp_path)
    kwargs = dict(discovery_window=WINDOW, panel_factory=lambda _u, _f: panel)
    first = run_battery(uspec, journal, factors, "t1", "mom21", **kwargs)
    events_after_first = len(list(journal.events()))
    second = run_battery(uspec, journal, factors, "t2", "mom21", **kwargs)
    assert len(discovery_trials(journal)) == 13          # identical configs dedupe
    # Verdict replaced by config hash: ONE battery event survives dedupe,
    # and it is the LATEST.
    verdict = battery_verdict(journal, first.event["config_hash"])
    assert verdict is not None and verdict["ts"] == "t2"
    # ...but the journal itself is append-only (both runs appended).
    assert len(list(journal.events())) > events_after_first
    assert second.eligible == first.eligible             # deterministic


def test_run_battery_narrow_universe_fails_check2_via_error_trials(tmp_path):
    # 16 names: half-draws of 8 < min_names 15 -> the five subset trials are
    # honest SortError error trials, the check fails, the battery still
    # completes and journals its verdict (an uncomputable perturbation is a
    # FAIL, not a crash).
    from trading.alphasearch.robustness import run_battery

    journal, panel, factors, uspec = _swept_survivor(tmp_path, n_symbols=16)
    outcome = run_battery(uspec, journal, factors, "t1", "mom21",
                          discovery_window=WINDOW,
                          panel_factory=lambda _u, _f: panel)
    subsets = next(c for c in outcome.checks if c.name == "universe_subsets")
    assert not subsets.passed and subsets.detail["n_pass"] == 0
    assert not outcome.eligible
    errored = [e for e in journal.events()
               if e.get("battery") is not None and e.get("error") is not None]
    assert len(errored) >= 5
