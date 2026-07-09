"""Hand-computable portfolio-sort fixtures (6 symbols x a few months)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trading.alphasearch.panel import PanelData
from trading.alphasearch.sort import (
    SortError,
    assign_quantiles,
    portfolio_sort,
)
from trading.alphasearch.spec import SignalSpec


def _panel(rates: dict[str, float], periods: int = 65) -> PanelData:
    """Constant daily growth per symbol: mom21 rank == rate rank, and the
    portfolio's daily return equals the mean of its members' rates."""
    idx = pd.date_range("2020-01-02", periods=periods, freq="B", tz="UTC")
    closes = {
        sym: pd.Series([100.0 * (1 + r) ** i for i in range(periods)], index=idx)
        for sym, r in rates.items()
    }
    return PanelData(closes=closes, options={}, fundamentals={},
                     symbols=tuple(sorted(closes)))


def _mom21() -> SignalSpec:
    from trading.alphasearch.spec import SIGNALS

    return SIGNALS["mom21"]


SIX = {"S1": -0.02, "S2": -0.01, "S3": 0.0, "S4": 0.01, "S5": 0.02, "S6": 0.03}


def test_assign_quantiles_terciles_of_six():
    scores = pd.Series({"S1": 1.0, "S2": 2.0, "S3": 3.0, "S4": 4.0,
                        "S5": 5.0, "S6": 6.0})
    top, bottom = assign_quantiles(scores, 3)
    assert set(top) == {"S5", "S6"}
    assert set(bottom) == {"S1", "S2"}


def test_assign_quantiles_uneven_extras_go_to_lower_buckets():
    scores = pd.Series({f"S{i}": float(i) for i in range(1, 8)})  # 7 names, q=3
    top, bottom = assign_quantiles(scores, 3)
    assert set(bottom) == {"S1", "S2", "S3"}  # 3-2-2 split, extras at the bottom
    assert set(top) == {"S6", "S7"}


def test_assign_quantiles_deterministic_tie_break():
    scores = pd.Series({"B": 1.0, "A": 1.0, "D": 1.0, "C": 1.0})
    top, bottom = assign_quantiles(scores, 2)
    assert bottom == ["A", "B"] and top == ["C", "D"]  # alphabetical on ties


def test_ls_and_lo_daily_returns_hand_computed():
    panel = _panel(SIX)
    dates = panel.decision_dates(panel.closes["S1"].index[30],
                                 panel.closes["S1"].index[-1])
    result = portfolio_sort(panel, _mom21(), dates, panel.closes["S1"].index[-1],
                            quantiles=3, tercile_below=0, min_names=3)
    # tercile_below=0 -> quantiles=3 everywhere; top={S5,S6}, bottom={S1,S2}.
    expected_lo = (0.02 + 0.03) / 2
    expected_ls = expected_lo - (-0.02 + -0.01) / 2
    assert len(result.ls) > 0
    assert all(math.isclose(v, expected_ls, rel_tol=1e-9) for v in result.ls)
    assert all(math.isclose(v, expected_lo, rel_tol=1e-9) for v in result.lo)
    # Constant rates -> identical top set every month -> zero turnover.
    assert result.turnover_monthly == 0.0
    assert result.skipped_dates == ()
    assert result.n_names_median == 6.0


def test_series_span_holding_periods_not_just_first_month():
    panel = _panel(SIX)
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[30], idx[-1])
    assert len(dates) >= 2
    result = portfolio_sort(panel, _mom21(), dates, idx[-1],
                            quantiles=3, tercile_below=0, min_names=3)
    # Daily series covers (first decision date, end]: every union trading day.
    expected_days = idx[(idx > dates[0]) & (idx <= idx[-1])]
    assert list(result.ls.index) == list(expected_days)
    assert result.n_dates == len(dates)


def test_thin_dates_are_skipped_and_recorded():
    # Only 2 symbols have 21 bars of history at the first decision date ->
    # mom21 is NaN for the rest -> cross-section 2 < min_names 3 -> skip.
    idx = pd.date_range("2020-01-02", periods=65, freq="B", tz="UTC")
    rich = {s: pd.Series([100.0 * (1 + r) ** i for i in range(65)], index=idx)
            for s, r in {"S1": 0.01, "S2": 0.02}.items()}
    poor = {s: pd.Series(100.0, index=idx[40:]) for s in ("S3", "S4", "S5", "S6")}
    panel = PanelData(closes={**rich, **poor}, options={}, fundamentals={},
                      symbols=tuple(sorted({**rich, **poor})))
    dates = panel.decision_dates(idx[25], idx[-1])
    result = portfolio_sort(panel, _mom21(), dates, idx[-1],
                            quantiles=3, tercile_below=0, min_names=3)
    assert len(result.skipped_dates) >= 1
    assert result.skipped_dates[0] == dates[0].date().isoformat()


def test_all_dates_skipped_raises_sort_error():
    panel = _panel(SIX)
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[30], idx[-1])
    with pytest.raises(SortError):
        portfolio_sort(panel, _mom21(), dates, idx[-1], min_names=15)  # only 6 names


def test_no_decision_dates_raises_sort_error():
    panel = _panel(SIX)
    with pytest.raises(SortError):
        portfolio_sort(panel, _mom21(), (), panel.closes["S1"].index[-1])


def test_turnover_hand_example():
    # Verify the turnover formula on a crafted signal that rotates the top set.
    calls = {"n": 0}

    def rotating(view, as_of):
        calls["n"] += 1
        base = {"S1": 1.0, "S2": 2.0, "S3": 3.0, "S4": 4.0, "S5": 5.0, "S6": 6.0}
        if calls["n"] == 2:  # second decision date: S6 drops to the bottom
            base["S6"] = 0.0
        return pd.Series(base, dtype="float64")

    panel = _panel(SIX)
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[30], idx[-1])[:2]
    spec = SignalSpec("rotating", rotating)
    result = portfolio_sort(panel, spec, dates, idx[-1],
                            quantiles=3, tercile_below=0, min_names=3)
    # tops: {S5,S6} then {S4,S5} -> overlap 1 of 2 -> one-way turnover 0.5.
    assert result.turnover_monthly == 0.5


def _skipping_spec(skip_calls: set[int],
                   rotate_calls: frozenset[int] = frozenset()) -> SignalSpec:
    """Signal that returns a thin (2-name) cross-section on the given calls.

    Base scores rank S1..S6 ascending -> top {S5,S6}, bottom {S1,S2}. On
    rotate_calls S6's score drops to 0 -> top {S4,S5}, bottom {S6,S1}.
    """
    calls = {"n": 0}

    def fn(view, as_of):
        calls["n"] += 1
        if calls["n"] in skip_calls:
            return pd.Series({"S1": 1.0, "S2": 2.0})  # 2 < min_names 3 -> skip
        base = {"S1": 1.0, "S2": 2.0, "S3": 3.0, "S4": 4.0, "S5": 5.0, "S6": 6.0}
        if calls["n"] in rotate_calls:
            base["S6"] = 0.0
        return pd.Series(base, dtype="float64")

    return SignalSpec("skipper", fn)


def test_skipped_middle_date_holds_prior_portfolio_no_gap():
    # A skipped date means "don't rebalance", never "delete the period": the
    # portfolio formed on dates[0] keeps accruing through the skipped dates[1]
    # period, and the daily series has no missing trading days.
    panel = _panel(SIX)
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[0], idx[-1])[:3]
    assert len(dates) == 3
    spec = _skipping_spec(skip_calls={2}, rotate_calls=frozenset({3}))
    result = portfolio_sort(panel, spec, dates, idx[-1],
                            quantiles=3, tercile_below=0, min_names=3)
    # Continuous: every union trading day from the first rebalance to end.
    expected_days = idx[(idx > dates[0]) & (idx <= idx[-1])]
    assert list(result.ls.index) == list(expected_days)
    assert result.skipped_dates == (dates[1].date().isoformat(),)
    # (dates[1], dates[2]]: prior portfolio {S5,S6}/{S1,S2} still held.
    held = result.ls[(result.ls.index > dates[1]) & (result.ls.index <= dates[2])]
    expected_held = (0.02 + 0.03) / 2 - (-0.02 + -0.01) / 2
    assert len(held) > 0
    assert all(math.isclose(v, expected_held, rel_tol=1e-9) for v in held)
    # After dates[2]: rotated portfolio top {S4,S5}, bottom {S6,S1}.
    after = result.ls[result.ls.index > dates[2]]
    expected_after = (0.01 + 0.02) / 2 - (0.03 + -0.02) / 2
    assert len(after) > 0
    assert all(math.isclose(v, expected_after, rel_tol=1e-9) for v in after)


def test_leading_skipped_date_series_starts_at_first_rebalance():
    # No portfolio exists before the first ACTUAL rebalance, so days after a
    # leading skipped date contribute nothing.
    panel = _panel(SIX)
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[0], idx[-1])[:2]
    spec = _skipping_spec(skip_calls={1})
    result = portfolio_sort(panel, spec, dates, idx[-1],
                            quantiles=3, tercile_below=0, min_names=3)
    expected_days = idx[(idx > dates[1]) & (idx <= idx[-1])]
    assert list(result.ls.index) == list(expected_days)
    assert result.skipped_dates == (dates[0].date().isoformat(),)
    assert math.isnan(result.turnover_monthly)  # single actual rebalance


def _degenerate_spec(tied_calls: set[int]) -> SignalSpec:
    """Signal returning an all-tied (degenerate) cross-section on the given
    calls -- plenty of names (>= min_names), but only ONE distinct score, so
    quantile buckets can't be formed without assign_quantiles' alphabetical
    tie-break becoming the entire "signal" (e.g. ind_mom on a single-sector
    segment universe). Other calls return 6 distinct scores (S1..S6 rising)."""
    calls = {"n": 0}

    def fn(view, as_of):
        calls["n"] += 1
        if calls["n"] in tied_calls:
            return pd.Series({s: 1.0 for s in SIX})
        return pd.Series({s: float(i) for i, s in enumerate(SIX)})

    return SignalSpec("tied", fn)


def test_degenerate_tied_cross_section_is_skipped_like_a_thin_date():
    # All 6 names score IDENTICALLY on the middle date: plenty of names, but
    # zero distinct scores. Skipped via the same machinery as the
    # <min_names rule (spec section 5.4 extension) -- never a junk trial
    # from assign_quantiles' alphabetical tie-break.
    panel = _panel(SIX)
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[0], idx[-1])[:3]
    assert len(dates) == 3
    spec = _degenerate_spec(tied_calls={2})
    result = portfolio_sort(panel, spec, dates, idx[-1],
                            quantiles=3, tercile_below=0, min_names=3)
    assert result.skipped_dates == (dates[1].date().isoformat(),)
    # "Skipped" means "don't rebalance": the portfolio formed on dates[0]
    # keeps holding through the degenerate dates[1] period.
    held = result.ls[(result.ls.index > dates[1]) & (result.ls.index <= dates[2])]
    assert len(held) > 0


def test_all_tied_cross_sections_raise_sort_error():
    # Every date is degenerate (single-sector ind_mom over the whole window)
    # -> no portfolio ever forms -> the existing "every date skipped"
    # SortError fires and journals an honest error trial.
    panel = _panel(SIX)
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[30], idx[-1])
    spec = SignalSpec("alltied", lambda view, as_of: pd.Series({s: 1.0 for s in SIX}))
    with pytest.raises(SortError):
        portfolio_sort(panel, spec, dates, idx[-1],
                       quantiles=3, tercile_below=0, min_names=3)


def test_partial_ties_with_enough_distinct_values_are_not_skipped():
    # 6 names, exactly 3 distinct scores (two pairs of ties) -- NOT
    # degenerate, since 3 distinct values exactly covers the 3 quantile
    # buckets in use. A normal mixed date is unaffected by the new guard.
    panel = _panel(SIX)
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[30], idx[-1])
    spec = SignalSpec(
        "partial_tie",
        lambda view, as_of: pd.Series({"S1": 1.0, "S2": 1.0, "S3": 2.0,
                                       "S4": 2.0, "S5": 3.0, "S6": 3.0}),
    )
    result = portfolio_sort(panel, spec, dates, idx[-1],
                            quantiles=3, tercile_below=0, min_names=3)
    assert result.skipped_dates == ()


def test_turnover_across_skip_uses_actual_rebalances_only():
    # Turnover pairs consecutive ACTUAL rebalances; the skipped middle date
    # neither contributes a pair nor breaks the pairing across the gap.
    panel = _panel(SIX)
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[0], idx[-1])[:3]
    spec = _skipping_spec(skip_calls={2}, rotate_calls=frozenset({3}))
    result = portfolio_sort(panel, spec, dates, idx[-1],
                            quantiles=3, tercile_below=0, min_names=3)
    # tops: {S5,S6} (dates[0]) then {S4,S5} (dates[2]) -> one pair -> 0.5.
    assert result.turnover_monthly == 0.5
    assert result.n_names_median == 6.0
