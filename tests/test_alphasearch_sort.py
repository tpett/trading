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
