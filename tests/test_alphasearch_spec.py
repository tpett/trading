"""Seed signal registry: hand-checked scores on tiny deterministic panels."""

from __future__ import annotations

import math

import pandas as pd

from trading.alphasearch.panel import PanelData
from trading.alphasearch.spec import SIGNALS


def _geometric_panel(n_days: int = 300, rates: dict[str, float] | None = None) -> PanelData:
    """Each symbol's close grows at a constant daily rate -> every price
    metric has a closed-form value."""
    rates = rates or {"SLOW": 0.001, "FAST": 0.01}
    idx = pd.date_range("2020-01-02", periods=n_days, freq="B", tz="UTC")
    closes = {
        sym: pd.Series([100.0 * (1 + r) ** i for i in range(n_days)], index=idx)
        for sym, r in rates.items()
    }
    return PanelData(closes=closes, options={}, fundamentals={},
                     symbols=tuple(sorted(closes)))


def _score(name: str, panel: PanelData, as_of: pd.Timestamp) -> pd.Series:
    spec = SIGNALS[name]
    return spec.fn(panel.view(as_of), as_of)


def test_price_signals_registered_with_no_data_requirements():
    for name in ("mom21", "mom63", "mom126", "mom252", "rev5", "rvol21", "disthigh"):
        spec = SIGNALS[name]
        assert spec.name == name
        assert not spec.requires_options
        assert not spec.requires_fundamentals


def test_mom21_closed_form_and_ordering():
    panel = _geometric_panel()
    as_of = panel.closes["SLOW"].index[-1]
    scores = _score("mom21", panel, as_of)
    assert math.isclose(scores["SLOW"], 1.001**21 - 1, rel_tol=1e-9)
    assert math.isclose(scores["FAST"], 1.01**21 - 1, rel_tol=1e-9)
    assert scores["FAST"] > scores["SLOW"]  # higher momentum = higher score


def test_rev5_is_negated_trailing_return():
    panel = _geometric_panel()
    as_of = panel.closes["SLOW"].index[-1]
    scores = _score("rev5", panel, as_of)
    assert math.isclose(scores["FAST"], -(1.01**5 - 1), rel_tol=1e-9)
    assert scores["FAST"] < scores["SLOW"]  # recent winners UNattractive


def test_rvol21_is_negated_and_zero_for_constant_growth():
    # Constant growth rate -> identical daily returns -> zero realized vol.
    panel = _geometric_panel()
    as_of = panel.closes["SLOW"].index[-1]
    scores = _score("rvol21", panel, as_of)
    assert scores["SLOW"] == 0.0
    assert scores["FAST"] == 0.0


def test_disthigh_zero_at_high_negative_below():
    idx = pd.date_range("2020-01-02", periods=30, freq="B", tz="UTC")
    rising = pd.Series([100.0 + i for i in range(30)], index=idx)
    dipped = pd.Series([100.0 + i for i in range(29)] + [64.5], index=idx)
    panel = PanelData(closes={"UP": rising, "DIP": dipped}, options={},
                      fundamentals={}, symbols=("DIP", "UP"))
    scores = _score("disthigh", panel, idx[-1])
    assert scores["UP"] == 0.0  # at its high
    assert math.isclose(scores["DIP"], 64.5 / 128.0 - 1, rel_tol=1e-9)
    assert scores["UP"] > scores["DIP"]  # near-high names attractive


def test_insufficient_history_yields_nan_not_crash():
    panel = _geometric_panel(n_days=10)
    as_of = panel.closes["SLOW"].index[-1]
    assert math.isnan(_score("mom21", panel, as_of)["SLOW"])
    assert math.isnan(_score("rvol21", panel, as_of)["SLOW"])
    # mom{21..252} on 10 bars: all NaN; rev5 still computable (needs 6 bars).
    assert not math.isnan(_score("rev5", panel, as_of)["SLOW"])


def test_scores_cover_every_panel_symbol():
    panel = _geometric_panel()
    as_of = panel.closes["SLOW"].index[-1]
    for name in ("mom21", "rev5", "rvol21", "disthigh"):
        scores = _score(name, panel, as_of)
        assert list(scores.index) == list(panel.symbols)
        assert scores.dtype == "float64"
