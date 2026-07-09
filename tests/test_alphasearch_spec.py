"""Seed signal registry: hand-checked scores on tiny deterministic panels."""

from __future__ import annotations

import math

import pandas as pd

from trading.alphasearch.panel import PanelData, options_from_cells
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
    # Float compounding of (1+r)**i makes the rolling std ~1e-16, not exact zero.
    assert math.isclose(scores["SLOW"], 0.0, abs_tol=1e-9)
    assert math.isclose(scores["FAST"], 0.0, abs_tol=1e-9)


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


# --------------------------------------------------------------------------- #
# Options + fundamentals families (Task 5)
# --------------------------------------------------------------------------- #


def _options_panel() -> tuple[PanelData, pd.Timestamp]:
    idx = pd.date_range("2020-01-02", periods=60, freq="B", tz="UTC")
    closes = {s: pd.Series(100.0, index=idx) for s in ("AAA", "BBB")}
    as_of = idx[-1]
    date = as_of.date().isoformat()
    cells = [
        {"symbol": "AAA", "decision_date": date, "skew_put_atm": 0.05,
         "skew_put_call": 0.02, "contracts": [
             {"role": "atm", "bid": 4.0, "ask": 4.2, "mid": 4.1, "iv": 0.30,
              "volume": 100},
             {"role": "otm_put", "iv": 0.34, "volume": 50},
             {"role": "otm_call", "iv": 0.28, "volume": 25}]},
        {"symbol": "BBB", "decision_date": date, "skew_put_atm": 0.10,
         "skew_put_call": -0.01, "contracts": [
             {"role": "atm", "bid": 2.0, "ask": 2.4, "mid": 2.2, "iv": 0.50,
              "volume": 10},
             {"role": "otm_put", "iv": 0.60, "volume": 5},
             {"role": "otm_call", "iv": 0.44, "volume": 2}]},
    ]
    panel = PanelData(closes=closes, options=options_from_cells(cells),
                      fundamentals={}, symbols=("AAA", "BBB"))
    return panel, as_of


def test_options_family_signs():
    panel, as_of = _options_panel()
    hedge = _score("hedge", panel, as_of)
    assert hedge["AAA"] == -0.05 and hedge["BBB"] == -0.10  # -skew_put_atm
    assert hedge["AAA"] > hedge["BBB"]  # less-hedged name is more attractive
    excite = _score("excite", panel, as_of)
    assert excite["AAA"] == -0.02 and excite["BBB"] == 0.01
    atm_iv = _score("atm_iv", panel, as_of)
    assert atm_iv["AAA"] == -0.30 and atm_iv["BBB"] == -0.50
    smile = _score("smile", panel, as_of)
    assert math.isclose(smile["AAA"], -((0.34 + 0.28) / 2 - 0.30), rel_tol=1e-12)
    spread = _score("atm_spread", panel, as_of)
    assert math.isclose(spread["AAA"], (4.2 - 4.0) / 4.1, rel_tol=1e-12)
    assert spread["BBB"] > spread["AAA"]  # wider spread = higher score


def test_vrp_is_iv_minus_realized_vol():
    panel, as_of = _options_panel()
    # Flat closes -> realized vol 0 -> vrp == atm_iv.
    vrp = _score("vrp", panel, as_of)
    assert math.isclose(vrp["AAA"], 0.30, rel_tol=1e-12)
    assert math.isclose(vrp["BBB"], 0.50, rel_tol=1e-12)


def test_options_signals_nan_without_cells():
    panel, as_of = _options_panel()
    bare = PanelData(closes=panel.closes, options={}, fundamentals={},
                     symbols=panel.symbols)
    for name in ("hedge", "excite", "atm_iv", "smile", "atm_spread", "vrp"):
        assert _score(name, bare, as_of).isna().all()


def test_fundamentals_family_values_and_neutrality():
    idx = pd.date_range("2020-01-02", periods=10, freq="B", tz="UTC")
    closes = {"AAA": pd.Series(50.0, index=idx), "BBB": pd.Series(50.0, index=idx)}
    filed = pd.DatetimeIndex([idx[0]])
    fund = {"AAA": pd.DataFrame(
        {"gross_profitability": [0.4], "ttm_net_income": [5e6],
         "book_equity": [2.5e7], "shares_outstanding": [1e6]}, index=filed)}
    panel = PanelData(closes=closes, options={}, fundamentals=fund,
                      symbols=("AAA", "BBB"))
    as_of = idx[-1]
    gp = _score("gross_profitability", panel, as_of)
    assert gp["AAA"] == 0.4
    assert math.isnan(gp["BBB"])  # no filing -> NaN (dropped, not imputed)
    ey = _score("earnings_yield", panel, as_of)
    assert math.isclose(ey["AAA"], 5e6 / (1e6 * 50.0), rel_tol=1e-12)
    bm = _score("book_to_market", panel, as_of)
    assert math.isclose(bm["AAA"], 2.5e7 / (1e6 * 50.0), rel_tol=1e-12)


def test_registry_is_complete_with_correct_requirements():
    assert len(SIGNALS) == 35  # 16 seeds + 9 price/volume + 5 options + 5 fundamentals
    options_family = {"vrp", "hedge", "excite", "atm_iv", "smile", "atm_spread",
                      "cp_vol", "osv", "otm_put_iv", "iv_change", "dskew"}
    volume_family = {"cp_vol", "osv"}
    fundamentals_family = {
        "gross_profitability", "earnings_yield", "book_to_market",
        "asset_growth", "net_issuance", "roa", "droa", "rev_growth",
    }
    for name, spec in SIGNALS.items():
        assert spec.requires_options == (name in options_family)
        assert spec.requires_option_volume == (name in volume_family)
        assert spec.requires_fundamentals == (name in fundamentals_family)
