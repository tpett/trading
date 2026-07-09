"""Tier-1 signal batch: hand-computed unit fixtures per signal, including the
pre-registered sign conventions (spec 2026-07-09-tier1-signal-batch-design)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from alphasearch_helpers import make_cell
from trading.alphasearch.panel import PanelData, options_from_cells
from trading.alphasearch.spec import SIGNALS


def _score(name: str, panel: PanelData, as_of: pd.Timestamp) -> pd.Series:
    return SIGNALS[name].fn(panel.view(as_of), as_of)


def _bar_frame(
    close: pd.Series,
    *,
    open_: pd.Series | None = None,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    volume: float | pd.Series = 1000.0,
    div_cash: float | pd.Series = 0.0,
    split_factor: float | pd.Series = 1.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": close if open_ is None else open_,
         "high": close if high is None else high,
         "low": close if low is None else low,
         "close": close, "volume": volume, "div_cash": div_cash,
         "split_factor": split_factor},
        index=close.index,
    )


def _bar_panel(frames: dict[str, pd.DataFrame], **kwargs) -> PanelData:
    return PanelData(
        closes={s: f["close"] for s, f in frames.items()},
        bars=frames, symbols=tuple(sorted(frames)), **kwargs,
    )


def _geometric_close(rate: float, n: int, start: str = "2019-01-02") -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    return pd.Series([100.0 * (1 + rate) ** i for i in range(n)], index=idx)


# --------------------------------------------------------------------------- #
# Price/volume family
# --------------------------------------------------------------------------- #


def test_mom_12_2_skips_the_most_recent_month():
    frames = {"SLOW": _bar_frame(_geometric_close(0.001, 300)),
              "FAST": _bar_frame(_geometric_close(0.01, 300))}
    panel = _bar_panel(frames)
    as_of = frames["SLOW"].index[-1]
    scores = _score("mom_12_2", panel, as_of)
    # close[p-21]/close[p-252] - 1 = (1+r)^231 - 1
    assert math.isclose(scores["SLOW"], 1.001**231 - 1, rel_tol=1e-9)
    assert math.isclose(scores["FAST"], 1.01**231 - 1, rel_tol=1e-9)
    assert scores["FAST"] > scores["SLOW"]  # + sign: winners attractive


def test_mom_12_2_nan_under_253_closes():
    frames = {"AAA": _bar_frame(_geometric_close(0.001, 252))}
    panel = _bar_panel(frames)
    assert math.isnan(_score("mom_12_2", panel, frames["AAA"].index[-1])["AAA"])


def test_overnight_sums_63_log_gaps():
    idx = pd.date_range("2020-01-02", periods=70, freq="B", tz="UTC")
    close = pd.Series(100.0, index=idx)
    frames = {
        "HOT": _bar_frame(close, open_=pd.Series(100.0 * math.exp(0.002), index=idx)),
        "COLD": _bar_frame(close, open_=pd.Series(100.0 * math.exp(0.0005), index=idx)),
    }
    panel = _bar_panel(frames)
    scores = _score("overnight", panel, idx[-1])
    assert math.isclose(scores["HOT"], 63 * 0.002, rel_tol=1e-9)
    assert math.isclose(scores["COLD"], 63 * 0.0005, rel_tol=1e-9)
    assert scores["HOT"] > scores["COLD"]  # + sign: overnight persistence


def test_overnight_nan_under_64_bars():
    idx = pd.date_range("2020-01-02", periods=63, freq="B", tz="UTC")
    frames = {"AAA": _bar_frame(pd.Series(100.0, index=idx))}
    assert math.isnan(_score("overnight", _bar_panel(frames), idx[-1])["AAA"])


def test_park_vol_closed_form_and_negated():
    idx = pd.date_range("2020-01-02", periods=30, freq="B", tz="UTC")
    close = pd.Series(100.0, index=idx)

    def with_range(log_range: float) -> pd.DataFrame:
        return _bar_frame(
            close,
            high=pd.Series(100.0 * math.exp(log_range), index=idx),
            low=pd.Series(100.0, index=idx),
        )

    panel = _bar_panel({"WILD": with_range(0.04), "TAME": with_range(0.01)})
    scores = _score("park_vol", panel, idx[-1])
    expected_wild = math.sqrt(0.04**2 / (4 * math.log(2))) * math.sqrt(252)
    assert math.isclose(scores["WILD"], -expected_wild, rel_tol=1e-9)
    assert scores["TAME"] > scores["WILD"]  # - sign: quiet names attractive


def test_max5_is_negated_mean_of_top_returns():
    rets = [0.001] * 16 + [0.05, 0.04, 0.03, 0.02, 0.01]  # 21 returns
    closes = [100.0]
    for r in rets:
        closes.append(closes[-1] * (1 + r))
    idx = pd.date_range("2020-01-02", periods=22, freq="B", tz="UTC")
    frames = {"LOTTO": _bar_frame(pd.Series(closes, index=idx)),
              "STEADY": _bar_frame(_geometric_close(0.001, 22, start="2020-01-02"))}
    panel = _bar_panel(frames)
    scores = _score("max5", panel, idx[-1])
    assert math.isclose(scores["LOTTO"], -(0.05 + 0.04 + 0.03 + 0.02 + 0.01) / 5,
                        rel_tol=1e-9)
    assert scores["STEADY"] > scores["LOTTO"]  # - sign: lottery names penalized


def test_ivol_and_beta_signals_negate_the_precomputed_feature():
    idx = pd.date_range("2020-01-06", periods=3, freq="B", tz="UTC")
    feats = {"AAA": pd.DataFrame({"ivol": [0.2, 0.2, 0.2],
                                  "beta": [1.5, 1.5, 1.5]}, index=idx)}
    panel = PanelData(closes={}, features=feats, symbols=("AAA", "BBB"))
    as_of = idx[-1]
    ivol = _score("ivol", panel, as_of)
    assert ivol["AAA"] == -0.2          # - sign: idio-vol puzzle
    assert math.isnan(ivol["BBB"])      # no features -> NaN, dropped
    beta = _score("beta", panel, as_of)
    assert beta["AAA"] == -1.5          # - sign: betting-against-beta


def test_amihud_constant_dollar_volume_closed_form_and_floor():
    n = 260
    close = _geometric_close(0.01, n)
    volume = 1e6 / close                 # constant dollar volume 1e6
    frames = {"AAA": _bar_frame(close, volume=volume)}
    panel = _bar_panel(frames)
    got = _score("amihud", panel, close.index[-1])["AAA"]
    assert math.isclose(got, 0.01 / 1e6, rel_tol=1e-9)  # |r|/D every day
    short = _geometric_close(0.01, 120)
    thin = _bar_panel({"AAA": _bar_frame(short, volume=1e6 / short)})
    assert math.isnan(_score("amihud", thin, short.index[-1])["AAA"])  # <126 obs


def test_vol_trend_ratio_of_dollar_volume_means():
    idx = pd.date_range("2019-01-02", periods=252, freq="B", tz="UTC")
    close = pd.Series(100.0, index=idx)
    volume = pd.Series(1000.0, index=idx)
    volume.iloc[-21:] = 2000.0
    frames = {"AAA": _bar_frame(close, volume=volume)}
    got = _score("vol_trend", _bar_panel(frames), idx[-1])["AAA"]
    base = (231 * 100.0 * 1000.0 + 21 * 100.0 * 2000.0) / 252
    assert math.isclose(got, (100.0 * 2000.0) / base, rel_tol=1e-12)
    short = _bar_panel({"AAA": _bar_frame(pd.Series(100.0, index=idx[:200]))})
    assert math.isnan(_score("vol_trend", short, idx[199])["AAA"])


def test_div_yield_sums_trailing_dividends_and_never_fabricates_zero():
    idx = pd.date_range("2019-01-02", periods=260, freq="B", tz="UTC")
    div = pd.Series(0.0, index=idx)
    div.iloc[-100] = 1.0
    div.iloc[-10] = 0.5
    frames = {"PAYER": _bar_frame(pd.Series(50.0, index=idx), div_cash=div),
              "LEGACY": _bar_frame(pd.Series(50.0, index=idx),
                                   div_cash=pd.Series(np.nan, index=idx))}
    scores = _score("div_yield", _bar_panel(frames), idx[-1])
    assert math.isclose(scores["PAYER"], 1.5 / 50.0, rel_tol=1e-12)
    # Legacy narrow cache (all-NaN div_cash): NaN, never sum()'s skipna 0.0.
    assert math.isnan(scores["LEGACY"])


def test_price_volume_family_covers_every_panel_symbol():
    frames = {"AAA": _bar_frame(_geometric_close(0.001, 300)),
              "BBB": _bar_frame(_geometric_close(0.002, 300))}
    panel = _bar_panel(frames)
    as_of = frames["AAA"].index[-1]
    for name in ("mom_12_2", "overnight", "park_vol", "max5", "amihud",
                 "vol_trend", "div_yield"):
        scores = _score(name, panel, as_of)
        assert list(scores.index) == list(panel.symbols)
        assert scores.dtype == "float64"


# --------------------------------------------------------------------------- #
# Options family
# --------------------------------------------------------------------------- #


def _options_tier1_panel(prior_age_days: int = 28):
    idx = pd.date_range("2020-01-02", periods=60, freq="B", tz="UTC")
    as_of = idx[-1]
    prior = (as_of - pd.Timedelta(prior_age_days, unit="D")).date().isoformat()
    current = as_of.date().isoformat()
    cells = [
        make_cell("AAA", prior, atm_iv=0.24, skew_put_atm=0.02),
        make_cell("AAA", current, atm_iv=0.30, skew_put_atm=0.05),
        make_cell("BBB", current, atm_iv=0.50, put_iv=0.50,
                  skew_put_atm=0.10),  # no prior; steeper smirk
    ]
    bars = {s: _bar_frame(pd.Series(100.0, index=idx), volume=500.0)
            for s in ("AAA", "BBB")}
    panel = PanelData(
        closes={s: f["close"] for s, f in bars.items()}, bars=bars,
        options=options_from_cells(cells), symbols=("AAA", "BBB"),
        has_option_volume=True,
    )
    return panel, as_of


def test_cp_vol_reads_the_committed_call_minus_put_log_volume():
    panel, as_of = _options_tier1_panel()
    scores = _score("cp_vol", panel, as_of)
    # ATM leg is a call: call side = atm(100) + otm_call(25); put side = 50.
    assert math.isclose(scores["AAA"], math.log(126 / 51), rel_tol=1e-12)
    assert scores["AAA"] > 0  # + sign: informed call demand attractive


def test_osv_is_negated_option_to_stock_dollar_volume():
    panel, as_of = _options_tier1_panel()
    scores = _score("osv", panel, as_of)
    opt_dollar = 100 * 100 * 4.1 + 50 * 100 * 2.0 + 25 * 100 * 1.5  # 54750
    assert math.isclose(scores["AAA"], -(opt_dollar / (100.0 * 500.0)),
                        rel_tol=1e-12)


def test_otm_put_iv_is_negated_smirk_level():
    panel, as_of = _options_tier1_panel()
    scores = _score("otm_put_iv", panel, as_of)
    assert scores["AAA"] == -0.34
    assert scores["AAA"] > scores["BBB"]  # steeper smirk = less attractive


def test_iv_change_and_dskew_are_negated_innovations_nan_without_prior():
    panel, as_of = _options_tier1_panel()
    iv_change = _score("iv_change", panel, as_of)
    assert math.isclose(iv_change["AAA"], -(0.30 - 0.24), rel_tol=1e-12)
    assert math.isnan(iv_change["BBB"])  # no prior cell -> NaN
    dskew = _score("dskew", panel, as_of)
    assert math.isclose(dskew["AAA"], -(0.05 - 0.02), rel_tol=1e-12)
    assert math.isnan(dskew["BBB"])


def test_innovations_nan_when_the_prior_cell_is_stale():
    panel, as_of = _options_tier1_panel(prior_age_days=50)  # > 45d cap
    assert math.isnan(_score("iv_change", panel, as_of)["AAA"])
    assert math.isnan(_score("dskew", panel, as_of)["AAA"])


def test_options_family_nan_without_cells():
    idx = pd.date_range("2020-01-02", periods=60, freq="B", tz="UTC")
    bars = {"AAA": _bar_frame(pd.Series(100.0, index=idx), volume=500.0)}
    bare = _bar_panel(bars)
    for name in ("cp_vol", "osv", "otm_put_iv", "iv_change", "dskew"):
        assert _score(name, bare, idx[-1]).isna().all(), name


# --------------------------------------------------------------------------- #
# Fundamentals family (300-calendar-day YoY filing rule)
# --------------------------------------------------------------------------- #

FILED_2019 = pd.Timestamp("2019-01-10", tz="UTC")
FILED_2019_Q4 = pd.Timestamp("2019-11-30", tz="UTC")  # 324d later: YoY-eligible


def _fund_frame(values_by_column: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        values_by_column, index=pd.DatetimeIndex([FILED_2019, FILED_2019_Q4])
    )


def _fund_panel(fundamentals: dict[str, pd.DataFrame],
                 split_factor: pd.Series | float = 1.0) -> tuple[PanelData, pd.Timestamp]:
    idx = pd.date_range("2019-01-02", periods=300, freq="B", tz="UTC")
    bars = {s: _bar_frame(pd.Series(100.0, index=idx), split_factor=split_factor)
            for s in fundamentals}
    panel = PanelData(
        closes={s: f["close"] for s, f in bars.items()}, bars=bars,
        fundamentals=fundamentals, symbols=tuple(sorted(fundamentals)),
    )
    return panel, pd.Timestamp("2020-01-15", tz="UTC")


def test_asset_growth_rev_growth_roa_droa_hand_values():
    fund = {
        "AAA": _fund_frame({
            "assets": [100.0, 110.0],
            "revenue_ttm": [200.0, 260.0],
            "ttm_net_income": [8.0, 13.2],
            "shares_outstanding": [1e6, 1e6],
        }),
        # Single YoY-ineligible filer: everything YoY-based is NaN.
        "BBB": pd.DataFrame(
            {"assets": [50.0], "revenue_ttm": [10.0], "ttm_net_income": [5.0],
             "shares_outstanding": [1e6]},
            index=pd.DatetimeIndex([FILED_2019_Q4]),
        ),
    }
    panel, as_of = _fund_panel(fund)
    ag = _score("asset_growth", panel, as_of)
    assert math.isclose(ag["AAA"], -(110.0 / 100.0 - 1), rel_tol=1e-12)  # negated
    assert math.isnan(ag["BBB"])
    rg = _score("rev_growth", panel, as_of)
    assert math.isclose(rg["AAA"], 260.0 / 200.0 - 1, rel_tol=1e-12)
    roa = _score("roa", panel, as_of)
    assert math.isclose(roa["AAA"], 13.2 / 110.0, rel_tol=1e-12)
    assert math.isclose(roa["BBB"], 5.0 / 50.0, rel_tol=1e-12)  # roa needs no prior
    droa = _score("droa", panel, as_of)
    assert math.isclose(droa["AAA"], 13.2 / 110.0 - 8.0 / 100.0, rel_tol=1e-12)
    assert math.isnan(droa["BBB"])


def test_net_issuance_is_split_adjusted_and_negated():
    fund = {"AAA": _fund_frame({
        "assets": [100.0, 110.0], "revenue_ttm": [200.0, 260.0],
        "ttm_net_income": [8.0, 13.2],
        "shares_outstanding": [1e6, 2.1e6],   # 2:1 split + 5% true issuance
    })}
    idx = pd.date_range("2019-01-02", periods=300, freq="B", tz="UTC")
    split = pd.Series(1.0, index=idx)
    split.loc[pd.Timestamp("2019-06-03", tz="UTC")] = 2.0  # between the filings
    panel, as_of = _fund_panel(fund, split_factor=split)
    got = _score("net_issuance", panel, as_of)["AAA"]
    assert math.isclose(got, -(2.1e6 / (1e6 * 2.0) - 1), rel_tol=1e-12)  # -0.05


def test_net_issuance_nan_when_split_history_is_unknown():
    fund = {"AAA": _fund_frame({
        "assets": [100.0, 110.0], "revenue_ttm": [200.0, 260.0],
        "ttm_net_income": [8.0, 13.2], "shares_outstanding": [1e6, 1.05e6],
    })}
    panel, as_of = _fund_panel(fund, split_factor=float("nan"))  # legacy cache
    assert math.isnan(_score("net_issuance", panel, as_of)["AAA"])


def test_fundamentals_family_nan_without_a_store():
    panel, as_of = _fund_panel({"AAA": _fund_frame({
        "assets": [100.0, 110.0], "revenue_ttm": [200.0, 260.0],
        "ttm_net_income": [8.0, 13.2], "shares_outstanding": [1e6, 1e6],
    })})
    bare = PanelData(closes=panel.closes, bars=panel.bars, symbols=panel.symbols)
    for name in ("asset_growth", "net_issuance", "roa", "droa", "rev_growth"):
        assert _score(name, bare, as_of).isna().all()


# --------------------------------------------------------------------------- #
# Industry-relative family (10 frozen SEGMENTS sectors via sic_map)
# --------------------------------------------------------------------------- #


def _sector_panel() -> tuple[PanelData, pd.Timestamp]:
    frames = {
        "FIN1": _bar_frame(_geometric_close(0.001, 300)),
        "FIN2": _bar_frame(_geometric_close(0.003, 300)),
        "TRD1": _bar_frame(_geometric_close(0.002, 300)),
        "UNMAPPED": _bar_frame(_geometric_close(0.004, 300)),
    }
    panel = _bar_panel(
        frames,
        sectors={"FIN1": "finance", "FIN2": "finance", "TRD1": "trade"},
    )
    return panel, frames["FIN1"].index[-1]


def _mom(rate: float) -> float:
    return (1 + rate) ** 231 - 1


def _trail21(rate: float) -> float:
    return (1 + rate) ** 21 - 1


def test_ind_mom_assigns_the_sector_mean_and_nan_to_unmapped():
    panel, as_of = _sector_panel()
    scores = _score("ind_mom", panel, as_of)
    finance_mean = (_mom(0.001) + _mom(0.003)) / 2
    assert math.isclose(scores["FIN1"], finance_mean, rel_tol=1e-9)
    assert math.isclose(scores["FIN2"], finance_mean, rel_tol=1e-9)
    assert math.isclose(scores["TRD1"], _mom(0.002), rel_tol=1e-9)
    assert math.isnan(scores["UNMAPPED"])  # never guessed


def test_ind_rel_rev_rewards_within_sector_laggards():
    panel, as_of = _sector_panel()
    scores = _score("ind_rel_rev", panel, as_of)
    finance_mean = (_trail21(0.001) + _trail21(0.003)) / 2
    assert math.isclose(scores["FIN1"], -(_trail21(0.001) - finance_mean),
                        rel_tol=1e-9)
    assert scores["FIN1"] > 0 > scores["FIN2"]  # laggard attractive, leader not
    # A one-member sector sits exactly at its own mean.
    assert math.isclose(scores["TRD1"], 0.0, abs_tol=1e-12)
    assert math.isnan(scores["UNMAPPED"])


def test_sector_stats_use_only_the_dates_cross_section():
    # A finance member with too little history for mom_12_2 contributes
    # NOTHING to the sector mean, but (mapped) still receives ind_mom's mean.
    frames = {
        "FIN1": _bar_frame(_geometric_close(0.001, 300)),
        "FIN2": _bar_frame(_geometric_close(0.003, 300)),
        "FINYOUNG": _bar_frame(_geometric_close(0.05, 30, start="2020-01-02")),
    }
    sectors = {s: "finance" for s in frames}
    panel = _bar_panel(frames, sectors=sectors)
    as_of = frames["FIN1"].index[-1]
    scores = _score("ind_mom", panel, as_of)
    finance_mean = (_mom(0.001) + _mom(0.003)) / 2  # FINYOUNG's NaN excluded
    assert math.isclose(scores["FIN1"], finance_mean, rel_tol=1e-9)
    assert math.isclose(scores["FINYOUNG"], finance_mean, rel_tol=1e-9)
    # ind_rel_rev needs the symbol's OWN trail21 too.
    rel = _score("ind_rel_rev", panel, as_of)
    assert not math.isnan(rel["FINYOUNG"])  # 30 bars >= 22: trail21 exists
