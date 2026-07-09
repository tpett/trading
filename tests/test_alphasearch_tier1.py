"""Tier-1 signal batch: hand-computed unit fixtures per signal, including the
pre-registered sign conventions (spec 2026-07-09-tier1-signal-batch-design)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trading.alphasearch.panel import PanelData
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
