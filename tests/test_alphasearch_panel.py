"""PanelData/PanelView: PIT truncation and the monthly decision calendar."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from alphasearch_helpers import make_cell
from trading.alphasearch.panel import (
    BAR_COLUMNS,
    MAX_OPTION_AGE_DAYS,
    OPTION_COLUMNS,
    ROLLING_FEATURES,
    PanelData,
    PanelError,
    build_panel,
    cell_metrics,
    load_bars,
    load_closes,
    load_options,
    options_from_cells,
)


def _closes(dates: list[str], values: list[float]) -> pd.Series:
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dates])
    return pd.Series(values, index=idx, dtype="float64", name="close")


def _panel(closes: dict[str, pd.Series]) -> PanelData:
    return PanelData(
        closes=closes, options={}, fundamentals={}, symbols=tuple(sorted(closes))
    )


def test_fundamentals_row_prior_300_day_boundary():
    base = pd.Timestamp("2019-01-10", tz="UTC")
    as_of = pd.Timestamp("2020-01-15", tz="UTC")
    ok = pd.DataFrame(
        {"assets": [100.0, 110.0]},
        index=pd.DatetimeIndex([base, base + pd.Timedelta(300, unit="D")]),
    )
    panel = PanelData(closes={}, fundamentals={"AAA": ok}, symbols=("AAA",))
    prior = panel.view(as_of).fundamentals_row_prior("AAA")
    assert prior is not None and prior["assets"] == 100.0  # exactly 300d: counts
    close_call = pd.DataFrame(
        {"assets": [100.0, 110.0]},
        index=pd.DatetimeIndex([base, base + pd.Timedelta(299, unit="D")]),
    )
    panel299 = PanelData(closes={}, fundamentals={"AAA": close_call}, symbols=("AAA",))
    assert panel299.view(as_of).fundamentals_row_prior("AAA") is None  # 299d: no
    # The rule anchors on the CURRENT filing: before the second filing is
    # visible, the first IS current and has no prior.
    early = base + pd.Timedelta(10, unit="D")
    assert panel.view(early).fundamentals_row_prior("AAA") is None
    assert panel.view(as_of).fundamentals_row_prior("NOPE") is None


def test_view_truncates_closes_at_as_of():
    s = _closes(["2020-01-02", "2020-01-03", "2020-01-06"], [1.0, 2.0, 3.0])
    panel = _panel({"AAA": s})
    view = panel.view(pd.Timestamp("2020-01-03", tz="UTC"))
    got = view.closes("AAA")
    assert list(got.values) == [1.0, 2.0]  # the 01-06 bar is unreachable
    assert view.last_close("AAA") == 2.0


def test_view_between_bars_uses_last_prior_bar():
    s = _closes(["2020-01-02", "2020-01-06"], [1.0, 2.0])
    view = _panel({"AAA": s}).view(pd.Timestamp("2020-01-04", tz="UTC"))
    assert view.last_close("AAA") == 1.0


def test_view_before_history_is_empty_and_nan():
    s = _closes(["2020-01-02"], [1.0])
    view = _panel({"AAA": s}).view(pd.Timestamp("2019-12-31", tz="UTC"))
    assert view.closes("AAA").empty
    assert math.isnan(view.last_close("AAA"))
    assert math.isnan(view.last_close("MISSING"))


def test_view_rejects_naive_as_of():
    panel = _panel({"AAA": _closes(["2020-01-02"], [1.0])})
    with pytest.raises(ValueError):
        panel.view(pd.Timestamp("2020-01-02"))


def test_decision_dates_first_trading_session_per_month():
    # AAA misses Feb 3; BBB trades it -> the UNION calendar supplies Feb 3.
    a = _closes(["2020-01-02", "2020-01-03", "2020-02-04", "2020-03-02"], [1, 2, 3, 4])
    b = _closes(["2020-01-03", "2020-02-03", "2020-03-02"], [1, 2, 3])
    panel = _panel({"AAA": a, "BBB": b})
    got = panel.decision_dates(
        pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-03-31", tz="UTC")
    )
    assert [d.date().isoformat() for d in got] == ["2020-01-02", "2020-02-03", "2020-03-02"]


def test_decision_dates_respects_window_bounds():
    a = _closes(["2020-01-02", "2020-02-03", "2020-03-02"], [1, 2, 3])
    panel = _panel({"AAA": a})
    got = panel.decision_dates(
        pd.Timestamp("2020-02-01", tz="UTC"), pd.Timestamp("2020-02-28", tz="UTC")
    )
    assert [d.date().isoformat() for d in got] == ["2020-02-03"]
    assert panel.decision_dates(
        pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-12-31", tz="UTC")
    ) == ()


def test_load_closes_reads_parquet_and_skips_missing(tmp_path):
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-02", tz="UTC")])
    frame = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.5], "volume": [10.0]},
        index=idx,
    )
    frame.to_parquet(tmp_path / "AAA.parquet")
    got = load_closes(tmp_path, ["AAA", "NOPE"])
    assert set(got) == {"AAA"}
    assert got["AAA"].iloc[0] == 1.5


def test_panel_error_is_a_value_error():
    assert issubclass(PanelError, ValueError)


# --------------------------------------------------------------------------- #
# Options cells + fundamentals (Task 5)
# --------------------------------------------------------------------------- #


def _cell(symbol: str, date: str, *, atm_iv=0.30, put_iv=0.34, call_iv=0.28,
          skew_put_atm=0.05, skew_put_call=0.02) -> dict:
    return {
        "symbol": symbol,
        "decision_date": date,
        "skew_put_atm": skew_put_atm,
        "skew_put_call": skew_put_call,
        "contracts": [
            {"role": "atm", "bid": 4.0, "ask": 4.2, "mid": 4.1, "iv": atm_iv,
             "volume": 100},
            {"role": "otm_put", "iv": put_iv, "volume": 50},
            {"role": "otm_call", "iv": call_iv, "volume": 25},
        ],
    }


def test_cell_metrics_matches_signal_scan_semantics():
    m = cell_metrics(_cell("AAA", "2020-01-02"))
    assert m["hedge"] == 0.05
    assert m["excite"] == -0.02  # -skew_put_call
    assert m["atm_spread"] == (4.2 - 4.0) / 4.1
    assert math.isclose(m["smile"], (0.34 + 0.28) / 2 - 0.30, rel_tol=1e-12)
    assert set(OPTION_COLUMNS) == set(m)


def test_options_from_cells_builds_float_frames():
    frames = options_from_cells([_cell("AAA", "2020-01-02"), _cell("AAA", "2020-02-03")])
    f = frames["AAA"]
    assert list(f.columns) == OPTION_COLUMNS
    assert str(f.index[0].tz) == "UTC"
    assert f.dtypes.eq("float64").all()  # None from a missing leg becomes NaN


def test_options_from_cells_duplicate_date_keeps_last_gathered():
    # A re-gathered (symbol, date) must serve the LAST cell in gather order.
    # The duplicates are interleaved across enough dates (>= ~10) that an
    # unstable sort visibly scrambles which duplicate survives -- small cases
    # pass by luck because quicksort falls back to insertion sort.
    dates = [d.date().isoformat() for d in pd.date_range("2020-01-02", periods=20, freq="B")]
    first = [_cell("AAA", d, atm_iv=0.10) for d in reversed(dates)]  # first gather
    regather = [_cell("AAA", d, atm_iv=0.99) for d in dates]  # re-gather: must win
    frame = options_from_cells(first + regather)["AAA"]
    assert len(frame) == len(dates)  # one row per distinct date
    assert (frame["atm_iv"] == 0.99).all()
    panel = PanelData(closes={}, options={"AAA": frame}, fundamentals={}, symbols=("AAA",))
    row = panel.view(pd.Timestamp(dates[10], tz="UTC")).option_row("AAA")
    assert row["atm_iv"] == 0.99


def test_load_options_skips_and_counts_corrupt_lines(tmp_path):
    path = tmp_path / "samples.jsonl"
    lines = [
        json.dumps(_cell("AAA", "2020-01-02")),
        '{"torn json',                       # corrupt: unparseable
        json.dumps({"decision_date": "2020-01-02"}),  # corrupt: no symbol
        json.dumps(_cell("BBB", "2020-01-02")),
        "",                                  # blank: ignored, not corrupt
    ]
    path.write_text("\n".join(lines) + "\n")
    frames, corrupt, _has_volume = load_options(path)
    assert set(frames) == {"AAA", "BBB"}
    assert corrupt == 2


def test_option_row_is_pit_and_staleness_capped():
    frames = options_from_cells([_cell("AAA", "2020-01-06", atm_iv=0.5)])
    idx = pd.date_range("2020-01-02", periods=40, freq="B", tz="UTC")
    closes = {"AAA": pd.Series(100.0, index=idx)}
    panel = PanelData(closes=closes, options=frames, fundamentals={}, symbols=("AAA",))
    # Before the cell: nothing to see.
    assert panel.view(pd.Timestamp("2020-01-03", tz="UTC")).option_row("AAA") is None
    # On/after the cell within the age cap: visible.
    row = panel.view(pd.Timestamp("2020-01-06", tz="UTC")).option_row("AAA")
    assert row["atm_iv"] == 0.5
    fresh_limit = pd.Timestamp("2020-01-06", tz="UTC") + pd.Timedelta(
        MAX_OPTION_AGE_DAYS, unit="D"
    )
    assert panel.view(fresh_limit).option_row("AAA") is not None
    # One day beyond the cap: stale -> missing, not forward-filled.
    assert panel.view(fresh_limit + pd.Timedelta(1, unit="D")).option_row("AAA") is None


def test_fundamentals_row_filed_date_cut():
    idx = pd.date_range("2020-01-02", periods=10, freq="B", tz="UTC")
    filed = pd.DatetimeIndex([pd.Timestamp("2020-01-06", tz="UTC")])
    fund = pd.DataFrame({"gross_profitability": [0.4], "ttm_net_income": [5e6],
                         "book_equity": [2e7], "shares_outstanding": [1e6]}, index=filed)
    panel = PanelData(closes={"AAA": pd.Series(100.0, index=idx)},
                      options={}, fundamentals={"AAA": fund}, symbols=("AAA",))
    assert panel.view(pd.Timestamp("2020-01-03", tz="UTC")).fundamentals_row("AAA") is None
    row = panel.view(pd.Timestamp("2020-01-07", tz="UTC")).fundamentals_row("AAA")
    assert row["gross_profitability"] == 0.4


def test_build_panel_from_files(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    idx = pd.date_range("2020-01-02", periods=5, freq="B", tz="UTC")
    for sym in ("AAA", "BBB"):
        pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 2.0,
                      "volume": 10.0}, index=idx).to_parquet(cache / f"{sym}.parquet")
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps(_cell("AAA", "2020-01-02")) + "\n"
        + json.dumps(_cell("BBB", "2020-01-02")) + "\n"
        + json.dumps(_cell("NOBAR", "2020-01-02")) + "\n"
    )
    panel = build_panel(cache, samples, tmp_path / "no-fundamentals")
    assert panel.symbols == ("AAA", "BBB")  # NOBAR dropped: allowlist ∩ bars
    assert set(panel.options) == {"AAA", "BBB"}
    assert panel.fundamentals == {}  # absent store dir -> empty, NOT created
    assert not (tmp_path / "no-fundamentals").exists()
    assert panel.corrupt_cells == 0


def test_build_panel_missing_samples_refused(tmp_path):
    with pytest.raises(PanelError):
        build_panel(tmp_path, tmp_path / "nope.jsonl", None)


# --------------------------------------------------------------------------- #
# Explicit-symbols universes (Piece 2)
# --------------------------------------------------------------------------- #


def _write_cache(tmp_path, symbols):
    cache = tmp_path / "cache"
    cache.mkdir()
    idx = pd.date_range("2020-01-02", periods=5, freq="B", tz="UTC")
    for sym in symbols:
        pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 2.0,
                      "volume": 10.0}, index=idx).to_parquet(cache / f"{sym}.parquet")
    return cache


def test_build_panel_explicit_symbols_overrides_the_samples_allowlist(tmp_path):
    cache = _write_cache(tmp_path, ("AAA", "BBB", "CCC"))
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps(_cell("AAA", "2020-01-02")) + "\n"
        + json.dumps(_cell("BBB", "2020-01-02")) + "\n"
    )
    panel = build_panel(cache, samples, None, symbols=("CCC", "BBB"))
    assert panel.symbols == ("BBB", "CCC")   # explicit set wins, sorted
    assert set(panel.options) == {"BBB"}     # option frames restricted to it


def test_build_panel_samples_none_builds_a_closes_only_panel(tmp_path):
    cache = _write_cache(tmp_path, ("AAA",))
    panel = build_panel(cache, None, None, symbols=("AAA", "NOBAR"))
    assert panel.symbols == ("AAA",)         # missing-data rule: NOBAR dropped
    assert panel.options == {}
    assert panel.fundamentals == {}
    assert panel.corrupt_cells == 0


def test_build_panel_empty_symbols_tuple_refused(tmp_path):
    # Mirrors the empty-signals refusal: a universe with no names is a caller
    # bug, refused at assembly, never a silent no-trade sweep.
    with pytest.raises(PanelError, match="empty"):
        build_panel(tmp_path, None, None, symbols=())


def test_build_panel_without_any_universe_source_refused(tmp_path):
    with pytest.raises(PanelError, match="universe source"):
        build_panel(tmp_path, None, None)


def test_load_bars_reads_full_schema_and_nan_fills_legacy_columns(tmp_path):
    idx = pd.date_range("2020-01-02", periods=3, freq="B", tz="UTC")
    wide = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": [1.5, 1.6, 1.7],
         "volume": 10.0, "div_cash": [0.0, 0.25, 0.0], "split_factor": 1.0,
         "close_raw": 1.5},
        index=idx,
    )
    narrow = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
        index=idx,
    )
    wide.to_parquet(tmp_path / "WIDE.parquet")
    narrow.to_parquet(tmp_path / "NARROW.parquet")
    got = load_bars(tmp_path, ["WIDE", "NARROW", "NOPE"])
    assert set(got) == {"WIDE", "NARROW"}
    assert list(got["WIDE"].columns) == BAR_COLUMNS      # close_raw dropped
    assert got["WIDE"]["div_cash"].iloc[1] == 0.25
    # A legacy narrow cache cannot claim "no dividends": NaN, never 0.0/1.0.
    assert got["NARROW"]["div_cash"].isna().all()
    assert got["NARROW"]["split_factor"].isna().all()


def test_view_bars_truncates_at_as_of_and_is_empty_for_unknown_symbol():
    idx = pd.date_range("2020-01-06", periods=5, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {c: float(i) for i, c in enumerate(BAR_COLUMNS)}, index=idx
    )
    panel = PanelData(closes={"AAA": frame["close"]}, bars={"AAA": frame},
                      symbols=("AAA",))
    view = panel.view(idx[2])
    assert len(view.bars("AAA")) == 3
    assert view.bars("AAA").index.max() == idx[2]
    empty = view.bars("NOPE")
    assert empty.empty and list(empty.columns) == BAR_COLUMNS


def test_view_factors_truncates_at_as_of():
    idx = pd.date_range("2020-01-06", periods=5, freq="B", tz="UTC")
    factors = pd.DataFrame(
        {"Mkt-RF": 0.001, "SMB": 0.0, "HML": 0.0, "RF": 0.0001}, index=idx
    )
    panel = PanelData(closes={}, factors=factors, symbols=())
    got = panel.view(idx[1]).factors()
    assert len(got) == 2 and got.index.max() == idx[1]
    before = pd.Timestamp("2019-12-31", tz="UTC")
    assert panel.view(before).factors().empty


def test_build_panel_derives_closes_from_bars_and_stores_factors(tmp_path):
    idx = pd.date_range("2020-01-02", periods=4, freq="B", tz="UTC")
    pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": [1.0, 2.0, 3.0, 4.0],
         "volume": 10.0},
        index=idx,
    ).to_parquet(tmp_path / "AAA.parquet")
    factors = pd.DataFrame(
        {"Mkt-RF": 0.001, "SMB": 0.0, "HML": 0.0, "RF": 0.0001, "Mom": 0.0},
        index=idx,
    )
    panel = build_panel(tmp_path, None, None, symbols=("AAA",), factors=factors)
    assert list(panel.bars["AAA"].columns) == BAR_COLUMNS
    assert panel.closes["AAA"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert panel.factors.equals(factors)


def test_build_panel_without_factors_stores_an_empty_frame(tmp_path):
    idx = pd.date_range("2020-01-02", periods=4, freq="B", tz="UTC")
    pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
        index=idx,
    ).to_parquet(tmp_path / "AAA.parquet")
    panel = build_panel(tmp_path, None, None, symbols=("AAA",))
    assert panel.factors.empty


def test_build_panel_precomputes_rolling_features_when_factors_supplied(tmp_path):
    idx = pd.date_range("2020-01-02", periods=30, freq="B", tz="UTC")
    pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5,
         "close": np.linspace(100.0, 110.0, 30), "volume": 10.0},
        index=idx,
    ).to_parquet(tmp_path / "AAA.parquet")
    rng = np.random.default_rng(2)
    factors = pd.DataFrame(
        {"Mkt-RF": rng.normal(0.0, 0.01, 30), "SMB": rng.normal(0.0, 0.005, 30),
         "HML": rng.normal(0.0, 0.005, 30), "RF": 0.0001, "Mom": 0.0},
        index=idx,
    )
    panel = build_panel(tmp_path, None, None, symbols=("AAA",), factors=factors)
    assert list(panel.features["AAA"].columns) == ROLLING_FEATURES
    without = build_panel(tmp_path, None, None, symbols=("AAA",))
    assert without.features == {}


# --------------------------------------------------------------------------- #
# Option-volume infrastructure (Task 4)
# --------------------------------------------------------------------------- #


def test_cell_metrics_opt_dollar_vol_sums_legs_with_both_volume_and_mid():
    metrics = cell_metrics(make_cell("AAA", "2020-01-06"))
    want = 100 * 100 * 4.1 + 50 * 100 * 2.0 + 25 * 100 * 1.5  # 54750
    assert math.isclose(metrics["opt_dollar_vol"], want, rel_tol=1e-12)
    no_volume = cell_metrics(make_cell("AAA", "2020-01-06", with_volume=False))
    assert np.isnan(no_volume["opt_dollar_vol"])  # no qualifying leg -> NaN
    partial = make_cell("AAA", "2020-01-06")
    del partial["contracts"][1]["volume"]  # put leg loses volume
    got = cell_metrics(partial)["opt_dollar_vol"]
    assert math.isclose(got, 100 * 100 * 4.1 + 25 * 100 * 1.5, rel_tol=1e-12)


def test_load_options_reports_leg_volume_presence(tmp_path):
    p1 = tmp_path / "with.jsonl"
    p1.write_text(json.dumps(make_cell("AAA", "2020-01-06")) + "\n")
    p2 = tmp_path / "without.jsonl"
    p2.write_text(json.dumps(make_cell("AAA", "2020-01-06", with_volume=False)) + "\n")
    _frames, _corrupt, has_volume = load_options(p1)
    assert has_volume is True
    _frames, _corrupt, has_volume = load_options(p2)
    assert has_volume is False


def test_build_panel_threads_has_option_volume(tmp_path):
    idx = pd.date_range("2020-01-02", periods=3, freq="B", tz="UTC")
    pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
        index=idx,
    ).to_parquet(tmp_path / "AAA.parquet")
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps(make_cell("AAA", "2020-01-02")) + "\n")
    panel = build_panel(tmp_path, samples, None)
    assert panel.has_option_volume is True
    assert panel.view(idx[0]).has_option_volume is True
    bare = build_panel(tmp_path, None, None, symbols=("AAA",))
    assert bare.has_option_volume is False


def _prior_panel(dates: list[str]) -> PanelData:
    cells = [make_cell("AAA", d) for d in dates]
    return PanelData(closes={}, options=options_from_cells(cells), symbols=("AAA",))


def test_option_row_prior_returns_the_cell_strictly_older_than_current():
    panel = _prior_panel(["2020-01-06", "2020-02-03", "2020-03-02"])
    as_of = pd.Timestamp("2020-03-02", tz="UTC")
    prior = panel.view(as_of).option_row_prior("AAA")
    assert prior is not None
    assert prior.name == pd.Timestamp("2020-02-03", tz="UTC")
    # Current cell = Feb when as_of sits between Feb and Mar cells.
    mid = pd.Timestamp("2020-02-05", tz="UTC")
    prior = panel.view(mid).option_row_prior("AAA")
    assert prior.name == pd.Timestamp("2020-01-06", tz="UTC")


def test_option_row_prior_none_without_an_older_cell_or_when_stale():
    single = _prior_panel(["2020-01-06"])
    as_of = pd.Timestamp("2020-01-06", tz="UTC")
    assert single.view(as_of).option_row_prior("AAA") is None
    assert single.view(as_of).option_row_prior("NOPE") is None
    # Boundary: exactly 45 calendar days before as_of is FRESH; 46 is stale.
    fresh = _prior_panel(["2020-01-17", "2020-03-02"])   # Jan 17 + 45d = Mar 2
    at = pd.Timestamp("2020-03-02", tz="UTC")
    assert fresh.view(at).option_row_prior("AAA") is not None
    stale = _prior_panel(["2020-01-16", "2020-03-02"])   # 46 days -> stale
    assert stale.view(at).option_row_prior("AAA") is None
