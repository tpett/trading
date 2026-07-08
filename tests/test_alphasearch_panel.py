"""PanelData/PanelView: PIT truncation and the monthly decision calendar."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trading.alphasearch.panel import PanelData, PanelError, load_closes


def _closes(dates: list[str], values: list[float]) -> pd.Series:
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dates])
    return pd.Series(values, index=idx, dtype="float64", name="close")


def _panel(closes: dict[str, pd.Series]) -> PanelData:
    return PanelData(
        closes=closes, options={}, fundamentals={}, symbols=tuple(sorted(closes))
    )


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
