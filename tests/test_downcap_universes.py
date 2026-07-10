from __future__ import annotations

import pandas as pd

from trading.alphasearch.panel import BAR_COLUMNS, PanelData
from trading.venues.universes.downcap_membership import (
    MEMBERSHIP_COLUMNS,
    load_band_membership,
)


def _write_membership(path):
    rows = [
        ("micro", "MIC", "2019-01-01", "2019-03-01"),  # micro Jan-Feb (end exclusive)
        ("small", "SML", "2019-01-01", ""),            # small, open through end
        ("micro", "MIC", "2019-04-01", ""),            # re-enters micro from Apr
    ]
    pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS).to_csv(path, index=False)


def test_load_band_membership_filters_bands(tmp_path):
    path = tmp_path / "band_membership.csv"
    _write_membership(path)
    micro_only = load_band_membership(path, frozenset({"micro"}))
    assert set(micro_only) == {"MIC"}
    assert micro_only["MIC"] == (("2019-01-01", "2019-03-01"), ("2019-04-01", ""))
    both = load_band_membership(path, frozenset({"micro", "small"}))
    assert set(both) == {"MIC", "SML"}


def test_panelview_symbols_are_per_date_band_filtered():
    # Two names, both have bars for the whole span; membership makes MIC in-band
    # only Jan-Feb, SML in-band always.
    idx = pd.date_range("2019-01-01", "2019-06-30", freq="B", tz="UTC")
    bars = {
        s: pd.DataFrame(
            {c: (1.0 if c != "volume" else 100_000.0) for c in BAR_COLUMNS}, index=idx
        )[BAR_COLUMNS]
        for s in ("MIC", "SML")
    }
    panel = PanelData(
        closes={s: bars[s]["close"] for s in bars},
        symbols=("MIC", "SML"),
        bars=bars,
        membership={
            "MIC": (("2019-01-01", "2019-03-01"),),
            "SML": (("2019-01-01", ""),),
        },
    )
    jan = panel.view(pd.Timestamp("2019-01-15", tz="UTC"))
    assert set(jan.symbols) == {"MIC", "SML"}
    apr = panel.view(pd.Timestamp("2019-04-15", tz="UTC"))
    assert set(apr.symbols) == {"SML"}          # MIC left the band -> excluded at D


def test_empty_membership_leaves_symbols_unfiltered():
    idx = pd.date_range("2019-01-01", periods=5, freq="B", tz="UTC")
    bars = {"X": pd.DataFrame(
        {c: 1.0 for c in BAR_COLUMNS}, index=idx)[BAR_COLUMNS]}
    panel = PanelData(closes={"X": bars["X"]["close"]}, symbols=("X",), bars=bars)
    view = panel.view(pd.Timestamp("2019-01-03", tz="UTC"))
    assert set(view.symbols) == {"X"}           # default {} -> unchanged behavior
