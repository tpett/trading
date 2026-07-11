from __future__ import annotations

import pandas as pd
import pytest

from trading.alphasearch.panel import BAR_COLUMNS, PanelData
from trading.alphasearch.sweep import UniverseSpec, build_universe_panel
from trading.venues.universes.downcap_membership import (
    MEMBERSHIP_COLUMNS,
    downcap_universes,
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


def _write_bar_cache(tmp_path, symbols):
    idx = pd.date_range("2019-01-01", periods=10, freq="B", tz="UTC")
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    for symbol in symbols:
        frame = pd.DataFrame(
            {c: (1.0 if c != "volume" else 1e5) for c in BAR_COLUMNS}, index=idx
        )[BAR_COLUMNS]
        frame.to_parquet(cache / f"{symbol}.parquet")
    return cache


def test_universespec_partial_membership_config_raises(tmp_path):
    cache = _write_bar_cache(tmp_path, ("AAA", "SML"))
    membership_csv = tmp_path / "band_membership.csv"
    _write_membership(membership_csv)

    only_intervals = UniverseSpec(
        "u", cache, None, None, symbols=("AAA",),
        membership_intervals=membership_csv, bands=None,
    )
    with pytest.raises(ValueError, match="membership_intervals"):
        build_universe_panel(only_intervals)

    only_bands = UniverseSpec(
        "u", cache, None, None, symbols=("AAA",),
        membership_intervals=None, bands=("micro",),
    )
    with pytest.raises(ValueError, match="bands"):
        build_universe_panel(only_bands)

    # Both None: unfiltered default, no error (Piece 1/2 behavior unchanged).
    neither = UniverseSpec("u", cache, None, None, symbols=("AAA",))
    panel = build_universe_panel(neither)
    assert panel.membership == {}

    # Both set: filtered as today, no error -- SML actually carries a "small"
    # row in the fixture CSV, so this proves the guard doesn't disturb the
    # real filtering path, not just that it fails to raise.
    both = UniverseSpec(
        "u", cache, None, None, symbols=("SML",),
        membership_intervals=membership_csv, bands=("small",),
    )
    panel = build_universe_panel(both)
    assert panel.membership == {"SML": (("2019-01-01", ""),)}


def _write_full_membership(path):
    rows = [
        ("micro", "MIC", "2019-01-01", ""),
        ("small", "SML", "2019-01-01", ""),
    ]
    pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS).to_csv(path, index=False)


def test_downcap_universes_registers_three_specs(tmp_path):
    path = tmp_path / "band_membership.csv"
    _write_full_membership(path)
    specs = downcap_universes(tmp_path, membership_path=path)
    assert set(specs) == {"downcap", "downcap:small", "downcap:micro"}
    # Full band = union of both names; sub-bands partition it.
    assert set(specs["downcap"].symbols) == {"MIC", "SML"}
    assert set(specs["downcap:small"].symbols) == {"SML"}
    assert set(specs["downcap:micro"].symbols) == {"MIC"}
    # Each carries its band filter + the membership CSV + fresh cache dir.
    assert specs["downcap:micro"].bands == ("micro",)
    assert specs["downcap"].bands == ("micro", "small")
    assert specs["downcap"].membership_intervals == path
    assert specs["downcap"].cache_dir == tmp_path / "data" / "equities-downcap-tiingo"
    assert specs["downcap"].samples is None            # options signals refused


def test_downcap_universes_absent_csv_returns_empty(tmp_path):
    # No membership CSV built yet -> no specs (the leaderboard/sweep then just
    # omits them, like segments do when their inputs are absent).
    assert downcap_universes(tmp_path, membership_path=tmp_path / "missing.csv") == {}
