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


def _write_bars_at(cache_dir, symbols, periods=40):
    # Like _write_bar_cache above, but at a CALLER-CHOSEN path -- needed here
    # because downcap_universes derives cache_dir itself
    # (root/data/equities-downcap-tiingo), so the bar fixture must live there
    # rather than at the ad hoc tmp_path/"cache" the other helper uses.
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2019-01-01", periods=periods, freq="B", tz="UTC")
    for symbol in symbols:
        frame = pd.DataFrame(
            {c: (1.0 if c != "volume" else 1e5) for c in BAR_COLUMNS}, index=idx
        )[BAR_COLUMNS]
        frame.to_parquet(cache_dir / f"{symbol}.parquet")
    return idx


def test_downcap_universe_loads_through_build_universe_panel_and_filters(tmp_path):
    # End-to-end proof for the R3 down-cap chain (B2 review: this was only
    # verified by inspection before). A downcap_universes() spec must load
    # through build_universe_panel without tripping B1's partial-config
    # ValueError (membership_intervals and bands both set), and the resulting
    # panel's per-date membership must actually restrict PanelView.symbols.
    cache_dir = tmp_path / "data" / "equities-downcap-tiingo"
    _write_bars_at(cache_dir, ("MIC", "SML"))

    membership_csv = tmp_path / "band_membership.csv"
    rows = [
        ("micro", "MIC", "2019-01-01", "2019-02-01"),  # in-band Jan only
        ("small", "SML", "2019-01-01", ""),             # in-band throughout
    ]
    pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS).to_csv(membership_csv, index=False)

    specs = downcap_universes(tmp_path, membership_path=membership_csv)
    spec = specs["downcap"]
    assert spec.membership_intervals == membership_csv
    assert spec.bands == ("micro", "small")

    # The B1 guard fires only on a ONE-SIDED config; the factory always sets
    # both membership_intervals and bands together, so this must not raise.
    panel = build_universe_panel(spec)

    assert panel.membership == {
        "MIC": (("2019-01-01", "2019-02-01"),),
        "SML": (("2019-01-01", ""),),
    }
    assert set(panel.symbols) == {"MIC", "SML"}

    jan = panel.view(pd.Timestamp("2019-01-15", tz="UTC"))
    assert set(jan.symbols) == {"MIC", "SML"}   # both in-band in January

    feb = panel.view(pd.Timestamp("2019-02-15", tz="UTC"))
    assert set(feb.symbols) == {"SML"}          # MIC left the band -> excluded at D

    # downcap:small (bands=("small",)) excludes the micro-only symbol
    # entirely -- proving the band filter flows through to symbol selection,
    # not just the per-date view.
    small_spec = specs["downcap:small"]
    assert small_spec.bands == ("small",)
    small_panel = build_universe_panel(small_spec)
    assert set(small_panel.symbols) == {"SML"}
    assert small_panel.membership == {"SML": (("2019-01-01", ""),)}
