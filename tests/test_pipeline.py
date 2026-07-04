import dataclasses
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading.config import load_venue_config
from trading.data.cache import OhlcvCache
from trading.pipeline import PipelineDataError, build_rankings
from trading.venues.base import DataFetchError, SymbolInfo, VenueConstraints

CONFIG = load_venue_config("equities", Path("config"))
AS_OF = datetime.date(2026, 7, 1)


def _bars(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end="2026-07-01", periods=320, freq="B", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, 320))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.uniform(1e5, 1e6, 320),
        },
        index=idx,
    )


class FakeAdapter:
    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        infos: list[SymbolInfo],
        fail: frozenset[str] = frozenset(),
    ):
        self.frames = frames
        self.infos = infos
        self.fail = fail

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        return self.infos

    def constraints(self) -> VenueConstraints:
        return VenueConstraints(0.0, 0.0, 5.0, 1, False)

    def fetch_ohlcv(self, symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        if symbol in self.fail:
            raise DataFetchError(symbol)
        df = self.frames[symbol]
        return df.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]


def _make(tmp_path, fail: frozenset[str] = frozenset()):
    symbols = [f"S{i}" for i in range(10)]
    frames = {s: _bars(i) for i, s in enumerate(symbols)}
    frames["SPY"] = _bars(999)  # benchmark per config/equities.toml
    infos = [SymbolInfo(s, "tradable") for s in symbols]
    adapter = FakeAdapter(frames, infos, fail=fail)
    cache = OhlcvCache(tmp_path / "cache", CONFIG.data.refetch_days)
    return adapter, cache, frames


def test_happy_path_ranks_full_universe(tmp_path):
    adapter, cache, _ = _make(tmp_path)
    result = build_rankings(CONFIG, adapter, cache, AS_OF)
    assert result.venue == "equities"
    assert result.as_of == pd.Timestamp(AS_OF, tz="UTC")
    assert len(result.table) == 10
    assert list(result.table.columns)[0] == "status"
    composites = result.table["composite"].tolist()
    assert composites == sorted(composites, reverse=True)
    assert result.coverage.ok
    assert result.regime.state in {"risk_on", "neutral", "risk_off"}
    assert result.regime.exposure_multiplier in {1.0, 0.5, 0.0}
    assert result.fetch_failures == ()
    assert result.quarantined == ()


def test_one_failure_in_ten_proceeds_with_exclusion(tmp_path):
    adapter, cache, _ = _make(tmp_path, fail=frozenset({"S3"}))
    result = build_rankings(CONFIG, adapter, cache, AS_OF)
    assert result.fetch_failures == ("S3",)
    assert "S3" not in result.table.index
    assert result.coverage.ratio == pytest.approx(0.9)


def test_below_min_coverage_raises(tmp_path):
    adapter, cache, _ = _make(tmp_path, fail=frozenset({"S3", "S7"}))
    with pytest.raises(PipelineDataError, match="coverage"):
        build_rankings(CONFIG, adapter, cache, AS_OF)


def test_quarantined_symbol_is_excluded_and_reported(tmp_path):
    adapter, cache, frames = _make(tmp_path)
    # Spike near the end of the frame (a handful of trading days back), well
    # inside the quarantine_window_days lookback, so it still trips the gate.
    spike_at = frames["S5"].index[-5]
    prior_at = frames["S5"].index[-6]
    frames["S5"].loc[spike_at, "close"] = frames["S5"]["close"].loc[prior_at] * 1.7
    result = build_rankings(CONFIG, adapter, cache, AS_OF)
    assert result.quarantined == ("S5",)
    assert "S5" not in result.table.index


def test_old_spike_outside_quarantine_window_is_not_quarantined(tmp_path):
    adapter, cache, frames = _make(tmp_path)
    # Same spike shape as above, but far in the past: outside the
    # quarantine_window_days lookback, so it must not exclude the symbol.
    spike_at = frames["S5"].index[200]
    prior_at = frames["S5"].index[199]
    frames["S5"].loc[spike_at, "close"] = frames["S5"]["close"].loc[prior_at] * 1.7
    result = build_rankings(CONFIG, adapter, cache, AS_OF)
    assert result.quarantined == ()
    assert "S5" in result.table.index


def test_insufficient_history_reported_not_ranked(tmp_path):
    adapter, cache, frames = _make(tmp_path)
    frames["S9"] = frames["S9"].iloc[-30:]  # 30 bars < equities min history
    result = build_rankings(CONFIG, adapter, cache, AS_OF)
    assert result.insufficient_history == ("S9",)
    assert "S9" not in result.table.index
    assert result.coverage.fetched == 10  # it fetched fine; it just can't be ranked yet


def test_sell_only_symbol_still_ranked_with_status(tmp_path):
    adapter, cache, _ = _make(tmp_path)
    adapter.infos[2] = SymbolInfo("S2", "sell_only")
    result = build_rankings(CONFIG, adapter, cache, AS_OF)
    assert result.table.loc["S2", "status"] == "sell_only"


def test_benchmark_failure_raises(tmp_path):
    adapter, cache, _ = _make(tmp_path, fail=frozenset({"SPY"}))
    with pytest.raises(PipelineDataError, match="benchmark"):
        build_rankings(CONFIG, adapter, cache, AS_OF)


def _with_drop(enabled: bool):
    return dataclasses.replace(
        CONFIG, data=dataclasses.replace(CONFIG.data, drop_incomplete_last_bar=enabled)
    )


def test_drop_incomplete_last_bar_matches_manual_removal(tmp_path):
    # Every frame here (from _bars/_make) ends exactly on AS_OF, so its last
    # row is the as-of-dated (potentially still-forming) bar.
    _, _, frames = _make(tmp_path)
    infos = [SymbolInfo(s, "tradable") for s in frames if s != "SPY"]

    stripped = {s: df.iloc[:-1] for s, df in frames.items()}
    baseline_adapter = FakeAdapter(stripped, infos)
    baseline_cache = OhlcvCache(tmp_path / "cache_baseline", CONFIG.data.refetch_days)
    baseline = build_rankings(_with_drop(False), baseline_adapter, baseline_cache, AS_OF)

    full_adapter = FakeAdapter(frames, infos)
    full_cache = OhlcvCache(tmp_path / "cache_full", CONFIG.data.refetch_days)
    with_drop = build_rankings(_with_drop(True), full_adapter, full_cache, AS_OF)

    pd.testing.assert_frame_equal(with_drop.table, baseline.table)


def test_drop_incomplete_last_bar_flag_changes_rankings(tmp_path):
    _, _, frames = _make(tmp_path)
    infos = [SymbolInfo(s, "tradable") for s in frames if s != "SPY"]

    adapter_a = FakeAdapter(frames, infos)
    cache_a = OhlcvCache(tmp_path / "cache_a", CONFIG.data.refetch_days)
    with_drop = build_rankings(_with_drop(True), adapter_a, cache_a, AS_OF)

    adapter_b = FakeAdapter(frames, infos)
    cache_b = OhlcvCache(tmp_path / "cache_b", CONFIG.data.refetch_days)
    without_drop = build_rankings(_with_drop(False), adapter_b, cache_b, AS_OF)

    assert not with_drop.table["composite"].equals(without_drop.table["composite"])
