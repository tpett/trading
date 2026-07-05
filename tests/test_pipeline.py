import dataclasses
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sim_helpers import frame
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
        self.fetch_calls: dict[str, int] = {}

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        return self.infos

    def constraints(self) -> VenueConstraints:
        return VenueConstraints(0.0, 0.0, 5.0, 1, False)

    def fetch_ohlcv(self, symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        self.fetch_calls[symbol] = self.fetch_calls.get(symbol, 0) + 1
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


def test_corrupted_benchmark_raises_pipeline_data_error(tmp_path):
    adapter, cache, frames = _make(tmp_path)
    # Recent (within the quarantine window), implausible benchmark move: a
    # corrupt print must abort the run, not silently distort the regime gate.
    spike_at = frames["SPY"].index[-5]
    prior_at = frames["SPY"].index[-6]
    frames["SPY"].loc[spike_at, "close"] = frames["SPY"]["close"].loc[prior_at] * 1.7
    with pytest.raises(PipelineDataError, match="benchmark"):
        build_rankings(CONFIG, adapter, cache, AS_OF)


def test_benchmark_symbol_already_in_universe_is_not_fetched_twice(tmp_path):
    adapter, cache, frames = _make(tmp_path)
    adapter.infos.append(SymbolInfo("SPY", "tradable"))
    result = build_rankings(CONFIG, adapter, cache, AS_OF)
    assert adapter.fetch_calls.get("SPY") == 1
    assert "SPY" in result.table.index


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


def test_result_exposes_clean_bars_and_benchmark_bars(tmp_path):
    adapter, cache, frames = _make(tmp_path)
    result = build_rankings(CONFIG, adapter, cache, AS_OF)
    assert set(result.bars) == {f"S{i}" for i in range(10)}
    pd.testing.assert_frame_equal(result.bars["S1"], frames["S1"])
    pd.testing.assert_frame_equal(result.benchmark_bars, frames["SPY"])


def test_quarantined_symbol_is_excluded_from_bars(tmp_path):
    adapter, cache, frames = _make(tmp_path)
    spike_at = frames["S5"].index[-5]
    prior_at = frames["S5"].index[-6]
    frames["S5"].loc[spike_at, "close"] = frames["S5"]["close"].loc[prior_at] * 1.7
    result = build_rankings(CONFIG, adapter, cache, AS_OF)
    assert "S5" not in result.bars


def test_assemble_rankings_is_pure_and_matches_build_rankings_semantics():
    # No adapter, no cache: hand it frames directly and get a full RankingsResult.
    from trading.pipeline import assemble_rankings

    config = load_venue_config("equities", Path("config"))
    bars = {"AAA": frame(periods=300), "BBB": frame(periods=300)}
    benchmark = frame(periods=300)
    infos = [
        SymbolInfo(symbol="AAA", status="tradable"),
        SymbolInfo(symbol="BBB", status="sell_only"),
    ]
    result = assemble_rankings(config, infos, bars, benchmark, datetime.date(2026, 7, 1))
    assert set(result.table.index) == {"AAA", "BBB"}
    assert result.table.loc["BBB", "status"] == "sell_only"
    assert result.coverage.ratio == 1.0
    assert result.bars.keys() == bars.keys()  # nothing quarantined
    assert result.venue == "equities"


def test_build_rankings_delegates_to_assemble_rankings_with_identical_results(tmp_path):
    # Pins the extraction seam: build_rankings must be fetch + delegate, so any
    # post-processing later added after the assemble_rankings call breaks loudly.
    from trading.pipeline import assemble_rankings

    # A fetch failure (S3) and a quarantined symbol (S5) make fetch_failures and
    # quarantined nontrivial, so their propagation through the seam is pinned too.
    adapter, cache, frames = _make(tmp_path, fail=frozenset({"S3"}))
    spike_at = frames["S5"].index[-5]
    prior_at = frames["S5"].index[-6]
    frames["S5"].loc[spike_at, "close"] = frames["S5"]["close"].loc[prior_at] * 1.7

    via_build = build_rankings(CONFIG, adapter, cache, AS_OF)

    # Reconstruct assemble_rankings' inputs through the same fake fetch path
    # build_rankings uses: universe -> cached fetch -> drop-incomplete-last-bar.
    infos = adapter.universe(AS_OF)
    start = AS_OF - datetime.timedelta(days=CONFIG.data.history_days)
    cutoff = pd.Timestamp(AS_OF, tz="UTC")
    mirror_cache = OhlcvCache(tmp_path / "mirror_cache", CONFIG.data.refetch_days)

    def _fetch(symbol: str) -> pd.DataFrame:
        df = mirror_cache.fetch(symbol, start, AS_OF, adapter.fetch_ohlcv)
        return df[df.index < cutoff] if CONFIG.data.drop_incomplete_last_bar else df

    bars: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    for info in infos:
        try:
            bars[info.symbol] = _fetch(info.symbol)
        except DataFetchError:
            failures.append(info.symbol)
    benchmark = _fetch(CONFIG.benchmark)

    direct = assemble_rankings(
        CONFIG, infos, bars, benchmark, AS_OF, fetch_failures=tuple(sorted(failures))
    )

    assert direct.venue == via_build.venue == "equities"
    assert direct.as_of == via_build.as_of
    assert direct.regime == via_build.regime
    assert direct.coverage == via_build.coverage
    assert direct.quarantined == via_build.quarantined == ("S5",)
    assert direct.fetch_failures == via_build.fetch_failures == ("S3",)
    assert direct.insufficient_history == via_build.insufficient_history
    pd.testing.assert_frame_equal(direct.table, via_build.table)
    assert direct.bars.keys() == via_build.bars.keys()
    for symbol in direct.bars:
        pd.testing.assert_frame_equal(direct.bars[symbol], via_build.bars[symbol])
    pd.testing.assert_frame_equal(direct.benchmark_bars, via_build.benchmark_bars)
