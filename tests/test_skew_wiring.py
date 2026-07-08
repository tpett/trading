"""End-to-end wiring: the IV-skew channel threaded through prepare()/replay().

A tiny synthetic backtest over a handful of symbols with a hand-made skew
store; a hand computation confirms the cross-sectional composite the engine
produces, that no-lookahead holds at the engine seam (a skew dated after a
session never reaches it), and that the momentum path is unaffected when a
non-skew ranker is configured.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import replace

import pandas as pd
import pytest

from backtest_helpers import FakeBacktestAdapter, noisy_frame, small_config
from trading.backtest.engine import prepare, replay
from trading.data.cache import OhlcvCache
from trading.signals.skew import SKEW_MIN_CROSS_SECTION

START = datetime.date(2025, 2, 1)
END = datetime.date(2025, 4, 30)

# A dense cross-section (>= SKEW_MIN_CROSS_SECTION names carry skew) so the
# ranker's thin-cross-section guard does NOT fire and the composite is a real
# skew-driven percentile. Skew increases with the index: S00 is flattest (should
# top the ranking), the last name steepest (bottom).
UNIVERSE_N = SKEW_MIN_CROSS_SECTION + 2
FLATTEST = "S00"
STEEPEST = f"S{UNIVERSE_N - 1:02d}"
SKEW_BY_SYMBOL = {f"S{i:02d}": 0.02 + 0.01 * i for i in range(UNIVERSE_N)}


def _frames() -> dict[str, pd.DataFrame]:
    frames = {f"S{i:02d}": noisy_frame(seed=i + 1, drift=0.004) for i in range(UNIVERSE_N)}
    frames["BENCH"] = noisy_frame(seed=99, drift=0.003)
    return frames


def _write_skew(path, extra_late=None) -> None:
    """One skew cell per symbol dated 2025-01-15 (visible to every session),
    plus an optional LATER cell to prove no-lookahead at the engine seam."""
    cells = [
        {
            "symbol": symbol,
            "decision_date": "2025-01-15",
            "spot_at_decision": 100.0,
            "days_to_expiry": 42,
            "contracts": [],
            "skew_put_atm": value,
            "skew_put_call": None,
        }
        for symbol, value in SKEW_BY_SYMBOL.items()
    ]
    if extra_late is not None:
        symbol, date, value = extra_late
        cells.append(
            {
                "symbol": symbol,
                "decision_date": date,
                "spot_at_decision": 100.0,
                "days_to_expiry": 42,
                "contracts": [],
                "skew_put_atm": value,
                "skew_put_call": None,
            }
        )
    path.write_text("\n".join(json.dumps(c) for c in cells) + "\n")


def _skew_config(tmp_path, extra_late=None):
    samples = tmp_path / "samples.jsonl"
    _write_skew(samples, extra_late=extra_late)
    config = replace(small_config(), benchmark="BENCH")
    config = replace(
        config,
        signals=replace(config.signals, ranker="skew_v1"),
        data=replace(config.data, skew_samples=str(samples)),
    )
    return config


def _first_ranked(prepared):
    for plan in prepared.sessions:
        if plan.rankings is not None:
            return plan
    raise AssertionError("no ranked session")


def test_skew_channel_drives_the_cross_sectional_composite(tmp_path):
    config = _skew_config(tmp_path)
    adapter = FakeBacktestAdapter(_frames(), "BENCH")
    cache = OhlcvCache(tmp_path / "cache", config.data.refetch_days)
    prepared = prepare(config, adapter, cache, START, END)

    table = _first_ranked(prepared).rankings.table
    # Composite = cross-sectional percentile of -skew over the N names: the
    # flattest tops out at 1.0, the steepest at 1/N.
    assert table.loc[FLATTEST, "composite"] == 1.0
    assert table.loc[STEEPEST, "composite"] == pytest.approx(1 / UNIVERSE_N)
    assert table.loc[FLATTEST, "composite"] > table.loc[STEEPEST, "composite"]
    # The raw as-of skew level is surfaced in the table.
    assert table.loc[FLATTEST, "skew"] == 0.02
    # Ranked table is sorted by composite: the flattest name is first.
    assert table.index[0] == FLATTEST
    # A backtest actually runs through to a result on this channel.
    result = replay(prepared, config)
    assert result.sessions_run > 0


def test_engine_seam_has_no_lookahead(tmp_path):
    # S00 gets a much flatter skew LATE in the run (2025-04-20); early sessions
    # must still see only the 2025-01-15 value (0.02), never the future one.
    config = _skew_config(tmp_path, extra_late=(FLATTEST, "2025-04-20", -0.50))
    adapter = FakeBacktestAdapter(_frames(), "BENCH")
    cache = OhlcvCache(tmp_path / "cache", config.data.refetch_days)
    prepared = prepare(config, adapter, cache, START, END)

    early = _first_ranked(prepared)
    assert early.ts < pd.Timestamp("2025-04-20", tz="UTC")
    assert early.rankings.table.loc[FLATTEST, "skew"] == 0.02  # not the future -0.50

    # A late session (on/after 2025-04-20) DOES see the new value.
    late = [
        p
        for p in prepared.sessions
        if p.rankings is not None and p.ts >= pd.Timestamp("2025-04-20", tz="UTC")
    ]
    assert late, "expected a session at/after the late skew update"
    assert late[0].rankings.table.loc[FLATTEST, "skew"] == -0.50


def test_momentum_path_unaffected_and_ignores_skew_samples(tmp_path):
    # A momentum config that even HAS skew_samples set must not load or thread
    # skew (requires_skew is False): its table carries the momentum columns and
    # no "skew" column.
    samples = tmp_path / "samples.jsonl"
    _write_skew(samples)
    config = replace(small_config(), benchmark="BENCH")
    config = replace(config, data=replace(config.data, skew_samples=str(samples)))
    assert config.signals.ranker == "momentum_v1"
    adapter = FakeBacktestAdapter(_frames(), "BENCH")
    cache = OhlcvCache(tmp_path / "cache", config.data.refetch_days)
    prepared = prepare(config, adapter, cache, START, END)

    table = _first_ranked(prepared).rankings.table
    assert "skew" not in table.columns
    assert "mom_short" in table.columns  # momentum feature columns present
    result = replay(prepared, config)
    assert result.sessions_run > 0
