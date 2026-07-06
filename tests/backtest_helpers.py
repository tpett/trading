"""Shared builders for backtest tests: noisy frames, a date-aware fake
adapter, small-window configs, and hand-built PreparedBacktest fixtures."""

from __future__ import annotations

import datetime
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd

from sim_helpers import CR, make_rankings
from trading.backtest.engine import PreparedBacktest, SessionPlan
from trading.config import VenueConfig
from trading.venues.base import SymbolInfo, VenueConstraints


def noisy_frame(
    *,
    seed: int,
    drift: float = 0.0,
    periods: int = 120,
    start: str = "2025-01-01",
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Seeded random-walk OHLCV, daily UTC bars (24/7 style)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    returns = rng.normal(loc=drift, scale=0.02, size=periods)
    close = start_price * np.cumprod(1.0 + returns)
    open_ = np.concatenate([[start_price], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.uniform(5e5, 1.5e6, size=periods),
        },
        index=idx,
    )


def small_config(base: VenueConfig = CR, **backtest_overrides) -> VenueConfig:
    """Shrink every window so 120-bar fixtures rank and trade."""
    signals = replace(
        base.signals,
        momentum_windows=(3, 5, 8),
        vol_window=5,
        volume_week=3,
        volume_baseline=10,
        breakout_windows=(5, 8),
        rsi_window=5,
        mean_window=5,
        raw_return_days=5,
    )
    regime = replace(base.regime, sma_fast=5, sma_slow=15, vol_lookback=30)
    portfolio = replace(
        base.portfolio,
        atr_window=5,
        time_stop_bars=8,
        entry_score_threshold=0.55,
        min_raw_return_cost_multiple=0.0,
    )
    backtest_defaults = {
        "start": datetime.date(2025, 2, 1),
        "holdout_start": datetime.date(2027, 1, 1),
        "min_session_coverage": 0.5,
        "periods_per_year": 365,
    }
    backtest = replace(base.backtest, **{**backtest_defaults, **backtest_overrides})
    data = replace(base.data, history_days=25, backfill_exchange="")
    return replace(
        base, signals=signals, regime=regime, portfolio=portfolio, backtest=backtest, data=data
    )


class FakeBacktestAdapter:
    """In-memory venue: membership may vary by date; fetch slices stored frames."""

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        benchmark: str,
        members_on: Callable[[datetime.date], list[str]] | None = None,
        intervals: Callable[[str], list[tuple[str, str]]] | None = None,
    ):
        self._frames = frames
        self._benchmark = benchmark
        self._members_on = members_on  # date -> list[str]; None = all non-benchmark symbols
        # symbol -> [(start, end), ...]; None = no per-symbol data (every
        # test that doesn't care about the recycling guard leaves this
        # unset, and membership_intervals then reports "no info" for every
        # symbol, which the guard treats as "never truncate").
        self._intervals = intervals

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        if self._members_on is not None:
            names = self._members_on(as_of)
        else:
            names = [s for s in sorted(self._frames) if s != self._benchmark]
        return [SymbolInfo(symbol=s, status="tradable") for s in names]

    def membership_intervals(self, symbol: str) -> list[tuple[str, str]]:
        return self._intervals(symbol) if self._intervals is not None else []

    def constraints(self) -> VenueConstraints:
        return VenueConstraints(
            taker_fee_bps=95.0,
            maker_fee_bps=50.0,
            slippage_bps=5.0,
            settlement_days=0,
            trades_24_7=True,
        )

    def fetch_ohlcv(self, symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        df = self._frames[symbol]
        return df.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]


def prepared_from_sessions(
    config: VenueConfig,
    session_specs: list[tuple[str, pd.DataFrame]],
    bars: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
) -> PreparedBacktest:
    """Hand-built PreparedBacktest: full control over each session's table.

    session_specs: [(iso_date, table)] built with sim_helpers.make_table.
    """
    sessions = []
    for iso, table in session_specs:
        ts = pd.Timestamp(iso, tz="UTC")
        rankings = make_rankings(config, {s: bars[s].loc[:ts] for s in bars}, table)
        slim = replace(rankings, bars={}, benchmark_bars=benchmark.iloc[0:0])
        sessions.append(
            SessionPlan(
                ts=ts,
                rankings=slim,
                clean_symbols=tuple(bars),
                survivorship_ratio=1.0,
                eligible_members=len(bars),
                skip_reason=None,
            )
        )
    return PreparedBacktest(
        venue=config.name,
        start=sessions[0].ts.date(),
        end=sessions[-1].ts.date(),
        sessions=tuple(sessions),
        bars=bars,
        benchmark_bars=benchmark,
        missing_symbols=(),
    )
