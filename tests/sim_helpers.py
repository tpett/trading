"""Shared builders for simulator tests: bar frames, ranking tables, states."""

from pathlib import Path

import numpy as np
import pandas as pd

from trading.config import load_venue_config
from trading.data.quality import CoverageReport
from trading.pipeline import RankingsResult
from trading.signals.regime import Regime
from trading.simulator.state import initial_state

EQ = load_venue_config("equities", Path("config"))
CR = load_venue_config("crypto", Path("config"))
AS_OF = pd.Timestamp("2026-07-01", tz="UTC")

REGIME_MULT = {"risk_on": 1.0, "neutral": 0.5, "risk_off": 0.0}


def frame(
    *,
    periods: int = 80,
    end: str = "2026-07-01",
    drift: float = 0.0,
    start_price: float = 100.0,
    volume: float = 1e6,
    freq: str = "B",
) -> pd.DataFrame:
    """Deterministic OHLCV frame ending at `end` with constant per-bar drift."""
    idx = pd.date_range(end=end, periods=periods, freq=freq, tz="UTC")
    close = start_price * np.cumprod(np.full(periods, 1.0 + drift))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(periods, volume),
        },
        index=idx,
    )


def make_table(rows: dict[str, dict]) -> pd.DataFrame:
    """rows: symbol -> {"status": ..., "composite": ..., "raw_return_30d": ...}.
    Returns a frame sorted by composite desc, like trading.signals.engine.rank."""
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.sort_values("composite", ascending=False, na_position="last", kind="mergesort")


def make_rankings(
    config,
    bars: dict[str, pd.DataFrame],
    table: pd.DataFrame,
    *,
    regime_state: str = "risk_on",
    benchmark: pd.DataFrame | None = None,
    quarantined: tuple[str, ...] = (),
    fetch_failures: tuple[str, ...] = (),
) -> RankingsResult:
    symbols = list(bars)
    return RankingsResult(
        venue=config.name,
        as_of=AS_OF,
        regime=Regime(state=regime_state, exposure_multiplier=REGIME_MULT[regime_state]),
        table=table,
        coverage=CoverageReport(
            requested=len(symbols), fetched=len(symbols), ratio=1.0, ok=True, missing=()
        ),
        quarantined=quarantined,
        fetch_failures=fetch_failures,
        insufficient_history=(),
        bars=bars,
        benchmark_bars=benchmark if benchmark is not None else frame(periods=260),
    )


def make_state(config, **overrides):
    state = initial_state(
        config.name,
        config.portfolio.starting_balance,
        100.0,
        "2026-06-01T00:00:00+00:00",
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state
