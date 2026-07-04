"""Frozen per-venue configuration loaded from config/<venue>.toml.

Every tunable number in the system lives in TOML, never as a code constant.
Unknown or missing TOML keys raise TypeError via dataclass construction.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CostsConfig:
    taker_fee_bps: float
    maker_fee_bps: float
    slippage_bps: float
    settlement_days: int
    trades_24_7: bool


@dataclass(frozen=True)
class UniverseConfig:
    min_dollar_volume: float


@dataclass(frozen=True)
class SignalConfig:
    momentum_windows: tuple[int, int, int]
    calendar_days: bool
    vol_window: int
    volume_week: int
    volume_baseline: int
    breakout_windows: tuple[int, int]
    rsi_window: int
    mean_window: int
    raw_return_days: int


@dataclass(frozen=True)
class RegimeConfig:
    sma_fast: int
    sma_slow: int
    vol_window: int
    vol_lookback: int
    vol_high_percentile: float
    exposure_risk_on: float
    exposure_neutral: float
    exposure_risk_off: float


@dataclass(frozen=True)
class PortfolioConfig:
    """Loaded in M1 for config-shape stability; consumed by the M2 simulator."""

    max_positions: int
    position_size_pct: float
    starting_balance: float
    time_stop_bars: int
    stop_atr_multiple: float
    regime_flush_atr_multiple: float
    cooldown_days: int
    max_daily_deployment_pct: float
    drawdown_halt_pct: float
    entry_score_threshold: float
    min_raw_return_cost_multiple: float
    earnings_blackout_sessions: int
    staleness_hours: int
    atr_window: int


@dataclass(frozen=True)
class DataConfig:
    cache_dir: str
    refetch_days: int
    min_coverage: float
    max_daily_move: float
    history_days: int
    quarantine_window_days: int
    drop_incomplete_last_bar: bool


@dataclass(frozen=True)
class VenueConfig:
    name: str
    benchmark: str
    costs: CostsConfig
    universe: UniverseConfig
    signals: SignalConfig
    regime: RegimeConfig
    portfolio: PortfolioConfig
    data: DataConfig


def load_venue_config(venue: str, config_dir: Path) -> VenueConfig:
    path = config_dir / f"{venue}.toml"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = tomllib.loads(path.read_text())
    signals = dict(raw["signals"])
    signals["momentum_windows"] = tuple(signals["momentum_windows"])
    signals["breakout_windows"] = tuple(signals["breakout_windows"])
    return VenueConfig(
        name=raw["venue"]["name"],
        benchmark=raw["venue"]["benchmark"],
        costs=CostsConfig(**raw["costs"]),
        universe=UniverseConfig(**raw["universe"]),
        signals=SignalConfig(**signals),
        regime=RegimeConfig(**raw["regime"]),
        portfolio=PortfolioConfig(**raw["portfolio"]),
        data=DataConfig(**raw["data"]),
    )
