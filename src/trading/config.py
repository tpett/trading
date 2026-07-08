"""Frozen per-venue configuration loaded from config/<venue>.toml.

Every tunable number in the system lives in TOML, never as a code constant.
Unknown or missing TOML keys raise TypeError via dataclass construction.
"""

from __future__ import annotations

import datetime
import tomllib
from dataclasses import dataclass
from pathlib import Path

VENUES = ["equities", "crypto"]


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
    # True when universe(as_of) returns real point-in-time membership
    # (equities PIT intervals): a member without data is a DATA problem and
    # degrades backtest coverage. False when the universe is a today-snapshot
    # (crypto): listing is inferred from data availability instead.
    point_in_time: bool
    # Which index columns of the equities membership CSV count as "in the
    # universe" (the CSV also carries sp400 rows). Defaults to today's live
    # behavior (sp500+ndx); sp400 is opt-in, added by a backtest experiment's
    # config, never a live/paper change. Ignored by venues (crypto) whose
    # universe() doesn't read that CSV.
    indices: tuple[str, ...] = ("sp500", "ndx")
    # Optional path to an allowlist file that FURTHER restricts the point-in-time
    # universe to the symbols it names (their intersection with PIT membership).
    # Used by the options-skew experiment to trade ONLY the names an options
    # gather covered: pointing this at that gather's samples.jsonl auto-restricts
    # the universe to exactly the gathered symbols, no hand-maintained list. ""
    # (the default) disables it, so live/paper is unaffected. See
    # trading.symbols.load_symbol_allowlist for the tolerant file format.
    symbols_allowlist_path: str = ""


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
    ranker: str  # key into trading.signals.registry.RANKERS; validated at load time


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
    earnings_blackout_enabled: bool
    staleness_hours: int
    atr_window: int
    session_close_buffer_minutes: int  # session venues only; see costs.trades_24_7
    exit_style: str  # "frozen" (default) or "trailing" (experiment flag)


@dataclass(frozen=True)
class DataConfig:
    cache_dir: str
    refetch_days: int
    min_coverage: float
    max_daily_move: float
    history_days: int
    quarantine_window_days: int
    drop_incomplete_last_bar: bool
    backfill_exchange: str  # ccxt exchange id for rows Kraken cannot serve; "" disables
    backfill_page_limit: int
    # Max calendar-day hole tolerated at the backfill/Kraken seam; doubles as
    # the head tolerance when judging whether Kraken alone covers a request.
    seam_max_gap_days: int
    # M4 fundamentals overlay. fundamentals_dir = "" means "this venue has no
    # fundamentals" (crypto); a ranker that requires fundamentals refuses to
    # load with it empty. refresh_days is the live top-up cadence -- data
    # plumbing, NOT a tunable hyperparameter (the walk-forward surface stays
    # entry_score_threshold x stop_atr_multiple only). Defaulted (like
    # membership_exit_buffer_days) so frozen test-venue TOMLs -- notably
    # tests/golden/golden.toml, which must stay byte-identical -- keep
    # loading; both real venue TOMLs still set them explicitly.
    fundamentals_dir: str = ""
    fundamentals_refresh_days: int = 0
    # Daily-bar source for the equities venue (crypto ignores it). "tiingo"
    # adds delisted tickers (survivorship-bias-free backtests); it needs an
    # API key in ~/.config/trading/config.toml or $TIINGO_API_KEY. Data
    # plumbing, NOT a tunable hyperparameter. Sources adjust prices on the
    # same split+dividend basis but not bit-identically, so a cache dir is
    # bound to one source via a .source marker -- switching sources means
    # pointing cache_dir somewhere fresh.
    bar_source: str = "yfinance"
    # Frozen/offline cache mode: when True the OHLCV cache serves every request
    # purely from its parquets and NEVER calls the fetch function -- a fully
    # backfilled cache can then drive experiments with zero network (enabling
    # e.g. a Tiingo cancellation). A request the cache does not cover raises
    # rather than silently returning a partial. Data plumbing, NOT a tunable
    # hyperparameter; defaulted so existing TOMLs keep loading (like bar_source).
    cache_offline: bool = False
    # Wall-clock ceiling (seconds) on one weekly refresh_fundamentals call:
    # data plumbing, same non-tunable status as refresh_days. A ~1,100-cik
    # companyfacts refresh over a slow/degraded network could otherwise run
    # long enough to threaten the run's own cadence; refresh_fundamentals
    # stops cleanly once this elapses, keeping symbols already processed and
    # deferring the remainder to next run (see trading.runner).
    fundamentals_refresh_budget_s: int = 900
    # IV-skew overlay (used only when [signals] ranker sets requires_skew --
    # skew_v1 / skew_change_v1; the default momentum_v1 never touches it). Path
    # to the gathered options samples.jsonl (one skew cell per symbol/decision-
    # month). "" means "this venue has no skew" (crypto, live equities); a
    # skew-requiring ranker refuses to load with it empty. Data plumbing, NOT a
    # tunable hyperparameter. Defaulted (like fundamentals_dir) so frozen
    # test-venue TOMLs keep loading.
    skew_samples: str = ""


@dataclass(frozen=True)
class BacktestConfig:
    """Spec: Backtesting & Validation. The tunable surface is exactly two
    hyperparameters (entry_score_threshold, stop_atr_multiple); their grids
    live here. Everything else is set by design, not fitted."""

    start: datetime.date
    holdout_start: datetime.date  # final 6 months; touched exactly once via --holdout
    train_months: int
    test_months: int
    entry_score_threshold_grid: tuple[float, ...]
    stop_atr_multiple_grid: tuple[float, ...]
    min_session_coverage: float  # skip a session when fewer members have data
    periods_per_year: int  # Sharpe annualization: 252 sessions / 365 UTC days
    stress_segments: tuple[tuple[datetime.date, datetime.date], ...]
    # Ticker-recycling guard (spec: survivorship): a symbol with no open-ended
    # membership interval has its cached bars truncated at (last interval end
    # + this many days) before prepare() hands them to the simulator, so a
    # delisted ticker later reused by an unrelated live company can never
    # contribute post-exit prices. Symbols still a current member (an
    # open-ended interval) are untouched.
    membership_exit_buffer_days: int = 30


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
    backtest: BacktestConfig


def load_venue_config(venue: str, config_dir: Path) -> VenueConfig:
    path = config_dir / f"{venue}.toml"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = tomllib.loads(path.read_text())
    signals = dict(raw["signals"])
    signals["momentum_windows"] = tuple(signals["momentum_windows"])
    signals["breakout_windows"] = tuple(signals["breakout_windows"])
    # Deferred import: trading.signals.registry imports trading.config (for
    # the SignalConfig type it dispatches on), so importing it at module
    # level here would be circular. Validating at load time (rather than at
    # first pipeline run) is the point: a typo'd ranker name must fail fast.
    # get_ranker's ValueError propagates as-is: the registry is the single
    # source of truth for the unknown-ranker message and known-names list.
    from trading.signals.registry import get_ranker

    spec = get_ranker(signals["ranker"])
    data_config = DataConfig(**raw["data"])
    if data_config.bar_source not in ("yfinance", "tiingo"):
        raise ValueError(
            f"data.bar_source must be 'yfinance' or 'tiingo', got {data_config.bar_source!r}"
        )
    if spec.requires_fundamentals and not data_config.fundamentals_dir:
        raise ValueError(f"ranker {signals['ranker']!r} requires [data] fundamentals_dir to be set")
    if spec.requires_fundamentals and data_config.fundamentals_refresh_days < 1:
        # 0 (the field's own default) means "never refresh", which for a
        # fundamentals-requiring ranker is a misconfiguration -- not a valid
        # cadence -- so it must fail at load, not silently never top up.
        # Validated against the CONSTRUCTED dataclass (not the raw TOML dict)
        # so a config that legitimately OMITS this key still fails here
        # (default 0 is itself invalid for a fundamentals-requiring ranker).
        raise ValueError(
            f"ranker {signals['ranker']!r} requires [data] fundamentals_refresh_days >= 1"
        )
    if spec.requires_fundamentals and data_config.fundamentals_refresh_budget_s < 1:
        # Same misconfiguration shape as refresh_days above, but the default
        # here is 900 (a valid budget), not 0 -- so this MUST read the
        # constructed dataclass rather than raw["data"].get(..., 0): a config
        # that omits the key entirely should load fine (getting the 900s
        # default), while an explicit 0 is still a real misconfiguration
        # (no wall-clock budget at all) and must still fail at load.
        raise ValueError(
            f"ranker {signals['ranker']!r} requires [data] fundamentals_refresh_budget_s >= 1"
        )
    if spec.requires_skew and not data_config.skew_samples:
        # Same fail-fast shape as the fundamentals_dir check: a skew-requiring
        # ranker with no [data] skew_samples is a misconfiguration, not a
        # legitimately empty channel, so it must fail at load rather than
        # silently rank every symbol neutral.
        raise ValueError(f"ranker {signals['ranker']!r} requires [data] skew_samples to be set")
    backtest = dict(raw["backtest"])
    backtest["entry_score_threshold_grid"] = tuple(backtest["entry_score_threshold_grid"])
    backtest["stop_atr_multiple_grid"] = tuple(backtest["stop_atr_multiple_grid"])
    backtest["stress_segments"] = tuple(
        (datetime.date.fromisoformat(a), datetime.date.fromisoformat(b))
        for a, b in backtest["stress_segments"]
    )
    portfolio = dict(raw["portfolio"])
    if portfolio["exit_style"] not in ("frozen", "trailing"):
        raise ValueError(
            f"portfolio.exit_style must be 'frozen' or 'trailing', got {portfolio['exit_style']!r}"
        )
    universe = dict(raw["universe"])
    if "indices" in universe:
        universe["indices"] = tuple(universe["indices"])
    return VenueConfig(
        name=raw["venue"]["name"],
        benchmark=raw["venue"]["benchmark"],
        costs=CostsConfig(**raw["costs"]),
        universe=UniverseConfig(**universe),
        signals=SignalConfig(**signals),
        regime=RegimeConfig(**raw["regime"]),
        portfolio=PortfolioConfig(**portfolio),
        data=data_config,
        backtest=BacktestConfig(**backtest),
    )
