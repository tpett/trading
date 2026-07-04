# M1: Foundation & Rankings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A user can run `trading rankings --venue equities` and `trading rankings --venue crypto` and see a ranked momentum table with sub-scores plus the regime state, computed from freshly fetched and Parquet-cached data.

**Architecture:** Single Python package with strictly separated layers: venue adapters (yfinance/ccxt behind one protocol) feed a Parquet cache-through layer; a pure, I/O-free signal engine turns bars into cross-sectional percentile features and a composite ranking; a regime gate classifies the benchmark; a thin pipeline + argparse CLI wires it together. No trading, no state files, no journal in M1 (those are M2). Point-in-time universe history is M3; M1 uses committed static universe files.

**Tech Stack:** Python 3.12, uv (env/deps), pandas, pyarrow, ccxt (Kraken), yfinance (pinned), rich (tables), pytest, ruff. CLI is **argparse** (stdlib): M1 has one subcommand, the spec's full CLI is simple flat subcommands, and argparse avoids another pinned dependency — rich is still used for output rendering. Typer would add ergonomics we don't need at this surface area.

## Global Constraints

These come from the approved spec (`docs/superpowers/specs/2026-07-04-momentum-swing-system-design.md`) and locked architectural decisions. Every task's requirements implicitly include this section.

- Repo root for all relative paths in this plan: `/Users/travis/Source/personal/trading/worktrees/momentum-system` (git worktree, branch `tpett/ai/momentum-system`). Run all commands from this directory.
- Python 3.12; `requires-python = ">=3.12"`. Env and deps managed with uv (`uv sync`, `uv run ...`). All deps pinned exactly in `pyproject.toml`; `uv.lock` is committed.
- src layout: package `src/trading/`, console script `trading` via pyproject entry point. Tests in `tests/` with pytest. Lint/format with ruff; run `uv run ruff check . && uv run ruff format .` before every commit.
- All timestamps UTC everywhere. Every DataFrame of bars has a tz-aware UTC `DatetimeIndex` and columns exactly `[open, high, low, close, volume]`. Adjusted prices for equities (`auto_adjust=True`, explicit).
- Signal engine and regime gate are pure: no I/O, no clock access — "now" (`as_of`) is always a parameter. Only the CLI may read the clock.
- Every spec number (fees, thresholds, position counts, windows) lives in per-venue TOML under `config/`, never as a code constant. M2's simulator numbers are included in the TOML/dataclasses now so M2 does not restructure config.
- Composite score = equal-weight sum of feature percentiles (weights fixed equal in v1 — do not add weight config).
- Error handling in M1 scope: partial universe fetch proceeds at >=90% coverage with exclusions listed, below 90% aborts with nonzero exit; a >40% (config) day-over-day close move quarantines the symbol at the data layer.
- `data/` (Parquet cache) is gitignored. The trailing 30-day (config) window is always re-fetched — adjusted history rewrites; cache is ephemeral for recent data.
- Commit after every task, one logical change per commit, message tagged `[AI]` (e.g. `Add venue adapter contract [AI]`). Executors append their standard Co-Authored-By footer.

## File Structure

```
pyproject.toml                          # project metadata, pinned deps, entry point, tool config
.python-version                         # 3.12
.gitignore                              # data/, .venv/, caches
README.md                               # 60-second usage doc (Task 12)
config/equities.toml                    # every equities number (signals, regime, costs, portfolio, data)
config/crypto.toml                      # every crypto number
scripts/build_equities_universe.py      # one-off: builds the committed equities universe CSV (provenance)
src/trading/__init__.py
src/trading/cli.py                      # argparse entry point, rich rendering, --json
src/trading/config.py                   # frozen dataclasses + TOML loader
src/trading/pipeline.py                 # build_rankings(): universe -> fetch -> quality -> signals -> regime
src/trading/data/__init__.py
src/trading/data/cache.py               # OhlcvCache: Parquet cache-through, trailing-window refetch
src/trading/data/quality.py             # coverage check, sanity quarantine
src/trading/venues/__init__.py          # make_adapter() factory (Task 12)
src/trading/venues/base.py              # SymbolInfo, VenueConstraints, VenueAdapter protocol, validate_ohlcv
src/trading/venues/equities.py          # yfinance adapter
src/trading/venues/crypto.py            # ccxt/Kraken adapter
src/trading/venues/universes/equities.csv   # committed static S&P500+NDX100 list (M1; point-in-time is M3)
src/trading/venues/universes/crypto.csv     # committed static Robinhood-listed pairs with status
src/trading/signals/__init__.py
src/trading/signals/features.py         # pure per-symbol feature functions
src/trading/signals/engine.py           # compute_features(), rank(), cross-sectional percentiles
src/trading/signals/regime.py           # Regime, compute_regime()
tests/test_cli.py
tests/test_config.py
tests/test_venue_base.py
tests/test_equities_adapter.py
tests/test_crypto_adapter.py
tests/test_cache.py
tests/test_quality.py
tests/test_features.py
tests/test_engine.py
tests/test_regime.py
tests/test_pipeline.py
```

---

### Task 1: Package scaffold and toolchain

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `src/trading/__init__.py`
- Create: `src/trading/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: installable package `trading` with console script `trading`; `trading.cli.main(argv: list[str] | None = None) -> int` (returned int is the process exit code); `uv run pytest` and `uv run ruff check .` work. Task 12 replaces `cli.py`'s body with the real `rankings` subcommand.

- [ ] **Step 1: Write project metadata and tool config**

Create `pyproject.toml`:

```toml
[project]
name = "trading"
version = "0.1.0"
description = "Momentum swing trading system: rankings, paper trading, backtests"
requires-python = ">=3.12"
dependencies = [
    "pandas==2.3.1",
    "pyarrow==20.0.0",
    "ccxt==4.4.95",
    "yfinance==0.2.65",
    "rich==14.0.0",
]

[project.scripts]
trading = "trading.cli:main"

[dependency-groups]
dev = [
    "pytest==8.4.1",
    "ruff==0.12.3",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/trading"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

Pins are the latest releases known at plan-writing time. If `uv sync` reports a pin does not exist, pin the closest current release instead and note the substitution in the commit message — but never unpin.

Create `.python-version`:

```
3.12
```

Create `.gitignore`:

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
dist/
data/
```

- [ ] **Step 2: Create the package and a minimal CLI**

Create `src/trading/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `src/trading/cli.py`:

```python
"""Command-line entry point. Subcommands are added milestone by milestone."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading", description="Momentum swing trading system")
    parser.add_subparsers(dest="command", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
```

- [ ] **Step 3: Install the environment**

Run: `uv sync`
Expected: creates `.venv/`, writes `uv.lock`, installs all pinned deps without resolver errors.

- [ ] **Step 4: Write the failing smoke test**

Create `tests/test_cli.py`:

```python
import pytest

from trading.cli import main


def test_no_command_exits_with_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_help_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 2 passed. (The implementation was written in Step 2; this verifies the scaffold end-to-end. If imports fail, the src layout or entry point is wrong — fix before proceeding.)

- [ ] **Step 6: Verify the console script**

Run: `uv run trading --help`
Expected: usage text starting `usage: trading` and exit code 0.

- [ ] **Step 7: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format .
git add pyproject.toml .python-version .gitignore uv.lock src/trading/__init__.py src/trading/cli.py tests/test_cli.py
git commit -m "Scaffold trading package with uv, pytest, ruff [AI]"
```

---

### Task 2: Per-venue TOML config and frozen dataclasses

**Files:**
- Create: `config/equities.toml`
- Create: `config/crypto.toml`
- Create: `src/trading/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `load_venue_config(venue: str, config_dir: Path) -> VenueConfig` and frozen dataclasses `VenueConfig` (fields: `name: str`, `benchmark: str`, `costs: CostsConfig`, `universe: UniverseConfig`, `signals: SignalConfig`, `regime: RegimeConfig`, `portfolio: PortfolioConfig`, `data: DataConfig`). Later tasks consume `config.signals` (`SignalConfig`), `config.regime` (`RegimeConfig`), `config.costs`, and `config.data`. The `portfolio` section is loaded now but only consumed by M2's simulator.

- [ ] **Step 1: Write the equities TOML**

Create `config/equities.toml`. Every number here is a spec value or a documented default — nothing is duplicated in code:

```toml
[venue]
name = "equities"
benchmark = "SPY"

[costs]
taker_fee_bps = 0.0      # zero commission (spec)
maker_fee_bps = 0.0
slippage_bps = 5.0       # spec: 5 bps slippage assumption
settlement_days = 1      # T+1 cash account (spec)
trades_24_7 = false

[universe]
min_dollar_volume = 20000000.0  # entry-filter floor (M2 applies it; spec: config)

[signals]
momentum_windows = [5, 20, 60]  # trading days (spec)
calendar_days = false           # lookbacks in trading-day rows
vol_window = 20                 # realized-vol window for vol adjustment
volume_week = 5                 # "current week" = 5 trading days
volume_baseline = 63            # trailing 3 months = 63 trading days
breakout_windows = [20, 60]     # spec: distance from 20/60-day highs
rsi_window = 14                 # spec: RSI-14 stretch
mean_window = 20                # spec: distance above 20-day mean
raw_return_days = 30            # raw 30-day return (calendar), feeds M2 crypto fee gate

[regime]
sma_fast = 50
sma_slow = 200
vol_window = 20
vol_lookback = 252
vol_high_percentile = 0.80
exposure_risk_on = 1.0
exposure_neutral = 0.5
exposure_risk_off = 0.0

[portfolio]  # consumed by the M2 simulator; shaped now so M2 does not restructure config
max_positions = 5
position_size_pct = 0.18
starting_balance = 1000.0
time_stop_bars = 20                 # 20 sessions (spec)
stop_atr_multiple = 1.5
regime_flush_atr_multiple = 1.0
cooldown_days = 7
max_daily_deployment_pct = 0.25
drawdown_halt_pct = 0.20
entry_score_threshold = 0.70        # tunable hyperparameter #1 (spec)
min_raw_return_cost_multiple = 0.0  # fee gate off for equities
earnings_blackout_sessions = 5
staleness_hours = 2

[data]
cache_dir = "data/equities"
refetch_days = 30       # trailing window always re-fetched (adjusted history rewrites)
min_coverage = 0.90     # below this, abort the run (spec)
max_daily_move = 0.40   # sanity quarantine bound (spec default)
history_days = 500      # calendar days fetched; covers 200-day SMA + 252-obs vol percentile
```

- [ ] **Step 2: Write the crypto TOML**

Create `config/crypto.toml`:

```toml
[venue]
name = "crypto"
benchmark = "BTC"  # fetched through the crypto adapter as BTC/USD on Kraken

[costs]
taker_fee_bps = 95.0   # 0.95% taker at <$10K 30-day volume (spec, June 2026 schedule)
maker_fee_bps = 50.0   # 0.50% maker
slippage_bps = 5.0
settlement_days = 0
trades_24_7 = true

[universe]
min_dollar_volume = 0.0  # no dollar-volume floor for crypto in v1

[signals]
momentum_windows = [7, 30, 90]  # calendar days (spec)
calendar_days = true
vol_window = 20
volume_week = 7                 # "current week" = 7 calendar days
volume_baseline = 90            # trailing 3 months = 90 calendar days
breakout_windows = [20, 60]
rsi_window = 14
mean_window = 20
raw_return_days = 30

[regime]
sma_fast = 50
sma_slow = 200
vol_window = 20
vol_lookback = 252
vol_high_percentile = 0.80
exposure_risk_on = 1.0
exposure_neutral = 0.5
exposure_risk_off = 0.0

[portfolio]  # consumed by the M2 simulator
max_positions = 3
position_size_pct = 0.30
starting_balance = 1000.0
time_stop_bars = 30                 # 30 calendar days = 30 daily bars (24/7 venue)
stop_atr_multiple = 1.5
regime_flush_atr_multiple = 1.0
cooldown_days = 7
max_daily_deployment_pct = 0.25
drawdown_halt_pct = 0.20
entry_score_threshold = 0.70
min_raw_return_cost_multiple = 3.0  # raw 30d return must exceed 3x round-trip cost (spec)
earnings_blackout_sessions = 0      # no earnings in crypto
staleness_hours = 6

[data]
cache_dir = "data/crypto"
refetch_days = 30
min_coverage = 0.90
max_daily_move = 0.40
history_days = 500
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_config.py`:

```python
import dataclasses
from pathlib import Path

import pytest

from trading.config import load_venue_config

CONFIG_DIR = Path("config")


def test_load_equities_config():
    config = load_venue_config("equities", CONFIG_DIR)
    assert config.name == "equities"
    assert config.benchmark == "SPY"
    assert config.costs.slippage_bps == 5.0
    assert config.costs.settlement_days == 1
    assert config.costs.trades_24_7 is False
    assert config.signals.momentum_windows == (5, 20, 60)
    assert config.signals.calendar_days is False
    assert config.signals.breakout_windows == (20, 60)
    assert config.regime.sma_slow == 200
    assert config.portfolio.max_positions == 5
    assert config.data.min_coverage == 0.90
    assert config.data.max_daily_move == 0.40


def test_load_crypto_config():
    config = load_venue_config("crypto", CONFIG_DIR)
    assert config.benchmark == "BTC"
    assert config.costs.taker_fee_bps == 95.0
    assert config.costs.maker_fee_bps == 50.0
    assert config.costs.trades_24_7 is True
    assert config.signals.momentum_windows == (7, 30, 90)
    assert config.signals.calendar_days is True
    assert config.portfolio.max_positions == 3
    assert config.portfolio.min_raw_return_cost_multiple == 3.0


def test_config_is_frozen():
    config = load_venue_config("equities", CONFIG_DIR)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.name = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.signals.vol_window = 99


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_venue_config("equities", tmp_path)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.config'`.

- [ ] **Step 5: Implement the config module**

Create `src/trading/config.py`:

```python
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


@dataclass(frozen=True)
class DataConfig:
    cache_dir: str
    refetch_days: int
    min_coverage: float
    max_daily_move: float
    history_days: int


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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 4 passed.

- [ ] **Step 7: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format .
git add config/equities.toml config/crypto.toml src/trading/config.py tests/test_config.py
git commit -m "Add per-venue TOML config with frozen dataclasses [AI]"
```

---
### Task 3: Venue adapter contract (`base.py`)

**Files:**
- Create: `src/trading/venues/__init__.py`
- Create: `src/trading/venues/base.py`
- Test: `tests/test_venue_base.py`

**Interfaces:**
- Consumes: nothing.
- Produces (locked protocol — later milestones depend on these exact shapes):
  - `SymbolInfo(symbol: str, status: Literal["tradable", "sell_only", "untradable"])` — frozen dataclass.
  - `VenueConstraints(taker_fee_bps: float, maker_fee_bps: float, slippage_bps: float, settlement_days: int, trades_24_7: bool)` — frozen dataclass.
  - `VenueAdapter` Protocol: `universe(as_of: datetime.date) -> list[SymbolInfo]`, `constraints() -> VenueConstraints`, `fetch_ohlcv(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame`.
  - `OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]`, `validate_ohlcv(df) -> pd.DataFrame`, `DataFetchError(RuntimeError)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_venue_base.py`:

```python
import dataclasses

import pandas as pd
import pytest

from trading.venues.base import OHLCV_COLUMNS, SymbolInfo, VenueConstraints, validate_ohlcv


def _good_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=3, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0}, index=idx
    )


def test_symbol_info_is_frozen():
    info = SymbolInfo(symbol="AAPL", status="tradable")
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.status = "sell_only"


def test_venue_constraints_is_frozen():
    c = VenueConstraints(
        taker_fee_bps=95.0, maker_fee_bps=50.0, slippage_bps=5.0,
        settlement_days=0, trades_24_7=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.taker_fee_bps = 0.0


def test_validate_ohlcv_accepts_contract_frame():
    df = _good_frame()
    assert validate_ohlcv(df) is df


def test_validate_ohlcv_rejects_wrong_columns():
    df = _good_frame().rename(columns={"close": "Close"})
    with pytest.raises(ValueError, match="columns"):
        validate_ohlcv(df)


def test_validate_ohlcv_rejects_naive_index():
    df = _good_frame()
    df.index = df.index.tz_localize(None)
    with pytest.raises(ValueError, match="UTC"):
        validate_ohlcv(df)


def test_validate_ohlcv_rejects_unsorted_index():
    df = _good_frame().iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        validate_ohlcv(df)


def test_ohlcv_columns_locked():
    assert OHLCV_COLUMNS == ["open", "high", "low", "close", "volume"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_venue_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.venues'`.

- [ ] **Step 3: Implement the contract module**

Create `src/trading/venues/__init__.py` as an empty file (the `make_adapter` factory is added in Task 12).

Create `src/trading/venues/base.py`:

```python
"""Venue adapter contract shared by all venues and all milestones.

Every adapter returns bars as: tz-aware UTC DatetimeIndex, columns exactly
[open, high, low, close, volume], sorted ascending. Equities prices are
corporate-action adjusted.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

SymbolStatus = Literal["tradable", "sell_only", "untradable"]


class DataFetchError(RuntimeError):
    """A symbol's bars could not be fetched or were empty."""


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    status: SymbolStatus


@dataclass(frozen=True)
class VenueConstraints:
    taker_fee_bps: float
    maker_fee_bps: float
    slippage_bps: float
    settlement_days: int
    trades_24_7: bool


class VenueAdapter(Protocol):
    def universe(self, as_of: datetime.date) -> list[SymbolInfo]: ...

    def constraints(self) -> VenueConstraints: ...

    def fetch_ohlcv(
        self, symbol: str, start: datetime.date, end: datetime.date
    ) -> pd.DataFrame: ...


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError unless df matches the adapter OHLCV contract."""
    if list(df.columns) != OHLCV_COLUMNS:
        raise ValueError(f"OHLCV columns must be {OHLCV_COLUMNS}, got {list(df.columns)}")
    index = df.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or str(index.tz) != "UTC":
        raise ValueError("OHLCV index must be a tz-aware UTC DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV index must be sorted ascending")
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_venue_base.py -v`
Expected: 7 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/venues/__init__.py src/trading/venues/base.py tests/test_venue_base.py
git commit -m "Add venue adapter contract with OHLCV validation [AI]"
```

---

### Task 4: Equities venue adapter and static universe

**Files:**
- Create: `src/trading/venues/equities.py`
- Create: `scripts/build_equities_universe.py`
- Create: `src/trading/venues/universes/equities.csv` (generated by the script, then committed)
- Test: `tests/test_equities_adapter.py`

**Interfaces:**
- Consumes: `VenueConfig` from `trading.config` (Task 2); `SymbolInfo`, `VenueConstraints`, `OHLCV_COLUMNS`, `validate_ohlcv`, `DataFetchError` from `trading.venues.base` (Task 3).
- Produces: `EquitiesAdapter(config: VenueConfig, universe_csv: Path | None = None)` implementing the `VenueAdapter` protocol. Module-level `_yf_download(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame` — the only network touchpoint, monkeypatched in every test. `DEFAULT_UNIVERSE_CSV: Path` module constant.

**Notes for the implementer:** M1 uses a committed static CSV of current S&P 500 + Nasdaq-100 members (point-in-time membership history is Milestone 3 — do not build it now). The build script records provenance per the spec's open item. yfinance is called with `auto_adjust=True` explicitly so all prices are adjusted (spec: adjusted OHLC used everywhere).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_equities_adapter.py`:

```python
import datetime
from pathlib import Path

import pandas as pd
import pytest

from trading.config import load_venue_config
from trading.venues.base import OHLCV_COLUMNS, DataFetchError, SymbolInfo, VenueConstraints
from trading.venues.equities import EquitiesAdapter

CONFIG = load_venue_config("equities", Path("config"))


def _yf_style_frame(symbol: str) -> pd.DataFrame:
    """Mimic yfinance.download output: naive index, MultiIndex (Price, Ticker) columns."""
    idx = pd.date_range("2026-01-05", periods=5, freq="B")  # naive, like yfinance
    data = {
        ("Open", symbol): [10.0, 10.5, 10.2, 10.8, 11.0],
        ("High", symbol): [10.6, 10.9, 10.7, 11.2, 11.4],
        ("Low", symbol): [9.8, 10.1, 9.9, 10.5, 10.7],
        ("Close", symbol): [10.5, 10.2, 10.6, 11.0, 11.2],
        ("Volume", symbol): [1e6, 1.1e6, 9e5, 1.3e6, 1.2e6],
    }
    return pd.DataFrame(data, index=idx)


def test_universe_reads_symbols_from_csv(tmp_path):
    csv = tmp_path / "universe.csv"
    csv.write_text("symbol\nAAPL\nMSFT\nNVDA\n")
    adapter = EquitiesAdapter(CONFIG, universe_csv=csv)
    infos = adapter.universe(datetime.date(2026, 7, 1))
    assert infos == [
        SymbolInfo("AAPL", "tradable"),
        SymbolInfo("MSFT", "tradable"),
        SymbolInfo("NVDA", "tradable"),
    ]


def test_constraints_come_from_config():
    adapter = EquitiesAdapter(CONFIG)
    assert adapter.constraints() == VenueConstraints(
        taker_fee_bps=0.0, maker_fee_bps=0.0, slippage_bps=5.0,
        settlement_days=1, trades_24_7=False,
    )


def test_fetch_ohlcv_normalizes_yfinance_frame(monkeypatch):
    monkeypatch.setattr(
        "trading.venues.equities._yf_download", lambda s, start, end: _yf_style_frame(s)
    )
    adapter = EquitiesAdapter(CONFIG)
    df = adapter.fetch_ohlcv("AAPL", datetime.date(2026, 1, 5), datetime.date(2026, 1, 9))
    assert list(df.columns) == OHLCV_COLUMNS
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == 11.2
    assert len(df) == 5


def test_fetch_ohlcv_slices_to_requested_range(monkeypatch):
    monkeypatch.setattr(
        "trading.venues.equities._yf_download", lambda s, start, end: _yf_style_frame(s)
    )
    adapter = EquitiesAdapter(CONFIG)
    df = adapter.fetch_ohlcv("AAPL", datetime.date(2026, 1, 6), datetime.date(2026, 1, 8))
    assert df.index.min() == pd.Timestamp("2026-01-06", tz="UTC")
    assert df.index.max() == pd.Timestamp("2026-01-08", tz="UTC")


def test_fetch_ohlcv_empty_raises(monkeypatch):
    monkeypatch.setattr(
        "trading.venues.equities._yf_download", lambda s, start, end: pd.DataFrame()
    )
    adapter = EquitiesAdapter(CONFIG)
    with pytest.raises(DataFetchError):
        adapter.fetch_ohlcv("AAPL", datetime.date(2026, 1, 5), datetime.date(2026, 1, 9))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_equities_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.venues.equities'`.

- [ ] **Step 3: Implement the adapter**

Create `src/trading/venues/equities.py`:

```python
"""Equities venue: yfinance daily bars, static S&P500+NDX100 universe CSV.

M1 uses a committed static membership snapshot (see universes/equities.csv,
built by scripts/build_equities_universe.py). Point-in-time membership
history is Milestone 3. Prices are corporate-action adjusted
(auto_adjust=True) so signals, stops and fills share one price basis.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

from trading.config import VenueConfig
from trading.venues.base import (
    OHLCV_COLUMNS,
    DataFetchError,
    SymbolInfo,
    VenueConstraints,
    validate_ohlcv,
)

DEFAULT_UNIVERSE_CSV = Path(__file__).parent / "universes" / "equities.csv"


def _yf_download(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Network touchpoint, isolated for monkeypatching. yfinance `end` is exclusive."""
    import yfinance as yf

    return yf.download(
        symbol,
        start=start.isoformat(),
        end=(end + datetime.timedelta(days=1)).isoformat(),
        auto_adjust=True,
        actions=False,
        progress=False,
    )


class EquitiesAdapter:
    def __init__(self, config: VenueConfig, universe_csv: Path | None = None):
        self._config = config
        self._universe_csv = universe_csv or DEFAULT_UNIVERSE_CSV

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        # as_of is part of the locked protocol; the static M1 snapshot ignores it.
        df = pd.read_csv(self._universe_csv)
        return [SymbolInfo(symbol=s, status="tradable") for s in df["symbol"]]

    def constraints(self) -> VenueConstraints:
        c = self._config.costs
        return VenueConstraints(
            taker_fee_bps=c.taker_fee_bps,
            maker_fee_bps=c.maker_fee_bps,
            slippage_bps=c.slippage_bps,
            settlement_days=c.settlement_days,
            trades_24_7=c.trades_24_7,
        )

    def fetch_ohlcv(
        self, symbol: str, start: datetime.date, end: datetime.date
    ) -> pd.DataFrame:
        raw = _yf_download(symbol, start, end)
        if raw is None or raw.empty:
            raise DataFetchError(f"no equities data for {symbol}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[OHLCV_COLUMNS].astype("float64")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index().loc[
            pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")
        ]
        return validate_ohlcv(df)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_equities_adapter.py -v`
Expected: 5 passed.

- [ ] **Step 5: Write the universe build script**

Create `scripts/build_equities_universe.py`:

```python
"""One-off builder for src/trading/venues/universes/equities.csv.

Provenance (spec open item): current S&P 500 and Nasdaq-100 constituent
tables from en.wikipedia.org, fetched on the run date. This is a static M1
snapshot; point-in-time membership history is Milestone 3.

Run from the repo root:
    uv run --with lxml python scripts/build_equities_universe.py
"""

from pathlib import Path

import pandas as pd

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
OUT = Path("src/trading/venues/universes/equities.csv")


def main() -> None:
    sp500 = pd.read_html(SP500_URL)[0]["Symbol"].tolist()
    ndx_tables = pd.read_html(NDX_URL)
    ndx = next(t for t in ndx_tables if "Ticker" in t.columns)["Ticker"].tolist()
    # yfinance uses '-' for share classes (BRK.B -> BRK-B).
    symbols = sorted({str(s).strip().replace(".", "-") for s in sp500 + ndx})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("symbol\n" + "\n".join(symbols) + "\n")
    print(f"wrote {len(symbols)} symbols to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the script (network required) and sanity-check the CSV**

Run: `uv run --with lxml python scripts/build_equities_universe.py`
Expected: `wrote N symbols to src/trading/venues/universes/equities.csv` with N between 520 and 620 (S&P 500 + NDX 100 minus overlap).

Run: `head -5 src/trading/venues/universes/equities.csv && grep -c '' src/trading/venues/universes/equities.csv && grep '\.' src/trading/venues/universes/equities.csv | head`
Expected: header `symbol` then tickers; line count = N+1; the final grep prints nothing (no dots remain).

- [ ] **Step 7: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/venues/equities.py scripts/build_equities_universe.py src/trading/venues/universes/equities.csv tests/test_equities_adapter.py
git commit -m "Add equities adapter with yfinance and static universe snapshot [AI]"
```

---

### Task 5: Crypto venue adapter and static universe

**Files:**
- Create: `src/trading/venues/crypto.py`
- Create: `src/trading/venues/universes/crypto.csv`
- Test: `tests/test_crypto_adapter.py`

**Interfaces:**
- Consumes: `VenueConfig` (Task 2); `SymbolInfo`, `VenueConstraints`, `OHLCV_COLUMNS`, `validate_ohlcv`, `DataFetchError` (Task 3).
- Produces: `CryptoAdapter(config: VenueConfig, universe_csv: Path | None = None)` implementing `VenueAdapter`. Module-level `_kraken_fetch(pair: str, since_ms: int) -> list[list[float]]` — the only network touchpoint, monkeypatched in tests. `DEFAULT_UNIVERSE_CSV: Path`. Symbols map to Kraken pairs as `f"{symbol}/USD"`.

**Notes for the implementer:** The universe is a committed, hand-maintained snapshot of Robinhood-listed coins with a `status` column (spec: model `sell_only`/`untradable`; sync from a maintained list, never hardcode in code — the CSV *is* that maintained list in M1). A Robinhood symbol not listed on Kraken simply fails its fetch and is handled by the >=90% coverage rule (Task 7/11) — no special-casing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_crypto_adapter.py`:

```python
import datetime
from pathlib import Path

import pandas as pd
import pytest

from trading.config import load_venue_config
from trading.venues.base import OHLCV_COLUMNS, DataFetchError, SymbolInfo, VenueConstraints
from trading.venues.crypto import DEFAULT_UNIVERSE_CSV, CryptoAdapter

CONFIG = load_venue_config("crypto", Path("config"))


def _kraken_rows(n: int, end: datetime.date) -> list[list[float]]:
    """Mimic ccxt fetch_ohlcv: [ms_timestamp, open, high, low, close, volume] rows."""
    start_ts = pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=n - 1)
    return [
        [
            int((start_ts + pd.Timedelta(days=i)).timestamp() * 1000),
            100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1000.0,
        ]
        for i in range(n)
    ]


def test_universe_reads_symbols_and_statuses(tmp_path):
    csv = tmp_path / "universe.csv"
    csv.write_text("symbol,status\nBTC,tradable\nETH,tradable\nSOL,sell_only\n")
    adapter = CryptoAdapter(CONFIG, universe_csv=csv)
    infos = adapter.universe(datetime.date(2026, 7, 1))
    assert infos == [
        SymbolInfo("BTC", "tradable"),
        SymbolInfo("ETH", "tradable"),
        SymbolInfo("SOL", "sell_only"),
    ]


def test_universe_rejects_unknown_status(tmp_path):
    csv = tmp_path / "universe.csv"
    csv.write_text("symbol,status\nBTC,halted\n")
    adapter = CryptoAdapter(CONFIG, universe_csv=csv)
    with pytest.raises(ValueError, match="halted"):
        adapter.universe(datetime.date(2026, 7, 1))


def test_committed_universe_csv_is_valid():
    adapter = CryptoAdapter(CONFIG, universe_csv=DEFAULT_UNIVERSE_CSV)
    infos = adapter.universe(datetime.date(2026, 7, 1))
    assert len(infos) >= 40
    assert SymbolInfo("BTC", "tradable") in infos


def test_constraints_come_from_config():
    adapter = CryptoAdapter(CONFIG)
    assert adapter.constraints() == VenueConstraints(
        taker_fee_bps=95.0, maker_fee_bps=50.0, slippage_bps=5.0,
        settlement_days=0, trades_24_7=True,
    )


def test_fetch_ohlcv_maps_symbol_to_kraken_usd_pair(monkeypatch):
    seen: list[str] = []

    def fake_fetch(pair: str, since_ms: int) -> list[list[float]]:
        seen.append(pair)
        return _kraken_rows(10, datetime.date(2026, 7, 1))

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_fetch)
    adapter = CryptoAdapter(CONFIG)
    df = adapter.fetch_ohlcv("BTC", datetime.date(2026, 6, 22), datetime.date(2026, 7, 1))
    assert seen == ["BTC/USD"]
    assert list(df.columns) == OHLCV_COLUMNS
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert len(df) == 10
    assert df["close"].iloc[-1] == 109.5


def test_fetch_ohlcv_slices_to_requested_range(monkeypatch):
    monkeypatch.setattr(
        "trading.venues.crypto._kraken_fetch",
        lambda pair, since_ms: _kraken_rows(10, datetime.date(2026, 7, 1)),
    )
    adapter = CryptoAdapter(CONFIG)
    df = adapter.fetch_ohlcv("BTC", datetime.date(2026, 6, 25), datetime.date(2026, 6, 30))
    assert df.index.min() == pd.Timestamp("2026-06-25", tz="UTC")
    assert df.index.max() == pd.Timestamp("2026-06-30", tz="UTC")


def test_fetch_ohlcv_empty_raises(monkeypatch):
    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", lambda pair, since_ms: [])
    adapter = CryptoAdapter(CONFIG)
    with pytest.raises(DataFetchError):
        adapter.fetch_ohlcv("BTC", datetime.date(2026, 6, 22), datetime.date(2026, 7, 1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_crypto_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.venues.crypto'`.

- [ ] **Step 3: Write the committed universe CSV**

Create `src/trading/venues/universes/crypto.csv`. This is the hand-maintained Robinhood-listed snapshot (statuses updated by hand when Robinhood flags a coin `sell_only`/`untradable`; the spec's live sync is a later-milestone concern). All entries below are listed on both Robinhood and Kraken as of 2026-07:

```csv
symbol,status
AAVE,tradable
ADA,tradable
ALGO,tradable
ARB,tradable
AVAX,tradable
BCH,tradable
BONK,tradable
BTC,tradable
COMP,tradable
CRV,tradable
DOGE,tradable
DOT,tradable
ETC,tradable
ETH,tradable
FET,tradable
FLOKI,tradable
HBAR,tradable
ICP,tradable
INJ,tradable
JTO,tradable
JUP,tradable
LDO,tradable
LINK,tradable
LTC,tradable
MEW,tradable
MOODENG,tradable
NEAR,tradable
ONDO,tradable
OP,tradable
PENGU,tradable
PEPE,tradable
PNUT,tradable
POPCAT,tradable
PYTH,tradable
RENDER,tradable
SEI,tradable
SHIB,tradable
SOL,tradable
SUI,tradable
TIA,tradable
TRUMP,tradable
UNI,tradable
USDC,tradable
WIF,tradable
WLD,tradable
XLM,tradable
XRP,tradable
XTZ,tradable
```

Before committing, spot-check the list against Robinhood's current crypto listing page and Kraken's USD pairs; add/remove rows as needed (this file is data, not code — edits to it never require code changes).

- [ ] **Step 4: Implement the adapter**

Create `src/trading/venues/crypto.py`:

```python
"""Crypto venue: ccxt/Kraken daily UTC bars, Robinhood-listed universe CSV.

The universe CSV is the maintained Robinhood listing snapshot with per-symbol
status (tradable / sell_only / untradable). Kraken is the M1 data source;
Bitstamp (Robinhood's routing venue) is a config-free swap later since both
sit behind ccxt's fetch_ohlcv.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import get_args

import pandas as pd

from trading.config import VenueConfig
from trading.venues.base import (
    OHLCV_COLUMNS,
    DataFetchError,
    SymbolInfo,
    SymbolStatus,
    VenueConstraints,
    validate_ohlcv,
)

DEFAULT_UNIVERSE_CSV = Path(__file__).parent / "universes" / "crypto.csv"

_VALID_STATUSES = set(get_args(SymbolStatus))
_KRAKEN_DAILY_LIMIT = 720  # Kraken returns at most 720 daily candles per call


def _kraken_fetch(pair: str, since_ms: int) -> list[list[float]]:
    """Network touchpoint, isolated for monkeypatching."""
    import ccxt

    exchange = ccxt.kraken({"enableRateLimit": True})
    return exchange.fetch_ohlcv(pair, timeframe="1d", since=since_ms, limit=_KRAKEN_DAILY_LIMIT)


class CryptoAdapter:
    def __init__(self, config: VenueConfig, universe_csv: Path | None = None):
        self._config = config
        self._universe_csv = universe_csv or DEFAULT_UNIVERSE_CSV

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        # as_of is part of the locked protocol; the static M1 snapshot ignores it.
        df = pd.read_csv(self._universe_csv)
        infos: list[SymbolInfo] = []
        for row in df.itertuples(index=False):
            if row.status not in _VALID_STATUSES:
                raise ValueError(f"unknown status {row.status!r} for {row.symbol}")
            infos.append(SymbolInfo(symbol=row.symbol, status=row.status))
        return infos

    def constraints(self) -> VenueConstraints:
        c = self._config.costs
        return VenueConstraints(
            taker_fee_bps=c.taker_fee_bps,
            maker_fee_bps=c.maker_fee_bps,
            slippage_bps=c.slippage_bps,
            settlement_days=c.settlement_days,
            trades_24_7=c.trades_24_7,
        )

    def fetch_ohlcv(
        self, symbol: str, start: datetime.date, end: datetime.date
    ) -> pd.DataFrame:
        pair = f"{symbol}/USD"
        since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        rows = _kraken_fetch(pair, since_ms)
        if not rows:
            raise DataFetchError(f"no crypto data for {pair}")
        df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
        df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
        df.index.name = None
        df = df.astype("float64").sort_index()
        df = df.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]
        return validate_ohlcv(df)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_crypto_adapter.py -v`
Expected: 7 passed.

- [ ] **Step 6: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/venues/crypto.py src/trading/venues/universes/crypto.csv tests/test_crypto_adapter.py
git commit -m "Add crypto adapter with ccxt Kraken and Robinhood universe snapshot [AI]"
```

---
### Task 6: Parquet OHLCV cache

**Files:**
- Create: `src/trading/data/__init__.py`
- Create: `src/trading/data/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: nothing from earlier tasks directly (the fetch function is injected, so the cache is adapter-agnostic).
- Produces: `OhlcvCache(cache_dir: Path, refetch_days: int)` with methods `path_for(symbol: str) -> Path` and `fetch(symbol: str, start: datetime.date, end: datetime.date, fetch_fn: Callable[[str, datetime.date, datetime.date], pd.DataFrame]) -> pd.DataFrame`. Task 11 passes `adapter.fetch_ohlcv` as `fetch_fn`.

**Notes for the implementer:** Cache-through semantics per spec: the trailing `refetch_days` window is *always* re-fetched because corporate actions rewrite adjusted history — cached rows inside that window are discarded and replaced. Rows older than the cutoff are served from Parquet. Writes are atomic (temp file + `os.replace`) so a crashed run never leaves a torn cache file.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cache.py`:

```python
import datetime

import pandas as pd

from trading.data.cache import OhlcvCache


def _frame(start: datetime.date, end: datetime.date, value: float) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": value, "high": value, "low": value, "close": value, "volume": value},
        index=idx,
    )


class RecordingFetcher:
    def __init__(self, value: float):
        self.value = value
        self.calls: list[tuple[str, datetime.date, datetime.date]] = []

    def __call__(self, symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        return _frame(start, end, self.value)


START = datetime.date(2026, 1, 1)
END = datetime.date(2026, 3, 1)


def test_cold_cache_fetches_full_range_and_writes_parquet(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    fetcher = RecordingFetcher(1.0)
    df = cache.fetch("AAPL", START, END, fetcher)
    assert fetcher.calls == [("AAPL", START, END)]
    assert cache.path_for("AAPL").exists()
    assert df.index.min() == pd.Timestamp(START, tz="UTC")
    assert df.index.max() == pd.Timestamp(END, tz="UTC")


def test_warm_cache_refetches_only_trailing_window(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    cache.fetch("AAPL", START, END, RecordingFetcher(1.0))

    second = RecordingFetcher(2.0)
    df = cache.fetch("AAPL", START, END, second)

    cutoff = END - datetime.timedelta(days=30)  # 2026-01-30
    assert second.calls == [("AAPL", cutoff, END)]
    # Rows before the cutoff come from the cache (old value)...
    assert df.loc[pd.Timestamp("2026-01-15", tz="UTC"), "close"] == 1.0
    # ...rows in the trailing window come from the fresh fetch (new value).
    assert df.loc[pd.Timestamp("2026-02-15", tz="UTC"), "close"] == 2.0
    assert not df.index.duplicated().any()


def test_result_is_sliced_to_requested_range(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    cache.fetch("AAPL", START, END, RecordingFetcher(1.0))
    df = cache.fetch("AAPL", datetime.date(2026, 2, 1), END, RecordingFetcher(2.0))
    assert df.index.min() == pd.Timestamp("2026-02-01", tz="UTC")
    assert df.index.max() == pd.Timestamp(END, tz="UTC")


def test_cache_missing_early_history_triggers_full_refetch(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    cache.fetch("AAPL", datetime.date(2026, 2, 1), END, RecordingFetcher(1.0))

    second = RecordingFetcher(2.0)
    cache.fetch("AAPL", START, END, second)  # asks for more history than cached
    assert second.calls == [("AAPL", START, END)]


def test_path_for_sanitizes_pair_symbols(tmp_path):
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    assert cache.path_for("BTC/USD").name == "BTC-USD.parquet"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.data'`.

- [ ] **Step 3: Implement the cache**

Create `src/trading/data/__init__.py` as an empty file.

Create `src/trading/data/cache.py`:

```python
"""Parquet cache-through layer for OHLCV bars.

The trailing `refetch_days` window is always re-fetched: adjusted equity
history rewrites on corporate actions, so recent cache contents are treated
as ephemeral (spec). Older rows are served from per-symbol Parquet files.
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd

FetchFn = Callable[[str, datetime.date, datetime.date], pd.DataFrame]

# First cached bar may legitimately start after the requested date (weekends,
# holidays, listing date); tolerate this gap before declaring history missing.
_START_TOLERANCE = pd.Timedelta(days=5)


class OhlcvCache:
    def __init__(self, cache_dir: Path, refetch_days: int):
        self._dir = cache_dir
        self._refetch_days = refetch_days
        self._dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str) -> Path:
        return self._dir / f"{symbol.replace('/', '-')}.parquet"

    def fetch(
        self,
        symbol: str,
        start: datetime.date,
        end: datetime.date,
        fetch_fn: FetchFn,
    ) -> pd.DataFrame:
        path = self.path_for(symbol)
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        cutoff = end - datetime.timedelta(days=self._refetch_days)

        keep: pd.DataFrame | None = None
        fetch_start = start
        if path.exists():
            cached = pd.read_parquet(path)
            if not cached.empty and cached.index.min() <= start_ts + _START_TOLERANCE:
                keep = cached[cached.index < pd.Timestamp(cutoff, tz="UTC")]
                fetch_start = max(cutoff, start)

        fresh = fetch_fn(symbol, fetch_start, end)
        merged = fresh if keep is None or keep.empty else pd.concat([keep, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()

        tmp = path.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp)
        os.replace(tmp, path)  # atomic: never leave a torn cache file

        return merged.loc[start_ts:end_ts]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/data/__init__.py src/trading/data/cache.py tests/test_cache.py
git commit -m "Add Parquet cache-through layer with trailing-window refetch [AI]"
```

---

### Task 7: Data quality — coverage check and sanity quarantine

**Files:**
- Create: `src/trading/data/quality.py`
- Test: `tests/test_quality.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure functions over plain data).
- Produces:
  - `CoverageReport(requested: int, fetched: int, ratio: float, ok: bool, missing: tuple[str, ...])` — frozen dataclass.
  - `check_coverage(requested: Sequence[str], fetched: Iterable[str], min_coverage: float) -> CoverageReport`.
  - `quarantine_outliers(bars: Mapping[str, pd.DataFrame], max_daily_move: float) -> tuple[dict[str, pd.DataFrame], tuple[str, ...]]` — returns (clean bars, quarantined symbols sorted).

**Notes for the implementer:** Spec rules mapped to M1: proceed at >=90% universe coverage with exclusions listed, abort below (the abort itself happens in Task 11's pipeline — this module only reports). A day-over-day close move beyond `max_daily_move` (default 0.40) quarantines the symbol. Prices are already corporate-action adjusted (Task 4), so a legitimate split never trips this bound — any remaining >40% adjusted move is treated as bad data until manually cleared (manual clearing UX arrives with M2's journal/digest).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_quality.py`:

```python
import pandas as pd

from trading.data.quality import CoverageReport, check_coverage, quarantine_outliers


def _bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=len(closes), freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": [1e6] * len(closes)},
        index=idx,
    )


def test_coverage_at_exactly_90_percent_is_ok():
    requested = [f"S{i}" for i in range(10)]
    report = check_coverage(requested, requested[:9], min_coverage=0.90)
    assert report == CoverageReport(requested=10, fetched=9, ratio=0.9, ok=True, missing=("S9",))


def test_coverage_below_90_percent_is_not_ok():
    requested = [f"S{i}" for i in range(10)]
    report = check_coverage(requested, requested[:8], min_coverage=0.90)
    assert report.ok is False
    assert report.missing == ("S8", "S9")


def test_coverage_with_empty_universe_is_not_ok():
    report = check_coverage([], [], min_coverage=0.90)
    assert report.ok is False


def test_quarantine_flags_moves_beyond_bound():
    # 100 -> 155 is a +55% day: quarantined at the 40% bound.
    bars = {"BAD": _bars([100.0, 155.0, 150.0]), "OK": _bars([100.0, 139.0, 140.0])}
    clean, quarantined = quarantine_outliers(bars, max_daily_move=0.40)
    assert quarantined == ("BAD",)
    assert set(clean) == {"OK"}


def test_quarantine_flags_crashes_too():
    # 100 -> 55 is a -45% day.
    bars = {"CRASH": _bars([100.0, 55.0, 56.0])}
    clean, quarantined = quarantine_outliers(bars, max_daily_move=0.40)
    assert quarantined == ("CRASH",)
    assert clean == {}


def test_quarantine_passes_moves_at_the_bound():
    bars = {"EDGE": _bars([100.0, 140.0, 141.0])}  # exactly +40%
    clean, quarantined = quarantine_outliers(bars, max_daily_move=0.40)
    assert quarantined == ()
    assert set(clean) == {"EDGE"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.data.quality'`.

- [ ] **Step 3: Implement the quality module**

Create `src/trading/data/quality.py`:

```python
"""Data-quality rules applied between fetch and signals (spec: Error Handling).

- Coverage: a run proceeds only if >= min_coverage of the universe fetched;
  excluded symbols are reported, never silently dropped.
- Sanity quarantine: prices are adjusted, so a day-over-day close move beyond
  max_daily_move without a corporate action is bad data; the symbol is
  excluded from ranking and surfaced as a warning.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CoverageReport:
    requested: int
    fetched: int
    ratio: float
    ok: bool
    missing: tuple[str, ...]


def check_coverage(
    requested: Sequence[str], fetched: Iterable[str], min_coverage: float
) -> CoverageReport:
    fetched_set = set(fetched)
    missing = tuple(sorted(s for s in requested if s not in fetched_set))
    count = len(requested) - len(missing)
    ratio = count / len(requested) if requested else 0.0
    return CoverageReport(
        requested=len(requested),
        fetched=count,
        ratio=ratio,
        ok=bool(requested) and ratio >= min_coverage,
        missing=missing,
    )


def quarantine_outliers(
    bars: Mapping[str, pd.DataFrame], max_daily_move: float
) -> tuple[dict[str, pd.DataFrame], tuple[str, ...]]:
    clean: dict[str, pd.DataFrame] = {}
    quarantined: list[str] = []
    for symbol, df in bars.items():
        moves = df["close"].pct_change().abs()
        if (moves > max_daily_move).any():
            quarantined.append(symbol)
        else:
            clean[symbol] = df
    return clean, tuple(sorted(quarantined))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_quality.py -v`
Expected: 6 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/data/quality.py tests/test_quality.py
git commit -m "Add coverage check and sanity quarantine rules [AI]"
```

---

### Task 8: Signal feature functions (pure, hand-computed fixtures)

**Files:**
- Create: `src/trading/signals/__init__.py`
- Create: `src/trading/signals/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure pandas functions; no config, no I/O).
- Produces (all take series already truncated to `<= as_of` by the caller and return `float`, `math.nan` when history is insufficient):
  - `vol_adjusted_return(close: pd.Series, lookback: int, vol_window: int, calendar_days: bool) -> float`
  - `volume_surge(close: pd.Series, volume: pd.Series, week: int, baseline: int) -> float`
  - `breakout_proximity(close: pd.Series, high: pd.Series, windows: tuple[int, int]) -> float`
  - `rsi(close: pd.Series, window: int) -> float`
  - `overextension(close: pd.Series, rsi_window: int, mean_window: int) -> float`
  - `raw_return(close: pd.Series, days: int) -> float`

**Notes for the implementer:** Definitions (fixed here; the spec names the features, these formulas make them implementable):
- *Momentum*: `(close_now / close_lookback_ago - 1) / std(daily pct changes over vol_window)`. Trading-day venues look back by rows; calendar-day venues look back by timestamp via `Series.asof` (crypto has a bar every calendar day, `asof` also tolerates rare gaps).
- *Volume surge*: mean daily dollar volume (`close * volume`) over the last `week` rows ÷ mean over the last `baseline` rows.
- *Breakout proximity*: mean over both windows of `close_now / max(high over window)` — 1.0 means at the high.
- *RSI*: Cutler's simple-mean RSI (`100 * mean(gains) / (mean(gains) + mean(losses))` over the last `window` deltas) — chosen over Wilder smoothing because it is hand-computable in fixtures and the spec only asks for "RSI-style stretch". Flat series returns 50.
- *Overextension*: `rsi/100 + max(close/SMA(mean_window) - 1, 0)` — higher = more stretched; the composite (Task 9) counts it negatively.
- *Raw return*: close vs `asof(as_of - days calendar days)` — un-normalized, feeds M2's crypto fee-adjusted entry gate.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_features.py`:

```python
import math
import statistics

import pandas as pd
import pytest

from trading.signals.features import (
    breakout_proximity,
    overextension,
    raw_return,
    rsi,
    vol_adjusted_return,
    volume_surge,
)


def _series(values: list[float], freq: str = "B") -> pd.Series:
    idx = pd.date_range("2026-01-05", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, dtype="float64")


def test_vol_adjusted_return_trading_days():
    close = _series([100, 102, 101, 103, 106, 108])
    changes = [102 / 100 - 1, 101 / 102 - 1, 103 / 101 - 1, 106 / 103 - 1, 108 / 106 - 1]
    expected = (108 / 100 - 1) / statistics.stdev(changes)
    got = vol_adjusted_return(close, lookback=5, vol_window=5, calendar_days=False)
    assert got == pytest.approx(expected)


def test_vol_adjusted_return_calendar_days():
    close = _series([100.0] * 9 + [130.0], freq="D")
    # 7 calendar days before the last bar lands exactly on index[2] (close=100).
    changes = [0.0, 0.0, 0.0, 0.0, 0.3]
    expected = 0.3 / statistics.stdev(changes)
    got = vol_adjusted_return(close, lookback=7, vol_window=5, calendar_days=True)
    assert got == pytest.approx(expected)


def test_vol_adjusted_return_insufficient_history_is_nan():
    close = _series([100.0, 101.0, 102.0])
    assert math.isnan(vol_adjusted_return(close, lookback=5, vol_window=5, calendar_days=False))


def test_vol_adjusted_return_zero_vol_is_nan():
    close = _series([100.0] * 10)
    assert math.isnan(vol_adjusted_return(close, lookback=5, vol_window=5, calendar_days=False))


def test_volume_surge():
    close = _series([10.0] * 12)
    volume = _series([100.0] * 10 + [300.0, 300.0])
    # recent 2-day dollar volume = 3000; trailing 10-day = (8*1000 + 2*3000)/10 = 1400
    assert volume_surge(close, volume, week=2, baseline=10) == pytest.approx(3000 / 1400)


def test_volume_surge_insufficient_history_is_nan():
    close = _series([10.0] * 5)
    volume = _series([100.0] * 5)
    assert math.isnan(volume_surge(close, volume, week=2, baseline=10))


def test_breakout_proximity():
    high = _series([120.0] * 40 + [110.0] * 20)
    close = _series([100.0] * 59 + [99.0])
    # 20-day high = 110, 60-day high = 120
    expected = (99 / 110 + 99 / 120) / 2
    assert breakout_proximity(close, high, windows=(20, 60)) == pytest.approx(expected)


def test_rsi_simple_average():
    close = _series([10.0, 11.0, 10.0, 12.0, 14.0])
    # deltas +1, -1, +2, +2: mean gain 1.25, mean loss 0.25
    assert rsi(close, window=4) == pytest.approx(100 * 1.25 / 1.5)


def test_rsi_all_gains_is_100():
    close = _series([10.0, 11.0, 12.0, 13.0, 14.0])
    assert rsi(close, window=4) == pytest.approx(100.0)


def test_rsi_flat_is_50():
    close = _series([10.0] * 5)
    assert rsi(close, window=4) == pytest.approx(50.0)


def test_overextension_above_mean():
    close = _series([10.0, 11.0, 10.0, 12.0, 14.0])
    sma = (10 + 11 + 10 + 12 + 14) / 5  # 11.4
    expected = (100 * 1.25 / 1.5) / 100 + (14 / sma - 1)
    assert overextension(close, rsi_window=4, mean_window=5) == pytest.approx(expected)


def test_overextension_below_mean_has_no_stretch_term():
    close = _series([14.0, 12.0, 13.0, 12.0, 10.0])
    # deltas -2, +1, -1, -2: mean gain 0.25, mean loss 1.25; close < mean so stretch = 0
    expected = (100 * 0.25 / 1.5) / 100
    assert overextension(close, rsi_window=4, mean_window=5) == pytest.approx(expected)


def test_raw_return_calendar_days():
    close = _series([100.0] * 34 + [130.0], freq="D")
    assert raw_return(close, days=30) == pytest.approx(0.3)


def test_raw_return_insufficient_history_is_nan():
    close = _series([100.0] * 5, freq="D")
    assert math.isnan(raw_return(close, days=30))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.signals'`.

- [ ] **Step 3: Implement the feature functions**

Create `src/trading/signals/__init__.py` as an empty file.

Create `src/trading/signals/features.py`:

```python
"""Pure per-symbol feature functions (spec: Signal Engine).

Every function takes series already truncated to <= as_of by the caller,
performs no I/O and reads no clock, and returns math.nan when history is
insufficient (NaN symbols drop out of percentile ranking naturally).
"""

from __future__ import annotations

import math

import pandas as pd


def _lookback_price(close: pd.Series, lookback: int, calendar_days: bool) -> float:
    if calendar_days:
        target = close.index[-1] - pd.Timedelta(days=lookback)
        if target < close.index[0]:
            return math.nan
        return float(close.asof(target))
    if len(close) <= lookback:
        return math.nan
    return float(close.iloc[-1 - lookback])


def vol_adjusted_return(
    close: pd.Series, lookback: int, vol_window: int, calendar_days: bool
) -> float:
    past = _lookback_price(close, lookback, calendar_days)
    if math.isnan(past) or past <= 0:
        return math.nan
    changes = close.pct_change().iloc[-vol_window:]
    if changes.isna().any() or len(changes) < vol_window:
        return math.nan
    vol = float(changes.std())
    if not vol > 0:
        return math.nan
    return (float(close.iloc[-1]) / past - 1.0) / vol


def volume_surge(close: pd.Series, volume: pd.Series, week: int, baseline: int) -> float:
    dollar_volume = close * volume
    if len(dollar_volume) < baseline:
        return math.nan
    base = float(dollar_volume.iloc[-baseline:].mean())
    if not base > 0:
        return math.nan
    return float(dollar_volume.iloc[-week:].mean()) / base


def breakout_proximity(close: pd.Series, high: pd.Series, windows: tuple[int, int]) -> float:
    if len(high) < max(windows):
        return math.nan
    last = float(close.iloc[-1])
    proximities = [last / float(high.iloc[-w:].max()) for w in windows]
    return sum(proximities) / len(proximities)


def rsi(close: pd.Series, window: int) -> float:
    """Cutler's RSI: simple means, hand-computable in fixtures."""
    deltas = close.diff().iloc[-window:]
    if deltas.isna().any() or len(deltas) < window:
        return math.nan
    gains = float(deltas.clip(lower=0.0).mean())
    losses = float((-deltas.clip(upper=0.0)).mean())
    if gains + losses == 0:
        return 50.0
    return 100.0 * gains / (gains + losses)


def overextension(close: pd.Series, rsi_window: int, mean_window: int) -> float:
    """Higher = more stretched. The composite counts this negatively."""
    stretch_rsi = rsi(close, rsi_window)
    if math.isnan(stretch_rsi) or len(close) < mean_window:
        return math.nan
    sma = float(close.iloc[-mean_window:].mean())
    stretch = max(float(close.iloc[-1]) / sma - 1.0, 0.0)
    return stretch_rsi / 100.0 + stretch


def raw_return(close: pd.Series, days: int) -> float:
    """Un-normalized calendar-day return; feeds the M2 crypto fee-adjusted entry gate."""
    target = close.index[-1] - pd.Timedelta(days=days)
    if target < close.index[0]:
        return math.nan
    past = float(close.asof(target))
    if not past > 0:
        return math.nan
    return float(close.iloc[-1]) / past - 1.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_features.py -v`
Expected: 14 passed. (If `test_vol_adjusted_return_insufficient_history_is_nan` fails on the vol guard rather than the lookback guard, both return NaN — the assertion still holds.)

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/signals/__init__.py src/trading/signals/features.py tests/test_features.py
git commit -m "Add pure signal feature functions with hand-computed fixtures [AI]"
```

---
### Task 9: Signal engine — cross-sectional features, composite, ranking

**Files:**
- Create: `src/trading/signals/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: all six feature functions from `trading.signals.features` (Task 8); `SignalConfig` from `trading.config` (Task 2).
- Produces (locked function shapes):
  - `compute_features(bars: dict[str, pd.DataFrame], as_of: pd.Timestamp, config: SignalConfig) -> pd.DataFrame` — rows = symbols; columns exactly `["mom_short", "mom_med", "mom_long", "volume_surge", "breakout", "overextension", "composite", "raw_return_30d"]` where the first six are cross-sectional percentiles in [0, 1], `composite` is the equal-weight blend, `raw_return_30d` is the raw (un-normalized) return.
  - `rank(features: pd.DataFrame) -> pd.DataFrame` — sorted descending by `composite`, NaNs last.
  - `FEATURE_COLUMNS: list[str]`, `OUTPUT_COLUMNS: list[str]`, `min_history_rows(config: SignalConfig) -> int`.
- Task 11 consumes `compute_features` and `rank`; M2's simulator consumes `raw_return_30d` for the crypto fee gate.

**Notes for the implementer:**
- No-lookahead is structural: the engine's first act per symbol is `df.loc[:as_of]`; nothing after `as_of` can influence output. The property test below is the regression guard the spec requires.
- Composite (equal weights, fixed in v1 — spec forbids fitting them): `(mom_short + mom_med + mom_long + volume_surge + breakout + (1 - overextension)) / 6` over the percentile columns. Overextension enters as `1 - percentile` because it is the negative guard.
- Symbols with fewer than `min_history_rows(config)` bars at `as_of` are dropped (Task 11 reports them). A NaN feature leaves `composite` NaN; `rank` sorts those last.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_engine.py`:

```python
import numpy as np
import pandas as pd

from trading.config import SignalConfig
from trading.signals.engine import FEATURE_COLUMNS, OUTPUT_COLUMNS, compute_features, rank

CONFIG = SignalConfig(
    momentum_windows=(5, 10, 20),
    calendar_days=False,
    vol_window=5,
    volume_week=5,
    volume_baseline=20,
    breakout_windows=(10, 20),
    rsi_window=14,
    mean_window=20,
    raw_return_days=30,
)


def _trending_bars(drift: float, periods: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2025-11-03", periods=periods, freq="B", tz="UTC")
    jitter = np.where(np.arange(periods) % 2 == 0, 0.002, -0.002)
    close = 100 * np.cumprod(1 + drift + jitter)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": np.full(periods, 1e6)},
        index=idx,
    )


def _random_walk_bars(seed: int, periods: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-11-03", periods=periods, freq="B", tz="UTC")
    rets = rng.normal(0.001, 0.02, periods)
    close = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(
        {
            "open": np.concatenate([[100.0], close[:-1]]),
            "high": close * (1 + rng.uniform(0.0, 0.02, periods)),
            "low": close * (1 - rng.uniform(0.0, 0.02, periods)),
            "close": close,
            "volume": rng.uniform(1e5, 1e6, periods),
        },
        index=idx,
    )


def test_columns_are_locked():
    assert OUTPUT_COLUMNS == [
        "mom_short", "mom_med", "mom_long", "volume_surge", "breakout",
        "overextension", "composite", "raw_return_30d",
    ]
    assert FEATURE_COLUMNS == OUTPUT_COLUMNS[:6]


def test_compute_features_ranks_momentum_cross_sectionally():
    bars = {
        "UP": _trending_bars(0.01),
        "FLAT": _trending_bars(0.0),
        "DOWN": _trending_bars(-0.01),
    }
    as_of = bars["UP"].index[-1]
    out = compute_features(bars, as_of, CONFIG)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert set(out.index) == {"UP", "FLAT", "DOWN"}
    assert out.loc["UP", "mom_med"] > out.loc["FLAT", "mom_med"] > out.loc["DOWN", "mom_med"]
    assert out.loc["UP", "composite"] > out.loc["FLAT", "composite"]
    assert out.loc["FLAT", "composite"] > out.loc["DOWN", "composite"]
    assert out.loc["UP", "raw_return_30d"] > 0 > out.loc["DOWN", "raw_return_30d"]
    feats = out[FEATURE_COLUMNS]
    assert ((feats >= 0) & (feats <= 1)).all().all()


def test_symbol_with_short_history_is_dropped():
    bars = {"UP": _trending_bars(0.01), "NEW": _trending_bars(0.01, periods=10)}
    out = compute_features(bars, bars["UP"].index[-1], CONFIG)
    assert "NEW" not in out.index
    assert "UP" in out.index


def test_empty_universe_returns_empty_frame():
    as_of = pd.Timestamp("2026-07-01", tz="UTC")
    out = compute_features({}, as_of, CONFIG)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out.empty


def test_no_lookahead_property():
    """Spec property test: perturbing data after as_of must not change features at as_of."""
    bars = {f"S{i}": _random_walk_bars(i) for i in range(6)}
    as_of = bars["S0"].index[100]
    base = compute_features(bars, as_of, CONFIG)

    perturbed = {}
    for symbol, df in bars.items():
        p = df.copy()
        future = p.index > as_of
        p.loc[future, ["open", "high", "low", "close"]] *= 7.5
        p.loc[future, "volume"] *= 100.0
        perturbed[symbol] = p

    after = compute_features(perturbed, as_of, CONFIG)
    pd.testing.assert_frame_equal(base, after)


def test_rank_sorts_by_composite_desc_nans_last():
    features = pd.DataFrame(
        {"composite": [0.2, 0.9, float("nan"), 0.5]}, index=["A", "B", "C", "D"]
    )
    assert list(rank(features).index) == ["B", "D", "A", "C"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.signals.engine'`.

- [ ] **Step 3: Implement the engine**

Create `src/trading/signals/engine.py`:

```python
"""Cross-sectional signal engine (spec: Signal Engine).

Pure: no I/O, no clock — as_of is always a parameter. Features are
normalized to cross-sectional percentiles; the composite is their
equal-weight blend (weights fixed equal in v1 by design).
"""

from __future__ import annotations

import pandas as pd

from trading.config import SignalConfig
from trading.signals.features import (
    breakout_proximity,
    overextension,
    raw_return,
    vol_adjusted_return,
    volume_surge,
)

FEATURE_COLUMNS = [
    "mom_short",
    "mom_med",
    "mom_long",
    "volume_surge",
    "breakout",
    "overextension",
]
OUTPUT_COLUMNS = [*FEATURE_COLUMNS, "composite", "raw_return_30d"]


def min_history_rows(config: SignalConfig) -> int:
    return max(
        config.momentum_windows[-1] + 1,
        config.vol_window + 1,
        config.volume_baseline,
        config.breakout_windows[-1],
        config.mean_window,
        config.rsi_window + 1,
    )


def compute_features(
    bars: dict[str, pd.DataFrame], as_of: pd.Timestamp, config: SignalConfig
) -> pd.DataFrame:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be tz-aware UTC")
    required = min_history_rows(config)
    short, med, long_ = config.momentum_windows

    raw_rows: dict[str, dict[str, float]] = {}
    for symbol, df in bars.items():
        window = df.loc[:as_of]  # structural no-lookahead cut
        if len(window) < required:
            continue
        close, high, volume = window["close"], window["high"], window["volume"]
        raw_rows[symbol] = {
            "mom_short": vol_adjusted_return(close, short, config.vol_window, config.calendar_days),
            "mom_med": vol_adjusted_return(close, med, config.vol_window, config.calendar_days),
            "mom_long": vol_adjusted_return(close, long_, config.vol_window, config.calendar_days),
            "volume_surge": volume_surge(close, volume, config.volume_week, config.volume_baseline),
            "breakout": breakout_proximity(close, high, config.breakout_windows),
            "overextension": overextension(close, config.rsi_window, config.mean_window),
            "raw_return_30d": raw_return(close, config.raw_return_days),
        }

    if not raw_rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS, dtype="float64")

    raw = pd.DataFrame.from_dict(raw_rows, orient="index")
    pct = raw[FEATURE_COLUMNS].rank(pct=True)
    pct["composite"] = (
        pct["mom_short"]
        + pct["mom_med"]
        + pct["mom_long"]
        + pct["volume_surge"]
        + pct["breakout"]
        + (1.0 - pct["overextension"])  # negative guard
    ) / 6.0
    pct["raw_return_30d"] = raw["raw_return_30d"]
    return pct[OUTPUT_COLUMNS]


def rank(features: pd.DataFrame) -> pd.DataFrame:
    return features.sort_values("composite", ascending=False, na_position="last")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_engine.py -v`
Expected: 6 passed. `test_no_lookahead_property` must pass exactly (`assert_frame_equal`, no tolerance).

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/signals/engine.py tests/test_engine.py
git commit -m "Add cross-sectional signal engine with no-lookahead property test [AI]"
```

---

### Task 10: Regime gate

**Files:**
- Create: `src/trading/signals/regime.py`
- Test: `tests/test_regime.py`

**Interfaces:**
- Consumes: `RegimeConfig` from `trading.config` (Task 2).
- Produces (locked shape, plus a config parameter because every number must live in TOML):
  - `Regime(state: Literal["risk_on", "neutral", "risk_off"], exposure_multiplier: float)` — frozen dataclass.
  - `compute_regime(benchmark_bars: pd.DataFrame, as_of: pd.Timestamp, config: RegimeConfig) -> Regime`.
- Task 11 consumes `compute_regime` with SPY (equities) / BTC (crypto) benchmark bars; M2's simulator consumes `exposure_multiplier`.

**Notes for the implementer:** Decision rules (fixed here — spec gives the ingredients, this makes them implementable):
- `vol_pct` = fraction of the trailing `vol_lookback` realized-vol observations (rolling `vol_window` std of daily pct changes) **strictly below** the current one. Strict comparison means a constant-vol series reads as low-vol, not high-vol.
- `risk_off` if `close < SMA(sma_slow)`, or `close < SMA(sma_fast)` while `vol_pct >= vol_high_percentile` → exposure `exposure_risk_off` (0.0: no new entries).
- `risk_on` if `close > SMA(sma_fast)` and `close > SMA(sma_slow)` and `vol_pct < vol_high_percentile` → exposure `exposure_risk_on` (1.0).
- otherwise `neutral` → `exposure_neutral` (0.5).
- Insufficient history (< `sma_slow` bars at `as_of`) → `neutral`, the conservative default.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_regime.py`:

```python
import dataclasses

import numpy as np
import pandas as pd
import pytest

from trading.config import RegimeConfig
from trading.signals.regime import Regime, compute_regime

CONFIG = RegimeConfig(
    sma_fast=50,
    sma_slow=200,
    vol_window=20,
    vol_lookback=252,
    vol_high_percentile=0.80,
    exposure_risk_on=1.0,
    exposure_neutral=0.5,
    exposure_risk_off=0.0,
)


def _bars_from_rets(rets: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(rets) + 1, freq="B", tz="UTC")
    close = 100 * np.concatenate([[1.0], np.cumprod(1 + rets)])
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.full(len(close), 1e6)},
        index=idx,
    )


def _decaying_jitter(n: int, drift: float) -> np.ndarray:
    """Trend with noise that shrinks over time, so current vol is the lowest."""
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return drift + signs * np.linspace(0.004, 0.001, n)


def test_uptrend_with_falling_vol_is_risk_on():
    bars = _bars_from_rets(_decaying_jitter(300, drift=0.004))
    regime = compute_regime(bars, bars.index[-1], CONFIG)
    assert regime == Regime(state="risk_on", exposure_multiplier=1.0)


def test_downtrend_is_risk_off():
    bars = _bars_from_rets(_decaying_jitter(300, drift=-0.004))
    regime = compute_regime(bars, bars.index[-1], CONFIG)
    assert regime == Regime(state="risk_off", exposure_multiplier=0.0)


def test_uptrend_with_vol_spike_is_neutral():
    # Long calm uptrend, then 11 alternating +/-8% bars ending on +8%: the close
    # stays above both SMAs but current vol is the highest on record.
    calm = _decaying_jitter(290, drift=0.004)
    spike = np.array([0.08 if i % 2 == 0 else -0.08 for i in range(11)])
    bars = _bars_from_rets(np.concatenate([calm, spike]))
    regime = compute_regime(bars, bars.index[-1], CONFIG)
    assert regime == Regime(state="neutral", exposure_multiplier=0.5)


def test_short_history_is_neutral():
    bars = _bars_from_rets(_decaying_jitter(100, drift=0.004))
    regime = compute_regime(bars, bars.index[-1], CONFIG)
    assert regime.state == "neutral"


def test_regime_no_lookahead():
    bars = _bars_from_rets(_decaying_jitter(300, drift=0.004))
    as_of = bars.index[250]
    base = compute_regime(bars, as_of, CONFIG)
    perturbed = bars.copy()
    perturbed.loc[perturbed.index > as_of, "close"] *= 0.1
    assert compute_regime(perturbed, as_of, CONFIG) == base


def test_regime_is_frozen():
    regime = Regime(state="risk_on", exposure_multiplier=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        regime.state = "risk_off"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_regime.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.signals.regime'`.

- [ ] **Step 3: Implement the regime gate**

Create `src/trading/signals/regime.py`:

```python
"""Market-regime gate (spec: Signal Engine / regime gate).

Benchmark trend (price vs 50/200-day SMAs) plus realized-vol percentile map
to an exposure multiplier: risk-on = full, neutral = half, risk-off = no new
entries (exits still honored — enforced by the M2 simulator). Pure: no I/O,
no clock; as_of is a parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from trading.config import RegimeConfig

RegimeState = Literal["risk_on", "neutral", "risk_off"]


@dataclass(frozen=True)
class Regime:
    state: RegimeState
    exposure_multiplier: float


def compute_regime(
    benchmark_bars: pd.DataFrame, as_of: pd.Timestamp, config: RegimeConfig
) -> Regime:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be tz-aware UTC")
    close = benchmark_bars.loc[:as_of, "close"]  # structural no-lookahead cut
    if len(close) < config.sma_slow:
        return Regime(state="neutral", exposure_multiplier=config.exposure_neutral)

    last = float(close.iloc[-1])
    sma_fast = float(close.iloc[-config.sma_fast :].mean())
    sma_slow = float(close.iloc[-config.sma_slow :].mean())

    vols = close.pct_change().rolling(config.vol_window).std().dropna()
    if vols.empty:
        return Regime(state="neutral", exposure_multiplier=config.exposure_neutral)
    trailing = vols.iloc[-config.vol_lookback :]
    current_vol = float(trailing.iloc[-1])
    vol_pct = float((trailing < current_vol).mean())
    high_vol = vol_pct >= config.vol_high_percentile

    if last < sma_slow or (last < sma_fast and high_vol):
        return Regime(state="risk_off", exposure_multiplier=config.exposure_risk_off)
    if last > sma_fast and last > sma_slow and not high_vol:
        return Regime(state="risk_on", exposure_multiplier=config.exposure_risk_on)
    return Regime(state="neutral", exposure_multiplier=config.exposure_neutral)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_regime.py -v`
Expected: 6 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/signals/regime.py tests/test_regime.py
git commit -m "Add benchmark regime gate with exposure multipliers [AI]"
```

---
### Task 11: Rankings pipeline

**Files:**
- Create: `src/trading/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `VenueConfig` (Task 2); `VenueAdapter`, `SymbolInfo`, `DataFetchError` (Task 3); `OhlcvCache.fetch` (Task 6); `check_coverage`, `quarantine_outliers`, `CoverageReport` (Task 7); `compute_features`, `rank` (Task 9); `compute_regime`, `Regime` (Task 10).
- Produces:
  - `PipelineDataError(RuntimeError)`.
  - `RankingsResult(venue: str, as_of: pd.Timestamp, regime: Regime, table: pd.DataFrame, coverage: CoverageReport, quarantined: tuple[str, ...], fetch_failures: tuple[str, ...], insufficient_history: tuple[str, ...])` — frozen dataclass; `table` is the ranked frame with a leading `status` column followed by the engine's `OUTPUT_COLUMNS`.
  - `build_rankings(config: VenueConfig, adapter: VenueAdapter, cache: OhlcvCache, as_of: datetime.date) -> RankingsResult`.
- Task 12's CLI consumes all three.

**Notes for the implementer:** Spec rules enforced here: coverage < `min_coverage` (90%) or a benchmark fetch failure raises `PipelineDataError` — the run must not proceed. Per-symbol fetch failures of *any* exception type are exclusions, by design (network libs raise arbitrary types; the coverage gate is the safety net). All universe statuses (`tradable`/`sell_only`/`untradable`) stay in the ranking with their status shown — pre-ranking filters are *entry* filters and belong to M2 (spec: held names always remain rankable).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline.py`:

```python
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
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
         "volume": rng.uniform(1e5, 1e6, 320)},
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

    def fetch_ohlcv(
        self, symbol: str, start: datetime.date, end: datetime.date
    ) -> pd.DataFrame:
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
    spike_at = frames["S5"].index[200]
    frames["S5"].loc[spike_at, "close"] = frames["S5"]["close"].iloc[199] * 1.7
    result = build_rankings(CONFIG, adapter, cache, AS_OF)
    assert result.quarantined == ("S5",)
    assert "S5" not in result.table.index


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.pipeline'`.

- [ ] **Step 3: Implement the pipeline**

Create `src/trading/pipeline.py`:

```python
"""Rankings pipeline: universe -> cached fetch -> quality gates -> signals -> regime.

Orchestrates I/O around the pure signal engine. Raises PipelineDataError when
the spec says the run must not proceed (< min_coverage universe fetch,
benchmark fetch failure); the CLI turns that into a warning + nonzero exit.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import pandas as pd

from trading.config import VenueConfig
from trading.data.cache import OhlcvCache
from trading.data.quality import CoverageReport, check_coverage, quarantine_outliers
from trading.signals.engine import compute_features, rank
from trading.signals.regime import Regime, compute_regime
from trading.venues.base import VenueAdapter


class PipelineDataError(RuntimeError):
    """Fresh data could not be assembled; the run must not proceed (spec)."""


@dataclass(frozen=True)
class RankingsResult:
    venue: str
    as_of: pd.Timestamp
    regime: Regime
    table: pd.DataFrame  # ranked; leading "status" column + engine OUTPUT_COLUMNS
    coverage: CoverageReport
    quarantined: tuple[str, ...]
    fetch_failures: tuple[str, ...]
    insufficient_history: tuple[str, ...]


def build_rankings(
    config: VenueConfig,
    adapter: VenueAdapter,
    cache: OhlcvCache,
    as_of: datetime.date,
) -> RankingsResult:
    start = as_of - datetime.timedelta(days=config.data.history_days)
    infos = adapter.universe(as_of)

    bars: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    for info in infos:
        try:
            bars[info.symbol] = cache.fetch(info.symbol, start, as_of, adapter.fetch_ohlcv)
        except Exception:
            # Any per-symbol failure (network, missing pair, bad frame) is an
            # exclusion; the coverage gate below is the safety net.
            failures.append(info.symbol)

    coverage = check_coverage([i.symbol for i in infos], bars, config.data.min_coverage)
    if not coverage.ok:
        raise PipelineDataError(
            f"universe coverage {coverage.ratio:.0%} below "
            f"{config.data.min_coverage:.0%}; missing: {', '.join(coverage.missing)}"
        )

    clean, quarantined = quarantine_outliers(bars, config.data.max_daily_move)

    try:
        benchmark = cache.fetch(config.benchmark, start, as_of, adapter.fetch_ohlcv)
    except Exception as exc:
        raise PipelineDataError(f"benchmark {config.benchmark} fetch failed: {exc}") from exc

    as_of_ts = pd.Timestamp(as_of, tz="UTC")
    regime = compute_regime(benchmark, as_of_ts, config.regime)
    features = compute_features(clean, as_of_ts, config.signals)
    table = rank(features).copy()

    statuses = {i.symbol: i.status for i in infos}
    table.insert(0, "status", [statuses[s] for s in table.index])
    insufficient = tuple(sorted(set(clean) - set(table.index)))

    return RankingsResult(
        venue=config.name,
        as_of=as_of_ts,
        regime=regime,
        table=table,
        coverage=coverage,
        quarantined=quarantined,
        fetch_failures=tuple(sorted(failures)),
        insufficient_history=insufficient,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 7 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/pipeline.py tests/test_pipeline.py
git commit -m "Add rankings pipeline with coverage and quarantine gates [AI]"
```

---
### Task 12: Rankings CLI, adapter factory, README

**Files:**
- Modify: `src/trading/cli.py` (replace the Task 1 stub body entirely)
- Modify: `src/trading/venues/__init__.py` (was empty; add `make_adapter`)
- Modify: `tests/test_cli.py` (replace entirely; keeps the two scaffold assertions plus end-to-end tests)
- Create: `README.md`

**Interfaces:**
- Consumes: `load_venue_config` (Task 2); `EquitiesAdapter` + `_yf_download` + `DEFAULT_UNIVERSE_CSV` (Task 4); `CryptoAdapter` + `_kraken_fetch` + `DEFAULT_UNIVERSE_CSV` (Task 5); `OhlcvCache` (Task 6); `build_rankings`, `RankingsResult`, `PipelineDataError` (Task 11).
- Produces: `trading rankings --venue equities|crypto [--as-of DATE] [--top N] [--json] [--config-dir DIR]`; `make_adapter(config: VenueConfig) -> VenueAdapter` in `trading.venues`. This is the M1 deliverable; M2 adds `run`/`status`/`digest`/`schedule`/`reset-breaker` subcommands to the same parser.

**Notes for the implementer:** The CLI is the only module allowed to read the clock (`--as-of` defaults to today UTC). `--json` prints the machine-readable payload per spec; the human default is a rich table plus regime line and warnings. A `PipelineDataError` prints `WARNING: ...` to stderr and exits 1 (spec failure behavior; macOS notification hooks arrive with M2 scheduling).

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_cli.py` entirely with:

```python
import dataclasses
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading.cli import main
from trading.config import load_venue_config
from trading.venues import make_adapter
from trading.venues.base import DataFetchError
from trading.venues.crypto import CryptoAdapter
from trading.venues.equities import EquitiesAdapter

AS_OF = "2026-07-01"


# --- scaffold behavior (from Task 1) ---


def test_no_command_exits_with_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_help_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_unknown_venue_rejected():
    with pytest.raises(SystemExit) as excinfo:
        main(["rankings", "--venue", "bonds"])
    assert excinfo.value.code == 2


# --- adapter factory ---


def test_make_adapter_dispatches_by_venue_name():
    assert isinstance(make_adapter(load_venue_config("equities", Path("config"))), EquitiesAdapter)
    assert isinstance(make_adapter(load_venue_config("crypto", Path("config"))), CryptoAdapter)


def test_make_adapter_unknown_venue_raises():
    config = dataclasses.replace(load_venue_config("equities", Path("config")), name="bonds")
    with pytest.raises(ValueError, match="bonds"):
        make_adapter(config)


# --- end-to-end fixtures (network fully monkeypatched) ---


def _write_config(tmp_path: Path, venue: str) -> Path:
    """Copy the real venue TOML, pointing its cache at the test tmp dir."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    text = Path("config", f"{venue}.toml").read_text()
    text = text.replace(
        f'cache_dir = "data/{venue}"', f'cache_dir = "{tmp_path}/cache/{venue}"'
    )
    (cfg_dir / f"{venue}.toml").write_text(text)
    return cfg_dir


def _fake_history(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Deterministic yfinance-shaped frame (naive index, capitalized columns)."""
    rng = np.random.default_rng(sum(map(ord, symbol)))
    idx = pd.date_range(start, end, freq="B")
    rets = rng.normal(0.0005, 0.015, len(idx))
    close = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
         "Volume": rng.uniform(1e5, 1e6, len(idx))},
        index=idx,
    )


def _fake_kraken(pair: str, since_ms: int) -> list[list[float]]:
    """Deterministic ccxt-shaped daily rows ending on AS_OF."""
    rng = np.random.default_rng(sum(map(ord, pair)))
    n = 520
    end = pd.Timestamp(AS_OF, tz="UTC")
    start = end - pd.Timedelta(days=n - 1)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    rows = []
    for i in range(n):
        ts = int((start + pd.Timedelta(days=i)).timestamp() * 1000)
        c = float(close[i])
        rows.append([ts, c, c * 1.01, c * 0.99, c, float(rng.uniform(1e5, 1e6))])
    return rows


def _setup_equities(tmp_path, monkeypatch) -> Path:
    cfg_dir = _write_config(tmp_path, "equities")
    universe = tmp_path / "equities_universe.csv"
    universe.write_text("symbol\nAAA\nBBB\nCCC\nDDD\n")
    monkeypatch.setattr("trading.venues.equities.DEFAULT_UNIVERSE_CSV", universe)
    monkeypatch.setattr("trading.venues.equities._yf_download", _fake_history)
    return cfg_dir


def _setup_crypto(tmp_path, monkeypatch) -> Path:
    cfg_dir = _write_config(tmp_path, "crypto")
    universe = tmp_path / "crypto_universe.csv"
    universe.write_text("symbol,status\nBTC,tradable\nETH,tradable\nSOL,sell_only\n")
    monkeypatch.setattr("trading.venues.crypto.DEFAULT_UNIVERSE_CSV", universe)
    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", _fake_kraken)
    return cfg_dir


# --- end-to-end tests ---


def test_rankings_json_equities_end_to_end(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    rc = main(["rankings", "--venue", "equities", "--as-of", AS_OF,
               "--json", "--config-dir", str(cfg_dir)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["venue"] == "equities"
    assert payload["as_of"] == AS_OF
    assert payload["regime"]["state"] in {"risk_on", "neutral", "risk_off"}
    assert payload["coverage"] == {"requested": 4, "fetched": 4, "ratio": 1.0}
    assert {row["symbol"] for row in payload["rankings"]} == {"AAA", "BBB", "CCC", "DDD"}
    composites = [row["composite"] for row in payload["rankings"]]
    assert composites == sorted(composites, reverse=True)
    assert all("raw_return_30d" in row for row in payload["rankings"])


def test_rankings_json_crypto_end_to_end(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_crypto(tmp_path, monkeypatch)
    rc = main(["rankings", "--venue", "crypto", "--as-of", AS_OF,
               "--json", "--config-dir", str(cfg_dir)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["venue"] == "crypto"
    by_symbol = {row["symbol"]: row for row in payload["rankings"]}
    assert set(by_symbol) == {"BTC", "ETH", "SOL"}
    assert by_symbol["SOL"]["status"] == "sell_only"  # sell_only stays rankable


def test_rankings_table_renders_human_output(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    rc = main(["rankings", "--venue", "equities", "--as-of", AS_OF,
               "--config-dir", str(cfg_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "regime" in out
    assert "AAA" in out


def test_coverage_failure_warns_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)

    def flaky(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        if symbol in {"BBB", "CCC"}:
            raise DataFetchError(f"boom {symbol}")
        return _fake_history(symbol, start, end)

    monkeypatch.setattr("trading.venues.equities._yf_download", flaky)
    rc = main(["rankings", "--venue", "equities", "--as-of", AS_OF,
               "--json", "--config-dir", str(cfg_dir)])
    assert rc == 1
    assert "WARNING" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_adapter' from 'trading.venues'`.

- [ ] **Step 3: Implement the adapter factory**

Replace `src/trading/venues/__init__.py` (was empty) with:

```python
from __future__ import annotations

from trading.config import VenueConfig
from trading.venues.base import VenueAdapter


def make_adapter(config: VenueConfig) -> VenueAdapter:
    if config.name == "equities":
        from trading.venues.equities import EquitiesAdapter

        return EquitiesAdapter(config)
    if config.name == "crypto":
        from trading.venues.crypto import CryptoAdapter

        return CryptoAdapter(config)
    raise ValueError(f"unknown venue {config.name!r}")
```

- [ ] **Step 4: Implement the CLI**

Replace `src/trading/cli.py` entirely with:

```python
"""Command-line entry point (spec: CLI & README).

M1 ships `trading rankings`. Later milestones add run/status/backtest/digest/
schedule/reset-breaker on the same parser. Human-readable rich tables by
default, --json for machine consumption. The CLI is the only module allowed
to read the clock; everything below it takes as_of as a parameter.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

from trading.config import load_venue_config
from trading.data.cache import OhlcvCache
from trading.pipeline import PipelineDataError, RankingsResult, build_rankings
from trading.venues import make_adapter

VENUES = ["equities", "crypto"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trading", description="Momentum swing trading system")
    sub = parser.add_subparsers(dest="command", required=True)

    rankings = sub.add_parser("rankings", help="current ranked table with sub-scores")
    rankings.add_argument("--venue", choices=VENUES, required=True)
    rankings.add_argument(
        "--as-of",
        type=datetime.date.fromisoformat,
        default=None,
        help="decision date, YYYY-MM-DD (default: today UTC)",
    )
    rankings.add_argument(
        "--top", type=int, default=25, help="rows to display, 0 = all (table output only)"
    )
    rankings.add_argument("--json", action="store_true", help="machine-readable output")
    rankings.add_argument("--config-dir", default="config", help="directory with <venue>.toml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "rankings":
        return _cmd_rankings(args)
    return 2  # unreachable: subparsers are required


def _cmd_rankings(args: argparse.Namespace) -> int:
    config = load_venue_config(args.venue, Path(args.config_dir))
    adapter = make_adapter(config)
    cache = OhlcvCache(Path(config.data.cache_dir), config.data.refetch_days)
    as_of = args.as_of or datetime.datetime.now(datetime.UTC).date()
    try:
        result = build_rankings(config, adapter, cache, as_of)
    except PipelineDataError as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(_to_json(result), indent=2))
    else:
        _render(result, top=args.top)
    return 0


def _to_json(result: RankingsResult) -> dict:
    rankings = []
    for pos, (symbol, row) in enumerate(result.table.iterrows(), start=1):
        entry: dict[str, object] = {"rank": pos, "symbol": symbol, "status": row["status"]}
        for col in result.table.columns:
            if col == "status":
                continue
            value = row[col]
            entry[col] = None if pd.isna(value) else round(float(value), 4)
        rankings.append(entry)
    return {
        "venue": result.venue,
        "as_of": result.as_of.date().isoformat(),
        "regime": {
            "state": result.regime.state,
            "exposure_multiplier": result.regime.exposure_multiplier,
        },
        "coverage": {
            "requested": result.coverage.requested,
            "fetched": result.coverage.fetched,
            "ratio": round(result.coverage.ratio, 4),
        },
        "quarantined": list(result.quarantined),
        "fetch_failures": list(result.fetch_failures),
        "insufficient_history": list(result.insufficient_history),
        "rankings": rankings,
    }


def _render(result: RankingsResult, top: int) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(
        f"[bold]{result.venue}[/bold] rankings as of {result.as_of.date().isoformat()} | "
        f"regime: [bold]{result.regime.state}[/bold] "
        f"(exposure x{result.regime.exposure_multiplier})"
    )
    value_columns = [c for c in result.table.columns if c != "status"]
    table = Table()
    table.add_column("#", justify="right")
    table.add_column("symbol")
    table.add_column("status")
    for col in value_columns:
        table.add_column(col, justify="right")
    rows = result.table if top == 0 else result.table.head(top)
    for pos, (symbol, row) in enumerate(rows.iterrows(), start=1):
        cells = [str(pos), str(symbol), str(row["status"])]
        cells += ["" if pd.isna(row[c]) else f"{row[c]:.3f}" for c in value_columns]
        table.add_row(*cells)
    console.print(table)
    console.print(
        f"coverage {result.coverage.fetched}/{result.coverage.requested} "
        f"({result.coverage.ratio:.0%})"
    )
    if result.quarantined:
        console.print(f"[yellow]quarantined:[/yellow] {', '.join(result.quarantined)}")
    if result.fetch_failures:
        console.print(f"[yellow]fetch failures:[/yellow] {', '.join(result.fetch_failures)}")
    if result.insufficient_history:
        console.print(
            f"[yellow]insufficient history:[/yellow] {', '.join(result.insufficient_history)}"
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 9 passed.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -v`
Expected: all tests from Tasks 1-12 pass, zero failures.

- [ ] **Step 7: Write the README**

Create `README.md`:

```markdown
# trading

Momentum swing trading system. It ranks liquid assets (S&P 500 + Nasdaq-100
equities; Robinhood-listed crypto) by likelihood of near-term upward moves
using price/volume momentum behind a market-regime gate. This milestone
ships rankings only; paper trading, digests and backtesting arrive next.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):

    uv sync

## Commands

    uv run trading rankings --venue equities   # ranked table + regime (SPY benchmark)
    uv run trading rankings --venue crypto     # ranked table + regime (BTC benchmark)

Options: `--as-of YYYY-MM-DD` (default: today UTC), `--top N` (default 25,
0 = all rows), `--json` for machine-readable output, `--config-dir DIR`
(default `config/`). Run from the repo root.

Exits 1 with a `WARNING` on stderr when fresh data cannot be assembled
(under 90% universe coverage, or the benchmark fetch fails).

## Where things live

- `config/<venue>.toml` — every tunable number (fees, windows, thresholds).
- `data/<venue>/*.parquet` — gitignored OHLCV cache. The trailing 30 days
  are re-fetched every run, so deleting `data/` is always safe.
- `src/trading/venues/universes/*.csv` — committed universe snapshots.

All timestamps are UTC. First equities run fetches ~580 symbols from
yfinance and takes several minutes; later runs hit the Parquet cache.
```

- [ ] **Step 8: Manual verification against live data (network required)**

Run: `uv run trading rankings --venue crypto --top 10`
Expected: a rich table of ~10 rows with sub-scores in [0,1], a regime line (`risk_on`, `neutral`, or `risk_off`), a coverage line at or above 90%, exit code 0. A handful of fetch-failure warnings for pairs Kraken doesn't list is acceptable while coverage holds.

Run: `uv run trading rankings --venue crypto --top 10 --json | head -30`
Expected: valid JSON with `venue`, `as_of`, `regime`, `coverage`, `rankings` keys.

(Equities can be verified the same way but fetches ~580 symbols; expect several minutes on the first run.)

- [ ] **Step 9: Lint, format, commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/cli.py src/trading/venues/__init__.py tests/test_cli.py README.md
git commit -m "Add rankings CLI with rich table, JSON output, and README [AI]"
```

---

## M1 Completion Criteria

- `uv run pytest` fully green; `uv run ruff check .` clean.
- `trading rankings --venue equities` and `trading rankings --venue crypto` print a ranked table with sub-scores, composite, raw_return_30d, per-symbol status, the regime state, and coverage/quarantine warnings, from freshly fetched + Parquet-cached data.
- No trading, no state files, no journal — those are M2. No point-in-time universe history — that is M3.
