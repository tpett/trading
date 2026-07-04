# M2: Paper Trading Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A user can run `trading run --venue equities|crypto` on a launchd schedule and get a per-venue $1,000 paper portfolio that trades the M1 rankings under all spec risk rules, with an append-only journal, atomic state, a daily digest, `trading status`, and macOS failure notifications.

**Architecture:** A pure per-venue portfolio simulator (`step(state, rankings, config) -> StepResult` — no I/O, no clock) sits between M1's rankings pipeline and a runner module that owns ALL I/O: state file (atomic temp+rename), append-only JSONL journal, lockfile, digest, notifications. Each run first fills the previous run's pending orders at the next bar's open, then evaluates exits, then entries — writing new pending orders for the next run. M3 will replay this exact `step` function in the backtester, so nothing inside the simulator may read a clock or touch a file.

**Tech Stack:** Python 3.12, uv, pandas, ccxt, yfinance (all pinned in M1's `pyproject.toml`), argparse + rich CLI, pytest, ruff. No new dependencies (launchd plists via stdlib `plistlib`, notifications via `osascript` subprocess).

## Global Constraints

From the approved spec (`docs/superpowers/specs/2026-07-04-momentum-swing-system-design.md`) and locked architectural decisions. Every task's requirements implicitly include this section.

- Repo root for all relative paths: `/Users/travis/Source/personal/trading/worktrees/paper-trading` (git worktree, branch `tpett/ai/paper-trading`). Run all commands from this directory.
- Python 3.12, uv (`uv sync`, `uv run ...`). Lint/format with ruff: run `uv run ruff check . && uv run ruff format .` before every commit.
- The simulator core (`src/trading/simulator/`) is pure: no I/O, no clock access. Decisions come from bars + state + config only; "now" is always a parameter. All I/O (state file, journal, digest, notifications, lockfile) lives in `runner.py` and below the CLI.
- All timestamps UTC everywhere (state, journal, digest filenames). ISO-8601 strings inside JSON payloads.
- Two-phase loop, locked precisely: each run first FILLS pending orders written by the previous run using the current run's freshly fetched bars (fill price = open of the first bar strictly after the pending order's decision bar, plus slippage, plus fees from config costs), then evaluates exits, then entries, writing new pending orders for the next run. Decisions use data through the last completed bar (M1's `drop_incomplete_last_bar` handles bar completeness).
- Idempotency: run_key = `(venue, decision-bar timestamp)`; the journal is consulted before acting — the same run_key never trades twice. A lockfile (`state/<venue>/.lock`) prevents concurrent runs. Late runs skip entries beyond the config staleness bound; exits always process. No catch-up trading of older bars, ever.
- Every number is config, never a code constant. All simulator numbers already live in `config/<venue>.toml [portfolio]` (field names verified against `src/trading/config.py: PortfolioConfig`); this plan adds exactly two new fields: `atr_window` and `earnings_blackout_enabled`.
- Paper starting balance $1,000 per venue (`starting_balance` in TOML). Initial state is created on the first `trading run` (explicit bootstrap path, journaled).
- State: `state/<venue>/portfolio.json`, atomic write (temp + `os.replace`). Journal: `journal/<venue>.jsonl`, append-only. Digest: `digest/YYYY-MM-DD.md` (UTC date). All three directories are gitignored. A corrupt state file refuses to run + notifies; it is never silently regenerated — recovery is journal replay with typed operator confirmation.
- Every failed or skipped run fires a macOS notification (`osascript display notification`), as does a circuit-breaker trip.
- Commit after every task, one logical change per commit, message tagged `[AI]`, with footer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- M3 (backtester, walk-forward, point-in-time universe) is out of scope. Do not build any of it, but never compromise simulator purity — M3 replays `step` verbatim.

## File Structure

```
config/equities.toml                    # MODIFY: [portfolio] atr_window, earnings_blackout_enabled; comments
config/crypto.toml                      # MODIFY: same two fields + max_daily_deployment_pct/staleness_hours notes
pyproject.toml                          # MODIFY: pytest filterwarnings (Task 1)
.gitignore                              # MODIFY: /state/, /journal/, /digest/ (Task 14)
README.md                               # MODIFY: full command set, state layout, digest guide (Task 14)
scripts/build_equities_universe.py      # MODIFY: provenance header line (Task 1)
src/trading/config.py                   # MODIFY: PortfolioConfig new fields (Tasks 4, 9)
src/trading/pipeline.py                 # MODIFY: RankingsResult gains bars + benchmark_bars (Task 2)
src/trading/signals/features.py         # MODIFY: Timedelta fix, raw_return refactor (Task 1)
src/trading/simulator/__init__.py       # NEW: empty package marker
src/trading/simulator/state.py          # NEW: Position, PendingOrder, Settlement, Skip, PortfolioState, dict round-trip
src/trading/simulator/fills.py          # NEW: atr(), Fill, apply_fills(), release_settlements()
src/trading/simulator/exits.py          # NEW: evaluate_exits() — stop/trend-break/time-stop/flush/forced
src/trading/simulator/entries.py        # NEW: evaluate_entries() — all entry gates + skip reasons
src/trading/simulator/core.py           # NEW: decision_bar(), make_run_key(), step(), StepResult, snapshot
src/trading/journal.py                  # NEW: Journal (JSONL append/read/has_run), config_hash()
src/trading/earnings.py                 # NEW: yfinance earnings dates behind the kill switch
src/trading/runner.py                   # NEW: run_venue() — lockfile, bootstrap, staleness, state I/O, journal event
src/trading/notify.py                   # NEW: macOS osascript notification
src/trading/digest.py                   # NEW: build_digest(), collect_run_events(), write_digest()
src/trading/schedule.py                 # NEW: launchd plist build/install/status/remove
src/trading/cli.py                      # MODIFY: run/status/digest/schedule/reset-breaker subcommands
tests/sim_helpers.py                    # NEW: shared builders (bars, table, RankingsResult, state) for simulator tests
tests/test_adapter_contract.py          # NEW: shared venue-adapter contract test (Task 1)
tests/test_state.py                     # NEW
tests/test_fills.py                     # NEW
tests/test_exits.py                     # NEW
tests/test_entries.py                   # NEW
tests/test_core.py                      # NEW
tests/test_journal.py                   # NEW
tests/test_earnings.py                  # NEW
tests/test_runner.py                    # NEW
tests/test_digest.py                    # NEW
tests/test_notify.py                    # NEW
tests/test_schedule.py                  # NEW
tests/test_cli.py                       # MODIFY: run/status/reset-breaker end-to-end (Tasks 10, 12) + Timedelta fix (Task 1)
tests/test_crypto_adapter.py            # MODIFY: Timedelta fix (Task 1)
tests/test_cache.py                     # MODIFY: gappy-fresh full-refetch test (Task 1)
tests/test_engine.py                    # MODIFY: partial-NaN cross-sectional test (Task 1)
tests/test_equities_adapter.py          # MODIFY: provenance/comment test (Task 1)
tests/test_pipeline.py                  # MODIFY: bars/benchmark_bars exposure tests (Task 2)
tests/test_config.py                    # MODIFY: new portfolio fields (Tasks 4, 9)
```

---

### Task 1: M1 review-debt cleanup

Six small, independent fixes from M1 review. One task, three commits.

**Files:**
- Modify: `src/trading/signals/features.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_cli.py:91-95`
- Modify: `tests/test_crypto_adapter.py:16-19`
- Modify: `src/trading/venues/equities.py:49`
- Modify: `src/trading/venues/universes/equities.csv` (prepend one comment line)
- Modify: `scripts/build_equities_universe.py`
- Create: `tests/test_adapter_contract.py`
- Modify: `tests/test_equities_adapter.py` (append one test)
- Modify: `tests/test_cache.py` (append one test)
- Modify: `tests/test_engine.py` (append one test)

**Interfaces:**
- Consumes: all existing M1 code.
- Produces: warning-clean test suite with `filterwarnings = ["error", ...]`; `raw_return()` delegating to `_lookback_price()`; equities universe CSV read with `comment="#"`. No signature changes — later tasks rely on M1 interfaces exactly as they are.

- [ ] **Step 1: Fix the pd.Timedelta deprecation at its sources**

Under numpy 2.5 / pandas 2.3.1, `pd.Timedelta(days=n)` emits `DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated` (~1743 occurrences across the suite). The fixed form is `pd.Timedelta(n, unit="D")` (already used in `src/trading/data/cache.py:21`).

In `src/trading/signals/features.py`, change `_lookback_price` (line 17):

```python
        target = close.index[-1] - pd.Timedelta(lookback, unit="D")
```

and refactor `raw_return` (lines 81-89) to delegate to `_lookback_price` (removing its duplicate lookup logic, which contains the second deprecated call):

```python
def raw_return(close: pd.Series, days: int) -> float:
    """Un-normalized calendar-day return; feeds the M2 crypto fee-adjusted entry gate."""
    past = _lookback_price(close, days, calendar_days=True)
    if math.isnan(past) or not past > 0:
        return math.nan
    return float(close.iloc[-1]) / past - 1.0
```

In `tests/test_cli.py` `_fake_kraken` (lines 91 and 95):

```python
    start = end - pd.Timedelta(n - 1, unit="D")
```
```python
        ts = int((start + pd.Timedelta(i, unit="D")).timestamp() * 1000)
```

In `tests/test_crypto_adapter.py` `_kraken_rows` (lines 16 and 19):

```python
    start_ts = pd.Timestamp(end, tz="UTC") - pd.Timedelta(n - 1, unit="D")
```
```python
            int((start_ts + pd.Timedelta(i, unit="D")).timestamp() * 1000),
```

- [ ] **Step 2: Make new warnings fail the suite**

In `pyproject.toml`, replace the `[tool.pytest.ini_options]` section:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = [
    "error",
    # Third-party noise we don't control; our own code must stay warning-clean.
    "ignore::DeprecationWarning:ccxt.*",
    "ignore::DeprecationWarning:yfinance.*",
    "ignore::DeprecationWarning:websockets.*",
]
```

- [ ] **Step 3: Run the full suite to verify it is warning-clean**

Run: `uv run pytest -q 2>&1 | tee /tmp/claude-m2-task1.log | tail -3`
Expected: `91 passed` with **no** `warnings` count in the summary line. If a previously masked warning from our own code now errors, fix it at the source (same `pd.Timedelta(n, unit="D")` pattern); only add an `ignore` entry if the warning originates inside a third-party module.

- [ ] **Step 4: Commit the warnings fix**

```bash
git add src/trading/signals/features.py pyproject.toml tests/test_cli.py tests/test_crypto_adapter.py
git commit -m "Fix pd.Timedelta numpy deprecation and fail suite on new warnings [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Write the shared adapter contract test (failing for equities comment handling)**

Create `tests/test_adapter_contract.py`:

```python
"""Shared venue-adapter data contract (spec: Testing Strategy).

Both venues run the same assertions in v1 since both paper-trade from day
one. Network touchpoints are monkeypatched; the contract covers universe
shape, constraints, and OHLCV frame invariants.
"""

import datetime
from pathlib import Path
from typing import get_args

import pandas as pd
import pytest

from trading.config import load_venue_config
from trading.venues.base import OHLCV_COLUMNS, DataFetchError, SymbolStatus
from trading.venues.crypto import CryptoAdapter
from trading.venues.equities import EquitiesAdapter

START = datetime.date(2026, 6, 1)
END = datetime.date(2026, 7, 1)
AS_OF = datetime.date(2026, 7, 1)
VALID_STATUSES = set(get_args(SymbolStatus))


def _fake_yf(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    idx = pd.date_range(start, END, freq="B")
    return pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1e6}, index=idx
    )


def _fake_kraken(pair: str, since_ms: int) -> list[list[float]]:
    idx = pd.date_range(START, END, freq="D", tz="UTC")
    return [[int(ts.timestamp() * 1000), 100.0, 101.0, 99.0, 100.5, 1e6] for ts in idx]


def make_adapter(venue: str, monkeypatch, tmp_path, empty: bool = False):
    config = load_venue_config(venue, Path("config"))
    universe = tmp_path / f"{venue}_universe.csv"
    if venue == "equities":
        universe.write_text("# provenance comment line\nsymbol\nAAA\nBBB\n")
        fetch = (lambda s, a, b: pd.DataFrame()) if empty else _fake_yf
        monkeypatch.setattr("trading.venues.equities._yf_download", fetch)
        return EquitiesAdapter(config, universe_csv=universe), config, "AAA"
    universe.write_text("symbol,status\nBTC,tradable\nETH,sell_only\n")
    fetch = (lambda p, s: []) if empty else _fake_kraken
    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fetch)
    return CryptoAdapter(config, universe_csv=universe), config, "BTC"


VENUES = ["equities", "crypto"]


@pytest.mark.parametrize("venue", VENUES)
def test_universe_returns_symbol_infos_with_valid_statuses(venue, monkeypatch, tmp_path):
    adapter, _, _ = make_adapter(venue, monkeypatch, tmp_path)
    infos = adapter.universe(AS_OF)
    assert len(infos) == 2
    assert all(info.status in VALID_STATUSES for info in infos)
    assert all(isinstance(info.symbol, str) and info.symbol for info in infos)


@pytest.mark.parametrize("venue", VENUES)
def test_constraints_mirror_config_costs(venue, monkeypatch, tmp_path):
    adapter, config, _ = make_adapter(venue, monkeypatch, tmp_path)
    constraints = adapter.constraints()
    assert constraints.taker_fee_bps == config.costs.taker_fee_bps
    assert constraints.maker_fee_bps == config.costs.maker_fee_bps
    assert constraints.slippage_bps == config.costs.slippage_bps
    assert constraints.settlement_days == config.costs.settlement_days
    assert constraints.trades_24_7 == config.costs.trades_24_7


@pytest.mark.parametrize("venue", VENUES)
def test_fetch_ohlcv_returns_utc_ohlcv_frame_sliced_to_range(venue, monkeypatch, tmp_path):
    adapter, _, symbol = make_adapter(venue, monkeypatch, tmp_path)
    df = adapter.fetch_ohlcv(symbol, START, END)
    assert list(df.columns) == OHLCV_COLUMNS
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert (df.dtypes == "float64").all()
    assert df.index.min() >= pd.Timestamp(START, tz="UTC")
    assert df.index.max() <= pd.Timestamp(END, tz="UTC")


@pytest.mark.parametrize("venue", VENUES)
def test_fetch_ohlcv_empty_raises_data_fetch_error(venue, monkeypatch, tmp_path):
    adapter, _, symbol = make_adapter(venue, monkeypatch, tmp_path)
    adapter_empty, _, symbol = make_adapter(venue, monkeypatch, tmp_path, empty=True)
    with pytest.raises(DataFetchError):
        adapter_empty.fetch_ohlcv(symbol, START, END)
```

Append to `tests/test_equities_adapter.py`:

```python
def test_committed_equities_csv_has_provenance_comment_and_loads():
    from trading.venues.equities import DEFAULT_UNIVERSE_CSV

    first_line = DEFAULT_UNIVERSE_CSV.read_text().splitlines()[0]
    assert first_line.startswith("#")  # build-date provenance comment
    config = load_venue_config("equities", Path("config"))
    infos = EquitiesAdapter(config).universe(datetime.date(2026, 7, 4))
    assert len(infos) >= 500
```

(Match that file's existing imports; it already imports `EquitiesAdapter`, `load_venue_config`, `datetime`, and `Path` — add any that are missing.)

- [ ] **Step 6: Run the new tests to verify equities fails on the comment line**

Run: `uv run pytest tests/test_adapter_contract.py tests/test_equities_adapter.py -q`
Expected: crypto contract tests PASS; equities `test_universe_returns_symbol_infos_with_valid_statuses` FAILS (without `comment="#"` the comment line is consumed as the CSV header, so the `symbol` column is missing — KeyError) and `test_committed_equities_csv_has_provenance_comment_and_loads` FAILS (no comment line in the committed CSV yet).

- [ ] **Step 7: Align equities universe reading with crypto and add provenance**

In `src/trading/venues/equities.py`, line 49:

```python
        df = pd.read_csv(self._universe_csv, comment="#")
```

Prepend one line to `src/trading/venues/universes/equities.csv` (before the `symbol` header):

```
# S&P 500 + Nasdaq-100 snapshot built 2026-07-04 from Wikipedia constituent lists by scripts/build_equities_universe.py
```

In `scripts/build_equities_universe.py`, add `import datetime` to the imports and change `main()` to write the header:

```python
def main() -> None:
    sp500 = _fetch_tables(SP500_URL)[0]["Symbol"].tolist()
    ndx_tables = _fetch_tables(NDX_URL)
    ndx = next(t for t in ndx_tables if "Ticker" in t.columns)["Ticker"].tolist()
    # yfinance uses '-' for share classes (BRK.B -> BRK-B).
    symbols = sorted({str(s).strip().replace(".", "-") for s in sp500 + ndx})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# S&P 500 + Nasdaq-100 snapshot built {datetime.date.today().isoformat()} "
        "from Wikipedia constituent lists by scripts/build_equities_universe.py\n"
    )
    OUT.write_text(header + "symbol\n" + "\n".join(symbols) + "\n")
    print(f"wrote {len(symbols)} symbols to {OUT}")
```

- [ ] **Step 8: Run and commit the contract/provenance changes**

Run: `uv run pytest tests/test_adapter_contract.py tests/test_equities_adapter.py tests/test_cli.py -q`
Expected: all PASS.

```bash
git add tests/test_adapter_contract.py tests/test_equities_adapter.py src/trading/venues/equities.py src/trading/venues/universes/equities.csv scripts/build_equities_universe.py
git commit -m "Add shared adapter contract test and equities universe provenance [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 9: Add the two missing edge-case tests**

Append to `tests/test_cache.py`:

```python
def test_full_refetch_gappy_fresh_is_authoritative_in_range(tmp_path):
    """Full refetch (request needs more history than cached): fresh data is
    authoritative for the fetched range — cached in-range days missing from
    fresh are dropped, while cached rows after the requested end survive."""
    cache = OhlcvCache(tmp_path / "cache", refetch_days=30)
    cache.fetch("AAPL", datetime.date(2026, 2, 1), END, RecordingFetcher(1.0))

    gap = pd.date_range("2026-02-10", "2026-02-12", freq="D", tz="UTC")

    def gappy(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        return _frame(start, end, 2.0).drop(gap)

    # start=START < cached min (2026-02-01) => full refetch path.
    cache.fetch("AAPL", START, datetime.date(2026, 2, 15), gappy)

    on_disk = pd.read_parquet(cache.path_for("AAPL"))
    # Fresh wins the refetched range: the gap days are gone, not backfilled.
    assert pd.Timestamp("2026-02-11", tz="UTC") not in on_disk.index
    assert on_disk.loc[pd.Timestamp("2026-02-05", tz="UTC"), "close"] == 2.0
    # Cached tail past the requested end is preserved.
    assert on_disk.loc[pd.Timestamp("2026-02-20", tz="UTC"), "close"] == 1.0
    assert not on_disk.index.duplicated().any()
```

Append to `tests/test_engine.py` (add `import math` to its imports):

```python
def test_partial_nan_feature_gives_nan_composite_and_clean_cross_section():
    """A symbol with one NaN feature must not poison the cross-section: its
    composite is NaN (ranks last) and the others' percentiles are computed
    over the non-NaN subset only."""
    bars = {
        "UP": _trending_bars(0.01),
        "FLAT": _trending_bars(0.0),
        "ZEROVOL": _trending_bars(0.005).assign(volume=0.0),  # volume_surge -> NaN
    }
    out = compute_features(bars, bars["UP"].index[-1], CONFIG)
    assert math.isnan(out.loc["ZEROVOL", "volume_surge"])
    assert math.isnan(out.loc["ZEROVOL", "composite"])
    # UP and FLAT have identical constant volume: their surge values tie, and
    # pct-rank over the 2-symbol non-NaN subset gives both (1.5 / 2) = 0.75.
    assert (out.loc[["UP", "FLAT"], "volume_surge"] == 0.75).all()
    assert list(rank(out).index)[-1] == "ZEROVOL"
```

- [ ] **Step 10: Run full suite, lint, and commit**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: all tests PASS, no lint errors. (If `ruff format --check` flags files this plan touched, run `uv run ruff format .` and re-run.)

```bash
git add tests/test_cache.py tests/test_engine.py
git commit -m "Pin cache full-refetch gappy-fresh and engine partial-NaN behavior [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Expose clean bars on RankingsResult

The simulator computes ATR, fills, and marks from the same frames the rankings used — without re-fetching.

**Files:**
- Modify: `src/trading/pipeline.py:37-47` (dataclass) and `:119-128` (return)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `build_rankings(config, adapter, cache, as_of) -> RankingsResult` (M1).
- Produces: `RankingsResult.bars: dict[str, pd.DataFrame]` — the quarantine-passed ("clean") universe bars, post `drop_incomplete_last_bar`, keyed by symbol; and `RankingsResult.benchmark_bars: pd.DataFrame` — the benchmark frame on the same basis. Backward compatible: all existing fields unchanged, `trading rankings` untouched. Tasks 4-11 consume both fields.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: both new tests FAIL with `AttributeError: 'RankingsResult' object has no attribute 'bars'`.

- [ ] **Step 3: Add the fields**

In `src/trading/pipeline.py`, extend the dataclass (append the two fields after `insufficient_history`):

```python
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
    bars: dict[str, pd.DataFrame]  # clean (quarantine-passed) universe bars, for the M2 simulator
    benchmark_bars: pd.DataFrame
```

and extend the `return` at the end of `build_rankings`:

```python
    return RankingsResult(
        venue=config.name,
        as_of=as_of_ts,
        regime=regime,
        table=table,
        coverage=coverage,
        quarantined=quarantined,
        fetch_failures=tuple(sorted(failures)),
        insufficient_history=insufficient,
        bars=clean,
        benchmark_bars=benchmark,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py -q`
Expected: all PASS (rankings CLI unaffected).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/pipeline.py tests/test_pipeline.py
git commit -m "Expose clean bars and benchmark bars on RankingsResult for the simulator [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Portfolio state model

Pure data structures + JSON dict round-trip. File I/O comes later (Task 10, runner).

**Files:**
- Create: `src/trading/simulator/__init__.py` (empty)
- Create: `src/trading/simulator/state.py`
- Create: `tests/test_state.py`

**Interfaces:**
- Consumes: nothing.
- Produces (consumed by Tasks 4-12):
  - `Position(symbol, qty, entry_price, entry_ts, entry_atr, stop_price, flushed, entry_composite, entry_rank)` — frozen.
  - `PendingOrder(symbol, side, notional, decision_ts, reason, atr_at_decision=0.0, composite=0.0, rank=0)` — frozen; `side` is `"buy" | "sell"`.
  - `Settlement(amount, available_on)` — frozen.
  - `Skip(symbol, action, reason)` — frozen; `symbol="*"` for venue-wide skips.
  - `PortfolioState(venue, cash, settlements, positions, pending_orders, cooldowns, high_water_mark, breaker_tripped, breaker_tripped_at, benchmark_start_price, created_at, last_run_key)` — mutable.
  - `initial_state(venue: str, starting_balance: float, benchmark_start_price: float, created_at: str) -> PortfolioState`
  - `to_state_dict(state: PortfolioState) -> dict` / `state_from_dict(payload: dict) -> PortfolioState` (raises `StateError`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_state.py`:

```python
import pytest

from trading.simulator.state import (
    PendingOrder,
    Position,
    Settlement,
    StateError,
    initial_state,
    state_from_dict,
    to_state_dict,
)


def _populated_state():
    state = initial_state("equities", 1000.0, 620.55, "2026-07-01T22:30:00+00:00")
    state.cash = 640.0
    state.settlements = [Settlement(amount=180.0, available_on="2026-07-03")]
    state.positions = {
        "AAPL": Position(
            symbol="AAPL",
            qty=0.85,
            entry_price=211.5,
            entry_ts="2026-06-25T00:00:00+00:00",
            entry_atr=4.2,
            stop_price=205.2,
            flushed=False,
            entry_composite=0.83,
            entry_rank=2,
        )
    }
    state.pending_orders = [
        PendingOrder(
            symbol="NVDA",
            side="buy",
            notional=180.0,
            decision_ts="2026-07-01T00:00:00+00:00",
            reason="entry",
            atr_at_decision=6.1,
            composite=0.91,
            rank=1,
        ),
        PendingOrder(
            symbol="MSFT",
            side="sell",
            notional=0.0,
            decision_ts="2026-07-01T00:00:00+00:00",
            reason="stop_loss",
        ),
    ]
    state.cooldowns = {"TSLA": "2026-07-05"}
    state.high_water_mark = 1050.0
    state.breaker_tripped = True
    state.breaker_tripped_at = "2026-06-30T00:00:00+00:00"
    state.last_run_key = "equities:2026-06-30T00:00:00+00:00"
    return state


def test_initial_state_shape():
    state = initial_state("crypto", 1000.0, 67000.0, "2026-07-01T06:00:00+00:00")
    assert state.venue == "crypto"
    assert state.cash == 1000.0
    assert state.high_water_mark == 1000.0
    assert state.positions == {}
    assert state.pending_orders == []
    assert state.settlements == []
    assert state.cooldowns == {}
    assert state.breaker_tripped is False
    assert state.breaker_tripped_at is None
    assert state.benchmark_start_price == 67000.0
    assert state.last_run_key is None


def test_dict_round_trip_preserves_everything():
    state = _populated_state()
    payload = to_state_dict(state)
    assert payload["version"] == 1
    restored = state_from_dict(payload)
    assert restored == state


def test_round_trip_survives_json():
    import json

    state = _populated_state()
    restored = state_from_dict(json.loads(json.dumps(to_state_dict(state))))
    assert restored == state


def test_missing_key_raises_state_error():
    payload = to_state_dict(_populated_state())
    del payload["cash"]
    with pytest.raises(StateError, match="corrupt"):
        state_from_dict(payload)


def test_malformed_position_raises_state_error():
    payload = to_state_dict(_populated_state())
    payload["positions"]["AAPL"] = {"symbol": "AAPL"}  # missing fields
    with pytest.raises(StateError, match="corrupt"):
        state_from_dict(payload)


def test_unknown_version_raises_state_error():
    payload = to_state_dict(_populated_state())
    payload["version"] = 99
    with pytest.raises(StateError, match="version"):
        state_from_dict(payload)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.simulator'`.

- [ ] **Step 3: Implement the state module**

Create empty `src/trading/simulator/__init__.py`, then `src/trading/simulator/state.py`:

```python
"""Paper-portfolio state (spec: Portfolio Simulator, State).

Pure data + dict round-trip only. File persistence (atomic write, corrupt
detection) lives in trading.runner; the simulator core never touches disk.
All timestamps are ISO-8601 UTC strings so the state survives JSON exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

STATE_VERSION = 1

Side = Literal["buy", "sell"]


class StateError(RuntimeError):
    """State payload is corrupt or structurally invalid."""


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    entry_price: float  # fill price including slippage
    entry_ts: str  # ISO-8601 UTC timestamp of the fill bar
    entry_atr: float  # ATR frozen at entry (bars through the entry decision bar)
    stop_price: float
    flushed: bool  # one-way regime-flush ratchet already applied
    entry_composite: float  # ranking evidence at decision time (journal/digest rationale)
    entry_rank: int


@dataclass(frozen=True)
class PendingOrder:
    symbol: str
    side: Side
    notional: float  # buys: committed dollars (fee charged on top); sells: 0.0 (full qty)
    decision_ts: str  # ISO-8601 UTC decision-bar timestamp; fills at first bar strictly after
    reason: str  # entry | stop_loss | trend_break | time_stop | forced_exit
    atr_at_decision: float = 0.0  # buys only: ATR to freeze at entry
    composite: float = 0.0  # buys only
    rank: int = 0  # buys only


@dataclass(frozen=True)
class Settlement:
    amount: float
    available_on: str  # ISO date; settled once the decision-bar date reaches it


@dataclass(frozen=True)
class Skip:
    symbol: str  # "*" for venue-wide skips
    action: str  # entry | exit | fill
    reason: str


@dataclass
class PortfolioState:
    venue: str
    cash: float  # settled cash only
    settlements: list[Settlement] = field(default_factory=list)
    positions: dict[str, Position] = field(default_factory=dict)
    pending_orders: list[PendingOrder] = field(default_factory=list)
    cooldowns: dict[str, str] = field(default_factory=dict)  # symbol -> ISO date re-entry allowed
    high_water_mark: float = 0.0
    breaker_tripped: bool = False
    breaker_tripped_at: str | None = None
    benchmark_start_price: float = 0.0  # benchmark close at bootstrap (buy-and-hold baseline)
    created_at: str = ""
    last_run_key: str | None = None


def initial_state(
    venue: str, starting_balance: float, benchmark_start_price: float, created_at: str
) -> PortfolioState:
    return PortfolioState(
        venue=venue,
        cash=starting_balance,
        high_water_mark=starting_balance,
        benchmark_start_price=benchmark_start_price,
        created_at=created_at,
    )


def to_state_dict(state: PortfolioState) -> dict:
    return {"version": STATE_VERSION, **asdict(state)}


def state_from_dict(payload: dict) -> PortfolioState:
    try:
        version = payload["version"]
        if version != STATE_VERSION:
            raise StateError(f"unsupported state version {version!r}")
        return PortfolioState(
            venue=payload["venue"],
            cash=float(payload["cash"]),
            settlements=[Settlement(**s) for s in payload["settlements"]],
            positions={k: Position(**p) for k, p in payload["positions"].items()},
            pending_orders=[PendingOrder(**o) for o in payload["pending_orders"]],
            cooldowns=dict(payload["cooldowns"]),
            high_water_mark=float(payload["high_water_mark"]),
            breaker_tripped=bool(payload["breaker_tripped"]),
            breaker_tripped_at=payload["breaker_tripped_at"],
            benchmark_start_price=float(payload["benchmark_start_price"]),
            created_at=payload["created_at"],
            last_run_key=payload["last_run_key"],
        )
    except StateError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise StateError(f"corrupt portfolio state: {exc}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_state.py -q`
Expected: 6 PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/simulator tests/test_state.py
git commit -m "Add paper-portfolio state model with JSON round-trip [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: ATR and the fill engine

Fills pending orders at the open of the first bar strictly after each order's decision bar, with slippage and taker fees; creates positions with frozen-ATR stops; routes sale proceeds through T+1 settlement; sets stop-out cooldowns. Also adds the `atr_window` config field.

**Files:**
- Create: `src/trading/simulator/fills.py`
- Create: `tests/sim_helpers.py`
- Create: `tests/test_fills.py`
- Modify: `src/trading/config.py:54-69` (PortfolioConfig)
- Modify: `config/equities.toml`, `config/crypto.toml` (`[portfolio] atr_window = 20`)
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: Task 3 types; `CostsConfig`/`PortfolioConfig` from `trading.config`.
- Produces (consumed by Tasks 5-7, 10-11):
  - `atr(bars: pd.DataFrame, window: int) -> float` — simple (Cutler-style) mean true range of the last `window` bars; `math.nan` if fewer than `window + 1` rows.
  - `Fill(symbol, side, qty, price, fee, bar_ts, reason, realized_pnl)` — frozen; `realized_pnl` is `None` for buys.
  - `apply_fills(state: PortfolioState, bars: dict[str, pd.DataFrame], config: VenueConfig) -> tuple[list[Fill], list[Skip]]` — mutates `state` in place (the caller, `core.step`, deep-copies first).
  - `release_settlements(state: PortfolioState, decision_date: datetime.date) -> None` — moves due settlements into `state.cash`.
  - Config: `config.portfolio.atr_window` (int, 20 for both venues).
- Fill-carryover rules (locked interpretation): a **buy** whose symbol has no bar after its decision bar this run is **cancelled** (skip `entry_cancelled_no_fill_bar` — no catch-up entries on stale decisions); a **sell** stays pending until a bar exists (skip `exit_deferred_no_fill_bar` — exits always process eventually). A sell whose position vanished is dropped with skip `exit_orphaned_no_position`.

- [ ] **Step 1: Add `atr_window` to config**

In `src/trading/config.py`, add to `PortfolioConfig` after `stop_atr_multiple`:

```python
    atr_window: int
```

In `config/equities.toml` `[portfolio]`, after the `stop_atr_multiple` line:

```toml
atr_window = 20                     # ATR-20 (spec); frozen at entry
```

In `config/crypto.toml` `[portfolio]`, after the `stop_atr_multiple` line:

```toml
atr_window = 20                     # ATR-20 (spec); frozen at entry
```

Append to `tests/test_config.py::test_load_equities_config`:

```python
    assert config.portfolio.atr_window == 20
```

Run: `uv run pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 2: Write shared simulator test helpers**

Create `tests/sim_helpers.py` (pytest puts each test file's directory on `sys.path`, so test modules can `import sim_helpers` directly):

```python
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
```

- [ ] **Step 3: Write the failing fill tests**

Create `tests/test_fills.py`:

```python
import math

import pandas as pd
import pytest
from sim_helpers import CR, EQ, frame, make_state

from trading.simulator.fills import Fill, apply_fills, atr, release_settlements
from trading.simulator.state import PendingOrder, Position, Settlement

DECISION = "2026-06-30T00:00:00+00:00"  # bars in frame(end="2026-07-01") have one bar after this


def _atr_bars() -> pd.DataFrame:
    idx = pd.date_range("2026-06-01", periods=4, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [10.5, 11.5, 12.5, 13.5],
            "low": [9.5, 10.5, 11.5, 12.5],
            "close": [10.0, 11.0, 12.0, 13.0],
            "volume": [1e6] * 4,
        },
        index=idx,
    )


def test_atr_hand_computed():
    # TR per bar (with prev close): max(high-low, |high-pc|, |low-pc|) = 1.5 for bars 2..4.
    assert atr(_atr_bars(), window=3) == pytest.approx(1.5)


def test_atr_insufficient_history_is_nan():
    assert math.isnan(atr(_atr_bars(), window=4))


def _buy_order(symbol="AAA", notional=180.0, entry_atr=4.0) -> PendingOrder:
    return PendingOrder(
        symbol=symbol,
        side="buy",
        notional=notional,
        decision_ts=DECISION,
        reason="entry",
        atr_at_decision=entry_atr,
        composite=0.9,
        rank=1,
    )


def _position(symbol="AAA", qty=2.0, entry_price=90.0, stop=80.0) -> Position:
    return Position(
        symbol=symbol,
        qty=qty,
        entry_price=entry_price,
        entry_ts="2026-06-20T00:00:00+00:00",
        entry_atr=4.0,
        stop_price=stop,
        flushed=False,
        entry_composite=0.8,
        entry_rank=1,
    )


def test_buy_fills_at_next_bar_open_with_slippage_and_creates_frozen_stop():
    bars = {"AAA": frame(end="2026-07-01")}  # constant close/open 100.0
    state = make_state(EQ, pending_orders=[_buy_order()])
    fills, skips = apply_fills(state, bars, EQ)

    assert skips == []
    assert len(fills) == 1
    fill = fills[0]
    price = 100.0 * (1 + 5.0 / 1e4)  # open + 5 bps slippage
    assert fill.price == pytest.approx(price)
    assert fill.qty == pytest.approx(180.0 / price)
    assert fill.fee == pytest.approx(0.0)  # equities: zero commission
    assert fill.realized_pnl is None
    # The fill bar is the first bar strictly after the decision bar (2026-07-01).
    assert fill.bar_ts == "2026-07-01T00:00:00+00:00"
    position = state.positions["AAA"]
    assert position.entry_atr == 4.0
    assert position.stop_price == pytest.approx(price - EQ.portfolio.stop_atr_multiple * 4.0)
    assert position.flushed is False
    assert state.cash == pytest.approx(1000.0 - 180.0)
    assert state.pending_orders == []


def test_crypto_buy_pays_taker_fee():
    bars = {"BTC": frame(end="2026-07-01")}
    state = make_state(CR, pending_orders=[_buy_order(symbol="BTC", notional=300.0)])
    fills, _ = apply_fills(state, bars, CR)
    assert fills[0].fee == pytest.approx(300.0 * 95.0 / 1e4)
    assert state.cash == pytest.approx(1000.0 - 300.0 - 300.0 * 95.0 / 1e4)


def test_equities_sell_settles_t_plus_1_and_realizes_pnl():
    bars = {"AAA": frame(end="2026-07-01")}
    state = make_state(EQ, positions={"AAA": _position()}, cash=0.0)
    state.pending_orders = [
        PendingOrder(symbol="AAA", side="sell", notional=0.0, decision_ts=DECISION, reason="trend_break")
    ]
    fills, _ = apply_fills(state, bars, EQ)

    price = 100.0 * (1 - 5.0 / 1e4)
    proceeds = 2.0 * price  # zero commission
    assert fills[0].realized_pnl == pytest.approx(proceeds - 2.0 * 90.0)
    assert state.positions == {}
    assert state.cash == 0.0  # unspendable until settled (T+1)
    assert state.settlements == [Settlement(amount=proceeds, available_on="2026-07-02")]
    # trend_break is not a stop-out: no cooldown.
    assert state.cooldowns == {}


def test_crypto_sell_is_immediately_settled():
    bars = {"BTC": frame(end="2026-07-01")}
    state = make_state(CR, positions={"BTC": _position(symbol="BTC")}, cash=0.0)
    state.pending_orders = [
        PendingOrder(symbol="BTC", side="sell", notional=0.0, decision_ts=DECISION, reason="time_stop")
    ]
    apply_fills(state, bars, CR)
    price = 100.0 * (1 - 5.0 / 1e4)
    gross = 2.0 * price
    assert state.settlements == []
    assert state.cash == pytest.approx(gross - gross * 95.0 / 1e4)


def test_stop_loss_fill_sets_reentry_cooldown():
    bars = {"AAA": frame(end="2026-07-01")}
    state = make_state(EQ, positions={"AAA": _position()})
    state.pending_orders = [
        PendingOrder(symbol="AAA", side="sell", notional=0.0, decision_ts=DECISION, reason="stop_loss")
    ]
    apply_fills(state, bars, EQ)
    # Fill bar 2026-07-01 + 7-day cooldown -> re-entry allowed 2026-07-08.
    assert state.cooldowns == {"AAA": "2026-07-08"}


def test_buy_without_fill_bar_is_cancelled_and_sell_is_deferred():
    bars = {"AAA": frame(end="2026-06-30"), "BBB": frame(end="2026-06-30")}
    state = make_state(EQ, positions={"BBB": _position(symbol="BBB")})
    state.pending_orders = [
        _buy_order(symbol="AAA"),
        PendingOrder(symbol="BBB", side="sell", notional=0.0, decision_ts=DECISION, reason="stop_loss"),
    ]
    fills, skips = apply_fills(state, bars, EQ)
    assert fills == []
    reasons = {(s.symbol, s.reason) for s in skips}
    assert ("AAA", "entry_cancelled_no_fill_bar") in reasons
    assert ("BBB", "exit_deferred_no_fill_bar") in reasons
    assert [o.symbol for o in state.pending_orders] == ["BBB"]  # sell kept, buy dropped
    assert "BBB" in state.positions


def test_orphaned_sell_is_dropped():
    from trading.simulator.state import Skip

    bars = {"AAA": frame(end="2026-07-01")}
    state = make_state(EQ)  # no positions
    state.pending_orders = [
        PendingOrder(symbol="AAA", side="sell", notional=0.0, decision_ts=DECISION, reason="stop_loss")
    ]
    fills, skips = apply_fills(state, bars, EQ)
    assert fills == []
    assert skips == [Skip("AAA", "fill", "exit_orphaned_no_position")]
    assert state.pending_orders == []


def test_release_settlements_moves_due_cash():
    import datetime

    state = make_state(EQ, cash=10.0)
    state.settlements = [
        Settlement(amount=100.0, available_on="2026-07-01"),
        Settlement(amount=50.0, available_on="2026-07-05"),
    ]
    release_settlements(state, datetime.date(2026, 7, 1))
    assert state.cash == pytest.approx(110.0)
    assert state.settlements == [Settlement(amount=50.0, available_on="2026-07-05")]


def test_fill_price_within_next_bar_range_plus_slippage():
    """Spec property: fills are anchored to the next bar's open, so they sit
    within [low*(1-slip), high*(1+slip)] of the fill bar."""
    bars = {"AAA": frame(end="2026-07-01")}
    state = make_state(EQ, pending_orders=[_buy_order()])
    fills, _ = apply_fills(state, bars, EQ)
    fill_bar = bars["AAA"].loc[pd.Timestamp("2026-07-01", tz="UTC")]
    slip = 5.0 / 1e4
    assert fill_bar["low"] * (1 - slip) <= fills[0].price <= fill_bar["high"] * (1 + slip)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_fills.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.simulator.fills'`.

- [ ] **Step 5: Implement the fill engine**

Create `src/trading/simulator/fills.py`:

```python
"""Fill engine (spec: Fill model / Execution Split, phase 1 of each run).

Pending orders written by the PREVIOUS run fill at the open of the first bar
strictly after their decision bar, plus slippage, plus taker fees. Pure: the
caller (core.step) deep-copies state before handing it here.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass

import pandas as pd

from trading.config import VenueConfig
from trading.simulator.state import PendingOrder, PortfolioState, Position, Settlement, Skip


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    bar_ts: str  # ISO-8601 UTC timestamp of the fill bar
    reason: str
    realized_pnl: float | None  # sells only


def atr(bars: pd.DataFrame, window: int) -> float:
    """Simple (Cutler-style) ATR: mean true range of the last `window` bars.

    Needs window + 1 rows (the previous close seeds the first true range).
    """
    if len(bars) < window + 1:
        return math.nan
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return float(true_range.iloc[-window:].mean())


def release_settlements(state: PortfolioState, decision_date: datetime.date) -> None:
    """Move settled sale proceeds into spendable cash (T+1 for equities)."""
    remaining: list[Settlement] = []
    for settlement in state.settlements:
        if datetime.date.fromisoformat(settlement.available_on) <= decision_date:
            state.cash += settlement.amount
        else:
            remaining.append(settlement)
    state.settlements = remaining


def apply_fills(
    state: PortfolioState, bars: dict[str, pd.DataFrame], config: VenueConfig
) -> tuple[list[Fill], list[Skip]]:
    """Fill pending orders against current bars. Mutates state; returns fills + skips.

    Sells process before buys (deterministic order: side, then symbol). A buy
    with no bar after its decision bar is CANCELLED (stale decision — no
    catch-up entries); a sell is kept pending (exits always process
    eventually); a sell without a matching position is dropped.
    """
    slip = config.costs.slippage_bps / 1e4
    fee_rate = config.costs.taker_fee_bps / 1e4
    fills: list[Fill] = []
    skips: list[Skip] = []
    remaining: list[PendingOrder] = []

    for order in sorted(state.pending_orders, key=lambda o: (o.side != "sell", o.symbol)):
        df = bars.get(order.symbol)
        after = df[df.index > pd.Timestamp(order.decision_ts)] if df is not None else None
        if after is None or after.empty:
            if order.side == "buy":
                skips.append(Skip(order.symbol, "fill", "entry_cancelled_no_fill_bar"))
            else:
                remaining.append(order)
                skips.append(Skip(order.symbol, "fill", "exit_deferred_no_fill_bar"))
            continue

        bar_ts: pd.Timestamp = after.index[0]
        open_price = float(after["open"].iloc[0])

        if order.side == "buy":
            price = open_price * (1 + slip)
            qty = order.notional / price
            fee = order.notional * fee_rate
            state.cash -= order.notional + fee
            state.positions[order.symbol] = Position(
                symbol=order.symbol,
                qty=qty,
                entry_price=price,
                entry_ts=bar_ts.isoformat(),
                entry_atr=order.atr_at_decision,
                stop_price=price - config.portfolio.stop_atr_multiple * order.atr_at_decision,
                flushed=False,
                entry_composite=order.composite,
                entry_rank=order.rank,
            )
            fills.append(
                Fill(order.symbol, "buy", qty, price, fee, bar_ts.isoformat(), order.reason, None)
            )
        else:
            position = state.positions.pop(order.symbol, None)
            if position is None:
                skips.append(Skip(order.symbol, "fill", "exit_orphaned_no_position"))
                continue
            price = open_price * (1 - slip)
            gross = position.qty * price
            fee = gross * fee_rate
            proceeds = gross - fee
            if config.costs.settlement_days == 0:
                state.cash += proceeds
            else:
                available = bar_ts.date() + datetime.timedelta(days=config.costs.settlement_days)
                state.settlements.append(
                    Settlement(amount=proceeds, available_on=available.isoformat())
                )
            if order.reason == "stop_loss":
                until = bar_ts.date() + datetime.timedelta(days=config.portfolio.cooldown_days)
                state.cooldowns[order.symbol] = until.isoformat()
            fills.append(
                Fill(
                    order.symbol,
                    "sell",
                    position.qty,
                    price,
                    fee,
                    bar_ts.isoformat(),
                    order.reason,
                    proceeds - position.qty * position.entry_price,
                )
            )

    state.pending_orders = remaining
    return fills, skips
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_fills.py tests/test_config.py -q`
Expected: all PASS.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/simulator/fills.py tests/sim_helpers.py tests/test_fills.py src/trading/config.py config/equities.toml config/crypto.toml tests/test_config.py
git commit -m "Add fill engine with frozen-ATR stops, slippage/fees, and T+1 settlement [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Exit rules

Checked before entries, against the UNFILTERED ranking (held names always stay rankable; filter mechanics can never manufacture a spurious exit). Emits pending sell orders that fill next run.

**Files:**
- Create: `src/trading/simulator/exits.py`
- Create: `tests/test_exits.py`

**Interfaces:**
- Consumes: Task 3 types, `RankingsResult` (Task 2), `VenueConfig`.
- Produces (consumed by Task 7):
  - `evaluate_exits(state: PortfolioState, rankings: RankingsResult, config: VenueConfig, decision_ts: pd.Timestamp) -> tuple[list[PendingOrder], list[Skip], list[str]]` — mutates `state.positions` in place only to apply the one-way regime-flush stop ratchet; returns new sell orders, skips, warnings.
- Exit priority per position (first match wins, one order max): forced (sell_only / untradable / delisted) > stop_loss > trend_break > time_stop. Quarantined or fetch-failed held symbols are NOT exited (no trades on bad data) — they produce a warning + skip instead. Positions with an already-pending sell are not re-evaluated.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_exits.py`:

```python
import dataclasses

import pandas as pd
from sim_helpers import EQ, frame, make_rankings, make_state, make_table

from trading.simulator.exits import evaluate_exits
from trading.simulator.state import PendingOrder, Position

DECISION = pd.Timestamp("2026-07-01", tz="UTC")


# Default entry_ts is recent (4 sessions held) so the 20-session time stop
# stays out of the way except in the tests that age the position explicitly.
OLD_ENTRY = "2026-06-01T00:00:00+00:00"  # 22 sessions before DECISION


def _position(symbol, *, entry_price=100.0, stop=94.0, entry_atr=4.0, flushed=False,
              entry_ts="2026-06-25T00:00:00+00:00"):
    return Position(
        symbol=symbol,
        qty=1.0,
        entry_price=entry_price,
        entry_ts=entry_ts,
        entry_atr=entry_atr,
        stop_price=stop,
        flushed=flushed,
        entry_composite=0.8,
        entry_rank=1,
    )


def _row(status="tradable", composite=0.9):
    return {"status": status, "composite": composite, "raw_return_30d": 0.10}


def _rankings(bars, rows, **kwargs):
    return make_rankings(EQ, bars, make_table(rows), **kwargs)


def test_stop_loss_on_close_at_or_below_frozen_stop():
    bars = {"AAA": frame(start_price=93.0)}  # constant close 93 <= stop 94
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    orders, _, _ = evaluate_exits(state, _rankings(bars, {"AAA": _row()}), EQ, DECISION)
    assert [(o.symbol, o.side, o.reason) for o in orders] == [("AAA", "sell", "stop_loss")]
    assert orders[0].decision_ts == DECISION.isoformat()


def test_no_stop_when_close_above_frozen_stop_even_in_high_vol():
    # The stop is FROZEN at entry: current volatility never widens or triggers it.
    bars = {"AAA": frame(start_price=95.0)}
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    orders, _, _ = evaluate_exits(state, _rankings(bars, {"AAA": _row()}), EQ, DECISION)
    assert orders == []
    assert state.positions["AAA"].stop_price == 94.0  # untouched


def test_regime_flush_ratchets_stop_once_and_never_loosens():
    bars = {"AAA": frame(start_price=99.0)}
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    rankings = _rankings(bars, {"AAA": _row()}, regime_state="risk_off")

    orders, _, _ = evaluate_exits(state, rankings, EQ, DECISION)
    # Ratchet: entry 100 - 1.0 * ATR 4 = 96; close 99 > 96 so no exit yet.
    assert orders == []
    assert state.positions["AAA"].stop_price == 96.0
    assert state.positions["AAA"].flushed is True

    # Second risk_off run must not re-apply; recovery must not loosen.
    state.positions["AAA"] = dataclasses.replace(state.positions["AAA"], stop_price=98.0)
    evaluate_exits(state, rankings, EQ, DECISION)
    assert state.positions["AAA"].stop_price == 98.0
    recovered = _rankings(bars, {"AAA": _row()}, regime_state="risk_on")
    evaluate_exits(state, recovered, EQ, DECISION)
    assert state.positions["AAA"].stop_price == 98.0


def test_regime_flush_can_trigger_immediate_stop():
    bars = {"AAA": frame(start_price=95.0)}  # close 95 < ratcheted 96
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    rankings = _rankings(bars, {"AAA": _row()}, regime_state="risk_off")
    orders, _, _ = evaluate_exits(state, rankings, EQ, DECISION)
    assert [(o.symbol, o.reason) for o in orders] == [("AAA", "stop_loss")]


def test_trend_break_requires_bottom_half_rank_and_close_below_mean():
    # 20-day mean of a downtrending series sits above the last close.
    down = frame(drift=-0.01, start_price=200.0)
    bars = {"AAA": down, "BBB": frame(), "CCC": frame(), "DDD": frame()}
    rows = {
        "BBB": _row(composite=0.9),
        "CCC": _row(composite=0.8),
        "DDD": _row(composite=0.7),
        "AAA": _row(composite=0.1),  # rank 4 of 4: bottom half
    }
    state = make_state(EQ, positions={"AAA": _position("AAA", stop=1.0)})  # stop can't fire
    orders, _, _ = evaluate_exits(state, _rankings(bars, rows), EQ, DECISION)
    assert [(o.symbol, o.reason) for o in orders] == [("AAA", "trend_break")]


def test_no_trend_break_when_bottom_half_but_above_mean():
    up = frame(drift=0.01)  # rising: close above its 20-day mean
    bars = {"AAA": up, "BBB": frame(), "CCC": frame(), "DDD": frame()}
    rows = {
        "BBB": _row(composite=0.9),
        "CCC": _row(composite=0.8),
        "DDD": _row(composite=0.7),
        "AAA": _row(composite=0.1),
    }
    state = make_state(EQ, positions={"AAA": _position("AAA", stop=1.0)})
    orders, _, _ = evaluate_exits(state, _rankings(bars, rows), EQ, DECISION)
    assert orders == []


def test_time_stop_fires_only_when_flat_to_down():
    # Held > 20 sessions (entry June 1, bars through July 1 = 22 sessions after).
    flat = frame(start_price=100.0)
    state = make_state(
        EQ, positions={"AAA": _position("AAA", entry_price=100.0, stop=1.0, entry_ts=OLD_ENTRY)}
    )
    rows = {"AAA": _row(composite=0.9)}
    orders, _, _ = evaluate_exits(state, _rankings({"AAA": flat}, rows), EQ, DECISION)
    assert [(o.symbol, o.reason) for o in orders] == [("AAA", "time_stop")]

    # Same age but profitable: dead-money rule does not fire.
    state = make_state(
        EQ, positions={"AAA": _position("AAA", entry_price=90.0, stop=1.0, entry_ts=OLD_ENTRY)}
    )
    orders, _, _ = evaluate_exits(state, _rankings({"AAA": flat}, rows), EQ, DECISION)
    assert orders == []


def test_young_position_has_no_time_stop():
    # Default _position entry_ts is 4 sessions old; flat at entry price would
    # trip the time stop only after 20 sessions.
    state = make_state(EQ, positions={"AAA": _position("AAA", stop=1.0)})
    orders, _, _ = evaluate_exits(state, _rankings({"AAA": frame()}, {"AAA": _row()}), EQ, DECISION)
    assert orders == []


def test_forced_exit_on_sell_only_untradable_and_delisted():
    bars = {"AAA": frame(), "BBB": frame()}
    rows = {"AAA": _row(status="sell_only"), "BBB": _row(status="untradable")}
    state = make_state(
        EQ,
        positions={
            "AAA": _position("AAA", stop=1.0),
            "BBB": _position("BBB", stop=1.0),
            "GONE": _position("GONE", stop=1.0),  # absent everywhere: delisted
        },
    )
    orders, _, _ = evaluate_exits(state, _rankings(bars, rows), EQ, DECISION)
    assert {(o.symbol, o.reason) for o in orders} == {
        ("AAA", "forced_exit"),
        ("BBB", "forced_exit"),
        ("GONE", "forced_exit"),
    }


def test_quarantined_held_symbol_warns_and_holds():
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    rankings = make_rankings(
        EQ, {"BBB": frame()}, make_table({"BBB": _row()}), quarantined=("AAA",)
    )
    orders, skips, warnings = evaluate_exits(state, rankings, EQ, DECISION)
    assert orders == []
    assert any(s.symbol == "AAA" and s.reason == "quarantined_no_trades" for s in skips)
    assert any("AAA" in w for w in warnings)
    assert "AAA" in state.positions


def test_pending_sell_is_not_duplicated():
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    state.pending_orders = [
        PendingOrder(symbol="AAA", side="sell", notional=0.0,
                     decision_ts="2026-06-30T00:00:00+00:00", reason="stop_loss")
    ]
    bars = {"AAA": frame(start_price=50.0)}  # way below stop
    orders, _, _ = evaluate_exits(state, _rankings(bars, {"AAA": _row()}), EQ, DECISION)
    assert orders == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_exits.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.simulator.exits'`.

- [ ] **Step 3: Implement the exit rules**

Create `src/trading/simulator/exits.py`:

```python
"""Exit rules (spec: Portfolio Simulator — exits checked before entries).

Evaluated against the UNFILTERED ranking: held names always remain rankable,
so entry-filter mechanics can never manufacture a spurious exit. Quarantined
or fetch-failed held symbols are never traded this run (bad data), only
warned about. Exits emit pending sell orders that fill next run.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from trading.config import VenueConfig
from trading.pipeline import RankingsResult
from trading.simulator.state import PendingOrder, PortfolioState, Skip

FORCED_STATUSES = {"sell_only", "untradable"}


def _sell(symbol: str, decision_ts: pd.Timestamp, reason: str) -> PendingOrder:
    return PendingOrder(
        symbol=symbol, side="sell", notional=0.0, decision_ts=decision_ts.isoformat(), reason=reason
    )


def evaluate_exits(
    state: PortfolioState,
    rankings: RankingsResult,
    config: VenueConfig,
    decision_ts: pd.Timestamp,
) -> tuple[list[PendingOrder], list[Skip], list[str]]:
    orders: list[PendingOrder] = []
    skips: list[Skip] = []
    warnings: list[str] = []
    pending_sells = {o.symbol for o in state.pending_orders if o.side == "sell"}
    ranked = list(rankings.table.index)

    for symbol, position in sorted(state.positions.items()):
        if symbol in pending_sells:
            continue
        if symbol in rankings.quarantined:
            warnings.append(f"{symbol}: held position quarantined; no trades until it clears")
            skips.append(Skip(symbol, "exit", "quarantined_no_trades"))
            continue
        if symbol in rankings.fetch_failures:
            warnings.append(f"{symbol}: held position fetch failed; exit rules not evaluated")
            skips.append(Skip(symbol, "exit", "fetch_failure_no_evaluation"))
            continue

        in_table = symbol in rankings.table.index
        df = rankings.bars.get(symbol)
        if not in_table and df is None:
            # Dropped from the venue universe entirely: delisted.
            orders.append(_sell(symbol, decision_ts, "forced_exit"))
            continue
        if in_table and str(rankings.table.loc[symbol, "status"]) in FORCED_STATUSES:
            orders.append(_sell(symbol, decision_ts, "forced_exit"))
            continue

        window = df[df.index <= decision_ts] if df is not None else None
        if window is None or window.empty:
            warnings.append(f"{symbol}: held but no bars this run; exit rules not evaluated")
            skips.append(Skip(symbol, "exit", "no_bars_no_evaluation"))
            continue
        last_close = float(window["close"].iloc[-1])

        # Regime flush: recompute the stop ONCE at 1.0x the frozen entry ATR.
        # One-way ratchet — never loosened until the position closes.
        if rankings.regime.state == "risk_off" and not position.flushed:
            ratchet = (
                position.entry_price
                - config.portfolio.regime_flush_atr_multiple * position.entry_atr
            )
            position = replace(
                position, stop_price=max(position.stop_price, ratchet), flushed=True
            )
            state.positions[symbol] = position

        if last_close <= position.stop_price:
            orders.append(_sell(symbol, decision_ts, "stop_loss"))
            continue

        if in_table:
            rank_pos = ranked.index(symbol) + 1
            mean = float(window["close"].iloc[-config.signals.mean_window :].mean())
            if rank_pos > len(ranked) / 2 and last_close < mean:
                orders.append(_sell(symbol, decision_ts, "trend_break"))
                continue
        else:
            warnings.append(f"{symbol}: held but unranked this run; trend-break not evaluated")

        bars_held = int((window.index > pd.Timestamp(position.entry_ts)).sum())
        if bars_held >= config.portfolio.time_stop_bars and last_close <= position.entry_price:
            orders.append(_sell(symbol, decision_ts, "time_stop"))

    return orders, skips, warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_exits.py -q`
Expected: 11 PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/simulator/exits.py tests/test_exits.py
git commit -m "Add exit rules: frozen stops, regime-flush ratchet, trend break, time stop, forced exits [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Entry rules

Regime-gated top-of-ranking entries with every hard rail and a skip reason for each blocked candidate.

**Files:**
- Create: `src/trading/simulator/entries.py`
- Create: `tests/test_entries.py`
- Modify: `config/crypto.toml` (`max_daily_deployment_pct`, see Step 1)

**Interfaces:**
- Consumes: Task 3 types, `atr()` (Task 4), `RankingsResult` (Task 2), `VenueConfig`.
- Produces (consumed by Tasks 7, 9):
  - `evaluate_entries(state: PortfolioState, rankings: RankingsResult, config: VenueConfig, decision_ts: pd.Timestamp, portfolio_value: float) -> tuple[list[PendingOrder], list[Skip]]` — pure w.r.t. state (reads only). Task 9 appends an `earnings` keyword parameter.
- Gate order per candidate (walking the ranked table top-down): composite ≥ `entry_score_threshold` (below it, iteration stops — the table is sorted); not already held / no pending sell (never average down); `status == "tradable"`; not in cooldown; dollar-volume floor (only if `min_dollar_volume > 0`; trailing `mean_window`-bar average of close×volume); crypto fee gate (only if `min_raw_return_cost_multiple > 0`: `raw_return_30d ≥ multiple × round-trip cost`, round-trip = `2 × (taker_fee_bps + slippage_bps) / 1e4`); free slot under `floor(max_positions × exposure_multiplier)`; ATR computable; 25% daily-deployment budget; settled cash ≥ `notional × (1 + taker fee)`. Slot/budget/cash exhaustion records one skip then stops (every later candidate would hit the same wall). Venue-wide gates first: tripped circuit breaker or risk_off regime yield a single `Skip("*", "entry", ...)`.
- Sizing: `notional = position_size_pct × portfolio_value`; buys carry `atr_at_decision`, `composite`, `rank` for the fill engine and journal.

- [ ] **Step 1: Fix the crypto deployment-cap deadlock in config**

The spec's ~30% crypto position size cannot fit through a 25% daily-deployment cap — with both at their spec defaults, crypto could NEVER open a position. The cap is config, not constant (spec: "Every number above is config"), so raise it just enough for exactly one full-size crypto entry per day. In `config/crypto.toml` `[portfolio]`, replace the `max_daily_deployment_pct` line:

```toml
max_daily_deployment_pct = 0.35     # must exceed position_size_pct (0.30) or entries deadlock;
                                    # still allows only ONE new crypto position per day
```

(Equities keeps 0.25: with 18% sizing that deliberately paces deployment to one entry per day.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_entries.py`:

```python
import math

import pandas as pd
import pytest
from sim_helpers import CR, EQ, frame, make_rankings, make_state, make_table

from trading.simulator.entries import evaluate_entries
from trading.simulator.state import PendingOrder, Position, Skip

DECISION = pd.Timestamp("2026-07-01", tz="UTC")
VALUE = 1000.0


def _row(status="tradable", composite=0.9, raw_return=0.10):
    return {"status": status, "composite": composite, "raw_return_30d": raw_return}


def _position(symbol):
    return Position(
        symbol=symbol, qty=1.0, entry_price=100.0, entry_ts="2026-06-25T00:00:00+00:00",
        entry_atr=4.0, stop_price=94.0, flushed=False, entry_composite=0.8, entry_rank=1,
    )


def _entries(config, bars, rows, state=None, regime_state="risk_on", value=VALUE):
    state = state if state is not None else make_state(config)
    rankings = make_rankings(config, bars, make_table(rows), regime_state=regime_state)
    return evaluate_entries(state, rankings, config, DECISION, value), state


def test_top_candidate_becomes_buy_order_with_decision_evidence():
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row(composite=0.95)})
    assert len(orders) == 1
    order = orders[0]
    assert order == PendingOrder(
        symbol="AAA",
        side="buy",
        notional=pytest.approx(0.18 * VALUE),
        decision_ts=DECISION.isoformat(),
        reason="entry",
        atr_at_decision=pytest.approx(4.0),  # frame(): TR = 0.04 * close = 4.0
        composite=pytest.approx(0.95),
        rank=1,
    )


def test_below_threshold_is_not_entered():
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row(composite=0.69)})
    assert orders == []
    assert skips == []  # below-threshold tail is not journal noise


def test_already_held_never_averages_down():
    state = make_state(EQ, positions={"AAA": _position("AAA")})
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row()}, state=state)
    assert orders == []
    assert Skip("AAA", "entry", "already_held") in skips


def test_non_tradable_status_blocked_from_entry():
    (orders, skips), _ = _entries(
        EQ, {"AAA": frame()}, {"AAA": _row(status="sell_only")}
    )
    assert orders == []
    assert Skip("AAA", "entry", "status_sell_only") in skips


def test_cooldown_blocks_reentry_until_expiry():
    state = make_state(EQ, cooldowns={"AAA": "2026-07-05"})
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row()}, state=state)
    assert orders == []
    assert Skip("AAA", "entry", "cooldown") in skips

    # On the expiry date itself, re-entry is allowed again.
    state = make_state(EQ, cooldowns={"AAA": "2026-07-01"})
    (orders, _), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row()}, state=state)
    assert len(orders) == 1


def test_equities_dollar_volume_floor():
    thin = frame(volume=1e3)  # 100 * 1e3 = 1e5 << 2e7 floor
    (orders, skips), _ = _entries(EQ, {"AAA": thin}, {"AAA": _row()})
    assert orders == []
    assert Skip("AAA", "entry", "below_dollar_volume_floor") in skips


def test_crypto_fee_gate_requires_raw_return_multiple():
    # Round trip = 2 * (95 + 5) bps = 2%; gate = 3x = 6%.
    (orders, skips), _ = _entries(CR, {"BTC": frame()}, {"BTC": _row(raw_return=0.05)})
    assert orders == []
    assert Skip("BTC", "entry", "fee_gate") in skips

    (orders, _), _ = _entries(CR, {"BTC": frame()}, {"BTC": _row(raw_return=0.10)})
    assert len(orders) == 1


def test_fee_gate_not_applied_to_equities():
    # Equities multiple is 0.0: negative raw momentum must not block entry.
    (orders, _), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row(raw_return=-0.02)})
    assert len(orders) == 1


def test_neutral_regime_halves_position_slots():
    # floor(5 * 0.5) = 2 slots; 2 already held -> venue full.
    state = make_state(
        EQ, positions={"XXX": _position("XXX"), "YYY": _position("YYY")}
    )
    bars = {"AAA": frame(), "XXX": frame(), "YYY": frame()}
    rows = {"AAA": _row(), "XXX": _row(composite=0.85), "YYY": _row(composite=0.84)}
    (orders, skips), _ = _entries(EQ, bars, rows, state=state, regime_state="neutral")
    assert orders == []
    assert Skip("AAA", "entry", "no_free_slot") in skips


def test_risk_off_blocks_all_entries():
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row()}, regime_state="risk_off")
    assert orders == []
    assert skips == [Skip("*", "entry", "regime_risk_off")]


def test_tripped_breaker_blocks_all_entries():
    state = make_state(EQ, breaker_tripped=True, breaker_tripped_at="2026-06-30T00:00:00+00:00")
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row()}, state=state)
    assert orders == []
    assert skips == [Skip("*", "entry", "circuit_breaker")]


def test_daily_deployment_cap_allows_one_equities_entry_per_day():
    bars = {"AAA": frame(), "BBB": frame()}
    rows = {"AAA": _row(composite=0.95), "BBB": _row(composite=0.94)}
    (orders, skips), _ = _entries(EQ, bars, rows)
    # 18% sizing: first entry (180) fits the 25% budget (250); second would breach it.
    assert [o.symbol for o in orders] == ["AAA"]
    assert Skip("BBB", "entry", "daily_deployment_cap") in skips


def test_crypto_full_size_entry_fits_deployment_cap():
    # Guards the Step-1 config fix: 30% sizing must clear the (raised) 35% cap
    # for exactly one entry; a second same-day entry is capped.
    bars = {"BTC": frame(), "ETH": frame()}
    rows = {"BTC": _row(composite=0.95), "ETH": _row(composite=0.94)}
    (orders, skips), _ = _entries(CR, bars, rows)
    assert [o.symbol for o in orders] == ["BTC"]
    assert orders[0].notional == pytest.approx(300.0)
    assert Skip("ETH", "entry", "daily_deployment_cap") in skips


def test_insufficient_settled_cash_blocks_entry():
    state = make_state(EQ, cash=100.0)  # value stays 1000 (rest is in settlements)
    state.settlements = []
    (orders, skips), _ = _entries(EQ, {"AAA": frame()}, {"AAA": _row()}, state=state)
    assert orders == []
    assert Skip("AAA", "entry", "insufficient_settled_cash") in skips


def test_atr_unavailable_blocks_entry():
    short = frame(periods=10)  # < atr_window + 1
    (orders, skips), _ = _entries(EQ, {"AAA": short}, {"AAA": _row()})
    assert orders == []
    assert Skip("AAA", "entry", "insufficient_history_for_atr") in skips


def test_nan_composite_stops_iteration():
    (orders, skips), _ = _entries(
        EQ, {"AAA": frame()}, {"AAA": _row(composite=math.nan)}
    )
    assert orders == []
    assert skips == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_entries.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.simulator.entries'`.

- [ ] **Step 4: Implement the entry rules**

Create `src/trading/simulator/entries.py`:

```python
"""Entry rules (spec: Portfolio Simulator — entries after exits, regime-gated).

Walks the ranked table top-down and emits buy orders for the next bar, with a
journaled skip reason for every blocked candidate above the score threshold.
Hard rails: never average down, never exceed the regime-scaled position count,
no margin (settled cash only), max 25% of portfolio deployed per day.
"""

from __future__ import annotations

import datetime
import math

import pandas as pd

from trading.config import VenueConfig
from trading.pipeline import RankingsResult
from trading.simulator.fills import atr
from trading.simulator.state import PendingOrder, PortfolioState, Skip


def evaluate_entries(
    state: PortfolioState,
    rankings: RankingsResult,
    config: VenueConfig,
    decision_ts: pd.Timestamp,
    portfolio_value: float,
) -> tuple[list[PendingOrder], list[Skip]]:
    p = config.portfolio
    if state.breaker_tripped:
        return [], [Skip("*", "entry", "circuit_breaker")]
    if rankings.regime.state == "risk_off":
        return [], [Skip("*", "entry", "regime_risk_off")]

    orders: list[PendingOrder] = []
    skips: list[Skip] = []
    pending_sells = {o.symbol for o in state.pending_orders if o.side == "sell"}
    pending_buys = sum(1 for o in state.pending_orders if o.side == "buy")
    slots = (
        math.floor(p.max_positions * rankings.regime.exposure_multiplier)
        - len(state.positions)
        - pending_buys
    )
    budget = p.max_daily_deployment_pct * portfolio_value
    cash = state.cash
    fee_rate = config.costs.taker_fee_bps / 1e4
    round_trip_cost = 2 * (config.costs.taker_fee_bps + config.costs.slippage_bps) / 1e4
    decision_date = decision_ts.date()

    for rank_pos, (symbol, row) in enumerate(rankings.table.iterrows(), start=1):
        composite = float(row["composite"])
        if math.isnan(composite) or composite < p.entry_score_threshold:
            break  # sorted desc: no candidate below this can qualify
        if symbol in state.positions or symbol in pending_sells:
            skips.append(Skip(symbol, "entry", "already_held"))
            continue
        if str(row["status"]) != "tradable":
            skips.append(Skip(symbol, "entry", f"status_{row['status']}"))
            continue
        cooldown = state.cooldowns.get(symbol)
        if cooldown is not None and decision_date < datetime.date.fromisoformat(cooldown):
            skips.append(Skip(symbol, "entry", "cooldown"))
            continue
        df = rankings.bars.get(symbol)
        window = df[df.index <= decision_ts] if df is not None else None
        if window is None or window.empty:
            skips.append(Skip(symbol, "entry", "no_bars"))
            continue
        if config.universe.min_dollar_volume > 0:
            recent = window["close"].iloc[-config.signals.mean_window :] * window[
                "volume"
            ].iloc[-config.signals.mean_window :]
            if float(recent.mean()) < config.universe.min_dollar_volume:
                skips.append(Skip(symbol, "entry", "below_dollar_volume_floor"))
                continue
        if p.min_raw_return_cost_multiple > 0:
            raw = row["raw_return_30d"]
            if pd.isna(raw) or float(raw) < p.min_raw_return_cost_multiple * round_trip_cost:
                skips.append(Skip(symbol, "entry", "fee_gate"))
                continue
        if slots <= 0:
            skips.append(Skip(symbol, "entry", "no_free_slot"))
            break  # every later candidate hits the same wall
        entry_atr = atr(window, p.atr_window)
        if math.isnan(entry_atr):
            skips.append(Skip(symbol, "entry", "insufficient_history_for_atr"))
            continue
        notional = p.position_size_pct * portfolio_value
        if notional > budget:
            skips.append(Skip(symbol, "entry", "daily_deployment_cap"))
            break
        if notional * (1 + fee_rate) > cash:
            skips.append(Skip(symbol, "entry", "insufficient_settled_cash"))
            break
        orders.append(
            PendingOrder(
                symbol=symbol,
                side="buy",
                notional=notional,
                decision_ts=decision_ts.isoformat(),
                reason="entry",
                atr_at_decision=entry_atr,
                composite=composite,
                rank=rank_pos,
            )
        )
        slots -= 1
        budget -= notional
        cash -= notional * (1 + fee_rate)

    return orders, skips
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_entries.py tests/test_config.py -q`
Expected: 16 entry tests + config tests PASS.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/simulator/entries.py tests/test_entries.py config/crypto.toml
git commit -m "Add regime-gated entry rules with hard rails and skip reasons [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: The pure step() and circuit breaker

Composes fills → exits → mark/breaker → entries into the single function M3 will replay. Deep-copies the input state (pure w.r.t. the caller), computes the run key and portfolio snapshot, and trips the drawdown circuit breaker.

**Files:**
- Create: `src/trading/simulator/core.py`
- Create: `tests/test_core.py`

**Interfaces:**
- Consumes: Tasks 3-6 (`apply_fills`, `release_settlements`, `evaluate_exits`, `evaluate_entries`, state types), `RankingsResult`.
- Produces (consumed by Tasks 9-12):
  - `decision_bar(rankings: RankingsResult) -> pd.Timestamp` — max last-bar timestamp across `rankings.bars` (all bars are already completeness-filtered upstream).
  - `make_run_key(venue: str, decision_ts: pd.Timestamp) -> str` — `f"{venue}:{decision_ts.isoformat()}"`.
  - `PositionMark(symbol, qty, entry_price, last_close, market_value, unrealized_pnl_pct, stop_price, stop_distance_pct, entry_rank, entry_composite, entry_ts)` — frozen.
  - `PortfolioSnapshot(value, cash, unsettled, positions: tuple[PositionMark, ...])` — frozen.
  - `StepResult(state, run_key, decision_ts, fills, new_orders, skips, warnings, breaker_tripped_now, snapshot)` — frozen; `state` is the UPDATED deep copy.
  - `step(state: PortfolioState, rankings: RankingsResult, config: VenueConfig, *, allow_entries: bool = True, stale_reason: str | None = None) -> StepResult`. Task 9 appends an `earnings` keyword parameter.
- Phase order inside `step` (locked): release due settlements → fill pending orders → evaluate exits → mark portfolio, update high-water mark, trip breaker → evaluate entries (skipped when `allow_entries=False` or breaker tripped) → append new orders to state.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_core.py`:

```python
import copy

import pandas as pd
import pytest
from sim_helpers import EQ, frame, make_rankings, make_state, make_table

from trading.simulator.core import decision_bar, make_run_key, step

JUL1 = pd.Timestamp("2026-07-01", tz="UTC")
JUL2 = pd.Timestamp("2026-07-02", tz="UTC")


def _row(composite=0.9):
    return {"status": "tradable", "composite": composite, "raw_return_30d": 0.10}


def _rankings(end="2026-07-01", **kwargs):
    bars = {"AAA": frame(end=end), "BBB": frame(end=end, start_price=50.0)}
    rows = {"AAA": _row(0.95), "BBB": _row(0.60)}  # BBB below entry threshold
    return make_rankings(EQ, bars, make_table(rows), **kwargs)


def test_decision_bar_is_max_last_bar_and_run_key_format():
    rankings = _rankings()
    assert decision_bar(rankings) == JUL1
    assert make_run_key("equities", JUL1) == "equities:2026-07-01T00:00:00+00:00"


def test_step_is_pure_and_deterministic():
    state = make_state(EQ)
    rankings = _rankings()
    before = copy.deepcopy(state)
    a = step(state, rankings, EQ)
    b = step(state, rankings, EQ)
    assert state == before  # input state untouched
    assert a.state == b.state
    assert a.fills == b.fills
    assert a.new_orders == b.new_orders
    assert a.skips == b.skips


def test_two_run_lifecycle_decides_then_fills():
    # Run 1 (decision bar Jul 1): no fills, one pending entry for AAA.
    state = make_state(EQ)
    first = step(state, _rankings(), EQ)
    assert first.run_key == "equities:2026-07-01T00:00:00+00:00"
    assert first.fills == ()
    assert [(o.symbol, o.side) for o in first.new_orders] == [("AAA", "buy")]
    assert first.state.positions == {}
    assert first.state.last_run_key == first.run_key

    # Run 2 (decision bar Jul 2): the pending order fills at Jul 2's open.
    second = step(first.state, _rankings(end="2026-07-02"), EQ)
    assert second.run_key == "equities:2026-07-02T00:00:00+00:00"
    assert [f.symbol for f in second.fills] == ["AAA"]
    assert second.fills[0].bar_ts == "2026-07-02T00:00:00+00:00"
    assert "AAA" in second.state.positions
    # AAA is now held: no re-entry (never average down).
    assert all(o.symbol != "AAA" or o.side != "buy" for o in second.state.pending_orders)


def test_snapshot_marks_positions_at_decision_close():
    state = make_state(EQ)
    first = step(state, _rankings(), EQ)
    second = step(first.state, _rankings(end="2026-07-02"), EQ)
    snap = second.snapshot
    position = second.state.positions["AAA"]
    assert snap.cash == pytest.approx(second.state.cash)
    mark = snap.positions[0]
    assert mark.symbol == "AAA"
    assert mark.last_close == pytest.approx(100.0)
    assert snap.value == pytest.approx(snap.cash + snap.unsettled + position.qty * 100.0)
    assert mark.stop_distance_pct == pytest.approx((100.0 - position.stop_price) / 100.0)


def test_breaker_trips_on_drawdown_and_blocks_entries():
    state = make_state(EQ, cash=700.0, high_water_mark=1000.0)  # 30% drawdown > 20%
    result = step(state, _rankings(), EQ)
    assert result.breaker_tripped_now is True
    assert result.state.breaker_tripped is True
    assert result.state.breaker_tripped_at == JUL1.isoformat()
    assert result.new_orders == ()
    assert any(s.reason == "circuit_breaker" for s in result.skips)

    # Already-tripped breaker does not re-fire the notification flag.
    again = step(result.state, _rankings(end="2026-07-02"), EQ)
    assert again.breaker_tripped_now is False
    assert again.state.breaker_tripped is True


def test_high_water_mark_ratchets_up():
    state = make_state(EQ, cash=1200.0, high_water_mark=1000.0)
    result = step(state, _rankings(), EQ)
    assert result.state.high_water_mark == pytest.approx(1200.0)


def test_stale_run_skips_entries_but_still_fills_and_exits():
    state = make_state(EQ)
    first = step(state, _rankings(), EQ)
    second = step(
        first.state,
        _rankings(end="2026-07-02"),
        EQ,
        allow_entries=False,
        stale_reason="stale_run_entries_skipped",
    )
    assert [f.symbol for f in second.fills] == ["AAA"]  # fills still processed
    assert any(s.reason == "stale_run_entries_skipped" for s in second.skips)
    assert all(o.side != "buy" for o in second.state.pending_orders)


def test_step_decisions_track_the_decision_bar():
    """No-lookahead companion: step's inputs end AT the decision bar by
    construction (decision_ts is the newest bar), so the property test here is
    sensitivity — decisions must be a function of data through the decision
    bar (change it, decisions change), and determinism is pinned by
    test_step_is_pure_and_deterministic. M3's replay equality test covers the
    perturb-after-T form end to end."""
    state = make_state(EQ)
    base = step(state, _rankings(), EQ)
    moved = _rankings()
    moved.bars["AAA"].loc[moved.bars["AAA"].index[-1], "close"] = 1.0  # crash the last bar
    after = step(state, moved, EQ)
    assert base.new_orders != after.new_orders or base.skips != after.skips
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.simulator.core'`.

- [ ] **Step 3: Implement step()**

Create `src/trading/simulator/core.py`:

```python
"""The pure per-venue simulator step (spec: Portfolio Simulator).

step() is the single function the M3 backtester replays: no I/O, no clock.
Everything it needs arrives as bars + state + config; staleness (the only
clock-dependent rule) is decided by the runner and passed in as allow_entries.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pandas as pd

from trading.config import VenueConfig
from trading.pipeline import RankingsResult
from trading.simulator.entries import evaluate_entries
from trading.simulator.exits import evaluate_exits
from trading.simulator.fills import Fill, apply_fills, release_settlements
from trading.simulator.state import PendingOrder, PortfolioState, Skip


@dataclass(frozen=True)
class PositionMark:
    symbol: str
    qty: float
    entry_price: float
    last_close: float
    market_value: float
    unrealized_pnl_pct: float
    stop_price: float
    stop_distance_pct: float
    entry_rank: int
    entry_composite: float
    entry_ts: str


@dataclass(frozen=True)
class PortfolioSnapshot:
    value: float
    cash: float
    unsettled: float
    positions: tuple[PositionMark, ...]


@dataclass(frozen=True)
class StepResult:
    state: PortfolioState  # updated deep copy; caller's state is untouched
    run_key: str
    decision_ts: pd.Timestamp
    fills: tuple[Fill, ...]
    new_orders: tuple[PendingOrder, ...]
    skips: tuple[Skip, ...]
    warnings: tuple[str, ...]
    breaker_tripped_now: bool
    snapshot: PortfolioSnapshot


def decision_bar(rankings: RankingsResult) -> pd.Timestamp:
    """The decision bar = newest completed bar across the clean universe."""
    return max(df.index[-1] for df in rankings.bars.values() if not df.empty)


def make_run_key(venue: str, decision_ts: pd.Timestamp) -> str:
    return f"{venue}:{decision_ts.isoformat()}"


def _mark_portfolio(
    state: PortfolioState, bars: dict[str, pd.DataFrame], decision_ts: pd.Timestamp
) -> tuple[PortfolioSnapshot, list[str]]:
    warnings: list[str] = []
    marks: list[PositionMark] = []
    for symbol, position in sorted(state.positions.items()):
        df = bars.get(symbol)
        window = df[df.index <= decision_ts] if df is not None else None
        if window is None or window.empty:
            last_close = position.entry_price
            warnings.append(f"{symbol}: no bars to mark position; using entry price")
        else:
            last_close = float(window["close"].iloc[-1])
        marks.append(
            PositionMark(
                symbol=symbol,
                qty=position.qty,
                entry_price=position.entry_price,
                last_close=last_close,
                market_value=position.qty * last_close,
                unrealized_pnl_pct=last_close / position.entry_price - 1.0,
                stop_price=position.stop_price,
                stop_distance_pct=(last_close - position.stop_price) / last_close,
                entry_rank=position.entry_rank,
                entry_composite=position.entry_composite,
                entry_ts=position.entry_ts,
            )
        )
    unsettled = sum(s.amount for s in state.settlements)
    value = state.cash + unsettled + sum(m.market_value for m in marks)
    return PortfolioSnapshot(
        value=value, cash=state.cash, unsettled=unsettled, positions=tuple(marks)
    ), warnings


def step(
    state: PortfolioState,
    rankings: RankingsResult,
    config: VenueConfig,
    *,
    allow_entries: bool = True,
    stale_reason: str | None = None,
) -> StepResult:
    state = copy.deepcopy(state)
    decision_ts = decision_bar(rankings)
    run_key = make_run_key(config.name, decision_ts)
    warnings: list[str] = []
    skips: list[Skip] = []

    # Phase 1: settle yesterday's sale proceeds, then fill pending orders.
    release_settlements(state, decision_ts.date())
    fills, fill_skips = apply_fills(state, rankings.bars, config)
    skips.extend(fill_skips)

    # Phase 2: exits (before entries, against the unfiltered ranking).
    exit_orders, exit_skips, exit_warnings = evaluate_exits(state, rankings, config, decision_ts)
    skips.extend(exit_skips)
    warnings.extend(exit_warnings)

    # Phase 3: mark to the decision bar; ratchet the high-water mark; breaker.
    snapshot, mark_warnings = _mark_portfolio(state, rankings.bars, decision_ts)
    warnings.extend(mark_warnings)
    breaker_tripped_now = False
    if snapshot.value > state.high_water_mark:
        state.high_water_mark = snapshot.value
    drawdown = 1.0 - snapshot.value / state.high_water_mark
    if not state.breaker_tripped and drawdown > config.portfolio.drawdown_halt_pct:
        state.breaker_tripped = True
        state.breaker_tripped_at = decision_ts.isoformat()
        breaker_tripped_now = True
        warnings.append(
            f"circuit breaker tripped: drawdown {drawdown:.1%} exceeds "
            f"{config.portfolio.drawdown_halt_pct:.0%}; entries halted until reset-breaker"
        )

    # Phase 4: entries (regime- and breaker-gated; staleness decided upstream).
    entry_orders: list[PendingOrder] = []
    if allow_entries:
        entry_orders, entry_skips = evaluate_entries(
            state, rankings, config, decision_ts, snapshot.value
        )
        skips.extend(entry_skips)
    else:
        skips.append(Skip("*", "entry", stale_reason or "entries_disabled"))

    new_orders = [*exit_orders, *entry_orders]
    state.pending_orders = [*state.pending_orders, *new_orders]
    state.last_run_key = run_key

    return StepResult(
        state=state,
        run_key=run_key,
        decision_ts=decision_ts,
        fills=tuple(fills),
        new_orders=tuple(new_orders),
        skips=tuple(skips),
        warnings=tuple(warnings),
        breaker_tripped_now=breaker_tripped_now,
        snapshot=snapshot,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_core.py tests/test_fills.py tests/test_exits.py tests/test_entries.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/simulator/core.py tests/test_core.py
git commit -m "Compose pure simulator step with drawdown circuit breaker [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Append-only journal

Per-venue JSONL with run-key idempotency lookups and a config hash for every event.

**Files:**
- Create: `src/trading/journal.py`
- Create: `tests/test_journal.py`

**Interfaces:**
- Consumes: `VenueConfig`.
- Produces (consumed by Tasks 10-12):
  - `Journal(path: Path)` with `append(event: dict) -> None` (fsynced JSON line), `events() -> Iterator[dict]`, `has_run(run_key: str) -> bool`, `last_event(types: frozenset[str] | None = None) -> dict | None`.
  - `JournalError(RuntimeError)` — raised for corruption anywhere except a torn final line (a crash mid-append), which is skipped.
  - `config_hash(config: VenueConfig) -> str` — first 12 hex chars of sha256 over the sorted-JSON dataclass dump.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_journal.py`:

```python
import dataclasses
from pathlib import Path

import pytest

from trading.config import load_venue_config
from trading.journal import Journal, JournalError, config_hash


def _journal(tmp_path) -> Journal:
    return Journal(tmp_path / "journal" / "equities.jsonl")


def test_append_and_read_round_trip(tmp_path):
    journal = _journal(tmp_path)
    journal.append({"event": "bootstrap", "venue": "equities"})
    journal.append({"event": "run", "run_key": "equities:2026-07-01T00:00:00+00:00"})
    events = list(journal.events())
    assert [e["event"] for e in events] == ["bootstrap", "run"]


def test_missing_file_yields_no_events(tmp_path):
    assert list(_journal(tmp_path).events()) == []
    assert _journal(tmp_path).has_run("equities:2026-07-01T00:00:00+00:00") is False


def test_has_run_finds_run_key(tmp_path):
    journal = _journal(tmp_path)
    journal.append({"event": "run", "run_key": "equities:2026-07-01T00:00:00+00:00"})
    assert journal.has_run("equities:2026-07-01T00:00:00+00:00") is True
    assert journal.has_run("equities:2026-07-02T00:00:00+00:00") is False


def test_last_event_with_type_filter(tmp_path):
    journal = _journal(tmp_path)
    journal.append({"event": "bootstrap"})
    journal.append({"event": "run", "n": 1})
    journal.append({"event": "run_failed"})
    journal.append({"event": "run", "n": 2})
    assert journal.last_event()["event"] == "run"
    assert journal.last_event(types=frozenset({"run"}))["n"] == 2
    assert journal.last_event(types=frozenset({"bootstrap"}))["event"] == "bootstrap"
    assert journal.last_event(types=frozenset({"nope"})) is None


def test_torn_final_line_is_skipped(tmp_path):
    journal = _journal(tmp_path)
    journal.append({"event": "run", "n": 1})
    with (tmp_path / "journal" / "equities.jsonl").open("a") as f:
        f.write('{"event": "run", "n"')  # crash mid-append
    assert [e["n"] for e in journal.events()] == [1]


def test_corruption_mid_file_raises(tmp_path):
    path = tmp_path / "journal" / "equities.jsonl"
    journal = Journal(path)
    journal.append({"event": "run", "n": 1})
    with path.open("a") as f:
        f.write("not json\n")
    journal.append({"event": "run", "n": 2})
    with pytest.raises(JournalError, match="line 2"):
        list(journal.events())


def test_config_hash_is_stable_and_sensitive(tmp_path):
    config = load_venue_config("equities", Path("config"))
    assert config_hash(config) == config_hash(config)
    assert len(config_hash(config)) == 12
    changed = dataclasses.replace(
        config, portfolio=dataclasses.replace(config.portfolio, max_positions=4)
    )
    assert config_hash(changed) != config_hash(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_journal.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.journal'`.

- [ ] **Step 3: Implement the journal**

Create `src/trading/journal.py`:

```python
"""Per-venue append-only JSONL journal (spec: Reporting & Operations).

Every run appends one event; state is reconstructible by replaying the
events' state_after snapshots. The journal is also the idempotency ledger:
has_run(run_key) is consulted before acting, so a decision bar that has been
traded is never traded again.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from trading.config import VenueConfig


class JournalError(RuntimeError):
    """The journal is corrupt somewhere other than a torn final line."""


class Journal:
    def __init__(self, path: Path):
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict) -> None:
        line = json.dumps(event, sort_keys=True, default=str)
        with self._path.open("a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def events(self) -> Iterator[dict]:
        if not self._path.exists():
            return
        lines = self._path.read_text().splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if number == len(lines):
                    return  # torn final line: crash mid-append, ignore
                raise JournalError(f"{self._path}: corrupt journal line {number}") from exc

    def has_run(self, run_key: str) -> bool:
        return any(event.get("run_key") == run_key for event in self.events())

    def last_event(self, types: frozenset[str] | None = None) -> dict | None:
        last: dict | None = None
        for event in self.events():
            if types is None or event.get("event") in types:
                last = event
        return last


def config_hash(config: VenueConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_journal.py -q`
Expected: 7 PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/journal.py tests/test_journal.py
git commit -m "Add append-only JSONL journal with run-key idempotency and config hash [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Earnings blackout filter (with kill switch)

Equities entries are blocked when earnings fall within `earnings_blackout_sessions` of the decision date. Dates come from yfinance at the runner boundary; a fetch failure NEVER crashes a run — it degrades to no-filter with a journal warning. The whole filter sits behind `earnings_blackout_enabled` so it can be turned off in TOML (and documented as dropped, per spec) if yfinance proves unreliable in Task 14's live verification.

**Files:**
- Create: `src/trading/earnings.py`
- Create: `tests/test_earnings.py`
- Modify: `src/trading/config.py` (PortfolioConfig)
- Modify: `config/equities.toml`, `config/crypto.toml`
- Modify: `src/trading/simulator/entries.py` (add `earnings` parameter + check)
- Modify: `src/trading/simulator/core.py` (pass-through)
- Modify: `tests/test_config.py`, `tests/test_entries.py` (new cases)

**Interfaces:**
- Consumes: `evaluate_entries` (Task 6), `step` (Task 7).
- Produces (consumed by Task 10):
  - `fetch_earnings_dates(symbols: Iterable[str]) -> tuple[dict[str, tuple[str, ...]], bool]` — `{symbol: ISO dates}` plus a `degraded` flag (True if any symbol's fetch failed; failed symbols are simply absent = unfiltered).
  - `evaluate_entries(..., earnings: Mapping[str, tuple[str, ...]] | None = None)` — `None` disables the filter entirely.
  - `step(..., earnings: Mapping[str, tuple[str, ...]] | None = None)` — passes through to `evaluate_entries`.
  - Config: `config.portfolio.earnings_blackout_enabled` (equities `true`, crypto `false`).
- Blackout window: any earnings date `d` with `decision_date <= d <= pd.bdate_range(decision_date, periods=sessions + 1)[-1]` (sessions approximated as business days). `sessions <= 0` disables.

- [ ] **Step 1: Add the config kill switch**

In `src/trading/config.py`, add to `PortfolioConfig` after `earnings_blackout_sessions`:

```python
    earnings_blackout_enabled: bool
```

In `config/equities.toml` `[portfolio]`, after the `earnings_blackout_sessions` line:

```toml
earnings_blackout_enabled = true    # kill switch: set false (and document in README) if yfinance dates prove unreliable
```

In `config/crypto.toml` `[portfolio]`, after the `earnings_blackout_sessions` line:

```toml
earnings_blackout_enabled = false   # no earnings in crypto
```

Append to `tests/test_config.py::test_load_equities_config`:

```python
    assert config.portfolio.earnings_blackout_enabled is True
```

and to `tests/test_config.py::test_load_crypto_config`:

```python
    assert config.portfolio.earnings_blackout_enabled is False
```

Run: `uv run pytest tests/test_config.py -q` — expected: PASS.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_earnings.py`:

```python
import pytest

from trading.earnings import fetch_earnings_dates


def test_fetch_collects_iso_dates(monkeypatch):
    import datetime

    monkeypatch.setattr(
        "trading.earnings._yf_earnings_dates",
        lambda symbol: [datetime.date(2026, 7, 3), datetime.date(2026, 10, 2)],
    )
    dates, degraded = fetch_earnings_dates(["AAPL", "MSFT"])
    assert degraded is False
    assert dates == {
        "AAPL": ("2026-07-03", "2026-10-02"),
        "MSFT": ("2026-07-03", "2026-10-02"),
    }


def test_per_symbol_failure_degrades_without_raising(monkeypatch):
    import datetime

    def flaky(symbol: str):
        if symbol == "MSFT":
            raise RuntimeError("yfinance flaked")
        return [datetime.date(2026, 7, 3)]

    monkeypatch.setattr("trading.earnings._yf_earnings_dates", flaky)
    dates, degraded = fetch_earnings_dates(["AAPL", "MSFT"])
    assert degraded is True
    assert dates == {"AAPL": ("2026-07-03",)}  # MSFT absent = unfiltered
```

Append to `tests/test_entries.py`:

```python
def test_earnings_within_blackout_blocks_entry():
    earnings = {"AAA": ("2026-07-06",)}  # 3 business days after 2026-07-01
    state = make_state(EQ)
    rankings = make_rankings(EQ, {"AAA": frame()}, make_table({"AAA": _row()}))
    orders, skips = evaluate_entries(state, rankings, EQ, DECISION, VALUE, earnings=earnings)
    assert orders == []
    assert Skip("AAA", "entry", "earnings_blackout") in skips


def test_earnings_beyond_blackout_and_none_do_not_block():
    state = make_state(EQ)
    rankings = make_rankings(EQ, {"AAA": frame()}, make_table({"AAA": _row()}))
    # 2026-07-01 + 5 business days = 2026-07-08; the 9th is outside the window.
    far = {"AAA": ("2026-07-09",)}
    orders, _ = evaluate_entries(state, rankings, EQ, DECISION, VALUE, earnings=far)
    assert len(orders) == 1
    # earnings=None (filter disabled/degraded symbol) never blocks.
    orders, _ = evaluate_entries(state, rankings, EQ, DECISION, VALUE, earnings=None)
    assert len(orders) == 1


def test_earnings_day_of_decision_blocks():
    earnings = {"AAA": ("2026-07-01",)}
    state = make_state(EQ)
    rankings = make_rankings(EQ, {"AAA": frame()}, make_table({"AAA": _row()}))
    orders, skips = evaluate_entries(state, rankings, EQ, DECISION, VALUE, earnings=earnings)
    assert orders == []
    assert Skip("AAA", "entry", "earnings_blackout") in skips
```

Append to `tests/test_core.py`:

```python
def test_step_passes_earnings_through_to_entries():
    state = make_state(EQ)
    blocked = step(state, _rankings(), EQ, earnings={"AAA": ("2026-07-02",)})
    assert all(o.symbol != "AAA" for o in blocked.new_orders)
    assert any(s.reason == "earnings_blackout" for s in blocked.skips)
```

(`make_state` is already imported at the top of `tests/test_core.py`.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_earnings.py tests/test_entries.py tests/test_core.py -q`
Expected: `test_earnings.py` FAILS with `ModuleNotFoundError: No module named 'trading.earnings'`; the new entries/core tests FAIL with `TypeError: evaluate_entries() got an unexpected keyword argument 'earnings'` (and the same for `step`).

- [ ] **Step 4: Implement the fetcher and wire the filter**

Create `src/trading/earnings.py`:

```python
"""Earnings-date sourcing for the equities entry blackout (spec: Venue Model).

Kill switch: config [portfolio] earnings_blackout_enabled. A fetch failure
must never crash or block a run — failed symbols are omitted (= unfiltered)
and the degraded flag is journaled upstream as a warning. If the source
proves unreliable in practice, set the switch to false in TOML and document
the drop in the README (spec: dropped entirely, both modes).
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable


def _yf_earnings_dates(symbol: str) -> list[datetime.date]:
    """Network touchpoint, isolated for monkeypatching."""
    import yfinance as yf

    df = yf.Ticker(symbol).get_earnings_dates(limit=8)
    if df is None or df.empty:
        return []
    return sorted({ts.date() for ts in df.index})


def fetch_earnings_dates(symbols: Iterable[str]) -> tuple[dict[str, tuple[str, ...]], bool]:
    dates: dict[str, tuple[str, ...]] = {}
    degraded = False
    for symbol in symbols:
        try:
            dates[symbol] = tuple(d.isoformat() for d in _yf_earnings_dates(symbol))
        except Exception:
            degraded = True  # degrade to no-filter for this symbol, never crash
    return dates, degraded
```

In `src/trading/simulator/entries.py`:

Add `from collections.abc import Mapping` to the imports, add the helper above `evaluate_entries`:

```python
def _within_earnings_blackout(
    dates: tuple[str, ...], decision_date: datetime.date, sessions: int
) -> bool:
    if sessions <= 0 or not dates:
        return False
    horizon = pd.bdate_range(decision_date, periods=sessions + 1)[-1].date()
    return any(decision_date <= datetime.date.fromisoformat(d) <= horizon for d in dates)
```

change the signature:

```python
def evaluate_entries(
    state: PortfolioState,
    rankings: RankingsResult,
    config: VenueConfig,
    decision_ts: pd.Timestamp,
    portfolio_value: float,
    earnings: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[list[PendingOrder], list[Skip]]:
```

and insert this gate immediately after the cooldown check (before the `df = rankings.bars.get(symbol)` line):

```python
        if earnings is not None and _within_earnings_blackout(
            earnings.get(symbol, ()), decision_date, p.earnings_blackout_sessions
        ):
            skips.append(Skip(symbol, "entry", "earnings_blackout"))
            continue
```

In `src/trading/simulator/core.py`, add `from collections.abc import Mapping` to the imports, change `step`'s signature:

```python
def step(
    state: PortfolioState,
    rankings: RankingsResult,
    config: VenueConfig,
    *,
    allow_entries: bool = True,
    stale_reason: str | None = None,
    earnings: Mapping[str, tuple[str, ...]] | None = None,
) -> StepResult:
```

and change the `evaluate_entries` call:

```python
        entry_orders, entry_skips = evaluate_entries(
            state, rankings, config, decision_ts, snapshot.value, earnings=earnings
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_earnings.py tests/test_entries.py tests/test_core.py tests/test_config.py -q`
Expected: all PASS.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/earnings.py tests/test_earnings.py src/trading/simulator/entries.py src/trading/simulator/core.py src/trading/config.py config/equities.toml config/crypto.toml tests/test_config.py tests/test_entries.py tests/test_core.py
git commit -m "Add earnings blackout filter with kill switch and fail-open degradation [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Runner, notifications, and `trading run`

All I/O around the pure step: lockfile, bootstrap, idempotency, staleness, atomic state writes, the journal run event, macOS notifications, and the `trading run` CLI command (including `--restore-from-journal` recovery).

**Files:**
- Create: `src/trading/notify.py`
- Create: `src/trading/runner.py`
- Create: `tests/test_notify.py`
- Create: `tests/test_runner.py`
- Modify: `src/trading/cli.py`
- Modify: `tests/test_cli.py` (run-command end-to-end tests)

**Interfaces:**
- Consumes: `build_rankings`/`PipelineDataError` (M1+Task 2), `step`/`decision_bar`/`make_run_key` (Task 7), `Journal`/`config_hash` (Task 8), `fetch_earnings_dates` (Task 9), state types (Task 3).
- Produces (consumed by Tasks 11-12):
  - `notify(title: str, message: str) -> None` in `trading.notify` — best-effort osascript, never raises.
  - In `trading.runner`:
    - `state_path(state_root: Path, venue: str) -> Path` (`state/<venue>/portfolio.json`), `lock_path(state_root, venue)` (`state/<venue>/.lock`).
    - `load_state(path: Path) -> PortfolioState | None` (None = not bootstrapped; raises `StateError` on corruption), `save_state(path: Path, state: PortfolioState) -> None` (temp + `os.replace`).
    - `RunLock(path)` with `acquire() -> bool` (stale-pid aware) and `release()`.
    - `RunOutcome(venue, status, message, run_key=None, result=None)` — `status in {"ok", "noop", "skipped", "failed"}`.
    - `run_venue(config, adapter, cache, *, now: datetime.datetime, state_root: Path, journal_root: Path, notify: Callable[[str, str], None]) -> RunOutcome` (Task 11 appends a `digest_root` parameter).
    - `restore_from_journal(venue: str, state_root: Path, journal_root: Path) -> str`; `RunnerError(RuntimeError)`.
  - CLI: `trading run --venue V [--json] [--restore-from-journal] [--config-dir D] [--state-dir D] [--journal-dir D]`; exit 0 for ok/noop, 1 for skipped/failed. `trading.cli._utcnow()` is the single clock read (monkeypatchable).
- Staleness (locked): `deadline = decision_ts + 1 day + staleness_hours`; entries allowed iff `now <= deadline`. Exits/fills always processed. Equities runs on non-trading days hit an already-journaled run_key and no-op cleanly.
- Journal `run` event schema (consumed by digest/status): `event, venue, run_key, ts, decision_ts, config_hash, regime{state, exposure_multiplier}, coverage{requested, fetched, ratio}, benchmark{symbol, close, start_price}, starting_balance, ranking[{rank, symbol, status, <sub-score columns>}], fills[], new_orders[], skips[], warnings[], snapshot{value, cash, unsettled, positions[]}, state_after{...}`.

- [ ] **Step 1: Write the failing notify test**

Create `tests/test_notify.py`:

```python
from trading import notify as notify_module
from trading.notify import notify


def test_notify_invokes_osascript_with_escaped_text(monkeypatch):
    calls = []
    monkeypatch.setattr(
        notify_module.subprocess, "run", lambda *a, **k: calls.append((a, k))
    )
    notify('run "failed"', "coverage 42%")
    (args,), kwargs = calls[0][0], calls[0][1]
    assert args[0] == "osascript"
    assert 'run \\"failed\\"' in args[2]
    assert "coverage 42%" in args[2]


def test_notify_never_raises(monkeypatch):
    def boom(*a, **k):
        raise OSError("no osascript here")

    monkeypatch.setattr(notify_module.subprocess, "run", boom)
    notify("title", "message")  # must not raise
```

Run: `uv run pytest tests/test_notify.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.notify'`.

- [ ] **Step 2: Implement notify**

Create `src/trading/notify.py`:

```python
"""macOS failure signaling (spec: a silent dead pipeline is the most likely
real-world failure, so every failed/skipped run and breaker trip notifies)."""

from __future__ import annotations

import subprocess


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def notify(title: str, message: str) -> None:
    """Best-effort `display notification`; never raises."""
    script = f'display notification "{_escape(message)}" with title "{_escape(title)}"'
    try:
        subprocess.run(
            ["osascript", "-e", script], capture_output=True, timeout=10, check=False
        )
    except Exception:
        pass  # notification failure must never break a run
```

Run: `uv run pytest tests/test_notify.py -q`
Expected: 2 PASS.

- [ ] **Step 3: Write the failing runner tests**

Create `tests/test_runner.py`:

```python
import datetime
import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from sim_helpers import EQ

from trading.data.cache import OhlcvCache
from trading.journal import Journal
from trading.runner import (
    RunLock,
    lock_path,
    load_state,
    restore_from_journal,
    run_venue,
    save_state,
    state_path,
)
from trading.simulator.state import StateError
from trading.venues.base import DataFetchError, SymbolInfo, VenueConstraints

NOW = datetime.datetime(2026, 7, 1, 22, 30, tzinfo=datetime.UTC)  # Wed evening UTC
SYMBOLS = ["UPUP", "FLAT", "MEH1", "MEH2"]


@pytest.fixture(autouse=True)
def _no_earnings_network(monkeypatch):
    # config/equities.toml enables the earnings filter; keep tests offline.
    monkeypatch.setattr("trading.runner.fetch_earnings_dates", lambda symbols: ({}, False))


def _bars(drift: float, end: datetime.date) -> pd.DataFrame:
    idx = pd.date_range(end=end, periods=320, freq="B", tz="UTC")
    jitter = np.where(np.arange(320) % 2 == 0, 0.002, -0.002)
    close = 100 * np.cumprod(1 + drift + jitter)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(320, 1e6),
        },
        index=idx,
    )


class FakeAdapter:
    """Serves deterministic frames through any end date; UPUP leads every feature."""

    def __init__(self, fail: frozenset[str] = frozenset()):
        self.fail = fail
        self.drifts = {"UPUP": 0.01, "FLAT": 0.0, "MEH1": -0.002, "MEH2": -0.004, "SPY": 0.001}

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        return [SymbolInfo(s, "tradable") for s in SYMBOLS]

    def constraints(self) -> VenueConstraints:
        return VenueConstraints(0.0, 0.0, 5.0, 1, False)

    def fetch_ohlcv(self, symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        if symbol in self.fail:
            raise DataFetchError(symbol)
        df = _bars(self.drifts[symbol], end)
        return df.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]


def _run(tmp_path, now=NOW, fail=frozenset(), notes=None):
    notes = notes if notes is not None else []
    adapter = FakeAdapter(fail=fail)
    cache = OhlcvCache(tmp_path / "cache", EQ.data.refetch_days)
    outcome = run_venue(
        EQ,
        adapter,
        cache,
        now=now,
        state_root=tmp_path / "state",
        journal_root=tmp_path / "journal",
        notify=lambda title, message: notes.append((title, message)),
    )
    return outcome, notes


def test_first_run_bootstraps_journals_and_orders_top_candidate(tmp_path):
    outcome, notes = _run(tmp_path)
    assert outcome.status == "ok"
    assert outcome.run_key == "equities:2026-07-01T00:00:00+00:00"
    assert notes == []

    state = load_state(state_path(tmp_path / "state", "equities"))
    assert state.cash == 1000.0  # nothing filled yet
    assert [(o.symbol, o.side) for o in state.pending_orders] == [("UPUP", "buy")]
    assert state.benchmark_start_price > 0

    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    assert [e["event"] for e in events] == ["bootstrap", "run"]
    run_event = events[1]
    assert run_event["run_key"] == outcome.run_key
    assert run_event["starting_balance"] == 1000.0
    assert len(run_event["ranking"]) == 4
    assert run_event["state_after"]["positions"] == {}
    assert not (tmp_path / "state" / "equities" / ".lock").exists()  # released


def test_same_decision_bar_is_a_noop(tmp_path):
    _run(tmp_path)
    outcome, _ = _run(tmp_path, now=NOW + datetime.timedelta(hours=1))
    assert outcome.status == "noop"
    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    assert len(events) == 2  # nothing new journaled; never trades twice


def test_next_day_run_fills_pending_order(tmp_path):
    _run(tmp_path)
    outcome, _ = _run(tmp_path, now=NOW + datetime.timedelta(days=1))
    assert outcome.status == "ok"
    state = load_state(state_path(tmp_path / "state", "equities"))
    assert "UPUP" in state.positions
    run_event = Journal(tmp_path / "journal" / "equities.jsonl").last_event(
        types=frozenset({"run"})
    )
    assert [f["symbol"] for f in run_event["fills"]] == ["UPUP"]


def test_stale_run_still_fills_and_exits_but_skips_entries(tmp_path):
    _run(tmp_path)
    # Friday bars (decision 2026-07-03) processed Sunday: way past 1d + 2h.
    late = datetime.datetime(2026, 7, 5, 18, 0, tzinfo=datetime.UTC)
    outcome, _ = _run(tmp_path, now=late)
    assert outcome.status == "ok"
    run_event = Journal(tmp_path / "journal" / "equities.jsonl").last_event(
        types=frozenset({"run"})
    )
    assert [f["symbol"] for f in run_event["fills"]] == ["UPUP"]  # fill still happened
    assert any(s["reason"] == "stale_run_entries_skipped" for s in run_event["skips"])
    state = load_state(state_path(tmp_path / "state", "equities"))
    assert all(o.side != "buy" for o in state.pending_orders)


def test_live_lock_skips_run_and_notifies(tmp_path):
    lock = lock_path(tmp_path / "state", "equities")
    lock.parent.mkdir(parents=True)
    import os

    lock.write_text(str(os.getpid()))  # a live process holds the lock
    outcome, notes = _run(tmp_path)
    assert outcome.status == "skipped"
    assert notes and "lock" in notes[0][1]
    assert lock.exists()  # never steal a live lock


def test_stale_lock_from_dead_process_is_broken(tmp_path):
    dead = subprocess.Popen(["true"])
    dead.wait()
    lock = lock_path(tmp_path / "state", "equities")
    lock.parent.mkdir(parents=True)
    lock.write_text(str(dead.pid))
    outcome, _ = _run(tmp_path)
    assert outcome.status == "ok"


def test_corrupt_state_refuses_to_run_and_notifies(tmp_path):
    _run(tmp_path)
    state_file = state_path(tmp_path / "state", "equities")
    state_file.write_text("{ not json")
    outcome, notes = _run(tmp_path, now=NOW + datetime.timedelta(days=1))
    assert outcome.status == "failed"
    assert "restore-from-journal" in outcome.message
    assert notes and "corrupt" in notes[0][1].lower()
    assert state_file.read_text() == "{ not json"  # never silently regenerated


def test_coverage_failure_journals_notifies_and_touches_nothing(tmp_path):
    outcome, notes = _run(tmp_path, fail=frozenset({"MEH1", "MEH2"}))  # 50% < 90%
    assert outcome.status == "failed"
    assert notes
    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    assert [e["event"] for e in events] == ["run_failed"]
    assert not state_path(tmp_path / "state", "equities").exists()


def test_restore_from_journal_rebuilds_state(tmp_path):
    _run(tmp_path)
    state_file = state_path(tmp_path / "state", "equities")
    good = state_file.read_text()
    state_file.write_text("garbage")
    message = restore_from_journal("equities", tmp_path / "state", tmp_path / "journal")
    assert "equities:2026-07-01" in message
    assert json.loads(state_file.read_text()) == json.loads(good)


def test_save_state_is_atomic_and_load_round_trips(tmp_path):
    from trading.simulator.state import initial_state

    path = state_path(tmp_path / "state", "equities")
    state = initial_state("equities", 1000.0, 620.0, "2026-07-01T00:00:00+00:00")
    save_state(path, state)
    assert load_state(path) == state
    assert not path.with_suffix(".json.tmp").exists()
    assert load_state(state_path(tmp_path / "state", "crypto")) is None
    path.write_text("{}")
    with pytest.raises(StateError):
        load_state(path)


def test_run_lock_is_reentrant_safe(tmp_path):
    lock = RunLock(tmp_path / ".lock")
    assert lock.acquire() is True
    assert RunLock(tmp_path / ".lock").acquire() is False  # our own pid is alive
    lock.release()
    assert RunLock(tmp_path / ".lock").acquire() is True
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.runner'`.

- [ ] **Step 5: Implement the runner**

Create `src/trading/runner.py`:

```python
"""One live-paper cycle around the pure simulator (spec: Execution Split).

Owns every side effect the simulator is not allowed to have: the lockfile,
state file (atomic write, corrupt -> refuse + notify), append-only journal,
staleness decision (the only clock-dependent rule), earnings fetch, and
failure notifications. run_venue never trades a decision bar twice: the
journal is consulted by run_key before acting.
"""

from __future__ import annotations

import datetime
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from trading.config import VenueConfig
from trading.data.cache import OhlcvCache
from trading.earnings import fetch_earnings_dates
from trading.journal import Journal, config_hash
from trading.pipeline import PipelineDataError, RankingsResult, build_rankings
from trading.simulator.core import StepResult, decision_bar, make_run_key, step
from trading.simulator.state import (
    PortfolioState,
    StateError,
    initial_state,
    state_from_dict,
    to_state_dict,
)
from trading.venues.base import VenueAdapter

Notifier = Callable[[str, str], None]

EARNINGS_CANDIDATE_DEPTH = 15  # top-of-ranking symbols worth an earnings lookup


class RunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunOutcome:
    venue: str
    status: str  # "ok" | "noop" | "skipped" | "failed"
    message: str
    run_key: str | None = None
    result: StepResult | None = None


def state_path(state_root: Path, venue: str) -> Path:
    return state_root / venue / "portfolio.json"


def lock_path(state_root: Path, venue: str) -> Path:
    return state_root / venue / ".lock"


def load_state(path: Path) -> PortfolioState | None:
    """None = not bootstrapped yet. Corruption raises StateError — the caller
    must refuse to run and notify; state is NEVER silently regenerated."""
    if not path.exists():
        return None
    try:
        return state_from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, StateError) as exc:
        raise StateError(f"{path}: {exc}") from exc


def save_state(path: Path, state: PortfolioState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(to_state_dict(state), indent=2, sort_keys=True))
    os.replace(tmp, path)  # atomic: never leave a torn state file


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    """state/<venue>/.lock — prevents a manual run racing the scheduled job."""

    def __init__(self, path: Path):
        self._path = path

    def acquire(self) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(self._path.read_text().strip())
                except (OSError, ValueError):
                    pid = None
                if pid is not None and _pid_alive(pid):
                    return False
                self._path.unlink(missing_ok=True)  # stale lock from a dead process
                continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        return False

    def release(self) -> None:
        self._path.unlink(missing_ok=True)


def _ranking_records(table: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for pos, (symbol, row) in enumerate(table.iterrows(), start=1):
        record: dict[str, object] = {"rank": pos, "symbol": symbol, "status": row["status"]}
        for col in table.columns:
            if col == "status":
                continue
            value = row[col]
            record[col] = None if pd.isna(value) else round(float(value), 4)
        records.append(record)
    return records


def _run_event(
    result: StepResult,
    rankings: RankingsResult,
    config: VenueConfig,
    now: datetime.datetime,
    extra_warnings: list[str],
) -> dict:
    benchmark_close = float(
        rankings.benchmark_bars.loc[: result.decision_ts, "close"].iloc[-1]
    )
    return {
        "event": "run",
        "venue": config.name,
        "run_key": result.run_key,
        "ts": now.isoformat(),
        "decision_ts": result.decision_ts.isoformat(),
        "config_hash": config_hash(config),
        "regime": {
            "state": rankings.regime.state,
            "exposure_multiplier": rankings.regime.exposure_multiplier,
        },
        "coverage": {
            "requested": rankings.coverage.requested,
            "fetched": rankings.coverage.fetched,
            "ratio": round(rankings.coverage.ratio, 4),
        },
        "benchmark": {
            "symbol": config.benchmark,
            "close": benchmark_close,
            "start_price": result.state.benchmark_start_price,
        },
        "starting_balance": config.portfolio.starting_balance,
        "ranking": _ranking_records(rankings.table),
        "fills": [asdict(f) for f in result.fills],
        "new_orders": [asdict(o) for o in result.new_orders],
        "skips": [asdict(s) for s in result.skips],
        "warnings": [
            *result.warnings,
            *extra_warnings,
            *(f"quarantined: {s}" for s in rankings.quarantined),
            *(f"fetch failed: {s}" for s in rankings.fetch_failures),
        ],
        "snapshot": asdict(result.snapshot),
        "state_after": to_state_dict(result.state),
    }


def run_venue(
    config: VenueConfig,
    adapter: VenueAdapter,
    cache: OhlcvCache,
    *,
    now: datetime.datetime,
    state_root: Path,
    journal_root: Path,
    notify: Notifier,
) -> RunOutcome:
    venue = config.name
    journal = Journal(journal_root / f"{venue}.jsonl")
    lock = RunLock(lock_path(state_root, venue))
    if not lock.acquire():
        notify("trading: run skipped", f"{venue}: another run holds the lock")
        return RunOutcome(venue, "skipped", "lockfile held by a live process")
    try:
        try:
            state = load_state(state_path(state_root, venue))
        except StateError as exc:
            notify("trading: state corrupt", f"{venue}: refusing to run")
            return RunOutcome(
                venue,
                "failed",
                f"corrupt state file ({exc}); recover with "
                f"'trading run --venue {venue} --restore-from-journal'",
            )

        try:
            rankings = build_rankings(config, adapter, cache, now.date())
        except PipelineDataError as exc:
            journal.append(
                {"event": "run_failed", "venue": venue, "ts": now.isoformat(), "reason": str(exc)}
            )
            notify("trading: run failed", f"{venue}: {exc}")
            return RunOutcome(venue, "failed", str(exc))

        decision_ts = decision_bar(rankings)
        run_key = make_run_key(venue, decision_ts)
        if journal.has_run(run_key):
            return RunOutcome(
                venue, "noop", f"decision bar {decision_ts.date()} already processed", run_key
            )

        if state is None:  # explicit bootstrap path, journaled (spec)
            benchmark_close = float(rankings.benchmark_bars["close"].iloc[-1])
            state = initial_state(
                venue, config.portfolio.starting_balance, benchmark_close, now.isoformat()
            )
            journal.append(
                {
                    "event": "bootstrap",
                    "venue": venue,
                    "ts": now.isoformat(),
                    "starting_balance": config.portfolio.starting_balance,
                    "benchmark_start_price": benchmark_close,
                    "config_hash": config_hash(config),
                }
            )

        # Staleness (spec: Execution Split #4): a late run still processes
        # exits and fills; entries are skipped beyond the config bound past
        # the decision bar's day boundary. No catch-up trading, ever.
        deadline = (
            decision_ts
            + pd.Timedelta(1, unit="D")
            + pd.Timedelta(config.portfolio.staleness_hours, unit="h")
        )
        allow_entries = pd.Timestamp(now) <= deadline

        earnings = None
        extra_warnings: list[str] = []
        if config.portfolio.earnings_blackout_enabled:
            candidates = list(rankings.table.index[:EARNINGS_CANDIDATE_DEPTH])
            earnings, degraded = fetch_earnings_dates(candidates)
            if degraded:
                extra_warnings.append(
                    "earnings fetch degraded: blackout filter partially disabled this run"
                )

        result = step(
            state,
            rankings,
            config,
            allow_entries=allow_entries,
            stale_reason=None if allow_entries else "stale_run_entries_skipped",
            earnings=earnings,
        )

        save_state(state_path(state_root, venue), result.state)
        journal.append(_run_event(result, rankings, config, now, extra_warnings))

        if result.breaker_tripped_now:
            notify(
                "trading: circuit breaker",
                f"{venue}: drawdown halt — entries stopped until reset-breaker",
            )

        message = f"{len(result.fills)} fill(s), {len(result.new_orders)} new order(s)"
        if not allow_entries:
            message += "; stale run: entries skipped"
        return RunOutcome(venue, "ok", message, run_key, result)
    finally:
        lock.release()


def restore_from_journal(venue: str, state_root: Path, journal_root: Path) -> str:
    """Recovery for a corrupt state file: replay the last journaled snapshot.
    The CLI gates this behind typed operator confirmation."""
    journal = Journal(journal_root / f"{venue}.jsonl")
    last = journal.last_event(types=frozenset({"run"}))
    if last is None or "state_after" not in last:
        raise RunnerError(f"no run event with state_after in {venue} journal")
    state = state_from_dict(last["state_after"])
    save_state(state_path(state_root, venue), state)
    return f"restored {venue} state from journaled run {last['run_key']}"
```

- [ ] **Step 6: Run the runner tests**

Run: `uv run pytest tests/test_runner.py tests/test_notify.py -q`
Expected: all PASS.

- [ ] **Step 7: Write the failing CLI tests**

Append to `tests/test_cli.py`:

```python
# --- trading run (Task 10) ---


def _run_args(tmp_path, cfg_dir, extra=()):
    return [
        "run",
        "--venue",
        "equities",
        "--config-dir",
        str(cfg_dir),
        "--state-dir",
        str(tmp_path / "state"),
        "--journal-dir",
        str(tmp_path / "journal"),
        *extra,
    ]


def _freeze_now(monkeypatch, iso: str):
    frozen = datetime.datetime.fromisoformat(iso)
    monkeypatch.setattr("trading.cli._utcnow", lambda: frozen)


def test_run_bootstraps_then_noops_same_bar(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    assert main(_run_args(tmp_path, cfg_dir)) == 0
    assert (tmp_path / "state" / "equities" / "portfolio.json").exists()
    assert (tmp_path / "journal" / "equities.jsonl").exists()
    capsys.readouterr()

    assert main(_run_args(tmp_path, cfg_dir)) == 0
    assert "noop" in capsys.readouterr().out


def test_run_next_day_fills_and_reports_json(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()
    _freeze_now(monkeypatch, "2026-07-02T22:30:00+00:00")
    rc = main(_run_args(tmp_path, cfg_dir, extra=["--json"]))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["run_key"] == "equities:2026-07-02T00:00:00+00:00"


def test_run_coverage_failure_exits_nonzero_and_notifies(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")

    def flaky(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        if symbol in {"BBB", "CCC"}:
            raise DataFetchError(f"boom {symbol}")
        return _fake_history(symbol, start, end)

    monkeypatch.setattr("trading.venues.equities._yf_download", flaky)
    notes = []
    monkeypatch.setattr("trading.cli.notify", lambda t, m: notes.append((t, m)))
    assert main(_run_args(tmp_path, cfg_dir)) == 1
    assert notes  # every failed run fires a notification


def test_run_restore_from_journal_requires_typed_confirmation(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    state_file = tmp_path / "state" / "equities" / "portfolio.json"
    good = state_file.read_text()
    state_file.write_text("garbage")
    capsys.readouterr()

    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    assert main(_run_args(tmp_path, cfg_dir, extra=["--restore-from-journal"])) == 1
    assert state_file.read_text() == "garbage"

    monkeypatch.setattr("builtins.input", lambda prompt="": "RESTORE")
    assert main(_run_args(tmp_path, cfg_dir, extra=["--restore-from-journal"])) == 0
    assert json.loads(state_file.read_text()) == json.loads(good)
```

Note for the earnings filter in these tests: `_setup_equities` uses the real `config/equities.toml`, where `earnings_blackout_enabled = true`, so `_cmd_run` would call yfinance. Stub it in `_freeze_now`'s neighborhood by adding this line to `_setup_equities` (network isolation, consistent with the existing monkeypatching there):

```python
    monkeypatch.setattr("trading.runner.fetch_earnings_dates", lambda symbols: ({}, False))
```

- [ ] **Step 8: Wire the CLI**

In `src/trading/cli.py`:

Add imports at the top (alongside the existing ones):

```python
from trading.notify import notify
from trading.runner import RunnerError, restore_from_journal, run_venue
```

Add the clock helper (below `VENUES = [...]`):

```python
def _utcnow() -> datetime.datetime:
    """The CLI is the only module allowed to read the clock."""
    return datetime.datetime.now(datetime.UTC)


def _add_store_dirs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-dir", default="config", help="directory with <venue>.toml")
    parser.add_argument("--state-dir", default="state", help="portfolio state root")
    parser.add_argument("--journal-dir", default="journal", help="journal root")
```

In `build_parser()`, after the `rankings` subparser:

```python
    run = sub.add_parser("run", help="one live-paper cycle now")
    run.add_argument("--venue", choices=VENUES, required=True)
    run.add_argument("--json", action="store_true", help="machine-readable output")
    run.add_argument(
        "--restore-from-journal",
        action="store_true",
        help="rebuild state/<venue>/portfolio.json from the last journal snapshot (confirms)",
    )
    _add_store_dirs(run)
```

Replace `main()`'s dispatch with:

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "rankings": _cmd_rankings,
        "run": _cmd_run,
    }
    return handlers[args.command](args)
```

Add the command handler (after `_cmd_rankings`):

```python
def _cmd_run(args: argparse.Namespace) -> int:
    config = load_venue_config(args.venue, Path(args.config_dir))
    state_root, journal_root = Path(args.state_dir), Path(args.journal_dir)

    if args.restore_from_journal:
        print(f"This will overwrite {state_root / args.venue / 'portfolio.json'} from the journal.")
        if input("Type RESTORE to confirm: ").strip() != "RESTORE":
            print("aborted")
            return 1
        try:
            print(restore_from_journal(args.venue, state_root, journal_root))
        except RunnerError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    adapter = make_adapter(config)
    cache = OhlcvCache(Path(config.data.cache_dir), config.data.refetch_days)
    try:
        outcome = run_venue(
            config,
            adapter,
            cache,
            now=_utcnow(),
            state_root=state_root,
            journal_root=journal_root,
            notify=notify,
        )
    except Exception as exc:  # a silent dead pipeline is the worst failure (spec)
        notify("trading: run crashed", f"{args.venue}: {exc}")
        raise
    if args.json:
        print(
            json.dumps(
                {
                    "venue": outcome.venue,
                    "status": outcome.status,
                    "message": outcome.message,
                    "run_key": outcome.run_key,
                }
            )
        )
    else:
        print(f"{outcome.venue}: {outcome.status} — {outcome.message}")
    return 0 if outcome.status in ("ok", "noop") else 1
```

- [ ] **Step 9: Run the full suite**

Run: `uv run pytest -q 2>&1 | tail -3`
Expected: all PASS (existing rankings CLI tests keep passing — the parser change is additive).

- [ ] **Step 10: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/runner.py src/trading/notify.py src/trading/cli.py tests/test_runner.py tests/test_notify.py tests/test_cli.py
git commit -m "Add live-paper runner with lockfile, bootstrap, staleness, and trading run CLI [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: Daily digest and `trading digest`

`digest/YYYY-MM-DD.md` (UTC date) regenerated after each venue run from journal events — a 60-second read.

**Files:**
- Create: `src/trading/digest.py`
- Create: `tests/test_digest.py`
- Modify: `src/trading/runner.py` (write digest after the journal append)
- Modify: `src/trading/cli.py` (`digest` subcommand)
- Modify: `tests/test_runner.py` (digest write assertion; `_run` helper gains `digest_root`)
- Modify: `tests/test_cli.py` (digest command test)

**Interfaces:**
- Consumes: Journal (Task 8), the `run` event schema (Task 10).
- Produces (consumed by Task 12's status rendering conventions and the README):
  - `collect_run_events(journal_root: Path, venues: list[str], date_iso: str) -> list[dict]` — the LAST `run` event per venue whose `ts` falls on `date_iso`.
  - `build_digest(date_iso: str, run_events: list[dict]) -> str` — pure markdown builder.
  - `write_digest(digest_root: Path, date_iso: str, run_events: list[dict]) -> Path`.
  - `run_venue(..., digest_root: Path | None = None)` — when set, regenerates today's digest (both venues' sections) after journaling.
  - CLI: `trading digest [--date YYYY-MM-DD] [--digest-dir D] [--json]` — prints the digest file (default: latest).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_digest.py`:

```python
import json
from pathlib import Path

from trading.digest import build_digest, collect_run_events, write_digest
from trading.journal import Journal


def _event(venue="equities", ts="2026-07-01T22:30:00+00:00", value=1012.34, n=1) -> dict:
    return {
        "event": "run",
        "venue": venue,
        "run_key": f"{venue}:2026-07-01T00:00:00+00:00",
        "ts": ts,
        "decision_ts": "2026-07-01T00:00:00+00:00",
        "config_hash": "abc123def456",
        "regime": {"state": "risk_on", "exposure_multiplier": 1.0},
        "coverage": {"requested": 4, "fetched": 4, "ratio": 1.0},
        "benchmark": {"symbol": "SPY", "close": 624.0, "start_price": 620.0},
        "starting_balance": 1000.0,
        "ranking": [
            {"rank": 1, "symbol": "UPUP", "status": "tradable", "composite": 0.83},
            {"rank": 2, "symbol": "FLAT", "status": "tradable", "composite": 0.55},
        ],
        "fills": [
            {
                "symbol": "UPUP", "side": "buy", "qty": 1.7982, "price": 100.05,
                "fee": 0.0, "bar_ts": "2026-07-01T00:00:00+00:00", "reason": "entry",
                "realized_pnl": None,
            }
        ],
        "new_orders": [
            {
                "symbol": "MEH1", "side": "sell", "notional": 0.0,
                "decision_ts": "2026-07-01T00:00:00+00:00", "reason": "stop_loss",
                "atr_at_decision": 0.0, "composite": 0.0, "rank": 0,
            }
        ],
        "skips": [{"symbol": "*", "action": "entry", "reason": "stale_run_entries_skipped"}],
        "warnings": ["quarantined: BADCO"],
        "snapshot": {
            "value": value,
            "cash": 640.0,
            "unsettled": 180.0,
            "positions": [
                {
                    "symbol": "UPUP", "qty": 1.7982, "entry_price": 100.05,
                    "last_close": 107.0, "market_value": 192.4,
                    "unrealized_pnl_pct": 0.0695, "stop_price": 94.05,
                    "stop_distance_pct": 0.121, "entry_rank": 1,
                    "entry_composite": 0.83, "entry_ts": "2026-07-01T00:00:00+00:00",
                }
            ],
        },
        "state_after": {"breaker_tripped": True},
        "n": n,
    }


def test_build_digest_renders_all_sections():
    text = build_digest("2026-07-01", [_event()])
    assert "# Trading digest — 2026-07-01 (UTC)" in text
    assert "## equities — risk_on (exposure x1.0)" in text
    assert "$1,012.34" in text and "+1.23%" in text  # portfolio value + P&L since start
    assert "SPY buy-and-hold: +0.65%" in text  # 624/620 - 1
    assert "| UPUP |" in text and "rank #1, composite 0.83" in text
    assert "buy" in text and "stop_loss" in text
    assert "quarantined: BADCO" in text
    assert "late run: entries skipped" in text
    assert "TRIPPED" in text


def test_build_digest_without_events():
    text = build_digest("2026-07-01", [])
    assert "No runs journaled" in text


def test_collect_run_events_takes_last_run_per_venue_for_date(tmp_path):
    journal = Journal(tmp_path / "journal" / "equities.jsonl")
    journal.append({"event": "bootstrap", "venue": "equities", "ts": "2026-07-01T22:00:00+00:00"})
    journal.append(_event(n=1))
    journal.append(_event(n=2))
    journal.append(_event(ts="2026-06-30T22:30:00+00:00", n=3))  # other date
    events = collect_run_events(tmp_path / "journal", ["equities", "crypto"], "2026-07-01")
    assert [e["n"] for e in events] == [2]  # latest same-date run; crypto absent


def test_write_digest_creates_dated_file(tmp_path):
    path = write_digest(tmp_path / "digest", "2026-07-01", [_event()])
    assert path == tmp_path / "digest" / "2026-07-01.md"
    assert "equities" in path.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_digest.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.digest'`.

- [ ] **Step 3: Implement the digest module**

Create `src/trading/digest.py`:

```python
"""Daily digest (spec: Reporting & Operations): a 60-second markdown read.

Built purely from journal run events, so `trading digest` shows exactly what
the pipeline decided — value + P&L vs benchmark buy-and-hold, open positions
with entry rationale and distance-to-stop, fills, top-5 ranking, regime, and
warnings (quarantines, staleness, breaker state, earnings degradation).
"""

from __future__ import annotations

from pathlib import Path

from trading.journal import Journal


def collect_run_events(journal_root: Path, venues: list[str], date_iso: str) -> list[dict]:
    events: list[dict] = []
    for venue in venues:
        journal = Journal(journal_root / f"{venue}.jsonl")
        last: dict | None = None
        for event in journal.events():
            if event.get("event") == "run" and str(event.get("ts", "")).startswith(date_iso):
                last = event
        if last is not None:
            events.append(last)
    return events


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _pct(x: float) -> str:
    return f"{x:+.2%}"


def _venue_section(event: dict) -> list[str]:
    snapshot = event["snapshot"]
    start = float(event["starting_balance"])
    value = float(snapshot["value"])
    bench = event["benchmark"]
    bench_pnl = float(bench["close"]) / float(bench["start_price"]) - 1.0
    regime = event["regime"]
    breaker = bool(event["state_after"]["breaker_tripped"])

    lines = [
        f"## {event['venue']} — {regime['state']} (exposure x{regime['exposure_multiplier']})",
        "",
        f"- Portfolio: {_money(value)} ({_pct(value / start - 1.0)} since start) | "
        f"{bench['symbol']} buy-and-hold: {_pct(bench_pnl)}",
        f"- Cash: {_money(float(snapshot['cash']))} settled, "
        f"{_money(float(snapshot['unsettled']))} unsettled | "
        f"breaker: {'TRIPPED' if breaker else 'armed'}",
        "",
        "### Open positions",
    ]
    if snapshot["positions"]:
        lines += [
            "| symbol | qty | entry | last | P&L | stop | to stop | rationale |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for m in snapshot["positions"]:
            lines.append(
                f"| {m['symbol']} | {m['qty']:.4f} | {m['entry_price']:.2f} "
                f"| {m['last_close']:.2f} | {_pct(m['unrealized_pnl_pct'])} "
                f"| {m['stop_price']:.2f} | {_pct(m['stop_distance_pct'])} "
                f"| rank #{m['entry_rank']}, composite {m['entry_composite']:.2f} |"
            )
    else:
        lines.append("- none")

    lines += ["", "### Today's fills"]
    if event["fills"]:
        lines += [
            "| symbol | side | qty | price | fee | reason | realized P&L |",
            "|---|---|---|---|---|---|---|",
        ]
        for f in event["fills"]:
            realized = "-" if f["realized_pnl"] is None else _money(float(f["realized_pnl"]))
            lines.append(
                f"| {f['symbol']} | {f['side']} | {f['qty']:.4f} | {f['price']:.2f} "
                f"| {f['fee']:.2f} | {f['reason']} | {realized} |"
            )
    else:
        lines.append("- none")

    lines += ["", "### New orders for the next bar"]
    if event["new_orders"]:
        lines += [f"- {o['side']} {o['symbol']} ({o['reason']})" for o in event["new_orders"]]
    else:
        lines.append("- none")

    lines += ["", "### Top 5 ranking", "| # | symbol | composite | status |", "|---|---|---|---|"]
    for row in event["ranking"][:5]:
        lines.append(f"| {row['rank']} | {row['symbol']} | {row['composite']} | {row['status']} |")

    warnings = list(event["warnings"])
    if any(s["reason"].startswith("stale_run") for s in event["skips"]):
        warnings.append("late run: entries skipped (staleness bound exceeded)")
    if breaker:
        warnings.append("circuit breaker is TRIPPED: entries halted until `trading reset-breaker`")
    lines += ["", "### Warnings"]
    lines += [f"- {w}" for w in warnings] if warnings else ["- none"]
    lines.append("")
    return lines


def build_digest(date_iso: str, run_events: list[dict]) -> str:
    lines = [f"# Trading digest — {date_iso} (UTC)", ""]
    if not run_events:
        lines += ["No runs journaled for this date.", ""]
    for event in run_events:
        lines.extend(_venue_section(event))
    return "\n".join(lines)


def write_digest(digest_root: Path, date_iso: str, run_events: list[dict]) -> Path:
    digest_root.mkdir(parents=True, exist_ok=True)
    path = digest_root / f"{date_iso}.md"
    path.write_text(build_digest(date_iso, run_events))
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_digest.py -q`
Expected: 4 PASS.

- [ ] **Step 5: Wire the digest into the runner and CLI**

In `src/trading/runner.py`:

Add the import:

```python
from trading.digest import collect_run_events, write_digest
```

Change `run_venue`'s signature to accept the digest root:

```python
def run_venue(
    config: VenueConfig,
    adapter: VenueAdapter,
    cache: OhlcvCache,
    *,
    now: datetime.datetime,
    state_root: Path,
    journal_root: Path,
    notify: Notifier,
    digest_root: Path | None = None,
) -> RunOutcome:
```

and insert immediately after the `journal.append(_run_event(...))` line:

```python
        if digest_root is not None:
            date_iso = now.date().isoformat()
            write_digest(
                digest_root,
                date_iso,
                collect_run_events(journal_root, ["equities", "crypto"], date_iso),
            )
```

In `src/trading/cli.py`:

Extend `_add_store_dirs` with the digest directory:

```python
def _add_store_dirs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-dir", default="config", help="directory with <venue>.toml")
    parser.add_argument("--state-dir", default="state", help="portfolio state root")
    parser.add_argument("--journal-dir", default="journal", help="journal root")
    parser.add_argument("--digest-dir", default="digest", help="daily digest directory")
```

In `_cmd_run`, pass it through to `run_venue`:

```python
            digest_root=Path(args.digest_dir),
```

In `build_parser()`, after the `run` subparser:

```python
    digest = sub.add_parser("digest", help="print a daily digest (default: latest)")
    digest.add_argument("--date", default=None, help="digest date, YYYY-MM-DD")
    digest.add_argument("--json", action="store_true", help="machine-readable output")
    digest.add_argument("--digest-dir", default="digest", help="daily digest directory")
```

Register `"digest": _cmd_digest` in `main()`'s handlers dict, and add:

```python
def _cmd_digest(args: argparse.Namespace) -> int:
    digest_dir = Path(args.digest_dir)
    if args.date:
        path = digest_dir / f"{args.date}.md"
    else:
        candidates = sorted(digest_dir.glob("*.md"))
        path = candidates[-1] if candidates else None
    if path is None or not path.exists():
        print("no digest found", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"date": path.stem, "markdown": path.read_text()}))
    else:
        print(path.read_text())
    return 0
```

- [ ] **Step 6: Extend the runner and CLI tests**

In `tests/test_runner.py`, change the `_run` helper's `run_venue` call to include `digest_root=tmp_path / "digest",` (after the `notify=...` argument), then append:

```python
def test_run_writes_daily_digest(tmp_path):
    _run(tmp_path)
    digest_file = tmp_path / "digest" / "2026-07-01.md"
    assert digest_file.exists()
    text = digest_file.read_text()
    assert "## equities" in text
    assert "Top 5 ranking" in text
```

In `tests/test_cli.py`, extend the `_run_args` helper (from Task 10) so `trading run` never writes a digest into the real repo `digest/` during tests — add these two items to its list, right before `*extra`:

```python
        "--digest-dir",
        str(tmp_path / "digest"),
```

then append:

```python
def test_digest_command_prints_latest_and_specific(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    digest_args = ["--digest-dir", str(tmp_path / "digest")]
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()

    assert main(["digest", *digest_args]) == 0
    assert "Trading digest — 2026-07-01" in capsys.readouterr().out

    assert main(["digest", "--date", "2026-07-01", *digest_args]) == 0
    assert "equities" in capsys.readouterr().out

    assert main(["digest", "--date", "1999-01-01", *digest_args]) == 1
```

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q 2>&1 | tail -3`
Expected: all PASS.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/digest.py tests/test_digest.py src/trading/runner.py src/trading/cli.py tests/test_runner.py tests/test_cli.py
git commit -m "Write daily digest from journal events and add trading digest CLI [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: `trading status` and `trading reset-breaker`

Operational visibility (both venues, time-since-last-successful-run, breaker state) and the manual circuit-breaker reset with typed confirmation.

**Files:**
- Modify: `src/trading/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_state`/`save_state`/`state_path` (Task 10), `Journal` (Task 8), the run event schema (Task 10).
- Produces:
  - CLI: `trading status [--json] [--state-dir D] [--journal-dir D]` — per venue: portfolio value, P&L vs benchmark buy-and-hold, open position count, breaker state, hours since last successful run (`run`/`bootstrap` event). Venues render as `not bootstrapped` or `CORRUPT STATE` when applicable; exit 0 always (status is informational).
  - CLI: `trading reset-breaker --venue V [--state-dir D] [--journal-dir D]` — requires typing `RESET`; clears the breaker, rebases the high-water mark to the last journaled snapshot value (otherwise the unchanged HWM would re-trip on the next run), journals a `breaker_reset` event. Exit 0 on reset or not-tripped; 1 on abort/missing state.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
# --- trading status / reset-breaker (Task 12) ---


def _store_args(tmp_path):
    return ["--state-dir", str(tmp_path / "state"), "--journal-dir", str(tmp_path / "journal")]


def test_status_reports_both_venues(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()
    _freeze_now(monkeypatch, "2026-07-02T01:30:00+00:00")  # 3h after the run

    assert main(["status", "--json", *_store_args(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    by_venue = {v["venue"]: v for v in payload["venues"]}
    equities = by_venue["equities"]
    assert equities["state"] == "ok"
    assert equities["value"] == pytest.approx(1000.0)
    assert equities["pnl_pct"] == pytest.approx(0.0)
    assert "benchmark_pnl_pct" in equities
    assert equities["breaker_tripped"] is False
    assert equities["hours_since_last_success"] == pytest.approx(3.0)
    assert by_venue["crypto"]["state"] == "not bootstrapped"


def test_status_flags_corrupt_state(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    (tmp_path / "state" / "equities" / "portfolio.json").write_text("garbage")
    capsys.readouterr()
    assert main(["status", "--json", *_store_args(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    by_venue = {v["venue"]: v for v in payload["venues"]}
    assert by_venue["equities"]["state"] == "corrupt"


def test_status_human_output_renders(tmp_path, monkeypatch, capsys):
    _freeze_now(monkeypatch, "2026-07-02T01:30:00+00:00")
    assert main(["status", *_store_args(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "equities" in out and "crypto" in out


def test_reset_breaker_requires_typed_confirmation(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()

    from trading.runner import load_state, save_state, state_path

    path = state_path(tmp_path / "state", "equities")
    state = load_state(path)
    state.breaker_tripped = True
    state.breaker_tripped_at = "2026-07-01T00:00:00+00:00"
    state.high_water_mark = 5000.0
    save_state(path, state)

    monkeypatch.setattr("builtins.input", lambda prompt="": "nope")
    assert main(["reset-breaker", "--venue", "equities", *_store_args(tmp_path)]) == 1
    assert load_state(path).breaker_tripped is True

    monkeypatch.setattr("builtins.input", lambda prompt="": "RESET")
    assert main(["reset-breaker", "--venue", "equities", *_store_args(tmp_path)]) == 0
    restored = load_state(path)
    assert restored.breaker_tripped is False
    assert restored.breaker_tripped_at is None
    # HWM rebased to the last journaled snapshot value, not the stale 5000.
    assert restored.high_water_mark == pytest.approx(1000.0)

    from trading.journal import Journal

    events = list(Journal(tmp_path / "journal" / "equities.jsonl").events())
    assert events[-1]["event"] == "breaker_reset"


def test_reset_breaker_when_not_tripped_is_a_noop(tmp_path, monkeypatch, capsys):
    cfg_dir = _setup_equities(tmp_path, monkeypatch)
    _freeze_now(monkeypatch, "2026-07-01T22:30:00+00:00")
    main(_run_args(tmp_path, cfg_dir))
    capsys.readouterr()
    assert main(["reset-breaker", "--venue", "equities", *_store_args(tmp_path)]) == 0
    assert "not tripped" in capsys.readouterr().out


def test_reset_breaker_without_state_errors(tmp_path, capsys):
    assert main(["reset-breaker", "--venue", "equities", *_store_args(tmp_path)]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -q`
Expected: the new tests FAIL — argparse rejects the unknown `status` / `reset-breaker` commands with exit code 2 (`SystemExit`), so the assertions never run and the tests error.

- [ ] **Step 3: Implement both commands**

In `src/trading/cli.py`:

Add imports — `Journal` and `StateError` are new lines; extend the existing `from trading.runner import ...` line (from Task 10) with the three extra names:

```python
from trading.journal import Journal
from trading.runner import (
    RunnerError,
    load_state,
    restore_from_journal,
    run_venue,
    save_state,
    state_path,
)
from trading.simulator.state import StateError
```

In `build_parser()`, after the `digest` subparser:

```python
    status = sub.add_parser("status", help="portfolios, P&L vs benchmark, last-run health")
    status.add_argument("--json", action="store_true", help="machine-readable output")
    status.add_argument("--state-dir", default="state", help="portfolio state root")
    status.add_argument("--journal-dir", default="journal", help="journal root")

    breaker = sub.add_parser("reset-breaker", help="manually reset the circuit breaker (confirms)")
    breaker.add_argument("--venue", choices=VENUES, required=True)
    breaker.add_argument("--state-dir", default="state", help="portfolio state root")
    breaker.add_argument("--journal-dir", default="journal", help="journal root")
```

Register `"status": _cmd_status` and `"reset-breaker": _cmd_reset_breaker` in `main()`'s handlers dict, then add:

```python
def _venue_status(venue: str, state_dir: Path, journal_dir: Path, now: datetime.datetime) -> dict:
    info: dict[str, object] = {"venue": venue, "state": "ok"}
    try:
        state = load_state(state_path(state_dir, venue))
    except StateError:
        return {"venue": venue, "state": "corrupt"}
    if state is None:
        return {"venue": venue, "state": "not bootstrapped"}
    info["breaker_tripped"] = state.breaker_tripped
    info["positions"] = len(state.positions)

    journal = Journal(journal_dir / f"{venue}.jsonl")
    last_run = journal.last_event(types=frozenset({"run"}))
    if last_run is not None:
        snapshot = last_run["snapshot"]
        start = float(last_run["starting_balance"])
        bench = last_run["benchmark"]
        info["value"] = float(snapshot["value"])
        info["pnl_pct"] = float(snapshot["value"]) / start - 1.0
        info["benchmark_pnl_pct"] = float(bench["close"]) / float(bench["start_price"]) - 1.0
    last_ok = journal.last_event(types=frozenset({"run", "bootstrap"}))
    if last_ok is not None:
        last_ts = datetime.datetime.fromisoformat(last_ok["ts"])
        info["hours_since_last_success"] = (now - last_ts).total_seconds() / 3600
    return info


def _cmd_status(args: argparse.Namespace) -> int:
    now = _utcnow()
    venues = [
        _venue_status(v, Path(args.state_dir), Path(args.journal_dir), now) for v in VENUES
    ]
    if args.json:
        print(json.dumps({"as_of": now.isoformat(), "venues": venues}))
        return 0

    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"trading status — {now.isoformat(timespec='seconds')}")
    for col in ["venue", "state", "value", "P&L", "benchmark", "positions", "breaker", "last run"]:
        table.add_column(col)
    for v in venues:
        table.add_row(
            str(v["venue"]),
            str(v["state"]),
            f"${v['value']:,.2f}" if "value" in v else "-",
            f"{v['pnl_pct']:+.2%}" if "pnl_pct" in v else "-",
            f"{v['benchmark_pnl_pct']:+.2%}" if "benchmark_pnl_pct" in v else "-",
            str(v.get("positions", "-")),
            "TRIPPED" if v.get("breaker_tripped") else "armed",
            f"{v['hours_since_last_success']:.1f}h ago"
            if "hours_since_last_success" in v
            else "never",
        )
    Console().print(table)
    return 0


def _cmd_reset_breaker(args: argparse.Namespace) -> int:
    path = state_path(Path(args.state_dir), args.venue)
    try:
        state = load_state(path)
    except StateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if state is None:
        print(f"no state for {args.venue}; nothing to reset", file=sys.stderr)
        return 1
    if not state.breaker_tripped:
        print(f"{args.venue}: breaker is not tripped")
        return 0
    print(f"Circuit breaker for {args.venue} tripped at {state.breaker_tripped_at}.")
    if input("Type RESET to re-enable entries: ").strip() != "RESET":
        print("aborted")
        return 1
    journal = Journal(Path(args.journal_dir) / f"{args.venue}.jsonl")
    last_run = journal.last_event(types=frozenset({"run"}))
    if last_run is not None:
        # Rebase the high-water mark to the last marked value; otherwise the
        # unchanged HWM re-trips the breaker on the very next run.
        state.high_water_mark = float(last_run["snapshot"]["value"])
    state.breaker_tripped = False
    state.breaker_tripped_at = None
    save_state(path, state)
    journal.append(
        {"event": "breaker_reset", "venue": args.venue, "ts": _utcnow().isoformat()}
    )
    print(f"{args.venue}: breaker reset; entries re-enabled")
    return 0
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q 2>&1 | tail -3`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/cli.py tests/test_cli.py
git commit -m "Add trading status and reset-breaker commands [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 13: launchd scheduling — `trading schedule install|status|remove`

Two per-venue LaunchAgents: equities weekday evenings 18:30 (machine-local time, assumed America/New_York — documented), crypto daily 01:00 local (lands after the 00:00 UTC bar close for US offsets). launchd coalesces missed intervals on wake into one late run, which the staleness rule then bounds.

**Files:**
- Create: `src/trading/schedule.py`
- Create: `tests/test_schedule.py`
- Modify: `src/trading/cli.py` (`schedule` subcommand)
- Modify: `config/crypto.toml` (`staleness_hours`, see Step 1)
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: nothing from the simulator (pure ops tooling).
- Produces:
  - In `trading.schedule`: `label(venue) -> str` (`com.travis.trading.<venue>`), `plist_path(agents_dir: Path, venue: str) -> Path`, `build_plist(venue: str, repo_root: Path, uv_path: str) -> bytes`, `install(repo_root: Path, agents_dir: Path) -> list[str]`, `status(agents_dir: Path) -> dict[str, dict]` (`{venue: {"installed": bool, "loaded": bool}}`), `remove(agents_dir: Path) -> list[str]`, `ScheduleError(RuntimeError)`. `_launchctl(*args)` is the subprocess touchpoint (monkeypatchable, like M1's `_yf_download`).
  - CLI: `trading schedule {install,status,remove} [--agents-dir D] [--json]` (agents dir defaults to `~/Library/LaunchAgents`; repo root = cwd).

- [ ] **Step 1: Widen the crypto staleness bound to cover the schedule**

The crypto decision bar completes at 00:00 UTC. The scheduled 01:00 America/New_York run lands at 05:00 UTC (EDT) or 06:00 UTC (EST) — winter runs would sit exactly on a 6-hour bound and any launchd jitter would misclassify a *scheduled* run as stale. Staleness is config (spec default 6h); widen it one hour so the venue's own schedule can never trip it. In `config/crypto.toml` `[portfolio]`, replace the `staleness_hours` line:

```toml
staleness_hours = 7    # bar completes 00:00 UTC; the 01:00 America/New_York run
                       # lands 05:00 (EDT) / 06:00 (EST) UTC — 6h would sit exactly
                       # on the winter boundary, so widen to 7
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_schedule.py`:

```python
import plistlib
import subprocess
from pathlib import Path

import pytest

from trading import schedule
from trading.schedule import (
    ScheduleError,
    build_plist,
    install,
    label,
    plist_path,
    remove,
    status,
)


def test_labels_and_paths():
    assert label("equities") == "com.travis.trading.equities"
    assert plist_path(Path("/tmp/agents"), "crypto") == Path(
        "/tmp/agents/com.travis.trading.crypto.plist"
    )


def test_equities_plist_runs_weekday_evenings_in_local_time():
    payload = plistlib.loads(build_plist("equities", Path("/repo"), "/usr/local/bin/uv"))
    assert payload["Label"] == "com.travis.trading.equities"
    assert payload["ProgramArguments"] == [
        "/usr/local/bin/uv", "run", "--project", "/repo", "trading", "run", "--venue", "equities",
    ]
    assert payload["WorkingDirectory"] == "/repo"
    # StartCalendarInterval is LOCAL time; 18:30 assumes America/New_York (README).
    assert payload["StartCalendarInterval"] == [
        {"Weekday": w, "Hour": 18, "Minute": 30} for w in (1, 2, 3, 4, 5)
    ]
    assert payload["StandardErrorPath"].endswith("state/equities/launchd.log")


def test_crypto_plist_runs_daily_0100_local():
    payload = plistlib.loads(build_plist("crypto", Path("/repo"), "/usr/local/bin/uv"))
    assert payload["StartCalendarInterval"] == [{"Hour": 1, "Minute": 0}]


def _ok(*args, **kwargs):
    return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


def test_install_writes_plists_and_bootstraps(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(schedule, "_launchctl", lambda *a: calls.append(a) or _ok())
    monkeypatch.setattr(schedule.shutil, "which", lambda name: "/usr/local/bin/uv")
    messages = install(Path("/repo"), tmp_path)
    for venue in ("equities", "crypto"):
        assert plist_path(tmp_path, venue).exists()
    actions = [c[0] for c in calls]
    assert actions.count("bootout") == 2  # idempotent reinstall
    assert actions.count("bootstrap") == 2
    assert len(messages) == 2


def test_install_requires_uv_on_path(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule.shutil, "which", lambda name: None)
    with pytest.raises(ScheduleError, match="uv"):
        install(Path("/repo"), tmp_path)


def test_status_reports_installed_and_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule.shutil, "which", lambda name: "/usr/local/bin/uv")
    monkeypatch.setattr(schedule, "_launchctl", lambda *a: _ok())
    install(Path("/repo"), tmp_path)

    def print_only_equities(*args):
        loaded = args[0] == "print" and args[1].endswith("equities")
        return subprocess.CompletedProcess(args=args, returncode=0 if loaded else 113,
                                           stdout="", stderr="")

    monkeypatch.setattr(schedule, "_launchctl", print_only_equities)
    result = status(tmp_path)
    assert result["equities"] == {"installed": True, "loaded": True}
    assert result["crypto"] == {"installed": True, "loaded": False}


def test_remove_boots_out_and_deletes(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule.shutil, "which", lambda name: "/usr/local/bin/uv")
    calls = []
    monkeypatch.setattr(schedule, "_launchctl", lambda *a: calls.append(a) or _ok())
    install(Path("/repo"), tmp_path)
    remove(tmp_path)
    assert [c[0] for c in calls].count("bootout") == 4  # 2 install + 2 remove
    assert not plist_path(tmp_path, "equities").exists()
    assert not plist_path(tmp_path, "crypto").exists()
```

Append to `tests/test_cli.py`:

```python
def test_schedule_status_cli_json(tmp_path, monkeypatch, capsys):
    import subprocess as sp

    monkeypatch.setattr(
        "trading.schedule._launchctl",
        lambda *a: sp.CompletedProcess(args=a, returncode=113, stdout="", stderr=""),
    )
    rc = main(["schedule", "status", "--agents-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "equities": {"installed": False, "loaded": False},
        "crypto": {"installed": False, "loaded": False},
    }
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_schedule.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading.schedule'`.

- [ ] **Step 4: Implement the scheduler**

Create `src/trading/schedule.py`:

```python
"""launchd LaunchAgents for the daily runs (spec: Runtime, CLI).

StartCalendarInterval is machine-LOCAL time: equities 18:30 assumes the Mac
is in America/New_York (documented in the README); crypto 01:00 local lands
after the 00:00 UTC bar close for US offsets. launchd coalesces intervals
missed while asleep into one run on wake — the runner's staleness rule then
skips entries and still processes exits. Failure signaling is the runner's
job (notifications), not launchd's.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
from pathlib import Path

VENUES = ("equities", "crypto")

_SCHEDULES: dict[str, list[dict[str, int]]] = {
    # Weekday evenings after NYSE close (local time; 1 = Monday ... 5 = Friday).
    "equities": [{"Weekday": w, "Hour": 18, "Minute": 30} for w in (1, 2, 3, 4, 5)],
    # Daily, after the 00:00 UTC crypto bar close (for US-negative offsets).
    "crypto": [{"Hour": 1, "Minute": 0}],
}


class ScheduleError(RuntimeError):
    pass


def label(venue: str) -> str:
    return f"com.travis.trading.{venue}"


def plist_path(agents_dir: Path, venue: str) -> Path:
    return agents_dir / f"{label(venue)}.plist"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    """Subprocess touchpoint, isolated for monkeypatching."""
    return subprocess.run(
        ["launchctl", *args], capture_output=True, text=True, check=False
    )


def build_plist(venue: str, repo_root: Path, uv_path: str) -> bytes:
    log = repo_root / "state" / venue / "launchd.log"
    return plistlib.dumps(
        {
            "Label": label(venue),
            "ProgramArguments": [
                uv_path, "run", "--project", str(repo_root), "trading", "run",
                "--venue", venue,
            ],
            "WorkingDirectory": str(repo_root),
            "StartCalendarInterval": _SCHEDULES[venue],
            "StandardOutPath": str(log),
            "StandardErrorPath": str(log),
        }
    )


def _domain() -> str:
    return f"gui/{os.getuid()}"


def install(repo_root: Path, agents_dir: Path) -> list[str]:
    uv_path = shutil.which("uv")
    if uv_path is None:
        raise ScheduleError("uv not found on PATH; cannot build LaunchAgents")
    agents_dir.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []
    for venue in VENUES:
        path = plist_path(agents_dir, venue)
        path.write_bytes(build_plist(venue, repo_root, uv_path))
        _launchctl("bootout", f"{_domain()}/{label(venue)}")  # idempotent reinstall
        result = _launchctl("bootstrap", _domain(), str(path))
        if result.returncode != 0:
            raise ScheduleError(f"launchctl bootstrap failed for {venue}: {result.stderr}")
        messages.append(f"{venue}: installed {path}")
    return messages


def status(agents_dir: Path) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for venue in VENUES:
        loaded = _launchctl("print", f"{_domain()}/{label(venue)}").returncode == 0
        report[venue] = {"installed": plist_path(agents_dir, venue).exists(), "loaded": loaded}
    return report


def remove(agents_dir: Path) -> list[str]:
    messages: list[str] = []
    for venue in VENUES:
        _launchctl("bootout", f"{_domain()}/{label(venue)}")
        plist_path(agents_dir, venue).unlink(missing_ok=True)
        messages.append(f"{venue}: removed")
    return messages
```

- [ ] **Step 5: Wire the CLI**

In `src/trading/cli.py`, add to `build_parser()` after the `reset-breaker` subparser:

```python
    sched = sub.add_parser("schedule", help="manage launchd jobs")
    sched.add_argument("action", choices=["install", "status", "remove"])
    sched.add_argument(
        "--agents-dir", default=None, help="LaunchAgents dir (default ~/Library/LaunchAgents)"
    )
    sched.add_argument("--json", action="store_true", help="machine-readable output")
```

Register `"schedule": _cmd_schedule` in `main()`'s handlers dict, then add:

```python
def _cmd_schedule(args: argparse.Namespace) -> int:
    from trading import schedule

    agents_dir = Path(args.agents_dir) if args.agents_dir else (
        Path.home() / "Library" / "LaunchAgents"
    )
    try:
        if args.action == "install":
            output: object = schedule.install(Path.cwd(), agents_dir)
        elif args.action == "remove":
            output = schedule.remove(agents_dir)
        else:
            output = schedule.status(agents_dir)
    except schedule.ScheduleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(output))
    elif isinstance(output, dict):
        for venue, info in output.items():
            state = "loaded" if info["loaded"] else ("installed, not loaded" if info["installed"] else "not installed")
            print(f"{venue}: {state}")
    else:
        for line in output:
            print(line)
    return 0
```

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q 2>&1 | tail -3`
Expected: all PASS.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check . && uv run ruff format .
git add src/trading/schedule.py tests/test_schedule.py src/trading/cli.py tests/test_cli.py config/crypto.toml
git commit -m "Add launchd scheduling with trading schedule install/status/remove [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 14: README, gitignore, and live verification

Documentation for the full M2 command set, gitignored runtime directories, and a real end-to-end shakeout (including the earnings-source reliability decision the spec defers to implementation time).

**Files:**
- Modify: `.gitignore`
- Modify: `README.md` (full rewrite below)
- Possibly modify: `config/equities.toml` + `README.md` (ONLY if Step 4's live earnings check fails)

**Interfaces:**
- Consumes: everything.
- Produces: the spec's "concise README at repo root" and a verified, runnable system.

- [ ] **Step 1: Gitignore the runtime state**

Append to `.gitignore`:

```
/state/
/journal/
/digest/
```

- [ ] **Step 2: Rewrite the README**

Replace `README.md` with:

```markdown
# trading

Momentum swing trading system. It ranks liquid assets (S&P 500 + Nasdaq-100
equities; Robinhood-listed crypto) by likelihood of near-term upward moves
using price/volume momentum behind a market-regime gate, and paper-trades
that ranking daily with $1,000 per venue under strict risk rules. Backtesting
and walk-forward validation arrive in the next milestone.

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and macOS (for
notifications and launchd scheduling):

    uv sync

## Commands

    uv run trading run --venue equities|crypto   # one live-paper cycle now
    uv run trading status                        # portfolios, P&L vs benchmark, last-run health
    uv run trading rankings --venue equities|crypto  # current ranked table w/ sub-scores
    uv run trading digest [--date YYYY-MM-DD]    # print digest (default: latest)
    uv run trading schedule install|status|remove    # manage launchd jobs
    uv run trading reset-breaker --venue VENUE   # manual circuit-breaker reset (confirms)

Every command prints human-readable tables; add `--json` for machine
consumption. Run from the repo root. `run` exits nonzero (and fires a macOS
notification) when a cycle fails or is skipped — a silent dead pipeline is
the failure mode this system is designed to avoid.

## How a run works

Each `trading run` fetches fresh bars, then: (1) fills the previous run's
pending orders at the first bar after their decision bar (open + 5 bps
slippage + venue fees), (2) checks exits — frozen 1.5x ATR-20 stops, regime
flush, trend break, time stop, forced exits, (3) checks entries — regime-
gated top of ranking, score threshold, cooldowns, settled cash (T+1 for
equities), max 25%/day deployment (35% crypto — fits one 30% position), and
(4) writes new pending orders for the next run. A decision bar is never
traded twice (journal-enforced); a late run (e.g. after sleep/wake) still
processes exits but skips entries beyond the staleness bound. Drawdown >20%
from the high-water mark halts entries venue-wide until `reset-breaker`.

## Scheduling

`trading schedule install` creates two LaunchAgents in ~/Library/LaunchAgents:
equities weekdays 18:30 and crypto daily 01:00. **launchd uses machine-local
time; the schedule assumes this Mac runs in America/New_York.** Crypto at
01:00 ET lands after the 00:00 UTC daily bar close. Runs missed while asleep
coalesce into one late run on wake, bounded by the staleness rule above.

## Where things live

- `config/<venue>.toml` — every tunable number (fees, windows, risk rules).
- `state/<venue>/portfolio.json` — paper portfolio (gitignored; atomic
  writes). If it corrupts, the run refuses to act and notifies; recover with
  `trading run --venue V --restore-from-journal`.
- `journal/<venue>.jsonl` — append-only record of every run: regime, full
  ranking with sub-scores, decisions made AND skipped (with reasons), fills,
  snapshot, config hash. State is reconstructible from it.
- `digest/YYYY-MM-DD.md` — daily digest (UTC date), regenerated after each
  venue run.
- `data/<venue>/*.parquet` — OHLCV cache (gitignored; safe to delete).

## Reading the digest (60 seconds)

Per venue: portfolio value and P&L vs buying-and-holding the benchmark
(SPY/BTC) since bootstrap; open positions with entry rationale (rank +
composite at entry) and distance-to-stop; today's fills; pending orders;
top-5 ranking; regime; warnings (quarantined symbols, stale-run entry skips,
circuit-breaker state, earnings-data degradation).

## Earnings blackout

Equities entries are blocked within 5 sessions of an earnings date
(yfinance-sourced). The filter is fail-open: a fetch failure only logs a
warning. Kill switch: `earnings_blackout_enabled` in `config/equities.toml`.

## Rankings output

Sub-scores are cross-sectional percentiles (0-1, higher = better):
`mom_short/med/long` (vol-adjusted momentum), `volume_surge`, `breakout`,
`overextension` (lower is better; inverted in the composite), `composite`
(equal-weight blend; the ranking key), `raw_return_30d` (raw return feeding
the crypto fee gate). `status` is `tradable` / `sell_only` / `untradable`;
regime is `risk_on` / `neutral` / `risk_off` (full / half / no new entries).
```

- [ ] **Step 3: Run the full suite and lint**

Run: `uv run pytest -q 2>&1 | tail -3 && uv run ruff check . && uv run ruff format --check .`
Expected: all tests PASS, no lint findings.

- [ ] **Step 4: Live earnings-source check (network)**

Run:

```bash
uv run python -c "
from trading.earnings import fetch_earnings_dates
dates, degraded = fetch_earnings_dates(['AAPL', 'MSFT', 'NVDA'])
print('degraded:', degraded)
for s, d in dates.items():
    print(s, d[:4])
" 2>&1 | tee /tmp/claude-m2-earnings-check.log
```

Expected: `degraded: False` and plausible upcoming/recent quarterly dates for all three symbols.

**If it fails or returns empty/garbage dates:** per the spec ("if that source proves unreliable in practice, the filter is dropped entirely, both modes, and the divergence documented"), set `earnings_blackout_enabled = false` in `config/equities.toml` and replace the README's "Earnings blackout" section body with:

```markdown
Dropped: yfinance earnings dates proved unreliable at implementation time
(2026-07), so the filter is disabled in BOTH live and backtest modes — a
filter that exists live but not in backtest is worse than no filter. The
code path remains behind `earnings_blackout_enabled` in
`config/equities.toml` should a reliable source appear.
```

- [ ] **Step 5: Live end-to-end shakeout (network)**

```bash
uv run trading run --venue crypto 2>&1 | tee /tmp/claude-m2-crypto-run.log
uv run trading run --venue crypto   # second invocation must print "noop"
uv run trading run --venue equities 2>&1 | tee /tmp/claude-m2-equities-run.log  # first run fetches ~500 symbols; slow
uv run trading status
uv run trading digest
uv run trading schedule install && uv run trading schedule status && uv run trading schedule remove
```

Expected: both runs exit 0 with a journaled run event and pending orders (or
justified skip reasons — e.g. `regime_risk_off`) visible in `trading digest`;
the second crypto invocation is a `noop`; `status` shows both venues with a
time-since-last-success; `schedule` installs, reports `loaded`, and removes
cleanly. Inspect `journal/*.jsonl` and `digest/*.md` by eye: ranking snapshot
present, skips have reasons, config_hash populated. `state/`, `journal/`,
`digest/` must NOT appear in `git status`.

- [ ] **Step 6: Commit**

```bash
git add .gitignore README.md
git add -u config/ 2>/dev/null || true   # only if Step 4 flipped the kill switch
git commit -m "Document M2 paper trading loop and gitignore runtime state [AI]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Judgment calls this plan locks in (deviations / gap-fills)

Recorded so reviewers see them without diffing against the spec:

1. **`RankingsResult.benchmark_bars`** added alongside the locked `bars` field — the benchmark (SPY) is not in the equities universe, and the digest's buy-and-hold comparison must come from "the same bars".
2. **Crypto `max_daily_deployment_pct` 0.25 → 0.35** — the spec's own numbers deadlock (30% position size can never pass a 25% daily cap); 0.35 admits exactly one full-size entry/day.
3. **Crypto `staleness_hours` 6 → 7** — the scheduled 01:00 America/New_York run lands at 06:00 UTC in winter, exactly on the spec-default 6h bound.
4. **`atr_window = 20` added to `[portfolio]`** — the spec fixes ATR-20 but had no config field, and every number must be config.
5. **Cooldown applies to stop-outs only** (`stop_loss` fills, which include regime-flush-ratchet stops) per the spec's entry rule "not stopped out within 7 calendar-day cooldown"; trend-break/time-stop/forced exits set no cooldown.
6. **Buy orders with no fill bar this run are cancelled; sell orders stay pending** — the concrete reading of "no catch-up trading" + "exits always processed".
7. **Staleness deadline = decision bar + 1 day + `staleness_hours`** — one uniform, config-driven form of the spec's venue-specific phrasing ("2h after open" / "6h past UTC midnight"); slightly conservative for equities.
8. **`reset-breaker` rebases the high-water mark** to the last journaled snapshot value — otherwise the untouched HWM re-trips the breaker on the next run and the reset is meaningless.
9. **Earnings fetched for the top 15 ranked symbols only** (entry candidates can't realistically reach deeper given ≤5 slots + the deployment cap); fetch failures degrade to no-filter per symbol with a journal warning.
10. **`sell_only` held positions are force-exited** (spec's exit rule 5) even though only entry-blocking was strictly required — spec text wins.
11. **State recovery is `trading run --venue V --restore-from-journal`** (typed `RESTORE` confirmation) rather than a new top-level command, keeping the CLI surface exactly the spec's seven commands.
12. **Quarantined/fetch-failed held symbols are never exited that run** (warn + hold): "no trades" for quarantined data beats a forced exit computed from data we don't trust.
