# Tier-1 Signal Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register the 21 pre-registered Tier-1 signals (9 price/volume, 5 fundamentals, 5 options, 2 industry-relative) plus the minimal engine extensions they require, per `docs/superpowers/specs/2026-07-09-tier1-signal-batch-design.md`.

**Architecture:** The alphasearch engine (`src/trading/alphasearch/`) scores signals through PIT `PanelView` accessors that truncate at `as_of` via `index.searchsorted(side="right")`. This batch extends `panel.py` with full OHLCV bars, a cached factor frame, precomputed rolling ivol/beta features (FeaturePanel pattern: vectorize full-span once, gather as-of), a prior-options-cell accessor, a leg-volume capability flag, and sector threading; `spec.py` gains 21 registrations; `sweep.py` gains the `requires_option_volume` refusal and factor threading into panel assembly; `segments.py` gets the pre-registered §3.4 fundamentals amendment.

**Tech Stack:** Python ≥3.12, pandas 2.3.1, numpy (via pandas), pytest 8.4.1, ruff 0.12.3. Hand-rolled OLS only — **no scipy, no statsmodels, no requests** (repo policy; `evaluate.ols` is the reference implementation).

## Global Constraints

- **The signal table is FROZEN pre-registered science** (spec §2). Every formula, window, minimum-observation floor, and sign in this plan transcribes it exactly. If an implementation problem seems to require changing any of them, STOP and consult the developer — that is a spec amendment, not a code fix.
- Signs (higher score = more attractive): `mom_12_2` +, `overnight` +, `park_vol` −, `ivol` −, `max5` −, `beta` −, `amihud` +, `vol_trend` +, `div_yield` +, `asset_growth` −, `net_issuance` −, `roa` +, `droa` +, `rev_growth` +, `cp_vol` +, `osv` −, `otm_put_iv` −, `iv_change` −, `dskew` −, `ind_mom` +, `ind_rel_rev` = −(trail21 − sector mean trail21) registered raw (the minus is in the formula). One-line rationale comment at each registration (existing convention in `spec.py`).
- Floors (spec §6): `beta` min 126 obs of its 252-obs window; `ivol` min 15 obs of its 21-obs window; `amihud` min 126 valid terms of its 252-bar window. All other windows are strict (insufficient history → NaN).
- PIT: every new accessor truncates via `searchsorted(side="right")` (or `.loc[:as_of]` for filed-date frames, matching `fundamentals_row`). NaN means "missing, dropped" — never imputed, never a fabricated zero.
- These signals add **NO new hashed sweep parameters**: 45d/300d/window constants are part of each signal's identity (its name); `_hashed_params` in `sweep.py` must NOT change.
- Test suite runs warnings-as-errors (`pyproject.toml`); ruff rules E,F,I,UP,B at 100 columns. Run `uv run pytest -q` and `uv run ruff check src tests scripts` before every commit.
- Style rules from the repo: `pd.Timedelta(N, unit="D")` (never `days=N` kwarg form used inconsistently — match existing call sites), `is None` checks (never truthiness for maybe-empty values), no `sort_index()`-then-`duplicated(keep=...)` dedupe (quicksort instability), no float-exact assertions on compounded synthetic fixtures (`math.isclose`), no constant-column regression fixtures (rank deficiency).
- Commits: granular, one logical change, message suffix ` [AI]` (repo convention, e.g. `Add load_bars panel accessor [AI]`).
- Work from the repo root `/Users/travis/Source/personal/trading`. All paths below are repo-relative.

## File Structure

| File | Change |
|---|---|
| `src/trading/alphasearch/panel.py` | `load_bars`, `BAR_COLUMNS`, `compute_rolling_features` (ivol/beta), `PanelData` fields (`bars`, `factors`, `features`, `sectors`, `has_option_volume`), `PanelView` accessors (`bars`, `factors`, `feature`, `option_row_prior`, `fundamentals_row_prior`, `sector`, `has_option_volume`), `cell_metrics` gains `opt_dollar_vol`, `load_options` returns a leg-volume flag, `build_panel` gains `factors`/`sectors` |
| `src/trading/alphasearch/spec.py` | `SignalSpec.requires_option_volume`; 21 new registrations + helpers |
| `src/trading/alphasearch/sweep.py` | `UniverseSpec.sic_map_path`; `build_universe_panel(spec, factors)`; `panel_factory` takes factors; `_check_universe_supports` option-volume refusal; `_universe_sectors` |
| `src/trading/alphasearch/segments.py` | §3.4 amendment: deep pools carry `fundamentals_dir` when the store exists; thread `sic_map_path` onto emitted specs |
| `src/trading/cli.py` | reword the `--segments` refusal hint (deep pools may now carry fundamentals) |
| `tests/alphasearch_helpers.py` | `make_panel` builds bars/factors/features/sectors; `assemble_panel`; `make_cell` gains `with_volume` + otm-leg mids |
| `tests/test_alphasearch_panel.py` | new accessor tests; `load_options` 3-tuple |
| `tests/test_alphasearch_features.py` | NEW: ivol/beta math vs `evaluate.ols` cross-check + boundaries |
| `tests/test_alphasearch_tier1.py` | NEW: per-signal hand-computed unit tests for all 21 |
| `tests/test_alphasearch_lookahead.py` | long-history panel; perturb bars/factors/options/fundamentals then REASSEMBLE (features recomputed); registry-wide anti-vacuity guard |
| `tests/test_alphasearch_sweep.py` | factory lambdas take factors; factor-threading test; option-volume refusal test; sic-map sector-derivation test |
| `tests/test_alphasearch_spec.py` | registry-count/flags test updated per task (16 → 25 → 30 → 35 → 37) |
| `tests/test_alphasearch_segments.py` | §3.4 amendment test; `sic_map_path` threading test |
| `tests/test_alphasearch_segments_golden.py` | Tier-1 golden root (extended bars + fundamentals store + two-sector sic map); one-signal-per-family sweep; fundamentals-on-deep-segment sweep |
| `docs/experiments.md`, `docs/glossary.md` | §10 pre-registration note; anomaly-zoo glossary entries |

**Known data landmine (also record in the final report):** the largecap bar cache `data/equities-tiingo/*.parquet` is the legacy NARROW schema (`open,high,low,close,volume` only); the midcap cache has the extended schema (`div_cash`, `split_factor`, `close_raw`). `load_bars` NaN-fills missing extended columns — never the venue layer's 0.0/1.0 defaults, because a cache that never stored dividends cannot claim "no dividends". Consequence: `div_yield` and `net_issuance` on largecap will journal honest all-NaN → `SortError` error trials until the largecap cache is re-backfilled with the extended schema (operational prerequisite, like the spec §5 fundamentals rsync).

---

### Task 1: `load_bars` + `PanelView.bars()` / `.factors()` + factor threading through panel assembly

**Files:**
- Modify: `src/trading/alphasearch/panel.py`
- Modify: `src/trading/alphasearch/sweep.py` (`build_universe_panel`, `run_sweep`, `run_holdout`)
- Modify: `tests/alphasearch_helpers.py`
- Test: `tests/test_alphasearch_panel.py`, `tests/test_alphasearch_sweep.py`

**Interfaces:**
- Produces: `BAR_COLUMNS: list[str] = ["open", "high", "low", "close", "volume", "div_cash", "split_factor"]`; `load_bars(cache_dir: Path, symbols: Iterable[str]) -> dict[str, pd.DataFrame]`; `PanelData.bars: dict[str, pd.DataFrame]`, `PanelData.factors: pd.DataFrame`; `PanelView.bars(symbol) -> pd.DataFrame`, `PanelView.factors() -> pd.DataFrame`; `build_panel(..., factors: pd.DataFrame | None = None)`; `build_universe_panel(spec: UniverseSpec, factors: pd.DataFrame | None = None)`; `panel_factory: Callable[[UniverseSpec, pd.DataFrame | None], PanelData]`; helpers `assemble_panel(bars, options, fundamentals, factors) -> PanelData` and `make_panel(..., factors=None)` whose panels carry bars+factors.
- Consumes: existing `PanelData`/`PanelView`/`build_panel` (`panel.py`), `run_sweep`/`run_holdout` (`sweep.py`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_alphasearch_panel.py` (add `BAR_COLUMNS`, `load_bars` to the existing `from trading.alphasearch.panel import (...)` block):

```python
def test_load_bars_reads_full_schema_and_nan_fills_legacy_columns(tmp_path):
    idx = pd.date_range("2020-01-02", periods=3, freq="B", tz="UTC")
    wide = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": [1.5, 1.6, 1.7],
         "volume": 10.0, "div_cash": [0.0, 0.25, 0.0], "split_factor": 1.0,
         "close_raw": 1.5},
        index=idx,
    )
    narrow = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
        index=idx,
    )
    wide.to_parquet(tmp_path / "WIDE.parquet")
    narrow.to_parquet(tmp_path / "NARROW.parquet")
    got = load_bars(tmp_path, ["WIDE", "NARROW", "NOPE"])
    assert set(got) == {"WIDE", "NARROW"}
    assert list(got["WIDE"].columns) == BAR_COLUMNS      # close_raw dropped
    assert got["WIDE"]["div_cash"].iloc[1] == 0.25
    # A legacy narrow cache cannot claim "no dividends": NaN, never 0.0/1.0.
    assert got["NARROW"]["div_cash"].isna().all()
    assert got["NARROW"]["split_factor"].isna().all()


def test_view_bars_truncates_at_as_of_and_is_empty_for_unknown_symbol():
    idx = pd.date_range("2020-01-06", periods=5, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {c: float(i) for i, c in enumerate(BAR_COLUMNS)}, index=idx
    )
    panel = PanelData(closes={"AAA": frame["close"]}, bars={"AAA": frame},
                      symbols=("AAA",))
    view = panel.view(idx[2])
    assert len(view.bars("AAA")) == 3
    assert view.bars("AAA").index.max() == idx[2]
    empty = view.bars("NOPE")
    assert empty.empty and list(empty.columns) == BAR_COLUMNS


def test_view_factors_truncates_at_as_of():
    idx = pd.date_range("2020-01-06", periods=5, freq="B", tz="UTC")
    factors = pd.DataFrame(
        {"Mkt-RF": 0.001, "SMB": 0.0, "HML": 0.0, "RF": 0.0001}, index=idx
    )
    panel = PanelData(closes={}, factors=factors, symbols=())
    got = panel.view(idx[1]).factors()
    assert len(got) == 2 and got.index.max() == idx[1]
    before = pd.Timestamp("2019-12-31", tz="UTC")
    assert panel.view(before).factors().empty


def test_build_panel_derives_closes_from_bars_and_stores_factors(tmp_path):
    idx = pd.date_range("2020-01-02", periods=4, freq="B", tz="UTC")
    pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": [1.0, 2.0, 3.0, 4.0],
         "volume": 10.0},
        index=idx,
    ).to_parquet(tmp_path / "AAA.parquet")
    factors = pd.DataFrame(
        {"Mkt-RF": 0.001, "SMB": 0.0, "HML": 0.0, "RF": 0.0001, "Mom": 0.0},
        index=idx,
    )
    panel = build_panel(tmp_path, None, None, symbols=("AAA",), factors=factors)
    assert list(panel.bars["AAA"].columns) == BAR_COLUMNS
    assert panel.closes["AAA"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert panel.factors.equals(factors)


def test_build_panel_without_factors_stores_an_empty_frame(tmp_path):
    idx = pd.date_range("2020-01-02", periods=4, freq="B", tz="UTC")
    pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
        index=idx,
    ).to_parquet(tmp_path / "AAA.parquet")
    panel = build_panel(tmp_path, None, None, symbols=("AAA",))
    assert panel.factors.empty
```

Append to `tests/test_alphasearch_sweep.py`:

```python
def test_run_sweep_passes_factors_to_the_panel_factory(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    seen: list = []

    def factory(_u, f):
        seen.append(f)
        return panel

    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=WINDOW, panel_factory=factory)
    assert len(seen) == 1 and seen[0] is factors
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_panel.py -q 2>&1 | tee /tmp/claude-t1-red.log`
Expected: FAIL — `ImportError: cannot import name 'load_bars'` (and the sweep test fails with `TypeError: factory() ... positional arguments` once panel tests import; run `uv run pytest tests/test_alphasearch_sweep.py::test_run_sweep_passes_factors_to_the_panel_factory -q` too).

- [ ] **Step 3: Implement `load_bars` + `PanelData`/`PanelView` extensions in `panel.py`**

Add after `load_closes` (keep `load_closes` itself — script compatibility):

```python
# Full bar schema served to signals. Legacy largecap caches predate the
# extended columns; load_bars NaN-fills them (see docstring) rather than
# fabricating "no dividends / no splits".
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "div_cash", "split_factor"]


def load_bars(cache_dir: Path, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Full bar frames (BAR_COLUMNS) per symbol from a Tiingo parquet cache.

    A symbol without a cached parquet is absent from the result (missing-data
    rule, spec section 5.5). A legacy narrow cache (OHLCV only) gets NaN
    div_cash/split_factor -- NOT the venue layer's 0.0/1.0 migration
    defaults: a cache that never stored dividends cannot claim "no
    dividends", so div_yield/net_issuance go honestly NaN there instead of
    scoring fabricated zeros. Extra columns (close_raw) are dropped.
    """
    out: dict[str, pd.DataFrame] = {}
    for symbol in sorted(set(symbols)):
        path = cache_dir / f"{symbol}.parquet"
        if path.exists():
            out[symbol] = pd.read_parquet(path).reindex(columns=BAR_COLUMNS)
    return out
```

In `PanelData`, append two fields after `corrupt_cells: int = 0`:

```python
    bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Daily Ken French factor frame (Mkt-RF, SMB, HML, RF[, Mom]; decimals,
    # UTC index) from evaluate.load_factors. Published history: PIT holds by
    # truncation like every other store (spec section 3, item 1).
    factors: pd.DataFrame = field(default_factory=pd.DataFrame)
```

In `PanelView`, add after `option_row`:

```python
    def bars(self, symbol: str) -> pd.DataFrame:
        """The symbol's bars (BAR_COLUMNS) up to and including as_of; an
        empty BAR_COLUMNS frame when the symbol has none."""
        frame = self._panel.bars.get(symbol)
        if frame is None:
            return pd.DataFrame(columns=BAR_COLUMNS, dtype="float64")
        pos = frame.index.searchsorted(self.as_of, side="right")
        return frame.iloc[:pos]

    def factors(self) -> pd.DataFrame:
        """Factor rows dated at or before as_of (empty frame when the panel
        carries no factors)."""
        frame = self._panel.factors
        if frame.empty:
            return frame
        pos = frame.index.searchsorted(self.as_of, side="right")
        return frame.iloc[:pos]
```

In `build_panel`: change the signature and body to load bars and thread factors:

```python
def build_panel(
    cache_dir: Path,
    samples: Path | None,
    fundamentals_dir: Path | None,
    *,
    symbols: tuple[str, ...] | None = None,
    factors: pd.DataFrame | None = None,
) -> PanelData:
```

Replace the `closes = load_closes(...)` / `if not closes:` lines with:

```python
    bars = load_bars(cache_dir, allowlist)
    if not bars:
        raise PanelError(f"no bar caches under {cache_dir} for the requested universe")
    closes = {s: frame["close"] for s, frame in bars.items()}
```

Replace `universe = tuple(s for s in allowlist if s in closes)` with `... if s in bars` and extend the return:

```python
    universe = tuple(s for s in allowlist if s in bars)
    return PanelData(
        closes={s: closes[s] for s in universe},
        options={s: options[s] for s in universe if s in options},
        fundamentals={s: fundamentals[s] for s in universe if s in fundamentals},
        symbols=universe,
        corrupt_cells=corrupt,
        bars={s: bars[s] for s in universe},
        factors=pd.DataFrame() if factors is None else factors,
    )
```

- [ ] **Step 4: Thread factors through `sweep.py`**

```python
def build_universe_panel(
    spec: UniverseSpec, factors: pd.DataFrame | None = None
) -> PanelData:
    return build_panel(
        spec.cache_dir, spec.samples, spec.fundamentals_dir,
        symbols=spec.symbols, factors=factors,
    )
```

In `run_sweep` and `run_holdout`, change the parameter annotation to
`panel_factory: Callable[[UniverseSpec, pd.DataFrame | None], PanelData] = build_universe_panel`
and the call sites to `panel_factory(uspec, factors)` (one in `run_sweep`'s validation loop, one in `run_holdout`).

- [ ] **Step 5: Update `tests/alphasearch_helpers.py` — `make_panel` builds bars, `assemble_panel` added**

Replace `make_panel` and add `assemble_panel` (keep `make_cell`, `month_firsts`, `make_factors` as they are; note the rng consumption order for closes is UNCHANGED so every downstream engineered-alpha assertion still holds):

```python
def assemble_panel(
    bars: dict[str, pd.DataFrame],
    options: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    factors: pd.DataFrame,
) -> PanelData:
    """PanelData from raw stores, deriving what build_panel derives (closes
    from bars). The lookahead test perturbs RAW stores and reassembles
    through here, so derived state (Task 2: precomputed rolling features) is
    recomputed from the perturbed inputs, never perturbed directly."""
    closes = {s: frame["close"] for s, frame in bars.items()}
    return PanelData(
        closes=closes, options=options, fundamentals=fundamentals,
        symbols=tuple(sorted(bars)), bars=bars, factors=factors,
    )


def make_panel(
    n_symbols: int = 16,
    start: str = "2020-01-02",
    periods: int = 130,
    seed: int = 7,
    with_options: bool = True,
    with_fundamentals: bool = True,
    factors: pd.DataFrame | None = None,
) -> PanelData:
    """Symbol S<i> drifts at (i - n/2)*2bp/day plus small seeded noise (same
    recipe/rng order as ever: closes are bit-identical to the pre-bars
    fixture). Bars extend the closes deterministically: open gaps up
    1bp*(i+1) from the prior close (a per-symbol overnight drift), high/low
    bracket the close at +-(0.2+0.05i)%, volume 1e5*(i+1), div_cash
    0.01*(i+1) daily, split_factor 1.0. Fundamentals file THREE times so the
    300-day YoY rule and the post-cutoff perturbation are both exercised on
    long fixtures (positions 0 / 63% / 95% of the index)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="B", tz="UTC")
    names = [f"S{i:02d}" for i in range(n_symbols)]
    bars: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(names):
        drift = (i - n_symbols / 2) * 2e-4
        rets = drift + rng.normal(0.0, 0.002, size=periods)
        close = pd.Series(100.0 * np.cumprod(1 + rets), index=idx)
        open_ = close.shift(1) * (1 + 1e-4 * (i + 1))
        open_.iloc[0] = close.iloc[0]
        span = 0.002 + 0.0005 * i
        bars[sym] = pd.DataFrame(
            {"open": open_, "high": close * (1 + span), "low": close * (1 - span),
             "close": close, "volume": 1e5 * (i + 1), "div_cash": 0.01 * (i + 1),
             "split_factor": 1.0},
            index=idx,
        )
    if factors is None:
        factors = make_factors()
    options: dict[str, pd.DataFrame] = {}
    if with_options:
        cells = []
        for date in month_firsts(idx):
            iso = date.date().isoformat()
            for i, sym in enumerate(names):
                cells.append(make_cell(
                    sym, iso,
                    atm_iv=0.20 + 0.01 * i,
                    put_iv=0.24 + 0.01 * i,
                    call_iv=0.18 + 0.01 * i,
                    skew_put_atm=0.02 + 0.005 * i,
                    skew_put_call=0.01 + 0.002 * i,
                ))
        options = options_from_cells(cells)
    fundamentals: dict[str, pd.DataFrame] = {}
    if with_fundamentals:
        filed = pd.DatetimeIndex(
            [idx[0], idx[(63 * len(idx)) // 100], idx[(95 * len(idx)) // 100]]
        )
        for i, sym in enumerate(names):
            fundamentals[sym] = pd.DataFrame(
                {
                    "gross_profitability": [0.10 + 0.02 * i, 0.12 + 0.02 * i,
                                            0.13 + 0.02 * i],
                    "ttm_net_income": [1e6 * (i + 1), 1.1e6 * (i + 1),
                                       1.2e6 * (i + 1)],
                    "book_equity": [5e6 * (i + 1), 5.2e6 * (i + 1),
                                    5.3e6 * (i + 1)],
                    "shares_outstanding": [1e6 * (i + 1), 1.02e6 * (i + 1),
                                           1.04e6 * (i + 1)],
                    "assets": [1e7 * (i + 1), 1.1e7 * (i + 1), 1.15e7 * (i + 1)],
                    "revenue_ttm": [2e7 * (i + 1), 2.2e7 * (i + 1),
                                    2.3e7 * (i + 1)],
                },
                index=filed,
            )
    return assemble_panel(bars, options, fundamentals, factors)
```

- [ ] **Step 6: Mechanically update the sweep-test factory lambdas**

Every injected factory in `tests/test_alphasearch_sweep.py` must now accept the factors argument. Run `grep -n "panel_factory=lambda" tests/test_alphasearch_sweep.py` and apply, at every hit:

- `panel_factory=lambda _u: panel` → `panel_factory=lambda _u, _f: panel`
- `panel_factory=lambda _u: make_panel()` → `panel_factory=lambda _u, _f: make_panel()`
- `panel_factory=lambda u: panels[u.name]` → `panel_factory=lambda u, _f: panels[u.name]`

(~26 occurrences; no other test file injects a factory.)

- [ ] **Step 7: Run the full suite + ruff**

Run: `uv run pytest -q 2>&1 | tee /tmp/claude-t1-green.log && uv run ruff check src tests scripts`
Expected: all PASS (the existing lookahead test still constructs `PanelData` without bars — new fields default; that test is rewritten in Task 3), ruff clean.

- [ ] **Step 8: Commit**

```bash
git add src/trading/alphasearch/panel.py src/trading/alphasearch/sweep.py \
    tests/alphasearch_helpers.py tests/test_alphasearch_panel.py \
    tests/test_alphasearch_sweep.py
git commit -m "Add full-bar loading and factor-frame threading to alphasearch panels [AI]"
```

---

### Task 2: Precomputed rolling features — ivol (21d FF3 residual std) and beta (252d Mkt-RF slope)

**Files:**
- Modify: `src/trading/alphasearch/panel.py`
- Modify: `tests/alphasearch_helpers.py` (`assemble_panel` recomputes features)
- Test: `tests/test_alphasearch_features.py` (NEW), `tests/test_alphasearch_panel.py`

**Interfaces:**
- Produces: `IVOL_WINDOW=21`, `IVOL_MIN_OBS=15`, `BETA_WINDOW=252`, `BETA_MIN_OBS=126`, `ROLLING_FEATURES=["ivol", "beta"]`; `compute_rolling_features(closes: dict[str, pd.Series], factors: pd.DataFrame) -> dict[str, pd.DataFrame]` (per-symbol frames indexed by the joint return∩factor calendar, columns `ivol`,`beta`); `PanelData.features: dict[str, pd.DataFrame]`; `PanelView.feature(symbol: str, column: str) -> float` (as-of gather, NaN when absent). `build_panel(factors=...)` precomputes features.
- Consumes: Task 1's `PanelData.factors`/`build_panel(factors=...)`; `evaluate.ols` (test cross-check only), `evaluate.TRADING_DAYS`.

**Frozen definitions (spec §2 + §6):** for each date t on a symbol's joint calendar (inner join of its `pct_change()` returns with the factor frame, NaN rows dropped): `ivol` = OLS of the trailing ≤21 EXCESS returns (`ret − RF`) on `[1, Mkt-RF, SMB, HML]`, then `sqrt(SSE/(n−4)) · sqrt(252)` — the OLS residual standard error, matching `evaluate.ols`'s `s2 = e'e/(n−k)` convention; NaN below 15 obs. `beta` = slope of the trailing ≤252 excess returns on Mkt-RF (with intercept) = `cov(mkt, y)/var(mkt)`; NaN below 126 obs. Both are precomputed full-span per symbol (vectorized rolling cross-products + one batched `np.linalg.solve`, the FeaturePanel pattern from `src/trading/signals/engine.py`) and gathered as-of; row t uses only rows ≤ t, so the gather is PIT.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alphasearch_features.py`:

```python
"""compute_rolling_features: hand-constructed fixtures plus an independent
per-window OLS cross-check (evaluate.ols is the repo's own hand-rolled
reference implementation, so the two paths share no rolling machinery)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trading.alphasearch.evaluate import ols
from trading.alphasearch.panel import (
    BETA_MIN_OBS,
    BETA_WINDOW,
    IVOL_MIN_OBS,
    IVOL_WINDOW,
    PanelData,
    ROLLING_FEATURES,
    compute_rolling_features,
)


def _factors(periods: int, seed: int = 11, start: str = "2019-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0004, 0.01, size=periods),
            "SMB": rng.normal(0.0, 0.005, size=periods),
            "HML": rng.normal(0.0, 0.005, size=periods),
            "RF": 0.0001,
            "Mom": rng.normal(0.0, 0.006, size=periods),
        },
        index=idx,
    )


def _closes_from_excess(excess: np.ndarray, factors: pd.DataFrame) -> pd.Series:
    """A close series whose pct_change() minus RF reproduces `excess` on the
    factor calendar exactly (one leading seed bar absorbs the NaN first
    return)."""
    rets = excess + factors["RF"].to_numpy()
    seed_day = factors.index[0] - pd.Timedelta(1, unit="D")
    idx = pd.DatetimeIndex([seed_day]).append(factors.index)
    return pd.Series(100.0 * np.cumprod([1.0, *(1 + rets)]), index=idx)


def test_pure_ff3_combination_has_zero_ivol_and_exact_beta():
    factors = _factors(300)
    mkt = factors["Mkt-RF"].to_numpy()
    excess = 0.0005 + 0.7 * mkt  # exactly linear in the design -> zero residuals
    closes = {"AAA": _closes_from_excess(excess, factors)}
    feats = compute_rolling_features(closes, factors)["AAA"]
    assert list(feats.columns) == ROLLING_FEATURES
    assert feats["ivol"].iloc[-1] < 1e-6
    assert math.isclose(feats["beta"].iloc[-1], 0.7, rel_tol=1e-6)


def test_ivol_matches_an_independent_per_window_ols():
    factors = _factors(120)
    rng = np.random.default_rng(5)
    excess = rng.normal(0.0002, 0.01, size=120)
    closes = {"AAA": _closes_from_excess(excess, factors)}
    ivol = compute_rolling_features(closes, factors)["AAA"]["ivol"]
    x3 = factors[["Mkt-RF", "SMB", "HML"]].to_numpy()
    for t in range(120):
        lo = max(0, t + 1 - IVOL_WINDOW)
        yy, xx = excess[lo:t + 1], x3[lo:t + 1]
        got = ivol.iloc[t]
        if len(yy) < IVOL_MIN_OBS:
            assert math.isnan(got), t
            continue
        design = np.column_stack([np.ones(len(yy)), xx])
        beta, _se, _t, _r2, n = ols(design, yy)
        resid = yy - design @ beta
        want = math.sqrt(float(resid @ resid) / (n - 4)) * math.sqrt(252)
        assert math.isclose(got, want, rel_tol=1e-8), t


def test_beta_matches_cov_over_var_and_respects_the_min_obs_floor():
    factors = _factors(300)
    rng = np.random.default_rng(9)
    mkt = factors["Mkt-RF"].to_numpy()
    excess = 1.3 * mkt + rng.normal(0.0, 0.004, size=300)
    closes = {"AAA": _closes_from_excess(excess, factors)}
    beta = compute_rolling_features(closes, factors)["AAA"]["beta"]
    assert beta.iloc[: BETA_MIN_OBS - 1].isna().all()   # <=125 obs -> NaN
    assert not math.isnan(beta.iloc[BETA_MIN_OBS - 1])  # 126th obs -> value
    for t in (BETA_MIN_OBS - 1, 200, 299):
        lo = max(0, t + 1 - BETA_WINDOW)
        yy, xx = excess[lo:t + 1], mkt[lo:t + 1]
        xc = xx - xx.mean()
        want = float(xc @ (yy - yy.mean()) / (xc @ xc))
        assert math.isclose(beta.iloc[t], want, rel_tol=1e-8), t


def test_ivol_min_obs_boundary():
    factors = _factors(40)
    rng = np.random.default_rng(3)
    excess = rng.normal(0.0, 0.01, size=40)
    closes = {"AAA": _closes_from_excess(excess, factors)}
    ivol = compute_rolling_features(closes, factors)["AAA"]["ivol"]
    assert ivol.iloc[: IVOL_MIN_OBS - 1].isna().all()   # 14 obs -> NaN
    assert not math.isnan(ivol.iloc[IVOL_MIN_OBS - 1])  # 15 obs -> value


def test_empty_factors_yield_no_features():
    idx = pd.date_range("2020-01-02", periods=2, freq="B", tz="UTC")
    closes = {"AAA": pd.Series([100.0, 101.0], index=idx)}
    assert compute_rolling_features(closes, pd.DataFrame()) == {}


def test_feature_gather_is_as_of_and_nan_when_absent():
    idx = pd.date_range("2020-01-06", periods=3, freq="B", tz="UTC")
    feats = {"AAA": pd.DataFrame(
        {"ivol": [0.1, 0.2, 0.3], "beta": [1.0, 1.1, 1.2]}, index=idx
    )}
    panel = PanelData(closes={}, features=feats, symbols=("AAA",))
    before = pd.Timestamp("2020-01-05", tz="UTC")
    assert math.isnan(panel.view(before).feature("AAA", "ivol"))
    mid = pd.Timestamp("2020-01-07", tz="UTC")
    assert panel.view(mid).feature("AAA", "ivol") == 0.2
    assert panel.view(mid).feature("AAA", "beta") == 1.1
    assert math.isnan(panel.view(mid).feature("NOPE", "ivol"))
```

Append to `tests/test_alphasearch_panel.py`:

```python
def test_build_panel_precomputes_rolling_features_when_factors_supplied(tmp_path):
    idx = pd.date_range("2020-01-02", periods=30, freq="B", tz="UTC")
    pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5,
         "close": np.linspace(100.0, 110.0, 30), "volume": 10.0},
        index=idx,
    ).to_parquet(tmp_path / "AAA.parquet")
    rng = np.random.default_rng(2)
    factors = pd.DataFrame(
        {"Mkt-RF": rng.normal(0.0, 0.01, 30), "SMB": rng.normal(0.0, 0.005, 30),
         "HML": rng.normal(0.0, 0.005, 30), "RF": 0.0001, "Mom": 0.0},
        index=idx,
    )
    panel = build_panel(tmp_path, None, None, symbols=("AAA",), factors=factors)
    assert list(panel.features["AAA"].columns) == ROLLING_FEATURES
    without = build_panel(tmp_path, None, None, symbols=("AAA",))
    assert without.features == {}
```

(add `np` import if missing, and `ROLLING_FEATURES` to the panel import block.)

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_features.py -q`
Expected: FAIL — `ImportError: cannot import name 'BETA_MIN_OBS'`.

- [ ] **Step 3: Implement in `panel.py`**

Add `import math` to the imports and `from trading.alphasearch.evaluate import TRADING_DAYS` (evaluate imports no trading modules, so this creates no cycle). Add after `load_bars`:

```python
IVOL_WINDOW = 21
IVOL_MIN_OBS = 15   # spec section 6: 15-obs floor for 21d windows
BETA_WINDOW = 252
BETA_MIN_OBS = 126  # spec section 6: 126-obs floor for 252d windows
ROLLING_FEATURES = ["ivol", "beta"]
_FF3 = ["Mkt-RF", "SMB", "HML"]


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing-window sums via cumsum differences: row t sums rows
    max(0, t-window+1)..t. Works for any trailing shape, so the same helper
    rolls scalars, design rows, and stacked 4x4 cross-product matrices."""
    cum = np.cumsum(values, axis=0, dtype="float64")
    out = cum.copy()
    out[window:] = cum[window:] - cum[:-window]
    return out


def _ivol_column(y: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """Rolling FF3 residual std, annualized (spec section 2 `ivol`).

    Per row t: OLS of the trailing <=IVOL_WINDOW excess returns on
    [1, Mkt-RF, SMB, HML]; ivol = sqrt(SSE / (n - 4)) * sqrt(TRADING_DAYS)
    -- the OLS residual standard error, the same e'e/(n-k) convention as
    evaluate.ols. NaN below IVOL_MIN_OBS obs, and NaN for a numerically
    singular window (a rank-deficient window has no defined FF3 residual:
    missing data, spec section 6 -- real factor history is never
    rank-deficient over 15+ days). Fully vectorized: rolling cross-product
    sums, one batched det, one batched solve.
    """
    n = len(y)
    k = 4
    design = np.column_stack([np.ones(n), x3])
    xtx = _rolling_sum(design[:, :, None] * design[:, None, :], IVOL_WINDOW)
    xty = _rolling_sum(design * y[:, None], IVOL_WINDOW)
    yty = _rolling_sum(y * y, IVOL_WINDOW)
    counts = np.minimum(np.arange(n) + 1, IVOL_WINDOW)
    solvable = (counts >= IVOL_MIN_OBS) & (np.abs(np.linalg.det(xtx)) > 0.0)
    # Identity-substitute the unsolvable stacks: batched solve raises on ANY
    # singular member (the first 14 windows always are), and masked rows are
    # overwritten with NaN below anyway.
    safe = np.where(solvable[:, None, None], xtx, np.eye(k))
    beta = np.linalg.solve(safe, xty[:, :, None])[:, :, 0]
    sse = np.maximum(yty - np.einsum("nk,nk->n", beta, xty), 0.0)
    dof = np.maximum(counts - k, 1)
    out = np.sqrt(sse / dof) * math.sqrt(TRADING_DAYS)
    out[~solvable] = np.nan
    return out


def _beta_column(y: np.ndarray, mkt: np.ndarray) -> np.ndarray:
    """Rolling OLS slope of excess returns on Mkt-RF with intercept (spec
    section 2 `beta`): cov(mkt, y) / var(mkt) over the trailing
    <=BETA_WINDOW rows; NaN below BETA_MIN_OBS obs or when the window's
    market variance is zero."""
    n = len(y)
    counts = np.minimum(np.arange(n) + 1, BETA_WINDOW).astype("float64")
    sx = _rolling_sum(mkt, BETA_WINDOW)
    sy = _rolling_sum(y, BETA_WINDOW)
    sxy = _rolling_sum(mkt * y, BETA_WINDOW)
    sxx = _rolling_sum(mkt * mkt, BETA_WINDOW)
    denom = counts * sxx - sx * sx
    out = np.full(n, np.nan)
    valid = (counts >= BETA_MIN_OBS) & (denom > 0)
    out[valid] = (counts * sxy - sx * sy)[valid] / denom[valid]
    return out


def compute_rolling_features(
    closes: dict[str, pd.Series], factors: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Per-symbol full-span ivol/beta frames (the FeaturePanel pattern from
    trading.signals.engine: precompute once, gather as-of). Rows live on the
    inner join of the symbol's pct_change() calendar with the factor
    calendar (NaN rows dropped); every rolling sum looks strictly backward,
    so the value gathered at as_of is identical whether or not data after
    as_of exists -- the no-look-ahead perturbation test proves it. ivol
    regresses EXCESS returns (ret - RF) on FF3; beta regresses them on
    Mkt-RF alone."""
    if factors.empty:
        return {}
    cols = factors[[*_FF3, "RF"]]
    out: dict[str, pd.DataFrame] = {}
    for symbol, series in closes.items():
        rets = series.pct_change().rename("ret")
        joined = cols.join(rets, how="inner").dropna()
        if joined.empty:
            continue
        y = (joined["ret"] - joined["RF"]).to_numpy()
        x3 = joined[_FF3].to_numpy()
        out[symbol] = pd.DataFrame(
            {"ivol": _ivol_column(y, x3), "beta": _beta_column(y, x3[:, 0])},
            index=joined.index,
        )
    return out
```

In `PanelData`, append after `factors`:

```python
    # Precomputed heavy rolling features (ROLLING_FEATURES columns), keyed by
    # symbol, indexed by the joint return/factor calendar.
    features: dict[str, pd.DataFrame] = field(default_factory=dict)
```

In `PanelView`, add after `factors()`:

```python
    def feature(self, symbol: str, column: str) -> float:
        """Precomputed rolling feature (ROLLING_FEATURES) at the last row
        dated at or before as_of; NaN when the symbol has no feature rows
        yet. Same searchsorted gather as FeaturePanel.gather."""
        frame = self._panel.features.get(symbol)
        if frame is None or frame.empty:
            return math.nan
        pos = int(frame.index.searchsorted(self.as_of, side="right")) - 1
        if pos < 0:
            return math.nan
        return float(frame.iloc[pos][column])
```

In `build_panel`, before the `return`, add and thread:

```python
    features = compute_rolling_features(closes, factors) if factors is not None else {}
```

and in the `PanelData(...)` return add `features={s: features[s] for s in universe if s in features},` after `factors=...`.

In `tests/alphasearch_helpers.py`, `assemble_panel` recomputes features — replace its body:

```python
    closes = {s: frame["close"] for s, frame in bars.items()}
    return PanelData(
        closes=closes, options=options, fundamentals=fundamentals,
        symbols=tuple(sorted(bars)), bars=bars, factors=factors,
        features=compute_rolling_features(closes, factors),
    )
```

(import `compute_rolling_features` from `trading.alphasearch.panel`.)

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_alphasearch_features.py tests/test_alphasearch_panel.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite + ruff, then commit**

Run: `uv run pytest -q && uv run ruff check src tests scripts`
Expected: PASS.

```bash
git add src/trading/alphasearch/panel.py tests/alphasearch_helpers.py \
    tests/test_alphasearch_features.py tests/test_alphasearch_panel.py
git commit -m "Precompute rolling ivol/beta features with as-of gather [AI]"
```

---

### Task 3: Register the price/volume family (9 signals) + rewrite the lookahead test around raw-store perturbation

**Files:**
- Modify: `src/trading/alphasearch/spec.py`
- Modify: `tests/test_alphasearch_lookahead.py` (full rewrite shown below)
- Modify: `tests/test_alphasearch_spec.py` (registry count)
- Test: `tests/test_alphasearch_tier1.py` (NEW)

**Interfaces:**
- Produces: registrations `mom_12_2`, `overnight`, `park_vol`, `ivol`, `max5`, `beta`, `amihud`, `vol_trend`, `div_yield` (no requires flags — price family); helpers `_bar_signal`, `_feature_signal`, `_mom_12_2` (reused by Task 7's `ind_mom`). Registry size 25.
- Consumes: `PanelView.bars/feature` (Tasks 1–2), existing `_price_signal`/`_trail`.

**Frozen formulas (spec §2; all windows trading days; the as_of bar IS included — the existing `_trail` convention, "position at or before the decision date is the last element"):**
- `mom_12_2` = close[t−21]/close[t−252] − 1; NaN under 253 closes.
- `overnight` = Σ over the last 63 overnight gaps of ln(open_i / close_{i−1}); NaN under 64 bars.
- `park_vol` = sqrt(mean over last 21 bars of ln(H/L)²/(4·ln 2)) · sqrt(252); NaN under 21 bars. Registered NEGATED.
- `ivol` = precomputed feature (Task 2). Registered NEGATED.
- `max5` = mean of the 5 largest daily close-to-close returns in the last 21 returns; NaN under 22 closes. Registered NEGATED.
- `beta` = precomputed feature (Task 2). Registered NEGATED.
- `amihud` = mean over the last 252 bars of |ret| / (close·volume), terms with non-positive dollar volume or NaN return skipped; NaN under 126 valid terms.
- `vol_trend` = mean dollar volume last 21 bars / mean dollar volume last 252 bars; NaN under 252 bars.
- `div_yield` = Σ div_cash over last 252 bars / last close; NaN under 252 bars; `sum(min_count=1)` so an all-NaN div_cash column (legacy narrow cache) is NaN, never a fabricated 0.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_alphasearch_tier1.py`:

```python
"""Tier-1 signal batch: hand-computed unit fixtures per signal, including the
pre-registered sign conventions (spec 2026-07-09-tier1-signal-batch-design)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trading.alphasearch.panel import PanelData
from trading.alphasearch.spec import SIGNALS


def _score(name: str, panel: PanelData, as_of: pd.Timestamp) -> pd.Series:
    return SIGNALS[name].fn(panel.view(as_of), as_of)


def _bar_frame(
    close: pd.Series,
    *,
    open_: pd.Series | None = None,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    volume: float | pd.Series = 1000.0,
    div_cash: float | pd.Series = 0.0,
    split_factor: float | pd.Series = 1.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": close if open_ is None else open_,
         "high": close if high is None else high,
         "low": close if low is None else low,
         "close": close, "volume": volume, "div_cash": div_cash,
         "split_factor": split_factor},
        index=close.index,
    )


def _bar_panel(frames: dict[str, pd.DataFrame], **kwargs) -> PanelData:
    return PanelData(
        closes={s: f["close"] for s, f in frames.items()},
        bars=frames, symbols=tuple(sorted(frames)), **kwargs,
    )


def _geometric_close(rate: float, n: int, start: str = "2019-01-02") -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    return pd.Series([100.0 * (1 + rate) ** i for i in range(n)], index=idx)


# --------------------------------------------------------------------------- #
# Price/volume family
# --------------------------------------------------------------------------- #


def test_mom_12_2_skips_the_most_recent_month():
    frames = {"SLOW": _bar_frame(_geometric_close(0.001, 300)),
              "FAST": _bar_frame(_geometric_close(0.01, 300))}
    panel = _bar_panel(frames)
    as_of = frames["SLOW"].index[-1]
    scores = _score("mom_12_2", panel, as_of)
    # close[p-21]/close[p-252] - 1 = (1+r)^231 - 1
    assert math.isclose(scores["SLOW"], 1.001**231 - 1, rel_tol=1e-9)
    assert math.isclose(scores["FAST"], 1.01**231 - 1, rel_tol=1e-9)
    assert scores["FAST"] > scores["SLOW"]  # + sign: winners attractive


def test_mom_12_2_nan_under_253_closes():
    frames = {"AAA": _bar_frame(_geometric_close(0.001, 252))}
    panel = _bar_panel(frames)
    assert math.isnan(_score("mom_12_2", panel, frames["AAA"].index[-1])["AAA"])


def test_overnight_sums_63_log_gaps():
    idx = pd.date_range("2020-01-02", periods=70, freq="B", tz="UTC")
    close = pd.Series(100.0, index=idx)
    frames = {
        "HOT": _bar_frame(close, open_=pd.Series(100.0 * math.exp(0.002), index=idx)),
        "COLD": _bar_frame(close, open_=pd.Series(100.0 * math.exp(0.0005), index=idx)),
    }
    panel = _bar_panel(frames)
    scores = _score("overnight", panel, idx[-1])
    assert math.isclose(scores["HOT"], 63 * 0.002, rel_tol=1e-9)
    assert math.isclose(scores["COLD"], 63 * 0.0005, rel_tol=1e-9)
    assert scores["HOT"] > scores["COLD"]  # + sign: overnight persistence


def test_overnight_nan_under_64_bars():
    idx = pd.date_range("2020-01-02", periods=63, freq="B", tz="UTC")
    frames = {"AAA": _bar_frame(pd.Series(100.0, index=idx))}
    assert math.isnan(_score("overnight", _bar_panel(frames), idx[-1])["AAA"])


def test_park_vol_closed_form_and_negated():
    idx = pd.date_range("2020-01-02", periods=30, freq="B", tz="UTC")
    close = pd.Series(100.0, index=idx)

    def with_range(log_range: float) -> pd.DataFrame:
        return _bar_frame(
            close,
            high=pd.Series(100.0 * math.exp(log_range), index=idx),
            low=pd.Series(100.0, index=idx),
        )

    panel = _bar_panel({"WILD": with_range(0.04), "TAME": with_range(0.01)})
    scores = _score("park_vol", panel, idx[-1])
    expected_wild = math.sqrt(0.04**2 / (4 * math.log(2))) * math.sqrt(252)
    assert math.isclose(scores["WILD"], -expected_wild, rel_tol=1e-9)
    assert scores["TAME"] > scores["WILD"]  # - sign: quiet names attractive


def test_max5_is_negated_mean_of_top_returns():
    rets = [0.001] * 16 + [0.05, 0.04, 0.03, 0.02, 0.01]  # 21 returns
    closes = [100.0]
    for r in rets:
        closes.append(closes[-1] * (1 + r))
    idx = pd.date_range("2020-01-02", periods=22, freq="B", tz="UTC")
    frames = {"LOTTO": _bar_frame(pd.Series(closes, index=idx)),
              "STEADY": _bar_frame(_geometric_close(0.001, 22, start="2020-01-02"))}
    panel = _bar_panel(frames)
    scores = _score("max5", panel, idx[-1])
    assert math.isclose(scores["LOTTO"], -(0.05 + 0.04 + 0.03 + 0.02 + 0.01) / 5,
                        rel_tol=1e-9)
    assert scores["STEADY"] > scores["LOTTO"]  # - sign: lottery names penalized


def test_ivol_and_beta_signals_negate_the_precomputed_feature():
    idx = pd.date_range("2020-01-06", periods=3, freq="B", tz="UTC")
    feats = {"AAA": pd.DataFrame({"ivol": [0.2, 0.2, 0.2],
                                  "beta": [1.5, 1.5, 1.5]}, index=idx)}
    panel = PanelData(closes={}, features=feats, symbols=("AAA", "BBB"))
    as_of = idx[-1]
    ivol = _score("ivol", panel, as_of)
    assert ivol["AAA"] == -0.2          # - sign: idio-vol puzzle
    assert math.isnan(ivol["BBB"])      # no features -> NaN, dropped
    beta = _score("beta", panel, as_of)
    assert beta["AAA"] == -1.5          # - sign: betting-against-beta


def test_amihud_constant_dollar_volume_closed_form_and_floor():
    n = 260
    close = _geometric_close(0.01, n)
    volume = 1e6 / close                 # constant dollar volume 1e6
    frames = {"AAA": _bar_frame(close, volume=volume)}
    panel = _bar_panel(frames)
    got = _score("amihud", panel, close.index[-1])["AAA"]
    assert math.isclose(got, 0.01 / 1e6, rel_tol=1e-9)  # |r|/D every day
    short = _geometric_close(0.01, 120)
    thin = _bar_panel({"AAA": _bar_frame(short, volume=1e6 / short)})
    assert math.isnan(_score("amihud", thin, short.index[-1])["AAA"])  # <126 obs


def test_vol_trend_ratio_of_dollar_volume_means():
    idx = pd.date_range("2019-01-02", periods=252, freq="B", tz="UTC")
    close = pd.Series(100.0, index=idx)
    volume = pd.Series(1000.0, index=idx)
    volume.iloc[-21:] = 2000.0
    frames = {"AAA": _bar_frame(close, volume=volume)}
    got = _score("vol_trend", _bar_panel(frames), idx[-1])["AAA"]
    base = (231 * 100.0 * 1000.0 + 21 * 100.0 * 2000.0) / 252
    assert math.isclose(got, (100.0 * 2000.0) / base, rel_tol=1e-12)
    short = _bar_panel({"AAA": _bar_frame(pd.Series(100.0, index=idx[:200]))})
    assert math.isnan(_score("vol_trend", short, idx[199])["AAA"])


def test_div_yield_sums_trailing_dividends_and_never_fabricates_zero():
    idx = pd.date_range("2019-01-02", periods=260, freq="B", tz="UTC")
    div = pd.Series(0.0, index=idx)
    div.iloc[-100] = 1.0
    div.iloc[-10] = 0.5
    frames = {"PAYER": _bar_frame(pd.Series(50.0, index=idx), div_cash=div),
              "LEGACY": _bar_frame(pd.Series(50.0, index=idx),
                                   div_cash=pd.Series(np.nan, index=idx))}
    scores = _score("div_yield", _bar_panel(frames), idx[-1])
    assert math.isclose(scores["PAYER"], 1.5 / 50.0, rel_tol=1e-12)
    # Legacy narrow cache (all-NaN div_cash): NaN, never sum()'s skipna 0.0.
    assert math.isnan(scores["LEGACY"])


def test_price_volume_family_covers_every_panel_symbol():
    frames = {"AAA": _bar_frame(_geometric_close(0.001, 300)),
              "BBB": _bar_frame(_geometric_close(0.002, 300))}
    panel = _bar_panel(frames)
    as_of = frames["AAA"].index[-1]
    for name in ("mom_12_2", "overnight", "park_vol", "max5", "amihud",
                 "vol_trend", "div_yield"):
        scores = _score(name, panel, as_of)
        assert list(scores.index) == list(panel.symbols)
        assert scores.dtype == "float64"
```

In `tests/test_alphasearch_spec.py`, replace `test_registry_is_complete_with_correct_requirements`:

```python
def test_registry_is_complete_with_correct_requirements():
    assert len(SIGNALS) == 25  # 16 seeds + 9 Tier-1 price/volume
    options_family = {"vrp", "hedge", "excite", "atm_iv", "smile", "atm_spread"}
    fundamentals_family = {"gross_profitability", "earnings_yield", "book_to_market"}
    for name, spec in SIGNALS.items():
        assert spec.requires_options == (name in options_family)
        assert spec.requires_fundamentals == (name in fundamentals_family)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_alphasearch_tier1.py -q`
Expected: FAIL — `KeyError: 'mom_12_2'`.

- [ ] **Step 3: Implement the registrations in `spec.py`**

Add `import numpy as np` to the imports. Append after the existing price-family block (before the options block; helpers first):

```python
# --------------------------------------------------------------------------- #
# Tier-1 price/volume family (spec 2026-07-09 section 2). Bar metrics receive
# the PIT-truncated BAR_COLUMNS frame; the last row IS the as_of bar (the
# _trail convention). Windows/floors/signs are frozen pre-registration.
# --------------------------------------------------------------------------- #
_PARKINSON_DENOM = 4.0 * math.log(2.0)


def _bar_signal(metric: Callable[[pd.DataFrame], float]) -> SignalFn:
    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores = {symbol: metric(view.bars(symbol)) for symbol in view.symbols}
        return pd.Series(scores, dtype="float64")

    return fn


def _feature_signal(column: str, sign: float) -> SignalFn:
    """Score = sign * the precomputed rolling feature gathered as-of."""

    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores = {s: sign * view.feature(s, column) for s in view.symbols}
        return pd.Series(scores, dtype="float64")

    return fn


def _mom_12_2(closes: pd.Series) -> float:
    """Total return t-252 -> t-21 (skip the most recent month)."""
    p = len(closes) - 1
    if p - 252 < 0:
        return math.nan
    return float(closes.iloc[p - 21] / closes.iloc[p - 252] - 1)


def _overnight(bars: pd.DataFrame) -> float:
    """Sum of the last 63 overnight log-returns ln(open_t / close_{t-1})."""
    if len(bars) < 64:
        return math.nan
    opens = bars["open"].to_numpy()[-63:]
    prev_close = bars["close"].to_numpy()[-64:-1]
    return float(np.sum(np.log(opens / prev_close)))


def _park_vol(bars: pd.DataFrame) -> float:
    """Parkinson range vol over 21 bars: sqrt(mean(ln(H/L)^2/(4 ln 2))) * sqrt(252)."""
    if len(bars) < 21:
        return math.nan
    high = bars["high"].to_numpy()[-21:]
    low = bars["low"].to_numpy()[-21:]
    terms = np.log(high / low) ** 2 / _PARKINSON_DENOM
    return float(math.sqrt(terms.mean()) * math.sqrt(252))


def _max5(closes: pd.Series) -> float:
    """Mean of the 5 largest daily returns among the last 21."""
    if len(closes) < 22:
        return math.nan
    rets = closes.iloc[-22:].pct_change().dropna().to_numpy()
    return float(np.sort(rets)[-5:].mean())


def _amihud(bars: pd.DataFrame) -> float:
    """Mean |ret| / dollar volume over the last 252 bars; min 126 valid terms
    (non-positive dollar volume or NaN return terms are skipped, never 0)."""
    window = bars.iloc[-252:]
    rets = window["close"].pct_change().to_numpy()
    dollar = (window["close"] * window["volume"]).to_numpy()
    valid = ~np.isnan(rets) & ~np.isnan(dollar) & (dollar > 0)
    if valid.sum() < 126:
        return math.nan
    return float(np.mean(np.abs(rets[valid]) / dollar[valid]))


def _vol_trend(bars: pd.DataFrame) -> float:
    """Mean dollar volume over 21 bars / mean over 252 bars (strict window)."""
    if len(bars) < 252:
        return math.nan
    dollar = (bars["close"] * bars["volume"]).to_numpy()[-252:]
    base = dollar.mean()
    if not base > 0:  # NaN or zero baseline both land here
        return math.nan
    return float(dollar[-21:].mean() / base)


def _div_yield(bars: pd.DataFrame) -> float:
    """Trailing 252-bar cash dividends / last close. min_count=1 keeps an
    all-NaN div_cash column (legacy narrow cache) NaN instead of sum()'s
    skipna zero -- a cache without dividend data must not claim 'no
    dividends' (that would fabricate a 0-yield cross-section)."""
    if len(bars) < 252:
        return math.nan
    paid = bars["div_cash"].iloc[-252:].sum(min_count=1)
    close = float(bars["close"].iloc[-1])
    if pd.isna(paid) or not close > 0:
        return math.nan
    return float(paid / close)


# Classic UMD with the skip-month that avoids short-term reversal
# contamination (Jegadeesh-Titman).
_register("mom_12_2", _price_signal(_mom_12_2))
# The overnight component of returns persists (Lou-Polk-Skouras).
_register("overnight", _bar_signal(_overnight))
# Low-vol anomaly on the Parkinson range estimator -> negate.
_register("park_vol", _bar_signal(lambda b: -_park_vol(b)))
# Idiosyncratic-vol puzzle (Ang-Hodrick-Xing-Zhang) -> negate.
_register("ivol", _feature_signal("ivol", -1.0))
# Lottery demand: extreme-day chasers overpay (Bali-Cakici-Whitelaw) -> negate.
_register("max5", _price_signal(lambda c: -_max5(c)))
# Betting-against-beta (Frazzini-Pedersen) -> negate.
_register("beta", _feature_signal("beta", -1.0))
# Illiquidity premium (Amihud): harder-to-trade names pay more.
_register("amihud", _bar_signal(_amihud))
# High-volume return premium (Gervais-Kaniel-Mingelgrin).
_register("vol_trend", _bar_signal(_vol_trend))
# Income/value tilt: cash actually paid out over the trailing year.
_register("div_yield", _bar_signal(_div_yield))
```

- [ ] **Step 4: Run the unit tests**

Run: `uv run pytest tests/test_alphasearch_tier1.py tests/test_alphasearch_spec.py -q`
Expected: PASS.

- [ ] **Step 5: Rewrite `tests/test_alphasearch_lookahead.py`**

Replace the whole file:

```python
"""THE no-look-ahead guarantee (spec section 7): perturb every RAW store
strictly after a cutoff date T -- bars (all columns), options cells,
fundamentals rows, factor rows -- REASSEMBLE the panel (so derived state,
incl. the precomputed ivol/beta features, is recomputed from the corrupted
inputs), and assert every registered signal's scores at <= T are
bit-identical. Iterates SIGNALS, so any future signal is automatically
covered; the anti-vacuity guard proves each signal actually produces values
pre-cutoff on this fixture."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from alphasearch_helpers import assemble_panel, make_factors, make_panel
from trading.alphasearch.panel import PanelData
from trading.alphasearch.spec import SIGNALS

START = pd.Timestamp("2020-01-01", tz="UTC")
CUTOFF = pd.Timestamp("2020-03-15", tz="UTC")


def _long_panel() -> PanelData:
    """~420 bars from 2019-01-02: enough pre-cutoff history that every
    registered signal (incl. beta's 126-obs floor, mom_12_2's 253 closes,
    and the 300-day YoY filing rule) produces real values at decision dates
    <= CUTOFF."""
    return make_panel(
        start="2019-01-02", periods=420,
        factors=make_factors(start="2018-12-03", periods=440),
    )


def _perturb_after(panel: PanelData, cutoff: pd.Timestamp) -> PanelData:
    """Corrupt every raw store strictly after cutoff, then reassemble."""
    bars: dict[str, pd.DataFrame] = {}
    for sym, frame in panel.bars.items():
        f = frame.copy()
        late = f.index > cutoff
        f.loc[late] = f.loc[late] * 3.7 + 11.0
        bars[sym] = f
    options: dict[str, pd.DataFrame] = {}
    for sym, frame in panel.options.items():
        f = frame.copy()
        f.loc[f.index > cutoff] = 9.9
        options[sym] = f
    fundamentals: dict[str, pd.DataFrame] = {}
    for sym, frame in panel.fundamentals.items():
        f = frame.copy()
        late = f.index > cutoff
        f.loc[late] = f.loc[late] * 5.0 + 1.0
        fundamentals[sym] = f
    factors = panel.factors.copy()
    late = factors.index > cutoff
    factors.loc[late] = factors.loc[late] * 3.0 + 0.001
    return assemble_panel(bars, options, fundamentals, factors)


def test_fixture_actually_has_data_after_the_cutoff():
    # Guard against a vacuous test: every store must carry post-cutoff rows.
    panel = _long_panel()
    assert any((f.index > CUTOFF).any() for f in panel.bars.values())
    assert any((f.index > CUTOFF).any() for f in panel.options.values())
    assert any((f.index > CUTOFF).any() for f in panel.fundamentals.values())
    assert (panel.factors.index > CUTOFF).any()


def test_every_signal_scores_real_values_at_the_last_pre_cutoff_date():
    # Anti-vacuity: an all-NaN signal would "pass" the perturbation test
    # without testing anything. Auto-extends as families register.
    panel = _long_panel()
    dates = list(panel.decision_dates(START, CUTOFF))
    as_of = dates[-1]
    for name, spec in sorted(SIGNALS.items()):
        scores = spec.fn(panel.view(as_of), as_of)
        assert scores.notna().any(), f"{name} is all-NaN at {as_of.date()}"


def test_no_registered_signal_can_see_past_as_of():
    assert SIGNALS, "registry unexpectedly empty"  # never pass vacuously
    panel = _long_panel()
    dates = list(panel.decision_dates(START, CUTOFF))
    assert len(dates) >= 3  # several decision months at or before the cutoff
    perturbed = _perturb_after(panel, CUTOFF)
    for name, spec in sorted(SIGNALS.items()):
        for as_of in dates:
            before = spec.fn(panel.view(as_of), as_of)
            after = spec.fn(perturbed.view(as_of), as_of)
            pdt.assert_series_equal(
                before, after, check_exact=True,
                obj=f"{name} @ {as_of.date().isoformat()}",
            )
```

- [ ] **Step 6: Run the lookahead test**

Run: `uv run pytest tests/test_alphasearch_lookahead.py -q`
Expected: PASS (3 tests). If `test_every_signal_...` fails for a seed signal, the fixture is wrong, NOT the signal — fix the fixture.

- [ ] **Step 7: Full suite + ruff, commit**

Run: `uv run pytest -q && uv run ruff check src tests scripts`
Expected: PASS (note: `test_signals_none_runs_the_full_registry` now counts 25 — it asserts against `len(SIGNALS)` so no edit is needed; `vol_trend`/`div_yield`/`mom_12_2`/`beta` journal SortError error trials on the 130-bar default fixture, which is correct behavior).

```bash
git add src/trading/alphasearch/spec.py tests/test_alphasearch_tier1.py \
    tests/test_alphasearch_spec.py tests/test_alphasearch_lookahead.py
git commit -m "Register Tier-1 price/volume family; perturb raw stores in lookahead test [AI]"
```

---

### Task 4: Option-volume infrastructure — `opt_dollar_vol` metric, `has_option_volume`, `option_row_prior`, `requires_option_volume` refusal

**Files:**
- Modify: `src/trading/alphasearch/panel.py`, `src/trading/alphasearch/spec.py` (SignalSpec only), `src/trading/alphasearch/sweep.py` (`_check_universe_supports`)
- Modify: `tests/alphasearch_helpers.py` (`make_cell` gains `with_volume` + otm-leg mids; `assemble_panel`/`make_panel` gain the flag)
- Test: `tests/test_alphasearch_panel.py`, `tests/test_alphasearch_sweep.py`

**Interfaces:**
- Produces: `cell_metrics` dict gains `"opt_dollar_vol"` (Σ over legs of volume·100·mid where BOTH present; NaN when no leg qualifies); `"opt_dollar_vol"` appended to `OPTION_COLUMNS`; `cells_have_volume(cells) -> bool`; `load_options(samples) -> tuple[dict, int, bool]` (frames, corrupt, has_volume); `PanelData.has_option_volume: bool = False` + `PanelView.has_option_volume` property; `MAX_PRIOR_OPTION_AGE_DAYS = 45`; `PanelView.option_row_prior(symbol, max_age_days=45) -> pd.Series | None`; `SignalSpec.requires_option_volume: bool = False` (+ `_register` kwarg); `_check_universe_supports` refusal naming the mid-cap gather and the `--signals` workaround; helpers `make_cell(..., with_volume=True)` (otm legs now carry mids 2.0/1.5), `assemble_panel(..., has_option_volume=False)`, `make_panel(..., with_option_volume=True)`.
- Consumes: Task 1 panel structure.

**Semantics frozen here:** `option_row_prior` returns the most recent cell strictly OLDER than the current cell (the one `option_row`'s position arithmetic selects — spec §3.5), never a future one; None when there is no current cell, no older cell, or the older cell is more than `max_age_days` calendar days before `as_of` (staleness measured from `as_of`, mirroring `MAX_OPTION_AGE_DAYS`; the 45-day cap is spec §2's `iv_change`/`dskew` staleness rule). Note the spec writes the signature as `option_row_prior(symbol, as_of, max_age_days=45)`; `PanelView` methods carry `as_of` on the view itself (like `option_row`), so the implemented signature drops the redundant parameter.

- [ ] **Step 1: Update `tests/alphasearch_helpers.py`**

Replace `make_cell` (otm legs gain mids so `opt_dollar_vol` has a hand value; `with_volume=False` models the largecap gather, whose cells carry NO volume keys at all):

```python
def make_cell(
    symbol: str,
    date: str,
    *,
    atm_iv: float = 0.30,
    put_iv: float = 0.34,
    call_iv: float = 0.28,
    skew_put_atm: float = 0.05,
    skew_put_call: float = 0.02,
    with_volume: bool = True,
) -> dict:
    """One samples.jsonl-shaped options cell with all three legs present.
    with_volume=False reproduces the largecap gather (no volume keys on any
    leg); True reproduces the mid-cap gather (volumes 100/50/25)."""
    contracts = [
        {"role": "atm", "bid": 4.0, "ask": 4.2, "mid": 4.1, "iv": atm_iv},
        {"role": "otm_put", "mid": 2.0, "iv": put_iv},
        {"role": "otm_call", "mid": 1.5, "iv": call_iv},
    ]
    if with_volume:
        for contract, volume in zip(contracts, (100, 50, 25), strict=True):
            contract["volume"] = volume
    return {
        "symbol": symbol,
        "decision_date": date,
        "skew_put_atm": skew_put_atm,
        "skew_put_call": skew_put_call,
        "contracts": contracts,
    }
```

`assemble_panel` gains a keyword flag, passed through to `PanelData(...)` — new signature and final line:

```python
def assemble_panel(
    bars: dict[str, pd.DataFrame],
    options: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    factors: pd.DataFrame,
    *,
    has_option_volume: bool = False,
) -> PanelData:
```

with `has_option_volume=has_option_volume,` added to its `PanelData(...)` call.

`make_panel` gains `with_option_volume: bool = True` (insert after `with_fundamentals`), builds cells with `make_cell(..., with_volume=with_option_volume)` (add the kwarg to the existing `make_cell(...)` call in its options loop), and its return becomes:

```python
    return assemble_panel(
        bars, options, fundamentals, factors,
        has_option_volume=with_options and with_option_volume,
    )
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_alphasearch_panel.py` (extend imports as needed: `math` and `json` at the top; `make_cell` from `alphasearch_helpers`; `cell_metrics` and `options_from_cells` in the `trading.alphasearch.panel` import block — check which are already present first):

```python
def test_cell_metrics_opt_dollar_vol_sums_legs_with_both_volume_and_mid():
    metrics = cell_metrics(make_cell("AAA", "2020-01-06"))
    want = 100 * 100 * 4.1 + 50 * 100 * 2.0 + 25 * 100 * 1.5  # 54750
    assert math.isclose(metrics["opt_dollar_vol"], want, rel_tol=1e-12)
    no_volume = cell_metrics(make_cell("AAA", "2020-01-06", with_volume=False))
    assert np.isnan(no_volume["opt_dollar_vol"])  # no qualifying leg -> NaN
    partial = make_cell("AAA", "2020-01-06")
    del partial["contracts"][1]["volume"]  # put leg loses volume
    got = cell_metrics(partial)["opt_dollar_vol"]
    assert math.isclose(got, 100 * 100 * 4.1 + 25 * 100 * 1.5, rel_tol=1e-12)


def test_load_options_reports_leg_volume_presence(tmp_path):
    p1 = tmp_path / "with.jsonl"
    p1.write_text(json.dumps(make_cell("AAA", "2020-01-06")) + "\n")
    p2 = tmp_path / "without.jsonl"
    p2.write_text(json.dumps(make_cell("AAA", "2020-01-06", with_volume=False)) + "\n")
    _frames, _corrupt, has_volume = load_options(p1)
    assert has_volume is True
    _frames, _corrupt, has_volume = load_options(p2)
    assert has_volume is False


def test_build_panel_threads_has_option_volume(tmp_path):
    idx = pd.date_range("2020-01-02", periods=3, freq="B", tz="UTC")
    pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
        index=idx,
    ).to_parquet(tmp_path / "AAA.parquet")
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps(make_cell("AAA", "2020-01-02")) + "\n")
    panel = build_panel(tmp_path, samples, None)
    assert panel.has_option_volume is True
    assert panel.view(idx[0]).has_option_volume is True
    bare = build_panel(tmp_path, None, None, symbols=("AAA",))
    assert bare.has_option_volume is False


def _prior_panel(dates: list[str]) -> PanelData:
    cells = [make_cell("AAA", d) for d in dates]
    return PanelData(closes={}, options=options_from_cells(cells), symbols=("AAA",))


def test_option_row_prior_returns_the_cell_strictly_older_than_current():
    panel = _prior_panel(["2020-01-06", "2020-02-03", "2020-03-02"])
    as_of = pd.Timestamp("2020-03-02", tz="UTC")
    prior = panel.view(as_of).option_row_prior("AAA")
    assert prior is not None
    assert prior.name == pd.Timestamp("2020-02-03", tz="UTC")
    # Current cell = Feb when as_of sits between Feb and Mar cells.
    mid = pd.Timestamp("2020-02-05", tz="UTC")
    prior = panel.view(mid).option_row_prior("AAA")
    assert prior.name == pd.Timestamp("2020-01-06", tz="UTC")


def test_option_row_prior_none_without_an_older_cell_or_when_stale():
    single = _prior_panel(["2020-01-06"])
    as_of = pd.Timestamp("2020-01-06", tz="UTC")
    assert single.view(as_of).option_row_prior("AAA") is None
    assert single.view(as_of).option_row_prior("NOPE") is None
    # Boundary: exactly 45 calendar days before as_of is FRESH; 46 is stale.
    fresh = _prior_panel(["2020-01-17", "2020-03-02"])   # Jan 17 + 45d = Mar 2
    at = pd.Timestamp("2020-03-02", tz="UTC")
    assert fresh.view(at).option_row_prior("AAA") is not None
    stale = _prior_panel(["2020-01-16", "2020-03-02"])   # 46 days -> stale
    assert stale.view(at).option_row_prior("AAA") is None
```

Update the existing unpacking at `tests/test_alphasearch_panel.py` (`test_load_options_skips_and_counts_corrupt_lines`): `frames, corrupt = load_options(path)` → `frames, corrupt, _has_volume = load_options(path)`.

Append to `tests/test_alphasearch_sweep.py` (import `SignalSpec` from `trading.alphasearch.spec`):

```python
def test_option_volume_signal_refused_on_a_volume_less_universe(tmp_path):
    # requires_option_volume mirrors requires_options: an assembly-time
    # refusal, never silent fake-log(1/1)=0 trials (spec section 2, options
    # family constraint).
    journal = trials_journal(tmp_path / "journal")
    fake = SignalSpec("fake_vol", SIGNALS["hedge"].fn,
                      requires_options=True, requires_option_volume=True)
    largecap_like = make_panel(with_option_volume=False)
    with pytest.raises(SweepError) as excinfo:
        run_sweep(_universe(tmp_path), journal, make_factors(), ts="t1",
                  signals={"fake_vol": fake, "mom21": SIGNALS["mom21"]},
                  window=WINDOW, panel_factory=lambda _u, _f: largecap_like)
    message = str(excinfo.value)
    assert "option volume" in message
    assert "--signals" in message                    # actionable workaround
    assert list(journal.events()) == []              # all-or-nothing: no trials
    # And the same signal RUNS where cells carry leg volume.
    midcap_like = make_panel()
    rows, n = run_sweep(_universe(tmp_path), journal, make_factors(), ts="t2",
                        signals={"fake_vol": fake}, window=WINDOW,
                        panel_factory=lambda _u, _f: midcap_like)
    assert n == 1 and rows[0].error is None
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_alphasearch_panel.py tests/test_alphasearch_sweep.py -q`
Expected: FAIL — `KeyError: 'opt_dollar_vol'`, `TypeError: make_cell() got an unexpected keyword argument 'with_volume'` (helpers step 1 fixes that one), unpacking errors.

- [ ] **Step 4: Implement in `panel.py`**

In `OPTION_COLUMNS`, append `"opt_dollar_vol"`:

```python
OPTION_COLUMNS = [
    "hedge", "excite", "atm_iv", "otm_put_iv", "otm_call_iv",
    "smile", "cp_vol", "wing_vol", "tot_vol", "atm_spread", "opt_dollar_vol",
]
```

In `cell_metrics`, add before the `return`:

```python
    def leg_dollar(role):
        contract = d.get(role, {})
        volume, mid = contract.get("volume"), contract.get("mid")
        if volume is None or mid is None:
            return None
        return float(volume) * 100.0 * float(mid)

    dollars = [x for x in (leg_dollar(r) for r in ("atm", "otm_put", "otm_call"))
               if x is not None]
```

and to the returned dict: `"opt_dollar_vol": (sum(dollars) if dollars else np.nan),` with the comment `# Johnson-So O/S numerator: only legs carrying BOTH volume and mid count; a cell with neither is missing, never $0.`

Add module-level:

```python
MAX_PRIOR_OPTION_AGE_DAYS = 45  # spec section 2: iv_change/dskew staleness cap


def cells_have_volume(cells: Iterable[dict]) -> bool:
    """True when ANY gathered leg carries a volume field. Leg volume ships
    only in the mid-cap gather; the sweep refuses the option-volume family
    on universes without it (a volume-less cell would otherwise score a
    fabricated log(1/1)=0, not NaN)."""
    return any(
        contract.get("volume") is not None
        for cell in cells
        for contract in cell.get("contracts", [])
    )
```

Change `load_options` to return the flag (update its docstring's first line accordingly):

```python
    return options_from_cells(cells), corrupt, cells_have_volume(cells)
```

In `PanelData`, append field: `has_option_volume: bool = False` (after `features`). In `PanelView`, add:

```python
    @property
    def has_option_volume(self) -> bool:
        return self._panel.has_option_volume

    def option_row_prior(
        self, symbol: str, max_age_days: int = MAX_PRIOR_OPTION_AGE_DAYS
    ) -> pd.Series | None:
        """The most recent cell strictly OLDER than the current cell (the
        one option_row's position arithmetic selects; spec section 3.5),
        never a future one. None when there is no current cell, no older
        cell, or the older cell is more than max_age_days calendar days
        before as_of -- staleness measured from as_of, mirroring
        MAX_OPTION_AGE_DAYS. The 45-day default freezes iv_change/dskew's
        'prior month' definition (spec section 2)."""
        frame = self._panel.options.get(symbol)
        if frame is None or frame.empty:
            return None
        pos = int(frame.index.searchsorted(self.as_of, side="right")) - 1
        if pos < 1:
            return None
        if (self.as_of - frame.index[pos - 1]).days > max_age_days:
            return None
        return frame.iloc[pos - 1]
```

In `build_panel`, change the options line and thread the flag:

```python
    options, corrupt, has_volume = (
        load_options(samples) if samples is not None else ({}, 0, False)
    )
```

and add `has_option_volume=has_volume,` to the returned `PanelData(...)`.

- [ ] **Step 5: Implement the `SignalSpec` flag in `spec.py`**

```python
@dataclass(frozen=True)
class SignalSpec:
    name: str
    fn: SignalFn
    requires_options: bool = False
    requires_fundamentals: bool = False
    # Leg volume exists only in the mid-cap gather; signals reading it are
    # refused at sweep assembly on volume-less universes (spec section 2).
    requires_option_volume: bool = False


def _register(
    name: str,
    fn: SignalFn,
    *,
    requires_options: bool = False,
    requires_fundamentals: bool = False,
    requires_option_volume: bool = False,
) -> None:
    SIGNALS[name] = SignalSpec(
        name,
        fn,
        requires_options=requires_options,
        requires_fundamentals=requires_fundamentals,
        requires_option_volume=requires_option_volume,
    )
```

- [ ] **Step 6: Implement the refusal in `sweep.py`**

Append to `_check_universe_supports`:

```python
    if spec.requires_option_volume and not panel.has_option_volume:
        raise SweepError(
            f"signal {spec.name!r} requires per-leg option volume; universe "
            f"{universe!r} cells carry none (leg volume ships only in the "
            "mid-cap gather -- see data/options-iv/samples-midcap.jsonl). "
            "Re-gather this universe's cells with a volume-carrying "
            "`scripts/gather_options_iv.py` run, or work around it by "
            "passing --signals without the option-volume family"
        )
```

- [ ] **Step 7: Run the tests, full suite, ruff**

Run: `uv run pytest tests/test_alphasearch_panel.py tests/test_alphasearch_sweep.py -q && uv run pytest -q && uv run ruff check src tests scripts`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/trading/alphasearch/panel.py src/trading/alphasearch/spec.py \
    src/trading/alphasearch/sweep.py tests/alphasearch_helpers.py \
    tests/test_alphasearch_panel.py tests/test_alphasearch_sweep.py
git commit -m "Add option-volume capability flag, prior-cell accessor, O/S dollar metric [AI]"
```

---

### Task 5: Register the options family (5 signals)

**Files:**
- Modify: `src/trading/alphasearch/spec.py`
- Modify: `tests/test_alphasearch_spec.py` (registry count)
- Test: `tests/test_alphasearch_tier1.py`

**Interfaces:**
- Produces: registrations `cp_vol` (+, requires_options + requires_option_volume), `osv` (−, requires_options + requires_option_volume), `otm_put_iv` (−, requires_options), `iv_change` (−, requires_options), `dskew` (−, requires_options). Registry size 30.
- Consumes: Task 4 (`opt_dollar_vol` column, `option_row_prior`, flags), `PanelView.bars` (Task 1), existing `_option_signal` and the committed `cp_vol`/`otm_put_iv`/`hedge` cell columns.

**Frozen definitions (spec §2), with two resolutions to record:** (1) `cp_vol` = log(1+call leg volume) − log(1+put leg volume). The gather's ATM leg IS a call (`options_gather._ROLE_IS_CALL["atm"] = True`), so the committed `cp_vol` cell column — `log((vol(atm)+vol(otm_call)+1)/(vol(otm_put)+1))` — is exactly this identity with call-side volume = ATM call + OTM call; the signal reads that committed column. (2) `osv` = cell option dollar volume (`opt_dollar_vol`) / decision-day stock dollar volume, where stock dollar volume = close·volume of the last bar at or before `as_of`. Registered NEGATED (Johnson-So: high O/S predicts underperformance). `iv_change` = atm_iv(current cell) − atm_iv(prior cell), `dskew` = skew_put_atm(current) − skew_put_atm(prior) (the `hedge` column), both via `option_row`/`option_row_prior`, both NEGATED, NaN when either cell is missing/stale.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_alphasearch_tier1.py` (add imports `options_from_cells` from `trading.alphasearch.panel` and `make_cell` from `alphasearch_helpers`):

```python
# --------------------------------------------------------------------------- #
# Options family
# --------------------------------------------------------------------------- #


def _options_tier1_panel(prior_age_days: int = 28):
    idx = pd.date_range("2020-01-02", periods=60, freq="B", tz="UTC")
    as_of = idx[-1]
    prior = (as_of - pd.Timedelta(prior_age_days, unit="D")).date().isoformat()
    current = as_of.date().isoformat()
    cells = [
        make_cell("AAA", prior, atm_iv=0.24, skew_put_atm=0.02),
        make_cell("AAA", current, atm_iv=0.30, skew_put_atm=0.05),
        make_cell("BBB", current, atm_iv=0.50, skew_put_atm=0.10),  # no prior
    ]
    bars = {s: _bar_frame(pd.Series(100.0, index=idx), volume=500.0)
            for s in ("AAA", "BBB")}
    panel = PanelData(
        closes={s: f["close"] for s, f in bars.items()}, bars=bars,
        options=options_from_cells(cells), symbols=("AAA", "BBB"),
        has_option_volume=True,
    )
    return panel, as_of


def test_cp_vol_reads_the_committed_call_minus_put_log_volume():
    panel, as_of = _options_tier1_panel()
    scores = _score("cp_vol", panel, as_of)
    # ATM leg is a call: call side = atm(100) + otm_call(25); put side = 50.
    assert math.isclose(scores["AAA"], math.log(126 / 51), rel_tol=1e-12)
    assert scores["AAA"] > 0  # + sign: informed call demand attractive


def test_osv_is_negated_option_to_stock_dollar_volume():
    panel, as_of = _options_tier1_panel()
    scores = _score("osv", panel, as_of)
    opt_dollar = 100 * 100 * 4.1 + 50 * 100 * 2.0 + 25 * 100 * 1.5  # 54750
    assert math.isclose(scores["AAA"], -(opt_dollar / (100.0 * 500.0)),
                        rel_tol=1e-12)


def test_otm_put_iv_is_negated_smirk_level():
    panel, as_of = _options_tier1_panel()
    scores = _score("otm_put_iv", panel, as_of)
    assert scores["AAA"] == -0.34
    assert scores["AAA"] > scores["BBB"]  # steeper smirk = less attractive


def test_iv_change_and_dskew_are_negated_innovations_nan_without_prior():
    panel, as_of = _options_tier1_panel()
    iv_change = _score("iv_change", panel, as_of)
    assert math.isclose(iv_change["AAA"], -(0.30 - 0.24), rel_tol=1e-12)
    assert math.isnan(iv_change["BBB"])  # no prior cell -> NaN
    dskew = _score("dskew", panel, as_of)
    assert math.isclose(dskew["AAA"], -(0.05 - 0.02), rel_tol=1e-12)
    assert math.isnan(dskew["BBB"])


def test_innovations_nan_when_the_prior_cell_is_stale():
    panel, as_of = _options_tier1_panel(prior_age_days=50)  # > 45d cap
    assert math.isnan(_score("iv_change", panel, as_of)["AAA"])
    assert math.isnan(_score("dskew", panel, as_of)["AAA"])


def test_options_family_nan_without_cells():
    idx = pd.date_range("2020-01-02", periods=60, freq="B", tz="UTC")
    bars = {"AAA": _bar_frame(pd.Series(100.0, index=idx), volume=500.0)}
    bare = _bar_panel(bars)
    for name in ("cp_vol", "osv", "otm_put_iv", "iv_change", "dskew"):
        assert _score(name, bare, idx[-1]).isna().all(), name
```

In `tests/test_alphasearch_spec.py`, update the registry test:

```python
def test_registry_is_complete_with_correct_requirements():
    assert len(SIGNALS) == 30  # 16 seeds + 9 price/volume + 5 options
    options_family = {"vrp", "hedge", "excite", "atm_iv", "smile", "atm_spread",
                      "cp_vol", "osv", "otm_put_iv", "iv_change", "dskew"}
    volume_family = {"cp_vol", "osv"}
    for name, spec in SIGNALS.items():
        assert spec.requires_options == (name in options_family)
        assert spec.requires_option_volume == (name in volume_family)
        assert spec.requires_fundamentals == (
            name in {"gross_profitability", "earnings_yield", "book_to_market"}
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_alphasearch_tier1.py -q`
Expected: FAIL — `KeyError: 'cp_vol'` (the SIGNALS registry has the cell COLUMN but no registered signal of that name yet).

- [ ] **Step 3: Implement in `spec.py`**

Append after the existing options-family block:

```python
# --------------------------------------------------------------------------- #
# Tier-1 options family (spec 2026-07-09 section 2). cp_vol/osv read per-leg
# volume, which only the mid-cap gather carries -> requires_option_volume
# refuses them elsewhere (fake log(1/1)=0 cross-sections must never trial).
# --------------------------------------------------------------------------- #
def _osv(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    """Option/stock dollar-volume ratio: the cell's opt_dollar_vol over the
    decision-day stock dollar volume (close*volume of the last bar <= as_of),
    one cell vs one day. NEGATED at registration below."""
    scores: dict[str, float] = {}
    for symbol in view.symbols:
        row = view.option_row(symbol)
        score = math.nan
        if row is not None:
            bars = view.bars(symbol)
            if len(bars):
                stock_dollar = float(bars["close"].iloc[-1] * bars["volume"].iloc[-1])
                opt_dollar = float(row["opt_dollar_vol"])
                if stock_dollar > 0 and not math.isnan(opt_dollar):
                    score = -(opt_dollar / stock_dollar)
        scores[symbol] = score
    return pd.Series(scores, dtype="float64")


def _option_innovation(column: str, sign: float) -> SignalFn:
    """sign * (current cell's column - prior cell's column); NaN when either
    cell is missing or the prior is stale (option_row_prior's 45d cap)."""

    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol in view.symbols:
            row = view.option_row(symbol)
            prior = view.option_row_prior(symbol)
            if row is None or prior is None:
                scores[symbol] = math.nan
            else:
                scores[symbol] = sign * (float(row[column]) - float(prior[column]))
        return pd.Series(scores, dtype="float64")

    return fn


# Informed call demand predicts positive returns (Pan-Poteshman). The cell's
# committed cp_vol column is log(1+call volume) - log(1+put volume) with the
# ATM leg counted as the call it is.
_register("cp_vol", _option_signal("cp_vol", +1.0),
          requires_options=True, requires_option_volume=True)
# High option/stock volume marks informed (mostly bearish) positioning
# (Johnson-So) -> negate.
_register("osv", _osv, requires_options=True, requires_option_volume=True)
# Steep OTM-put smirk predicts negative returns (Xing-Zhang-Zhao) -> negate.
_register("otm_put_iv", _option_signal("otm_put_iv", -1.0), requires_options=True)
# Rising implied vol = rising perceived risk (An-Ang-Bali-Cakici) -> negate.
_register("iv_change", _option_innovation("atm_iv", -1.0), requires_options=True)
# A steepening put smirk is bearish, consistent with `hedge`'s sign -> negate.
_register("dskew", _option_innovation("hedge", -1.0), requires_options=True)
```

- [ ] **Step 4: Run the tests, full suite, ruff**

Run: `uv run pytest tests/test_alphasearch_tier1.py tests/test_alphasearch_spec.py -q && uv run pytest -q && uv run ruff check src tests scripts`
Expected: PASS (the lookahead anti-vacuity guard now also covers the five new names on the long fixture: monthly cells give every symbol a ≤45d prior from February on).

- [ ] **Step 5: Commit**

```bash
git add src/trading/alphasearch/spec.py tests/test_alphasearch_tier1.py \
    tests/test_alphasearch_spec.py
git commit -m "Register Tier-1 options family (cp_vol, osv, otm_put_iv, iv_change, dskew) [AI]"
```

---

### Task 6: Fundamentals family (5 signals, 300-day YoY filing rule) + segments §3.4 amendment

**Files:**
- Modify: `src/trading/alphasearch/panel.py` (`fundamentals_row_prior`), `src/trading/alphasearch/spec.py`, `src/trading/alphasearch/segments.py`, `src/trading/cli.py` (hint wording)
- Modify: `tests/test_alphasearch_spec.py` (registry count)
- Test: `tests/test_alphasearch_panel.py`, `tests/test_alphasearch_tier1.py`, `tests/test_alphasearch_segments.py`

**Interfaces:**
- Produces: `MIN_YOY_AGE_DAYS = 300`; `PanelView.fundamentals_row_prior(symbol, min_age_days=300) -> pd.Series | None`; registrations `asset_growth` (−), `net_issuance` (−, split-adjusted), `roa` (+), `droa` (+), `rev_growth` (+), all `requires_fundamentals=True`. Registry size 35. Segments: deep pools carry `fundamentals_dir` when the store dir exists.
- Consumes: `fundamentals_row` (existing), `PanelView.bars` (split_factor, Task 1), the store's `SERIES_COLUMNS` (`assets`, `ttm_net_income`, `shares_outstanding`, `revenue_ttm` — see `src/trading/fundamentals/metrics.py`).

**Frozen YoY rule (spec §2):** "one-year-prior filing" = the latest filing FILED at least 300 calendar days before the CURRENT filing's filed date (current = latest filed ≤ as_of, exactly `fundamentals_row`'s cut). Exactly-300-days qualifies ("at least"). If none exists → NaN, dropped, never imputed, never reach further back on NaN values. `net_issuance` split adjustment: comparable prior shares = prior shares × Π split_factor over bar dates in (prior_filed, current_filed]; ANY NaN split_factor in that window → NaN (a legacy narrow cache cannot claim "no splits").

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_alphasearch_panel.py`:

```python
def test_fundamentals_row_prior_300_day_boundary():
    base = pd.Timestamp("2019-01-10", tz="UTC")
    as_of = pd.Timestamp("2020-01-15", tz="UTC")
    ok = pd.DataFrame(
        {"assets": [100.0, 110.0]},
        index=pd.DatetimeIndex([base, base + pd.Timedelta(300, unit="D")]),
    )
    panel = PanelData(closes={}, fundamentals={"AAA": ok}, symbols=("AAA",))
    prior = panel.view(as_of).fundamentals_row_prior("AAA")
    assert prior is not None and prior["assets"] == 100.0  # exactly 300d: counts
    close_call = pd.DataFrame(
        {"assets": [100.0, 110.0]},
        index=pd.DatetimeIndex([base, base + pd.Timedelta(299, unit="D")]),
    )
    panel299 = PanelData(closes={}, fundamentals={"AAA": close_call}, symbols=("AAA",))
    assert panel299.view(as_of).fundamentals_row_prior("AAA") is None  # 299d: no
    # The rule anchors on the CURRENT filing: before the second filing is
    # visible, the first IS current and has no prior.
    early = base + pd.Timedelta(10, unit="D")
    assert panel.view(early).fundamentals_row_prior("AAA") is None
    assert panel.view(as_of).fundamentals_row_prior("NOPE") is None
```

Append to `tests/test_alphasearch_tier1.py`:

```python
# --------------------------------------------------------------------------- #
# Fundamentals family (300-calendar-day YoY filing rule)
# --------------------------------------------------------------------------- #

FILED_2019 = pd.Timestamp("2019-01-10", tz="UTC")
FILED_2019_Q4 = pd.Timestamp("2019-11-30", tz="UTC")  # 324d later: YoY-eligible


def _fund_frame(values_by_column: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(
        values_by_column, index=pd.DatetimeIndex([FILED_2019, FILED_2019_Q4])
    )


def _fund_panel(fundamentals: dict[str, pd.DataFrame],
                split_factor: pd.Series | float = 1.0) -> tuple[PanelData, pd.Timestamp]:
    idx = pd.date_range("2019-01-02", periods=300, freq="B", tz="UTC")
    bars = {s: _bar_frame(pd.Series(100.0, index=idx), split_factor=split_factor)
            for s in fundamentals}
    panel = PanelData(
        closes={s: f["close"] for s, f in bars.items()}, bars=bars,
        fundamentals=fundamentals, symbols=tuple(sorted(fundamentals)),
    )
    return panel, pd.Timestamp("2020-01-15", tz="UTC")


def test_asset_growth_rev_growth_roa_droa_hand_values():
    fund = {
        "AAA": _fund_frame({
            "assets": [100.0, 110.0],
            "revenue_ttm": [200.0, 260.0],
            "ttm_net_income": [8.0, 13.2],
            "shares_outstanding": [1e6, 1e6],
        }),
        # Single YoY-ineligible filer: everything YoY-based is NaN.
        "BBB": pd.DataFrame(
            {"assets": [50.0], "revenue_ttm": [10.0], "ttm_net_income": [5.0],
             "shares_outstanding": [1e6]},
            index=pd.DatetimeIndex([FILED_2019_Q4]),
        ),
    }
    panel, as_of = _fund_panel(fund)
    ag = _score("asset_growth", panel, as_of)
    assert math.isclose(ag["AAA"], -(110.0 / 100.0 - 1), rel_tol=1e-12)  # negated
    assert math.isnan(ag["BBB"])
    rg = _score("rev_growth", panel, as_of)
    assert math.isclose(rg["AAA"], 260.0 / 200.0 - 1, rel_tol=1e-12)
    roa = _score("roa", panel, as_of)
    assert math.isclose(roa["AAA"], 13.2 / 110.0, rel_tol=1e-12)
    assert math.isclose(roa["BBB"], 5.0 / 50.0, rel_tol=1e-12)  # roa needs no prior
    droa = _score("droa", panel, as_of)
    assert math.isclose(droa["AAA"], 13.2 / 110.0 - 8.0 / 100.0, rel_tol=1e-12)
    assert math.isnan(droa["BBB"])


def test_net_issuance_is_split_adjusted_and_negated():
    fund = {"AAA": _fund_frame({
        "assets": [100.0, 110.0], "revenue_ttm": [200.0, 260.0],
        "ttm_net_income": [8.0, 13.2],
        "shares_outstanding": [1e6, 2.1e6],   # 2:1 split + 5% true issuance
    })}
    idx = pd.date_range("2019-01-02", periods=300, freq="B", tz="UTC")
    split = pd.Series(1.0, index=idx)
    split.loc[pd.Timestamp("2019-06-03", tz="UTC")] = 2.0  # between the filings
    panel, as_of = _fund_panel(fund, split_factor=split)
    got = _score("net_issuance", panel, as_of)["AAA"]
    assert math.isclose(got, -(2.1e6 / (1e6 * 2.0) - 1), rel_tol=1e-12)  # -0.05


def test_net_issuance_nan_when_split_history_is_unknown():
    fund = {"AAA": _fund_frame({
        "assets": [100.0, 110.0], "revenue_ttm": [200.0, 260.0],
        "ttm_net_income": [8.0, 13.2], "shares_outstanding": [1e6, 1.05e6],
    })}
    panel, as_of = _fund_panel(fund, split_factor=float("nan"))  # legacy cache
    assert math.isnan(_score("net_issuance", panel, as_of)["AAA"])


def test_fundamentals_family_nan_without_a_store():
    panel, as_of = _fund_panel({"AAA": _fund_frame({
        "assets": [100.0, 110.0], "revenue_ttm": [200.0, 260.0],
        "ttm_net_income": [8.0, 13.2], "shares_outstanding": [1e6, 1e6],
    })})
    bare = PanelData(closes=panel.closes, bars=panel.bars, symbols=panel.symbols)
    for name in ("asset_growth", "net_issuance", "roa", "droa", "rev_growth"):
        assert _score(name, bare, as_of).isna().all(), name
```

Append to `tests/test_alphasearch_segments.py`:

```python
def test_deep_segments_carry_fundamentals_dir_when_the_store_exists(tmp_path):
    # Tier-1 batch spec section 3.4: pre-registered prospective amendment to
    # Piece 2 section 3.2. fundamentals_dir=None on deep pools only ever
    # meant "no store backfilled yet", never a design choice; with a local
    # store the fundamentals family sweeps segments too. No fundamentals
    # segment trial ever ran before this amendment, so nothing is spent.
    _pharma, _banks, sic, membership = _fixture_root(tmp_path)
    store = tmp_path / "data" / "fundamentals" / "equities"
    store.mkdir(parents=True)
    universes, _ = segment_universes(tmp_path, sic, membership_path=membership)
    assert universes["largecap:biotech"].fundamentals_dir == store
    assert universes["opt-largecap:biotech"].fundamentals_dir == store
```

(The existing `test_segment_universes_emits_deep_and_options_pools` asserts `deep.fundamentals_dir is None` on a store-less root — that stays true and stays untouched: the amendment is conditional on the store existing.)

In `tests/test_alphasearch_spec.py`, bump the registry test: `assert len(SIGNALS) == 35` and `fundamentals_family = {"gross_profitability", "earnings_yield", "book_to_market", "asset_growth", "net_issuance", "roa", "droa", "rev_growth"}` (use the named set in the loop as in Task 5's version).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_alphasearch_panel.py::test_fundamentals_row_prior_300_day_boundary tests/test_alphasearch_tier1.py tests/test_alphasearch_segments.py -q`
Expected: FAIL — `AttributeError: 'PanelView' object has no attribute 'fundamentals_row_prior'`, `KeyError: 'asset_growth'`, amendment assert None != store.

- [ ] **Step 3: Implement `fundamentals_row_prior` in `panel.py`**

Add module constant `MIN_YOY_AGE_DAYS = 300  # spec section 2: the YoY filing rule` and, in `PanelView` after `fundamentals_row`:

```python
    def fundamentals_row_prior(
        self, symbol: str, min_age_days: int = MIN_YOY_AGE_DAYS
    ) -> pd.Series | None:
        """The 'one-year-prior filing' of the YoY rule (spec section 2): the
        latest row FILED at least min_age_days calendar days before the
        CURRENT filing's filed date (current = fundamentals_row's cut).
        Exactly-min_age_days qualifies. None when no current filing is
        visible or no filing is old enough -- YoY signals then go NaN,
        dropped, never imputed."""
        frame = self._panel.fundamentals.get(symbol)
        if frame is None or frame.empty:
            return None
        window = frame.loc[: self.as_of]
        if window.empty:
            return None
        current_filed = window.index[-1]
        cutoff = current_filed - pd.Timedelta(min_age_days, unit="D")
        prior = window.loc[:cutoff]
        return None if prior.empty else prior.iloc[-1]
```

- [ ] **Step 4: Implement the five registrations in `spec.py`**

Append after the existing fundamentals block:

```python
# --------------------------------------------------------------------------- #
# Tier-1 fundamentals family (spec 2026-07-09 section 2). YoY = latest filing
# vs the latest filing FILED >= 300 calendar days earlier
# (PanelView.fundamentals_row_prior); missing/ineligible -> NaN, dropped.
# --------------------------------------------------------------------------- #
def _fund_value(row: pd.Series | None, key: str) -> float:
    if row is None or key not in row.index:
        return math.nan
    return float(row[key])


def _yoy_growth(key: str, sign: float) -> SignalFn:
    """sign * (current/prior - 1) of one stored primitive; NaN unless both
    filings carry a value and the prior is strictly positive (a ratio against
    a non-positive base has no growth interpretation)."""

    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol in view.symbols:
            now = _fund_value(view.fundamentals_row(symbol), key)
            then = _fund_value(view.fundamentals_row_prior(symbol), key)
            score = math.nan
            if not math.isnan(now) and not math.isnan(then) and then > 0:
                score = sign * (now / then - 1.0)
            scores[symbol] = score
        return pd.Series(scores, dtype="float64")

    return fn


def _roa_of(row: pd.Series | None) -> float:
    ni = _fund_value(row, "ttm_net_income")
    assets = _fund_value(row, "assets")
    if math.isnan(ni) or not assets > 0:
        return math.nan
    return ni / assets


def _roa(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    scores = {s: _roa_of(view.fundamentals_row(s)) for s in view.symbols}
    return pd.Series(scores, dtype="float64")


def _droa(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    scores = {
        s: _roa_of(view.fundamentals_row(s)) - _roa_of(view.fundamentals_row_prior(s))
        for s in view.symbols
    }  # NaN propagates from either leg
    return pd.Series(scores, dtype="float64")


def _net_issuance(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    """Split-adjusted shares_outstanding YoY, NEGATED (issuers underperform,
    Pontiff-Woodgate). Comparable prior shares = prior * product of
    split_factor over bar dates in (prior_filed, current_filed]. ANY NaN
    split_factor in that window -> NaN: a legacy narrow cache cannot claim
    "no splits", and prod()'s skipna would fabricate exactly that."""
    scores: dict[str, float] = {}
    for symbol in view.symbols:
        current = view.fundamentals_row(symbol)
        prior = view.fundamentals_row_prior(symbol)
        score = math.nan
        if current is not None and prior is not None:
            shares_now = _fund_value(current, "shares_outstanding")
            shares_then = _fund_value(prior, "shares_outstanding")
            if not math.isnan(shares_now) and shares_then > 0:
                factors = view.bars(symbol)["split_factor"]
                window = factors[(factors.index > prior.name)
                                 & (factors.index <= current.name)]
                if not window.isna().any():
                    adjustment = float(window.prod()) if len(window) else 1.0
                    if adjustment > 0:
                        score = -(shares_now / (shares_then * adjustment) - 1.0)
        scores[symbol] = score
    return pd.Series(scores, dtype="float64")


# Investment factor: asset growers underperform (Cooper-Gulen-Schill) -> negate.
_register("asset_growth", _yoy_growth("assets", -1.0), requires_fundamentals=True)
# Issuance anomaly (Pontiff-Woodgate): negation lives inside _net_issuance.
_register("net_issuance", _net_issuance, requires_fundamentals=True)
# Quality: profitable-per-asset names outperform.
_register("roa", _roa, requires_fundamentals=True)
# Fundamental momentum: improving profitability.
_register("droa", _droa, requires_fundamentals=True)
# Growth: rising trailing revenue.
_register("rev_growth", _yoy_growth("revenue_ttm", +1.0), requires_fundamentals=True)
```

- [ ] **Step 5: Implement the §3.4 amendment in `segments.py`**

In `segment_universes`, the deep-pool `UniverseSpec(...)` currently passes `fundamentals_dir=None`. Replace that argument with:

```python
                    # Piece 2 section 3.2 as amended by the Tier-1 batch spec
                    # section 3.4 (2026-07-09, prospective, pre-registered):
                    # None only ever meant "no store backfilled yet". With a
                    # local store the fundamentals family sweeps deep
                    # segments too; no fundamentals segment trial predates
                    # this, so nothing is spent.
                    fundamentals_dir=(
                        fundamentals_dir if fundamentals_dir.is_dir() else None
                    ),
```

Also update the docstring sentence "samples=None and fundamentals_dir=None, so the sweep's assembly-time checks confine them to price signals" to: "samples=None (options signals refused); fundamentals_dir attaches when the local store exists (Tier-1 spec §3.4 amendment), else None".

In `src/trading/cli.py`, the `--segments` refusal hint says "deep-pool segments carry only price data" — now wrong. Replace that `print(...)` string with:

```python
                print(
                    "hint: deep-pool segments carry no options data (and "
                    "fundamentals only once data/fundamentals/equities is "
                    f"synced) — pair --segments with --signals {price_signals} "
                    "(price family), or target options segments individually "
                    "via --universe opt-largecap:<segment>",
                    file=sys.stderr,
                )
```

(`tests/test_alphasearch_cli.py` asserts only `"hint:"`, the price-signal list prefix, and `"opt-largecap:<segment>"` — all preserved.)

- [ ] **Step 6: Run the tests, full suite, ruff**

Run: `uv run pytest tests/test_alphasearch_tier1.py tests/test_alphasearch_panel.py tests/test_alphasearch_segments.py tests/test_alphasearch_spec.py tests/test_alphasearch_cli.py -q && uv run pytest -q && uv run ruff check src tests scripts`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/trading/alphasearch/panel.py src/trading/alphasearch/spec.py \
    src/trading/alphasearch/segments.py src/trading/cli.py \
    tests/test_alphasearch_panel.py tests/test_alphasearch_tier1.py \
    tests/test_alphasearch_segments.py tests/test_alphasearch_spec.py
git commit -m "Register Tier-1 fundamentals family with 300-day YoY rule; segments fundamentals amendment [AI]"
```

---

### Task 7: Industry-relative family (2 signals) + sector threading

**Files:**
- Modify: `src/trading/alphasearch/panel.py` (`PanelData.sectors`, `PanelView.sector`, `build_panel(sectors=...)`), `src/trading/alphasearch/sweep.py` (`UniverseSpec.sic_map_path`, `_universe_sectors`, `build_universe_panel`), `src/trading/alphasearch/segments.py` (thread `sic_map_path`), `src/trading/alphasearch/spec.py`
- Modify: `tests/alphasearch_helpers.py` (`assemble_panel(sectors=...)`, `make_panel` sectors), `tests/test_alphasearch_lookahead.py` (perturb preserves sectors), `tests/test_alphasearch_spec.py` (final registry test)
- Test: `tests/test_alphasearch_tier1.py`, `tests/test_alphasearch_sweep.py`, `tests/test_alphasearch_segments.py`

**Interfaces:**
- Produces: `PanelData.sectors: dict[str, str]` (symbol → the frozen SEGMENTS sector; absent = unmapped); `PanelView.sector(symbol) -> str | None`; `build_panel(..., sectors: dict[str, str] | None = None)`; `UniverseSpec.sic_map_path: Path | None = None` (None → committed `segments.DEFAULT_SIC_MAP_CSV`); `sweep._universe_sectors(sic_map_path) -> dict[str, str]` (lazy `segments` import — `segments.py` imports `sweep` at module scope, so a top-level import back would cycle); registrations `ind_mom` (+), `ind_rel_rev` (formula-carried sign), no requires flags. Registry FINAL: 37.
- Consumes: `segments.SEGMENTS/segments_for/load_sic_map` (sectors = the 10 `kind == "sector"` segments; `biotech`/`banks` are industries, not sectors), Task 3's `_mom_12_2`, existing `_trail`.

**Frozen definitions (spec §2):** `ind_mom` = the sector cross-sectional mean of `mom_12_2`, assigned to every member of the sector; `ind_rel_rev` = −(trail21 − sector mean trail21). Sector stats are computed ONLY from symbols present in that date's cross-section (`view.symbols`), NaN-input members contributing nothing; unmapped/unsegmented symbols score NaN; a sector with no finite member has no mean (its members go NaN). For `ind_rel_rev` a symbol also needs its OWN trail21. For `ind_mom` a mapped member without its own mom still receives the sector mean (industry momentum is a sector attribute).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_alphasearch_tier1.py`:

```python
# --------------------------------------------------------------------------- #
# Industry-relative family (10 frozen SEGMENTS sectors via sic_map)
# --------------------------------------------------------------------------- #


def _sector_panel() -> tuple[PanelData, pd.Timestamp]:
    frames = {
        "FIN1": _bar_frame(_geometric_close(0.001, 300)),
        "FIN2": _bar_frame(_geometric_close(0.003, 300)),
        "TRD1": _bar_frame(_geometric_close(0.002, 300)),
        "UNMAPPED": _bar_frame(_geometric_close(0.004, 300)),
    }
    panel = _bar_panel(
        frames,
        sectors={"FIN1": "finance", "FIN2": "finance", "TRD1": "trade"},
    )
    return panel, frames["FIN1"].index[-1]


def _mom(rate: float) -> float:
    return (1 + rate) ** 231 - 1


def _trail21(rate: float) -> float:
    return (1 + rate) ** 21 - 1


def test_ind_mom_assigns_the_sector_mean_and_nan_to_unmapped():
    panel, as_of = _sector_panel()
    scores = _score("ind_mom", panel, as_of)
    finance_mean = (_mom(0.001) + _mom(0.003)) / 2
    assert math.isclose(scores["FIN1"], finance_mean, rel_tol=1e-9)
    assert math.isclose(scores["FIN2"], finance_mean, rel_tol=1e-9)
    assert math.isclose(scores["TRD1"], _mom(0.002), rel_tol=1e-9)
    assert math.isnan(scores["UNMAPPED"])  # never guessed


def test_ind_rel_rev_rewards_within_sector_laggards():
    panel, as_of = _sector_panel()
    scores = _score("ind_rel_rev", panel, as_of)
    finance_mean = (_trail21(0.001) + _trail21(0.003)) / 2
    assert math.isclose(scores["FIN1"], -(_trail21(0.001) - finance_mean),
                        rel_tol=1e-9)
    assert scores["FIN1"] > 0 > scores["FIN2"]  # laggard attractive, leader not
    # A one-member sector sits exactly at its own mean.
    assert math.isclose(scores["TRD1"], 0.0, abs_tol=1e-12)
    assert math.isnan(scores["UNMAPPED"])


def test_sector_stats_use_only_the_dates_cross_section():
    # A finance member with too little history for mom_12_2 contributes
    # NOTHING to the sector mean, but (mapped) still receives ind_mom's mean.
    frames = {
        "FIN1": _bar_frame(_geometric_close(0.001, 300)),
        "FIN2": _bar_frame(_geometric_close(0.003, 300)),
        "FINYOUNG": _bar_frame(_geometric_close(0.05, 30, start="2020-01-02")),
    }
    sectors = {s: "finance" for s in frames}
    panel = _bar_panel(frames, sectors=sectors)
    as_of = frames["FIN1"].index[-1]
    scores = _score("ind_mom", panel, as_of)
    finance_mean = (_mom(0.001) + _mom(0.003)) / 2  # FINYOUNG's NaN excluded
    assert math.isclose(scores["FIN1"], finance_mean, rel_tol=1e-9)
    assert math.isclose(scores["FINYOUNG"], finance_mean, rel_tol=1e-9)
    # ind_rel_rev needs the symbol's OWN trail21 too.
    rel = _score("ind_rel_rev", panel, as_of)
    assert not math.isnan(rel["FINYOUNG"])  # 30 bars >= 22: trail21 exists
```

Append to `tests/test_alphasearch_sweep.py`:

```python
def test_build_universe_panel_derives_sectors_from_the_sic_map(tmp_path):
    from trading.alphasearch.sweep import build_universe_panel

    idx = pd.date_range("2020-01-02", periods=5, freq="B", tz="UTC")
    cache = tmp_path / "cache"
    cache.mkdir()
    for sym in ("AAA", "BBB", "CCC"):
        pd.DataFrame(
            {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
            index=idx,
        ).to_parquet(cache / f"{sym}.parquet")
    sic = tmp_path / "sic.csv"
    sic.write_text(
        "symbol,cik,sic,sic_description,fetched_at\n"
        "AAA,1,2836,biotech,2026-07-09\n"   # pharma-chemicals sector (+biotech industry)
        "BBB,2,6022,bank,2026-07-09\n"      # finance sector (+banks industry)
        "CCC,3,700,farm,2026-07-09\n"       # covered by no segment
    )
    uspec = UniverseSpec("u", cache, None, None, symbols=("AAA", "BBB", "CCC"),
                         sic_map_path=sic)
    panel = build_universe_panel(uspec, make_factors())
    # Sectors only (the 10-way partition); industries never masquerade as one.
    assert panel.sectors == {"AAA": "pharma-chemicals", "BBB": "finance"}
```

Append to `tests/test_alphasearch_segments.py`:

```python
def test_segment_universes_thread_the_sic_map_path(tmp_path):
    _pharma, _banks, sic, membership = _fixture_root(tmp_path)
    universes, _ = segment_universes(tmp_path, sic, membership_path=membership)
    assert all(u.sic_map_path == sic for u in universes.values())
```

In `tests/test_alphasearch_spec.py`, the FINAL registry test:

```python
def test_registry_is_complete_with_correct_requirements():
    assert len(SIGNALS) == 37  # 16 seeds + 21 Tier-1 (9+5+5+2)
    options_family = {"vrp", "hedge", "excite", "atm_iv", "smile", "atm_spread",
                      "cp_vol", "osv", "otm_put_iv", "iv_change", "dskew"}
    volume_family = {"cp_vol", "osv"}
    fundamentals_family = {"gross_profitability", "earnings_yield",
                           "book_to_market", "asset_growth", "net_issuance",
                           "roa", "droa", "rev_growth"}
    for name, spec in SIGNALS.items():
        assert spec.requires_options == (name in options_family)
        assert spec.requires_option_volume == (name in volume_family)
        assert spec.requires_fundamentals == (name in fundamentals_family)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_alphasearch_tier1.py tests/test_alphasearch_sweep.py tests/test_alphasearch_segments.py -q`
Expected: FAIL — `TypeError: _bar_panel() got an unexpected keyword argument 'sectors'` resolves to `TypeError: PanelData.__init__() got an unexpected keyword argument 'sectors'`; `KeyError: 'ind_mom'`; `sic_map_path` AttributeError.

- [ ] **Step 3: Implement sector threading**

`panel.py` — `PanelData` field after `has_option_volume`:

```python
    # symbol -> frozen SEGMENTS sector (the 10-way partition); absent =
    # unmapped, and industry-relative signals then score NaN, never a guess.
    sectors: dict[str, str] = field(default_factory=dict)
```

`PanelView`:

```python
    def sector(self, symbol: str) -> str | None:
        return self._panel.sectors.get(symbol)
```

`build_panel` gains keyword `sectors: dict[str, str] | None = None` and passes
`sectors={} if sectors is None else {s: sectors[s] for s in universe if s in sectors},` in the returned `PanelData(...)`.

`sweep.py` — add field to `UniverseSpec` (after `symbols`):

```python
    # Committed sic_map.csv override for sector derivation (industry-relative
    # family); None = segments.DEFAULT_SIC_MAP_CSV. Not part of the hashed
    # trial config: like bar caches, the map is committed data, and the
    # universe's identity is its NAME.
    sic_map_path: Path | None = None
```

and:

```python
def _universe_sectors(sic_map_path: Path | None) -> dict[str, str]:
    """symbol -> its (unique) frozen SEGMENTS sector for every mapped symbol.
    Industry segments (biotech, banks) overlap sectors and are NOT the
    partition, so only kind == "sector" qualifies. Imported lazily:
    segments.py imports this module at module scope, so a top-level import
    back would be a cycle."""
    from trading.alphasearch.segments import SEGMENTS, load_sic_map, segments_for

    sector_of: dict[str, str] = {}
    for symbol, code in load_sic_map(sic_map_path).items():
        sector = next(
            (n for n in segments_for(code) if SEGMENTS[n].kind == "sector"), None
        )
        if sector is not None:
            sector_of[symbol] = sector
    return sector_of


def build_universe_panel(
    spec: UniverseSpec, factors: pd.DataFrame | None = None
) -> PanelData:
    return build_panel(
        spec.cache_dir, spec.samples, spec.fundamentals_dir,
        symbols=spec.symbols, factors=factors,
        sectors=_universe_sectors(spec.sic_map_path),
    )
```

`segments.py` — at the top of `segment_universes`, resolve the default once:

```python
    if sic_map_path is None:
        sic_map_path = DEFAULT_SIC_MAP_CSV
```

(and keep passing it to `load_sic_map(sic_map_path)`), then add `sic_map_path=sic_map_path,` to BOTH emitted `UniverseSpec(...)` constructions.

`spec.py` — append:

```python
# --------------------------------------------------------------------------- #
# Tier-1 industry-relative family (spec 2026-07-09 section 2): the 10 frozen
# SEGMENTS sectors are the industry partition (via the committed sic_map,
# threaded onto the panel as PanelData.sectors). Sector stats come ONLY from
# symbols present in the date's cross-section; unmapped symbols score NaN.
# --------------------------------------------------------------------------- #
def _sector_means(view: PanelView, values: dict[str, float]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for symbol in view.symbols:
        sector = view.sector(symbol)
        value = values[symbol]
        if sector is None or math.isnan(value):
            continue
        sums[sector] = sums.get(sector, 0.0) + value
        counts[sector] = counts.get(sector, 0) + 1
    return {sector: sums[sector] / counts[sector] for sector in sums}


def _ind_mom(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    values = {s: _mom_12_2(view.closes(s)) for s in view.symbols}
    means = _sector_means(view, values)
    scores = {
        s: (means.get(sector, math.nan)
            if (sector := view.sector(s)) is not None else math.nan)
        for s in view.symbols
    }
    return pd.Series(scores, dtype="float64")


def _ind_rel_rev(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    values = {s: _trail(view.closes(s), 21) for s in view.symbols}
    means = _sector_means(view, values)
    scores: dict[str, float] = {}
    for symbol in view.symbols:
        sector = view.sector(symbol)
        own = values[symbol]
        if sector is None or sector not in means or math.isnan(own):
            scores[symbol] = math.nan
        else:
            scores[symbol] = -(own - means[sector])
    return pd.Series(scores, dtype="float64")


# Industry momentum (Moskowitz-Grinblatt): a hot sector lifts every member.
_register("ind_mom", _ind_mom)
# Within-industry reversal (Da-Liu-Schaumburg, 21d at monthly cadence):
# laggards vs their sector mean recover -- the minus lives IN the formula,
# so registration is raw (spec section 2 defines the signal WITH the minus).
_register("ind_rel_rev", _ind_rel_rev)
```

Also replace the now-stale final paragraph of `spec.py`'s MODULE docstring ("Task 4 registers the price family; Task 5 completes the registry...") with:

```text
The 16 seed signals (Piece 1) are followed by the 21 Tier-1 batch signals
(docs/superpowers/specs/2026-07-09-tier1-signal-batch-design.md section 2):
9 price/volume, 5 options (cp_vol/osv gated on requires_option_volume),
5 fundamentals (300-calendar-day YoY filing rule), 2 industry-relative
(the 10 frozen SEGMENTS sectors). All formulas, windows, floors, and signs
are frozen pre-registration.
```

`tests/alphasearch_helpers.py` — `assemble_panel` gains a final keyword `sectors: dict[str, str] | None = None`, and its `PanelData(...)` call gains `sectors={} if sectors is None else sectors,`. In `make_panel`, add before the return:

```python
    sectors = {sym: ("manufacturing-tech" if i % 2 == 0 else "finance")
               for i, sym in enumerate(names)}
```

and change the return to:

```python
    return assemble_panel(
        bars, options, fundamentals, factors,
        has_option_volume=with_options and with_option_volume,
        sectors=sectors,
    )
```

`tests/test_alphasearch_lookahead.py` — `_perturb_after`'s reassembly must preserve the (time-invariant) sector map and the volume flag: change its final line to:

```python
    return assemble_panel(
        bars, options, fundamentals, factors,
        has_option_volume=panel.has_option_volume, sectors=panel.sectors,
    )
```

- [ ] **Step 4: Run the tests, full suite, ruff**

Run: `uv run pytest tests/test_alphasearch_tier1.py tests/test_alphasearch_sweep.py tests/test_alphasearch_segments.py tests/test_alphasearch_spec.py tests/test_alphasearch_lookahead.py -q && uv run pytest -q && uv run ruff check src tests scripts`
Expected: PASS — the lookahead anti-vacuity guard now covers `ind_mom`/`ind_rel_rev` via `make_panel`'s sectors, and the perturbation test covers them end-to-end. (Real-path tests like `test_deep_universe_runs_price_signals_end_to_end` now read the COMMITTED `src/trading/venues/universes/sic_map.csv` through `_universe_sectors` — it is committed, so no fixture is needed; fixture symbols simply come back unmapped.)

- [ ] **Step 5: Commit**

```bash
git add src/trading/alphasearch/panel.py src/trading/alphasearch/spec.py \
    src/trading/alphasearch/sweep.py src/trading/alphasearch/segments.py \
    tests/alphasearch_helpers.py tests/test_alphasearch_tier1.py \
    tests/test_alphasearch_sweep.py tests/test_alphasearch_segments.py \
    tests/test_alphasearch_spec.py tests/test_alphasearch_lookahead.py
git commit -m "Register Tier-1 industry-relative family with sector threading [AI]"
```

---

### Task 8: Golden-sweep fixture extension + docs

**Files:**
- Modify: `tests/test_alphasearch_segments_golden.py`
- Modify: `docs/experiments.md`, `docs/glossary.md`
- Test: the new golden tests themselves

**Interfaces:**
- Consumes: everything above through the REAL file path (`segment_universes` → `run_sweep` → `build_universe_panel` → `build_panel`, no `panel_factory` injection).

- [ ] **Step 1: Write the failing golden tests**

Append to `tests/test_alphasearch_segments_golden.py`:

```python
# --------------------------------------------------------------------------- #
# Tier-1 batch golden coverage: one signal per new family through the REAL
# file path (extended bar schema + fundamentals store + factor threading).
# --------------------------------------------------------------------------- #


def _write_tier1_root(tmp_path):
    """_write_root plus: extended bar schema (per-symbol overnight drift,
    real high/low span, div_cash, split_factor), a fundamentals store (so
    the Tier-1 spec section 3.4 amendment attaches it to deep pools), and a
    SECOND two-sector sic map for the flat pool's industry signals (the
    original all-2836 map keeps the segment expectations intact)."""
    panel, samples, sic, membership = _write_root(tmp_path)
    cache = tmp_path / "data" / "equities-tiingo"
    store = tmp_path / "data" / "fundamentals" / "equities"
    store.mkdir(parents=True)
    for i, sym in enumerate(panel.symbols):
        closes = panel.closes[sym]
        opens = closes.shift(1) * (1 + 1e-4 * (i + 1))
        opens.iloc[0] = closes.iloc[0]
        pd.DataFrame(
            {"open": opens, "high": closes * 1.01, "low": closes * 0.99,
             "close": closes, "volume": 1000.0, "div_cash": 0.001 * (i + 1),
             "split_factor": 1.0},
            index=closes.index,
        ).to_parquet(cache / f"{sym}.parquet")
        pd.DataFrame(
            {"gross_profitability": [0.10 + 0.02 * i],
             "ttm_net_income": [1e6 * (i + 1)], "book_equity": [5e6 * (i + 1)],
             "shares_outstanding": [1e6], "assets": [1e7 * (i + 1)],
             "revenue_ttm": [2e7 * (i + 1)]},
            index=pd.DatetimeIndex([closes.index[0]], name="filed"),
        ).to_parquet(store / f"{sym}.parquet")
    sic_split = tmp_path / "sic_split.csv"
    sic_split.write_text(
        "symbol,cik,sic,sic_description,fetched_at\n"
        + "".join(
            f"{s},{i + 1},{2836 if i % 2 == 0 else 6022},d,2026-07-09\n"
            for i, s in enumerate(panel.symbols)
        )
    )
    return panel, samples, sic, membership, sic_split


def test_tier1_families_sweep_the_flat_pool_end_to_end(tmp_path):
    _panel, samples, _sic, _membership, sic_split = _write_tier1_root(tmp_path)
    flat = UniverseSpec(
        "largecap", tmp_path / "data" / "equities-tiingo", samples,
        tmp_path / "data" / "fundamentals" / "equities", sic_map_path=sic_split,
    )
    journal = trials_journal(tmp_path / "journal")
    names = ("overnight", "ivol", "cp_vol", "iv_change", "roa", "ind_rel_rev")
    rows, n_trials = run_sweep(
        {"largecap": flat}, journal, make_factors(), ts="t1",
        signals={n: SIGNALS[n] for n in names}, window=WINDOW,
    )
    assert n_trials == 6
    assert {r.signal for r in rows} == set(names)
    # Every new family produced a CLEAN trial on real files: bars (overnight),
    # factor-threaded features (ivol), option-volume cells (cp_vol), prior
    # cells (iv_change), the store (roa), and two sic sectors (ind_rel_rev).
    assert all(r.error is None for r in rows)
    assert len({e["config_hash"] for e in discovery_trials(journal)}) == 6


def test_fundamentals_signal_sweeps_a_deep_segment_after_the_amendment(tmp_path):
    _panel, _samples, sic, membership, _split = _write_tier1_root(tmp_path)
    seg_universes, _ = segment_universes(tmp_path, sic, membership_path=membership)
    deep = seg_universes["largecap:biotech"]
    assert deep.samples is None  # still a deep pool: options stay refused
    assert deep.fundamentals_dir == tmp_path / "data" / "fundamentals" / "equities"
    journal = trials_journal(tmp_path / "journal")
    rows, n = run_sweep(
        {deep.name: deep}, journal, make_factors(), ts="t1",
        signals={"roa": SIGNALS["roa"]}, window=WINDOW,
    )
    assert n == 1
    assert rows[0].universe == "largecap:biotech" and rows[0].error is None
```

- [ ] **Step 2: Run to verify state**

Run: `uv run pytest tests/test_alphasearch_segments_golden.py -q`
Expected: the two NEW tests PASS immediately if Tasks 1–7 are complete — golden tests are integration seals, not feature drivers. If either FAILS, debug the engine (most likely: a decision date where every symbol is NaN for a chosen signal — `overnight` legitimately skips Jan–Apr on the 130-bar fixture and trades May/June; ALL dates skipping would be a real bug).

- [ ] **Step 3: Docs — experiments.md §10 registration note**

Append at the end of `docs/experiments.md` §10 (before `## Known caveats...`):

```markdown
**Tier-1 signal batch registered 2026-07-09, pre-sweep.** 21 new signals
(9 price/volume incl. `mom_12_2`/`overnight`/`park_vol`/`ivol`/`max5`/`beta`/
`amihud`/`vol_trend`/`div_yield`, 5 fundamentals incl. the 300-calendar-day
YoY filing rule, 5 options incl. the option-volume-gated `cp_vol`/`osv`, 2
industry-relative over the 10 frozen SEGMENTS sectors) are frozen in
`docs/superpowers/specs/2026-07-09-tier1-signal-batch-design.md` §2 —
formulas, windows, minimum-observation floors, and signs pre-registered
BEFORE any trial ran. The pre-registered discovery sweep (that spec §4: all
21 × every compatible universe, BH q=0.10 across the whole journal including
the 238 existing trials) has NOT run as of this registration; its results get
their own entry. Includes one pre-registered prospective amendment to Piece 2
§3.2: deep-pool segments carry `fundamentals_dir` when the local store exists
(no fundamentals segment trial predates it, so nothing is spent). New terms
in the glossary ("The anomaly zoo" section). NOTE: the largecap bar cache is
the legacy narrow schema (no `div_cash`/`split_factor`), so `div_yield` and
`net_issuance` journal honest error trials there until it is re-backfilled.
```

- [ ] **Step 4: Docs — glossary anomaly section**

Append to `docs/glossary.md` (new themed section at the end):

```markdown
## The anomaly zoo (Tier-1 signal batch)

**Momentum 12−2** — total return from 12 months ago to 1 month ago, skipping the
most recent month because short-term reversal contaminates it (Jegadeesh-Titman).
The canonical UMD construction.

**Overnight return persistence** — split each day into overnight (prev close→open)
and intraday; the overnight component carries persistent, clientele-driven
momentum of its own (Lou-Polk-Skouras).

**Parkinson volatility** — a range-based vol estimator using ln(High/Low)²/(4·ln 2);
~5x more statistically efficient per day than close-to-close vol. Feeds the
low-vol anomaly.

**Idiosyncratic volatility (IVOL) puzzle** — the std of a stock's daily FF3
regression residuals; HIGH-ivol stocks anomalously UNDERPERFORM
(Ang-Hodrick-Xing-Zhang), the opposite of risk-reward intuition.

**MAX / lottery demand** — the mean of a stock's few biggest recent daily gains;
lottery-seekers overpay for jackpot-shaped names, which then underperform
(Bali-Cakici-Whitelaw).

**Betting against beta (BAB)** — leverage-constrained investors overpay for high-β
stocks, flattening the security market line; low-β outperforms per unit of risk
(Frazzini-Pedersen).

**Amihud illiquidity** — mean |return| / dollar volume: price impact per dollar
traded. Illiquid names earn a premium for being hard to trade.

**High-volume return premium** — unusually high recent trading volume (vs its own
baseline) attracts attention and predicts higher returns
(Gervais-Kaniel-Mingelgrin).

**Asset growth / investment factor** — firms that grow total assets fastest
subsequently underperform (Cooper-Gulen-Schill); empire-building is expensive.
The CMA factor's characteristic cousin.

**Net share issuance** — split-adjusted growth in shares outstanding; issuers
underperform, buyback firms outperform (Pontiff-Woodgate) — management times the
market with its own stock.

**Fundamental momentum (ΔROA)** — the year-over-year CHANGE in profitability;
improving firms keep outperforming beyond what the level explains.

**Option-to-stock volume ratio (O/S)** — option dollar volume relative to stock
dollar volume; high O/S marks informed (disproportionately bearish) positioning
ahead of returns (Johnson-So).

**Volatility smirk** — OTM-put implied vol above ATM implied vol; a steep or
steepening smirk means crash protection is being bid, predicting negative
returns (Xing-Zhang-Zhao). IV *innovations* (An-Ang-Bali-Cakici) time-difference
the level: rising implied vol = rising perceived risk.

**Industry momentum** — a sector's own trailing return, assigned to every member;
industries trend (Moskowitz-Grinblatt), and much of stock momentum is industry
momentum.

**Within-industry reversal** — a stock's SHORT-term return relative to its sector
mean reverses (Da-Liu-Schaumburg): sector-adjusted laggards bounce, leaders fade.
```

- [ ] **Step 5: Full suite + ruff + final verification**

Run: `uv run pytest -q 2>&1 | tee /tmp/claude-tier1-final.log && uv run ruff check src tests scripts`
Expected: PASS, warnings-clean. Also verify, per the Global Constraints, that `git diff master -- src/trading/alphasearch/sweep.py | grep -n "_hashed_params" ` shows NO change to the hashed-params constructor (the batch adds no sweep parameters).

- [ ] **Step 6: Commit**

```bash
git add tests/test_alphasearch_segments_golden.py docs/experiments.md docs/glossary.md
git commit -m "Add Tier-1 golden sweep coverage and docs (pre-registration note, anomaly glossary) [AI]"
```

---

## Post-implementation notes (for the operator, not tasks)

1. **Before the pre-registered sweep runs** (spec §4): rsync the fundamentals store from the mini (spec §5), and re-backfill `data/equities-tiingo` with the extended bar schema or accept honest `div_yield`/`net_issuance` error trials on largecap.
2. The sweep itself is run ONCE by the operator after merge — it is deliberately NOT part of this plan.
3. `ivol`/`beta` precompute cost: ~1,400 symbols × ~1,800 joint days with batched det+solve is seconds per universe panel; the segment sweep builds ~30 panels, still minutes overall, dominated as before by trial evaluation.

## Sweep-day run-book (added 2026-07-09, final-review fix 5)

The all-or-nothing assembly refusal (spec section 6) means the Tier-1
registry cannot be swept in one call: options-volume signals need leg volume
(mid-cap-gathered universes only), fundamentals signals need a local store,
and `ind_mom` is structurally degenerate on single-sector segments (§2 note
above — it still journals honest `SortError` trials there, it just isn't
worth *steering* an operator at). Five invocations cover the full pre-
registered grid (spec §4) without ever tripping that refusal. Signal-family
shorthand used below (registry names, `src/trading/alphasearch/spec.py`):

- `PRICE` (16): `mom21,mom63,mom126,mom252,rev5,rvol21,disthigh,mom_12_2,overnight,park_vol,ivol,max5,beta,amihud,vol_trend,div_yield`
- `OPT_NONVOL` (9): `vrp,hedge,excite,atm_iv,smile,atm_spread,otm_put_iv,iv_change,dskew`
- `OPT_VOL` (2): `cp_vol,osv`
- `FUND` (8): `gross_profitability,earnings_yield,book_to_market,asset_growth,net_issuance,roa,droa,rev_growth`
- `INDUSTRY` (2): `ind_mom,ind_rel_rev`

Registry check: `PRICE(16) + OPT_NONVOL(9) + OPT_VOL(2) + FUND(8) + INDUSTRY(2) = 37`, matching `test_registry_is_complete_with_correct_requirements`.

### Operational prerequisites (both must be done BEFORE step (v), and the second before any largecap `div_yield`/`net_issuance` trial is trusted)

1. **Fundamentals store rsync** (design spec §5): `ssh mac-m1` (repo `~/trading`, store at `data/fundamentals/equities`) → this machine's `data/fundamentals/equities`, before step (v).
2. **Largecap cache re-backfill** for `close_raw`/`div_cash`/`split_factor` (the plan's "known data landmine", sharpened by fix 1): `data/equities-tiingo/*.parquet` is still the legacy narrow schema. Until it carries the extended columns, `div_yield` and `net_issuance` on every `largecap*` universe (flat and `largecap:<segment>`) journal honest all-NaN → `SortError` error trials — expected, not a bug, but counted against the BH bar for nothing. Re-backfill first if those two signals' largecap results are meant to be readable.

### Invocations

**(i) Flat pools, all non-fundamentals-non-optvol signals** — `PRICE + OPT_NONVOL + INDUSTRY` = 27 signals × 2 universes (`largecap`, `midcap`) = **54 trial-configs attempted**. Of these, the 13 pre-Tier-1 signals (7 price + 6 options) × 2 flat pools = 26 exactly re-hash the FIRST sweep's step (a) trials (idempotent dedup — journaled once, not double-counted); the remaining 14 Tier-1 signals (9 price + 3 opt-nonvol + 2 industry) × 2 = **28 genuinely new**.

```
trading alphasearch sweep --signals mom21,mom63,mom126,mom252,rev5,rvol21,disthigh,mom_12_2,overnight,park_vol,ivol,max5,beta,amihud,vol_trend,div_yield,vrp,hedge,excite,atm_iv,smile,atm_spread,otm_put_iv,iv_change,dskew,ind_mom,ind_rel_rev
```

**(ii) Midcap flat, option-volume family** — `OPT_VOL` = 2 signals × 1 universe (`midcap` only — `largecap`'s `samples.jsonl` carries no leg volume and would trip `requires_option_volume`) = **2 trial-configs, all new**.

```
trading alphasearch sweep --universe midcap --signals cp_vol,osv
```

**(iii) `--segments`, non-options families** — `PRICE + INDUSTRY` = 18 signals × 26 universes (2 flat + 24 emitted deep-pool segments, per the first sweep's count) = **468 trial-configs attempted**. Of these, 7 pre-Tier-1 price signals × 26 = 182 re-hash the first sweep's step (b) exactly (dedup); the rest — 11 Tier-1 signals (9 price + 2 industry) × 26 = **286 genuinely new**. Expect `ind_mom` to journal an honest `SortError` on every single-sector segment (§2 note, fix 2) — that is the guard working, not a run failure.

```
trading alphasearch sweep --segments --signals mom21,mom63,mom126,mom252,rev5,rvol21,disthigh,mom_12_2,overnight,park_vol,ivol,max5,beta,amihud,vol_trend,div_yield,ind_mom,ind_rel_rev
```

**(iv) Per opt-segment, options families** — the all-or-nothing refusal forbids combining `OPT_VOL` with an `opt-largecap:*` universe in one call (no leg volume there), so this runs once per one of the 5 viable opt-segment universes (first sweep's count; enumerate the current set from this run's own `segment excluded:` stderr lines from step (iii), or a throwaway `--universe bogus` unknown-universe error, which lists every known name). For each `opt-largecap:<segment>` universe: `OPT_NONVOL` only (9 trials). For each `opt-midcap:<segment>` universe (the only cap whose gather carries leg volume): `OPT_NONVOL + OPT_VOL` (11 trials).

```
trading alphasearch sweep --segments --universe opt-largecap:<segment> --signals vrp,hedge,excite,atm_iv,smile,atm_spread,otm_put_iv,iv_change,dskew
trading alphasearch sweep --segments --universe opt-midcap:<segment> --signals vrp,hedge,excite,atm_iv,smile,atm_spread,otm_put_iv,iv_change,dskew,cp_vol,osv
```

Arithmetic: letting `k` = the number of the 5 opt segments that are `opt-midcap:*`, step (iv) attempts `9×5 + 2×k = 45 + 2k` trial-configs; of these `6×5 = 30` (the pre-Tier-1 options family) re-hash the first sweep's step (c) (dedup), leaving `3×5 + 2×k = 15 + 2k` **genuinely new** (`0 <= k <= 5`).

**(v) Fundamentals family, everywhere a store is attached** — run AFTER prerequisite 1. Post-amendment (spec §3, item 4 / segments.py §3.4), every universe carries `fundamentals_dir`: 2 flat + 24 deep segments + 5 opt segments = 31 universes. `FUND` (8 signals, all NEW — no fundamentals trial predates this batch) × 31 = **248 trial-configs, all new**. One call covers every universe (fundamentals eligibility isn't cap-split like leg volume, so it needs none of step (iv)'s per-universe treatment):

```
trading alphasearch sweep --segments --signals gross_profitability,earnings_yield,book_to_market,asset_growth,net_issuance,roa,droa,rev_growth
```

**Partial-store abort behavior:** if the rsync in prerequisite 1 is incomplete such that even ONE targeted universe's member symbols have zero overlap with what's locally present, `panel.fundamentals` is empty for that universe and the all-or-nothing assembly check folds it into ONE combined `SweepError` naming every incompatible (signal, universe) pair — the entire step (v) call refuses with the store message (`"... has none. Expected store: data/fundamentals/equities ... Populate it with scripts/backfill_fundamentals.py"`), not just that universe. Fix: complete the rsync, or fall back to per-universe calls (like step (iv)) excluding the uncovered universe.

**Plain-command refusal:** `trading alphasearch sweep --segments` with NO `--signals` (the full default registry) is a deliberate refusal, not a bug — deep-pool segments carry no options data and (pre-rsync) no fundamentals store, so the all-or-nothing cross-product check catches the mismatch before any trial journals. The CLI prints an actionable `hint:` line naming a segment-safe signal subset (fix 4b: labeled "segment-safe signals", excludes `ind_mom`) and the `--universe opt-largecap:<segment>` per-universe workaround — this run-book's five-call split IS that hint's recipe, generalized to the full Tier-1 registry.

### Reconciliation

Sum the "genuinely new" counts above: `28 (i) + 2 (ii) + 286 (iii) + (15 + 2k) (iv) + 248 (v) = 579 + 2k` new discovery trials, `k` in `[0, 5]` → **579 to 589 new trials**, for a grand total of **817 to 827** discovery trials once combined with the first sweep's 238 (one BH computation spans all of them, spec 3.5). After all five invocations, reconcile:

```
trading alphasearch leaderboard --json | python3 -c "import json,sys; print(json.load(sys.stdin)['trials'])"
```

against the expected range above. A mismatch means either a universe emitted differently than the first sweep recorded (re-check `segment excluded:` stderr against experiments.md's count) or an invocation was skipped/repeated outside this run-book — investigate before trusting the leaderboard's BH gate. The estimate binds nothing (as the design spec itself notes for its own ~450 figure) — the journal's count is the only authority; this arithmetic exists to catch a materially wrong run, not to substitute for reading the journal.
