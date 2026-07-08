# Piece 1 — Core Alpha-Search Engine (design spec)

**Status:** approved design, 2026-07-08. Child of the program charter
(`2026-07-08-alpha-discovery-engine-program.md` — read it first for the why).
**Scope:** Piece 1 only. Segments (Piece 2), robustness automation (Piece 3),
and portfolio construction (Piece 4) are explicitly out of scope.

## 1. Purpose

Turn "signal + universe" into "factor-adjusted alpha + t-stat" in seconds, sweep
many such evaluations, gate the results with a multiple-testing-aware bar, and
persist an honest, reproducible trial record. A null result is a first-class
outcome.

## 2. Decisions made during brainstorming (locked)

| Question (charter §5) | Decision |
|---|---|
| Return-series construction | **Both** L/S and long-only computed; **long/short (Q5−Q1) is the gate**; long-only reported as tradability annotation only |
| Multiple-testing machinery | **BH-FDR at q = 0.10 gates**, computed across the entire persisted trial journal; **DSR reported** (advisory) for BH survivors |
| Discovery / holdout split | Discovery **2019-01-01..2023-12-31**; holdout **2024-01-01..latest**, touched once per candidate, journal-enforced |
| Architecture | New package `src/trading/alphasearch/`, composing existing primitives; `trading alphasearch` CLI |
| Quantile depth | Quintiles; terciles when a date's cross-section < 50 names; skip date entirely (journaled) when < 15 |
| Rebalance / series cadence | Monthly decision dates; **daily** portfolio return series |
| Weighting | Equal weight within quantile |
| Sweep execution | Serial (parallelism deferred until Piece 2 multiplies the grid) |
| Seed signal library | Wrap existing signals/metrics only (~15); no new signal research in Piece 1 |

## 3. Package layout

```
src/trading/alphasearch/
  spec.py       SignalSpec + SIGNALS registry
  panel.py      PanelData assembly (bars + options cells + fundamentals, PIT)
  sort.py       portfolio-sort daily return series (L/S and long-only)
  evaluate.py   factor regression → AlphaResult
  stats.py      p-values, BH-FDR, DSR
  sweep.py      sweep runner, leaderboard, holdout re-prove
journal/alphasearch-trials.jsonl   (via trading.journal.Journal)
```

CLI: `trading alphasearch sweep | leaderboard | holdout` in `src/trading/cli.py`,
following the existing subcommand pattern.

### 3.1 `spec.py` — signal specification

```python
@dataclass(frozen=True)
class SignalSpec:
    name: str
    fn: Callable[[PanelData, pd.Timestamp], pd.Series]  # per-symbol score, higher = long
    requires_options: bool = False
    requires_fundamentals: bool = False

SIGNALS: dict[str, SignalSpec]
```

PIT contract: `fn` may only read data at or before `as_of` (structurally: the
`PanelData` accessors it receives are as-of views). Seed registry wraps existing
metrics — price family from `scripts/signal_scan.py` (`mom21`, `mom63`,
`mom126`, `mom252`, `rev5`, `rvol21`, `vrp`, `disthigh`), options family from
`_cell_metrics` (`hedge`, `excite`, `atm_iv`, `smile`, `atm_spread`),
fundamentals family (`gross_profitability`, `earnings_yield`,
`book_to_market` — computed as in `quality.py`/`value.py`). Sign convention is
part of the wrapper (e.g. the skew signal registers as `-skew_put_atm` so
higher = more attractive), recorded per signal in the registry.

### 3.2 `panel.py` — data assembly

Generalizes `signal_scan.load_panel`. Inputs: Tiingo bar caches
(`data/equities-tiingo/`, `data/equities-midcap-tiingo/`; parquet
`[open, high, low, close, volume]`, UTC index), options cells
(`data/options-iv/samples.jsonl`, `samples-midcap.jsonl`), the fundamentals
store (`data/fundamentals/equities`). Monthly decision calendar (first trading
session per month). PIT positioning via `index.searchsorted(side="right") - 1`,
matching the existing pattern. Signal scoring and forward-return construction
are separate passes over the panel — signal functions can never reach forward
data.

Universe definitions for Piece 1: the two gathered options pools (large-cap 99
names, mid-cap per `samples-midcap.jsonl`) for options signals; for price/
fundamentals signals the pool is the same allowlists, so every signal within a
universe is measured on an identical cross-section (differences in coverage are
handled by §5 missing-data rules, not by widening the pool per signal).

### 3.3 `sort.py` — portfolio-sort return generator

Per decision date: rank the cross-section of scores; form equal-weight
quantile portfolios per §2; hold until the next decision date. Output: two
dated **daily** return series over the requested window — `ls = mean(Q_top) −
mean(Q_bottom)` and `lo = mean(Q_top)`. Daily returns give ~1250 regression
observations over discovery vs ~60 monthly. No transaction costs in the cheap
series (costs belong to the survivor-stage full backtest); the leaderboard
prints monthly one-way turnover of the top quantile so cost fragility is
visible early.

### 3.4 `evaluate.py` — alpha evaluator

The reusable statistics in `scripts/factor_regression.py` (`ols`,
`load_factors`, `parse_ff_csv`, `run_regression`, `RegressionResult`) are
**promoted into this module**; the script becomes a thin CLI wrapper importing
from the package (no duplicated stats code). Factors: Ken French daily
Carhart set (`Mkt-RF, SMB, HML, Mom` + `RF`), cached under `data/factors/`.
For the L/S series (self-financing) the regression excess return is the raw
spread; for long-only it is `returns − RF`. Output `AlphaResult`: CAPM
alpha/t, four-factor alpha/t, factor loadings + t's, R², n, raw annualized
Sharpe.

### 3.5 `stats.py` — the trials-aware bar

- `p_from_t(t, df)` two-sided.
- `bh_fdr(pvals, q=0.10) -> mask` — standard Benjamini-Hochberg step-up.
  Computed over **all discovery-window trials in the journal**, not just the
  current sweep.
- `deflated_sharpe(...)` — DSR per Bailey & López de Prado, using the journal's
  trial count and the variance/skew/kurtosis of the candidate's return series,
  with the cross-trial Sharpe variance estimated from journaled trials.
  Advisory only; printed for BH survivors.

### 3.6 `sweep.py` + CLI

`sweep`: enumerate `SIGNALS` × universes, build panel once per universe,
evaluate each trial on the discovery window, **append each trial to the journal
before the leaderboard is computed**, then print the leaderboard (sorted by
|4F L/S t|) with BH pass/fail marks, DSR for survivors, CAPM-vs-4F gap, factor
loadings, turnover, and the honest total trial count.

`leaderboard`: recompute the leaderboard + BH gate from the journal alone (no
new trials) — the auditable view.

`holdout <trial-id>`: evaluate one BH survivor on the holdout window. Refuses
to run if the journal already has a `holdout` event for that (signal, universe)
— override requires the literal confirmation `RERUN HOLDOUT`, mirroring
`backtest/experiments.py::prior_holdout`. Pass rule (pre-registered): the holdout
four-factor L/S alpha has the same sign as discovery AND its point estimate is
≥ 0.5 × the discovery alpha point estimate. The holdout window end (the "latest
data" date) is recorded in the journal event, making the evaluation exactly
reproducible.

## 4. Trial journal

`journal/alphasearch-trials.jsonl` via `trading.journal.Journal` (append-only,
fsync, torn-tail repair, sorted keys). One event per evaluation:

```json
{"event": "trial", "kind": "discovery" | "holdout",
 "signal": "...", "universe": "largecap|midcap", "window": "2019-01-01..2023-12-31",
 "params": {"quantiles": 5, "weighting": "equal", "cadence": "monthly"},
 "config_hash": "…", "n_dates": 60, "n_names_median": 97,
 "ls": {"alpha_annual_pct": ..., "alpha_t": ..., "capm_alpha_t": ...,
         "loadings": {...}, "r2": ..., "sharpe": ...},
 "lo": {…same…}, "turnover_monthly": ..., "skipped_dates": [...],
 "ts": "…"}
```

Identical `config_hash` re-runs update in place (idempotent), never
double-count. The trial count used by BH and DSR is derived from this file —
deleting or editing it invalidates the program's statistics, so it is
append-only and committed to git.

## 5. Pre-registered rules (amend only in writing, prospectively)

1. **Gate statistic:** four-factor (Carhart) alpha t-stat of the daily L/S
   series over discovery 2019-01-01..2023-12-31.
2. **Significance:** BH-FDR q = 0.10 across all journaled discovery trials.
3. **Holdout:** 2024-01-01..latest data; once per (signal, universe); pass rule
   in §3.6. Known caveat, disclosed: 2024-25 data was partially examined by the
   earlier skew studies (docs/experiments.md §9), so holdout passes for
   skew-family signals carry residual contamination risk and are read
   conservatively.
4. **Sort mechanics:** monthly rebalance on first trading session; quintiles;
   tercile fallback < 50 names; skip + journal < 15 names; equal weight.
5. **Missing data:** a symbol lacking a signal's inputs on a date is dropped
   from that date's cross-section — never imputed, never filled forward across
   the decision boundary.
6. **No survivor re-slicing:** any post-hoc variation (different quantiles,
   window, universe subset) is a *new journaled trial*, subject to the same BH
   accounting.

## 6. Error handling

- Empty/corrupt options cells: skip the cell (as `load_panel` does), count it,
  report coverage per signal in the leaderboard.
- Regression with n ≤ k or degenerate design: the trial fails loudly and is
  journaled as `"error"` — it still counts as a trial.
- Universe/signal mismatch (e.g. options signal on a universe without cells):
  refused at sweep-assembly time, not silently skipped.
- Factor cache absent + offline: hard error instructing `--refresh` (same
  behavior as `factor_regression.py` today).

## 7. Testing

- **Unit fixtures, hand-computable:** tiny synthetic panel (e.g. 6 symbols × 8
  dates) where quantile assignment, L/S daily returns, OLS alpha, BH cutoffs,
  and turnover are verifiable by hand. Every `stats.py` function tested against
  worked examples (BH against the canonical textbook example; DSR against a
  published reference value).
- **No-look-ahead test:** perturb all panel data strictly after date T; assert
  every registered signal's scores at ≤ T are bit-identical.
- **Journal honesty tests:** identical re-run doesn't inflate trial count;
  distinct params do; holdout double-touch refused without the override.
- **Golden sweep test:** full `sweep` on a fixture panel pinned end-to-end.
- **Promotion refactor test:** `scripts/factor_regression.py` CLI output
  unchanged after its internals move to `evaluate.py` (existing
  `test_factor_regression.py` keeps passing against the new import path).
- Repo discipline: `uv run pytest` (warnings-as-errors), `uv run ruff check`,
  granular `[AI]` commits.

## 8. Implementation & review discipline (per charter §6)

Subagent-driven: fresh-implementer subagents per plan task; an adversarial
reviewer specifically hunting (a) look-ahead in panel/sort construction,
(b) trial-count dishonesty, (c) BH/DSR misapplication, (d) holdout leakage.
Update `docs/experiments.md` + `docs/glossary.md` (new terms: FDR, BH, DSR,
portfolio sort, trial) when the first real sweep runs.

## 9. Out of scope (deferred)

Segmented universes and their classification data (Piece 2); perturbation
robustness gates and failure diagnostics (Piece 3); portfolio blending
(Piece 4); parallel sweep execution; signal combinations / parameter grids
(the sweep runner's trial schema already accommodates them via `params`);
any new data sources.
