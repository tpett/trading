# Core Alpha-Search Engine (Piece 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn "signal + universe" into "factor-adjusted alpha + t-stat" in seconds; sweep ~16 seed signals across two universes into a BH-FDR-gated leaderboard with an honest, journal-enforced trial count and a touched-once holdout.

**Architecture:** New package `src/trading/alphasearch/` composing existing primitives: signal registry (`spec.py`) scores a point-in-time `PanelData` (`panel.py`, generalizing `scripts/signal_scan.py::load_panel`); a portfolio sort (`sort.py`) turns scores into daily L/S and long-only return series; factor regression (`evaluate.py`, promoted from `scripts/factor_regression.py`) turns series into alpha t-stats; `stats.py` supplies p-values, BH-FDR, and DSR; `sweep.py` orchestrates trials, the append-only trial journal (`journal/alphasearch-trials.jsonl` via `trading.journal.Journal`), the leaderboard, and the journal-enforced holdout. CLI: `trading alphasearch sweep|leaderboard|holdout`.

**Tech Stack:** Python >=3.12, pandas 2.3.1, numpy (via pandas), pyarrow parquet caches, hand-rolled OLS (no scipy/statsmodels — Student-t p via incomplete beta with `math.lgamma`; normal CDF via stdlib `statistics.NormalDist`), pytest with warnings-as-errors, ruff.

**Spec:** `docs/superpowers/specs/2026-07-08-alpha-search-engine-design.md` (approved 2026-07-08). Program charter: `docs/superpowers/specs/2026-07-08-alpha-discovery-engine-program.md`.

## Global Constraints

- `uv run pytest` must pass — `pyproject.toml` sets `filterwarnings = ["error", ...]`, so ANY warning from our code is a test failure (pandas deprecations included).
- `uv run ruff check` must be clean — line-length 100, rules `E, F, I, UP, B` (imports sorted, no unused imports, `zip(..., strict=...)` explicit, no loop-closure lambdas).
- Python >= 3.12; pandas == 2.3.1; **scipy is NOT a dependency and must not be added** — `p_from_t` is implemented via the regularized incomplete beta (continued fraction + `math.lgamma`); normal CDF/inv-CDF via `statistics.NormalDist` (stdlib).
- Granular commits, one logical change each, message explains *why*, tagged `[AI]`.
- Pre-registered rules (spec §5) are LOCKED: gate statistic = four-factor L/S alpha t over discovery `2019-01-01..2023-12-31`; BH-FDR q=0.10 across ALL journaled discovery trials; holdout `2024-01-01..latest`, once per (signal, universe); monthly rebalance on first trading session; quintiles / terciles <50 names / skip+journal <15 names; equal weight; missing data dropped per date, never imputed; any variation is a new journaled trial.
- The trial journal `journal/alphasearch-trials.jsonl` is append-only and committed to git; the BH/DSR trial count derives from it.
- No transaction costs in the cheap series (costs belong to the survivor-stage full backtest); monthly one-way turnover of the top quantile is reported instead.
- `scripts/factor_regression.py` CLI behavior must not change; existing `tests/test_factor_regression.py` must keep passing unmodified.
- The CLI is the only module that reads the clock (`_utcnow` in `src/trading/cli.py`); everything below takes `as_of`/`ts` as parameters.

## File Structure

```
src/trading/alphasearch/
  __init__.py    (empty package marker)
  stats.py       p_from_t, bh_fdr, deflated_sharpe                  (Task 1)
  evaluate.py    ols/FF loading/RegressionResult promoted from
                 scripts/factor_regression.py + AlphaResult          (Task 2)
  panel.py       PanelData/PanelView assembly, PIT accessors,
                 decision calendar, cell_metrics                     (Tasks 3, 5)
  spec.py        SignalSpec + SIGNALS registry (16 seed signals)     (Tasks 4, 5)
  sort.py        portfolio-sort daily L/S + long-only series         (Task 6)
  sweep.py       trial journal layer, sweep runner, leaderboard,
                 holdout re-prove                                    (Tasks 7, 8, 9)
scripts/factor_regression.py   becomes a thin CLI wrapper            (Task 2)
scripts/signal_scan.py         imports cell_metrics from the package (Task 5)
src/trading/cli.py             `trading alphasearch` subcommand      (Task 10)
tests/alphasearch_helpers.py   shared synthetic fixtures             (Task 8)
tests/test_alphasearch_*.py    per-module tests                      (each task)
docs/glossary.md, docs/experiments.md                                (Task 12)
```

Interfaces flow strictly forward: stats and evaluate are leaf modules; panel feeds spec; panel+spec feed sort; everything feeds sweep; the CLI only touches sweep + evaluate.

---

### Task 1: `stats.py` — p-values, BH-FDR, Deflated Sharpe Ratio

**Files:**
- Create: `src/trading/alphasearch/__init__.py`
- Create: `src/trading/alphasearch/stats.py`
- Test: `tests/test_alphasearch_stats.py`

**Interfaces:**
- Consumes: nothing (leaf module; stdlib `math`/`statistics` + numpy only).
- Produces (Tasks 8/9 rely on these exact signatures):
  - `p_from_t(t: float, df: int) -> float` — two-sided Student-t p; NaN in -> NaN out.
  - `bh_fdr(pvals, q: float = 0.10) -> np.ndarray` — boolean pass mask aligned to input order; NaN p-values treated as 1.0 (never pass, still count in n).
  - `deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, n_trials: int, var_trials_sr: float) -> float` — DSR in [0,1]; all Sharpe inputs per-period (daily, non-annualized); `kurt` is Pearson kurtosis (normal = 3).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alphasearch_stats.py`:

```python
"""Unit tests for the trials-aware statistics (worked reference values)."""

from __future__ import annotations

import math

import numpy as np

from trading.alphasearch.stats import bh_fdr, deflated_sharpe, p_from_t

# --------------------------------------------------------------------------- #
# p_from_t: checked against published t-table critical values
# --------------------------------------------------------------------------- #
def test_p_from_t_matches_critical_values():
    # Two-sided 5% critical values: t(10)=2.228139, t(60)=2.000298, t(1)=12.7062.
    assert math.isclose(p_from_t(2.228139, 10), 0.05, abs_tol=1e-5)
    assert math.isclose(p_from_t(2.000298, 60), 0.05, abs_tol=1e-5)
    assert math.isclose(p_from_t(12.7062, 1), 0.05, abs_tol=1e-5)
    # Interior value: P(|T_10| >= 1.0) = 0.34089.
    assert math.isclose(p_from_t(1.0, 10), 0.34089, abs_tol=1e-4)
    # Large df converges to the normal: z=1.959964 -> 0.05.
    assert math.isclose(p_from_t(1.959964, 10**6), 0.05, abs_tol=1e-4)


def test_p_from_t_edges():
    assert p_from_t(0.0, 5) == 1.0
    assert p_from_t(-2.5, 30) == p_from_t(2.5, 30)  # two-sided symmetry
    assert math.isnan(p_from_t(float("nan"), 10))  # error trials carry no t


def test_p_from_t_rejects_bad_df():
    import pytest

    with pytest.raises(ValueError):
        p_from_t(1.0, 0)


# --------------------------------------------------------------------------- #
# bh_fdr: canonical Benjamini-Hochberg (1995) worked example, 15 p-values
# --------------------------------------------------------------------------- #
BH_PVALS = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
            0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]


def test_bh_canonical_example_q05_rejects_four():
    mask = bh_fdr(BH_PVALS, q=0.05)
    # k = max{i: p_(i) <= (i/15)*0.05} = 4 -> the four smallest are rejected.
    assert mask.sum() == 4
    assert list(np.nonzero(mask)[0]) == [0, 1, 2, 3]


def test_bh_canonical_example_q10_rejects_nine():
    # At the pre-registered q=0.10: p_(9)=0.0459 <= 9/15*0.10=0.06 -> k=9.
    assert bh_fdr(BH_PVALS, q=0.10).sum() == 9


def test_bh_mask_alignment_is_input_order():
    # Shuffled input: the mask marks the same VALUES regardless of position.
    p = [0.5719, 0.0001, 0.3240, 0.0004]
    mask = bh_fdr(p, q=0.05)
    assert list(mask) == [False, True, False, True]


def test_bh_nan_counts_as_a_trial_but_never_passes():
    # Two borderline p-values pass alone...
    assert bh_fdr([0.05, 0.09], q=0.10).sum() == 2
    # ...but adding two error trials (NaN p) raises the bar and kills both:
    # honest trial accounting means a failed trial still spends a trial.
    mask = bh_fdr([0.05, 0.09, float("nan"), float("nan")], q=0.10)
    assert mask.sum() == 0
    assert len(mask) == 4


def test_bh_empty_input():
    assert bh_fdr([], q=0.10).sum() == 0


# --------------------------------------------------------------------------- #
# deflated_sharpe: Bailey & Lopez de Prado (2014) worked example
# --------------------------------------------------------------------------- #
def test_dsr_paper_reference_value():
    # The paper's example: annualized SR 2.5 over 1250 daily obs, skew -3,
    # kurtosis 10, best of N=100 trials with cross-trial variance of the
    # ANNUALIZED SR = 0.5. In daily units: sr=2.5/sqrt(250), var=0.5/250.
    # Published result: DSR ~= 0.9004.
    dsr = deflated_sharpe(
        sr=2.5 / math.sqrt(250), n_obs=1250, skew=-3.0, kurt=10.0,
        n_trials=100, var_trials_sr=0.5 / 250,
    )
    assert math.isclose(dsr, 0.9004, abs_tol=1e-3)


def test_dsr_single_trial_has_no_deflation():
    # n_trials=1 -> SR0=0 -> plain probabilistic Sharpe ratio; positive SR > 0.5.
    dsr = deflated_sharpe(sr=0.1, n_obs=252, skew=0.0, kurt=3.0,
                          n_trials=1, var_trials_sr=0.0)
    assert 0.5 < dsr < 1.0


def test_dsr_more_trials_deflate_harder():
    kw = dict(sr=0.1, n_obs=252, skew=0.0, kurt=3.0, var_trials_sr=0.002)
    assert deflated_sharpe(n_trials=100, **kw) < deflated_sharpe(n_trials=10, **kw)


def test_dsr_degenerate_inputs_are_nan():
    assert math.isnan(deflated_sharpe(0.1, 1, 0.0, 3.0, 10, 0.001))  # n_obs < 2
    assert math.isnan(deflated_sharpe(0.1, 252, 0.0, 3.0, 0, 0.001))  # no trials
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_stats.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'trading.alphasearch'`

- [ ] **Step 3: Implement the module**

Create `src/trading/alphasearch/__init__.py` (empty file).

Create `src/trading/alphasearch/stats.py`:

```python
"""Trials-aware statistics: t -> p-value, BH-FDR gate, Deflated Sharpe Ratio.

This repo has no scipy (deliberately), so the Student-t two-sided p-value is
computed from the regularized incomplete beta function, implemented with
math.lgamma plus the standard continued fraction (Numerical Recipes 6.4,
modified Lentz's method). The normal CDF / inverse CDF needed by the DSR come
from stdlib statistics.NormalDist. Every function is verified against
published reference values in tests/test_alphasearch_stats.py.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni, in the DSR E[max] term
_TINY = 1e-30
_MAX_ITER = 200
_CF_EPS = 1e-12


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (modified Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_EPS:
            return h
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def p_from_t(t: float, df: int) -> float:
    """Two-sided p-value of a Student-t statistic with df degrees of freedom.

    P(|T| >= |t|) = I_{df/(df+t^2)}(df/2, 1/2). NaN in -> NaN out (a journaled
    error trial has no t-stat; bh_fdr maps that NaN to p=1.0 itself).
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    if math.isnan(t):
        return float("nan")
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def bh_fdr(pvals: np.ndarray | list[float], q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg step-up: boolean pass mask aligned to input order.

    Reject H_(1)..H_(k) where k = max{i : p_(i) <= (i/n) * q}. NaN p-values
    (journaled error trials) are treated as p=1.0: they can never pass, but
    they DO count in n -- an error trial still spends a trial (spec section 6).
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    p = np.where(np.isnan(p), 1.0, p)
    order = np.argsort(p, kind="stable")
    passed = p[order] <= q * np.arange(1, n + 1) / n
    mask = np.zeros(n, dtype=bool)
    if passed.any():
        k = int(np.nonzero(passed)[0].max())
        mask[order[: k + 1]] = True
    return mask


def deflated_sharpe(
    sr: float,
    n_obs: int,
    skew: float,
    kurt: float,
    n_trials: int,
    var_trials_sr: float,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014). Advisory only.

    Probability that the candidate's true Sharpe exceeds zero after accounting
    for having selected the best of n_trials and for non-normal returns.
    ALL Sharpe inputs are PER-PERIOD (daily, non-annualized): `sr` is the
    candidate's daily Sharpe, `var_trials_sr` the cross-trial variance of the
    daily Sharpes. `kurt` is Pearson kurtosis (normal = 3).
    """
    if n_trials < 1 or n_obs < 2:
        return float("nan")
    if n_trials == 1:
        sr0 = 0.0  # inv_cdf(0) is -inf; a single trial has no selection bias
    else:
        sr0 = math.sqrt(max(var_trials_sr, 0.0)) * (
            (1 - _EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / n_trials)
            + _EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
        )
    denom = 1 - skew * sr + (kurt - 1) / 4.0 * sr * sr
    if denom <= 0:
        return float("nan")
    return _NORMAL.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_alphasearch_stats.py -v`
Expected: all 12 tests PASS

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check src/trading/alphasearch tests/test_alphasearch_stats.py`
Expected: clean

```bash
git add src/trading/alphasearch/__init__.py src/trading/alphasearch/stats.py tests/test_alphasearch_stats.py
git commit -m "Add alphasearch.stats: t->p, BH-FDR, deflated Sharpe [AI]

No scipy in this repo, so the Student-t p-value is computed via the
regularized incomplete beta (lgamma + Lentz continued fraction), verified
against t-table critical values; BH against the canonical 1995 worked
example; DSR against the Bailey & Lopez de Prado paper value (0.9004).
NaN p-values count as trials but never pass -- the honest-trial rule."
```

---

### Task 2: `evaluate.py` — promote the factor-regression core; `AlphaResult`

**Files:**
- Create: `src/trading/alphasearch/evaluate.py`
- Modify: `scripts/factor_regression.py` (becomes a thin CLI wrapper)
- Test: `tests/test_alphasearch_evaluate.py`
- Must keep passing UNMODIFIED: `tests/test_factor_regression.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (Tasks 8/9 and the script rely on these exact signatures):
  - `ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]` — `(beta, se, tstat, r2, n)`; raises `ValueError` when n <= k; raises `np.linalg.LinAlgError` on a singular design.
  - `parse_ff_csv(text: str) -> pd.DataFrame`
  - `load_factors(factors_dir: Path, refresh: bool = False) -> pd.DataFrame` — columns `Mkt-RF, SMB, HML, RF, Mom` (decimals, UTC index).
  - `RegressionResult` frozen dataclass — fields `names, beta, se, tstat, r2, adj_r2, n`; properties `alpha_daily`, `alpha_annual_pct`, `alpha_tstat`.
  - `run_regression(returns: pd.DataFrame, factors: pd.DataFrame, factor_names: list[str], *, subtract_rf: bool = True) -> RegressionResult` — `returns` must have a `strategy` column; `subtract_rf=False` regresses the raw series (self-financing L/S spread).
  - `ALL_FACTORS = ("Mkt-RF", "SMB", "HML", "Mom")`, `TRADING_DAYS = 252`.
  - `annualized_sharpe(returns: pd.Series) -> float` — mean/std(ddof=1) * sqrt(252) of the raw daily series.
  - `AlphaResult` frozen dataclass — fields `four_factor: RegressionResult`, `capm: RegressionResult`, `sharpe_annual: float`; properties `alpha_annual_pct`, `alpha_tstat`, `capm_alpha_annual_pct`, `capm_alpha_tstat`, `n`.
  - `evaluate_alpha(returns: pd.Series, factors: pd.DataFrame, *, self_financing: bool) -> AlphaResult` — L/S passes `self_financing=True` (raw series); long-only `False` (returns − RF).

The promoted function/dataclass bodies are copied VERBATIM from
`scripts/factor_regression.py` except: (a) `run_regression` gains the
`subtract_rf` keyword, (b) the module has no `ROOT`/`DEFAULT_FACTORS_DIR`
(callers pass `factors_dir`; the CLI defaults are the callers' concern).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alphasearch_evaluate.py`:

```python
"""Tests for the promoted factor-regression core + AlphaResult (no network)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from trading.alphasearch.evaluate import (
    ALL_FACTORS,
    annualized_sharpe,
    evaluate_alpha,
    run_regression,
)


def _utc_index(n, start="2020-01-02"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _factors(n, mkt, rf=0.0001):
    idx = _utc_index(n)
    return pd.DataFrame(
        {"Mkt-RF": mkt, "SMB": 0.0, "HML": 0.0, "Mom": 0.0, "RF": rf}, index=idx
    )


def test_run_regression_subtract_rf_false_uses_raw_series():
    # A self-financing spread built as exactly alpha + 0.5 * Mkt-RF (plus a
    # whisper of noise so standard errors are defined). With subtract_rf=False
    # the intercept recovers alpha; the default (True) would shift it by -RF.
    rng = np.random.default_rng(11)
    n = 750
    mkt = rng.normal(0.0005, 0.01, size=n)
    alpha = 0.0003
    spread = alpha + 0.5 * mkt + rng.normal(0.0, 1e-5, size=n)
    factors = _factors(n, mkt)
    returns = pd.DataFrame({"strategy": spread}, index=factors.index)

    raw = run_regression(returns, factors, ["Mkt-RF"], subtract_rf=False)
    assert abs(raw.alpha_daily - alpha) < 1e-5
    assert abs(raw.beta[1] - 0.5) < 1e-3

    excess = run_regression(returns, factors, ["Mkt-RF"])  # default True
    assert abs(excess.alpha_daily - (alpha - 0.0001)) < 1e-5


def test_evaluate_alpha_ls_vs_lo_rf_handling():
    rng = np.random.default_rng(5)
    n = 600
    mkt = rng.normal(0.0004, 0.01, size=n)
    factors = _factors(n, mkt)
    series = pd.Series(0.0002 + 1.0 * mkt + rng.normal(0, 1e-5, size=n),
                       index=factors.index)

    ls = evaluate_alpha(series, factors, self_financing=True)
    lo = evaluate_alpha(series, factors, self_financing=False)
    # Same series: the long-only treatment subtracts RF=0.0001 from the alpha.
    assert abs(ls.four_factor.alpha_daily - 0.0002) < 1e-5
    assert abs(lo.four_factor.alpha_daily - 0.0001) < 1e-5
    # Both models present, with CAPM for contrast, and n populated.
    assert ls.four_factor.names == ["alpha", *ALL_FACTORS]
    assert ls.capm.names == ["alpha", "Mkt-RF"]
    assert ls.n == n
    assert not math.isnan(ls.sharpe_annual)


def test_annualized_sharpe_known_value():
    # Alternating +1%/-0.5% forever: mean 0.0025, std(ddof=1) known.
    r = pd.Series([0.01, -0.005] * 50)
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert math.isclose(annualized_sharpe(r), expected, rel_tol=1e-12)
    assert math.isnan(annualized_sharpe(pd.Series([0.01])))  # too short


def test_script_reexports_are_the_package_objects():
    # The promotion contract: the script is a thin wrapper, not a fork.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import factor_regression as script

    from trading.alphasearch import evaluate as package

    assert script.ols is package.ols
    assert script.parse_ff_csv is package.parse_ff_csv
    assert script.run_regression is package.run_regression
    assert script.load_factors is package.load_factors
    assert script.RegressionResult is package.RegressionResult
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_evaluate.py -v`
Expected: FAIL at collection with `ImportError` (module `trading.alphasearch.evaluate` does not exist)

- [ ] **Step 3: Create `src/trading/alphasearch/evaluate.py`**

The long "why alpha, why these factors" essay moves here from the script
(the script keeps only its usage docstring). Bodies marked *verbatim* are
copied character-for-character from today's `scripts/factor_regression.py`.

```python
"""Factor regression -> alpha: the reusable core behind scripts/factor_regression.py.

ALPHA is the piece of a strategy's return that known, freely-buyable factor
exposures do NOT explain -- the intercept of an OLS regression of the daily
return series on the canonical Ken French daily factors (Mkt-RF, SMB, HML,
Mom, with RF for excess returns). A positive, statistically significant alpha
is the only trustworthy "real edge" measure; Sharpe-vs-benchmark is not
(docs/experiments.md section 9). This module always fits BOTH the four-factor
model and market-only CAPM: the gap between their alphas is the size/value/
momentum tilt a single-factor model mis-reads as skill.

RF handling is the caller's declaration, not a guess: a long/short spread is
self-financing, so its regression return is the RAW spread
(run_regression(..., subtract_rf=False)); a long-only series regresses its
excess return over RF (the default). evaluate_alpha() packages both models
plus the raw annualized Sharpe into an AlphaResult for the alphasearch sweep.

Standard errors are classical OLS (Newey-West is a documented v2; see the
script). Promoted from scripts/factor_regression.py -- the script remains the
CLI and re-imports everything from here, so there is exactly one copy of the
statistics.
"""

from __future__ import annotations

import datetime
import io
import math
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RESEARCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
MOMENTUM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)
RESEARCH_CSV = "F-F_Research_Data_Factors_daily.csv"
MOMENTUM_CSV = "F-F_Momentum_Factor_daily.csv"

ALL_FACTORS = ("Mkt-RF", "SMB", "HML", "Mom")
TRADING_DAYS = 252
# Ken French encodes missing observations as these percent sentinels; any real
# daily factor return is far above -99%, so a single threshold catches both.
MISSING_SENTINEL = -99.0
USER_AGENT = "trading-factor-regression/1.0 (research; stdlib urllib)"
_DATE_RE = re.compile(r"^\d{8}$")


# --------------------------------------------------------------------------- #
# OLS core (hand-rolled; no statsmodels/scipy in this repo)
# --------------------------------------------------------------------------- #
def ols(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Classical OLS. ``x`` is the full design matrix (constant column included).

    Returns ``(beta, se, tstat, r2, n)``:
      beta  = (X'X)^-1 X'y
      e     = y - X beta                      (residuals)
      s2    = e'e / (n - k)                    (unbiased error variance)
      Var(beta) = s2 (X'X)^-1 ; se = sqrt(diag) ; t = beta / se
      r2    = 1 - e'e / TSS
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = x.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than parameters ({k})")
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    ss_res = float(resid @ resid)
    sigma2 = ss_res / (n - k)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    tstat = beta / se
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, se, tstat, r2, n


# --------------------------------------------------------------------------- #
# Fama-French factor parsing / fetching
# --------------------------------------------------------------------------- #
def parse_ff_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French daily-factor CSV into a UTC-indexed decimal DataFrame.

    Format quirks handled: a multi-line text preamble, a header row like
    ``,Mkt-RF,SMB,HML,RF``, daily rows ``YYYYMMDD, <space-padded numbers>``,
    and -- LATER in the same file -- a blank line followed by an *annual* block
    (4-digit year rows). We keep ONLY rows whose first field is an 8-digit date,
    which excludes the annual block outright. Values are in PERCENT and are
    divided by 100; the -99.99/-999 missing sentinels become NaN. Header
    whitespace is stripped (the momentum file uses ``   Mom   ``).
    """
    header: list[str] | None = None
    dates: list[pd.Timestamp] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        first = fields[0]
        if _DATE_RE.match(first):
            when = datetime.datetime.strptime(first, "%Y%m%d").replace(tzinfo=datetime.UTC)
            values = [float(f) for f in fields[1:] if f != ""]
            dates.append(pd.Timestamp(when))
            rows.append(values)
        elif header is None and first == "":
            names = [f for f in fields[1:] if f != ""]
            if names:  # first ",Mkt-RF,..." style header wins; annual re-header ignored
                header = names
    if header is None or not rows:
        raise ValueError("no Fama-French header or daily rows found")
    width = len(header)
    # Guard against ragged rows: keep exactly the header's worth of columns.
    trimmed = [r[:width] for r in rows]
    frame = pd.DataFrame(trimmed, index=pd.DatetimeIndex(dates, name="date"), columns=header)
    frame = frame.where(frame > MISSING_SENTINEL)  # sentinels -> NaN
    return frame / 100.0  # percent -> decimal


def _fetch_ff_csv(url: str, cache_path: Path, refresh: bool) -> str:
    """Download+extract a Ken French zip's CSV, caching the extracted text.

    After the first successful fetch the cache makes re-runs fully offline;
    ``--refresh`` forces a re-download.
    """
    if cache_path.exists() and not refresh:
        return cache_path.read_text()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:  # noqa: S310 (trusted canonical host)
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"no CSV inside zip at {url}")
        text = archive.read(members[0]).decode("latin-1")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text


def load_factors(factors_dir: Path, refresh: bool = False) -> pd.DataFrame:
    """Return a UTC-indexed frame with columns Mkt-RF, SMB, HML, RF, Mom (decimals)."""
    research = parse_ff_csv(
        _fetch_ff_csv(RESEARCH_URL, factors_dir / RESEARCH_CSV, refresh)
    )
    momentum = parse_ff_csv(
        _fetch_ff_csv(MOMENTUM_URL, factors_dir / MOMENTUM_CSV, refresh)
    )
    return research.join(momentum, how="inner")


# --------------------------------------------------------------------------- #
# Alignment + regression
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegressionResult:
    names: list[str]  # ["alpha", <factor>, ...]
    beta: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    r2: float
    adj_r2: float
    n: int

    @property
    def alpha_daily(self) -> float:
        return float(self.beta[0])

    @property
    def alpha_annual_pct(self) -> float:
        return self.alpha_daily * TRADING_DAYS * 100.0

    @property
    def alpha_tstat(self) -> float:
        return float(self.tstat[0])


def run_regression(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    factor_names: list[str],
    *,
    subtract_rf: bool = True,
) -> RegressionResult:
    """Inner-join returns with factors, form the regression return, and regress.

    subtract_rf=True (default, unchanged behavior): regress the EXCESS return
    strategy - RF -- correct for a long-only or benchmark-relative series.
    subtract_rf=False: regress the raw series -- correct for a self-financing
    long/short spread, which already nets out the financing leg. RF must be
    present and non-NaN either way so both modes see identical observations.
    """
    merged = returns.join(factors, how="inner")
    needed = list(factor_names) + ["RF", "strategy"]
    missing = [c for c in needed if c not in merged.columns]
    if missing:
        raise ValueError(f"missing columns after join: {missing}")
    merged = merged.dropna(subset=needed)
    if merged.empty:
        raise ValueError("no overlapping dates between returns and factors")
    if subtract_rf:
        y = (merged["strategy"] - merged["RF"]).to_numpy()
    else:
        y = merged["strategy"].to_numpy()
    columns = [np.ones(len(merged))] + [merged[f].to_numpy() for f in factor_names]
    x = np.column_stack(columns)
    beta, se, tstat, r2, n = ols(x, y)
    k = x.shape[1]
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else float("nan")
    return RegressionResult(
        names=["alpha", *factor_names],
        beta=beta,
        se=se,
        tstat=tstat,
        r2=r2,
        adj_r2=adj_r2,
        n=n,
    )


# --------------------------------------------------------------------------- #
# AlphaResult: the alphasearch evaluation unit (spec section 3.4)
# --------------------------------------------------------------------------- #
def annualized_sharpe(returns: pd.Series) -> float:
    """Raw annualized Sharpe of a daily return series (0% cash rate).

    Reported for tradability context only; the gate statistic is the
    four-factor alpha t, never this number.
    """
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(r.mean()) / sd * math.sqrt(TRADING_DAYS)


@dataclass(frozen=True)
class AlphaResult:
    """CAPM + four-factor decomposition of one return series."""

    four_factor: RegressionResult
    capm: RegressionResult
    sharpe_annual: float

    @property
    def alpha_annual_pct(self) -> float:
        return self.four_factor.alpha_annual_pct

    @property
    def alpha_tstat(self) -> float:
        return self.four_factor.alpha_tstat

    @property
    def capm_alpha_annual_pct(self) -> float:
        return self.capm.alpha_annual_pct

    @property
    def capm_alpha_tstat(self) -> float:
        return self.capm.alpha_tstat

    @property
    def n(self) -> int:
        return self.four_factor.n


def evaluate_alpha(
    returns: pd.Series, factors: pd.DataFrame, *, self_financing: bool
) -> AlphaResult:
    """Regress one daily return series on CAPM and the Carhart four factors.

    self_financing=True (the L/S spread): regress the raw series.
    self_financing=False (long-only): regress returns - RF.
    """
    frame = returns.rename("strategy").to_frame()
    subtract = not self_financing
    four = run_regression(frame, factors, list(ALL_FACTORS), subtract_rf=subtract)
    capm = run_regression(frame, factors, ["Mkt-RF"], subtract_rf=subtract)
    return AlphaResult(
        four_factor=four, capm=capm, sharpe_annual=annualized_sharpe(returns)
    )
```

Note: `ols`, `parse_ff_csv`, `_fetch_ff_csv`, `load_factors`, and
`RegressionResult` above are character-for-character copies of today's
`scripts/factor_regression.py` — zero statistical drift is the point of the
promotion. Diff them against the script before deleting the originals in the
next step.

- [ ] **Step 4: Rewrite `scripts/factor_regression.py` as the thin wrapper**

Replace the whole file with (the `import ... as ...` self-alias form marks
deliberate re-exports so ruff F401 stays quiet and
`tests/test_factor_regression.py`'s imports keep resolving unchanged):

```python
"""Decompose a strategy's returns into factor exposures (betas) plus ALPHA.

Thin CLI over trading.alphasearch.evaluate, which owns the statistics (OLS,
Ken French factor fetching/parsing, RegressionResult) -- see that module's
docstring for the full "why alpha, not Sharpe-vs-benchmark" story. This
script only loads a returns CSV, fetches/caches factors, and prints the
four-factor vs CAPM contrast.

Two-step workflow to decompose a strategy:

    trading backtest --venue equities --walk-forward --dump-returns rets.csv ...
    uv run python scripts/factor_regression.py --returns rets.csv

The first command writes daily strategy/benchmark returns; this script fetches +
caches the factors (offline after the first run) and prints the decomposition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trading.alphasearch.evaluate import (
    ALL_FACTORS as ALL_FACTORS,
)
from trading.alphasearch.evaluate import (
    MISSING_SENTINEL as MISSING_SENTINEL,
)
from trading.alphasearch.evaluate import (
    RegressionResult as RegressionResult,
)
from trading.alphasearch.evaluate import (
    TRADING_DAYS as TRADING_DAYS,
)
from trading.alphasearch.evaluate import (
    load_factors as load_factors,
)
from trading.alphasearch.evaluate import (
    ols as ols,
)
from trading.alphasearch.evaluate import (
    parse_ff_csv as parse_ff_csv,
)
from trading.alphasearch.evaluate import (
    run_regression as run_regression,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FACTORS_DIR = ROOT / "data" / "factors"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def load_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns or "strategy" not in frame.columns:
        raise ValueError("returns CSV must have at least 'date' and 'strategy' columns")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.set_index("date")


def _verdict(result: RegressionResult) -> str:
    sig = "significant (t>2)" if abs(result.alpha_tstat) > 2 else "not distinguishable from zero"
    return (
        f"ALPHA: {result.alpha_annual_pct:+.1f}%/yr, "
        f"t={result.alpha_tstat:+.1f} — {sig}"
    )


def _print_model(title: str, result: RegressionResult) -> None:
    print(f"\n{title}  (N={result.n})")
    print(f"  {'term':<10}{'coef':>14}{'t-stat':>10}   note")
    for i, name in enumerate(result.names):
        if name == "alpha":
            coef = f"{result.alpha_annual_pct:+.2f}%/yr"
            note = "annualized intercept (daily x 252)"
        else:
            coef = f"{result.beta[i]:+.4f}"
            flag = "significant loading" if abs(result.tstat[i]) > 2 else ""
            note = flag
        print(f"  {name:<10}{coef:>14}{result.tstat[i]:>+10.2f}   {note}")
    print(f"  R^2 = {result.r2:.3f}   adj R^2 = {result.adj_r2:.3f}")
    print(f"  {_verdict(result)}")
    loaders = [
        n for n, t in zip(result.names[1:], result.tstat[1:], strict=True) if abs(t) > 2
    ]
    print(f"  significant factor loadings: {', '.join(loaders) if loaders else 'none'}")


def report(returns: pd.DataFrame, factors: pd.DataFrame, factor_names: list[str]) -> None:
    full = run_regression(returns, factors, factor_names)
    _print_model(f"Factor model: {', '.join(factor_names)}", full)
    # Always show the market-only (CAPM) line for contrast: the gap between its
    # alpha and the full-model alpha is exactly the return the extra factors
    # explain (i.e. tilt, not skill).
    if factor_names != ["Mkt-RF"]:
        capm = run_regression(returns, factors, ["Mkt-RF"])
        _print_model("CAPM (market only)", capm)
        print(
            f"\nCAPM alpha {capm.alpha_annual_pct:+.1f}%/yr vs "
            f"{len(factor_names)}-factor alpha {full.alpha_annual_pct:+.1f}%/yr: "
            "the difference is the size/value/momentum tilt the market-only model "
            "mis-reads as alpha."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--returns",
        type=Path,
        required=True,
        help="returns CSV (date,strategy[,benchmark])",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=list(ALL_FACTORS),
        choices=list(ALL_FACTORS),
        help="factors to include (default: all four; e.g. 'Mkt-RF' alone for CAPM)",
    )
    parser.add_argument("--refresh", action="store_true", help="re-download the factor CSVs")
    parser.add_argument(
        "--factors-dir", type=Path, default=DEFAULT_FACTORS_DIR, help="factor cache directory"
    )
    args = parser.parse_args(argv)

    returns = load_returns(args.returns)
    factors = load_factors(args.factors_dir, refresh=args.refresh)
    print(
        f"loaded {len(returns)} return rows "
        f"({returns.index.min().date()}..{returns.index.max().date()}); "
        f"factors {factors.index.min().date()}..{factors.index.max().date()}"
    )
    report(returns, factors, list(args.factors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

(`load_returns`, `_verdict`, `_print_model`, `report`, `main` are today's
bodies unchanged — only the promoted statistics left the file.)

- [ ] **Step 5: Run both test files**

Run: `uv run pytest tests/test_alphasearch_evaluate.py tests/test_factor_regression.py -v`
Expected: ALL PASS — `test_factor_regression.py` passes UNMODIFIED (its
`from factor_regression import load_returns, ols, parse_ff_csv, run_regression`
now resolves through the wrapper's re-exports).

- [ ] **Step 6: Verify the script CLI still works end-to-end (cached factors exist under `data/factors/`)**

Run:
```bash
printf 'date,strategy\n2024-01-02,0.01\n2024-01-03,-0.002\n2024-01-04,0.004\n2024-01-05,0.001\n2024-01-08,0.003\n2024-01-09,-0.001\n' > /tmp/claude-rets.csv
uv run python scripts/factor_regression.py --returns /tmp/claude-rets.csv 2>&1 | head -20
```
Expected: the familiar report (factor model table, CAPM contrast) — no traceback.

- [ ] **Step 7: Lint and commit**

Run: `uv run ruff check src/trading/alphasearch scripts/factor_regression.py tests/test_alphasearch_evaluate.py`
Expected: clean

```bash
git add src/trading/alphasearch/evaluate.py scripts/factor_regression.py tests/test_alphasearch_evaluate.py
git commit -m "Promote factor-regression core into trading.alphasearch.evaluate [AI]

The sweep needs ols/load_factors/run_regression as a library; keeping the
stats in a script would mean sys.path hacks or a fork. The script is now a
thin CLI re-exporting the same objects (identity-tested), so existing tests
and CLI behavior are unchanged. run_regression gains subtract_rf=False for
self-financing L/S spreads; evaluate_alpha packages 4F+CAPM+Sharpe."
```

---

### Task 3: `panel.py` — bars, `PanelData`/`PanelView`, decision calendar

**Files:**
- Create: `src/trading/alphasearch/panel.py`
- Test: `tests/test_alphasearch_panel.py`

**Interfaces:**
- Consumes: `trading.symbols.load_symbol_allowlist(path) -> frozenset[str]` (existing).
- Produces (Tasks 4/5/6/8 rely on these exact signatures):
  - `PanelError(ValueError)`
  - `PanelData` frozen dataclass — fields `closes: dict[str, pd.Series]` (full-span adjusted closes, sorted tz-aware UTC index), `options: dict[str, pd.DataFrame]` (added Task 5; `{}` for now), `fundamentals: dict[str, pd.DataFrame]` (added Task 5; `{}` for now), `symbols: tuple[str, ...]`, `corrupt_cells: int = 0`; methods `view(as_of: pd.Timestamp) -> PanelView` and `decision_dates(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]`.
  - `PanelView` — properties `symbols`, `as_of`; methods `closes(symbol) -> pd.Series` (truncated to <= as_of), `last_close(symbol) -> float` (NaN when none). Task 5 adds `option_row` / `fundamentals_row`.
  - `load_closes(cache_dir: Path, symbols: Iterable[str]) -> dict[str, pd.Series]`

This task ships price-only assembly; Task 5 extends the same file with
options + fundamentals. The PIT rule is STRUCTURAL: signal functions (Task 4)
receive only a `PanelView`, whose every accessor truncates at `as_of` via
`searchsorted` — the same `side="right"` convention as
`trading.signals.engine.FeaturePanel.gather` and `signal_scan.load_panel`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alphasearch_panel.py`:

```python
"""PanelData/PanelView: PIT truncation and the monthly decision calendar."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trading.alphasearch.panel import PanelData, PanelError, load_closes


def _closes(dates: list[str], values: list[float]) -> pd.Series:
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dates])
    return pd.Series(values, index=idx, dtype="float64", name="close")


def _panel(closes: dict[str, pd.Series]) -> PanelData:
    return PanelData(
        closes=closes, options={}, fundamentals={}, symbols=tuple(sorted(closes))
    )


def test_view_truncates_closes_at_as_of():
    s = _closes(["2020-01-02", "2020-01-03", "2020-01-06"], [1.0, 2.0, 3.0])
    panel = _panel({"AAA": s})
    view = panel.view(pd.Timestamp("2020-01-03", tz="UTC"))
    got = view.closes("AAA")
    assert list(got.values) == [1.0, 2.0]  # the 01-06 bar is unreachable
    assert view.last_close("AAA") == 2.0


def test_view_between_bars_uses_last_prior_bar():
    s = _closes(["2020-01-02", "2020-01-06"], [1.0, 2.0])
    view = _panel({"AAA": s}).view(pd.Timestamp("2020-01-04", tz="UTC"))
    assert view.last_close("AAA") == 1.0


def test_view_before_history_is_empty_and_nan():
    s = _closes(["2020-01-02"], [1.0])
    view = _panel({"AAA": s}).view(pd.Timestamp("2019-12-31", tz="UTC"))
    assert view.closes("AAA").empty
    assert math.isnan(view.last_close("AAA"))
    assert math.isnan(view.last_close("MISSING"))


def test_view_rejects_naive_as_of():
    panel = _panel({"AAA": _closes(["2020-01-02"], [1.0])})
    with pytest.raises(ValueError):
        panel.view(pd.Timestamp("2020-01-02"))


def test_decision_dates_first_trading_session_per_month():
    # AAA misses Feb 3; BBB trades it -> the UNION calendar supplies Feb 3.
    a = _closes(["2020-01-02", "2020-01-03", "2020-02-04", "2020-03-02"], [1, 2, 3, 4])
    b = _closes(["2020-01-03", "2020-02-03", "2020-03-02"], [1, 2, 3])
    panel = _panel({"AAA": a, "BBB": b})
    got = panel.decision_dates(
        pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-03-31", tz="UTC")
    )
    assert [d.date().isoformat() for d in got] == ["2020-01-02", "2020-02-03", "2020-03-02"]


def test_decision_dates_respects_window_bounds():
    a = _closes(["2020-01-02", "2020-02-03", "2020-03-02"], [1, 2, 3])
    panel = _panel({"AAA": a})
    got = panel.decision_dates(
        pd.Timestamp("2020-02-01", tz="UTC"), pd.Timestamp("2020-02-28", tz="UTC")
    )
    assert [d.date().isoformat() for d in got] == ["2020-02-03"]
    assert panel.decision_dates(
        pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-12-31", tz="UTC")
    ) == ()


def test_load_closes_reads_parquet_and_skips_missing(tmp_path):
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-02", tz="UTC")])
    frame = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.5], "volume": [10.0]},
        index=idx,
    )
    frame.to_parquet(tmp_path / "AAA.parquet")
    got = load_closes(tmp_path, ["AAA", "NOPE"])
    assert set(got) == {"AAA"}
    assert got["AAA"].iloc[0] == 1.5


def test_panel_error_is_a_value_error():
    assert issubclass(PanelError, ValueError)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_panel.py -v`
Expected: FAIL at collection with `ImportError` (no `trading.alphasearch.panel`)

- [ ] **Step 3: Create `src/trading/alphasearch/panel.py`**

```python
"""Point-in-time panel assembly for the alpha-search engine (spec section 3.2).

Generalizes scripts/signal_scan.py::load_panel: Tiingo parquet bar caches,
options samples.jsonl cells (Task 5), and the fundamentals store (Task 5),
unified behind one PIT discipline. Signal scoring and forward-return
construction are SEPARATE passes over this data: a signal fn only ever
receives a PanelView, whose every accessor truncates at as_of via
index.searchsorted(side="right") (the repo-wide convention -- see
trading.signals.engine.FeaturePanel.gather), so a signal structurally cannot
reach forward data. The sort (trading.alphasearch.sort) reads panel.closes
directly for forward returns -- signals never see that pass.

Pure I/O + indexing: no clock reads; as_of is always a parameter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


class PanelError(ValueError):
    """Panel assembly refused (missing inputs, unusable universe)."""


def load_closes(cache_dir: Path, symbols: Iterable[str]) -> dict[str, pd.Series]:
    """Adjusted-close series per symbol from a Tiingo parquet cache.

    A symbol without a cached parquet is simply absent from the result --
    the missing-data rule (spec section 5.5) drops it from every
    cross-section rather than fabricating history.
    """
    out: dict[str, pd.Series] = {}
    for symbol in sorted(set(symbols)):
        path = cache_dir / f"{symbol}.parquet"
        if path.exists():
            out[symbol] = pd.read_parquet(path)["close"]
    return out


class PanelView:
    """Read-only as-of window onto a PanelData.

    Every accessor truncates to data timestamped at or before as_of. Signal
    functions receive ONLY this view (never the PanelData), which is what
    makes the no-look-ahead guarantee structural rather than by-convention.
    """

    def __init__(self, panel: PanelData, as_of: pd.Timestamp) -> None:
        self._panel = panel
        self.as_of = as_of

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._panel.symbols

    def closes(self, symbol: str) -> pd.Series:
        """The symbol's closes up to and including as_of (empty if none)."""
        series = self._panel.closes.get(symbol)
        if series is None:
            return pd.Series(dtype="float64")
        pos = series.index.searchsorted(self.as_of, side="right")
        return series.iloc[:pos]

    def last_close(self, symbol: str) -> float:
        closes = self.closes(symbol)
        return float(closes.iloc[-1]) if len(closes) else float("nan")


@dataclass(frozen=True)
class PanelData:
    """One universe's data: full-span series keyed by symbol.

    Frames deliberately extend past any given decision date -- truncation is
    PanelView's job (identical to the ranker registry's contract for bars and
    fundamentals). `options` and `fundamentals` are populated in Task 5.
    """

    closes: dict[str, pd.Series]
    options: dict[str, pd.DataFrame] = field(default_factory=dict)
    fundamentals: dict[str, pd.DataFrame] = field(default_factory=dict)
    symbols: tuple[str, ...] = ()
    corrupt_cells: int = 0

    def view(self, as_of: pd.Timestamp) -> PanelView:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be tz-aware UTC")
        return PanelView(self, as_of)

    def decision_dates(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.Timestamp, ...]:
        """First trading session of each month in [start, end].

        "Trading session" = any date on which at least one panel symbol has a
        bar (the union calendar), so one symbol's missing day never shifts the
        whole universe's rebalance date.
        """
        union = sorted({d for s in self.closes.values() for d in s.index})
        in_window = [d for d in union if start <= d <= end]
        if not in_window:
            return ()
        firsts: dict[str, pd.Timestamp] = {}
        for date in in_window:  # ascending, so the first hit per month sticks
            firsts.setdefault(date.strftime("%Y-%m"), date)
        return tuple(firsts[m] for m in sorted(firsts))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_alphasearch_panel.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check src/trading/alphasearch tests/test_alphasearch_panel.py`
Expected: clean

```bash
git add src/trading/alphasearch/panel.py tests/test_alphasearch_panel.py
git commit -m "Add alphasearch.panel: PIT PanelData/PanelView + decision calendar [AI]

Signals will only ever receive a PanelView whose accessors truncate at
as_of (searchsorted side='right', the repo convention) -- making
no-look-ahead structural instead of by-convention. Monthly decision dates
come from the union bar calendar so one symbol's gap can't shift the
universe's rebalance."
```

---

### Task 4: `spec.py` — `SignalSpec` registry + price signals

**Files:**
- Create: `src/trading/alphasearch/spec.py`
- Test: `tests/test_alphasearch_spec.py`

**Interfaces:**
- Consumes: `PanelView` from Task 3 (`.symbols`, `.closes(symbol)`, `.last_close(symbol)`).
- Produces (Tasks 5/6/8/11 rely on these exact names):
  - `SignalFn = Callable[[PanelView, pd.Timestamp], pd.Series]` — per-symbol score indexed by symbol, float64, HIGHER = more attractive to be long; NaN = symbol lacks this signal's inputs on this date (dropped from that date's cross-section, never imputed).
  - `SignalSpec` frozen dataclass — `name: str`, `fn: SignalFn`, `requires_options: bool = False`, `requires_fundamentals: bool = False`.
  - `SIGNALS: dict[str, SignalSpec]` — after Task 4: 7 of the 8 price-family signals `mom21, mom63, mom126, mom252, rev5, rvol21, disthigh` (`vrp` needs `atm_iv` from options cells, so it lands with the options family in Task 5). After Task 5: all 16.
  - Price helpers reused by Task 5's `vrp`: `_trail(closes: pd.Series, window: int) -> float`, `_rvol(closes: pd.Series, window: int = 21) -> float`, `_disthigh(closes: pd.Series, window: int = 252) -> float` — metric semantics copied from `scripts/signal_scan.py` (`trail`/`rvol`/`disthigh`), operating on an already-truncated close series.

Sign conventions are PART of the wrapper and recorded next to each
registration (spec section 3.1). Decisions locked here:

| signal | score | direction rationale |
|---|---|---|
| mom21/63/126/252 | `+trail(w)` | momentum: recent winners attractive |
| rev5 | `-trail(5)` | short-term reversal: recent losers attractive |
| rvol21 | `-rvol(21)` | low-vol anomaly: quiet names attractive |
| disthigh | `+disthigh(252)` | near the 52-week high = momentum-like |

(The gate is |t|, so a wrong direction guess flips the sign of alpha, not the
significance — but the recorded direction is what a survivor would trade.)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alphasearch_spec.py`:

```python
"""Seed signal registry: hand-checked scores on tiny deterministic panels."""

from __future__ import annotations

import math

import pandas as pd

from trading.alphasearch.panel import PanelData
from trading.alphasearch.spec import SIGNALS


def _geometric_panel(n_days: int = 300, rates: dict[str, float] | None = None) -> PanelData:
    """Each symbol's close grows at a constant daily rate -> every price
    metric has a closed-form value."""
    rates = rates or {"SLOW": 0.001, "FAST": 0.01}
    idx = pd.date_range("2020-01-02", periods=n_days, freq="B", tz="UTC")
    closes = {
        sym: pd.Series([100.0 * (1 + r) ** i for i in range(n_days)], index=idx)
        for sym, r in rates.items()
    }
    return PanelData(closes=closes, options={}, fundamentals={},
                     symbols=tuple(sorted(closes)))


def _score(name: str, panel: PanelData, as_of: pd.Timestamp) -> pd.Series:
    spec = SIGNALS[name]
    return spec.fn(panel.view(as_of), as_of)


def test_price_signals_registered_with_no_data_requirements():
    for name in ("mom21", "mom63", "mom126", "mom252", "rev5", "rvol21", "disthigh"):
        spec = SIGNALS[name]
        assert spec.name == name
        assert not spec.requires_options
        assert not spec.requires_fundamentals


def test_mom21_closed_form_and_ordering():
    panel = _geometric_panel()
    as_of = panel.closes["SLOW"].index[-1]
    scores = _score("mom21", panel, as_of)
    assert math.isclose(scores["SLOW"], 1.001**21 - 1, rel_tol=1e-9)
    assert math.isclose(scores["FAST"], 1.01**21 - 1, rel_tol=1e-9)
    assert scores["FAST"] > scores["SLOW"]  # higher momentum = higher score


def test_rev5_is_negated_trailing_return():
    panel = _geometric_panel()
    as_of = panel.closes["SLOW"].index[-1]
    scores = _score("rev5", panel, as_of)
    assert math.isclose(scores["FAST"], -(1.01**5 - 1), rel_tol=1e-9)
    assert scores["FAST"] < scores["SLOW"]  # recent winners UNattractive


def test_rvol21_is_negated_and_zero_for_constant_growth():
    # Constant growth rate -> identical daily returns -> zero realized vol.
    panel = _geometric_panel()
    as_of = panel.closes["SLOW"].index[-1]
    scores = _score("rvol21", panel, as_of)
    assert scores["SLOW"] == 0.0
    assert scores["FAST"] == 0.0


def test_disthigh_zero_at_high_negative_below():
    idx = pd.date_range("2020-01-02", periods=30, freq="B", tz="UTC")
    rising = pd.Series([100.0 + i for i in range(30)], index=idx)
    dipped = pd.Series([100.0 + i for i in range(29)] + [64.5], index=idx)
    panel = PanelData(closes={"UP": rising, "DIP": dipped}, options={},
                      fundamentals={}, symbols=("DIP", "UP"))
    scores = _score("disthigh", panel, idx[-1])
    assert scores["UP"] == 0.0  # at its high
    assert math.isclose(scores["DIP"], 64.5 / 128.0 - 1, rel_tol=1e-9)
    assert scores["UP"] > scores["DIP"]  # near-high names attractive


def test_insufficient_history_yields_nan_not_crash():
    panel = _geometric_panel(n_days=10)
    as_of = panel.closes["SLOW"].index[-1]
    assert math.isnan(_score("mom21", panel, as_of)["SLOW"])
    assert math.isnan(_score("rvol21", panel, as_of)["SLOW"])
    # mom{21..252} on 10 bars: all NaN; rev5 still computable (needs 6 bars).
    assert not math.isnan(_score("rev5", panel, as_of)["SLOW"])


def test_scores_cover_every_panel_symbol():
    panel = _geometric_panel()
    as_of = panel.closes["SLOW"].index[-1]
    for name in ("mom21", "rev5", "rvol21", "disthigh"):
        scores = _score(name, panel, as_of)
        assert list(scores.index) == list(panel.symbols)
        assert scores.dtype == "float64"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_spec.py -v`
Expected: FAIL at collection with `ImportError` (no `trading.alphasearch.spec`)

- [ ] **Step 3: Create `src/trading/alphasearch/spec.py`**

```python
"""SignalSpec + the seed SIGNALS registry (spec section 3.1).

A signal is a pure function (PanelView, as_of) -> per-symbol float score,
HIGHER = more attractive to be long. The sign convention is part of the
wrapper and is documented at each registration -- e.g. rev5 registers the
NEGATED trailing 5-day return because short-term reversal makes recent
losers attractive. NaN means "this symbol lacks the signal's inputs on this
date": the sort drops it from that date's cross-section (spec section 5.5),
never imputes.

PIT contract: fn receives a PanelView (never the raw PanelData), whose
accessors truncate at as_of -- forward data is structurally unreachable.
Price-metric semantics (trail / rvol / disthigh) are copied from
scripts/signal_scan.py so alphasearch scores agree with the scanner's panel.

Task 4 registers the price family; Task 5 completes the registry with the
options family (vrp, hedge, excite, atm_iv, smile, atm_spread) and the
fundamentals family (gross_profitability, earnings_yield, book_to_market).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from trading.alphasearch.panel import PanelView

SignalFn = Callable[[PanelView, pd.Timestamp], pd.Series]


@dataclass(frozen=True)
class SignalSpec:
    name: str
    fn: SignalFn
    requires_options: bool = False
    requires_fundamentals: bool = False


SIGNALS: dict[str, SignalSpec] = {}


def _register(
    name: str,
    fn: SignalFn,
    *,
    requires_options: bool = False,
    requires_fundamentals: bool = False,
) -> None:
    SIGNALS[name] = SignalSpec(name, fn, requires_options, requires_fundamentals)


# --------------------------------------------------------------------------- #
# Price metrics -- semantics identical to scripts/signal_scan.py, but applied
# to an already-truncated close series (PanelView.closes), so the "position
# at or before the decision date" is simply the last element.
# --------------------------------------------------------------------------- #
def _trail(closes: pd.Series, window: int) -> float:
    """Trailing `window`-bar return ending at the last close."""
    p = len(closes) - 1
    if p - window < 0:
        return math.nan
    return float(closes.iloc[p] / closes.iloc[p - window] - 1)


def _rvol(closes: pd.Series, window: int = 21) -> float:
    """Annualized realized vol of the `window` bars BEFORE the last close."""
    p = len(closes) - 1
    if p < window:
        return math.nan
    return float(closes.iloc[p - window : p].pct_change().std() * math.sqrt(252))


def _disthigh(closes: pd.Series, window: int = 252) -> float:
    """Distance from the trailing `window`-bar high (<= 0; 0 = at the high)."""
    p = len(closes) - 1
    if p < 0:
        return math.nan
    return float(closes.iloc[p] / closes.iloc[max(0, p - window) : p + 1].max() - 1)


def _price_signal(metric: Callable[[pd.Series], float]) -> SignalFn:
    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores = {symbol: metric(view.closes(symbol)) for symbol in view.symbols}
        return pd.Series(scores, dtype="float64")

    return fn


def _mom(window: int) -> SignalFn:
    def metric(closes: pd.Series, _window: int = window) -> float:
        return _trail(closes, _window)

    return _price_signal(metric)


# Price family. Direction rationale recorded per registration (spec 3.1).
_register("mom21", _mom(21))  # momentum: recent winners attractive
_register("mom63", _mom(63))
_register("mom126", _mom(126))
_register("mom252", _mom(252))
# Short-term reversal: recent LOSERS attractive -> negate the trailing return.
_register("rev5", _price_signal(lambda c: -_trail(c, 5)))
# Low-vol anomaly: quiet names attractive -> negate realized vol.
_register("rvol21", _price_signal(lambda c: -_rvol(c)))
# Proximity to the 52-week high is momentum-like; disthigh is <= 0 with 0 at
# the high, so raw sign already puts near-high names on top.
_register("disthigh", _price_signal(_disthigh))
```

Note: `-_rvol` makes the constant-growth test value `-0.0`; `-0.0 == 0.0` in
Python so the test's `== 0.0` holds.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_alphasearch_spec.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check src/trading/alphasearch tests/test_alphasearch_spec.py`
Expected: clean

```bash
git add src/trading/alphasearch/spec.py tests/test_alphasearch_spec.py
git commit -m "Add alphasearch.spec: SignalSpec registry + price seed signals [AI]

Signals are pure (PanelView, as_of) -> score functions with the sign
convention (higher = attractive) baked into each wrapper and documented at
the registration site, so the leaderboard's L/S direction is always the
direction a survivor would actually trade. Metric semantics copied from
signal_scan.py so both tools agree."
```

---

### Task 5: options + fundamentals in `panel.py`; complete the 16-signal registry

**Files:**
- Modify: `src/trading/alphasearch/panel.py` (add `cell_metrics`, `load_options`, `options_from_cells`, `load_fundamentals`, `build_panel`, `PanelView.option_row`, `PanelView.fundamentals_row`, `OPTION_COLUMNS`, `MAX_OPTION_AGE_DAYS`)
- Modify: `src/trading/alphasearch/spec.py` (register `vrp` + options family + fundamentals family)
- Modify: `scripts/signal_scan.py` (its `_cell_metrics` becomes an import of the promoted `cell_metrics`)
- Test: extend `tests/test_alphasearch_panel.py` and `tests/test_alphasearch_spec.py`
- Must keep passing UNMODIFIED: `tests/test_signal_scan.py` (it imports `_cell_metrics` from the script)

**Interfaces:**
- Consumes: Task 3's `PanelData`/`PanelView`; Task 4's `_register`, `_price_signal` helpers; existing `trading.symbols.load_symbol_allowlist` and `trading.fundamentals.store.FundamentalsStore` (`.read(symbol) -> pd.DataFrame` indexed by tz-aware UTC FILED dates with columns incl. `gross_profitability`, `ttm_net_income`, `book_equity`, `shares_outstanding`).
- Produces (Tasks 8/11 rely on these exact signatures):
  - `cell_metrics(cell: dict) -> dict` — verbatim promotion of `signal_scan._cell_metrics` (keys `hedge, excite, atm_iv, otm_put_iv, otm_call_iv, smile, cp_vol, wing_vol, tot_vol, atm_spread`).
  - `OPTION_COLUMNS: list[str]` — those ten keys, in that order.
  - `MAX_OPTION_AGE_DAYS = 7` — a cell older than this at `as_of` is stale -> treated as missing.
  - `options_from_cells(cells: Iterable[dict]) -> dict[str, pd.DataFrame]` — per-symbol float64 frames indexed by UTC `decision_date`.
  - `load_options(samples: Path) -> tuple[dict[str, pd.DataFrame], int]` — `(frames, corrupt_count)`; corrupt/incomplete lines are skipped AND counted (spec §6).
  - `load_fundamentals(root: Path, symbols: Iterable[str]) -> dict[str, pd.DataFrame]` — non-empty store frames only; `{}` when `root` does not exist (never creates directories).
  - `build_panel(cache_dir: Path, samples: Path, fundamentals_dir: Path | None) -> PanelData` — universe = the samples allowlist ∩ cached bars; raises `PanelError` when samples are missing or no bars resolve.
  - `PanelView.option_row(symbol) -> pd.Series | None` — latest cell with `decision_date <= as_of`, `None` if none or stale.
  - `PanelView.fundamentals_row(symbol) -> pd.Series | None` — latest row FILED `<= as_of` (same cut as `trading.signals.quality.latest_filed_row`).
  - `SIGNALS` completed: + `vrp, hedge, excite, atm_iv, smile, atm_spread` (all `requires_options=True`) and `gross_profitability, earnings_yield, book_to_market` (all `requires_fundamentals=True`). 16 total.

Sign conventions locked for the new families (recorded at each registration):

| signal | score | direction rationale |
|---|---|---|
| vrp | `+(atm_iv − rvol21)` | rich vol premium = overpaid insurance on the name |
| hedge | `−skew_put_atm` | spec's example: less downside-hedged = attractive (the risk-veto intuition) |
| excite | `+cell excite` (= `−skew_put_call` already) | call-side richness = speculative demand |
| atm_iv | `−atm_iv` | lottery/vol-premium: high-IV names underperform |
| smile | `−smile` | convex smile = tail fear priced in |
| atm_spread | `+atm_spread` | option illiquidity premium — the one real mid-cap predictor (docs/experiments.md §9), even though it decomposed into SMB |
| gross_profitability | `+` | quality (as `quality.py`) |
| earnings_yield | `+ ttm_net_income / (shares × close)` | value (as `value.py`) |
| book_to_market | `+ book_equity / (shares × close)` | value (as `value.py`) |

- [ ] **Step 1: Write the failing panel tests**

Append to `tests/test_alphasearch_panel.py`. IMPORTANT: merge the imports
into the file's TOP import block (`import json` with stdlib, the panel names
into the existing `from trading.alphasearch.panel import ...`) — a mid-file
import is a ruff E402 error.

```python
# needed at the top of the file:
#   import json
#   from trading.alphasearch.panel import (
#       MAX_OPTION_AGE_DAYS, OPTION_COLUMNS, PanelData, PanelError,
#       build_panel, cell_metrics, load_closes, load_options, options_from_cells,
#   )

# --------------------------------------------------------------------------- #
# Options cells + fundamentals (Task 5)
# --------------------------------------------------------------------------- #


def _cell(symbol: str, date: str, *, atm_iv=0.30, put_iv=0.34, call_iv=0.28,
          skew_put_atm=0.05, skew_put_call=0.02) -> dict:
    return {
        "symbol": symbol,
        "decision_date": date,
        "skew_put_atm": skew_put_atm,
        "skew_put_call": skew_put_call,
        "contracts": [
            {"role": "atm", "bid": 4.0, "ask": 4.2, "mid": 4.1, "iv": atm_iv,
             "volume": 100},
            {"role": "otm_put", "iv": put_iv, "volume": 50},
            {"role": "otm_call", "iv": call_iv, "volume": 25},
        ],
    }


def test_cell_metrics_matches_signal_scan_semantics():
    m = cell_metrics(_cell("AAA", "2020-01-02"))
    assert m["hedge"] == 0.05
    assert m["excite"] == -0.02  # -skew_put_call
    assert m["atm_spread"] == (4.2 - 4.0) / 4.1
    assert math.isclose(m["smile"], (0.34 + 0.28) / 2 - 0.30, rel_tol=1e-12)
    assert set(OPTION_COLUMNS) == set(m)


def test_options_from_cells_builds_float_frames():
    frames = options_from_cells([_cell("AAA", "2020-01-02"), _cell("AAA", "2020-02-03")])
    f = frames["AAA"]
    assert list(f.columns) == OPTION_COLUMNS
    assert str(f.index[0].tz) == "UTC"
    assert f.dtypes.eq("float64").all()  # None from a missing leg becomes NaN


def test_load_options_skips_and_counts_corrupt_lines(tmp_path):
    path = tmp_path / "samples.jsonl"
    lines = [
        json.dumps(_cell("AAA", "2020-01-02")),
        '{"torn json',                       # corrupt: unparseable
        json.dumps({"decision_date": "2020-01-02"}),  # corrupt: no symbol
        json.dumps(_cell("BBB", "2020-01-02")),
        "",                                  # blank: ignored, not corrupt
    ]
    path.write_text("\n".join(lines) + "\n")
    frames, corrupt = load_options(path)
    assert set(frames) == {"AAA", "BBB"}
    assert corrupt == 2


def test_option_row_is_pit_and_staleness_capped():
    frames = options_from_cells([_cell("AAA", "2020-01-06", atm_iv=0.5)])
    idx = pd.date_range("2020-01-02", periods=40, freq="B", tz="UTC")
    closes = {"AAA": pd.Series(100.0, index=idx)}
    panel = PanelData(closes=closes, options=frames, fundamentals={}, symbols=("AAA",))
    # Before the cell: nothing to see.
    assert panel.view(pd.Timestamp("2020-01-03", tz="UTC")).option_row("AAA") is None
    # On/after the cell within the age cap: visible.
    row = panel.view(pd.Timestamp("2020-01-06", tz="UTC")).option_row("AAA")
    assert row["atm_iv"] == 0.5
    fresh_limit = pd.Timestamp("2020-01-06", tz="UTC") + pd.Timedelta(days=MAX_OPTION_AGE_DAYS)
    assert panel.view(fresh_limit).option_row("AAA") is not None
    # One day beyond the cap: stale -> missing, not forward-filled.
    assert panel.view(fresh_limit + pd.Timedelta(days=1)).option_row("AAA") is None


def test_fundamentals_row_filed_date_cut():
    idx = pd.date_range("2020-01-02", periods=10, freq="B", tz="UTC")
    filed = pd.DatetimeIndex([pd.Timestamp("2020-01-06", tz="UTC")])
    fund = pd.DataFrame({"gross_profitability": [0.4], "ttm_net_income": [5e6],
                         "book_equity": [2e7], "shares_outstanding": [1e6]}, index=filed)
    panel = PanelData(closes={"AAA": pd.Series(100.0, index=idx)},
                      options={}, fundamentals={"AAA": fund}, symbols=("AAA",))
    assert panel.view(pd.Timestamp("2020-01-03", tz="UTC")).fundamentals_row("AAA") is None
    row = panel.view(pd.Timestamp("2020-01-07", tz="UTC")).fundamentals_row("AAA")
    assert row["gross_profitability"] == 0.4


def test_build_panel_from_files(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    idx = pd.date_range("2020-01-02", periods=5, freq="B", tz="UTC")
    for sym in ("AAA", "BBB"):
        pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 2.0,
                      "volume": 10.0}, index=idx).to_parquet(cache / f"{sym}.parquet")
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps(_cell("AAA", "2020-01-02")) + "\n"
        + json.dumps(_cell("BBB", "2020-01-02")) + "\n"
        + json.dumps(_cell("NOBAR", "2020-01-02")) + "\n"
    )
    panel = build_panel(cache, samples, tmp_path / "no-fundamentals")
    assert panel.symbols == ("AAA", "BBB")  # NOBAR dropped: allowlist ∩ bars
    assert set(panel.options) == {"AAA", "BBB"}
    assert panel.fundamentals == {}  # absent store dir -> empty, NOT created
    assert not (tmp_path / "no-fundamentals").exists()
    assert panel.corrupt_cells == 0


def test_build_panel_missing_samples_refused(tmp_path):
    with pytest.raises(PanelError):
        build_panel(tmp_path, tmp_path / "nope.jsonl", None)
```

- [ ] **Step 2: Write the failing signal tests**

Append to `tests/test_alphasearch_spec.py`. IMPORTANT: add
`from trading.alphasearch.panel import PanelData, options_from_cells` to the
TOP import block (replacing the existing PanelData import) — a mid-file
import is a ruff E402 error.

```python
# --------------------------------------------------------------------------- #
# Options + fundamentals families (Task 5)
# --------------------------------------------------------------------------- #


def _options_panel() -> tuple[PanelData, pd.Timestamp]:
    idx = pd.date_range("2020-01-02", periods=60, freq="B", tz="UTC")
    closes = {s: pd.Series(100.0, index=idx) for s in ("AAA", "BBB")}
    as_of = idx[-1]
    date = as_of.date().isoformat()
    cells = [
        {"symbol": "AAA", "decision_date": date, "skew_put_atm": 0.05,
         "skew_put_call": 0.02, "contracts": [
             {"role": "atm", "bid": 4.0, "ask": 4.2, "mid": 4.1, "iv": 0.30,
              "volume": 100},
             {"role": "otm_put", "iv": 0.34, "volume": 50},
             {"role": "otm_call", "iv": 0.28, "volume": 25}]},
        {"symbol": "BBB", "decision_date": date, "skew_put_atm": 0.10,
         "skew_put_call": -0.01, "contracts": [
             {"role": "atm", "bid": 2.0, "ask": 2.4, "mid": 2.2, "iv": 0.50,
              "volume": 10},
             {"role": "otm_put", "iv": 0.60, "volume": 5},
             {"role": "otm_call", "iv": 0.44, "volume": 2}]},
    ]
    panel = PanelData(closes=closes, options=options_from_cells(cells),
                      fundamentals={}, symbols=("AAA", "BBB"))
    return panel, as_of


def test_options_family_signs():
    panel, as_of = _options_panel()
    hedge = _score("hedge", panel, as_of)
    assert hedge["AAA"] == -0.05 and hedge["BBB"] == -0.10  # -skew_put_atm
    assert hedge["AAA"] > hedge["BBB"]  # less-hedged name is more attractive
    excite = _score("excite", panel, as_of)
    assert excite["AAA"] == -0.02 and excite["BBB"] == 0.01
    atm_iv = _score("atm_iv", panel, as_of)
    assert atm_iv["AAA"] == -0.30 and atm_iv["BBB"] == -0.50
    smile = _score("smile", panel, as_of)
    assert math.isclose(smile["AAA"], -((0.34 + 0.28) / 2 - 0.30), rel_tol=1e-12)
    spread = _score("atm_spread", panel, as_of)
    assert math.isclose(spread["AAA"], (4.2 - 4.0) / 4.1, rel_tol=1e-12)
    assert spread["BBB"] > spread["AAA"]  # wider spread = higher score


def test_vrp_is_iv_minus_realized_vol():
    panel, as_of = _options_panel()
    # Flat closes -> realized vol 0 -> vrp == atm_iv.
    vrp = _score("vrp", panel, as_of)
    assert math.isclose(vrp["AAA"], 0.30, rel_tol=1e-12)
    assert math.isclose(vrp["BBB"], 0.50, rel_tol=1e-12)


def test_options_signals_nan_without_cells():
    panel, as_of = _options_panel()
    bare = PanelData(closes=panel.closes, options={}, fundamentals={},
                     symbols=panel.symbols)
    for name in ("hedge", "excite", "atm_iv", "smile", "atm_spread", "vrp"):
        assert _score(name, bare, as_of).isna().all()


def test_fundamentals_family_values_and_neutrality():
    idx = pd.date_range("2020-01-02", periods=10, freq="B", tz="UTC")
    closes = {"AAA": pd.Series(50.0, index=idx), "BBB": pd.Series(50.0, index=idx)}
    filed = pd.DatetimeIndex([idx[0]])
    fund = {"AAA": pd.DataFrame(
        {"gross_profitability": [0.4], "ttm_net_income": [5e6],
         "book_equity": [2.5e7], "shares_outstanding": [1e6]}, index=filed)}
    panel = PanelData(closes=closes, options={}, fundamentals=fund,
                      symbols=("AAA", "BBB"))
    as_of = idx[-1]
    gp = _score("gross_profitability", panel, as_of)
    assert gp["AAA"] == 0.4
    assert math.isnan(gp["BBB"])  # no filing -> NaN (dropped, not imputed)
    ey = _score("earnings_yield", panel, as_of)
    assert math.isclose(ey["AAA"], 5e6 / (1e6 * 50.0), rel_tol=1e-12)
    bm = _score("book_to_market", panel, as_of)
    assert math.isclose(bm["AAA"], 2.5e7 / (1e6 * 50.0), rel_tol=1e-12)


def test_registry_is_complete_with_correct_requirements():
    assert len(SIGNALS) == 16
    options_family = {"vrp", "hedge", "excite", "atm_iv", "smile", "atm_spread"}
    fundamentals_family = {"gross_profitability", "earnings_yield", "book_to_market"}
    for name, spec in SIGNALS.items():
        assert spec.requires_options == (name in options_family)
        assert spec.requires_fundamentals == (name in fundamentals_family)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_panel.py tests/test_alphasearch_spec.py -v`
Expected: FAIL with `ImportError: cannot import name 'cell_metrics'` (and the spec tests fail on missing registry entries)

- [ ] **Step 4: Extend `src/trading/alphasearch/panel.py`**

Add to the imports:

```python
import json

import numpy as np

from trading.fundamentals.store import FundamentalsStore
from trading.symbols import load_symbol_allowlist
```

Add after `PanelError` (constants + promoted cell metrics):

```python
MAX_OPTION_AGE_DAYS = 7  # a cell older than this at as_of is stale -> missing

OPTION_COLUMNS = [
    "hedge", "excite", "atm_iv", "otm_put_iv", "otm_call_iv",
    "smile", "cp_vol", "wing_vol", "tot_vol", "atm_spread",
]


def cell_metrics(cell: dict) -> dict:
    """The option-derived metrics for one samples.jsonl cell (NaN when a leg is
    missing so a partial cell never fabricates a value).

    Promoted verbatim from scripts/signal_scan.py::_cell_metrics; the script
    now imports it from here so there is exactly one definition.
    """
    d = {c["role"]: c for c in cell.get("contracts", [])}

    def iv(role):
        return d.get(role, {}).get("iv")

    def vol(role):
        return d.get(role, {}).get("volume") or 0

    atm = d.get("atm", {})
    put_iv, call_iv, atm_iv = iv("otm_put"), iv("otm_call"), iv("atm")
    rr = cell.get("skew_put_call")
    spread = ((atm["ask"] - atm["bid"]) / atm["mid"]
              if atm.get("mid") and atm.get("bid") is not None and atm.get("ask") is not None
              else np.nan)
    return {
        "hedge": cell.get("skew_put_atm"),
        "excite": (-rr if rr is not None else np.nan),  # call-vs-put IV richness
        "atm_iv": atm_iv,
        "otm_put_iv": put_iv,
        "otm_call_iv": call_iv,
        "smile": ((put_iv + call_iv) / 2 - atm_iv
                  if None not in (put_iv, call_iv, atm_iv) else np.nan),
        "cp_vol": np.log((vol("atm") + vol("otm_call") + 1) / (vol("otm_put") + 1)),
        "wing_vol": np.log((vol("otm_call") + 1) / (vol("otm_put") + 1)),
        "tot_vol": vol("atm") + vol("otm_put") + vol("otm_call"),
        "atm_spread": spread,
    }


def options_from_cells(cells: Iterable[dict]) -> dict[str, pd.DataFrame]:
    """Per-symbol metric frames indexed by UTC decision_date.

    astype(float64) turns the None a missing leg leaves into NaN instead of an
    object column; a duplicated (symbol, date) keeps the LAST gathered cell.
    """
    rows: dict[str, list[dict]] = {}
    for cell in cells:
        date = pd.Timestamp(cell["decision_date"], tz="UTC")
        rows.setdefault(cell["symbol"], []).append({"date": date, **cell_metrics(cell)})
    frames: dict[str, pd.DataFrame] = {}
    for symbol, symbol_rows in rows.items():
        frame = pd.DataFrame(symbol_rows).set_index("date").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        frames[symbol] = frame[OPTION_COLUMNS].astype("float64")
    return frames


def load_options(samples: Path) -> tuple[dict[str, pd.DataFrame], int]:
    """Parse a samples.jsonl into per-symbol metric frames.

    An unparseable line or one without symbol/decision_date is SKIPPED and
    COUNTED (spec section 6: corrupt cells never fabricate data, and the count
    is surfaced by the sweep so coverage loss is visible).
    """
    cells: list[dict] = []
    corrupt = 0
    for line in samples.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cell = json.loads(line)
        except ValueError:
            corrupt += 1
            continue
        if not isinstance(cell, dict) or not cell.get("symbol") or not cell.get("decision_date"):
            corrupt += 1
            continue
        cells.append(cell)
    return options_from_cells(cells), corrupt


def load_fundamentals(root: Path, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Per-symbol fundamentals frames (FILED-date index) for symbols that have
    any. Returns {} without creating anything when the store dir is absent --
    assembly must never invent an empty store."""
    if not root.exists():
        return {}
    store = FundamentalsStore(root)
    out: dict[str, pd.DataFrame] = {}
    for symbol in sorted(set(symbols)):
        frame = store.read(symbol)
        if not frame.empty:
            out[symbol] = frame
    return out


def build_panel(
    cache_dir: Path, samples: Path, fundamentals_dir: Path | None
) -> PanelData:
    """Assemble one universe's PanelData.

    The universe is the gathered options pool: the samples.jsonl allowlist
    (spec section 3.2) intersected with the symbols that have cached bars, so
    every signal family within a universe is measured on the identical
    cross-section (missing per-signal inputs are handled per-date by the NaN
    rule, never by widening the pool).
    """
    if not samples.exists():
        raise PanelError(
            f"options samples not found: {samples} "
            "(Piece 1 universes are the gathered options pools)"
        )
    allowlist = sorted(load_symbol_allowlist(samples))
    closes = load_closes(cache_dir, allowlist)
    if not closes:
        raise PanelError(f"no bar caches under {cache_dir} for the {samples.name} allowlist")
    options, corrupt = load_options(samples)
    fundamentals = (
        load_fundamentals(fundamentals_dir, closes) if fundamentals_dir is not None else {}
    )
    symbols = tuple(s for s in allowlist if s in closes)
    return PanelData(
        closes={s: closes[s] for s in symbols},
        options={s: options[s] for s in symbols if s in options},
        fundamentals={s: fundamentals[s] for s in symbols if s in fundamentals},
        symbols=symbols,
        corrupt_cells=corrupt,
    )
```

Add the two accessors to `PanelView`:

```python
    def option_row(self, symbol: str) -> pd.Series | None:
        """Latest options cell with decision_date <= as_of, or None.

        A cell more than MAX_OPTION_AGE_DAYS calendar days old is STALE and
        returns None: option state is a snapshot, and forward-filling a
        months-old cell across the decision boundary would fabricate data
        (spec section 5.5)."""
        frame = self._panel.options.get(symbol)
        if frame is None or frame.empty:
            return None
        pos = int(frame.index.searchsorted(self.as_of, side="right")) - 1
        if pos < 0:
            return None
        if (self.as_of - frame.index[pos]).days > MAX_OPTION_AGE_DAYS:
            return None
        return frame.iloc[pos]

    def fundamentals_row(self, symbol: str) -> pd.Series | None:
        """Latest fundamentals row FILED at or before as_of (the step function
        on FILING dates -- same cut as trading.signals.quality.latest_filed_row),
        or None when nothing is visible yet."""
        frame = self._panel.fundamentals.get(symbol)
        if frame is None or frame.empty:
            return None
        window = frame.loc[: self.as_of]
        return None if window.empty else window.iloc[-1]
```

- [ ] **Step 5: Complete the registry in `src/trading/alphasearch/spec.py`**

Append at the end of the file:

```python
# --------------------------------------------------------------------------- #
# Options family (requires_options). Column values come straight from
# panel.cell_metrics; each registration bakes in the higher-=-attractive sign.
# --------------------------------------------------------------------------- #
def _option_signal(column: str, sign: float) -> SignalFn:
    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol in view.symbols:
            row = view.option_row(symbol)
            scores[symbol] = math.nan if row is None else sign * float(row[column])
        return pd.Series(scores, dtype="float64")

    return fn


def _vrp(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    """Volatility risk premium: ATM implied vol minus trailing realized vol
    (exactly signal_scan's vrp = atm_iv - rvol21)."""
    scores: dict[str, float] = {}
    for symbol in view.symbols:
        row = view.option_row(symbol)
        if row is None:
            scores[symbol] = math.nan
            continue
        scores[symbol] = float(row["atm_iv"]) - _rvol(view.closes(symbol))
    return pd.Series(scores, dtype="float64")


# Rich vol premium = overpaid insurance on the name -> attractive to own.
_register("vrp", _vrp, requires_options=True)
# Spec 3.1's example: the skew signal registers NEGATED (-skew_put_atm), so
# LESS downside-hedged names score higher -- the risk-veto intuition.
_register("hedge", _option_signal("hedge", -1.0), requires_options=True)
# cell excite is already -skew_put_call (call-side richness): keep raw sign.
_register("excite", _option_signal("excite", +1.0), requires_options=True)
# Lottery/vol-premium effect: high-IV names underperform -> negate.
_register("atm_iv", _option_signal("atm_iv", -1.0), requires_options=True)
# Convex smile = tail fear priced in -> negate.
_register("smile", _option_signal("smile", -1.0), requires_options=True)
# Option-illiquidity premium: WIDER spread predicted higher returns in the
# mid-cap scan (docs/experiments.md section 9) -> keep raw sign.
_register("atm_spread", _option_signal("atm_spread", +1.0), requires_options=True)


# --------------------------------------------------------------------------- #
# Fundamentals family (requires_fundamentals). Ratios computed at scoring time
# from price-free stored primitives x the as-of close, exactly as
# trading.signals.quality / trading.signals.value do. NaN (dropped) when no
# filing is visible -- the sort's missing-data rule handles it; there is no
# 0.5-neutral here because a cross-sectional SORT needs a real value.
# --------------------------------------------------------------------------- #
def _fundamental_field(view: PanelView, symbol: str, key: str) -> float:
    row = view.fundamentals_row(symbol)
    if row is None or key not in row.index:
        return math.nan
    return float(row[key])


def _gross_profitability(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    scores = {s: _fundamental_field(view, s, "gross_profitability") for s in view.symbols}
    return pd.Series(scores, dtype="float64")


def _value_ratio(numerator: str) -> SignalFn:
    def fn(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
        scores: dict[str, float] = {}
        for symbol in view.symbols:
            shares = _fundamental_field(view, symbol, "shares_outstanding")
            value = _fundamental_field(view, symbol, numerator)
            close = view.last_close(symbol)
            score = math.nan
            if not math.isnan(shares) and shares > 0 and not math.isnan(close):
                market_cap = shares * close
                if market_cap > 0 and not math.isnan(value):
                    # Negative income/equity is a real (bottom-ranked) value,
                    # not missing data -- same policy as trading.signals.value.
                    score = value / market_cap
            scores[symbol] = score
        return pd.Series(scores, dtype="float64")

    return fn


_register("gross_profitability", _gross_profitability, requires_fundamentals=True)
_register("earnings_yield", _value_ratio("ttm_net_income"), requires_fundamentals=True)
_register("book_to_market", _value_ratio("book_equity"), requires_fundamentals=True)
```

- [ ] **Step 6: Point `scripts/signal_scan.py` at the promoted metrics**

In `scripts/signal_scan.py`, delete the whole `_cell_metrics` function
(lines 54-83) and replace with an import so the script and the engine can
never drift. After `import pandas as pd` add:

```python
from trading.alphasearch.panel import cell_metrics as _cell_metrics
```

Everything else in the script is unchanged (`load_panel` keeps calling
`_cell_metrics(cell)`, and `tests/test_signal_scan.py`'s
`from signal_scan import _cell_metrics` keeps resolving).

- [ ] **Step 7: Run the affected suites**

Run: `uv run pytest tests/test_alphasearch_panel.py tests/test_alphasearch_spec.py tests/test_signal_scan.py -v`
Expected: ALL PASS (`test_signal_scan.py` unmodified)

- [ ] **Step 8: Lint and commit**

Run: `uv run ruff check src/trading/alphasearch scripts/signal_scan.py tests/test_alphasearch_panel.py tests/test_alphasearch_spec.py`
Expected: clean

```bash
git add src/trading/alphasearch/panel.py src/trading/alphasearch/spec.py scripts/signal_scan.py tests/test_alphasearch_panel.py tests/test_alphasearch_spec.py
git commit -m "Complete panel (options+fundamentals) and the 16-signal registry [AI]

Options cells and fundamentals join bars behind the same PIT view: cells
are capped at 7 days' staleness (a months-old snapshot forward-filled
across a decision boundary would fabricate data) and fundamentals cut on
FILED dates. cell_metrics is promoted from signal_scan.py (script now
imports it) so scanner and engine can never disagree. Sign conventions
recorded per registration; higher always = attractive."
```

---

### Task 6: `sort.py` — portfolio-sort daily return series

**Files:**
- Create: `src/trading/alphasearch/sort.py`
- Test: `tests/test_alphasearch_sort.py`

**Interfaces:**
- Consumes: `PanelData` (`.closes`, `.view(as_of)`) from Task 3; `SignalSpec` (`.fn`) from Task 4.
- Produces (Tasks 8/9 rely on these exact signatures):
  - Constants `QUANTILES = 5`, `TERCILE_BELOW = 50`, `MIN_NAMES = 15` (the pre-registered §5.4 defaults; parameters exist so hand-sized test fixtures are possible, and any non-default value is a DIFFERENT journaled trial via `params`).
  - `SortError(ValueError)`
  - `SortResult` frozen dataclass — `ls: pd.Series` (daily, UTC index), `lo: pd.Series`, `turnover_monthly: float`, `skipped_dates: tuple[str, ...]` (ISO dates), `n_dates: int`, `n_names_median: float`.
  - `assign_quantiles(scores: pd.Series, quantiles: int) -> tuple[list[str], list[str]]` — `(top, bottom)` symbol lists.
  - `portfolio_sort(panel: PanelData, spec: SignalSpec, dates: Sequence[pd.Timestamp], end: pd.Timestamp, *, quantiles: int = QUANTILES, tercile_below: int = TERCILE_BELOW, min_names: int = MIN_NAMES) -> SortResult`

Mechanics (pre-registered §5.4, restated so the implementer needs no spec):
per decision date `d`, score the cross-section as of `d` (via `panel.view(d)`),
drop NaN scores; if `< min_names` names remain, SKIP the date and record it;
use quintiles, or terciles when `< tercile_below` names. Rank ascending with a
deterministic symbol tie-break; `np.array_split` makes near-equal buckets
(extras go to the LOWER buckets). Hold equal-weight from the close of `d` to
the close of the next decision date (or `end` for the last one). Daily
portfolio return on day `t` = mean over members of each member's own
close-to-close return; a member with no bar that day is skipped by the mean
(equal weight over present members). `ls = mean(top) − mean(bottom)`;
`lo = mean(top)`. Turnover = mean over consecutive rebalances of
`1 − |top_prev ∩ top_cur| / |top_cur|`. No transaction costs.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alphasearch_sort.py`:

```python
"""Hand-computable portfolio-sort fixtures (6 symbols x a few months)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trading.alphasearch.panel import PanelData
from trading.alphasearch.sort import (
    SortError,
    assign_quantiles,
    portfolio_sort,
)
from trading.alphasearch.spec import SignalSpec


def _panel(rates: dict[str, float], periods: int = 65) -> PanelData:
    """Constant daily growth per symbol: mom21 rank == rate rank, and the
    portfolio's daily return equals the mean of its members' rates."""
    idx = pd.date_range("2020-01-02", periods=periods, freq="B", tz="UTC")
    closes = {
        sym: pd.Series([100.0 * (1 + r) ** i for i in range(periods)], index=idx)
        for sym, r in rates.items()
    }
    return PanelData(closes=closes, options={}, fundamentals={},
                     symbols=tuple(sorted(closes)))


def _mom21() -> SignalSpec:
    from trading.alphasearch.spec import SIGNALS

    return SIGNALS["mom21"]


SIX = {"S1": -0.02, "S2": -0.01, "S3": 0.0, "S4": 0.01, "S5": 0.02, "S6": 0.03}


def test_assign_quantiles_terciles_of_six():
    scores = pd.Series({"S1": 1.0, "S2": 2.0, "S3": 3.0, "S4": 4.0,
                        "S5": 5.0, "S6": 6.0})
    top, bottom = assign_quantiles(scores, 3)
    assert set(top) == {"S5", "S6"}
    assert set(bottom) == {"S1", "S2"}


def test_assign_quantiles_uneven_extras_go_to_lower_buckets():
    scores = pd.Series({f"S{i}": float(i) for i in range(1, 8)})  # 7 names, q=3
    top, bottom = assign_quantiles(scores, 3)
    assert set(bottom) == {"S1", "S2", "S3"}  # 3-2-2 split, extras at the bottom
    assert set(top) == {"S6", "S7"}


def test_assign_quantiles_deterministic_tie_break():
    scores = pd.Series({"B": 1.0, "A": 1.0, "D": 1.0, "C": 1.0})
    top, bottom = assign_quantiles(scores, 2)
    assert bottom == ["A", "B"] and top == ["C", "D"]  # alphabetical on ties


def test_ls_and_lo_daily_returns_hand_computed():
    panel = _panel(SIX)
    dates = panel.decision_dates(panel.closes["S1"].index[30],
                                 panel.closes["S1"].index[-1])
    result = portfolio_sort(panel, _mom21(), dates, panel.closes["S1"].index[-1],
                            quantiles=3, tercile_below=0, min_names=3)
    # tercile_below=0 -> quantiles=3 everywhere; top={S5,S6}, bottom={S1,S2}.
    expected_lo = (0.02 + 0.03) / 2
    expected_ls = expected_lo - (-0.02 + -0.01) / 2
    assert len(result.ls) > 0
    assert all(math.isclose(v, expected_ls, rel_tol=1e-9) for v in result.ls)
    assert all(math.isclose(v, expected_lo, rel_tol=1e-9) for v in result.lo)
    # Constant rates -> identical top set every month -> zero turnover.
    assert result.turnover_monthly == 0.0
    assert result.skipped_dates == ()
    assert result.n_names_median == 6.0


def test_series_span_holding_periods_not_just_first_month():
    panel = _panel(SIX)
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[30], idx[-1])
    assert len(dates) >= 2
    result = portfolio_sort(panel, _mom21(), dates, idx[-1],
                            quantiles=3, tercile_below=0, min_names=3)
    # Daily series covers (first decision date, end]: every union trading day.
    expected_days = idx[(idx > dates[0]) & (idx <= idx[-1])]
    assert list(result.ls.index) == list(expected_days)
    assert result.n_dates == len(dates)


def test_thin_dates_are_skipped_and_recorded():
    # Only 2 symbols have 21 bars of history at the first decision date ->
    # mom21 is NaN for the rest -> cross-section 2 < min_names 3 -> skip.
    idx = pd.date_range("2020-01-02", periods=65, freq="B", tz="UTC")
    rich = {s: pd.Series([100.0 * (1 + r) ** i for i in range(65)], index=idx)
            for s, r in {"S1": 0.01, "S2": 0.02}.items()}
    poor = {s: pd.Series(100.0, index=idx[40:]) for s in ("S3", "S4", "S5", "S6")}
    panel = PanelData(closes={**rich, **poor}, options={}, fundamentals={},
                      symbols=tuple(sorted({**rich, **poor})))
    dates = panel.decision_dates(idx[25], idx[-1])
    result = portfolio_sort(panel, _mom21(), dates, idx[-1],
                            quantiles=3, tercile_below=0, min_names=3)
    assert len(result.skipped_dates) >= 1
    assert result.skipped_dates[0] == dates[0].date().isoformat()


def test_all_dates_skipped_raises_sort_error():
    panel = _panel(SIX)
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[30], idx[-1])
    with pytest.raises(SortError):
        portfolio_sort(panel, _mom21(), dates, idx[-1], min_names=15)  # only 6 names


def test_no_decision_dates_raises_sort_error():
    panel = _panel(SIX)
    with pytest.raises(SortError):
        portfolio_sort(panel, _mom21(), (), panel.closes["S1"].index[-1])


def test_turnover_hand_example():
    # Verify the turnover formula on a crafted signal that rotates the top set.
    calls = {"n": 0}

    def rotating(view, as_of):
        calls["n"] += 1
        base = {"S1": 1.0, "S2": 2.0, "S3": 3.0, "S4": 4.0, "S5": 5.0, "S6": 6.0}
        if calls["n"] == 2:  # second decision date: S6 drops to the bottom
            base["S6"] = 0.0
        return pd.Series(base, dtype="float64")

    panel = _panel(SIX)
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[30], idx[-1])[:2]
    spec = SignalSpec("rotating", rotating)
    result = portfolio_sort(panel, spec, dates, idx[-1],
                            quantiles=3, tercile_below=0, min_names=3)
    # tops: {S5,S6} then {S4,S5} -> overlap 1 of 2 -> one-way turnover 0.5.
    assert result.turnover_monthly == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_sort.py -v`
Expected: FAIL at collection with `ImportError` (no `trading.alphasearch.sort`)

- [ ] **Step 3: Create `src/trading/alphasearch/sort.py`**

```python
"""Portfolio-sort return generator (spec section 3.3): scores -> daily series.

Per decision date: rank the cross-section of signal scores, form equal-weight
quantile portfolios (quintiles; terciles below 50 names; skip + record below
15 -- spec section 5.4), hold to the next decision date. Output is two dated
DAILY return series -- ls = mean(top) - mean(bottom), lo = mean(top) -- which
give ~1250 regression observations over the 5-year discovery window vs ~60
monthly. No transaction costs here (costs belong to the survivor-stage full
backtest); monthly one-way turnover of the top quantile is reported so cost
fragility is visible early.

Forward returns are computed HERE, from panel.closes -- a separate pass from
signal scoring, which only ever sees a PanelView truncated at the decision
date. That separation is the engine's no-look-ahead guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading.alphasearch.panel import PanelData
from trading.alphasearch.spec import SignalSpec

QUANTILES = 5
TERCILE_BELOW = 50  # cross-section thinner than this -> terciles
MIN_NAMES = 15      # thinner than this -> skip the date entirely (journaled)


class SortError(ValueError):
    """The sort could not produce a return series (empty calendar/universe)."""


@dataclass(frozen=True)
class SortResult:
    ls: pd.Series                    # daily long/short spread (top - bottom)
    lo: pd.Series                    # daily long-only (top quantile)
    turnover_monthly: float          # mean one-way turnover of the top quantile
    skipped_dates: tuple[str, ...]   # ISO dates skipped for thin cross-sections
    n_dates: int                     # decision dates attempted (incl. skipped)
    n_names_median: float            # median cross-section size on traded dates


def assign_quantiles(scores: pd.Series, quantiles: int) -> tuple[list[str], list[str]]:
    """(top, bottom) equal-weight buckets of a NaN-free score series.

    Sort ascending by (score, symbol) -- the mergesort-on-sorted-index trick
    trading.signals.engine.rank uses, so ties resolve alphabetically and the
    sort is reproducible. np.array_split makes `quantiles` near-equal buckets;
    when n % q != 0 the EARLIER (lower-score) buckets take the extras.
    """
    ordered = scores.sort_index().sort_values(kind="mergesort")
    buckets = np.array_split(ordered.index.to_numpy(), quantiles)
    return list(buckets[-1]), list(buckets[0])


def portfolio_sort(
    panel: PanelData,
    spec: SignalSpec,
    dates: Sequence[pd.Timestamp],
    end: pd.Timestamp,
    *,
    quantiles: int = QUANTILES,
    tercile_below: int = TERCILE_BELOW,
    min_names: int = MIN_NAMES,
) -> SortResult:
    """Build the daily L/S and long-only series over [dates[0], end]."""
    if not dates:
        raise SortError(f"{spec.name}: no decision dates in the window")
    # Per-symbol close-to-close returns on each symbol's OWN calendar, aligned
    # to the union calendar afterwards; a symbol's missing day stays NaN and
    # mean(skipna) simply equal-weights the members that traded.
    returns = pd.DataFrame({s: c.pct_change() for s, c in panel.closes.items()})

    ls_parts: list[pd.Series] = []
    lo_parts: list[pd.Series] = []
    tops: list[set[str]] = []
    skipped: list[str] = []
    names_per_date: list[int] = []

    for i, date in enumerate(dates):
        scores = spec.fn(panel.view(date), date).dropna()
        if len(scores) < min_names:
            skipped.append(date.date().isoformat())
            continue
        q = quantiles if len(scores) >= tercile_below else 3
        top, bottom = assign_quantiles(scores, q)
        tops.append(set(top))
        names_per_date.append(len(scores))
        hold_end = dates[i + 1] if i + 1 < len(dates) else end
        segment = returns.loc[(returns.index > date) & (returns.index <= hold_end)]
        top_mean = segment[top].mean(axis=1)
        bottom_mean = segment[bottom].mean(axis=1)
        ls_parts.append(top_mean - bottom_mean)
        lo_parts.append(top_mean)

    if not tops:
        raise SortError(
            f"{spec.name}: every decision date skipped (cross-section < {min_names})"
        )
    ls = pd.concat(ls_parts).dropna()
    lo = pd.concat(lo_parts).dropna()
    pairs = list(zip(tops, tops[1:], strict=False))
    turnover = (
        float(np.mean([1 - len(prev & cur) / len(cur) for prev, cur in pairs]))
        if pairs
        else math.nan
    )
    return SortResult(
        ls=ls,
        lo=lo,
        turnover_monthly=turnover,
        skipped_dates=tuple(skipped),
        n_dates=len(dates),
        n_names_median=float(np.median(names_per_date)),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_alphasearch_sort.py -v`
Expected: all 9 tests PASS

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check src/trading/alphasearch tests/test_alphasearch_sort.py`
Expected: clean

```bash
git add src/trading/alphasearch/sort.py tests/test_alphasearch_sort.py
git commit -m "Add alphasearch.sort: quantile portfolio -> daily L/S + LO series [AI]

Daily (not monthly) series give ~1250 regression observations over the
discovery window. Quantile assignment reuses the repo's deterministic
mergesort tie-break; thin dates are skipped AND recorded (spec 5.4), and
the top-quantile one-way turnover is computed so cost fragility is visible
before any survivor reaches a full backtest."
```

---

### Task 7: `sweep.py` part 1 — the trial journal layer

**Files:**
- Create: `src/trading/alphasearch/sweep.py` (journal layer only; Tasks 8/9 extend this file)
- Test: `tests/test_alphasearch_journal.py`

**Interfaces:**
- Consumes: `trading.journal.Journal` (existing: `append(event)`, `events()`, append-only JSONL with fsync + torn-tail repair).
- Produces (Tasks 8/9/10 rely on these exact signatures):
  - Constants: `DISCOVERY_WINDOW = "2019-01-01..2023-12-31"`, `HOLDOUT_START = "2024-01-01"`, `BH_Q = 0.10`, `HOLDOUT_PASS_RATIO = 0.5`, `DEFAULT_PARAMS = {"quantiles": 5, "weighting": "equal", "cadence": "monthly"}`.
  - `SweepError(RuntimeError)`
  - `trials_journal(journal_dir: Path) -> Journal` — `journal/alphasearch-trials.jsonl`.
  - `trial_config(signal: str, universe: str, window: str, params: dict | None = None) -> dict`
  - `trial_config_hash(config: dict) -> str` — sha256 of sorted-keys JSON, first 12 hex (mirrors `trading.journal.config_hash`).
  - `log_trial(journal: Journal, *, kind: str, config: dict, ts: str, result: dict | None = None, error: str | None = None) -> dict` — appends and returns the event.
  - `load_trials(journal: Journal) -> list[dict]` — all trial events, deduplicated: latest event per `(config_hash, kind)` wins.
  - `discovery_trials(journal: Journal) -> list[dict]`
  - `prior_holdout_trial(journal: Journal, signal: str, universe: str) -> dict | None`
  - `find_discovery_trial(journal: Journal, signal: str, universe: str, window: str = DISCOVERY_WINDOW) -> dict | None` — exact default-params config-hash lookup.

Idempotency resolution (spec §4 says "identical config_hash re-runs update in
place" while the journal is append-only): re-runs APPEND a new event —
append-only is never violated — and every reader deduplicates via
`load_trials`, keeping the LATEST event per `(config_hash, kind)`. Logical
update-in-place, physical append-only. The BH/DSR trial count is
`len(discovery_trials(journal))`, so identical re-runs never double-count and
distinct params always do.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alphasearch_journal.py`:

```python
"""Journal honesty: idempotent re-runs, distinct params count, holdout tracking."""

from __future__ import annotations

import math

from trading.alphasearch.sweep import (
    DEFAULT_PARAMS,
    DISCOVERY_WINDOW,
    discovery_trials,
    find_discovery_trial,
    load_trials,
    log_trial,
    prior_holdout_trial,
    trial_config,
    trial_config_hash,
    trials_journal,
)


def _journal(tmp_path):
    return trials_journal(tmp_path / "journal")


def _result(alpha_t: float = 2.0) -> dict:
    return {
        "n_dates": 60,
        "n_names_median": 97.0,
        "ls": {"alpha_annual_pct": 5.0, "alpha_t": alpha_t, "p": 0.04},
        "lo": {"alpha_annual_pct": 2.0, "alpha_t": 1.0, "p": 0.3},
        "turnover_monthly": 0.4,
        "skipped_dates": [],
    }


def test_trial_config_hash_is_deterministic_and_param_sensitive():
    base = trial_config("mom21", "largecap", DISCOVERY_WINDOW)
    again = trial_config("mom21", "largecap", DISCOVERY_WINDOW)
    assert trial_config_hash(base) == trial_config_hash(again)
    assert base["params"] == DEFAULT_PARAMS
    tercile = trial_config("mom21", "largecap", DISCOVERY_WINDOW,
                           params={**DEFAULT_PARAMS, "quantiles": 3})
    assert trial_config_hash(tercile) != trial_config_hash(base)


def test_log_trial_event_matches_spec_schema(tmp_path):
    journal = _journal(tmp_path)
    config = trial_config("mom21", "largecap", DISCOVERY_WINDOW)
    event = log_trial(journal, kind="discovery", config=config,
                      ts="2026-07-08T00:00:00+00:00", result=_result())
    stored = next(iter(journal.events()))
    assert stored == event
    assert stored["event"] == "trial"
    assert stored["kind"] == "discovery"
    assert stored["signal"] == "mom21"
    assert stored["universe"] == "largecap"
    assert stored["window"] == DISCOVERY_WINDOW
    assert stored["params"] == DEFAULT_PARAMS
    assert stored["config_hash"] == trial_config_hash(config)
    assert stored["ls"]["alpha_t"] == 2.0
    assert stored["error"] is None
    assert stored["ts"] == "2026-07-08T00:00:00+00:00"


def test_identical_rerun_appends_but_never_double_counts(tmp_path):
    journal = _journal(tmp_path)
    config = trial_config("mom21", "largecap", DISCOVERY_WINDOW)
    log_trial(journal, kind="discovery", config=config, ts="t1", result=_result(2.0))
    log_trial(journal, kind="discovery", config=config, ts="t2", result=_result(2.5))
    assert len(list(journal.events())) == 2      # append-only: nothing rewritten
    trials = discovery_trials(journal)
    assert len(trials) == 1                       # ...but it is ONE trial
    assert trials[0]["ts"] == "t2"                # latest event wins


def test_distinct_params_are_distinct_trials(tmp_path):
    journal = _journal(tmp_path)
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", DISCOVERY_WINDOW),
              ts="t1", result=_result())
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", DISCOVERY_WINDOW,
                                  params={**DEFAULT_PARAMS, "quantiles": 3}),
              ts="t2", result=_result())
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "midcap", DISCOVERY_WINDOW),
              ts="t3", result=_result())
    assert len(discovery_trials(journal)) == 3


def test_error_trial_still_counts(tmp_path):
    journal = _journal(tmp_path)
    log_trial(journal, kind="discovery",
              config=trial_config("vrp", "largecap", DISCOVERY_WINDOW),
              ts="t1", error="ValueError: need more observations (3) than parameters (5)")
    trials = discovery_trials(journal)
    assert len(trials) == 1
    assert trials[0]["error"].startswith("ValueError")
    assert "ls" not in trials[0] or trials[0].get("ls") is None


def test_nan_results_are_journaled_as_null(tmp_path):
    journal = _journal(tmp_path)
    result = _result()
    result["turnover_monthly"] = math.nan
    result["ls"]["p"] = math.nan
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", DISCOVERY_WINDOW),
              ts="t1", result=result)
    stored = next(iter(journal.events()))
    assert stored["turnover_monthly"] is None    # NaN never reaches the JSONL
    assert stored["ls"]["p"] is None


def test_holdout_tracking_and_discovery_lookup(tmp_path):
    journal = _journal(tmp_path)
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", DISCOVERY_WINDOW),
              ts="t1", result=_result())
    assert prior_holdout_trial(journal, "mom21", "largecap") is None
    log_trial(journal, kind="holdout",
              config=trial_config("mom21", "largecap", "2024-01-01..2026-07-07"),
              ts="t2", result=_result())
    prior = prior_holdout_trial(journal, "mom21", "largecap")
    assert prior is not None and prior["ts"] == "t2"
    assert prior_holdout_trial(journal, "mom21", "midcap") is None
    found = find_discovery_trial(journal, "mom21", "largecap")
    assert found is not None and found["kind"] == "discovery"
    assert find_discovery_trial(journal, "rev5", "largecap") is None


def test_holdout_and_discovery_never_collide_in_dedupe(tmp_path):
    journal = _journal(tmp_path)
    window = "2024-01-01..2026-07-07"
    config = trial_config("mom21", "largecap", window)
    log_trial(journal, kind="discovery", config=config, ts="t1", result=_result())
    log_trial(journal, kind="holdout", config=config, ts="t2", result=_result())
    assert len(load_trials(journal)) == 2  # same hash, different kind
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_journal.py -v`
Expected: FAIL at collection with `ImportError` (no `trading.alphasearch.sweep`)

- [ ] **Step 3: Create `src/trading/alphasearch/sweep.py` (journal layer)**

```python
"""Sweep runner, trial journal, leaderboard, holdout re-prove (spec 3.6 + 4).

The trial journal (journal/alphasearch-trials.jsonl, via trading.journal.
Journal) is the program's scientific ledger: EVERY evaluation -- success or
error -- is appended BEFORE any leaderboard is computed, and the BH-FDR /
DSR trial count is derived from this file alone. It is append-only and
committed to git; deleting or editing it invalidates the statistics.

Idempotency: an identical config re-run APPENDS a new event (append-only is
never violated) and every reader deduplicates via load_trials(), keeping the
LATEST event per (config_hash, kind) -- logical update-in-place, physical
append-only, and re-runs never inflate the trial count. Any changed parameter
changes the hash and honestly counts as a NEW trial (spec 5.6).

This module never reads the clock: `ts` always arrives from the CLI.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from trading.journal import Journal

DISCOVERY_WINDOW = "2019-01-01..2023-12-31"   # pre-registered (spec 5.1)
HOLDOUT_START = "2024-01-01"                  # pre-registered (spec 5.3)
BH_Q = 0.10                                   # pre-registered (spec 5.2)
HOLDOUT_PASS_RATIO = 0.5                      # pre-registered (spec 3.6)
DEFAULT_PARAMS = {"quantiles": 5, "weighting": "equal", "cadence": "monthly"}


class SweepError(RuntimeError):
    """A sweep/holdout invariant was violated; refuse loudly."""


def trials_journal(journal_dir: Path) -> Journal:
    return Journal(journal_dir / "alphasearch-trials.jsonl")


def trial_config(
    signal: str, universe: str, window: str, params: dict | None = None
) -> dict:
    return {
        "signal": signal,
        "universe": universe,
        "window": window,
        "params": dict(params or DEFAULT_PARAMS),
    }


def trial_config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _json_safe(value: object) -> object:
    """NaN -> None, recursively; numpy scalars -> Python scalars. The journal
    must stay strict JSON (json.dumps would happily emit invalid bare NaN)."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def log_trial(
    journal: Journal,
    *,
    kind: str,  # "discovery" | "holdout"
    config: dict,
    ts: str,  # ISO-8601 UTC, supplied by the CLI (the only clock reader)
    result: dict | None = None,
    error: str | None = None,
) -> dict:
    """Append one trial event (spec section 4 schema) and return it."""
    event = {
        "event": "trial",
        "kind": kind,
        **config,
        "config_hash": trial_config_hash(config),
        "ts": ts,
        "error": error,
        **(result or {}),
    }
    event = _json_safe(event)
    journal.append(event)
    return event


def load_trials(journal: Journal) -> list[dict]:
    """All trial events, deduplicated: latest per (config_hash, kind) wins."""
    latest: dict[tuple[str, str], dict] = {}
    for event in journal.events():
        if event.get("event") != "trial":
            continue
        latest[(event["config_hash"], event["kind"])] = event
    return list(latest.values())


def discovery_trials(journal: Journal) -> list[dict]:
    """The honest trial count for BH/DSR = len() of this list."""
    return [e for e in load_trials(journal) if e.get("kind") == "discovery"]


def prior_holdout_trial(journal: Journal, signal: str, universe: str) -> dict | None:
    """Any prior holdout event for (signal, universe) -- ANY window/params:
    the holdout is touched once per candidate, not once per configuration."""
    last: dict | None = None
    for event in journal.events():
        if (
            event.get("event") == "trial"
            and event.get("kind") == "holdout"
            and event.get("signal") == signal
            and event.get("universe") == universe
        ):
            last = event
    return last


def find_discovery_trial(
    journal: Journal, signal: str, universe: str, window: str = DISCOVERY_WINDOW
) -> dict | None:
    """The default-params discovery trial for (signal, universe), by exact
    config hash -- the reference a holdout is compared against."""
    wanted = trial_config_hash(trial_config(signal, universe, window))
    for event in load_trials(journal):
        if event.get("kind") == "discovery" and event.get("config_hash") == wanted:
            return event
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_alphasearch_journal.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check src/trading/alphasearch tests/test_alphasearch_journal.py`
Expected: clean

```bash
git add src/trading/alphasearch/sweep.py tests/test_alphasearch_journal.py
git commit -m "Add alphasearch trial journal layer: honest, idempotent counting [AI]

Every evaluation appends to journal/alphasearch-trials.jsonl before any
leaderboard exists; BH/DSR derive their trial count from this file alone.
Idempotency without breaking append-only: re-runs append, readers keep the
latest event per (config_hash, kind), so identical re-runs never inflate
the count and any changed parameter honestly creates a new trial."
```

---

### Task 8: `sweep.py` part 2 — trial evaluation, sweep runner, leaderboard

**Files:**
- Modify: `src/trading/alphasearch/sweep.py` (add everything below the journal layer)
- Create: `tests/alphasearch_helpers.py` (shared fixtures; NOT a test module, mirroring `tests/backtest_helpers.py`)
- Test: `tests/test_alphasearch_sweep.py`

**Interfaces:**
- Consumes: Task 1 `stats.p_from_t/bh_fdr/deflated_sharpe`; Task 2 `evaluate_alpha`, `AlphaResult`; Task 5 `build_panel`, `PanelData`; Task 6 `portfolio_sort`, `SortError`, `SortResult`, `QUANTILES`, `TERCILE_BELOW`, `MIN_NAMES`; Task 4/5 `SIGNALS`, `SignalSpec`; Task 7 journal layer.
- Produces (Tasks 9/10/11 rely on these exact signatures):
  - `UniverseSpec` frozen dataclass — `name: str`, `cache_dir: Path`, `samples: Path`, `fundamentals_dir: Path | None`.
  - `default_universes(root: Path) -> dict[str, UniverseSpec]` — `largecap` (`data/equities-tiingo` + `data/options-iv/samples.jsonl`) and `midcap` (`data/equities-midcap-tiingo` + `data/options-iv/samples-midcap.jsonl`), both with `data/fundamentals/equities`.
  - `build_universe_panel(spec: UniverseSpec) -> PanelData`
  - `evaluate_trial(panel: PanelData, spec: SignalSpec, window: str, factors: pd.DataFrame, *, quantiles: int = QUANTILES, tercile_below: int = TERCILE_BELOW, min_names: int = MIN_NAMES) -> dict` — the spec §4 result payload (`n_dates`, `n_names_median`, `ls`, `lo`, `turnover_monthly`, `skipped_dates`).
  - `LeaderboardRow` frozen dataclass (fields in the code below).
  - `build_leaderboard(journal: Journal) -> tuple[list[LeaderboardRow], int]` — recomputed from the journal ALONE; second element is the honest discovery-trial count.
  - `run_sweep(universes: dict[str, UniverseSpec], journal: Journal, factors: pd.DataFrame, ts: str, *, signals: dict[str, SignalSpec] | None = None, window: str = DISCOVERY_WINDOW, quantiles: int = QUANTILES, tercile_below: int = TERCILE_BELOW, min_names: int = MIN_NAMES, panel_factory: Callable[[UniverseSpec], PanelData] = build_universe_panel) -> tuple[list[LeaderboardRow], int]`
  - Helpers module: `make_cell(symbol, date, **metrics) -> dict`, `month_firsts(idx) -> list[pd.Timestamp]`, `make_panel(n_symbols=16, start="2020-01-02", periods=130, seed=7, with_options=True, with_fundamentals=True) -> PanelData`, `make_factors(start="2019-12-02", periods=160, seed=3) -> pd.DataFrame`.

Error-handling contract implemented here (spec §6):
- `SortError` / `ValueError` (n<=k regression) / `np.linalg.LinAlgError` (degenerate design) during a trial -> the trial is journaled with `error` set and NO `ls` payload; it still counts toward BH's n (its p reads as NaN -> 1.0) and appears on the leaderboard flagged as an error. The sweep continues.
- A signal whose data family the universe cannot supply (options signal, no cells; fundamentals signal, empty store) -> `SweepError` raised BEFORE any trial in that universe runs — refused at assembly, nothing journaled.

- [ ] **Step 1: Create the shared fixture helpers**

Create `tests/alphasearch_helpers.py`:

```python
"""Deterministic synthetic fixtures shared by the alphasearch tests.

Not a test module (no test_ prefix): imported by test_alphasearch_sweep.py,
test_alphasearch_lookahead.py and the golden sweep test, mirroring
tests/backtest_helpers.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading.alphasearch.panel import PanelData, options_from_cells


def make_cell(
    symbol: str,
    date: str,
    *,
    atm_iv: float = 0.30,
    put_iv: float = 0.34,
    call_iv: float = 0.28,
    skew_put_atm: float = 0.05,
    skew_put_call: float = 0.02,
) -> dict:
    """One samples.jsonl-shaped options cell with all three legs present."""
    return {
        "symbol": symbol,
        "decision_date": date,
        "skew_put_atm": skew_put_atm,
        "skew_put_call": skew_put_call,
        "contracts": [
            {"role": "atm", "bid": 4.0, "ask": 4.2, "mid": 4.1, "iv": atm_iv,
             "volume": 100},
            {"role": "otm_put", "iv": put_iv, "volume": 50},
            {"role": "otm_call", "iv": call_iv, "volume": 25},
        ],
    }


def month_firsts(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """First index entry of each calendar month (the fixture's decision dates)."""
    firsts: dict[str, pd.Timestamp] = {}
    for date in idx:
        firsts.setdefault(date.strftime("%Y-%m"), date)
    return [firsts[m] for m in sorted(firsts)]


def make_panel(
    n_symbols: int = 16,
    start: str = "2020-01-02",
    periods: int = 130,
    seed: int = 7,
    with_options: bool = True,
    with_fundamentals: bool = True,
) -> PanelData:
    """Symbol S<i> drifts at (i - n/2)*2bp/day plus small seeded noise: momentum
    ranks are stable (so a momentum L/S spread has a large true alpha), values
    are bit-reproducible across runs, and 16 names >= MIN_NAMES=15 so default
    sort parameters trade every eligible date (as terciles, being < 50)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="B", tz="UTC")
    names = [f"S{i:02d}" for i in range(n_symbols)]
    closes: dict[str, pd.Series] = {}
    for i, sym in enumerate(names):
        drift = (i - n_symbols / 2) * 2e-4
        rets = drift + rng.normal(0.0, 0.002, size=periods)
        closes[sym] = pd.Series(100.0 * np.cumprod(1 + rets), index=idx)
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
        # Two filings (initial + mid-fixture) so the no-look-ahead test has
        # post-cutoff fundamentals to perturb -- one filing would make that
        # family's check vacuous.
        filed = pd.DatetimeIndex([idx[0], idx[len(idx) // 2]])
        for i, sym in enumerate(names):
            fundamentals[sym] = pd.DataFrame(
                {
                    "gross_profitability": [0.10 + 0.02 * i, 0.12 + 0.02 * i],
                    "ttm_net_income": [1e6 * (i + 1), 1.1e6 * (i + 1)],
                    "book_equity": [5e6 * (i + 1), 5.2e6 * (i + 1)],
                    "shares_outstanding": [1e6, 1e6],
                },
                index=filed,
            )
    return PanelData(closes=closes, options=options, fundamentals=fundamentals,
                     symbols=tuple(names))


def make_factors(
    start: str = "2019-12-02", periods: int = 160, seed: int = 3
) -> pd.DataFrame:
    """Synthetic Ken-French-shaped daily factors covering the fixture window.
    Uncorrelated with the fixture returns by construction (different seed), so
    the fixture's drift spread shows up as ALPHA, not loadings."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0004, 0.01, size=periods),
            "SMB": rng.normal(0.0, 0.005, size=periods),
            "HML": rng.normal(0.0, 0.005, size=periods),
            "Mom": rng.normal(0.0, 0.006, size=periods),
            "RF": 0.0001,
        },
        index=idx,
    )
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_alphasearch_sweep.py`:

```python
"""Sweep runner + leaderboard: journaling order, honesty, refusal, recompute."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphasearch_helpers import make_factors, make_panel
from trading.alphasearch.spec import SIGNALS
from trading.alphasearch.sweep import (
    SweepError,
    UniverseSpec,
    build_leaderboard,
    discovery_trials,
    run_sweep,
    trials_journal,
)

WINDOW = "2020-01-01..2020-06-30"


def _universe(tmp_path) -> dict[str, UniverseSpec]:
    # Paths are unused: tests inject panel_factory instead of touching disk.
    dummy = UniverseSpec("largecap", tmp_path, tmp_path / "s.jsonl", None)
    return {"largecap": dummy}


def _subset(*names: str) -> dict:
    return {n: SIGNALS[n] for n in names}


def test_sweep_journals_every_trial_and_ranks_by_abs_t(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    rows, n_trials = run_sweep(
        _universe(tmp_path), journal, make_factors(), ts="t1",
        signals=_subset("mom21", "rev5", "rvol21"), window=WINDOW,
        panel_factory=lambda _u: panel,
    )
    assert n_trials == 3
    assert len(discovery_trials(journal)) == 3      # journaled, not just returned
    assert len(rows) == 3
    ts = [abs(r.alpha_t) for r in rows if r.alpha_t is not None]
    assert ts == sorted(ts, reverse=True)           # sorted by |4F L/S t|
    # The engineered momentum spread must be strongly significant.
    mom = next(r for r in rows if r.signal == "mom21")
    assert abs(mom.alpha_t) > 5
    assert mom.bh_pass
    assert mom.dsr is not None and 0.0 <= mom.dsr <= 1.0  # DSR shown for survivors
    assert set(mom.loadings) == {"Mkt-RF", "SMB", "HML", "Mom"}
    assert mom.turnover_monthly is not None


def test_sweep_rerun_is_idempotent_for_trial_count(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    kwargs = dict(signals=_subset("mom21"), window=WINDOW,
                  panel_factory=lambda _u: panel)
    _, first = run_sweep(_universe(tmp_path), journal, make_factors(), "t1", **kwargs)
    _, second = run_sweep(_universe(tmp_path), journal, make_factors(), "t2", **kwargs)
    assert first == second == 1                     # identical config: ONE trial
    assert len(list(journal.events())) == 2         # ...but both runs appended


def test_error_trial_is_journaled_flagged_and_counted(tmp_path):
    # mom252 needs 253 bars; the fixture has 130 -> every date skips -> SortError.
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    rows, n_trials = run_sweep(
        _universe(tmp_path), journal, make_factors(), ts="t1",
        signals=_subset("mom21", "mom252"), window=WINDOW,
        panel_factory=lambda _u: panel,
    )
    assert n_trials == 2                            # the failure still counts
    failed = next(r for r in rows if r.signal == "mom252")
    assert failed.error is not None and "SortError" in failed.error
    assert failed.alpha_t is None
    assert not failed.bh_pass
    event = next(e for e in discovery_trials(journal) if e["signal"] == "mom252")
    assert event["error"] is not None and "ls" not in event


def test_options_signal_without_cells_refused_before_any_trial(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(with_options=False)
    with pytest.raises(SweepError):
        run_sweep(
            _universe(tmp_path), journal, make_factors(), ts="t1",
            signals=_subset("mom21", "hedge"), window=WINDOW,
            panel_factory=lambda _u: panel,
        )
    assert list(journal.events()) == []             # refused at assembly: no trials


def test_fundamentals_signal_without_store_refused(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(with_fundamentals=False)
    with pytest.raises(SweepError):
        run_sweep(
            _universe(tmp_path), journal, make_factors(), ts="t1",
            signals=_subset("earnings_yield"), window=WINDOW,
            panel_factory=lambda _u: panel,
        )


def test_leaderboard_recomputes_from_journal_alone(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    rows, n_trials = run_sweep(
        _universe(tmp_path), journal, make_factors(), ts="t1",
        signals=_subset("mom21", "rev5"), window=WINDOW,
        panel_factory=lambda _u: panel,
    )
    again, n_again = build_leaderboard(journal)     # no panels, no factors
    assert n_again == n_trials
    assert [(r.signal, r.alpha_t, r.bh_pass) for r in again] == [
        (r.signal, r.alpha_t, r.bh_pass) for r in rows
    ]


def test_default_universes_point_at_gathered_pools():
    from trading.alphasearch.sweep import default_universes

    got = default_universes(Path("."))
    assert set(got) == {"largecap", "midcap"}
    assert got["largecap"].samples.name == "samples.jsonl"
    assert got["midcap"].samples.name == "samples-midcap.jsonl"
    assert got["midcap"].cache_dir.name == "equities-midcap-tiingo"
    assert got["largecap"].fundamentals_dir is not None


def test_bh_gate_spans_the_whole_journal_not_one_sweep(tmp_path):
    # Trials from an EARLIER sweep (different window -> different hashes)
    # must raise n for the BH gate of a later sweep.
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    run_sweep(_universe(tmp_path), journal, make_factors(), "t1",
              signals=_subset("mom21"), window=WINDOW,
              panel_factory=lambda _u: panel)
    _, n_trials = run_sweep(_universe(tmp_path), journal, make_factors(), "t2",
                            signals=_subset("rev5"), window=WINDOW,
                            panel_factory=lambda _u: panel)
    assert n_trials == 2  # the gate sees ALL journaled discovery trials
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_sweep.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'UniverseSpec'`

- [ ] **Step 4: Extend `src/trading/alphasearch/sweep.py`**

Update the imports block to:

```python
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading.alphasearch import stats
from trading.alphasearch.evaluate import AlphaResult, evaluate_alpha
from trading.alphasearch.panel import PanelData, build_panel
from trading.alphasearch.sort import (
    MIN_NAMES,
    QUANTILES,
    TERCILE_BELOW,
    SortError,
    portfolio_sort,
)
from trading.alphasearch.spec import SIGNALS, SignalSpec
from trading.journal import Journal
```

Append below the journal layer:

```python
# --------------------------------------------------------------------------- #
# Universes (spec 3.2): the two gathered options pools. Every signal family in
# a universe is measured on this same allowlist cross-section.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UniverseSpec:
    name: str
    cache_dir: Path
    samples: Path
    fundamentals_dir: Path | None


def default_universes(root: Path) -> dict[str, UniverseSpec]:
    return {
        "largecap": UniverseSpec(
            "largecap",
            root / "data" / "equities-tiingo",
            root / "data" / "options-iv" / "samples.jsonl",
            root / "data" / "fundamentals" / "equities",
        ),
        "midcap": UniverseSpec(
            "midcap",
            root / "data" / "equities-midcap-tiingo",
            root / "data" / "options-iv" / "samples-midcap.jsonl",
            root / "data" / "fundamentals" / "equities",
        ),
    }


def build_universe_panel(spec: UniverseSpec) -> PanelData:
    return build_panel(spec.cache_dir, spec.samples, spec.fundamentals_dir)


def _check_universe_supports(panel: PanelData, spec: SignalSpec, universe: str) -> None:
    """Spec section 6: a universe/signal mismatch is refused at assembly time,
    never silently skipped (a silent skip would corrupt the trial count)."""
    if spec.requires_options and not panel.options:
        raise SweepError(
            f"signal {spec.name!r} requires options cells; universe {universe!r} has none"
        )
    if spec.requires_fundamentals and not panel.fundamentals:
        raise SweepError(
            f"signal {spec.name!r} requires fundamentals; universe {universe!r} has none"
        )


# --------------------------------------------------------------------------- #
# One trial: panel + signal + window -> the spec section-4 result payload
# --------------------------------------------------------------------------- #
def _window_bounds(window: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_s, _, end_s = window.partition("..")
    if not end_s:
        raise SweepError(f"window must be 'YYYY-MM-DD..YYYY-MM-DD', got {window!r}")
    return pd.Timestamp(start_s, tz="UTC"), pd.Timestamp(end_s, tz="UTC")


def _series_moments(returns: pd.Series) -> tuple[float, float]:
    """(skew, Pearson kurtosis) of a daily series -- the DSR's inputs."""
    r = returns.dropna().to_numpy()
    if len(r) < 4:
        return math.nan, math.nan
    mean = r.mean()
    sd = r.std()  # population, per the DSR definition
    if sd == 0:
        return math.nan, math.nan
    skew = float(((r - mean) ** 3).mean() / sd**3)
    kurt = float(((r - mean) ** 4).mean() / sd**4)  # normal = 3
    return skew, kurt


def _daily_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return math.nan
    sd = float(r.std(ddof=1))
    return float(r.mean()) / sd if sd > 0 else math.nan


def _leg_stats(alpha: AlphaResult, returns: pd.Series) -> dict:
    """The journaled per-leg payload (spec section 4 'ls'/'lo' blocks).

    Everything the leaderboard and DSR need is HERE, so `leaderboard` can be
    recomputed from the journal alone -- no panel rebuild, no factor refetch.
    """
    four = alpha.four_factor
    df = four.n - len(four.names)
    skew, kurt = _series_moments(returns)
    return {
        "alpha_annual_pct": four.alpha_annual_pct,
        "alpha_t": four.alpha_tstat,
        "p": stats.p_from_t(four.alpha_tstat, df),
        "capm_alpha_annual_pct": alpha.capm_alpha_annual_pct,
        "capm_alpha_t": alpha.capm_alpha_tstat,
        "loadings": {
            name: float(b) for name, b in zip(four.names[1:], four.beta[1:], strict=True)
        },
        "loadings_t": {
            name: float(t) for name, t in zip(four.names[1:], four.tstat[1:], strict=True)
        },
        "r2": four.r2,
        "n_obs": four.n,
        "sharpe": alpha.sharpe_annual,
        "sharpe_daily": _daily_sharpe(returns),
        "skew": skew,
        "kurt": kurt,
    }


def evaluate_trial(
    panel: PanelData,
    spec: SignalSpec,
    window: str,
    factors: pd.DataFrame,
    *,
    quantiles: int = QUANTILES,
    tercile_below: int = TERCILE_BELOW,
    min_names: int = MIN_NAMES,
) -> dict:
    """Score -> sort -> regress. Raises SortError/ValueError/LinAlgError on
    failure; the caller journals that as an error trial."""
    start, end = _window_bounds(window)
    dates = panel.decision_dates(start, end)
    sort = portfolio_sort(
        panel, spec, dates, end,
        quantiles=quantiles, tercile_below=tercile_below, min_names=min_names,
    )
    ls_alpha = evaluate_alpha(sort.ls, factors, self_financing=True)
    lo_alpha = evaluate_alpha(sort.lo, factors, self_financing=False)
    return {
        "n_dates": sort.n_dates,
        "n_names_median": sort.n_names_median,
        "ls": _leg_stats(ls_alpha, sort.ls),
        "lo": _leg_stats(lo_alpha, sort.lo),
        "turnover_monthly": sort.turnover_monthly,
        "skipped_dates": list(sort.skipped_dates),
    }


# --------------------------------------------------------------------------- #
# Leaderboard: recomputed from the journal ALONE (the auditable view)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LeaderboardRow:
    signal: str
    universe: str
    window: str
    alpha_annual_pct: float | None
    alpha_t: float | None
    p: float | None
    bh_pass: bool
    dsr: float | None
    capm_alpha_annual_pct: float | None
    capm_alpha_t: float | None
    loadings: dict
    turnover_monthly: float | None
    lo_alpha_t: float | None
    n_names_median: float | None
    n_dates: int | None
    skipped_dates: int
    error: str | None


def _pval(trial: dict) -> float:
    p = (trial.get("ls") or {}).get("p")
    return float("nan") if p is None else float(p)


def _abs_t_key(row: LeaderboardRow) -> float:
    if row.alpha_t is None:
        return 0.0
    t = float(row.alpha_t)
    return -abs(t) if not math.isnan(t) else 0.0


def build_leaderboard(journal: Journal) -> tuple[list[LeaderboardRow], int]:
    """(rows sorted by |4F L/S t| desc, honest discovery-trial count).

    BH is computed across EVERY journaled discovery trial -- prior sweeps
    included -- never just the current run (spec 3.5). Error trials carry
    p=NaN -> 1.0: they cannot pass but they raise the bar for everyone.
    """
    trials = discovery_trials(journal)
    n_trials = len(trials)
    if n_trials == 0:
        return [], 0
    mask = stats.bh_fdr(np.array([_pval(t) for t in trials]), q=BH_Q)
    daily_sharpes = [
        t["ls"]["sharpe_daily"]
        for t in trials
        if t.get("ls") and t["ls"].get("sharpe_daily") is not None
    ]
    var_sr = float(np.var(daily_sharpes, ddof=1)) if len(daily_sharpes) >= 2 else 0.0
    rows: list[LeaderboardRow] = []
    for trial, passed in zip(trials, mask, strict=True):
        ls = trial.get("ls") or {}
        lo = trial.get("lo") or {}
        dsr = None
        if passed and ls.get("sharpe_daily") is not None:
            dsr = stats.deflated_sharpe(
                sr=float(ls["sharpe_daily"]),
                n_obs=int(ls["n_obs"]),
                skew=float(ls["skew"]) if ls.get("skew") is not None else 0.0,
                kurt=float(ls["kurt"]) if ls.get("kurt") is not None else 3.0,
                n_trials=n_trials,
                var_trials_sr=var_sr,
            )
            if math.isnan(dsr):
                dsr = None  # keep leaderboard rows strictly JSON-serializable
        rows.append(
            LeaderboardRow(
                signal=trial["signal"],
                universe=trial["universe"],
                window=trial["window"],
                alpha_annual_pct=ls.get("alpha_annual_pct"),
                alpha_t=ls.get("alpha_t"),
                p=ls.get("p"),
                bh_pass=bool(passed),
                dsr=dsr,
                capm_alpha_annual_pct=ls.get("capm_alpha_annual_pct"),
                capm_alpha_t=ls.get("capm_alpha_t"),
                loadings=ls.get("loadings") or {},
                turnover_monthly=trial.get("turnover_monthly"),
                lo_alpha_t=lo.get("alpha_t"),
                n_names_median=trial.get("n_names_median"),
                n_dates=trial.get("n_dates"),
                skipped_dates=len(trial.get("skipped_dates") or []),
                error=trial.get("error"),
            )
        )
    rows.sort(key=_abs_t_key)
    return rows, n_trials


# --------------------------------------------------------------------------- #
# The sweep runner (spec 3.6)
# --------------------------------------------------------------------------- #
def run_sweep(
    universes: dict[str, UniverseSpec],
    journal: Journal,
    factors: pd.DataFrame,
    ts: str,
    *,
    signals: dict[str, SignalSpec] | None = None,
    window: str = DISCOVERY_WINDOW,
    quantiles: int = QUANTILES,
    tercile_below: int = TERCILE_BELOW,
    min_names: int = MIN_NAMES,
    panel_factory: Callable[[UniverseSpec], PanelData] = build_universe_panel,
) -> tuple[list[LeaderboardRow], int]:
    """Enumerate signals x universes serially; build each panel once; journal
    EVERY trial BEFORE the leaderboard is computed (spec 3.6) so a crash
    mid-sweep can never yield counted-but-unjournaled trials."""
    chosen = signals or SIGNALS
    params = {"quantiles": quantiles, "weighting": "equal", "cadence": "monthly"}
    for _, uspec in sorted(universes.items()):
        panel = panel_factory(uspec)
        for name in sorted(chosen):  # refuse the whole universe up front
            _check_universe_supports(panel, chosen[name], uspec.name)
        for name in sorted(chosen):
            config = trial_config(name, uspec.name, window, params=params)
            try:
                result: dict | None = evaluate_trial(
                    panel, chosen[name], window, factors,
                    quantiles=quantiles, tercile_below=tercile_below,
                    min_names=min_names,
                )
                # Spec section 6: corrupt cells are skipped AND counted; the
                # count rides on every trial event so coverage loss is audible.
                result["corrupt_cells"] = panel.corrupt_cells
                error = None
            except (SortError, ValueError, np.linalg.LinAlgError) as exc:
                result = None
                error = f"{type(exc).__name__}: {exc}"
            log_trial(journal, kind="discovery", config=config, ts=ts,
                      result=result, error=error)
    return build_leaderboard(journal)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_alphasearch_sweep.py -v`
Expected: all 8 tests PASS

- [ ] **Step 6: Run the full suite (guard against regressions)**

Run: `uv run pytest`
Expected: ALL PASS

- [ ] **Step 7: Lint and commit**

Run: `uv run ruff check src/trading/alphasearch tests/alphasearch_helpers.py tests/test_alphasearch_sweep.py`
Expected: clean

```bash
git add src/trading/alphasearch/sweep.py tests/alphasearch_helpers.py tests/test_alphasearch_sweep.py
git commit -m "Add alphasearch sweep runner + journal-derived leaderboard [AI]

Trials are journaled before any leaderboard exists and the leaderboard is
recomputed from the journal alone, so BH always spans every discovery
trial ever run -- prior sweeps included -- and a crash mid-sweep can't
produce counted-but-unjournaled results. Universe/signal mismatches are
refused at assembly (a silent skip would corrupt the trial count); errored
trials are flagged, carry p=1, and still spend a trial."
```

---

### Task 9: `sweep.py` part 3 — the journal-enforced holdout

**Files:**
- Modify: `src/trading/alphasearch/sweep.py` (append the holdout section)
- Test: extend `tests/test_alphasearch_sweep.py`

**Interfaces:**
- Consumes: Task 7 `prior_holdout_trial`, `find_discovery_trial`, `log_trial`; Task 8 `build_leaderboard`, `evaluate_trial`, `build_universe_panel`, `UniverseSpec`.
- Produces (Task 10 relies on these exact signatures):
  - `RERUN_CONFIRMATION = "RERUN HOLDOUT"`
  - `holdout_passes(discovery_alpha: float, holdout_alpha: float) -> bool`
  - `latest_bar_date(panel: PanelData) -> pd.Timestamp`
  - `HoldoutOutcome` frozen dataclass — `event: dict`, `passed: bool | None` (None when the holdout trial errored), `discovery_alpha: float`, `holdout_alpha: float | None`, `window: str`.
  - `run_holdout(uspec: UniverseSpec, journal: Journal, factors: pd.DataFrame, ts: str, signal_name: str, *, holdout_start: str = HOLDOUT_START, discovery_window: str = DISCOVERY_WINDOW, confirm: Callable[[], str] = lambda: "", quantiles: int = QUANTILES, tercile_below: int = TERCILE_BELOW, min_names: int = MIN_NAMES, panel_factory: Callable[[UniverseSpec], PanelData] = build_universe_panel) -> HoldoutOutcome`

Behavior (spec §3.6 + §5.3, restated):
- Refuse (`SweepError`) when: the signal is unknown; no default-params discovery trial exists for (signal, universe); the discovery trial errored; the (signal, universe) is not a CURRENT BH survivor (spending the once-only holdout on a non-survivor would waste it — the CLI surfaces the message).
- If the journal already holds a holdout event for (signal, universe): call `confirm()`; anything but the literal `RERUN HOLDOUT` -> `SweepError` (mirrors `backtest/experiments.py::prior_holdout` + the CLI confirmation in `_cmd_backtest`).
- Window: `holdout_start..latest_bar_date(panel)` — the actual end date is IN the journaled window string, making the evaluation exactly reproducible.
- Pass rule (pre-registered): holdout 4F L/S alpha has the same sign as discovery AND its point estimate retains >= 0.5x the discovery magnitude. Both conditions are exactly `holdout_alpha / discovery_alpha >= 0.5` (the ratio is positive iff same-signed). Note: the spec's literal "holdout >= 0.5 x discovery" reading would PASS a −4%/yr holdout against a −10%/yr discovery while failing the equivalent positive case; the ratio form applies the intended magnitude test symmetrically and is the pre-registered rule implemented here.
- The holdout trial is journaled (kind `"holdout"`) whether it passes, fails, or errors.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_alphasearch_sweep.py`. IMPORTANT: merge
`RERUN_CONFIRMATION, holdout_passes, prior_holdout_trial, run_holdout` into
the file's existing top `from trading.alphasearch.sweep import (...)` block —
a mid-file import is a ruff E402 error.

```python
# --------------------------------------------------------------------------- #
# Holdout (Task 9)
# --------------------------------------------------------------------------- #

DISCOVERY = "2020-01-01..2020-03-31"
HOLDOUT_FROM = "2020-04-01"


def _sweep_then_holdout_setup(tmp_path):
    """Discovery on Q1 2020; the fixture's remaining bars are the holdout."""
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=DISCOVERY,
              panel_factory=lambda _u: panel)
    return journal, panel, factors


def test_holdout_pass_rule_is_signed_ratio():
    assert holdout_passes(10.0, 6.0) is True      # kept 60% of the effect
    assert holdout_passes(10.0, 4.9) is False     # faded below half
    assert holdout_passes(10.0, -6.0) is False    # flipped sign
    assert holdout_passes(-10.0, -6.0) is True    # negative alphas: same rule
    assert holdout_passes(-10.0, -4.0) is False
    assert holdout_passes(-10.0, 6.0) is False
    assert holdout_passes(0.0, 1.0) is False      # degenerate discovery
    assert holdout_passes(float("nan"), 1.0) is False


def test_holdout_runs_once_and_journals_with_end_date(tmp_path):
    journal, panel, factors = _sweep_then_holdout_setup(tmp_path)
    outcome = run_holdout(
        _universe(tmp_path)["largecap"], journal, factors, "t2", "mom21",
        holdout_start=HOLDOUT_FROM, discovery_window=DISCOVERY,
        panel_factory=lambda _u: panel,
    )
    latest = max(s.index.max() for s in panel.closes.values())
    assert outcome.window == f"{HOLDOUT_FROM}..{latest.date().isoformat()}"
    prior = prior_holdout_trial(journal, "mom21", "largecap")
    assert prior is not None and prior["kind"] == "holdout"
    assert prior["window"] == outcome.window       # reproducible end date
    assert outcome.passed in (True, False)
    assert outcome.holdout_alpha is not None


def test_holdout_double_touch_refused_without_literal_confirmation(tmp_path):
    journal, panel, factors = _sweep_then_holdout_setup(tmp_path)
    kwargs = dict(holdout_start=HOLDOUT_FROM, discovery_window=DISCOVERY,
                  panel_factory=lambda _u: panel)
    run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t2",
                "mom21", **kwargs)
    events_before = len(list(journal.events()))
    # Default confirm refuses; a wrong phrase refuses; nothing is journaled.
    with pytest.raises(SweepError):
        run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t3",
                    "mom21", **kwargs)
    with pytest.raises(SweepError):
        run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t3",
                    "mom21", confirm=lambda: "yes please", **kwargs)
    assert len(list(journal.events())) == events_before
    # The literal phrase re-runs (and appends a fresh holdout event).
    run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t4",
                "mom21", confirm=lambda: RERUN_CONFIRMATION, **kwargs)
    assert len(list(journal.events())) == events_before + 1


def test_holdout_refused_without_discovery_trial(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    with pytest.raises(SweepError):
        run_holdout(_universe(tmp_path)["largecap"], journal, make_factors(),
                    "t1", "mom21", holdout_start=HOLDOUT_FROM,
                    discovery_window=DISCOVERY, panel_factory=lambda _u: panel)
    assert list(journal.events()) == []


def test_holdout_refused_for_non_bh_survivor(tmp_path):
    # Fabricate a deterministic non-survivor: a clean discovery event whose
    # p-value can never clear BH, alongside mom21's real (surviving) trial.
    from trading.alphasearch.sweep import log_trial, trial_config

    journal, panel, factors = _sweep_then_holdout_setup(tmp_path)
    dull = _result_like(alpha_annual_pct=0.3, alpha_t=0.1, p=0.92)
    log_trial(journal, kind="discovery",
              config=trial_config("rvol21", "largecap", DISCOVERY),
              ts="t1b", result=dull)
    rows, _ = build_leaderboard(journal)
    rvol = next(r for r in rows if r.signal == "rvol21")
    assert not rvol.bh_pass  # precondition for the refusal below
    with pytest.raises(SweepError):
        run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t2",
                    "rvol21", holdout_start=HOLDOUT_FROM,
                    discovery_window=DISCOVERY, panel_factory=lambda _u: panel)


def _result_like(*, alpha_annual_pct: float, alpha_t: float, p: float) -> dict:
    """Minimal spec-section-4 result payload for fabricated journal events."""
    leg = {
        "alpha_annual_pct": alpha_annual_pct, "alpha_t": alpha_t, "p": p,
        "capm_alpha_annual_pct": alpha_annual_pct, "capm_alpha_t": alpha_t,
        "loadings": {}, "loadings_t": {}, "r2": 0.0, "n_obs": 120,
        "sharpe": 0.1, "sharpe_daily": 0.006, "skew": 0.0, "kurt": 3.0,
    }
    return {"n_dates": 3, "n_names_median": 16.0, "ls": leg, "lo": dict(leg),
            "turnover_monthly": 0.3, "skipped_dates": []}


def test_unknown_signal_refused(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    with pytest.raises(SweepError):
        run_holdout(_universe(tmp_path)["largecap"], journal, make_factors(),
                    "t1", "no_such_signal")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_sweep.py -v`
Expected: the six new tests FAIL with `ImportError: cannot import name 'RERUN_CONFIRMATION'`; the Task 8 tests still pass.

- [ ] **Step 3: Append the holdout section to `src/trading/alphasearch/sweep.py`**

```python
# --------------------------------------------------------------------------- #
# Holdout re-prove (spec 3.6 + 5.3): touched once per (signal, universe),
# journal-enforced, mirroring backtest/experiments.py::prior_holdout.
# --------------------------------------------------------------------------- #
RERUN_CONFIRMATION = "RERUN HOLDOUT"


def holdout_passes(discovery_alpha: float, holdout_alpha: float) -> bool:
    """Pre-registered pass rule: same sign AND the holdout point estimate
    retains >= HOLDOUT_PASS_RATIO of the discovery magnitude. Both conditions
    collapse to the signed ratio (positive iff same-signed), which applies the
    magnitude test symmetrically for negative-alpha candidates."""
    if (
        discovery_alpha == 0
        or math.isnan(discovery_alpha)
        or math.isnan(holdout_alpha)
    ):
        return False
    return holdout_alpha / discovery_alpha >= HOLDOUT_PASS_RATIO


def latest_bar_date(panel: PanelData) -> pd.Timestamp:
    return max(series.index.max() for series in panel.closes.values())


@dataclass(frozen=True)
class HoldoutOutcome:
    event: dict
    passed: bool | None          # None when the holdout evaluation errored
    discovery_alpha: float
    holdout_alpha: float | None
    window: str


def run_holdout(
    uspec: UniverseSpec,
    journal: Journal,
    factors: pd.DataFrame,
    ts: str,
    signal_name: str,
    *,
    holdout_start: str = HOLDOUT_START,
    discovery_window: str = DISCOVERY_WINDOW,
    confirm: Callable[[], str] = lambda: "",
    quantiles: int = QUANTILES,
    tercile_below: int = TERCILE_BELOW,
    min_names: int = MIN_NAMES,
    panel_factory: Callable[[UniverseSpec], PanelData] = build_universe_panel,
) -> HoldoutOutcome:
    """Evaluate ONE BH survivor on the reserved holdout window.

    Refusals (SweepError) protect the once-only holdout: unknown signal; no
    clean default-params discovery trial; not a current BH survivor; already
    holdout-touched unless confirm() returns the literal RERUN_CONFIRMATION.
    The realized window end (latest bar) is journaled so the evaluation is
    exactly reproducible.
    """
    if signal_name not in SIGNALS:
        known = ", ".join(sorted(SIGNALS))
        raise SweepError(f"unknown signal {signal_name!r}; known: {known}")
    discovery = find_discovery_trial(journal, signal_name, uspec.name, discovery_window)
    if discovery is None:
        raise SweepError(
            f"no discovery trial for {signal_name}:{uspec.name} over "
            f"{discovery_window}; run the sweep first"
        )
    if discovery.get("error"):
        raise SweepError(
            f"discovery trial for {signal_name}:{uspec.name} errored "
            f"({discovery['error']}); nothing to re-prove"
        )
    rows, _ = build_leaderboard(journal)
    survivor = any(
        row.signal == signal_name and row.universe == uspec.name and row.bh_pass
        for row in rows
    )
    if not survivor:
        raise SweepError(
            f"{signal_name}:{uspec.name} is not a current BH survivor "
            f"(q={BH_Q}); the once-only holdout is reserved for survivors"
        )
    prior = prior_holdout_trial(journal, signal_name, uspec.name)
    if prior is not None and confirm() != RERUN_CONFIRMATION:
        raise SweepError(
            f"holdout for {signal_name}:{uspec.name} already evaluated at "
            f"{prior['ts']}; rerunning invalidates the evidence — aborted"
        )

    panel = panel_factory(uspec)
    spec = SIGNALS[signal_name]
    _check_universe_supports(panel, spec, uspec.name)
    window = f"{holdout_start}..{latest_bar_date(panel).date().isoformat()}"
    config = trial_config(signal_name, uspec.name, window)
    try:
        result: dict | None = evaluate_trial(
            panel, spec, window, factors,
            quantiles=quantiles, tercile_below=tercile_below, min_names=min_names,
        )
        error = None
    except (SortError, ValueError, np.linalg.LinAlgError) as exc:
        result = None
        error = f"{type(exc).__name__}: {exc}"
    event = log_trial(journal, kind="holdout", config=config, ts=ts,
                      result=result, error=error)

    discovery_alpha = float(discovery["ls"]["alpha_annual_pct"])
    if result is None:
        return HoldoutOutcome(event, None, discovery_alpha, None, window)
    holdout_alpha = float(result["ls"]["alpha_annual_pct"])
    return HoldoutOutcome(
        event,
        holdout_passes(discovery_alpha, holdout_alpha),
        discovery_alpha,
        holdout_alpha,
        window,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_alphasearch_sweep.py -v`
Expected: all 14 tests PASS

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check src/trading/alphasearch tests/test_alphasearch_sweep.py`
Expected: clean

```bash
git add src/trading/alphasearch/sweep.py tests/test_alphasearch_sweep.py
git commit -m "Add journal-enforced once-only holdout to alphasearch [AI]

The holdout is spent the first time it is read: a second touch refuses
unless the literal RERUN HOLDOUT confirmation arrives (mirroring the
backtest go-live gate), only current BH survivors may spend it, and the
realized window end date is journaled so the evaluation is exactly
reproducible. Pass rule pre-registered as the signed alpha ratio >= 0.5."
```

---

### Task 10: CLI — `trading alphasearch sweep | leaderboard | holdout`

**Files:**
- Modify: `src/trading/cli.py` (parser in `build_parser`, handler mapping in `main`, `_cmd_alphasearch` + `_print_alphasearch_leaderboard` near the other `_cmd_*` functions)
- Test: `tests/test_alphasearch_cli.py`

**Interfaces:**
- Consumes: Task 7-9 `trials_journal`, `build_leaderboard`, `run_sweep`, `run_holdout`, `default_universes`, `SweepError`, `LeaderboardRow`; Task 2 `load_factors`; Task 5 `PanelError`; existing `_utcnow` (the CLI is the only clock reader).
- Produces: the `trading alphasearch` subcommand. Follows the repo CLI conventions: `action` positional with choices (like `schedule`), lazy submodule import inside the handler (like `_cmd_schedule`), errors to stderr + exit 1, `--json` for machine output, confirmation prompts read stdin with `EOFError` treated as refusal (like `_cmd_backtest`'s `RERUN HOLDOUT`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alphasearch_cli.py`:

```python
"""CLI wiring for `trading alphasearch` (journal-only paths; no data dirs)."""

from __future__ import annotations

import json

import pandas as pd

from trading import cli
from trading.alphasearch.sweep import (
    DISCOVERY_WINDOW,
    log_trial,
    trial_config,
    trials_journal,
)


def _seed_journal(journal_dir, *, with_holdout: bool = False) -> None:
    journal = trials_journal(journal_dir)
    leg = {"alpha_annual_pct": 8.0, "alpha_t": 4.2, "p": 1e-4,
           "capm_alpha_annual_pct": 9.0, "capm_alpha_t": 4.4,
           "loadings": {"Mkt-RF": 0.1, "SMB": 0.0, "HML": 0.0, "Mom": 0.0},
           "loadings_t": {"Mkt-RF": 1.0, "SMB": 0.0, "HML": 0.0, "Mom": 0.0},
           "r2": 0.1, "n_obs": 1200, "sharpe": 1.1, "sharpe_daily": 0.07,
           "skew": -0.2, "kurt": 4.0}
    result = {"n_dates": 60, "n_names_median": 97.0, "ls": leg, "lo": dict(leg),
              "turnover_monthly": 0.35, "skipped_dates": []}
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", DISCOVERY_WINDOW),
              ts="t1", result=result)
    if with_holdout:
        log_trial(journal, kind="holdout",
                  config=trial_config("mom21", "largecap", "2024-01-01..2026-07-07"),
                  ts="t2", result=result)


def test_leaderboard_json_from_journal(tmp_path, capsys):
    _seed_journal(tmp_path)
    rc = cli.main(["alphasearch", "leaderboard", "--journal-dir", str(tmp_path),
                   "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["trials"] == 1
    assert payload["rows"][0]["signal"] == "mom21"
    assert payload["rows"][0]["bh_pass"] is True


def test_leaderboard_table_renders(tmp_path, capsys):
    _seed_journal(tmp_path)
    rc = cli.main(["alphasearch", "leaderboard", "--journal-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mom21" in out
    assert "honest trial count" in out


def test_leaderboard_empty_journal_is_fine(tmp_path, capsys):
    rc = cli.main(["alphasearch", "leaderboard", "--journal-dir", str(tmp_path),
                   "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["trials"] == 0 and payload["rows"] == []


def test_sweep_unknown_signal_rejected_before_any_io(tmp_path, capsys):
    rc = cli.main(["alphasearch", "sweep", "--signals", "nope,mom21",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "unknown signals: nope" in capsys.readouterr().err
    assert not (tmp_path / "alphasearch-trials.jsonl").exists()


def test_holdout_requires_trial_id(tmp_path, capsys):
    rc = cli.main(["alphasearch", "holdout", "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "signal:universe" in capsys.readouterr().err


def test_holdout_unknown_universe_rejected(tmp_path, capsys):
    rc = cli.main(["alphasearch", "holdout", "mom21:smallcap",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "unknown universe" in capsys.readouterr().err


def test_holdout_double_touch_refused_via_prompt(tmp_path, capsys, monkeypatch):
    _seed_journal(tmp_path, with_holdout=True)
    # Factors are irrelevant to the refusal path; keep the test offline.
    monkeypatch.setattr("trading.alphasearch.evaluate.load_factors",
                        lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    rc = cli.main(["alphasearch", "holdout", "mom21:largecap",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "already evaluated" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_cli.py -v`
Expected: FAIL — `cli.main` raises `SystemExit` from argparse (`invalid choice: 'alphasearch'`)

- [ ] **Step 3: Wire the subcommand into `src/trading/cli.py`**

In `build_parser()`, after the `backtest` parser block and before `sched`:

```python
    alphasearch = sub.add_parser(
        "alphasearch",
        help="cheap alpha search: sweep signals x universes, leaderboard, holdout",
    )
    alphasearch.add_argument("action", choices=["sweep", "leaderboard", "holdout"])
    alphasearch.add_argument(
        "trial",
        nargs="?",
        default=None,
        help="holdout target as signal:universe (e.g. mom126:largecap)",
    )
    alphasearch.add_argument(
        "--universe",
        choices=["largecap", "midcap", "all"],
        default="all",
        help="sweep scope (sweep only)",
    )
    alphasearch.add_argument(
        "--signals",
        default=None,
        help="comma-separated signal subset for the sweep; every run is still "
        "a journaled trial",
    )
    alphasearch.add_argument("--journal-dir", default="journal", help="journal root")
    alphasearch.add_argument(
        "--factors-dir", default="data/factors", help="Ken French factor cache directory"
    )
    alphasearch.add_argument(
        "--refresh-factors", action="store_true", help="re-download the factor CSVs"
    )
    alphasearch.add_argument("--json", action="store_true", help="machine-readable output")
```

In `main()`, add to the handlers dict:

```python
        "alphasearch": _cmd_alphasearch,
```

Add the handler + renderer (place after `_cmd_backtest`'s helpers). The
submodule import is lazy inside the handler, matching `_cmd_schedule`:

```python
def _cmd_alphasearch(args: argparse.Namespace) -> int:
    from trading.alphasearch import sweep as engine
    from trading.alphasearch.panel import PanelError

    journal = engine.trials_journal(Path(args.journal_dir))

    if args.action == "leaderboard":
        # The auditable view: recomputed from the journal alone, no new trials.
        rows, count = engine.build_leaderboard(journal)
        _print_alphasearch_leaderboard(rows, count, as_json=args.json)
        return 0

    if args.action == "sweep":
        signals = None
        if args.signals:
            from trading.alphasearch.spec import SIGNALS

            names = [n.strip() for n in args.signals.split(",") if n.strip()]
            unknown = sorted(set(names) - set(SIGNALS))
            if unknown:
                print(f"ERROR: unknown signals: {', '.join(unknown)}", file=sys.stderr)
                return 1
            signals = {n: SIGNALS[n] for n in names}
        factors = _load_alphasearch_factors(args)
        if factors is None:
            return 1
        universes = engine.default_universes(Path("."))
        if args.universe != "all":
            universes = {args.universe: universes[args.universe]}
        try:
            rows, count = engine.run_sweep(
                universes, journal, factors, _utcnow().isoformat(), signals=signals
            )
        except (engine.SweepError, PanelError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        _print_alphasearch_leaderboard(rows, count, as_json=args.json)
        return 0

    # holdout
    if not args.trial or ":" not in args.trial:
        print(
            "ERROR: holdout needs a trial id: signal:universe (e.g. mom126:largecap)",
            file=sys.stderr,
        )
        return 1
    signal_name, _, universe = args.trial.partition(":")
    universes = engine.default_universes(Path("."))
    if universe not in universes:
        print(
            f"ERROR: unknown universe {universe!r}; choose from "
            f"{', '.join(sorted(universes))}",
            file=sys.stderr,
        )
        return 1
    factors = _load_alphasearch_factors(args)
    if factors is None:
        return 1

    def confirm() -> str:
        # Same ceremony as the backtest holdout: spent the first time it is
        # read; stdout stays clean for --json, prompts live on stderr.
        print(
            "Holdout already evaluated for this candidate (journaled). "
            "Rerunning it invalidates the evidence (spec).",
            file=sys.stderr,
        )
        try:
            return input("Type RERUN HOLDOUT to run it anyway: ").strip()
        except EOFError:  # non-interactive stdin: refusal, not a crash
            return ""

    try:
        outcome = engine.run_holdout(
            universes[universe], journal, factors, _utcnow().isoformat(),
            signal_name, confirm=confirm,
        )
    except (engine.SweepError, PanelError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(
            json.dumps(
                {
                    "signal": signal_name,
                    "universe": universe,
                    "window": outcome.window,
                    "discovery_alpha_annual_pct": outcome.discovery_alpha,
                    "holdout_alpha_annual_pct": outcome.holdout_alpha,
                    "passed": outcome.passed,
                    "error": outcome.event.get("error"),
                }
            )
        )
        return 0 if outcome.holdout_alpha is not None else 1
    print(f"holdout {signal_name}:{universe} over {outcome.window}")
    if outcome.holdout_alpha is None:
        print(
            f"ERRORED: {outcome.event.get('error')} (journaled; still spends a trial)",
            file=sys.stderr,
        )
        return 1
    print(
        f"discovery alpha {outcome.discovery_alpha:+.1f}%/yr -> "
        f"holdout {outcome.holdout_alpha:+.1f}%/yr"
    )
    print(
        "pass rule (same sign, >=50% of the effect retained): "
        + ("PASS" if outcome.passed else "FAIL")
    )
    return 0


def _load_alphasearch_factors(args: argparse.Namespace):
    """Factors or None (error already printed). Offline with a warm cache;
    a cold cache offline is a hard error instructing --refresh-factors."""
    from trading.alphasearch.evaluate import load_factors

    try:
        return load_factors(Path(args.factors_dir), refresh=args.refresh_factors)
    except OSError as exc:
        print(
            f"ERROR: factor data unavailable ({exc}); run once online with "
            "--refresh-factors to (re)build the cache",
            file=sys.stderr,
        )
        return None


def _print_alphasearch_leaderboard(rows, count: int, *, as_json: bool) -> None:
    if as_json:
        from dataclasses import asdict

        print(json.dumps({"trials": count, "bh_q": 0.10,
                          "rows": [asdict(r) for r in rows]}))
        return

    from rich.console import Console
    from rich.table import Table

    def num(value, fmt="{:+.2f}"):
        return "-" if value is None else fmt.format(value)

    console = Console()
    table = Table(
        title=f"alphasearch leaderboard — {count} discovery trials, BH q=0.10"
    )
    for col in ["signal", "universe", "4F a%/yr", "t", "p", "BH", "DSR", "CAPM t",
                "bMkt", "bSMB", "bHML", "bMom", "turn", "names", "error"]:
        table.add_column(col, justify="right")
    for r in rows:
        table.add_row(
            r.signal,
            r.universe,
            num(r.alpha_annual_pct, "{:+.1f}"),
            num(r.alpha_t),
            num(r.p, "{:.4f}"),
            "PASS" if r.bh_pass else "-",
            num(r.dsr, "{:.3f}"),
            num(r.capm_alpha_t),
            num(r.loadings.get("Mkt-RF")),
            num(r.loadings.get("SMB")),
            num(r.loadings.get("HML")),
            num(r.loadings.get("Mom")),
            num(r.turnover_monthly, "{:.2f}"),
            num(r.n_names_median, "{:.0f}"),
            r.error or "",
        )
    console.print(table)
    console.print(
        f"honest trial count: {count} journaled discovery trials — the BH gate "
        "is computed across ALL of them, not just this run"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_alphasearch_cli.py tests/test_cli.py -v`
Expected: ALL PASS (existing CLI tests untouched)

- [ ] **Step 5: Smoke the help output**

Run: `uv run trading alphasearch --help`
Expected: usage listing `{sweep,leaderboard,holdout}`, the `trial` positional, and the `--universe/--signals/--journal-dir/--factors-dir/--refresh-factors/--json` flags.

Run: `uv run trading alphasearch leaderboard --journal-dir /tmp/claude-empty-journal`
Expected: an empty leaderboard table + `honest trial count: 0 ...`, exit 0.

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check src/trading/cli.py tests/test_alphasearch_cli.py`
Expected: clean

```bash
git add src/trading/cli.py tests/test_alphasearch_cli.py
git commit -m "Wire trading alphasearch sweep|leaderboard|holdout into the CLI [AI]

Follows the existing subcommand conventions: action positional (like
schedule), lazy engine import, stderr errors, --json machine output, and
the same stdin RERUN HOLDOUT ceremony as the backtest go-live gate. The
CLI stays the only clock reader; ts flows into the journal from here."
```

---

### Task 11: the no-look-ahead test + the golden end-to-end sweep

**Files:**
- Test: `tests/test_alphasearch_lookahead.py` (new)
- Test: `tests/test_alphasearch_golden.py` (new)

**Interfaces:**
- Consumes: `SIGNALS` (all 16), `PanelData`, `make_panel`/`make_cell`/`make_factors`/`month_firsts` from `tests/alphasearch_helpers.py`, `run_sweep`/`UniverseSpec`/`trials_journal`/`discovery_trials`, `FundamentalsStore`.
- Produces: nothing new — these are the adversarial guarantees (spec §7): (a) every registered signal, present and future, is mechanically checked for look-ahead; (b) the whole pipeline (real file I/O -> panel -> sort -> regression -> journal -> leaderboard) is pinned end-to-end on a deterministic fixture.

These are pure test tasks: both files exercise ONLY code that exists after
Task 10. If either fails, the bug is in earlier tasks — fix it there, do not
weaken the test.

- [ ] **Step 1: Write the no-look-ahead test**

Create `tests/test_alphasearch_lookahead.py`:

```python
"""THE no-look-ahead guarantee (spec section 7): perturb every store strictly
after a cutoff date T and assert every registered signal's scores at <= T are
bit-identical. Iterates SIGNALS, so any future signal is automatically
covered -- a new signal that peeks past as_of fails here by construction."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from alphasearch_helpers import make_panel
from trading.alphasearch.panel import PanelData
from trading.alphasearch.spec import SIGNALS

START = pd.Timestamp("2020-01-01", tz="UTC")
CUTOFF = pd.Timestamp("2020-03-15", tz="UTC")


def _perturb_after(panel: PanelData, cutoff: pd.Timestamp) -> PanelData:
    """Corrupt every data point strictly after cutoff, in every store."""
    closes: dict[str, pd.Series] = {}
    for sym, series in panel.closes.items():
        s = series.copy()
        late = s.index > cutoff
        s.loc[late] = s.loc[late] * 3.7 + 11.0
        closes[sym] = s
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
    return PanelData(closes=closes, options=options, fundamentals=fundamentals,
                     symbols=panel.symbols, corrupt_cells=panel.corrupt_cells)


def test_fixture_actually_has_data_after_the_cutoff():
    # Guard against a vacuous test: every store must carry post-cutoff rows.
    panel = make_panel()
    assert any((s.index > CUTOFF).any() for s in panel.closes.values())
    assert any((f.index > CUTOFF).any() for f in panel.options.values())
    assert any((f.index > CUTOFF).any() for f in panel.fundamentals.values())


def test_no_registered_signal_can_see_past_as_of():
    panel = make_panel()
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

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_alphasearch_lookahead.py -v`
Expected: 2 tests PASS. (If `test_no_registered_signal_can_see_past_as_of`
fails, a signal or a PanelView accessor is reading forward data — a
correctness bug in Task 3/4/5, not in this test.)

- [ ] **Step 3: Write the golden sweep test**

Create `tests/test_alphasearch_golden.py`:

```python
"""Golden end-to-end sweep (spec section 7): real files -> build_panel ->
sort -> regression -> journal -> leaderboard, on a deterministic fixture."""

from __future__ import annotations

import json

import pandas as pd

from alphasearch_helpers import make_cell, make_factors, make_panel, month_firsts
from trading.alphasearch.spec import SIGNALS
from trading.alphasearch.sweep import (
    UniverseSpec,
    discovery_trials,
    run_sweep,
    trials_journal,
)
from trading.fundamentals.store import FundamentalsStore

WINDOW = "2020-01-01..2020-06-30"


def _write_universe(tmp_path) -> UniverseSpec:
    """Materialize make_panel()'s exact data as real files: parquet bar
    caches, a samples.jsonl, and a fundamentals store."""
    panel = make_panel()
    cache = tmp_path / "cache"
    cache.mkdir()
    for sym in panel.symbols:
        closes = panel.closes[sym]
        pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes,
             "volume": 1000.0},
            index=closes.index,
        ).to_parquet(cache / f"{sym}.parquet")
    idx = panel.closes[panel.symbols[0]].index
    lines = []
    for date in month_firsts(idx):
        iso = date.date().isoformat()
        for i, sym in enumerate(panel.symbols):
            lines.append(json.dumps(make_cell(
                sym, iso,
                atm_iv=0.20 + 0.01 * i, put_iv=0.24 + 0.01 * i,
                call_iv=0.18 + 0.01 * i, skew_put_atm=0.02 + 0.005 * i,
                skew_put_call=0.01 + 0.002 * i,
            )))
    samples = tmp_path / "samples.jsonl"
    samples.write_text("\n".join(lines) + "\n")
    store = FundamentalsStore(tmp_path / "fundamentals")
    for sym in panel.symbols:
        store.append(sym, panel.fundamentals[sym])
    return UniverseSpec("largecap", cache, samples, tmp_path / "fundamentals")


def test_golden_sweep_end_to_end(tmp_path):
    uspec = _write_universe(tmp_path)
    journal = trials_journal(tmp_path / "journal")
    rows, n_trials = run_sweep({"largecap": uspec}, journal, make_factors(),
                               ts="t1", window=WINDOW)

    # Every registered signal became exactly one journaled trial.
    assert n_trials == len(SIGNALS) == 16
    assert {(r.signal, r.universe) for r in rows} == {
        (s, "largecap") for s in SIGNALS
    }
    # Deep-history momentum cannot exist on a 6-month fixture: honest errors
    # that are flagged AND still spend trials.
    errored = {r.signal for r in rows if r.error is not None}
    assert {"mom126", "mom252"} <= errored
    # The engineered momentum spread is a standout survivor with full stats.
    mom = next(r for r in rows if r.signal == "mom21")
    assert mom.bh_pass
    assert abs(mom.alpha_t) > 5
    assert mom.dsr is not None and 0.0 <= mom.dsr <= 1.0
    assert set(mom.loadings) == {"Mkt-RF", "SMB", "HML", "Mom"}
    assert mom.turnover_monthly is not None and 0.0 <= mom.turnover_monthly <= 0.6
    assert mom.n_names_median == 16.0
    # Ranking is by |4F L/S t| descending.
    ts = [abs(r.alpha_t) for r in rows if r.alpha_t is not None]
    assert ts == sorted(ts, reverse=True)

    # Identical re-run: journal grows (append-only) but nothing double-counts
    # and the statistics are bit-identical.
    rows2, n2 = run_sweep({"largecap": uspec}, journal, make_factors(),
                          ts="t2", window=WINDOW)
    assert n2 == n_trials
    assert [(r.signal, r.alpha_t, r.p) for r in rows2] == [
        (r.signal, r.alpha_t, r.p) for r in rows
    ]
    assert len(list(journal.events())) == 32
    assert len(discovery_trials(journal)) == 16
```

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_alphasearch_golden.py -v`
Expected: PASS. If the turnover or |t| bound fails marginally, inspect the
fixture values FIRST (`uv run python -c ...` printing the leaderboard row) —
the bound may need a one-time calibration to the seeded fixture, which is a
legitimate fixture adjustment, not a product change. Record any adjusted
bound in the test with a comment.

- [ ] **Step 5: Full suite + lint, then commit**

Run: `uv run pytest`
Expected: ALL PASS

Run: `uv run ruff check tests/test_alphasearch_lookahead.py tests/test_alphasearch_golden.py`
Expected: clean

```bash
git add tests/test_alphasearch_lookahead.py tests/test_alphasearch_golden.py
git commit -m "Add adversarial alphasearch tests: no-look-ahead + golden sweep [AI]

The look-ahead test perturbs bars/options/fundamentals strictly after a
cutoff and requires bit-identical scores at or before it, iterating the
registry so any FUTURE signal is automatically covered. The golden test
drives the full file->panel->sort->regression->journal->leaderboard path
on a deterministic fixture, including honest error trials and idempotent
re-runs."
```

---

### Task 12: documentation — glossary terms + experiments ledger stub

**Files:**
- Modify: `docs/glossary.md` (new themed section)
- Modify: `docs/experiments.md` (new §10 stub, inserted after §9 and before `## Known caveats affecting these numbers`)

**Interfaces:**
- Consumes: nothing (prose only). No code changes.
- Produces: the ledger discipline required by spec §8 — terms defined before the first real sweep runs; the experiments section that sweep will fill in.

- [ ] **Step 1: Add the new glossary section**

Append to `docs/glossary.md` (after the last existing section, keeping the
file's voice: term — plain-English definition, grounded in our results):

```markdown
---

## Multiple testing, and how the alpha-search engine stays honest

**Trial** — one evaluation of one (signal, universe, window, parameters)
combination. Every trial is journaled (`journal/alphasearch-trials.jsonl`)
whether it succeeds, fails, or errors — because the significance bar depends on
how many things were tried, an uncounted trial silently corrupts every later
p-value. Identical re-runs update in place (same `config_hash`); ANY parameter
change is a new trial.

**Multiple testing / data snooping** — run enough tests and "significant"
results appear by pure luck: at |t|>2, roughly 1 in 20 dead signals looks alive.
A sweep engine is a false-positive factory unless the bar rises with the number
of trials. This is the central risk the alpha-search engine is built around.

**False discovery rate (FDR)** — the expected fraction of your *accepted*
signals that are actually false. Controlling FDR at q=0.10 means: of the
candidates the gate passes, ~10% are expected to be flukes — a deliberate,
quantified tolerance, instead of the unquantified optimism of eyeballing top
rows.

**Benjamini-Hochberg (BH)** — the standard step-up procedure that controls FDR:
sort the n trial p-values ascending, find the largest k with
p_(k) <= (k/n)·q, accept the k smallest. The bar RISES as the journal grows —
running more trials makes every existing candidate harder to accept, which is
exactly the honesty we want. Our gate: q=0.10 across ALL journaled discovery
trials.

**Deflated Sharpe Ratio (DSR)** — Bailey & López de Prado's correction to
"best backtest of N": the probability the candidate's true Sharpe exceeds zero
after accounting for how many trials were run, the spread of Sharpes across
them, and fat tails/skew in the candidate's returns. Reported (advisory) for
BH survivors; DSR near 1 = likely real, near 0.5 = coin flip.

**Portfolio sort** — the cheap way to turn a signal into a return series
without a backtest: each month, rank the universe by the signal, go long the
top quantile and short the bottom (equal weight), hold to the next rebalance.
The daily long/short spread isolates the signal from the market; its
four-factor alpha t-stat is our gate statistic. Quintiles normally, terciles
under 50 names, skip (and journal) under 15.

**Discovery vs holdout window** — discovery (2019-01-01..2023-12-31) is where
the sweep is allowed to look; the holdout (2024-01-01..latest) is spent the
FIRST time a candidate reads it, enforced by the trial journal exactly like
the go-live holdout. Pass rule, pre-registered: same alpha sign AND >= 50% of
the discovery alpha magnitude retained.
```

- [ ] **Step 2: Add the experiments-ledger stub**

In `docs/experiments.md`, insert immediately after the end of section 9
(before the line `## Known caveats affecting these numbers`):

```markdown
## 10. Alpha-search engine (Piece 1) — built; first real sweep pending

The core alpha-search engine is implemented (`src/trading/alphasearch/`,
`trading alphasearch sweep|leaderboard|holdout`; design:
`docs/superpowers/specs/2026-07-08-alpha-search-engine-design.md`). It turns
signal + universe into a four-factor L/S alpha t-stat via a monthly portfolio
sort, gates candidates with BH-FDR (q=0.10) over the persisted trial journal
(`journal/alphasearch-trials.jsonl`), reports DSR for survivors, and enforces
a touched-once 2024+ holdout. Pre-registered rules are in the design spec §5;
terms in the glossary ("Multiple testing" section).

**No real sweep has been run yet.** When the first discovery sweep over the
large-cap and mid-cap options pools lands, record here: the honest trial
count, the leaderboard summary, BH survivors (if any), and the null-result
reading if nothing survives — a null is a first-class outcome. Known caveat
to carry forward: 2024-25 data was partially examined by the §9 skew studies,
so holdout passes for skew-family signals carry residual contamination risk
and must be read conservatively (spec §5.3).
```

- [ ] **Step 3: Verify docs render and nothing else changed**

Run: `uv run pytest -q`
Expected: ALL PASS (prose-only change)

Run: `git diff --stat`
Expected: only `docs/glossary.md` and `docs/experiments.md` modified.

- [ ] **Step 4: Commit**

```bash
git add docs/glossary.md docs/experiments.md
git commit -m "Document alphasearch terms + experiments ledger stub [AI]

Spec section 8 requires the ledger discipline to exist before the first
real sweep: FDR/BH/DSR/portfolio-sort/trial definitions in the glossary
(grounded in how our engine applies them) and an experiments.md section
ready to receive the first sweep's honest trial count and results --
including the null-result reading, which is a first-class outcome."
```

---

## Execution notes

- Task order is strict: each task's Interfaces block consumes only earlier tasks. Tasks 1 and 2 are independent of each other; everything else is sequential.
- After Task 10 the tool is operator-usable end-to-end; Tasks 11-12 add the adversarial guarantees and ledger discipline REQUIRED by the spec (§7, §8) — the feature is not "done" without them.
- Real-data caveat for the first operator sweep (not part of this plan's tests): `data/fundamentals/equities` may be absent on a given machine; the sweep then refuses fundamentals signals by design — either backfill the store or scope with `--signals` to the price/options families. Factor cache under `data/factors/` must exist (run any factor command once online, or `--refresh-factors`).
- The adversarial reviewer (per spec §8) should specifically attack: PanelView truncation (`side="right"` off-by-one), sort segment boundaries (`(date, hold_end]`), trial-count honesty (dedupe key, error trials), BH input ordering, holdout touched-once enforcement.

## Resolved ambiguities (spec -> plan decisions)

1. **Append-only vs "update in place" (spec §4):** re-runs append; all readers dedupe to the latest event per `(config_hash, kind)`. Logical replace, physical append-only.
2. **"Journaled as error" (spec §6):** error trials keep `kind="discovery"` with an `error` field (no `ls`/`lo` payload) so the BH trial count derives from one kind filter; their p reads as NaN -> 1.0 in `bh_fdr`.
3. **Holdout pass rule for negative alphas (spec §3.6):** implemented as the signed ratio `holdout/discovery >= 0.5`, which is the literal rule for positive discoveries and applies the intended magnitude test (rather than a trivially-passing inequality) for negative ones.
4. **Options-cell staleness:** a cell is visible from its decision_date for `MAX_OPTION_AGE_DAYS = 7` calendar days, then missing — forward-filling months-old cells would violate spec §5.5.
5. **Sign conventions beyond the spec's `hedge` example:** locked in the Task 4/5 tables and recorded as comments at each registration.
6. **Holdout eligibility:** restricted to current BH survivors (spec §3.6 says "evaluate one BH survivor"); refusing non-survivors protects the once-only budget.
7. **Fundamentals-signal mismatch:** refused at assembly like the options case (spec §6 names only options; the same "never silently skip" principle applies).
8. **DSR trial-count/variance inputs:** trial count = deduped journaled discovery trials; cross-trial Sharpe variance = variance of journaled daily Sharpes (`ddof=1`; 0.0 when fewer than 2 are available).
9. **Signal fn signature:** the spec writes `fn(PanelData, as_of)` while requiring "the PanelData accessors it receives are as-of views"; the plan types it `fn(PanelView, as_of)` — `PanelView` IS that as-of view, and passing it (never the raw `PanelData`) is what makes the PIT contract structural.











