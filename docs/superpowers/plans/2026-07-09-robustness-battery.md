# Robustness & Failure-Analysis Battery (Piece 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pre-registered robustness battery (`trading alphasearch robustness <signal>:<universe>`), the cost/capacity analysis, and the battery-passed holdout-eligibility gate, per `docs/superpowers/specs/2026-07-09-robustness-battery-design.md`, then pilot it on the three parked amihud BH survivors.

**Architecture:** New `src/trading/alphasearch/robustness.py` is pure composition of existing machinery: `sweep.evaluate_trial` for the re-evaluation checks (1-4, journaled as tagged discovery trials), `sort.portfolio_sort` outputs for the arithmetic checks (5-6) and the cost/capacity series, `evaluate.evaluate_alpha`/`run_regression` for every re-regression, and `spec.amihud_lambda` for the capacity λ. `sweep.py` gains two hash-preserving perturbation params (`symbol_subset`, `calendar_offset`), a display-only `battery` tag on journal events, the `kind="battery"` verdict event, and a battery-passed pre-check in `run_holdout`. One new CLI action.

**Tech Stack:** Python ≥3.12, pandas 2.3.1, numpy (via pandas), rich 14.0.0, pytest 8.4.1, ruff 0.12.3. Hand-rolled OLS only — **no scipy, no statsmodels** (repo policy).

## Global Constraints

- **The battery table is FROZEN pre-registered science** (spec §3; amend only in writing, prospectively). The checks, constructions, and pass rules below transcribe it exactly. If implementation seems to require changing any threshold, STOP and consult the developer.

| # | check | construction | pass rule (frozen) |
|---|---|---|---|
| 1 | Sub-period halves | discovery split `2019-01-01..2021-06-30` and `2021-07-01..2023-12-31` | both halves: alpha sign matches full-window sign AND \|t\| ≥ 1.0 |
| 2 | Universe subsets | 5 seeded (seed=42+i) random half-universe draws, same params | ≥ 4 of 5 draws: alpha sign matches |
| 3 | Parameter jitter | quantiles ∈ {4, 6}; min_names ∈ {10, 20} (4 trials) | all 4: alpha sign matches |
| 4 | Decision-date offset | rebalance on the 2nd trading session of each month | sign matches AND \|alpha\| ≥ 0.5 × full-window \|alpha\| |
| 5 | Name concentration | recompute the L/S daily series excluding the top-3 names by cumulative contribution to the top-quantile leg | remaining alpha point estimate ≥ 0.5 × original |
| 6 | Month concentration | top-3 calendar months' share of the cumulative L/S log return | ≤ 60% |
| 7 | Factor-proxy flag | any factor loading with \|t_loading\| > 2 × \|t_alpha\| while regression R² > 0.5 | WARNING only (printed prominently in red; does not block) |

- **Promotion rule (frozen):** holdout-eligible iff checks 1-6 all pass AND the 30 bps row of the cost table retains t ≥ 2.0. Cost table: one-way costs c ∈ {10, 30, 50} bps, each rebalance charges turnover × c to EACH leg (2 × turnover × c total), using the same `turnover_monthly` the leaderboard reports. Capacity curve: book sizes B ∈ {$10k, $100k, $1M} per side; each rebalanced name's own Amihud λ (the signal's existing 252d construction) prices impact as λ × (B / names-per-leg) charged on entry and exit.
- **THE HASH-PRESERVATION CONSTRAINT (most delicate in this plan):** `_hashed_params` feeds `trial_config_hash` for all 799 journaled discovery trials. Adding an always-present key to the params dict would change EVERY existing hash, orphaning the whole journal's dedupe identity. New params (`symbol_subset`, `calendar_offset`) are therefore **omitted from the dict when default-valued** (None / 0) and only included when set. Task 1 pins the live journaled hash `4f3d0819382a` (amihud:midcap discovery, verified present in `journal/alphasearch-trials.jsonl`) as a literal regression test. Never "clean up" this asymmetry.
- **Journal honesty (Piece 1 §5.6, unchanged):** battery re-evaluations are BH-counted discovery trials — tagged with a `battery` field on the journal EVENT, deliberately NOT in the hashed config (an identical evaluation inside or outside a battery is ONE trial). The battery refuses non-BH-survivors BEFORE journaling anything. An errored re-evaluation journals the error trial AND fails its check (an uncomputable perturbation is not a pass). The `kind="battery"` verdict event replaces by config hash on re-run. The battery never mutates holdout state.
- Style rules (lessons from prior plans): no truthiness on maybe-empty collections (`is None` / `len(x) == 0`); `pd.Timedelta(N, unit="D")`; no float-exact assertions on compounded synthetic fixtures (`math.isclose`/`pytest.approx`); seeded draws must be dict-order-independent (`rng = np.random.default_rng(42+i)` over a SORTED symbol list, output sorted); hand-computable fixtures everywhere.
- Test suite runs warnings-as-errors (`pyproject.toml`); ruff rules E,F,I,UP,B at 100 columns. Run `uv run pytest -q` and `uv run ruff check src tests scripts` before every commit.
- Commits: granular, one logical change, message suffix ` [AI]`.
- Work from the repo root `/Users/travis/Source/personal/trading`. All paths below are repo-relative. Use `uv run` for python/pytest.

## File Structure

| File | Change |
|---|---|
| `src/trading/alphasearch/sweep.py` | `_hashed_params` gains omit-when-default `symbol_subset`/`calendar_offset`; `evaluate_trial` threads both; `log_trial` gains `battery` tag kwarg (+ reserved key); `battery_verdict`; `run_holdout` battery-passed pre-check |
| `src/trading/alphasearch/sort.py` | `portfolio_sort` gains `symbol_subset`; `SortResult.rebalances` (recorded memberships) |
| `src/trading/alphasearch/panel.py` | `PanelData.decision_dates` gains `offset` (nth trading session of each month) |
| `src/trading/alphasearch/spec.py` | rename `_amihud` → public `amihud_lambda` (capacity λ reuse; registration unchanged) |
| `src/trading/alphasearch/robustness.py` | NEW — frozen constants, `BatteryContext`, checks 1-7, cost table, capacity curve, `run_battery` |
| `src/trading/cli.py` | `robustness` action, `_resolve_alphasearch_universe` helper (shared with holdout), report card + red factor-proxy warning + `--json` |
| `tests/test_alphasearch_journal.py` | pinned-hash test; omit-when-default tests; battery-tag tests |
| `tests/test_alphasearch_sort.py` | `symbol_subset` + `rebalances` tests |
| `tests/test_alphasearch_panel.py` | `decision_dates` offset tests |
| `tests/test_alphasearch_sweep.py` | battery-gate holdout tests + `_log_passing_battery` in existing holdout setups |
| `tests/test_alphasearch_robustness.py` | NEW — per-check unit/integration tests, cost/capacity hand tests, `run_battery` composition tests |
| `tests/test_alphasearch_robustness_golden.py` | NEW — end-to-end golden battery on real fixture files |
| `tests/test_alphasearch_cli.py` | `_seed_journal` battery event; robustness action tests |
| `docs/glossary.md` | robustness-battery / capacity-curve terms |
| `docs/experiments.md` | §11: pilot results recorded live by Task 8 (no battery-pending note beforehand) |

**Notes the implementer must know up front**

- The real journal has **799 discovery trials and three amihud BH survivors** (midcap, opt-midcap:trade, midcap:trade). The pilot (Task 8) adds ~12 battery trials per battery to that count — honest and intended; the BH bar rises slightly with each.
- `opt-midcap:trade` has 26 names, so its check-2 half-universe draws are 13 names < `MIN_NAMES` 15 → every draw journals an honest `SortError` error trial and the check fails. That is the frozen battery working as designed — do NOT touch thresholds; record the outcome.
- `sort.SortError` subclasses `ValueError`; every `except (SortError, ValueError, np.linalg.LinAlgError)` in this plan mirrors `run_sweep`'s error-trial contract.

---

### Task 1: Hash-preserving perturbation params (`symbol_subset`, `calendar_offset`)

**Files:**
- Modify: `src/trading/alphasearch/sweep.py` (`_hashed_params` at line ~52, `evaluate_trial` at line ~376)
- Modify: `src/trading/alphasearch/sort.py` (`portfolio_sort`)
- Modify: `src/trading/alphasearch/panel.py` (`PanelData.decision_dates`)
- Test: `tests/test_alphasearch_journal.py`, `tests/test_alphasearch_sort.py`, `tests/test_alphasearch_panel.py`, `tests/test_alphasearch_sweep.py`

**Interfaces:**
- Consumes: existing `_hashed_params(quantiles, tercile_below, min_names)`, `portfolio_sort`, `decision_dates`.
- Produces (later tasks rely on these exact signatures):
  - `_hashed_params(quantiles: int, tercile_below: int, min_names: int, symbol_subset: tuple[str, ...] | None = None, calendar_offset: int = 0) -> dict` — omits both new keys at defaults; includes `"symbol_subset": sorted(symbol_subset)` / `"calendar_offset": calendar_offset` when set.
  - `evaluate_trial(panel, spec, window, factors, *, quantiles=QUANTILES, tercile_below=TERCILE_BELOW, min_names=MIN_NAMES, symbol_subset: tuple[str, ...] | None = None, calendar_offset: int = 0) -> dict`
  - `portfolio_sort(..., symbol_subset: tuple[str, ...] | None = None)` — restricts each date's scored cross-section to the subset.
  - `PanelData.decision_dates(start, end, offset: int = 0)` — the (offset+1)-th trading session of each month; months with too few in-window sessions are dropped.

- [ ] **Step 1: Write the failing hash-preservation tests**

Append to `tests/test_alphasearch_journal.py`:

```python
def test_default_config_hash_is_pinned_to_the_live_journal():
    # journal/alphasearch-trials.jsonl carries 799 discovery trials hashed
    # through this exact params dict. This is amihud:midcap's LIVE hash (the
    # parked BH survivor, verified present in the committed journal). If
    # _hashed_params ever emits a different default dict — e.g. a new key
    # present at its default value — every existing trial silently orphans
    # from its dedupe identity. Never "fix" this by re-pinning.
    config = trial_config("amihud", "midcap", "2019-01-01..2023-12-31")
    assert trial_config_hash(config) == "4f3d0819382a"
    assert config["params"] == {
        "quantiles": 5, "weighting": "equal", "cadence": "monthly",
        "tercile_below": 50, "min_names": 15,
    }


def test_default_valued_perturbation_params_are_omitted_from_the_hash():
    from trading.alphasearch.sweep import DEFAULT_PARAMS, _hashed_params

    assert _hashed_params(5, 50, 15) == DEFAULT_PARAMS
    assert _hashed_params(5, 50, 15, symbol_subset=None, calendar_offset=0) == DEFAULT_PARAMS
    assert set(_hashed_params(5, 50, 15)) == {
        "quantiles", "weighting", "cadence", "tercile_below", "min_names",
    }


def test_subset_and_offset_change_the_hash_when_set():
    from trading.alphasearch.sweep import _hashed_params

    window = "2020-01-01..2020-06-30"
    base = trial_config_hash(trial_config("mom21", "largecap", window))
    sub = _hashed_params(5, 50, 15, symbol_subset=("B", "A"))
    assert sub["symbol_subset"] == ["A", "B"]        # sorted: draw-order-proof
    sub_hash = trial_config_hash(trial_config("mom21", "largecap", window, params=sub))
    off = _hashed_params(5, 50, 15, calendar_offset=1)
    assert off["calendar_offset"] == 1
    off_hash = trial_config_hash(trial_config("mom21", "largecap", window, params=off))
    assert len({base, sub_hash, off_hash}) == 3      # three distinct trials
```

`trial_config` and `trial_config_hash` are already imported at the top of this test file.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_alphasearch_journal.py -q`
Expected: `test_default_config_hash_is_pinned_to_the_live_journal` PASSES already (it pins current behavior — good); the other two FAIL with `TypeError: _hashed_params() got an unexpected keyword argument`.

- [ ] **Step 3: Extend `_hashed_params` in `src/trading/alphasearch/sweep.py`**

Replace the existing `_hashed_params` (keep the comment block above it, extend it):

```python
# Every parameter that can change a trial's outcome MUST appear here, or a
# re-run with a changed value would dedupe against the stale trial -- breaking
# the "any changed parameter is a NEW trial" rule. run_sweep AND run_holdout
# both build their hashed params through this one constructor so the journaled
# config always records what evaluate_trial truly ran.
#
# The Piece 3 perturbation params (symbol_subset, calendar_offset) are
# OMITTED when default-valued: an always-present new key would change the
# hash of every one of the journal's existing trials (799 at Piece 3 time),
# severing them from their dedupe identities. A default-valued perturbation
# IS the plain trial, so omission is also semantically exact. Pinned by
# test_default_config_hash_is_pinned_to_the_live_journal.
def _hashed_params(
    quantiles: int,
    tercile_below: int,
    min_names: int,
    symbol_subset: tuple[str, ...] | None = None,
    calendar_offset: int = 0,
) -> dict:
    params = {
        "quantiles": quantiles,
        "weighting": "equal",
        "cadence": "monthly",
        "tercile_below": tercile_below,
        "min_names": min_names,
    }
    if symbol_subset is not None:
        params["symbol_subset"] = sorted(symbol_subset)
    if calendar_offset != 0:
        params["calendar_offset"] = calendar_offset
    return params
```

`DEFAULT_PARAMS = _hashed_params(QUANTILES, TERCILE_BELOW, MIN_NAMES)` stays exactly as is.

- [ ] **Step 4: Run the journal tests**

Run: `uv run pytest tests/test_alphasearch_journal.py -q`
Expected: all PASS.

- [ ] **Step 5: Write the failing `decision_dates` offset tests**

Append to `tests/test_alphasearch_panel.py`:

```python
def test_decision_dates_offset_picks_the_nth_session_and_drops_short_months():
    # Jan has 3 union sessions, Feb exactly 1: offset=1 (2nd session) keeps
    # Jan's 2nd date and DROPS Feb (no 2nd session to rebalance on).
    idx = pd.DatetimeIndex(
        ["2020-01-06", "2020-01-07", "2020-01-08", "2020-02-03"], tz="UTC"
    )
    panel = PanelData(closes={"A": pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)},
                      symbols=("A",))
    start, end = idx[0], idx[-1]
    assert panel.decision_dates(start, end) == (idx[0], idx[3])          # unchanged
    assert panel.decision_dates(start, end, offset=0) == (idx[0], idx[3])
    assert panel.decision_dates(start, end, offset=1) == (idx[1],)
    assert panel.decision_dates(start, end, offset=3) == ()
```

(`pd` and `PanelData` are already imported in this file.)

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/test_alphasearch_panel.py -q -k offset`
Expected: FAIL with `TypeError: decision_dates() got an unexpected keyword argument 'offset'`.

- [ ] **Step 7: Extend `decision_dates` in `src/trading/alphasearch/panel.py`**

Replace the method body (docstring included):

```python
    def decision_dates(
        self, start: pd.Timestamp, end: pd.Timestamp, offset: int = 0
    ) -> tuple[pd.Timestamp, ...]:
        """The (offset+1)-th trading session of each month in [start, end]
        (offset=0, the default, is the first session -- Piece 1 behavior,
        bit-identical). offset=1 is Piece 3's decision-date-offset battery
        check. A month with too few in-window sessions is dropped, never
        approximated with a different session.

        "Trading session" = any date on which at least one panel symbol has a
        bar (the union calendar), so one symbol's missing day never shifts the
        whole universe's rebalance date.
        """
        union = sorted({d for s in self.closes.values() for d in s.index})
        by_month: dict[str, list[pd.Timestamp]] = {}
        for date in union:
            if start <= date <= end:
                by_month.setdefault(date.strftime("%Y-%m"), []).append(date)
        return tuple(
            by_month[m][offset] for m in sorted(by_month) if len(by_month[m]) > offset
        )
```

- [ ] **Step 8: Run the panel tests (full file — the rewrite must not disturb offset=0 callers)**

Run: `uv run pytest tests/test_alphasearch_panel.py -q`
Expected: all PASS.

- [ ] **Step 9: Write the failing `portfolio_sort` subset tests**

Append to `tests/test_alphasearch_sort.py` (reuse its `_panel`, `_mom21`, `SIX` helpers):

```python
def test_symbol_subset_restricts_the_cross_section():
    panel = _panel(SIX)
    dates = panel.decision_dates(panel.closes["S1"].index[0],
                                 panel.closes["S1"].index[-1])
    end = panel.closes["S1"].index[-1]
    subset = ("S1", "S3", "S5")
    got = portfolio_sort(panel, _mom21(), dates, end, min_names=3,
                         symbol_subset=subset)
    assert got.n_names_median == 3.0          # only subset members scored
    full = portfolio_sort(panel, _mom21(), dates, end, min_names=3)
    assert full.n_names_median == 6.0
    # Terciles of {S1, S3, S5} by growth rate: top = S5, bottom = S1 -- the
    # L/S daily return is exactly the rate spread on constant-growth closes.
    assert math.isclose(got.ls.iloc[-1], 0.02 - (-0.02), rel_tol=1e-9)


def test_symbol_subset_below_min_names_skips_like_a_thin_date():
    panel = _panel(SIX)
    dates = panel.decision_dates(panel.closes["S1"].index[0],
                                 panel.closes["S1"].index[-1])
    end = panel.closes["S1"].index[-1]
    with pytest.raises(SortError):
        portfolio_sort(panel, _mom21(), dates, end, min_names=3,
                       symbol_subset=("S1", "S2"))
```

- [ ] **Step 10: Run them to verify they fail**

Run: `uv run pytest tests/test_alphasearch_sort.py -q -k symbol_subset`
Expected: FAIL with `TypeError: portfolio_sort() got an unexpected keyword argument 'symbol_subset'`.

- [ ] **Step 11: Extend `portfolio_sort` in `src/trading/alphasearch/sort.py`**

Add the keyword parameter after `min_names`:

```python
def portfolio_sort(
    panel: PanelData,
    spec: SignalSpec,
    dates: Sequence[pd.Timestamp],
    end: pd.Timestamp,
    *,
    quantiles: int = QUANTILES,
    tercile_below: int = TERCILE_BELOW,
    min_names: int = MIN_NAMES,
    symbol_subset: tuple[str, ...] | None = None,
) -> SortResult:
```

and inside the loop, immediately after `scores = spec.fn(panel.view(date), date).dropna()`:

```python
        if symbol_subset is not None:
            # Piece 3 subset check: the scored cross-section is restricted to
            # the draw; every downstream rule (min_names skip, tercile fall-
            # back, distinct-score guard) then applies to the SUBSET, exactly
            # as if the universe had been this half all along.
            scores = scores[scores.index.isin(symbol_subset)]
```

- [ ] **Step 12: Run the sort tests**

Run: `uv run pytest tests/test_alphasearch_sort.py -q`
Expected: all PASS.

- [ ] **Step 13: Write the failing `evaluate_trial` threading tests**

Append to `tests/test_alphasearch_sweep.py`:

```python
def test_evaluate_trial_threads_subset_and_offset():
    from trading.alphasearch.sweep import evaluate_trial

    panel = make_panel()
    factors = make_factors()
    subset = tuple(panel.symbols[:8])
    got = evaluate_trial(panel, SIGNALS["mom21"], WINDOW, factors,
                         min_names=5, symbol_subset=subset)
    assert got["n_names_median"] == 8.0        # subset reached the sort
    offset = evaluate_trial(panel, SIGNALS["mom21"], WINDOW, factors,
                            calendar_offset=1)
    base = evaluate_trial(panel, SIGNALS["mom21"], WINDOW, factors)
    # Same months, different sessions: the offset run rebalances on the 2nd
    # session, so its decision-date count matches and its series differs.
    assert offset["n_dates"] == base["n_dates"]
    assert offset["ls"]["alpha_annual_pct"] != base["ls"]["alpha_annual_pct"]
```

- [ ] **Step 14: Run it to verify it fails**

Run: `uv run pytest tests/test_alphasearch_sweep.py -q -k threads_subset`
Expected: FAIL with `TypeError` (unexpected keyword argument).

- [ ] **Step 15: Extend `evaluate_trial` in `src/trading/alphasearch/sweep.py`**

```python
def evaluate_trial(
    panel: PanelData,
    spec: SignalSpec,
    window: str,
    factors: pd.DataFrame,
    *,
    quantiles: int = QUANTILES,
    tercile_below: int = TERCILE_BELOW,
    min_names: int = MIN_NAMES,
    symbol_subset: tuple[str, ...] | None = None,
    calendar_offset: int = 0,
) -> dict:
    """Score -> sort -> regress. Raises SortError/ValueError/LinAlgError on
    failure; the caller journals that as an error trial. symbol_subset and
    calendar_offset are Piece 3 battery perturbations: callers that set them
    MUST hash them into the trial config via _hashed_params (they change the
    outcome)."""
    start, end = _window_bounds(window)
    _check_factor_coverage(factors, end)
    dates = panel.decision_dates(start, end, offset=calendar_offset)
    sort = portfolio_sort(
        panel, spec, dates, end,
        quantiles=quantiles, tercile_below=tercile_below, min_names=min_names,
        symbol_subset=symbol_subset,
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
```

(Only the signature, docstring, and `dates`/`sort` lines change; the rest is verbatim today's body.)

- [ ] **Step 16: Run the full suite + ruff**

Run: `uv run pytest -q && uv run ruff check src tests scripts`
Expected: all PASS, ruff clean.

- [ ] **Step 17: Commit**

```bash
git add src/trading/alphasearch/sweep.py src/trading/alphasearch/sort.py \
        src/trading/alphasearch/panel.py tests/test_alphasearch_journal.py \
        tests/test_alphasearch_sort.py tests/test_alphasearch_panel.py \
        tests/test_alphasearch_sweep.py
git commit -m "Add hash-preserving symbol_subset/calendar_offset trial params [AI]"
```

---

### Task 2: `robustness.py` — battery tag, survivor refusal, re-evaluation checks 1-4

**Files:**
- Create: `src/trading/alphasearch/robustness.py`
- Modify: `src/trading/alphasearch/sweep.py` (`log_trial`, `RESERVED_RESULT_KEYS`)
- Create: `tests/test_alphasearch_robustness.py`
- Test: also `tests/test_alphasearch_journal.py`

**Interfaces:**
- Consumes (from Task 1): `_hashed_params(..., symbol_subset=, calendar_offset=)`, `evaluate_trial(..., symbol_subset=, calendar_offset=)`.
- Produces (Task 5 composes these exact names):
  - `log_trial(journal, *, kind, config, ts, result=None, error=None, battery: str | None = None) -> dict` — stamps `event["battery"] = battery` when set (display/grouping only, never hashed).
  - In `robustness.py`: frozen constants (below); `BatteryContext` (frozen dataclass); `CheckResult(number: int, name: str, passed: bool, detail: dict)`; `subperiod_windows(window: str) -> tuple[str, str]`; `subset_draw(symbols, i) -> tuple[str, ...]`; `signed_retention(original, perturbed) -> float`; `check_subperiods(ctx) -> CheckResult`; `check_subsets(ctx) -> CheckResult`; `check_jitter(ctx) -> CheckResult`; `check_offset(ctx) -> CheckResult`; `require_survivor(journal, signal_name, universe, window, params) -> dict`.

- [ ] **Step 1: Write the failing battery-tag journal tests**

Append to `tests/test_alphasearch_journal.py`:

```python
def test_battery_tag_rides_on_the_event_never_the_hash(tmp_path):
    from trading.alphasearch.sweep import discovery_trials, trials_journal

    journal = trials_journal(tmp_path / "journal")
    config = trial_config("mom21", "largecap", "2020-01-01..2020-06-30")
    tagged = log_trial(journal, kind="discovery", config=config, ts="t1",
                       result=None, error="SortError: x",
                       battery="amihud:midcap")
    assert tagged["battery"] == "amihud:midcap"
    assert tagged["config_hash"] == trial_config_hash(config)  # tag not hashed
    # The identical config outside a battery dedupes INTO the same trial
    # (spec section 5: one trial, never two) -- and the latest event wins.
    log_trial(journal, kind="discovery", config=config, ts="t2")
    trials = discovery_trials(journal)
    assert len(trials) == 1
    assert "battery" not in trials[0]          # latest (untagged) event won


def test_result_payload_cannot_clobber_the_battery_tag(tmp_path):
    from trading.alphasearch.sweep import SweepError, trials_journal

    journal = trials_journal(tmp_path / "journal")
    with pytest.raises(SweepError):
        log_trial(journal, kind="discovery",
                  config=trial_config("mom21", "largecap", "2020-01-01..2020-06-30"),
                  ts="t1", result={"battery": "sneaky"})
```

(`log_trial`, `trial_config`, `trial_config_hash`, `pytest` are already imported there.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_alphasearch_journal.py -q -k battery`
Expected: FAIL (`TypeError: log_trial() got an unexpected keyword argument 'battery'`; the reserved-key test fails because `"battery"` is not reserved yet).

- [ ] **Step 3: Extend `log_trial` in `src/trading/alphasearch/sweep.py`**

Add `"battery"` to the reserved set:

```python
RESERVED_RESULT_KEYS = frozenset(
    {"event", "kind", "config_hash", "ts", "error", "signal", "universe",
     "window", "params", "battery"}
)
```

Extend `log_trial` (kind comment gains `"battery"`; new kwarg; two lines before `_json_safe`):

```python
def log_trial(
    journal: Journal,
    *,
    kind: str,  # "discovery" | "holdout" | "battery"
    config: dict,
    ts: str,  # ISO-8601 UTC, supplied by the CLI (the only clock reader)
    result: dict | None = None,
    error: str | None = None,
    battery: str | None = None,  # Piece 3: display/grouping tag, NEVER hashed
) -> dict:
    """Append one trial event (spec section 4 schema) and return it."""
    result = result or {}
    clobbered = RESERVED_RESULT_KEYS & result.keys()
    if clobbered:
        raise SweepError(
            f"result payload cannot set reserved journal keys: {sorted(clobbered)}"
        )
    event = {
        "event": "trial",
        "kind": kind,
        **config,
        "config_hash": trial_config_hash(config),
        "ts": ts,
        "error": error,
        **result,
    }
    if battery is not None:
        # On the EVENT only (spec Piece 3 section 5): the hash comes from
        # `config` above, so an identical evaluation made inside or outside
        # a battery stays ONE trial -- the tag is for display and grouping.
        event["battery"] = battery
    event = _json_safe(event)
    journal.append(event)
    return event
```

- [ ] **Step 4: Run the journal tests**

Run: `uv run pytest tests/test_alphasearch_journal.py -q`
Expected: all PASS.

- [ ] **Step 5: Create `src/trading/alphasearch/robustness.py` with constants, context, helpers, and the refusal**

```python
"""Piece 3 robustness battery (design spec 2026-07-09): pre-registered,
frozen interrogation of a BH survivor BEFORE it may spend a holdout touch.

Pure composition of existing machinery: evaluate_trial for the re-evaluation
checks (1-4, journaled as `battery`-tagged, BH-counted discovery trials),
portfolio_sort outputs for the arithmetic checks (5-6) and the cost/capacity
series, evaluate_alpha for every re-regression, and spec.amihud_lambda for
the capacity curve's impact prices. Thresholds here are FROZEN (spec section
3); amend only in writing, prospectively.

This module never reads the clock: `ts` always arrives from the CLI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading.alphasearch.panel import PanelData
from trading.alphasearch.sort import SortError
from trading.alphasearch.spec import SIGNALS, SignalSpec
from trading.alphasearch.sweep import (
    BH_Q,
    SweepError,
    _bh_survivor_hashes,
    _hashed_params,
    _window_bounds,
    discovery_trials,
    evaluate_trial,
    find_discovery_trial,
    log_trial,
    trial_config,
)
from trading.journal import Journal

# ---------------------------------------------------------------------------#
# Frozen battery parameters (spec section 3 / section 4). Amend only in
# writing, prospectively -- these are pre-registered science, not tunables.
# ---------------------------------------------------------------------------#
SUBPERIOD_MIN_ABS_T = 1.0
SUBSET_DRAWS = 5
SUBSET_SEED_BASE = 42
SUBSET_PASS_MIN = 4
JITTER_GRID = ((4, 10), (4, 20), (6, 10), (6, 20))  # (quantiles, min_names)
OFFSET_SESSIONS = 1                 # rebalance on the 2nd trading session
OFFSET_MIN_RETENTION = 0.5
NAME_EXCLUDE_TOP = 3
NAME_MIN_RETENTION = 0.5
MONTH_TOP = 3
MONTH_MAX_SHARE = 0.60
PROXY_LOADING_MULTIPLE = 2.0
PROXY_MIN_R2 = 0.5
COST_BPS = (10, 30, 50)             # one-way, per leg, per rebalance
ELIGIBLE_COST_BPS = 30
ELIGIBLE_MIN_COST_T = 2.0
BOOK_SIZES = (10_000.0, 100_000.0, 1_000_000.0)  # $ per side


@dataclass(frozen=True)
class CheckResult:
    number: int          # spec section 3 row number (1-6)
    name: str
    passed: bool
    detail: dict         # per-check numbers, JSON-safe via log_trial


@dataclass(frozen=True)
class BatteryContext:
    """Everything the check runners share. full_alpha is the DISCOVERY
    trial's journaled L/S four-factor alpha (%/yr) -- the baseline every
    sign/retention rule compares against."""

    journal: Journal
    panel: PanelData
    spec: SignalSpec
    factors: pd.DataFrame
    ts: str
    universe: str
    window: str          # the discovery window being interrogated
    full_alpha: float
    quantiles: int
    tercile_below: int
    min_names: int
    tag: str             # "signal:universe" -- the battery grouping tag


def subperiod_windows(window: str) -> tuple[str, str]:
    """Split a window at its calendar midpoint (floor). For the production
    discovery window this reproduces the spec section 3 literals exactly --
    pinned by test_subperiod_windows_pin_the_frozen_discovery_split."""
    start, end = _window_bounds(window)
    mid = start + pd.Timedelta((end - start).days // 2, unit="D")
    first_end = mid - pd.Timedelta(1, unit="D")
    return (
        f"{start.date().isoformat()}..{first_end.date().isoformat()}",
        f"{mid.date().isoformat()}..{end.date().isoformat()}",
    )


def subset_draw(symbols: tuple[str, ...], i: int) -> tuple[str, ...]:
    """Half-universe draw i, seed = SUBSET_SEED_BASE + i (frozen). Sorted
    input AND sorted output: the draw is a set, reproducible regardless of
    dict/tuple ordering upstream."""
    universe = sorted(symbols)
    rng = np.random.default_rng(SUBSET_SEED_BASE + i)
    picked = rng.choice(np.array(universe, dtype=object),
                        size=len(universe) // 2, replace=False)
    return tuple(sorted(str(s) for s in picked))


def signed_retention(original: float | None, perturbed: float | None) -> float:
    """perturbed / original: > 0 iff the signs match, >= r iff the magnitude
    retains an r fraction WITH the matching sign (holdout_passes' collapse,
    symmetric for negative-alpha candidates). NaN when either side is
    missing/NaN or original == 0 -- callers treat NaN as a FAIL (an
    uncomputable perturbation is not a pass, spec section 6)."""
    if original is None or perturbed is None:
        return math.nan
    original, perturbed = float(original), float(perturbed)
    if original == 0 or math.isnan(original) or math.isnan(perturbed):
        return math.nan
    return perturbed / original


def _reevaluate(
    ctx: BatteryContext,
    *,
    window: str | None = None,
    quantiles: int | None = None,
    min_names: int | None = None,
    symbol_subset: tuple[str, ...] | None = None,
    calendar_offset: int = 0,
) -> dict:
    """One battery re-evaluation: journaled as a tagged, BH-counted discovery
    trial (config-hash dedupe applies as everywhere) BEFORE its check is
    judged. Errors journal an error trial exactly like run_sweep -- and the
    caller fails the check."""
    q = ctx.quantiles if quantiles is None else quantiles
    mn = ctx.min_names if min_names is None else min_names
    w = ctx.window if window is None else window
    params = _hashed_params(q, ctx.tercile_below, mn,
                            symbol_subset=symbol_subset,
                            calendar_offset=calendar_offset)
    config = trial_config(ctx.spec.name, ctx.universe, w, params=params)
    try:
        result: dict | None = evaluate_trial(
            ctx.panel, ctx.spec, w, ctx.factors,
            quantiles=q, tercile_below=ctx.tercile_below, min_names=mn,
            symbol_subset=symbol_subset, calendar_offset=calendar_offset,
        )
        result["corrupt_cells"] = ctx.panel.corrupt_cells
        error = None
    except (SortError, ValueError, np.linalg.LinAlgError) as exc:
        result = None
        error = f"{type(exc).__name__}: {exc}"
    return log_trial(ctx.journal, kind="discovery", config=config, ts=ctx.ts,
                     result=result, error=error, battery=ctx.tag)


def _alpha_and_t(event: dict) -> tuple[float | None, float | None]:
    ls = event.get("ls") or {}
    return ls.get("alpha_annual_pct"), ls.get("alpha_t")


def check_subperiods(ctx: BatteryContext) -> CheckResult:
    """Check 1 (frozen): both halves -- alpha sign matches the full-window
    sign AND |t| >= 1.0."""
    halves = []
    passed = True
    for w in subperiod_windows(ctx.window):
        event = _reevaluate(ctx, window=w)
        alpha, t = _alpha_and_t(event)
        r = signed_retention(ctx.full_alpha, alpha)
        ok = (
            not math.isnan(r) and r > 0
            and t is not None and abs(float(t)) >= SUBPERIOD_MIN_ABS_T
        )
        passed = passed and ok
        halves.append({"window": w, "alpha_annual_pct": alpha, "alpha_t": t,
                       "error": event.get("error"), "passed": ok})
    return CheckResult(1, "sub_period_halves", passed, {"halves": halves})


def check_subsets(ctx: BatteryContext) -> CheckResult:
    """Check 2 (frozen): 5 seeded half-universe draws; >= 4 of 5 sign-match.
    A draw whose evaluation errors (e.g. half-universe below min_names, or
    missing panel data) FAILS -- the others proceed (spec section 6)."""
    draws = []
    n_pass = 0
    for i in range(SUBSET_DRAWS):
        subset = subset_draw(ctx.panel.symbols, i)
        event = _reevaluate(ctx, symbol_subset=subset)
        alpha, _t = _alpha_and_t(event)
        r = signed_retention(ctx.full_alpha, alpha)
        ok = not math.isnan(r) and r > 0
        n_pass += 1 if ok else 0
        draws.append({"seed": SUBSET_SEED_BASE + i, "n_symbols": len(subset),
                      "alpha_annual_pct": alpha, "error": event.get("error"),
                      "passed": ok})
    return CheckResult(2, "universe_subsets", n_pass >= SUBSET_PASS_MIN,
                       {"draws": draws, "n_pass": n_pass})


def check_jitter(ctx: BatteryContext) -> CheckResult:
    """Check 3 (frozen): quantiles x min_names jitter grid, all 4 sign-match."""
    trials = []
    passed = True
    for q, mn in JITTER_GRID:
        event = _reevaluate(ctx, quantiles=q, min_names=mn)
        alpha, _t = _alpha_and_t(event)
        r = signed_retention(ctx.full_alpha, alpha)
        ok = not math.isnan(r) and r > 0
        passed = passed and ok
        trials.append({"quantiles": q, "min_names": mn,
                       "alpha_annual_pct": alpha,
                       "error": event.get("error"), "passed": ok})
    return CheckResult(3, "parameter_jitter", passed, {"trials": trials})


def check_offset(ctx: BatteryContext) -> CheckResult:
    """Check 4 (frozen): rebalance on the 2nd trading session; sign matches
    AND |alpha| >= 0.5 x full-window |alpha| (the signed-ratio collapse)."""
    event = _reevaluate(ctx, calendar_offset=OFFSET_SESSIONS)
    alpha, t = _alpha_and_t(event)
    r = signed_retention(ctx.full_alpha, alpha)
    ok = not math.isnan(r) and r >= OFFSET_MIN_RETENTION
    return CheckResult(4, "decision_offset", ok, {
        "offset_sessions": OFFSET_SESSIONS, "alpha_annual_pct": alpha,
        "alpha_t": t, "retention": None if math.isnan(r) else r,
        "error": event.get("error"),
    })


def require_survivor(
    journal: Journal, signal_name: str, universe: str, window: str, params: dict
) -> dict:
    """The battery's admission gate (mirrors the holdout gate; refusal
    journals NOTHING). Returns the clean discovery trial being interrogated.
    Refusals: unknown signal; no clean same-params discovery trial with a
    usable L/S alpha; that EXACT trial (by config hash) not a current BH
    survivor -- the refusal lists the current survivors."""
    if signal_name not in SIGNALS:
        known = ", ".join(sorted(SIGNALS))
        raise SweepError(f"unknown signal {signal_name!r}; known: {known}")
    discovery = find_discovery_trial(journal, signal_name, universe, window,
                                     params=params)
    if discovery is None:
        raise SweepError(
            f"no discovery trial for {signal_name}:{universe} over {window} "
            f"with matching params; run the sweep first"
        )
    if discovery.get("error"):
        raise SweepError(
            f"discovery trial for {signal_name}:{universe} errored "
            f"({discovery['error']}); nothing to interrogate"
        )
    if (discovery.get("ls") or {}).get("alpha_annual_pct") is None:
        raise SweepError(
            f"discovery trial for {signal_name}:{universe} has no usable "
            f"L/S alpha (journaled as null); nothing to interrogate"
        )
    survivors = _bh_survivor_hashes(journal)
    if discovery["config_hash"] not in survivors:
        current = sorted({
            f"{t['signal']}:{t['universe']}"
            for t in discovery_trials(journal)
            if t["config_hash"] in survivors
        })
        listing = ", ".join(current) if len(current) > 0 else "none"
        raise SweepError(
            f"{signal_name}:{universe} is not a current BH survivor "
            f"(q={BH_Q}); the battery is reserved for survivors. "
            f"Current survivors: {listing}"
        )
    return discovery
```

- [ ] **Step 6: Write the check tests**

Create `tests/test_alphasearch_robustness.py`:

```python
"""Piece 3 robustness battery: frozen thresholds, per-check behavior,
cost/capacity arithmetic, verdict + gate honesty."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alphasearch_helpers import make_factors, make_panel
from trading.alphasearch.panel import PanelData
from trading.alphasearch.robustness import (
    BatteryContext,
    check_jitter,
    check_offset,
    check_subperiods,
    check_subsets,
    require_survivor,
    signed_retention,
    subperiod_windows,
    subset_draw,
)
from trading.alphasearch.spec import SIGNALS, SignalSpec
from trading.alphasearch.sweep import (
    DISCOVERY_WINDOW,
    SweepError,
    evaluate_trial,
    log_trial,
    trial_config,
    trials_journal,
)

WINDOW = "2020-01-01..2020-06-30"


def _ctx(journal, panel, factors, *, spec=None, full_alpha=None,
         window=WINDOW, tercile_below=50, min_names=15, quantiles=5):
    spec = spec if spec is not None else SIGNALS["mom21"]
    if full_alpha is None:
        full = evaluate_trial(panel, spec, window, factors,
                              quantiles=quantiles, tercile_below=tercile_below,
                              min_names=min_names)
        full_alpha = full["ls"]["alpha_annual_pct"]
    return BatteryContext(
        journal=journal, panel=panel, spec=spec, factors=factors, ts="t1",
        universe="largecap", window=window, full_alpha=float(full_alpha),
        quantiles=quantiles, tercile_below=tercile_below, min_names=min_names,
        tag=f"{spec.name}:largecap",
    )


def test_subperiod_windows_pin_the_frozen_discovery_split():
    # Spec section 3 check 1, verbatim: the production discovery window
    # splits into EXACTLY these two halves. FROZEN.
    assert subperiod_windows(DISCOVERY_WINDOW) == (
        "2019-01-01..2021-06-30", "2021-07-01..2023-12-31",
    )


def test_subperiod_windows_fixture_split():
    assert subperiod_windows(WINDOW) == (
        "2020-01-01..2020-03-30", "2020-03-31..2020-06-30",
    )


def test_signed_retention_rules():
    assert signed_retention(10.0, 6.0) == pytest.approx(0.6)
    assert signed_retention(10.0, -6.0) == pytest.approx(-0.6)   # sign flip
    assert signed_retention(-10.0, -6.0) == pytest.approx(0.6)   # symmetric
    assert math.isnan(signed_retention(0.0, 1.0))
    assert math.isnan(signed_retention(None, 1.0))
    assert math.isnan(signed_retention(10.0, None))
    assert math.isnan(signed_retention(float("nan"), 1.0))


def test_subset_draw_is_deterministic_sorted_and_half_sized():
    symbols = tuple(f"S{i:02d}" for i in range(40))
    scrambled = tuple(reversed(symbols))
    first = subset_draw(symbols, 0)
    assert first == subset_draw(scrambled, 0)      # input-order-proof
    assert first == subset_draw(symbols, 0)        # deterministic
    assert list(first) == sorted(first)
    assert len(first) == 20
    assert set(first) <= set(symbols)
    assert first != subset_draw(symbols, 1)        # seeds 42 vs 43 differ


def test_check_subperiods_passes_on_a_stable_fixture(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(n_symbols=40)
    ctx = _ctx(journal, panel, make_factors())
    got = check_subperiods(ctx)
    assert got.passed
    assert got.number == 1 and got.name == "sub_period_halves"
    assert len(got.detail["halves"]) == 2
    events = list(journal.events())
    assert len(events) == 2                        # both halves journaled
    assert all(e["battery"] == "mom21:largecap" for e in events)
    assert all(e["kind"] == "discovery" for e in events)


def _two_regime_panel(n=40, periods=130):
    """Planted two-regime cross-section (spec section 7): the drift spread
    REVERSES at the midpoint, so a fixed-rank signal's L/S alpha flips sign
    in the second half -- a deterministic check-1 failure via the sign rule
    (a zero-drift second half would leave |t| to noise, ~32% flaky)."""
    idx = pd.date_range("2020-01-02", periods=periods, freq="B", tz="UTC")
    names = [f"S{i:02d}" for i in range(n)]
    half = periods // 2
    rng = np.random.default_rng(11)
    closes = {}
    for i, sym in enumerate(names):
        drift = (i - n / 2) * 4e-4
        rets = np.concatenate(
            [np.full(half, drift), np.full(periods - half, -drift)]
        ) + rng.normal(0.0, 1e-4, size=periods)
        closes[sym] = pd.Series(100.0 * np.cumprod(1 + rets), index=idx)
    return PanelData(closes=closes, symbols=tuple(names))


def _planted_rank_spec() -> SignalSpec:
    """History-free fixed ranking: symbol S<i> scores i. Unlike momentum it
    cannot re-learn a reversed regime, so the two-regime failure is exact."""

    def fn(view, as_of):
        return pd.Series(
            {s: float(i) for i, s in enumerate(sorted(view.symbols))},
            dtype="float64",
        )

    return SignalSpec("planted_rank", fn)


def test_check_subperiods_fails_a_two_regime_signal(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = _two_regime_panel()
    # The discovery baseline is the first-half sign (positive): the battery
    # compares halves against the JOURNALED full-window alpha, which the
    # test supplies directly.
    ctx = _ctx(journal, panel, make_factors(), spec=_planted_rank_spec(),
               full_alpha=10.0)
    got = check_subperiods(ctx)
    assert not got.passed
    first, second = got.detail["halves"]
    assert first["passed"] is True                 # regime 1: strong + right sign
    assert second["passed"] is False               # regime 2: sign flipped
    assert second["alpha_annual_pct"] < 0


def test_check_subsets_passes_on_a_wide_fixture(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(n_symbols=40)
    ctx = _ctx(journal, panel, make_factors())
    got = check_subsets(ctx)
    assert got.passed and got.detail["n_pass"] == 5
    assert len(list(journal.events())) == 5        # one tagged trial per draw
    # Draw subsets entered the hashed configs (sorted lists).
    for event in journal.events():
        assert event["params"]["symbol_subset"] == sorted(
            event["params"]["symbol_subset"]
        )
        assert len(event["params"]["symbol_subset"]) == 20


def test_check_subsets_error_draws_journal_and_fail(tmp_path):
    # 16-symbol panel: half-draws of 8 < min_names 15 -> every draw journals
    # an honest SortError trial AND fails (spec section 6: an uncomputable
    # perturbation is not a pass). The check itself fails 0/5.
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()                            # 16 symbols
    ctx = _ctx(journal, panel, make_factors(), full_alpha=10.0)
    got = check_subsets(ctx)
    assert not got.passed and got.detail["n_pass"] == 0
    events = list(journal.events())
    assert len(events) == 5
    assert all(e["error"] is not None and "SortError" in e["error"]
               for e in events)
    assert all(e["battery"] == "mom21:largecap" for e in events)


def test_check_jitter_runs_the_frozen_grid(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(n_symbols=40)
    # tercile_below=10 so the jittered quantile counts actually bind on a
    # 40-name fixture (40 >= 10 -> quantiles apply, not the tercile fallback).
    ctx = _ctx(journal, panel, make_factors(), tercile_below=10)
    got = check_jitter(ctx)
    assert got.passed
    grid = {(t["quantiles"], t["min_names"]) for t in got.detail["trials"]}
    assert grid == {(4, 10), (4, 20), (6, 10), (6, 20)}   # FROZEN
    assert len(list(journal.events())) == 4
    hashes = {e["config_hash"] for e in journal.events()}
    assert len(hashes) == 4                        # each jitter is a NEW trial


def test_check_offset_passes_on_a_stable_fixture(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(n_symbols=40)
    ctx = _ctx(journal, panel, make_factors())
    got = check_offset(ctx)
    assert got.passed
    assert got.detail["retention"] >= 0.5
    (event,) = list(journal.events())
    assert event["params"]["calendar_offset"] == 1  # hashed perturbation


def _result_like(*, alpha_annual_pct: float, alpha_t: float, p: float) -> dict:
    leg = {
        "alpha_annual_pct": alpha_annual_pct, "alpha_t": alpha_t, "p": p,
        "capm_alpha_annual_pct": alpha_annual_pct, "capm_alpha_t": alpha_t,
        "loadings": {}, "loadings_t": {}, "r2": 0.0, "n_obs": 120,
        "sharpe": 0.1, "sharpe_daily": 0.006, "skew": 0.0, "kurt": 3.0,
    }
    return {"n_dates": 3, "n_names_median": 16.0, "ls": leg, "lo": dict(leg),
            "turnover_monthly": 0.3, "skipped_dates": []}


def test_require_survivor_refuses_nonsurvivor_listing_survivors(tmp_path):
    from trading.alphasearch.sweep import DEFAULT_PARAMS

    journal = trials_journal(tmp_path / "journal")
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", WINDOW), ts="t1",
              result=_result_like(alpha_annual_pct=12.0, alpha_t=8.0, p=1e-8))
    log_trial(journal, kind="discovery",
              config=trial_config("rvol21", "largecap", WINDOW), ts="t1",
              result=_result_like(alpha_annual_pct=0.3, alpha_t=0.1, p=0.92))
    events_before = len(list(journal.events()))
    with pytest.raises(SweepError) as excinfo:
        require_survivor(journal, "rvol21", "largecap", WINDOW,
                         dict(DEFAULT_PARAMS))
    assert "not a current BH survivor" in str(excinfo.value)
    assert "mom21:largecap" in str(excinfo.value)   # lists current survivors
    assert len(list(journal.events())) == events_before  # journaled NOTHING
    # The survivor itself is admitted and returned.
    got = require_survivor(journal, "mom21", "largecap", WINDOW,
                           dict(DEFAULT_PARAMS))
    assert got["signal"] == "mom21"


def test_require_survivor_refuses_unknown_missing_and_errored(tmp_path):
    from trading.alphasearch.sweep import DEFAULT_PARAMS

    journal = trials_journal(tmp_path / "journal")
    with pytest.raises(SweepError, match="unknown signal"):
        require_survivor(journal, "no_such", "largecap", WINDOW,
                         dict(DEFAULT_PARAMS))
    with pytest.raises(SweepError, match="no discovery trial"):
        require_survivor(journal, "mom21", "largecap", WINDOW,
                         dict(DEFAULT_PARAMS))
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", WINDOW), ts="t1",
              error="SortError: boom")
    with pytest.raises(SweepError, match="errored"):
        require_survivor(journal, "mom21", "largecap", WINDOW,
                         dict(DEFAULT_PARAMS))
    assert len(list(journal.events())) == 1          # refusals journal nothing
```

- [ ] **Step 7: Run the new tests**

Run: `uv run pytest tests/test_alphasearch_robustness.py -q`
Expected: all PASS. (If `test_check_subperiods_passes_on_a_stable_fixture` fails because a half-window has too few post-warmup dates, the fixture halves are `2020-01-01..2020-03-30` / `2020-03-31..2020-06-30`; mom21 needs 22 bars of the `2020-01-02`-starting fixture, so the first half rebalances in February and March — verify with `-k subperiods -x --tb=long` and consult the developer before touching any threshold.)

- [ ] **Step 8: Run the full suite + ruff**

Run: `uv run pytest -q && uv run ruff check src tests scripts`
Expected: all PASS, ruff clean.

- [ ] **Step 9: Commit**

```bash
git add src/trading/alphasearch/robustness.py src/trading/alphasearch/sweep.py \
        tests/test_alphasearch_robustness.py tests/test_alphasearch_journal.py
git commit -m "Add battery journal tag + robustness re-evaluation checks 1-4 [AI]"
```

---

### Task 3: Arithmetic checks 5-7 — rebalance memberships, name/month concentration, factor-proxy flag

**Files:**
- Modify: `src/trading/alphasearch/sort.py` (`SortResult`, `portfolio_sort`)
- Modify: `src/trading/alphasearch/robustness.py`
- Test: `tests/test_alphasearch_sort.py`, `tests/test_alphasearch_robustness.py`

**Interfaces:**
- Consumes: `SortResult` (sort.py), `evaluate_alpha` (evaluate.py), Task 2's `CheckResult`, `BatteryContext`, `signed_retention`.
- Produces (Task 4 and 5 rely on these exact names):
  - `sort.Membership = tuple[pd.Timestamp, tuple[str, ...], tuple[str, ...]]` (type alias) and `SortResult.rebalances: tuple[Membership, ...] = ()` — (decision date, top, bottom) recorded at each ACTUAL rebalance (skipped dates record nothing).
  - In `robustness.py`: `ls_series(closes: dict[str, pd.Series], rebalances, end: pd.Timestamp, *, excluded: frozenset[str] = frozenset()) -> pd.Series`; `top_leg_contributions(closes, rebalances, end) -> pd.Series` (descending); `month_share(ls: pd.Series, top_n: int = MONTH_TOP) -> float`; `check_name_concentration(ctx, sort: SortResult) -> CheckResult`; `check_month_concentration(ls: pd.Series) -> CheckResult`; `factor_proxy_flag(ls_stats: dict) -> dict` returning `{"flagged": bool, "offenders": dict[str, float], "alpha_t": ..., "r2": ...}`.

- [ ] **Step 1: Write the failing `SortResult.rebalances` test**

Append to `tests/test_alphasearch_sort.py`:

```python
def test_rebalances_record_actual_memberships_only():
    panel = _panel(SIX)
    idx = panel.closes["S1"].index
    dates = panel.decision_dates(idx[0], idx[-1])
    got = portfolio_sort(panel, _mom21(), dates, idx[-1], min_names=3)
    # mom21 needs 22 bars: the first month is skipped (no membership
    # recorded), later months rebalance with the SIX terciles.
    assert len(got.rebalances) < len(dates)
    date, top, bottom = got.rebalances[0]
    assert date in dates
    assert top == ("S5", "S6") and bottom == ("S1", "S2")   # tercile extremes
    # Every attempted date either rebalanced (recorded) or was skipped.
    assert len(got.rebalances) + len(got.skipped_dates) == got.n_dates
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_alphasearch_sort.py -q -k rebalances_record`
Expected: FAIL with `AttributeError: 'SortResult' object has no attribute 'rebalances'`.

- [ ] **Step 3: Record memberships in `src/trading/alphasearch/sort.py`**

Add the alias above `SortResult` and the field (defaulted, so existing constructors stay valid):

```python
# (decision date, top, bottom) at one ACTUAL rebalance. Piece 3's arithmetic
# checks and cost/capacity analysis replay holdings from this record.
Membership = tuple[pd.Timestamp, tuple[str, ...], tuple[str, ...]]
```

```python
@dataclass(frozen=True)
class SortResult:
    ls: pd.Series                    # daily long/short spread (top - bottom)
    lo: pd.Series                    # daily long-only (top quantile)
    turnover_monthly: float          # mean one-way turnover of the top quantile
    skipped_dates: tuple[str, ...]   # ISO dates skipped for thin cross-sections
    n_dates: int                     # decision dates attempted (incl. skipped)
    n_names_median: float            # median cross-section size on traded dates
    rebalances: tuple[Membership, ...] = ()  # memberships at ACTUAL rebalances
```

In `portfolio_sort`, add `rebalances: list[Membership] = []` next to the other accumulators, then record right where `tops.append(...)` happens:

```python
                current_top, current_bottom = assign_quantiles(scores, q)
                tops.append(set(current_top))
                names_per_date.append(len(scores))
                rebalances.append(
                    (date, tuple(current_top), tuple(current_bottom))
                )
```

and thread it into the return:

```python
    return SortResult(
        ls=ls,
        lo=lo,
        turnover_monthly=turnover,
        skipped_dates=tuple(skipped),
        n_dates=len(dates),
        n_names_median=float(np.median(names_per_date)),
        rebalances=tuple(rebalances),
    )
```

- [ ] **Step 4: Run the sort tests**

Run: `uv run pytest tests/test_alphasearch_sort.py -q`
Expected: all PASS.

- [ ] **Step 5: Write the failing replay/concentration tests**

Append to `tests/test_alphasearch_robustness.py`:

```python
def _hand_closes(values: dict[str, list[float]], start="2020-01-06") -> dict:
    idx = pd.date_range(start, periods=len(next(iter(values.values()))),
                        freq="B", tz="UTC")
    return {sym: pd.Series(v, index=idx) for sym, v in values.items()}


def test_ls_series_replays_portfolio_sort_exactly(tmp_path):
    from trading.alphasearch.robustness import ls_series
    from trading.alphasearch.sort import portfolio_sort

    panel = make_panel(n_symbols=40)
    idx = panel.closes[panel.symbols[0]].index
    dates = panel.decision_dates(idx[0], idx[-1])
    sort = portfolio_sort(panel, SIGNALS["mom21"], dates, idx[-1])
    replayed = ls_series(panel.closes, sort.rebalances, idx[-1])
    pd.testing.assert_series_equal(replayed, sort.ls)


def test_ls_series_excludes_names_from_both_legs():
    from trading.alphasearch.robustness import ls_series

    # 4 names, constant daily growth: A 4%, B 3%, C 1%, D 0%. One rebalance:
    # top=(A,B), bottom=(C,D). Excluding A leaves top=B alone.
    closes = _hand_closes({
        "A": [100.0, 104.0], "B": [100.0, 103.0],
        "C": [100.0, 101.0], "D": [100.0, 100.0],
    })
    date, end = closes["A"].index[0], closes["A"].index[-1]
    rebalances = ((date, ("A", "B"), ("C", "D")),)
    full = ls_series(closes, rebalances, end)
    assert full.iloc[0] == pytest.approx((0.04 + 0.03) / 2 - (0.01 + 0.0) / 2)
    reduced = ls_series(closes, rebalances, end, excluded=frozenset({"A"}))
    assert reduced.iloc[0] == pytest.approx(0.03 - 0.005)
    # Excluding a bottom name symmetrically:
    reduced2 = ls_series(closes, rebalances, end, excluded=frozenset({"D"}))
    assert reduced2.iloc[0] == pytest.approx(0.035 - 0.01)


def test_ls_series_emptied_leg_contributes_nothing():
    from trading.alphasearch.robustness import ls_series
    from trading.alphasearch.sort import SortError

    closes = _hand_closes({"A": [100.0, 104.0], "B": [100.0, 101.0]})
    date, end = closes["A"].index[0], closes["A"].index[-1]
    rebalances = ((date, ("A",), ("B",)),)
    with pytest.raises(SortError):
        ls_series(closes, rebalances, end, excluded=frozenset({"A"}))


def test_top_leg_contributions_hand_computed():
    from trading.alphasearch.robustness import top_leg_contributions

    # top=(A,B) held two days; equal weight 1/2. A returns 4% then ~1.923%,
    # B returns 1% then ~0.990%: contributions are the summed ret/2.
    closes = _hand_closes({
        "A": [100.0, 104.0, 106.0],
        "B": [100.0, 101.0, 102.0],
        "C": [100.0, 100.0, 100.0],
    })
    date, end = closes["A"].index[0], closes["A"].index[-1]
    rebalances = ((date, ("A", "B"), ("C",)),)
    got = top_leg_contributions(closes, rebalances, end)
    assert got.index[0] == "A"                       # ranked descending
    assert got["A"] == pytest.approx((0.04 + 2.0 / 104.0) / 2)
    assert got["B"] == pytest.approx((0.01 + 1.0 / 101.0) / 2)
    assert "C" not in got.index                      # bottom leg never counted


def test_check_name_concentration_fails_a_three_name_alpha(tmp_path):
    # Spec section 7's three-name fixture: 3 monsters carry the whole top
    # leg; excluding them collapses the alpha below half -> FAIL.
    from trading.alphasearch.robustness import check_name_concentration
    from trading.alphasearch.sort import portfolio_sort

    idx = pd.date_range("2020-01-02", periods=130, freq="B", tz="UTC")
    names = [f"S{i:02d}" for i in range(40)]
    rng = np.random.default_rng(5)
    closes = {}
    for i, sym in enumerate(names):
        drift = 0.02 if i >= 37 else 0.0             # 3 monsters, 37 duds
        rets = drift + rng.normal(0.0, 1e-4, size=130)
        closes[sym] = pd.Series(100.0 * np.cumprod(1 + rets), index=idx)
    panel = PanelData(closes=closes, symbols=tuple(names))
    journal = trials_journal(tmp_path / "journal")
    factors = make_factors()
    ctx = _ctx(journal, panel, factors, spec=_planted_rank_spec())
    # The sort is built over the ctx WINDOW bounds, exactly as run_battery
    # does (check 5 replays memberships to the window end, not the last bar).
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2020-06-30", tz="UTC")
    sort = portfolio_sort(panel, _planted_rank_spec(),
                          panel.decision_dates(start, end), end)
    got = check_name_concentration(ctx, sort)
    assert not got.passed
    assert set(got.detail["excluded"]) == {"S37", "S38", "S39"}
    assert got.detail["retention"] < 0.5
    assert len(list(journal.events())) == 0          # arithmetic: NO new trials


def test_check_name_concentration_passes_a_broad_alpha(tmp_path):
    from trading.alphasearch.robustness import check_name_concentration
    from trading.alphasearch.sort import portfolio_sort

    panel = make_panel(n_symbols=40)                 # linear drift spread
    journal = trials_journal(tmp_path / "journal")
    ctx = _ctx(journal, panel, make_factors())
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2020-06-30", tz="UTC")
    sort = portfolio_sort(panel, SIGNALS["mom21"],
                          panel.decision_dates(start, end), end)
    got = check_name_concentration(ctx, sort)
    assert got.passed
    assert got.number == 5 and got.name == "name_concentration"


def test_month_share_hand_computed():
    from trading.alphasearch.robustness import month_share

    # One trading day per month with known LOG returns .01/.02/.03/.04:
    # top-3 share = .09/.10. expm1 round-trips through log1p.
    idx = pd.DatetimeIndex(
        ["2020-01-15", "2020-02-14", "2020-03-16", "2020-04-15"], tz="UTC"
    )
    ls = pd.Series(np.expm1([0.01, 0.02, 0.03, 0.04]), index=idx)
    assert month_share(ls) == pytest.approx(0.09 / 0.10)
    # Non-positive cumulative log return: concentration is undefined -> NaN
    # (the caller FAILS the check; spec section 6).
    flat = pd.Series(np.expm1([-0.02, 0.01]), index=idx[:2])
    assert math.isnan(month_share(flat))


def test_check_month_concentration_fails_a_single_month_spike():
    from trading.alphasearch.robustness import check_month_concentration

    idx = pd.date_range("2020-01-02", periods=105, freq="B", tz="UTC")
    values = np.full(105, 0.0001)
    march = (idx.month == 3)
    values[march] = 0.01                             # the spike month
    got = check_month_concentration(pd.Series(values, index=idx))
    assert not got.passed
    assert got.number == 6 and got.name == "month_concentration"
    assert got.detail["top3_share"] > 0.60


def test_check_month_concentration_passes_an_even_series():
    from trading.alphasearch.robustness import check_month_concentration

    idx = pd.date_range("2020-01-02", periods=130, freq="B", tz="UTC")
    got = check_month_concentration(pd.Series(0.001, index=idx))
    assert got.passed                                # ~6 even months: 3/6 = 50%


def test_factor_proxy_flag_is_the_smb_costume_detector():
    from trading.alphasearch.robustness import factor_proxy_flag

    costume = {"alpha_t": 3.0, "r2": 0.6,
               "loadings_t": {"Mkt-RF": 1.0, "SMB": 9.0, "HML": 0.5, "Mom": 0.2}}
    got = factor_proxy_flag(costume)
    assert got["flagged"] is True
    assert got["offenders"] == {"SMB": 9.0}
    # R^2 below the floor: high loading t alone does not flag.
    low_r2 = dict(costume, r2=0.4)
    assert factor_proxy_flag(low_r2)["flagged"] is False
    # Loading below 2x |alpha t|: no flag.
    mild = dict(costume, loadings_t={"SMB": 5.9})
    assert factor_proxy_flag(mild)["flagged"] is False
    # Missing stats (errored trial): never flags, never crashes.
    assert factor_proxy_flag({})["flagged"] is False
```

- [ ] **Step 6: Run them to verify they fail**

Run: `uv run pytest tests/test_alphasearch_robustness.py -q -k "ls_series or contributions or concentration or month_share or proxy"`
Expected: FAIL with `ImportError` (names not yet defined).

- [ ] **Step 7: Implement checks 5-7 in `src/trading/alphasearch/robustness.py`**

Add imports at the top: `from trading.alphasearch.evaluate import evaluate_alpha` and `from trading.alphasearch.sort import Membership, SortError, SortResult` (replace the existing bare `SortError` import). Then append:

```python
# ---------------------------------------------------------------------------#
# Checks 5-7: arithmetic on already-computed series. NO new trials journaled.
# ---------------------------------------------------------------------------#
def _segments(rebalances: tuple[Membership, ...], end: pd.Timestamp):
    """(date, top, bottom, hold_end): each membership is held to the next
    ACTUAL rebalance (or the window end) -- skipped decision dates never
    truncate a holding period, matching portfolio_sort's hold-through rule."""
    for j, (date, top, bottom) in enumerate(rebalances):
        hold_end = rebalances[j + 1][0] if j + 1 < len(rebalances) else end
        yield date, top, bottom, hold_end


def ls_series(
    closes: dict[str, pd.Series],
    rebalances: tuple[Membership, ...],
    end: pd.Timestamp,
    *,
    excluded: frozenset[str] = frozenset(),
) -> pd.Series:
    """Replay the daily L/S spread from recorded memberships (bit-identical
    to portfolio_sort's ls -- proven by test), optionally excluding names
    from BOTH legs. A rebalance segment whose top or bottom leg is emptied
    by the exclusion has no defined spread and is dropped; if every segment
    empties, SortError (the caller fails the check)."""
    returns = pd.DataFrame({s: c.pct_change() for s, c in closes.items()})
    parts: list[pd.Series] = []
    for date, top, bottom, hold_end in _segments(rebalances, end):
        top_kept = [s for s in top if s not in excluded]
        bottom_kept = [s for s in bottom if s not in excluded]
        if len(top_kept) == 0 or len(bottom_kept) == 0:
            continue
        segment = returns.loc[(returns.index > date) & (returns.index <= hold_end)]
        parts.append(
            segment[top_kept].mean(axis=1) - segment[bottom_kept].mean(axis=1)
        )
    if len(parts) == 0:
        raise SortError("every rebalance segment emptied by the exclusion")
    return pd.concat(parts).dropna()


def top_leg_contributions(
    closes: dict[str, pd.Series],
    rebalances: tuple[Membership, ...],
    end: pd.Timestamp,
) -> pd.Series:
    """Cumulative per-name contribution to the TOP-quantile leg's daily
    return: on each held day an equal-weight member contributes ret / n_top.
    Descending order -- index[:3] are check 5's exclusion candidates."""
    returns = pd.DataFrame({s: c.pct_change() for s, c in closes.items()})
    totals: dict[str, float] = {}
    for date, top, _bottom, hold_end in _segments(rebalances, end):
        segment = returns.loc[(returns.index > date) & (returns.index <= hold_end)]
        per_name = segment[list(top)].sum() / len(top)
        for sym, value in per_name.items():
            totals[sym] = totals.get(sym, 0.0) + float(value)
    return pd.Series(totals, dtype="float64").sort_values(ascending=False)


def check_name_concentration(ctx: BatteryContext, sort: SortResult) -> CheckResult:
    """Check 5 (frozen): recompute the L/S excluding the top-3 contributors
    to the top leg; remaining four-factor alpha must retain >= 0.5 of the
    journaled original (signed ratio -- symmetric for negative alphas)."""
    end = _window_bounds(ctx.window)[1]
    contributions = top_leg_contributions(ctx.panel.closes, sort.rebalances, end)
    excluded = list(contributions.index[:NAME_EXCLUDE_TOP])
    alpha: float | None = None
    error: str | None = None
    try:
        reduced = ls_series(ctx.panel.closes, sort.rebalances, end,
                            excluded=frozenset(excluded))
        alpha = evaluate_alpha(reduced, ctx.factors, self_financing=True
                               ).alpha_annual_pct
    except (SortError, ValueError, np.linalg.LinAlgError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    r = signed_retention(ctx.full_alpha, alpha)
    ok = not math.isnan(r) and r >= NAME_MIN_RETENTION
    return CheckResult(5, "name_concentration", ok, {
        "excluded": excluded, "alpha_annual_pct": alpha,
        "retention": None if math.isnan(r) else r, "error": error,
    })


def month_share(ls: pd.Series, top_n: int = MONTH_TOP) -> float:
    """Top-`top_n` calendar months' share of the cumulative L/S log return.
    NaN when the cumulative log return is not positive (concentration of a
    non-gain is undefined; callers fail the check -- an amendment would be
    needed before running the battery on a negative-alpha survivor)."""
    log_returns = np.log1p(ls)
    monthly = log_returns.groupby(ls.index.strftime("%Y-%m")).sum()
    total = float(monthly.sum())
    if not total > 0:
        return math.nan
    return float(monthly.nlargest(top_n).sum()) / total


def check_month_concentration(ls: pd.Series) -> CheckResult:
    """Check 6 (frozen): top-3 months' share of cumulative L/S log return
    <= 60%."""
    share = month_share(ls)
    ok = not math.isnan(share) and share <= MONTH_MAX_SHARE
    return CheckResult(6, "month_concentration", ok, {
        "top3_share": None if math.isnan(share) else share,
    })


def factor_proxy_flag(ls_stats: dict) -> dict:
    """Check 7 (frozen; WARNING only, never blocks): any factor loading with
    |t_loading| > 2 x |t_alpha| while R^2 > 0.5 -- the section 9 SMB-costume
    detector. Input is the discovery trial's journaled `ls` block."""
    alpha_t = ls_stats.get("alpha_t")
    r2 = ls_stats.get("r2")
    loadings_t = ls_stats.get("loadings_t") or {}
    offenders: dict[str, float] = {}
    if alpha_t is not None and r2 is not None and float(r2) > PROXY_MIN_R2:
        for name, t in loadings_t.items():
            if t is not None and abs(float(t)) > (
                PROXY_LOADING_MULTIPLE * abs(float(alpha_t))
            ):
                offenders[name] = float(t)
    return {"flagged": len(offenders) > 0, "offenders": offenders,
            "alpha_t": alpha_t, "r2": r2}
```

- [ ] **Step 8: Run the robustness tests, full suite, ruff**

Run: `uv run pytest tests/test_alphasearch_robustness.py tests/test_alphasearch_sort.py -q && uv run pytest -q && uv run ruff check src tests scripts`
Expected: all PASS, ruff clean.

- [ ] **Step 9: Commit**

```bash
git add src/trading/alphasearch/sort.py src/trading/alphasearch/robustness.py \
        tests/test_alphasearch_sort.py tests/test_alphasearch_robustness.py
git commit -m "Add battery arithmetic checks: name/month concentration, factor-proxy [AI]"
```

---

### Task 4: Cost-adjusted alpha table + Amihud capacity curve

**Files:**
- Modify: `src/trading/alphasearch/spec.py` (rename `_amihud` → `amihud_lambda`, lines ~194 and ~276)
- Modify: `src/trading/alphasearch/robustness.py`
- Test: `tests/test_alphasearch_robustness.py`

**Interfaces:**
- Consumes: `SortResult.rebalances`/`Membership` (Task 3), `evaluate_alpha`, `PanelData.view(...).bars(symbol)`.
- Produces (Task 5 relies on):
  - `spec.amihud_lambda(bars: pd.DataFrame) -> float` — the existing `_amihud` body, public (registration unchanged: `_register("amihud", _bar_signal(amihud_lambda))`).
  - `apply_rebalance_charges(ls: pd.Series, charges: list[tuple[pd.Timestamp, float]]) -> pd.Series`
  - `cost_adjusted_table(ls, rebalances, turnover_monthly, factors) -> list[dict]` — rows `{"cost_bps": int, "alpha_annual_pct": float|None, "alpha_t": float|None[, "error": str]}` for c ∈ COST_BPS.
  - `capacity_curve(panel, rebalances, ls, factors) -> list[dict]` — rows `{"book_usd": float, "alpha_annual_pct": float|None, "alpha_t": float|None, "total_impact_charge": float, "skipped_no_lambda": int[, "error": str]}` for B ∈ BOOK_SIZES.

**Frozen formula interpretations (write these into the docstrings verbatim — they are the concrete reading of spec §4's sentences, chosen here once so re-runs are reproducible):**
- Cost table: every rebalance in `rebalances` (formation included) charges `2 × turnover_monthly × c` (both legs trade) against the L/S series on the first return day strictly after that decision date. `turnover_monthly` is exactly the leaderboard's measurement; using its mean per rebalance keeps the total charge identical to charging actuals.
- Capacity: at each rebalance, per leg, each ENTERING name is charged `λ × (B / n_new)` impact on its `1/n_new`-weight position → leg-return drag `λ × B / n_new²`; each EXITING name analogously with the OLD leg's `n_prev` (the size actually being liquidated). λ = `amihud_lambda(view.bars(symbol))` at that decision date (PIT). NaN-λ names are skipped and counted (`skipped_no_lambda`), never fabricated. Final holdings are never charged an exit (the window ends holding them). First-order impact model, documented as a model.

- [ ] **Step 1: Rename `_amihud` → `amihud_lambda` in `src/trading/alphasearch/spec.py`**

Change the def line and extend the docstring's first line:

```python
def amihud_lambda(bars: pd.DataFrame) -> float:
    """Mean |ret| / dollar volume over the last 252 bars; min 126 valid terms
    (non-positive dollar volume or NaN return terms are skipped, never 0).
    Doubles as Piece 3's capacity-curve impact price: for an illiquidity
    signal, the names' own lambda is the most self-consistent EOD impact
    estimate available (robustness.capacity_curve)."""
```

and the registration:

```python
# Illiquidity premium (Amihud): harder-to-trade names pay more.
_register("amihud", _bar_signal(amihud_lambda))
```

`grep -rn "_amihud" src tests scripts` must return nothing afterwards (tests reference the signal only through the registry).

- [ ] **Step 2: Run the tier1 tests to prove the rename is behavior-neutral**

Run: `uv run pytest tests/test_alphasearch_tier1.py -q && uv run ruff check src`
Expected: all PASS (the amihud closed-form test exercises the renamed function through the registry).

- [ ] **Step 3: Write the failing cost/capacity tests**

Append to `tests/test_alphasearch_robustness.py`:

```python
def test_apply_rebalance_charges_lands_on_the_first_following_day():
    from trading.alphasearch.robustness import apply_rebalance_charges

    idx = pd.date_range("2020-01-06", periods=10, freq="B", tz="UTC")
    ls = pd.Series(0.001, index=idx)
    charged = apply_rebalance_charges(
        ls, [(idx[0], 0.003), (idx[5], 0.003), (idx[-1], 0.5)]
    )
    assert charged.iloc[1] == pytest.approx(0.001 - 0.003)
    assert charged.iloc[6] == pytest.approx(0.001 - 0.003)
    # A charge dated on the last day has no following return day: dropped.
    # Untouched entries are copies, bit-identical to the input.
    assert charged.drop(charged.index[[1, 6]]).eq(0.001).all()
    assert ls.eq(0.001).all()                        # input never mutated


def test_cost_adjusted_table_hand_arithmetic_and_monotone(tmp_path):
    from trading.alphasearch.robustness import apply_rebalance_charges, cost_adjusted_table
    from trading.alphasearch.sort import portfolio_sort

    panel = make_panel(n_symbols=40)
    factors = make_factors()
    idx = panel.closes[panel.symbols[0]].index
    dates = panel.decision_dates(idx[0], idx[-1])
    sort = portfolio_sort(panel, SIGNALS["mom21"], dates, idx[-1])
    rows = cost_adjusted_table(sort.ls, sort.rebalances,
                               sort.turnover_monthly, factors)
    assert [r["cost_bps"] for r in rows] == [10, 30, 50]        # FROZEN
    # Hand check the 30 bps charge: 2 legs x turnover x 0.003 per rebalance.
    per_rebalance = 2.0 * sort.turnover_monthly * 30 / 1e4
    charged = apply_rebalance_charges(
        sort.ls, [(d, per_rebalance) for d, _t, _b in sort.rebalances]
    )
    from trading.alphasearch.evaluate import evaluate_alpha
    expected = evaluate_alpha(charged, factors, self_financing=True)
    row30 = rows[1]
    assert row30["alpha_annual_pct"] == pytest.approx(expected.alpha_annual_pct)
    assert row30["alpha_t"] == pytest.approx(expected.alpha_tstat)
    # Costs only ever hurt: alpha decreases with c.
    alphas = [r["alpha_annual_pct"] for r in rows]
    assert alphas[0] > alphas[1] > alphas[2]


def test_cost_adjusted_table_without_turnover_reports_error():
    from trading.alphasearch.robustness import cost_adjusted_table

    idx = pd.date_range("2020-01-06", periods=10, freq="B", tz="UTC")
    ls = pd.Series(0.001, index=idx)
    rows = cost_adjusted_table(ls, ((idx[0], ("A",), ("B",)),),
                               float("nan"), make_factors())
    assert all(r["alpha_t"] is None for r in rows)
    assert all("turnover" in r["error"] for r in rows)


def _lambda_bars(n: int, daily_ret: float, dollar: float) -> pd.DataFrame:
    """Bars with constant pct return and constant dollar volume, so
    amihud_lambda == daily_ret / dollar exactly (the tier1 closed form)."""
    idx = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    close = pd.Series(100.0 * (1 + daily_ret) ** np.arange(n), index=idx)
    return pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
        "volume": dollar / close, "div_cash": 0.0, "split_factor": 1.0,
        "close_raw": close,
    })


def test_capacity_curve_two_name_hand_computed():
    from trading.alphasearch.robustness import capacity_curve

    # lambda_A = .01/1e6 = 1e-8, lambda_B = .01/5e5 = 2e-8 (known lambdas).
    # 200 bars so the formation date sees >= 126 valid return terms (the
    # amihud_lambda floor) -- at idx[130] each name has 130 valid terms.
    bars = {"A": _lambda_bars(200, 0.01, 1e6), "B": _lambda_bars(200, 0.01, 5e5)}
    closes = {s: f["close"] for s, f in bars.items()}
    panel = PanelData(closes=closes, bars=bars, symbols=("A", "B"))
    idx = closes["A"].index
    factors = make_factors(periods=300)
    # One formation: both names enter 1-name legs; 69 held days follow.
    date = idx[130]
    rebalances = ((date, ("A",), ("B",)),)
    ls = pd.Series(0.001, index=idx[131:])
    rows = capacity_curve(panel, rebalances, ls, factors)
    assert [r["book_usd"] for r in rows] == [1e4, 1e5, 1e6]     # FROZEN
    # Entry drag per $1 of book: lambda_A/1^2 + lambda_B/1^2 = 3e-8.
    assert rows[2]["total_impact_charge"] == pytest.approx(3e-8 * 1e6, rel=1e-6)
    assert rows[0]["total_impact_charge"] == pytest.approx(3e-8 * 1e4, rel=1e-6)
    assert all(r["skipped_no_lambda"] == 0 for r in rows)
    # More book, more impact, less alpha.
    alphas = [r["alpha_annual_pct"] for r in rows]
    assert alphas[0] > alphas[1] > alphas[2]


def test_capacity_curve_charges_exits_at_the_old_leg_size():
    from trading.alphasearch.robustness import capacity_curve

    # 200 bars: both rebalance dates clear the 126-valid-term lambda floor.
    bars = {"A": _lambda_bars(200, 0.01, 1e6), "B": _lambda_bars(200, 0.01, 1e6),
            "C": _lambda_bars(200, 0.01, 1e6), "D": _lambda_bars(200, 0.01, 1e6)}
    closes = {s: f["close"] for s, f in bars.items()}
    panel = PanelData(closes=closes, bars=bars, symbols=("A", "B", "C", "D"))
    idx = closes["A"].index
    lam = 1e-8
    # Formation: top=(A,B) bottom=(C,) -- then A,B exit for (D,) top.
    rebalances = (
        (idx[130], ("A", "B"), ("C",)),
        (idx[140], ("D",), ("C",)),
    )
    ls = pd.Series(0.001, index=idx[131:])
    rows = capacity_curve(panel, rebalances, ls, make_factors(periods=300))
    # Formation: A,B enter n=2 legs (2 x lam/4), C enters n=1 (lam).
    # Rebalance 2: D enters n=1 (lam); A,B exit at OLD n=2 (2 x lam/4).
    per_dollar = (2 * lam / 4 + lam) + (lam + 2 * lam / 4)
    assert rows[2]["total_impact_charge"] == pytest.approx(per_dollar * 1e6,
                                                           rel=1e-6)


def test_capacity_curve_counts_missing_lambda_names():
    from trading.alphasearch.robustness import capacity_curve

    bars = {"A": _lambda_bars(130, 0.01, 1e6),
            "B": _lambda_bars(60, 0.01, 1e6)}      # < 126 valid terms -> NaN
    closes = {s: f["close"] for s, f in bars.items()}
    panel = PanelData(closes=closes, bars=bars, symbols=("A", "B"))
    idx = closes["A"].index
    rebalances = ((idx[128], ("A",), ("B",)),)
    ls = pd.Series(0.001, index=idx[129:])
    rows = capacity_curve(panel, rebalances, ls, make_factors(periods=200))
    assert all(r["skipped_no_lambda"] == 1 for r in rows)
    assert rows[2]["total_impact_charge"] == pytest.approx(1e-8 * 1e6, rel=1e-6)
```

- [ ] **Step 4: Run them to verify they fail**

Run: `uv run pytest tests/test_alphasearch_robustness.py -q -k "charges or cost_adjusted or capacity"`
Expected: FAIL with `ImportError` (names not yet defined).

- [ ] **Step 5: Implement cost + capacity in `src/trading/alphasearch/robustness.py`**

Add `from trading.alphasearch.spec import SIGNALS, SignalSpec, amihud_lambda` (extend the existing spec import). Append:

```python
# ---------------------------------------------------------------------------#
# Cost & capacity analysis (spec section 4): arithmetic, NO new trials.
# ---------------------------------------------------------------------------#
def apply_rebalance_charges(
    ls: pd.Series, charges: list[tuple[pd.Timestamp, float]]
) -> pd.Series:
    """Deduct each charge from the first daily return strictly after its
    rebalance date (the first day the traded book exists). A charge dated at
    or after the last return day has no day to land on and is dropped (a
    final-day rebalance is never held). Returns a copy."""
    charged = ls.copy()
    for date, charge in charges:
        pos = int(charged.index.searchsorted(date, side="right"))
        if pos < len(charged):
            charged.iloc[pos] -= charge
    return charged


def _regress_charged(charged: pd.Series, factors: pd.DataFrame, row: dict) -> dict:
    try:
        alpha = evaluate_alpha(charged, factors, self_financing=True)
        row["alpha_annual_pct"] = alpha.alpha_annual_pct
        row["alpha_t"] = alpha.alpha_tstat
    except (ValueError, np.linalg.LinAlgError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def cost_adjusted_table(
    ls: pd.Series,
    rebalances: tuple[Membership, ...],
    turnover_monthly: float,
    factors: pd.DataFrame,
) -> list[dict]:
    """Cost-adjusted alpha (spec section 4, frozen reading): for each one-way
    cost c in COST_BPS, EVERY rebalance (formation included) charges
    2 x turnover_monthly x c -- both legs trade -- against the L/S series on
    the first return day after that decision date, and the charged series is
    re-regressed. turnover_monthly is exactly the leaderboard's measurement;
    charging its mean per rebalance totals the same as charging actuals."""
    rows: list[dict] = []
    dates = [date for date, _top, _bottom in rebalances]
    usable = turnover_monthly is not None and not math.isnan(turnover_monthly)
    for bps in COST_BPS:
        row: dict = {"cost_bps": bps, "alpha_annual_pct": None, "alpha_t": None}
        if usable:
            per_rebalance = 2.0 * float(turnover_monthly) * bps / 1e4
            charged = apply_rebalance_charges(
                ls, [(d, per_rebalance) for d in dates]
            )
            row = _regress_charged(charged, factors, row)
        else:
            row["error"] = "turnover unavailable (single rebalance?)"
        rows.append(row)
    return rows


def capacity_curve(
    panel: PanelData,
    rebalances: tuple[Membership, ...],
    ls: pd.Series,
    factors: pd.DataFrame,
) -> list[dict]:
    """Amihud-implied capacity (spec section 4, frozen reading): for each
    book size B per side, every name ENTERING a leg is charged its own
    lambda x (B / n_new) impact on its 1/n_new-weight position -- a
    leg-return drag of lambda x B / n_new^2 -- and every EXITING name the
    analogue at the OLD leg's size (the position actually liquidated).
    lambda = spec.amihud_lambda(view.bars(sym)) at the rebalance date (PIT).
    NaN-lambda names are skipped and counted, never fabricated; final
    holdings are never charged an exit. This is a FIRST-ORDER impact model
    -- honest about being a model; for an illiquidity signal the names' own
    lambda is the most self-consistent EOD impact estimate available."""
    unit_charges: list[tuple[pd.Timestamp, float]] = []  # per $1 of book
    skipped = 0
    prev: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    for date, top, bottom in rebalances:
        view = panel.view(date)
        unit = 0.0
        for leg_index, current in enumerate((top, bottom)):
            previous = prev[leg_index] if prev is not None else ()
            cur_set, prev_set = set(current), set(previous)
            for sym in sorted(cur_set - prev_set):        # entries
                lam = amihud_lambda(view.bars(sym))
                if math.isnan(lam):
                    skipped += 1
                    continue
                unit += lam / (len(current) * len(current))
            for sym in sorted(prev_set - cur_set):        # exits
                lam = amihud_lambda(view.bars(sym))
                if math.isnan(lam):
                    skipped += 1
                    continue
                unit += lam / (len(previous) * len(previous))
        unit_charges.append((date, unit))
        prev = (top, bottom)
    rows: list[dict] = []
    for book in BOOK_SIZES:
        row: dict = {
            "book_usd": book, "alpha_annual_pct": None, "alpha_t": None,
            "total_impact_charge": sum(u for _d, u in unit_charges) * book,
            "skipped_no_lambda": skipped,
        }
        charged = apply_rebalance_charges(
            ls, [(d, u * book) for d, u in unit_charges]
        )
        rows.append(_regress_charged(charged, factors, row))
    return rows
```

- [ ] **Step 6: Run the tests, full suite, ruff**

Run: `uv run pytest tests/test_alphasearch_robustness.py -q && uv run pytest -q && uv run ruff check src tests scripts`
Expected: all PASS, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/trading/alphasearch/spec.py src/trading/alphasearch/robustness.py \
        tests/test_alphasearch_robustness.py
git commit -m "Add cost-adjusted alpha table and Amihud capacity curve [AI]"
```

---

### Task 5: `run_battery` composition, `kind="battery"` verdict event, holdout battery gate

**Files:**
- Modify: `src/trading/alphasearch/robustness.py` (`BatteryOutcome`, `run_battery`)
- Modify: `src/trading/alphasearch/sweep.py` (`battery_verdict`, `run_holdout` pre-check)
- Modify: `tests/test_alphasearch_sweep.py` (existing holdout setups + new gate tests)
- Modify: `tests/test_alphasearch_cli.py` (`_seed_journal`)
- Test: `tests/test_alphasearch_robustness.py`

**Interfaces:**
- Consumes: everything Tasks 1-4 produced.
- Produces (Task 6 renders these exact fields):

```python
@dataclass(frozen=True)
class BatteryOutcome:
    signal: str
    universe: str
    window: str                       # the discovery window interrogated
    checks: tuple[CheckResult, ...]   # checks 1-6, in spec order
    factor_proxy: dict                # {"flagged", "offenders", "alpha_t", "r2"}
    cost_table: list[dict]
    capacity_curve: list[dict]
    eligible: bool
    event: dict                       # the journaled kind="battery" verdict
```

  - `run_battery(uspec: UniverseSpec, journal: Journal, factors: pd.DataFrame, ts: str, signal_name: str, *, discovery_window: str = DISCOVERY_WINDOW, quantiles: int = QUANTILES, tercile_below: int = TERCILE_BELOW, min_names: int = MIN_NAMES, panel_factory=build_universe_panel) -> BatteryOutcome`
  - `sweep.battery_verdict(journal: Journal, config_hash: str) -> dict | None` — latest `kind="battery"` event for the hash (load_trials dedupe).
  - `run_holdout` refuses survivors whose battery is missing or `eligible is not True`; the refusal names `trading alphasearch robustness <signal>:<universe>`.

- [ ] **Step 1: Write the failing `run_battery` composition tests**

Append to `tests/test_alphasearch_robustness.py`:

```python
# --------------------------------------------------------------------------- #
# run_battery composition + verdict event
# --------------------------------------------------------------------------- #
def _swept_survivor(tmp_path, n_symbols=40):
    """A real BH survivor: mom21 swept on the wide fixture panel."""
    from trading.alphasearch.sweep import UniverseSpec, run_sweep

    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(n_symbols=n_symbols)
    factors = make_factors()
    uspec = UniverseSpec("largecap", tmp_path, tmp_path / "s.jsonl", None)
    run_sweep({"largecap": uspec}, journal, factors, ts="t0",
              signals={"mom21": SIGNALS["mom21"]}, window=WINDOW,
              panel_factory=lambda _u, _f: panel)
    return journal, panel, factors, uspec


def test_run_battery_refuses_nonsurvivor_before_journaling(tmp_path):
    from trading.alphasearch.robustness import run_battery
    from trading.alphasearch.sweep import UniverseSpec

    journal = trials_journal(tmp_path / "journal")
    log_trial(journal, kind="discovery",
              config=trial_config("mom21", "largecap", WINDOW), ts="t1",
              result=_result_like(alpha_annual_pct=12.0, alpha_t=8.0, p=1e-8))
    log_trial(journal, kind="discovery",
              config=trial_config("rvol21", "largecap", WINDOW), ts="t1",
              result=_result_like(alpha_annual_pct=0.3, alpha_t=0.1, p=0.92))
    uspec = UniverseSpec("largecap", tmp_path, tmp_path / "s.jsonl", None)
    before = len(list(journal.events()))
    with pytest.raises(SweepError, match="not a current BH survivor"):
        run_battery(uspec, journal, make_factors(), "t2", "rvol21",
                    discovery_window=WINDOW,
                    panel_factory=lambda _u, _f: make_panel())
    assert len(list(journal.events())) == before      # journaled NOTHING


def test_run_battery_journals_tagged_trials_and_a_verdict(tmp_path):
    from trading.alphasearch.robustness import run_battery
    from trading.alphasearch.sweep import discovery_trials

    journal, panel, factors, uspec = _swept_survivor(tmp_path)
    outcome = run_battery(uspec, journal, factors, "t1", "mom21",
                          discovery_window=WINDOW,
                          panel_factory=lambda _u, _f: panel)
    # 2 halves + 5 draws + 4 jitter + 1 offset = 12 tagged re-evaluations,
    # all BH-counted discovery trials on top of the 1 swept trial.
    trials = discovery_trials(journal)
    assert len(trials) == 13
    tagged = [t for t in trials if t.get("battery") == "mom21:largecap"]
    assert len(tagged) == 12
    # Checks 1-6 in spec order; 7 is the warning dict.
    assert [c.number for c in outcome.checks] == [1, 2, 3, 4, 5, 6]
    assert [c.name for c in outcome.checks] == [
        "sub_period_halves", "universe_subsets", "parameter_jitter",
        "decision_offset", "name_concentration", "month_concentration",
    ]
    assert isinstance(outcome.eligible, bool)
    assert set(outcome.factor_proxy) == {"flagged", "offenders", "alpha_t", "r2"}
    assert [r["cost_bps"] for r in outcome.cost_table] == [10, 30, 50]
    assert [r["book_usd"] for r in outcome.capacity_curve] == [1e4, 1e5, 1e6]
    # The verdict event: kind="battery", SAME config hash as the discovery
    # trial (default params), full payload, json-safe.
    event = outcome.event
    assert event["kind"] == "battery"
    discovery = next(t for t in trials if t.get("battery") is None
                     and t["signal"] == "mom21" and t["window"] == WINDOW)
    assert event["config_hash"] == discovery["config_hash"]
    assert event["eligible"] == outcome.eligible
    assert set(event["checks"]) == {c.name for c in outcome.checks}


def test_run_battery_rerun_replaces_the_verdict_and_double_counts_nothing(tmp_path):
    from trading.alphasearch.robustness import run_battery
    from trading.alphasearch.sweep import battery_verdict, discovery_trials

    journal, panel, factors, uspec = _swept_survivor(tmp_path)
    kwargs = dict(discovery_window=WINDOW, panel_factory=lambda _u, _f: panel)
    first = run_battery(uspec, journal, factors, "t1", "mom21", **kwargs)
    events_after_first = len(list(journal.events()))
    second = run_battery(uspec, journal, factors, "t2", "mom21", **kwargs)
    assert len(discovery_trials(journal)) == 13          # identical configs dedupe
    # Verdict replaced by config hash: ONE battery event survives dedupe,
    # and it is the LATEST.
    verdict = battery_verdict(journal, first.event["config_hash"])
    assert verdict is not None and verdict["ts"] == "t2"
    # ...but the journal itself is append-only (both runs appended).
    assert len(list(journal.events())) > events_after_first
    assert second.eligible == first.eligible             # deterministic


def test_run_battery_narrow_universe_fails_check2_via_error_trials(tmp_path):
    # 16 names: half-draws of 8 < min_names 15 -> the five subset trials are
    # honest SortError error trials, the check fails, the battery still
    # completes and journals its verdict (an uncomputable perturbation is a
    # FAIL, not a crash).
    from trading.alphasearch.robustness import run_battery

    journal, panel, factors, uspec = _swept_survivor(tmp_path, n_symbols=16)
    outcome = run_battery(uspec, journal, factors, "t1", "mom21",
                          discovery_window=WINDOW,
                          panel_factory=lambda _u, _f: panel)
    subsets = next(c for c in outcome.checks if c.name == "universe_subsets")
    assert not subsets.passed and subsets.detail["n_pass"] == 0
    assert not outcome.eligible
    errored = [e for e in journal.events()
               if e.get("battery") is not None and e.get("error") is not None]
    assert len(errored) >= 5
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_alphasearch_robustness.py -q -k run_battery`
Expected: FAIL with `ImportError: cannot import name 'run_battery'`.

- [ ] **Step 3: Implement `battery_verdict` in `src/trading/alphasearch/sweep.py`**

Below `_bh_survivor_hashes`:

```python
def battery_verdict(journal: Journal, config_hash: str) -> dict | None:
    """The latest kind="battery" verdict event for a discovery config hash
    (Piece 3). load_trials' (config_hash, kind) dedupe makes re-runs replace
    in place; None when the battery has never been run for this config."""
    for event in load_trials(journal):
        if event.get("kind") == "battery" and event.get("config_hash") == config_hash:
            return event
    return None
```

- [ ] **Step 4: Implement `BatteryOutcome` + `run_battery` in `src/trading/alphasearch/robustness.py`**

Extend the sweep import block with `DISCOVERY_WINDOW`, `UniverseSpec`, `_check_factor_coverage`, `_check_universe_supports`, `build_universe_panel`; add `from collections.abc import Callable`; add `from trading.alphasearch.sort import MIN_NAMES, QUANTILES, TERCILE_BELOW, portfolio_sort` (extend the existing sort import). Append:

```python
# ---------------------------------------------------------------------------#
# The battery runner: refusal gate -> checks 1-4 (journaled re-evaluations)
# -> checks 5-7 + cost/capacity (arithmetic) -> journaled verdict.
# ---------------------------------------------------------------------------#
@dataclass(frozen=True)
class BatteryOutcome:
    signal: str
    universe: str
    window: str
    checks: tuple[CheckResult, ...]   # checks 1-6, spec order
    factor_proxy: dict                # check 7: warning only, never blocks
    cost_table: list[dict]
    capacity_curve: list[dict]
    eligible: bool                    # the frozen promotion rule
    event: dict                       # the journaled kind="battery" verdict


def run_battery(
    uspec: UniverseSpec,
    journal: Journal,
    factors: pd.DataFrame,
    ts: str,
    signal_name: str,
    *,
    discovery_window: str = DISCOVERY_WINDOW,
    quantiles: int = QUANTILES,
    tercile_below: int = TERCILE_BELOW,
    min_names: int = MIN_NAMES,
    panel_factory: Callable[[UniverseSpec, pd.DataFrame | None], PanelData] = (
        build_universe_panel
    ),
) -> BatteryOutcome:
    """Run the frozen battery on ONE current BH survivor (spec section 3).

    Refuses non-survivors BEFORE journaling anything (require_survivor).
    Checks 1-4 journal battery-tagged, BH-counted discovery trials; checks
    5-7 and the cost/capacity analysis are arithmetic on a locally
    recomputed full-window sort (identical config to the journaled discovery
    trial -- deliberately NOT journaled again). The verdict is one
    kind="battery" event per (signal, universe); re-runs replace by config
    hash. The promotion rule (frozen): eligible iff checks 1-6 all pass AND
    the 30 bps cost row retains t >= ELIGIBLE_MIN_COST_T. Never touches
    holdout state."""
    params = _hashed_params(quantiles, tercile_below, min_names)
    discovery = require_survivor(journal, signal_name, uspec.name,
                                 discovery_window, params)
    full_alpha = float(discovery["ls"]["alpha_annual_pct"])
    start, end = _window_bounds(discovery_window)
    # Stale factors refuse HERE, before the first re-evaluation journals:
    # letting evaluate_trial hit the coverage check inside the loop would
    # journal 12 predictable error trials for one fixable cache problem.
    try:
        _check_factor_coverage(factors, end)
    except ValueError as exc:
        raise SweepError(str(exc)) from exc
    panel = panel_factory(uspec, factors)
    spec = SIGNALS[signal_name]
    _check_universe_supports(panel, spec, uspec.name)
    ctx = BatteryContext(
        journal=journal, panel=panel, spec=spec, factors=factors, ts=ts,
        universe=uspec.name, window=discovery_window, full_alpha=full_alpha,
        quantiles=quantiles, tercile_below=tercile_below, min_names=min_names,
        tag=f"{signal_name}:{uspec.name}",
    )
    checks = [check_subperiods(ctx), check_subsets(ctx), check_jitter(ctx),
              check_offset(ctx)]
    # Full-window sort recomputed ONCE for checks 5-6 and cost/capacity --
    # same config as the journaled discovery trial, so journaling it again
    # would only append a duplicate; spec: checks 5-7 journal no new trials.
    try:
        sort = portfolio_sort(
            panel, spec, panel.decision_dates(start, end), end,
            quantiles=quantiles, tercile_below=tercile_below,
            min_names=min_names,
        )
    except SortError as exc:
        # The journaled discovery trial was clean, so current caches must
        # have drifted from the evidence. Refuse loudly; journal nothing new.
        raise SweepError(
            f"full-window sort failed against current data ({exc}) although "
            f"the journaled discovery trial is clean; re-run the sweep before "
            f"the battery"
        ) from exc
    checks.append(check_name_concentration(ctx, sort))
    checks.append(check_month_concentration(sort.ls))
    proxy = factor_proxy_flag(discovery.get("ls") or {})
    cost_rows = cost_adjusted_table(sort.ls, sort.rebalances,
                                    sort.turnover_monthly, factors)
    capacity_rows = capacity_curve(panel, sort.rebalances, sort.ls, factors)
    cost_t = next(
        (r["alpha_t"] for r in cost_rows if r["cost_bps"] == ELIGIBLE_COST_BPS),
        None,
    )
    eligible = (
        all(c.passed for c in checks)
        and cost_t is not None
        and float(cost_t) >= ELIGIBLE_MIN_COST_T
    )
    config = trial_config(signal_name, uspec.name, discovery_window,
                          params=params)
    verdict = {
        "checks": {
            c.name: {"number": c.number, "passed": c.passed, **c.detail}
            for c in checks
        },
        "factor_proxy": proxy,
        "cost_table": cost_rows,
        "capacity_curve": capacity_rows,
        "eligible": eligible,
    }
    event = log_trial(journal, kind="battery", config=config, ts=ts,
                      result=verdict)
    return BatteryOutcome(
        signal=signal_name, universe=uspec.name, window=discovery_window,
        checks=tuple(checks), factor_proxy=proxy, cost_table=cost_rows,
        capacity_curve=capacity_rows, eligible=eligible, event=event,
    )
```

- [ ] **Step 5: Run the composition tests**

Run: `uv run pytest tests/test_alphasearch_robustness.py -q`
Expected: all PASS.

- [ ] **Step 6: Write the failing holdout-gate tests**

Append to `tests/test_alphasearch_sweep.py`:

```python
def _log_battery_verdict(journal, signal, universe, window, *, eligible,
                         params=None):
    """Fabricate a Piece 3 battery verdict for the (default-params) config."""
    from trading.alphasearch.sweep import log_trial, trial_config

    log_trial(journal, kind="battery",
              config=trial_config(signal, universe, window, params=params),
              ts="tb", result={"eligible": eligible})


def test_holdout_refused_for_battery_less_survivor(tmp_path):
    # Prospective amendment to Piece 1 spec 3.6 (Piece 3 design spec): no
    # holdout may be spent on a survivor that has not passed its battery.
    journal, panel, factors = _sweep_then_holdout_setup(tmp_path,
                                                        with_battery=False)
    with pytest.raises(SweepError) as excinfo:
        run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t2",
                    "mom21", holdout_start=HOLDOUT_FROM,
                    discovery_window=DISCOVERY, min_factor_span_days=MIN_SPAN,
                    panel_factory=lambda _u, _f: panel)
    # The refusal names the command that fixes it, and journals no touch.
    assert "trading alphasearch robustness mom21:largecap" in str(excinfo.value)
    assert all(e.get("kind") != "holdout" for e in journal.events())


def test_holdout_refused_for_battery_failed_survivor(tmp_path):
    journal, panel, factors = _sweep_then_holdout_setup(tmp_path,
                                                        with_battery=False)
    _log_battery_verdict(journal, "mom21", "largecap", DISCOVERY,
                         eligible=False)
    with pytest.raises(SweepError, match="did not pass"):
        run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t2",
                    "mom21", holdout_start=HOLDOUT_FROM,
                    discovery_window=DISCOVERY, min_factor_span_days=MIN_SPAN,
                    panel_factory=lambda _u, _f: panel)
    assert all(e.get("kind") != "holdout" for e in journal.events())


def test_holdout_battery_gate_binds_to_the_exact_config(tmp_path):
    # A battery verdict for DIFFERENT params must not qualify the default-
    # params holdout (hash-keyed, like the BH survivor gate).
    journal, panel, factors = _sweep_then_holdout_setup(tmp_path,
                                                        with_battery=False)
    other_params = {"quantiles": 4, "weighting": "equal", "cadence": "monthly",
                    "tercile_below": 50, "min_names": 15}
    _log_battery_verdict(journal, "mom21", "largecap", DISCOVERY,
                         eligible=True, params=other_params)
    with pytest.raises(SweepError, match="robustness"):
        run_holdout(_universe(tmp_path)["largecap"], journal, factors, "t2",
                    "mom21", holdout_start=HOLDOUT_FROM,
                    discovery_window=DISCOVERY, min_factor_span_days=MIN_SPAN,
                    panel_factory=lambda _u, _f: panel)
```

Update `_sweep_then_holdout_setup` so every EXISTING holdout test keeps passing (they now need a passing battery):

```python
def _sweep_then_holdout_setup(tmp_path, with_battery: bool = True):
    """Discovery on Q1 2020; the fixture's remaining bars are the holdout.
    with_battery fabricates the Piece 3 battery-passed verdict the holdout
    now requires (its own tests set False to exercise the refusal)."""
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel()
    factors = make_factors()
    run_sweep(_universe(tmp_path), journal, factors, ts="t1",
              signals=_subset("mom21"), window=DISCOVERY,
              panel_factory=lambda _u, _f: panel)
    if with_battery:
        _log_battery_verdict(journal, "mom21", "largecap", DISCOVERY,
                             eligible=True)
    return journal, panel, factors
```

(`_log_battery_verdict` must be defined ABOVE `_sweep_then_holdout_setup` in the file.)

Two other existing tests need the verdict too, because they bypass the setup helper:

- `test_holdout_journals_the_actual_params` — it sweeps with `quantiles=4`, so give it a matching-params verdict right after its `run_sweep(...)` call:

```python
    _log_battery_verdict(
        journal, "mom21", "largecap", DISCOVERY, eligible=True,
        params={"quantiles": 4, "weighting": "equal", "cadence": "monthly",
                "tercile_below": 50, "min_names": 15},
    )
```

- `test_holdout_refused_when_discovery_alpha_missing` — no change needed: its refusal fires BEFORE the battery gate (null alpha). Verify it still passes.

And in `tests/test_alphasearch_cli.py`, extend `_seed_journal` (its holdout CLI test must get past the battery gate to reach the double-touch prompt):

```python
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
    # Piece 3: the holdout now requires a battery-passed verdict.
    log_trial(journal, kind="battery",
              config=trial_config("mom21", "largecap", DISCOVERY_WINDOW),
              ts="t1b", result={"eligible": True})
    if with_holdout:
        log_trial(journal, kind="holdout",
                  config=trial_config("mom21", "largecap", "2024-01-01..2026-07-07"),
                  ts="t2", result=result)
```

- [ ] **Step 7: Run the gate tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_sweep.py -q -k battery`
Expected: FAIL — `run_holdout` does not refuse yet (no gate), and `_sweep_then_holdout_setup` signature errors until updated.

- [ ] **Step 8: Add the battery pre-check to `run_holdout` in `src/trading/alphasearch/sweep.py`**

Insert immediately AFTER the `_bh_survivor_hashes` refusal block (line ~689, after `raise SweepError(... reserved for survivors ...)`) and BEFORE the `prior = prior_holdout_trial(...)` line — the gate must precede the rerun prompt so an ineligible candidate is never asked for the confirmation ceremony:

```python
    # Piece 3 battery gate -- a WRITTEN PROSPECTIVE AMENDMENT to Piece 1 spec
    # 3.6, recorded in docs/superpowers/specs/2026-07-09-robustness-battery-
    # design.md section 3: no holdout may be spent on a survivor that has not
    # passed its robustness battery. No holdout had ever been spent when this
    # gate was added, so nothing is affected retroactively. Hash-keyed to the
    # EXACT discovery trial, like the BH gate above.
    verdict = battery_verdict(journal, discovery["config_hash"])
    if verdict is None or verdict.get("eligible") is not True:
        state = ("has not been run" if verdict is None
                 else "did not pass (not holdout-eligible)")
        raise SweepError(
            f"robustness battery for {signal_name}:{uspec.name} {state}; the "
            f"once-only holdout is reserved for battery-passed survivors. "
            f"Run `trading alphasearch robustness {signal_name}:{uspec.name}` "
            f"first"
        )
```

Also update `run_holdout`'s docstring refusal list: append `; battery not passed (Piece 3)` to the refusals sentence.

- [ ] **Step 9: Run the full suite + ruff**

Run: `uv run pytest -q && uv run ruff check src tests scripts`
Expected: all PASS (including every pre-existing holdout test, now riding the updated setup), ruff clean.

- [ ] **Step 10: Commit**

```bash
git add src/trading/alphasearch/robustness.py src/trading/alphasearch/sweep.py \
        tests/test_alphasearch_robustness.py tests/test_alphasearch_sweep.py \
        tests/test_alphasearch_cli.py
git commit -m "Add run_battery verdict event and holdout battery gate [AI]"
```

---

### Task 6: CLI `robustness` action — report card, red factor-proxy warning, `--json`

**Files:**
- Modify: `src/trading/cli.py` (parser at line ~152, `_cmd_alphasearch` at line ~850, new renderers)
- Test: `tests/test_alphasearch_cli.py`

**Interfaces:**
- Consumes: `robustness.run_battery` / `BatteryOutcome` (Task 5), existing `_load_alphasearch_factors`, `engine.default_universes`, `segments.segment_universes`.
- Produces: `trading alphasearch robustness <signal>:<universe> [--json]`; `_resolve_alphasearch_universe(universe: str) -> UniverseSpec | None` (shared by holdout + robustness; prints its own error); `_render_battery(outcome) -> None`.

- [ ] **Step 1: Write the failing CLI tests**

Append to `tests/test_alphasearch_cli.py`:

```python
# --------------------------------------------------------------------------- #
# robustness (Piece 3)
# --------------------------------------------------------------------------- #


def test_robustness_requires_trial_id(tmp_path, capsys):
    rc = cli.main(["alphasearch", "robustness", "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "signal:universe" in capsys.readouterr().err


def test_robustness_unknown_universe_rejected(tmp_path, capsys):
    rc = cli.main(["alphasearch", "robustness", "mom21:smallcap",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "unknown universe" in capsys.readouterr().err


def test_robustness_refusal_prints_error(tmp_path, capsys, monkeypatch):
    # Empty journal: run_battery refuses (no discovery trial) before any IO
    # beyond factors, which the stub keeps offline.
    monkeypatch.setattr("trading.alphasearch.evaluate.load_factors",
                        lambda *a, **k: pd.DataFrame())
    rc = cli.main(["alphasearch", "robustness", "mom21:largecap",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "no discovery trial" in capsys.readouterr().err


def _fake_battery_outcome(*, eligible=False, flagged=True):
    from trading.alphasearch.robustness import BatteryOutcome, CheckResult

    checks = (
        CheckResult(1, "sub_period_halves", True, {"halves": [
            {"window": "2019-01-01..2021-06-30", "alpha_annual_pct": 30.0,
             "alpha_t": 4.0, "error": None, "passed": True},
            {"window": "2021-07-01..2023-12-31", "alpha_annual_pct": 25.0,
             "alpha_t": 3.0, "error": None, "passed": True}]}),
        CheckResult(2, "universe_subsets", True,
                    {"draws": [], "n_pass": 5}),
        CheckResult(3, "parameter_jitter", True, {"trials": []}),
        CheckResult(4, "decision_offset", True,
                    {"offset_sessions": 1, "alpha_annual_pct": 28.0,
                     "alpha_t": 3.5, "retention": 0.9, "error": None}),
        CheckResult(5, "name_concentration", False,
                    {"excluded": ["AAA", "BBB", "CCC"],
                     "alpha_annual_pct": 5.0, "retention": 0.16, "error": None}),
        CheckResult(6, "month_concentration", True, {"top3_share": 0.41}),
    )
    return BatteryOutcome(
        signal="amihud", universe="midcap", window="2019-01-01..2023-12-31",
        checks=checks,
        factor_proxy={"flagged": flagged, "offenders": {"SMB": 13.1},
                      "alpha_t": 3.0, "r2": 0.6},
        cost_table=[{"cost_bps": 10, "alpha_annual_pct": 55.0, "alpha_t": 7.1},
                    {"cost_bps": 30, "alpha_annual_pct": 48.0, "alpha_t": 6.0},
                    {"cost_bps": 50, "alpha_annual_pct": 41.0, "alpha_t": 4.9}],
        capacity_curve=[
            {"book_usd": 1e4, "alpha_annual_pct": 60.0, "alpha_t": 8.0,
             "total_impact_charge": 0.01, "skipped_no_lambda": 0},
            {"book_usd": 1e5, "alpha_annual_pct": 52.0, "alpha_t": 6.8,
             "total_impact_charge": 0.1, "skipped_no_lambda": 0},
            {"book_usd": 1e6, "alpha_annual_pct": 20.0, "alpha_t": 2.1,
             "total_impact_charge": 1.0, "skipped_no_lambda": 0}],
        eligible=eligible,
        event={"event": "trial", "kind": "battery", "signal": "amihud",
               "universe": "midcap", "window": "2019-01-01..2023-12-31",
               "config_hash": "4f3d0819382a", "ts": "t1", "error": None,
               "eligible": eligible},
    )


def test_robustness_report_card_renders_with_red_proxy_warning(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr("trading.alphasearch.evaluate.load_factors",
                        lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("trading.alphasearch.robustness.run_battery",
                        lambda *a, **k: _fake_battery_outcome())
    rc = cli.main(["alphasearch", "robustness", "amihud:midcap",
                   "--journal-dir", str(tmp_path)])
    assert rc == 0                                   # completed battery = 0
    out = capsys.readouterr().out
    assert "name_concentration" in out and "FAIL" in out
    assert "sub_period_halves" in out and "PASS" in out
    assert "FACTOR-PROXY WARNING" in out and "SMB" in out
    assert "30" in out and "capacity" in out.lower()
    assert "holdout-eligible: NO" in out


def test_robustness_json_dumps_the_verdict_event(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("trading.alphasearch.evaluate.load_factors",
                        lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr("trading.alphasearch.robustness.run_battery",
                        lambda *a, **k: _fake_battery_outcome(eligible=True))
    rc = cli.main(["alphasearch", "robustness", "amihud:midcap",
                   "--journal-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "battery"
    assert payload["eligible"] is True
    assert payload["signal"] == "amihud"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_alphasearch_cli.py -q -k robustness`
Expected: FAIL — `argparse` rejects the `robustness` action (`invalid choice`).

- [ ] **Step 3: Wire the action into `src/trading/cli.py`**

Parser changes in `build_parser()` (the alphasearch block):

```python
    alphasearch.add_argument(
        "action", choices=["sweep", "leaderboard", "holdout", "robustness"]
    )
    alphasearch.add_argument(
        "trial",
        nargs="?",
        default=None,
        help="holdout/robustness target as signal:universe (e.g. mom126:largecap)",
    )
```

In `_cmd_alphasearch`, extract the holdout branch's universe resolution into a helper (place it after `_cmd_alphasearch`) and use it for BOTH actions. Replace the holdout branch's resolution block (from `signal_name, _, universe = args.trial.partition(":")` through the second `unknown universe` check) with calls to:

```python
def _resolve_alphasearch_universe(universe: str):
    """UniverseSpec for a flat pool or (flag-free) segment universe name, or
    None with the error already printed -- shared by holdout and robustness
    (both resolve journal-derived targets; a missing sic_map only errors when
    a segment name actually needs it)."""
    from trading.alphasearch import sweep as engine

    universes = engine.default_universes(Path("."))
    if universe not in universes:
        from trading.alphasearch.segments import segment_universes

        try:
            seg_universes, _excluded = segment_universes(Path("."))
        except engine.SweepError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return None
        universes = {**universes, **seg_universes}
    if universe not in universes:
        print(
            f"ERROR: unknown universe {universe!r}; choose from "
            f"{', '.join(sorted(universes))}",
            file=sys.stderr,
        )
        return None
    return universes[universe]
```

(`sys`, `json`, and `Path` are already imported at the top of `cli.py`; the helper needs no new module-level imports.)

The robustness branch, inserted in `_cmd_alphasearch` after the `sweep` branch and before the holdout code:

```python
    if args.action == "robustness":
        if not args.trial or ":" not in args.trial:
            print(
                "ERROR: robustness needs a trial id: signal:universe "
                "(e.g. amihud:midcap)",
                file=sys.stderr,
            )
            return 1
        signal_name, _, universe = args.trial.partition(":")
        uspec = _resolve_alphasearch_universe(universe)
        if uspec is None:
            return 1
        factors = _load_alphasearch_factors(args)
        if factors is None:
            return 1
        from trading.alphasearch import robustness

        try:
            outcome = robustness.run_battery(
                uspec, journal, factors, _utcnow().isoformat(), signal_name
            )
        except (engine.SweepError, PanelError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(outcome.event))
            return 0
        _render_battery(outcome)
        return 0
```

The holdout branch keeps its own `if not args.trial or ":" not in args.trial:` guard (message unchanged — the existing test pins it) and swaps its resolution block for `uspec = _resolve_alphasearch_universe(universe)` / `if uspec is None: return 1`, then passes `uspec` (not `universes[universe]`) to `engine.run_holdout`.

The renderer (place next to `_print_alphasearch_leaderboard`):

```python
def _render_battery(outcome) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console() if sys.stdout.isatty() else Console(width=200)

    def num(value, fmt="{:+.2f}"):
        return "-" if value is None else fmt.format(value)

    def summary(check) -> str:
        d = check.detail
        if check.name == "sub_period_halves":
            return "; ".join(
                f"{h['window']}: a={num(h['alpha_annual_pct'], '{:+.1f}')} "
                f"t={num(h['alpha_t'])}" for h in d["halves"]
            )
        if check.name == "universe_subsets":
            return f"{d['n_pass']}/5 draws sign-matched (need >=4)"
        if check.name == "parameter_jitter":
            ok = sum(1 for t in d["trials"] if t["passed"])
            return f"{ok}/4 jitter trials sign-matched (need 4)"
        if check.name == "decision_offset":
            return (f"a={num(d['alpha_annual_pct'], '{:+.1f}')} "
                    f"retention={num(d['retention'], '{:.2f}')} (need >=0.50)")
        if check.name == "name_concentration":
            return (f"excl {', '.join(d['excluded'])}: "
                    f"a={num(d['alpha_annual_pct'], '{:+.1f}')} "
                    f"retention={num(d['retention'], '{:.2f}')} (need >=0.50)")
        if check.name == "month_concentration":
            return f"top-3 months share={num(d['top3_share'], '{:.0%}')} (need <=60%)"
        return ""

    table = Table(title=(
        f"robustness battery — {outcome.signal}:{outcome.universe} "
        f"over {outcome.window}"
    ))
    for col in ["#", "check", "result", "numbers"]:
        table.add_column(col)
    for check in outcome.checks:
        table.add_row(str(check.number), check.name,
                      "PASS" if check.passed else "FAIL", summary(check))
    console.print(table)

    cost = Table(title="cost-adjusted alpha (one-way bps, both legs charged)")
    for col in ["cost", "4F a%/yr", "t"]:
        cost.add_column(col, justify="right")
    for row in outcome.cost_table:
        cost.add_row(f"{row['cost_bps']}bp", num(row.get("alpha_annual_pct"), "{:+.1f}"),
                     num(row.get("alpha_t")))
    console.print(cost)

    capacity = Table(title="Amihud-implied capacity curve (first-order model)")
    for col in ["book/side", "4F a%/yr", "t", "no-λ names"]:
        capacity.add_column(col, justify="right")
    for row in outcome.capacity_curve:
        capacity.add_row(f"${row['book_usd']:,.0f}",
                         num(row.get("alpha_annual_pct"), "{:+.1f}"),
                         num(row.get("alpha_t")),
                         str(row.get("skipped_no_lambda", 0)))
    console.print(capacity)

    if outcome.factor_proxy.get("flagged"):
        offenders = ", ".join(
            f"{name} (t={t:+.1f})"
            for name, t in outcome.factor_proxy["offenders"].items()
        )
        console.print(
            f"[bold red]FACTOR-PROXY WARNING: {offenders} loads harder than "
            f"the alpha (|t_loading| > 2x|t_alpha|, R²="
            f"{outcome.factor_proxy['r2']:.2f} > 0.5) — the §9 SMB-costume "
            f"pattern. Does not block, but read the alpha as a factor bet "
            f"until proven otherwise.[/bold red]"
        )
    console.print(
        f"holdout-eligible: {'YES' if outcome.eligible else 'NO'} "
        f"(checks 1-6 all pass AND 30bp cost t >= 2.0)"
    )
```

- [ ] **Step 4: Run the CLI tests**

Run: `uv run pytest tests/test_alphasearch_cli.py -q`
Expected: all PASS (including the pre-existing holdout CLI tests over the refactored resolution helper).

- [ ] **Step 5: Run the full suite + ruff**

Run: `uv run pytest -q && uv run ruff check src tests scripts`
Expected: all PASS, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/trading/cli.py tests/test_alphasearch_cli.py
git commit -m "Add alphasearch robustness CLI action with report card [AI]"
```

---

### Task 7: Golden battery test on real fixture files + glossary docs

**Files:**
- Create: `tests/test_alphasearch_robustness_golden.py`
- Modify: `docs/glossary.md` (append a new section at the end)
- Test: the golden file itself

**Interfaces:**
- Consumes: `run_battery`, `run_sweep`, `battery_verdict`, `discovery_trials`; the real-files pattern from `tests/test_alphasearch_golden.py::_write_universe` (parquet caches + explicit-symbols `UniverseSpec`, no `panel_factory` injection).

- [ ] **Step 1: Write the golden test (failing only if earlier tasks are broken — it is the integration gate)**

Create `tests/test_alphasearch_robustness_golden.py`:

```python
"""Golden end-to-end battery (spec section 7): real files -> build_panel ->
sweep -> run_battery -> verdict, deterministic fixture, no factory injection."""

from __future__ import annotations

import pandas as pd

from alphasearch_helpers import make_factors, make_panel
from trading.alphasearch.robustness import run_battery
from trading.alphasearch.spec import SIGNALS
from trading.alphasearch.sweep import (
    UniverseSpec,
    battery_verdict,
    discovery_trials,
    run_sweep,
    trials_journal,
)

WINDOW = "2020-01-01..2020-06-30"


def _write_universe(tmp_path) -> UniverseSpec:
    """make_panel(40)'s closes as real parquet caches + an explicit symbols
    tuple (the deep-pool universe shape): 40 names so the half-universe
    draws (20) clear MIN_NAMES 15 and check 2 genuinely runs."""
    panel = make_panel(n_symbols=40)
    cache = tmp_path / "cache"
    cache.mkdir()
    for sym in panel.symbols:
        closes = panel.closes[sym]
        pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes,
             "volume": 1000.0},
            index=closes.index,
        ).to_parquet(cache / f"{sym}.parquet")
    return UniverseSpec("largecap:golden", cache, None, None,
                        symbols=panel.symbols)


def test_golden_battery_end_to_end(tmp_path):
    uspec = _write_universe(tmp_path)
    journal = trials_journal(tmp_path / "journal")
    factors = make_factors()
    run_sweep({uspec.name: uspec}, journal, factors, ts="t0",
              signals={"mom21": SIGNALS["mom21"]}, window=WINDOW)
    assert len(discovery_trials(journal)) == 1

    outcome = run_battery(uspec, journal, factors, "t1", "mom21",
                          discovery_window=WINDOW)

    # 12 battery-tagged, BH-counted discovery trials on top of the sweep's 1.
    trials = discovery_trials(journal)
    assert len(trials) == 13
    assert sum(1 for t in trials
               if t.get("battery") == "mom21:largecap:golden") == 12
    # Every check computed with numbers; the report is fully populated.
    assert [c.number for c in outcome.checks] == [1, 2, 3, 4, 5, 6]
    assert all(isinstance(c.passed, bool) for c in outcome.checks)
    subsets = next(c for c in outcome.checks if c.name == "universe_subsets")
    assert subsets.passed                     # 20-name draws genuinely ran
    concentration = next(c for c in outcome.checks
                         if c.name == "name_concentration")
    assert concentration.passed               # linear drift spread is broad
    # Costs monotonically eat alpha; capacity is populated (volume=1000
    # yields real lambdas on this fixture).
    alphas = [r["alpha_annual_pct"] for r in outcome.cost_table]
    assert alphas[0] > alphas[1] > alphas[2]
    assert all(r["alpha_t"] is not None for r in outcome.capacity_curve)
    # Verdict journaled under the discovery config's hash.
    verdict = battery_verdict(journal, outcome.event["config_hash"])
    assert verdict is not None and verdict["eligible"] == outcome.eligible

    # Bit-identical, count-stable re-run: dedupe by config hash everywhere.
    again = run_battery(uspec, journal, factors, "t2", "mom21",
                        discovery_window=WINDOW)
    assert len(discovery_trials(journal)) == 13
    assert again.eligible == outcome.eligible
    assert [c.passed for c in again.checks] == [c.passed for c in outcome.checks]
    verdict2 = battery_verdict(journal, outcome.event["config_hash"])
    assert verdict2["ts"] == "t2"             # replaced in place
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_alphasearch_robustness_golden.py -q`
Expected: PASS. If `subsets.passed` or `concentration.passed` fails, debug with `-x --tb=long` — the fixture's drift spread is engineered to be broad and stable, so a failure here means a real composition bug (e.g. subset params not reaching the sort), not a threshold problem.

- [ ] **Step 3: Append the glossary section to `docs/glossary.md`**

Append at the end of the file:

```markdown
## The robustness battery (Piece 3)

- **Robustness battery** — the pre-registered, frozen set of seven checks a
  BH survivor must face before it may spend its once-only holdout touch
  (`trading alphasearch robustness <signal>:<universe>`). Pre-committed
  interrogation instead of ad-hoc survivor-poking: the checks and thresholds
  were written down before any survivor was examined, so passing them can't
  be the product of tweaking the exam after seeing the student.
- **Sub-period halves (check 1)** — re-run the discovery evaluation on each
  half of the window (2019-01..2021-06 / 2021-07..2023-12). A real effect
  shows the same sign in both halves with |t| ≥ 1; a one-regime wonder
  doesn't.
- **Universe-subset draws (check 2)** — five seeded random half-universes
  (seed 42+i, sorted draws). An alpha carried by the breadth of the
  cross-section survives ≥ 4 of 5; one carried by a few lucky names doesn't.
- **Parameter jitter (check 3)** — the same evaluation at quantiles {4, 6} ×
  min_names {10, 20}. Real effects don't care exactly where the bucket
  boundaries fall.
- **Decision-date offset (check 4)** — rebalance on the second trading
  session of each month instead of the first. Guards against calendar-turn
  artifacts; the alpha must keep its sign and half its magnitude.
- **Name concentration (check 5)** — recompute the L/S series with the top-3
  contributors to the long leg removed. A "three-name alpha" collapses; a
  broad one retains ≥ half its point estimate.
- **Month concentration (check 6)** — the top-3 calendar months' share of
  the cumulative L/S log return must be ≤ 60%; otherwise the "alpha" is a
  couple of episodes, not a process.
- **Factor-proxy flag (check 7)** — warning-only: any factor loading with
  |t| more than twice the alpha's |t| while regression R² > 0.5. The §9
  SMB-costume detector — the series is mostly a factor bet wearing an alpha
  costume.
- **Cost-adjusted alpha** — the L/S series re-regressed after charging
  parametric one-way costs (10/30/50 bps × turnover × both legs) at every
  rebalance. The promotion rule requires t ≥ 2.0 at 30 bps.
- **Amihud λ (price impact)** — |daily return| / dollar volume, averaged
  over 252 days: the price move a dollar of trading buys. The amihud
  signal's own construction, reused as an impact price.
- **Capacity curve** — net alpha at book sizes $10k/$100k/$1M per side,
  charging each rebalanced name its own λ × (book / names-per-leg) on entry
  and exit. First-order model — an honest sketch of how fast paper alpha
  drowns in impact, not a fill simulator.
- **Holdout-eligible** — the battery's verdict (checks 1-6 pass AND the
  30 bps cost row keeps t ≥ 2.0), journaled as one `kind="battery"` event
  per candidate. The holdout command refuses candidates without it — a
  written prospective amendment to the Piece 1 holdout protocol.
```

- [ ] **Step 4: Run the full suite + ruff**

Run: `uv run pytest -q && uv run ruff check src tests scripts`
Expected: all PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add tests/test_alphasearch_robustness_golden.py docs/glossary.md
git commit -m "Add golden battery test and robustness glossary entries [AI]"
```

---

### Task 8: Pilot — run the battery on the three amihud survivors, record in experiments.md §11

This is the spec §8 acceptance run. It spends NO holdout touches regardless of verdicts (the park stands until the developer says otherwise). It DOES append real trials to the committed journal — that is the design (battery trials are BH-counted), so the journal diff is part of the deliverable.

- [ ] **Step 1: Pre-flight**

```bash
uv run pytest -q && uv run ruff check src tests scripts
uv run trading alphasearch leaderboard 2>&1 | tee /tmp/claude-piece3-leaderboard-before.log
```

Expected: suite green; leaderboard shows 799 discovery trials with exactly the three amihud BH survivors (midcap, opt-midcap:trade, midcap:trade). If the counts differ, STOP and consult the developer — the journal moved since this plan was written.

- [ ] **Step 2: Run the three batteries (order: largest universe first)**

```bash
uv run trading alphasearch robustness amihud:midcap 2>&1 | tee /tmp/claude-piece3-battery-midcap.log
uv run trading alphasearch robustness amihud:midcap:trade 2>&1 | tee /tmp/claude-piece3-battery-midcap-trade.log
uv run trading alphasearch robustness amihud:opt-midcap:trade 2>&1 | tee /tmp/claude-piece3-battery-opt-midcap-trade.log
```

Also capture the machine-readable verdicts:

```bash
uv run trading alphasearch robustness amihud:midcap --json 2>/dev/null | tee /tmp/claude-piece3-midcap.json
uv run trading alphasearch robustness amihud:midcap:trade --json 2>/dev/null | tee /tmp/claude-piece3-midcap-trade.json
uv run trading alphasearch robustness amihud:opt-midcap:trade --json 2>/dev/null | tee /tmp/claude-piece3-opt-midcap-trade.json
```

Notes:
- The `--json` re-runs replace each verdict in place and dedupe every re-evaluation by config hash — the trial count must NOT change between the table run and the json run (verify on the leaderboard afterwards).
- Expected structural outcome: `opt-midcap:trade` (26 names) fails check 2 outright — its 13-name half-draws are below `MIN_NAMES` 15, so all five draws journal `SortError` error trials. Record it; do not touch thresholds.
- If a run refuses with "not a current BH survivor", the BH frontier moved as battery trials raised the trial count — a real, honest interaction. STOP and consult the developer before re-running anything.

- [ ] **Step 3: Post-flight leaderboard**

```bash
uv run trading alphasearch leaderboard 2>&1 | tee /tmp/claude-piece3-leaderboard-after.log
```

Record the new honest trial count (799 + the journaled battery re-evaluations; up to 36 if no config collided) and whether the three amihud rows still pass BH.

- [ ] **Step 4: Record the pilot in `docs/experiments.md` §11**

Append a subsection at the end of §11 (before "## Known caveats"). Use the MEASURED numbers from the logs — every cell below is filled from the run, none invented:

```markdown
### Piece 3 pilot: the robustness battery on the amihud family (2026-07-XX)

**Ran with the frozen battery (design spec 2026-07-09 §3); trial count
NNN → MMM (the battery's re-evaluations are BH-counted discovery trials).
No holdout touched.**

| check (pass rule) | midcap | midcap:trade | opt-midcap:trade |
|---|---|---|---|
| 1 sub-period halves (sign + \|t\|≥1 both) | ... | ... | ... |
| 2 universe subsets (≥4/5 sign) | ... | ... | ... |
| 3 parameter jitter (4/4 sign) | ... | ... | ... |
| 4 decision offset (sign + ≥0.5×) | ... | ... | ... |
| 5 name concentration (≥0.5× excl top-3) | ... | ... | ... |
| 6 month concentration (top-3 ≤60%) | ... | ... | ... |
| 7 factor-proxy flag (warning) | ... | ... | ... |
| cost table 10/30/50bp (t) | ... | ... | ... |
| capacity $10k/$100k/$1M (α%/yr) | ... | ... | ... |
| **holdout-eligible** | ... | ... | ... |

[2-4 sentences of interpretation: which §9 failure modes the battery did or
did not find, what the cost/capacity numbers say about the park decision.]

**Park decision update:** [restate or revise the amihud park with this
evidence — the park stands unless the developer explicitly lifts it.]
```

Fill the date, counts, per-cell PASS/FAIL + key numbers (halves' t's, subset tallies, retention ratios, month shares, cost-row t's, capacity alphas), and write the interpretation from the actual results. For `opt-midcap:trade`'s check 2, note the structural cause (26-name universe → 13-name draws < 15 minimum) explicitly.

- [ ] **Step 5: Verify the working tree tells the whole story, then commit**

```bash
git status
git diff --stat
uv run pytest -q
git add journal/alphasearch-trials.jsonl docs/experiments.md
git commit -m "Run Piece 3 battery pilot on the amihud survivors [AI]"
```

Expected in the diff: journal appends only (no deletions — `git diff journal/ | grep '^-' | grep -v '^---'` must be empty), plus the experiments.md subsection.

---

## Execution notes for the orchestrator

- Tasks 1→5 are strictly ordered (each consumes the previous task's interfaces). Task 6 depends on 5; Task 7 on 6 (glossary references the CLI verb); Task 8 is the acceptance run and must be last, on a green suite.
- Task 8 mutates the committed journal — run it once, deliberately, from a clean tree.
- If any FROZEN number in this plan disagrees with `docs/superpowers/specs/2026-07-09-robustness-battery-design.md`, the spec wins and the discrepancy must be reported to the developer, not silently fixed.
