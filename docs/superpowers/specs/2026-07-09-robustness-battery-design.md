# Piece 3 — Robustness & Failure-Analysis Battery (design spec)

**Status:** approved design, 2026-07-09. Child of the program charter; builds
on Pieces 1-2 and the Tier-1 batch (registry at 37, journal at 799 discovery
trials, three parked amihud BH survivors).
**Scope:** the pre-registered robustness battery, cost/capacity analysis, the
holdout-eligibility promotion rule, one CLI verb, and the amihud pilot run.
Out of scope: the charter's failure-mode taxonomy engine and automated guard
feedback (build after batteries have run on real survivors); the full
backtester bridge (deferred as the final pre-live step, per brainstorm).

## 1. Purpose

Convert "BH survivor" into "evidence-graded candidate or documented kill"
WITHOUT touching the once-only holdout. Encode the program's learned failure
modes (the §9 SMB-costume, one-regime wonders, three-name alphas, paper-only
illiquidity premia) as frozen, repeatable checks — pre-committed
interrogation instead of ad-hoc survivor-poking.

## 2. Decisions made during brainstorming (locked)

| Question | Decision |
|---|---|
| Battery evaluations vs the trial journal | **BH-counted discovery trials, tagged** with a `battery` field (Piece 1 §5.6 honored — no second ledger, no trial-hiding); the leaderboard may group them |
| Cost machinery | **Cost-adjusted alpha (parametric bps) + Amihud-implied capacity curve** inside the factor-alpha framework; NO backtester wrapper (the §9 long-only-beta trap) |
| Eligibility gate | Battery must pass before a survivor may spend a holdout touch |

## 3. The frozen battery (pre-registered; amend only in writing, prospectively)

Run as `trading alphasearch robustness <signal>:<universe>`; refuses any
target that is not a current BH survivor (mirrors the holdout gate; the
refusal journals nothing). Checks 1-4 are re-evaluations journaled as tagged
discovery trials (config-hash dedupe applies as everywhere); checks 5-7 are
arithmetic on already-computed series and journal no new trials.

| # | check | construction | pass rule (frozen) |
|---|---|---|---|
| 1 | Sub-period halves | discovery split 2019-01-01..2021-06-30 and 2021-07-01..2023-12-31 | both halves: alpha sign matches full-window sign AND \|t\| ≥ 1.0 |
| 2 | Universe subsets | 5 seeded (seed=42+i) random half-universe draws, same params | ≥ 4 of 5 draws: alpha sign matches |
| 3 | Parameter jitter | quantiles ∈ {4, 6}; min_names ∈ {10, 20} (4 trials) | all 4: alpha sign matches |
| 4 | Decision-date offset | rebalance on the 2nd trading session of each month | sign matches AND \|alpha\| ≥ 0.5 × full-window \|alpha\| |
| 5 | Name concentration | recompute the L/S daily series excluding the top-3 names by cumulative contribution to the top-quantile leg | remaining alpha point estimate ≥ 0.5 × original |
| 6 | Month concentration | top-3 calendar months' share of the cumulative L/S log return | ≤ 60% |
| 7 | Factor-proxy flag | any factor loading with \|t_loading\| > 2 × \|t_alpha\| while regression R² > 0.5 | WARNING only (printed prominently, does not block) — the §9 SMB-costume detector |

**Promotion rule (frozen):** a survivor is **holdout-eligible** iff checks
1-6 all pass AND the 30 bps row of the cost table (§4) retains t ≥ 2.0. The
battery verdict (per-check numbers + eligible yes/no) is journaled as one
`kind="battery"` event per (signal, universe) — re-runs replace by config
hash like everything else. The holdout command adds battery-passed to its
pre-checks (a written prospective amendment to Piece 1 §3.6: no holdout may
be spent on a survivor that has not passed its battery; no holdout has ever
been spent, so nothing is affected retroactively).

## 4. Cost & capacity analysis (arithmetic; no new trials)

- **Cost-adjusted alpha table:** one-way costs c ∈ {10, 30, 50} bps. Each
  rebalance charges turnover × c to each leg of the L/S series (long and
  short sides both trade); the charged series is re-regressed and the table
  reports alpha/t per c. Uses the same turnover measurement the leaderboard
  already reports.
- **Amihud capacity curve:** for book sizes B ∈ {$10k, $100k, $1M} per side,
  each rebalanced name's own Amihud ratio λ (|ret|/dollar-volume, the
  signal's existing 252d construction) prices its impact as λ × (B / names
  per leg) charged on entry and exit; the curve reports net alpha per B.
  This is a first-order impact model — documented as such, honest about
  being a model, chosen because for an ILLIQUIDITY signal the names' own λ
  is the most self-consistent impact estimate available in EOD data.

## 5. Architecture

`src/trading/alphasearch/robustness.py` — pure composition of existing
machinery: `evaluate_trial`/`run_sweep` internals for re-evaluations (with
explicit windows/params/universe subsets), `sort.portfolio_sort` outputs for
the arithmetic checks, `evaluate.run_regression` for cost-adjusted
re-estimates, `panel` accessors for λ. New sweep.py surface: a `battery`
tag stored on the journal EVENT (display/grouping only — deliberately NOT
part of the hashed trial config, so an identical evaluation made inside or
outside a battery is one trial, never two), the `kind="battery"` verdict
event, and the holdout pre-check extension. CLI: `robustness` action on the
existing alphasearch subcommand, report-card output (rich table + the
factor-proxy warning in red), `--json` for machines.

Universe-subset and offset re-evaluations need small, explicit extensions:
`portfolio_sort`/`evaluate_trial` accept an optional symbol-subset and a
decision-calendar offset (both threaded into the hashed params — they change
trial outcomes, so they MUST enter the config hash; the default values hash
identically to existing trials).

## 6. Error handling

Non-survivor target → refused, listing current survivors. A battery
re-evaluation that errors (SortError etc.) journals the error trial as usual
and FAILS that check (an uncomputable perturbation is not a pass). Missing
panel data for a subset draw → that draw fails, others proceed. The battery
never mutates the holdout state.

## 7. Testing

Hand-computable fixtures per check: a planted two-regime signal (strong
2019-21, dead 2021-23) fails check 1; a three-name-driven fixture fails
check 5; a single-month-spike fixture fails check 6; a deliberate
SMB-costume fixture (scores ∝ size) trips flag 7; cost arithmetic verified
by hand on a tiny series; capacity curve on a two-name fixture with known λ.
Journal honesty: battery trials BH-counted + tagged + deduped; battery event
replaces on re-run; holdout refuses battery-less survivors (new pre-check
test). Refusal of non-survivors. Full suite warnings-as-errors; ruff clean;
subagent implementation + adversarial review with the standing science lens.

## 8. Pilot (acceptance)

Run the battery on all three amihud survivors (midcap, opt-midcap:trade,
midcap:trade). Record per-check numbers, cost table, capacity curve, and
verdicts in experiments.md §11; update the park decision with evidence. The
pilot spends no holdout touches regardless of verdicts (the park stands
until the developer says otherwise).

## Amendments

- **2026-07-10 (R1, `2026-07-10-longonly-gate-amendment.md`):** amends §3's
  frozen promotion rule prospectively. Checks 1-7 and their frozen
  thresholds are UNTOUCHED. What changes is the final eligibility gate: "the
  30 bps row of the cost table retains t ≥ 2.0" is superseded by "the
  Corwin-Schultz (2012) spread-charged long-only series' annualized Sharpe
  over the discovery window is ≥ SPY buy-and-hold's Sharpe over the
  identical window AND its total return exceeds SPY's total return" — a
  survivor is holdout-eligible iff checks 1-6 all pass AND this long-only-
  vs-SPY comparison passes. §4's cost-adjusted alpha table and capacity
  curve are retained exactly as specified, now DIAGNOSTIC only (no longer
  read by the eligibility gate). §5's holdout pre-check extension is
  unaffected in mechanism — it still reads the verdict's `eligible` bit —
  only what that bit MEANS changed. New machinery lives in
  `trading.alphasearch.costs` (a leaf module: the spread estimator,
  per-name rebalance charging, and the SPY benchmark loader), imported by
  both `robustness.py` (the battery gate) and `sweep.py` (the leaderboard's
  `--long-only` view, which re-reads every journaled trial under the new
  lens without re-journaling anything).
