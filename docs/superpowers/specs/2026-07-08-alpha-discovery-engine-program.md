# Alpha-Discovery Engine — Program Charter

**Status:** high-level program spec (charter). Written 2026-07-08.
**Audience:** a fresh agent (and its subagents) picking this up with NONE of the
originating conversation in context. Read this end-to-end first.

## 0. What this document is and how to use it

This is a **program charter**, not a single implementation spec. It defines a
multi-piece system for discovering statistically-real trading signals, the
scientific discipline that makes it trustworthy, the existing code/data it builds
on, and a decomposition into **four independently-buildable pieces**.

**How to execute it:** each of the four pieces gets its OWN full
`brainstorming → writing-plans → implementation` cycle, in order (they depend on
each other). Drive the work through **subagents** to protect the top-level context
window — the top-level agent orchestrates and reviews; subagents do context-heavy
exploration, implementation, and adversarial review. Follow the repo's existing
discipline: fresh-implementer + adversarial-reviewer subagents, tests with
warnings-as-errors, ruff clean, granular `[AI]`-tagged commits, no merge of code
you don't understand.

## 1. Why we're building this (the lessons that shaped it)

We spent a long research arc trying to find a tradeable edge in US equities —
first cross-sectional momentum, then an options implied-volatility program
(skew, "excitement"/risk-reversal, option illiquidity), across large-cap
(S&P 500 + NDX) and mid-cap (S&P 400) universes. **Every apparent winner
collapsed under scrutiny.** The durable lessons — this system exists to
operationalize them:

1. **"Beats SPY's Sharpe" is a hurdle, not proof of skill.** A long-only book is
   mostly market *beta*; in a bull market almost anything that holds stocks clears
   the bar. Our best-looking backtest (a skew strategy at 0.79 OOS Sharpe vs SPY
   0.70) had a **four-factor alpha t-stat of 0.88 — indistinguishable from zero.**
   It was market + size + momentum beta wearing a signal's costume.
2. **Every "edge" we found decomposed into known factor exposure.** The one real
   cross-sectional predictor (mid-cap option illiquidity) turned out to be a noisy
   proxy for the **size factor** (SMB loading t=13.1, alpha ~0); a plain size sort
   beat it. See `docs/experiments.md` §9 and the glossary.
3. **The only trustworthy measure is factor-adjusted alpha with a t-stat**, not
   Sharpe-vs-benchmark. We built the tool for it (see §4).
4. **Overly-analyzed universes (S&P 500) are near-efficient** — an edge that works
   "across the board" is unlikely. Edges, if they exist, are more plausibly
   *conditional* (a specific segment, regime, or signal interaction).
5. **Running many trials is how you manufacture false positives.** This is THE
   central risk of the very approach below, and the system's discipline must be
   built to neutralize it (see §3).

**The thesis that follows:** stop hunting one universal signal. Build a fast,
disciplined **discovery engine** that measures *true* (factor-adjusted) alpha
across many signal permutations and many universe segments, with rigorous
overfitting control, and assembles the survivors into a **diversified portfolio
of segment-specific strategies.** The value is the *engine and its discipline*,
not any single signal.

## 2. Success criteria

- Given a signal (or signal combination) and a universe, the engine returns a
  **factor-adjusted alpha estimate with a t-stat in seconds** — cheap enough to
  sweep thousands of permutations × segments.
- The engine **honestly accounts for the number of trials run** and only surfaces
  candidates that survive a multiple-testing-aware bar AND a reserved,
  touched-once holdout.
- A surfaced candidate comes with its factor decomposition, robustness profile,
  and a reproducible record — so "is this real?" is answered by the system, not by
  hopeful eyeballing.
- **A null result across a sweep is a valid, first-class outcome** — the system is
  as valuable for cheaply *killing* bad ideas as for finding good ones.

## 3. Non-negotiable methodology (the scientific spine)

This is the part that must not be compromised for speed or convenience. Every
piece must respect it.

- **Measure alpha, not benchmark-beating.** The unit of evaluation is the
  intercept of a **Carhart four-factor regression** (market, size, value,
  momentum) on the signal's excess-return series, with its t-stat. Print CAPM
  alpha alongside four-factor alpha — the gap reveals hidden factor tilts.
- **Cheap return series, real alpha.** You do NOT need a full backtest to get
  alpha. Build the signal's **long/short (or long-only) portfolio return series**
  by a cheap cross-sectional sort (seconds), then factor-regress it. Full,
  cost-and-risk-modeled backtests are reserved for a small set of survivors.
- **Multiple-testing discipline is built into the search, not bolted on.**
  - Track an honest **trial count** (every signal × universe × parameter setting
    is a test). Persist it — you cannot fix multiple-testing after the fact.
  - Apply a trials-aware bar: **Benjamini-Hochberg false-discovery-rate** control
    and/or the **Deflated Sharpe Ratio / Probability of Backtest Overfitting**
    (Bailey & López de Prado), which are purpose-built for "I ran N trials; how
    much do I discount the best?".
  - Prefer **out-of-sample by construction** (the cheap sweep runs on a discovery
    window; survivors must re-prove on a **reserved holdout touched exactly once**,
    journal-enforced like the existing go-live holdout).
- **Robustness over point-significance.** A real signal survives small
  perturbations — horizon, universe subset, sub-period, parameter jitter. A
  "winner" that works only at one exact setting is the fingerprint of overfitting
  and must be flagged as fragile.
- **Pre-register the rules; no garden-of-forking-paths.** Decide the sort method,
  horizons, factor set, and significance bar BEFORE the sweep. "Digging into
  survivors" is legitimate only under pre-committed rules; ad-hoc re-slicing of
  the survivor set re-introduces the bias the holdout was meant to remove.
- **Point-in-time / no look-ahead, always.** Every signal value uses only data
  available at the decision date; the repo's existing PIT discipline
  (`FeaturePanel`, the fundamentals/skew stores' as-of gather) is the model.
- **Survivorship-free data only** (the repo's Tiingo caches are delisted-inclusive).

## 4. Existing infrastructure to build on (exact pointers)

Do NOT rebuild these — the engine composes them.

- **`scripts/signal_scan.py`** — cross-sectional signal scanner: for a panel of
  candidate metrics, computes the information coefficient (mean monthly Spearman
  of metric vs forward return) among a filtered pool, with t-stats. The seed of
  the "cheap sweep." Factored core (`information_coefficient`, `survivors`,
  `_cell_metrics`) is unit-tested.
- **`scripts/factor_regression.py`** — the alpha tool: hand-rolled OLS, fetches +
  caches Ken French's canonical daily Fama-French factors (Mkt-RF, SMB, HML, RF)
  + momentum under `data/factors/`; regresses a return series and reports alpha
  (annualized, t-stat) + factor loadings + R², CAPM vs four-factor. `ols(X,y)` is
  a clean reusable primitive.
- **Backtest / walk-forward engine** — `src/trading/backtest/` (`engine.py`,
  `walkforward.py`, `experiments.py`). `trading backtest --venue equities
  --config-dir <cfg> --walk-forward [--dump-returns PATH]`; `--dump-returns`
  writes the stitched OOS daily strategy+benchmark returns for the factor tool.
  Reserved for survivors (minutes per run; ~9 min / 99 names, ~13 / 150).
- **Ranker registry** — `src/trading/signals/registry.py`: `RankerSpec(fn,
  requires_fundamentals, requires_skew)`; a ranker is `fn(bars, as_of, config,
  fundamentals=None, *, skew=None, panel=None) -> DataFrame` indexed by symbol
  with feature columns + `composite` + `raw_return_30d`. This is the plug-in point
  for a signal that graduates to a full backtest.
- **Signal channels** — `src/trading/signals/`: `engine.py` (`FeaturePanel`,
  vectorized PIT price features), `skew.py` (options skew/illiquidity channel +
  rankers), `quality.py`/`value.py` (fundamentals rankers). Existing seed signals:
  momentum, skew, risk-reversal/"excitement", option illiquidity, quality (gross
  profitability), value (earnings yield, book-to-market), plus the metric panel in
  `signal_scan.py` (IV level, smile, put/call flow, realized vol, VRP, distance
  from high, reversal, several momentum windows).
- **Universes** — `src/trading/venues/universes/equities_membership.csv` (PIT
  S&P 500 / NDX / S&P 400 membership; `build_universe(..., indices=...)` in
  `options_gather.py` ranks by dollar volume). `UniverseConfig.symbols_allowlist_path`
  restricts a run to an explicit name set. Sector/industry segmentation for Piece 2
  needs a new classification source (see open questions).
- **Data** — survivorship-free Tiingo adjusted bars: `data/equities-tiingo/`
  (713 large-cap names) and `data/equities-midcap-tiingo/` (705 mid-cap;
  `.source` marker required — copy from the large-cap cache when moving it).
  Options-IV cells: `data/options-iv/samples*.jsonl` (gathered via
  `scripts/gather_options_iv.py` against a local ThetaData terminal). Fundamentals
  store (SEC EDGAR/companyfacts) under the configured `fundamentals_dir`.
- **Anti-overfitting ledger** — `journal/experiments-*.jsonl` (experiment counting)
  and `docs/experiments.md` (human-readable results log). `docs/glossary.md` defines
  all the terms.

## 5. The four pieces (decomposition)

Build in order; each is its own spec → plan → build cycle.

### Piece 1 — Core alpha-search engine  *(build first; everything depends on it)*
**Purpose:** turn "signal + universe" into "factor-adjusted alpha + t-stat" in
seconds, and sweep many of them into a multiple-testing-aware leaderboard.
**Likely components:**
- A **signal specification** — a uniform way to define a candidate signal (a
  function/config that maps `(bars/options/fundamentals, as_of, universe) → a
  per-symbol score`, PIT). Seed it by wrapping the existing signals/metrics.
- A **portfolio-sort return generator** — from a signal's scores, build the cheap
  long/short (top-minus-bottom quantile) and long-only return series over the
  discovery window (extend the `signal_scan.py` sort logic; return a dated series).
- An **alpha evaluator** — factor-regress that series (reuse
  `factor_regression.ols` + the cached FF factors); return alpha, t-stat,
  loadings, R², CAPM-vs-4-factor gap.
- A **sweep runner + leaderboard** — run a set of signals (and later,
  combinations/parameters) × a universe, collect results, apply the
  **trials-aware significance bar** (FDR / deflated-Sharpe), and persist a ranked
  record with the honest trial count.
- A **trial journal** — every evaluation logged (signal id, universe, window,
  params, alpha, t) so the multiple-testing correction is honest and the sweep is
  reproducible.
**Key design questions (resolve in its spec):** long/short vs long-only (or both)
for the return series; quantile depth; which factor set is canonical; the exact
multiple-testing method and threshold; discovery-window vs holdout split;
how signals are registered/enumerated; serial vs parallel sweep.

### Piece 2 — Segmented universes
**Purpose:** run the engine across market segments (S&P 500, mid-cap, sectors,
industries like biotech/banks) so a signal dead in aggregate can be found where it
lives. Produces per-segment leaderboards.
**Depends on:** Piece 1. **Key considerations:** a PIT sector/industry
classification source (GICS/SIC — need data; SEC filings carry SIC codes, or a
vendor); avoiding *segment* multiple-testing (many segments × many signals
multiplies trials — the trial count and FDR must span the whole sweep, not reset
per segment); minimum names per segment for a meaningful cross-section.

### Piece 3 — Robustness & failure-analysis automation
**Purpose:** automate the "is a survivor real or lucky/fragile?" interrogation —
perturbation gates (horizon, sub-period, universe subset, parameter jitter), the
reserved-holdout re-prove, and diagnostics that characterize *how* a false positive
arose (e.g. factor-loading drift, concentration in a few names/months, look-ahead
leaks). Feeds guards back into the engine.
**Depends on:** Piece 1 (and benefits from 2). **Key considerations:** what makes a
survivor "robust enough" to promote to full backtesting; how to encode learned
failure modes as reusable automated checks.

### Piece 4 — Portfolio construction
**Purpose:** blend the segment-specific survivors into one diversified strategy —
the right signal in the right segment, combined with sane risk management, sized
and rebalanced, then validated with full cost-and-risk backtests and the go-live
gate.
**Depends on:** Pieces 1-3. **Key considerations:** allocation across
segment-strategies (equal-risk, alpha-weighted, correlation-aware); interaction
with the existing regime gate / stops / position sizing; capacity & cost realism;
the final once-only holdout and paper-trading shakeout before any real capital.

## 6. Orchestration guidance (for the driving agent)

- **Subagent-first.** The top-level agent orchestrates, decides, and reviews;
  push context-heavy exploration, implementation, and adversarial review into
  subagents so the top-level context stays clean. This charter + the referenced
  files are enough for a subagent to start a piece.
- **Per piece:** run `brainstorming` (refine into a focused spec) →
  `writing-plans` (implementation plan) → implement (fresh-implementer subagent) →
  **adversarial review** (a skeptical subagent hunting correctness/overfitting/
  look-ahead bugs) → verify tests + ruff green → commit. This mirrors how the
  existing options infrastructure was built and reviewed.
- **Guard the science in review.** The adversarial pass must specifically check:
  no look-ahead in signal or return construction; the trial count is honest;
  the multiple-testing correction is correctly applied; the holdout was not
  touched during discovery.
- **Keep the ledger.** Update `docs/experiments.md` and the experiment journal;
  extend `docs/glossary.md` as new terms arise.

## 7. Key risks

1. **The multiple-testing false-positive factory (highest risk).** The engine's
   speed is also its danger: sweep enough signals and "significant" alphas appear
   from pure luck. If §3's trials-aware bar + once-only holdout are compromised,
   the system becomes a very fast machine for fooling ourselves. This is the one
   thing that cannot be cut.
2. **Segment overfitting.** Slicing into many segments multiplies trials; the
   correction must span the entire sweep, and segment "winners" need the same
   holdout discipline.
3. **Data quality / look-ahead / survivorship.** New signals may need new data
   (sentiment, short interest, revisions, insider, real options flow); each source
   must be verified PIT and survivorship-safe before it's trusted.
4. **The honest null.** It is entirely possible no robust, tradeable alpha exists
   in reach — and the correct scientific response is to accept that, not to lower
   the bar until something passes. The engine must make *killing* ideas as
   satisfying as finding them.

## 8. Open program-level decisions (deferred to per-piece specs)

- The **signal library** scope and sourcing (which candidates; what new data —
  analyst revisions, short interest, insider transactions, sentiment, real options
  flow — and in what order).
- The exact **statistical machinery** (FDR vs deflated-Sharpe vs both;
  significance/FDR thresholds; discovery/holdout split dates; factor set — Carhart
  4 vs FF5).
- **Segment definitions** and the classification data source (GICS/SIC).
- **Portfolio blend** method and risk model.
- Long/short vs long-only as the canonical return-series construction (long/short
  isolates alpha best; long-only matches the real no-shorting constraint — the
  engine may compute both).

---

*Companion reading in-repo: `docs/experiments.md` (what's been tried and why it
failed), `docs/glossary.md` (terms), `scripts/signal_scan.py` and
`scripts/factor_regression.py` (the two primitives this engine composes).*
