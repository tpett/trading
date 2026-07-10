# R3 — Down-cap / illiquid universe (design spec)

**Status:** approved direction (strategy audit 2026-07-10, developer-approved
sequence R2 → R1 → R3; design approved 2026-07-10 with full enumeration +
full battery). Pre-registers the universe, the verification gate, and the
sweep BEFORE any data is fetched.

## 1. Thesis under test

A **simple** momentum tilt — the alphasearch long-only top-quintile series
(equal weight, monthly rebalance, rank-exit; the wrapper-free construction
R2 vindicated in §14) — **beats SPY** in the small/micro-cap band where a
~$1k long-only account's size is a capacity edge institutions cannot
exploit. R2 showed bare momentum only *ties* SPY in large-cap waters; R3
asks whether the same simple construction *beats* SPY where the account's
smallness is an advantage rather than an irrelevance.

**Success (the R1 gate):** cost-charged long-only Sharpe ≥ SPY AND total
return > SPY over the discovery window, then the robustness battery, then —
only on explicit developer approval — the once-only reserved holdout. This
spec spends NO holdout.

## 2. The universe (frozen construction rules)

**Survivorship-free roster.** Fetch Tiingo's `supported_tickers.csv` (the
full historical ticker list, delisted names included, with
ticker/exchange/assetType/priceCurrency/startDate/endDate). Filter
**structurally** (no performance-dependent selection):

- `assetType == "Stock"` (excludes ETFs, funds, mutual funds).
- `priceCurrency == "USD"`.
- `exchange ∈ {NYSE, NASDAQ, NYSE ARCA, AMEX/NYSE MKT}` — the major US
  common-stock venues (frozen exact string set determined at build time from
  the file's actual `exchange` values, recorded in the roster build log; OTC
  / pink-sheet venues are excluded — untradeable, fraud-prone, and outside
  the account's reach).
- A name is a **candidate as-of date D** iff D ∈ [startDate, endDate]
  (endDate empty = still listed). Delisted names are first-class — their
  presence is what makes the roster survivorship-free.

**PIT band membership (recomputed as-of every monthly decision date D).** A
candidate is *in the down-cap universe* at D iff ALL of:

1. **Market cap** = `shares_outstanding(latest companyfacts row FILED ≤ D) ×
   raw_close(≤ D)` ∈ **[$50M, $2B]**. **Correctness requirement (frozen):**
   the price MUST be the raw/unadjusted share price (`close_raw`), NOT the
   split/dividend-adjusted `close` — an adjusted close is not the actual
   share price and would misstate the absolute cap (the same vendor-
   adjustment look-ahead class as the div_yield bug in the Tier-1 batch).
   The existing PIT computation (`value.py:78` / `spec.py:494`) is reused
   only if it is confirmed to consume the raw price; otherwise the band
   selector computes cap on the raw basis directly.
2. **Tradeability — spread:** trailing-63-session Corwin-Schultz effective
   spread (the R1 `effective_spread`) **≤ 2.0%**.
3. **Tradeability — depth:** trailing-63-session **median** dollar-volume
   (`close × volume`) **≥ $50,000/day** (a $1k order is then ≤ ~2% of a
   day's volume; well below the $20M large-cap gate).

Band membership is **dynamic** — a name enters/leaves as its cap and
liquidity cross the bounds — which is the point-in-time-correct way and
admits no "today's small-caps" selection leak.

**Sub-universes (frozen, three UniverseSpecs).** To locate where any edge
lives, register three:

- `downcap` — the full band, $50M–$2B.
- `downcap:small` — $300M–$2B.
- `downcap:micro` — $50M–$300M.

Each is an ordinary `UniverseSpec` (name + `cache_dir` +
explicit-symbols-per-date via the PIT selector + `fundamentals_dir`). The
two sub-bands partition the full band by the same cap rule; the tradeability
screens (2, 3) apply identically to all three.

**Missing-shares handling (the market-cap path's survivorship risk).** A
candidate with NO companyfacts shares FILED ≤ D cannot be cap-banded and is
**excluded at D** (fail-closed — never guess a cap). This is a real
survivorship hazard: names lacking XBRL shares skew smallest / most
distressed / oldest-delisted, so excluding them can quietly re-introduce the
bias R3 exists to avoid. Phase A therefore **measures** the exclusion (§4)
and gates on it.

## 3. Data acquisition (Phase A, the heavy step)

The candidate set is the full US-common-stock roster (~6–8k names across
history). R3 backfills, for every candidate:

- **Bars** into a fresh cache `data/equities-downcap-tiingo/` with its own
  `.source = tiingo` marker (the symbol-agnostic `fetch_ohlcv` path; a new
  backfill driver feeds it the roster instead of index membership).
- **companyfacts shares** into the fundamentals store for any candidate not
  already present (SEC companyfacts, 0.11s throttle; delisted filers are
  covered because they filed 10-Ks while public).

Run overnight on the always-on mini (as prior gathers were), Tiingo
rate-limited. Coverage gaps (Tiingo 404 / thin history) are recorded, not
guessed. This is a one-time cost.

## 4. Verification gate (frozen pass criteria — the audit's mandate, BEFORE any sweep)

Phase A ends with a written report and a hard **GO / NO-GO**. Criteria frozen
now so the decision is not post-hoc:

- **Survivorship present:** delisted names constitute **≥ 15%** of
  candidate-months in the band. (Below that, the roster is suspiciously
  survivor-tilted — investigate the filter before proceeding.)
- **Shares-coverage (the §2 risk):** across band-eligible candidate-months,
  the fraction with PIT shares available is **≥ 70%** (i.e. < 30% dropped
  for missing shares). **Below 70% → NO-GO on the market-cap band**; fall
  back, via a written prospective amendment, to a dollar-volume-only band
  (tradeability screens 2–3 without the cap bound). Report the size/vintage
  skew of the dropped names either way.
- **Spread realism:** report the CS effective-spread distribution across the
  band (median, IQR, % ≤ 2%). The 2% screen must be a real filter, not a
  no-op or a near-total cull.
- **Breadth:** **≥ 15 tradeable names in EVERY month** of the discovery
  window 2019-01-01..2023-12-31, for each of the three universes it will be
  swept as. A universe with any sub-15 month is dropped from the sweep
  (recorded, not silently skipped). ≥ 50/month is the comfort target for
  stable quintiles.

If frozen criteria (band bounds, spread cap, volume floor) turn out
degenerate (e.g. breadth fails), that is a **NO-GO finding recorded in
experiments.md and resolved by a written amendment** — never a silent
re-tune.

## 5. Sweep + gate (Phase B, only if Phase A returns GO)

Same locked engine mechanics as every prior sweep: discovery
2019-01-01..2023-12-31, quintiles (tercile < 50 names, skip < 15), equal
weight, monthly, BH-FDR q=0.10 across the whole journal, DSR advisory,
**holdout 2024+ untouched**.

- **Primary pre-registered hypothesis:** `momentum_v1`'s long-only
  top-quintile series vs SPY, on each of the three down-cap universes, under
  the R1 cost-charged long-only gate.
- **Full battery alongside (exploratory, BH-counted):** every registry
  signal whose stores the down-cap universe supports — all price/volume
  signals, plus fundamentals signals to the extent companyfacts covers the
  band (`_check_universe_supports` refuses the rest, as today). Options and
  insider signals are unsupported here (no such stores for the band) and are
  refused. Momentum is the pre-registered primary; everything else is
  exploratory and subject to the standing pseudo-replication disclosure.
- **Read:** the R1 `--long-only` leaderboard (cost-charged long-only vs SPY)
  over the down-cap trials; the four-factor L/S alpha remains the mandatory
  diagnostic; BH survivorship a reported property. Any candidate that clears
  the gate goes to the battery; the holdout is spent only on explicit
  developer approval.

## 6. Build vs reuse

**BUILD:** (1) Tiingo `supported_tickers` fetch + parse; (2) the
survivorship-free roster + the PIT band-membership selector (cap +
spread + volume screens, dynamic per date); (3) the bar+shares backfill
driver for a non-index roster + a fresh cache dir; (4) the three
`UniverseSpec` registrations; (5) the Phase-A verification report + GO/NO-GO
computation. **REUSE:** `UniverseSpec` / `build_universe_panel` /
`_check_universe_supports`; the PIT market-cap computation; the entire R1
Corwin-Schultz + long-only-vs-SPY machinery (`effective_spread`,
`cost_charged_lo`, `spy_benchmark`, `build_long_only_leaderboard`,
`run_battery`); the alphasearch sweep + BH-FDR + journal.

## 7. Out of scope

Spending the holdout (developer-gated, separate); any market-neutral /
short construction (the account is long-only); R4 (the SPY-plus-tilt paper
control, separate spec); a paid market-cap/shares vendor (companyfacts is
the free PIT source; if its coverage fails the §4 gate the answer is the
dollar-volume-only fallback, not a new vendor).

## 8. Risks (disclosed)

- **Shares-coverage survivorship leak** — the §2/§4 central risk; gated at
  70% with a dollar-volume-only fallback.
- **Tiingo small-cap bar coverage** — thin/delisted micro-caps Tiingo lacks
  become gaps; recorded, and they lower breadth honestly rather than being
  imputed.
- **Spread-model stress in illiquid names** — Corwin-Schultz was validated
  on liquid names (R1); across micro-caps its high-low estimator is nearer
  its noisy regime. The 2% screen and the Phase-A distribution report are
  the guards; the capacity curve remains a diagnostic.
- **Down-cap factor exposure** — a down-cap momentum win may be the size
  premium in disguise; the four-factor diagnostic (SMB loading) is retained
  precisely to read that, and the gate is vs SPY (not size-neutral) by
  deliberate design — beating SPY from this seat is the goal, whatever the
  loading.
