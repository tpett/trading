# Experiment log

A human-readable record of every strategy experiment: what we tried, **why**,
and **what we learned**. The machine-readable source of truth is the
append-only counter at `journal/experiments-<venue>.jsonl` on the trading host
(one record per walk-forward window + one stitched `walk_forward` summary per
run); this document is the narrative layer over it. **Keep it updated: add a
row here whenever a new configuration is walk-forwarded.**

## How to read this

- **The gate** (spec: Backtesting & Validation): a configuration passes only if
  its **stitched out-of-sample (OOS) Sharpe beats SPY's** *and* total return is
  positive. Over our window SPY buy-and-hold scores **0.96** — that is the bar.
- **Methodology**: walk-forward, 30-month train / 3-month test, rolling; each
  window grid-searches exactly two hyperparameters (entry-score threshold ×
  stop-ATR multiple, 5 × 4 = 20 combinations), selects the best on train, and is
  scored on the untouched test slice. The test slices are stitched into one
  OOS curve — that stitched Sharpe is the number quoted below. The 2022 bear
  market must fall inside the OOS span (a strategy that only ever "sees" a
  downturn in training data is not tested).
- **Anti-overfitting discipline**: every window is journaled and counted, so the
  total number of configurations tried is auditable — quote it alongside any
  result. The final holdout (≥ 2026-01-05) is reserved and touched exactly once,
  only for a configuration that has already passed walk-forward. None has, so
  the holdout is still unspent.
- **All figures below are journal-sourced** (equities, `sp500 + ndx` universe
  unless noted). "trades" is the stitched OOS trade count. Fees are $0 on
  Robinhood equities, so fee drag is zero throughout.

## Summary

| # | Date | Configuration | Rationale (in one line) | OOS Sharpe | OOS total | Trades | Verdict |
|---|------|---------------|-------------------------|-----------:|----------:|-------:|---------|
| — | — | **SPY buy-and-hold** | the benchmark to beat | **0.96** | +124% | — | the bar |
| 0 | 07-05 | Baseline momentum | starting point (M3 backtester) | 0.52 | +39% | 169 | best-so-far |
| 1 | 07-05 | Wide hyperparameter grids | does a broader search find better params? | 0.41 | +29% | 160 | refuted |
| 2 | 07-06 | Trailing exits | protect profits by trailing the stop | 0.09 | +2% | 237 | refuted (hard) |
| 3 | 07-06 | **Factor-scale momentum** | measure momentum at academic 3/6/12-mo scale | **0.59** | +39% | 444 | **promoted to live** |
| 4 | 07-06 | + S&P MidCap 400 | more names, momentum stronger in mid-caps | 0.50 | +36% | 536 | refuted |
| 5 | 07-06 | Quality overlay | profitable firms complement momentum, filter junk | 0.49 | +29% | 337 | refuted |
| 6 | 07-06 | Value overlay | value+momentum is a classic diversifying pair | 0.45 | +25% | 360 | refuted |
| 7 | 07-07 | Survivorship-free momentum (Tiingo) | is 0.59 real once delisted names are included? | **0.45** | +27% | 335 | refuted |

### 8. Pure mid-cap (S&P 400) momentum — 0.49 (refuted, the universe pivot)

**Rationale:** large-caps are too efficient for a momentum edge and too failure-free
for the quality screen (see conclusion 5). Momentum is academically stronger in
less-efficient names, so pure S&P MidCap 400 (survivorship-free Tiingo, 98.2%
coverage) with a hard $20M liquidity floor and **15 bps slippage** (3x large-cap,
an honest cost guard) was the natural place to look. This is the "does our built
strategy work where momentum should" test.
**Result:** OOS Sharpe **0.49**, +35% total, net of 15 bps — GATE FAIL. Decisive
context from buy-and-hold over 2018-2026: SPY 0.78 / +188%, **MDY (mid-cap) 0.48 /
+94%**. So the strategy (0.49) merely *matches* mid-cap buy-and-hold on Sharpe
(no universe alpha) and badly trails it on total return (regime gate sits in
cash); and mid-caps themselves lagged large-caps. **The momentum-is-stronger-in-
mid-caps thesis is refuted at realistic costs** -- the tradeable result is no
better than holding the mid-cap index, which is worse than holding SPY. Caveat:
0.49 is net of 15 bps; a 0-slippage re-run would show whether the raw signal is
stronger but eaten by costs (the small-cap-momentum graveyard) vs genuinely no
stronger -- the tradeable conclusion is the same either way. The quality screen
(MC-2), gated on this showing life, is therefore not pursued: there is no base
alpha to sharpen.

**No configuration has beaten SPY out of sample, and the best one shrinks once
survivorship bias is removed.** Experiment 3 (factor-scale momentum) scores 0.59
on survivor-only data but **0.45** on delisted-inclusive data — so a meaningful
chunk of the apparent edge was survivorship bias. Live paper trading runs on the
exp-3 config, but its honest OOS Sharpe is ~0.45, not 0.59.

## The experiments in detail

### 0. Baseline momentum — OOS Sharpe 0.52
The starting configuration from the M3 backtester: cross-sectional
vol-adjusted momentum with a volume-surge / breakout-proximity / RSI-guard
composite, equal-weight, a regime gate (SPY 50/200 SMA + volatility percentile
scaling exposure 1.0 / 0.5 / 0.0), and frozen ATR stops.
**Learned:** a real but sub-benchmark edge — momentum captures *something*, but
not enough to beat simply holding the index.

### 1. Wide hyperparameter grids — 0.41 (refuted)
**Rationale:** widen the entry-threshold and stop-ATR search grids to see if a
larger search surfaces a better operating point.
**Learned:** the opposite — a wider grid overfits the training windows and
generalizes *worse* out of sample (0.41 < 0.52). This is why the grid is
deliberately kept small (2 hyperparameters, 5×4) and never expanded to chase a
number.

### 2. Trailing exits — 0.09 (refuted, hard)
**Rationale:** replace frozen ATR stops with a trailing stop + trend-break rule
to lock in gains as winners run.
**Learned:** catastrophic. Trailing stops repeatedly gave back gains from local
peaks and churned the book (237 trades for +2% total). Frozen stops are
decisively better for this strategy; exit style is now `frozen` and the
trailing path stays behind a config flag, not on by default.

### 3. Factor-scale momentum — 0.59 (promoted to live) ⭐
**Rationale:** academic momentum is measured over 3/6/12-month lookbacks (the
"factor scale"), not the shorter windows the baseline used; and more concurrent
positions should diversify idiosyncratic noise. Config: `momentum_windows =
[63, 126, 252]`, `max_positions = 20`, `position_size_pct = 0.045`,
`time_stop_bars = 60`.
**Learned:** the single biggest improvement (0.52 → 0.59) and the best config we
have — **promoted to the live paper config**. Longer, factor-style momentum
windows plus broader diversification is the right shape. Still short of SPY's
0.96, so promotion means "best known," not "ready for real money."

### 4. + S&P MidCap 400 — 0.50 (refuted)
**Rationale:** momentum is theoretically *stronger* in smaller, less-efficient
names; adding the 400 mid-caps triples the opportunity set.
**Learned:** it diluted rather than sharpened (0.59 → 0.50). More names meant
more trades (536) and more ways for a low-quality name to spike to the top of
the ranking, and the wider spreads/noise of mid-caps outweighed the extra edge.
Directly informs the "should we expand the universe?" question: on this
strategy, breadth-by-index-addition hurt. (A *liquidity-floored*,
survivorship-free breadth test is a different, still-open hypothesis.)

### 5. Quality overlay — 0.49 (refuted)
**Rationale:** blend a quality factor (gross-profitability percentile) into the
ranker (`quality_momentum_v1`) so profitable firms are favored and junk momentum
is filtered.
**Learned:** diluted the momentum core (0.49 < 0.59). Caveat: genuine
fundamentals coverage was only ~19% in the earliest windows (pre-ASC-606
revenue tags fall outside the locked tag chain), so early windows pulled the
quality signal toward neutral — but even granting that, no improvement emerged.

### 6. Value overlay — 0.45 (refuted)
**Rationale:** value and momentum are the classic negatively-correlated factor
pair; a value tilt (earnings yield + book-to-market, `value_momentum_v1`) should
diversify the momentum signal.
**Learned:** hurt more than quality did (0.45). Over this window a value tilt
worked *against* the momentum core rather than complementing it. This closed the
fundamentals-overlay hypothesis family: neither quality nor value helped.

### 7. Survivorship-free momentum (Tiingo) — 0.45 (refuted)
**Rationale:** every experiment above ran on yfinance data, which serves **no
delisted tickers** — so the historical universe silently dropped every company
that failed or was removed (journal survivorship ratio ≈ 0.88, i.e. ~12% of
member-sessions had no data). That biases results *optimistically*. This re-ran
experiment 3 (the promoted config, unchanged) on Tiingo's delisted-inclusive
data (97.8% coverage) to measure how much of the 0.59 was survivorship bias.
**Result:** OOS Sharpe **0.45** (vs 0.59 survivor-only), total +27% (vs +39%).
The prediction held: including the delisted/failed names that momentum sometimes
picks — and that then crash — knocks ~0.14 off the Sharpe. **The apparent edge
was partly survivorship bias; the honest number is 0.45, further from SPY's 0.96
than the biased backtests showed.**
**Caveats:** (1) the 0.59→0.45 drop conflates survivorship (the intended change)
with a small vendor-adjustment difference (Tiingo vs yfinance are both
split/div-adjusted but not bit-identical); isolating survivorship cleanly would
need a Tiingo-with vs Tiingo-without-delisted A/B. (2) 16 rename symbols were
still missing (the ticker-alias resolution had not been deployed when this ran);
a clean re-run with resolution would reach ~100% coverage and pin the number,
but 16/729 ≈ 2% won't change the conclusion.

## What we've learned overall

1. **Momentum has a weak, sub-benchmark edge on liquid US large-caps — weaker
   than the biased backtests showed.** On survivor-only data configurations
   land 0.09–0.59 OOS Sharpe; the best drops to **0.45 once survivorship bias
   is removed** (exp 7). None reaches SPY's 0.96. On this universe and horizon,
   the strategy does not beat buy-and-hold out of sample.
2. **Simplicity wins.** Wider grids (exp 1) and trailing exits (exp 2) both made
   it worse. The improvements came from *structure* (factor-scale windows, more
   positions — exp 3), not from more tuning or more machinery.
3. **Adding inputs diluted, it didn't sharpen.** Mid-cap breadth (4), quality
   (5), and value (6) all *reduced* the risk-adjusted return. The momentum core
   is cleanest on its own.
4. **Survivorship bias was real and material** (exp 7): ~0.14 of Sharpe. Any
   backtest number from this system should be read as survivor-only-optimistic
   unless it came from the Tiingo (delisted-inclusive) path.
5. **The quality/junk-screen idea is structurally dead for large-caps** (settled
   2026-07-07). The thesis was: a screen that removes failing junk momentum
   chases into a crash could recover some of the survivorship loss. But a direct
   census of the 230 sp500+ndx index exits over 2018–2026 found only **4 genuine
   distressed failures** (CHK 2020; SIVB/SBNY/FRC in the 2023 bank runs) — **2%**
   — versus 67 acquisitions and 106 still-trading reconstitutions. Large-cap
   members essentially don't go bankrupt while large; they get acquired or shrink
   out. So a junk screen has almost nothing to remove here, which is why the
   quality blend (exp 5) diluted and why the look-ahead "ceiling test" found no
   real failures to strip. **Neither fundamentals blend NOR screen helps this
   universe** — the fundamentals-overlay family is closed. (A quality screen may
   still matter in a small/mid-cap universe where junk-failures are common; it is
   the large-cap structure, not fundamentals per se, that defeats it here.)
   Aside: the ceiling test itself was compromised (it ran on the pre-ticker-alias
   cache so renames masqueraded as failures, and its offline mode wrongly dropped
   96 names that listed after 2018 — a real offline bug to fix before any
   reproducible offline re-run); the census, not the ceiling number, is the
   evidence.

## 9. Options IV-skew — the first differential signal (OPT-1 0.79, OPT-2 0.73; momentum-on-same-names control 0.46)

The orthogonal pivot (momentum is exhausted on liquid US equities; §8). Hypothesis:
a steep OTM-put vol skew — the market paying up for downside protection — precedes
LOWER forward stock returns, so ranking a universe by *flat* skew (buying the
low-skew names) should earn a cross-sectional premium. We trade the **stock** on
the options **signal**; we never trade the options. Data: ThetaData Standard EOD
option quotes for the **100 most-liquid S&P 500 + NDX names**, 2019-01..2025-12,
first-trading-day-of-month decision dates, three contracts per name (ATM, ~10%-OTM
put, ~10%-OTM call). IV is inverted from the quote **mid** (Black-Scholes,
`trading.research.options_iv`); `skew_put_atm = IV(otm_put) − IV(atm)`. **6767
cells, 99.9% with a valid skew, mean +0.059 (97% positive — the textbook equity
smirk).** The signal plugs into the SAME walk-forward as every equities experiment
(survivorship-free Tiingo bars for returns/stops/regime, same regime gate, same
2-hyperparameter grid, same 5 bps cost, same fixed 2026-01-05 holdout). Universe
restricted to exactly the gathered names via a symbols allowlist.

| Experiment | Ranker | Stitched OOS Sharpe | Total | vs stitched-SPY 0.70 |
|---|---|---|---|---|
| **OPT-1 skew level** | `skew_v1` (percentile of −skew_put_atm) | **0.79** | **+64%** | **PASS** |
| **OPT-2 skew change** | `skew_change_v1` (skew − own trailing mean) | **0.73** | +33% | **PASS** |
| **Control** | `momentum_v1`, *identical universe + window* | 0.46 | +16% | FAIL |

OOS span 2021-07..2025-12 (30-month train / 3-month test, rolling; 18 windows; 2022
stress covered). **These are the first gate PASSes in the program** — momentum never
beat its benchmark (best 0.45).

**What is and isn't established (read this before believing it):**
- **Skew is a real cross-sectional signal.** The clean control — momentum_v1 on the
  *exact same 99 names and window and methodology* — scores **0.46** (reproducing the
  §7 survivorship-free momentum ~0.45, which also validates the harness is
  consistent). Skew scores **0.79**. So the edge is the SIGNAL, not the universe or
  the window: skew beats momentum on the same book by **+0.33 Sharpe and 4× the total
  return**. This is the one airtight comparison (everything held equal), and it is the
  real result.
- **But long-only skew only ~ties passive SPY.** By the engine's stitched-benchmark
  metric skew_v1 beats SPY 0.79 vs 0.70. A naive continuous SPY buy-hold over the same
  span, however, is ~**0.75 Sharpe / +69%** (the 0.70 is a few points lower because the
  stitch drops each window's first-day return + low-coverage sessions — applied
  identically to all three runs, so the *comparisons* hold, but the absolute edge over
  a passive index is thin). skew_v1's +64% total actually **trails** SPY's +69%. So a
  long-only book of the 20 lowest-skew large-caps is still ~a large-cap portfolio — the
  skew premium is real but largely washed out by market beta in a long-only wrapper.
- **2022 was ugly** (all four quarters −2.9 to −3.6 Sharpe); the outperformance is
  concentrated 2023-2025. The per-window Sharpes are noisy.
- **Holdout not yet evaluable.** The ≥2026-01-05 holdout has no data (skew cells end
  2025-12); it stays reserved until 2026 options are gathered, then run ONCE.
- **Threshold caveat:** skew_v1's composite is a single-feature percentile (near-uniform),
  so the `entry_score_threshold` grid axis is nearly inert — the book is effectively
  "long the ~20 lowest-skew names." Don't compare its tuned threshold to momentum's.
- **Anti-overfitting:** OOS-only, 2 hyperparameters, journaled (`experiments-equities.jsonl`);
  but this is ONE universe/window/frequency. Data-quality: 6 of 6761 cells (0.09%) have
  physically implausible |skew|>0.25 (bad inversions); left unfiltered (percentile-rank
  caps their effect to one position each).

**Model-free cross-sectional check (the decisive test the long-only Sharpe can't do).**
A long-only backtest can't separate a skew *premium* from market *beta*, so we measured
the skew→forward-return relationship directly: each decision month, rank names by skew
and compute the tercile spread (mean forward return of the LOW-skew third minus the
HIGH-skew third — exactly what a market-neutral long-low/short-high book earns) and the
Spearman(skew, forward return). Forward returns are total-return (adjusted close).
Results by holding horizon:

| Horizon | FULL 2019-2025 spread (t) | OOS 2021-07+ spread (t) |
|---|---|---|
| 21d (backtest's monthly rebalance) | +0.86%/mo (t=1.5) | **+0.03%/mo (t=0.04)** |
| 42d | +1.82%/2mo (t=2.1) | +0.85% (t=0.7) |
| 63d | **+3.56%/qtr (t=3.2)** | +2.04% (t=1.4) |

This **reframes OPT-1.** At the backtest's monthly horizon the OOS premium is
**statistically zero** (t=0.04; low- and high-skew terciles returned +1.32% vs
+1.28%/mo — indistinguishable), and the Spearman is even weakly WRONG-signed. So
**OPT-1's 0.79 gate-pass is not attributable to a skew premium** — it comes from the
regime gate + per-window stop tuning applied to a large-cap book whose low-skew members
happened to be favorable beta in 2023-2025, not from skew's cross-sectional predictive
power (which is ~nil in that window). The genuine signal that *does* exist is (a)
concentrated in the 2019-2021 crisis-inclusive period and (b) only emerges at a
**2-3 month** horizon (t=3.2/quarter over the full sample) — consistent with skew being
a slow, tail-/crisis-sensitive signal, not a monthly cross-sectional stock-picker.

**Verdict: a real but weak medium-horizon signal — NOT the monthly alpha the backtest
appeared to show.** The honest read: the long-only OPT-1/2 gate-passes are beta/tuning
artifacts (the model-free monthly premium is zero OOS); there IS a skew premium, but
it needs a 2-3 month hold and is significant only over the crisis-inclusive full sample.
This is still the most signal any track has shown — but it argues for testing skew as a
**longer-horizon and/or regime-conditional (high-vol) signal**, and a **long/short
quarterly spread**, rather than deploying the monthly long-only book. A regime split
confirms the shape: at the 63d horizon the tercile spread is significant in BOTH vol
regimes and ~2× larger in high-vol months (+5.5%/qtr, t=2.3, vs +2.7%/qtr, t=2.2 in
calm months; at the monthly horizon neither is significant) — skew earns its keep at a
quarterly hold and in stress, exactly where the monthly long-only book didn't use it. The momentum-vs-skew
gap (0.79 vs 0.46) remains real but, given the null cross-sectional premium, reflects
name-set beta differences more than a repeatable skew edge. (Signal build:
`src/trading/signals/skew.py`, adversarially reviewed — no lookahead, correct sign,
thin-cross-section guard; study: `scripts/skew_premium_study.py`.)

## 10. Alpha-search engine (Piece 1) — built; first sweep = honest null (238 trials, 0 BH survivors)

The core alpha-search engine is implemented (`src/trading/alphasearch/`,
`trading alphasearch sweep|leaderboard|holdout`; design:
`docs/superpowers/specs/2026-07-08-alpha-search-engine-design.md`). It turns
signal + universe into a four-factor L/S alpha t-stat via a monthly portfolio
sort, gates candidates with BH-FDR (q=0.10) over the persisted trial journal
(`journal/alphasearch-trials.jsonl`), reports DSR for survivors, and enforces
a touched-once 2024+ holdout. Pre-registered rules are in the design spec §5;
terms in the glossary ("Multiple testing" section).

**First pre-registered sweep ran 2026-07-09 (master dcaf5cf): 238 journaled
discovery trials, ZERO BH survivors at q=0.10 — a comprehensive honest null.**
Three passes over discovery 2019-01-01..2023-12-31, all in one journal, one
BH computation: (a) flat pools × 13 signals (7 price + 6 options; fundamentals
signals refused — no local store) = 26 trials; (b) `--segments` × 7 price
signals over 26 universes (flat + 24 SIC segments) = 208 cumulative;
(c) options family × the 5 viable opt segments = 238 cumulative (245 events
incl. 7 honest `SortError` error trials on `midcap:construction`, whose
cross-section fell below 15 on every decision date). Best candidates vs the
BH bar (k=1 needs p ≤ 0.10/238 ≈ 0.00042): `atm_iv` midcap t=−2.70 p=0.0071;
`disthigh` midcap:services t=−2.54 p=0.0113; `atm_spread`
opt-largecap:services t=+2.52 p=0.0118; `vrp` largecap t=+2.47 p=0.0138.
Nothing within an order of magnitude of the bar. **No holdout touches were
spent** (nothing qualified; the 2024+ holdout remains fully reserved). Reading:
across every seed signal, both caps, and every pre-registered segment, no
cross-sectional signal in this library carries four-factor alpha the
multiple-testing bar can distinguish from luck at monthly rebalance — the
engine did exactly what it was built to do: kill 238 hypotheses in ~15
minutes of compute for the cost of zero forward capital. Suggestive (NOT
significant, and classical-SE inflated) patterns for future pre-registered
work, recorded only so they aren't re-discovered as "new" ideas: low-IV/-vol
longs showed NEGATIVE alpha (high-vol names outperformed 2019-23 on a 4F
basis, consistent with the beta-heavy bull window), and vrp/atm_spread longs
were the only positive family. Known caveat carried forward: 2024-25 data was
partially examined by the §9 skew studies, so any FUTURE holdout passes for
skew-family signals carry residual contamination risk and must be read
conservatively (spec §5.3).

**Survivorship caveat (the universes, not the bars, are tilted).** The
alphasearch large-cap and mid-cap universes are each the gathered options
pool — ~100 most-liquid sp500+ndx names (large-cap) selected at gather time
in 2025, held fixed back to 2019 (§9's ThetaData gather). The underlying
equity *bars* are survivorship-free Tiingo (§7), but pool *membership* itself
is survivorship-tilted: a name only entered the pool because it was still
liquid enough to gather in 2025, so every discovery alpha here is measured on
names already known to have stayed liquid through 2025, not on the universe
as it actually looked in 2019. Read any discovery or holdout result on these
universes as conditional on that survival, not as a clean point-in-time
backtest.

**Classical-SE caveat (read at the point of use).** Every t-stat and BH
p-value on the leaderboard comes from classical OLS standard errors on daily
data (`trading.alphasearch.evaluate.ols`); volatility clustering typically
inflates these vs. Newey-West/HAC SEs by something like 10-30%, so a marginal
BH pass (t close to the threshold) should be read skeptically until re-checked
with a heteroskedasticity/autocorrelation-robust estimator. This does not
change the gate statistic itself (spec §5 is pre-registered and locked) — it
is a caveat on how to *read* a marginal pass, not a different pass rule.

**Adjusted-price / raw-quantity mixing caveat (2026-07-09).** Several signals
combine a Tiingo *adjusted* price series with a *raw* (unadjusted-basis)
quantity, and adjusted prices bake in FUTURE corporate actions relative to
any given as-of date (Tiingo recomputes `close` from the full downloaded
history on every new split/dividend, so an earlier row's adjusted value
already reflects actions that happen after that row). `div_yield` mixed raw
`div_cash` with adjusted `close` this way and has been FIXED (this entry) to
use `close_raw` with each trailing payment split-adjusted into as-of terms —
see the design spec's dated amendment and the glossary. Two related but
UNFIXED, bounded distortions are recorded here rather than corrected, since
neither has run a sweep affected enough to justify unwinding journaled
trials: (a) `amihud`/`osv`'s dollar-volume term is `close * volume` — Tiingo
scales volume by the inverse of the same split factor it divides price by,
so the SPLIT component of the adjustment cancels in the product (dollar
volume is basis-invariant to splits); the DIVIDEND component of the price
adjustment has no inverse in volume, so a small future-dividend-adjustment
tilt survives in both signals' dollar-volume terms — bounded (dividend
adjustments are typically sub-1% price effects, unlike stock splits) and
not worth a fix at this scale. (b) The pre-existing Piece 1 value signals
(`earnings_yield`, `book_to_market`, `src/trading/alphasearch/spec.py`
`_value_ratio`) compute market cap as raw FILED `shares_outstanding`
(actual share count as of the filing date, never split-adjusted forward)
times the as-of `close` (adjusted, i.e. already discounted for any split
that happens after that date) — the identical raw-quantity/adjusted-price
mismatch `div_yield` had, on the denominator side. Their 238 already-
journaled discovery trials (first sweep, above) should be read with this
caveat: a name that splits later has an understated historical market cap
in any pre-split trial period, distorting `earnings_yield`/`book_to_market`
at those dates in the same direction (inflated) as the pre-fix `div_yield`
bug. No trial has been re-run to correct this — recorded as a caveat on
existing results, not a fix, per the same discipline as the survivorship
and classical-SE caveats above.

**Segment sweep (Piece 2) — pre-registered disclosure.** `trading alphasearch
sweep --segments` expands the sweep to the frozen `SEGMENTS` table
(`trading.alphasearch.segments`; design:
`docs/superpowers/specs/2026-07-08-segmented-universes-design.md`): 10 coarse
SIC sectors + 2 fine industries (biotech SIC 2836/8731, banks SIC 6020–6039)
× 2 caps — deep pools (membership ∩ Tiingo caches, price signals only, since
they carry no options or fundamentals stores) plus options pools
(`opt-largecap:`/`opt-midcap:`) where ≥ 15 gathered names. Expected cost
≈ 200–300 new journaled trials, all discounted by the same BH q=0.10 bar over
the whole journal — segment multiplication makes the gate harder, never
easier, and a null across the entire segment sweep is a valid outcome.
Because assembly-time validation is all-or-nothing across the signal ×
universe cross-product, the sweep runs as separate passes: price signals over
everything (`--segments --signals
mom21,mom63,mom126,mom252,rev5,rvol21,disthigh`), then the options/
fundamentals families per options-pool segment (`--segments --universe
opt-largecap:manufacturing-tech --signals vrp,hedge,excite,atm_iv,smile,atm_spread,...`).
Disclosed caveats on every segment result (spec §4 rule 5): (a) SIC is each
filer's **current** code applied backward over the window (no PIT
reclassification); (b) segment membership is static across the window; (c)
fine industries double-count names with their parent sector — distinct,
honestly-counted trials; (d) a symbol without a SIC mapping belongs to no
segment (never guessed; see `sic_map.csv` provenance); (e) biotech (SIC
2836/8731) emits no universes under current data and the 15-name floor — the
charter hypothesis is structurally untestable as pre-registered, since large
biotechs file under 2834, which is fused into pharma-chemicals; any
redefinition must be a written prospective amendment filed *before* a biotech
sweep runs (none has run; nothing is spent); (f) the banks segment is
failure-censored — FRC and SBNY, two of the three 2023 bank failures, are FDIC
filers with no EDGAR presence and are structurally absent from the SIC map,
while SIVB is included, so banks-segment results are measured on a partially
failure-censored cross-section; (g) one mapped symbol (CTVA, SIC 0100
agriculture) falls in no segment's ranges — mapped-but-unsegmented is a known
small residual alongside the ~2% unmapped. Below-threshold segments (< 15
names) are excluded at build time and printed, never silently dropped.

**Operator recipe (the two canonical invocations).** In practice the pass
split above means two shapes of command cover the whole table without ever
tripping the all-or-nothing refusal. Price family, everywhere in one call —
every price signal is legal on every universe, flat or segmented, so this
sweeps the flat pools plus every `largecap:<segment>`/`midcap:<segment>` deep
pool and every options segment at once:
`trading alphasearch sweep --segments --signals
mom21,mom63,mom126,mom252,rev5,rvol21,disthigh`. Options/fundamentals family,
one options-pool segment at a time — repeat per `opt-largecap:<segment>` /
`opt-midcap:<segment>` name (the run's stderr prints the excluded segments, so
below-threshold ones are never silently skipped):
`trading alphasearch sweep --segments --universe opt-largecap:manufacturing-tech
--signals vrp,hedge,excite,atm_iv,smile,atm_spread,gross_profitability,earnings_yield,book_to_market`.
Running `--segments` bare against the full default registry is a deliberate
refusal, not a bug — deep pools have no options or fundamentals store, so the
cross-product check catches the mismatch before any trial is journaled; the
CLI prints an actionable `hint:` line naming this same two-pass split.

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

## 11. Tier-1 batch sweep — FIRST BH SURVIVORS: the amihud family (799 trials)

**Ran 2026-07-09 (master 6ac8427 + ops).** Pre-sweep ops completed first:
fundamentals store rsynced from the mini (1,110 symbol files) and the
largecap bar cache re-backfilled to the extended schema (728/729 = 99.9%
coverage, only FRC unavailable; the refetch also recovered 15 previously
missing rename symbols, growing the cache 713→728 — which changed segment
emission vs. the §10 sweep's universe set, see reconciliation).

**Execution:** the plan's five-invocation run-book, one leaderboard read at
the end. Reconciliation: journal landed at exactly **799 discovery trials**
(238 prior + 561 new), matching the re-derived arithmetic (the run-book's
printed ranges were computed against a 26-universe assumption; the actual
emitted set is 28 universes — 2 flat + 21 deep + 5 options — and observed
per-invocation deltas were exact under that correction: +28, +2, +286, +21,
+224). All flat-pool overlaps deduped by config hash as designed.

**RESULT — three BH survivors at q=0.10, all one signal family (amihud,
long-illiquid/short-liquid, monthly, equal weight):**

| signal | universe | 4F α/yr | t | p | DSR |
|---|---|---:|---:|---:|---:|
| amihud | midcap (flat, 137 names) | +61.5% | +8.34 | ≈0 | 0.999 |
| amihud | opt-midcap:trade (26) | +42.5% | +3.87 | 0.0001 | 0.679 |
| amihud | midcap:trade (61) | +37.2% | +3.69 | 0.0002 | 0.584 |

Nothing else passed (next best: droa largecap:finance t=−3.05 — wrong-signed
fundamental momentum, p=0.0023, below the m=799 bar).

**Read this skeptically before celebrating (the pre-registered caveats bite
hardest exactly here):**
1. **The cheap series carries NO transaction costs**, and amihud's long leg
   is by construction the most illiquid mid-caps — the exact names where
   costs, spreads, and capacity destroy paper alpha. The illiquidity premium
   is the classic "real on paper, unharvestable in size" anomaly. Mitigant:
   measured monthly one-way turnover is tiny (~4-6%), so the cost drag is
   bounded; a cost-realistic full backtest is the required next test.
2. Classical OLS SEs on daily data (t inflated 10-30% vs HAC) — though
   t=8.3 survives any plausible haircut.
3. Universe caveats: pool survivorship tilts (§10), and the 2019-2023 window
   includes the 2020-21 small/mid-cap liquidity mania.
4. Positive SMB (+0.48) and HML (+0.69) loadings on the flat-midcap trial —
   the 4F regression already nets these out, but the strategy is
   structurally a small/value-tilted book; capacity is small.

**The holdout (2024+) has NOT been touched** — the three survivors are
eligible for their once-only re-prove, but that decision (and whether
robustness/cost tests should come first, i.e., Piece 3) belongs to the
developer. Nothing is spent.

**Robustness battery pilot (2026-07-09, Piece 3 at 3cdf017; journal now 835
deduped discovery trials + 3 battery verdicts).** The frozen 7-check battery
(spec `2026-07-09-robustness-battery-design.md` §3) ran on all three
survivors — the program's first evidence-graded interrogation, no holdout
touched:

| target | verdict | what decided it |
|---|---|---|
| amihud:midcap | **HOLDOUT-ELIGIBLE** | all 6 checks pass — both halves strong (2019-21 a=+70.7 t=+6.15; 2021-23 a=+49.1 t=+5.60, NOT a one-regime wonder); 5/5 subset draws; 4/4 jitter; offset retention 1.00; survives GME/CELH/OVV exclusion at 0.90 retention; months 21%; 50 bps barely dents it (t=+8.27); capacity flat to $1M/side |
| amihud:midcap:trade | NOT eligible | check 1 FAIL: one-regime (2019-21 a=+59.9 t=+3.60 vs 2021-23 a=+7.5 t=+0.67); name-concentration squeaked by at 0.53 excluding GME/BBBY/DDS — meme-era names |
| amihud:opt-midcap:trade | NOT eligible | check 1 FAIL (same one-regime shape) + check 2 structurally infeasible (26-name universe; 13-name draws < 15 floor → 5 honest error trials, designed-in) |

Reading: the battery killed both trade-sector echoes as 2019-21/meme-era
artifacts and passed the broad mid-cap version with a surprisingly stable
profile — both halves independently significant, tiny turnover (~5%/mo)
making costs nearly irrelevant, and first-order Amihud impact negligible at
personal book sizes. The pseudo-replication caveat (above) applies to the
post-battery leaderboard: amihud:midcap's jitter variants now also display
as BH passes (near-duplicates of the same effect, expected); any UNRELATED
new PASS rows are artifacts of the enlarged m and earn nothing without
their own battery.

**Look-ahead audit (2026-07-09, developer-requested).** Question: is the
amihud alpha computed from data unknowable at decision time? Answer: **no.**
Decisive test — the identical trial recomputed with a raw-basis denominator
(close_raw × tape volume, removing the future-dividend adjustment channel
that leaked into div_yield) — reproduces the result: +60.20%/yr t=+8.08 vs
journaled +61.53%/yr t=+8.34 (per-date rank correlation 0.987, top-quintile
overlap 94.8%). Channels checked and quantified: dividend-adjustment tilt
(present, immaterial — median factor 1.0, worst 1.51× vs 3 decades of λ
spread; direction slightly DAMPENS the effect), zero-volume days
(structurally filtered, cannot poison), zombie scores from delisted names
(WPX in the long leg on 47/60 rebalances contributing nothing — removing it
RAISES alpha to +62.5, a drag not an edge), forward-return interaction
(none). The audit was read-only (no journal writes); a raw-basis or
recency-guarded redefinition would each be a prospective amendment + new
trial and buys ~nothing. The caveats that still matter are the disclosed
ones: pool survivorship, classical SEs, and paper-vs-realizable costs. This is the
program's second illiquidity finding. The first (§9: mid-cap option
illiquidity, the `illiquidity_veto_v1` ranker family) looked real until it
decomposed into the size factor (SMB loading t=13.1, four-factor alpha ≈ 0 —
a plain size sort beat it), and its tradeable form needed 15 bps costs it
couldn't pay. The amihud result is NOT the same decomposition story — its
+61.5%/yr alpha survives a regression that already includes SMB (+0.48
loading netted out) — but it shares the same tradability profile: the long
leg is the most illiquid names in the book, where quoted alpha and
realizable alpha diverge most. Given that history, amihud is parked pending
cost-realistic backtesting and robustness interrogation (Piece 3 machinery);
the holdout stays reserved, and no live promotion path is open for this
family until it survives realistic costs. Recorded so a future sweep doesn't
"rediscover" illiquidity as a novel finding.

**Full walk-forward (2026-07-09, `amihud_v1` ranker, config
`config/experiments/amihud-midcap`, master d020d8a): OOS Sharpe 0.16 vs SPY
0.96, total +7.43% — GATE FAIL, decisively.** The battery-passed L/S factor
alpha does NOT survive the long-only tradeable wrapper: at 15 bps with the
strategy machinery (stops, regime gate, threshold entries), the long-illiquid
book bleeds across most windows (2020's rebound quarters carried what little
total return there was; 2022-2025 is a string of negative quarters against a
rising benchmark). This completes the illiquidity story in both directions:
the FACTOR is real (L/S, battery-passed, look-ahead-audited above), but as a
long-only strategy in this account's format it is worse than holding the
index — echoing §9's lesson that the long-only wrapper is where
cross-sectional signals go to die. Journaled: 94 walk-forward windows +
summary in `journal/experiments-equities.jsonl`. The alphasearch holdout
remains unspent; with the tradeable form refuted, spending it on amihud is
academic unless a market-neutral or overlay construction is designed (its
own pre-registered spec, if ever).

**Battery pseudo-replication caveat (read post-battery promotions
skeptically).** Checks 1-4 journal 12 new BH-counted discovery trials per
battery run, deliberately (spec section 5.6 — no second ledger, no
trial-hiding), but those 12 are near-duplicate re-evaluations of ONE
already-significant effect, not 12 independent hypotheses. Piling enough
near-duplicate, near-zero p-values from a true survivor's own battery onto
the sorted journal can raise the BH step-up rank k faster than it raises n,
which LOWERS the effective p-value bar for every OTHER candidate still
waiting in the queue — the opposite of the "the bar rises as the journal
grows" caution already documented (glossary, "Multiple testing" section,
which only covers the demotion direction). A review pilot demonstrated this
concretely: battery-testing the three amihud survivors (36 added battery
trials, m=799→835) flips `droa` largecap:finance — the §11 next-best
non-survivor, t=−3.05, p=0.0023, untouched and unre-examined — from FAIL to
PASS at q=0.10, purely because the denominator moved under it. Consequence:
any post-battery leaderboard promotion must be read as skeptically as a
first-pass survivor and must earn its OWN battery + holdout before being
trusted — it may NOT be waved through on the strength of an unrelated
candidate's battery run.

**Form 4 insider family registered 2026-07-09, pre-sweep.** 3 purchase-side
signals (`npr_90`, `cluster_buys_90`, `officer_buy_90`) are frozen in
`docs/superpowers/specs/2026-07-09-insider-pipeline-design.md` §3 —
definitions, signs (+ all three), and NaN conventions (cluster's
0-vs-never-covered distinction; officer's raw-price basis and dual
`requires_insider`+`requires_fundamentals` flags) pre-registered BEFORE any
trial. The data store is SEC DERA insider-transaction quarterly ZIPs 2018q3+
(open-market P/S only, FILED-date PIT keying — the transaction date precedes
filing and is never scored), built by `scripts/build_insider_store.py` into
`data/insider/equities/` and mapped through the committed cik_map intervals
(unmapped CIKs counted, never guessed). Disclosed limitation: Form 4/A
amendments carry their own accessions and the frozen spec has no form-type
filter, so an amended transaction can appear twice — slight over-counting,
accepted for the batch sweep. **Nothing sweeps under this
registration**: the pre-registered discovery sweep belongs to the combined
options-v2 + insider batch spec (one read after both data sets land), which
will disclose the full trial count. New glossary section "Insider
transactions (the Form 4 family)".

**Form 4 store BUILT 2026-07-09 (real data):** 31 quarterly ZIPs
(2018q3-2026q1; 2026q2 unpublished, tolerated), 924,625 P/S rows parsed with
zero unparseable, 1,261 symbol parquets, window-membership coverage 96.6%
(quiet companies legitimately lack rows), no per-year holes. EDGAR-grade
spot-check against famous public filings passed: Berkshire's March 2022 OXY
purchases (owner CIK 315090, $55.78/$57.38 — exact public dates/prices), the
Saudi PIF Lucid injections, JAB's 2019 Coty tender. filed >= trans_date
everywhere. The family now awaits the combined options-v2 + insider batch
sweep.

**Gather v2 COMPLETE + verified 2026-07-09:** largecap 7,399 cells /
midcap 11,235 (both > v1: 6,767 / 10,301), zero errors either pool. Coverage
gate PASSED: IV median |delta| = 0.0 on ~50k overlapping legs (v2 reproduces
v1's IVs exactly on the same contracts), leg volume 100% on BOTH pools (the
largecap volume gap is closed -> cp_vol/osv become largecap-eligible), OI on
99.9% of legs, far-expiration blocks on 81% (largecap) / 50% (midcap) of
cells. One ops incident, fully recovered: the midcap run initially launched
without its cache-dir/raw-close prerequisites (91-cell junk enumeration,
caught by the cell-count check, v1 backup pristine, corrected + re-run;
run-book amended). v1 files retained beside the new ones for forensics.

## 12. Options-v2 + insider batch sweep — null beyond amihud (946 trials)

**Ran 2026-07-09 (master b508037, registry 43) per the approved batch spec:**
options families (3 new signals + largecap cp_vol/osv enablement + v2-data
hash-replacement re-runs) across the 7 options universes, then the first
insider sweep (3 signals x 28 universes). Journal 835 -> 946 deduped
discovery trials, one leaderboard read.

**RESULT: no new-family signal passes BH at q=0.10.** Best new candidates:
`npr_90` midcap:pharma-chemicals t=+2.62 (p=0.009), `cluster_buys_90`
opt-midcap:trade t=+2.22 — an order of magnitude short of the m=946 bar.
Directional note (recorded, not evidence): the insider purchase signals lean
positive across segments, consistent with the literature's sign, just
statistically indistinguishable at this trial count. `iv_term_slope` (the
flagged-weakest anchor) is inert (best |t|=1.34). The amihud family's rows
persist as known PASSes (battery near-duplicates included, disclosed) and
the tradeable form is already refuted (§11 walk-forward).

**The predicted pseudo-replication artifact materialized exactly as
disclosed:** `droa` largecap:finance shows PASS (t=-3.05, p=0.0023, DSR
0.000) purely because amihud's battery trials moved the BH denominator — it
is WRONG-SIGNED versus its registration (fundamental momentum registered
positive), carries a zero deflated-Sharpe, and per the standing rule earns
nothing without its own battery + holdout (which its sign makes impossible
under the frozen cost gate). Read as an artifact, recorded as such.

**Program state after 946 trials:** one real factor found (amihud,
untradeable long-only in this account), everything else honestly null. The
holdout (2024+) remains fully unspent. v2 data (OI, term structure, largecap
volume) and the insider store are in place for future pre-registered
batches; the engine's cost per hypothesis continues to fall.

## 13. R1 — the long-only gate amendment (2026-07-10, no new trials)

**Not a sweep run — a gate-definition change**, recorded here because it
changes how every number above (and every future promotion decision) is
read. Full rationale: `docs/superpowers/specs/2026-07-10-longonly-gate-
amendment.md`.

**The finding that forced it:** twice now (§10's OPT-1/OPT-2 skew studies,
§11-12's amihud family) the program proved a signal "real" under the
four-factor L/S gate and watched it die in tradeable, long-only form —
amihud's own record above says it plainly: "one real factor found (amihud,
untradeable long-only in this account)." The L/S gate certifies a
construction (equal-dollar long AND short, no financing cost, no spread)
this account cannot trade. "Beating the market" from this seat means: the
long-only portfolio, after realistic spread-based costs, outperforms SPY
buy-and-hold over the identical window. That's what the amended gate now
measures — see the glossary's "the long-only gate" section for the full
term set (Corwin-Schultz spread estimator, spread-based rebalance charge,
SPY benchmark, `--long-only` leaderboard view).

**Prospective, not retroactive:** no already-journaled trial is re-scored.
The existing journal (946 discovery trials, amihud the lone BH survivor
family) is unchanged; `trading alphasearch leaderboard --long-only` re-READS
it under the new lens as a display, never re-journaling anything. The
four-factor L/S regression stays a mandatory diagnostic on every future
candidate (know what you're being paid for) and BH-FDR keeps running on the
L/S p-values exactly as before — it's now a reported property, not the
promotion filter. The robustness battery changes in two ratified ways (see
the R1 spec's Ratifications section): its checks 1-6 keep their frozen
thresholds but re-anchor their statistics onto the cost-charged LO-minus-SPY
ACTIVE return series (raw-LO retention would mostly test whether the market
regime repeated — beta — not the signal's edge over the benchmark), and its
final eligibility comparator moved from "30bps L/S cost row retains t≥2" to
"cost-charged long-only beats SPY on Sharpe and total return."

**Immediate implication for the amihud family:** its L/S alpha remains a
real, statistically significant DIAGNOSTIC finding (persists on the
leaderboard, BH-survivor status unchanged), but per the ALREADY-recorded
§11-12 walk-forward, its tradeable long-only form was refuted before this
amendment existed — the new gate formalizes exactly the judgment that
refutation was already making informally. No park decision changes as a
result of this amendment alone.

### The `--long-only` re-read (2026-07-10, display only — no journaling, no promotion)

Ran `trading alphasearch leaderboard --long-only` over the full journal.
Of 946 trials, **863 re-derived** a cost-charged long-only series from
current data; **83 show honestly as n/a** (data can no longer re-derive the
book — never silently scored 0 or dropped).

**The number that matters most is a warning, not a result: 390 of 863
(45%) "beat SPY" long-only, after Corwin-Schultz costs, over their own
discovery window.** That is the overfitting surface made visible, not
evidence of edge. Selecting a signal's top quintile and holding it over a
bull-heavy 2019–2023 window will beat SPY roughly half the time by
construction; an in-sample beat is a *necessary, nowhere-near-sufficient*
screen. This is precisely why the amended gate is prospective and why
promotion still requires the battery **and** the once-only holdout on top of
this display. The board is "candidates that clear step one," nothing more.

**amihud:midcap tops the board yet is already refuted.** Its in-sample
long-only Sharpe reaches +2.46 (vs SPY +1.12 on that sub-window) with
total returns into the thousands of percent — the microcap-illiquidity
explosion, on names a $1k account cannot actually accumulate. Its OOS
walk-forward (§ above) already failed at 0.16. There is no cleaner
illustration that in-sample leaderboard rank ≠ tradeable edge; the amihud
park decision is unchanged.

**What the long-only lens surfaces that the L/S gate under-weighted**
(recorded as candidates to carry into the R2/R3 work, NOT promotions):
`atm_spread` on **largecap** (+1.38 vs SPY +0.80) and on
opt-largecap:services (+1.45); `vrp` on largecap (+1.29); `mom252` on
midcap (+1.39). These are tradeable large-cap constructions that the
four-factor L/S screen treated as only "suggestive" (§10's vrp/atm_spread
note) — worth a real look under the new objective. **Caveat the board makes
visible:** the `opt-largecap:manufacturing-tech` segment recurs under nearly
every signal near the top — that reads as tech beta over 2019–2021, not
signal, and is a reminder that segment/beta, not the ranker, drives much of
the in-sample outperformance.

## 14. R2 — the strategy-wrapper ablation (2026-07-10): the momentum "refutation" was largely the wrapper

**Pre-registered** in `docs/superpowers/specs/2026-07-10-wrapper-ablation-
design.md` before any cell ran. The question: how much of every "tradeable
form" verdict is the signal, and how much is the inherited momentum-era
wrapper (regime gate, ATR/time stops, entry-score threshold, the M2
sizing/cadence machinery)? Three signals with established life —
`momentum_v1`, `amihud_v1`, `skew_v1` — each run across five wrapper
configurations (W0 bare → W4 full), on their historical universes/spans,
holding everything but the wrapper fixed. All 15 journaled in
`journal/experiments-equities.jsonl`.

**The matrix** (stitched-OOS Sharpe; W4 = the full historical wrapper, the
control that must reproduce the recorded verdict):

| cell | wrapper state | momentum (bmk 0.96) | amihud·midcap (bmk 0.96) | skew (bmk 0.70) |
|---|---|---|---|---|
| **W0** | bare: rank-only, equal-weight, monthly, rank-exit | **0.96** | 0.35 | 0.63 |
| W1 | + regime gate | 0.69 | 0.27 | 0.59 |
| W2 | + M2 stops/sizing bundle (regime off) | 0.70 | 0.42 | 0.64 |
| W3 | + entry threshold (regime off) | 0.69 | 0.44 | 0.83 |
| **W4** | full wrapper (control) | **0.54** | **0.16** | **0.79** |

**W4-reproduction gate — PASSED** (the spec's hard STOP condition: if the
full-wrapper cell can't reproduce the recorded verdict, the instrument is
broken and no cell may be read). Two of three anchors reproduce essentially
exactly: **amihud W4 = 0.16429, bit-identical** to the standalone amihud_v1
walk-forward (§ above); **skew W4 = 0.786** vs the recorded OPT-1 0.7875.
That bit-identity proves the ablation engine with all flags ON ≡ the plain
engine. Momentum W4 = **0.54** vs the §7-recorded **0.45** is the one
non-exact anchor, and it is a documented *data-provenance* shift, not a
harness fault: momentum-w4's config is byte-identical at runtime to
`config/experiments/tiingo`, and the current survivorship-free universe is
more complete than when §7 ran (the run reports 11 missing PIT members vs
§7's 16 — the ticker-alias re-backfill recovered renamed tickers absent in
§7). The instrument is faithful; the momentum baseline itself rose 0.45 →
0.54 as the data completed. (Caveat carried forward: the §7 record's 0.45 is
now stale for this reason.)

**The headline (the pre-registered correction):** the bare momentum
portfolio — top-20 by `momentum_v1` rank, equal weight, monthly rebalance,
sold only on rank exit, **no regime gate, no stops, no entry threshold** —
scores **Sharpe 0.96, matching SPY's 0.96** over the identical stitched-OOS
segments. The full wrapper drags that to 0.54. W0 exceeds the recorded 0.45
by +0.51 — far past the pre-registered +0.1 trigger — so per the protocol
the record gets a **written correction: the recorded momentum verdict
("honest best ~0.45, does not beat SPY, refuted") was substantially a
statement about the WRAPPER, not about momentum.** Held simply, momentum
tracks the market on a risk-adjusted basis; the elaborate risk-management
wrapper is what turned a market-matching signal into a market-lagging one.

**Precise, and not overstated:** matching SPY's *Sharpe* is not *beating*
SPY. Bare momentum's total return (+85.9%) still **trails** SPY's stitched
+124.5% — it matches SPY's risk-adjusted return at lower volatility, it does
not out-return it. The "momentum does not beat SPY" finding stands; what
falls is the "0.45, refuted" framing — that number was the wrapper's, and
bare momentum is a full +0.4 Sharpe above it.

**Component attribution** (read only from clean single-flag pairs; the
pre-run amendment warned that `bare_mode` bundles stops+sizing+cadence, so
per-component isolation of those is not available — disclosed):
- **The M2 stops/sizing/cadence bundle is the dominant drag: W0→W2 =
  0.96 → 0.70, −0.27 Sharpe.** Turning on the simulator's ATR/time stops,
  fractional sizing, deployment caps and cooldown — as a bundle — destroys a
  quarter of the bare Sharpe. (Stops vs sizing cannot be separated here.)
- **Regime gate and entry threshold are each ≈0 alone but −0.16 together.**
  Regime on/off with the threshold off costs nothing (W2→W1: −0.01);
  threshold on/off with regime off costs nothing (W2→W3: −0.005); but both
  on (W2→W4) costs −0.16. They compound: the regime gate cuts market
  exposure at the same time the threshold shrinks the eligible book, so the
  portfolio ends up too small and too defensive at once.
- Whole wrapper, W0→W4: **−0.42 Sharpe.**

**Per-signal readings (each against its own pre-registered logic):**
- **momentum — corrected** (above): wrapper is the drag; bare ties SPY's
  Sharpe.
- **amihud — refutation STANDS, wrapper not the cause:** the wrapper hurts
  amihud too (bare 0.35 → full 0.16, the full wrapper churning stops on thin
  microcaps is maximally destructive), but **even bare, amihud long-only
  (0.35) is nowhere near SPY (0.96)** — it fails the gate in every wrapper
  form. §11-12's long-only refutation is unaffected; the L/S factor remains
  real, the tradeable long-only construction remains refuted.
- **skew — consistent with §9 (edge is tuning/beta, not raw signal):** here
  the wrapper *helps* (bare 0.63 → full 0.79), and **bare skew (0.63)
  underperforms even its own options-pool benchmark (0.70)** — only the
  wrapped/tuned form clears it. That the signal needs the wrapper to look
  good corroborates §9's model-free verdict that skew's monthly long-only
  premium is ≈0 OOS and its apparent edge is stop-tuning plus beta.

**No promotion follows from any cell** — this is instrument calibration, not
discovery (spec §2). What it changes going forward: the construction worth
testing for a long-only account is a **simple** momentum tilt, not the
inherited wrapper — which is exactly what R3 (down-cap universe) and R4
(SPY-plus-tilt paper control) should carry. The holdout remains unspent.

## 15. R3 — down-cap / illiquid universe (Phase A run; cap band NO-GO → dollar-volume fallback GO; sweep pending)

**Pre-registered record + Phase-A results.** Design:
`docs/superpowers/specs/2026-07-10-downcap-universe-design.md`; plan:
`docs/superpowers/plans/2026-07-10-downcap-universe.md`. The code is built and
unit-tested
(`src/trading/venues/universes/downcap_{roster,band,backfill,membership,verify}.py`,
six dedicated test files plus alphasearch panel additions). Phase A (roster +
overnight bar backfill + best-effort companyfacts shares + membership + the
frozen gate) RAN on mac-m1 2026-07-10/11; the Phase-B sweep has not yet run.
The gate verdict is below.

**Pre-registered thesis.** R2 (§14) showed bare momentum only *ties* SPY in
large-cap waters. R3 asks whether the identical simple, wrapper-free momentum
tilt (the alphasearch long-only top-quintile series — equal weight, monthly
rebalance, rank-exit) *beats* SPY in the $50M–$2B down-cap band, where a
~$1k account's smallness is a capacity edge institutions cannot exploit: same
construction, different universe, betting the account's size is an advantage
rather than an irrelevance.

**Frozen construction.**
- **Roster:** survivorship-free — Tiingo's full historical
  `supported_tickers` list (delisted names included), filtered
  *structurally* only (asset type = Stock, currency = USD, major US
  common-stock exchanges), never by performance.
- **Band membership (dynamic, PIT, recomputed every monthly decision date
  D):** a candidate is in-band at D iff ALL of (1) raw-price market cap =
  `shares_outstanding(latest companyfacts row FILED ≤ D) × close_raw(≤ D)` ∈
  **[$50M, $2B]** — the raw/unadjusted price, never the split/dividend-adjusted
  `close` (the same look-ahead class as the fixed `div_yield` bug, §10); (2)
  trailing-63-session Corwin-Schultz effective spread **≤ 2%**; (3)
  trailing-63-session median dollar-volume **≥ $50,000/day**. A candidate
  with no PIT shares is excluded at D — fail-closed, never a guessed cap.
- **Three frozen universes:** `downcap` (full $50M–$2B band), `downcap:small`
  ($300M–$2B), `downcap:micro` ($50M–$300M) — ordinary `UniverseSpec`s
  sharing a fresh `data/equities-downcap-tiingo/` bar cache and a
  `(band, symbol, start, end)` membership CSV (glossary: "Band-membership
  interval CSV").

**Phase-A GO/NO-GO gate (frozen, decided before any sweep; computed by
`downcap_verify.compute_gate` from the diagnostics artifact).** Four
criteria:
1. **Survivorship present:** delisted names ≥ **15%** of in-band
   candidate-months.
2. **Shares-coverage:** ≥ **70%** of tradeable candidate-months have PIT
   shares; below 70% is a NO-GO on the market-cap band, with an automatic,
   developer-pre-approved fallback to a dollar-volume-only band (drop the cap
   bound, keep the two tradeability screens).
3. **Spread realism:** the 2% CS-spread screen must be a real filter (report
   median/IQR/% ≤ 2%), not a no-op or a near-total cull.
4. **Breadth:** ≥ **15** tradeable names in every month of the discovery
   window (2019-01-01..2023-12-31), per universe; any universe with a
   sub-15 month is dropped from the sweep — recorded, never silently
   skipped.

**Phase-A verdict: NO-GO on the market-cap band → automatic developer-
pre-approved dollar-volume-only fallback: GO.** The roster resolved to 15,335
survivorship-free US-common-stock names (structural filter; ~49% delisted);
10,636 were discovery-window candidates. Bars backfilled for 13,295 names
(99.5% Tiingo coverage). The gate (over 403,767 diagnostics candidate-months):
- survivorship 23.1% ≥ 15% ✓
- **shares-coverage 61.7% < 70% ✗ ← the sole failing criterion**
- spread median 0.0002 (2 bps, the CS floor), 100% ≤ 2% ✓ (screen is real but
  these names are liquid enough that spread never binds)
- breadth (min tradeable names/month): downcap 1470, downcap:small 943,
  downcap:micro 509 — all ≫ 15 ✓

**Shares-coverage figure + dropped-names size skew (the §2 survivorship risk,
now measured).** The market-cap band needs PIT shares to compute a cap; those
come from SEC companyfacts, which requires a resolved CIK. Full best-effort
resolution: 5,147/10,636 (48.4%) via SEC `company_tickers.json` (current
tickers) **plus** a dedicated FSDS historical resolution recovering 1,646 of
5,486 delisted names (all verified against submissions JSON; 3,774 were absent
from the FSDS 10-K/10-Q filings entirely — they never filed XBRL, so no shares
are obtainable) → 6,793 CIKs → companyfacts shares for 5,777 names. That still
lands at **61.7%** shares-coverage over tradeable candidate-months. Critically,
the 118,644 dropped (tradeable, no PIT shares) candidate-months skew **smaller**
— median dollar-volume **$1.13M vs $7.08M** for shares-kept months — so
including only shares-having names would bias the cap band toward the larger,
better-documented end. **This is the finding: a survivorship-clean market-cap
band cannot be built from free data for this universe** — the smallest,
most-delisted micro-caps are exactly the ones SEC XBRL / companyfacts don't
cover, and the frozen 70% gate correctly refused rather than ship a
survivor-tilted cap band. (Vintage/listing-date skew is not carried in the
diagnostics artifact, so only the dollar-volume size proxy is reported.)

**The fallback (spec §4, developer-pre-approved) is survivorship-clean by
construction** — it drops the cap bound and keeps only the two tradeability
screens (CS spread ≤ 2%, dollar-volume ≥ $50k/day), which need no shares, so
delisted names are retained on equal footing. Fallback verdict: **GO** —
survivorship 24.7% ≥ 15% ✓, breadth **4,443 tradeable names/month** ✓,
shares-coverage non-gating. This defines a single `downcap-dv` universe.

**Phase-B result (2026-07-11): the momentum thesis is NOT confirmed —
momentum ties SPY here too; the in-sample standouts are illiquidity + beta,
consistent with the overfitting surface, not a new tradeable edge.** The
sweep ran 18 bars-only signals on the `downcap-dv` universe (the fundamentals
family was excluded — `downcap_universes` currently wires `fundamentals_dir`
to the INDEX store, a known bug to fix before any fundamentals sweep on
down-cap; options/insider unsupported). Cost-charged long-only vs SPY over
2019-01-01..2023-12-31 (SPY 0.80 / +106% on the full window):

- **Pre-registered momentum primary — a wash, exactly as R2 (§14) predicted
  for the simple construction.** `mom252` (12-month) 0.74 vs SPY 0.73 (barely
  beats); `mom_12_2` 0.72, `mom126` 0.75, `mom63` 0.51, `mom21` 0.50 — all at
  or below SPY. Simple momentum *matches* the market in the down-cap universe
  just as it does in large-caps; the account's smallness does NOT turn
  momentum into a market-beater. **The R3 thesis is refuted for momentum.**
- **In-sample standouts (necessary-not-sufficient, NOT promotions):**
  `amihud` (illiquidity) 1.25 / +270%, `beta` 1.15 / +144%, `ind_mom` 0.86,
  `vol_trend` 0.83, `mom252` 0.74 clear the beats-SPY bar in-sample. Read them
  skeptically: (a) this is the DISCOVERY window, where the R1 re-read (§13)
  showed ~45% of ALL trials beat SPY — an in-sample beat is the overfitting
  surface, not evidence; (b) `amihud` is the program's one known real factor,
  and its tradeable long-only form was ALREADY refuted out-of-sample in
  mid-caps (0.16, § above) — down-cap changing that is a holdout question, not
  answered here; (c) `beta` beating in a 2019–2021 bull window is a beta tilt,
  not alpha; (d) the microcap-illiquidity artifact is loud — `rev5` posts
  **+3854%** total return at a mediocre 0.49 Sharpe, the classic thin-name
  in-sample explosion, so the large totals (`amihud` +270% included) are not
  tradeable at any real size.
- **BH-FDR** runs across all 964 deduped journaled trials as always; these 18
  are counted. Classical-OLS-SE caveat applies (t inflated 10–30% vs HAC).

**Net:** R3 answers its own question — a simple momentum tilt does NOT beat
SPY in the survivorship-free down-cap universe (it ties, like everywhere
else), so the capacity-edge-for-momentum thesis is refuted. The only thing
that "beats" in-sample is illiquidity (amihud), which the program already
knows is real-but-untradeable-long-only and OOS-refuted, plus beta. No signal
is promoted; none is battery-tested or holdout-checked here. The follow-ups
(if any): a market-neutral amihud construction (its own spec), and fixing the
down-cap fundamentals-store wiring to sweep the fundamentals family.

**Holdout: unspent.** No trial has touched the reserved 2024+ holdout; R3
spends none of it (spec §7). Next: register the `downcap-dv` universe from a
`--no-cap-band` membership and run the Phase-B sweep.

## 16. Concentration axis — the negative verdict was a construction artifact (in-sample): concentrated large-cap momentum BEATS SPY

**This is the most promising in-sample result the program has produced, and it
came from an adversarial audit, not the engine.** After R1/R2/R3 all returned
"momentum ties SPY, never beats it," an independent adversarial reviewer
(2026-07-11) caught that EVERY long-only-vs-SPY number in the program — 1,314
of 1,326 journaled trials — used a top-QUINTILE equal-weight book (20 to
*hundreds* of names), which by construction converges to market beta (that is
*why* R2's bare momentum landed at Sharpe 0.96 = SPY). A ~$1k account holds
~10–20 names. **Concentration was never a search axis, and R3's capacity-edge
thesis was tested with the diluted opposite of the construction it was about.**
Pre-registered amendment (before any run):
`docs/superpowers/specs/2026-07-11-concentration-axis-amendment.md` froze a
fixed-count top-N construction and the exact test (N ∈ {10, 20}; `mom_12_2`,
`mom252`; largecap + `downcap-dv`; the R1 cost-charged long-only-vs-SPY gate).

**Result (cost-charged long-only vs SPY, discovery 2019-01-01..2023-12-31):**

| signal | universe | N | lo Sharpe | SPY | lo total | SPY total | beats |
|---|---|---|---:|---:|---:|---:|:--:|
| mom252 | largecap | **10** | **1.14** | 0.80 | **+373%** | +106% | ✅ |
| mom_12_2 | largecap | 10 | 1.05 | 0.80 | +307% | +106% | ✅ |
| mom252 | largecap | 20 | 1.00 | 0.80 | +223% | +106% | ✅ |
| mom_12_2 | largecap | 20 | 0.97 | 0.80 | +208% | +106% | ✅ |
| mom252 | downcap-dv | 10 | 0.87 | 0.73 | +589% | +91% | ✅ |
| mom252 | downcap-dv | 20 | 0.72 | 0.73 | +221% | +91% | ✗ |
| mom_12_2 | downcap-dv | 10 | 0.63 | 0.73 | +157% | +91% | ✗ |
| mom_12_2 | downcap-dv | 20 | 0.61 | 0.73 | +131% | +91% | ✗ |

**The audit was right.** In large-caps, concentrated momentum beats SPY on
BOTH Sharpe and total return, and the beat is **monotonic in concentration**
(top-10 > top-20 > the quintile that only tied) — exactly the tail-concentration
mechanism (momentum's premium lives in the winner decile; averaging the top
quintile dilutes it toward beta). The recorded "momentum ties SPY" verdict was
a statement about the top-quintile construction, NOT about momentum at the
account's real book. In `downcap-dv`, concentrated momentum crushes SPY on
total return (+131% to +589% vs +91%) but a 10-name micro-cap book is high-vol,
so risk-adjusted it only ties (mom252 top-10 at 0.87 the one exception).

**Why this is a LEAD, not yet an edge — the mandatory caveats:**
- **In-sample.** This is the discovery window (it does include the 2022
  bear, which helps, but it is still in-sample). The R1 re-read (§13) found
  ~45% of ALL trials beat SPY in-sample; an in-sample beat is necessary, not
  sufficient.
- **Concentration amplifies overfitting.** A top-10 book is the 10
  best-in-hindsight-ranked names each month; it is mechanically more sensitive
  to the ranking than a quintile, so *some* of the monotonic top-10 > top-20
  gain is overfitting amplification, not pure tail-concentration edge. The two
  effects are entangled here and only OOS can separate them.
- **The real tests have NOT run.** Every prior candidate died out-of-sample
  (survivorship-free momentum 0.45, amihud 0.16). What is still owed: (1) an
  OOS walk-forward on the concentrated book; (2) the robustness battery —
  which needs `top_n` threading (Piece 3 `robustness.py` was deliberately left
  out of the concentration build, spec §6); (3) only on an explicit developer
  decision, the once-only 2024+ holdout.
- **Concentration cuts both ways** — a 10-name book has real single-name and
  drawdown risk the Sharpe alone doesn't capture.

**Holdout: still unspent.** These 8 top-N trials are journaled (972 deduped),
BH-counted, classical-SE-caveated.

### The OOS resolution (2026-07-11): the concentration lead was in-sample overfitting — it does NOT survive out of sample

The in-sample beat demanded an out-of-sample test before belief. Ran a true
walk-forward (train/test-split, the M2 bare simulator = R2's W0 harness: bare
momentum, regime off, `momentum_v1`, monthly, rank-exit) at
`max_positions` ∈ {20, 10, 5}, on large-cap survivorship-free bars, scored on
stitched OOS segments vs SPY 0.96:

| construction | in-sample (alphasearch) | **OOS (walk-forward)** |
|---|---|---|
| top-20 (R2 W0) | ~0.97–1.00 (beats) | **0.96 — ties SPY** |
| top-10 | 1.05–1.14 (beats most) | **0.76 — FAIL, below SPY** |
| top-5 | — | **0.76 — FAIL, below SPY** |

**Concentration helps in-sample and HURTS out-of-sample — the textbook
overfitting signature.** Holding the signal fixed (`momentum_v1`), tightening
the book 20 → 10 → 5 names *raises* in-sample Sharpe but *drops* OOS Sharpe
from 0.96 to 0.76 (below SPY). The construction that looked best in-sample
(top-10, monotonically increasing) is the worst out-of-sample: fitting the ten
best-in-hindsight-ranked names to the discovery window does not generalize.
The in-sample top-10 beat (1.14) was the overfitting the §16 caveats warned
about, now measured. (Minor confound: the OOS harness uses the `momentum_v1`
composite vs the in-sample `mom252`/`mom_12_2`; but at top-20 both engines
agree (~0.96–1.0) and the same-signal 20→10→5 OOS decline is clean, so the
divergence is concentration/overfitting, not the signal.)

**Verdict: the negative conclusion STANDS, now properly tested at the account's
real construction.** The adversarial audit was right that concentration had
never been tested — and testing it correctly (in-sample AND out-of-sample)
confirms rather than overturns the result: a concentrated long-only momentum
book does not beat SPY out of sample; it does worse. This is the
pre-registration + OOS discipline doing exactly its job — catching an
in-sample artifact before it became a live bet. **The holdout was NOT spent**
(a candidate that fails the OOS walk-forward does not warrant it) and remains
reserved. The audit's other two leads are untouched by this and remain the
open frontier: a holding-horizon axis (their §9 skew study: OOS tercile spread
t=0.04 at 21d but t=3.2 at 63d — the monthly-only engine is blind to
medium-horizon anomalies) and a PEAD / earnings-surprise signal on the
point-in-time earnings calendar already being accumulated.

## 17. Skew traded natively (options) — dies on execution costs (the last lead, closed)

After the concentration lead resolved as overfitting (§16), a second adversarial
review made the sharpest structural argument in the program: (a) a long-only US
equity book is ~90% market beta, so the residual cross-sectional premium any
factor leaves can't clear SPY's bull-market Sharpe net of forced beta — which is
why every construction (momentum quintile/decile/top-N, amihud, value, quality,
skew-long-only) lands in the same 0.45–0.96 band around SPY; and (b) the go-live
gate is a naked `lo_sharpe > spy_sharpe` point comparison with NO error bars,
while over a ~4.5yr OOS window the standard error of an annualized Sharpe is
≈0.5 — so 0.96, 0.76, 0.45 and SPY are all within ~1 SE of each other. **The
whole promote/refute ledger has been decided on Sharpe differences we lack the
data to measure.** Rigor was spent on the in-sample diagnostic (BH-FDR, DSR),
not on the gate that matters. Both points say: the long-only-equity-tilt question
is answered AND statistically unmeasurable — more signals is motion, not progress.

The one remaining lead the review endorsed: **skew is the only signal in the
program with model-free, statistically-significant power (§9: t=3.2 at 63d, 2× in
stress), and it was never traded in its native instrument (options) — we trade
the stock, monthly, long-only, the exact construction §9 proved buries it.** The
account can trade options directly. So we made the one clean pass — starting, per
the review's own first pre-commitment, with the decisive killer: realistic
options execution cost, measured from the ThetaData bid/ask we already gathered
(7,399 largecap + 11,235 midcap cells).

**Result — option bid/ask spread as % of the option's mid (the in/out cost):**
best case is a large-cap ATM call at **3.8% median (7.6% round-trip)**; large-cap
OTM calls 9.9% median; mid-cap ATM 9.5% (median volume 5 contracts, OI 94 — not
fillable); mid-cap OTM 21.8% median, 115% at p90. **The best-case round-trip
execution cost (~7.6%) is roughly double the entire skew premium it would harvest
(~3.5%/quarter, §9)** — before theta, before the option's embedded time premium
(a ~3.5% predicted quarterly move can't overcome a 63-day ATM option's time value
+ spread), before the tail. And the crisis-tilt that makes skew real (2× in
high-vol) coincides with the widest option markets, so the signal is strongest
exactly where it's least harvestable. No options expression escapes it (buying
convexity pays the spread in stress; selling premium is short-vol crushed in the
same stress, and unsuitable risk on $1k).

**Verdict: the amihud trap in its purest form — a real signal whose only native
instrument costs more than the edge. The skew-in-options lead is closed on a
data-backed execution-cost check, without building a backtest (which would only
confirm it at cost). This closes the last lead. The disciplined terminal finding:
across every construction and both instruments, this account has no harvestable
long-only-or-options edge that beats SPY net of realistic costs — and the
differences involved are, in any case, below the statistical resolution of the
available data. The engine and the discipline were the deliverables; the P&L
answer is "index it." Holdout: never spent.**

## 18. R6 Stage 1 — shorting (market-neutral) does NOT rescue amihud; the "edge" is the illiquidity artifact

Assuming a shorting-capable venue (IBKR-class), we built a **market-neutral
(long/short) gate WITH error bars** — the engine's `ls` series (long top, short
bottom), both legs cost-charged (Corwin-Schultz) + a PIT short-borrow model,
benchmark cash, promoted only if the **95% bootstrap Sharpe CI lower bound > 0**
on the full window AND both discovery halves (spec
`2026-07-11-market-neutral-gate-amendment.md`; the CI is the statistical-power
fix §17 exposed). This directly tests what the long-only Robinhood constraint
forbade, on the pre-registered primary `amihud` (the lone BH survivor, refuted
only long-only).

**Result — it passes on illiquid universes and fails on liquid ones, which is
the tell.** Market-neutral amihud:
- **largecap (LIQUID, tradeable): 0/13 pass** — plain `amihud:largecap` Sharpe
  +0.62, CI **[−0.22, +1.52]** (includes zero), total +62%. On names whose
  recorded returns are actually achievable, shorting unlocks **nothing** — a
  statistical wash.
- **midcap (illiquid): Sharpe +2.83, CI [+2.09, +4.26], "passes"** — but the
  diagnostics expose it: **+1,723% total return over 5 years (~78%/yr),
  ~50–108% EVERY year, at 8 bps modeled spread cost + 60 bps borrow.** A
  market-neutral book returning 78%/yr at Sharpe 2.8 with 8 bps of cost is
  physically impossible for a real premium. It is the **microcap-illiquidity
  artifact**: the illiquid long leg's close-to-close returns are bid-ask bounce
  / stale-price noise a trader cannot capture (the same artifact that inflated
  long-only amihud to +270% and `rev5` to +3854%, §15). The "edge" scales with
  illiquidity (midcap median total +331%, opt-midcap:trade +534%) — the more
  un-tradeable the universe, the bigger the mirage.

**Two findings.** (1) **Shorting does not rescue our one real factor.** On
liquid, tradeable names amihud market-neutral is indistinguishable from zero;
the constraint was never the long-only leg — it's that amihud's premium lives
in illiquid names whose returns aren't achievable, and shorting the liquid leg
doesn't change that. The pre-registered base case (§5) is confirmed. (2)
**Methodological catch — the gate has an illiquidity blind spot.** Both the
long-only and market-neutral gates charge the *quoted* spread but trust the
*recorded returns* of illiquid names, which overstate what's achievable. The
gate is trustworthy on LIQUID universes (where amihud correctly fails) and
foolable on illiquid ones (the +1,723% "pass"). Before trusting any
market-neutral result from an illiquid universe, it needs an illiquidity/impact
haircut or a hard liquid-tradeability restriction — a real limitation now
documented.

**Net:** the expanded (shorting) capability was tested rigorously with proper
error bars, and it does not surface a tradeable edge on liquid names; the one
"passing" candidate is the known illiquidity artifact, correctly diagnosed and
not promoted. Holdout: not spent (a candidate that only "works" via
un-achievable illiquid returns does not warrant it). The multi-leg options /
VRP track (R6 Stage 2) remains the other unexplored capability.

## Known caveats affecting these numbers

- **Survivorship bias** (being measured by exp 7): experiments 0–6 ran on
  yfinance's survivor-only universe. The comparison *between* them is still
  valid (all share the bias); their absolute levels are likely optimistic.
- **Membership double-listing**: three companies dual-listed in the S&P 500 and
  NASDAQ-100 are labeled with their *old* ticker in one index and *new* ticker
  in the other across a rename (FB/META, FISV/FI, WLTW/WTW), so each appears
  twice in the ranking. This affects all runs equally (so it doesn't distort
  their comparison) and is flagged for a resolution-aware-universe-dedup
  follow-up (see `trading.symbols.resolution_collisions`).

## Not yet run / candidate experiments

- **Liquidity-floored, survivorship-free breadth** — expand beyond the S&P 500
  by trailing dollar volume (a hard liquidity floor) rather than by adding an
  index, tested against the exp 4 negative result. Only worth it if exp 7 shows
  the core edge survives. (Superseded/formalized by §15 R3's frozen down-cap
  band construction, which has since been built but not yet run.)
- **Earnings-aware entry blackout** — the earnings filter was dropped (stale
  yfinance dates); a point-in-time earnings history is now being accumulated
  from the Robinhood calendar to reinstate and backtest it.
- **Crypto walk-forward** — the crypto venue paper-trades but has never had a
  deep-history walk-forward (Kraken's ~720-candle retention; deep history now
  splices from Coinbase). Its go-live additionally requires
  `fee_drag_vs_gross < 30%`.
- **Long/short skew spread** (the priority follow-up to §9) — long the lowest-skew,
  short the highest-skew names, market-neutral, to harvest the skew premium that the
  long-only book (§9) buries under market beta. skew_v1 already beats momentum on the
  same names; the question is whether the spread earns clean alpha above SPY.
- **OPT-3 — skew × momentum overlay** — use the skew signal as a filter/tilt on a
  momentum (or other) core, now that skew is an established differential signal.
- **OPT holdout** — once 2026 option cells are gathered, evaluate the reserved
  ≥2026-01-05 holdout ONCE with the frozen OPT-1 params (go-live gate step 2).
- **Skew robustness** — broaden beyond 100 names (survivorship-free breadth), test
  daily vs monthly decision frequency, and sensitivity to the handful of
  implausible-IV cells.
