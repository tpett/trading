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
  the core edge survives.
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
