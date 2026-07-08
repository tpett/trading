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
