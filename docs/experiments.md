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
5. **The open strategic question** is no longer "does fundamentals help" but
   "does this momentum core have enough of an edge to be worth overlaying at
   all." The one fundamentals idea with a live thesis in a survivorship-free
   world is a *quality/junk filter* (exp #1 in the candidate list) — precisely
   because the delisted failures now dragging the result down are what such a
   screen would remove. Everything else is lower priority given the 0.45 core.

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
