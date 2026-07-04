# Momentum Swing Trading System — Design Spec

**Date:** 2026-07-04
**Status:** Approved design, revised after adversarial review (same day), pending implementation plan

## Purpose

A scheduled, deterministic system that ranks liquid assets by likelihood of
near-term (days-to-weeks) upward moves and paper-trades that ranking, with a
validated path to live execution in Travis's Robinhood Agentic account.
Equities are the priority venue (live execution possible today via the
Robinhood MCP); crypto runs in parallel as paper-only until Robinhood's
agentic crypto support reaches GA.

Claude's role is builder/analyst — the trading loop itself is plain Python
with no model calls, fully backtestable and cheap to run.

## Decisions (from brainstorming, 2026-07-02..04)

| Decision | Choice |
|---|---|
| Execution | Paper trade both venues first; equities go live first |
| Horizon | Swing: 2–20 day holds |
| Universe | Equities: S&P 500 + Nasdaq 100 (point-in-time membership), dollar-volume filtered. Crypto: Robinhood-listed (~83 pairs) |
| Signals | Lean core: price/volume momentum + market-regime gate. Sentiment/on-chain are v2 candidates gated on backtest evidence |
| Runtime | Daily cron/launchd Python pipeline + daily digest; Claude sessions for analysis and (later) live order execution |
| Architecture | Backtest-first: one signal/simulation engine shared by backtest and live-paper modes |
| Interface | Single CLI with concise README; digest is a 60-second read |

## Architecture

Single Python package, four layers, strictly separated:

```
Data Layer ──▶ Signal Engine ──▶ Portfolio Simulator ──▶ Reporting
fetch/cache     pure functions     rules → trades         journal/digest
OHLCV           bars → scores      per-venue state
```

- **Stack:** Python 3.12, pandas, ccxt, yfinance (version-pinned), pyarrow.
  No framework, no DB, no web UI.
- **State:** per-venue paper portfolio (JSON) + append-only journal (JSONL).
  State writes are atomic (temp file + rename). Journal snapshots make state
  reconstructible by replay; a corrupt state file is rebuilt from the journal
  with explicit operator confirmation, never regenerated silently.
- **Determinism:** signal engine and simulator are pure — no I/O, no clock
  access; "now" is always a parameter. Backtest and live-paper run identical
  code, differing only in the timestamp handed in.
- **Time:** all timestamps UTC everywhere (state, journal, digest filenames).
  Crypto trading day boundary = 00:00 UTC. Equities decisions keyed to the
  official NYSE session calendar.
- **Idempotency (designed, not assumed):** each run computes a run-key
  `(venue, decision-bar timestamp)` and consults the journal before acting —
  a bar that has already been traded is never traded again (including
  partial re-runs entering *different* symbols). A lockfile prevents a manual
  `trading run` racing the scheduled job.

## Venue Model (constraints verified 2026-07-04)

Each venue adapter implements: `universe(as_of)`, `constraints()`,
`fetch_ohlcv(symbol, timeframe)`.

### Equities venue
- **Universe:** S&P 500 + Nasdaq 100 **point-in-time** membership (free
  historical constituent datasets; membership evaluated as-of each backtest
  date), min dollar-volume floor (config). Backtesting today's members over
  past years is prohibited — index inclusion is itself a momentum artifact
  and inflates results for exactly this strategy class.
- **Data:** daily OHLCV via yfinance behind the adapter interface —
  version-pinned, `auto_adjust` set explicitly, adjusted OHLC used
  everywhere (signals, stops, fills) for internal consistency. The trailing
  window is re-fetched every run (corporate actions rewrite adjusted
  history; cache is treated as ephemeral for recent data). Planned v2 swap:
  Robinhood `get_equity_historicals` — the actual execution venue's data.
- **Earnings dates:** sourced programmatically (yfinance earnings calendar)
  in both live and backtest modes, using point-in-time dates for backtests.
  If that source proves unreliable in practice, the filter is dropped
  entirely (both modes) and the divergence documented — a filter that exists
  live but not in backtest is worse than no filter.
- **Costs:** 5 bps slippage assumption; zero commission.
- **Rules modeled:** market hours per NYSE calendar; T+1 cash-account
  settlement (sale proceeds unspendable until next session).
- **Order types available via Robinhood MCP (verified against tool schemas):**
  market, limit, stop_market, stop_limit; gfd/gtc. Fractional shares ONLY on
  market orders in regular hours; resting stops require whole shares.
- **Consequence:** entries are dollar-based market orders; exits are market
  orders after a close-based trigger. No resting stop orders in v1.

### Crypto venue
- **Universe:** Robinhood-listed coins (~83 as of 2026-07; sync from a
  maintained list, never hardcode; model `sell_only`/`untradable` states).
- **Data:** daily OHLCV (UTC close) via ccxt from Kraken/Bitstamp (Bitstamp
  is Robinhood's actual exchange-routing venue). Intraday bars deferred to
  v2 with intraday cadence.
- **Costs (June 2026 fee schedule, charged on every paper fill):** exchange
  routing 0.95% taker / 0.50% maker at <$10K 30-day volume, plus 5 bps
  slippage. Fee schedule is config so backtests can model higher tiers.
  v1 places market (taker) orders only.
- **Rules modeled:** 24/7 clock (UTC day boundary); no PDT; fee-adjusted
  entry gate (defined precisely under Signal Engine).
- **Execution path (future):** Robinhood Crypto Trading API (market, limit,
  stop_loss, stop_limit; 100 req/min) or agentic MCP crypto when GA
  (announced 2026-07-01, "rolling out soon"). Paper-only until then.

## Signal Engine

Pure functions, shared across venues; config differs per venue. All features
normalized to cross-sectional percentiles.

1. **Momentum:** returns over 5/20/60 trading days (crypto: 7/30/90 calendar
   days), volatility-adjusted (return ÷ realized vol).
2. **Volume surge:** current week dollar volume vs trailing 3-month average.
3. **Breakout proximity:** distance from 20-day/60-day highs.
4. **Overextension guard (negative):** RSI-style stretch + distance above
   20-day mean.

**Composite score** = weighted sum of feature percentiles. **Weights are
fixed equal in v1** — the training sample (30–80 trades per window) cannot
support fitting them; unequal weights require v2-scale evidence. Output is a
ranked table with sub-scores — every pick explainable in the journal.

**Regime gate** (per venue): benchmark trend (SPY / BTC) vs 50/200-day
averages + realized-vol level → exposure multiplier: risk-on = full,
neutral = half, risk-off = no new entries (exits still honored).

**Pre-ranking filters (entries only — see exit note in Simulator):**
equities — earnings within 5 sessions, dollar-volume floor. Crypto —
non-tradable states, and the **fee-adjusted entry gate**: the symbol's raw
(un-normalized) 30-day return must exceed 3× the round-trip cost
(≈ 6% at the v1 fee tier). Defined on raw momentum, not the percentile
score, so it is directly implementable and backtestable.

## Portfolio Simulator & Risk Rules

One simulator, instantiated per venue with own config + state file.

| | Equities | Crypto |
|---|---|---|
| Max positions | 5 | 3 |
| Position size | ~18% | ~30% |
| Starting paper balance | $1,000 | $1,000 |
| Run cadence | Weekday evenings after NYSE close | Daily, after 00:00 UTC |
| Time stop | 20 sessions | 30 calendar days |

**Entries** (each run, after exits): regime-gated top of ranking; require
score ≥ threshold, not already held, not stopped out within 7 calendar-day
cooldown, free slot, settled cash available.

**Exits** (checked before entries, **against the unfiltered ranking** — held
names always remain rankable even when entry filters would exclude them,
so filter mechanics can never manufacture a spurious exit):
1. Stop-loss: close beyond 1.5× ATR from entry, **ATR frozen at entry**
   (rolling ATR silently widens stops in rising volatility) → market exit
   next bar.
2. Trend break: falls out of top half of unfiltered ranking AND closes below
   20-day mean.
3. Time stop: flat-to-down at limit → exit (dead money blocks a slot).
4. Regime flush: risk-off recomputes each stop once at 1.0× the frozen
   entry ATR (one-way ratchet; never loosened until position closes).
5. Forced states: symbol goes `sell_only`/delisted → exit next bar, logged
   as a forced exit distinct from signal exits.

**Fill model:** decisions use data through the last completed bar; fills at
next session open (equities) / next UTC daily bar (crypto) + venue costs.
v1 is market-orders-only; the limit/maker fill model is v2 alongside any
maker-order strategy. No same-bar round trips, no lookahead.

**Hard rails (paper and live):** never average down; never exceed position
count; no margin; max 25% of portfolio into new entries per day.

**Circuit breakers & sanity checks:**
- Portfolio drawdown > 20% from high-water mark (config) → halt all entries,
  venue-wide, until manual reset via CLI. Applies in paper and live.
- Per-symbol data sanity: a day-over-day move beyond a config bound (default
  40%) without a known corporate action quarantines the symbol (no trades,
  warn in digest) until it passes or is manually cleared.

Every number above is config, not constant.

## Execution Split (live path)

1. **Pipeline** (cron): fetch → rank → update paper book → journal + digest.
   In live mode it additionally writes `pending_orders.json` (intended
   trades); it never talks to a brokerage.
2. **Paper mode:** pipeline fills its own orders against next-bar prices.
3. **Live mode (equities, post-validation):** a scheduled Claude session
   reads `pending_orders.json`, sanity-checks, and executes via Robinhood MCP
   (review_equity_order → user-visible approval → place_equity_order) in the
   Agentic account (••••6655). Orders are submitted **before the open** as
   queued market orders so live fills correspond to the open prints the
   backtest models. The agent executes the pipeline's decisions verbatim and
   reports discrepancies; it never originates trades.
4. **Missed/late runs:** launchd coalesces missed jobs on wake. A late run
   still processes **exits**; it skips **entries** if more than a config
   staleness bound past the decision bar (default: 2 hours after open for
   equities, 6 hours past UTC midnight for crypto). Skipped-late entries are
   journaled as such. No catch-up trading of older bars, ever.

## Backtesting & Validation

- **Span:** 2018-present. The OOS stitch must include at least one stress
  segment (2022 bear; April 2025 drawdown) — a bear market that only ever
  appears in training data does not count as tested.
- **Walk-forward:** tune on a 24–36 month window, test on the following
  3 months untouched, roll. Report stitched out-of-sample segments only.
- **Tunable surface (v1):** exactly two hyperparameters — entry score
  threshold and stop ATR multiple. Feature weights stay equal (see Signal
  Engine). Everything else is set by design, not fitted.
- **Final holdout:** the most recent 6 months are touched **exactly once**,
  as the last step before the go-live decision. Journal records every
  experiment run against the walk-forward OOS (config hash + results); the
  experiment count is reported alongside any result quoted as evidence —
  50 experiments deep, "OOS beats SPY" is selection, not signal.
- **Gate metric (defined):** annualized Sharpe of daily returns of the
  stitched OOS equity curve vs SPY total-return over the identical period,
  cash yielding 0%. "Beats" = higher Sharpe AND positive total return.
- **Metrics reported per venue:** total/annualized return, max drawdown,
  Sharpe, win rate, avg win/loss, turnover, fee drag (own line), vs
  buy-and-hold SPY/BTC benchmark.
- **Survivorship:** handled structurally via point-in-time universe
  membership (see Venue Model), not via an invented haircut. Residual data
  gaps (delisted tickers missing from yfinance) are logged and counted, and
  results are annotated with the coverage ratio.
- **Go-live criteria (equities):**
  1. Walk-forward OOS (including a stress segment) passes the gate metric.
  2. Final holdout, evaluated once, is consistent with OOS results.
  3. ~4 weeks of live-paper as a **plumbing shakeout, not a performance
     test** (5–10 trades prove nothing statistically): live-paper decisions
     must match a same-day simulator replay 1:1, and simulated fills must
     sit within the slippage tolerance of observed prices.
  Then real money at $1,000 scale. **Crypto additionally:** fee drag < 30%
  of gross returns; paper-only until agentic crypto GA regardless.

## Reporting & Operations

- **Journal:** per-venue append-only JSONL. Every run: run-key, timestamp,
  regime, full ranking + sub-scores, decisions made AND skipped (with
  reasons), fills, portfolio snapshot, config hash. Backtest experiments
  land here too.
- **Digest:** `digest/YYYY-MM-DD.md` (UTC date) after the evening run. P&L
  vs benchmark per venue, open positions with rationale + distance-to-stop,
  trades, top-5 ranking, regime, warnings (including quarantined symbols and
  circuit-breaker state). 60-second read.
- **Failure behavior:** a run that cannot fetch fresh data (or achieves
  < 90% universe coverage) writes a warning and touches nothing. **Every
  failed or skipped run fires a macOS notification** (osascript /
  terminal-notifier on nonzero exit) — a silent dead pipeline is the most
  likely real-world failure, so failure signaling is v1, not v2.
  `trading status` always shows time-since-last-successful-run per venue.
- **Roles:** pipeline decides (deterministic, daily); Travis skims digest
  (optional, 60s); Claude analyzes journal, proposes config experiments,
  backs them with backtest evidence, and executes live orders post-go-live.
  No hand-tuning live config without backtest evidence.

## CLI & README (user interface)

Single entry point (`trading`, via console script), concise README at repo
root documenting exactly these commands:

```
trading backtest --venue equities|crypto [--from DATE] [--to DATE]
trading run --venue equities|crypto      # one live-paper cycle now
trading status                           # portfolios, P&L vs benchmark, last-run health
trading rankings --venue equities|crypto # current ranked table w/ sub-scores
trading digest [--date YYYY-MM-DD]       # print digest (default: latest)
trading schedule install|status|remove   # manage launchd jobs
trading reset-breaker --venue VENUE      # manual circuit-breaker reset (confirms)
```

Every command prints human-readable tables to stdout; `--json` flag for
machine consumption. README covers: what the system does (3 sentences),
setup (uv/pip install), the commands above, where state lives, and how to
read the digest. No other documentation required reading.

## Error Handling

- Data fetch failure → skip run, warn, notify, exit nonzero.
- Partial universe fetch → proceed if ≥90% coverage, excluded symbols listed
  in journal; below 90% → skip run + notify.
- Corrupt/missing state file → refuse to run, notify; recovery = journal
  replay with operator confirmation.
- Venue calendar: equities runs on non-trading days no-op cleanly.
- Stale/late run semantics as specified under Execution Split.

## Testing Strategy

- **Unit:** signal functions against hand-computed fixtures; simulator rules
  (settlement, cooldown, rails, circuit breaker, frozen-ATR stops) against
  scripted scenarios.
- **Property:** no-lookahead test — perturbing data after time T must never
  change decisions at T; fills always within next bar's high/low.
- **Golden backtest:** small frozen fixture dataset with committed expected
  output; CI fails if results drift unintentionally.
- **Venue adapters:** shared contract tests for the *data* interface (both
  venues run them in v1, since both paper-trade from day one). Live
  execution contract tests arrive with the live crypto path itself.

## Out of Scope (v1)

Sentiment/news signals, on-chain/derivatives data, intraday trading and
intraday crypto bars, resting stop orders, maker/limit-order strategies and
their fill model, options, email/push digest delivery (macOS failure
notifications ARE in v1), web UI, live crypto execution, margin, tuned
feature weights.

## Open Items

- Robinhood Crypto API `is_api_tradable` subset + per-pair min order sizes:
  confirm when/if API keys are created.
- Agentic crypto GA date and its constraints: watch Robinhood announcements.
- Point-in-time constituent dataset selection (several free options; pick at
  implementation time and record provenance in the repo).
- yfinance earnings-date reliability: evaluate during implementation; drop
  the earnings filter (both modes) if unreliable.
- Digest delivery beyond repo file: decide after 2 weeks of use.
