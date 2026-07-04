# Momentum Swing Trading System — Design Spec

**Date:** 2026-07-04
**Status:** Approved design, pending implementation plan

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
| Universe | Equities: S&P 500 + Nasdaq 100, dollar-volume filtered. Crypto: Robinhood-listed (~83 pairs) |
| Signals | Lean core: price/volume momentum + market-regime gate. Sentiment/on-chain are v2 candidates gated on backtest evidence |
| Runtime | Cron/launchd Python pipeline + daily digest; Claude sessions for analysis and (later) live order execution |
| Architecture | Backtest-first: one signal/simulation engine shared by backtest and live-paper modes |
| Interface | Single CLI with concise README; digest is a 60-second read |

## Architecture

Single Python package, four layers, strictly separated:

```
Data Layer ──▶ Signal Engine ──▶ Portfolio Simulator ──▶ Reporting
fetch/cache     pure functions     rules → trades         journal/digest
OHLCV           bars → scores      per-venue state
```

- **Stack:** Python 3.12, pandas, ccxt, yfinance, pyarrow. No framework, no
  DB, no web UI.
- **State:** per-venue paper portfolio (JSON) + append-only journal (JSONL).
- **Determinism:** signal engine and simulator are pure — no I/O, no clock
  access; "now" is always a parameter. Backtest and live-paper run identical
  code, differing only in the timestamp handed in.

## Venue Model (constraints verified 2026-07-04)

Each venue adapter implements: `universe()`, `constraints()`,
`fetch_ohlcv(symbol, timeframe)`.

### Equities venue
- **Universe:** S&P 500 + Nasdaq 100 constituents, min dollar-volume floor
  (config; target a few hundred names).
- **Data:** daily OHLCV via yfinance (swappable behind adapter interface).
- **Costs:** 5 bps slippage assumption; zero commission.
- **Rules modeled:** market hours; T+1 cash-account settlement (sale proceeds
  unspendable until next session); earnings blackout — no entry within 5
  sessions of a report (MCP earnings calendar feeds this).
- **Order types available via Robinhood MCP (verified against tool schemas):**
  market, limit, stop_market, stop_limit; gfd/gtc. Fractional shares ONLY on
  market orders in regular hours; resting stops require whole shares.
- **Consequence:** entries are dollar-based market orders at next open; exits
  are market orders at next open after a close-based trigger. No resting stop
  orders in v1 (fidelity between backtest and live behavior beats intraday
  protection at $180 position scale).

### Crypto venue
- **Universe:** Robinhood-listed coins (~83 as of 2026-07; sync from a
  maintained list, never hardcode; model `sell_only`/`untradable` states).
- **Data:** daily + 4h OHLCV via ccxt from Kraken/Bitstamp (Bitstamp is
  Robinhood's actual exchange-routing venue).
- **Costs (June 2026 fee schedule, must be charged on every paper fill):**
  exchange routing 0.95% taker / 0.50% maker at <$10K 30-day volume, plus
  5 bps slippage. Fee schedule is config so backtests can model higher tiers.
- **Rules modeled:** 24/7 clock; no PDT; fee-adjusted entry threshold
  (projected edge must exceed ~3× round-trip cost).
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

**Composite score** = weighted sum of feature percentiles. Weights in config,
start equal, tuned only via walk-forward backtest. Output is a ranked table
with sub-scores — every pick explainable in the journal.

**Regime gate** (per venue): benchmark trend (SPY / BTC) vs 50/200-day
averages + realized-vol level → exposure multiplier: risk-on = full,
neutral = half, risk-off = no new entries (exits still honored).

**Pre-ranking filters:** equities — earnings within 5 sessions, dollar-volume
floor. Crypto — non-tradable states, fee-adjusted threshold (~6% projected
move vs ~1% equities).

## Portfolio Simulator & Risk Rules

One simulator, instantiated per venue with own config + state file.

| | Equities | Crypto |
|---|---|---|
| Max positions | 5 | 3 |
| Position size | ~18% | ~30% |
| Starting paper balance | $1,000 | $1,000 |
| Run cadence | Weekday evenings after close | Every 8 hours |
| Time stop | 20 sessions | 30 days |

**Entries** (each run, after exits): regime-gated top of ranking; require
score ≥ threshold, not already held, not stopped out within 7-day cooldown,
free slot, settled cash available.

**Exits** (checked before entries):
1. Stop-loss: close beyond 1.5× 20-day ATR from entry → market exit next open.
2. Trend break: falls out of top half of ranking AND closes below 20-day mean.
3. Time stop: flat-to-down at limit → exit (dead money blocks a slot).
4. Regime flush: risk-off flips stops to 1.0× ATR; no forced liquidation.

**Fill model:** decisions use data through yesterday's close; fills at next
open (equities) / next 4h bar (crypto) + venue costs. Limit fills count only
if the bar traded through the price. No same-bar round trips, no lookahead.

**Hard rails (paper and live):** never average down; never exceed position
count; no margin; max 25% of portfolio into new entries per day.

Every number above is config, not constant.

## Execution Split (live path)

1. **Pipeline** (cron): fetch → rank → update paper book → journal + digest.
   In live mode it additionally writes `pending_orders.json` (intended
   trades); it never talks to a brokerage.
2. **Paper mode:** pipeline fills its own orders against next-bar prices.
3. **Live mode (equities, post-validation):** a scheduled Claude session
   reads `pending_orders.json`, sanity-checks, and executes via Robinhood MCP
   (review_equity_order → user-visible approval → place_equity_order) in the
   Agentic account (••••6655). The agent executes the pipeline's decisions
   verbatim and reports discrepancies; it never originates trades.

## Backtesting & Validation

- **Span:** 2022-present (must include a bear market).
- **Walk-forward:** tune on 12-month window, test on following 3 months
  untouched, roll. Report stitched out-of-sample segments only.
- **Metrics per venue:** total/annualized return, max drawdown, Sharpe, win
  rate, avg win/loss, turnover, fee drag (own line), vs buy-and-hold
  SPY/BTC benchmark.
- **Overfitting defenses:** parameter-plateau sensitivity checks; every
  experiment logged to journal (config hash + results); stated survivorship
  haircut (~1–2%/yr) on equity results (free data under-represents
  delistings).
- **Go-live criteria (equities):** out-of-sample beats SPY risk-adjusted AND
  ~4 weeks live-paper consistent with backtest expectations. Then $1,000
  scale. **Crypto additionally:** fee drag < 30% of gross returns; paper-only
  until agentic crypto GA regardless.

## Reporting & Operations

- **Journal:** per-venue append-only JSONL. Every run: timestamp, regime,
  full ranking + sub-scores, decisions made AND skipped (with reasons),
  fills, portfolio snapshot, config hash. Backtest experiments land here too.
- **Digest:** `digest/YYYY-MM-DD.md` after the evening run. P&L vs benchmark
  per venue, open positions with rationale + distance-to-stop, trades, top-5
  ranking, regime, warnings. 60-second read.
- **Failure behavior:** a run that cannot fetch fresh data writes a loud
  warning and touches nothing. Runs are idempotent; stale-data guard
  self-skips. Never trade on a stale view.
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
trading status                           # both portfolios, P&L vs benchmark
trading rankings --venue equities|crypto # current ranked table w/ sub-scores
trading digest [--date YYYY-MM-DD]       # print digest (default: latest)
trading schedule install|status|remove   # manage launchd jobs
```

Every command prints human-readable tables to stdout; `--json` flag for
machine consumption. README covers: what the system does (3 sentences),
setup (uv/pip install), the 6 commands, where state lives, and how to read
the digest. No other documentation required reading.

## Error Handling

- Data fetch failure → skip run, warn in digest, exit nonzero.
- Partial universe fetch (some symbols fail) → proceed if ≥90% coverage,
  excluded symbols listed in journal; below 90% → skip run.
- Corrupt/missing state file → refuse to run, never regenerate silently.
- Venue clock checks: equities run outside market-calendar trading days
  no-ops cleanly (holidays).

## Testing Strategy

- **Unit:** signal functions against hand-computed fixtures; simulator rules
  (settlement, cooldown, rails) against scripted scenarios.
- **Property:** no-lookahead test — perturbing data after time T must never
  change decisions at T; fills always within next bar's high/low.
- **Golden backtest:** small frozen fixture dataset with committed expected
  output; CI fails if results drift unintentionally.
- **Venue adapters:** contract tests both venues must pass (including crypto,
  so the dormant-live path can't rot).

## Out of Scope (v1)

Sentiment/news signals, on-chain/derivatives data, intraday trading, resting
stop orders, options, notification delivery (email/push), web UI, live crypto
execution, margin.

## Open Items

- Robinhood Crypto API `is_api_tradable` subset + per-pair min order sizes:
  confirm when/if API keys are created.
- Agentic crypto GA date and its constraints: watch Robinhood announcements.
- Digest delivery beyond repo file: decide after 2 weeks of use.
