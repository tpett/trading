# trading

Momentum swing trading system. It ranks liquid assets (S&P 500 + Nasdaq-100
equities; Robinhood-listed crypto) by likelihood of near-term upward moves
using price/volume momentum behind a market-regime gate, and paper-trades
that ranking daily with $1,000 per venue under strict risk rules. Backtesting
and walk-forward validation arrive in the next milestone.

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and macOS (for
notifications and launchd scheduling):

    uv sync

## Commands

    uv run trading run --venue equities|crypto   # one live-paper cycle now
    uv run trading status                        # portfolios, P&L vs benchmark, last-run health
    uv run trading rankings --venue equities|crypto  # current ranked table w/ sub-scores
    uv run trading digest [--date YYYY-MM-DD]    # print digest (default: latest)
    uv run trading schedule install|status|remove    # manage launchd jobs
    uv run trading reset-breaker --venue VENUE   # manual circuit-breaker reset (confirms)

Every command prints human-readable tables; add `--json` for machine
consumption. Run from the repo root. `run` exits nonzero (and fires a macOS
notification) when a cycle fails or is skipped — a silent dead pipeline is
the failure mode this system is designed to avoid.

## How a run works

Each `trading run` fetches fresh bars, then: (1) fills the previous run's
pending orders at the first bar after their decision bar (open + 5 bps
slippage + venue fees), (2) checks exits — frozen 1.5x ATR-20 stops, regime
flush, trend break, time stop, forced exits, (3) checks entries — regime-
gated top of ranking, score threshold, cooldowns, settled cash (T+1 for
equities), max 25%/day deployment (35% crypto — fits one 30% position), and
(4) writes new pending orders for the next run. A decision bar is never
traded twice (journal-enforced); a late run (e.g. after sleep/wake) still
processes exits but skips entries beyond the staleness bound. Drawdown >20%
from the high-water mark halts entries venue-wide until `reset-breaker`.

## Scheduling

`trading schedule install` creates two LaunchAgents in ~/Library/LaunchAgents:
equities weekdays 18:30 and crypto daily 01:00. **launchd uses machine-local
time; the schedule assumes this Mac runs in America/New_York.** Crypto at
01:00 ET lands after the 00:00 UTC daily bar close. Runs missed while asleep
coalesce into one late run on wake, bounded by the staleness rule above.
The machine must be awake by ~02:00 ET or that day's crypto entries are
skipped (exits are unaffected).

## Where things live

- `config/<venue>.toml` — every tunable number (fees, windows, risk rules).
- `state/<venue>/portfolio.json` — paper portfolio (gitignored; atomic
  writes). If it corrupts, the run refuses to act and notifies; recover with
  `trading run --venue V --restore-from-journal`.
- `journal/<venue>.jsonl` — append-only record of every run: regime, full
  ranking with sub-scores, decisions made AND skipped (with reasons), fills,
  snapshot, config hash. State is reconstructible from it.
- `digest/YYYY-MM-DD.md` — daily digest (UTC date), regenerated after each
  venue run.
- `data/<venue>/*.parquet` — OHLCV cache (gitignored; safe to delete).

## Reading the digest (60 seconds)

Per venue: portfolio value and P&L vs buying-and-holding the benchmark
(SPY/BTC) since bootstrap; open positions with entry rationale (rank +
composite at entry) and distance-to-stop; today's fills; pending orders;
top-5 ranking; regime; warnings (quarantined symbols, stale-run entry skips,
circuit-breaker state, earnings-data degradation). Per-trade P&L excludes
the entry fee (cash totals are exact; see Open Items in the spec). A
permanently-delisted holding warns every run until manually handled — M3
adds a close command.

## Earnings blackout

Dropped: yfinance earnings dates proved unreliable at implementation time
(2026-07), so the filter is disabled in BOTH live and backtest modes — a
filter that exists live but not in backtest is worse than no filter. The
code path remains behind `earnings_blackout_enabled` in
`config/equities.toml` should a reliable source appear.

## Rankings output

Sub-scores are cross-sectional percentiles (0-1, higher = better):
`mom_short/med/long` (vol-adjusted momentum), `volume_surge`, `breakout`,
`overextension` (lower is better; inverted in the composite), `composite`
(equal-weight blend; the ranking key), `raw_return_30d` (raw return feeding
the crypto fee gate). `status` is `tradable` / `sell_only` / `untradable`;
regime is `risk_on` / `neutral` / `risk_off` (full / half / no new entries).
