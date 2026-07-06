# trading

Momentum swing trading system. It ranks liquid assets (S&P 500 + Nasdaq-100
equities; Robinhood-listed crypto) by likelihood of near-term upward moves
using price/volume momentum behind a market-regime gate, and paper-trades
that ranking daily with $1,000 per venue under strict risk rules. Backtesting
and walk-forward validation replay the same simulator against 2018-present
history to validate go-live evidence before it trades real money.

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and macOS (for
notifications and launchd scheduling):

    uv sync

## Commands

    uv run trading backtest --venue equities|crypto [--from DATE] [--to DATE] [--json]
                                                  # replay history through the live simulator;
                                                  # see Backtesting below for --walk-forward/--holdout
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

## Backtesting

```
trading backtest --venue equities|crypto [--from DATE] [--to DATE] [--json]
trading backtest --venue equities --walk-forward     # stitched OOS, tuned per window
trading backtest --venue equities --holdout          # final 6 months, evaluated ONCE
```

- One engine, replayed: the backtest drives the same simulator `step()` as
  `trading run`, session by session, filling at next-bar opens. Decisions at
  session T never see data after T.
- `--to` defaults to yesterday (UTC) — today's daily bar may still be forming.
- Non-holdout runs stop before `[backtest].holdout_start`. `--holdout` spends
  the holdout: a second invocation demands a typed `RERUN HOLDOUT` and both
  evaluations stay journaled forever.
- `--walk-forward` tunes exactly two hyperparameters (`entry_score_threshold`,
  `stop_atr_multiple`) on rolling train windows and reports stitched
  out-of-sample segments only; it refuses to report a stitch that skips every
  configured stress segment (2022 bear).
- Every run appends config hash + grid point + metrics to
  `journal/experiments-<venue>.jsonl`; the experiment count prints with every
  result. Quote results WITH their experiment count.
- Gate: annualized Sharpe of daily returns (0% cash) above buy-and-hold
  SPY/BTC over the identical period AND positive total return.
- A plain multi-year `--from`/`--to` backtest is one continuous `replay()`:
  a circuit-breaker trip halts entries for the rest of that run, exactly
  like live (only `reset-breaker` clears it there). `--walk-forward` windows
  are insulated from this — each window replays against fresh state, so a
  trip in one window never bleeds into the next.

### Data caveats (read before trusting a number)

- **Equities universe is point-in-time** (S&P 500 + NDX membership as-of each
  session; sources + licences in
  `src/trading/venues/universes/sources/PROVENANCE.md`). Residual survivorship
  (delisted tickers missing from yfinance) is measured per session and printed
  as the coverage ratio on every equities result; sessions below
  `[backtest].min_session_coverage` are skipped, not faked.
- **Crypto universe is today's Robinhood listing**: coins delisted before
  today are absent (survivorship bias — annotated on every crypto result).
  Listing dates come from data availability. Kraken retains only the ~720
  most recent daily candles (anchored to today), so each fetch judges
  coverage from the rows Kraken actually returned and splices any head it
  could not serve from a second exchange; Kraken wins overlaps (see `[data]`
  in `config/crypto.toml`). Live runs skip the backfill exchange because
  Kraken serves recent windows fully — a data-driven outcome, not an
  unconditional code path. The crypto adapter's `universe()` also ignores
  `as_of` for status: today's `sell_only`/`untradable` snapshot is projected
  onto every historical session, not just the current one.
- NYSE half-days are handled conservatively by the live session guard (waits
  for 16:00 ET + buffer); the backtest calendar is SPY's actual bar dates, so
  half-days are traded normally there.

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

`[portfolio].exit_style` selects which of the two exit mechanisms above
governs a position: `"frozen"` (default) is the behavior just described.
`"trailing"` is an experiment flag — it replaces the frozen stop and trend
break with a single stop that ratchets up with the position's peak close
(tightening one-way on a regime flush) and is never loosened; time stop and
forced exits are unchanged. It is evidence-driven, not a parameter tweak:
walk-forward diagnostics showed frozen stops and the trend-break rule both
giving back money from held peaks, so it is under evaluation as a config
flag rather than being turned on by default.

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
  `[signals].ranker` selects the ranking strategy by name (default
  `"momentum_v1"`), validated against the registry at config-load time. To
  add a new ranker, register a callable matching `compute_features`'s
  contract in `RANKERS` in `src/trading/signals/registry.py`, then reference
  its key from a venue's TOML.
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
circuit-breaker state, earnings-data degradation). Per-trade P&L includes
the entry fee (folded into `realized_pnl`; cash totals are exact). A
permanently-unfillable holding (e.g. a delisted symbol that never prints
another bar) warns every run until an operator manually intervenes on
`state/<venue>/portfolio.json`; a dedicated `close` command remains future
work.

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
