# trading

Momentum swing trading system. It ranks liquid assets (S&P 500 + Nasdaq-100
equities; Robinhood-listed crypto) by likelihood of near-term upward moves
using price/volume momentum behind a market-regime gate. This milestone
ships rankings only; paper trading, digests and backtesting arrive next.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):

    uv sync

## Commands

    uv run trading rankings --venue equities   # ranked table + regime (SPY benchmark)
    uv run trading rankings --venue crypto     # ranked table + regime (BTC benchmark)

Options: `--as-of YYYY-MM-DD` (default: today UTC), `--top N` (default 25,
0 = all rows), `--json` for machine-readable output, `--config-dir DIR`
(default `config/`). Run from the repo root.

Exits 1 with a `WARNING` on stderr when fresh data cannot be assembled
(under 90% universe coverage, or the benchmark fetch fails).

## Where things live

- `config/<venue>.toml` — every tunable number (fees, windows, thresholds).
- `data/<venue>/*.parquet` — gitignored OHLCV cache. The trailing 30 days
  are re-fetched every run, so deleting `data/` is always safe.
- `src/trading/venues/universes/*.csv` — committed universe snapshots.

All timestamps are UTC. First equities run fetches 517 symbols from
yfinance and takes several minutes; later runs hit the Parquet cache.

## Reading the output

Each row's sub-scores are cross-sectional percentiles (0-1, higher = more
favorable) computed within that run's ranked universe:

- `mom_short` / `mom_med` / `mom_long` — volatility-adjusted momentum
  percentile over the venue's short/med/long lookback windows.
- `volume_surge` — current week's dollar volume vs its trailing 3-month
  average, percentile.
- `breakout` — closeness to the 20/60-day highs, percentile.
- `overextension` — RSI-stretch guard; **lower is better**, and it enters
  the composite inverted (`1 - percentile`).
- `composite` — equal-weight average of the six scores above; this is what
  the table is ranked on.
- `raw_return_30d` — un-normalized 30-day return (not a percentile); used
  downstream by the crypto fee gate in M2, not by the ranking itself.

`status` (from the venue's listing) is one of:

- `tradable` — normal entries and exits allowed.
- `sell_only` — still ranked and shown, but M2 will not open new positions.
- `untradable` — excluded from the venue universe entirely.

The `regime` line shows the benchmark-driven gate: `risk_on` / `neutral` /
`risk_off`, mapping to exposure multipliers `1.0` / `0.5` / `0.0`. This
governs how aggressively M2 deploys new entries; it does not affect the
rankings themselves.

Warning lines below the table:

- `fetch failures` — symbols whose fetch errored; they count against the
  90% coverage gate (the run aborts if coverage drops below it).
- `quarantined` — symbols that failed the recent-window data-sanity check
  (an implausible price move within the trailing quarantine window) and are
  excluded from ranking.
- `insufficient history` — fetched fine, but too few bars to compute all
  required features yet (e.g. a recent listing); excluded from ranking
  until enough history accumulates.
