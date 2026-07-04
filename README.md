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

All timestamps are UTC. First equities run fetches ~580 symbols from
yfinance and takes several minutes; later runs hit the Parquet cache.
