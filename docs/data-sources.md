# Data sources: known gaps and the paid-upgrade path

Status as of 2026-07-06. This documents the data limitations that bound on
backtest validity, and the vetted plan for closing them if/when a paid
subscription is approved.

## The binding gap: delisted equities (survivorship bias)

yfinance serves NO price history for delisted tickers (XLNX, XEC, XL, ...
fail with `YFTzMissingError`). Robinhood's API has the same limitation
(probed 2026-07-06: XLNX -> not_found). Our PIT index membership correctly
includes those names in their historical windows, but with no bars they
silently drop from the ranking pool.

Consequence: **every walk-forward number in the experiment ledger carries
survivorship bias**, and the bias most likely flatters us (the pool is
missing names that died). The momentum-vs-SPY gap (0.59 vs 0.96 OOS Sharpe)
is therefore, if anything, understated. Closing this gap is about honesty,
not alpha: the first action after purchase is re-running the exp3
walk-forward survivorship-free to learn the true baseline.

## Vetted providers (evaluated 2026-07-06)

| Provider | Fixes | ~Cost | Verdict |
|---|---|---|---|
| Sharadar Core US Equities bundle (Nasdaq Data Link) | delisted prices since 1998 + as-reported PIT fundamentals + S&P 500 constituent history, REST API | ~$50–100/mo (confirm on site) | **first choice** — one API covers prices, constituents, and cross-validates our SEC pipeline |
| Norgate Data (Platinum) | delisted prices since 1950 + constituent history for S&P 500/400/600, NDX, Russell | ~$630/yr | best-in-class data, but the updater app is **Windows-only** — does not fit the Mac pipeline without a VM |
| Tiingo | delisted coverage (less proven), clean EOD, REST API | ~$10–30/mo | budget trial option |

Non-fixes, for the record: EODHD/FMP fundamentals are restated (lookahead
risk); Polygon starter tier is only 5 years deep; Robinhood has no delisted
tickers.

## Integration plan (deferred until an API key exists)

Deliberately NOT pre-built: coding an ingest adapter against a guessed API
shape without live verification is how silent data bugs happen.

1. Sign up; put the key in `~/.config/trading/` (outside the repo), exposed
   to the pipeline as `NASDAQ_DATA_LINK_API_KEY` (or `TIINGO_API_KEY`).
2. Add a bar-source adapter alongside the yfinance path in
   `src/trading/data/`, keyed by config, with the same cache/parquet
   contract. Delisted names keep their final bars; exits on delisting are
   simulated at the last print (conservative).
3. Verification gate before any experiment uses it: cross-check N live
   symbols' bars against the yfinance cache (tolerance for adjustment
   differences), spot-check delisted names against known events (XLNX/AMD
   close 2022-02-14), and re-run the exp3 config unchanged — the delta vs
   0.59 measures the survivorship effect and gets journaled like any other
   experiment.

## What Robinhood's MCP is good for (and not)

Good: the earnings calendar (verified report dates, am/pm timing, EPS
estimate/actual) — journaled daily by `scripts/dump_earnings_calendar.py`
since 2026-07 to accumulate a point-in-time earnings history (see README,
Earnings blackout). Deep split-adjusted history for LISTED names also works.

Not: anything point-in-time (fundamentals are today-only snapshots) or any
delisted name. Do not backfill history from it into backtests.
