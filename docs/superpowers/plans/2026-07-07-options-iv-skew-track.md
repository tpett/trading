# Options IV-skew track — data pull + first experiments

Written 2026-07-07 as the durable spec for the options-signal pivot (momentum is
exhausted on liquid US equities; see docs/experiments.md). Self-contained so it
survives a context compaction. **Hypothesis:** implied-volatility skew predicts
the cross-section of forward stock returns — steep OTM-put skew (informed
downside bets) → lower forward returns. Trade the liquid *stock* on the options
*signal*; never trade options themselves.

## Proven state (validated live 2026-07-07)

**ThetaData Standard is active** (Options: STANDARD, 4 concurrent requests,
history back to **2012** — past our 2018 need). Verified: pulled the AAPL
2018-06-15 $185 call EOD series with OHLC + NBBO bid/ask + volume.

**Terminal / hosts:**
- ThetaData serves a LOCAL HTTP API at `http://127.0.0.1:25503` via the Theta
  Terminal (Java). There is NO cloud endpoint — the terminal must run on
  whatever host queries it.
- Key: `theta_data_api_key` in `~/.config/trading/config.toml` (present on BOTH
  laptop and mini). Launch with it via the `THETADATA_API_KEY` env var.
- Jar staged at `mini:~/thetadata/ThetaTerminalv3.jar`. Java on the mini is
  openjdk 26 at `/opt/homebrew/opt/openjdk/bin/java` (NOT on default PATH).
- **ONLY ONE terminal per account** — a second instance → "Invalid session ID".
  Before (re)launching, clear ports 25503 AND 25520: `lsof -ti :25503 :25520 |
  xargs kill -9`, then `pkill -9 -f ThetaTerminal`. Wait for the log line
  "Starting server at: http://0.0.0.0:25503/" (not "already in use").
- Currently running on the LAPTOP (pid ~94829). For overnight/unattended work it
  must move to the always-on MINI (kill the laptop one first to avoid the
  session conflict).

**v3 API (confirmed endpoints):**
- `GET /v3/option/list/expirations?symbol=SYM&format=json` → `{"response":
  [{"symbol","expiration":"YYYY-MM-DD"}...]}`
- `GET /v3/option/list/strikes?symbol=SYM&expiration=YYYY-MM-DD&format=json` →
  `{"response":[{"strike":185.0}...]}` (strikes in DOLLARS, raw/unadjusted)
- `GET /v3/option/history/eod?symbol=SYM&expiration=YYYY-MM-DD&strike=185&right=C|P&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&format=json`
  → `{"response":[{"contract":{...},"data":[{open,high,low,close,bid,ask,
  bid_size,ask_size,volume,count,last_trade}...]}]}`
- Rules: v3 uses `symbol` (NOT `root`); dates DASHED `YYYY-MM-DD`; **history
  requests CAPPED at 1 month** — chunk monthly. Strikes/prices are RAW
  (unadjusted) — 2018 AAPL strikes are ~$185 (pre-2020 split).
- **No working historical IV/greeks endpoint found** (the guessed paths 404'd).
  We **compute IV ourselves** via `trading.research.options_iv.implied_vol`
  (Black-Scholes inversion, already built for the POC) from the quote **mid**
  ((bid+ask)/2) + the underlying spot + strike + days-to-expiry + rate.

**Underlying prices:** from Tiingo (survivorship-free), not ThetaData (whose
stock plan stayed FREE). The `data/equities-tiingo` cache (713 sp500+ndx symbols,
2016-2026) exists on both laptop and mini. Use it for spot-at-decision and
forward returns.

**Existing reusable code:** `src/trading/research/options_iv.py` (BS inversion,
`compute_skew`, `skew_from_cell`, interpolated/placeholder-leg drop) and
`scripts/option_skew_analysis.py` (cross-sectional skew→return study: Spearman +
tercile spread, level and de-meaned "change"). The FeaturePanel vectorization
pattern (`src/trading/signals/engine.py`) is the model for a per-symbol
skew-series panel gathered per session.

## Data-pull plan

**First-read universe (bound the gather so it completes overnight):**
- ~**100 most-liquid sp500+ndx names**, point-in-time (rank membership by
  trailing dollar volume; survivorship-free — include delisted members over the
  window). Expand to the full ~500 only if the first read shows signal.
- Window **2019-01 .. 2026-01** (7y, includes the 2022 stress segment; holdout
  ≥2026-01-05 stays reserved).
- **Monthly** decision dates (1st trading day). Skew is a slow, ~monthly-horizon
  signal; monthly sampling cuts data ~20× vs daily and is piecewise-constant
  between updates in the walk-forward (the fundamentals pattern).

**Per (symbol, decision month):**
1. Spot on the decision date from the Tiingo cache.
2. Target expiration: the monthly (3rd-Friday) closest to D+35 in [D+25, D+50].
3. Three strikes from spot: ATM (nearest spot), OTM put (~0.90×spot), OTM call
   (~1.10×spot), snapped to the real strike ladder via `/list/strikes`.
4. `/option/history/eod` for each contract over the decision date (±3 days),
   take the bar on D; compute IV from the (bid,ask) mid via `options_iv`.
5. `skew_put_atm = IV(otm_put) - IV(atm)`, `skew_put_call = IV(otm_put) -
   IV(otm_call)`.

**Gather engine:** a script hitting `localhost:25503` on the mini (where the
terminal runs), 4-way concurrent, monthly-chunked, with rate-limit backoff and
incremental JSONL append (partial progress survives). Store under
`data/options-iv/` : `samples.jsonl` (per symbol-month: spot, expiration, dte,
the 3 contracts' bid/ask/close + computed IV + skew) and reuse the Tiingo cache
for underlyings. Rough volume: 100 symbols × ~84 months × 3 contracts ≈ 25k EOD
requests — hours on 4 concurrent, an overnight job on the mini.

## Signal + pipeline

- Build an **IVSkewPanel** (per-symbol monthly skew time-series, gathered at
  `as_of` — analogous to FeaturePanel; PIT, no lookahead).
- New ranker(s) in the registry: `skew_v1` — cross-sectional rank on skew
  (steep put skew → rank LOW / avoid). Missing skew → neutral (fail-open), like
  the fundamentals rankers.
- Plug into the existing walk-forward engine (same regime gate, same 2-hyperparam
  grid, same SPY gate, same holdout discipline). Reuse the vectorized panel path.

## First experiments (gated, journaled)

1. **OPT-1 — skew-level cross-sectional signal.** Rank the ~100-name universe by
   `skew_put_atm`; walk-forward vs SPY. Does put-skew predict forward returns?
   Report OOS Sharpe vs SPY (0.96 bar) AND vs the momentum baseline (0.45).
2. **OPT-2 — skew-change (de-meaned).** Per-name skew minus its own trailing
   mean (strips structural per-name skew levels, the sector-tilt problem). Often
   the cleaner signal.
3. **OPT-3 — skew as an overlay on momentum.** Combine: does a skew filter/tilt
   improve the momentum core? (Only if OPT-1/2 show standalone life.)
- Gate: stitched OOS Sharpe > SPY with positive return; also read vs momentum
  0.45 (does it add anything). Anti-overfitting: experiment counting, holdout
  reserved. Expand universe/frequency only if a first read shows signal.

## Execution sequence (post-compaction)

1. **Move the terminal to the mini:** kill the laptop terminal; on the mini,
   clear ports and launch `THETADATA_API_KEY=<from config> /opt/homebrew/opt/openjdk/bin/java
   -jar ~/thetadata/ThetaTerminalv3.jar` (background); confirm "Starting server"
   + Options STANDARD; validate one 2018 EOD pull.
2. **Build the gather client + IV/skew pipeline** (subagent, tests): the
   ThetaData v3 EOD client, monthly-chunked concurrent gather, IV-from-mid, skew,
   the `data/options-iv/` schema. Mock the HTTP for tests.
3. **Gather the first universe** on the mini (overnight background job).
4. **Build `skew_v1` + IVSkewPanel**, run OPT-1 walk-forward; then OPT-2/3.
5. Report each result against SPY (0.96) and momentum (0.45); update
   docs/experiments.md.

## Open decisions / risks
- IV from EOD quote **mid** is noisier than a vendor IV surface; deep-OTM wings
  have wide spreads — apply the POC's interpolated/placeholder guards and
  consider a min-quote-size floor.
- Monthly sampling is a scoping choice; if signal is marginal, daily is the
  richer (20× costlier) test.
- Universe is ~100 liquid names first; the true test needs survivorship-free
  breadth — expand only on a positive first read.
- Keep it signal-only (trade the stock); options-trading is a separate risk model
  we are NOT taking on.
