# Options Gather v2 — OI, greeks, term structure (design spec)

**Status:** approved design, 2026-07-09. Tier 2 sub-project A (charter §8 /
Tier-1 spec §8 queue). DATA-ONLY: this spec acquires and verifies data; no
signal definitions are frozen here. The follow-on batch spec (options-v2 +
insider signals) freezes definitions before any sweep.
**Scope:** enrich the options gather with open interest, vendor greeks, and a
second expiration; fully re-gather both pools; the ThetaData delisted-equity
test; verification. Out of scope: signal registration, sweeps, stock-feed
adoption.

## 1. Purpose

The paid ThetaData Standard tier serves fields the gather never requests —
open interest, precomputed greeks, and additional expirations. These feed the
next signal families (OI put/call, OI change, IV term slope) and close the
large-cap leg-volume gap (the current samples.jsonl predates volume capture,
which is why cp_vol/osv run mid-cap-only today).

## 2. Cell enrichment (additive schema)

Each gathered cell (`src/trading/research/options_gather.py::gather_cell`)
gains, additively (existing readers tolerate unknown keys; `cell_metrics`
NaNs missing ones):

- Per leg: `open_interest` (from the EOD tape or the dedicated OI endpoint —
  implementer verifies which v3 endpoint serves it on Standard against the
  live terminal and documents the choice), and vendor greeks
  (`delta, gamma, theta, vega`) where the IV/greeks endpoint serves them.
  Absent field → key absent (never 0).
- Per cell: a **second expiration block** `far` — the next monthly expiration
  after the target (DTE band 55..90, same nearest-strike-with-data walk, same
  3 roles), carrying the same per-leg fields. Missing/no-data far expiration →
  `far` absent, cell still valid.
- Large-cap legs now carry `volume`/`count` (the gather code already writes
  them; the old large-cap FILE predates it — the re-gather closes the gap).

## 3. Re-gather and ops

- Full re-gather, both pools (large-cap ~100 names, mid-cap ~149), monthly
  decision dates 2019-01..latest, into the SAME paths
  (`data/options-iv/samples.jsonl`, `samples-midcap.jsonl`); the old files are
  backed up beside them (`samples.v1.jsonl`) before overwrite. Downstream
  paths unchanged; alphasearch trials are keyed by universe NAME (documented:
  regenerated data under unchanged hashes — the Piece 3 drift guard exists
  precisely to catch stale-baseline batteries afterward).
- Runs ON the mini against the local terminal (the proven pattern):
  `THETADATA_API_KEY=<key> /opt/homebrew/opt/openjdk/bin/java -jar
  ~/thetadata/202607071.jar` (the downloaded jar, NOT the bootstrap; key from
  mini `~/.config/trading/config.toml`, 41 chars with underscores; ONE
  terminal per account; clear ports 25503+25520 via lsof before relaunch).
  Gather as a background job with a log (`state/options-iv/gather-v2.log`),
  ~2× the original runtime (second expiration ≈ double the EOD requests;
  original was ~7.8k cells at ~3.4 cells/s — budget an overnight).
- The four original live-data caveats stay honored (raw strikes vs Tiingo
  close_raw snapping; HTTP 472 = empty not error; strike walking
  MAX_STRIKE_CANDIDATES=4; IV from mid never close). DTE bands: near 25..55
  (unchanged), far 55..90.
- After completion: rsync both samples files to the laptop; run the coverage
  report; commit nothing under data/ (gitignored) — the verification numbers
  go in docs.

## 4. The ThetaData delisted-equity test (5 minutes, while the terminal is up)

Query the stock EOD endpoint for XLNX (delisted 2022-02) and one control
(AAPL): `/v3/stock/history/eod`. Record the verdict in
`docs/options-data-vendors.md`: delisted history served (augment case
strengthens) or not (stock feed is live-universe-only, microstructure add-on
at best). Either way Tiingo remains the equity system of record.

## 5. Verification (before the data is trusted)

- Coverage report vs v1: cell count per pool, leg-volume presence rate
  (large-cap must jump from 0% to ~mid-cap levels), OI presence rate, far-
  expiration presence rate, IV agreement on overlapping near-leg cells
  (median |iv_v2 − iv_v1| — large drift on the SAME contract/date is a red
  flag to investigate, not accept).
- Spot-checks: one liquid name's OI vs public data (order of magnitude +
  direction over an expiry cycle); far-DTE band respected; strikes still raw
  dollars snapped against close_raw.
- Unit tests: schema round-trip (enriched cell parses; old v1 cell still
  parses), far-block absence tolerated, greeks/OI absence tolerated,
  gather_cell request composition (mocked client) for near+far.
- Repo discipline: full `uv run pytest` warnings-as-errors, ruff clean,
  subagent build + adversarial review, granular [AI] commits.

## 6. Error handling

Per-leg OI/greeks fetch failures degrade to absent keys (cell still valid,
counted in the coverage report); far-expiration resolution failures drop the
far block only; a mid-gather crash is resumable (the gather already skips
completed (symbol, month) cells present in the output file — verify this
holds with the enriched schema).

## 7. Follow-on (recorded, not in scope)

The options-v2 + insider batch spec will freeze: `oi_put_call`, `d_oi`
(OI innovations), `iv_term_slope` (far ATM IV − near ATM IV), large-cap
`cp_vol`/`osv` enablement, and the three Form 4 insider signals (npr_90,
cluster_buys_90, officer_buy_90) — one pre-registered sweep, one leaderboard
read, after both data sets land and pass verification.
