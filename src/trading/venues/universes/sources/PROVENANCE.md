# Point-in-time universe data provenance

## S&P 500 membership
- Source: https://github.com/fja05680/sp500
- File: `S&P 500 Historical Components & Changes (Updated).csv` -> snapshotted as
  `sp500_history.csv`. Note: as of retrieval the upstream repo no longer names
  this file with an embedded date (earlier revisions used a
  `(MM-DD-YYYY)`-suffixed name); `(Updated)` is the current, actively
  maintained variant and is the one used here. A second, unsuffixed file in
  the same repo (`S&P 500 Historical Components & Changes.csv`) embeds
  delisting dates in the tickers themselves (e.g. `AAL-199702`) and was not
  used, since the build script expects plain ticker symbols per snapshot row.
- Pinned commit: `c403a121c2e766840f34837738cdd4725eeda818` (last commit to
  touch this file; repo HEAD at retrieval was `b792557e915703398ef9a67e4b583a37c6ec80d5`)
- Retrieved: 2026-07-04 (UTC)
- Licence: MIT

## Nasdaq-100 membership
- Source: https://en.wikipedia.org/wiki/Nasdaq-100 (current constituents +
  yearly change tables), retrieved by scripts/build_pit_membership.py
- Page revision: https://en.wikipedia.org/w/index.php?title=Nasdaq-100&oldid=1362572078
- Retrieved: 2026-07-04 (UTC)
- Licence: CC BY-SA 4.0 (Wikipedia text/data)

## S&P 400 (MidCap) membership
- Source: https://en.wikipedia.org/wiki/List_of_S%26P_400_companies (current
  constituents + changes table), retrieved by scripts/build_pit_membership.py
- Page revision: https://en.wikipedia.org/w/index.php?title=List_of_S%26P_400_companies&oldid=1362781477
- Retrieved: 2026-07-06 (UTC)
- Licence: CC BY-SA 4.0 (Wikipedia text/data) -- same terms already accepted
  for NDX above.
- Cell splitting: dual-class Wikipedia ticker cells ("UAA/UA") are split into
  one interval row per symbol; validate() hard-fails on any surviving
  non-ticker artifact (slashes, spaces, footnote markers).
- Coverage boundary: the changes table is only reliable back to 2019, so
  sp400 intervals never start before 2019-01-01 (SP400_SINCE in the build
  script) even when a symbol's real membership predates that. Backtests
  reaching further back than 2019 for sp400 simply see fewer sp400 members,
  same shape as the 2017-01-01 floor already applied to NDX/sp500.
- sp400 is opt-in: the live/paper venue config still defaults to sp500+ndx
  only (see config/equities.toml `[universe] indices`); sp400 participation
  is a backtest-only config choice.
- Build validation output (2026-07-06 run):
  `validation OK: drift vs M1 snapshot = 7 symbols; sp400 members today = 400;
  CDK removed 2022-07-06, MDP removed 2020-04-27`
  Spot-check anchors (scout-verified, asserted exactly by the build script):
  CDK Global (ticker CDK) removed 2022-07-06; Meredith Corp (ticker MDP)
  removed 2020-04-27.

## Output
- `../equities_membership.csv` — merged intervals, regenerated only by
  `uv run python scripts/build_pit_membership.py`; treat as frozen data, review diffs.

## Known limitations (annotated on backtest results)
- Ticker renames (FB->META, ANTM->ELV, ...) appear as remove+add: the backtest
  force-exits on the rename date. Conservative, and rare enough to accept.
- Residual survivorship: delisted tickers absent from yfinance are counted per
  session and reported as the coverage ratio on every equities result.
- Ticker recycling: a delisted symbol's letters can later be reassigned to an
  unrelated live company (yfinance then serves that new company's prices
  under the old ticker -- e.g. CNR, flagged during scouting for this work).
  Any symbol here with a closed (non-empty end) interval is at risk: the
  backtest cache would otherwise fetch that ticker's full history and hand
  the simulator prices from whatever company holds the symbol today. The
  backtest engine truncates such a symbol's bars at (last closed membership
  interval end + `membership_exit_buffer_days`), so post-exit prices --
  recycled or not -- can never leak into a result (see prepare() in
  src/trading/backtest/engine.py).

## CIK <-> symbol point-in-time map (M4 fundamentals)
- Output: `src/trading/fundamentals/cik_map.csv` — regenerated only by
  `uv run python scripts/build_cik_map.py`; treat as frozen data, review diffs.
- Sources: SEC `company_tickers.json` (https://www.sec.gov/files/company_tickers.json,
  public domain, retrieved 2026-07-06 UTC) + the script's reviewed RENAMES
  table (ticker renames among membership symbols, boundary dates cross-checked
  against the membership CSV's remove/add rows) + `equities_membership.csv`
  symbols (which define what needs mapping).
- Validation (asserted by the script): FB and META share CIK 1326801 with the
  interval boundary at the 2022-06-09 rename; ABC and COR share CIK 1140859;
  >= 95% of current members map.
- Build validation output (2026-07-06 run):
  `validation OK: 100.0% of 918 current members mapped; wrote
  src/trading/fundamentals/cik_map.csv (1131 intervals; 211 membership symbols
  unmapped)`.
- RENAMES additions beyond the initial reviewed set, found by cross-checking
  the unmapped list against SEC's live `company_tickers.json` during this
  build's validation pass (scout-verified against known corporate events):
  HCP->PEAK (2019-11-05, Healthpeak Properties' first rebrand) and PEAK->DOC
  (2024-03-04, Healthpeak's post-Physicians-Realty-Trust-merger ticker,
  chained through both hops to the current CIK 765880); FBHS->FBIN
  (2022-12-19, Fortune Brands Home & Security -> Fortune Brands Innovations,
  CIK 1519751). All three dates match the membership CSV's remove/add
  transition exactly.
- Known limitation: acquired/delisted symbols absent from company_tickers.json
  and the RENAMES table are unmapped -> no fundamentals -> neutral (0.5)
  fundamentals percentiles. The build prints the list; extend RENAMES
  deliberately.
- Structural assumption (ticker recycling): a ticker present in today's
  `company_tickers.json` is assumed to have denoted the SAME company for its
  whole interval back to 2017 unless a RENAMES row says otherwise. A ticker
  that was vacated by one membership company and later reassigned to an
  unrelated live company would silently attach the new company's CIK to the
  old company's membership window (the same recycling hazard the membership
  section flags for prices, e.g. CNR). No such case is known among current
  intervals, but the assumption is unverified in bulk; automated
  reconciliation of interval-vs-filing identity against the backfilled
  fundamentals store lands in the M4 verification task.
- Known limitation (out-and-back rename): Fiserv renamed FISV->FI in 2023 then
  reverted FI->FISV in 2025 (confirmed live against SEC's current listing,
  which now shows CIK 798354 under ticker FISV again). The one-row-per-symbol
  build model can represent a symbol's *single* continuous interval per CIK,
  not a ticker a company vacates and later re-adopts, so this reversion is
  left unresolved rather than guessed at: "FISV" maps continuously from
  2017-01-01 with no end (silently spanning the 2023-06-06..2025-11-11 window
  when the live ticker was actually "FI"), and "FI" itself is unmapped. This
  is fail-open, not a misjoin: "FI" is not a current member so the coverage
  gate is unaffected, and the membership CSV has no active "FISV" member
  during that window for stale data to misjoin into -- but any FI-ticker
  filings from that window get no fundamentals. Left as a documented gap
  rather than extending the CSV schema to support multiple intervals per
  symbol string, which is beyond this task's scope.
- Upstream membership artifact (Fiserv, M3): the membership CSV's NDX rows
  label Fiserv "FI" all the way back to 2017 (`FI,ndx,2017-01-01,2023-06-07`)
  because the Wikipedia NDX source backcasts a constituent's
  current-at-retrieval ticker, while the sp500 source correctly carries
  "FISV" for the same period. Harmless today: "FI" is unmapped here so those
  rows fail open to neutral fundamentals (no misjoin), and "FI" is not a
  current member so the coverage gate is unaffected. Flagged for a future
  membership rebuild with rename-aware historical labeling.
- Deliberately NOT added: multi-company mergers where the surviving CIK is
  ambiguous without further research (e.g. Cimarex Energy (XEC) + Cabot Oil &
  Gas (COG) -> Coterra Energy (CTRA), 2021). Guessing a successor CIK here
  risks exactly the kind of misjoin this map exists to prevent, so these stay
  unmapped/fail-open until deliberately researched and added.

## SEC EDGAR fundamentals (M4)
- Backfill: SEC Financial Statement Data Sets quarterly ZIPs, 2018q1 -> present
  (https://www.sec.gov/dera/data/financial-statement-data-sets, US-government
  public domain), cached under data/edgar-raw/ (gitignored).
- Top-up: https://data.sec.gov/api/xbrl/companyfacts/ (same terms).
- Access policy: User-Agent "trading-system travis@launchsupply.com", requests
  spaced under the 10 req/s ceiling.
- Verification (run 2026-07-06): AAPL 2023-02-03 TTM gross profitability
  0.4812 (expected 0.4813; single-quarter scout basis was 0.1452 — see
  scripts/verify_fundamentals.py for the recomputation arithmetic); AAPL value
  primitives matched the 2023-02-03 10-Q (15,842,407,000 shares; $56,727M
  book equity; $95,171M TTM net income; earnings yield 0.0389 at the pinned
  $154.50 close); restatement invariant passed with 186 re-filed fiscal
  periods, zero later accessions in the store; 1110 symbols with fundamentals
  (from 33 quarterly ZIPs 2018q1->2026q1 -- 2026q2 not yet published, covered
  by the weekly companyfacts top-up instead; rerun of the backfill script
  confirmed idempotence: 0 rows appended).
- Ticker-recycling reconciliation (routed here from Task 4's review, see
  scripts/verify_fundamentals.py's check_recycling_reconciliation): every
  cik_map.csv symbol/CIK interval was checked for at least one filing FILED
  inside its membership window intersected with the backfill's coverage
  range. 21 of 1135 intervals had zero filings in-window. Investigated each
  via data.sec.gov/submissions/CIK##########.json:
  - 16 are foreign private issuers that file 20-F/40-F instead of 10-K/10-Q
    (structurally excluded by edgar.py's form filter, benign): ARM, ASML,
    AZN, BIDU, CCEP, CHKP, FER, GFS, JD, NBIS, NTES, PDD, SE, TCOM, TRI, VOD.
  - 1 is a domestic bank (OZK / Bank OZK, CIK 1569650, confirmed Nasdaq-listed
    under this CIK) that reports periodic financials to the FDIC rather than
    filing 10-K/10-Q with the SEC -- benign, a different regulator, not a
    mapping error.
  - 1 is a genuine spinoff too new to have filed an annual/quarterly report
    yet (FDXF / FedEx Freight Holding Company, Inc., CIK 2082247, added to
    the sp500 membership 2026-06-01; its only EDGAR filings so far are Form
    10 registration statements) -- benign, will self-resolve once it files.
  - 3 are CONFIRMED ticker-recycling mismaps: APC (Anadarko Petroleum,
    sp500 member 2017-01-01..2019-08-09, acquired by Occidental) currently
    resolves via company_tickers.json to CIK 2080921, "ARKO Petroleum
    Corp.", an unrelated company now trading under the recycled APC ticker;
    BID (Sotheby's, sp400 member 2019-01-01..2019-10-03, taken private)
    resolves to CIK 2094919, "Tribeca Strategic Acquisition Corp.", an
    unrelated SPAC; CONE (CyrusOne, sp400 member 2019-01-01..2022-03-30,
    taken private) resolves to CIK 2103884, "Compass Sub North, Inc.", an
    unrelated merger shell. In all three cases build_cik_map.py's
    current-ticker lookup (no RENAMES entry covers a ticker vacated by an
    acquired/delisted company) attached today's live owner of the ticker to
    the historical membership interval instead of leaving it unmapped. In
    practice this is fail-open, not silent corruption: the wrongly-mapped
    CIK had zero filings during the historical (pre-2022) window the real
    company occupied, so no misattributed data actually reached the store --
    but it is a real defect in cik_map.csv worth a deliberate follow-up
    (e.g. an explicit exclusion/"delisted, do not re-resolve" list in
    scripts/build_cik_map.py) rather than relying on this fail-open outcome
    indefinitely.
