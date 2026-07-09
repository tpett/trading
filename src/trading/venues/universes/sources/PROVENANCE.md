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

## Historical (delisted/acquired) CIK intervals (Piece 2 extension)

- `cik_map_historical.csv` GENERATED by `scripts/build_cik_map_historical.py`
  on 2026-07-08, and its 157 rows appended verbatim (minus the review-evidence
  `name` column) to `cik_map.csv` -- existing rows untouched, byte-for-byte.
  `build_cik_map.py` re-merges the historical file on every regeneration
  (`merge_historical()`), skipping EXCLUSIONS and any interval overlapping a
  live resolution for the same symbol.
- Why: build_cik_map.py resolves only symbols chaining to a ticker alive in
  TODAY's company_tickers.json, so window-era members that were acquired,
  taken private, or delisted (XLNX, CELG, TWTR, SIVB, RTN, CHK, ...) had no
  row at all -- 182 of 1203 discovery-window members (2019-01-01..2023-12-31),
  found when the Piece 2 SIC map's coverage gate refused to build at 84.9%.
- Source: SEC Financial Statement Data Sets quarterly ZIPs 2019q1..2024q1
  (sub.txt), cached in data/edgar-raw/. Each 10-K/10-Q row's `instance`
  filename prefix is the ticker the filer itself used at filing time
  (xlnx-20201226.xml -> XLNX) next to its cik and registered name -- a
  point-in-time ticker->CIK record that includes dead companies.
- Resolution rules (never guess): unique candidate CIK that FILED strictly
  within the symbol's membership tenure; else unique within tenure + 90-day
  grace (a final report straggling past index removal: SCG, DRQ); else the
  reviewed RENAMES successor's committed CIK among the candidates (CTL: Lumen
  vs its Qwest/Level 3 co-filers sharing the ctl- prefix); else SEC's own
  submissions-JSON ticker attribution naming exactly one candidate. Every
  resolution was then verified against its CIK's live submissions JSON
  (ticker or name/formerNames match, retry-once, fail-closed): 157/157
  passed, 0 dropped. RENAMES pairs are cross-checked for CIK agreement AT the
  rename boundary (PEAK->DOC legitimately spans two companies: DOC was
  Physicians Realty Trust, CIK 1574540, until Healthpeak took the ticker
  2024-03-04 -- the appended DOC interval ends exactly at the committed row's
  start). Manual cross-checks (submissions name/formerNames): XLNX=743988
  Xilinx, CTL=18926 Lumen/CenturyLink, TWTR=1418091 Twitter, CELG=816284
  Celgene, SCG=754737 SCANA, DOC=1574540 Physicians Realty, BK=1390777 BNY
  Mellon (ticker changed BK->BNY, why it left company_tickers.json),
  SIVB=719739 SVB Financial.
- Interval dating: start is the 2017-01-01 SINCE floor (same convention as
  the committed map's rename-successor rows, which also predate their
  ticker-live era); end is the symbol's RENAMES change date, else the
  committed successor row's start, else its last membership end -- every
  appended interval is CLOSED so a future recycler of the ticker can never
  inherit it. Where two symbols share a CIK with overlapping intervals
  (SYMC/NLOK, VIAC/PARA), attribution outside a symbol's membership window is
  inert: consumers gate by point-in-time membership.
- Coverage: 1021/1203 (84.9%) window members mapped before, 1178/1203
  (97.9%) after. The 25 that remain unmapped (fail-open, no segment, neutral
  rank), by reason:
  - no SEC 10-K/10-Q at all: FRC (First Republic) and SBNY (Signature Bank)
    report(ed) to the FDIC -- the OZK pattern; structurally unresolvable
    from EDGAR.
  - absent from FSDS instance prefixes: CBS, CHX (files as championx-), DISCK
    (class shares live under DISCA's filings), LPT (files under its pre-2003
    lry- prefix), NFX, NLSN (files as nlsnnv-), RCM, VVC (last pre-window-era
    filings predate 2019q1).
  - ambiguous beyond the tie-breaks: ACC, AZPN (two same-name Aspen
    Technology CIKs both held AZPN in tenure), BERY, BHGE (Baker Hughes
    parent + LLC co-filer, neither claims BHGE today), DNB (2017-2019 member
    is the pre-LBO entity; the FSDS candidate is the 2020 re-IPO CIK), ETRN,
    FI (Frank's International held FI in the window before Fiserv took it),
    HFC (HollyFrontier -> HF Sinclair is a NEW holdco CIK; resolving would
    break the DINO boundary check), K, MNK, TCF (two same-name TCF Financial
    CIKs across the 2019 Chemical Financial merger), VAR.
  - EXCLUSIONS (confirmed recycled tickers, unchanged): APC, BID, CONE.
- KNOWN CAVEAT: like the committed map's other rows, these intervals date
  symbol<->CIK identity, not index membership; the SINCE-floor starts mean a
  filing filed before the ticker was live can attach to the later symbol --
  harmless under membership gating, disclosed here. Regenerate deliberately,
  review the diff (the name column exists for exactly that), and update this
  entry.

## SEC EDGAR fundamentals (M4)
- Backfill (PRIMARY, final-review fix wave): https://data.sec.gov/api/xbrl/
  companyfacts/ (US-government public domain), one fetch per cik_map.csv cik.
  Retired-primary alternative (`--source zips`): SEC Financial Statement Data
  Sets quarterly ZIPs, 2018q1 -> present
  (https://www.sec.gov/dera/data/financial-statement-data-sets), cached under
  data/edgar-raw/ (gitignored). The switch: the ZIPs strip nearly all dei
  cover-page facts (dei:EntityCommonStockSharesOutstanding on 1 of 5631
  filings), leaving shares_outstanding at 59% vs companyfacts' 89.7%.
- Top-up: same companyfacts API (same terms), weekly from the runner.
- Access policy: User-Agent "trading-system travis@launchsupply.com", requests
  spaced under the 10 req/s ceiling.
- Verification (companyfacts rebuild, 2026-07-06): AAPL 2023-02-03 TTM gross
  profitability 0.4812 (expected 0.4813; single-quarter scout basis was
  0.1452 — see scripts/verify_fundamentals.py for the recomputation
  arithmetic); AAPL value primitives matched the 2023-02-03 10-Q
  (15,821,946,000 shares — the dei cover-page count verified verbatim against
  the filed document; the ZIP-era expectation 15,842,407,000 was the us-gaap
  balance-sheet fallback at 2022-12-31, mis-attributed as cover page —
  $56,727M book equity; $95,171M TTM net income; earnings yield 0.0389 at the
  pinned $154.50 close); restatement invariant (visibility-timing form: any
  store row for a re-filed (cik, period, form) group sits at the ORIGINAL
  filing's filed date) passed; 1110 symbols with fundamentals, 38,972 rows
  from 1109 fetched CIKs (1 persistent 404: OZK/Bank OZK reports to the FDIC,
  no companyfacts document — benign, on the reconciliation audit list).
- Known structural limits of the companyfacts source (accepted, fail-open to
  neutral): (a) multi-class filers (~77 current members incl. META, BRK-B,
  CMCSA, ACN, ABNB) tag cover-page share counts per share class with a class
  dimension, and companyfacts serves undimensioned facts only -> no shares ->
  value ratios neutral. NAMED FOLLOW-UP — per-class share summation: resolve
  a consolidated share count for multi-class filers by summing the per-class
  dimensioned dei:EntityCommonStockSharesOutstanding facts (requires the XBRL
  frames API or per-filing parsing; the companyfacts API cannot serve them).
  (b) Dual-registrant REIT/LP combined filings can attribute consolidated
  facts to the partnership CIK (MAA 2014-2019 has no undimensioned Assets
  under the parent CIK), plus fy/fp label-noise dedup collisions: 185 of
  33,996 ZIP-original filings (0.54%, 147 symbols, worst MAA=7) have no store
  row at their filed date; the step-function read serves the prior filing's
  values across such gaps.
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
  - 3 WERE confirmed ticker-recycling mismaps: APC (Anadarko Petroleum,
    sp500 member 2017-01-01..2019-08-09, acquired by Occidental) resolved via
    company_tickers.json to CIK 2080921, "ARKO Petroleum Corp.", an unrelated
    company now trading under the recycled APC ticker; BID (Sotheby's, sp400
    member 2019-01-01..2019-10-03, taken private) resolved to CIK 2094919,
    "Tribeca Strategic Acquisition Corp.", an unrelated SPAC; CONE (CyrusOne,
    sp400 member 2019-01-01..2022-03-30, taken private) resolved to CIK
    2103884, "Compass Sub North, Inc.", an unrelated merger shell. In all
    three cases build_cik_map.py's current-ticker lookup (no RENAMES entry
    covers a ticker vacated by an acquired/delisted company) attached today's
    live owner of the ticker to the historical membership interval instead of
    leaving it unmapped. In practice this was fail-open, not silent
    corruption: the wrongly-mapped CIK had zero filings during the historical
    (pre-2022) window the real company occupied, so no misattributed data
    actually reached the store -- but it was a real defect in cik_map.csv.
    FIXED (final-review fix wave): APC/BID/CONE are now seeded into
    `build_cik_map.py`'s `EXCLUSIONS` dict (symbol -> reason) and always
    unmapped regardless of what company_tickers.json currently resolves them
    to; regenerated cik_map.csv (2026-07-06) confirms all three are absent
    (100.0% of 918 current members still mapped). Extend EXCLUSIONS
    deliberately if a future recycled ticker is found the same way.

## SIC classification map (Piece 2 segments)

- `sic_map.csv` GENERATED by `scripts/build_sic_map.py` on 2026-07-09 UTC (see
  the `fetched_at` column): for each cik_map.csv symbol whose CIK interval
  overlaps the discovery window 2019-01-01..2023-12-31, the `sic` +
  `sicDescription` fields of https://data.sec.gov/submissions/CIK##########.json
  (throttled 0.11 s/req, User-Agent per SEC policy, retry-once per symbol).
- Coverage at generation: 97.8% of the 1203 window membership symbols mapped
  (1283 rows total, incl. non-member share classes like GOOG/GOOGL sharing a
  CIK; 1284 window symbols attempted across 1250 distinct CIKs; 1 fetch-side
  unmapped: OZK, whose filer carries no SEC 10-K/10-Q -- reports to the FDIC,
  the known-benign case above). The other 25 unmapped window members have no
  window-overlapping cik_map.csv interval; the full list and per-symbol
  reasons live in the "Historical (delisted/acquired) CIK intervals" section
  (10 absent from FSDS incl. FDIC filers FRC/SBNY, 12 ambiguous beyond the
  never-guess tie-breaks, 3 EXCLUSIONS: APC/BID/CONE). A first generation
  attempt on 2026-07-08 FATALed at 84.8% coverage and wrote nothing; the
  historical CIK extension above closed the gap (84.9% -> 97.9% CIK
  coverage), after which this map regenerated cleanly.
- Anchors verified: AAPL=3571 (Electronic Computers), AMGN=2836 (Biological
  Products), JPM=6021 (National Commercial Banks); recovered historical
  members classify correctly (XLNX=3674 Semiconductors, CELG=2834
  Pharmaceutical Preparations).
- KNOWN CAVEATS (disclosed on every segment result, spec section 4 rule 5):
  the code is each filer's CURRENT classification applied backward over the
  whole window (no PIT reclassification history); segment membership is
  static across the window; unmapped symbols belong to NO segment (never
  guessed). Regenerate deliberately, review the diff, and update this entry.
