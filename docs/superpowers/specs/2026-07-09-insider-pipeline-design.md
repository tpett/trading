# Form 4 Insider Pipeline (design spec)

**Status:** approved design, 2026-07-09. Tier 2 sub-project B. DATA + SIGNAL
REGISTRATION ONLY: the sweep belongs to the combined options-v2 + insider
batch spec (one pre-registered read after both data sets land).
**Scope:** the DERA insider-transactions store, its panel accessor, three
purchase-side signals registered with a `requires_insider` refusal flag, and
verification. Out of scope: opportunistic-vs-routine modeling
(Cohen-Malloy-Pomorski — deferred), sales-side signals, any sweep.

## 1. Purpose

Insider purchases are the strongest free, PIT-clean anomaly source we
identified: official SEC DERA structured bulk data, filed within 2 business
days of the transaction, as-reported forever, covering delisted names. The
pipeline turns it into a per-symbol event store the alphasearch engine can
score.

## 2. Data source and store

- **Source:** SEC DERA "Insider Transactions Data Sets" quarterly ZIPs
  (`https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/YYYYqQ_form345.zip`),
  2018q3 through the latest available (2018q3 start gives the trailing-90d
  windows full coverage from 2019-01). Stdlib urllib, the SEC fair-access
  throttle conventions, downloads to a scratch dir (NOT committed), parsed
  and discarded.
- **Parse:** `SUBMISSION` (ACCESSION_NUMBER, FILING_DATE, ISSUERCIK) joined
  to `NONDERIV_TRANS` (transaction date, TRANS_CODE, shares, price per
  share, acquired/disposed flag) and `REPORTINGOWNER` (owner CIK,
  relationship flags: officer/director/ten-percent).
- **Filter:** open-market transactions only — TRANS_CODE `P` (purchase) and
  `S` (sale). Everything else (awards, exercises, gifts, plans) is excluded;
  10b5-1 flagged rows are NOT excluded (the flag is unreliable pre-2023;
  documented limitation).
- **Symbol mapping:** ISSUERCIK → symbol via the existing
  `cik_map.csv` + `cik_map_historical.csv` intervals (the transaction's
  FILING_DATE selects the interval). Unmapped CIKs are counted and reported,
  never guessed (~2% expected, the known residual).
- **Store:** per-symbol parquet under `data/insider/equities/` (gitignored),
  append-only rebuild semantics (the build script regenerates the store
  whole; a `.source` marker like the bar caches). Columns:
  `filed` (index, the PIT key), `trans_date`, `code` (P/S), `shares`,
  `price`, `value` (shares×price), `owner_cik`, `is_officer`, `is_director`,
  `is_ten_pct`. One row per (accession, transaction row).

## 3. Panel accessor and signals

- `panel.py`: `load_insider(insider_dir, symbols)` +
  `PanelData.insider` + `PanelView.insider_window(symbol, days=90)` — rows
  whose FILED date lies in `(as_of − days, as_of]` (calendar days, PIT by
  filing). Absent store → empty dict → `requires_insider` signals refused at
  sweep assembly (a new `SignalSpec.requires_insider` flag + universe
  support check, mirroring requires_fundamentals; `UniverseSpec` gains
  `insider_dir` the same way).
- **Signals (frozen here; sign = higher more attractive):**

| name | definition | sign rationale |
|---|---|---|
| `npr_90` | (Σ buy value − Σ sell value) / (Σ buy value + Σ sell value) over trailing 90d filed | + net purchase ratio (Lakonishok-Lee); NaN when no P/S rows in window |
| `cluster_buys_90` | count of DISTINCT owner_cik with ≥1 `P` row in the window | + cluster buying = conviction; 0 is a real value (names with insider data but no buys), NaN only when the symbol has NO insider rows EVER filed ≤ as_of (never-covered ≠ quiet) |
| `officer_buy_90` | Σ `P` value where is_officer / (shares_outstanding × close_raw at as_of) | + officer purchases carry the most information; raw-price basis (the div_yield lesson); NaN when fundamentals shares or close_raw unavailable |

- `officer_buy_90` additionally requires fundamentals (shares_outstanding) —
  it registers with BOTH `requires_insider` and `requires_fundamentals`.

## 4. Verification (before the data is trusted)

- Build-script coverage report: quarters fetched, rows parsed, P/S counts,
  mapped-symbol rate vs the membership window (expect ~97-98%), per-year row
  counts (a missing quarter shows as a hole).
- Spot-checks against EDGAR's website for 2-3 known filings (a famous CEO
  purchase: filing date, shares, price match the rendered Form 4).
- PIT: signals read FILED date only; the no-look-ahead perturbation test
  gains insider-store perturbation coverage (the registry test auto-covers
  the signals; the panel assembly extension must perturb insider rows after
  T — same pattern as fundamentals).
- Unit tests: parse fixtures (synthetic sub/trans/owner tables incl. a
  non-P/S code row excluded, an unmapped CIK counted, an officer flag),
  window boundary (filed exactly as_of included; as_of−90 excluded),
  NaN conventions per the table above, refusal wiring end-to-end.
- Full suite warnings-as-errors; ruff clean; subagent build + adversarial
  review (look-ahead: TRANS_DATE vs FILED — scoring must never key on
  transaction date, which precedes filing).

## 5. Error handling

Missing/corrupt quarterly ZIP → named error, build continues with the gap
REPORTED (a hole in coverage is loud, never silent); unparseable rows
counted + skipped; a symbol with zero rows simply has no parquet (absent =
never-covered semantics per the signal table).

## 6. Trial cost (recorded for the batch spec)

3 signals × compatible universes journal in the combined batch sweep (the
batch spec discloses the full count with the options-v2 additions). Nothing
sweeps under this spec.
