# Form 4 Insider Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SEC DERA Form 4 insider-transaction store, its PIT panel accessor, and the three frozen purchase-side signals (`npr_90`, `cluster_buys_90`, `officer_buy_90`) with a `requires_insider` refusal flag, per `docs/superpowers/specs/2026-07-09-insider-pipeline-design.md`. DATA + SIGNAL REGISTRATION ONLY — nothing sweeps under this plan (the sweep belongs to the combined options-v2 + insider batch spec).

**Architecture:** A new parser `src/trading/fundamentals/form345.py` (sibling of `edgar.py` — same package that owns SEC ZIP parsing, `cik_map`, and the fundamentals store) turns each quarterly `YYYYqQ_form345.zip` into open-market P/S transaction rows. `scripts/build_insider_store.py` downloads 2018q3→latest (throttled, cached scratch dir), maps ISSUERCIK→symbol through the existing `cik_map.csv` FILED-date intervals, and regenerates the per-symbol parquet store `data/insider/equities/` whole. `panel.py` gains `load_insider` + `PanelData.insider` + `PanelView.insider_window(symbol, days=90)` (rows FILED in `(as_of−days, as_of]`); `spec.py` gains the three registrations behind a new `SignalSpec.requires_insider` flag; `sweep.py` refuses insider signals on store-less universes at assembly time, exactly like `requires_fundamentals`.

**Tech Stack:** Python ≥3.12, pandas 2.3.1, pyarrow 20.0.0, stdlib `urllib` ONLY for HTTP (repo policy — no requests), pytest 8.4.1 (warnings-as-errors), ruff 0.12.3 (E,F,I,UP,B @ 100 cols).

## Global Constraints

- **The spec §3 signal table is FROZEN pre-registered science.** Definitions, signs, and NaN conventions transcribe exactly; if implementation seems to require changing any of them, STOP and consult the developer (that is a spec amendment, not a code fix). The frozen facts:
  - `npr_90` = (Σ buy value − Σ sell value) / (Σ buy value + Σ sell value) over trailing 90 FILED days; sign +; NaN when no P/S value in the window.
  - `cluster_buys_90` = count of DISTINCT `owner_cik` with ≥1 `P` row in the window; sign +; **0 is a real value** (covered but no buys); **NaN only when the symbol has NO insider row EVER filed ≤ as_of** (never-covered ≠ quiet).
  - `officer_buy_90` = Σ `P` value where `is_officer` / (shares_outstanding × **close_raw** at as_of); sign +; NaN when fundamentals shares or close_raw unavailable; registers with BOTH `requires_insider` AND `requires_fundamentals`. Raw-price basis is the div_yield lesson: Tiingo's adjusted close bakes in future corporate actions, and Form 4 dollar values are raw dollars.
- **PIT: scoring keys the FILED date ONLY.** `TRANS_DATE` precedes filing (2-business-day rule; longer for late filers) — keying on it is look-ahead. The store index is `filed`; `trans_date` is a reporting column. The lookahead test must perturb rows' post-T FILED dates specifically (Task 5).
- **Filter frozen (spec §2):** `TRANS_CODE` ∈ {`P`, `S`} only. Awards/exercises/gifts/plans excluded; 10b5-1-flagged rows NOT excluded (flag unreliable pre-2023 — documented limitation).
- **Absent ≠ 0:** a quarter with no buys is 0 for `cluster_buys_90`, but a never-covered symbol is NaN. `PanelView.insider_window` encodes this: `None` = never covered at as_of; empty frame = covered-but-quiet.
- **The registry lands at 40 signals** (16 seeds + 21 Tier-1 + 3 insider). Every hardcoded count updates in Task 4: `tests/test_alphasearch_spec.py:178` (37→40), `tests/test_alphasearch_golden.py:62` (37→40), `:92` (74→80), `:93` (37→40).
- **DERA schemas drift.** Task 1 Step 1 downloads ONE real quarter ZIP and prints the actual headers BEFORE any parse code is committed. STOP and consult the developer if headers differ materially from the pinned names.
- **Data acquisition is an OPS step** the orchestrator runs post-merge (see "Post-merge ops" at the end) — the multi-GB/multi-minute download+build is NOT a plan task; the plan only ships the script and the exact command.
- Repo style rules: `is None` checks, never truthiness for maybe-None values (empty-dict truthiness like the existing `not panel.fundamentals` is the established convention in `_check_universe_supports` — match it); `pd.Timedelta(N, unit="D")`; never `pd.concat` with an empty frame (pandas 2.x FutureWarning + warnings-as-errors); `kind="mergesort"` for reproducible sorts; no float-exact assertions on compounded synthetic fixtures (`math.isclose`); hand-computable fixtures.
- Run `uv run pytest -q` and `uv run ruff check src tests scripts` before every commit. Commits granular, message suffix ` [AI]`.
- Work from the repo root `/Users/travis/Source/personal/trading`. All paths below are repo-relative. Do NOT commit anything under `data/` (gitignored via `/data/`).

## File Structure

| File | Change |
|---|---|
| `src/trading/fundamentals/form345.py` | NEW: `parse_quarter(zip_path) -> (transactions, skipped)`, `TRANSACTION_COLUMNS`, `INSIDER_COLUMNS`, `empty_transactions()` |
| `scripts/build_insider_store.py` | NEW: download loop 2018q3→latest, CIK→symbol interval mapping, whole-store parquet write + `.source` marker, coverage report |
| `src/trading/alphasearch/panel.py` | `load_insider`, `PanelData.insider`, `PanelView.insider_window`, `build_panel(..., insider_dir=...)` |
| `src/trading/alphasearch/sweep.py` | `UniverseSpec.insider_dir`, `default_universes` wiring, `build_universe_panel` threading, `_check_universe_supports` insider refusal |
| `src/trading/alphasearch/segments.py` | deep pools carry `insider_dir` when the store exists (the fundamentals §3.4-amendment pattern); opt pools carry it like `fundamentals_dir` |
| `src/trading/alphasearch/spec.py` | `SignalSpec.requires_insider`, `_register` param, 3 registrations |
| `src/trading/cli.py` | segment-safe hint list excludes `requires_insider` signals; hint text names the insider store |
| `tests/test_form345.py` | NEW: parser unit tests on synthetic form345 ZIP fixtures |
| `tests/test_build_insider_store.py` | NEW: mapping/store-write/gap-reporting unit tests (build_cik_map_historical fixture style) |
| `tests/test_alphasearch_panel.py` | `load_insider` + `insider_window` boundary/None-vs-empty tests |
| `tests/test_alphasearch_insider.py` | NEW: hand-computed unit tests for the 3 signals (signs! NaN conventions!) |
| `tests/test_alphasearch_spec.py` | registry count 37→40, insider family flags |
| `tests/test_alphasearch_sweep.py` | insider refusal tests (factory + deep-universe end-to-end) |
| `tests/test_alphasearch_cli.py` | hint excludes insider signals |
| `tests/test_alphasearch_segments.py` | `insider_dir` attach tests |
| `tests/test_alphasearch_golden.py` | insider store in `_write_universe`, counts 40/80, one insider signal asserted non-error |
| `tests/test_alphasearch_lookahead.py` | insider perturbation (values + post-T FILED dates) + filed-vs-trans-date PIT test |
| `tests/alphasearch_helpers.py` | `assemble_panel(..., insider=...)`, `make_panel(with_insider=True)` deterministic insider fixture |
| `docs/glossary.md`, `docs/experiments.md` | Form 4 / NPR / cluster-buying entries; registered-pre-sweep note |

**Known caveats to carry into the final report:** (1) Form 4/A amendments have their own accessions, so an amended transaction can appear twice (original + amendment) — the spec's parse table has no form-type filter; do NOT invent one (report it as a documented limitation). (2) The largecap golden fixture bar cache is the narrow schema, so `officer_buy_90` journals an honest error trial there (like `div_yield`); `npr_90`/`cluster_buys_90` are the golden's non-error insider representatives. (3) `RPTOWNER_RELATIONSHIP` value spellings are pinned by the Task 1 discovery step, not from memory.

---

### Task 1: DERA schema discovery + `form345.py` parser

**Files:**
- Create: `src/trading/fundamentals/form345.py`
- Test: `tests/test_form345.py`

**Interfaces:**
- Produces: `parse_quarter(zip_path: Path) -> tuple[pd.DataFrame, int]` — (TRANSACTION_COLUMNS frame, skipped-row count). `TRANSACTION_COLUMNS = ["accession", "issuer_cik", "filed", "trans_date", "code", "shares", "price", "value", "owner_cik", "is_officer", "is_director", "is_ten_pct"]` with `filed`/`trans_date` tz-aware UTC, ciks int64, flags bool, `value = shares × price` (NaN when either is missing). `INSIDER_COLUMNS = ["trans_date", "code", "shares", "price", "value", "owner_cik", "is_officer", "is_director", "is_ten_pct"]` (the store schema minus the `filed` index). `empty_transactions() -> pd.DataFrame`.
- Consumes: nothing from other tasks (pure file parsing, like `edgar.load_quarter_facts`).

- [ ] **Step 1: Discovery — download ONE real quarter and print the actual schema (STOP gate)**

DERA schemas drift between vintages; pin the truth before writing parse code. Run:

```bash
mkdir -p /tmp/dera-form345
curl -sS -A "trading-system travis@launchsupply.com" \
  -o /tmp/dera-form345/2024q1_form345.zip \
  "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2024q1_form345.zip"
ls -la /tmp/dera-form345/
```

Then:

```bash
uv run python - <<'EOF' 2>&1 | tee /tmp/claude-form345-discovery.log
import zipfile
import pandas as pd

zp = "/tmp/dera-form345/2024q1_form345.zip"
with zipfile.ZipFile(zp) as zf:
    print("MEMBERS:", zf.namelist())
    for name in ("SUBMISSION.tsv", "NONDERIV_TRANS.tsv", "REPORTINGOWNER.tsv"):
        with zf.open(name) as fh:
            head = pd.read_csv(fh, sep="\t", dtype=str, encoding="latin-1", nrows=3)
        print(f"\n=== {name} columns ===\n{list(head.columns)}")
        print(head.to_string())
    with zf.open("SUBMISSION.tsv") as fh:
        sub = pd.read_csv(fh, sep="\t", dtype=str, encoding="latin-1",
                          usecols=["FILING_DATE"], nrows=200)
    print("\nFILING_DATE samples:", sub["FILING_DATE"].dropna().unique()[:5])
    with zf.open("REPORTINGOWNER.tsv") as fh:
        own = pd.read_csv(fh, sep="\t", dtype=str, encoding="latin-1",
                          usecols=["RPTOWNER_RELATIONSHIP"])
    print("\nRPTOWNER_RELATIONSHIP distinct values:",
          sorted(own["RPTOWNER_RELATIONSHIP"].fillna("").unique())[:30])
    with zf.open("NONDERIV_TRANS.tsv") as fh:
        tr = pd.read_csv(fh, sep="\t", dtype=str, encoding="latin-1",
                         usecols=["TRANS_CODE", "TRANS_ACQUIRED_DISP_CD"])
    print("\nTRANS_CODE counts:\n", tr["TRANS_CODE"].value_counts().head(15))
    print("\nP/S vs ACQUIRED_DISP crosstab:\n",
          pd.crosstab(tr["TRANS_CODE"].where(tr["TRANS_CODE"].isin(["P", "S"])),
                      tr["TRANS_ACQUIRED_DISP_CD"]))
EOF
```

**Expected** (what the parser below pins): members include `SUBMISSION.tsv`, `NONDERIV_TRANS.tsv`, `REPORTINGOWNER.tsv`; SUBMISSION carries `ACCESSION_NUMBER`, `FILING_DATE`, `ISSUERCIK`; NONDERIV_TRANS carries `ACCESSION_NUMBER`, `TRANS_DATE`, `TRANS_CODE`, `TRANS_SHARES`, `TRANS_PRICEPERSHARE`, `TRANS_ACQUIRED_DISP_CD`; REPORTINGOWNER carries `ACCESSION_NUMBER`, `RPTOWNERCIK`, `RPTOWNER_RELATIONSHIP`; dates look like `02-JAN-2024` (`%d-%b-%Y`); relationship values are text containing `Officer` / `Director` / `TenPercentOwner` (possibly comma-joined); P rows are overwhelmingly `A`-coded and S rows `D`-coded in the crosstab.

**STOP CONDITIONS — consult the developer before proceeding if:** any of the three member files is missing or named differently; any pinned column is absent (e.g. relationship ships as separate `IS_OFFICER`-style boolean columns instead of one text field); the date format is not `DD-MON-YYYY`. Record the printed headers verbatim in the task's commit message body. If everything matches, adjust NOTHING and continue.

- [ ] **Step 2: Write the failing parser tests**

Create `tests/test_form345.py`:

```python
"""form345 parser: synthetic DERA ZIP fixtures (headers pinned by the plan's
discovery step against the real 2024q1 ZIP)."""

from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from trading.fundamentals.form345 import (
    INSIDER_COLUMNS,
    TRANSACTION_COLUMNS,
    empty_transactions,
    parse_quarter,
)

# Real files carry many more columns; fixtures include extras to prove the
# parser selects by NAME (usecols), not position.
SUB_HEADER = ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "ISSUERCIK", "ISSUERNAME"]
TRANS_HEADER = [
    "ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "TRANS_DATE", "TRANS_CODE",
    "EQUITY_SWAP_INVOLVED", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
    "TRANS_ACQUIRED_DISP_CD",
]
OWNER_HEADER = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP"]


def _tsv(header: list[str], rows: list[tuple]) -> str:
    lines = ["\t".join(header)]
    lines += ["\t".join("" if v is None else str(v) for v in row) for row in rows]
    return "\n".join(lines) + "\n"


def _write_zip(path, sub_rows, trans_rows, owner_rows,
               *, owner_header=OWNER_HEADER) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("SUBMISSION.tsv", _tsv(SUB_HEADER, sub_rows))
        zf.writestr("NONDERIV_TRANS.tsv", _tsv(TRANS_HEADER, trans_rows))
        zf.writestr("REPORTINGOWNER.tsv", _tsv(owner_header, owner_rows))


def _basic_zip(path) -> None:
    _write_zip(
        path,
        sub_rows=[
            ("acc-1", "05-FEB-2024", "01-FEB-2024", "1750", "AEROQUIP"),
            ("acc-2", "12-FEB-2024", "08-FEB-2024", "320193", "APPLE INC"),
        ],
        trans_rows=[
            # acc-1: an open-market buy and a sale
            ("acc-1", "1", "01-FEB-2024", "P", "0", "100", "50.5", "A"),
            ("acc-1", "2", "01-FEB-2024", "S", "0", "40", "51.0", "D"),
            # acc-1: an award -- non-P/S, must be excluded
            ("acc-1", "3", "01-FEB-2024", "A", "0", "999", "0.0", "A"),
            # acc-2: a buy with a footnote (missing) price -> value NaN, kept
            ("acc-2", "1", "08-FEB-2024", "P", "0", "200", None, "A"),
        ],
        owner_rows=[
            ("acc-1", "9001", "DOE JANE", "Officer, Director"),
            ("acc-2", "9002", "ROE RICHARD", "TenPercentOwner"),
        ],
    )


def test_parse_quarter_joins_filters_and_types(tmp_path):
    path = tmp_path / "2024q1_form345.zip"
    _basic_zip(path)
    tx, skipped = parse_quarter(path)
    assert skipped == 0
    assert list(tx.columns) == TRANSACTION_COLUMNS
    assert len(tx) == 3                      # the award row is excluded
    assert set(tx["code"]) == {"P", "S"}
    buy = tx[(tx["accession"] == "acc-1") & (tx["code"] == "P")].iloc[0]
    assert buy["issuer_cik"] == 1750
    assert buy["filed"] == pd.Timestamp("2024-02-05", tz="UTC")   # SUBMISSION date
    assert buy["trans_date"] == pd.Timestamp("2024-02-01", tz="UTC")
    assert buy["shares"] == 100.0
    assert buy["value"] == pytest.approx(100 * 50.5)
    assert buy["owner_cik"] == 9001
    assert bool(buy["is_officer"]) and bool(buy["is_director"])
    assert not bool(buy["is_ten_pct"])
    ten = tx[tx["accession"] == "acc-2"].iloc[0]
    assert bool(ten["is_ten_pct"]) and not bool(ten["is_officer"])


def test_missing_price_keeps_the_row_with_nan_value(tmp_path):
    # A footnote-priced transaction is REAL (cluster_buys counts it) but its
    # dollar value is unmeasured: NaN, never a fabricated 0.
    path = tmp_path / "q.zip"
    _basic_zip(path)
    tx, _ = parse_quarter(path)
    noprice = tx[tx["accession"] == "acc-2"].iloc[0]
    assert pd.isna(noprice["price"]) and pd.isna(noprice["value"])
    assert noprice["shares"] == 200.0


def test_multi_owner_filing_collapses_to_one_row_with_flags_ored(tmp_path):
    # A trust + its officer trustee filing jointly is ONE trade (spec: one row
    # per (accession, transaction row)); it counts as officer buying when ANY
    # reporting owner is an officer. Representative owner = lowest cik.
    path = tmp_path / "q.zip"
    _write_zip(
        path,
        sub_rows=[("acc-9", "20-MAR-2024", "18-MAR-2024", "55", "XCO")],
        trans_rows=[("acc-9", "1", "18-MAR-2024", "P", "0", "10", "5.0", "A")],
        owner_rows=[
            ("acc-9", "7002", "SMITH TRUST", ""),
            ("acc-9", "7001", "SMITH SAM", "Officer"),
        ],
    )
    tx, skipped = parse_quarter(path)
    assert skipped == 0
    assert len(tx) == 1                      # never duplicated per owner
    row = tx.iloc[0]
    assert row["owner_cik"] == 7001          # lowest cik, deterministic
    assert bool(row["is_officer"])


def test_unjoinable_rows_are_skipped_and_counted(tmp_path):
    # spec section 5: unparseable rows counted + skipped, never silent.
    path = tmp_path / "q.zip"
    _write_zip(
        path,
        sub_rows=[
            ("acc-bad-date", "NOT-A-DATE", "", "77", "BADCO"),
            ("acc-ok", "11-APR-2024", "", "88", "OKCO"),
        ],
        trans_rows=[
            ("acc-bad-date", "1", "09-APR-2024", "P", "0", "10", "1.0", "A"),
            ("acc-no-sub", "1", "09-APR-2024", "P", "0", "10", "1.0", "A"),
            ("acc-ok", "1", "09-APR-2024", "P", "0", "10", "1.0", "A"),   # no owner row
            ("acc-ok", "2", "09-APR-2024", "S", "0", "10", "1.0", "D"),   # no owner row
        ],
        owner_rows=[("acc-bad-date", "9001", "X", "Officer")],
    )
    tx, skipped = parse_quarter(path)
    # bad filing date kills acc-bad-date, no SUBMISSION kills acc-no-sub, and
    # acc-ok has no REPORTINGOWNER row: all four P/S rows are skipped.
    assert skipped == 4
    assert tx.empty
    assert list(tx.columns) == TRANSACTION_COLUMNS


def test_schema_drift_fails_loudly(tmp_path):
    # A renamed DERA column must raise (usecols mismatch), never mis-parse.
    path = tmp_path / "q.zip"
    _write_zip(
        path,
        sub_rows=[("acc-1", "05-FEB-2024", "", "1750", "X")],
        trans_rows=[("acc-1", "1", "01-FEB-2024", "P", "0", "1", "1.0", "A")],
        owner_rows=[("acc-1", "9001", "X", "Officer")],
        owner_header=["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RELATIONSHIP_TXT"],
    )
    with pytest.raises(ValueError):
        parse_quarter(path)


def test_quarter_with_no_open_market_rows_is_empty_not_an_error(tmp_path):
    path = tmp_path / "q.zip"
    _write_zip(
        path,
        sub_rows=[("acc-1", "05-FEB-2024", "", "1750", "X")],
        trans_rows=[("acc-1", "1", "01-FEB-2024", "A", "0", "1", "1.0", "A")],
        owner_rows=[("acc-1", "9001", "X", "Officer")],
    )
    tx, skipped = parse_quarter(path)
    assert tx.empty and skipped == 0
    assert list(tx.columns) == TRANSACTION_COLUMNS


def test_empty_transactions_dtypes_and_store_columns():
    empty = empty_transactions()
    assert list(empty.columns) == TRANSACTION_COLUMNS
    assert str(empty["filed"].dtype) == "datetime64[ns, UTC]"
    assert str(empty["issuer_cik"].dtype) == "int64"
    assert str(empty["is_officer"].dtype) == "bool"
    # The store schema is the transaction row minus its keys (filed indexes it).
    assert INSIDER_COLUMNS == [
        "trans_date", "code", "shares", "price", "value", "owner_cik",
        "is_officer", "is_director", "is_ten_pct",
    ]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_form345.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.fundamentals.form345'`

- [ ] **Step 4: Write the parser**

Create `src/trading/fundamentals/form345.py`:

```python
"""SEC DERA "Insider Transactions Data Sets" quarterly ZIP parsing (spec:
2026-07-09 insider pipeline, section 2).

Parses one quarterly form345 ZIP (SUBMISSION.tsv + NONDERIV_TRANS.tsv +
REPORTINGOWNER.tsv; tab-delimited latin-1, the FSDS convention) into the
open-market transaction table (TRANSACTION_COLUMNS) that
scripts/build_insider_store.py maps onto symbols. Pure file parsing:
downloads live in the build script, mirroring edgar.py.

Filter is FROZEN by the spec: TRANS_CODE `P` (open-market purchase) and `S`
(open-market sale) only. Awards, exercises, gifts and plan transactions never
enter the store; 10b5-1-flagged rows are NOT excluded (the flag is unreliable
pre-2023 -- documented limitation). Form 4/A amendments carry their OWN
accessions and are not deduplicated against originals (the spec's parse table
has no form filter); a documented limitation, not a bug.

PIT: `filed` (SUBMISSION.FILING_DATE) is the ONLY scoring key downstream.
TRANS_DATE precedes filing by up to 2 business days (longer for late filers)
and is stored for reporting only -- keying anything on it is look-ahead
(tests/test_alphasearch_lookahead.py proves the panel never does).

Multi-owner filings (a trust and its officer trustee filing jointly) collapse
to ONE representative per accession -- lowest RPTOWNERCIK, relationship flags
OR-ed -- so a jointly-reported trade is one store row and counts as officer
buying when ANY reporting owner is an officer.

Missing TRANS_SHARES / TRANS_PRICEPERSHARE (footnote-priced rows) keep the
row with value=NaN: the transaction is real (cluster_buys counts it) but its
dollar value is unmeasured, never fabricated. Rows are SKIPPED and COUNTED
only when join keys are unusable: no valid SUBMISSION row (missing or
unparseable FILING_DATE / ISSUERCIK) or no REPORTINGOWNER row.

Column names and the %d-%b-%Y date format (e.g. 02-JAN-2024) were pinned
against the real 2024q1 ZIP by the plan's discovery step; a drifted schema
fails LOUDLY (read_csv usecols raises on a missing column).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

_SUB_FILE = "SUBMISSION.tsv"
_TRANS_FILE = "NONDERIV_TRANS.tsv"
_OWNER_FILE = "REPORTINGOWNER.tsv"

_SUB_COLS = ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK"]
_TRANS_COLS = [
    "ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
    "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD",
]
_OWNER_COLS = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"]

_DATE_FORMAT = "%d-%b-%Y"  # 02-JAN-2024 (DERA Oracle export; discovery-pinned)

TRANSACTION_COLUMNS = [
    "accession", "issuer_cik", "filed", "trans_date", "code",
    "shares", "price", "value", "owner_cik",
    "is_officer", "is_director", "is_ten_pct",
]
# The store schema (spec section 2): TRANSACTION_COLUMNS minus the join keys;
# `filed` becomes the per-symbol parquet index.
INSIDER_COLUMNS = [
    "trans_date", "code", "shares", "price", "value", "owner_cik",
    "is_officer", "is_director", "is_ten_pct",
]

_TX_DTYPES = {
    "accession": "object",
    "issuer_cik": "int64",
    "filed": "datetime64[ns, UTC]",
    "trans_date": "datetime64[ns, UTC]",
    "code": "object",
    "shares": "float64",
    "price": "float64",
    "value": "float64",
    "owner_cik": "int64",
    "is_officer": "bool",
    "is_director": "bool",
    "is_ten_pct": "bool",
}


def empty_transactions() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSACTION_COLUMNS).astype(_TX_DTYPES)


def _read_table(zf: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    # usecols raises ValueError on a missing column: DERA schema drift fails
    # LOUDLY here instead of silently mis-parsing.
    with zf.open(name) as fh:
        return pd.read_csv(fh, sep="\t", dtype=str, usecols=cols, encoding="latin-1")


def _collapse_owners(owners: pd.DataFrame) -> pd.DataFrame:
    """One representative owner per accession: lowest RPTOWNERCIK, flags OR-ed."""
    owners = owners.dropna(subset=["ACCESSION_NUMBER"]).copy()
    owners["owner_cik"] = pd.to_numeric(owners["RPTOWNERCIK"], errors="coerce")
    owners = owners[owners["owner_cik"].notna()]
    rel = owners["RPTOWNER_RELATIONSHIP"].fillna("").str.upper()
    owners["is_officer"] = rel.str.contains("OFFICER", regex=False)
    owners["is_director"] = rel.str.contains("DIRECTOR", regex=False)
    owners["is_ten_pct"] = rel.str.contains("TENPERCENTOWNER", regex=False)
    grouped = owners.groupby("ACCESSION_NUMBER", sort=False).agg(
        owner_cik=("owner_cik", "min"),
        is_officer=("is_officer", "any"),
        is_director=("is_director", "any"),
        is_ten_pct=("is_ten_pct", "any"),
    )
    return grouped.reset_index()


def parse_quarter(zip_path: Path) -> tuple[pd.DataFrame, int]:
    """One quarterly form345 ZIP -> (open-market transactions, skipped rows).

    Output: TRANSACTION_COLUMNS, one row per (accession, NONDERIV_TRANS row),
    P/S only, filed/trans_date tz-aware UTC. `skipped` counts P/S transaction
    rows dropped for unusable join keys (spec section 5: counted, never
    silent)."""
    with zipfile.ZipFile(zip_path) as zf:
        sub = _read_table(zf, _SUB_FILE, _SUB_COLS)
        trans = _read_table(zf, _TRANS_FILE, _TRANS_COLS)
        owners = _read_table(zf, _OWNER_FILE, _OWNER_COLS)

    trans = trans[trans["TRANS_CODE"].isin(["P", "S"])]
    trans = trans.dropna(subset=["ACCESSION_NUMBER"]).copy()
    if trans.empty:
        return empty_transactions(), 0

    sub = sub.dropna(subset=_SUB_COLS).copy()
    sub["filed"] = pd.to_datetime(sub["FILING_DATE"], format=_DATE_FORMAT, errors="coerce")
    sub["issuer_cik"] = pd.to_numeric(sub["ISSUERCIK"], errors="coerce")
    sub = sub[sub["filed"].notna() & sub["issuer_cik"].notna()]

    trans["trans_date"] = pd.to_datetime(
        trans["TRANS_DATE"], format=_DATE_FORMAT, errors="coerce"
    )
    trans["shares"] = pd.to_numeric(trans["TRANS_SHARES"], errors="coerce")
    trans["price"] = pd.to_numeric(trans["TRANS_PRICEPERSHARE"], errors="coerce")

    merged = trans.merge(
        sub[["ACCESSION_NUMBER", "filed", "issuer_cik"]], on="ACCESSION_NUMBER", how="left"
    ).merge(_collapse_owners(owners), on="ACCESSION_NUMBER", how="left")
    unjoined = merged["filed"].isna() | merged["owner_cik"].isna()
    skipped = int(unjoined.sum())
    merged = merged[~unjoined].copy()
    if merged.empty:
        return empty_transactions(), skipped

    merged["value"] = merged["shares"] * merged["price"]
    merged["issuer_cik"] = merged["issuer_cik"].astype("int64")
    merged["owner_cik"] = merged["owner_cik"].astype("int64")
    for flag in ("is_officer", "is_director", "is_ten_pct"):
        merged[flag] = merged[flag].astype(bool)
    merged["filed"] = merged["filed"].dt.tz_localize("UTC")
    merged["trans_date"] = merged["trans_date"].dt.tz_localize("UTC")
    merged = merged.rename(columns={"ACCESSION_NUMBER": "accession", "TRANS_CODE": "code"})
    return merged[TRANSACTION_COLUMNS].reset_index(drop=True), skipped
```

Note: `TRANS_ACQUIRED_DISP_CD` is read (so a drift there fails loudly and the discovery step could crosstab it) but the filter is TRANS_CODE alone — the spec's frozen filter.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_form345.py -q`
Expected: all PASS.

- [ ] **Step 6: (One-time sanity, not a committed test) run the parser against the real 2024q1 ZIP from Step 1**

```bash
uv run python - <<'EOF'
from pathlib import Path
from trading.fundamentals.form345 import parse_quarter
tx, skipped = parse_quarter(Path("/tmp/dera-form345/2024q1_form345.zip"))
print(len(tx), "P/S rows;", skipped, "skipped")
print(tx["code"].value_counts().to_dict())
print(tx.head(3).to_string())
EOF
```

Expected: tens of thousands of rows (S heavily outnumbers P), a small skipped count, plausible filed/trans_date pairs (filed ≥ trans_date, usually by ≤ 2 business days). If the numbers are wildly off (0 rows, >10% skipped), STOP and investigate before committing.

- [ ] **Step 7: Lint + full suite + commit**

```bash
uv run ruff check src tests scripts
uv run pytest -q
git add src/trading/fundamentals/form345.py tests/test_form345.py
git commit -m "Add DERA form345 quarterly ZIP parser (open-market P/S rows) [AI]"
```

Include the discovery-step header printout in the commit message body.

---

### Task 2: `scripts/build_insider_store.py`

**Files:**
- Create: `scripts/build_insider_store.py`
- Test: `tests/test_build_insider_store.py`

**Interfaces:**
- Consumes: `parse_quarter`, `INSIDER_COLUMNS` (Task 1); `quarter_range`, `last_complete_quarter` (`trading.fundamentals.backfill`); `load_cik_map` (`trading.fundamentals.cik_map`); `USER_AGENT` (`trading.fundamentals.edgar`).
- Produces: the store layout every later task relies on — `data/insider/equities/<SYMBOL>.parquet` (`/` in symbols → `-`, the `FundamentalsStore.path_for` rule), tz-aware UTC `filed` DatetimeIndex, columns exactly `INSIDER_COLUMNS`, plus a `.source` marker containing `form345`. Testable pure functions: `map_to_symbols(transactions, cik_map) -> tuple[dict[str, pd.DataFrame], int, int]` and `write_store(frames, store_root) -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_insider_store.py`:

```python
"""scripts/build_insider_store.py: offline unit tests on synthetic
transactions + cik_map fixtures (the build_cik_map_historical test pattern:
pure functions, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_insider_store as bis  # noqa: E402
from trading.fundamentals.form345 import INSIDER_COLUMNS  # noqa: E402


def _tx(rows: list[tuple]) -> pd.DataFrame:
    """(accession, issuer_cik, filed_iso, code, shares, price, owner_cik,
    is_officer) -> a TRANSACTION_COLUMNS-shaped frame."""
    out = pd.DataFrame(
        rows,
        columns=["accession", "issuer_cik", "filed", "code", "shares", "price",
                 "owner_cik", "is_officer"],
    )
    out["filed"] = pd.to_datetime(out["filed"]).dt.tz_localize("UTC")
    out["trans_date"] = out["filed"] - pd.Timedelta(2, unit="D")
    out["value"] = out["shares"] * out["price"]
    out["is_director"] = False
    out["is_ten_pct"] = False
    return out


def _cik_map(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["symbol", "cik", "start", "end"]).astype(str)
    df["cik"] = df["cik"].astype(int)
    return df


def test_map_to_symbols_uses_filed_date_intervals():
    # FB/META shape: one cik, two symbol intervals; the FILED date selects.
    tx = _tx([
        ("a1", 1326801, "2021-06-01", "P", 100.0, 10.0, 9001, True),
        ("a2", 1326801, "2022-06-01", "P", 200.0, 10.0, 9002, False),
    ])
    cmap = _cik_map([
        ("FB", 1326801, "2017-01-01", "2022-06-09"),
        ("META", 1326801, "2022-06-09", ""),
    ])
    frames, unmapped_rows, unmapped_ciks = bis.map_to_symbols(tx, cmap)
    assert set(frames) == {"FB"}             # both filings pre-rename
    assert len(frames["FB"]) == 2
    assert unmapped_rows == 0 and unmapped_ciks == 0
    assert list(frames["FB"].columns) == INSIDER_COLUMNS
    assert frames["FB"].index.name == "filed"
    assert str(frames["FB"].index.tz) == "UTC"


def test_map_to_symbols_counts_unmapped_never_guesses():
    tx = _tx([
        ("a1", 999999, "2021-06-01", "P", 100.0, 10.0, 9001, True),   # unknown cik
        ("a2", 55, "2010-06-01", "S", 50.0, 10.0, 9002, False),       # pre-interval
        ("a3", 55, "2021-06-01", "S", 50.0, 10.0, 9002, False),       # mapped
    ])
    cmap = _cik_map([("XCO", 55, "2017-01-01", "")])
    frames, unmapped_rows, unmapped_ciks = bis.map_to_symbols(tx, cmap)
    assert set(frames) == {"XCO"}
    assert len(frames["XCO"]) == 1
    assert unmapped_rows == 2
    assert unmapped_ciks == 2                # 999999 entirely + 55 partially


def test_map_to_symbols_shared_cik_maps_to_both_symbols():
    # GOOG/GOOGL: two concurrent intervals share one cik -- both symbols get
    # the row (the fundamentals backfill rule); nothing is unmapped.
    tx = _tx([("a1", 1652044, "2021-06-01", "P", 10.0, 100.0, 9001, True)])
    cmap = _cik_map([
        ("GOOG", 1652044, "2017-01-01", ""),
        ("GOOGL", 1652044, "2017-01-01", ""),
    ])
    frames, unmapped_rows, _ = bis.map_to_symbols(tx, cmap)
    assert set(frames) == {"GOOG", "GOOGL"}
    assert unmapped_rows == 0


def test_write_store_atomic_layout_and_marker(tmp_path):
    tx = _tx([
        ("a1", 55, "2021-06-01", "P", 100.0, 10.0, 9001, True),
        ("a2", 55, "2021-03-01", "S", 40.0, 10.0, 9002, False),
    ])
    cmap = _cik_map([("BRK/B", 55, "2017-01-01", "")])
    frames, _, _ = bis.map_to_symbols(tx, cmap)
    store = tmp_path / "insider" / "equities"
    bis.write_store(frames, store)
    path = store / "BRK-B.parquet"           # '/' sanitized like FundamentalsStore
    assert path.exists()
    got = pd.read_parquet(path)
    assert list(got.columns) == INSIDER_COLUMNS
    assert got.index.is_monotonic_increasing  # sorted by filed
    assert (store / ".source").read_text() == "form345"
    assert not list(store.glob("*.tmp"))     # atomic: no torn files left


def test_ensure_empty_refuses_a_populated_store(tmp_path):
    store = tmp_path / "equities"
    store.mkdir(parents=True)
    (store / "AAA.parquet").write_bytes(b"")
    with pytest.raises(SystemExit, match="Delete"):
        bis.ensure_empty(store)
    (store / "AAA.parquet").unlink()
    bis.ensure_empty(store)                  # empty dir: fine
    bis.ensure_empty(tmp_path / "missing")   # absent dir: fine


def test_gap_recording_shape():
    # main() records failed quarters and exits 1 -- the pure helper just
    # formats; pin the message carries quarter + reason (loud, greppable).
    line = bis.gap_line("2020q2", ValueError("boom"))
    assert "2020q2" in line and "ValueError" in line and "boom" in line
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_build_insider_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_insider_store'`

- [ ] **Step 3: Write the script**

Create `scripts/build_insider_store.py`:

```python
"""Build the per-symbol Form 4 insider-transaction store at
data/insider/equities/ (spec: 2026-07-09 insider pipeline, sections 2/4/5).

Downloads the SEC DERA "Insider Transactions Data Sets" quarterly ZIPs
2018q3 -> the last complete quarter (the 2018q3 start gives trailing-90d
windows full coverage from 2019-01) into data/edgar-insider-raw/ (gitignored
scratch; cached -- reruns never re-fetch), parses each with
trading.fundamentals.form345.parse_quarter (open-market P/S rows only), maps
ISSUERCIK -> symbol through the committed cik_map.csv FILED-date intervals
(unmapped ciks are COUNTED, never guessed), and regenerates the store WHOLE:
one parquet per symbol (filed index, form345.INSIDER_COLUMNS), atomic
tmp+os.replace writes, and a .source marker ("form345") like the bar caches.

Error handling (spec section 5): a quarter whose download or parse fails is
a NAMED coverage GAP -- the build continues, the gap is printed loudly in
the final report, and the exit code is 1 so the orchestrator cannot miss it.
The NEWEST quarter 404ing is publication lag, not a gap. The store must be
empty before a rebuild (whole-regeneration semantics: a rebuild on top of an
older build could leave stale symbols behind) -- delete its contents first.

Coverage report (spec section 4): per-quarter row/P/S/skipped counts,
per-year row counts (a missing quarter shows as a hole), unmapped row/cik
counts, and the window-membership coverage rate (symbols with >= 1 stored
row / membership symbols overlapping the discovery window; quiet companies
legitimately lack rows, so this is context, not a gate).

Usage: uv run python scripts/build_insider_store.py
(~31 quarterly ZIPs, a few MB each; minutes, throttled per SEC fair access)
"""

from __future__ import annotations

import datetime
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from trading.fundamentals.backfill import last_complete_quarter, quarter_range
from trading.fundamentals.cik_map import load_cik_map
from trading.fundamentals.edgar import USER_AGENT
from trading.fundamentals.form345 import INSIDER_COLUMNS, parse_quarter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_sic_map import WINDOW_END, WINDOW_START  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "edgar-insider-raw"
STORE_DIR = ROOT / "data" / "insider" / "equities"
MEMBERSHIP = ROOT / "src" / "trading" / "venues" / "universes" / "equities_membership.csv"
ZIP_URL = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/{quarter}_form345.zip"
)
FIRST_QUARTER = "2018q3"  # spec section 2: trailing-90d windows full from 2019-01
REQUEST_SPACING_S = 0.11  # SEC ceiling is 10 req/s; stay under it
SOURCE_MARKER = ".source"


def download(quarter: str, newest: bool) -> Path | None:
    """Fetch one quarterly ZIP into RAW_DIR (cached: on-disk files are never
    re-downloaded). Returns None on a 404 of the NEWEST quarter only
    (publication lag -- normal, not a gap). Any other failure raises and
    main() records it as a coverage gap."""
    dest = RAW_DIR / f"{quarter}_form345.zip"
    if dest.exists():
        return dest
    url = ZIP_URL.format(quarter=quarter)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"downloading {url}")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and newest:
            print(f"WARNING: {quarter}_form345.zip not published yet; skipping")
            return None
        raise
    tmp = dest.with_suffix(".zip.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    time.sleep(REQUEST_SPACING_S)
    return dest


def gap_line(quarter: str, exc: Exception) -> str:
    return f"{quarter}: {type(exc).__name__}: {exc}"


def map_to_symbols(
    transactions: pd.DataFrame, cik_map: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], int, int]:
    """Slice each issuer cik's rows into its cik_map symbol intervals by
    FILED date (the fundamentals rule: a renamed company's pre-rename filings
    land on the old symbol). Returns (per-symbol filed-indexed frames,
    unmapped row count, ciks with >= 1 unmapped row). A row can map to
    SEVERAL symbols (GOOG/GOOGL share one cik) -- both get it, like the
    fundamentals backfill; it is unmapped only when NO interval covers its
    (cik, filed)."""
    indexed = transactions.set_index("filed").sort_index(kind="mergesort")
    by_cik = {int(cik): frame for cik, frame in indexed.groupby("issuer_cik")}
    covered = {cik: np.zeros(len(frame), dtype=bool) for cik, frame in by_cik.items()}
    pieces: dict[str, list[pd.DataFrame]] = {}
    for row in cik_map.itertuples():
        frame = by_cik.get(int(row.cik))
        if frame is None:
            continue
        # A DatetimeIndex comparison already yields an ndarray[bool].
        mask = np.asarray(frame.index >= pd.Timestamp(row.start, tz="UTC"))
        if row.end:
            mask &= np.asarray(frame.index < pd.Timestamp(row.end, tz="UTC"))
        if mask.any():
            pieces.setdefault(row.symbol, []).append(frame[mask])
            covered[int(row.cik)] |= mask
    frames = {
        symbol: pd.concat(parts).sort_index(kind="mergesort")[INSIDER_COLUMNS]
        for symbol, parts in pieces.items()
    }
    unmapped_rows = sum(int((~flags).sum()) for flags in covered.values())
    unmapped_ciks = sum(1 for flags in covered.values() if not flags.all())
    return frames, unmapped_rows, unmapped_ciks


def write_store(frames: dict[str, pd.DataFrame], store_root: Path) -> None:
    """Whole-store write: per-symbol parquet, atomic tmp+os.replace, then the
    .source marker (the bar-cache convention)."""
    store_root.mkdir(parents=True, exist_ok=True)
    for symbol in sorted(frames):
        path = store_root / f"{symbol.replace('/', '-')}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        frames[symbol].to_parquet(tmp)
        os.replace(tmp, path)
    marker = store_root / SOURCE_MARKER
    tmp = marker.with_suffix(".tmp")
    tmp.write_text("form345")
    os.replace(tmp, marker)


def ensure_empty(store_root: Path) -> None:
    """The store is regenerated WHOLE: refuse a rebuild on top of an existing
    store (stale symbols from an older cik_map could silently survive)."""
    existing = sorted(store_root.glob("*.parquet"))
    if existing:
        raise SystemExit(
            f"ERROR: {store_root} already has {len(existing)} symbol file(s). "
            "The insider store is regenerated whole; a rebuild on top could "
            "leave stale symbols behind. Delete the directory's contents "
            f"(including {SOURCE_MARKER}) first, then rerun."
        )


def window_members(membership_path: Path) -> set[str]:
    """Membership symbols whose interval overlaps the discovery window."""
    df = pd.read_csv(membership_path, comment="#", dtype=str).fillna("")
    overlap = (df["start"] <= WINDOW_END) & ((df["end"] == "") | (df["end"] > WINDOW_START))
    return set(df.loc[overlap, "symbol"])


def main() -> None:
    ensure_empty(STORE_DIR)
    cik_map = load_cik_map()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    quarters = quarter_range(FIRST_QUARTER, last_complete_quarter(datetime.date.today()))

    quarter_frames: list[pd.DataFrame] = []
    gaps: list[str] = []
    total_skipped = 0
    for quarter in quarters:
        try:
            zip_path = download(quarter, newest=quarter == quarters[-1])
            if zip_path is None:
                continue  # newest not published yet: lag, not a gap
            tx, skipped = parse_quarter(zip_path)
        except (OSError, zipfile.BadZipFile, ValueError, KeyError) as exc:
            gaps.append(gap_line(quarter, exc))
            print(f"ERROR: {quarter} failed ({exc}); continuing -- GAP RECORDED")
            continue
        total_skipped += skipped
        n_p = int((tx["code"] == "P").sum())
        print(f"parsed {quarter}: {len(tx)} P/S rows ({n_p} P / {len(tx) - n_p} S), "
              f"{skipped} skipped")
        if not tx.empty:  # never concat an empty frame (warnings-as-errors)
            quarter_frames.append(tx)
    if not quarter_frames:
        sys.exit("FATAL: no quarter parsed; nothing written")

    transactions = pd.concat(quarter_frames, ignore_index=True)
    frames, unmapped_rows, unmapped_ciks = map_to_symbols(transactions, cik_map)
    write_store(frames, STORE_DIR)

    # ---- coverage report (spec section 4) ----
    n_p = int((transactions["code"] == "P").sum())
    mapped_rows = len(transactions) - unmapped_rows
    members = window_members(MEMBERSHIP)
    with_rows = members & set(frames)
    print(f"\nrows parsed: {len(transactions)} ({n_p} P / {len(transactions) - n_p} S); "
          f"{total_skipped} unparseable rows skipped")
    print(f"mapped to >=1 symbol: {mapped_rows} rows -> {len(frames)} symbols; "
          f"unmapped: {unmapped_rows} rows across {unmapped_ciks} ciks "
          "(non-members + the known ~2% cik_map residual; never guessed)")
    print(f"window-membership coverage: {len(with_rows)}/{len(members)} "
          f"({len(with_rows) / len(members):.1%}) members with >= 1 insider row "
          "(quiet companies legitimately lack rows)")
    print("\nrows per FILED year (a missing quarter shows as a hole):")
    per_year = transactions["filed"].dt.year.value_counts().sort_index()
    for year, count in per_year.items():
        print(f"  {year}: {count}")
    if gaps:
        print("\nCOVERAGE GAPS (spec section 5: loud, never silent):")
        for gap in gaps:
            print(f"  {gap}")
        sys.exit(1)
    print(f"\nwrote {len(frames)} symbol parquets to {STORE_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_build_insider_store.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint + full suite + commit**

```bash
uv run ruff check src tests scripts
uv run pytest -q
git add scripts/build_insider_store.py tests/test_build_insider_store.py
git commit -m "Add insider store build script (download, map, coverage report) [AI]"
```

---

### Task 3: Panel accessor + universe wiring + fixture

**Files:**
- Modify: `src/trading/alphasearch/panel.py` (`load_insider` near `load_fundamentals` ~line 164; `PanelData.insider` field; `PanelView.insider_window` after `fundamentals_row_prior` ~line 422; `build_panel` signature ~line 514)
- Modify: `src/trading/alphasearch/sweep.py` (`UniverseSpec` ~line 220, `default_universes` ~line 237, `build_universe_panel` ~line 272)
- Modify: `src/trading/alphasearch/segments.py` (`segment_universes` ~line 136)
- Modify: `tests/alphasearch_helpers.py` (`assemble_panel`, `make_panel`)
- Test: `tests/test_alphasearch_panel.py`, `tests/test_alphasearch_segments.py`

**Interfaces:**
- Produces: `INSIDER_WINDOW_DAYS = 90` (panel.py constant); `load_insider(root: Path, symbols: Iterable[str]) -> dict[str, pd.DataFrame]`; `PanelData.insider: dict[str, pd.DataFrame]`; `PanelView.insider_window(symbol: str, days: int = INSIDER_WINDOW_DAYS) -> pd.DataFrame | None` — **None = the symbol has no insider row filed ≤ as_of (never-covered); an empty frame = covered-but-quiet**; `build_panel(..., insider_dir: Path | None = None)`; `UniverseSpec.insider_dir: Path | None = None`; helpers `assemble_panel(..., insider=None)` and `make_panel(..., with_insider=True)` whose panels carry a deterministic insider fixture (recipe below — Task 4's anti-vacuity and Task 5's perturbation depend on it).
- Consumes: the Task 2 store layout (parquet columns `INSIDER_COLUMNS`, filed index).

- [ ] **Step 1: Write the failing panel tests**

Append to `tests/test_alphasearch_panel.py`:

```python
# --------------------------------------------------------------------------- #
# Insider store (Form 4 pipeline)
# --------------------------------------------------------------------------- #


def _insider_frame(rows: list[tuple[str, str, float, float, int]]) -> pd.DataFrame:
    """(filed_iso, code, shares, price, owner_cik) -> a store-shaped frame."""
    frame = pd.DataFrame(
        rows, columns=["filed", "code", "shares", "price", "owner_cik"]
    )
    frame["filed"] = pd.to_datetime(frame["filed"]).dt.tz_localize("UTC")
    frame["trans_date"] = frame["filed"] - pd.Timedelta(2, unit="D")
    frame["value"] = frame["shares"] * frame["price"]
    frame["is_officer"] = True
    frame["is_director"] = False
    frame["is_ten_pct"] = False
    return frame.set_index("filed").sort_index(kind="mergesort")


def test_insider_window_filed_date_boundaries():
    as_of = pd.Timestamp("2020-06-30", tz="UTC")
    frame = _insider_frame([
        ("2020-04-01", "P", 100.0, 10.0, 1),   # exactly as_of-90d: EXCLUDED
        ("2020-04-02", "P", 100.0, 10.0, 2),   # first included day
        ("2020-06-30", "S", 50.0, 10.0, 3),    # filed exactly at as_of: included
        ("2020-07-01", "P", 999.0, 10.0, 4),   # filed after as_of: invisible
    ])
    panel = PanelData(closes={}, insider={"AAA": frame}, symbols=("AAA",))
    window = panel.view(as_of).insider_window("AAA")
    assert window is not None
    assert list(window["owner_cik"]) == [2, 3]


def test_insider_window_none_vs_empty_is_the_covered_distinction():
    as_of = pd.Timestamp("2020-06-30", tz="UTC")
    old = _insider_frame([("2019-01-15", "P", 100.0, 10.0, 1)])
    panel = PanelData(closes={}, insider={"QUIET": old}, symbols=("QUIET", "NEVER"))
    view = panel.view(as_of)
    quiet = view.insider_window("QUIET")
    assert quiet is not None and quiet.empty     # covered-but-quiet: a real 0
    assert view.insider_window("NEVER") is None  # never covered: NaN downstream
    # Before ANY filing, a later-covered symbol is never-covered too.
    early = panel.view(pd.Timestamp("2019-01-10", tz="UTC"))
    assert early.insider_window("QUIET") is None


def test_load_insider_reads_store_and_never_creates_it(tmp_path):
    frame = _insider_frame([("2020-01-06", "P", 10.0, 5.0, 1)])
    store = tmp_path / "insider"
    store.mkdir()
    frame.to_parquet(store / "AAA.parquet")
    frame.to_parquet(store / "BRK-B.parquet")   # '/'-sanitized filename
    from trading.alphasearch.panel import load_insider
    got = load_insider(store, ["AAA", "BRK/B", "NOPE"])
    assert set(got) == {"AAA", "BRK/B"}
    assert got["AAA"]["value"].iloc[0] == 50.0
    absent = load_insider(tmp_path / "missing", ["AAA"])
    assert absent == {}
    assert not (tmp_path / "missing").exists()  # never invented


def test_build_panel_threads_insider_dir(tmp_path):
    cache = _write_cache(tmp_path, ("AAA",))
    store = tmp_path / "insider"
    store.mkdir()
    _insider_frame([("2020-01-06", "P", 10.0, 5.0, 1)]).to_parquet(store / "AAA.parquet")
    panel = build_panel(cache, None, None, insider_dir=store, symbols=("AAA",))
    assert set(panel.insider) == {"AAA"}
    bare = build_panel(cache, None, None, symbols=("AAA",))
    assert bare.insider == {}
```

Add `load_insider` to the existing `from trading.alphasearch.panel import (...)` block at the top of the file instead of the inline import if preferred — either is fine; be consistent with one style.

Append to `tests/test_alphasearch_segments.py`:

```python
def test_deep_segments_carry_insider_dir_when_the_store_exists(tmp_path):
    # Mirrors the fundamentals section 3.4-amendment conditional: None only
    # ever means "no store built yet". Opt pools carry it unconditionally,
    # exactly like fundamentals_dir (an absent store loads {} and the
    # requires_insider refusal fires at sweep assembly).
    _pharma, _banks, sic, membership = _fixture_root(tmp_path)
    universes, _ = segment_universes(tmp_path, sic, membership_path=membership)
    assert universes["largecap:biotech"].insider_dir is None
    store = tmp_path / "data" / "insider" / "equities"
    assert universes["opt-largecap:biotech"].insider_dir == store
    store.mkdir(parents=True)
    universes, _ = segment_universes(tmp_path, sic, membership_path=membership)
    assert universes["largecap:biotech"].insider_dir == store
    assert universes["opt-largecap:biotech"].insider_dir == store
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_alphasearch_panel.py tests/test_alphasearch_segments.py -q`
Expected: FAIL — `TypeError: PanelData.__init__() got an unexpected keyword argument 'insider'` (and the segments test on the missing `insider_dir` field).

- [ ] **Step 3: Implement panel.py**

In `src/trading/alphasearch/panel.py`, after `load_fundamentals`:

```python
def load_insider(root: Path, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Per-symbol Form 4 insider frames (FILED-date index, the store schema
    scripts/build_insider_store.py writes) for symbols that have any.
    Returns {} without creating anything when the store dir is absent --
    assembly must never invent an empty store (the fundamentals rule)."""
    if not root.exists():
        return {}
    out: dict[str, pd.DataFrame] = {}
    for symbol in sorted(set(symbols)):
        path = root / f"{symbol.replace('/', '-')}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            if not frame.empty:
                out[symbol] = frame
    return out
```

Add the module constant near `MIN_YOY_AGE_DAYS` (~line 123):

```python
INSIDER_WINDOW_DAYS = 90  # insider spec section 3: trailing filed-date window
```

Add to `PanelView` (after `fundamentals_row_prior`):

```python
    def insider_window(
        self, symbol: str, days: int = INSIDER_WINDOW_DAYS
    ) -> pd.DataFrame | None:
        """Form 4 rows FILED in (as_of - days, as_of] -- calendar days, PIT
        by FILING date (TRANS_DATE precedes filing and must never key
        anything). None when the symbol has NO insider row filed at or
        before as_of (never-covered: the signal-table NaN case); an EMPTY
        frame means covered-but-quiet, a real observation (cluster_buys_90
        scores it 0, never NaN)."""
        frame = self._panel.insider.get(symbol)
        if frame is None or frame.empty:
            return None
        visible = frame.loc[: self.as_of]
        if visible.empty:
            return None
        return visible[visible.index > self.as_of - pd.Timedelta(days, unit="D")]
```

Add the `PanelData` field (after `fundamentals`; every construction site in the repo is keyword-based — verified — so inserting mid-dataclass is safe):

```python
    # symbol -> Form 4 open-market transactions (filed-date index, the
    # data/insider/equities store schema). Absent store -> {} -> the
    # requires_insider refusal at sweep assembly.
    insider: dict[str, pd.DataFrame] = field(default_factory=dict)
```

In `build_panel`: add the keyword-only parameter `insider_dir: Path | None = None` (after `fundamentals_dir` in the signature's keyword section, next to `symbols`/`factors`/`sectors`), and inside:

```python
    insider = load_insider(insider_dir, closes) if insider_dir is not None else {}
```

and in the `PanelData(...)` construction:

```python
        insider={s: insider[s] for s in universe if s in insider},
```

Also extend `build_panel`'s docstring with one line: `insider_dir` mirrors `fundamentals_dir` (absent/None → empty dict → `requires_insider` refusal at sweep assembly).

- [ ] **Step 4: Implement sweep.py + segments.py wiring**

`sweep.py` — `UniverseSpec` gains (between `fundamentals_dir` and `symbols`):

```python
    # Form 4 insider store; None = no store, requires_insider signals are
    # refused at sweep assembly (mirrors fundamentals_dir).
    insider_dir: Path | None = None
```

`default_universes`: add to BOTH specs (keyword):

```python
            insider_dir=root / "data" / "insider" / "equities",
```

`build_universe_panel`:

```python
    return build_panel(
        spec.cache_dir, spec.samples, spec.fundamentals_dir,
        insider_dir=spec.insider_dir,
        symbols=spec.symbols, factors=factors,
        sectors=_universe_sectors(spec.sic_map_path),
    )
```

`segments.py` — in `segment_universes`, next to the existing `fundamentals_dir = ...` line:

```python
    insider_dir = root / "data" / "insider" / "equities"
```

Deep-pool `UniverseSpec` gains (next to the fundamentals conditional, same rationale comment style):

```python
                    # Same store-exists conditional as fundamentals_dir
                    # (Tier-1 spec section 3.4 pattern): None only ever means
                    # "no insider store built yet". No insider segment trial
                    # predates this, so nothing is spent.
                    insider_dir=insider_dir if insider_dir.is_dir() else None,
```

Opt-pool `UniverseSpec` gains (unconditional, like `fundamentals_dir` there):

```python
                    insider_dir=insider_dir,
```

- [ ] **Step 5: Extend the test helpers**

In `tests/alphasearch_helpers.py`:

`assemble_panel` gains a keyword param and threads it:

```python
def assemble_panel(
    bars: dict[str, pd.DataFrame],
    options: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    factors: pd.DataFrame,
    *,
    insider: dict[str, pd.DataFrame] | None = None,
    has_option_volume: bool = False,
    sectors: dict[str, str] | None = None,
) -> PanelData:
    """PanelData from raw stores, deriving what build_panel derives (closes
    from bars). The lookahead test perturbs RAW stores and reassembles
    through here, so derived state (the precomputed rolling features) is
    recomputed from the perturbed inputs, never perturbed directly."""
    closes = {s: frame["close"] for s, frame in bars.items()}
    return PanelData(
        closes=closes, options=options, fundamentals=fundamentals,
        insider={} if insider is None else insider,
        symbols=tuple(sorted(bars)), bars=bars, factors=factors,
        features=compute_rolling_features(closes, factors),
        has_option_volume=has_option_volume,
        sectors={} if sectors is None else sectors,
    )
```

`make_panel` gains `with_insider: bool = True` and, after the fundamentals block (NO rng use — closes must stay bit-identical; extend the docstring with the recipe):

```python
    insider: dict[str, pd.DataFrame] = {}
    if with_insider:
        # Deterministic Form 4 fixture: at every month-first FILED date,
        # symbol i gets (i % 8) distinct buyers (owner_cik 1000*(i+1)+j, the
        # first an officer, the second a director) buying 100*(i+1) sh @ 10,
        # plus ONE ten-pct seller of 50*(i+1) sh @ 10. i % 8 == 0 symbols are
        # covered-but-buyless (cluster_buys' real 0; npr -1). trans_date =
        # filed - 20d, so month-first filings just after a cutoff carry
        # trans_dates BEFORE it -- the straddling rows the lookahead
        # perturbation needs to catch trans_date keying. 8 distinct cluster
        # values keeps segment-free sorts non-degenerate (>= 3 buckets).
        for i, sym in enumerate(names):
            rows: list[dict] = []
            for date in month_firsts(idx):
                trans = date - pd.Timedelta(20, unit="D")
                for j in range(i % 8):
                    rows.append({
                        "filed": date, "trans_date": trans, "code": "P",
                        "shares": 100.0 * (i + 1), "price": 10.0,
                        "value": 1000.0 * (i + 1),
                        "owner_cik": 1000 * (i + 1) + j,
                        "is_officer": j == 0, "is_director": j == 1,
                        "is_ten_pct": False,
                    })
                rows.append({
                    "filed": date, "trans_date": trans, "code": "S",
                    "shares": 50.0 * (i + 1), "price": 10.0,
                    "value": 500.0 * (i + 1), "owner_cik": 9000 + i,
                    "is_officer": False, "is_director": False,
                    "is_ten_pct": True,
                })
            insider[sym] = pd.DataFrame(rows).set_index("filed")
```

Insert that block immediately after the existing fundamentals block; the `sectors = ...` line stays as-is, and the final call becomes:

```python
    return assemble_panel(
        bars, options, fundamentals, factors,
        insider=insider,
        has_option_volume=with_options and with_option_volume,
        sectors=sectors,
    )
```

- [ ] **Step 6: Run the new tests, then the full suite**

Run: `uv run pytest tests/test_alphasearch_panel.py tests/test_alphasearch_segments.py -q` — expected: PASS.
Run: `uv run pytest -q` — expected: PASS (no signal reads `insider` yet, so nothing else changes).

- [ ] **Step 7: Lint + commit**

```bash
uv run ruff check src tests scripts
git add src/trading/alphasearch/panel.py src/trading/alphasearch/sweep.py \
        src/trading/alphasearch/segments.py tests/alphasearch_helpers.py \
        tests/test_alphasearch_panel.py tests/test_alphasearch_segments.py
git commit -m "Add PIT insider panel accessor + universe insider_dir wiring [AI]"
```

---

### Task 4: The three signals + `requires_insider` refusal + count updates

**Files:**
- Modify: `src/trading/alphasearch/spec.py` (`SignalSpec` ~line 38, `_register` ~line 52, new family section at end)
- Modify: `src/trading/alphasearch/sweep.py` (`_check_universe_supports` ~line 282)
- Modify: `src/trading/cli.py` (segment-safe hint ~line 926)
- Create: `tests/test_alphasearch_insider.py`
- Modify: `tests/test_alphasearch_spec.py` (~line 178), `tests/test_alphasearch_sweep.py`, `tests/test_alphasearch_cli.py` (~line 209), `tests/test_alphasearch_golden.py`

**Interfaces:**
- Produces: `SignalSpec.requires_insider: bool = False`; `_register(..., requires_insider: bool = False)`; registered signals `npr_90`, `cluster_buys_90`, `officer_buy_90` (the last with `requires_insider=True, requires_fundamentals=True`); the `_check_universe_supports` insider refusal naming `data/insider/equities`, `scripts/build_insider_store.py`, and the `--signals` workaround. Note on the `_register` assertion: `requires_option_volume` implies `requires_options` structurally (volume lives ON option cells); no analogous implication exists for `requires_insider` (the insider store is independent of every other store), so no new assert — `officer_buy_90`'s dual flags are pinned by the registry-completeness test instead.
- Consumes: `PanelView.insider_window` (Task 3), `_fundamental_field` (existing, spec.py ~line 396), `PanelView.bars` (existing).

- [ ] **Step 1: Write the failing signal unit tests**

Create `tests/test_alphasearch_insider.py`:

```python
"""Form 4 insider family: hand-computed fixtures pinning the FROZEN spec
section 3 table -- definitions, signs (+ all three), and the NaN conventions
(cluster's 0-vs-NaN never-covered distinction; officer's raw-price basis)."""

from __future__ import annotations

import math

import pandas as pd

from trading.alphasearch.panel import BAR_COLUMNS, PanelData
from trading.alphasearch.spec import SIGNALS

AS_OF = pd.Timestamp("2020-06-30", tz="UTC")


def _insider(rows: list[tuple[str, str, float, float, int, bool]]) -> pd.DataFrame:
    """(filed_iso, code, shares, price, owner_cik, is_officer) -> store frame."""
    frame = pd.DataFrame(
        rows,
        columns=["filed", "code", "shares", "price", "owner_cik", "is_officer"],
    )
    frame["filed"] = pd.to_datetime(frame["filed"]).dt.tz_localize("UTC")
    frame["trans_date"] = frame["filed"] - pd.Timedelta(2, unit="D")
    frame["value"] = frame["shares"] * frame["price"]
    frame["is_director"] = False
    frame["is_ten_pct"] = False
    return frame.set_index("filed").sort_index(kind="mergesort")


def _bars(close_raw: float = 50.0) -> pd.DataFrame:
    idx = pd.date_range("2020-06-01", periods=21, freq="B", tz="UTC")
    close = pd.Series(50.0, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": 1000.0, "div_cash": 0.0, "split_factor": 1.0,
         "close_raw": close_raw},
        index=idx,
    )


def _panel(insider: dict[str, pd.DataFrame], symbols: tuple[str, ...],
           *, with_fundamentals: bool = True, with_bars: bool = True) -> PanelData:
    filed = pd.DatetimeIndex([pd.Timestamp("2020-01-06", tz="UTC")])
    fundamentals = (
        {s: pd.DataFrame({"shares_outstanding": [1e6]}, index=filed) for s in symbols}
        if with_fundamentals else {}
    )
    bars = {s: _bars() for s in symbols} if with_bars else {}
    return PanelData(
        closes={s: b["close"] for s, b in bars.items()},
        fundamentals=fundamentals, insider=insider, bars=bars, symbols=symbols,
    )


def _score(name: str, panel: PanelData) -> pd.Series:
    return SIGNALS[name].fn(panel.view(AS_OF), AS_OF)


def test_npr_90_net_purchase_ratio_and_sign():
    insider = {
        # AAA: buys 100*10=1000, sells 50*10=500 -> (1000-500)/1500 = 1/3
        "AAA": _insider([("2020-06-15", "P", 100.0, 10.0, 1, True),
                         ("2020-06-16", "S", 50.0, 10.0, 2, False)]),
        # BBB: sells only -> -1 (net seller, ranked at the bottom: + sign)
        "BBB": _insider([("2020-06-15", "S", 50.0, 10.0, 3, False)]),
        # CCC: covered, but nothing filed in the trailing 90d -> NaN (no P/S
        # rows in window is the spec's npr NaN case, distinct from cluster's 0)
        "CCC": _insider([("2019-06-15", "P", 10.0, 10.0, 4, False)]),
    }
    scores = _score("npr_90", _panel(insider, ("AAA", "BBB", "CCC", "DDD")))
    assert math.isclose(scores["AAA"], 1.0 / 3.0, rel_tol=1e-12)
    assert scores["BBB"] == -1.0
    assert scores["AAA"] > scores["BBB"]          # net buying ranks higher
    assert math.isnan(scores["CCC"])
    assert math.isnan(scores["DDD"])              # never covered


def test_npr_90_all_nan_values_in_window_is_nan_not_zero():
    # A footnote-priced (value=NaN) window must not fabricate 0/0.
    frame = _insider([("2020-06-15", "P", 100.0, 10.0, 1, True)])
    frame["value"] = float("nan")
    scores = _score("npr_90", _panel({"AAA": frame}, ("AAA",)))
    assert math.isnan(scores["AAA"])


def test_cluster_buys_90_distinct_owners_and_the_zero_vs_nan_distinction():
    insider = {
        # Two P rows, SAME owner -> 1; the sale's owner never counts.
        "ONE": _insider([("2020-06-10", "P", 10.0, 5.0, 42, False),
                         ("2020-06-20", "P", 10.0, 5.0, 42, False),
                         ("2020-06-21", "S", 10.0, 5.0, 43, False)]),
        "TWO": _insider([("2020-06-10", "P", 10.0, 5.0, 1, False),
                         ("2020-06-20", "P", 10.0, 5.0, 2, False)]),
        # Covered (a sale in-window) but NO buys -> 0.0, a REAL value.
        "QUIET": _insider([("2020-06-10", "S", 10.0, 5.0, 9, False)]),
        # Covered only by an out-of-window old row -> still 0.0 (covered).
        "OLD": _insider([("2019-01-10", "P", 10.0, 5.0, 9, False)]),
    }
    scores = _score("cluster_buys_90", _panel(insider, ("ONE", "TWO", "QUIET", "OLD", "NEVER")))
    assert scores["ONE"] == 1.0
    assert scores["TWO"] == 2.0
    assert scores["TWO"] > scores["ONE"]          # more buyers = conviction
    assert scores["QUIET"] == 0.0                 # quiet, not missing
    assert scores["OLD"] == 0.0                   # never-covered != quiet: covered
    assert math.isnan(scores["NEVER"])            # NO row ever filed <= as_of


def test_officer_buy_90_raw_price_basis_and_nan_conventions():
    insider = {
        # Officer buys 100 sh @ 10 = 1000; a NON-officer buy of 2000 must not
        # count. shares_outstanding 1e6, close_raw 50 -> 1000 / 5e7 = 2e-5.
        "AAA": _insider([("2020-06-15", "P", 100.0, 10.0, 1, True),
                         ("2020-06-16", "P", 200.0, 10.0, 2, False)]),
        # Covered, no officer buying -> 0.0 (real quiet), not NaN.
        "BBB": _insider([("2020-06-16", "S", 10.0, 10.0, 3, False)]),
    }
    scores = _score("officer_buy_90", _panel(insider, ("AAA", "BBB", "NEVER")))
    assert math.isclose(scores["AAA"], 1000.0 / (1e6 * 50.0), rel_tol=1e-12)
    assert scores["BBB"] == 0.0
    assert math.isnan(scores["NEVER"])


def test_officer_buy_90_nan_without_shares_or_close_raw():
    insider = {"AAA": _insider([("2020-06-15", "P", 100.0, 10.0, 1, True)])}
    no_fund = _score("officer_buy_90", _panel(insider, ("AAA",), with_fundamentals=False))
    assert math.isnan(no_fund["AAA"])             # no shares_outstanding
    no_bars = _score("officer_buy_90", _panel(insider, ("AAA",), with_bars=False))
    assert math.isnan(no_bars["AAA"])             # no close_raw at as_of
    # A legacy narrow cache: bars exist but close_raw is NaN -> NaN, never a
    # fabricated adjusted-close basis (the div_yield lesson).
    panel = _panel(insider, ("AAA",))
    narrow = panel.bars["AAA"].copy()
    narrow["close_raw"] = float("nan")
    panel = PanelData(closes=panel.closes, fundamentals=panel.fundamentals,
                      insider=panel.insider, bars={"AAA": narrow}, symbols=("AAA",))
    assert math.isnan(_score("officer_buy_90", panel)["AAA"])
    assert list(narrow.columns) == BAR_COLUMNS


def test_insider_signals_all_nan_without_any_insider_data():
    panel = _panel({}, ("AAA", "BBB"))
    for name in ("npr_90", "cluster_buys_90", "officer_buy_90"):
        assert _score(name, panel).isna().all()
```

- [ ] **Step 2: Update the registry-count tests (they must fail first)**

In `tests/test_alphasearch_spec.py`, replace the `test_registry_is_complete_with_correct_requirements` body:

```python
def test_registry_is_complete_with_correct_requirements():
    assert len(SIGNALS) == 40  # 16 seeds + 21 Tier-1 (9+5+5+2) + 3 insider
    options_family = {"vrp", "hedge", "excite", "atm_iv", "smile", "atm_spread",
                      "cp_vol", "osv", "otm_put_iv", "iv_change", "dskew"}
    volume_family = {"cp_vol", "osv"}
    # officer_buy_90 needs shares_outstanding: it is IN the fundamentals
    # family (dual-flagged) as well as the insider family (spec section 3).
    fundamentals_family = {"gross_profitability", "earnings_yield",
                           "book_to_market", "asset_growth", "net_issuance",
                           "roa", "droa", "rev_growth", "officer_buy_90"}
    insider_family = {"npr_90", "cluster_buys_90", "officer_buy_90"}
    for name, spec in SIGNALS.items():
        assert spec.requires_options == (name in options_family)
        assert spec.requires_option_volume == (name in volume_family)
        assert spec.requires_fundamentals == (name in fundamentals_family)
        assert spec.requires_insider == (name in insider_family)
```

- [ ] **Step 3: Write the failing refusal tests**

Append to `tests/test_alphasearch_sweep.py` (next to `test_fundamentals_signal_without_store_refused`):

```python
def test_insider_signal_without_store_refused(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    panel = make_panel(with_insider=False)
    with pytest.raises(SweepError) as excinfo:
        run_sweep(
            _universe(tmp_path), journal, make_factors(), ts="t1",
            signals=_subset("npr_90"), window=WINDOW,
            panel_factory=lambda _u, _f: panel,
        )
    assert list(journal.events()) == []             # refused at assembly: no trials
    message = str(excinfo.value)
    assert "data/insider/equities" in message
    assert "scripts/build_insider_store.py" in message
    assert "--signals" in message
```

and (next to `test_deep_universe_refuses_fundamentals_signals_end_to_end`):

```python
def test_deep_universe_refuses_insider_signals_end_to_end(tmp_path):
    # insider_dir=None -> panel.insider == {} -> the requires_insider refusal.
    journal = trials_journal(tmp_path / "journal")
    uspec = _write_deep_universe(tmp_path)
    with pytest.raises(SweepError, match="requires insider"):
        run_sweep({uspec.name: uspec}, journal, make_factors(), ts="t1",
                  signals=_subset("cluster_buys_90"), window=WINDOW)
    assert list(journal.events()) == []
```

In `tests/test_alphasearch_cli.py::test_sweep_segments_refusal_prints_signal_family_hint`, add after the `assert "ind_mom" not in err` line:

```python
    # requires_insider signals are excluded from the segment-safe list for
    # the same reason as fundamentals: their store may not be synced, and a
    # hint containing them would hand the operator a still-failing command.
    assert "npr_90" not in err
    assert "cluster_buys_90" not in err
```

- [ ] **Step 4: Update the golden test (also fails first)**

In `tests/test_alphasearch_golden.py`:

`_write_universe` — after the fundamentals-store block, add:

```python
    insider_dir = tmp_path / "insider"
    insider_dir.mkdir()
    for sym in panel.symbols:
        panel.insider[sym].to_parquet(insider_dir / f"{sym}.parquet")
    return UniverseSpec("largecap", cache, samples, tmp_path / "fundamentals",
                        insider_dir=insider_dir)
```

(replacing the old `return UniverseSpec(...)` line.)

In `test_golden_sweep_end_to_end`:
- `assert n_trials == len(SIGNALS) == 40` (was 37)
- after the `errored` assertions add:

```python
    # The narrow golden bar cache has no close_raw -> officer_buy_90 is an
    # honest all-NaN error trial (like div_yield); the other two insider
    # signals run end-to-end on the real-files store.
    assert "officer_buy_90" in errored
    for name in ("npr_90", "cluster_buys_90"):
        row = next(r for r in rows if r.signal == name)
        assert row.error is None
        assert row.alpha_t is not None
        assert row.n_names_median == 16.0
```

- `assert len(list(journal.events())) == 80` (was 74; 2 runs × 40)
- `assert len(discovery_trials(journal)) == 40` (was 37)

- [ ] **Step 5: Run to verify the new tests fail**

Run: `uv run pytest tests/test_alphasearch_insider.py tests/test_alphasearch_spec.py tests/test_alphasearch_sweep.py tests/test_alphasearch_cli.py tests/test_alphasearch_golden.py -q`
Expected: FAIL — `KeyError: 'npr_90'`, count assertion 37 != 40, `TypeError` on `requires_insider`, etc.

- [ ] **Step 6: Implement — `spec.py` flag + registrations**

`SignalSpec` gains (after `requires_option_volume`):

```python
    # Form 4 insider store (data/insider/equities); signals reading it are
    # refused at sweep assembly on store-less universes (insider spec
    # section 3), mirroring requires_fundamentals.
    requires_insider: bool = False
```

`_register` gains the parameter and threads it (the existing assert is untouched — `requires_option_volume` implies `requires_options` because volume lives ON option cells; no analogous implication exists for the independent insider store):

```python
def _register(
    name: str,
    fn: SignalFn,
    *,
    requires_options: bool = False,
    requires_fundamentals: bool = False,
    requires_option_volume: bool = False,
    requires_insider: bool = False,
) -> None:
    assert not requires_option_volume or requires_options, (
        f"{name}: requires_option_volume=True must also set requires_options=True"
    )
    SIGNALS[name] = SignalSpec(
        name,
        fn,
        requires_options=requires_options,
        requires_fundamentals=requires_fundamentals,
        requires_option_volume=requires_option_volume,
        requires_insider=requires_insider,
    )
```

Append the family at the end of `spec.py` (after the industry-relative section), and update the module docstring's registry inventory line to mention the 3 insider signals:

```python
# --------------------------------------------------------------------------- #
# Insider (Form 4) family (requires_insider; spec 2026-07-09-insider-pipeline
# section 3, FROZEN). All three read the trailing-90-FILED-day window via
# PanelView.insider_window: None = never covered at as_of -> NaN; an empty
# frame = covered-but-quiet, a REAL observation. Scoring keys the FILED date
# ONLY -- TRANS_DATE precedes filing and keying on it is look-ahead.
# --------------------------------------------------------------------------- #
def _npr_90(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    """(sum buy value - sum sell value) / (sum buy value + sum sell value)
    over the trailing 90 filed days. NaN when the window holds no P/S value
    (denominator <= 0): footnote-priced NaN values are skipped by sum(), so
    an all-NaN window is honestly missing, never 0/0."""
    scores: dict[str, float] = {}
    for symbol in view.symbols:
        window = view.insider_window(symbol)
        score = math.nan
        if window is not None and len(window):
            buys = float(window.loc[window["code"] == "P", "value"].sum())
            sells = float(window.loc[window["code"] == "S", "value"].sum())
            total = buys + sells
            if total > 0:
                score = (buys - sells) / total
        scores[symbol] = score
    return pd.Series(scores, dtype="float64")


def _cluster_buys_90(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    """Count of DISTINCT owner ciks with >= 1 open-market purchase in the
    window. 0 is a REAL value (covered names with no buys); NaN only when
    the symbol has no insider row EVER filed <= as_of (never-covered !=
    quiet -- the spec section 3 distinction, encoded by insider_window's
    None-vs-empty contract)."""
    scores: dict[str, float] = {}
    for symbol in view.symbols:
        window = view.insider_window(symbol)
        if window is None:
            scores[symbol] = math.nan
        else:
            scores[symbol] = float(
                window.loc[window["code"] == "P", "owner_cik"].nunique()
            )
    return pd.Series(scores, dtype="float64")


def _officer_buy_90(view: PanelView, as_of: pd.Timestamp) -> pd.Series:
    """Officer open-market purchase value over the window, scaled by market
    cap on the RAW price basis: shares_outstanding (latest visible filing) x
    close_raw at as_of. Raw basis is the div_yield lesson -- the adjusted
    close bakes in FUTURE corporate actions, and Form 4 dollar values are
    raw dollars, so the denominator must be too. NaN when never covered or
    when shares_outstanding / close_raw are unavailable; 0.0 when covered
    with no officer buying (a real quiet)."""
    scores: dict[str, float] = {}
    for symbol in view.symbols:
        window = view.insider_window(symbol)
        score = math.nan
        if window is not None:
            shares = _fundamental_field(view, symbol, "shares_outstanding")
            bars = view.bars(symbol)
            close_raw = float(bars["close_raw"].iloc[-1]) if len(bars) else math.nan
            if (not math.isnan(shares) and shares > 0
                    and not math.isnan(close_raw) and close_raw > 0):
                officer = window[(window["code"] == "P") & window["is_officer"]]
                score = float(officer["value"].sum()) / (shares * close_raw)
        scores[symbol] = score
    return pd.Series(scores, dtype="float64")


# Net purchase ratio (Lakonishok-Lee): insider net buying predicts returns.
_register("npr_90", _npr_90, requires_insider=True)
# Cluster buying: several DISTINCT insiders buying together is conviction
# (the strongest Form 4 configuration in the literature).
_register("cluster_buys_90", _cluster_buys_90, requires_insider=True)
# Officer purchases carry the most information per dollar; market-cap scale
# needs fundamentals shares_outstanding -> BOTH flags (spec section 3).
_register("officer_buy_90", _officer_buy_90,
          requires_insider=True, requires_fundamentals=True)
```

- [ ] **Step 7: Implement — `sweep.py` refusal + `cli.py` hint**

`_check_universe_supports`, after the fundamentals block:

```python
    if spec.requires_insider and not panel.insider:
        raise SweepError(
            f"signal {spec.name!r} requires insider transactions; universe "
            f"{universe!r} has none. Expected store: data/insider/equities "
            "(UniverseSpec.insider_dir points at it). Populate it with "
            "`scripts/build_insider_store.py`; or work around it by passing "
            "--signals with a non-insider signal subset"
        )
```

`cli.py` (~line 926): add the flag to the filter and name the store in the text:

```python
                segment_safe_signals = ",".join(
                    name
                    for name, s in SIGNALS.items()
                    if not s.requires_options
                    and not s.requires_fundamentals
                    and not s.requires_insider
                    and name != "ind_mom"
                )
                print(
                    "hint: deep-pool segments carry no options data (and "
                    "fundamentals/insider only once data/fundamentals/equities "
                    "and data/insider/equities are synced) — pair --segments "
                    f"with --signals {segment_safe_signals} "
                    "(segment-safe signals), or target options segments individually "
                    "via --universe opt-largecap:<segment>",
                    file=sys.stderr,
                )
```

(Keep the existing `ind_mom` rationale comment above the filter unchanged.)

- [ ] **Step 8: Run the tests to verify they pass, then the full suite**

Run: `uv run pytest tests/test_alphasearch_insider.py tests/test_alphasearch_spec.py tests/test_alphasearch_sweep.py tests/test_alphasearch_cli.py tests/test_alphasearch_golden.py -q` — expected: PASS.
Run: `uv run pytest -q` — expected: PASS. If `tests/test_alphasearch_lookahead.py::test_every_signal_scores_real_values_at_the_last_pre_cutoff_date` fails on an insider signal, the Task 3 fixture recipe was not followed (every symbol must have month-first filings and the fundamentals/bars the officer signal needs) — fix the fixture, not the test.

- [ ] **Step 9: Lint + commit**

```bash
uv run ruff check src tests scripts
git add src/trading/alphasearch/spec.py src/trading/alphasearch/sweep.py \
        src/trading/cli.py tests/test_alphasearch_insider.py \
        tests/test_alphasearch_spec.py tests/test_alphasearch_sweep.py \
        tests/test_alphasearch_cli.py tests/test_alphasearch_golden.py
git commit -m "Register frozen Form 4 insider family behind requires_insider [AI]"
```

---

### Task 5: Look-ahead coverage — perturb post-T FILED insider rows

**Files:**
- Modify: `tests/test_alphasearch_lookahead.py`

**Interfaces:**
- Consumes: `assemble_panel(..., insider=...)` and the `make_panel` insider fixture (Task 3 — its `trans_date = filed − 20d` produces rows whose trans_date precedes the cutoff while their FILED date follows it: exactly the rows a trans_date-keyed signal would leak), plus the three registered signals (Task 4; the registry-iterating test auto-covers them once the stores are perturbed).

- [ ] **Step 1: Extend the perturbation + guards (failing first is impossible here — this test's failure mode is silence; make the perturbation REAL and prove the fixture is non-vacuous)**

In `tests/test_alphasearch_lookahead.py`:

`_perturb_after` — add before the `factors` block:

```python
    insider: dict[str, pd.DataFrame] = {}
    for sym, frame in panel.insider.items():
        f = frame.copy()
        late = f.index > cutoff
        # Corrupt post-T-FILED rows' values AND ownership...
        for col in ("shares", "price", "value"):
            f.loc[late, col] = f.loc[late, col] * 7.0 + 3.0
        f.loc[late, "owner_cik"] = 424242      # collapses any distinct-owner count
        f.loc[late, "is_officer"] = True
        # ...and shift their FILED dates themselves (the PIT key): several of
        # these rows have trans_date <= cutoff (the make_panel fixture files
        # 20 days after the transaction), so a signal keyed on TRANS_DATE --
        # the classic Form 4 look-ahead -- would see them move and change its
        # pre-cutoff scores.
        f.index = f.index.where(f.index <= cutoff, f.index + pd.Timedelta(30, unit="D"))
        insider[sym] = f
    return assemble_panel(
        bars, options, fundamentals, factors,
        insider=insider,
        has_option_volume=panel.has_option_volume, sectors=panel.sectors,
    )
```

`test_fixture_actually_has_data_after_the_cutoff` — add:

```python
    assert any((f.index > CUTOFF).any() for f in panel.insider.values())
    # The trans_date-keying trap must exist: rows FILED after the cutoff whose
    # TRANSACTION predates it (Form 4's 2-day+ filing lag).
    assert any(
        ((f.index > CUTOFF) & (f["trans_date"] <= CUTOFF)).any()
        for f in panel.insider.values()
    )
```

Add a dedicated boundary test at the end of the file:

```python
def test_insider_rows_filed_after_as_of_are_invisible_even_when_transacted_before():
    # A row transacted BEFORE as_of but FILED after it does not exist yet:
    # only the filing makes it public (spec: scoring keys FILED only).
    as_of = pd.Timestamp("2020-03-02", tz="UTC")
    frame = pd.DataFrame(
        {"trans_date": [pd.Timestamp("2020-02-28", tz="UTC")], "code": ["P"],
         "shares": [100.0], "price": [10.0], "value": [1000.0],
         "owner_cik": [1], "is_officer": [True], "is_director": [False],
         "is_ten_pct": [False]},
        index=pd.DatetimeIndex([pd.Timestamp("2020-03-04", tz="UTC")], name="filed"),
    )
    panel = PanelData(closes={}, insider={"AAA": frame}, symbols=("AAA",))
    assert panel.view(as_of).insider_window("AAA") is None   # not even "covered"
    after = panel.view(pd.Timestamp("2020-03-04", tz="UTC")).insider_window("AAA")
    assert after is not None and len(after) == 1             # visible once FILED
```

- [ ] **Step 2: Run the lookahead suite**

Run: `uv run pytest tests/test_alphasearch_lookahead.py -q`
Expected: PASS — `test_no_registered_signal_can_see_past_as_of` now exercises all 40 signals against insider-perturbed reassembled panels. To prove the harness bites, temporarily change `insider_window` to filter on `trans_date` instead of the index (`visible = frame[frame["trans_date"] <= self.as_of]`), rerun, and confirm the perturbation test FAILS for the insider signals; revert.

- [ ] **Step 3: Full suite + lint + commit**

```bash
uv run pytest -q
uv run ruff check src tests scripts
git add tests/test_alphasearch_lookahead.py
git commit -m "Extend no-look-ahead perturbation to insider FILED dates [AI]"
```

---

### Task 6: Docs — glossary + experiments registration note

**Files:**
- Modify: `docs/glossary.md` (new section between "The anomaly zoo (Tier-1 signal batch)" and "## The robustness battery (Piece 3)")
- Modify: `docs/experiments.md` (registration paragraph at the end of §11, before "## Known caveats affecting these numbers")

**Interfaces:** none (prose only; folded into this task because Task 4's registrations are what the note pre-registers — commit them adjacent in history).

- [ ] **Step 1: Add the glossary section**

Insert into `docs/glossary.md` before `## The robustness battery (Piece 3)`:

```markdown
## Insider transactions (the Form 4 family)

**Form 4** — the SEC filing corporate insiders (officers, directors, >10%
owners) must submit within 2 business days of trading their own company's
stock. Our source is the SEC DERA "Insider Transactions Data Sets" quarterly
bulk files: official, free, as-reported forever (never restated), covering
delisted names — the cleanest PIT anomaly source we identified. The store
keeps only open-market purchases (`P`) and sales (`S`); awards, exercises,
gifts and plan transactions are excluded. 10b5-1 (pre-scheduled) trades are
NOT excluded — the flag is unreliable before 2023 (documented limitation).
PIT discipline: every signal keys the FILED date, never the transaction date
(which precedes filing and would be look-ahead).

**Net purchase ratio (NPR, `npr_90`)** — (buy dollars − sell dollars) /
(buy dollars + sell dollars) over the trailing 90 filed days
(Lakonishok-Lee). +1 = insiders only bought, −1 = only sold. Sales carry
less information than purchases (diversification, taxes, option vesting);
purchases are the deliberate act.

**Cluster buying (`cluster_buys_90`)** — the count of DISTINCT insiders with
at least one open-market purchase in the window. Several insiders buying
independently is far stronger evidence than one large buy. 0 is a real value
(a covered, quiet name); NaN is reserved for names with no Form 4 history at
all as of the date (never-covered ≠ quiet).

**Officer purchases (`officer_buy_90`)** — officer open-market purchase
dollars scaled by market cap on the RAW price basis (shares outstanding ×
raw unadjusted close — the div_yield lesson: adjusted closes bake in future
splits, and Form 4 dollars are raw dollars). Officers have the best
information per dollar traded. Requires both the insider store and
fundamentals (`shares_outstanding`), so it registers with both refusal
flags.
```

- [ ] **Step 2: Add the experiments.md registration note**

Append at the end of §11 (immediately before `## Known caveats affecting these numbers`):

```markdown
**Form 4 insider family registered 2026-07-09, pre-sweep.** 3 purchase-side
signals (`npr_90`, `cluster_buys_90`, `officer_buy_90`) are frozen in
`docs/superpowers/specs/2026-07-09-insider-pipeline-design.md` §3 —
definitions, signs (+ all three), and NaN conventions (cluster's
0-vs-never-covered distinction; officer's raw-price basis and dual
`requires_insider`+`requires_fundamentals` flags) pre-registered BEFORE any
trial. The data store is SEC DERA insider-transaction quarterly ZIPs 2018q3+
(open-market P/S only, FILED-date PIT keying — the transaction date precedes
filing and is never scored), built by `scripts/build_insider_store.py` into
`data/insider/equities/` and mapped through the committed cik_map intervals
(unmapped CIKs counted, never guessed). **Nothing sweeps under this
registration**: the pre-registered discovery sweep belongs to the combined
options-v2 + insider batch spec (one read after both data sets land), which
will disclose the full trial count. New glossary section "Insider
transactions (the Form 4 family)".
```

- [ ] **Step 3: Full suite + lint + commit**

```bash
uv run pytest -q
uv run ruff check src tests scripts
git add docs/glossary.md docs/experiments.md
git commit -m "Document Form 4 insider family (glossary + pre-sweep registration note) [AI]"
```

---

## Post-merge ops (orchestrator run-book — NOT a plan task)

The store build is data acquisition and runs after merge, on the machine that owns `data/`:

```bash
cd /Users/travis/Source/personal/trading
uv run python scripts/build_insider_store.py 2>&1 | tee /tmp/claude-insider-build.log
```

Expected: ~31-32 quarterly ZIPs (2018q3 → the last complete quarter; the newest may 404-skip with a WARNING — publication lag, not a gap), a few MB each (~100-300MB total in `data/edgar-insider-raw/`, cached for reruns), a few minutes download (SEC-throttled) plus 1-2 minutes parsing. Expected report shape:

```
parsed 2018q3: ~40-80k P/S rows (~5-15k P / ~35-70k S), <1% skipped
...
rows parsed: ~1.5-3M total; small skipped count
mapped to >=1 symbol: ... rows -> ~900-1100 symbols; unmapped: most rows (non-members) + the known ~2% cik_map residual
window-membership coverage: expect ~90-98% members with >= 1 insider row
rows per FILED year: 2018 partial; 2019-2025 full years of similar magnitude; a low year = a HOLE, investigate
COVERAGE GAPS: (must be empty; exit 1 otherwise)
```

Then the spec §4 spot-check (2-3 filings against EDGAR's website): pick 2-3 store rows with large `value` from well-known symbols, e.g.

```bash
uv run python - <<'EOF'
import pandas as pd
f = pd.read_parquet("data/insider/equities/JPM.parquet")
print(f[f["code"] == "P"].nlargest(3, "value").to_string())
EOF
```

and verify each row's filed date / shares / price against the rendered Form 4 in the company's EDGAR filing index (https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=4). A famous check: Jamie Dimon's ~500,000-share JPM purchase filed 2016 is pre-window; use any large 2019+ CEO buy. Mismatches → STOP, report.

Rebuild rule: the script refuses a non-empty `data/insider/equities/` — delete its contents (including `.source`) first; cached ZIPs in `data/edgar-insider-raw/` make reruns cheap.

**Orchestrator should also know:** (1) DERA schema drift is the top risk — Task 1's discovery step is a hard STOP gate; if the 2024q1 headers don't match the pinned names, the plan pauses for consultation. (2) The registry grows 37→40; any concurrently-executing plan that pins the registry count (options-v2 batch) must land its own count update relative to whichever merges second. (3) Form 4/A amendments can duplicate transactions (own accessions; the frozen spec has no form filter) — documented limitation, surfaces as slight over-counting, acceptable for the batch sweep. (4) Nothing sweeps under this plan; `journal/alphasearch-trials.jsonl` must show ZERO new trials from this work.
