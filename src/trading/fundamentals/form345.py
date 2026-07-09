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
