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
