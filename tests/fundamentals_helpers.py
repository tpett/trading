"""Builders for synthetic SEC Financial Statement Data Set fixtures.

write_quarter_zip fabricates a minimal quarterly ZIP (sub.txt + num.txt with
only the columns the parser reads via usecols); fact/filing_facts build
already-normalized facts rows for metrics-level tests that skip the ZIP layer.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from trading.fundamentals.edgar import FACT_COLUMNS

SUB_HEADER = "adsh\tcik\tname\tform\tperiod\tfy\tfp\tfiled"
NUM_HEADER = "adsh\ttag\tversion\tddate\tqtrs\tuom\tsegments\tcoreg\tvalue"


def sub_line(adsh: str, cik: int, form: str, period: str, fy: str, fp: str, filed: str) -> str:
    return f"{adsh}\t{cik}\tTEST CO\t{form}\t{period}\t{fy}\t{fp}\t{filed}"


def num_line(
    adsh: str,
    tag: str,
    ddate: str,
    qtrs: int,
    value: float,
    uom: str = "USD",
    segments: str = "",
    coreg: str = "",
) -> str:
    return f"{adsh}\t{tag}\tus-gaap/2023\t{ddate}\t{qtrs}\t{uom}\t{segments}\t{coreg}\t{value}"


def write_quarter_zip(path: Path, sub_lines: list[str], num_lines: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("sub.txt", "\n".join([SUB_HEADER, *sub_lines]) + "\n")
        zf.writestr("num.txt", "\n".join([NUM_HEADER, *num_lines]) + "\n")
    return path


def fact(
    cik: int,
    adsh: str,
    form: str,
    fy: str,
    fp: str,
    period: str,
    filed: str,
    concept: str,
    tag: str,
    qtrs: int,
    value: float,
) -> dict:
    return {
        "cik": cik,
        "adsh": adsh,
        "form": form,
        "fy": fy,
        "fp": fp,
        "period": pd.Timestamp(period),
        "filed": pd.Timestamp(filed),
        "concept": concept,
        "tag": tag,
        "qtrs": qtrs,
        "value": value,
    }


def filing_facts(
    cik: int,
    adsh: str,
    form: str,
    fy: str,
    fp: str,
    period: str,
    filed: str,
    revenue: float | None = None,
    cogs: float | None = None,
    assets: float | None = None,
    net_income: float | None = None,
    equity: float | None = None,
    shares: float | None = None,
) -> list[dict]:
    """One filing's normalized facts: flows (revenue/cogs/net income) at the
    form's duration (10-K = 4 quarters, 10-Q = 1), instants (assets/equity/
    shares) at qtrs=0."""
    qtrs = 4 if form == "10-K" else 1
    rows = []
    if revenue is not None:
        rows.append(
            fact(cik, adsh, form, fy, fp, period, filed, "revenue", "Revenues", qtrs, revenue)
        )
    if cogs is not None:
        rows.append(
            fact(cik, adsh, form, fy, fp, period, filed, "cogs", "CostOfRevenue", qtrs, cogs)
        )
    if assets is not None:
        rows.append(fact(cik, adsh, form, fy, fp, period, filed, "assets", "Assets", 0, assets))
    if net_income is not None:
        rows.append(
            fact(
                cik,
                adsh,
                form,
                fy,
                fp,
                period,
                filed,
                "net_income",
                "NetIncomeLoss",
                qtrs,
                net_income,
            )
        )
    if equity is not None:
        rows.append(
            fact(cik, adsh, form, fy, fp, period, filed, "equity", "StockholdersEquity", 0, equity)
        )
    if shares is not None:
        rows.append(
            fact(
                cik,
                adsh,
                form,
                fy,
                fp,
                period,
                filed,
                "shares",
                "EntityCommonStockSharesOutstanding",
                0,
                shares,
            )
        )
    return rows


def facts_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FACT_COLUMNS)
