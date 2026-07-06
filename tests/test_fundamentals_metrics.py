import math

import pandas as pd
import pytest

from fundamentals_helpers import facts_frame, filing_facts
from trading.fundamentals.metrics import SERIES_COLUMNS, compute_pit_series, empty_series

CIK = 100


def _year_of_filings() -> list[dict]:
    """Q1-Q3 2023 10-Qs, the FY2023 10-K, then Q1 2024 — a full TTM ramp."""
    rows = []
    rows += filing_facts(
        CIK,
        "a-01",
        "10-Q",
        "2023",
        "Q1",
        "2023-03-31",
        "2023-05-10",
        revenue=10.0,
        cogs=4.0,
        assets=90.0,
        net_income=3.0,
        equity=50.0,
        shares=100.0,
    )
    rows += filing_facts(
        CIK,
        "a-02",
        "10-Q",
        "2023",
        "Q2",
        "2023-06-30",
        "2023-08-09",
        revenue=12.0,
        cogs=5.0,
        assets=95.0,
        net_income=4.0,
        equity=52.0,
        shares=100.0,
    )
    rows += filing_facts(
        CIK,
        "a-03",
        "10-Q",
        "2023",
        "Q3",
        "2023-09-30",
        "2023-11-08",
        revenue=11.0,
        cogs=5.0,
        assets=98.0,
        net_income=3.0,
        equity=54.0,
        shares=100.0,
    )
    rows += filing_facts(
        CIK,
        "a-04",
        "10-K",
        "2023",
        "FY",
        "2023-12-31",
        "2024-02-20",
        revenue=48.0,
        cogs=20.0,
        assets=100.0,
        net_income=14.0,
        equity=60.0,
        shares=101.0,
    )
    rows += filing_facts(
        CIK,
        "a-05",
        "10-Q",
        "2024",
        "Q1",
        "2024-03-31",
        "2024-05-09",
        revenue=14.0,
        cogs=6.0,
        assets=110.0,
        net_income=5.0,
        equity=62.0,
        shares=102.0,
    )
    return rows


def test_ttm_incomplete_is_nan_then_completes_at_the_10k():
    series = compute_pit_series(facts_frame(_year_of_filings()))[CIK]
    assert list(series.columns) == SERIES_COLUMNS
    assert series.index.tz is not None and series.index.name == "filed"
    # First three filings: fewer than 4 known quarters -> NaN TTMs, row still
    # present; the instant primitives are already real.
    for filed in ("2023-05-10", "2023-08-09", "2023-11-08"):
        row = series.loc[pd.Timestamp(filed, tz="UTC")]
        assert math.isnan(row["gross_profitability"])
        assert math.isnan(row["ttm_net_income"])
        assert row["shares_outstanding"] == 100.0
    # 10-K: Q4 derived = FY - (Q1+Q2+Q3) -> rev 48-33=15, cogs 20-14=6,
    # ni 14-10=4; TTM = the FY totals.
    at_10k = series.loc[pd.Timestamp("2024-02-20", tz="UTC")]
    assert at_10k["revenue_ttm"] == 48.0
    assert at_10k["cogs_ttm"] == 20.0
    assert at_10k["gross_profitability"] == pytest.approx((48.0 - 20.0) / 100.0)
    assert at_10k["ttm_net_income"] == 14.0
    assert at_10k["book_equity"] == 60.0
    assert at_10k["shares_outstanding"] == 101.0


def test_ttm_rolls_forward_at_the_next_quarter():
    series = compute_pit_series(facts_frame(_year_of_filings()))[CIK]
    # Q2'23 + Q3'23 + derived Q4'23 + Q1'24: rev 12+11+15+14, cogs 5+5+6+6,
    # ni 4+3+4+5; instants come from the latest filing.
    row = series.loc[pd.Timestamp("2024-05-09", tz="UTC")]
    assert row["revenue_ttm"] == pytest.approx(52.0)
    assert row["cogs_ttm"] == pytest.approx(22.0)
    assert row["gross_profitability"] == pytest.approx(30.0 / 110.0)
    assert row["ttm_net_income"] == pytest.approx(16.0)
    assert row["book_equity"] == 62.0
    assert row["shares_outstanding"] == 102.0


def test_net_income_ttm_computes_independently_of_missing_cogs():
    # Financials shape: NetIncomeLoss present, no COGS concept anywhere ->
    # gross profitability NaN forever, but the value primitives are real.
    rows = []
    rows += filing_facts(
        CIK,
        "e-01",
        "10-Q",
        "2023",
        "Q1",
        "2023-03-31",
        "2023-05-10",
        revenue=10.0,
        assets=90.0,
        net_income=3.0,
        equity=50.0,
        shares=100.0,
    )
    rows += filing_facts(
        CIK,
        "e-02",
        "10-Q",
        "2023",
        "Q2",
        "2023-06-30",
        "2023-08-09",
        revenue=12.0,
        assets=95.0,
        net_income=4.0,
        equity=52.0,
        shares=100.0,
    )
    rows += filing_facts(
        CIK,
        "e-03",
        "10-Q",
        "2023",
        "Q3",
        "2023-09-30",
        "2023-11-08",
        revenue=11.0,
        assets=98.0,
        net_income=3.0,
        equity=54.0,
        shares=100.0,
    )
    rows += filing_facts(
        CIK,
        "e-04",
        "10-K",
        "2023",
        "FY",
        "2023-12-31",
        "2024-02-20",
        revenue=48.0,
        assets=100.0,
        net_income=14.0,
        equity=60.0,
        shares=101.0,
    )
    series = compute_pit_series(facts_frame(rows))[CIK]
    at_10k = series.loc[pd.Timestamp("2024-02-20", tz="UTC")]
    assert math.isnan(at_10k["gross_profitability"])  # no COGS -> quality NaN
    assert at_10k["ttm_net_income"] == 14.0  # value primitives still real
    assert at_10k["book_equity"] == 60.0
    assert at_10k["shares_outstanding"] == 101.0
    assert at_10k["net_income_tag"] == "NetIncomeLoss"
    assert at_10k["cogs_tag"] == ""


def test_restatement_never_rewrites_history():
    # Same (cik, fy, fp) filed twice: the ORIGINAL (earliest-filed) wins; the
    # later re-filing is discarded entirely -- no row at its filed date, and
    # every later TTM uses the original value.
    rows = _year_of_filings()
    rows += filing_facts(
        CIK,
        "a-99",
        "10-Q",
        "2023",
        "Q1",
        "2023-03-31",
        "2023-09-01",
        revenue=999.0,
        cogs=999.0,
        assets=999.0,
    )
    series = compute_pit_series(facts_frame(rows))[CIK]
    assert pd.Timestamp("2023-09-01", tz="UTC") not in series.index
    assert series.loc[pd.Timestamp("2024-02-20", tz="UTC"), "revenue_ttm"] == 48.0
    assert "a-99" not in set(series["adsh"])


def test_earliest_accession_breaks_a_same_day_tie():
    rows = filing_facts(
        CIK,
        "b-02",
        "10-Q",
        "2023",
        "Q1",
        "2023-03-31",
        "2023-05-10",
        revenue=999.0,
        cogs=1.0,
        assets=50.0,
    )
    rows += filing_facts(
        CIK,
        "b-01",
        "10-Q",
        "2023",
        "Q1",
        "2023-03-31",
        "2023-05-10",
        revenue=10.0,
        cogs=4.0,
        assets=90.0,
    )
    series = compute_pit_series(facts_frame(rows))[CIK]
    assert list(series["adsh"]) == ["b-01"]


def test_10k_without_all_three_prior_quarters_gives_nan_q4():
    rows = []
    rows += filing_facts(
        CIK,
        "c-01",
        "10-Q",
        "2023",
        "Q1",
        "2023-03-31",
        "2023-05-10",
        revenue=10.0,
        cogs=4.0,
        assets=90.0,
    )
    # Q2 missing entirely.
    rows += filing_facts(
        CIK,
        "c-03",
        "10-Q",
        "2023",
        "Q3",
        "2023-09-30",
        "2023-11-08",
        revenue=11.0,
        cogs=5.0,
        assets=98.0,
    )
    rows += filing_facts(
        CIK,
        "c-04",
        "10-K",
        "2023",
        "FY",
        "2023-12-31",
        "2024-02-20",
        revenue=48.0,
        cogs=20.0,
        assets=100.0,
    )
    series = compute_pit_series(facts_frame(rows))[CIK]
    assert math.isnan(series.loc[pd.Timestamp("2024-02-20", tz="UTC"), "gross_profitability"])


def test_ragged_quarter_window_is_nan():
    # Four known quarters but with a year gap inside: span > 330 days -> NaN.
    rows = []
    rows += filing_facts(
        CIK,
        "d-01",
        "10-Q",
        "2022",
        "Q3",
        "2022-09-30",
        "2022-11-08",
        revenue=10.0,
        cogs=4.0,
        assets=90.0,
    )
    rows += filing_facts(
        CIK,
        "d-02",
        "10-Q",
        "2023",
        "Q1",
        "2023-03-31",
        "2023-05-10",
        revenue=10.0,
        cogs=4.0,
        assets=90.0,
    )
    rows += filing_facts(
        CIK,
        "d-03",
        "10-Q",
        "2023",
        "Q2",
        "2023-06-30",
        "2023-08-09",
        revenue=12.0,
        cogs=5.0,
        assets=95.0,
    )
    rows += filing_facts(
        CIK,
        "d-04",
        "10-Q",
        "2023",
        "Q3",
        "2023-09-30",
        "2023-11-08",
        revenue=11.0,
        cogs=5.0,
        assets=98.0,
    )
    series = compute_pit_series(facts_frame(rows))[CIK]
    assert math.isnan(series.loc[pd.Timestamp("2023-11-08", tz="UTC"), "gross_profitability"])


def test_missing_cogs_gives_nan_metric_with_provenance():
    rows = _year_of_filings()
    facts = facts_frame(rows)
    facts = facts[~((facts["concept"] == "cogs") & (facts["adsh"] == "a-04"))]
    series = compute_pit_series(facts)[CIK]
    at_10k = series.loc[pd.Timestamp("2024-02-20", tz="UTC")]
    assert math.isnan(at_10k["gross_profitability"])
    assert at_10k["cogs_tag"] == ""
    assert at_10k["adsh"] == "a-04"
    assert at_10k["form"] == "10-K"
    assert at_10k["period"] == "2023-12-31"


def test_empty_facts_and_empty_series_shapes():
    assert compute_pit_series(facts_frame([])) == {}
    frame = empty_series()
    assert list(frame.columns) == SERIES_COLUMNS
    assert frame.index.tz is not None
