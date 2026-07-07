from fundamentals_helpers import num_line, sub_line, write_quarter_zip
from trading.fundamentals.edgar import FACT_COLUMNS, empty_facts, load_quarter_facts

# One 10-Q (cik 100, Q1 2023) + one 10-K (cik 200, FY 2022).
SUBS = [
    sub_line("0001-23-000001", 100, "10-Q", "20230331", "2023", "Q1", "20230510"),
    sub_line("0002-23-000001", 200, "10-K", "20221231", "2022", "FY", "20230225"),
    sub_line("0003-23-000001", 300, "10-K/A", "20221231", "2022", "FY", "20230301"),
    sub_line("0004-23-000001", 400, "8-K", "20230331", "2023", "Q1", "20230410"),
]
NUMS = [
    # cik 100 10-Q: both revenue tags present -> priority tag must win.
    num_line(
        "0001-23-000001",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "20230331",
        1,
        100.0,
    ),
    num_line("0001-23-000001", "Revenues", "20230331", 1, 999.0),
    # segment breakout + co-registrant + non-USD + comparative period: all excluded.
    num_line("0001-23-000001", "Revenues", "20230331", 1, 555.0, segments="Region=US;"),
    num_line("0001-23-000001", "Revenues", "20230331", 1, 556.0, coreg="SubCo"),
    num_line("0001-23-000001", "Revenues", "20230331", 1, 557.0, uom="EUR"),
    num_line("0001-23-000001", "Revenues", "20220331", 1, 558.0),  # ddate != period
    num_line("0001-23-000001", "CostOfGoodsAndServicesSold", "20230331", 1, 40.0),
    num_line("0001-23-000001", "Assets", "20230331", 0, 1000.0),
    # 10-Q revenue at annual duration must NOT be picked for a 10-Q.
    num_line("0001-23-000001", "Revenues", "20230331", 4, 400.0),
    # Value primitives: net income (flow), equity (instant; primary tag absent
    # here so the fallback must win), shares (dei cover-page instant dated
    # AFTER the period end -- month-end rounded -- beating the balance-sheet
    # fallback; a second, earlier dei row and a stale pre-period row lose).
    num_line("0001-23-000001", "NetIncomeLoss", "20230331", 1, 25.0),
    num_line(
        "0001-23-000001",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "20230331",
        0,
        500.0,
    ),
    num_line(
        "0001-23-000001", "EntityCommonStockSharesOutstanding", "20230430", 0, 1000.0, uom="shares"
    ),
    num_line(
        "0001-23-000001", "EntityCommonStockSharesOutstanding", "20230331", 0, 990.0, uom="shares"
    ),
    num_line("0001-23-000001", "CommonStockSharesOutstanding", "20230331", 0, 985.0, uom="shares"),
    num_line(
        "0001-23-000001", "EntityCommonStockSharesOutstanding", "20221231", 0, 900.0, uom="shares"
    ),
    # cik 200 10-K: full-year (qtrs=4) facts; a qtrs=1 stray must be ignored.
    num_line("0002-23-000001", "Revenues", "20221231", 4, 480.0),
    num_line("0002-23-000001", "Revenues", "20221231", 1, 120.0),
    num_line("0002-23-000001", "CostOfRevenue", "20221231", 4, 200.0),
    num_line("0002-23-000001", "Assets", "20221231", 0, 2000.0),
    # amendment + wrong form: their facts must never appear.
    num_line("0003-23-000001", "Revenues", "20221231", 4, 111.0),
    num_line("0004-23-000001", "Revenues", "20230331", 1, 222.0),
]


def _facts(tmp_path, ciks=None):
    zip_path = write_quarter_zip(tmp_path / "2023q2.zip", SUBS, NUMS)
    return load_quarter_facts(zip_path, ciks=ciks)


def test_tag_priority_and_consolidated_only(tmp_path):
    facts = _facts(tmp_path)
    rows = facts[(facts["cik"] == 100) & (facts["concept"] == "revenue")]
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert row["value"] == 100.0
    assert row["qtrs"] == 1


def test_form_duration_selection(tmp_path):
    facts = _facts(tmp_path)
    annual_rows = facts[(facts["cik"] == 200) & (facts["concept"] == "revenue")]
    assert len(annual_rows) == 1
    annual = annual_rows.iloc[0]
    assert annual["qtrs"] == 4
    assert annual["value"] == 480.0
    assets_rows = facts[(facts["cik"] == 200) & (facts["concept"] == "assets")]
    assert len(assets_rows) == 1
    assets = assets_rows.iloc[0]
    assert assets["qtrs"] == 0
    assert assets["value"] == 2000.0


def test_amendments_and_other_forms_never_parse(tmp_path):
    facts = _facts(tmp_path)
    assert set(facts["form"]) == {"10-Q", "10-K"}
    assert 300 not in set(facts["cik"])
    assert 400 not in set(facts["cik"])


def test_value_primitive_concepts(tmp_path):
    facts = _facts(tmp_path)
    ni_rows = facts[(facts["cik"] == 100) & (facts["concept"] == "net_income")]
    assert len(ni_rows) == 1
    ni = ni_rows.iloc[0]
    assert ni["tag"] == "NetIncomeLoss"
    assert ni["value"] == 25.0
    assert ni["qtrs"] == 1  # flow: 10-Q -> single quarter, like revenue
    eq_rows = facts[(facts["cik"] == 100) & (facts["concept"] == "equity")]
    assert len(eq_rows) == 1
    eq = eq_rows.iloc[0]
    assert eq["tag"] == "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
    assert eq["value"] == 500.0
    assert eq["qtrs"] == 0
    sh_rows = facts[(facts["cik"] == 100) & (facts["concept"] == "shares")]
    assert len(sh_rows) == 1
    sh = sh_rows.iloc[0]
    assert sh["tag"] == "EntityCommonStockSharesOutstanding"  # dei beats the us-gaap fallback
    assert sh["value"] == 1000.0  # LATEST cover-date instant wins; stale 900.0 dropped
    assert sh["qtrs"] == 0


def test_own_fiscal_period_only(tmp_path):
    facts = _facts(tmp_path)
    q = facts[facts["cik"] == 100]
    # Every fact carries the FILING's fiscal period (sub.period), including
    # the shares row whose underlying ddate is the later cover date.
    assert set(q["period"].dt.strftime("%Y%m%d")) == {"20230331"}
    assert 558.0 not in set(q["value"])


def test_cik_filter_and_dtypes(tmp_path):
    facts = _facts(tmp_path, ciks={100})
    assert set(facts["cik"]) == {100}
    assert list(facts.columns) == FACT_COLUMNS
    assert len(facts) >= 1
    assert str(facts["filed"].iloc[0].date()) == "2023-05-10"


def test_no_matching_filings_returns_empty(tmp_path):
    facts = _facts(tmp_path, ciks={999})
    assert facts.empty
    for frame in (facts, empty_facts()):
        assert list(frame.columns) == FACT_COLUMNS
        assert frame["cik"].dtype == "int64"
        assert frame["qtrs"].dtype == "int64"
        assert frame["value"].dtype == "float64"
        assert frame["period"].dtype == "datetime64[ns]"
        assert frame["filed"].dtype == "datetime64[ns]"


def test_dimensional_rows_neither_win_nor_block(tmp_path):
    subs = [
        sub_line("0010-23-000001", 500, "10-Q", "20230331", "2023", "Q1", "20230510"),
        sub_line("0011-23-000001", 600, "10-Q", "20230331", "2023", "Q1", "20230510"),
    ]
    nums = [
        # cik 500: the HIGHEST-priority revenue tag exists ONLY as a
        # dimensional row -> it must not win; the next-priority tag's
        # consolidated row must be selected instead (not blocked).
        num_line(
            "0010-23-000001",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "20230331",
            1,
            70.0,
            segments="Product=A;",
        ),
        num_line("0010-23-000001", "Revenues", "20230331", 1, 60.0),
        # cik 600: NO consolidated row for ANY revenue chain tag -> the
        # concept must be absent for this filing entirely.
        num_line(
            "0011-23-000001",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "20230331",
            1,
            71.0,
            segments="Product=A;",
        ),
        num_line("0011-23-000001", "Revenues", "20230331", 1, 61.0, segments="Region=US;"),
    ]
    facts = load_quarter_facts(write_quarter_zip(tmp_path / "dim.zip", subs, nums))
    rev = facts[(facts["cik"] == 500) & (facts["concept"] == "revenue")]
    assert len(rev) == 1
    assert rev.iloc[0]["tag"] == "Revenues"
    assert rev.iloc[0]["value"] == 60.0
    assert facts[(facts["cik"] == 600) & (facts["concept"] == "revenue")].empty


def test_pre_asc606_revenue_and_cogs_fallbacks(tmp_path):
    # Pre-ASC-606 filings (before ~2018) used SalesRevenueNet / CostOfGoodsSold.
    # When NONE of the modern tags are present, these legacy fallbacks resolve;
    # when a modern tag IS present alongside the legacy one, the modern tag
    # must still win (post-606 tags stay first in TAG_PRIORITY).
    subs = [
        # cik 900: legacy-only filing -> the pre-606 fallbacks resolve.
        sub_line("0030-18-000001", 900, "10-K", "20171231", "2017", "FY", "20180215"),
        # cik 910: transition filing carrying BOTH a modern tag and the legacy
        # fallback -> the modern tag must win, value unchanged.
        sub_line("0031-18-000001", 910, "10-K", "20171231", "2017", "FY", "20180215"),
    ]
    nums = [
        num_line("0030-18-000001", "SalesRevenueNet", "20171231", 4, 300.0),
        num_line("0030-18-000001", "CostOfGoodsSold", "20171231", 4, 180.0),
        num_line("0030-18-000001", "Assets", "20171231", 0, 1000.0),
        # cik 910: modern + legacy both present for each concept.
        num_line("0031-18-000001", "Revenues", "20171231", 4, 500.0),
        num_line("0031-18-000001", "SalesRevenueNet", "20171231", 4, 499.0),
        num_line("0031-18-000001", "CostOfRevenue", "20171231", 4, 250.0),
        num_line("0031-18-000001", "CostOfGoodsSold", "20171231", 4, 249.0),
        num_line("0031-18-000001", "Assets", "20171231", 0, 2000.0),
    ]
    facts = load_quarter_facts(write_quarter_zip(tmp_path / "pre606.zip", subs, nums))

    rev = facts[(facts["cik"] == 900) & (facts["concept"] == "revenue")]
    assert len(rev) == 1
    assert rev.iloc[0]["tag"] == "SalesRevenueNet"
    assert rev.iloc[0]["value"] == 300.0
    cogs = facts[(facts["cik"] == 900) & (facts["concept"] == "cogs")]
    assert len(cogs) == 1
    assert cogs.iloc[0]["tag"] == "CostOfGoodsSold"
    assert cogs.iloc[0]["value"] == 180.0

    # cik 910: modern tags win over the legacy fallbacks.
    rev2 = facts[(facts["cik"] == 910) & (facts["concept"] == "revenue")]
    assert rev2.iloc[0]["tag"] == "Revenues"
    assert rev2.iloc[0]["value"] == 500.0
    cogs2 = facts[(facts["cik"] == 910) & (facts["concept"] == "cogs")]
    assert cogs2.iloc[0]["tag"] == "CostOfRevenue"
    assert cogs2.iloc[0]["value"] == 250.0


def test_shares_fallback_tag_requires_period_end_date(tmp_path):
    # The cover-date relaxation (ddate >= period) is dei-only: the us-gaap
    # CommonStockSharesOutstanding fallback is a normal balance-sheet instant
    # and must match the period end exactly.
    subs = [
        sub_line("0020-23-000001", 700, "10-Q", "20230331", "2023", "Q1", "20230510"),
        sub_line("0021-23-000001", 800, "10-Q", "20230331", "2023", "Q1", "20230510"),
    ]
    nums = [
        # cik 700: fallback tag exists ONLY post-period -> NOT selected.
        num_line(
            "0020-23-000001", "CommonStockSharesOutstanding", "20230430", 0, 800.0, uom="shares"
        ),
        # cik 800: fallback tag at the period end -> selected.
        num_line(
            "0021-23-000001", "CommonStockSharesOutstanding", "20230331", 0, 850.0, uom="shares"
        ),
    ]
    facts = load_quarter_facts(write_quarter_zip(tmp_path / "shares.zip", subs, nums))
    assert facts[(facts["cik"] == 700) & (facts["concept"] == "shares")].empty
    sh = facts[(facts["cik"] == 800) & (facts["concept"] == "shares")]
    assert len(sh) == 1
    assert sh.iloc[0]["tag"] == "CommonStockSharesOutstanding"
    assert sh.iloc[0]["value"] == 850.0
