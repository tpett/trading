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
    row = facts[(facts["cik"] == 100) & (facts["concept"] == "revenue")].iloc[0]
    assert row["tag"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert row["value"] == 100.0
    assert row["qtrs"] == 1


def test_form_duration_selection(tmp_path):
    facts = _facts(tmp_path)
    annual = facts[(facts["cik"] == 200) & (facts["concept"] == "revenue")].iloc[0]
    assert annual["qtrs"] == 4
    assert annual["value"] == 480.0
    assets = facts[(facts["cik"] == 200) & (facts["concept"] == "assets")].iloc[0]
    assert assets["qtrs"] == 0
    assert assets["value"] == 2000.0


def test_amendments_and_other_forms_never_parse(tmp_path):
    facts = _facts(tmp_path)
    assert set(facts["form"]) == {"10-Q", "10-K"}
    assert 300 not in set(facts["cik"])
    assert 400 not in set(facts["cik"])


def test_value_primitive_concepts(tmp_path):
    facts = _facts(tmp_path)
    ni = facts[(facts["cik"] == 100) & (facts["concept"] == "net_income")].iloc[0]
    assert ni["tag"] == "NetIncomeLoss"
    assert ni["value"] == 25.0
    assert ni["qtrs"] == 1  # flow: 10-Q -> single quarter, like revenue
    eq = facts[(facts["cik"] == 100) & (facts["concept"] == "equity")].iloc[0]
    assert eq["tag"] == "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
    assert eq["value"] == 500.0
    assert eq["qtrs"] == 0
    sh = facts[(facts["cik"] == 100) & (facts["concept"] == "shares")].iloc[0]
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
    assert str(facts["filed"].iloc[0].date()) == "2023-05-10"


def test_no_matching_filings_returns_empty(tmp_path):
    facts = _facts(tmp_path, ciks={999})
    assert facts.empty
    assert list(empty_facts().columns) == FACT_COLUMNS
