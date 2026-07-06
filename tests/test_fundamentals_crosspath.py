"""Cross-path identity: the invariant the companyfacts top-up exists to protect.

ONE canonical filing set is rendered in BOTH representations -- a quarterly
ZIP fixture (sub.txt/num.txt via fundamentals_helpers) and its companyfacts
JSON equivalent -- and the two parsers must produce byte-identical normalized
facts tables. The 10-K is deliberately rich: a tag-priority collision (both
revenue tags present with different values), the dei cover-page share count
dated after the period end (re-dating exercised), qtrs=4 flows, and a
prior-year same-tag comparative. The Q3 10-Q carries a prior-year
SAME-quarter comparative (~92-day duration, same tag).

If a future edit to either parser diverges them, these tests fail loudly --
and the store-convergence test proves ingest ORDER can never matter: a filing
seen first via companyfacts then via the next quarter's ZIP (or vice versa)
leaves the append-only store in the identical final state.
"""

import pandas as pd

from fundamentals_helpers import num_line, sub_line, write_quarter_zip
from trading.fundamentals.companyfacts import facts_from_companyfacts
from trading.fundamentals.edgar import load_quarter_facts
from trading.fundamentals.metrics import compute_pit_series
from trading.fundamentals.store import FundamentalsStore

CIK = 777
REV_TAG = "RevenueFromContractWithCustomerExcludingAssessedTax"

# --- ZIP representation: 3 10-Qs + 1 rich 10-K ---
SUBS = [
    sub_line("cp-01", CIK, "10-Q", "20230331", "2023", "Q1", "20230510"),
    sub_line("cp-02", CIK, "10-Q", "20230630", "2023", "Q2", "20230809"),
    sub_line("cp-03", CIK, "10-Q", "20230930", "2023", "Q3", "20231108"),
    sub_line("cp-04", CIK, "10-K", "20231231", "2023", "FY", "20240220"),
]
NUMS = [
    # Q1
    num_line("cp-01", REV_TAG, "20230331", 1, 10.0),
    num_line("cp-01", "CostOfGoodsAndServicesSold", "20230331", 1, 4.0),
    num_line("cp-01", "Assets", "20230331", 0, 90.0),
    num_line("cp-01", "NetIncomeLoss", "20230331", 1, 3.0),
    num_line("cp-01", "StockholdersEquity", "20230331", 0, 50.0),
    num_line("cp-01", "EntityCommonStockSharesOutstanding", "20230430", 0, 100.0, uom="shares"),
    # Q2
    num_line("cp-02", REV_TAG, "20230630", 1, 12.0),
    num_line("cp-02", "CostOfGoodsAndServicesSold", "20230630", 1, 5.0),
    num_line("cp-02", "Assets", "20230630", 0, 95.0),
    num_line("cp-02", "NetIncomeLoss", "20230630", 1, 4.0),
    num_line("cp-02", "StockholdersEquity", "20230630", 0, 52.0),
    num_line("cp-02", "EntityCommonStockSharesOutstanding", "20230731", 0, 100.0, uom="shares"),
    # Q3, including a prior-year SAME-quarter comparative under the SAME tag
    # (ddate != period drops it in the ZIP path).
    num_line("cp-03", REV_TAG, "20230930", 1, 11.0),
    num_line("cp-03", REV_TAG, "20220930", 1, 8.0),
    num_line("cp-03", "CostOfGoodsAndServicesSold", "20230930", 1, 5.0),
    num_line("cp-03", "Assets", "20230930", 0, 98.0),
    num_line("cp-03", "NetIncomeLoss", "20230930", 1, 3.0),
    num_line("cp-03", "StockholdersEquity", "20230930", 0, 54.0),
    num_line("cp-03", "EntityCommonStockSharesOutstanding", "20231031", 0, 100.0, uom="shares"),
    # 10-K (the canonical rich filing): tag-priority collision (both revenue
    # tags, DIFFERENT values -- the priority tag must win in both paths),
    # prior-year same-tag comparative, qtrs=4 flows, dei cover-date shares.
    num_line("cp-04", REV_TAG, "20231231", 4, 48.0),
    num_line("cp-04", "Revenues", "20231231", 4, 47.5),
    num_line("cp-04", REV_TAG, "20221231", 4, 40.0),
    num_line("cp-04", "CostOfGoodsAndServicesSold", "20231231", 4, 20.0),
    num_line("cp-04", "Assets", "20231231", 0, 100.0),
    num_line("cp-04", "NetIncomeLoss", "20231231", 4, 14.0),
    num_line("cp-04", "StockholdersEquity", "20231231", 0, 60.0),
    num_line("cp-04", "EntityCommonStockSharesOutstanding", "20240131", 0, 101.0, uom="shares"),
]


def _e(end, val, accn, fp, form, filed, start=None):
    entry = {
        "end": end,
        "val": val,
        "accn": accn,
        "fy": 2023,
        "fp": fp,
        "form": form,
        "filed": filed,
    }
    if start is not None:
        entry["start"] = start
    return entry


# --- companyfacts JSON representation of the SAME filings ---
PAYLOAD = {
    "facts": {
        "us-gaap": {
            REV_TAG: {
                "units": {
                    "USD": [
                        _e(
                            "2023-03-31",
                            10.0,
                            "cp-01",
                            "Q1",
                            "10-Q",
                            "2023-05-10",
                            start="2023-01-01",
                        ),
                        _e(
                            "2023-06-30",
                            12.0,
                            "cp-02",
                            "Q2",
                            "10-Q",
                            "2023-08-09",
                            start="2023-04-01",
                        ),
                        _e(
                            "2023-09-30",
                            11.0,
                            "cp-03",
                            "Q3",
                            "10-Q",
                            "2023-11-08",
                            start="2023-07-01",
                        ),
                        # Prior-year same-quarter comparative (same tag, ~92
                        # days): own-period filter must drop it.
                        _e(
                            "2022-09-30",
                            8.0,
                            "cp-03",
                            "Q3",
                            "10-Q",
                            "2023-11-08",
                            start="2022-07-01",
                        ),
                        _e(
                            "2023-12-31",
                            48.0,
                            "cp-04",
                            "FY",
                            "10-K",
                            "2024-02-20",
                            start="2023-01-01",
                        ),
                        # Prior-year full-year comparative in the 10-K.
                        _e(
                            "2022-12-31",
                            40.0,
                            "cp-04",
                            "FY",
                            "10-K",
                            "2024-02-20",
                            start="2022-01-01",
                        ),
                    ]
                }
            },
            "Revenues": {
                "units": {
                    "USD": [
                        # Tag-priority collision: present with a DIFFERENT
                        # value -- the priority tag (REV_TAG) must win.
                        _e(
                            "2023-12-31",
                            47.5,
                            "cp-04",
                            "FY",
                            "10-K",
                            "2024-02-20",
                            start="2023-01-01",
                        ),
                    ]
                }
            },
            "CostOfGoodsAndServicesSold": {
                "units": {
                    "USD": [
                        _e(
                            "2023-03-31",
                            4.0,
                            "cp-01",
                            "Q1",
                            "10-Q",
                            "2023-05-10",
                            start="2023-01-01",
                        ),
                        _e(
                            "2023-06-30",
                            5.0,
                            "cp-02",
                            "Q2",
                            "10-Q",
                            "2023-08-09",
                            start="2023-04-01",
                        ),
                        _e(
                            "2023-09-30",
                            5.0,
                            "cp-03",
                            "Q3",
                            "10-Q",
                            "2023-11-08",
                            start="2023-07-01",
                        ),
                        _e(
                            "2023-12-31",
                            20.0,
                            "cp-04",
                            "FY",
                            "10-K",
                            "2024-02-20",
                            start="2023-01-01",
                        ),
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        _e("2023-03-31", 90.0, "cp-01", "Q1", "10-Q", "2023-05-10"),
                        _e("2023-06-30", 95.0, "cp-02", "Q2", "10-Q", "2023-08-09"),
                        _e("2023-09-30", 98.0, "cp-03", "Q3", "10-Q", "2023-11-08"),
                        _e("2023-12-31", 100.0, "cp-04", "FY", "10-K", "2024-02-20"),
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        _e(
                            "2023-03-31",
                            3.0,
                            "cp-01",
                            "Q1",
                            "10-Q",
                            "2023-05-10",
                            start="2023-01-01",
                        ),
                        _e(
                            "2023-06-30",
                            4.0,
                            "cp-02",
                            "Q2",
                            "10-Q",
                            "2023-08-09",
                            start="2023-04-01",
                        ),
                        _e(
                            "2023-09-30",
                            3.0,
                            "cp-03",
                            "Q3",
                            "10-Q",
                            "2023-11-08",
                            start="2023-07-01",
                        ),
                        _e(
                            "2023-12-31",
                            14.0,
                            "cp-04",
                            "FY",
                            "10-K",
                            "2024-02-20",
                            start="2023-01-01",
                        ),
                    ]
                }
            },
            "StockholdersEquity": {
                "units": {
                    "USD": [
                        _e("2023-03-31", 50.0, "cp-01", "Q1", "10-Q", "2023-05-10"),
                        _e("2023-06-30", 52.0, "cp-02", "Q2", "10-Q", "2023-08-09"),
                        _e("2023-09-30", 54.0, "cp-03", "Q3", "10-Q", "2023-11-08"),
                        _e("2023-12-31", 60.0, "cp-04", "FY", "10-K", "2024-02-20"),
                    ]
                }
            },
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {
                    "shares": [  # cover-page counts: dated AFTER each period end
                        _e("2023-04-30", 100.0, "cp-01", "Q1", "10-Q", "2023-05-10"),
                        _e("2023-07-31", 100.0, "cp-02", "Q2", "10-Q", "2023-08-09"),
                        _e("2023-10-31", 100.0, "cp-03", "Q3", "10-Q", "2023-11-08"),
                        _e("2024-01-31", 101.0, "cp-04", "FY", "10-K", "2024-02-20"),
                    ]
                }
            }
        },
    }
}


def _aligned(facts: pd.DataFrame) -> pd.DataFrame:
    return facts.sort_values(["adsh", "concept"], kind="mergesort").reset_index(drop=True)


def _zip_facts(tmp_path) -> pd.DataFrame:
    zip_path = write_quarter_zip(tmp_path / "crosspath.zip", SUBS, NUMS)
    return load_quarter_facts(zip_path, ciks={CIK})


def test_both_paths_produce_byte_identical_facts(tmp_path):
    zip_facts = _aligned(_zip_facts(tmp_path))
    cf_facts = _aligned(facts_from_companyfacts(PAYLOAD, CIK))
    pd.testing.assert_frame_equal(cf_facts, zip_facts)
    # Pin the hard cases explicitly (guards against BOTH paths regressing in
    # lockstep to a wrong answer the frame-equality alone would not catch):
    fy_rev = cf_facts[(cf_facts["adsh"] == "cp-04") & (cf_facts["concept"] == "revenue")]
    assert list(fy_rev["tag"]) == [REV_TAG]  # priority tag beat Revenues=47.5
    assert list(fy_rev["value"]) == [48.0]
    assert 40.0 not in set(cf_facts["value"])  # 10-K prior-year comparative dropped
    q3_rev = cf_facts[(cf_facts["adsh"] == "cp-03") & (cf_facts["concept"] == "revenue")]
    assert list(q3_rev["value"]) == [11.0]  # prior-year same-quarter comparative dropped
    sh = cf_facts[(cf_facts["adsh"] == "cp-04") & (cf_facts["concept"] == "shares")].iloc[0]
    assert sh["value"] == 101.0
    assert sh["period"] == pd.Timestamp("2023-12-31")  # cover date re-dated


def test_store_converges_regardless_of_ingest_order(tmp_path):
    zip_series = compute_pit_series(_zip_facts(tmp_path))[CIK]
    cf_series = compute_pit_series(facts_from_companyfacts(PAYLOAD, CIK))[CIK]

    zip_first = FundamentalsStore(tmp_path / "zip_first")
    assert zip_first.append("TEST", zip_series) == 4
    assert zip_first.append("TEST", cf_series) == 0  # pure no-op collision

    cf_first = FundamentalsStore(tmp_path / "cf_first")
    assert cf_first.append("TEST", cf_series) == 4
    assert cf_first.append("TEST", zip_series) == 0

    pd.testing.assert_frame_equal(zip_first.read("TEST"), cf_first.read("TEST"))
