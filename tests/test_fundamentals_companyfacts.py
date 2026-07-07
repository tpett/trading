import datetime

import pandas as pd

from trading.fundamentals.companyfacts import (
    _http_get_json,
    facts_from_companyfacts,
    http_get_json,
    refresh_fundamentals,
)
from trading.fundamentals.store import FundamentalsStore

CIK = 320193
CIK_MAP = pd.DataFrame({"symbol": ["AAPL"], "cik": [CIK], "start": ["2017-01-01"], "end": [""]})


def _entry(end, val, accn, fy, fp, form, filed, start=None):
    entry = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}
    if start is not None:
        entry["start"] = start
    return entry


def _payload():
    """Four 10-Qs + one 10-K, plus traps: a comparative period re-reported in
    the 10-K, an amendment, and a year-to-date (qtrs=2) duration."""
    rev = [
        _entry("2023-03-31", 10.0, "a-01", 2023, "Q1", "10-Q", "2023-05-10", start="2023-01-01"),
        _entry("2023-06-30", 12.0, "a-02", 2023, "Q2", "10-Q", "2023-08-09", start="2023-04-01"),
        # YTD duration inside the same 10-Q: must be dropped (6 months != 1 quarter).
        _entry("2023-06-30", 22.0, "a-02", 2023, "Q2", "10-Q", "2023-08-09", start="2023-01-01"),
        _entry("2023-09-30", 11.0, "a-03", 2023, "Q3", "10-Q", "2023-11-08", start="2023-07-01"),
        # Prior-year SAME-quarter comparative inside the Q3 10-Q: same tag,
        # ~92-day duration (passes the quarter-length check), so ONLY the
        # own-period filter (max end per accession) can drop it.
        _entry("2022-09-30", 8.0, "a-03", 2023, "Q3", "10-Q", "2023-11-08", start="2022-07-01"),
        _entry("2023-12-31", 48.0, "a-04", 2023, "FY", "10-K", "2024-02-20", start="2023-01-01"),
        # Comparative full-year 2022 re-reported inside the FY2023 10-K: the
        # own-period filter (max end per accession) must drop it.
        _entry("2022-12-31", 40.0, "a-04", 2023, "FY", "10-K", "2024-02-20", start="2022-01-01"),
        # Amendment: never parses.
        _entry("2023-03-31", 999.0, "a-05", 2023, "Q1", "10-Q/A", "2023-09-01", start="2023-01-01"),
    ]
    cogs = [
        _entry("2023-03-31", 4.0, "a-01", 2023, "Q1", "10-Q", "2023-05-10", start="2023-01-01"),
        _entry("2023-06-30", 5.0, "a-02", 2023, "Q2", "10-Q", "2023-08-09", start="2023-04-01"),
        _entry("2023-09-30", 5.0, "a-03", 2023, "Q3", "10-Q", "2023-11-08", start="2023-07-01"),
        _entry("2023-12-31", 20.0, "a-04", 2023, "FY", "10-K", "2024-02-20", start="2023-01-01"),
    ]
    assets = [
        _entry("2023-03-31", 90.0, "a-01", 2023, "Q1", "10-Q", "2023-05-10"),
        _entry("2023-06-30", 95.0, "a-02", 2023, "Q2", "10-Q", "2023-08-09"),
        _entry("2023-09-30", 98.0, "a-03", 2023, "Q3", "10-Q", "2023-11-08"),
        _entry("2023-12-31", 100.0, "a-04", 2023, "FY", "10-K", "2024-02-20"),
    ]
    net_income = [
        _entry("2023-03-31", 3.0, "a-01", 2023, "Q1", "10-Q", "2023-05-10", start="2023-01-01"),
        _entry("2023-06-30", 4.0, "a-02", 2023, "Q2", "10-Q", "2023-08-09", start="2023-04-01"),
        _entry("2023-09-30", 3.0, "a-03", 2023, "Q3", "10-Q", "2023-11-08", start="2023-07-01"),
        _entry("2023-12-31", 14.0, "a-04", 2023, "FY", "10-K", "2024-02-20", start="2023-01-01"),
    ]
    equity = [
        _entry("2023-03-31", 50.0, "a-01", 2023, "Q1", "10-Q", "2023-05-10"),
        _entry("2023-06-30", 52.0, "a-02", 2023, "Q2", "10-Q", "2023-08-09"),
        _entry("2023-09-30", 54.0, "a-03", 2023, "Q3", "10-Q", "2023-11-08"),
        _entry("2023-12-31", 60.0, "a-04", 2023, "FY", "10-K", "2024-02-20"),
    ]
    shares = [  # dei cover-page counts: dated AFTER each fiscal period end
        _entry("2023-04-30", 100.0, "a-01", 2023, "Q1", "10-Q", "2023-05-10"),
        _entry("2023-07-31", 100.0, "a-02", 2023, "Q2", "10-Q", "2023-08-09"),
        _entry("2023-10-31", 100.0, "a-03", 2023, "Q3", "10-Q", "2023-11-08"),
        _entry("2024-01-31", 101.0, "a-04", 2023, "FY", "10-K", "2024-02-20"),
    ]
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": rev}},
                "CostOfRevenue": {"units": {"USD": cogs}},
                "Assets": {"units": {"USD": assets}},
                "NetIncomeLoss": {"units": {"USD": net_income}},
                "StockholdersEquity": {"units": {"USD": equity}},
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {"units": {"shares": shares}},
            },
        }
    }


def test_http_get_json_public_alias_is_the_same_function():
    # scripts/build_cik_map.py imports the public name across the module
    # boundary; it must stay the identical callable, not a copy.
    assert http_get_json is _http_get_json


def test_normalizes_to_own_period_facts_only():
    facts = facts_from_companyfacts(_payload(), CIK)
    assert set(facts["adsh"]) == {"a-01", "a-02", "a-03", "a-04"}
    fy = facts[(facts["adsh"] == "a-04") & (facts["concept"] == "revenue")]
    assert list(fy["value"]) == [48.0]
    assert list(fy["qtrs"]) == [4]
    q2 = facts[(facts["adsh"] == "a-02") & (facts["concept"] == "revenue")]
    assert list(q2["value"]) == [12.0]  # the qtrs=2 YTD duration was dropped
    q3 = facts[(facts["adsh"] == "a-03") & (facts["concept"] == "revenue")]
    assert list(q3["value"]) == [11.0]  # prior-year same-quarter comparative dropped


def test_shares_cover_date_maps_to_the_filing_period():
    facts = facts_from_companyfacts(_payload(), CIK)
    sh = facts[(facts["adsh"] == "a-04") & (facts["concept"] == "shares")].iloc[0]
    assert sh["value"] == 101.0
    assert sh["qtrs"] == 0
    # The dei cover date (2024-01-31) must neither define the filing's own
    # period nor survive in the period column: it is re-dated to 2023-12-31.
    assert sh["period"] == pd.Timestamp("2023-12-31")
    ni = facts[(facts["adsh"] == "a-04") & (facts["concept"] == "net_income")].iloc[0]
    assert ni["value"] == 14.0
    assert ni["qtrs"] == 4


def _legacy_payload(revenue_tag: str, cogs_tag: str, extra_us_gaap: dict | None = None) -> dict:
    """A single pre-ASC-606 FY 10-K reporting revenue/COGS under the given
    tags (plus Assets). `extra_us_gaap` can add a competing modern tag."""
    us_gaap = {
        revenue_tag: {
            "units": {
                "USD": [
                    _entry(
                        "2017-12-31",
                        300.0,
                        "z-01",
                        2017,
                        "FY",
                        "10-K",
                        "2018-02-15",
                        start="2017-01-01",
                    )
                ]
            }
        },
        cogs_tag: {
            "units": {
                "USD": [
                    _entry(
                        "2017-12-31",
                        180.0,
                        "z-01",
                        2017,
                        "FY",
                        "10-K",
                        "2018-02-15",
                        start="2017-01-01",
                    )
                ]
            }
        },
        "Assets": {
            "units": {
                "USD": [_entry("2017-12-31", 1000.0, "z-01", 2017, "FY", "10-K", "2018-02-15")]
            }
        },
    }
    if extra_us_gaap:
        us_gaap.update(extra_us_gaap)
    return {"facts": {"us-gaap": us_gaap}}


def test_companyfacts_pre_asc606_fallbacks_resolve():
    # SalesRevenueNet / CostOfGoodsSold are us-gaap, so the companyfacts path
    # (which defaults unknown tags to us-gaap and iterates TAG_PRIORITY) must
    # pick them up with NO _TAXONOMY_BY_TAG entry.
    facts = facts_from_companyfacts(_legacy_payload("SalesRevenueNet", "CostOfGoodsSold"), CIK)
    rev = facts[facts["concept"] == "revenue"].iloc[0]
    assert rev["tag"] == "SalesRevenueNet"
    assert rev["value"] == 300.0
    cogs = facts[facts["concept"] == "cogs"].iloc[0]
    assert cogs["tag"] == "CostOfGoodsSold"
    assert cogs["value"] == 180.0


def test_companyfacts_modern_tags_still_win_over_legacy():
    # A transition filing carrying both the modern tag and the legacy fallback
    # must resolve to the modern tag (priority order preserved).
    extra = {
        "Revenues": {
            "units": {
                "USD": [
                    _entry(
                        "2017-12-31",
                        500.0,
                        "z-01",
                        2017,
                        "FY",
                        "10-K",
                        "2018-02-15",
                        start="2017-01-01",
                    )
                ]
            }
        },
        "CostOfRevenue": {
            "units": {
                "USD": [
                    _entry(
                        "2017-12-31",
                        250.0,
                        "z-01",
                        2017,
                        "FY",
                        "10-K",
                        "2018-02-15",
                        start="2017-01-01",
                    )
                ]
            }
        },
    }
    facts = facts_from_companyfacts(
        _legacy_payload("SalesRevenueNet", "CostOfGoodsSold", extra_us_gaap=extra), CIK
    )
    assert facts[facts["concept"] == "revenue"].iloc[0]["tag"] == "Revenues"
    assert facts[facts["concept"] == "revenue"].iloc[0]["value"] == 500.0
    assert facts[facts["concept"] == "cogs"].iloc[0]["tag"] == "CostOfRevenue"
    assert facts[facts["concept"] == "cogs"].iloc[0]["value"] == 250.0


def test_refresh_appends_only_new_filed_dates(tmp_path):
    store = FundamentalsStore(tmp_path)
    calls: list[str] = []

    def fetch(url: str) -> dict:
        calls.append(url)
        return _payload()

    added, degraded = refresh_fundamentals(
        store, CIK_MAP, ["AAPL"], as_of=datetime.date(2024, 3, 1), fetch_json=fetch
    )
    assert (added, degraded) == (4, False)
    assert calls == [f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK:010d}.json"]
    at_10k = store.read("AAPL").loc[pd.Timestamp("2024-02-20", tz="UTC")]
    assert at_10k["gross_profitability"] == (48.0 - 20.0) / 100.0
    assert at_10k["ttm_net_income"] == 14.0  # derived Q4 = 14 - (3+4+3) = 4; TTM = FY
    assert at_10k["book_equity"] == 60.0
    assert at_10k["shares_outstanding"] == 101.0

    added, degraded = refresh_fundamentals(
        store, CIK_MAP, ["AAPL"], as_of=datetime.date(2024, 3, 8), fetch_json=fetch
    )
    assert (added, degraded) == (0, False)  # append-only: nothing new to add


def test_refresh_is_fail_open_per_symbol(tmp_path):
    store = FundamentalsStore(tmp_path)
    cik_map = pd.DataFrame(
        {
            "symbol": ["AAPL", "BOOM", "UNMAPPED"],
            "cik": [CIK, 999, 0],
            "start": ["2017-01-01", "2017-01-01", "2017-01-01"],
            "end": ["", "", ""],
        }
    )
    cik_map = cik_map[cik_map["symbol"] != "UNMAPPED"]  # UNMAPPED has no row at all

    def fetch(url: str) -> dict:
        if "0000000999" in url:
            raise OSError("edgar down")
        return _payload()

    added, degraded = refresh_fundamentals(
        store,
        cik_map,
        ["AAPL", "BOOM", "UNMAPPED"],
        as_of=datetime.date(2024, 3, 1),
        fetch_json=fetch,
    )
    assert degraded is True  # BOOM failed -> degraded, run continues
    assert added == 4  # AAPL still refreshed; UNMAPPED silently skipped


def test_refresh_stops_cleanly_once_the_budget_elapses(tmp_path, monkeypatch):
    # A fake slow fetcher: each fetch call "takes" far longer than the
    # budget, simulated by advancing a fake clock instead of really sleeping.
    clock = {"t": 0.0}
    monkeypatch.setattr("trading.fundamentals.companyfacts.time.monotonic", lambda: clock["t"])

    store = FundamentalsStore(tmp_path)
    cik_map = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "cik": [1, 2, 3],
            "start": ["2017-01-01"] * 3,
            "end": [""] * 3,
        }
    )
    calls: list[str] = []

    def slow_fetch(url: str) -> dict:
        calls.append(url)
        clock["t"] += 1000.0  # each fetch blows through any reasonable budget
        return _payload()

    added, degraded = refresh_fundamentals(
        store,
        cik_map,
        ["A", "B", "C"],
        as_of=datetime.date(2024, 3, 1),
        fetch_json=slow_fetch,
        budget_s=500.0,
    )
    assert degraded is True
    assert len(calls) == 1  # only the first symbol got processed before the budget tripped
    assert added == 4  # that symbol's rows are still kept -- not a partial write
    assert store.read("A").empty is False
    assert store.read("B").empty  # deferred to the next scheduled refresh
    assert store.read("C").empty


def test_refresh_default_budget_is_unbounded(tmp_path):
    # No budget_s passed: existing callers/behavior are unaffected.
    store = FundamentalsStore(tmp_path)
    added, degraded = refresh_fundamentals(
        store, CIK_MAP, ["AAPL"], as_of=datetime.date(2024, 3, 1), fetch_json=lambda url: _payload()
    )
    assert (added, degraded) == (4, False)
