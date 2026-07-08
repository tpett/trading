import datetime
from pathlib import Path

import pandas as pd
import pytest

from trading.config import load_venue_config
from trading.venues.base import OHLCV_COLUMNS, DataFetchError, VenueConstraints
from trading.venues.equities import EquitiesAdapter

CONFIG = load_venue_config("equities", Path("config"))


def _yf_style_frame(symbol: str) -> pd.DataFrame:
    """Mimic yfinance.download output: naive index, MultiIndex (Price, Ticker) columns."""
    idx = pd.date_range("2026-01-05", periods=5, freq="B")  # naive, like yfinance
    data = {
        ("Open", symbol): [10.0, 10.5, 10.2, 10.8, 11.0],
        ("High", symbol): [10.6, 10.9, 10.7, 11.2, 11.4],
        ("Low", symbol): [9.8, 10.1, 9.9, 10.5, 10.7],
        ("Close", symbol): [10.5, 10.2, 10.6, 11.0, 11.2],
        ("Volume", symbol): [1e6, 1.1e6, 9e5, 1.3e6, 1.2e6],
    }
    return pd.DataFrame(data, index=idx)


MEMBERSHIP_CSV = """# test fixture
symbol,index,start,end
AAA,sp500,2018-01-01,
BBB,sp500,2018-01-01,2020-06-01
CCC,ndx,2020-06-01,
"""


def _adapter_with(tmp_path, text: str) -> EquitiesAdapter:
    path = tmp_path / "membership.csv"
    path.write_text(text)
    config = load_venue_config("equities", Path("config"))
    return EquitiesAdapter(config, membership_csv=path)


def test_universe_is_point_in_time(tmp_path):
    adapter = _adapter_with(tmp_path, MEMBERSHIP_CSV)
    in_2019 = {i.symbol for i in adapter.universe(datetime.date(2019, 1, 2))}
    in_2021 = {i.symbol for i in adapter.universe(datetime.date(2021, 1, 4))}
    assert in_2019 == {"AAA", "BBB"}
    assert in_2021 == {"AAA", "CCC"}
    assert all(i.status == "tradable" for i in adapter.universe(datetime.date(2021, 1, 4)))


MEMBERSHIP_CSV_WITH_SP400 = """# test fixture
symbol,index,start,end
AAA,sp500,2018-01-01,
BBB,sp500,2018-01-01,2020-06-01
CCC,ndx,2020-06-01,
DDD,sp400,2018-01-01,
"""


def test_universe_excludes_sp400_by_default(tmp_path):
    # Live/paper invariant: sp400 rows exist in the CSV, but the default
    # config.universe.indices is ("sp500", "ndx") -- unchanged behavior.
    adapter = _adapter_with(tmp_path, MEMBERSHIP_CSV_WITH_SP400)
    symbols = {i.symbol for i in adapter.universe(datetime.date(2021, 1, 4))}
    assert symbols == {"AAA", "CCC"}
    assert "DDD" not in symbols


def test_universe_includes_sp400_when_config_opts_in(tmp_path):
    from dataclasses import replace

    path = tmp_path / "membership.csv"
    path.write_text(MEMBERSHIP_CSV_WITH_SP400)
    config = replace(CONFIG, universe=replace(CONFIG.universe, indices=("sp500", "ndx", "sp400")))
    adapter = EquitiesAdapter(config, membership_csv=path)
    symbols = {i.symbol for i in adapter.universe(datetime.date(2021, 1, 4))}
    assert symbols == {"AAA", "CCC", "DDD"}


def test_universe_restricted_by_symbols_allowlist(tmp_path):
    # The options-skew experiment path: symbols_allowlist_path narrows the PIT
    # universe to the gathered names, still respecting point-in-time membership.
    from dataclasses import replace

    path = tmp_path / "membership.csv"
    path.write_text(MEMBERSHIP_CSV)
    # A samples.jsonl-shaped allowlist: only AAA is "gathered".
    allow = tmp_path / "samples.jsonl"
    allow.write_text('{"symbol": "AAA", "decision_date": "2019-01-02", "skew_put_atm": 0.05}\n')
    config = replace(
        CONFIG, universe=replace(CONFIG.universe, symbols_allowlist_path=str(allow))
    )
    adapter = EquitiesAdapter(config, membership_csv=path)
    # 2019: PIT members are {AAA, BBB} but only AAA is allowlisted.
    assert {i.symbol for i in adapter.universe(datetime.date(2019, 1, 2))} == {"AAA"}
    # An allowlisted name that is NOT a PIT member on a date stays excluded
    # (membership filter still applies): AAA is a member throughout, CCC is not
    # allowlisted, so 2021 also yields just AAA.
    assert {i.symbol for i in adapter.universe(datetime.date(2021, 1, 4))} == {"AAA"}


def test_membership_intervals_returns_all_indices_unfiltered(tmp_path):
    # membership_intervals is index-agnostic by design (the recycling guard
    # cares whether a ticker is a current member of ANYTHING, not just the
    # indices this particular backtest opted into).
    adapter = _adapter_with(tmp_path, MEMBERSHIP_CSV_WITH_SP400)
    assert adapter.membership_intervals("DDD") == [("2018-01-01", "")]
    assert adapter.membership_intervals("BBB") == [("2018-01-01", "2020-06-01")]
    assert adapter.membership_intervals("ZZZ") == []


def test_membership_interval_boundaries_start_inclusive_end_exclusive(tmp_path):
    adapter = _adapter_with(tmp_path, MEMBERSHIP_CSV)
    on_start = {i.symbol for i in adapter.universe(datetime.date(2020, 6, 1))}
    day_before = {i.symbol for i in adapter.universe(datetime.date(2020, 5, 31))}
    assert "CCC" in on_start and "BBB" not in on_start  # start inclusive, end exclusive
    assert "BBB" in day_before and "CCC" not in day_before


def test_committed_membership_file_sanity():
    # The real committed file: plausible sizes, and known index churn visible.
    adapter = EquitiesAdapter(load_venue_config("equities", Path("config")))
    for day, low, high in [
        (datetime.date(2018, 6, 1), 450, 650),
        (datetime.date(2022, 6, 1), 450, 650),
        (datetime.date(2026, 7, 1), 450, 650),
    ]:
        count = len(adapter.universe(day))
        assert low <= count <= high, f"{day}: {count} members"
    early = {i.symbol for i in adapter.universe(datetime.date(2018, 6, 1))}
    today = {i.symbol for i in adapter.universe(datetime.date(2026, 7, 1))}
    assert early != today
    assert len(early - today) > 20  # real churn: many 2018 members are gone
    # sp400 rows exist in the committed CSV but are excluded by the default
    # config -- live/paper universe is untouched by their addition.
    assert not {i.symbol for i in adapter.universe(datetime.date(2026, 7, 1))} & {"CDK", "MDP"}


def test_committed_membership_file_sp400_opt_in():
    from dataclasses import replace

    base = load_venue_config("equities", Path("config"))
    sp400_only = replace(base, universe=replace(base.universe, indices=("sp400",)))
    adapter = EquitiesAdapter(sp400_only)
    for day, low, high in [
        (datetime.date(2019, 6, 1), 380, 420),
        (datetime.date(2022, 6, 1), 380, 420),
        (datetime.date(2026, 7, 1), 380, 420),
    ]:
        count = len(adapter.universe(day))
        assert low <= count <= high, f"{day}: {count} sp400 members"
    # Spot-check anchors (scout-verified against the Wikipedia changes table):
    # exact known removal dates, not just a plausible count.
    before_cdk = {i.symbol for i in adapter.universe(datetime.date(2022, 7, 5))}
    after_cdk = {i.symbol for i in adapter.universe(datetime.date(2022, 7, 6))}
    assert "CDK" in before_cdk and "CDK" not in after_cdk
    before_mdp = {i.symbol for i in adapter.universe(datetime.date(2020, 4, 26))}
    after_mdp = {i.symbol for i in adapter.universe(datetime.date(2020, 4, 27))}
    assert "MDP" in before_mdp and "MDP" not in after_mdp


def test_constraints_come_from_config():
    adapter = EquitiesAdapter(CONFIG)
    assert adapter.constraints() == VenueConstraints(
        taker_fee_bps=0.0,
        maker_fee_bps=0.0,
        slippage_bps=5.0,
        settlement_days=1,
        trades_24_7=False,
    )


def test_fetch_ohlcv_normalizes_yfinance_frame(monkeypatch):
    monkeypatch.setattr(
        "trading.venues.equities._yf_download", lambda s, start, end: _yf_style_frame(s)
    )
    adapter = EquitiesAdapter(CONFIG)
    df = adapter.fetch_ohlcv("AAPL", datetime.date(2026, 1, 5), datetime.date(2026, 1, 9))
    assert list(df.columns) == OHLCV_COLUMNS
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == 11.2
    assert len(df) == 5


def test_fetch_ohlcv_slices_to_requested_range(monkeypatch):
    monkeypatch.setattr(
        "trading.venues.equities._yf_download", lambda s, start, end: _yf_style_frame(s)
    )
    adapter = EquitiesAdapter(CONFIG)
    df = adapter.fetch_ohlcv("AAPL", datetime.date(2026, 1, 6), datetime.date(2026, 1, 8))
    assert df.index.min() == pd.Timestamp("2026-01-06", tz="UTC")
    assert df.index.max() == pd.Timestamp("2026-01-08", tz="UTC")


def _yf_style_frame_extended(symbol: str) -> pd.DataFrame:
    """_yf_download's combined output: the frozen adjusted OHLCV (identical to
    _yf_style_frame) PLUS the extended corporate-action columns. A 2:1 split on
    day 4 makes close_raw diverge from the adjusted close before it, and one
    dividend day exercises div_cash; split_factor is 1.0 except on the split."""
    base = _yf_style_frame(symbol)
    base[("close_raw", symbol)] = [21.0, 20.4, 21.2, 11.0, 11.2]
    base[("div_cash", symbol)] = [0.0, 0.25, 0.0, 0.0, 0.0]
    base[("split_factor", symbol)] = [1.0, 1.0, 1.0, 2.0, 1.0]
    return base


def test_fetch_ohlcv_yfinance_populates_extended_fields_without_moving_adjusted(monkeypatch):
    monkeypatch.setattr(
        "trading.venues.equities._yf_download", lambda s, start, end: _yf_style_frame_extended(s)
    )
    adapter = EquitiesAdapter(CONFIG)
    df = adapter.fetch_ohlcv("AAPL", datetime.date(2026, 1, 5), datetime.date(2026, 1, 9))
    assert list(df.columns) == OHLCV_COLUMNS + ["div_cash", "split_factor", "close_raw"]
    # The adjusted basis is UNCHANGED from the pre-existing expected values
    # (test_fetch_ohlcv_normalizes_yfinance_frame pins these same numbers).
    assert df["close"].tolist() == [10.5, 10.2, 10.6, 11.0, 11.2]
    assert df["open"].iloc[0] == 10.0
    assert df["volume"].iloc[-1] == 1.2e6
    # Extended fields carry the raw/action values, all float64.
    assert df["close_raw"].tolist() == [21.0, 20.4, 21.2, 11.0, 11.2]
    assert df["div_cash"].tolist() == [0.0, 0.25, 0.0, 0.0, 0.0]
    assert df["split_factor"].tolist() == [1.0, 1.0, 1.0, 2.0, 1.0]
    assert (df.dtypes == "float64").all()


def test_fetch_ohlcv_empty_raises(monkeypatch):
    monkeypatch.setattr(
        "trading.venues.equities._yf_download", lambda s, start, end: pd.DataFrame()
    )
    adapter = EquitiesAdapter(CONFIG)
    with pytest.raises(DataFetchError):
        adapter.fetch_ohlcv("AAPL", datetime.date(2026, 1, 5), datetime.date(2026, 1, 9))


# A 2:1 split sits between the two bars, so the RAW fields (open/high/low/
# close/volume) differ from the split-adjusted ones. This is what pins the
# single most important property: the adapter must read the ADJUSTED fields
# (matching yfinance auto_adjust) -- a mapping bug swapping adjClose->close
# would flip these assertions.
_TIINGO_ROWS = [
    {
        "date": "2022-02-14T00:00:00.000Z",
        "adjOpen": 97.25,
        "adjHigh": 97.5,
        "adjLow": 95.6,
        "adjClose": 97.0,
        "adjVolume": 10_000_000,
        "open": 194.5,  # raw, pre-adjustment -- must NOT be used
        "high": 195.0,
        "low": 191.2,
        "close": 194.0,
        "volume": 5_000_000,
        "divCash": 0.0,
        "splitFactor": 2.0,
    },
    {
        "date": "2022-02-11T00:00:00.000Z",  # deliberately out of order
        "adjOpen": 105.0,
        "adjHigh": 106.0,
        "adjLow": 104.0,
        "adjClose": 104.75,
        "adjVolume": 8_000_000,
        "open": 210.0,
        "high": 212.0,
        "low": 208.0,
        "close": 209.5,
        "volume": 4_000_000,
        "divCash": 0.5,  # a dividend day: div_cash must carry the raw amount
        "splitFactor": 1.0,
    },
]


def _tiingo_config():
    from dataclasses import replace

    return replace(CONFIG, data=replace(CONFIG.data, bar_source="tiingo"))


def test_tiingo_download_uses_adjusted_fields_not_raw(monkeypatch):
    import json

    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    captured = {}

    def fake_get(url, params):
        captured["url"] = url
        captured["params"] = params
        return 200, json.dumps(_TIINGO_ROWS).encode()

    monkeypatch.setattr("trading.venues.equities._tiingo_get", fake_get)
    adapter = EquitiesAdapter(_tiingo_config())
    df = adapter.fetch_ohlcv("XLNX", datetime.date(2022, 2, 11), datetime.date(2022, 2, 14))
    # Canonical OHLCV comes first; the extended corporate-action columns ride along.
    assert list(df.columns) == OHLCV_COLUMNS + ["div_cash", "split_factor", "close_raw"]
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing  # sorted despite unordered input
    # ADJUSTED values, not raw: 104.75 (adjClose) not 209.5 (close); 97.0 not 194.0.
    assert df["close"].iloc[0] == 104.75
    assert df["close"].iloc[-1] == 97.0
    assert df["open"].iloc[-1] == 97.25  # adjOpen, not raw open 194.5
    assert df["volume"].iloc[-1] == 10_000_000  # adjVolume
    # Extended fields are the RAW/action values, NOT the adjusted ones: the raw
    # close differs from adjClose across the 2:1 split, split_factor is the ratio
    # (2.0 on the split day, 1.0 otherwise), and div_cash carries the dividend.
    assert df["close_raw"].iloc[0] == 209.5  # raw close, not adjClose 104.75
    assert df["close_raw"].iloc[-1] == 194.0  # raw close, not adjClose 97.0
    assert df["split_factor"].iloc[0] == 1.0  # no split on 02-11
    assert df["split_factor"].iloc[-1] == 2.0  # 2:1 split on 02-14
    assert df["div_cash"].iloc[0] == 0.5  # dividend on 02-11
    assert df["div_cash"].iloc[-1] == 0.0
    # Every column float64, matching the yfinance path (adjVolume is int in JSON).
    assert (df.dtypes == "float64").all()
    # Token never in the URL or query params -- header only.
    assert "test-token" not in captured["url"]
    assert "token" not in captured["params"]
    assert "XLNX" in captured["url"]


def test_tiingo_404_is_empty_frame_then_raises(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setattr("trading.venues.equities._tiingo_get", lambda url, params: (404, b"{}"))
    adapter = EquitiesAdapter(_tiingo_config())
    with pytest.raises(DataFetchError, match="no equities data"):
        adapter.fetch_ohlcv("NOPE", datetime.date(2022, 1, 1), datetime.date(2022, 2, 1))


def test_tiingo_http_error_surfaces(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setattr(
        "trading.venues.equities._tiingo_get", lambda url, params: (503, b"upstream boom")
    )
    adapter = EquitiesAdapter(_tiingo_config())
    with pytest.raises(DataFetchError, match="HTTP 503"):
        adapter.fetch_ohlcv("AAPL", datetime.date(2022, 1, 1), datetime.date(2022, 2, 1))


def test_tiingo_token_missing_raises_with_hint(monkeypatch, tmp_path):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.setattr("trading.venues.equities.TIINGO_CONFIG_PATH", tmp_path / "absent.toml")
    from trading.venues.equities import tiingo_token

    with pytest.raises(DataFetchError, match="TIINGO_API_KEY"):
        tiingo_token()


def test_tiingo_token_read_from_config_file(monkeypatch, tmp_path):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    cfg = tmp_path / "config.toml"
    cfg.write_text('tiingo_api_key = "file-token-xyz"\n')
    monkeypatch.setattr("trading.venues.equities.TIINGO_CONFIG_PATH", cfg)
    from trading.venues.equities import tiingo_token

    assert tiingo_token() == "file-token-xyz"


def test_tiingo_retries_transient_then_succeeds(monkeypatch):
    import json

    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    slept = []
    monkeypatch.setattr("trading.venues.equities._tiingo_sleep", lambda s: slept.append(s))
    statuses = [
        (429, b"rate limited"),
        (503, b"unavailable"),
        (200, json.dumps(_TIINGO_ROWS).encode()),
    ]
    calls = iter(statuses)
    monkeypatch.setattr("trading.venues.equities._tiingo_get", lambda url, params: next(calls))
    adapter = EquitiesAdapter(_tiingo_config())
    df = adapter.fetch_ohlcv("XLNX", datetime.date(2022, 2, 11), datetime.date(2022, 2, 14))
    assert len(df) == 2
    assert slept == [1.0, 2.0]  # exponential backoff between the two transient failures


def test_tiingo_network_error_retried_then_raised(monkeypatch):
    import urllib.error

    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setattr("trading.venues.equities._tiingo_sleep", lambda s: None)

    def always_timeout(url, params):
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr("trading.venues.equities._tiingo_get", always_timeout)
    adapter = EquitiesAdapter(_tiingo_config())
    with pytest.raises(urllib.error.URLError):
        adapter.fetch_ohlcv("AAPL", datetime.date(2022, 1, 1), datetime.date(2022, 2, 1))


def test_tiingo_fetch_resolves_renamed_ticker_to_successor(monkeypatch):
    # A renamed PIT ticker (ABC = AmerisourceBergen) must fetch the successor
    # (COR = Cencora) Tiingo serves the continuous history under -- the literal
    # ABC would return the recycled Adbri (ASX) or nothing.
    import json

    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    captured = {}

    def fake_get(url, params):
        captured["url"] = url
        return 200, json.dumps(_TIINGO_ROWS).encode()

    monkeypatch.setattr("trading.venues.equities._tiingo_get", fake_get)
    adapter = EquitiesAdapter(_tiingo_config())
    adapter.fetch_ohlcv("ABC", datetime.date(2022, 2, 11), datetime.date(2022, 2, 14))
    assert "/COR/" in captured["url"]  # successor requested
    assert "/ABC/" not in captured["url"]  # NOT the literal recycled ticker


def test_tiingo_fetch_applies_namespace_override(monkeypatch):
    # MMC (US Marsh & McLennan) must fetch MRSH: Tiingo's bare MMC is the ASX
    # Mitre Mining squatter with no US daily bars.
    import json

    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    captured = {}

    def fake_get(url, params):
        captured["url"] = url
        return 200, json.dumps(_TIINGO_ROWS).encode()

    monkeypatch.setattr("trading.venues.equities._tiingo_get", fake_get)
    adapter = EquitiesAdapter(_tiingo_config())
    adapter.fetch_ohlcv("MMC", datetime.date(2022, 2, 11), datetime.date(2022, 2, 14))
    assert "/MRSH/" in captured["url"]


def test_tiingo_fetch_is_noop_for_current_ticker(monkeypatch):
    import json

    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    captured = {}

    def fake_get(url, params):
        captured["url"] = url
        return 200, json.dumps(_TIINGO_ROWS).encode()

    monkeypatch.setattr("trading.venues.equities._tiingo_get", fake_get)
    adapter = EquitiesAdapter(_tiingo_config())
    adapter.fetch_ohlcv("AAPL", datetime.date(2022, 2, 11), datetime.date(2022, 2, 14))
    assert "/AAPL/" in captured["url"]


def test_yfinance_fetch_does_not_resolve_renamed_ticker(monkeypatch):
    # Resolution is gated to tiingo: the yfinance path must request the literal
    # symbol (current members already use current tickers there, and the
    # MMC->MRSH override would be WRONG on yfinance).
    captured = {}

    def fake_yf(symbol, start, end):
        captured["symbol"] = symbol
        return _yf_style_frame(symbol)

    monkeypatch.setattr("trading.venues.equities._yf_download", fake_yf)
    adapter = EquitiesAdapter(CONFIG)  # default bar_source = yfinance
    adapter.fetch_ohlcv("ABC", datetime.date(2026, 1, 5), datetime.date(2026, 1, 9))
    assert captured["symbol"] == "ABC"  # NOT resolved to COR


def test_tiingo_persistent_429_surfaces_as_datafetcherror(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setattr("trading.venues.equities._tiingo_sleep", lambda s: None)
    monkeypatch.setattr(
        "trading.venues.equities._tiingo_get", lambda url, params: (429, b"slow down")
    )
    adapter = EquitiesAdapter(_tiingo_config())
    with pytest.raises(DataFetchError, match="HTTP 429"):
        adapter.fetch_ohlcv("AAPL", datetime.date(2022, 1, 1), datetime.date(2022, 2, 1))


def test_tiingo_malformed_config_raises_clean_error(monkeypatch, tmp_path):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    cfg = tmp_path / "config.toml"
    cfg.write_text("this is not = = valid toml [[[")
    monkeypatch.setattr("trading.venues.equities.TIINGO_CONFIG_PATH", cfg)
    from trading.venues.equities import tiingo_token

    with pytest.raises(DataFetchError, match="not valid TOML"):
        tiingo_token()


def test_tiingo_persistent_429_raises_ratelimiterror(monkeypatch):
    from trading.venues.base import RateLimitError

    monkeypatch.setenv("TIINGO_API_KEY", "test-token")
    monkeypatch.setattr("trading.venues.equities._tiingo_sleep", lambda s: None)
    monkeypatch.setattr(
        "trading.venues.equities._tiingo_get", lambda url, params: (429, b"over allocation")
    )
    adapter = EquitiesAdapter(_tiingo_config())
    with pytest.raises(RateLimitError):  # distinct from a plain DataFetchError 404-miss
        adapter.fetch_ohlcv("AAPL", datetime.date(2022, 1, 1), datetime.date(2022, 2, 1))


def _yf_adjusted_frame(symbol="AAPL"):
    idx = pd.date_range("2026-01-05", periods=3, freq="B")
    return pd.DataFrame(
        {
            ("Open", symbol): [10.0, 10.5, 10.2],
            ("High", symbol): [10.6, 10.9, 10.7],
            ("Low", symbol): [9.8, 10.1, 9.9],
            ("Close", symbol): [10.5, 10.2, 10.6],
            ("Volume", symbol): [1e6, 1.1e6, 9e5],
        },
        index=idx,
    )


def _yf_raw_actions_frame(symbol="AAPL"):
    # Raw (unadjusted) close differs from adjusted; a dividend and a 2:1 split.
    idx = pd.date_range("2026-01-05", periods=3, freq="B")
    return pd.DataFrame(
        {
            ("Open", symbol): [20.0, 21.0, 10.2],
            ("Close", symbol): [21.0, 20.4, 10.6],
            ("Dividends", symbol): [0.0, 0.22, 0.0],
            ("Stock Splits", symbol): [0.0, 0.0, 2.0],
        },
        index=idx,
    )


def test_merge_yf_extended_reads_raw_and_normalizes_splits():
    from trading.venues.equities import _merge_yf_extended

    # _merge_yf_extended runs BEFORE fetch_ohlcv's str.lower rename, so canonical
    # columns are still yfinance-capitalized here; extras are already lowercase.
    out = _merge_yf_extended(_yf_adjusted_frame(), _yf_raw_actions_frame())
    assert out["Close"].tolist() == [10.5, 10.2, 10.6]  # canonical untouched (adjusted)
    # Extended come from the RAW/actions frame, not the adjusted one.
    assert out["close_raw"].tolist() == [21.0, 20.4, 10.6]
    assert out["div_cash"].tolist() == [0.0, 0.22, 0.0]
    # 0.0 (no split) normalized to 1.0; a real 2.0 split preserved.
    assert out["split_factor"].tolist() == [1.0, 1.0, 2.0]


def test_merge_yf_extended_defaults_when_actions_call_failed():
    from trading.venues.equities import _merge_yf_extended

    # Second call raised -> raw is None. Canonical bars must still flow with
    # neutral extended defaults (close_raw falls back to the adjusted close).
    out = _merge_yf_extended(_yf_adjusted_frame(), None)
    assert out["Close"].tolist() == [10.5, 10.2, 10.6]
    assert out["close_raw"].tolist() == [10.5, 10.2, 10.6]
    assert (out["div_cash"] == 0.0).all()
    assert (out["split_factor"] == 1.0).all()


def test_merge_yf_extended_misaligned_dates_never_leave_nan():
    from trading.venues.equities import _merge_yf_extended

    adjusted = _yf_adjusted_frame()
    raw = _yf_raw_actions_frame().iloc[:2]  # raw missing the last trading day
    out = _merge_yf_extended(adjusted, raw)
    assert not out[["close_raw", "div_cash", "split_factor"]].isna().any().any()
    # The unmatched row defaults: close_raw<-adjusted close, div 0, split 1.
    assert out["close_raw"].iloc[-1] == 10.6
    assert out["div_cash"].iloc[-1] == 0.0
    assert out["split_factor"].iloc[-1] == 1.0


def test_yf_download_isolates_actions_call_failure(monkeypatch):
    # The nightly-run risk: the second (actions) yf.download raises. The first
    # call's canonical bars must survive with defaulted extras.
    import trading.venues.equities as eq

    calls = {"n": 0}

    def fake_download(symbol, **kwargs):
        calls["n"] += 1
        if kwargs.get("actions"):  # the second call
            raise RuntimeError("yahoo hiccup")
        return _yf_adjusted_frame(symbol)

    monkeypatch.setattr("yfinance.download", fake_download, raising=False)
    import yfinance  # noqa: F401  (ensure the module exists to patch)

    out = eq._yf_download("AAPL", datetime.date(2026, 1, 5), datetime.date(2026, 1, 8))
    assert list(out.columns[:5]) == ["Open", "High", "Low", "Close", "Volume"]
    assert (out["div_cash"] == 0.0).all()
    assert (out["split_factor"] == 1.0).all()
    assert calls["n"] == 2  # both calls attempted; the failing one was absorbed
