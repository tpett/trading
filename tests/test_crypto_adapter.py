import datetime
from pathlib import Path

import pandas as pd
import pytest

from trading.config import load_venue_config
from trading.venues.base import OHLCV_COLUMNS, DataFetchError, SymbolInfo, VenueConstraints
from trading.venues.crypto import DEFAULT_UNIVERSE_CSV, CryptoAdapter

CONFIG = load_venue_config("crypto", Path("config"))


def _kraken_rows(n: int, end: datetime.date) -> list[list[float]]:
    """Mimic ccxt fetch_ohlcv: [ms_timestamp, open, high, low, close, volume] rows."""
    start_ts = pd.Timestamp(end, tz="UTC") - pd.Timedelta(n - 1, unit="D")
    return [
        [
            int((start_ts + pd.Timedelta(i, unit="D")).timestamp() * 1000),
            100.0 + i,
            101.0 + i,
            99.0 + i,
            100.5 + i,
            1000.0,
        ]
        for i in range(n)
    ]


def test_universe_reads_symbols_and_statuses(tmp_path):
    csv = tmp_path / "universe.csv"
    csv.write_text("symbol,status\nBTC,tradable\nETH,tradable\nSOL,sell_only\n")
    adapter = CryptoAdapter(CONFIG, universe_csv=csv)
    infos = adapter.universe(datetime.date(2026, 7, 1))
    assert infos == [
        SymbolInfo("BTC", "tradable"),
        SymbolInfo("ETH", "tradable"),
        SymbolInfo("SOL", "sell_only"),
    ]


def test_universe_rejects_unknown_status(tmp_path):
    csv = tmp_path / "universe.csv"
    csv.write_text("symbol,status\nBTC,halted\n")
    adapter = CryptoAdapter(CONFIG, universe_csv=csv)
    with pytest.raises(ValueError, match="halted"):
        adapter.universe(datetime.date(2026, 7, 1))


def test_committed_universe_csv_is_valid():
    adapter = CryptoAdapter(CONFIG, universe_csv=DEFAULT_UNIVERSE_CSV)
    infos = adapter.universe(datetime.date(2026, 7, 1))
    assert len(infos) >= 80
    assert SymbolInfo("BTC", "tradable") in infos


def test_constraints_come_from_config():
    adapter = CryptoAdapter(CONFIG)
    assert adapter.constraints() == VenueConstraints(
        taker_fee_bps=95.0,
        maker_fee_bps=50.0,
        slippage_bps=5.0,
        settlement_days=0,
        trades_24_7=True,
    )


def test_fetch_ohlcv_maps_symbol_to_kraken_usd_pair(monkeypatch):
    seen: list[str] = []

    def fake_fetch(pair: str, since_ms: int) -> list[list[float]]:
        seen.append(pair)
        return _kraken_rows(10, datetime.date(2026, 7, 1))

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_fetch)
    adapter = CryptoAdapter(CONFIG)
    df = adapter.fetch_ohlcv("BTC", datetime.date(2026, 6, 22), datetime.date(2026, 7, 1))
    assert seen == ["BTC/USD"]
    assert list(df.columns) == OHLCV_COLUMNS
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert len(df) == 10
    assert df["close"].iloc[-1] == 109.5


def test_fetch_ohlcv_slices_to_requested_range(monkeypatch):
    monkeypatch.setattr(
        "trading.venues.crypto._kraken_fetch",
        lambda pair, since_ms: _kraken_rows(10, datetime.date(2026, 7, 1)),
    )
    adapter = CryptoAdapter(CONFIG)
    df = adapter.fetch_ohlcv("BTC", datetime.date(2026, 6, 25), datetime.date(2026, 6, 30))
    assert df.index.min() == pd.Timestamp("2026-06-25", tz="UTC")
    assert df.index.max() == pd.Timestamp("2026-06-30", tz="UTC")


def test_fetch_ohlcv_empty_raises(monkeypatch):
    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", lambda pair, since_ms: [])
    adapter = CryptoAdapter(CONFIG)
    with pytest.raises(DataFetchError):
        adapter.fetch_ohlcv("BTC", datetime.date(2026, 6, 22), datetime.date(2026, 7, 1))


def test_fetch_ohlcv_paginates_over_page_limit(monkeypatch):
    """Ranges longer than one Kraken page are stitched from multiple fetches."""
    all_rows = _kraken_rows(10, datetime.date(2026, 7, 1))
    day_ms = 86_400_000
    calls: list[int] = []

    def fake_fetch(pair: str, since_ms: int) -> list[list[float]]:
        calls.append(since_ms)
        # 4-row pages, overlapping one day back like a real exchange might.
        return [r for r in all_rows if r[0] >= since_ms - day_ms][:4]

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_fetch)
    adapter = CryptoAdapter(CONFIG)
    df = adapter.fetch_ohlcv("BTC", datetime.date(2026, 6, 22), datetime.date(2026, 7, 1))
    assert len(calls) > 1  # actually paginated
    assert len(df) == 10  # full range covered
    assert not df.index.duplicated().any()  # page overlap deduplicated
    assert df.index.min() == pd.Timestamp("2026-06-22", tz="UTC")
    assert df.index.max() == pd.Timestamp("2026-07-01", tz="UTC")


def test_fetch_ohlcv_wraps_ccxt_errors_as_data_fetch_error(monkeypatch):
    import ccxt

    def fake_fetch(pair: str, since_ms: int) -> list[list[float]]:
        raise ccxt.BadSymbol(f"kraken does not have market symbol {pair}")

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_fetch)
    adapter = CryptoAdapter(CONFIG)
    with pytest.raises(DataFetchError, match="NOPE/USD"):
        adapter.fetch_ohlcv("NOPE", datetime.date(2026, 6, 22), datetime.date(2026, 7, 1))


def _daily_rows(start: str, periods: int, price: float) -> list[list[float]]:
    base = pd.Timestamp(start, tz="UTC")
    return [
        [
            int((base + pd.Timedelta(i, unit="D")).timestamp() * 1000),
            price,
            price,
            price,
            price,
            1e6,
        ]
        for i in range(periods)
    ]


def test_deep_request_splices_backfill_and_kraken_with_kraken_precedence(monkeypatch):
    config = load_venue_config("crypto", Path("config"))
    # Kraken serves only from the boundary; backfill serves 2018 up to and
    # INCLUDING the boundary day at a different price -- Kraken must win it.
    boundary = datetime.date(2026, 7, 1) - datetime.timedelta(days=config.data.backfill_before_days)
    kraken_rows = _daily_rows(boundary.isoformat(), 30, price=200.0)
    deep_rows = _daily_rows(
        "2018-01-01", (boundary - datetime.date(2018, 1, 1)).days + 1, price=100.0
    )

    def fake_kraken(pair, since_ms):
        return [r for r in kraken_rows if r[0] >= since_ms][:720]

    def fake_backfill(exchange_id, pair, since_ms, limit):
        assert exchange_id == config.data.backfill_exchange
        return [r for r in deep_rows if r[0] >= since_ms][:limit]

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_kraken)
    monkeypatch.setattr("trading.venues.crypto._backfill_fetch", fake_backfill)
    adapter = CryptoAdapter(config)
    # end = 2026-07-01 so the adapter's internal boundary (end - backfill_before_days)
    # equals the `boundary` computed above.
    df = adapter.fetch_ohlcv("BTC", datetime.date(2018, 1, 1), datetime.date(2026, 7, 1))

    assert df.index[0] == pd.Timestamp("2018-01-01", tz="UTC")
    boundary_ts = pd.Timestamp(boundary, tz="UTC")
    assert float(df.loc[boundary_ts, "close"]) == 200.0  # Kraken wins the overlap
    assert float(df.loc[boundary_ts - pd.Timedelta(1, unit="D"), "close"]) == 100.0
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()


def test_recent_request_never_touches_backfill(monkeypatch):
    config = load_venue_config("crypto", Path("config"))
    kraken_rows = _daily_rows("2026-05-01", 62, price=200.0)

    def fake_kraken(pair, since_ms):
        return [r for r in kraken_rows if r[0] >= since_ms][:720]

    def forbidden(exchange_id, pair, since_ms, limit):
        raise AssertionError("backfill must not be called for a recent window")

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_kraken)
    monkeypatch.setattr("trading.venues.crypto._backfill_fetch", forbidden)
    adapter = CryptoAdapter(config)
    df = adapter.fetch_ohlcv("BTC", datetime.date(2026, 5, 1), datetime.date(2026, 7, 1))
    assert len(df) == 62


def test_backfill_pair_missing_falls_back_to_kraken_only(monkeypatch):
    import ccxt

    config = load_venue_config("crypto", Path("config"))
    kraken_rows = _daily_rows("2026-05-01", 62, price=200.0)

    def fake_kraken(pair, since_ms):
        return [r for r in kraken_rows if r[0] >= since_ms][:720]

    def missing_pair(exchange_id, pair, since_ms, limit):
        # Mirror _backfill_fetch's wrapping: BadSymbol preserved as __cause__.
        raise DataFetchError(f"{exchange_id} fetch failed for {pair}") from ccxt.BadSymbol(
            f"{exchange_id} does not have market symbol {pair}"
        )

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_kraken)
    monkeypatch.setattr("trading.venues.crypto._backfill_fetch", missing_pair)
    adapter = CryptoAdapter(config)
    # Deep request, but the pair only exists on Kraken: short history, no error.
    df = adapter.fetch_ohlcv("BTC", datetime.date(2018, 1, 1), datetime.date(2026, 7, 1))
    assert df.index[0] == pd.Timestamp("2026-05-01", tz="UTC")


def test_backfill_transient_failure_propagates(monkeypatch):
    """Only pair-not-listed downgrades to Kraken depth; a transient failure
    (network, rate limit) must fail the fetch rather than silently truncate
    a 2018 request to ~700 days."""
    import ccxt

    config = load_venue_config("crypto", Path("config"))
    kraken_rows = _daily_rows("2026-05-01", 62, price=200.0)

    def fake_kraken(pair, since_ms):
        return [r for r in kraken_rows if r[0] >= since_ms][:720]

    def transient(exchange_id, pair, since_ms, limit):
        raise DataFetchError(f"{exchange_id} fetch failed for {pair}: timeout") from (
            ccxt.NetworkError("timeout")
        )

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_kraken)
    monkeypatch.setattr("trading.venues.crypto._backfill_fetch", transient)
    adapter = CryptoAdapter(config)
    with pytest.raises(DataFetchError, match="timeout"):
        adapter.fetch_ohlcv("BTC", datetime.date(2018, 1, 1), datetime.date(2026, 7, 1))


def test_deep_request_seam_gap_raises(monkeypatch):
    """If the backfill history stops short of Kraken's window (retention
    shrinkage or config drift), the fetch must fail loudly, not hand signal
    computation a frame with a silent multi-day hole."""
    config = load_venue_config("crypto", Path("config"))
    boundary = datetime.date(2026, 7, 1) - datetime.timedelta(days=config.data.backfill_before_days)
    kraken_rows = _daily_rows(boundary.isoformat(), 30, price=200.0)
    # Backfill history ends 10 days before Kraken's first row.
    deep_end = boundary - datetime.timedelta(days=10)
    deep_rows = _daily_rows(
        "2018-01-01", (deep_end - datetime.date(2018, 1, 1)).days + 1, price=100.0
    )

    def fake_kraken(pair, since_ms):
        return [r for r in kraken_rows if r[0] >= since_ms][:720]

    def fake_backfill(exchange_id, pair, since_ms, limit):
        return [r for r in deep_rows if r[0] >= since_ms][:limit]

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_kraken)
    monkeypatch.setattr("trading.venues.crypto._backfill_fetch", fake_backfill)
    adapter = CryptoAdapter(config)
    with pytest.raises(DataFetchError, match="seam gap"):
        adapter.fetch_ohlcv("BTC", datetime.date(2018, 1, 1), datetime.date(2026, 7, 1))


def test_start_exactly_at_boundary_skips_backfill(monkeypatch):
    config = load_venue_config("crypto", Path("config"))
    end = datetime.date(2026, 7, 1)
    boundary = end - datetime.timedelta(days=config.data.backfill_before_days)
    kraken_rows = _daily_rows(
        boundary.isoformat(), config.data.backfill_before_days + 1, price=200.0
    )

    def fake_kraken(pair, since_ms):
        return [r for r in kraken_rows if r[0] >= since_ms][:720]

    def forbidden(exchange_id, pair, since_ms, limit):
        raise AssertionError("start == boundary must not trigger backfill")

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_kraken)
    monkeypatch.setattr("trading.venues.crypto._backfill_fetch", forbidden)
    adapter = CryptoAdapter(config)
    df = adapter.fetch_ohlcv("BTC", boundary, end)
    assert df.index[0] == pd.Timestamp(boundary, tz="UTC")
    assert len(df) == config.data.backfill_before_days + 1


def test_start_one_day_before_boundary_triggers_single_backfill_call(monkeypatch):
    config = load_venue_config("crypto", Path("config"))
    end = datetime.date(2026, 7, 1)
    boundary = end - datetime.timedelta(days=config.data.backfill_before_days)
    start = boundary - datetime.timedelta(days=1)
    kraken_rows = _daily_rows(
        boundary.isoformat(), config.data.backfill_before_days + 1, price=200.0
    )
    deep_rows = _daily_rows(start.isoformat(), 2, price=100.0)  # start day and boundary day
    backfill_calls: list[int] = []

    def fake_kraken(pair, since_ms):
        return [r for r in kraken_rows if r[0] >= since_ms][:720]

    def fake_backfill(exchange_id, pair, since_ms, limit):
        backfill_calls.append(since_ms)
        return [r for r in deep_rows if r[0] >= since_ms][:limit]

    monkeypatch.setattr("trading.venues.crypto._kraken_fetch", fake_kraken)
    monkeypatch.setattr("trading.venues.crypto._backfill_fetch", fake_backfill)
    adapter = CryptoAdapter(config)
    df = adapter.fetch_ohlcv("BTC", start, end)
    assert len(backfill_calls) == 1
    assert df.index[0] == pd.Timestamp(start, tz="UTC")
    assert float(df.loc[pd.Timestamp(start, tz="UTC"), "close"]) == 100.0
    assert float(df.loc[pd.Timestamp(boundary, tz="UTC"), "close"]) == 200.0  # Kraken wins
    assert not df.index.duplicated().any()
