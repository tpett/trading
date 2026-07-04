from trading.earnings import fetch_earnings_dates


def test_fetch_collects_iso_dates(monkeypatch):
    import datetime

    monkeypatch.setattr(
        "trading.earnings._yf_earnings_dates",
        lambda symbol: [datetime.date(2026, 7, 3), datetime.date(2026, 10, 2)],
    )
    dates, degraded = fetch_earnings_dates(["AAPL", "MSFT"])
    assert degraded is False
    assert dates == {
        "AAPL": ("2026-07-03", "2026-10-02"),
        "MSFT": ("2026-07-03", "2026-10-02"),
    }


def test_per_symbol_failure_degrades_without_raising(monkeypatch):
    import datetime

    def flaky(symbol: str):
        if symbol == "MSFT":
            raise RuntimeError("yfinance flaked")
        return [datetime.date(2026, 7, 3)]

    monkeypatch.setattr("trading.earnings._yf_earnings_dates", flaky)
    dates, degraded = fetch_earnings_dates(["AAPL", "MSFT"])
    assert degraded is True
    assert dates == {"AAPL": ("2026-07-03",)}  # MSFT absent = unfiltered
