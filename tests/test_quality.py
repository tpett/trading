import pandas as pd

from trading.data.quality import CoverageReport, check_coverage, quarantine_outliers


def _bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=len(closes), freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1e6] * len(closes),
        },
        index=idx,
    )


def test_coverage_at_exactly_90_percent_is_ok():
    requested = [f"S{i}" for i in range(10)]
    report = check_coverage(requested, requested[:9], min_coverage=0.90)
    assert report == CoverageReport(requested=10, fetched=9, ratio=0.9, ok=True, missing=("S9",))


def test_coverage_below_90_percent_is_not_ok():
    requested = [f"S{i}" for i in range(10)]
    report = check_coverage(requested, requested[:8], min_coverage=0.90)
    assert report.ok is False
    assert report.missing == ("S8", "S9")


def test_coverage_with_empty_universe_is_not_ok():
    report = check_coverage([], [], min_coverage=0.90)
    assert report.ok is False


def test_quarantine_flags_moves_beyond_bound():
    # 100 -> 155 is a +55% day: quarantined at the 40% bound.
    bars = {"BAD": _bars([100.0, 155.0, 150.0]), "OK": _bars([100.0, 139.0, 140.0])}
    clean, quarantined = quarantine_outliers(bars, max_daily_move=0.40, quarantine_window_days=40)
    assert quarantined == ("BAD",)
    assert set(clean) == {"OK"}


def test_quarantine_flags_crashes_too():
    # 100 -> 55 is a -45% day.
    bars = {"CRASH": _bars([100.0, 55.0, 56.0])}
    clean, quarantined = quarantine_outliers(bars, max_daily_move=0.40, quarantine_window_days=40)
    assert quarantined == ("CRASH",)
    assert clean == {}


def test_quarantine_passes_moves_at_the_bound():
    bars = {"EDGE": _bars([100.0, 140.0, 141.0])}  # exactly +40%
    clean, quarantined = quarantine_outliers(bars, max_daily_move=0.40, quarantine_window_days=40)
    assert quarantined == ()
    assert set(clean) == {"EDGE"}


def _bars_with_spike_at(total_days: int, spike_calendar_days_ago: int) -> pd.DataFrame:
    """A long, calm daily-close series with one +70% spike-and-revert placed
    `spike_calendar_days_ago` calendar days before the frame's last bar."""
    idx = pd.date_range(end="2026-07-01", periods=total_days, freq="D", tz="UTC")
    closes = [100.0] * total_days
    spike_date = idx[-1] - pd.Timedelta(spike_calendar_days_ago, unit="D")
    spike_pos = idx.searchsorted(spike_date)
    closes[spike_pos] = 170.0
    if spike_pos + 1 < total_days:
        closes[spike_pos + 1] = 100.0  # revert the next day
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1e6] * total_days,
        },
        index=idx,
    )


def test_old_spike_outside_window_is_not_quarantined():
    # Spike sits 200 calendar days back; a 40-day window must not see it.
    bars = {"OLD_SPIKE": _bars_with_spike_at(total_days=480, spike_calendar_days_ago=200)}
    clean, quarantined = quarantine_outliers(bars, max_daily_move=0.40, quarantine_window_days=40)
    assert quarantined == ()
    assert set(clean) == {"OLD_SPIKE"}


def test_recent_spike_inside_window_is_quarantined():
    # Same shape, but the spike sits 5 calendar days back: inside a 40-day window.
    bars = {"RECENT_SPIKE": _bars_with_spike_at(total_days=480, spike_calendar_days_ago=5)}
    clean, quarantined = quarantine_outliers(bars, max_daily_move=0.40, quarantine_window_days=40)
    assert quarantined == ("RECENT_SPIKE",)
    assert clean == {}
