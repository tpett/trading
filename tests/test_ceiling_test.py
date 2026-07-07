"""Offline, synthetic tests for the CEILING TEST diagnostic.

Covers the terminal-failure detector (flags a delisted-with-decline name;
does NOT flag an acquired flat/up exit or a still-trading survivor) and the
wrapping adapter's universe() filtering + method passthrough. No network, no
real cache: every bar frame and the inner adapter are synthetic.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ceiling_test import (  # noqa: E402
    EXCLUSION_WINDOW_DAYS,
    TRAILING_RETURN_DAYS,
    HindsightFilterAdapter,
    TerminalFailure,
    detect_terminal_failure,
    exclusion_windows,
)

from trading.venues.base import SymbolInfo  # noqa: E402

END = datetime.date(2025, 12, 31)
# Benchmark trading-day calendar up to `end`, as the script builds it.
SESSIONS = pd.date_range("2016-01-04", END, freq="B", tz="UTC")


def _frame(last_day: str, n: int, plateau: float, last_close: float) -> pd.DataFrame:
    """Synthetic OHLCV: `n` business days ending on `last_day`. The close holds
    flat at `plateau` and then ramps to `last_close` over the final
    TRAILING_RETURN_DAYS+1 bars, so the trailing-126-bar return is EXACTLY
    last_close/plateau - 1, independent of `n`."""
    idx = pd.bdate_range(end=last_day, periods=n, tz="UTC")
    ramp_len = TRAILING_RETURN_DAYS + 1
    flat = [plateau] * (n - ramp_len)
    ramp = [plateau + (last_close - plateau) * i / (ramp_len - 1) for i in range(ramp_len)]
    close = pd.Series(flat + ramp, index=idx)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


def test_flags_delisted_name_with_six_month_decline():
    # Delisted mid-2021 (far more than 30 trading days before END) and down ~50%
    # over its trailing 126 bars: a terminal failure.
    frame = _frame("2021-06-01", 400, 100.0, 50.0)
    failure = detect_terminal_failure("DEAD", frame, SESSIONS, -0.20)
    assert failure is not None
    assert failure.symbol == "DEAD"
    assert failure.last_bar == datetime.date(2021, 6, 1)
    assert failure.six_month_return < -0.20
    # Exclusion window opens EXCLUSION_WINDOW_DAYS trading rows before the last bar.
    expected_start = frame.index[len(frame) - 1 - EXCLUSION_WINDOW_DAYS].date()
    assert failure.window_start == expected_start


def test_does_not_flag_acquired_name_exiting_flat_or_up():
    # Same delisting gap, but exits UP (takeover premium): not distress.
    frame = _frame("2021-06-01", 400, 100.0, 112.0)
    assert detect_terminal_failure("BOUGHT", frame, SESSIONS, -0.20) is None


def test_does_not_flag_survivor_running_to_backtest_end():
    # Bars run to END even though the price fell hard: still trading, so the
    # last-bar gap is within tolerance -> not a terminal failure.
    frame = _frame("2025-12-31", 400, 100.0, 40.0)
    assert detect_terminal_failure("ALIVE", frame, SESSIONS, -0.20) is None


def test_looser_distress_threshold_is_more_permissive():
    # A -30% decline is a failure at -20% but NOT at the looser -40% cutoff.
    frame = _frame("2021-06-01", 400, 100.0, 70.0)
    assert detect_terminal_failure("MILD", frame, SESSIONS, -0.20) is not None
    assert detect_terminal_failure("MILD", frame, SESSIONS, -0.40) is None


class _FakeInnerAdapter:
    """Minimal stand-in for EquitiesAdapter: a fixed universe plus a couple of
    non-universe methods the wrapper must pass through untouched."""

    def __init__(self, symbols: list[str]):
        self._symbols = symbols
        self.fetch_calls: list[str] = []

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        return [SymbolInfo(symbol=s, status="tradable") for s in self._symbols]

    def fetch_ohlcv(self, symbol, start, end):
        self.fetch_calls.append(symbol)
        return f"bars:{symbol}"

    def membership_intervals(self, symbol):
        return [("2018-01-01", "")]


def _windowed_adapter():
    inner = _FakeInnerAdapter(["AAA", "FAIL", "CCC"])
    failures = {
        "FAIL": TerminalFailure(
            symbol="FAIL",
            last_bar=datetime.date(2021, 6, 1),
            six_month_return=-0.5,
            window_start=datetime.date(2020, 6, 1),
        )
    }
    return inner, HindsightFilterAdapter(inner, exclusion_windows(failures))


def test_universe_excludes_failed_symbol_inside_its_window():
    _, adapter = _windowed_adapter()
    inside = {i.symbol for i in adapter.universe(datetime.date(2021, 1, 4))}
    assert inside == {"AAA", "CCC"}  # FAIL removed during its decline


def test_universe_includes_failed_symbol_before_its_window():
    _, adapter = _windowed_adapter()
    before = {i.symbol for i in adapter.universe(datetime.date(2020, 1, 2))}
    assert before == {"AAA", "FAIL", "CCC"}  # participates normally pre-window


def test_universe_includes_failed_symbol_after_its_last_bar():
    _, adapter = _windowed_adapter()
    after = {i.symbol for i in adapter.universe(datetime.date(2021, 7, 1))}
    assert "FAIL" in after  # window closes at the last bar


def test_non_universe_methods_pass_through_unchanged():
    inner, adapter = _windowed_adapter()
    assert adapter.fetch_ohlcv("FAIL", None, None) == "bars:FAIL"
    assert inner.fetch_calls == ["FAIL"]  # delegated to the real adapter
    assert adapter.membership_intervals("FAIL") == [("2018-01-01", "")]
