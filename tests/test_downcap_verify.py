import pandas as pd
import pytest

from trading.venues.universes import downcap_verify as dv
from trading.venues.universes.downcap_membership import DIAGNOSTICS_COLUMNS


def _diag(rows):
    return pd.DataFrame(rows, columns=DIAGNOSTICS_COLUMNS)


def _row(date, symbol, band, *, delisted=False, tradeable=True, has_shares=True,
         spread=0.01, dv=1_000_000.0, cap=200_000_000.0):
    return {
        "date": date, "symbol": symbol, "delisted": delisted, "tradeable": tradeable,
        "has_shares": has_shares, "band": band, "spread": spread,
        "dollar_volume": dv, "market_cap": cap,
    }


def test_go_when_all_thresholds_met():
    rows = []
    for m in ("2019-01-01", "2019-02-01"):
        for i in range(20):                       # 20 in-band names/month, >= 15
            rows.append(_row(m, f"MIC{i}", "micro", delisted=(i < 4)))  # 20% delisted
    gate = dv.compute_gate(_diag(rows))
    assert gate.survivorship_ok is True           # 20% >= 15%
    assert gate.shares_coverage_ok is True        # all tradeable have shares
    assert gate.breadth_ok is True
    assert gate.fallback_triggered is False
    assert gate.go is True


def test_low_shares_coverage_triggers_fallback():
    rows = []
    for m in ("2019-01-01", "2019-02-01"):
        for i in range(20):
            has = i < 12                          # 12/20 = 60% < 70%
            band = "micro" if has else None
            rows.append(_row(m, f"S{i}", band, has_shares=has, delisted=(i < 3)))
    gate = dv.compute_gate(_diag(rows))
    assert gate.shares_coverage_pct == pytest.approx(0.60)
    assert gate.shares_coverage_ok is False
    assert gate.fallback_triggered is True        # developer pre-approved dv-only path
    assert "dollar-volume-only" in dv.render_amendment(gate)


def test_sub_15_breadth_month_fails_universe():
    rows = []
    # January: 20 micro names (ok). February: only 10 micro (sub-15).
    for i in range(20):
        rows.append(_row("2019-01-01", f"MIC{i}", "micro"))
    for i in range(10):
        rows.append(_row("2019-02-01", f"MIC{i}", "micro"))
    gate = dv.compute_gate(_diag(rows))
    micro = next(b for b in gate.breadth if b.name == "downcap:micro")
    assert micro.min_month_count == 10
    assert micro.ok is False
    assert gate.breadth_ok is False               # a universe with a sub-15 month
    assert gate.go is False


def test_spread_distribution_reported():
    rows = [_row("2019-01-01", f"M{i}", "micro", spread=s)
            for i, s in enumerate([0.005, 0.01, 0.015, 0.02, 0.05])]
    # only 4/5 <= 2% (0.05 excluded from in-band would normally not appear, but
    # here we force it into the band rows to exercise the distribution math)
    gate = dv.compute_gate(_diag(rows))
    assert gate.spread_median == pytest.approx(0.015)
    assert 0.0 <= gate.spread_pct_le_2 <= 1.0


def test_render_report_states_go_and_metrics():
    rows = [_row("2019-01-01", f"MIC{i}", "micro", delisted=(i < 4)) for i in range(20)]
    text = dv.render_report(dv.compute_gate(_diag(rows)))
    assert "GO" in text or "NO-GO" in text
    assert "survivorship" in text.lower()
    assert "shares-coverage" in text.lower()
    assert "breadth" in text.lower()


def test_breadth_excludes_untradeable_in_band_rows():
    # Real A4 output can never have band set on an untradeable row (band is
    # non-None only when has_shares AND tradeable -- see downcap_band.evaluate_band),
    # but the gate must not rely on that invariant: an untradeable-yet-banded
    # row must not count toward the breadth denominator per the A5 spec's
    # "tradeable in-band names" breadth definition.
    rows = []
    for i in range(20):
        rows.append(_row("2019-01-01", f"MIC{i}", "micro", tradeable=(i < 10)))
    gate = dv.compute_gate(_diag(rows))
    micro = next(b for b in gate.breadth if b.name == "downcap:micro")
    assert micro.min_month_count == 10             # only the 10 tradeable rows count
    assert micro.ok is False
    assert gate.breadth_ok is False
