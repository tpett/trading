"""THE no-look-ahead guarantee (spec section 7): perturb every store strictly
after a cutoff date T and assert every registered signal's scores at <= T are
bit-identical. Iterates SIGNALS, so any future signal is automatically
covered -- a new signal that peeks past as_of fails here by construction."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from alphasearch_helpers import make_panel
from trading.alphasearch.panel import PanelData
from trading.alphasearch.spec import SIGNALS

START = pd.Timestamp("2020-01-01", tz="UTC")
CUTOFF = pd.Timestamp("2020-03-15", tz="UTC")


def _perturb_after(panel: PanelData, cutoff: pd.Timestamp) -> PanelData:
    """Corrupt every data point strictly after cutoff, in every store."""
    closes: dict[str, pd.Series] = {}
    for sym, series in panel.closes.items():
        s = series.copy()
        late = s.index > cutoff
        s.loc[late] = s.loc[late] * 3.7 + 11.0
        closes[sym] = s
    options: dict[str, pd.DataFrame] = {}
    for sym, frame in panel.options.items():
        f = frame.copy()
        f.loc[f.index > cutoff] = 9.9
        options[sym] = f
    fundamentals: dict[str, pd.DataFrame] = {}
    for sym, frame in panel.fundamentals.items():
        f = frame.copy()
        late = f.index > cutoff
        f.loc[late] = f.loc[late] * 5.0 + 1.0
        fundamentals[sym] = f
    return PanelData(closes=closes, options=options, fundamentals=fundamentals,
                     symbols=panel.symbols, corrupt_cells=panel.corrupt_cells)


def test_fixture_actually_has_data_after_the_cutoff():
    # Guard against a vacuous test: every store must carry post-cutoff rows.
    panel = make_panel()
    assert any((s.index > CUTOFF).any() for s in panel.closes.values())
    assert any((f.index > CUTOFF).any() for f in panel.options.values())
    assert any((f.index > CUTOFF).any() for f in panel.fundamentals.values())


def test_no_registered_signal_can_see_past_as_of():
    assert SIGNALS, "registry unexpectedly empty"  # never pass vacuously
    panel = make_panel()
    dates = list(panel.decision_dates(START, CUTOFF))
    assert len(dates) >= 3  # several decision months at or before the cutoff
    perturbed = _perturb_after(panel, CUTOFF)
    for name, spec in sorted(SIGNALS.items()):
        for as_of in dates:
            before = spec.fn(panel.view(as_of), as_of)
            after = spec.fn(perturbed.view(as_of), as_of)
            pdt.assert_series_equal(
                before, after, check_exact=True,
                obj=f"{name} @ {as_of.date().isoformat()}",
            )
