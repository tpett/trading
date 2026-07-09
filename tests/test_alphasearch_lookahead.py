"""THE no-look-ahead guarantee (spec section 7): perturb every RAW store
strictly after a cutoff date T -- bars (all columns), options cells,
fundamentals rows, factor rows -- REASSEMBLE the panel (so derived state,
incl. the precomputed ivol/beta features, is recomputed from the corrupted
inputs), and assert every registered signal's scores at <= T are
bit-identical. Iterates SIGNALS, so any future signal is automatically
covered; the anti-vacuity guard proves each signal actually produces values
pre-cutoff on this fixture."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from alphasearch_helpers import assemble_panel, make_factors, make_panel
from trading.alphasearch.panel import PanelData
from trading.alphasearch.spec import SIGNALS

START = pd.Timestamp("2020-01-01", tz="UTC")
CUTOFF = pd.Timestamp("2020-03-15", tz="UTC")


def _long_panel() -> PanelData:
    """~420 bars from 2019-01-02: enough pre-cutoff history that every
    registered signal (incl. beta's 126-obs floor, mom_12_2's 253 closes,
    and the 300-day YoY filing rule) produces real values at decision dates
    <= CUTOFF."""
    return make_panel(
        start="2019-01-02", periods=420,
        factors=make_factors(start="2018-12-03", periods=440),
    )


def _perturb_after(panel: PanelData, cutoff: pd.Timestamp) -> PanelData:
    """Corrupt every raw store strictly after cutoff, then reassemble."""
    bars: dict[str, pd.DataFrame] = {}
    for sym, frame in panel.bars.items():
        f = frame.copy()
        late = f.index > cutoff
        f.loc[late] = f.loc[late] * 3.7 + 11.0
        bars[sym] = f
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
    factors = panel.factors.copy()
    late = factors.index > cutoff
    factors.loc[late] = factors.loc[late] * 3.0 + 0.001
    return assemble_panel(
        bars, options, fundamentals, factors,
        has_option_volume=panel.has_option_volume, sectors=panel.sectors,
    )


def test_fixture_actually_has_data_after_the_cutoff():
    # Guard against a vacuous test: every store must carry post-cutoff rows.
    panel = _long_panel()
    assert any((f.index > CUTOFF).any() for f in panel.bars.values())
    assert any((f.index > CUTOFF).any() for f in panel.options.values())
    assert any((f.index > CUTOFF).any() for f in panel.fundamentals.values())
    assert (panel.factors.index > CUTOFF).any()


def test_every_signal_scores_real_values_at_the_last_pre_cutoff_date():
    # Anti-vacuity: an all-NaN signal would "pass" the perturbation test
    # without testing anything. Auto-extends as families register.
    panel = _long_panel()
    dates = list(panel.decision_dates(START, CUTOFF))
    as_of = dates[-1]
    for name, spec in sorted(SIGNALS.items()):
        scores = spec.fn(panel.view(as_of), as_of)
        assert scores.notna().any(), f"{name} is all-NaN at {as_of.date()}"


def test_no_registered_signal_can_see_past_as_of():
    assert SIGNALS, "registry unexpectedly empty"  # never pass vacuously
    panel = _long_panel()
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
