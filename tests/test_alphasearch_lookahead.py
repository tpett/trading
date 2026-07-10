"""THE no-look-ahead guarantee (spec section 7): perturb every RAW store
strictly after a cutoff date T -- bars (all columns), options cells,
fundamentals rows, factor rows -- REASSEMBLE the panel (so derived state,
incl. the precomputed ivol/beta features, is recomputed from the corrupted
inputs), and assert every registered signal's scores at <= T are
bit-identical. Iterates SIGNALS, so any future signal is automatically
covered; the anti-vacuity guard proves each signal actually produces values
pre-cutoff on this fixture."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pandas.testing as pdt

from alphasearch_helpers import assemble_panel, make_factors, make_panel
from trading.alphasearch.panel import PanelData
from trading.alphasearch.spec import SIGNALS

START = pd.Timestamp("2020-01-01", tz="UTC")
CUTOFF = pd.Timestamp("2020-03-15", tz="UTC")


def _add_straddling_insider_row(
    insider: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """make_panel's insider rows file on the SAME month-first cadence as the
    decision dates with only a 20-day trans-to-filed lag: by the time a
    post-CUTOFF filing's transaction happens, the last pre-CUTOFF decision
    date (2020-03-02 on this fixture) has already passed, so no as_of this
    suite ever scores falls inside a trans_date..filed gap -- a trans_date-
    keyed mutant would agree with the real code everywhere and go
    undetected. Add one extra row per symbol -- filed a few days after
    CUTOFF, transacted three weeks before it -- so the gap is genuinely live
    at that last pre-cutoff decision date."""
    filed = CUTOFF + pd.Timedelta(3, unit="D")
    trans = CUTOFF - pd.Timedelta(20, unit="D")
    out: dict[str, pd.DataFrame] = {}
    for sym, frame in insider.items():
        extra = pd.DataFrame(
            {"trans_date": [trans], "code": ["P"], "shares": [321.0],
             "price": [10.0], "value": [3210.0], "owner_cik": [777],
             "is_officer": [True], "is_director": [False], "is_ten_pct": [False]},
            index=pd.DatetimeIndex([filed], name="filed"),
        )
        out[sym] = pd.concat([frame, extra]).sort_index(kind="mergesort")
    return out


def _long_panel() -> PanelData:
    """~420 bars from 2019-01-02: enough pre-cutoff history that every
    registered signal (incl. beta's 126-obs floor, mom_12_2's 253 closes,
    and the 300-day YoY filing rule) produces real values at decision dates
    <= CUTOFF. Insider rows also get one straddling row per symbol so the
    trans_date..filed gap is live (see _add_straddling_insider_row)."""
    panel = make_panel(
        start="2019-01-02", periods=420,
        factors=make_factors(start="2018-12-03", periods=440),
    )
    return dataclasses.replace(
        panel, insider=_add_straddling_insider_row(panel.insider)
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
    insider: dict[str, pd.DataFrame] = {}
    for sym, frame in panel.insider.items():
        f = frame.copy()
        late = f.index > cutoff
        # Corrupt every reader-visible field on post-cutoff rows: value (npr_90's
        # ratio / officer_buy_90's numerator), owner_cik (cluster_buys_90's
        # distinct-owner count), code (both P/S filters), is_officer (flipped),
        # and trans_date (shifted). Mixed dtypes rule out a bars-style blanket
        # `* k + c` for all fields.
        f.loc[late, "value"] = f.loc[late, "value"] * 5.0 + 1.0
        f.loc[late, "owner_cik"] = -1
        f.loc[late, "code"] = f.loc[late, "code"].map({"P": "S", "S": "P"})
        f.loc[late, "is_officer"] = ~f.loc[late, "is_officer"]
        f.loc[late, "trans_date"] = f.loc[late, "trans_date"] + pd.Timedelta(25, unit="D")
        # ...and shift the FILED dates themselves (the PIT key). Several of
        # these rows have trans_date <= cutoff (the make_panel fixture files
        # 20 days after the transaction, and _add_straddling_insider_row adds
        # one further out): a signal keyed on TRANS_DATE instead of FILED --
        # the classic Form 4 look-ahead -- would see these rows move and
        # change its pre-cutoff scores.
        f.index = f.index.where(f.index <= cutoff, f.index + pd.Timedelta(30, unit="D"))
        insider[sym] = f
    factors = panel.factors.copy()
    late = factors.index > cutoff
    factors.loc[late] = factors.loc[late] * 3.0 + 0.001
    return assemble_panel(
        bars, options, fundamentals, factors, insider=insider,
        has_option_volume=panel.has_option_volume, sectors=panel.sectors,
    )


def test_fixture_actually_has_data_after_the_cutoff():
    # Guard against a vacuous test: every store must carry post-cutoff rows.
    panel = _long_panel()
    assert any((f.index > CUTOFF).any() for f in panel.bars.values())
    assert any((f.index > CUTOFF).any() for f in panel.options.values())
    assert any((f.index > CUTOFF).any() for f in panel.fundamentals.values())
    assert any((f.index > CUTOFF).any() for f in panel.insider.values())
    assert (panel.factors.index > CUTOFF).any()
    # The trans_date-keying trap must exist: rows FILED after the cutoff
    # whose TRANSACTION predates it (Form 4's 2-day+ filing lag).
    assert any(
        ((f.index > CUTOFF) & (f["trans_date"] <= CUTOFF)).any()
        for f in panel.insider.values()
    )
    # Stronger, and the actual anti-vacuity condition the mutation check
    # relies on: the gap must be LIVE at a decision date this suite actually
    # scores (<= CUTOFF), not merely relative to CUTOFF itself. Without this,
    # a trans_date-keyed mutant agrees with the real FILED-keyed code at
    # every as_of the perturbation test iterates, and that test would pass
    # vacuously even with a broken PIT key.
    dates = list(panel.decision_dates(START, CUTOFF))
    assert any(
        ((f.index > CUTOFF) & (f["trans_date"] <= dates[-1])).any()
        for f in panel.insider.values()
    ), "no as_of <= CUTOFF falls inside a trans_date..filed gap"


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


def test_insider_rows_filed_after_as_of_are_invisible_even_when_transacted_before():
    # A row transacted BEFORE as_of but FILED after it does not exist yet:
    # only the filing makes it public (spec: scoring keys FILED only).
    as_of = pd.Timestamp("2020-03-02", tz="UTC")
    frame = pd.DataFrame(
        {"trans_date": [pd.Timestamp("2020-02-28", tz="UTC")], "code": ["P"],
         "shares": [100.0], "price": [10.0], "value": [1000.0],
         "owner_cik": [1], "is_officer": [True], "is_director": [False],
         "is_ten_pct": [False]},
        index=pd.DatetimeIndex([pd.Timestamp("2020-03-04", tz="UTC")], name="filed"),
    )
    panel = PanelData(closes={}, insider={"AAA": frame}, symbols=("AAA",))
    assert panel.view(as_of).insider_window("AAA") is None   # not even "covered"
    after = panel.view(pd.Timestamp("2020-03-04", tz="UTC")).insider_window("AAA")
    assert after is not None and len(after) == 1             # visible once FILED
