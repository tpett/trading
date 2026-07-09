"""Illiquidity signal channel: ``amihud_v1`` adapts the battery-passed
alphasearch ``amihud`` signal (the FIRST BH-survivor discovery family, and the
only one HOLDOUT-ELIGIBLE after the 7-check robustness battery -- see
``docs/experiments.md`` section 11) to the full walk-forward backtester's
ranker contract.

WHY a thin adapter, not a reimplementation
-------------------------------------------
``amihud_lambda`` (``trading.alphasearch.spec``) is the audited construction:
mean ``|ret| / (close * volume)`` over the trailing 252 bars, minimum 126
valid terms, zero/negative-dollar-volume days filtered. This module imports
it rather than recomputing it, so the number this ranker backtests is
provably the SAME number that passed discovery and the battery -- any drift
between the alphasearch scan and this adapter would undermine that evidence
chain. Nothing here reimplements the formula; nothing in alphasearch or the
backtest engine changes to support it (purely additive).

WHY this ranker exists (docs/experiments.md section 11)
---------------------------------------------------------
amihud:midcap passed discovery on a COST-FREE cheap series and then the
robustness battery (both regime halves independently significant, tiny
turnover, first-order impact negligible at personal book sizes) -- but the
battery's cost/capacity checks are still simplified relative to the FULL
walk-forward engine's fee/slippage/regime/portfolio-construction machinery.
This ranker is the full cost-modeled tradability test that closes that gap.

WHY the sign is NOT negated
----------------------------
The tested hypothesis is long-illiquid: a HIGH amihud lambda (harder to
trade, more price impact per dollar) precedes HIGHER forward returns (the
classical illiquidity/neglect premium). So ``composite`` = cross-sectional
percentile of lambda directly -- unlike the skew rankers' negated composite,
the buy-side end here IS the most illiquid name, not the least.

NaN policy: EXCLUDED, never neutral (unlike every other ranker here)
-----------------------------------------------------------------------
skew_v1 / skew_change_v1 / illiquidity_veto_v1 all fail OPEN: a missing
feature gets the neutral 0.5 percentile so it neither buys nor shorts.
amihud_v1 deliberately does NOT follow that convention. A name with fewer
than 126 valid trailing terms (too short a listed history, or too many
zero/negative-dollar-volume days) has no reliable impact/capacity estimate --
and since this ranker's entire purpose is testing whether the illiquidity
premium survives REAL trading costs, silently neutralizing an
illiquid-in-a-different-way name would launder exactly the risk under test.
So a NaN lambda drops the symbol from the output entirely. This mirrors how
``compute_features`` itself drops (never fills) a symbol with too little
price history for the momentum features.

Purity: no I/O, no clock -- as_of is the only time input. Bars are truncated
to <= as_of here before ``amihud_lambda`` ever sees them, mirroring
``skew_v1`` and ``trading.alphasearch.panel.PanelView.bars``'s
``searchsorted(side="right")`` no-lookahead cut.
"""

from __future__ import annotations

import pandas as pd

from trading.alphasearch.spec import amihud_lambda
from trading.config import SignalConfig
from trading.signals.engine import FeaturePanel, compute_features

AMIHUD_V1_COLUMNS = ["amihud_lambda", "composite", "raw_return_30d"]


def _truncated_bars(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """``frame`` cut to rows dated at or before ``as_of``.

    ``side="right"`` matches ``PanelView.bars`` / ``FeaturePanel.gather``: a
    row dated exactly ``as_of`` is included (known by the decision), and
    nothing after ``as_of`` is ever visible to ``amihud_lambda``.
    """
    pos = frame.index.searchsorted(as_of, side="right")
    return frame.iloc[:pos]


def amihud_v1(
    bars: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    config: SignalConfig,
    fundamentals: dict[str, pd.DataFrame] | None = None,
    *,
    skew: dict[str, pd.DataFrame] | None = None,
    panel: FeaturePanel | None = None,
) -> pd.DataFrame:
    """Battery-passed alphasearch ``amihud`` signal, adapted to the backtester.

    ``composite`` = cross-sectional percentile of ``amihud_lambda`` (higher
    lambda = more illiquid = higher composite = bought first). The ranked
    universe starts from the momentum PRICE-history gate
    (``compute_features``) and is further narrowed to names carrying a valid
    trailing lambda -- see the module docstring for why a NaN lambda EXCLUDES
    rather than neutralizes. ``raw_return_30d`` comes from that same momentum
    base (the M2 fee gate reads it).
    """
    base = compute_features(bars, as_of, config, panel=panel)
    if base.empty:
        return pd.DataFrame(columns=AMIHUD_V1_COLUMNS, dtype="float64")
    raw = {symbol: amihud_lambda(_truncated_bars(bars[symbol], as_of)) for symbol in base.index}
    lam = pd.Series(raw, dtype="float64")
    out = pd.DataFrame(index=base.index)
    out["amihud_lambda"] = lam
    out["composite"] = lam.rank(pct=True)
    out["raw_return_30d"] = base["raw_return_30d"]
    out = out.loc[lam.notna()]
    return out[AMIHUD_V1_COLUMNS]
