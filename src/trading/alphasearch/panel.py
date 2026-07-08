"""Point-in-time panel assembly for the alpha-search engine (spec section 3.2).

Generalizes scripts/signal_scan.py::load_panel: Tiingo parquet bar caches,
options samples.jsonl cells (Task 5), and the fundamentals store (Task 5),
unified behind one PIT discipline. Signal scoring and forward-return
construction are SEPARATE passes over this data: a signal fn only ever
receives a PanelView, whose every accessor truncates at as_of via
index.searchsorted(side="right") (the repo-wide convention -- see
trading.signals.engine.FeaturePanel.gather), so a signal structurally cannot
reach forward data. The sort (trading.alphasearch.sort) reads panel.closes
directly for forward returns -- signals never see that pass.

Pure I/O + indexing: no clock reads; as_of is always a parameter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


class PanelError(ValueError):
    """Panel assembly refused (missing inputs, unusable universe)."""


def load_closes(cache_dir: Path, symbols: Iterable[str]) -> dict[str, pd.Series]:
    """Adjusted-close series per symbol from a Tiingo parquet cache.

    A symbol without a cached parquet is simply absent from the result --
    the missing-data rule (spec section 5.5) drops it from every
    cross-section rather than fabricating history.
    """
    out: dict[str, pd.Series] = {}
    for symbol in sorted(set(symbols)):
        path = cache_dir / f"{symbol}.parquet"
        if path.exists():
            out[symbol] = pd.read_parquet(path)["close"]
    return out


class PanelView:
    """Read-only as-of window onto a PanelData.

    Every accessor truncates to data timestamped at or before as_of. Signal
    functions receive ONLY this view (never the PanelData), which is what
    makes the no-look-ahead guarantee structural rather than by-convention.
    """

    def __init__(self, panel: PanelData, as_of: pd.Timestamp) -> None:
        self._panel = panel
        self.as_of = as_of

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._panel.symbols

    def closes(self, symbol: str) -> pd.Series:
        """The symbol's closes up to and including as_of (empty if none)."""
        series = self._panel.closes.get(symbol)
        if series is None:
            return pd.Series(dtype="float64")
        pos = series.index.searchsorted(self.as_of, side="right")
        return series.iloc[:pos]

    def last_close(self, symbol: str) -> float:
        closes = self.closes(symbol)
        return float(closes.iloc[-1]) if len(closes) else float("nan")


@dataclass(frozen=True)
class PanelData:
    """One universe's data: full-span series keyed by symbol.

    Frames deliberately extend past any given decision date -- truncation is
    PanelView's job (identical to the ranker registry's contract for bars and
    fundamentals). `options` and `fundamentals` are populated in Task 5.
    """

    closes: dict[str, pd.Series]
    options: dict[str, pd.DataFrame] = field(default_factory=dict)
    fundamentals: dict[str, pd.DataFrame] = field(default_factory=dict)
    symbols: tuple[str, ...] = ()
    corrupt_cells: int = 0

    def view(self, as_of: pd.Timestamp) -> PanelView:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be tz-aware UTC")
        return PanelView(self, as_of)

    def decision_dates(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.Timestamp, ...]:
        """First trading session of each month in [start, end].

        "Trading session" = any date on which at least one panel symbol has a
        bar (the union calendar), so one symbol's missing day never shifts the
        whole universe's rebalance date.
        """
        union = sorted({d for s in self.closes.values() for d in s.index})
        in_window = [d for d in union if start <= d <= end]
        if not in_window:
            return ()
        firsts: dict[str, pd.Timestamp] = {}
        for date in in_window:  # ascending, so the first hit per month sticks
            firsts.setdefault(date.strftime("%Y-%m"), date)
        return tuple(firsts[m] for m in sorted(firsts))
