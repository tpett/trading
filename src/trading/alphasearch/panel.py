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

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from trading.fundamentals.store import FundamentalsStore
from trading.symbols import load_symbol_allowlist


class PanelError(ValueError):
    """Panel assembly refused (missing inputs, unusable universe)."""


MAX_OPTION_AGE_DAYS = 7  # a cell older than this at as_of is stale -> missing

OPTION_COLUMNS = [
    "hedge", "excite", "atm_iv", "otm_put_iv", "otm_call_iv",
    "smile", "cp_vol", "wing_vol", "tot_vol", "atm_spread",
]


def cell_metrics(cell: dict) -> dict:
    """The option-derived metrics for one samples.jsonl cell (NaN when a leg is
    missing so a partial cell never fabricates a value).

    Promoted verbatim from scripts/signal_scan.py::_cell_metrics; the script
    now imports it from here so there is exactly one definition.
    """
    d = {c["role"]: c for c in cell.get("contracts", [])}

    def iv(role):
        return d.get(role, {}).get("iv")

    def vol(role):
        return d.get(role, {}).get("volume") or 0

    atm = d.get("atm", {})
    put_iv, call_iv, atm_iv = iv("otm_put"), iv("otm_call"), iv("atm")
    rr = cell.get("skew_put_call")
    spread = ((atm["ask"] - atm["bid"]) / atm["mid"]
              if atm.get("mid") and atm.get("bid") is not None and atm.get("ask") is not None
              else np.nan)
    return {
        "hedge": cell.get("skew_put_atm"),
        "excite": (-rr if rr is not None else np.nan),  # call-vs-put IV richness
        "atm_iv": atm_iv,
        "otm_put_iv": put_iv,
        "otm_call_iv": call_iv,
        "smile": ((put_iv + call_iv) / 2 - atm_iv
                  if None not in (put_iv, call_iv, atm_iv) else np.nan),
        "cp_vol": np.log((vol("atm") + vol("otm_call") + 1) / (vol("otm_put") + 1)),
        "wing_vol": np.log((vol("otm_call") + 1) / (vol("otm_put") + 1)),
        "tot_vol": vol("atm") + vol("otm_put") + vol("otm_call"),
        "atm_spread": spread,
    }


def options_from_cells(cells: Iterable[dict]) -> dict[str, pd.DataFrame]:
    """Per-symbol metric frames indexed by UTC decision_date.

    astype(float64) turns the None a missing leg leaves into NaN instead of an
    object column; a duplicated (symbol, date) keeps the LAST gathered cell --
    deduped in gather order via a per-date dict BEFORE building the frame,
    because sort_index's default quicksort is unstable, so sort-then-
    duplicated(keep="last") would keep an arbitrary duplicate.
    """
    rows: dict[str, dict[pd.Timestamp, dict]] = {}
    for cell in cells:
        date = pd.Timestamp(cell["decision_date"], tz="UTC")
        rows.setdefault(cell["symbol"], {})[date] = {"date": date, **cell_metrics(cell)}
    frames: dict[str, pd.DataFrame] = {}
    for symbol, by_date in rows.items():
        frame = pd.DataFrame(list(by_date.values())).set_index("date").sort_index()
        frames[symbol] = frame[OPTION_COLUMNS].astype("float64")
    return frames


def load_options(samples: Path) -> tuple[dict[str, pd.DataFrame], int]:
    """Parse a samples.jsonl into per-symbol metric frames.

    An unparseable line or one without symbol/decision_date is SKIPPED and
    COUNTED (spec section 6: corrupt cells never fabricate data, and the count
    is surfaced by the sweep so coverage loss is visible).
    """
    cells: list[dict] = []
    corrupt = 0
    for line in samples.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cell = json.loads(line)
        except ValueError:
            corrupt += 1
            continue
        if not isinstance(cell, dict) or not cell.get("symbol") or not cell.get("decision_date"):
            corrupt += 1
            continue
        cells.append(cell)
    return options_from_cells(cells), corrupt


def load_fundamentals(root: Path, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Per-symbol fundamentals frames (FILED-date index) for symbols that have
    any. Returns {} without creating anything when the store dir is absent --
    assembly must never invent an empty store."""
    if not root.exists():
        return {}
    store = FundamentalsStore(root)
    out: dict[str, pd.DataFrame] = {}
    for symbol in sorted(set(symbols)):
        frame = store.read(symbol)
        if not frame.empty:
            out[symbol] = frame
    return out


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

    def option_row(self, symbol: str) -> pd.Series | None:
        """Latest options cell with decision_date <= as_of, or None.

        A cell more than MAX_OPTION_AGE_DAYS calendar days old is STALE and
        returns None: option state is a snapshot, and forward-filling a
        months-old cell across the decision boundary would fabricate data
        (spec section 5.5)."""
        frame = self._panel.options.get(symbol)
        if frame is None or frame.empty:
            return None
        pos = int(frame.index.searchsorted(self.as_of, side="right")) - 1
        if pos < 0:
            return None
        if (self.as_of - frame.index[pos]).days > MAX_OPTION_AGE_DAYS:
            return None
        return frame.iloc[pos]

    def fundamentals_row(self, symbol: str) -> pd.Series | None:
        """Latest fundamentals row FILED at or before as_of (the step function
        on FILING dates -- same cut as trading.signals.quality.latest_filed_row),
        or None when nothing is visible yet."""
        frame = self._panel.fundamentals.get(symbol)
        if frame is None or frame.empty:
            return None
        window = frame.loc[: self.as_of]
        return None if window.empty else window.iloc[-1]


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


def build_panel(
    cache_dir: Path, samples: Path, fundamentals_dir: Path | None
) -> PanelData:
    """Assemble one universe's PanelData.

    The universe is the gathered options pool: the samples.jsonl allowlist
    (spec section 3.2) intersected with the symbols that have cached bars, so
    every signal family within a universe is measured on the identical
    cross-section (missing per-signal inputs are handled per-date by the NaN
    rule, never by widening the pool).
    """
    if not samples.exists():
        raise PanelError(
            f"options samples not found: {samples} "
            "(Piece 1 universes are the gathered options pools)"
        )
    allowlist = sorted(load_symbol_allowlist(samples))
    closes = load_closes(cache_dir, allowlist)
    if not closes:
        raise PanelError(f"no bar caches under {cache_dir} for the {samples.name} allowlist")
    options, corrupt = load_options(samples)
    fundamentals = (
        load_fundamentals(fundamentals_dir, closes) if fundamentals_dir is not None else {}
    )
    symbols = tuple(s for s in allowlist if s in closes)
    return PanelData(
        closes={s: closes[s] for s in symbols},
        options={s: options[s] for s in symbols if s in options},
        fundamentals={s: fundamentals[s] for s in symbols if s in fundamentals},
        symbols=symbols,
        corrupt_cells=corrupt,
    )
