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
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from trading.alphasearch.evaluate import TRADING_DAYS
from trading.fundamentals.store import FundamentalsStore
from trading.symbols import load_symbol_allowlist

logger = logging.getLogger(__name__)


class PanelError(ValueError):
    """Panel assembly refused (missing inputs, unusable universe)."""


MAX_OPTION_AGE_DAYS = 7  # a cell older than this at as_of is stale -> missing

OPTION_COLUMNS = [
    "hedge", "excite", "atm_iv", "otm_put_iv", "otm_call_iv",
    "smile", "cp_vol", "wing_vol", "tot_vol", "atm_spread", "opt_dollar_vol",
]


def cell_metrics(cell: dict) -> dict:
    """The option-derived metrics for one samples.jsonl cell (NaN when a leg is
    missing so a partial cell never fabricates a value).

    Promoted verbatim from scripts/signal_scan.py::_cell_metrics; the script
    now imports it from here so there is exactly one definition.
    """
    d = {c["role"]: c for c in cell.get("contracts", [])}
    # Leg volume ships only in the mid-cap gather; a largecap cell carries NO
    # "volume" key on ANY leg. key-absent means unmeasured, not zero trades --
    # a volume-less cell must never fabricate log(1/1)=0 (2026-07-09 fix).
    has_volume = any("volume" in c for c in d.values())

    def iv(role):
        return d.get(role, {}).get("iv")

    def vol(role):
        # A leg present without its OWN volume key (rare, partial gather) still
        # defaults to 0 once has_volume gates the cell as measured -- a
        # PRESENT volume of 0 on some other leg is a real observation.
        return d.get(role, {}).get("volume") or 0

    atm = d.get("atm", {})
    put_iv, call_iv, atm_iv = iv("otm_put"), iv("otm_call"), iv("atm")
    rr = cell.get("skew_put_call")
    spread = ((atm["ask"] - atm["bid"]) / atm["mid"]
              if atm.get("mid") and atm.get("bid") is not None and atm.get("ask") is not None
              else np.nan)

    def leg_dollar(role):
        contract = d.get(role, {})
        volume, mid = contract.get("volume"), contract.get("mid")
        if volume is None or mid is None:
            return None
        return float(volume) * 100.0 * float(mid)

    dollars = [x for x in (leg_dollar(r) for r in ("atm", "otm_put", "otm_call"))
               if x is not None]
    return {
        "hedge": cell.get("skew_put_atm"),
        "excite": (-rr if rr is not None else np.nan),  # call-vs-put IV richness
        "atm_iv": atm_iv,
        "otm_put_iv": put_iv,
        "otm_call_iv": call_iv,
        "smile": ((put_iv + call_iv) / 2 - atm_iv
                  if None not in (put_iv, call_iv, atm_iv) else np.nan),
        "cp_vol": (np.log((vol("atm") + vol("otm_call") + 1) / (vol("otm_put") + 1))
                   if has_volume else np.nan),
        "wing_vol": (np.log((vol("otm_call") + 1) / (vol("otm_put") + 1))
                     if has_volume else np.nan),
        "tot_vol": (vol("atm") + vol("otm_put") + vol("otm_call")) if has_volume else np.nan,
        "atm_spread": spread,
        # Johnson-So O/S numerator: only legs carrying BOTH volume and mid
        # count; a cell with neither is missing, never $0.
        "opt_dollar_vol": (sum(dollars) if dollars else np.nan),
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


MAX_PRIOR_OPTION_AGE_DAYS = 45  # spec section 2: iv_change/dskew staleness cap
MIN_YOY_AGE_DAYS = 300  # spec section 2: the YoY filing rule
INSIDER_WINDOW_DAYS = 90  # insider spec section 3: trailing filed-date window


def cells_have_volume(cells: Iterable[dict]) -> bool:
    """True when ANY gathered leg carries a volume field. Leg volume ships
    only in the mid-cap gather; the sweep refuses the option-volume family
    on universes without it (a volume-less cell would otherwise score a
    fabricated log(1/1)=0, not NaN)."""
    return any(
        contract.get("volume") is not None
        for cell in cells
        for contract in cell.get("contracts", [])
    )


def load_options(samples: Path) -> tuple[dict[str, pd.DataFrame], int, bool]:
    """Parse a samples.jsonl into per-symbol metric frames, plus a corrupt-
    line count and whether any gathered cell carries leg volume.

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
    return options_from_cells(cells), corrupt, cells_have_volume(cells)


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


INSIDER_SOURCE_MARKER = ".source"


def load_insider(root: Path, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Per-symbol Form 4 insider frames (FILED-date index, the store schema
    scripts/build_insider_store.py writes) for symbols that have any.
    Returns {} without creating anything when the store dir is absent --
    assembly must never invent an empty store (the fundamentals rule).

    Warns ONCE per call when the store's .source marker carries a
    "GAPS:<quarters>" suffix (scripts/build_insider_store.py stamps this when
    a quarterly download/parse failed): the store is still usable, but an
    incomplete build must announce itself rather than silently under-count
    insider activity."""
    if not root.exists():
        return {}
    marker = root / INSIDER_SOURCE_MARKER
    if marker.exists():
        text = marker.read_text().strip()
        _, _, gaps = text.partition("GAPS:")
        if gaps:
            logger.warning(
                "insider store %s is missing coverage for quarter(s): %s "
                "(incomplete build -- see scripts/build_insider_store.py)",
                root, gaps,
            )
    out: dict[str, pd.DataFrame] = {}
    for symbol in sorted(set(symbols)):
        path = root / f"{symbol.replace('/', '-')}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
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


# Full bar schema served to signals. Legacy largecap caches predate the
# extended columns; load_bars NaN-fills them (see docstring) rather than
# fabricating "no dividends / no splits". close_raw (the RAW, unadjusted
# close) is div_yield's divisor (2026-07-09 fix): Tiingo's `close` is
# split+dividend adjusted using the FULL downloaded history, so it bakes in
# corporate actions that happen AFTER any given as-of date -- close_raw never
# does.
BAR_COLUMNS = [
    "open", "high", "low", "close", "volume", "div_cash", "split_factor", "close_raw",
]


def load_bars(cache_dir: Path, symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Full bar frames (BAR_COLUMNS) per symbol from a Tiingo parquet cache.

    A symbol without a cached parquet is absent from the result (missing-data
    rule, spec section 5.5). A legacy narrow cache (OHLCV only) gets NaN
    div_cash/split_factor/close_raw -- NOT the venue layer's 0.0/1.0/adjusted-
    close migration defaults: a cache that never stored these cannot claim
    "no dividends" / "no splits" / a raw-price basis, so div_yield/
    net_issuance go honestly NaN there instead of scoring fabricated or
    look-ahead-contaminated values.
    """
    out: dict[str, pd.DataFrame] = {}
    for symbol in sorted(set(symbols)):
        path = cache_dir / f"{symbol}.parquet"
        if path.exists():
            out[symbol] = pd.read_parquet(path).reindex(columns=BAR_COLUMNS)
    return out


IVOL_WINDOW = 21
IVOL_MIN_OBS = 15   # spec section 6: 15-obs floor for 21d windows
BETA_WINDOW = 252
BETA_MIN_OBS = 126  # spec section 6: 126-obs floor for 252d windows
ROLLING_FEATURES = ["ivol", "beta"]
_FF3 = ["Mkt-RF", "SMB", "HML"]


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing-window sums via cumsum differences: row t sums rows
    max(0, t-window+1)..t. Works for any trailing shape, so the same helper
    rolls scalars, design rows, and stacked 4x4 cross-product matrices."""
    cum = np.cumsum(values, axis=0, dtype="float64")
    out = cum.copy()
    out[window:] = cum[window:] - cum[:-window]
    return out


def _ivol_column(y: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """Rolling FF3 residual std, annualized (spec section 2 `ivol`).

    Per row t: OLS of the trailing <=IVOL_WINDOW excess returns on
    [1, Mkt-RF, SMB, HML]; ivol = sqrt(SSE / (n - 4)) * sqrt(TRADING_DAYS)
    -- the OLS residual standard error, the same e'e/(n-k) convention as
    evaluate.ols. NaN below IVOL_MIN_OBS obs, and NaN for a numerically
    singular window (a rank-deficient window has no defined FF3 residual:
    missing data, spec section 6 -- real factor history is never
    rank-deficient over 15+ days). Fully vectorized: rolling cross-product
    sums, one batched det, one batched solve.
    """
    n = len(y)
    k = 4
    design = np.column_stack([np.ones(n), x3])
    xtx = _rolling_sum(design[:, :, None] * design[:, None, :], IVOL_WINDOW)
    xty = _rolling_sum(design * y[:, None], IVOL_WINDOW)
    yty = _rolling_sum(y * y, IVOL_WINDOW)
    counts = np.minimum(np.arange(n) + 1, IVOL_WINDOW)
    solvable = (counts >= IVOL_MIN_OBS) & (np.abs(np.linalg.det(xtx)) > 0.0)
    # Identity-substitute the unsolvable stacks: batched solve raises on ANY
    # singular member (the first 14 windows always are), and masked rows are
    # overwritten with NaN below anyway.
    safe = np.where(solvable[:, None, None], xtx, np.eye(k))
    beta = np.linalg.solve(safe, xty[:, :, None])[:, :, 0]
    sse = np.maximum(yty - np.einsum("nk,nk->n", beta, xty), 0.0)
    dof = np.maximum(counts - k, 1)
    out = np.sqrt(sse / dof) * math.sqrt(TRADING_DAYS)
    out[~solvable] = np.nan
    return out


def _beta_column(y: np.ndarray, mkt: np.ndarray) -> np.ndarray:
    """Rolling OLS slope of excess returns on Mkt-RF with intercept (spec
    section 2 `beta`): cov(mkt, y) / var(mkt) over the trailing
    <=BETA_WINDOW rows; NaN below BETA_MIN_OBS obs or when the window's
    market variance is zero."""
    n = len(y)
    counts = np.minimum(np.arange(n) + 1, BETA_WINDOW).astype("float64")
    sx = _rolling_sum(mkt, BETA_WINDOW)
    sy = _rolling_sum(y, BETA_WINDOW)
    sxy = _rolling_sum(mkt * y, BETA_WINDOW)
    sxx = _rolling_sum(mkt * mkt, BETA_WINDOW)
    denom = counts * sxx - sx * sx
    out = np.full(n, np.nan)
    valid = (counts >= BETA_MIN_OBS) & (denom > 0)
    out[valid] = (counts * sxy - sx * sy)[valid] / denom[valid]
    return out


def compute_rolling_features(
    closes: dict[str, pd.Series], factors: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Per-symbol full-span ivol/beta frames (the FeaturePanel pattern from
    trading.signals.engine: precompute once, gather as-of). Rows live on the
    inner join of the symbol's pct_change() calendar with the factor
    calendar (NaN rows dropped); every rolling sum looks strictly backward,
    so the value gathered at as_of is identical whether or not data after
    as_of exists -- the no-look-ahead perturbation test proves it. ivol
    regresses EXCESS returns (ret - RF) on FF3; beta regresses them on
    Mkt-RF alone."""
    if factors.empty:
        return {}
    cols = factors[[*_FF3, "RF"]]
    out: dict[str, pd.DataFrame] = {}
    for symbol, series in closes.items():
        rets = series.pct_change().rename("ret")
        joined = cols.join(rets, how="inner").dropna()
        if joined.empty:
            continue
        y = (joined["ret"] - joined["RF"]).to_numpy()
        x3 = joined[_FF3].to_numpy()
        out[symbol] = pd.DataFrame(
            {"ivol": _ivol_column(y, x3), "beta": _beta_column(y, x3[:, 0])},
            index=joined.index,
        )
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

    @property
    def has_option_volume(self) -> bool:
        return self._panel.has_option_volume

    def sector(self, symbol: str) -> str | None:
        return self._panel.sectors.get(symbol)

    def option_row_prior(
        self, symbol: str, max_age_days: int = MAX_PRIOR_OPTION_AGE_DAYS
    ) -> pd.Series | None:
        """The most recent cell strictly OLDER than the current cell (the
        one option_row's position arithmetic selects; spec section 3.5),
        never a future one. None when there is no current cell, no older
        cell, or the older cell is more than max_age_days calendar days
        before as_of -- staleness measured from as_of, mirroring
        MAX_OPTION_AGE_DAYS. The 45-day default freezes iv_change/dskew's
        'prior month' definition (spec section 2)."""
        frame = self._panel.options.get(symbol)
        if frame is None or frame.empty:
            return None
        pos = int(frame.index.searchsorted(self.as_of, side="right")) - 1
        if pos < 1:
            return None
        if (self.as_of - frame.index[pos - 1]).days > max_age_days:
            return None
        return frame.iloc[pos - 1]

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

    def fundamentals_row_prior(
        self, symbol: str, min_age_days: int = MIN_YOY_AGE_DAYS
    ) -> pd.Series | None:
        """The 'one-year-prior filing' of the YoY rule (spec section 2): the
        latest row FILED at least min_age_days calendar days before the
        CURRENT filing's filed date (current = fundamentals_row's cut).
        Exactly-min_age_days qualifies. None when no current filing is
        visible or no filing is old enough -- YoY signals then go NaN,
        dropped, never imputed."""
        frame = self._panel.fundamentals.get(symbol)
        if frame is None or frame.empty:
            return None
        window = frame.loc[: self.as_of]
        if window.empty:
            return None
        current_filed = window.index[-1]
        cutoff = current_filed - pd.Timedelta(min_age_days, unit="D")
        prior = window.loc[:cutoff]
        return None if prior.empty else prior.iloc[-1]

    def insider_window(
        self, symbol: str, days: int = INSIDER_WINDOW_DAYS
    ) -> pd.DataFrame | None:
        """Form 4 rows FILED in (as_of - days, as_of] -- calendar days, PIT
        by FILING date (TRANS_DATE precedes filing and must never key
        anything). None when the symbol has NO insider row filed at or
        before as_of (never-covered: the signal-table NaN case); an EMPTY
        frame means covered-but-quiet, a real observation (cluster_buys_90
        scores it 0, never NaN)."""
        frame = self._panel.insider.get(symbol)
        if frame is None or frame.empty:
            return None
        visible = frame.loc[: self.as_of]
        if visible.empty:
            return None
        return visible[visible.index > self.as_of - pd.Timedelta(days, unit="D")]

    def bars(self, symbol: str) -> pd.DataFrame:
        """The symbol's bars (BAR_COLUMNS) up to and including as_of; an
        empty BAR_COLUMNS frame when the symbol has none."""
        frame = self._panel.bars.get(symbol)
        if frame is None:
            return pd.DataFrame(columns=BAR_COLUMNS, dtype="float64")
        pos = frame.index.searchsorted(self.as_of, side="right")
        return frame.iloc[:pos]

    def factors(self) -> pd.DataFrame:
        """Factor rows dated at or before as_of (empty frame when the panel
        carries no factors)."""
        frame = self._panel.factors
        if frame.empty:
            return frame
        pos = frame.index.searchsorted(self.as_of, side="right")
        return frame.iloc[:pos]

    def feature(self, symbol: str, column: str) -> float:
        """Precomputed rolling feature (ROLLING_FEATURES) at the last row
        dated at or before as_of; NaN when the symbol has no feature rows
        yet. Same searchsorted gather as FeaturePanel.gather."""
        frame = self._panel.features.get(symbol)
        if frame is None or frame.empty:
            return math.nan
        pos = int(frame.index.searchsorted(self.as_of, side="right")) - 1
        if pos < 0:
            return math.nan
        return float(frame.iloc[pos][column])


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
    # symbol -> Form 4 open-market transactions (filed-date index, the
    # data/insider/equities store schema). Absent store -> {} -> the
    # requires_insider refusal at sweep assembly.
    insider: dict[str, pd.DataFrame] = field(default_factory=dict)
    symbols: tuple[str, ...] = ()
    corrupt_cells: int = 0
    bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    # Daily Ken French factor frame (Mkt-RF, SMB, HML, RF[, Mom]; decimals,
    # UTC index) from evaluate.load_factors. Published history: PIT holds by
    # truncation like every other store (spec section 3, item 1).
    factors: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Precomputed heavy rolling features (ROLLING_FEATURES columns), keyed by
    # symbol, indexed by the joint return/factor calendar.
    features: dict[str, pd.DataFrame] = field(default_factory=dict)
    # True when ANY gathered options cell carries leg volume (mid-cap gather).
    # Drives the requires_option_volume assembly-time refusal (sweep.py).
    has_option_volume: bool = False
    # symbol -> frozen SEGMENTS sector (the 10-way partition); absent =
    # unmapped, and industry-relative signals then score NaN, never a guess.
    sectors: dict[str, str] = field(default_factory=dict)

    def view(self, as_of: pd.Timestamp) -> PanelView:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be tz-aware UTC")
        return PanelView(self, as_of)

    def decision_dates(
        self, start: pd.Timestamp, end: pd.Timestamp, offset: int = 0
    ) -> tuple[pd.Timestamp, ...]:
        """The (offset+1)-th trading session of each month in [start, end]
        (offset=0, the default, is the first session -- Piece 1 behavior,
        bit-identical). offset=1 is Piece 3's decision-date-offset battery
        check. A month with too few in-window sessions is dropped, never
        approximated with a different session.

        "Trading session" = any date on which at least one panel symbol has a
        bar (the union calendar), so one symbol's missing day never shifts the
        whole universe's rebalance date.
        """
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        union = sorted({d for s in self.closes.values() for d in s.index})
        by_month: dict[str, list[pd.Timestamp]] = {}
        for date in union:
            if start <= date <= end:
                by_month.setdefault(date.strftime("%Y-%m"), []).append(date)
        return tuple(
            by_month[m][offset] for m in sorted(by_month) if len(by_month[m]) > offset
        )


def build_panel(
    cache_dir: Path,
    samples: Path | None,
    fundamentals_dir: Path | None,
    *,
    insider_dir: Path | None = None,
    symbols: tuple[str, ...] | None = None,
    factors: pd.DataFrame | None = None,
    sectors: dict[str, str] | None = None,
) -> PanelData:
    """Assemble one universe's PanelData.

    `insider_dir` mirrors `fundamentals_dir` (absent/None -> empty dict ->
    `requires_insider` refusal at sweep assembly).

    Two universe sources (Piece 2 spec section 3.3):

    * samples allowlist (Piece 1): the gathered options pool -- the
      samples.jsonl allowlist intersected with the symbols that have cached
      bars, so every signal family within a universe is measured on the
      identical cross-section.
    * explicit ``symbols`` (segments): the caller supplies the universe
      outright, overriding the allowlist derivation. ``samples`` may then be
      None -- no option frames are loaded, so options signals are refused by
      the sweep's existing assembly-time check -- or a path, in which case
      option cells are loaded and restricted to the explicit universe.

    An explicitly-empty symbols tuple is refused (mirrors the empty-signals
    refusal): sweeping a universe with no names is a caller bug, never a
    silent no-trade run. `symbols is None` checks throughout -- `symbols or`
    would conflate empty-and-refuse with absent-and-derive.
    """
    if symbols is not None and len(symbols) == 0:
        raise PanelError(
            "explicit symbols tuple is empty: a universe with no names cannot "
            "trade (segment assembly should have excluded it)"
        )
    if symbols is None and samples is None:
        raise PanelError(
            "no universe source: pass a samples allowlist or an explicit symbols tuple"
        )
    if samples is not None and not samples.exists():
        raise PanelError(
            f"options samples not found: {samples} "
            "(Piece 1 universes are the gathered options pools)"
        )
    if symbols is not None:
        allowlist = sorted(symbols)
    else:
        allowlist = sorted(load_symbol_allowlist(samples))
    bars = load_bars(cache_dir, allowlist)
    if not bars:
        raise PanelError(f"no bar caches under {cache_dir} for the requested universe")
    closes = {s: frame["close"] for s, frame in bars.items()}
    options, corrupt, has_volume = (
        load_options(samples) if samples is not None else ({}, 0, False)
    )
    fundamentals = (
        load_fundamentals(fundamentals_dir, closes) if fundamentals_dir is not None else {}
    )
    insider = load_insider(insider_dir, closes) if insider_dir is not None else {}
    universe = tuple(s for s in allowlist if s in bars)
    features = compute_rolling_features(closes, factors) if factors is not None else {}
    return PanelData(
        closes={s: closes[s] for s in universe},
        options={s: options[s] for s in universe if s in options},
        fundamentals={s: fundamentals[s] for s in universe if s in fundamentals},
        insider={s: insider[s] for s in universe if s in insider},
        symbols=universe,
        corrupt_cells=corrupt,
        bars={s: bars[s] for s in universe},
        factors=pd.DataFrame() if factors is None else factors,
        features={s: features[s] for s in universe if s in features},
        has_option_volume=has_volume,
        sectors={} if sectors is None else {s: sectors[s] for s in universe if s in sectors},
    )
