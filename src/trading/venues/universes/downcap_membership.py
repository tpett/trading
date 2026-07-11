"""The dynamic PIT band-membership builder (R3 spec section 2/4). Runs the
band selector at the first trading session of every discovery month for every
roster candidate, then:

  * MEMBERSHIP artifact -- (band, symbol, start, end) intervals (end
    EXCLUSIVE, "" = open), the science input the UniverseSpecs consume. Its
    (band, symbol, start, end) shape deliberately mirrors
    equities_membership.csv so the same interval-overlap logic loads it.
  * DIAGNOSTICS artifact -- one row per (date, symbol) with the raw screen
    outcomes, the audit input the Phase-A gate (A5) reads for survivorship,
    shares-coverage, spread distribution, and breadth.

`require_cap_band=False` is the developer-pre-approved dollar-volume-only
fallback (spec section 4): membership = tradeable regardless of cap, single
band "downcap"."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading.alphasearch.sweep import DISCOVERY_WINDOW
from trading.fundamentals.store import FundamentalsStore
from trading.signals.quality import latest_filed_row
from trading.venues.universes.downcap_band import evaluate_band
from trading.venues.universes.downcap_roster import candidates_at, delisted_symbols

MEMBERSHIP_COLUMNS = ["band", "symbol", "start", "end"]
DIAGNOSTICS_COLUMNS = [
    "date", "symbol", "delisted", "tradeable", "has_shares",
    "band", "spread", "dollar_volume", "market_cap",
]
FALLBACK_BAND = "downcap"


def load_calendar(bars: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    """The union trading calendar: any date on which at least one symbol has a
    bar (matches PanelData.decision_dates' union-calendar rule)."""
    return sorted({d for frame in bars.values() for d in frame.index})


def monthly_decision_dates(
    calendar, start: pd.Timestamp, end: pd.Timestamp
) -> list[pd.Timestamp]:
    """First trading session of each month in [start, end] (matches
    PanelData.decision_dates offset=0)."""
    by_month: dict[str, list[pd.Timestamp]] = {}
    for date in sorted(calendar):
        if start <= date <= end:
            by_month.setdefault(date.strftime("%Y-%m"), []).append(date)
    return [by_month[m][0] for m in sorted(by_month)]


@dataclass
class MembershipBuild:
    membership: pd.DataFrame
    diagnostics: pd.DataFrame


def _coalesce(
    month_rows: list[tuple[str, str, str]], all_months: list[str]
) -> list[tuple[str, str, str, str]]:
    """(band, symbol, month_iso) rows in date order, plus `all_months` -- the
    FULL ordered monthly decision-date calendar for the discovery window
    (not just the months some symbol qualified in) -- -> coalesced
    (band, symbol, start, end) intervals.

    A run is a maximal sequence of CONSECUTIVE decision dates (consecutive in
    `all_months`, the real calendar) where the symbol is in the same band. A
    gap in the calendar OR a band change ends the run and starts a new one.
    `end` is the decision date immediately after the run's last in-band date
    in `all_months` -- EXCLUSIVE -- or "" ONLY when that last in-band date is
    the FINAL entry of `all_months` (still a member at window end). Because
    `end` is always resolved from the real calendar rather than from the next
    *qualifying* month, a permanent exit closes its interval instead of being
    left open, and a re-entry after a gap starts a genuinely separate
    interval."""
    pos = {m: i for i, m in enumerate(all_months)}

    def close(prev_pos: int) -> str:
        nxt = prev_pos + 1
        return all_months[nxt] if nxt < len(all_months) else ""

    per_symbol: dict[str, list[tuple[str, str]]] = {}
    for band, symbol, month in month_rows:
        per_symbol.setdefault(symbol, []).append((month, band))
    out: list[tuple[str, str, str, str]] = []
    for symbol, entries in per_symbol.items():
        entries.sort()
        cur_band = None
        cur_start = None
        prev_pos = None
        for month, band in entries:
            contiguous = prev_pos is not None and pos[month] == prev_pos + 1
            if cur_band == band and contiguous:
                prev_pos = pos[month]
                continue
            if cur_band is not None:
                out.append((cur_band, symbol, cur_start, close(prev_pos)))
            cur_band, cur_start, prev_pos = band, month, pos[month]
        if cur_band is not None:
            out.append((cur_band, symbol, cur_start, close(prev_pos)))
    return out


def build_band_membership(
    roster: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    store: FundamentalsStore,
    *,
    discovery_window: str = DISCOVERY_WINDOW,
    require_cap_band: bool = True,
) -> MembershipBuild:
    """WINDOW-SCOPED, do not reuse across windows: an interval's `end` is ""
    (open) when the symbol is still in-band at the FINAL decision date of
    `discovery_window`, and `load_band_membership`/the panel filter treat
    `end == ""` as in-band for ALL future dates -- not just through the
    window this build actually evaluated. A membership CSV built over the
    2019-2023 discovery window is therefore stale (look-ahead-adjacent) if
    ever queried for a later window, e.g. the 2024+ holdout: it would assert
    band membership for dates it never checked. Rebuild membership over the
    new window before reusing it for anything beyond the window it was built
    over."""
    start_s, _, end_s = discovery_window.partition("..")
    start = pd.Timestamp(start_s, tz="UTC")
    end = pd.Timestamp(end_s, tz="UTC")
    dates = monthly_decision_dates(load_calendar(bars), start, end)
    delisted = delisted_symbols(roster)

    diag_rows: list[dict] = []
    month_rows: list[tuple[str, str, str]] = []  # (band, symbol, month_iso)
    for d in dates:
        month_iso = d.date().isoformat()
        d_date = d.date()
        for symbol in sorted(candidates_at(roster, d_date)):
            frame = bars.get(symbol)
            if frame is None:
                continue                       # recorded gap: no bars, no row
            truncated = frame.loc[:d]          # PIT: strictly <= decision date
            if truncated.empty:
                continue
            filed = store.read(symbol)
            row = latest_filed_row(filed, d)
            shares = (
                float(row["shares_outstanding"])
                if row is not None and "shares_outstanding" in row.index
                else float("nan")
            )
            ev = evaluate_band(truncated, shares)
            diag_rows.append(
                {
                    "date": month_iso, "symbol": symbol,
                    "delisted": symbol in delisted,
                    "tradeable": ev.tradeable, "has_shares": ev.has_shares,
                    "band": ev.band, "spread": ev.spread,
                    "dollar_volume": ev.dollar_volume, "market_cap": ev.market_cap,
                }
            )
            if require_cap_band:
                if ev.band is not None:
                    month_rows.append((ev.band, symbol, month_iso))
            elif ev.tradeable:
                month_rows.append((FALLBACK_BAND, symbol, month_iso))

    all_months = [d.date().isoformat() for d in dates]
    membership = pd.DataFrame(
        _coalesce(month_rows, all_months), columns=MEMBERSHIP_COLUMNS
    )
    membership = membership.sort_values(["band", "symbol", "start"]).reset_index(drop=True)
    diagnostics = pd.DataFrame(diag_rows, columns=DIAGNOSTICS_COLUMNS)
    return MembershipBuild(membership=membership, diagnostics=diagnostics)


def load_band_membership(
    path: Path, bands: frozenset[str]
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Read the (band, symbol, start, end) CSV, keep only rows whose band is in
    `bands`, and return symbol -> tuple of (start_iso, end_iso) intervals (end
    EXCLUSIVE, "" = open) -- the interval shape PanelView.symbols filters on,
    mirroring equities_membership.csv's overlap logic.

    WINDOW-SCOPED: `end == ""` means "still in-band at the end of the
    discovery window the CSV was built over" (see build_band_membership), and
    this loader treats it as in-band for every date the caller later queries
    -- including dates past that window. Do not point this at a CSV built
    over a different (e.g. earlier/holdout) window without rebuilding it
    first."""
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["band"].isin(bands)]
    out: dict[str, list[tuple[str, str]]] = {}
    for row in df.itertuples():
        out.setdefault(row.symbol, []).append((row.start, row.end))
    return {s: tuple(sorted(iv)) for s, iv in out.items()}


_DOWNCAP_UNIVERSES = {
    "downcap": ("micro", "small"),
    "downcap:small": ("small",),
    "downcap:micro": ("micro",),
}


def downcap_universes(
    root: Path, *, membership_path: Path | None = None
) -> dict:
    """The three frozen down-cap UniverseSpecs (spec section 2). Each shares
    the fresh bars cache and the band-membership CSV; identity is the NAME
    (like segments). `samples=None` -> options signals refused; fundamentals
    attach when the store exists so the fundamentals family sweeps the band to
    the extent companyfacts covers it. Returns {} when the membership CSV is
    absent (nothing built yet) -- callers then simply omit these universes,
    mirroring segment_universes' absent-inputs behavior."""
    from trading.alphasearch.sweep import UniverseSpec  # lazy: avoids import cycle

    cache_dir = root / "data" / "equities-downcap-tiingo"
    if membership_path is None:
        membership_path = cache_dir / "band_membership.csv"
    if not membership_path.exists():
        return {}
    fundamentals_dir = root / "data" / "fundamentals" / "equities"
    specs: dict[str, UniverseSpec] = {}
    for name, bands in _DOWNCAP_UNIVERSES.items():
        members = load_band_membership(membership_path, frozenset(bands))
        symbols = tuple(sorted(members))
        if not symbols:
            continue  # no names in this band's CSV yet -> omit (never empty spec)
        specs[name] = UniverseSpec(
            name=name,
            cache_dir=cache_dir,
            samples=None,                               # options signals refused
            fundamentals_dir=fundamentals_dir if fundamentals_dir.is_dir() else None,
            symbols=symbols,                            # ever-in-band union (bar loading)
            membership_intervals=membership_path,       # per-date filter
            bands=bands,
        )
    return specs


def write_membership(
    build: MembershipBuild, membership_path: Path, diagnostics_path: Path
) -> None:
    """Writes the CSV pair as-is -- WINDOW-SCOPED to whatever `discovery_window`
    `build` was constructed over (see build_band_membership); the written
    membership CSV must be rebuilt, not reused, for a different window."""
    membership_path.parent.mkdir(parents=True, exist_ok=True)
    build.membership.to_csv(membership_path, index=False)
    build.diagnostics.to_csv(diagnostics_path, index=False)
