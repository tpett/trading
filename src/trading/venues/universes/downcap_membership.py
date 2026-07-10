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


def _coalesce(month_rows: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """(band, symbol, month_iso) rows in date order -> coalesced
    (band, symbol, start, end) intervals. A break in contiguous months OR a
    band change starts a new interval. `end` is the month-start of the first
    month NO LONGER in that (band) -- EXCLUSIVE -- or "" when in-band through
    the final month processed."""
    # index months so contiguity is "adjacent in the ordered month list"
    all_months = sorted({m for _, _, m in month_rows})
    pos = {m: i for i, m in enumerate(all_months)}
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
                out.append((cur_band, symbol, cur_start, month))  # exclusive end
            cur_band, cur_start, prev_pos = band, month, pos[month]
        if cur_band is not None:
            out.append((cur_band, symbol, cur_start, ""))  # open: in-band through the end
    return out


def build_band_membership(
    roster: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    store: FundamentalsStore,
    *,
    discovery_window: str = DISCOVERY_WINDOW,
    require_cap_band: bool = True,
) -> MembershipBuild:
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

    membership = pd.DataFrame(_coalesce(month_rows), columns=MEMBERSHIP_COLUMNS)
    membership = membership.sort_values(["band", "symbol", "start"]).reset_index(drop=True)
    diagnostics = pd.DataFrame(diag_rows, columns=DIAGNOSTICS_COLUMNS)
    return MembershipBuild(membership=membership, diagnostics=diagnostics)


def write_membership(
    build: MembershipBuild, membership_path: Path, diagnostics_path: Path
) -> None:
    membership_path.parent.mkdir(parents=True, exist_ok=True)
    build.membership.to_csv(membership_path, index=False)
    build.diagnostics.to_csv(diagnostics_path, index=False)
