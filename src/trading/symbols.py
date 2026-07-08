"""Single source of truth for symbol resolution: ticker renames and vendor
namespace overrides.

Two distinct hazards this module resolves, both discovered live against Tiingo:

1. RENAMES -- a company changed ticker (or merged into a successor) and the
   vendor now serves its FULL continuous price history under the SUCCESSOR
   ticker, while the OLD ticker is either dead or RECYCLED to an unrelated
   company. Fetching the literal old ticker therefore returns nothing OR --
   the real danger -- a DIFFERENT company's bars (silent backtest
   contamination). resolve_current() follows the rename chain forward so the
   price-fetch path requests the successor and gets the real, continuous
   series. Verified live 2026-07-07: e.g. ABC now resolves to Adbri Ltd
   (ASX) while AmerisourceBergen's history lives under COR (Cencora, back to
   1995); CTL now resolves to Qwest while CenturyLink's history is under LUMN.

2. NAMESPACE_OVERRIDES -- NOT a rename: a CURRENT US ticker whose bare form
   collides with a foreign listing the vendor happens to rank first, with no
   point-in-time boundary and no chain. Vendor-specific (gated to tiingo in
   the equities adapter). Empty today -- the one candidate (MMC) turned out
   to be a genuine ticker change and moved to RENAMES -- but kept as the seam
   for a future genuine collision.

resolution_collisions() surfaces where the same company can enter the ranking
twice -- from cross-index membership double-listing (a few names even in the
default sp500+ndx universe) or ticker recycling (more under sp400). See that
function; it is a detector, not yet an auto-fix.

build_cik_map.py imports RENAMES from here so the fundamentals/CIK overlay
and the price-fetch path share one reviewed table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# (old_symbol, new_symbol, change_date). Reviewed ticker renames among 2017+
# membership symbols. change_date is the membership CSV's remove/add
# transition for the OLD ticker (its interval `end`): the boundary that
# decides which symbol a filing filed near it attaches to (fundamentals) and
# documents when the old ticker stopped being live. Chains (A->B->C) are
# supported and resolved to the terminal ticker by resolve_current().
RENAMES = [
    ("DWDP", "DD", "2019-06-03"),
    ("HCP", "PEAK", "2019-11-05"),
    ("HRS", "LHX", "2019-07-01"),
    ("UTX", "RTX", "2020-04-03"),
    ("MYL", "VTRS", "2020-11-16"),
    ("WLTW", "WTW", "2022-01-05"),
    ("FB", "META", "2022-06-09"),
    ("ANTM", "ELV", "2022-06-28"),
    ("FBHS", "FBIN", "2022-12-19"),
    ("PKI", "RVTY", "2023-05-16"),
    ("FISV", "FI", "2023-06-06"),
    ("RE", "EG", "2023-07-10"),
    ("ABC", "COR", "2023-08-30"),
    ("PEAK", "DOC", "2024-03-04"),
    ("MMC", "MRSH", "2026-01-14"),  # Marsh & McLennan ticker change; see MMC note below
    # --- Verified additions (2026-07-07): each successor was confirmed to
    # carry the OLD ticker's FULL continuous history on Tiingo (metadata
    # startDate well before the membership window), so resolving the old
    # ticker to it recovers the real series instead of a recycled-ticker
    # contaminant. change_date = the old ticker's membership-CSV interval end.
    ("ADS", "BFH", "2020-06-22"),  # Alliance Data Systems -> Bread Financial (BFH back to 2001)
    ("BLL", "BALL", "2022-05-10"),  # Ball Corp ticker change (BALL back to 1984)
    ("CBS", "PARA", "2019-12-05"),  # CBS Corp -> ViacomCBS/Paramount (PARA back to 2006); see note
    ("COG", "CTRA", "2021-10-04"),  # Cabot Oil & Gas -> Coterra (CTRA back to 1990)
    ("CTL", "LUMN", "2020-09-18"),  # CenturyLink -> Lumen (LUMN back to 1987)
    ("GPS", "GAP", "2022-02-02"),  # Gap Inc ticker change (GAP back to 1987)
    ("HFC", "DINO", "2021-06-04"),  # HollyFrontier -> HF Sinclair (DINO back to 1992)
    ("TMK", "GL", "2019-08-08"),  # Torchmark -> Globe Life (GL back to 1987)
    # CBS chain note: Paramount (PARA) merged with Skydance on 2025-08-06 and
    # the current ticker is PSKY -- but this is deliberately NOT chained
    # CBS->PARA->PSKY. Unlike the ABC->COR pattern, Tiingo did NOT carry the
    # continuous history forward under PSKY: PSKY is a FRESH listing that
    # returns EMPTY for every date before 2025-08-06, whereas PARA carries the
    # full 2006..2025-08-07 lineage covering BOTH the CBS (2017..2019) and PARA
    # (2022..2025) membership windows. Chaining to PSKY would redirect those
    # fetches to an empty series and zero out coverage. PSKY IS its own current
    # membership row (sp500 from 2025-08-08) and self-covers via Tiingo from
    # its listing date, so terminating the CBS chain at PARA -- and leaving
    # PSKY to resolve to itself -- covers every window correctly.
]

# MMC note: Marsh & McLennan is a genuine ticker change, MMC -> MRSH on
# 2026-01-14 (membership CSV: MMC ends and MRSH begins that day), and Tiingo
# serves Marsh's full history back to 1987 under MRSH -- the ABC->COR pattern,
# NOT a bare-symbol collision. It therefore lives in RENAMES above (so the
# fundamentals/CIK overlay chains it too, not just the price path) rather than
# here. Bare "MMC" on Tiingo does resolve to Mitre Mining Corp (ASX), but the
# rename entry supersedes that for the whole membership window.
#
# Vendor namespace collisions (NOT renames): a current US ticker whose bare
# form the vendor maps to a foreign listing with no US daily bars, with no
# point-in-time boundary or chain. Applied by resolve_current() AFTER the
# rename chain, and gated to the tiingo bar_source in the equities adapter --
# on a vendor that resolves the bare ticker correctly this map must NOT be
# applied. Empty today (MMC turned out to be a real rename); kept as the seam
# for a future genuine collision.
NAMESPACE_OVERRIDES: dict[str, str] = {}


def normalize(symbol: str) -> str:
    """Canonical symbol form: upper, stripped, dots -> dashes (BRK.B -> BRK-B)."""
    return str(symbol).strip().upper().replace(".", "-")


def resolve_current(symbol: str) -> str:
    """Follow the rename chain forward to the current ticker, then apply any
    vendor namespace override.

    A->B->C returns C; a symbol in no chain returns itself. Multi-hop chains
    are walked to the terminal; a cycle (A->B->A, which should never occur in
    reviewed data) terminates deterministically via a seen-set guard and
    returns the last symbol reached rather than looping forever.

    The result is only ever the STRING sent to the vendor -- the returned
    price frame is date-indexed and carries no ticker label, so resolution
    changes coverage, never a bar's value or the caller's cache key.
    """
    cursor = normalize(symbol)
    new_by_old = {normalize(old): normalize(new) for old, new, _ in RENAMES}
    seen: set[str] = set()
    while cursor in new_by_old and cursor not in seen:
        seen.add(cursor)
        cursor = new_by_old[cursor]
    return NAMESPACE_OVERRIDES.get(cursor, cursor)


def resolution_collisions(membership: pd.DataFrame, indices: tuple[str, ...]) -> list[dict]:
    """Old tickers whose successor is ALSO an independent member (in `indices`)
    during an interval that OVERLAPS the old ticker's tenure.

    Two causes, both surfaced here:
    - Membership double-listing (affects the default sp500+ndx universe): a
      company in BOTH indices is labeled with the OLD ticker in one index and
      the CURRENT ticker in the other across a rename (e.g. FB in sp500, META
      in ndx), so universe() -- which dedups by ticker string -- returns it
      twice. Both resolve to the same successor and rank as two identical
      symbols. Pre-existing in the CSV, independent of resolution.
    - Ticker recycling (adds more under sp400-inclusive configs): an old
      ticker (ABC, sp500) and a mid-cap recycler of its successor's symbol
      (COR, independently sp400 2019..2021) are both members at once and both
      fetch the same successor series.
    Either way the same price stream enters the ranking twice. A proper fix is
    resolution-aware universe dedup (collapse symbols sharing a resolved
    identity) and/or reconciling the CSV's cross-index labeling; until then
    this surfaces the affected pairs so no run double-counts silently.
    """
    rows = membership[membership["index"].isin(indices)]
    intervals: dict[str, list[tuple[str, str]]] = {}
    for _, r in rows.iterrows():
        intervals.setdefault(normalize(r["symbol"]), []).append((r["start"], r["end"] or "9999"))
    out: list[dict] = []
    for old, new, _ in RENAMES:
        old_n, new_n = normalize(old), normalize(new)
        if old_n not in intervals or new_n not in intervals:
            continue
        for os_, oe in intervals[old_n]:
            for ns, ne in intervals[new_n]:
                if os_ < ne and ns < oe:  # half-open interval overlap
                    out.append(
                        {
                            "old": old_n,
                            "new": new_n,
                            "old_window": (os_, oe),
                            "new_window": (ns, ne),
                        }
                    )
    return out


def load_symbol_allowlist(path: str | Path) -> frozenset[str]:
    """Distinct symbols named by an allowlist file, for restricting a universe.

    Tolerant of two formats so one helper serves both a hand-written list and a
    gather's own output:

    * JSONL -- a line parsing to a JSON object contributes its ``"symbol"``
      value (an options ``samples.jsonl`` line is exactly this). An unparseable
      line -- e.g. the torn final line a killed gather leaves -- is skipped, the
      same discipline the JSONL loaders use.
    * Plain -- any non-JSON, non-blank, non-``#``-comment line is taken as a
      bare ticker.

    A missing file raises FileNotFoundError: an allowlist that was ASKED for but
    is absent must fail loudly, not silently widen the universe back to
    everything. The result is symbols only -- intersecting it with real
    point-in-time membership stays the adapter's job (a name in the list but not
    a PIT member on a given date is still correctly excluded that session).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"symbols allowlist not found: {path}")
    symbols: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            symbols.add(line)  # bare ticker
            continue
        if isinstance(parsed, dict):
            symbol = parsed.get("symbol")
            if symbol:
                symbols.add(str(symbol))
        elif isinstance(parsed, str):
            symbols.add(parsed)
    return frozenset(symbols)
