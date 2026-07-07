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
   collides with a foreign listing the vendor happens to rank first. The
   entity never changed name or ticker; the vendor just needs a different
   identifier string. Kept separate from RENAMES because the semantics
   differ (no point-in-time boundary, no chain) and because the override is
   vendor-specific (see resolve_current's tiingo gating in the equities
   adapter -- applying an override to a vendor that already resolves the
   bare ticker correctly would BREAK it).

build_cik_map.py imports RENAMES from here so the fundamentals/CIK overlay
and the price-fetch path share one reviewed table.
"""

from __future__ import annotations

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
    # the CURRENT ticker is PSKY -- but this is deliberately NOT chained
    # CBS->PARA->PSKY. Unlike the ABC->COR pattern, Tiingo did NOT carry the
    # continuous history forward under PSKY: PSKY is a FRESH listing
    # (startDate 2025-08-06) that returns EMPTY for every historical date,
    # whereas PARA carries the full 2006..2025-08-07 lineage covering BOTH
    # the CBS (2017..2019) and PARA (2022..2025) membership windows. Adding
    # the PSKY hop would redirect those fetches to an empty series and zero
    # out coverage. PARA remains fetchable post-delisting, and PSKY is not a
    # membership symbol, so terminating the chain at PARA is correct.
]

# Vendor namespace collisions (NOT renames): a current US ticker whose bare
# form the vendor maps to a foreign listing with no US daily bars. Applied by
# resolve_current() AFTER the rename chain. Vendor-specific -- gated to the
# tiingo bar_source in the equities adapter (see fetch_ohlcv); on a vendor
# that resolves the bare ticker correctly this map must NOT be applied.
NAMESPACE_OVERRIDES: dict[str, str] = {
    # Tiingo's bare "MMC" resolves to Mitre Mining Corporation Ltd (ASX, no US
    # daily bars); US Marsh & McLennan is served under "MRSH" (NYSE, back to
    # 1987, verified 2026-07-07). Not a rename -- the real MMC never changed
    # ticker; Tiingo just ranks the ASX listing first for the bare symbol.
    "MMC": "MRSH",
}


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
