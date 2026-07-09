"""Pre-registered SIC market segments (Piece 2, spec section 3.2).

SEGMENTS is frozen science (spec section 4 rule 1): the names and ranges were
pinned in the implementation plan BEFORE any segment sweep ran. Adding,
removing, or re-ranging a segment after the first sweep is a written
prospective spec amendment; the journal keeps counting everything ever run.

Ranges are INCLUSIVE over the 4-digit SIC code. Sectors tile the used SIC
space without overlap; the fine industries (biotech, banks — the charter
hypotheses) deliberately overlap their parent sectors, so one name can carry
several segments: distinct, honestly-counted trials under the one BH bar.
Codes outside every range (agriculture 0100-0999, unused 1800-1999 and
6800-6999, public administration 9000+) belong to no segment — same policy
as an unmapped symbol, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentDef:
    """Inclusive 4-digit SIC ranges + granularity marker."""

    ranges: tuple[tuple[int, int], ...]
    kind: str  # "sector" | "industry"


SEGMENTS: dict[str, SegmentDef] = {
    # ~10 coarse sectors (SIC divisions, with manufacturing split three ways).
    "energy-mining": SegmentDef(((1000, 1499),), "sector"),
    "construction": SegmentDef(((1500, 1799),), "sector"),
    "manufacturing-consumer": SegmentDef(((2000, 2799),), "sector"),
    "pharma-chemicals": SegmentDef(((2800, 2899),), "sector"),
    "other-manufacturing": SegmentDef(
        ((2900, 3499), (3700, 3799), (3900, 3999)), "sector"
    ),
    "manufacturing-tech": SegmentDef(((3500, 3699), (3800, 3899)), "sector"),
    "transport-utilities": SegmentDef(((4000, 4999),), "sector"),
    "trade": SegmentDef(((5000, 5999),), "sector"),
    "finance": SegmentDef(((6000, 6799),), "sector"),
    "services": SegmentDef(((7000, 8999),), "sector"),
    # Pre-registered fine industries (charter hypotheses); overlap by design.
    "biotech": SegmentDef(((2836, 2836), (8731, 8731)), "industry"),
    "banks": SegmentDef(((6020, 6039),), "industry"),
}


def segments_for(sic: int) -> tuple[str, ...]:
    """Every segment whose ranges contain this code, in SEGMENTS order."""
    return tuple(
        name
        for name, seg in SEGMENTS.items()
        if any(lo <= sic <= hi for lo, hi in seg.ranges)
    )
