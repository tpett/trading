"""Pre-registered SEGMENTS table (Piece 2 spec sections 3.2/4/7).

The EXPECTED literal below is the frozen pre-registration: changing it after
the first segment sweep is a written prospective spec amendment, never a
casual edit to make a test pass.
"""

from __future__ import annotations

import dataclasses

import pytest

from trading.alphasearch.segments import (  # noqa: F401
    SEGMENTS,
    SegmentDef,
    segments_for,
)

EXPECTED: dict[str, tuple[tuple[tuple[int, int], ...], str]] = {
    "energy-mining": (((1000, 1499),), "sector"),
    "construction": (((1500, 1799),), "sector"),
    "manufacturing-consumer": (((2000, 2799),), "sector"),
    "pharma-chemicals": (((2800, 2899),), "sector"),
    "other-manufacturing": (((2900, 3499), (3700, 3799), (3900, 3999)), "sector"),
    "manufacturing-tech": (((3500, 3699), (3800, 3899)), "sector"),
    "transport-utilities": (((4000, 4999),), "sector"),
    "trade": (((5000, 5999),), "sector"),
    "finance": (((6000, 6799),), "sector"),
    "services": (((7000, 8999),), "sector"),
    "biotech": (((2836, 2836), (8731, 8731)), "industry"),
    "banks": (((6020, 6039),), "industry"),
}


def test_segments_table_is_exactly_the_frozen_preregistration():
    assert {n: (s.ranges, s.kind) for n, s in SEGMENTS.items()} == EXPECTED


def test_ten_sectors_two_industries():
    kinds = [s.kind for s in SEGMENTS.values()]
    assert kinds.count("sector") == 10
    assert kinds.count("industry") == 2


def test_ranges_are_well_formed_four_digit_and_ordered():
    for name, seg in SEGMENTS.items():
        assert seg.kind in ("sector", "industry"), name
        assert seg.ranges, name
        for lo, hi in seg.ranges:
            assert 1000 <= lo <= hi <= 9999, name
        # multi-range segments list their ranges ascending and disjoint
        flat = [b for r in seg.ranges for b in r]
        assert flat == sorted(flat), name


def test_sectors_partition_without_overlap():
    sectors = [
        (name, r)
        for name, seg in SEGMENTS.items()
        if seg.kind == "sector"
        for r in seg.ranges
    ]
    for i, (n1, (lo1, hi1)) in enumerate(sectors):
        for n2, (lo2, hi2) in sectors[i + 1 :]:
            assert hi1 < lo2 or hi2 < lo1, f"sector ranges overlap: {n1} vs {n2}"


def test_fine_industries_overlap_their_parents_by_design():
    # A code can land in several segments: distinct, honestly-counted trials.
    assert segments_for(2836) == ("pharma-chemicals", "biotech")
    assert segments_for(8731) == ("services", "biotech")
    assert segments_for(6022) == ("finance", "banks")
    assert segments_for(6021) == ("finance", "banks")


def test_representative_codes_land_in_the_right_sector():
    assert segments_for(1311) == ("energy-mining",)        # crude oil & gas
    assert segments_for(1531) == ("construction",)         # homebuilders
    assert segments_for(2080) == ("manufacturing-consumer",)  # beverages
    assert segments_for(2834) == ("pharma-chemicals",)     # pharma preparations
    assert segments_for(2911) == ("other-manufacturing",)  # petroleum refining
    assert segments_for(3711) == ("other-manufacturing",)  # motor vehicles
    assert segments_for(3674) == ("manufacturing-tech",)   # semiconductors
    assert segments_for(3841) == ("manufacturing-tech",)   # medical instruments
    assert segments_for(4813) == ("transport-utilities",)  # telecom
    assert segments_for(5812) == ("trade",)                # restaurants
    assert segments_for(6798) == ("finance",)              # REITs
    assert segments_for(7372) == ("services",)             # prepackaged software


def test_uncovered_codes_belong_to_no_segment():
    assert segments_for(700) == ()    # agriculture: uncovered by design
    assert segments_for(1900) == ()   # unused SIC space
    assert segments_for(9721) == ()   # public administration


def test_segmentdef_is_frozen():
    seg = SEGMENTS["banks"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        seg.kind = "sector"  # type: ignore[misc]
