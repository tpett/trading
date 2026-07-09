"""Pre-registered SEGMENTS table (Piece 2 spec sections 3.2/4/7).

The EXPECTED literal below is the frozen pre-registration: changing it after
the first segment sweep is a written prospective spec amendment, never a
casual edit to make a test pass.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from trading.alphasearch.segments import (
    DEFAULT_SIC_MAP_CSV,
    SEGMENTS,
    SegmentDef,  # noqa: F401
    SegmentError,
    load_sic_map,
    segment_universes,
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


# --------------------------------------------------------------------------- #
# segment_universes (spec section 3.2): fixture CSVs + touched cache files.
# --------------------------------------------------------------------------- #


def _write_sic_map(path: Path, rows: list[tuple[str, int]]) -> None:
    lines = ["symbol,cik,sic,sic_description,fetched_at"]
    lines += [f"{s},{i + 1},{sic},desc,2026-07-08" for i, (s, sic) in enumerate(rows)]
    path.write_text("\n".join(lines) + "\n")


def _write_membership(path: Path, symbols: list[str]) -> None:
    lines = ["symbol,index,start,end"] + [f"{s},sp500,2017-01-01," for s in symbols]
    path.write_text("\n".join(lines) + "\n")


def _fixture_root(tmp_path):
    """16 cached pharma names (SIC 2836: pharma-chemicals + biotech overlap),
    3 cached banks (6022: finance + banks, below min), one symbol with no SIC
    row (NOMAP), one with a SIC row but no bar cache (NOCACHE). The largecap
    samples allowlist gathers exactly 15 pharma names (== min_names: emitted);
    the midcap samples file is deliberately absent."""
    pharma = [f"P{i:02d}" for i in range(16)]
    banks = ["B00", "B01", "B02"]
    cached = pharma + banks + ["NOMAP"]
    cache = tmp_path / "data" / "equities-tiingo"
    cache.mkdir(parents=True)
    for s in cached:
        (cache / f"{s}.parquet").write_bytes(b"")  # existence is all that's checked
    (tmp_path / "data" / "options-iv").mkdir()
    (tmp_path / "data" / "options-iv" / "samples.jsonl").write_text(
        "\n".join(pharma[:15]) + "\n"  # plain-ticker allowlist form
    )
    sic = tmp_path / "sic_map.csv"
    _write_sic_map(
        sic,
        [(s, 2836) for s in pharma] + [("NOCACHE", 2836)] + [(s, 6022) for s in banks],
    )
    membership = tmp_path / "membership.csv"
    _write_membership(membership, cached + ["NOCACHE"])
    return pharma, banks, sic, membership


def test_segment_universes_emits_deep_and_options_pools(tmp_path):
    pharma, _banks, sic, membership = _fixture_root(tmp_path)
    universes, _excluded = segment_universes(tmp_path, sic, membership_path=membership)
    assert set(universes) == {
        "largecap:pharma-chemicals",
        "largecap:biotech",
        "opt-largecap:pharma-chemicals",
        "opt-largecap:biotech",
    }
    deep = universes["largecap:biotech"]
    assert deep.symbols == tuple(sorted(pharma))  # NOCACHE (no bars) is out
    assert deep.samples is None                   # price signals only
    assert deep.fundamentals_dir is None
    assert deep.cache_dir == tmp_path / "data" / "equities-tiingo"
    opt = universes["opt-largecap:biotech"]
    assert opt.symbols == tuple(sorted(pharma[:15]))  # exactly min_names: inclusive
    assert opt.samples == tmp_path / "data" / "options-iv" / "samples.jsonl"
    assert opt.fundamentals_dir == tmp_path / "data" / "fundamentals" / "equities"


def test_below_min_names_is_excluded_and_reported_never_silent(tmp_path):
    _pharma, _banks, sic, membership = _fixture_root(tmp_path)
    universes, excluded = segment_universes(tmp_path, sic, membership_path=membership)
    assert "largecap:banks" not in universes  # 3 names < 15
    row = next(r for r in excluded if r["cap"] == "largecap" and r["segment"] == "banks")
    assert row == {"segment": "banks", "cap": "largecap", "count": 3, "reason": "below-min"}
    fin = next(r for r in excluded if r["cap"] == "largecap" and r["segment"] == "finance")
    assert fin["count"] == 3
    # Every non-emitted (cap x pool x segment) slot is reported:
    # 2 caps x 2 pools x 12 segments = 48 slots, 4 emitted -> 44 rows.
    assert len(excluded) == 44


def test_missing_samples_file_reports_no_samples_instead_of_crashing(tmp_path):
    _pharma, _banks, sic, membership = _fixture_root(tmp_path)
    _universes, excluded = segment_universes(tmp_path, sic, membership_path=membership)
    midcap_opt = [r for r in excluded if r["cap"] == "opt-midcap"]
    assert len(midcap_opt) == len(SEGMENTS)
    assert all(r["reason"] == "no-samples" and r["count"] == 0 for r in midcap_opt)


def test_unmapped_or_uncached_symbols_belong_to_no_segment(tmp_path):
    _pharma, _banks, sic, membership = _fixture_root(tmp_path)
    universes, _excluded = segment_universes(tmp_path, sic, membership_path=membership)
    emitted = {s for u in universes.values() for s in u.symbols}
    assert "NOMAP" not in emitted    # no SIC row: never guessed
    assert "NOCACHE" not in emitted  # no bar cache: not a deep-pool member


def test_min_names_is_tunable(tmp_path):
    _pharma, banks, sic, membership = _fixture_root(tmp_path)
    universes, _ = segment_universes(tmp_path, sic, membership_path=membership, min_names=3)
    assert universes["largecap:banks"].symbols == tuple(sorted(banks))
    assert universes["largecap:finance"].symbols == tuple(sorted(banks))


def test_missing_sic_map_raises_with_the_exact_build_command(tmp_path):
    with pytest.raises(SegmentError, match="scripts/build_sic_map.py"):
        segment_universes(tmp_path, tmp_path / "missing.csv")


def test_every_segment_matches_at_least_one_committed_symbol():
    """A segment matching ZERO committed sic_map symbols is a range typo:
    fail HERE at test time, never at runtime (spec section 6)."""
    sics = set(load_sic_map(DEFAULT_SIC_MAP_CSV).values())
    for name in SEGMENTS:
        assert any(name in segments_for(s) for s in sics), f"{name} matches no symbol"
