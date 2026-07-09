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
from pathlib import Path

import pandas as pd

from trading.alphasearch.sweep import DISCOVERY_WINDOW, SweepError, UniverseSpec
from trading.symbols import load_symbol_allowlist


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


DEFAULT_SIC_MAP_CSV = Path(__file__).parent.parent / "venues" / "universes" / "sic_map.csv"
DEFAULT_MEMBERSHIP_CSV = (
    Path(__file__).parent.parent / "venues" / "universes" / "equities_membership.csv"
)
BUILD_COMMAND = "uv run python scripts/build_sic_map.py"

# (cap, membership indices, bar-cache dirname, samples filename) -- mirrors
# sweep.default_universes' pool layout.
_CAPS: tuple[tuple[str, frozenset[str], str, str], ...] = (
    ("largecap", frozenset({"sp500", "ndx"}), "equities-tiingo", "samples.jsonl"),
    ("midcap", frozenset({"sp400"}), "equities-midcap-tiingo", "samples-midcap.jsonl"),
)


class SegmentError(SweepError):
    """Segment-universe assembly refused (missing classification inputs)."""


def load_sic_map(path: Path | None = None) -> dict[str, int]:
    """symbol -> 4-digit SIC code from the committed sic_map.csv."""
    if path is None:
        path = DEFAULT_SIC_MAP_CSV
    if not path.exists():
        raise SegmentError(
            f"SIC map not found: {path}; build it with `{BUILD_COMMAND}` "
            "(fetches SEC submissions classifications, ~2-3 min)"
        )
    df = pd.read_csv(path, comment="#", dtype=str).fillna("")
    return {row.symbol: int(row.sic) for row in df.itertuples()}


def _window_members(membership_path: Path, indices: frozenset[str]) -> set[str]:
    """Ever-members of `indices` whose interval overlaps the discovery window
    (start inclusive, end exclusive, empty end = current)."""
    start_s, _, end_s = DISCOVERY_WINDOW.partition("..")
    df = pd.read_csv(membership_path, comment="#", dtype=str).fillna("")
    df = df[df["index"].isin(indices)]
    overlap = (df["start"] <= end_s) & ((df["end"] == "") | (df["end"] > start_s))
    return set(df.loc[overlap, "symbol"])


def segment_universes(
    root: Path,
    sic_map_path: Path | None = None,
    *,
    membership_path: Path | None = None,
    min_names: int = 15,
) -> tuple[dict[str, UniverseSpec], list[dict]]:
    """The pre-registered segment UniverseSpecs + the exclusions report.

    Deep pools per cap (`largecap:<segment>` / `midcap:<segment>`): membership
    symbols overlapping the discovery window intersected with cached bar
    parquets intersected with segment SIC ranges; samples=None and
    fundamentals_dir=None, so the sweep's assembly-time checks confine them to
    price signals. Options pools (`opt-<cap>:<segment>`): the gathered samples
    allowlist intersected with segment ranges, emitted only at >= min_names
    gathered names, with the samples path and fundamentals store attached
    (full signal registry, like Piece 1 pools).

    Every non-emitted (cap, pool, segment) slot lands in the report -- spec
    section 3.2: excluded segments are REPORTED, never silently dropped.
    Reasons: "below-min" (count < min_names) or "no-samples" (that cap's
    samples file has not been gathered). A missing sic_map raises
    SegmentError naming the exact build command.
    """
    sic_by_symbol = load_sic_map(sic_map_path)
    if membership_path is None:
        membership_path = DEFAULT_MEMBERSHIP_CSV
    universes: dict[str, UniverseSpec] = {}
    excluded: list[dict] = []
    fundamentals_dir = root / "data" / "fundamentals" / "equities"
    for cap, indices, cache_name, samples_name in _CAPS:
        cache_dir = root / "data" / cache_name
        members = _window_members(membership_path, indices)
        deep_pool = sorted(
            s
            for s in members
            if s in sic_by_symbol and (cache_dir / f"{s}.parquet").exists()
        )
        samples = root / "data" / "options-iv" / samples_name
        if samples.exists():
            opt_pool = sorted(s for s in load_symbol_allowlist(samples) if s in sic_by_symbol)
            opt_reason = "below-min"
        else:
            opt_pool = []
            opt_reason = "no-samples"
        deep_segments = {s: segments_for(sic_by_symbol[s]) for s in deep_pool}
        opt_segments = {s: segments_for(sic_by_symbol[s]) for s in opt_pool}
        for seg_name in SEGMENTS:
            deep = tuple(s for s in deep_pool if seg_name in deep_segments[s])
            if len(deep) >= min_names:
                name = f"{cap}:{seg_name}"
                universes[name] = UniverseSpec(
                    name=name,
                    cache_dir=cache_dir,
                    samples=None,
                    fundamentals_dir=None,
                    symbols=deep,
                )
            else:
                excluded.append(
                    {"segment": seg_name, "cap": cap, "count": len(deep),
                     "reason": "below-min"}
                )
            opt = tuple(s for s in opt_pool if seg_name in opt_segments[s])
            if len(opt) >= min_names:
                name = f"opt-{cap}:{seg_name}"
                universes[name] = UniverseSpec(
                    name=name,
                    cache_dir=cache_dir,
                    samples=samples,
                    fundamentals_dir=fundamentals_dir,
                    symbols=opt,
                )
            else:
                excluded.append(
                    {"segment": seg_name, "cap": f"opt-{cap}", "count": len(opt),
                     "reason": opt_reason}
                )
    return universes, excluded
