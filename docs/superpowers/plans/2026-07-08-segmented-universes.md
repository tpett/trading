# Segmented Universes (Piece 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the Piece 1 alpha-search engine across pre-registered SIC market segments, with per-segment leaderboard rows while the BH-FDR bar continues to span the entire trial journal.

**Architecture:** A frozen `SEGMENTS` table + `segment_universes()` in a new `src/trading/alphasearch/segments.py` produce ordinary `UniverseSpec`s (deep price-only pools per cap, options pools where ≥ 15 gathered names). The only engine change is `UniverseSpec.samples: Path | None` plus a new `symbols: tuple[str, ...] | None` explicit-universe override threaded through `build_panel`. Classification comes from a new committed `sic_map.csv` built by `scripts/build_sic_map.py` (SEC submissions endpoint, mirroring `scripts/build_cik_map.py`). Zero changes to journal/BH/holdout machinery: a segment is just a universe name inside the hashed trial config.

**Tech Stack:** Python 3.12, pandas 2.3.1, stdlib `urllib` (NO `requests`), pytest 8.4.1 (warnings-as-errors), ruff 0.12.3 (`E,F,I,UP,B`, line 100), `uv` runner.

**Spec:** `docs/superpowers/specs/2026-07-08-segmented-universes-design.md` (approved; do not re-litigate its locked decisions).

## Global Constraints

- Pre-registered constants copied verbatim from the spec: discovery window `2019-01-01..2023-12-31` (already `sweep.DISCOVERY_WINDOW` — import it, never re-type it), BH `q=0.10` across the WHOLE journal, segment minimum cross-section `min_names=15`, coverage floor for `build_sic_map.py` = 90% of window membership, biotech = SIC `2836` + `8731`, banks = SIC `6020–6039`.
- The `SEGMENTS` table in Task 1 is **frozen pre-registered science**. Implement it exactly as written there; any deviation is a spec amendment, not an implementation choice.
- Stdlib `urllib` only for HTTP; every SEC request goes through `trading.fundamentals.companyfacts.http_get_json` (process-global 0.11 s throttle + mandatory User-Agent). Never add `requests`.
- Library modules under `src/trading/` never read the clock; `ts`/`fetched_at` arrive as parameters. Scripts' `main()` may read the clock.
- The trial journal is append-only; nothing in this plan touches `trading.journal`, `stats.bh_fdr`, holdout ceremony, or `trial_config`/`trial_config_hash`. Segment identity = the universe NAME inside the hashed config (spec §3.3); the symbol list is derived from committed CSVs, exactly like bar caches are for Piece 1 universes, so it does NOT enter the hash.
- `run_sweep` validates the FULL signal × universe cross-product before any trial is journaled (all-or-nothing). Do not regress this; segment tests assert it.
- Piece 1 plan-review lessons (apply everywhere): never assert float-exact zero on compounded synthetic prices; no rank-deficient (constant-column) regression fixtures; dedupe via dicts in explicit order, never via pandas sort stability; never write `x or DEFAULT` for an optional param — use `if x is None`; `pd.Timedelta(N, unit="D")` not `pd.Timedelta(days=N)` (DeprecationWarning under pinned pandas = test failure).
- Every commit message tagged `[AI]`, one logical change per commit, run `uv run ruff check` + the named tests before each commit.
- Commands run from the repo root `/Users/travis/Source/personal/trading`.

## File Structure

| File | Responsibility |
|---|---|
| `src/trading/alphasearch/segments.py` (create) | `SegmentDef`, frozen `SEGMENTS`, `segments_for`, `load_sic_map`, `segment_universes`, `SegmentError` |
| `scripts/build_sic_map.py` (create) | Fetch SIC codes → committed `src/trading/venues/universes/sic_map.csv` |
| `src/trading/venues/universes/sic_map.csv` (generated, committed) | symbol,cik,sic,sic_description,fetched_at |
| `src/trading/alphasearch/panel.py` (modify) | `build_panel` gains `samples: Path | None` + keyword-only `symbols` override |
| `src/trading/alphasearch/sweep.py` (modify) | `UniverseSpec.samples: Path | None`, new `symbols` field, `build_universe_panel` threads it |
| `src/trading/cli.py` (modify) | `--segments` flag, relaxed `--universe`, holdout segment-name resolution |
| `tests/test_alphasearch_segments.py` (create) | Table validation + `segment_universes` unit tests |
| `tests/test_build_sic_map.py` (create) | Mocked-fetch script tests + committed-artifact anchors |
| `tests/test_alphasearch_segments_golden.py` (create) | End-to-end golden segment sweep |
| `tests/test_alphasearch_panel.py`, `tests/test_alphasearch_sweep.py`, `tests/test_alphasearch_cli.py` (modify) | New-surface tests |
| `docs/experiments.md`, `docs/glossary.md`, `src/trading/venues/universes/sources/PROVENANCE.md` (modify) | Disclosures |

---

### Task 1: The frozen SEGMENTS table

**Files:**
- Create: `src/trading/alphasearch/segments.py`
- Test: `tests/test_alphasearch_segments.py`

**Interfaces:**
- Consumes: nothing (pure data + one lookup function).
- Produces: `SegmentDef` (frozen dataclass: `ranges: tuple[tuple[int, int], ...]`, `kind: str`), `SEGMENTS: dict[str, SegmentDef]`, `segments_for(sic: int) -> tuple[str, ...]`. Task 4 extends this same module with `segment_universes`; Tasks 4–7 rely on these exact names and the exact segment keys below.

The table (THE pre-registered science — ranges are inclusive over the 4-digit SIC code):

| segment | ranges | kind | why |
|---|---|---|---|
| energy-mining | 1000–1499 | sector | SIC Division B: metal mining through oil & gas extraction |
| construction | 1500–1799 | sector | Division C: builders, heavy construction, trades |
| manufacturing-consumer | 2000–2799 | sector | nondurables: food, tobacco, textiles, apparel, lumber, furniture, paper, printing |
| pharma-chemicals | 2800–2899 | sector | chemicals & allied products; split out because it hosts pharma (2834) and biologics (2836) |
| other-manufacturing | 2900–3499, 3700–3799, 3900–3999 | sector | heavy/durable remainder: refining, rubber, stone/glass, metals, autos/aerospace, misc |
| manufacturing-tech | 3500–3699, 3800–3899 | sector | machinery & computers (35xx), electronics/semis (36xx), instruments/med-devices (38xx) |
| transport-utilities | 4000–4999 | sector | Division E: transport, telecom/media (48xx), utilities (49xx) |
| trade | 5000–5999 | sector | Divisions F+G: wholesale + retail |
| finance | 6000–6799 | sector | Division H: banks, insurance, real estate, REITs, holding cos |
| services | 7000–8999 | sector | Division I: software (7372) through health services and R&D (8731) |
| biotech | 2836, 8731 | industry | charter hypothesis: biological products + commercial physical/biological research; overlaps pharma-chemicals and services BY DESIGN |
| banks | 6020–6039 | industry | charter hypothesis: national/state commercial banks; overlaps finance BY DESIGN |

Deliberately uncovered: 0100–0999 (agriculture), 1800–1999 (unused SIC space), 6800–6999 (unused), 9000+ (public administration) — no discovery-window membership names carry those codes; an uncovered code lands in no segment, same policy as an unmapped symbol (never guessed).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alphasearch_segments.py`:

```python
"""Pre-registered SEGMENTS table (Piece 2 spec sections 3.2/4/7).

The EXPECTED literal below is the frozen pre-registration: changing it after
the first segment sweep is a written prospective spec amendment, never a
casual edit to make a test pass.
"""

from __future__ import annotations

import dataclasses

import pytest

from trading.alphasearch.segments import SEGMENTS, SegmentDef, segments_for

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_alphasearch_segments.py -v 2>&1 | tee /tmp/claude-p2-task1.log`
Expected: collection error — `ModuleNotFoundError: No module named 'trading.alphasearch.segments'`.

- [ ] **Step 3: Create the module with the frozen table**

Create `src/trading/alphasearch/segments.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_alphasearch_segments.py -v`
Expected: 8 passed.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/trading/alphasearch/segments.py tests/test_alphasearch_segments.py
git add src/trading/alphasearch/segments.py tests/test_alphasearch_segments.py
git commit -m "Freeze the pre-registered SIC SEGMENTS table (Piece 2) [AI]

10 coarse sectors + biotech/banks fine industries, pinned before any
segment sweep per spec section 4 rule 1; validation tests assert the
exact frozen ranges so a drive-by edit cannot silently amend science."
```

---

### Task 2: `scripts/build_sic_map.py` + the committed `sic_map.csv`

**Files:**
- Create: `scripts/build_sic_map.py`
- Create (generated): `src/trading/venues/universes/sic_map.csv`
- Modify: `src/trading/venues/universes/sources/PROVENANCE.md` (append a section)
- Test: `tests/test_build_sic_map.py`

**Interfaces:**
- Consumes: `trading.fundamentals.companyfacts.http_get_json` (the throttled, monkeypatchable seam — public alias of `_http_get_json`), `src/trading/fundamentals/cik_map.csv`, `src/trading/venues/universes/equities_membership.csv`.
- Produces: committed `src/trading/venues/universes/sic_map.csv` with header `symbol,cik,sic,sic_description,fetched_at`, one row per symbol. Script functions Task 4's tests never touch; Task 4 consumes only the CSV.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_sic_map.py`:

```python
"""scripts/build_sic_map.py: offline unit tests on the mocked fetch seam,
plus committed-artifact anchors (same pattern as tests/test_fundamentals_cik_map.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_sic_map  # noqa: E402

SIC_MAP_CSV = (
    Path(__file__).resolve().parent.parent
    / "src" / "trading" / "venues" / "universes" / "sic_map.csv"
)


def _cik_map(rows: list[tuple[str, int, str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["symbol", "cik", "start", "end"])
    return frame.astype({"cik": str})


def test_window_pairs_selects_intervals_overlapping_the_discovery_window():
    frame = _cik_map(
        [
            ("OLD", 111, "2017-01-01", "2018-06-01"),  # ends pre-window: skipped
            ("FB", 222, "2017-01-01", "2022-06-09"),   # overlaps: chosen
            ("META", 222, "2022-06-09", ""),           # overlaps (open end): chosen
            ("NEWCO", 333, "2024-05-01", ""),          # starts post-window: skipped
        ]
    )
    assert build_sic_map.window_pairs(frame) == [("FB", 222), ("META", 222)]


def test_window_pairs_first_overlapping_interval_wins_deterministically():
    # Dict-in-insertion-order dedupe (Piece 1 lesson: never rely on sort
    # stability): the FIRST overlapping interval in file order is the one used.
    frame = _cik_map(
        [
            ("DUAL", 111, "2017-01-01", "2020-01-01"),
            ("DUAL", 999, "2020-01-01", ""),
        ]
    )
    assert build_sic_map.window_pairs(frame) == [("DUAL", 111)]


def test_build_rows_parses_sic_and_fetches_each_cik_once():
    calls: list[str] = []

    def fetch(url: str) -> dict:
        calls.append(url)
        return {"sic": "2836", "sicDescription": "Biological Products, (No Diagnostic)"}

    rows, unmapped = build_sic_map.build_rows(
        [("GOOG", 1652044), ("GOOGL", 1652044)], fetch, fetched_at="2026-07-08"
    )
    assert unmapped == []
    assert rows == [
        ("GOOG", 1652044, 2836, "Biological Products, (No Diagnostic)", "2026-07-08"),
        ("GOOGL", 1652044, 2836, "Biological Products, (No Diagnostic)", "2026-07-08"),
    ]
    # GOOG/GOOGL share one CIK: exactly one request.
    assert calls == [build_sic_map.SUBMISSIONS_URL.format(cik=1652044)]


def test_fetch_sic_retries_once_then_succeeds():
    attempts = {"n": 0}

    def flaky(url: str) -> dict:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("edgar hiccup")
        return {"sic": "6022", "sicDescription": "State Commercial Banks"}

    assert build_sic_map.fetch_sic(1, flaky) == (6022, "State Commercial Banks")
    assert attempts["n"] == 2


def test_fetch_sic_two_failures_is_unmapped():
    def boom(url: str) -> dict:
        raise OSError("edgar down")

    assert build_sic_map.fetch_sic(1, boom) is None


def test_filer_without_sic_is_recorded_unmapped_never_guessed():
    rows, unmapped = build_sic_map.build_rows(
        [("XX", 1)], lambda url: {"sic": "", "sicDescription": ""}, fetched_at="2026-07-08"
    )
    assert rows == []
    assert unmapped == ["XX"]


def test_default_fetch_seam_is_the_throttled_companyfacts_one():
    # SEC policy: 0.11 s process-global throttle + mandatory User-Agent live
    # in ONE place; this script must ride that seam, not roll its own.
    from trading.fundamentals import companyfacts

    assert build_sic_map.http_get_json is companyfacts.http_get_json


def test_validate_exits_nonzero_below_90_percent_coverage():
    rows = [("A", 1, 2836, "Biological Products", "2026-07-08")]
    members = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"}  # 1/11 mapped
    with pytest.raises(SystemExit) as excinfo:
        build_sic_map.validate(rows, members)
    assert "90%" in str(excinfo.value)


def test_validate_passes_at_full_coverage(capsys):
    rows = [("A", 1, 2836, "Biological Products", "2026-07-08")]
    build_sic_map.validate(rows, {"A"})
    assert "coverage OK" in capsys.readouterr().out


def test_write_csv_round_trips_comma_bearing_descriptions(tmp_path):
    # SEC sicDescription values contain commas; string-joined CSV would tear.
    out = tmp_path / "sic_map.csv"
    rows = [
        ("XX", 1, 7370, "Services-Computer Programming, Data Processing, Etc.", "2026-07-08")
    ]
    build_sic_map.write_csv(rows, out)
    df = pd.read_csv(out, comment="#", dtype=str)
    assert list(df.columns) == ["symbol", "cik", "sic", "sic_description", "fetched_at"]
    assert df.iloc[0]["sic_description"] == (
        "Services-Computer Programming, Data Processing, Etc."
    )


# --- committed artifact (exists only after the real generation run) ----------


def test_committed_sic_map_shape_and_anchors():
    df = pd.read_csv(SIC_MAP_CSV, comment="#", dtype=str)
    assert list(df.columns) == ["symbol", "cik", "sic", "sic_description", "fetched_at"]
    assert len(df) > 900  # ~1130 window symbols minus the unmapped tail
    by_symbol = dict(zip(df["symbol"], df["sic"].astype(int), strict=True))
    assert by_symbol["AAPL"] == 3571          # Electronic Computers
    assert by_symbol["AMGN"] == 2836          # Biological Products: biotech anchor
    assert 6020 <= by_symbol["JPM"] <= 6039   # commercial bank: banks anchor
    assert df["symbol"].is_unique             # one row per symbol
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_build_sic_map.py -v 2>&1 | tee /tmp/claude-p2-task2.log`
Expected: collection error — `ModuleNotFoundError: No module named 'build_sic_map'`.

- [ ] **Step 3: Write the script**

Create `scripts/build_sic_map.py`:

```python
"""Build the committed symbol -> SIC classification map
(src/trading/venues/universes/sic_map.csv) for the Piece 2 segment universes.

Source: https://data.sec.gov/submissions/CIK##########.json `sic` +
`sicDescription` -- each filer's CURRENT code, applied backward over the
discovery window (a disclosed Piece 2 caveat) -- joined through the committed
cik_map.csv. A symbol with multiple CIK intervals uses the interval
overlapping the discovery window 2019-01-01..2023-12-31. Mirrors
scripts/build_cik_map.py's conventions: stdlib urllib only, the companyfacts
throttle seam (process-global 0.11 s + mandatory User-Agent), self-validating
main() that refuses to emit a bad map.

A symbol whose fetch fails twice (retry-once, spec section 6) or whose filer
carries no SIC is recorded unmapped and belongs to NO segment -- never
guessed. main() prints the coverage report and exits non-zero below 90% of
window membership. Update
src/trading/venues/universes/sources/PROVENANCE.md on every regeneration.

Usage: uv run python scripts/build_sic_map.py   (~1130 requests, ~2-3 min)
"""

from __future__ import annotations

import csv
import datetime
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from trading.fundamentals.companyfacts import http_get_json

ROOT = Path(__file__).resolve().parent.parent
CIK_MAP = ROOT / "src" / "trading" / "fundamentals" / "cik_map.csv"
MEMBERSHIP = ROOT / "src" / "trading" / "venues" / "universes" / "equities_membership.csv"
OUTPUT = ROOT / "src" / "trading" / "venues" / "universes" / "sic_map.csv"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
# The pre-registered discovery window (sweep.DISCOVERY_WINDOW); intervals are
# start-inclusive, end-exclusive, empty end = current.
WINDOW_START, WINDOW_END = "2019-01-01", "2023-12-31"
MIN_COVERAGE = 0.90

Row = tuple[str, int, int, str, str]  # symbol, cik, sic, sic_description, fetched_at


def window_pairs(cik_map: pd.DataFrame) -> list[tuple[str, int]]:
    """(symbol, cik) pairs via the interval overlapping the discovery window.

    A symbol with no overlapping interval was never a window-era filer under
    any mapped CIK and is skipped. If several intervals overlap, the FIRST in
    file order wins -- dict-in-insertion-order dedupe, deterministic.
    """
    chosen: dict[str, int] = {}
    for row in cik_map.itertuples():
        overlaps = row.start <= WINDOW_END and (row.end == "" or row.end > WINDOW_START)
        if overlaps and row.symbol not in chosen:
            chosen[row.symbol] = int(row.cik)
    return sorted(chosen.items())


def fetch_sic(
    cik: int, fetch_json: Callable[[str], dict] = http_get_json
) -> tuple[int, str] | None:
    """(sic, sicDescription) for one CIK, or None -> unmapped (the fetch
    failed twice, or the filer has no SIC on record)."""
    payload: dict | None = None
    for _attempt in range(2):  # retry-once per spec section 6
        try:
            payload = fetch_json(SUBMISSIONS_URL.format(cik=cik))
            break
        except Exception:
            continue
    if payload is None:
        return None
    sic_raw = str(payload.get("sic") or "").strip()
    if not sic_raw.isdigit():
        return None
    return int(sic_raw), str(payload.get("sicDescription") or "")


def build_rows(
    pairs: list[tuple[str, int]],
    fetch_json: Callable[[str], dict] = http_get_json,
    *,
    fetched_at: str,
) -> tuple[list[Row], list[str]]:
    """One row per symbol; a CIK shared by several symbols (GOOG/GOOGL,
    FB/META) is fetched exactly once."""
    by_cik: dict[int, tuple[int, str] | None] = {}
    rows: list[Row] = []
    unmapped: list[str] = []
    for symbol, cik in pairs:
        if cik not in by_cik:
            by_cik[cik] = fetch_sic(cik, fetch_json)
        got = by_cik[cik]
        if got is None:
            unmapped.append(symbol)
            continue
        rows.append((symbol, cik, got[0], got[1], fetched_at))
    return rows, unmapped


def membership_window_symbols() -> set[str]:
    """Membership symbols whose interval overlaps the discovery window -- the
    honest coverage denominator (a 2017-2018-only member can never appear in
    a window segment, so it must not dilute the ratio)."""
    df = pd.read_csv(MEMBERSHIP, comment="#", dtype=str).fillna("")
    overlap = (df["start"] <= WINDOW_END) & ((df["end"] == "") | (df["end"] > WINDOW_START))
    return set(df.loc[overlap, "symbol"])


def validate(rows: list[Row], members: set[str]) -> None:
    mapped = {r[0] for r in rows} & members
    ratio = len(mapped) / len(members)
    if ratio < MIN_COVERAGE:
        sys.exit(
            f"FATAL: only {ratio:.1%} of {len(members)} window membership symbols "
            f"mapped to a SIC (need >= {MIN_COVERAGE:.0%}); do not commit this map"
        )
    print(f"coverage OK: {ratio:.1%} of {len(members)} window membership symbols mapped")


def write_csv(rows: list[Row], output: Path = OUTPUT) -> None:
    """csv.writer, NOT string-join: SEC sicDescription values contain commas
    (e.g. 'Services-Computer Programming, Data Processing, Etc.')."""
    with output.open("w", newline="") as fh:
        fh.write("# symbol -> SIC classification. GENERATED by scripts/build_sic_map.py\n")
        fh.write("# Source + caveats: src/trading/venues/universes/sources/PROVENANCE.md.\n")
        fh.write("# CURRENT SEC code applied backward over the window (disclosed caveat).\n")
        writer = csv.writer(fh)
        writer.writerow(["symbol", "cik", "sic", "sic_description", "fetched_at"])
        writer.writerows(rows)


def main() -> None:
    cik_map = pd.read_csv(CIK_MAP, comment="#", dtype=str).fillna("")
    pairs = window_pairs(cik_map)
    distinct_ciks = len({cik for _, cik in pairs})
    print(f"fetching SIC for {len(pairs)} symbols ({distinct_ciks} CIKs, ~0.11s/req)...")
    fetched_at = datetime.datetime.now(datetime.UTC).date().isoformat()
    rows, unmapped = build_rows(pairs, fetched_at=fetched_at)
    validate(rows, membership_window_symbols())
    write_csv(rows)
    print(f"wrote {OUTPUT} ({len(rows)} symbols mapped; {len(unmapped)} unmapped)")
    if unmapped:
        print("unmapped (belong to NO segment, never guessed): " + ", ".join(unmapped))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the offline tests (the committed-artifact test still fails)**

Run: `uv run pytest tests/test_build_sic_map.py -v`
Expected: 10 passed, 1 failed — only `test_committed_sic_map_shape_and_anchors` (FileNotFoundError: sic_map.csv does not exist yet).

- [ ] **Step 5: Run the script for real (network, ~2–3 min)**

Run: `uv run python scripts/build_sic_map.py 2>&1 | tee /tmp/claude-p2-sicmap.log`
Expected output shape (counts approximate):

```
fetching SIC for ~1120 symbols (~1100 CIKs, ~0.11s/req)...
coverage OK: 9x.x% of ~1100 window membership symbols mapped
wrote .../src/trading/venues/universes/sic_map.csv (~1090 symbols mapped; ~30 unmapped)
unmapped (belong to NO segment, never guessed): ...
```

If it exits `FATAL: only ...%` — STOP and consult the developer (golden rule); do not lower `MIN_COVERAGE`.

- [ ] **Step 6: Run the full test file to verify all pass**

Run: `uv run pytest tests/test_build_sic_map.py -v`
Expected: 11 passed. If an anchor assertion fails (e.g. AAPL not 3571), inspect the committed CSV row — fix the TEST only if the SEC value genuinely differs, and note it in the PROVENANCE entry.

- [ ] **Step 7: Append the provenance entry**

Append to `src/trading/venues/universes/sources/PROVENANCE.md`:

```markdown
## SIC classification map (Piece 2 segments)

- `sic_map.csv` GENERATED by `scripts/build_sic_map.py` on 2026-07-08 (see the
  `fetched_at` column): for each cik_map.csv symbol whose CIK interval
  overlaps the discovery window 2019-01-01..2023-12-31, the `sic` +
  `sicDescription` fields of https://data.sec.gov/submissions/CIK##########.json
  (throttled 0.11 s/req, User-Agent per SEC policy, retry-once per symbol).
- Coverage at generation: <fill in from the script's printed report — ratio,
  mapped count, and the full unmapped symbol list>.
- KNOWN CAVEATS (disclosed on every segment result, spec section 4 rule 5):
  the code is each filer's CURRENT classification applied backward over the
  whole window (no PIT reclassification history); segment membership is
  static across the window; unmapped symbols belong to NO segment (never
  guessed). Regenerate deliberately, review the diff, and update this entry.
```

(Replace the `<fill in ...>` placeholder with the actual numbers from Step 5's log before committing.)

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check scripts/build_sic_map.py tests/test_build_sic_map.py
git add scripts/build_sic_map.py tests/test_build_sic_map.py \
    src/trading/venues/universes/sic_map.csv \
    src/trading/venues/universes/sources/PROVENANCE.md
git commit -m "Add build_sic_map.py + committed sic_map.csv [AI]

SEC submissions SIC codes joined via cik_map.csv's window-overlapping
intervals; throttled companyfacts seam, retry-once, coverage-gated at
90% so a degraded fetch can never silently commit a thin map."
```

---

### Task 3: `UniverseSpec.symbols` + `build_panel` explicit-universe override

**Files:**
- Modify: `src/trading/alphasearch/panel.py:248-279` (`build_panel`)
- Modify: `src/trading/alphasearch/sweep.py:193-219` (`UniverseSpec`, `build_universe_panel`)
- Test: `tests/test_alphasearch_panel.py`, `tests/test_alphasearch_sweep.py`

**Interfaces:**
- Consumes: existing `load_symbol_allowlist`, `load_closes`, `load_options`, `load_fundamentals`, `PanelData`.
- Produces (Tasks 4–6 rely on these exact signatures):
  - `UniverseSpec(name: str, cache_dir: Path, samples: Path | None, fundamentals_dir: Path | None, symbols: tuple[str, ...] | None = None)` — frozen dataclass; the new field defaults to `None` so every existing positional constructor call (`default_universes`, `tests/test_alphasearch_sweep.py:29,177`, `tests/test_alphasearch_golden.py:52`) compiles unchanged.
  - `build_panel(cache_dir: Path, samples: Path | None, fundamentals_dir: Path | None, *, symbols: tuple[str, ...] | None = None) -> PanelData`.
  - `_check_universe_supports` is deliberately UNCHANGED: `samples=None` yields `panel.options == {}` and `fundamentals_dir=None` yields `panel.fundamentals == {}`, so the existing truthiness checks already refuse options/fundamentals signals — verified end-to-end below.

- [ ] **Step 1: Write the failing panel tests**

Append to `tests/test_alphasearch_panel.py`:

```python
# --------------------------------------------------------------------------- #
# Explicit-symbols universes (Piece 2)
# --------------------------------------------------------------------------- #


def _write_cache(tmp_path, symbols):
    cache = tmp_path / "cache"
    cache.mkdir()
    idx = pd.date_range("2020-01-02", periods=5, freq="B", tz="UTC")
    for sym in symbols:
        pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 2.0,
                      "volume": 10.0}, index=idx).to_parquet(cache / f"{sym}.parquet")
    return cache


def test_build_panel_explicit_symbols_overrides_the_samples_allowlist(tmp_path):
    cache = _write_cache(tmp_path, ("AAA", "BBB", "CCC"))
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps(_cell("AAA", "2020-01-02")) + "\n"
        + json.dumps(_cell("BBB", "2020-01-02")) + "\n"
    )
    panel = build_panel(cache, samples, None, symbols=("CCC", "BBB"))
    assert panel.symbols == ("BBB", "CCC")   # explicit set wins, sorted
    assert set(panel.options) == {"BBB"}     # option frames restricted to it


def test_build_panel_samples_none_builds_a_closes_only_panel(tmp_path):
    cache = _write_cache(tmp_path, ("AAA",))
    panel = build_panel(cache, None, None, symbols=("AAA", "NOBAR"))
    assert panel.symbols == ("AAA",)         # missing-data rule: NOBAR dropped
    assert panel.options == {}
    assert panel.fundamentals == {}
    assert panel.corrupt_cells == 0


def test_build_panel_empty_symbols_tuple_refused(tmp_path):
    # Mirrors the empty-signals refusal: a universe with no names is a caller
    # bug, refused at assembly, never a silent no-trade sweep.
    with pytest.raises(PanelError, match="empty"):
        build_panel(tmp_path, None, None, symbols=())


def test_build_panel_without_any_universe_source_refused(tmp_path):
    with pytest.raises(PanelError, match="universe source"):
        build_panel(tmp_path, None, None)
```

- [ ] **Step 2: Write the failing sweep tests**

Append to `tests/test_alphasearch_sweep.py` (add `import pandas as pd` and extend the `trading.alphasearch.sweep` import list with `default_universes` if absent; add `from trading.alphasearch.panel import PanelError`):

```python
# --------------------------------------------------------------------------- #
# Explicit-symbols universes (Piece 2): the real files path, no factory.
# --------------------------------------------------------------------------- #


def _write_deep_universe(tmp_path) -> UniverseSpec:
    """make_panel()'s closes as real parquets + an explicit symbols tuple:
    exactly the shape segment_universes emits for a deep pool."""
    panel = make_panel()
    cache = tmp_path / "cache"
    cache.mkdir()
    for sym in panel.symbols:
        closes = panel.closes[sym]
        pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes,
             "volume": 1000.0},
            index=closes.index,
        ).to_parquet(cache / f"{sym}.parquet")
    return UniverseSpec("largecap:test", cache, None, None, symbols=panel.symbols)


def test_deep_universe_runs_price_signals_end_to_end(tmp_path):
    # No panel_factory injection: build_universe_panel/build_panel must
    # assemble a closes-only panel from the explicit symbols tuple.
    journal = trials_journal(tmp_path / "journal")
    uspec = _write_deep_universe(tmp_path)
    rows, n_trials = run_sweep({uspec.name: uspec}, journal, make_factors(),
                               ts="t1", signals=_subset("mom21"), window=WINDOW)
    assert n_trials == 1
    assert rows[0].universe == "largecap:test"
    assert rows[0].error is None
    assert abs(rows[0].alpha_t) > 5           # the engineered momentum spread


def test_deep_universe_refuses_options_signals_end_to_end(tmp_path):
    # samples=None -> panel.options == {} -> the EXISTING requires_options
    # assembly-time refusal fires; zero trials journaled (all-or-nothing).
    journal = trials_journal(tmp_path / "journal")
    uspec = _write_deep_universe(tmp_path)
    with pytest.raises(SweepError, match="requires options"):
        run_sweep({uspec.name: uspec}, journal, make_factors(), ts="t1",
                  signals=_subset("mom21", "hedge"), window=WINDOW)
    assert list(journal.events()) == []


def test_deep_universe_refuses_fundamentals_signals_end_to_end(tmp_path):
    # fundamentals_dir=None -> panel.fundamentals == {} -> existing refusal.
    journal = trials_journal(tmp_path / "journal")
    uspec = _write_deep_universe(tmp_path)
    with pytest.raises(SweepError, match="requires fundamentals"):
        run_sweep({uspec.name: uspec}, journal, make_factors(), ts="t1",
                  signals=_subset("earnings_yield"), window=WINDOW)
    assert list(journal.events()) == []


def test_empty_symbols_tuple_refused_at_assembly_no_trials(tmp_path):
    journal = trials_journal(tmp_path / "journal")
    cache = tmp_path / "cache"
    cache.mkdir()
    uspec = UniverseSpec("largecap:empty", cache, None, None, symbols=())
    with pytest.raises(PanelError, match="empty"):
        run_sweep({uspec.name: uspec}, journal, make_factors(), ts="t1",
                  signals=_subset("mom21"), window=WINDOW)
    assert list(journal.events()) == []


def test_universespec_symbols_defaults_none_for_piece1_call_sites():
    got = default_universes(Path("."))
    assert got["largecap"].symbols is None
    assert got["midcap"].symbols is None
```

- [ ] **Step 3: Run to verify the new tests fail**

Run: `uv run pytest tests/test_alphasearch_panel.py tests/test_alphasearch_sweep.py -v 2>&1 | tee /tmp/claude-p2-task3.log`
Expected: the 9 new tests fail (`TypeError: build_panel() got an unexpected keyword argument 'symbols'` / `TypeError: __init__() got an unexpected keyword argument 'symbols'` / `AttributeError: 'UniverseSpec' object has no attribute 'symbols'`); all pre-existing tests still pass.

- [ ] **Step 4: Implement `build_panel`**

In `src/trading/alphasearch/panel.py`, replace the whole `build_panel` function (lines 248–279) with:

```python
def build_panel(
    cache_dir: Path,
    samples: Path | None,
    fundamentals_dir: Path | None,
    *,
    symbols: tuple[str, ...] | None = None,
) -> PanelData:
    """Assemble one universe's PanelData.

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
    closes = load_closes(cache_dir, allowlist)
    if not closes:
        raise PanelError(f"no bar caches under {cache_dir} for the requested universe")
    options, corrupt = load_options(samples) if samples is not None else ({}, 0)
    fundamentals = (
        load_fundamentals(fundamentals_dir, closes) if fundamentals_dir is not None else {}
    )
    universe = tuple(s for s in allowlist if s in closes)
    return PanelData(
        closes={s: closes[s] for s in universe},
        options={s: options[s] for s in universe if s in options},
        fundamentals={s: fundamentals[s] for s in universe if s in fundamentals},
        symbols=universe,
        corrupt_cells=corrupt,
    )
```

- [ ] **Step 5: Implement `UniverseSpec.symbols`**

In `src/trading/alphasearch/sweep.py`, replace the `UniverseSpec` dataclass and `build_universe_panel` (lines 193–199 and 218–219):

```python
@dataclass(frozen=True)
class UniverseSpec:
    name: str
    cache_dir: Path
    samples: Path | None
    fundamentals_dir: Path | None
    # Piece 2: an explicit universe (segment pools). None = derive from the
    # samples allowlist (Piece 1 behavior). The segment's identity in the
    # hashed trial config is its NAME (spec section 3.3); the symbol list is
    # derived from committed CSVs, like bar caches are for Piece 1 pools.
    symbols: tuple[str, ...] | None = None
```

```python
def build_universe_panel(spec: UniverseSpec) -> PanelData:
    return build_panel(
        spec.cache_dir, spec.samples, spec.fundamentals_dir, symbols=spec.symbols
    )
```

Also update the section comment above `UniverseSpec` (line 190-192) from "the two gathered options pools" to:

```python
# --------------------------------------------------------------------------- #
# Universes (spec 3.2): Piece 1's two gathered options pools, plus Piece 2's
# segment universes (explicit symbols; samples optional). Every signal family
# in a universe is measured on the same cross-section.
# --------------------------------------------------------------------------- #
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_alphasearch_panel.py tests/test_alphasearch_sweep.py tests/test_alphasearch_golden.py tests/test_alphasearch_lookahead.py -v`
Expected: all pass (existing Piece 1 tests prove the default path is unchanged).

- [ ] **Step 7: Lint, full suite, commit**

```bash
uv run ruff check src tests
uv run pytest -q 2>&1 | tail -5
git add src/trading/alphasearch/panel.py src/trading/alphasearch/sweep.py \
    tests/test_alphasearch_panel.py tests/test_alphasearch_sweep.py
git commit -m "Extend UniverseSpec/build_panel with explicit symbols + optional samples [AI]

Segments are ordinary UniverseSpecs: an explicit symbol tuple overrides
the samples-allowlist derivation, samples=None yields a closes-only
panel so the existing assembly-time options/fundamentals refusals fire
unchanged. Empty tuple refused at assembly (mirrors empty-signals)."
```

---

### Task 4: `segment_universes` — segments become UniverseSpecs

**Files:**
- Modify: `src/trading/alphasearch/segments.py` (extend the Task 1 module)
- Test: `tests/test_alphasearch_segments.py` (append)

**Interfaces:**
- Consumes: `SEGMENTS`/`segments_for` (Task 1), committed `sic_map.csv` (Task 2), `UniverseSpec(..., symbols=...)` (Task 3), `sweep.DISCOVERY_WINDOW`, `sweep.SweepError`, `trading.symbols.load_symbol_allowlist`.
- Produces (Tasks 5–6 rely on these exact names):
  - `SegmentError(SweepError)`
  - `DEFAULT_SIC_MAP_CSV: Path`, `DEFAULT_MEMBERSHIP_CSV: Path`, `BUILD_COMMAND: str`
  - `load_sic_map(path: Path | None = None) -> dict[str, int]`
  - `segment_universes(root: Path, sic_map_path: Path | None = None, *, membership_path: Path | None = None, min_names: int = 15) -> tuple[dict[str, UniverseSpec], list[dict]]` — exclusion rows are `{"segment": str, "cap": str, "count": int, "reason": "below-min" | "no-samples"}` where `cap` is one of `largecap|midcap|opt-largecap|opt-midcap`.
  - Universe naming: deep `largecap:<segment>` / `midcap:<segment>` (samples=None, fundamentals_dir=None → price signals only); options `opt-largecap:<segment>` / `opt-midcap:<segment>` (samples path set, fundamentals_dir set → full registry).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_alphasearch_segments.py`. Keep Task 1's `import dataclasses` and `import pytest` lines; add `from pathlib import Path` and widen the segments import to:

```python
from pathlib import Path

from trading.alphasearch.segments import (
    DEFAULT_SIC_MAP_CSV,
    SEGMENTS,
    SegmentDef,
    SegmentError,
    load_sic_map,
    segment_universes,
    segments_for,
)
```

then append:

```python
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
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/test_alphasearch_segments.py -v 2>&1 | tee /tmp/claude-p2-task4.log`
Expected: collection error — `ImportError: cannot import name 'segment_universes'`.

- [ ] **Step 3: Implement**

Append to `src/trading/alphasearch/segments.py` (and extend the module's imports to):

```python
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trading.alphasearch.sweep import DISCOVERY_WINDOW, SweepError, UniverseSpec
from trading.symbols import load_symbol_allowlist
```

then add below `segments_for`:

```python
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
    symbols overlapping the discovery window ∩ cached bar parquets ∩ segment
    SIC ranges; samples=None and fundamentals_dir=None, so the sweep's
    assembly-time checks confine them to price signals. Options pools
    (`opt-<cap>:<segment>`): the gathered samples allowlist ∩ segment ranges,
    emitted only at >= min_names gathered names, with the samples path and
    fundamentals store attached (full signal registry, like Piece 1 pools).

    Every non-emitted (cap, pool, segment) slot lands in the report — spec
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_alphasearch_segments.py -v`
Expected: 15 passed (8 table tests + 7 new).

- [ ] **Step 5: Lint, full suite, commit**

```bash
uv run ruff check src/trading/alphasearch/segments.py tests/test_alphasearch_segments.py
uv run pytest -q 2>&1 | tail -5
git add src/trading/alphasearch/segments.py tests/test_alphasearch_segments.py
git commit -m "Add segment_universes: SIC segments as ordinary UniverseSpecs [AI]

Deep pools (membership x caches x SIC range, price signals only) and
options pools (gathered allowlist x range at >=15 names) per cap; every
non-emitted slot returned in an exclusions report so nothing is dropped
silently; missing sic_map raises naming the exact build command."
```

---

### Task 5: Golden segment sweep (end-to-end)

**Files:**
- Test: `tests/test_alphasearch_segments_golden.py` (create)

**Interfaces:**
- Consumes: `segment_universes` (Task 4), `UniverseSpec`/`run_sweep`/`discovery_trials`/`trials_journal` (Piece 1 + Task 3), `alphasearch_helpers.make_panel/make_factors/make_cell/month_firsts`, `SIGNALS`.
- Produces: nothing new — this task is pure verification that segment trials journal under their universe names and one BH computation spans flat + segment trials.

- [ ] **Step 1: Write the test file**

Create `tests/test_alphasearch_segments_golden.py`:

```python
"""Golden segment sweep (Piece 2 spec section 7): committed-format fixture
files -> segment_universes -> run_sweep on real panels (no panel_factory
injection). Segment trials journal under their universe names; the BH gate
spans flat + segment trials in ONE computation over the one journal."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from alphasearch_helpers import make_cell, make_factors, make_panel, month_firsts
from trading.alphasearch.segments import segment_universes
from trading.alphasearch.spec import SIGNALS
from trading.alphasearch.sweep import (
    SweepError,
    UniverseSpec,
    discovery_trials,
    run_sweep,
    trials_journal,
)

WINDOW = "2020-01-01..2020-06-30"


def _write_root(tmp_path):
    """A repo-shaped data root: make_panel()'s 16 names as parquet caches +
    a real cells samples.jsonl, all classified SIC 2836 -- so the segments
    are pharma-chemicals AND biotech (the deliberate parent/child overlap)."""
    panel = make_panel()
    cache = tmp_path / "data" / "equities-tiingo"
    cache.mkdir(parents=True)
    for sym in panel.symbols:
        closes = panel.closes[sym]
        pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes,
             "volume": 1000.0},
            index=closes.index,
        ).to_parquet(cache / f"{sym}.parquet")
    idx = panel.closes[panel.symbols[0]].index
    lines = []
    for date in month_firsts(idx):
        iso = date.date().isoformat()
        for i, sym in enumerate(panel.symbols):
            lines.append(json.dumps(make_cell(
                sym, iso,
                atm_iv=0.20 + 0.01 * i, put_iv=0.24 + 0.01 * i,
                call_iv=0.18 + 0.01 * i, skew_put_atm=0.02 + 0.005 * i,
                skew_put_call=0.01 + 0.002 * i,
            )))
    options_dir = tmp_path / "data" / "options-iv"
    options_dir.mkdir()
    samples = options_dir / "samples.jsonl"
    samples.write_text("\n".join(lines) + "\n")
    (options_dir / "samples-midcap.jsonl").write_text("")  # gathered-nothing midcap
    sic = tmp_path / "sic_map.csv"
    sic.write_text(
        "symbol,cik,sic,sic_description,fetched_at\n"
        + "".join(
            f"{s},{i + 1},2836,Biological Products,2026-07-08\n"
            for i, s in enumerate(panel.symbols)
        )
    )
    membership = tmp_path / "membership.csv"
    membership.write_text(
        "symbol,index,start,end\n"
        + "".join(f"{s},sp500,2017-01-01,\n" for s in panel.symbols)
    )
    return panel, samples, sic, membership


def test_golden_segment_sweep_journals_and_gates_across_flat_plus_segments(tmp_path):
    panel, samples, sic, membership = _write_root(tmp_path)
    seg_universes, excluded = segment_universes(tmp_path, sic, membership_path=membership)
    assert set(seg_universes) == {
        "largecap:pharma-chemicals",
        "largecap:biotech",
        "opt-largecap:pharma-chemicals",
        "opt-largecap:biotech",
    }
    # 2 caps x 2 pools x 12 segments = 48 slots, 4 emitted -> 44 reported.
    assert len(excluded) == 44
    flat = UniverseSpec(
        "largecap", tmp_path / "data" / "equities-tiingo", samples, None
    )
    journal = trials_journal(tmp_path / "journal")
    rows, n_trials = run_sweep(
        {"largecap": flat, **seg_universes}, journal, make_factors(), ts="t1",
        signals={"mom21": SIGNALS["mom21"]}, window=WINDOW,
    )
    # ONE BH computation across flat + segment trials, one honest count.
    assert n_trials == 5
    assert {(r.signal, r.universe) for r in rows} == {
        ("mom21", u) for u in ("largecap", *seg_universes)
    }
    # Same signal, different universe NAME -> distinct hashed configs/trials.
    assert len({e["config_hash"] for e in discovery_trials(journal)}) == 5
    # Identical 16-name pools: the engineered momentum spread survives on all.
    assert all(r.error is None for r in rows)
    assert all(abs(r.alpha_t) > 5 for r in rows)
    # Deep pools journal zero corrupt cells (no options file was parsed).
    deep_event = next(
        e for e in discovery_trials(journal) if e["universe"] == "largecap:biotech"
    )
    assert deep_event["corrupt_cells"] == 0


def test_options_signal_runs_on_opt_segment_but_refuses_deep_segment(tmp_path):
    _panel, _samples, sic, membership = _write_root(tmp_path)
    seg_universes, _ = segment_universes(tmp_path, sic, membership_path=membership)
    opt = seg_universes["opt-largecap:biotech"]
    deep = seg_universes["largecap:biotech"]
    journal = trials_journal(tmp_path / "journal")
    rows, n = run_sweep(
        {opt.name: opt}, journal, make_factors(), ts="t1",
        signals={"hedge": SIGNALS["hedge"]}, window=WINDOW,
    )
    assert n == 1 and rows[0].error is None  # options cells present: it runs
    with pytest.raises(SweepError, match="requires options"):
        run_sweep(
            {deep.name: deep, opt.name: opt}, journal, make_factors(), ts="t2",
            signals={"hedge": SIGNALS["hedge"]}, window=WINDOW,
        )
    # All-or-nothing assembly: the refusal journaled NOTHING new.
    assert len(discovery_trials(journal)) == 1
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_alphasearch_segments_golden.py -v 2>&1 | tee /tmp/claude-p2-task5.log`
Expected: 2 passed. (If `n_trials == 5` fails, check that `segment_universes` returned exactly the 4 expected universes — the midcap membership is empty by fixture design.)

- [ ] **Step 3: Full suite, lint, commit**

```bash
uv run ruff check tests/test_alphasearch_segments_golden.py
uv run pytest -q 2>&1 | tail -5
git add tests/test_alphasearch_segments_golden.py
git commit -m "Add golden segment sweep test [AI]

End-to-end on real fixture files: segment universes journal under their
names, one BH computation spans flat + segment trials, the parent/child
segment overlap double-counts honestly, and an options signal refuses a
deep segment at assembly without journaling anything."
```

---

### Task 6: CLI `--segments`

**Files:**
- Modify: `src/trading/cli.py:159-178` (argparse) and `src/trading/cli.py:854-899` (`_cmd_alphasearch` sweep + holdout branches)
- Test: `tests/test_alphasearch_cli.py` (append)

**Interfaces:**
- Consumes: `segment_universes` / `SegmentError` (Task 4; `SegmentError` subclasses `SweepError`, so the existing `except (engine.SweepError, PanelError)` shape catches it).
- Produces: `trading alphasearch sweep --segments` (merge + printed exclusions report on stderr); `--universe` accepts any merged universe name (validated at runtime; a bad name now exits 1 with a message instead of argparse's exit 2); `holdout signal:universe` resolves segment universe names with no flag (spec §3.3: leaderboard/holdout need no flags). Note `partition(":")` splits at the FIRST colon, so `mom21:largecap:biotech` parses as signal `mom21`, universe `largecap:biotech`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_alphasearch_cli.py` (add `from pathlib import Path` if absent):

```python
# --------------------------------------------------------------------------- #
# --segments (Piece 2)
# --------------------------------------------------------------------------- #


def _stub_segments(monkeypatch, tmp_path):
    from trading.alphasearch.sweep import UniverseSpec

    seg = UniverseSpec("largecap:banks", tmp_path, None, None, symbols=("A", "B"))
    monkeypatch.setattr(
        "trading.alphasearch.segments.segment_universes",
        lambda root, sic_map_path=None, **kwargs: (
            {"largecap:banks": seg},
            [{"segment": "construction", "cap": "opt-largecap", "count": 3,
              "reason": "below-min"}],
        ),
    )
    monkeypatch.setattr(
        "trading.alphasearch.evaluate.load_factors", lambda *a, **k: pd.DataFrame()
    )
    return seg


def test_sweep_segments_merges_universes_and_prints_exclusions(
    tmp_path, monkeypatch, capsys
):
    _stub_segments(monkeypatch, tmp_path)
    captured = {}

    def fake_run_sweep(universes, journal, factors, ts, **kwargs):
        captured["names"] = set(universes)
        return [], len(universes)

    monkeypatch.setattr("trading.alphasearch.sweep.run_sweep", fake_run_sweep)
    rc = cli.main(["alphasearch", "sweep", "--segments",
                   "--journal-dir", str(tmp_path), "--json"])
    assert rc == 0
    assert captured["names"] == {"largecap", "midcap", "largecap:banks"}
    err = capsys.readouterr().err
    assert "opt-largecap:construction" in err   # the exclusions report, on stderr
    assert "3 names" in err and "below-min" in err


def test_sweep_universe_flag_selects_a_single_segment(tmp_path, monkeypatch, capsys):
    _stub_segments(monkeypatch, tmp_path)
    captured = {}

    def fake_run_sweep(universes, journal, factors, ts, **kwargs):
        captured["names"] = set(universes)
        return [], len(universes)

    monkeypatch.setattr("trading.alphasearch.sweep.run_sweep", fake_run_sweep)
    rc = cli.main(["alphasearch", "sweep", "--segments",
                   "--universe", "largecap:banks",
                   "--journal-dir", str(tmp_path), "--json"])
    assert rc == 0
    assert captured["names"] == {"largecap:banks"}


def test_sweep_unknown_universe_lists_known_names(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "trading.alphasearch.evaluate.load_factors", lambda *a, **k: pd.DataFrame()
    )
    rc = cli.main(["alphasearch", "sweep", "--universe", "nope",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown universe" in err and "largecap" in err


def test_sweep_segments_missing_sic_map_is_an_actionable_error(
    tmp_path, monkeypatch, capsys
):
    from trading.alphasearch.segments import SegmentError

    monkeypatch.setattr(
        "trading.alphasearch.evaluate.load_factors", lambda *a, **k: pd.DataFrame()
    )

    def boom(root, sic_map_path=None, **kwargs):
        raise SegmentError("SIC map not found; build it with "
                           "`uv run python scripts/build_sic_map.py`")

    monkeypatch.setattr("trading.alphasearch.segments.segment_universes", boom)
    rc = cli.main(["alphasearch", "sweep", "--segments", "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert "build_sic_map.py" in capsys.readouterr().err


def test_holdout_resolves_segment_universe_names_without_a_flag(
    tmp_path, monkeypatch, capsys
):
    from trading.alphasearch import sweep as engine

    _stub_segments(monkeypatch, tmp_path)
    captured = {}

    def fake_run_holdout(uspec, journal, factors, ts, signal_name, **kwargs):
        captured["universe"] = uspec.name
        raise engine.SweepError("stub refusal after resolution")

    monkeypatch.setattr("trading.alphasearch.sweep.run_holdout", fake_run_holdout)
    # partition(":") splits at the FIRST colon: mom21 / largecap:banks.
    rc = cli.main(["alphasearch", "holdout", "mom21:largecap:banks",
                   "--journal-dir", str(tmp_path)])
    assert rc == 1
    assert captured["universe"] == "largecap:banks"
    assert "stub refusal" in capsys.readouterr().err
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/test_alphasearch_cli.py -v 2>&1 | tee /tmp/claude-p2-task6.log`
Expected: the 5 new tests fail (argparse exits 2 on `--segments` / non-choice `--universe` values); the 7 pre-existing tests pass.

- [ ] **Step 3: Update the argparse block**

In `src/trading/cli.py`, replace the `--universe` argument (lines 159–164) with:

```python
    alphasearch.add_argument(
        "--universe",
        default="all",
        help="sweep scope (sweep only): 'all', a flat pool (largecap/midcap), "
        "or any segment universe name when --segments is set "
        "(e.g. opt-largecap:biotech)",
    )
    alphasearch.add_argument(
        "--segments",
        action="store_true",
        help="merge the pre-registered SIC segment universes "
        "(trading.alphasearch.segments.SEGMENTS) into the sweep and print "
        "the exclusions report (sweep only)",
    )
```

(The hardcoded `choices` list is dropped; validation moves to runtime against the merged dict so segment names are selectable. A bad name now exits 1 with an actionable message instead of argparse's bare exit 2.)

- [ ] **Step 4: Update the sweep branch**

In `_cmd_alphasearch`, replace lines 868–870:

```python
        universes = engine.default_universes(Path("."))
        if args.universe != "all":
            universes = {args.universe: universes[args.universe]}
```

with:

```python
        universes = engine.default_universes(Path("."))
        if args.segments:
            from trading.alphasearch.segments import segment_universes

            try:
                seg_universes, excluded = segment_universes(Path("."))
            except engine.SweepError as exc:  # SegmentError included
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            universes = {**universes, **seg_universes}
            # Spec 3.2: below-threshold segments are REPORTED, never silent.
            for row in excluded:
                print(
                    f"segment excluded: {row['cap']}:{row['segment']} "
                    f"({row['count']} names, {row['reason']})",
                    file=sys.stderr,
                )
        if args.universe != "all":
            if args.universe not in universes:
                known = ", ".join(sorted(universes))
                print(
                    f"ERROR: unknown universe {args.universe!r}; choose from "
                    f"{known} (or 'all'; segment names need --segments)",
                    file=sys.stderr,
                )
                return 1
            universes = {args.universe: universes[args.universe]}
```

- [ ] **Step 5: Update the holdout branch**

Replace lines 889–896 (`universes = engine.default_universes(...)` through the unknown-universe `return 1`) with:

```python
    universes = engine.default_universes(Path("."))
    if universe not in universes:
        # Segment holdouts need no flag (spec 3.3: journal-derived targets);
        # resolving specs from the committed CSVs is cheap and touches no
        # panel. A missing sic_map only errors when a segment name actually
        # needs it -- flat-pool holdouts never reach this branch.
        from trading.alphasearch.segments import segment_universes

        try:
            seg_universes, _excluded = segment_universes(Path("."))
        except engine.SweepError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        universes = {**universes, **seg_universes}
    if universe not in universes:
        print(
            f"ERROR: unknown universe {universe!r}; choose from "
            f"{', '.join(sorted(universes))}",
            file=sys.stderr,
        )
        return 1
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_alphasearch_cli.py -v`
Expected: 12 passed. Note `test_holdout_unknown_universe_rejected` (pre-existing) now walks the segment-resolution branch against the REAL committed CSVs before printing "unknown universe" — it must still pass; if it errors on missing local data files, that is a bug in your `segment_universes` (it must only `exists()`-check caches, never open them, and report — not raise — on a missing samples file).

- [ ] **Step 7: Lint, full suite, commit**

```bash
uv run ruff check src/trading/cli.py tests/test_alphasearch_cli.py
uv run pytest -q 2>&1 | tail -5
git add src/trading/cli.py tests/test_alphasearch_cli.py
git commit -m "CLI: alphasearch sweep --segments [AI]

Merges the pre-registered segment universes into the sweep and prints
the exclusions report on stderr; --universe now validates at runtime so
individual segments are selectable (needed to run options signals on
opt- segments, since assembly validation is all-or-nothing); holdout
resolves segment universe names with no flag."
```

---

### Task 7: Docs — experiments.md disclosure + glossary entries

**Files:**
- Modify: `docs/experiments.md` (end of §10, before `## Known caveats affecting these numbers`)
- Modify: `docs/glossary.md` (end of the `## Multiple testing...` section)

**Interfaces:** none (prose only; keep every code/path reference exactly as the earlier tasks named them).

- [ ] **Step 1: Append the segment-sweep disclosure to experiments.md §10**

Insert after the "Classical-SE caveat" paragraph (docs/experiments.md:318-325), before `## Known caveats affecting these numbers`:

```markdown
**Segment sweep (Piece 2) — pre-registered disclosure.** `trading alphasearch
sweep --segments` expands the sweep to the frozen `SEGMENTS` table
(`trading.alphasearch.segments`; design:
`docs/superpowers/specs/2026-07-08-segmented-universes-design.md`): 10 coarse
SIC sectors + 2 fine industries (biotech SIC 2836/8731, banks SIC 6020–6039)
× 2 caps — deep pools (membership ∩ Tiingo caches, price signals only, since
they carry no options or fundamentals stores) plus options pools
(`opt-largecap:`/`opt-midcap:`) where ≥ 15 gathered names. Expected cost
≈ 200–300 new journaled trials, all discounted by the same BH q=0.10 bar over
the whole journal — segment multiplication makes the gate harder, never
easier, and a null across the entire segment sweep is a valid outcome.
Because assembly-time validation is all-or-nothing across the signal ×
universe cross-product, the sweep runs as separate passes: price signals over
everything (`--segments --signals
mom21,mom63,mom126,mom252,rev5,rvol21,disthigh`), then the options/
fundamentals families per options-pool segment (`--segments --universe
opt-largecap:biotech --signals vrp,hedge,excite,atm_iv,smile,atm_spread,...`).
Disclosed caveats on every segment result (spec §4 rule 5): (a) SIC is each
filer's **current** code applied backward over the window (no PIT
reclassification); (b) segment membership is static across the window; (c)
fine industries double-count names with their parent sector — distinct,
honestly-counted trials; (d) a symbol without a SIC mapping belongs to no
segment (never guessed; see `sic_map.csv` provenance). Below-threshold
segments (< 15 names) are excluded at build time and printed, never silently
dropped.
```

- [ ] **Step 2: Append glossary entries**

Append to the end of `docs/glossary.md` (bottom of the `## Multiple testing...` section):

```markdown
**SIC code** — Standard Industrial Classification: the 4-digit industry code the SEC
records for each filer (e.g. 2836 = Biological Products, 6021/6022 = commercial banks,
7372 = Prepackaged Software). We read each filer's *current* code from
`data.sec.gov/submissions` into the committed `sic_map.csv` and apply it backward over
the whole discovery window — a disclosed caveat, since companies occasionally
reclassify.

**Segment universe** — a pre-registered slice of a cap pool by SIC range
(`trading.alphasearch.segments.SEGMENTS`, frozen before any segment sweep): ten coarse
sectors plus the fine industries biotech and banks, which deliberately overlap their
parents. Each segment is an ordinary sweep universe, so every (signal, segment) pair is
an honestly-counted extra trial — the BH bar spans flat + segment trials in the one
journal, meaning segmentation *raises* the significance bar; it can never lower it.
```

- [ ] **Step 3: Verify nothing broke and commit**

```bash
uv run pytest -q 2>&1 | tail -5
git add docs/experiments.md docs/glossary.md
git commit -m "Docs: segment-sweep disclosure + SIC/segment glossary entries [AI]

Pre-registers the ~200-300 trial cost, the multi-pass invocation shape,
and the four SIC caveats (spec section 4 rule 5) before the first
segment sweep runs."
```

---

## Self-Review (completed at plan-write time)

1. **Spec coverage:** §3.1 build_sic_map → Task 2 (throttle seam, retry-once, interval selection, coverage gate, PROVENANCE). §3.2 SEGMENTS + segment_universes + exclusions report → Tasks 1, 4. §3.3 engine changes + CLI → Tasks 3, 6. §4 pre-registered rules → frozen-table test (Task 1), name-only hash identity (Task 3 comment + golden hash assertion), min-15 (Task 4), no-mapping-no-segment (Task 4), caveats disclosed (Tasks 2, 7). §5 trial cost → Task 7. §6 error handling → missing sic_map (Task 4), retry-once + <90% exit (Task 2), zero-symbol segment = table-validation/committed-coverage test (Tasks 1, 4), empty symbols tuple refused (Task 3). §7 testing → Tasks 1–6; existing no-look-ahead/golden suites run unchanged (Task 3 Step 6).
2. **Placeholder scan:** one deliberate operator fill-in remains — the PROVENANCE coverage numbers in Task 2 Step 7 marked `<fill in ...>`, which MUST be replaced with the real run's numbers before that commit; no other TBD/TODO/"similar to" references.
3. **Type consistency:** `UniverseSpec(name, cache_dir, samples: Path | None, fundamentals_dir: Path | None, symbols: tuple[str, ...] | None = None)` is used identically in Tasks 3, 4, 5, 6. `segment_universes(root, sic_map_path=None, *, membership_path=None, min_names=15) -> tuple[dict[str, UniverseSpec], list[dict]]` matches every call site (Tasks 4, 5, 6 including the CLI stubs' `lambda root, sic_map_path=None, **kwargs`). Exclusion-row keys `segment/cap/count/reason` match between Task 4's implementation/tests, Task 5's count assertion, and Task 6's CLI printer/test. `build_sic_map` row tuple `(symbol, cik, sic, sic_description, fetched_at)` matches `write_csv`, tests, and the committed-CSV header.
