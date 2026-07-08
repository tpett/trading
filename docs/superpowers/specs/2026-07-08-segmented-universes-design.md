# Piece 2 — Segmented Universes (design spec)

**Status:** approved design, 2026-07-08. Child of the program charter
(`2026-07-08-alpha-discovery-engine-program.md`); builds directly on Piece 1
(`2026-07-08-alpha-search-engine-design.md`, merged at 3a5b5a1).
**Scope:** Piece 2 only — run the alpha-search engine across market segments.
Robustness automation (Piece 3) and portfolio construction (Piece 4) are out of
scope.

## 1. Purpose

A signal dead in aggregate may be alive in a segment (the charter's biotech /
banks hypotheses). Piece 2 expands the sweep from two flat pools to a
pre-registered set of sector/industry segments, with per-segment leaderboard
rows, while the trial count and BH-FDR bar continue to span the entire sweep —
segment multiplication must make the gate harder, never easier.

## 2. Decisions made during brainstorming (locked)

| Question (charter §5) | Decision |
|---|---|
| Classification source | **SEC SIC codes** from the `data.sec.gov/submissions` endpoint, joined via the existing `cik_map.csv`; committed CSV, provenance-tracked |
| PIT caveat | SIC is each filer's **current** code applied backward over the window; segment membership is static across the window. Disclosed in spec + experiments.md, accepted |
| Segment base | **Deep pools for price signals** (membership ∩ Tiingo bar caches, ~713 largecap + ~705 midcap names), **options pools where viable** (segments with ≥ 15 gathered names in `samples*.jsonl`) |
| Granularity | **~10 coarse SIC sectors + 2 pre-registered fine industries (biotech, banks)**, frozen before any sweep |
| Architecture | **Segments are ordinary `UniverseSpec`s** produced by a new `segments.py`; zero changes to journal/BH/holdout machinery |
| CLI | `trading alphasearch sweep --segments` opts in; default sweep remains the Piece 1 flat pools |

## 3. Components

### 3.1 `scripts/build_sic_map.py` — classification fetch

- Input: `src/trading/fundamentals/cik_map.csv` (~1128 symbol→CIK intervals).
- Fetches `https://data.sec.gov/submissions/CIK{cik:010d}.json`, reads `sic` +
  `sicDescription`. Follows the `companyfacts.py` conventions exactly: stdlib
  `urllib`, module `USER_AGENT`, process-global 0.11 s throttle, a
  monkeypatchable `_http_get_json` seam. One full run ≈ 1128 requests ≈ 2–3 min.
- Output: **committed** `src/trading/venues/universes/sic_map.csv` with columns
  `symbol,cik,sic,sic_description,fetched_at` (one row per symbol; a symbol with
  multiple CIK intervals uses the interval overlapping the discovery window).
- Self-validating `main()` prints coverage (% of membership symbols mapped) and
  the unmapped list; updates
  `src/trading/venues/universes/sources/PROVENANCE.md`. Unmapped symbols are
  simply absent from every segment (never guessed).

### 3.2 `src/trading/alphasearch/segments.py` — the pre-registered table

- `SEGMENTS: dict[str, SegmentDef]` — frozen at spec/plan time. `SegmentDef`
  carries the SIC code ranges (inclusive int ranges over the 4-digit code) and a
  `kind: "sector" | "industry"` marker.
- Sector rollup (~10 buckets over SIC divisions; exact ranges pinned in the
  implementation plan and validated by tests): energy/mining, construction,
  manufacturing-consumer, pharma/chemicals, manufacturing-tech (electronics,
  instruments), other-manufacturing, transport/utilities, trade
  (wholesale+retail), finance, services.
- Fine industries (charter hypotheses): **biotech** (SIC 2836, 8731) and
  **banks** (SIC 6020–6039). Fine industries overlap their parent sectors by
  design — a name may appear in both; these are distinct, honestly-counted
  trials.
- `segment_universes(root, sic_map, *, min_names=15) -> dict[str, UniverseSpec]`:
  - **Deep-pool segments** per cap: symbols = membership-CSV ever-members
    overlapping the discovery window ∩ cached bar parquets ∩ segment SIC range.
    `samples=None` (no options data → options/fundamentals signals are refused
    by Piece 1's existing assembly-time check). Names:
    `largecap:<segment>` / `midcap:<segment>`.
  - **Options-pool segments**: filter each samples allowlist by segment; only
    segments with ≥ `min_names` gathered symbols are emitted, as
    `opt-largecap:<segment>` / `opt-midcap:<segment>` with the samples path set.
  - Segments below `min_names` at build time are **excluded and returned in an
    exclusions report** (segment, cap, count) that the CLI prints — never
    silently dropped.

### 3.3 Engine changes (deliberately minimal)

- `UniverseSpec.samples: Path | None` (was required) and a new
  `symbols: tuple[str, ...] | None` — an explicit symbol set that
  `build_panel` accepts as the universe (overriding the samples-allowlist
  derivation). No other change to `sweep.py`'s machinery: a segment is just a
  universe name inside the hashed trial config, so
  - BH-FDR automatically spans flat + segment trials (one journal),
  - holdout stays once per (signal, segment-universe) with the same
    confirmation ceremony,
  - re-running a segment sweep dedupes by config hash as before.
- CLI: `trading alphasearch sweep --segments` merges
  `segment_universes(...)` into the default universes dict and prints the
  exclusions report. `leaderboard` and `holdout` need no flags — journal-derived.

## 4. Pre-registered rules (amend only in writing, prospectively)

1. The `SEGMENTS` table (names + SIC ranges) is frozen with this spec's
   implementation plan. Adding, removing, or re-ranging a segment after the
   first segment sweep is a written prospective amendment; the journal keeps
   counting everything ever run.
2. Segment trials use the identical gate machinery as Piece 1 (4F L/S alpha t,
   discovery 2019-01-01..2023-12-31, BH q=0.10 across the whole journal,
   touched-once holdout 2024+). No per-segment FDR reset.
3. Minimum cross-section: segments below 15 names are excluded at build time
   (reported). Within a sweep the existing per-date skip/tercile rules apply
   unchanged (most segments will run terciles, being < 50 names — expected and
   fine).
4. A symbol without a SIC mapping belongs to no segment (no imputation).
5. Disclosed caveats attached to every segment result: current-SIC-applied-
   backward; static membership over the window; fine industries double-count
   names with their parent sector.

## 5. Trial cost (disclosed)

~10 sectors × 2 caps clearing the floor (some won't) + 2 fine industries × caps
+ ~3–6 options-viable segments, × ~10 price signals (× 6 options signals on
options segments) ≈ **200–300 new journaled trials**. All discounted by the same
BH bar. A null across the whole segment sweep is a valid outcome.

## 6. Error handling

- `sic_map.csv` missing/stale → `segment_universes` raises with the exact
  build command (`uv run python scripts/build_sic_map.py`).
- SEC fetch failures: per-symbol retry-once then recorded unmapped (script
  exits non-zero if coverage < 90% of membership).
- A segment emitting zero symbols (range typo) fails table-validation tests,
  not runtime.
- Empty `symbols` tuple on a UniverseSpec → refused at assembly (mirrors the
  existing empty-signals refusal).

## 7. Testing

- SEGMENTS table validation: ranges well-formed and typed, sector names stable,
  biotech/banks ranges exactly as pre-registered here.
- `segment_universes`: fixture sic_map + fixture caches → expected segment
  membership, min-15 exclusion + report, options-segment viability threshold,
  no-mapping-no-segment.
- `build_panel` with explicit `symbols` and with `samples=None` (options
  accessors absent → requires_options refusal path covered end-to-end).
- Golden segment sweep on fixture data: segment universes journal under their
  names, BH mask spans flat + segment trials in one computation.
- `build_sic_map.py`: mocked `_http_get_json` seam — parse, throttle-seam
  presence, coverage validation, interval selection.
- Existing no-look-ahead + golden guarantees run unchanged (segments reuse
  PanelView/PanelData verbatim).
- Repo discipline: full `uv run pytest` warnings-as-errors, `uv run ruff
  check`, granular `[AI]` commits, subagent-driven implementation + adversarial
  review per charter §6.

## 8. Out of scope (deferred)

Historical/PIT reclassification of SIC codes; GICS or vendor taxonomies; size-
bucket segments; regime-conditional segments; per-segment robustness gates
(Piece 3); blending segment survivors (Piece 4).
