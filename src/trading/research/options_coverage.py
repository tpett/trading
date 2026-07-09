"""Coverage/agreement report: options gather v2 vs the v1 backup (spec
2026-07-09-options-gather-v2-design.md section 5).

Answers, per pool, before the regathered data is trusted: how many cells; what
fraction carry leg volume (large-cap must jump from 0% to ~mid-cap levels),
open interest, and a far block; and whether IV agrees with v1 on the SAME
contract/date (large drift is a red flag to investigate, not accept). There is
no greeks coverage column: both vendor greeks endpoints 404 on this tier (Task
1 discovery), so no greeks capture path exists anywhere in the gather. Pure
functions over parsed cells -- no network, no pandas needed.

Two denominators, named explicitly: ``*_cell_rate`` fields are fractions of
CELLS (any near leg satisfies the predicate), ``*_leg_rate`` fields are
fractions of NEAR LEGS; the report's ``_denominators`` note restates this so
the CLI's JSON output is self-describing.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

# Median |iv_v2 - iv_v1| on the SAME (symbol, date, role, strike, expiration)
# above one vol point means the two gathers priced the same contract
# differently -- investigate (quote-window drift, spot mismatch) before use.
IV_DRIFT_RED_FLAG = 0.01


def load_cells(path: Path) -> list[dict]:
    """Tolerant samples.jsonl reader (the load_existing_keys discipline):
    torn/keyless lines are skipped, an absent file is just empty."""
    cells: list[dict] = []
    if not path.exists():
        return cells
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if len(line) == 0:
                continue
            try:
                cell = json.loads(line)
            except ValueError:
                continue
            if isinstance(cell, dict) and cell.get("symbol") and cell.get("decision_date"):
                cells.append(cell)
    return cells


def _near_legs(cells: list[dict]) -> list[dict]:
    return [c for cell in cells for c in cell.get("contracts", [])]


def _leg_rate(cells: list[dict], key: str) -> float | None:
    """Fraction of near legs carrying ``key`` with a non-None value; None when
    there are no legs at all (empty pool -> no rate, not a fake 0)."""
    legs = _near_legs(cells)
    if len(legs) == 0:
        return None
    return sum(1 for c in legs if c.get(key) is not None) / len(legs)


def _cell_rate(cells: list[dict], predicate) -> float | None:
    if len(cells) == 0:
        return None
    return sum(1 for cell in cells if predicate(cell)) / len(cells)


def iv_deltas(v2_cells: list[dict], v1_cells: list[dict]) -> list[float]:
    """|iv_v2 - iv_v1| per overlapping near leg: the SAME
    (symbol, decision_date, role, strike, target_expiration) where both
    gathers inverted an IV. A leg the v2 walk resolved to a different strike
    or expiration is a different contract and deliberately does not match."""

    def index(cells: list[dict]) -> dict[tuple, float]:
        out: dict[tuple, float] = {}
        for cell in cells:
            for c in cell.get("contracts", []):
                iv = c.get("iv")
                if iv is None:
                    continue
                key = (
                    cell.get("symbol"),
                    cell.get("decision_date"),
                    c.get("role"),
                    c.get("strike"),
                    cell.get("target_expiration"),
                )
                out[key] = float(iv)
        return out

    v1 = index(v1_cells)
    v2 = index(v2_cells)
    return [abs(v2[key] - v1[key]) for key in v2.keys() & v1.keys()]


def coverage_report(v2_cells: list[dict], v1_cells: list[dict]) -> dict:
    """The spec-section-5 verification numbers, JSON-serializable."""
    deltas = iv_deltas(v2_cells, v1_cells)
    iv_median = median(deltas) if len(deltas) > 0 else None
    return {
        "cells_v2": len(v2_cells),
        "cells_v1": len(v1_cells),
        "volume_cell_rate": _cell_rate(
            v2_cells,
            lambda cell: any(
                c.get("volume") is not None for c in cell.get("contracts", [])
            ),
        ),
        "volume_leg_rate": _leg_rate(v2_cells, "volume"),
        "oi_leg_rate": _leg_rate(v2_cells, "open_interest"),
        "far_rate": _cell_rate(v2_cells, lambda cell: "far" in cell),
        "iv_overlap_legs": len(deltas),
        "iv_median_abs_delta": iv_median,
        "iv_red_flag": iv_median is not None and iv_median > IV_DRIFT_RED_FLAG,
        "_denominators": (
            "volume_cell_rate/far_rate are fractions of v2 CELLS; "
            "volume_leg_rate/oi_leg_rate are fractions of v2 NEAR LEGS"
        ),
    }
