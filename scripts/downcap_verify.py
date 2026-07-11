"""Compute the Phase-A GO/NO-GO gate from the diagnostics artifact and write
the human report. Returns exit 0 on GO, 2 on NO-GO (still writes the report
and, if triggered, the fallback amendment).

    uv run python scripts/downcap_verify.py
    # dollar-volume-only fallback verdict (the amendment's re-run, only
    # after a cap-mode NO-GO on shares-coverage records the amendment):
    uv run python scripts/downcap_verify.py --fallback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from trading.venues.universes.downcap_verify import compute_gate, render_report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics", default="data/equities-downcap-tiingo/diagnostics.csv")
    p.add_argument("--out", default="docs/reports/downcap_phase_a.md")
    p.add_argument(
        "--fallback", action="store_true",
        help="compute the dollar-volume-only fallback verdict (require_cap_band=False) "
        "instead of the market-cap-band gate",
    )
    args = p.parse_args()

    diagnostics = pd.read_csv(args.diagnostics)
    # CSV round-trips bools as strings/objects; coerce the boolean columns.
    for col in ("delisted", "tradeable", "has_shares"):
        diagnostics[col] = diagnostics[col].astype(str).str.lower().isin(("true", "1"))
    gate = compute_gate(diagnostics, require_cap_band=not args.fallback)
    report = render_report(gate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n")
    print(report)
    return 0 if gate.go else 2


if __name__ == "__main__":
    sys.exit(main())
