"""CLI: print the options gather v2-vs-v1 coverage/agreement report as JSON.

Run once per pool after the re-gather (spec section 5):

    uv run python scripts/options_coverage_report.py \
        --v2 data/options-iv/samples.jsonl --v1 data/options-iv/samples.v1.jsonl
    uv run python scripts/options_coverage_report.py \
        --v2 data/options-iv/samples-midcap.jsonl --v1 data/options-iv/samples-midcap.v1.jsonl

iv_red_flag=true (median |iv_v2 - iv_v1| > 0.01 on the same contract/date)
means investigate before trusting the data -- do not proceed to signals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trading.research.options_coverage import coverage_report, load_cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, required=True, help="regathered samples.jsonl")
    parser.add_argument("--v1", type=Path, required=True, help="the .v1 backup baseline")
    args = parser.parse_args(argv)
    report = coverage_report(load_cells(args.v2), load_cells(args.v1))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
