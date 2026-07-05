"""Golden-backtest fixture generator (spec: Testing Strategy / Golden backtest).

Writes tests/golden/bars/<SYMBOL>.csv (seeded random walks -- fully
deterministic) and, with --write-expected, tests/golden/expected.json from the
CURRENT engine. Regenerating expected.json is a deliberate act: do it only
when a behavior change is intended, and review the diff in the commit. The
golden test exists to fail when results drift unintentionally.

Usage:
  uv run python scripts/gen_golden_fixture.py                  # bars only
  uv run python scripts/gen_golden_fixture.py --write-expected # bars + expected
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden"
SYMBOLS = {"AAA": 0.006, "BBB": 0.002, "CCC": 0.0, "DDD": -0.002, "EEE": 0.004, "FFF": -0.001}
PERIODS = 150
START = "2025-01-01"


def build_frame(seed: int, drift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=PERIODS, freq="D", tz="UTC")
    returns = rng.normal(loc=drift, scale=0.02, size=PERIODS)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.concatenate([[100.0], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.uniform(5e5, 1.5e6, size=PERIODS),
        },
        index=idx,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-expected", action="store_true")
    args = parser.parse_args()
    bars_dir = GOLDEN / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)
    for n, (symbol, drift) in enumerate(sorted(SYMBOLS.items()), start=1):
        build_frame(seed=n, drift=drift).to_csv(bars_dir / f"{symbol}.csv")
    build_frame(seed=99, drift=0.002).to_csv(bars_dir / "BENCH.csv")
    print(f"wrote {len(SYMBOLS) + 1} fixture frames to {bars_dir}")
    if args.write_expected:
        sys.path.insert(0, str(ROOT / "tests"))
        from golden_helpers import run_golden

        with tempfile.TemporaryDirectory() as tmp:
            expected = run_golden(Path(tmp) / "cache")
        (GOLDEN / "expected.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        print(f"wrote {GOLDEN / 'expected.json'} ({len(expected['trades'])} trades)")
        if not expected["trades"] and not expected["open_positions"]:
            sys.exit("FATAL: golden fixture produced zero trades -- raise drifts/lower threshold")


if __name__ == "__main__":
    main()
