"""Golden-backtest fixture generator (spec: Testing Strategy / Golden backtest).

Writes tests/golden/bars/<SYMBOL>.csv (seeded random walks and shaped
piecewise-drift paths -- fully deterministic), tests/golden/universe.csv
(per-symbol status flips), and, with --write-expected,
tests/golden/expected.json from the CURRENT engine. Regenerating
expected.json is a deliberate act: do it only when a behavior change is
intended, and review the diff in the commit. The golden test exists to fail
when results drift unintentionally.

The fixture is shaped so the frozen expectation exercises every exit path:
  SLC ramps then crashes hard        -> stop_loss
  TSF spikes then drifts flat/down   -> time_stop
  FEX ramps, flips untradable Apr 1  -> forced_exit
  background walkers churn           -> trend_break
  LAT lists Feb 18 (after start)     -> skipped sessions (coverage floor)
Regeneration fails loudly if any of those paths goes missing.

Usage:
  uv run python scripts/gen_golden_fixture.py                  # bars only
  uv run python scripts/gen_golden_fixture.py --write-expected # bars + expected
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "golden"
PERIODS = 150
START = "2025-01-01"

# Background random walkers: symbol -> (seed, constant drift).
BACKGROUND = {
    "AAA": (1, 0.006),
    "BBB": (2, 0.002),
    "CCC": (3, 0.0),
    "DDD": (4, -0.002),
    "EEE": (5, 0.004),
}
# Shaped paths: symbol -> (seed, segments of (days, drift, noise scale)).
# Days must sum to PERIODS.
SHAPED = {
    # Ramp into an entry, then a two-day 18%/day crash: day one already
    # breaches the frozen ATR stop, which is checked BEFORE trend_break, so
    # the momentum collapse cannot turn this into a trend-break exit.
    "SLC": (11, [(54, 0.014, 0.008), (2, -0.18, 0.004), (94, 0.0, 0.008)]),
    # Time-stop by construction: flat while the three slots fill, a 4-day
    # ramp while they are still full, then an EXACTLY flat glide starting
    # the session SLC's stop-out frees a slot. TSF is entered at the flat
    # level, so close <= entry every bar; close >= 5-bar mean blocks
    # trend_break while ranked, and five identical zero returns then zero
    # the vol window (momentum NaN -> held-but-unranked, the engine's
    # documented degenerate-input path) until the time stop fires at bar 8.
    "TSF": (
        12,
        [
            (51, 0.0, 0.005),
            (4, 0.014, 0.002),
            (16, 0.0, 0.0),
            (79, 0.003, 0.008),
        ],
    ),
    # Steady strong ramp: held when its status flips to untradable.
    "FEX": (13, [(150, 0.010, 0.008)]),
}
# Late lister: symbol -> (seed, first bar's day offset from START, drift).
# 8 of 9 members before this date -> coverage 89% < 90% -> skipped sessions.
LATE = {"LAT": (14, 48, 0.001)}
BENCH_SEED = 99
BENCH_DRIFT = 0.002
# Status flips written to universe.csv: symbol -> first untradable date.
UNTRADABLE_FROM = {"FEX": "2025-04-01"}

REQUIRED_REASONS = {"stop_loss", "time_stop", "trend_break", "forced_exit"}


def _frame(rng: np.random.Generator, returns: np.ndarray, start_day: int) -> pd.DataFrame:
    idx = pd.date_range(START, periods=PERIODS, freq="D", tz="UTC")[start_day:]
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.concatenate([[100.0], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close,
            "volume": rng.uniform(5e5, 1.5e6, size=len(returns)),
        },
        index=idx,
    )


def build_frame(seed: int, drift: float, start_day: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=0.02, size=PERIODS - start_day)
    return _frame(rng, returns, start_day)


def build_shaped_frame(seed: int, segments: list[tuple[int, float, float]]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = np.concatenate(
        [rng.normal(loc=drift, scale=scale, size=days) for days, drift, scale in segments]
    )
    if len(returns) != PERIODS:
        sys.exit(f"FATAL: shaped segments sum to {len(returns)} days, want {PERIODS}")
    return _frame(rng, returns, 0)


def _seeds() -> dict[str, int]:
    seeds = {sym: spec[0] for sym, spec in {**BACKGROUND, **SHAPED, **LATE}.items()}
    seeds["BENCH"] = BENCH_SEED
    return dict(sorted(seeds.items()))


def _engine_version() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-expected", action="store_true")
    parser.add_argument(
        "--engine-version",
        default=None,
        help="provenance stamp; defaults to `git rev-parse HEAD`",
    )
    args = parser.parse_args()
    bars_dir = GOLDEN / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)
    for symbol, (seed, drift) in sorted(BACKGROUND.items()):
        build_frame(seed, drift).to_csv(bars_dir / f"{symbol}.csv")
    for symbol, (seed, segments) in sorted(SHAPED.items()):
        build_shaped_frame(seed, segments).to_csv(bars_dir / f"{symbol}.csv")
    for symbol, (seed, start_day, drift) in sorted(LATE.items()):
        build_frame(seed, drift, start_day).to_csv(bars_dir / f"{symbol}.csv")
    build_frame(BENCH_SEED, BENCH_DRIFT).to_csv(bars_dir / "BENCH.csv")
    members = sorted({**BACKGROUND, **SHAPED, **LATE})
    lines = ["symbol,untradable_from"]
    lines += [f"{symbol},{UNTRADABLE_FROM.get(symbol, '')}" for symbol in members]
    (GOLDEN / "universe.csv").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(members) + 1} fixture frames to {bars_dir}")
    if args.write_expected:
        sys.path.insert(0, str(ROOT / "tests"))
        from golden_helpers import run_golden

        with tempfile.TemporaryDirectory() as tmp:
            expected = run_golden(Path(tmp) / "cache")
        expected["provenance"] = {
            "generated_by": "scripts/gen_golden_fixture.py",
            "seeds": _seeds(),
            "engine_version": args.engine_version or _engine_version(),
        }
        (GOLDEN / "expected.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        print(f"wrote {GOLDEN / 'expected.json'} ({len(expected['trades'])} trades)")
        reasons = sorted({trade["reason"] for trade in expected["trades"]})
        print(f"exit reasons: {reasons}; skipped sessions: {expected['sessions_skipped']}")
        missing = REQUIRED_REASONS - set(reasons)
        if missing:
            sys.exit(f"FATAL: golden fixture lost exit path(s) {sorted(missing)} -- reshape SHAPED")
        if not expected["sessions_skipped"]:
            sys.exit("FATAL: golden fixture has no skipped session -- delay LAT's listing")


if __name__ == "__main__":
    main()
