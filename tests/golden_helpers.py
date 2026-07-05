"""Golden backtest: load the committed fixture and run the real engine.

Shared by tests/test_golden_backtest.py and scripts/gen_golden_fixture.py so
the expected output and the assertion can never diverge in HOW they run.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from trading.backtest.engine import prepare, replay
from trading.backtest.metrics import compute_metrics
from trading.config import load_venue_config
from trading.data.cache import OhlcvCache
from trading.venues.base import SymbolInfo, VenueConstraints, validate_ohlcv

GOLDEN = Path(__file__).parent / "golden"
GOLDEN_END = datetime.date(2025, 5, 30)


class GoldenAdapter:
    def __init__(self) -> None:
        self._frames = {
            path.stem: self._load(path) for path in sorted((GOLDEN / "bars").glob("*.csv"))
        }
        # universe.csv: symbol,untradable_from ("" = tradable forever). The
        # status flip mid-fixture exercises the forced-exit path.
        with (GOLDEN / "universe.csv").open() as handle:
            self._untradable_from = {
                row["symbol"]: (
                    datetime.date.fromisoformat(row["untradable_from"])
                    if row["untradable_from"]
                    else None
                )
                for row in csv.DictReader(handle)
            }

    @staticmethod
    def _load(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True))
        return validate_ohlcv(df[["open", "high", "low", "close", "volume"]].astype("float64"))

    def universe(self, as_of: datetime.date) -> list[SymbolInfo]:
        infos = []
        for symbol, flip in sorted(self._untradable_from.items()):
            status = "untradable" if flip is not None and as_of >= flip else "tradable"
            infos.append(SymbolInfo(symbol=symbol, status=status))
        return infos

    def constraints(self) -> VenueConstraints:
        return VenueConstraints(
            taker_fee_bps=25.0,
            maker_fee_bps=0.0,
            slippage_bps=5.0,
            settlement_days=0,
            trades_24_7=True,
        )

    def fetch_ohlcv(self, symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
        return self._frames[symbol].loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]


def _round(value: object) -> object:
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 8)
    return value


def _curve_digest(curve: pd.Series) -> dict:
    values = [float(v) for v in curve]
    pairs = [[ts.date().isoformat(), _round(v)] for ts, v in zip(curve.index, values, strict=True)]
    canonical = json.dumps(pairs, separators=(",", ":"))
    return {
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "first": _round(values[0]),
        "last": _round(values[-1]),
        "min": _round(min(values)),
        "max": _round(max(values)),
    }


def run_golden(cache_dir: Path) -> dict:
    config = load_venue_config("golden", GOLDEN)
    adapter = GoldenAdapter()
    cache = OhlcvCache(cache_dir, config.data.refetch_days)
    prepared = prepare(config, adapter, cache, config.backtest.start, GOLDEN_END)
    result = replay(prepared, config)
    metrics = compute_metrics(result, config.backtest.periods_per_year)
    skipped = []
    for entry in result.sessions_skipped:  # "YYYY-MM-DD: reason ..."
        date, _, reason = entry.partition(": ")
        skipped.append([date, reason.split(" ", 1)[0]])
    return {
        "final_value": _round(float(result.equity_curve.iloc[-1])),
        "equity_curve": _curve_digest(result.equity_curve),
        "sessions_run": result.sessions_run,
        "sessions_skipped": skipped,
        "open_positions": list(result.open_positions),
        "trades": [
            {key: _round(value) for key, value in asdict(trade).items()} for trade in result.trades
        ],
        "metrics": {key: _round(value) for key, value in asdict(metrics).items()},
    }
