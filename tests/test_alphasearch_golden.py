"""Golden end-to-end sweep (spec section 7): real files -> build_panel ->
sort -> regression -> journal -> leaderboard, on a deterministic fixture."""

from __future__ import annotations

import json

import pandas as pd

from alphasearch_helpers import make_cell, make_factors, make_panel, month_firsts
from trading.alphasearch.spec import SIGNALS
from trading.alphasearch.sweep import (
    UniverseSpec,
    discovery_trials,
    run_sweep,
    trials_journal,
)
from trading.fundamentals.store import FundamentalsStore

WINDOW = "2020-01-01..2020-06-30"


def _write_universe(tmp_path) -> UniverseSpec:
    """Materialize make_panel()'s exact data as real files: parquet bar
    caches, a samples.jsonl, and a fundamentals store."""
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
    samples = tmp_path / "samples.jsonl"
    samples.write_text("\n".join(lines) + "\n")
    store = FundamentalsStore(tmp_path / "fundamentals")
    for sym in panel.symbols:
        store.append(sym, panel.fundamentals[sym])
    return UniverseSpec("largecap", cache, samples, tmp_path / "fundamentals")


def test_golden_sweep_end_to_end(tmp_path):
    uspec = _write_universe(tmp_path)
    journal = trials_journal(tmp_path / "journal")
    rows, n_trials = run_sweep({"largecap": uspec}, journal, make_factors(),
                               ts="t1", window=WINDOW)

    # Every registered signal became exactly one journaled trial.
    assert n_trials == len(SIGNALS) == 16
    assert {(r.signal, r.universe) for r in rows} == {
        (s, "largecap") for s in SIGNALS
    }
    # Deep-history momentum cannot exist on a 6-month fixture: honest errors
    # that are flagged AND still spend trials.
    errored = {r.signal for r in rows if r.error is not None}
    assert {"mom126", "mom252"} <= errored
    # The engineered momentum spread is a standout survivor with full stats.
    mom = next(r for r in rows if r.signal == "mom21")
    assert mom.bh_pass
    assert abs(mom.alpha_t) > 5
    assert mom.dsr is not None and 0.0 <= mom.dsr <= 1.0
    assert set(mom.loadings) == {"Mkt-RF", "SMB", "HML", "Mom"}
    assert mom.turnover_monthly is not None and 0.0 <= mom.turnover_monthly <= 0.6
    assert mom.n_names_median == 16.0
    # Ranking is by |4F L/S t| descending.
    ts = [abs(r.alpha_t) for r in rows if r.alpha_t is not None]
    assert ts == sorted(ts, reverse=True)

    # Identical re-run: journal grows (append-only) but nothing double-counts
    # and the statistics are bit-identical.
    rows2, n2 = run_sweep({"largecap": uspec}, journal, make_factors(),
                          ts="t2", window=WINDOW)
    assert n2 == n_trials
    assert [(r.signal, r.alpha_t, r.p) for r in rows2] == [
        (r.signal, r.alpha_t, r.p) for r in rows
    ]
    assert len(list(journal.events())) == 32
    assert len(discovery_trials(journal)) == 16
