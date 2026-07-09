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
    build_leaderboard,
    discovery_trials,
    log_trial,
    run_sweep,
    trial_config,
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


def _seed_trial(journal, universe: str, p: float) -> None:
    """One hand-built discovery trial event carrying exactly the ls fields
    build_leaderboard reads for the BH mask."""
    log_trial(
        journal, kind="discovery",
        config=trial_config("mom21", universe, WINDOW), ts="t1",
        result={"ls": {"alpha_annual_pct": 5.0, "alpha_t": 1.9, "p": p}},
    )


def test_bh_mask_spans_combined_journal_not_per_segment(tmp_path):
    """Boundary construction: a segment trial at p=0.08 clears BH alone
    (m=1: 0.08 <= 0.10) but must FAIL in the combined 5-trial journal
    (k=1 threshold 0.10/5 = 0.02, and the p=0.5 flat trials never rescue a
    higher k). A per-segment FDR reset would flip the combined assertion."""
    alone = trials_journal(tmp_path / "journal-alone")
    _seed_trial(alone, "largecap:biotech", 0.08)
    rows, n = build_leaderboard(alone)
    assert n == 1
    assert rows[0].universe == "largecap:biotech" and rows[0].bh_pass

    combined = trials_journal(tmp_path / "journal-combined")
    _seed_trial(combined, "largecap:biotech", 0.08)
    for i in range(4):
        _seed_trial(combined, f"flat-{i}", 0.5)
    rows, n = build_leaderboard(combined)
    assert n == 5
    seg = next(r for r in rows if r.universe == "largecap:biotech")
    assert seg.p == 0.08 and not seg.bh_pass
    assert not any(r.bh_pass for r in rows)
