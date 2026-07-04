from trading.digest import build_digest, collect_run_events, write_digest
from trading.journal import Journal


def _event(venue="equities", ts="2026-07-01T22:30:00+00:00", value=1012.34, n=1) -> dict:
    return {
        "event": "run",
        "venue": venue,
        "run_key": f"{venue}:2026-07-01T00:00:00+00:00",
        "ts": ts,
        "decision_ts": "2026-07-01T00:00:00+00:00",
        "config_hash": "abc123def456",
        "regime": {"state": "risk_on", "exposure_multiplier": 1.0},
        "coverage": {"requested": 4, "fetched": 4, "ratio": 1.0},
        "benchmark": {"symbol": "SPY", "close": 624.0, "start_price": 620.0},
        "starting_balance": 1000.0,
        "ranking": [
            {"rank": 1, "symbol": "UPUP", "status": "tradable", "composite": 0.83},
            {"rank": 2, "symbol": "FLAT", "status": "tradable", "composite": 0.55},
        ],
        "fills": [
            {
                "symbol": "UPUP",
                "side": "buy",
                "qty": 1.7982,
                "price": 100.05,
                "fee": 0.0,
                "bar_ts": "2026-07-01T00:00:00+00:00",
                "reason": "entry",
                "realized_pnl": None,
            }
        ],
        "new_orders": [
            {
                "symbol": "MEH1",
                "side": "sell",
                "notional": 0.0,
                "decision_ts": "2026-07-01T00:00:00+00:00",
                "reason": "stop_loss",
                "atr_at_decision": 0.0,
                "composite": 0.0,
                "rank": 0,
            }
        ],
        "skips": [{"symbol": "*", "action": "entry", "reason": "stale_run_entries_skipped"}],
        "warnings": ["quarantined: BADCO"],
        "snapshot": {
            "value": value,
            "cash": 640.0,
            "unsettled": 180.0,
            "positions": [
                {
                    "symbol": "UPUP",
                    "qty": 1.7982,
                    "entry_price": 100.05,
                    "last_close": 107.0,
                    "market_value": 192.4,
                    "unrealized_pnl_pct": 0.0695,
                    "stop_price": 94.05,
                    "stop_distance_pct": 0.121,
                    "entry_rank": 1,
                    "entry_composite": 0.83,
                    "entry_ts": "2026-07-01T00:00:00+00:00",
                }
            ],
        },
        "state_after": {"breaker_tripped": True},
        "n": n,
    }


def test_build_digest_renders_all_sections():
    text = build_digest("2026-07-01", [_event()])
    assert "# Trading digest — 2026-07-01 (UTC)" in text
    assert "## equities — risk_on (exposure x1.0)" in text
    assert "$1,012.34" in text and "+1.23%" in text  # portfolio value + P&L since start
    assert "SPY buy-and-hold: +0.65%" in text  # 624/620 - 1
    assert "| UPUP |" in text and "rank #1, composite 0.83" in text
    assert "buy" in text and "stop_loss" in text
    assert "quarantined: BADCO" in text
    assert "late run: entries skipped" in text
    assert "TRIPPED" in text


def test_build_digest_without_events():
    text = build_digest("2026-07-01", [])
    assert "No runs journaled" in text


def test_collect_run_events_takes_last_run_per_venue_for_date(tmp_path):
    journal = Journal(tmp_path / "journal" / "equities.jsonl")
    journal.append({"event": "bootstrap", "venue": "equities", "ts": "2026-07-01T22:00:00+00:00"})
    journal.append(_event(n=1))
    journal.append(_event(n=2))
    journal.append(_event(ts="2026-06-30T22:30:00+00:00", n=3))  # other date
    events = collect_run_events(tmp_path / "journal", ["equities", "crypto"], "2026-07-01")
    assert [e["n"] for e in events] == [2]  # latest same-date run; crypto absent


def test_write_digest_creates_dated_file(tmp_path):
    path = write_digest(tmp_path / "digest", "2026-07-01", [_event()])
    assert path == tmp_path / "digest" / "2026-07-01.md"
    assert "equities" in path.read_text()
