"""Daily digest (spec: Reporting & Operations): a 60-second markdown read.

Built purely from journal run events, so `trading digest` shows exactly what
the pipeline decided — value + P&L vs benchmark buy-and-hold, open positions
with entry rationale and distance-to-stop, fills, top-5 ranking, regime, and
warnings (quarantines, staleness, breaker state, earnings degradation).
"""

from __future__ import annotations

from pathlib import Path

from trading.journal import Journal


def collect_run_events(journal_root: Path, venues: list[str], date_iso: str) -> list[dict]:
    events: list[dict] = []
    for venue in venues:
        journal = Journal(journal_root / f"{venue}.jsonl")
        last: dict | None = None
        for event in journal.events():
            if event.get("event") == "run" and str(event.get("ts", "")).startswith(date_iso):
                last = event
        if last is not None:
            events.append(last)
    return events


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _pct(x: float) -> str:
    return f"{x:+.2%}"


def _venue_section(event: dict) -> list[str]:
    snapshot = event["snapshot"]
    start = float(event["starting_balance"])
    value = float(snapshot["value"])
    bench = event["benchmark"]
    bench_pnl = float(bench["close"]) / float(bench["start_price"]) - 1.0
    regime = event["regime"]
    breaker = bool(event["state_after"]["breaker_tripped"])

    lines = [
        f"## {event['venue']} — {regime['state']} (exposure x{regime['exposure_multiplier']})",
        "",
        f"- Portfolio: {_money(value)} ({_pct(value / start - 1.0)} since start) | "
        f"{bench['symbol']} buy-and-hold: {_pct(bench_pnl)}",
        f"- Cash: {_money(float(snapshot['cash']))} settled, "
        f"{_money(float(snapshot['unsettled']))} unsettled | "
        f"breaker: {'TRIPPED' if breaker else 'armed'}",
        "",
        "### Open positions",
    ]
    if snapshot["positions"]:
        lines += [
            "| symbol | qty | entry | last | P&L | stop | to stop | rationale |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for m in snapshot["positions"]:
            lines.append(
                f"| {m['symbol']} | {m['qty']:.4f} | {m['entry_price']:.2f} "
                f"| {m['last_close']:.2f} | {_pct(m['unrealized_pnl_pct'])} "
                f"| {m['stop_price']:.2f} | {_pct(m['stop_distance_pct'])} "
                f"| rank #{m['entry_rank']}, composite {m['entry_composite']:.2f} |"
            )
    else:
        lines.append("- none")

    lines += ["", "### Today's fills"]
    if event["fills"]:
        lines += [
            "| symbol | side | qty | price | fee | reason | realized P&L |",
            "|---|---|---|---|---|---|---|",
        ]
        for f in event["fills"]:
            realized = "-" if f["realized_pnl"] is None else _money(float(f["realized_pnl"]))
            lines.append(
                f"| {f['symbol']} | {f['side']} | {f['qty']:.4f} | {f['price']:.2f} "
                f"| {f['fee']:.2f} | {f['reason']} | {realized} |"
            )
    else:
        lines.append("- none")

    lines += ["", "### New orders for the next bar"]
    if event["new_orders"]:
        lines += [f"- {o['side']} {o['symbol']} ({o['reason']})" for o in event["new_orders"]]
    else:
        lines.append("- none")

    lines += ["", "### Top 5 ranking", "| # | symbol | composite | status |", "|---|---|---|---|"]
    for row in event["ranking"][:5]:
        lines.append(f"| {row['rank']} | {row['symbol']} | {row['composite']} | {row['status']} |")

    warnings = list(event["warnings"])
    if any(s["reason"].startswith("stale_run") for s in event["skips"]):
        warnings.append("late run: entries skipped (staleness bound exceeded)")
    if breaker:
        warnings.append("circuit breaker is TRIPPED: entries halted until `trading reset-breaker`")
    lines += ["", "### Warnings"]
    lines += [f"- {w}" for w in warnings] if warnings else ["- none"]
    lines.append("")
    return lines


def build_digest(date_iso: str, run_events: list[dict]) -> str:
    lines = [f"# Trading digest — {date_iso} (UTC)", ""]
    if not run_events:
        lines += ["No runs journaled for this date.", ""]
    for event in run_events:
        lines.extend(_venue_section(event))
    return "\n".join(lines)


def write_digest(digest_root: Path, date_iso: str, run_events: list[dict]) -> Path:
    digest_root.mkdir(parents=True, exist_ok=True)
    path = digest_root / f"{date_iso}.md"
    path.write_text(build_digest(date_iso, run_events))
    return path
