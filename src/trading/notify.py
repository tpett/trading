"""macOS failure signaling (spec: a silent dead pipeline is the most likely
real-world failure, so every failed/skipped run and breaker trip notifies)."""

from __future__ import annotations

import subprocess


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def notify(title: str, message: str) -> None:
    """Best-effort `display notification`; never raises."""
    script = f'display notification "{_escape(message)}" with title "{_escape(title)}"'
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10, check=False)
    except Exception:
        pass  # notification failure must never break a run
