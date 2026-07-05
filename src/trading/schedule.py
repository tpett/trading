"""launchd LaunchAgents for the daily runs (spec: Runtime, CLI).

StartCalendarInterval is machine-LOCAL time: equities 18:30 assumes the Mac
is in America/New_York (documented in the README); crypto 01:00 local lands
after the 00:00 UTC bar close for US offsets. launchd coalesces intervals
missed while asleep into one run on wake — the runner's staleness rule then
skips entries and still processes exits. Failure signaling is the runner's
job (notifications), not launchd's.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
from pathlib import Path

VENUES = ("equities", "crypto")

_SCHEDULES: dict[str, list[dict[str, int]]] = {
    # Weekday evenings after NYSE close (local time; 1 = Monday ... 5 = Friday).
    "equities": [{"Weekday": w, "Hour": 18, "Minute": 30} for w in (1, 2, 3, 4, 5)],
    # Daily, after the 00:00 UTC crypto bar close (for US-negative offsets).
    "crypto": [{"Hour": 1, "Minute": 0}],
}


class ScheduleError(RuntimeError):
    pass


def label(venue: str) -> str:
    return f"com.travis.trading.{venue}"


def plist_path(agents_dir: Path, venue: str) -> Path:
    return agents_dir / f"{label(venue)}.plist"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    """Subprocess touchpoint, isolated for monkeypatching."""
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)


def build_plist(venue: str, repo_root: Path, uv_path: str) -> bytes:
    log = repo_root / "state" / venue / "launchd.log"
    return plistlib.dumps(
        {
            "Label": label(venue),
            "ProgramArguments": [
                uv_path,
                "run",
                "--project",
                str(repo_root),
                "trading",
                "run",
                "--venue",
                venue,
            ],
            "WorkingDirectory": str(repo_root),
            "StartCalendarInterval": _SCHEDULES[venue],
            "StandardOutPath": str(log),
            "StandardErrorPath": str(log),
        }
    )


def _domain() -> str:
    return f"gui/{os.getuid()}"


def install(repo_root: Path, agents_dir: Path) -> list[str]:
    uv_path = shutil.which("uv")
    if uv_path is None:
        raise ScheduleError("uv not found on PATH; cannot build LaunchAgents")
    agents_dir.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []
    for venue in VENUES:
        path = plist_path(agents_dir, venue)
        path.write_bytes(build_plist(venue, repo_root, uv_path))
        _launchctl("bootout", f"{_domain()}/{label(venue)}")  # idempotent reinstall
        result = _launchctl("bootstrap", _domain(), str(path))
        if result.returncode != 0:
            raise ScheduleError(f"launchctl bootstrap failed for {venue}: {result.stderr}")
        messages.append(f"{venue}: installed {path}")
    return messages


def status(agents_dir: Path) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for venue in VENUES:
        loaded = _launchctl("print", f"{_domain()}/{label(venue)}").returncode == 0
        report[venue] = {"installed": plist_path(agents_dir, venue).exists(), "loaded": loaded}
    return report


def remove(agents_dir: Path) -> list[str]:
    messages: list[str] = []
    for venue in VENUES:
        _launchctl("bootout", f"{_domain()}/{label(venue)}")
        plist_path(agents_dir, venue).unlink(missing_ok=True)
        messages.append(f"{venue}: removed")
    return messages
