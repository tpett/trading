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
# Every scheduled job: the two venue runs plus the earnings-calendar dump.
JOBS = (*VENUES, "earnings")

_SCHEDULES: dict[str, list[dict[str, int]]] = {
    # Weekday evenings after NYSE close (local time; 1 = Monday ... 5 = Friday).
    "equities": [{"Weekday": w, "Hour": 18, "Minute": 30} for w in (1, 2, 3, 4, 5)],
    # Daily, after the 00:00 UTC crypto bar close (for US-negative offsets).
    "crypto": [{"Hour": 1, "Minute": 0}],
    # Weekdays before the equities run, so a reinstated earnings blackout
    # would see today's calendar. Reports land on weekdays; the trailing
    # window on Monday still covers anything journaled over a weekend gap.
    "earnings": [{"Weekday": w, "Hour": 17, "Minute": 30} for w in (1, 2, 3, 4, 5)],
}

_PROGRAM_ARGS: dict[str, list[str]] = {
    "equities": ["trading", "run", "--venue", "equities"],
    "crypto": ["trading", "run", "--venue", "crypto"],
    "earnings": ["python", "scripts/dump_earnings_calendar.py"],
}


class ScheduleError(RuntimeError):
    pass


def label(job: str) -> str:
    return f"com.travis.trading.{job}"


def plist_path(agents_dir: Path, job: str) -> Path:
    return agents_dir / f"{label(job)}.plist"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    """Subprocess touchpoint, isolated for monkeypatching."""
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)


def build_plist(job: str, repo_root: Path, uv_path: str) -> bytes:
    log = repo_root / "state" / job / "launchd.log"
    return plistlib.dumps(
        {
            "Label": label(job),
            "ProgramArguments": [
                uv_path,
                "run",
                "--project",
                str(repo_root),
                *_PROGRAM_ARGS[job],
            ],
            "WorkingDirectory": str(repo_root),
            "StartCalendarInterval": _SCHEDULES[job],
            "StandardOutPath": str(log),
            "StandardErrorPath": str(log),
            # launchd defaults the soft file limit to 256; a cold-cache fetch
            # across hundreds of symbols exhausts it (curl handles + parquet +
            # yfinance's sqlite), failing live symbols with DNS/thread errors.
            "SoftResourceLimits": {"NumberOfFiles": 4096},
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
    for job in JOBS:
        # launchd cannot open StandardOut/ErrorPath if the parent directory is
        # missing — on a fresh machine every scheduled run would silently fail
        # to spawn until the job is run manually. Pre-create the log dir.
        (repo_root / "state" / job).mkdir(parents=True, exist_ok=True)
        path = plist_path(agents_dir, job)
        path.write_bytes(build_plist(job, repo_root, uv_path))
        _launchctl("bootout", f"{_domain()}/{label(job)}")  # idempotent reinstall
        result = _launchctl("bootstrap", _domain(), str(path))
        if result.returncode != 0:
            raise ScheduleError(f"launchctl bootstrap failed for {job}: {result.stderr}")
        messages.append(f"{job}: installed {path}")
    return messages


def status(agents_dir: Path) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for job in JOBS:
        loaded = _launchctl("print", f"{_domain()}/{label(job)}").returncode == 0
        report[job] = {"installed": plist_path(agents_dir, job).exists(), "loaded": loaded}
    return report


def remove(agents_dir: Path) -> list[str]:
    messages: list[str] = []
    for job in JOBS:
        _launchctl("bootout", f"{_domain()}/{label(job)}")
        plist_path(agents_dir, job).unlink(missing_ok=True)
        messages.append(f"{job}: removed")
    return messages
