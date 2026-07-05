import plistlib
import subprocess
from pathlib import Path

import pytest

from trading import schedule
from trading.schedule import (
    ScheduleError,
    build_plist,
    install,
    label,
    plist_path,
    remove,
    status,
)


def test_labels_and_paths():
    assert label("equities") == "com.travis.trading.equities"
    assert plist_path(Path("/tmp/agents"), "crypto") == Path(
        "/tmp/agents/com.travis.trading.crypto.plist"
    )


def test_equities_plist_runs_weekday_evenings_in_local_time():
    payload = plistlib.loads(build_plist("equities", Path("/repo"), "/usr/local/bin/uv"))
    assert payload["Label"] == "com.travis.trading.equities"
    assert payload["ProgramArguments"] == [
        "/usr/local/bin/uv",
        "run",
        "--project",
        "/repo",
        "trading",
        "run",
        "--venue",
        "equities",
    ]
    assert payload["WorkingDirectory"] == "/repo"
    # StartCalendarInterval is LOCAL time; 18:30 assumes America/New_York (README).
    assert payload["StartCalendarInterval"] == [
        {"Weekday": w, "Hour": 18, "Minute": 30} for w in (1, 2, 3, 4, 5)
    ]
    assert payload["StandardErrorPath"].endswith("state/equities/launchd.log")


def test_crypto_plist_runs_daily_0100_local():
    payload = plistlib.loads(build_plist("crypto", Path("/repo"), "/usr/local/bin/uv"))
    assert payload["StartCalendarInterval"] == [{"Hour": 1, "Minute": 0}]


def _ok(*args, **kwargs):
    return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


def test_install_writes_plists_and_bootstraps(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(schedule, "_launchctl", lambda *a: calls.append(a) or _ok())
    monkeypatch.setattr(schedule.shutil, "which", lambda name: "/usr/local/bin/uv")
    messages = install(Path("/repo"), tmp_path)
    for venue in ("equities", "crypto"):
        assert plist_path(tmp_path, venue).exists()
    actions = [c[0] for c in calls]
    assert actions.count("bootout") == 2  # idempotent reinstall
    assert actions.count("bootstrap") == 2
    assert len(messages) == 2


def test_install_requires_uv_on_path(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule.shutil, "which", lambda name: None)
    with pytest.raises(ScheduleError, match="uv"):
        install(Path("/repo"), tmp_path)


def test_status_reports_installed_and_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule.shutil, "which", lambda name: "/usr/local/bin/uv")
    monkeypatch.setattr(schedule, "_launchctl", lambda *a: _ok())
    install(Path("/repo"), tmp_path)

    def print_only_equities(*args):
        loaded = args[0] == "print" and args[1].endswith("equities")
        return subprocess.CompletedProcess(
            args=args, returncode=0 if loaded else 113, stdout="", stderr=""
        )

    monkeypatch.setattr(schedule, "_launchctl", print_only_equities)
    result = status(tmp_path)
    assert result["equities"] == {"installed": True, "loaded": True}
    assert result["crypto"] == {"installed": True, "loaded": False}


def test_remove_boots_out_and_deletes(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule.shutil, "which", lambda name: "/usr/local/bin/uv")
    calls = []
    monkeypatch.setattr(schedule, "_launchctl", lambda *a: calls.append(a) or _ok())
    install(Path("/repo"), tmp_path)
    remove(tmp_path)
    assert [c[0] for c in calls].count("bootout") == 4  # 2 install + 2 remove
    assert not plist_path(tmp_path, "equities").exists()
    assert not plist_path(tmp_path, "crypto").exists()
