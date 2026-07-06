import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import backfill_fundamentals as backfill_script  # noqa: E402
import verify_fundamentals  # noqa: E402


def test_check_source_regime_refuses_zips_store(tmp_path, capsys):
    (tmp_path / backfill_script.SOURCE_MARKER).write_text("zips")
    with pytest.raises(SystemExit, match="companyfacts-built store"):
        verify_fundamentals.check_source_regime(tmp_path)
    # The message must name the found regime so the operator knows what to
    # rebuild away from, and how.
    with pytest.raises(SystemExit, match="found: zips"):
        verify_fundamentals.check_source_regime(tmp_path)
    with pytest.raises(SystemExit, match="rebuild with --source companyfacts"):
        verify_fundamentals.check_source_regime(tmp_path)


def test_check_source_regime_warns_but_continues_with_no_marker(tmp_path, capsys):
    verify_fundamentals.check_source_regime(tmp_path)  # no raise
    assert "WARNING" in capsys.readouterr().out


def test_check_source_regime_is_silent_and_passes_for_companyfacts_marker(tmp_path, capsys):
    (tmp_path / backfill_script.SOURCE_MARKER).write_text("companyfacts")
    verify_fundamentals.check_source_regime(tmp_path)  # no raise
    assert capsys.readouterr().out == ""
