import json

from golden_helpers import GOLDEN, run_golden


def test_golden_backtest_matches_committed_expected(tmp_path):
    expected = json.loads((GOLDEN / "expected.json").read_text())
    actual = run_golden(tmp_path / "cache")
    assert actual == expected, (
        "Golden backtest drifted. If the change is INTENDED, regenerate with "
        "'uv run python scripts/gen_golden_fixture.py --write-expected' and "
        "explain the drift in the commit message."
    )


def test_golden_fixture_actually_trades(tmp_path):
    # Guards against the golden test passing vacuously on an empty run.
    actual = run_golden(tmp_path / "cache")
    assert actual["trades"] or actual["open_positions"]
    assert actual["sessions_run"] > 60
