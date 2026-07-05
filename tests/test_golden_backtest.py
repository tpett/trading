import json

from golden_helpers import GOLDEN, run_golden

REQUIRED_REASONS = {"stop_loss", "time_stop", "trend_break", "forced_exit"}


def _expected() -> dict:
    return json.loads((GOLDEN / "expected.json").read_text())


def test_golden_backtest_matches_committed_expected(tmp_path):
    expected = _expected()
    # Provenance stamps WHEN/WHAT regenerated the expectation; it is not an
    # engine output, so it is excluded from the drift comparison.
    provenance = expected.pop("provenance")
    assert provenance["generated_by"] == "scripts/gen_golden_fixture.py"
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


def test_golden_expectation_covers_every_exit_path():
    # A regeneration that accidentally loses an exit path must fail loudly:
    # the frozen expectation itself asserts the fixture still exercises
    # stop_loss, time_stop, trend_break, forced_exit, and the skip path.
    expected = _expected()
    reasons = {trade["reason"] for trade in expected["trades"]}
    assert REQUIRED_REASONS <= reasons, f"missing exit paths: {REQUIRED_REASONS - reasons}"
    assert expected["sessions_skipped"], "fixture no longer exercises the skipped-session path"


# Fixture constructs pinned by scripts/gen_golden_fixture.py: LAT's first bar
# (LATE, day 48 from 2025-01-01) and the deliberate DDD bar hole (HOLE).
LAT_LISTS = "2025-02-18"
HOLE_DATES = {"2025-04-15", "2025-04-16", "2025-04-17"}


def test_golden_skips_are_data_quality_not_listing():
    """Listing-aware coverage: LAT listing mid-window (Feb 18, after the
    Feb 15 backtest start) must NOT skip the pre-listing sessions -- LAT is
    absent from the eligible denominator until its first bar. The ONLY skips
    are the deliberate DDD bar-hole sessions: a LISTED symbol missing a
    session's bar is a data-quality failure and still gates."""
    expected = _expected()
    skipped_dates = {date for date, _ in expected["sessions_skipped"]}
    assert skipped_dates == HOLE_DATES
    assert all(reason == "coverage" for _, reason in expected["sessions_skipped"])
    # The old semantics skipped 2025-02-15..17 (before LAT listed); those
    # sessions now run.
    assert not any(date < LAT_LISTS for date in skipped_dates)
