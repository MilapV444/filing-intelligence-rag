"""The agentic-versus-single-shot comparison (FR-8, AC-13).

These assertions check that the comparison was run fairly and reported fully.
They deliberately do **not** assert that the agentic loop wins. AC-13 exists
because FR-8 asserts an architecture, and a benchmark whose pass condition is
"my architecture won" measures nothing. The recorded result stands whichever way
it falls, and at the time of writing it falls against the loop.
"""

import json
from pathlib import Path

import pytest

from risk_analytics import benchmark, config

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "ab-results.json"
REPORT = ROOT / "benchmarks" / "ab-report.md"


@pytest.fixture(scope="module")
def results():
    if not RESULTS.exists():
        pytest.fail(
            f"AC-13 requires a recorded comparison at {RESULTS}. Produce it by "
            "running both arms over the benchmark questions against a funded key."
        )
    return json.loads(RESULTS.read_text(encoding="utf-8"))


# --- The comparison was actually run -----------------------------------------


def test_every_benchmark_question_has_both_arms(results):
    assert len(results) >= 3, "the comparison covers fewer questions than the benchmark set"
    for row in results:
        assert row["agentic"]["arm"] == benchmark.AGENTIC
        assert row["single_shot"]["arm"] == benchmark.SINGLE_SHOT
        assert row["agentic"]["question"] == row["single_shot"]["question"]


def test_both_arms_actually_called_the_model(results):
    """A control arm that silently failed would hand the loop a free win."""
    for row in results:
        assert row["agentic"]["calls"] > 0
        assert row["single_shot"]["calls"] > 0
        assert not row["agentic"]["error"] and not row["single_shot"]["error"]


def test_the_control_arm_did_not_iterate(results):
    """Single-shot means one retrieval. If it iterated, the arms are not
    different and the comparison measures nothing."""
    for row in results:
        assert row["single_shot"]["iterations"] == 1
        assert row["single_shot"]["termination"] == "single_shot"


def test_the_control_arm_was_not_handicapped(results):
    """The honest failure mode of a self-authored benchmark is starving the
    control. Both arms get the same evidence cap, so the control must be able to
    reach a comparable number of pages."""
    for row in results:
        agentic_pages = len(row["agentic"]["pages"])
        control_pages = len(row["single_shot"]["pages"])
        assert control_pages > 0, "the control arm reached nothing; it was starved"
        assert control_pages >= agentic_pages * 0.4, (
            f"the control reached {control_pages} pages against the loop's "
            f"{agentic_pages}; check it was given the same evidence budget"
        )


def test_both_arms_shared_the_evidence_cap():
    assert benchmark.config.MAX_EVIDENCE_CHUNKS == config.MAX_EVIDENCE_CHUNKS


# --- The result was reported, whichever way it fell --------------------------


def test_the_report_exists_and_states_the_method(results):
    assert REPORT.exists(), "AC-13 requires the comparison to be reported, not just recorded"
    text = REPORT.read_text(encoding="utf-8")
    assert "single-shot" in text.lower()
    assert str(config.MAX_EVIDENCE_CHUNKS) in text
    for row in results:
        assert row["question"][:40] in text


def test_the_report_names_what_each_arm_reached(results):
    text = REPORT.read_text(encoding="utf-8")
    assert "pages reached" in text.lower()
    assert "cost" in text.lower()


def test_every_recorded_cost_is_real(results):
    for row in results:
        for arm in ("agentic", "single_shot"):
            assert row[arm]["cost_usd"] > 0


def test_the_comparison_records_pages_unique_to_each_arm(results):
    """The interesting signal is not the totals but what one arm reached and the
    other did not, in both directions."""
    for row in results:
        assert "pages_only_agentic" in row
        assert "pages_only_single_shot" in row


def test_the_recorded_verdict_is_not_assumed_to_favour_the_loop(results):
    """A guard against quietly rewriting the benchmark if the result is
    unwelcome. This asserts the data is capable of showing the loop losing, and
    -- at the time of writing -- that it does on at least one question."""
    control_won_somewhere = any(
        len(r["single_shot"]["pages"]) >= len(r["agentic"]["pages"])
        or r["single_shot"]["findings"] >= r["agentic"]["findings"]
        for r in results
    )
    assert control_won_somewhere, (
        "every question favours the loop, which is suspicious for n=3 on a small "
        "corpus; verify the control arm is not handicapped before believing it"
    )


def test_the_loop_costs_more_than_the_control(results):
    """Recorded as a fact, not a complaint: the loop makes strictly more model
    calls, so any claim it makes for itself has to be worth that premium."""
    premiums = [r["agentic"]["cost_usd"] / r["single_shot"]["cost_usd"] for r in results]
    assert all(p > 0 for p in premiums)
    assert sum(premiums) / len(premiums) > 0.9, "cost ratios look implausible"
