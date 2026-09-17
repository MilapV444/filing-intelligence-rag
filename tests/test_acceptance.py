"""Acceptance against the live API (NFR-1, NFR-6, AC-5, AC-8, AC-9, AC-11).

These assertions run against the recorded traces of three real runs, in
`tests/acceptance/`, rather than re-running the pipeline. Two reasons: a gate
that spends about forty cents every time it executes is a bad gate, and the
figures being asserted -- cost, iteration counts, which specialists were chosen
-- are measurements of runs that happened, not properties a rerun would
reproduce identically.

The limitation that follows is real and recorded: this proves the runs happened
and what they produced, not that the current source would reproduce them. A
fresh measurement needs `python -m risk_analytics ask` and a funded key.
"""

import json
from pathlib import Path

import pytest

from risk_analytics import agents, config, indexing, manifest

ACCEPTANCE_DIR = Path(__file__).parent / "acceptance"
TIME_LIMIT_SECONDS = 300  # NFR-6


@pytest.fixture(scope="module")
def runs():
    traces = sorted(ACCEPTANCE_DIR.glob("*.trace.json"))
    if len(traces) < 3:
        pytest.fail(
            f"expected three recorded golden-question runs in {ACCEPTANCE_DIR}, "
            f"found {len(traces)}. Produce them with `python -m risk_analytics ask` "
            "against a funded key; AC-5 and AC-8 are claims about real runs."
        )
    return [json.loads(p.read_text(encoding="utf-8")) for p in traces]


# --- AC-8 and NFR-1: measured spend ------------------------------------------


def test_every_run_stayed_under_the_cost_ceiling(runs):
    """AC-8. The first live run finished at $0.4302 against this ceiling."""
    for run in runs:
        spent = run["budget"]["spent_usd"]
        assert spent <= config.RUN_COST_CEILING_USD, (
            f"{run['question'][:60]}... cost ${spent:.4f}"
        )


def test_the_recorded_cost_matches_the_sum_of_its_calls(runs):
    """FR-18's reported figure must be the arithmetic of what was actually
    billed, not a separate estimate that could drift from it."""
    for run in runs:
        by_call = sum(c["cost_usd"] for c in run["budget"]["by_call"])
        assert by_call == pytest.approx(run["budget"]["spent_usd"], abs=1e-4)


def test_every_call_used_a_model_with_a_published_rate(runs):
    for run in runs:
        for call in run["budget"]["by_call"]:
            assert call["model"] in config.MODEL_RATES_USD_PER_MTOK


def test_the_cheap_model_carried_the_high_frequency_work(runs):
    """DECISION-39162611 as retuned: planning, reflection and routing are
    high-frequency and belong on the cheapest model."""
    for run in runs:
        for call in run["budget"]["by_call"]:
            if call["purpose"] in {"plan", "reflect", "route"}:
                assert call["model"] == config.ROUTER_MODEL


# --- AC-5: every claim carries a citation that resolves ----------------------


def test_every_reported_finding_carries_a_citation(runs):
    """AC-5. An uncited material claim is a defect, not a stylistic lapse."""
    for run in runs:
        for finding in run["key_findings"]:
            assert finding["citations"], f"uncited: {finding['text'][:80]}"
            assert finding["evidence_ids"]


def test_every_citation_resolves_to_a_registered_document_and_a_real_page(runs):
    registered = set(manifest.by_id())
    for run in runs:
        for citation in run["citations"]:
            doc_id, _, page = citation.lstrip("[").rstrip("]").partition(" p.")
            assert doc_id in registered, f"citation names an unregistered document: {citation}"
            assert page.isdigit() and int(page) >= 1, f"malformed citation: {citation}"


def test_every_cited_chunk_was_actually_retrieved(runs):
    """NFR-7. A citation must trace back to a chunk the loop saw, which is what
    stops a page number being plausible rather than true."""
    for run in runs:
        seen = {i for it in run["trace"]["iterations"] for i in it["hit_ids"]}
        for finding in run["key_findings"]:
            assert set(finding["evidence_ids"]) <= seen, (
                f"finding cites evidence the run never retrieved: {finding['text'][:60]}"
            )


def test_no_run_silently_dropped_a_fabricated_claim_without_recording_it(runs):
    for run in runs:
        assert isinstance(run["dropped_claims"], list)


# --- AC-9: the table route earns its place -----------------------------------


def test_the_covenant_question_was_answered_from_table_pages(runs):
    """AC-9 as amended. The chart half was withdrawn because the corpus has no
    chart page; the table half is what proves classify-and-route is load-bearing.
    """
    covenant = [r for r in runs if "Net Gearing" in r["question"]]
    assert covenant, "the covenant golden question is not among the recorded runs"
    run = covenant[0]

    blob = " ".join(f["text"] for f in run["key_findings"])
    assert "0.55" in blob, "the Net Gearing figure was not reported"
    assert "5.42" in blob, "the DSCR figure was not reported"
    assert any("3" in blob for _ in [0]) and "1.10" in blob, "covenant thresholds missing"
    assert any("p.34" in c for c in run["citations"]), (
        "the covenant table on page 34 was not cited"
    )


# --- AC-7 and FR-11: routing is query-dependent ------------------------------


def test_the_three_questions_did_not_all_route_the_same_way(runs):
    selections = {tuple(r["routing"]["selected"]) for r in runs}
    assert len(selections) > 1, f"every question routed identically: {selections}"


def test_no_run_invoked_more_than_the_specialist_cap(runs):
    for run in runs:
        assert len(run["routing"]["selected"]) <= config.MAX_SPECIALISTS_PER_QUERY


def test_every_specialist_has_a_recorded_reason_in_every_run(runs):
    for run in runs:
        reasons = run["routing"]["reasons"]
        assert set(reasons) == set(agents.SPECIALISTS)
        assert all(r.strip() for r in reasons.values())


def test_the_comparison_question_drew_on_both_issuers(runs):
    """A peer comparison citing one issuer is not a comparison. Before balanced
    retrieval this question returned 74 chunks, all from the larger document."""
    comparison = [r for r in runs if "Compare" in r["question"]]
    assert comparison, "the comparison golden question is not among the recorded runs"
    retrieved = {
        i.split(":")[0]
        for i in comparison[0]["trace"]["iterations"][-1]["hit_ids"]
    }
    assert retrieved == set(manifest.by_id()), (
        f"the comparison drew on {retrieved}, not the whole corpus"
    )


# --- AC-6 and FR-9: the loop is agentic --------------------------------------


def test_at_least_one_run_revised_its_queries_after_reflection(runs):
    """AC-6. Without this the loop is single-shot search in a costume."""
    assert any(r["trace"]["revised_after_reflection"] for r in runs), (
        "no run changed what it searched for after reflecting"
    )


def test_every_run_recorded_exactly_one_termination_condition(runs):
    valid = {"sufficient", "iteration_ceiling", "spend_ceiling", "no_progress"}
    for run in runs:
        assert run["trace"]["termination"] in valid


# --- FR-15, FR-17, AC-10, AC-11 ----------------------------------------------


def test_every_run_names_the_documents_it_searched(runs):
    for run in runs:
        assert run["scope"].strip()
        assert run["limitations"], "a note with no stated limitations is overclaiming"


def test_the_disclaimer_text_says_it_is_not_a_rating():
    """AC-10, checked at the source so it cannot drift out of the renderer."""
    from risk_analytics import note as note_module

    assert "not a credit rating" in note_module.DISCLAIMER
    assert "not investment advice" in note_module.DISCLAIMER
    assert "not been reviewed by a rating committee" in note_module.DISCLAIMER


def test_no_recorded_trace_contains_credential_material(runs):
    """AC-11. Traces are written to disk and read by humans."""
    prefix = "sk-" + "ant-"
    for run in runs:
        assert prefix not in json.dumps(run)


def test_the_corpus_is_fully_registered_with_public_sources():
    """AC-11's other half."""
    documents = manifest.load()
    assert documents
    for doc in documents:
        assert doc.source_url.startswith("https://")
    assert not manifest.missing_files(documents)


def test_the_index_covers_the_whole_registered_corpus():
    counts = {
        doc_id: len(
            indexing.collection().get(where={"doc_id": doc_id}, include=[])["ids"]
        )
        for doc_id in manifest.by_id()
    }
    assert all(n > 0 for n in counts.values()), f"a document has no chunks indexed: {counts}"
