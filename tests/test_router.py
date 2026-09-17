"""Router and specialists (FR-11, FR-12, FR-13, AC-7).

Scripted model, real index. The fabrication tests matter most: they check what
happens when the model behaves badly, not when it behaves well.
"""

import pytest

from risk_analytics import agents, config, indexing, llm, retrieval
from test_retrieval_loop import fake_llm, plan_reply, reflect_reply  # noqa: F401


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("index")
    indexing.ingest(["apsez-q1fy27"], index_dir=path)
    return path


@pytest.fixture(scope="module")
def hits(index_dir):
    return indexing.search("net gearing DSCR covenant", k=5, index_dir=index_dir)


def route_reply(selected, **reasons):
    full = {k: reasons.get(k, f"no reason recorded for {k}") for k in agents.SPECIALISTS}
    return {"selected": list(selected), "reasons": full}


def finding_reply(claims, insufficient=False, note=""):
    return {
        "insufficient_evidence": insufficient,
        "note": note,
        "claims": [{"text": t, "evidence_ids": list(ids)} for t, ids in claims],
    }


# --- Four specialists exist and are narrow (FR-12) ---------------------------


def test_four_specialists_are_registered():
    assert set(agents.SPECIALISTS) == {agents.CREDIT, agents.ESG, agents.COMPLIANCE, agents.PEER}
    for specialist in agents.SPECIALISTS.values():
        assert specialist.covers and specialist.system
        assert specialist.id in agents.ROUTE_SCHEMA["properties"]["reasons"]["properties"]


# --- Routing (FR-11) ----------------------------------------------------------


def test_the_router_records_a_reason_for_every_specialist(hits):
    """FR-11 asks for a reason for each selection *and* each non-selection: a
    routing decision you cannot argue with is not inspectable."""
    model = fake_llm([route_reply(
        [agents.CREDIT],
        credit="the question is about leverage and covenant headroom",
        esg="no emissions or transition content is requested",
        compliance="no audit or related-party angle in the question",
        peer="only one issuer is named",
    )])
    decision = agents.route("What is APSEZ covenant headroom?", model)

    assert decision.selected == [agents.CREDIT]
    assert set(decision.reasons) == set(agents.SPECIALISTS)
    assert all(reason.strip() for reason in decision.reasons.values())
    assert "leverage" in decision.reasons[agents.CREDIT]


def test_routing_differs_across_questions(hits):
    """AC-7. If every question routes the same way, the router is decoration."""
    scripts = {
        "covenant headroom": [agents.CREDIT],
        "auditor emphasis of matter": [agents.COMPLIANCE],
        "compare the two issuers": [agents.PEER, agents.CREDIT],
    }
    selections = []
    for question, picked in scripts.items():
        model = fake_llm([route_reply(picked)])
        selections.append(tuple(agents.route(question, model).selected))

    assert len(set(selections)) > 1, "every question produced the same specialist set"


def test_no_more_than_two_specialists_are_invoked(hits):
    """FR-13, enforced in code. The prompt asks; the ceiling does not negotiate."""
    model = fake_llm([route_reply([agents.CREDIT, agents.ESG, agents.COMPLIANCE, agents.PEER])])
    decision = agents.route("a broad question", model)

    assert len(decision.selected) == config.MAX_SPECIALISTS_PER_QUERY
    assert decision.truncated == [agents.COMPLIANCE, agents.PEER]
    for dropped in decision.truncated:
        assert "not invoked" in decision.reasons[dropped]


def test_an_unknown_specialist_name_is_ignored(hits):
    model = fake_llm([{
        "selected": ["macro_strategy", agents.CREDIT],
        "reasons": {k: "reason" for k in agents.SPECIALISTS},
    }])
    assert agents.route("q", model).selected == [agents.CREDIT]


def test_duplicate_selections_are_collapsed(hits):
    model = fake_llm([route_reply([agents.CREDIT, agents.CREDIT, agents.ESG])])
    decision = agents.route("q", model)
    assert decision.selected == [agents.CREDIT, agents.ESG]


# --- Specialists answer only from evidence -----------------------------------


def test_a_specialist_reports_claims_with_resolved_citations(hits):
    ids = [h.chunk_id for h in hits[:2]]
    model = fake_llm([finding_reply([("Net gearing is 0.55 against a covenant of 3.0x", ids)])])
    finding = agents.consult(agents.CREDIT, "headroom?", hits, model)

    assert not finding.insufficient_evidence
    assert len(finding.claims) == 1
    claim = finding.claims[0]
    assert claim.evidence_ids == ids
    assert claim.citations
    assert all(c.startswith("[apsez-q1fy27 p.") for c in claim.citations)


def test_a_claim_citing_evidence_that_does_not_exist_is_dropped(hits):
    """The fabrication case. A plausible sentence citing an id that was never
    retrieved is exactly the failure that would discredit the demo, so it is
    removed and recorded rather than printed."""
    model = fake_llm([finding_reply([
        ("Net gearing is 0.55", [hits[0].chunk_id]),
        ("AGEL's emissions intensity fell 12 percent", ["apsez-q1fy27:p999:table0"]),
    ])])
    finding = agents.consult(agents.CREDIT, "q", hits, model)

    texts = [c.text for c in finding.claims]
    assert "Net gearing is 0.55" in texts
    assert all("emissions intensity" not in t for t in texts)
    assert finding.dropped_claims == ["AGEL's emissions intensity fell 12 percent"]


def test_a_claim_citing_nothing_real_leaves_the_finding_insufficient(hits):
    model = fake_llm([finding_reply([("Something confident and unsupported", ["not-a-real-id"])])])
    finding = agents.consult(agents.CREDIT, "q", hits, model)

    assert finding.claims == []
    assert finding.insufficient_evidence
    assert finding.dropped_claims


def test_partially_valid_citations_keep_only_the_real_ones(hits):
    real = hits[0].chunk_id
    model = fake_llm([finding_reply([("A supported claim", [real, "ghost-id"])])])
    finding = agents.consult(agents.CREDIT, "q", hits, model)

    assert finding.claims[0].evidence_ids == [real]


def test_a_specialist_with_no_evidence_says_so_without_calling_the_model():
    """No retrieval means nothing to analyse. Calling the model here would spend
    budget to produce something necessarily unsupported."""
    model = fake_llm([])  # any call would raise
    finding = agents.consult(agents.ESG, "q", [], model)

    assert finding.insufficient_evidence
    assert finding.claims == []
    assert model.budget.calls == 0


def test_the_esg_specialist_can_report_insufficient_evidence(hits):
    """KNOWLEDGE-4ef75831: this corpus has essentially no ESG disclosure. The
    specialist must say so rather than reach for general knowledge about a
    renewable-energy company."""
    model = fake_llm([finding_reply(
        [], insufficient=True,
        note="the excerpts contain no emissions, capacity or transition disclosure",
    )])
    finding = agents.consult(agents.ESG, "What are AGEL's emissions?", hits, model)

    assert finding.insufficient_evidence
    assert finding.claims == []
    assert "no emissions" in finding.note


def test_an_empty_claim_list_is_treated_as_insufficient(hits):
    model = fake_llm([finding_reply([], insufficient=False, note="nothing to report")])
    finding = agents.consult(agents.CREDIT, "q", hits, model)
    assert finding.insufficient_evidence, (
        "a finding with no claims must not read as a successful analysis"
    )


# --- Orchestration ------------------------------------------------------------


def test_consult_selected_runs_each_chosen_specialist(hits):
    model = fake_llm([
        finding_reply([("credit claim", [hits[0].chunk_id])]),
        finding_reply([("compliance claim", [hits[1].chunk_id])]),
    ])
    decision = agents.RoutingDecision(
        selected=[agents.CREDIT, agents.COMPLIANCE],
        reasons={k: "r" for k in agents.SPECIALISTS},
    )
    findings = agents.consult_selected(decision, "q", hits, model)

    assert [f.specialist for f in findings] == [agents.CREDIT, agents.COMPLIANCE]
    assert all(f.claims for f in findings)


def test_the_spend_ceiling_stops_specialists_and_is_recorded(hits):
    """NFR-1 again: specialists run on the expensive model, so this is where a
    run is most likely to breach."""
    model = fake_llm(
        [finding_reply([("a claim", [hits[0].chunk_id])])],
        ceiling=0.01, input_tokens=500_000, output_tokens=100_000,
    )
    decision = agents.RoutingDecision(
        selected=[agents.CREDIT, agents.COMPLIANCE],
        reasons={k: "r" for k in agents.SPECIALISTS},
    )
    findings = agents.consult_selected(decision, "q", hits, model)

    assert findings, "the run should record what it could not do"
    assert findings[-1].insufficient_evidence
    assert "spend ceiling" in findings[-1].note
    assert model.budget.spent_usd <= model.budget.ceiling_usd


def test_specialists_run_on_the_analysis_model_and_routing_on_the_cheap_one(hits):
    """DECISION-39162611. Routing is high-frequency and simple; analysis is what
    the demo is judged on. Inverting this breaks the cost ceiling."""
    model = fake_llm([route_reply([agents.CREDIT])])
    agents.route("q", model)
    assert model.client().messages.calls[-1]["model"] == config.ROUTER_MODEL

    model2 = fake_llm([finding_reply([("c", [hits[0].chunk_id])])])
    agents.consult(agents.CREDIT, "q", hits, model2)
    assert model2.client().messages.calls[-1]["model"] == config.SPECIALIST_MODEL


def test_routing_and_consulting_compose_over_real_retrieval(index_dir):
    """Loop, route, consult -- the full T6A-to-T7A path against a real index."""
    loop_model = fake_llm([
        plan_reply("net gearing ratio", "DSCR"),
        reflect_reply(True, "both present"),
    ])
    result = retrieval.answer(
        "What is APSEZ's covenant headroom?", loop_model,
        k=8, where={"kind": indexing.TABLE}, index_dir=index_dir,
    )
    assert result.evidence

    agent_model = fake_llm([
        route_reply([agents.CREDIT], credit="leverage and covenants"),
        finding_reply([("Net gearing is 0.55 against a 3.0x covenant",
                        [result.evidence[0].chunk_id])]),
    ])
    decision = agents.route(result.question, agent_model)
    findings = agents.consult_selected(decision, result.question, result.evidence, agent_model)

    assert decision.selected == [agents.CREDIT]
    assert findings[0].claims[0].citations
