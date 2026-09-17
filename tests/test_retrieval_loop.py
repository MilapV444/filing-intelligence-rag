"""The agentic retrieval loop (FR-8, FR-9, FR-10, NFR-1, AC-6).

Every test runs against a recorded stand-in for the model, so the suite spends
nothing and behaves identically on every machine. The real API is exercised at
T9A, where the cost is the point of the measurement.
"""

import json

import pytest

from risk_analytics import config, indexing, llm, retrieval

# --- A recorded stand-in for the model ---------------------------------------


class FakeResponseUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeResponse:
    def __init__(self, payload, input_tokens=1000, output_tokens=200):
        self.content = [FakeBlock(payload)]
        self.usage = FakeResponseUsage(input_tokens, output_tokens)


class FakeMessages:
    def __init__(self, script, input_tokens=1000, output_tokens=200):
        self.script = list(script)
        self.calls = []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("the loop made more model calls than the script allows")
        payload = self.script.pop(0)
        return FakeResponse(
            json.dumps(payload) if not isinstance(payload, str) else payload,
            self.input_tokens,
            self.output_tokens,
        )


class FakeClient:
    def __init__(self, script, **kw):
        self.messages = FakeMessages(script, **kw)


def fake_llm(script, ceiling=config.RUN_COST_CEILING_USD, **kw):
    return llm.LLM(client=FakeClient(script, **kw), budget=llm.Budget(ceiling_usd=ceiling))


def plan_reply(*queries, reasoning="decomposed"):
    return {"sub_queries": list(queries), "reasoning": reasoning}


def reflect_reply(sufficient, reason="", missing=(), next_queries=()):
    return {
        "sufficient": sufficient,
        "reason": reason,
        "missing": list(missing),
        "next_queries": list(next_queries),
    }


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    """One small real index, so retrieval is genuine even though the model is not."""
    path = tmp_path_factory.mktemp("index")
    indexing.ingest(["apsez-q1fy27"], index_dir=path)
    return path


# --- Planning and iteration (FR-8) -------------------------------------------


def test_the_loop_plans_sub_queries_before_searching(index_dir):
    model = fake_llm([
        plan_reply("net gearing ratio", "debt service coverage ratio"),
        reflect_reply(True, "both ratios and their thresholds are present"),
    ])
    result = retrieval.answer("What is APSEZ covenant headroom?", model, index_dir=index_dir)

    assert result.trace.plan[:2] == ["net gearing ratio", "debt service coverage ratio"]
    assert result.evidence, "planning produced no evidence"
    assert model.client().messages.calls[0]["model"] == config.ROUTER_MODEL


def test_a_sufficient_verdict_stops_the_loop_at_one_iteration(index_dir):
    model = fake_llm([plan_reply("net gearing"), reflect_reply(True, "figure present")])
    result = retrieval.answer("q", model, index_dir=index_dir)

    assert result.trace.termination == retrieval.SUFFICIENT
    assert result.trace.iteration_count == 1


def test_an_insufficient_verdict_drives_a_second_iteration_with_new_queries(index_dir):
    """AC-6. This is the difference between an agentic loop and single-shot
    search: reflection must change what the next iteration looks for."""
    model = fake_llm([
        plan_reply("covenant headroom"),
        reflect_reply(
            False,
            "the gearing threshold is present but the computed ratio is not",
            missing=["computed net gearing value"],
            next_queries=["net gearing total net debt tangible net worth"],
        ),
        reflect_reply(True, "the computed ratio is now present"),
    ])
    result = retrieval.answer("q", model, index_dir=index_dir)
    trace = result.trace

    assert trace.iteration_count == 2
    assert trace.termination == retrieval.SUFFICIENT
    assert trace.iterations[0].sufficient is False
    assert trace.iterations[1].queries == ["net gearing total net debt tangible net worth"]
    assert trace.revised_after_reflection, "the second pass repeated the first pass's queries"


def test_evidence_accumulates_across_iterations_without_duplicates(index_dir):
    model = fake_llm([
        plan_reply("net gearing"),
        reflect_reply(False, "need DSCR too", missing=["DSCR"], next_queries=["debt service coverage"]),
        reflect_reply(True, "complete"),
    ])
    result = retrieval.answer("q", model, index_dir=index_dir)

    ids = [h.chunk_id for h in result.evidence]
    assert len(ids) == len(set(ids)), "the same chunk was carried twice"
    # Superset, not lexicographic order: iteration two keeps everything
    # iteration one found and adds to it.
    assert set(result.trace.iterations[1].hit_ids) >= set(result.trace.iterations[0].hit_ids)
    assert result.trace.iterations[1].new_hits > 0


# --- Termination conditions (FR-9) -------------------------------------------


def test_the_iteration_ceiling_stops_a_loop_that_never_succeeds(index_dir):
    never = reflect_reply(False, "still missing", missing=["x"], next_queries=["another angle"])
    model = fake_llm([
        plan_reply("first"),
        {**never, "next_queries": ["second"]},
        {**never, "next_queries": ["third"]},
        {**never, "next_queries": ["fourth"]},
    ])
    result = retrieval.answer("q", model, max_iterations=3, index_dir=index_dir)

    assert result.trace.termination == retrieval.ITERATION_CEILING
    assert result.trace.iteration_count == 3


def test_the_spend_ceiling_stops_the_loop_and_is_recorded(index_dir):
    """NFR-1. The ceiling has to bite before the money is spent, not after."""
    model = fake_llm(
        [plan_reply("first"), reflect_reply(False, "more", next_queries=["second"])],
        ceiling=0.004,
        input_tokens=500_000,
        output_tokens=100_000,
    )
    result = retrieval.answer("q", model, index_dir=index_dir)

    assert result.trace.termination == retrieval.SPEND_CEILING
    assert model.budget.spent_usd > 0


def test_a_reflection_with_no_new_queries_does_not_loop_forever(index_dir):
    """Insufficient but nothing new to try: looping would spend money to retrieve
    exactly the same chunks."""
    model = fake_llm([
        plan_reply("net gearing"),
        reflect_reply(False, "insufficient", missing=["something"], next_queries=["net gearing"]),
    ])
    result = retrieval.answer("q", model, index_dir=index_dir)

    assert result.trace.termination == retrieval.NO_PROGRESS
    assert result.trace.iteration_count == 1


def test_exactly_one_termination_condition_is_recorded(index_dir):
    model = fake_llm([plan_reply("q1"), reflect_reply(True, "done")])
    trace = retrieval.answer("q", model, index_dir=index_dir).trace
    assert trace.termination in {
        retrieval.SUFFICIENT,
        retrieval.ITERATION_CEILING,
        retrieval.SPEND_CEILING,
        retrieval.NO_PROGRESS,
    }


# --- The trace (FR-10) --------------------------------------------------------


def test_the_trace_is_machine_readable_and_complete(index_dir):
    model = fake_llm([
        plan_reply("covenant headroom", reasoning="split by ratio"),
        reflect_reply(False, "missing computed ratio", missing=["value"], next_queries=["net gearing value"]),
        reflect_reply(True, "found"),
    ])
    trace = retrieval.answer("What is the headroom?", model, index_dir=index_dir).trace

    payload = json.loads(json.dumps(trace.to_dict()))  # must survive a round trip
    assert payload["question"] == "What is the headroom?"
    assert payload["plan_reasoning"] == "split by ratio"
    assert payload["termination"] == retrieval.SUFFICIENT
    assert payload["iteration_count"] == 2
    assert payload["revised_after_reflection"] is True

    first = payload["iterations"][0]
    assert first["queries"] and first["hit_ids"]
    assert first["sufficient"] is False
    assert first["missing"] == ["value"]
    assert first["reason"]


def test_the_trace_records_what_every_call_cost(index_dir):
    """FR-18 reports run cost; it can only do that if the trace carries it."""
    model = fake_llm([plan_reply("q1"), reflect_reply(True, "ok")])
    trace = retrieval.answer("q", model, index_dir=index_dir).trace

    budget = trace.budget
    assert budget["calls"] == 2
    assert budget["spent_usd"] > 0
    assert {c["purpose"] for c in budget["by_call"]} == {"plan", "reflect"}
    assert all(c["model"] == config.ROUTER_MODEL for c in budget["by_call"])


def test_citations_come_from_stored_metadata(index_dir):
    model = fake_llm([plan_reply("net gearing"), reflect_reply(True, "ok")])
    result = retrieval.answer("q", model, index_dir=index_dir)

    assert result.citations
    for citation in result.citations:
        assert citation.startswith("[apsez-q1fy27 p.")


# --- Budget accounting --------------------------------------------------------


def test_cost_is_computed_from_reported_tokens():
    budget = llm.Budget(ceiling_usd=1.0)
    budget.record(llm.Usage(config.ROUTER_MODEL, 100_000, 20_000, "plan"))
    expected = config.cost_usd(config.ROUTER_MODEL, 100_000, 20_000)
    assert budget.spent_usd == pytest.approx(expected)
    assert budget.remaining_usd == pytest.approx(1.0 - expected)


def test_remaining_budget_never_reports_a_negative_number():
    """An overrun reports zero left, not a negative balance -- a negative
    remaining reads like credit and would invite one more call."""
    budget = llm.Budget(ceiling_usd=0.01)
    budget.record(llm.Usage(config.SPECIALIST_MODEL, 1_000_000, 1_000_000, "specialist"))
    assert budget.spent_usd > budget.ceiling_usd
    assert budget.remaining_usd == 0.0


def test_the_budget_refuses_a_call_once_the_ceiling_is_reached():
    budget = llm.Budget(ceiling_usd=0.001)
    budget.record(llm.Usage(config.SPECIALIST_MODEL, 1_000_000, 0, "specialist"))
    with pytest.raises(llm.BudgetExceeded):
        budget.check("another call")


def test_a_run_under_the_ceiling_proceeds():
    budget = llm.Budget(ceiling_usd=config.RUN_COST_CEILING_USD)
    budget.record(llm.Usage(config.ROUTER_MODEL, 2_000, 500, "plan"))
    budget.check("reflect")  # must not raise
    assert budget.spent_usd < config.RUN_COST_CEILING_USD


# --- Retrieval is real even when the model is not ----------------------------


def test_the_loop_retrieves_the_covenant_figures_end_to_end(index_dir):
    """The retrieval half of golden question 1, with a scripted model but a real
    index and real embeddings."""
    model = fake_llm([
        plan_reply("net gearing ratio total net debt tangible net worth", "DSCR"),
        reflect_reply(True, "both present"),
    ])
    result = retrieval.answer(
        "What are APSEZ's Net Gearing and DSCR, and what headroom is there?",
        model,
        k=10,
        where={"kind": indexing.TABLE},
        index_dir=index_dir,
    )
    blob = "\n".join(h.text for h in result.evidence)
    assert "0.55" in blob
    assert "5.42" in blob
    assert any(h.page_number == 34 for h in result.evidence)


def test_the_loop_needs_no_api_key_when_the_client_is_injected(monkeypatch, index_dir):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    model = fake_llm([plan_reply("q"), reflect_reply(True, "ok")])
    assert retrieval.answer("q", model, index_dir=index_dir).evidence
