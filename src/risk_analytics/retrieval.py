"""Agentic retrieval: plan, search, reflect, iterate, stop (FR-8, FR-9, FR-10).

This is the part of the demo that is meant to beat single-shot RAG, so it is
worth being precise about what makes it agentic rather than decorative:

    plan     the question is decomposed into sub-queries before any search
    search   each sub-query retrieves independently
    reflect  a model judges whether the evidence actually answers the question
    iterate  an insufficient verdict produces *revised* queries, not a retry
    stop     on sufficiency, an iteration ceiling, or a spend ceiling

A loop that always runs once and declares success is single-shot search wearing
a costume. AC-6 exists to catch that: it requires a recorded trace showing a
second iteration whose queries differ from the first.

Every decision is written to a trace (FR-10) so the loop can be inspected after
the fact rather than believed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from . import config, indexing, llm

# Termination conditions (FR-9). Exactly one ends a run and it is recorded.
SUFFICIENT = "sufficient"
ITERATION_CEILING = "iteration_ceiling"
SPEND_CEILING = "spend_ceiling"
NO_PROGRESS = "no_progress"

# The Messages API rejects maxItems in a structured-output schema
# (KNOWLEDGE-2137310e); minItems and enum are accepted. Upper bounds are asked
# for in the prompt and enforced in code below, the same arrangement FR-13 uses
# for the specialist cap.
MAX_SUB_QUERIES = 5
MAX_NEXT_QUERIES = 4

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "reasoning": {"type": "string"},
    },
    "required": ["sub_queries", "reasoning"],
    "additionalProperties": False,
}

REFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "reason": {"type": "string"},
        "missing": {"type": "array", "items": {"type": "string"}},
        "next_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sufficient", "reason", "missing", "next_queries"],
    "additionalProperties": False,
}

PLAN_SYSTEM = (
    "You plan evidence retrieval over a corpus of public financial filings. "
    "Break the analyst's question into a small number of distinct search queries, "
    "each aimed at a different piece of evidence the answer needs. Use the "
    "vocabulary of the filings themselves -- statement line items, note headings, "
    "covenant and ratio names -- rather than restating the question. Do not answer "
    "the question."
)

REFLECT_SYSTEM = (
    "You judge whether retrieved excerpts from financial filings are sufficient to "
    "answer an analyst's question. Be strict: evidence that is merely topically "
    "related is not sufficient. If a specific figure, period or definition the "
    "question needs is absent, say it is insufficient, name precisely what is "
    "missing, and propose different search queries -- not rewordings of the ones "
    "already tried. Never answer from your own knowledge of these companies; only "
    "the excerpts count."
)


@dataclass
class Iteration:
    index: int
    queries: list[str]
    hit_ids: list[str]
    new_hits: int
    sufficient: bool | None = None
    reason: str = ""
    missing: list[str] = field(default_factory=list)
    next_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trace:
    question: str
    plan: list[str]
    plan_reasoning: str
    iterations: list[Iteration]
    termination: str
    started_at: str
    budget: dict

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    @property
    def revised_after_reflection(self) -> bool:
        """True when a later iteration searched something the plan did not.

        This is the property AC-6 actually cares about: not that the loop ran
        twice, but that reflection changed what it looked for.
        """
        if len(self.iterations) < 2:
            return False
        first = {q.strip().lower() for q in self.iterations[0].queries}
        return any(
            any(q.strip().lower() not in first for q in it.queries)
            for it in self.iterations[1:]
        )

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "plan": self.plan,
            "plan_reasoning": self.plan_reasoning,
            "iterations": [i.to_dict() for i in self.iterations],
            "termination": self.termination,
            "iteration_count": self.iteration_count,
            "revised_after_reflection": self.revised_after_reflection,
            "started_at": self.started_at,
            "budget": self.budget,
        }


@dataclass
class Evidence:
    """One retrieved chunk, carrying the provenance a citation is built from."""

    hit: indexing.Hit

    @property
    def citation(self) -> str:
        return self.hit.citation()


@dataclass
class RetrievalResult:
    question: str
    evidence: list[indexing.Hit]
    trace: Trace

    @property
    def citations(self) -> list[str]:
        return sorted({h.citation() for h in self.evidence})


def rank(hits: list[indexing.Hit], limit: int | None = None) -> list[indexing.Hit]:
    """Best-matching first, optionally truncated.

    Insertion order is arrival order, which is not relevance. Rendering the first
    N of an accumulating set showed reflection the same earliest chunks on every
    iteration while newly retrieved evidence was never seen at all, so the loop
    could not converge and burned its full iteration budget.
    """
    ordered = sorted(hits, key=lambda h: h.distance)
    return ordered[:limit] if limit else ordered


def rank_balanced(
    hits: list[indexing.Hit], limit: int, doc_ids: list[str] | None = None
) -> list[indexing.Hit]:
    """Take the best `limit` hits while giving each document a share.

    Balanced retrieval alone is not enough. A comparison question retrieved 40
    chunks from one issuer and 28 from the other, then the flat distance ranking
    that trims evidence to the cap discarded all 28: the specialists saw one
    issuer and correctly reported that no comparison could be made. Whatever the
    loop worked to retrieve has to survive the trim.
    """
    if not doc_ids or len(doc_ids) < 2:
        return rank(hits, limit)

    share = max(1, limit // len(doc_ids))
    picked: list[indexing.Hit] = []
    for doc_id in doc_ids:
        picked.extend(rank([h for h in hits if h.doc_id == doc_id], share))

    # Any unused share goes to the best remaining hits, whichever document they
    # come from, so a document with little to say does not waste the budget.
    if len(picked) < limit:
        chosen = {h.chunk_id for h in picked}
        picked.extend(rank([h for h in hits if h.chunk_id not in chosen], limit - len(picked)))
    return rank(picked, limit)


def _render(hits: list[indexing.Hit], limit: int = config.REFLECTION_EVIDENCE_CHUNKS) -> str:
    parts = []
    for hit in rank(hits, limit):
        parts.append(f"{hit.citation()} ({hit.kind})\n{hit.text}")
    return "\n\n---\n\n".join(parts)


def plan(question: str, model: llm.LLM) -> tuple[list[str], str, llm.Usage]:
    data, usage = model.json(
        model=config.ROUTER_MODEL,
        system=PLAN_SYSTEM,
        prompt=f"Analyst question:\n{question}\n\nProduce the search queries.",
        schema=PLAN_SCHEMA,
        purpose="plan",
    )
    queries = [q.strip() for q in data.get("sub_queries", []) if q.strip()]
    return queries[:MAX_SUB_QUERIES] or [question], data.get("reasoning", ""), usage


def reflect(question: str, hits: list[indexing.Hit], tried: list[str], model: llm.LLM):
    prompt = (
        f"Analyst question:\n{question}\n\n"
        f"Queries already tried:\n- " + "\n- ".join(tried) + "\n\n"
        f"Retrieved excerpts:\n\n{_render(hits)}\n\n"
        "Is this sufficient to answer the question from the excerpts alone?"
    )
    data, usage = model.json(
        model=config.REFLECTION_MODEL,
        system=REFLECT_SYSTEM,
        prompt=prompt,
        schema=REFLECT_SCHEMA,
        purpose="reflect",
    )
    return data, usage


def answer(
    question: str,
    model: llm.LLM | None = None,
    k: int = 6,
    max_iterations: int = config.MAX_RETRIEVAL_ITERATIONS,
    where: dict | None = None,
    index_dir=None,
    doc_ids: list[str] | None = None,
) -> RetrievalResult:
    """Run the loop until one of the three stop conditions fires."""
    model = model or llm.LLM()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    seen: dict[str, indexing.Hit] = {}
    iterations: list[Iteration] = []
    termination = ITERATION_CEILING

    try:
        queries, reasoning, _ = plan(question, model)
    except llm.BudgetExceeded:
        # Nothing has been searched yet; fall back to the raw question so the
        # run degrades to single-shot rather than returning nothing.
        queries, reasoning = [question], "planning skipped: spend ceiling reached"

    planned = list(queries)

    for index in range(1, max_iterations + 1):
        before = len(seen)
        for query in queries:
            for hit in indexing.search(
                query, k=k, where=where, index_dir=index_dir, doc_ids=doc_ids
            ):
                seen.setdefault(hit.chunk_id, hit)

        iteration = Iteration(
            index=index,
            queries=list(queries),
            hit_ids=sorted(seen),
            new_hits=len(seen) - before,
        )
        iterations.append(iteration)

        if index == max_iterations:
            termination = ITERATION_CEILING
            break

        try:
            verdict, _ = reflect(question, list(seen.values()), planned, model)
        except llm.BudgetExceeded:
            termination = SPEND_CEILING
            iteration.reason = "spend ceiling reached before reflection"
            break

        iteration.sufficient = bool(verdict.get("sufficient"))
        iteration.reason = verdict.get("reason", "")
        iteration.missing = list(verdict.get("missing", []))
        iteration.next_queries = [
            q.strip() for q in verdict.get("next_queries", []) if q.strip()
        ][:MAX_NEXT_QUERIES]

        if iteration.sufficient:
            termination = SUFFICIENT
            break

        fresh = [q for q in iteration.next_queries if q.strip().lower() not in
                 {p.strip().lower() for p in planned}]
        if not fresh:
            # Reflection says insufficient but offers nothing new to try. Looping
            # on the same queries would spend money to retrieve the same chunks.
            termination = NO_PROGRESS
            break

        queries = fresh
        planned.extend(fresh)

    trace = Trace(
        question=question,
        plan=planned,
        plan_reasoning=reasoning,
        iterations=iterations,
        termination=termination,
        started_at=started,
        budget=model.budget.to_dict(),
    )
    ranked = sorted(seen.values(), key=lambda h: h.distance)
    return RetrievalResult(question=question, evidence=ranked, trace=trace)
