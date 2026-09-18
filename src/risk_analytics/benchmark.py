"""Agentic retrieval versus single-shot retrieval (FR-8, AC-13).

FR-8 asserts that planning, reflecting and iterating beats a single vector
search. The whole project rests on that claim and nothing in it has ever tested
the claim. This measures it.

The comparison holds everything constant except retrieval. Both arms share one
index, one embedding model, one evidence cap, one router, one set of
specialists and one synthesis step. The single-shot arm issues the analyst's
question verbatim as one search and proceeds; the agentic arm runs the loop.
Any difference in the note is therefore attributable to retrieval and not to
some other advantage handed to one side.

AC-13 requires the result to be reported whichever way it falls. A benchmark
written by the author of the thing being benchmarked is worth little unless it
can return a verdict the author did not want, so the single-shot arm is given
the same evidence budget rather than a deliberately impoverished one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from . import agents, config, indexing, llm, manifest, note as note_module, retrieval

AGENTIC = "agentic"
SINGLE_SHOT = "single_shot"

# These artefacts are committed to a public repository and hold automated
# assertions about real listed companies. A reader landing on the raw JSON must
# see what it is without having to find the README, so the notice travels with
# the data rather than sitting beside it.
PUBLICATION_NOTICE = (
    "DEMONSTRATION ARTIFACT. " + note_module.DISCLAIMER + " Findings below were "
    "generated automatically from public filings by an experimental pipeline and "
    "have not been reviewed by an analyst. They are published as engineering "
    "evidence, not as an opinion on any issuer."
)


@dataclass
class ArmResult:
    arm: str
    question: str
    pages: list[str]
    citations: list[str]
    findings: int
    dropped_claims: int
    cost_usd: float
    calls: int
    iterations: int
    termination: str
    specialists: list[str]
    insufficient: bool
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Comparison:
    question: str
    agentic: ArmResult
    single_shot: ArmResult

    @property
    def pages_only_agentic(self) -> list[str]:
        return sorted(set(self.agentic.pages) - set(self.single_shot.pages))

    @property
    def pages_only_single_shot(self) -> list[str]:
        return sorted(set(self.single_shot.pages) - set(self.agentic.pages))

    @property
    def cost_ratio(self) -> float:
        return self.agentic.cost_usd / self.single_shot.cost_usd if self.single_shot.cost_usd else 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "agentic": self.agentic.to_dict(),
            "single_shot": self.single_shot.to_dict(),
            "pages_only_agentic": self.pages_only_agentic,
            "pages_only_single_shot": self.pages_only_single_shot,
            "cost_ratio": round(self.cost_ratio, 2),
        }


def single_shot_evidence(question: str, k: int, where: dict | None, doc_ids: list[str], index_dir=None):
    """One search on the raw question. No plan, no reflection, no iteration.

    `k` is raised to the agentic arm's evidence cap so the comparison is about
    *how* evidence is found rather than how much either side is allowed.
    """
    hits = indexing.search(
        question, k=max(k, config.MAX_EVIDENCE_CHUNKS), where=where,
        index_dir=index_dir, doc_ids=doc_ids,
    )
    return retrieval.rank_balanced(hits, config.MAX_EVIDENCE_CHUNKS, doc_ids)


def _measure(arm: str, question: str, note: note_module.Note, budget: llm.Budget) -> ArmResult:
    pages = sorted({c.strip("[]") for c in note.citations})
    return ArmResult(
        arm=arm,
        question=question,
        pages=pages,
        citations=note.citations,
        findings=len(note.findings),
        dropped_claims=len(note.dropped_claims),
        cost_usd=round(budget.spent_usd, 6),
        calls=budget.calls,
        iterations=note.trace.iteration_count,
        termination=note.trace.termination,
        specialists=list(note.routing.selected),
        insufficient=not note.findings,
    )


def run_single_shot(question: str, k: int = 8, where: dict | None = None, index_dir=None) -> ArmResult:
    """The control arm: retrieve once, then the identical downstream pipeline."""
    budget = llm.Budget()
    model = llm.LLM(budget=budget)

    where, scoped = note_module.scope_filter(question, where, None)
    balance = scoped or list(manifest.by_id())
    evidence = single_shot_evidence(question, k, where, balance, index_dir)

    decision = agents.route(question, model)
    findings = agents.consult_selected(decision, question, evidence, model)
    scope, key, watch, limits, dropped = note_module.synthesise(
        question, findings, evidence, model
    )

    trace = retrieval.Trace(
        question=question, plan=[question],
        plan_reasoning="single-shot control: the question is issued verbatim as one search",
        iterations=[retrieval.Iteration(
            index=1, queries=[question],
            hit_ids=sorted(h.chunk_id for h in evidence), new_hits=len(evidence),
            sufficient=None, reason="not assessed; the control arm does not reflect",
        )],
        termination="single_shot",
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        budget=budget.to_dict(),
    )
    note = note_module.Note(
        question=question, scope=scope, findings=key, by_specialist=findings,
        watch_items=watch, limitations=limits,
        contradictions=note_module.find_contradictions(findings),
        routing=decision, trace=trace,
        generated_at=trace.started_at, dropped_claims=dropped,
    )
    return _measure(SINGLE_SHOT, question, note, budget)


def run_agentic(question: str, k: int = 8, where: dict | None = None, index_dir=None) -> ArmResult:
    budget = llm.Budget()
    model = llm.LLM(budget=budget)
    note = note_module.analyse(question, model, k=k, where=where, index_dir=index_dir)
    return _measure(AGENTIC, question, note, budget)


def compare(question: str, k: int = 8, where: dict | None = None, index_dir=None) -> Comparison:
    return Comparison(
        question=question,
        agentic=run_agentic(question, k, where, index_dir),
        single_shot=run_single_shot(question, k, where, index_dir),
    )


def write_results(comparisons: list[Comparison], path) -> None:
    """Persist the comparison with its notice attached."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"_disclaimer": PUBLICATION_NOTICE,
             "comparisons": [c.to_dict() for c in comparisons]},
            indent=2,
        ),
        encoding="utf-8",
    )


def render_report(comparisons: list[Comparison]) -> str:
    """A report a sceptical reader can check, including against its author."""
    out = [
        "# Agentic retrieval versus single-shot retrieval",
        "",
        f"> **Demonstration artifact.** {note_module.DISCLAIMER}",
        "> Figures quoted below are automated output published as engineering",
        "> evidence, not as an opinion on any issuer.",
        "",
        "Both arms share one index, one embedding model, one evidence cap, one",
        "router, one set of specialists and one synthesis step. The only variable",
        "is how evidence is found: the control issues the analyst's question",
        "verbatim as a single search; the agentic arm plans sub-queries, reflects",
        "on sufficiency and iterates.",
        "",
        f"Evidence cap for both arms: {config.MAX_EVIDENCE_CHUNKS} chunks.",
        "",
        "| question | arm | pages reached | findings | cost | calls | termination |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in comparisons:
        short = c.question[:44] + "..."
        for arm in (c.agentic, c.single_shot):
            out.append(
                f"| {short} | {arm.arm} | {len(arm.pages)} | {arm.findings} | "
                f"${arm.cost_usd:.4f} | {arm.calls} | {arm.termination} |"
            )

    out += ["", "## Per question", ""]
    for c in comparisons:
        out += [f"### {c.question}", ""]
        out.append(f"- agentic reached: {', '.join(c.agentic.pages) or 'nothing'}")
        out.append(f"- single-shot reached: {', '.join(c.single_shot.pages) or 'nothing'}")
        if c.pages_only_agentic:
            out.append(f"- **only the loop reached**: {', '.join(c.pages_only_agentic)}")
        if c.pages_only_single_shot:
            out.append(f"- **only single-shot reached**: {', '.join(c.pages_only_single_shot)}")
        out.append(
            f"- cost: agentic ${c.agentic.cost_usd:.4f} against single-shot "
            f"${c.single_shot.cost_usd:.4f} ({c.cost_ratio:.1f}x)"
        )
        out.append("")
    return "\n".join(out)
