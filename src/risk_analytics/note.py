"""Synthesis and the analysis note (FR-14 to FR-18, NFR-7, AC-4, AC-10).

The note is what a credit professional actually reads, so two things govern its
construction:

* Citations are assembled from retrieved-chunk metadata, never from model output
  (FR-16). A specialist's claim carries chunk ids; those ids were already
  validated against the retrieved set in `agents.consult`, and the page numbers
  printed here are read from the store. The model never writes a page number.
* Disagreement is surfaced, not resolved (FR-14). Two specialists contradicting
  each other is information a reader needs. Silently keeping the more confident
  one is how a synthesis step launders uncertainty into false authority.

Everything the note asserts is traceable back through the trace to the chunk and
page it came from (NFR-7).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import agents, config, indexing, llm, manifest, retrieval

DISCLAIMER = (
    "This note is a demonstration artifact produced by an automated system from "
    "public documents. It is not a credit rating, not investment advice, and has "
    "not been reviewed by a rating committee. Every claim carries a citation to "
    "the source document and page; verify anything you intend to rely on."
)

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "scope": {"type": "string"},
        "key_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                },
                "required": ["text", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "watch_items": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["scope", "key_findings", "watch_items", "limitations"],
    "additionalProperties": False,
}

SYNTHESIS_SYSTEM = (
    "You assemble the summary of a first-pass credit rationale note from specialist "
    "findings. Write for a credit analyst: specific, sourced, no marketing language. "
    "Every key finding must cite the evidence ids it rests on. Do not introduce any "
    "fact the specialists did not report. Do not assign a rating, a score, or an "
    "outlook. Where specialists disagree, say so rather than choosing between them."
)


@dataclass
class Contradiction:
    specialists: list[str]
    claims: list[str]


@dataclass
class Note:
    question: str
    scope: str
    findings: list[agents.Claim]
    by_specialist: list[agents.Finding]
    watch_items: list[str]
    limitations: list[str]
    contradictions: list[Contradiction]
    routing: agents.RoutingDecision
    trace: retrieval.Trace
    generated_at: str
    dropped_claims: list[str] = field(default_factory=list)

    @property
    def citations(self) -> list[str]:
        cited = {c for claim in self.findings for c in claim.citations}
        for finding in self.by_specialist:
            for claim in finding.claims:
                cited.update(claim.citations)
        return sorted(cited)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "scope": self.scope,
            "generated_at": self.generated_at,
            "key_findings": [c.to_dict() for c in self.findings],
            "by_specialist": [f.to_dict() for f in self.by_specialist],
            "watch_items": self.watch_items,
            "limitations": self.limitations,
            "contradictions": [
                {"specialists": c.specialists, "claims": c.claims} for c in self.contradictions
            ],
            "routing": self.routing.to_dict(),
            "trace": self.trace.to_dict(),
            "dropped_claims": self.dropped_claims,
            "citations": self.citations,
        }


NEGATION = re.compile(r"\b(no|not|never|without|absent|insufficient|cannot|unable)\b", re.I)


# Tuned against a real run. At three shared words the detector flagged 11 pairs
# from 16 claims, nearly all of them two specialists agreeing in different words;
# printing those buries whatever real disagreement exists. Eight leaves three.
MIN_SHARED_WORDS = 8
MAX_REPORTED = 3


def find_contradictions(findings: list[agents.Finding]) -> list[Contradiction]:
    """Flag pairs of claims that may disagree, for a human to judge.

    This is a lexical heuristic, not a finding: it pairs claims from different
    specialists that share substantial vocabulary and differ in polarity. It
    cannot tell a contradiction from two correct statements about different
    measures, so the note labels what it prints as possible and unverified.

    It is kept, rather than trusted to the synthesis step alone, because FR-14's
    risk is a synthesis quietly picking a winner: an independent pass at least
    puts the candidates in front of the reader.
    """
    contradictions: list[Contradiction] = []
    seen: set[frozenset[str]] = set()
    claims = [(f.specialist, c.text) for f in findings for c in f.claims]

    for i, (spec_a, text_a) in enumerate(claims):
        for spec_b, text_b in claims[i + 1 :]:
            if spec_a == spec_b:
                continue
            words_a = {w.lower() for w in re.findall(r"[A-Za-z]{5,}", text_a)}
            words_b = {w.lower() for w in re.findall(r"[A-Za-z]{5,}", text_b)}
            if len(words_a & words_b) < MIN_SHARED_WORDS:
                continue
            if bool(NEGATION.search(text_a)) == bool(NEGATION.search(text_b)):
                continue
            pair = frozenset({text_a, text_b})
            if pair in seen:
                continue
            seen.add(pair)
            contradictions.append(
                Contradiction(specialists=[spec_a, spec_b], claims=[text_a, text_b])
            )
    return contradictions[:MAX_REPORTED]


def synthesise(
    question: str,
    findings: list[agents.Finding],
    hits: list[indexing.Hit],
    model: llm.LLM,
) -> tuple[str, list[agents.Claim], list[str], list[str], list[str]]:
    """Merge specialist findings. Returns scope, findings, watch items,
    limitations and any claims dropped for citing evidence that does not exist."""
    by_id = {h.chunk_id: h for h in hits}
    supported = [c for f in findings for c in f.claims]

    if not supported:
        return (
            "No supported findings were produced for this question.",
            [], [],
            ["Every specialist reported insufficient evidence in the retrieved excerpts."],
            [],
        )

    reported = "\n\n".join(
        f"[{f.specialist}] {'INSUFFICIENT EVIDENCE: ' + f.note if f.insufficient_evidence else ''}\n"
        + "\n".join(f"- {c.text} (evidence: {', '.join(c.evidence_ids)})" for c in f.claims)
        for f in findings
    )
    data, _ = model.json(
        model=config.SYNTHESIS_MODEL,
        system=SYNTHESIS_SYSTEM,
        prompt=(
            f"Analyst question:\n{question}\n\nSpecialist findings:\n\n{reported}\n\n"
            "Assemble the note summary. Cite evidence ids on every key finding."
        ),
        schema=SYNTHESIS_SCHEMA,
        purpose="synthesis",
    )

    key, dropped = [], []
    for raw in data.get("key_findings", []):
        text = str(raw.get("text", "")).strip()
        ids = [i for i in raw.get("evidence_ids", []) if i in by_id]
        if not text:
            continue
        if not ids:
            dropped.append(text)
            continue
        key.append(
            agents.Claim(
                text=text,
                evidence_ids=ids,
                citations=sorted({by_id[i].citation() for i in ids}),
            )
        )

    return (
        str(data.get("scope", "")).strip(),
        key,
        [str(w).strip() for w in data.get("watch_items", []) if str(w).strip()],
        [str(l).strip() for l in data.get("limitations", []) if str(l).strip()],
        dropped,
    )


def scope_filter(question: str, where: dict | None, doc_ids: list[str] | None) -> tuple[dict | None, list[str]]:
    """Combine an explicit filter with issuer scoping inferred from the question.

    Returns the filter and the documents it restricts to, so the note can state
    plainly which parts of the corpus were searched.
    """
    scoped = doc_ids if doc_ids is not None else manifest.match_documents(question)
    if not scoped:
        return where, []
    doc_clause = {"doc_id": scoped[0]} if len(scoped) == 1 else {"doc_id": {"$in": scoped}}
    combined = {"$and": [where, doc_clause]} if where else doc_clause
    return combined, scoped


def analyse(
    question: str,
    model: llm.LLM | None = None,
    k: int = 8,
    where: dict | None = None,
    index_dir: Path | None = None,
    doc_ids: list[str] | None = None,
) -> Note:
    """Question in, cited note out. The whole pipeline in one call (FR-19)."""
    model = model or llm.LLM()

    where, scoped_docs = scope_filter(question, where, doc_ids)
    # Unscoped means the question spans the corpus, so every document gets a
    # share of each query rather than letting the largest one crowd the rest out.
    balance = scoped_docs or list(manifest.by_id())
    result = retrieval.answer(
        question, model, k=k, where=where, index_dir=index_dir, doc_ids=balance
    )

    # Only the best-ranked evidence reaches the expensive models. The loop
    # accumulates across iterations, and sending all of it is what took the
    # first live run to $0.43 against a $0.25 ceiling.
    evidence = retrieval.rank_balanced(
        result.evidence, config.MAX_EVIDENCE_CHUNKS, balance
    )

    decision = agents.route(question, model)
    findings = agents.consult_selected(decision, question, evidence, model)

    try:
        scope, key, watch, limits, dropped = synthesise(question, findings, evidence, model)
    except llm.BudgetExceeded:
        scope = "Synthesis was not run: the run reached its spend ceiling."
        key, watch, dropped = [], [], []
        limits = ["The note stops at specialist findings; no synthesis was produced."]

    limits = list(limits)
    for finding in findings:
        if finding.insufficient_evidence and finding.note:
            limits.append(f"{agents.SPECIALISTS[finding.specialist].title}: {finding.note}")
    for finding in findings:
        dropped.extend(finding.dropped_claims)
    if scoped_docs:
        limits.append(
            "Retrieval was restricted to "
            + ", ".join(sorted(scoped_docs))
            + " because the question names that issuer; other corpus documents were "
            "not searched."
        )
    if result.trace.termination != retrieval.SUFFICIENT:
        limits.append(
            f"Retrieval stopped on {result.trace.termination.replace('_', ' ')} rather than "
            "on sufficient evidence, so the note may rest on an incomplete search."
        )

    return Note(
        question=question,
        scope=scope,
        findings=key,
        by_specialist=findings,
        watch_items=watch,
        limitations=limits,
        contradictions=find_contradictions(findings),
        routing=decision,
        trace=result.trace,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dropped_claims=dropped,
    )


def _sources_table(cited_doc_ids: set[str]) -> list[str]:
    """List the corpus and mark which documents this note actually drew on.

    Printing the whole corpus without that column implies every document was
    consulted. A reader glancing at the table would take a note citing one filing
    as covering both issuers, which is a claim the note does not make.
    """
    lines = [
        "| id | issuer | document | published | used here | source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for doc in manifest.load():
        used = "cited" if doc.doc_id in cited_doc_ids else "not cited"
        lines.append(
            f"| `{doc.doc_id}` | {doc.issuer} | {doc.title} | {doc.published} | "
            f"{used} | {doc.source_url} |"
        )
    return lines


def render(note: Note, budget: llm.Budget) -> str:
    """Render the note as Markdown (FR-15, FR-16, FR-17, FR-18)."""
    out: list[str] = [
        f"# First-pass analysis note",
        "",
        f"**Question.** {note.question}",
        "",
        f"> {DISCLAIMER}",
        "",
        "## Scope",
        "",
        note.scope or "No scope statement was produced.",
        "",
        "## Key findings",
        "",
    ]

    if note.findings:
        for claim in note.findings:
            out.append(f"- {claim.text} {' '.join(claim.citations)}")
    else:
        out.append("_No finding survived citation checking._")
    out.append("")

    if note.contradictions:
        out += [
            "## Possible specialist disagreement",
            "",
            "_Flagged by a lexical heuristic and not verified. These pairs share "
            "substantial wording and differ in polarity, which often means two "
            "correct statements about different measures rather than a genuine "
            "conflict. Surfaced rather than resolved so a reader can judge; both "
            "readings are reported as given._",
            "",
        ]
        for clash in note.contradictions:
            out.append(f"- **{' vs '.join(clash.specialists)}**")
            for text in clash.claims:
                out.append(f"  - {text}")
        out.append("")

    out += ["## Supporting analysis", ""]
    for finding in note.by_specialist:
        specialist = agents.SPECIALISTS[finding.specialist]
        out.append(f"### {specialist.title}")
        out.append("")
        if finding.insufficient_evidence and not finding.claims:
            out.append(f"_Insufficient evidence._ {finding.note}")
        else:
            for claim in finding.claims:
                out.append(f"- {claim.text} {' '.join(claim.citations)}")
            if finding.note:
                out += ["", f"_{finding.note}_"]
        out.append("")

    if note.watch_items:
        out += ["## Watch items", ""] + [f"- {w}" for w in note.watch_items] + [""]

    out += ["## Limitations", ""]
    out += [f"- {l}" for l in note.limitations] or ["- None recorded."]
    out.append("")

    out += ["## Routing", "",
            f"Specialists invoked: {', '.join(note.routing.selected) or 'none'}.", ""]
    for specialist_id, reason in sorted(note.routing.reasons.items()):
        mark = "invoked" if specialist_id in note.routing.selected else "not invoked"
        out.append(f"- **{specialist_id}** ({mark}): {reason}")
    out.append("")

    cited_docs = {c.split()[0].lstrip("[") for c in note.citations}
    out += ["## Sources", ""] + _sources_table(cited_docs) + [""]
    if note.citations:
        out += ["Pages cited: " + ", ".join(f"`{c}`" for c in note.citations), ""]

    out += [
        "## Run record",
        "",
        f"- Retrieval iterations: {note.trace.iteration_count}",
        f"- Terminated on: {note.trace.termination}",
        f"- Query revised after reflection: {'yes' if note.trace.revised_after_reflection else 'no'}",
        f"- Model calls: {budget.calls}",
        f"- Cost: ${budget.spent_usd:.4f} of a ${budget.ceiling_usd:.2f} ceiling",
        f"- Generated: {note.generated_at}",
        "",
    ]
    if note.dropped_claims:
        out += [
            f"- Claims dropped for citing evidence that was not retrieved: {len(note.dropped_claims)}",
            "",
        ]
    return "\n".join(out)


def write(note: Note, budget: llm.Budget, notes_dir: Path | None = None) -> tuple[Path, Path]:
    """Write the Markdown note and its machine-readable trace side by side."""
    directory = notes_dir or config.NOTES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = note.generated_at.replace(":", "").replace("-", "")
    stem = f"note-{stamp}"

    markdown_path = directory / f"{stem}.md"
    markdown_path.write_text(render(note, budget), encoding="utf-8")

    trace_path = directory / f"{stem}.trace.json"
    payload = note.to_dict()
    payload["budget"] = budget.to_dict()
    trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return markdown_path, trace_path
