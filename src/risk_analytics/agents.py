"""Router and narrow specialists (FR-11, FR-12, FR-13).

A router reads the question and picks which specialists to invoke, recording a
reason for every selection *and* every non-selection, so the routing decision can
be argued with rather than taken on trust (FR-11).

Two rules are enforced in code rather than asked for in a prompt, because a
prompt is a request and the cost ceiling is not negotiable:

* at most two specialists per query (FR-13), truncated here if the model picks
  more;
* every claim must cite evidence that actually exists. A claim whose cited chunk
  ids are not in the retrieved set is dropped and recorded as dropped. This is
  the anti-fabrication mechanism the whole demo rests on -- a specialist writing
  a plausible sentence about Adani Green from general knowledge is precisely the
  failure that would discredit the work in front of a ratings professional.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import config, indexing, llm

CREDIT = "credit"
ESG = "esg"
COMPLIANCE = "compliance"
PEER = "peer"


@dataclass(frozen=True)
class Specialist:
    id: str
    title: str
    covers: str
    system: str


SPECIALISTS: dict[str, Specialist] = {
    CREDIT: Specialist(
        id=CREDIT,
        title="Credit analysis",
        covers=(
            "leverage, coverage and liquidity; debt quantum, maturity profile and "
            "interest-rate exposure; cash-flow quality; covenant levels and headroom"
        ),
        system=(
            "You are a credit analyst producing the evidence section of a first-pass "
            "rating rationale. Report what the filings state about leverage, coverage, "
            "liquidity, debt maturity and covenant headroom. Quote figures with the "
            "period they belong to. Do not assign a rating or a score."
        ),
    ),
    ESG: Specialist(
        id=ESG,
        title="ESG and climate",
        covers=(
            "emissions and intensity disclosure, transition and renewable capex, "
            "climate-related risk language, sustainability-linked instruments"
        ),
        system=(
            "You are an ESG analyst. Report only what the filings disclose about "
            "emissions, capacity, transition plans and climate risk. This corpus is "
            "financial statements and a quarterly results filing, which usually carry "
            "little of this. If the excerpts do not contain such disclosure, say so "
            "plainly and report nothing further. Never describe a company's "
            "environmental profile from general knowledge."
        ),
    ),
    COMPLIANCE: Specialist(
        id=COMPLIANCE,
        title="Compliance and disclosure",
        covers=(
            "auditor opinion, emphasis of matter and key audit matters; covenant "
            "compliance statements; related-party transactions; contingent "
            "liabilities; regulatory and legal proceedings"
        ),
        system=(
            "You are a disclosure analyst. Report what the filings state about the "
            "audit opinion and its qualifications, covenant compliance, related-party "
            "dealings, contingent liabilities and regulatory proceedings. Distinguish "
            "what the company asserts from what the auditor asserts."
        ),
    ),
    PEER: Specialist(
        id=PEER,
        title="Peer and sector comparison",
        covers="comparison of disclosed metrics between the issuers in the corpus",
        system=(
            "You compare issuers using only what each one disclosed. The corpus holds "
            "a full-year annual-report extract for one issuer and a single quarter's "
            "filing for the other, so every comparison must state which period each "
            "side is drawn from. If the excerpts cover only one issuer, say the "
            "comparison cannot be made."
        ),
    ),
}

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        # maxItems is rejected by the API (KNOWLEDGE-2137310e); route()
        # truncates to MAX_SPECIALISTS_PER_QUERY regardless, which is where the
        # limit has to live anyway -- a schema the model satisfies is not a
        # ceiling on what the run may spend.
        "selected": {
            "type": "array",
            "items": {"type": "string", "enum": list(SPECIALISTS)},
        },
        "reasons": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in SPECIALISTS},
            "required": list(SPECIALISTS),
            "additionalProperties": False,
        },
    },
    "required": ["selected", "reasons"],
    "additionalProperties": False,
}

FINDING_SCHEMA = {
    "type": "object",
    "properties": {
        "insufficient_evidence": {"type": "boolean"},
        "note": {"type": "string"},
        "claims": {
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
    },
    "required": ["insufficient_evidence", "note", "claims"],
    "additionalProperties": False,
}

ROUTE_SYSTEM = (
    "You route an analyst's question to the specialists best placed to answer it "
    "from a corpus of public financial filings. Select only those whose scope the "
    "question actually requires; selecting one with nothing to contribute wastes "
    "the run's budget and dilutes the answer. Give a one-sentence reason for every "
    "specialist, whether selected or not."
)


@dataclass
class RoutingDecision:
    selected: list[str]
    reasons: dict[str, str]
    truncated: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Claim:
    text: str
    evidence_ids: list[str]
    citations: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Finding:
    specialist: str
    claims: list[Claim]
    insufficient_evidence: bool
    note: str
    dropped_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "specialist": self.specialist,
            "claims": [c.to_dict() for c in self.claims],
            "insufficient_evidence": self.insufficient_evidence,
            "note": self.note,
            "dropped_claims": self.dropped_claims,
        }


def _catalogue() -> str:
    return "\n".join(f"- {s.id}: {s.title}. Covers {s.covers}." for s in SPECIALISTS.values())


def route(question: str, model: llm.LLM, max_specialists: int = config.MAX_SPECIALISTS_PER_QUERY) -> RoutingDecision:
    """Choose specialists for one question (FR-11, FR-13)."""
    data, _ = model.json(
        model=config.ROUTER_MODEL,
        system=ROUTE_SYSTEM,
        prompt=f"Specialists available:\n{_catalogue()}\n\nAnalyst question:\n{question}",
        schema=ROUTE_SCHEMA,
        purpose="route",
    )

    picked = [s for s in data.get("selected", []) if s in SPECIALISTS]
    seen, ordered = set(), []
    for s in picked:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    # FR-13 enforced here, not requested in the prompt. The ceiling is the
    # reason the run stays affordable; a model that ignores the instruction
    # must not be able to spend the budget anyway.
    truncated = ordered[max_specialists:]
    selected = ordered[:max_specialists]

    reasons = {k: str(data.get("reasons", {}).get(k, "")).strip() for k in SPECIALISTS}
    for s in truncated:
        reasons[s] = (
            f"{reasons.get(s, '')} (not invoked: the router selected more than "
            f"{max_specialists} specialists and this one fell outside the limit)"
        ).strip()

    return RoutingDecision(selected=selected, reasons=reasons, truncated=truncated)


def _evidence_block(hits: list[indexing.Hit]) -> str:
    return "\n\n---\n\n".join(
        f"id: {h.chunk_id}\nsource: {h.citation()} ({h.kind})\n{h.text}" for h in hits
    )


def consult(
    specialist_id: str,
    question: str,
    hits: list[indexing.Hit],
    model: llm.LLM,
) -> Finding:
    """Run one specialist over retrieved evidence.

    Claims citing evidence that is not in `hits` are dropped rather than
    reported. The model is told to cite ids; it is not trusted to have done so.
    """
    specialist = SPECIALISTS[specialist_id]
    by_id = {h.chunk_id: h for h in hits}

    if not hits:
        return Finding(
            specialist=specialist_id,
            claims=[],
            insufficient_evidence=True,
            note="no evidence was retrieved for this question",
        )

    prompt = (
        f"Analyst question:\n{question}\n\n"
        f"Evidence excerpts, each with an id:\n\n{_evidence_block(hits)}\n\n"
        "Report only what these excerpts support, and cite the id of every excerpt "
        "a claim rests on. If they do not support an answer within your scope, set "
        "insufficient_evidence and explain what is missing."
    )
    data, _ = model.json(
        model=config.SPECIALIST_MODEL,
        system=specialist.system,
        prompt=prompt,
        schema=FINDING_SCHEMA,
        purpose=f"specialist:{specialist_id}",
    )

    claims, dropped = [], []
    for raw in data.get("claims", []):
        ids = [i for i in raw.get("evidence_ids", []) if i in by_id]
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        if not ids:
            # Cited nothing that exists. This is the fabrication case.
            dropped.append(text)
            continue
        claims.append(
            Claim(
                text=text,
                evidence_ids=ids,
                citations=sorted({by_id[i].citation() for i in ids}),
            )
        )

    insufficient = bool(data.get("insufficient_evidence")) or not claims
    return Finding(
        specialist=specialist_id,
        claims=claims,
        insufficient_evidence=insufficient,
        note=str(data.get("note", "")).strip(),
        dropped_claims=dropped,
    )


def consult_selected(
    decision: RoutingDecision,
    question: str,
    hits: list[indexing.Hit],
    model: llm.LLM,
) -> list[Finding]:
    findings = []
    for specialist_id in decision.selected:
        try:
            findings.append(consult(specialist_id, question, hits, model))
        except llm.BudgetExceeded:
            findings.append(
                Finding(
                    specialist=specialist_id,
                    claims=[],
                    insufficient_evidence=True,
                    note="not consulted: the run reached its spend ceiling first",
                )
            )
            break
    return findings
