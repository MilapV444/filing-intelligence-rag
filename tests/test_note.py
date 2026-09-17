"""Synthesis, the rendered note, and the CLI (FR-14 to FR-19, NFR-7, AC-4, AC-10)."""

import json

import pytest

from risk_analytics import agents, cli, config, indexing, llm, note as note_module
from test_retrieval_loop import fake_llm, plan_reply, reflect_reply  # noqa: F401
from test_router import finding_reply, route_reply  # noqa: F401


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("index")
    indexing.ingest(["apsez-q1fy27"], index_dir=path)
    return path


@pytest.fixture(scope="module")
def hits(index_dir):
    return indexing.search("net gearing DSCR covenant headroom", k=6, index_dir=index_dir)


def synthesis_reply(findings, scope="APSEZ covenant position as at 30 June 2026",
                    watch=("refinancing of current maturities",), limits=("single quarter only",)):
    return {
        "scope": scope,
        "key_findings": [{"text": t, "evidence_ids": list(ids)} for t, ids in findings],
        "watch_items": list(watch),
        "limitations": list(limits),
    }


def full_script(hits, *, selected=(agents.CREDIT,), claims=None, key=None):
    claims = claims if claims is not None else [("Net gearing is 0.55 against a 3.0x covenant", [hits[0].chunk_id])]
    key = key if key is not None else [("Headroom against the gearing covenant is wide", [hits[0].chunk_id])]
    return [
        plan_reply("net gearing", "DSCR"),
        reflect_reply(True, "both present"),
        route_reply(list(selected), credit="leverage and covenants"),
        *[finding_reply(claims) for _ in selected],
        synthesis_reply(key),
    ]


# --- Citations are structural (FR-16) ----------------------------------------


def test_every_key_finding_carries_a_citation(hits, index_dir):
    model = fake_llm(full_script(hits))
    note = note_module.analyse("What is APSEZ's covenant headroom?", model, index_dir=index_dir)

    assert note.findings
    for claim in note.findings:
        assert claim.citations, f"uncited claim: {claim.text}"
        assert all(c.startswith("[apsez-q1fy27 p.") for c in claim.citations)


def test_a_synthesised_finding_citing_unretrieved_evidence_is_dropped(hits, index_dir):
    """The model never writes a page number; it names an id, and the id must be
    one that was actually retrieved."""
    model = fake_llm(full_script(
        hits,
        key=[("A confident claim about a page nobody retrieved", ["apsez-q1fy27:p999:table0"])],
    ))
    note = note_module.analyse("q", model, index_dir=index_dir)

    assert note.findings == []
    assert "A confident claim about a page nobody retrieved" in note.dropped_claims


def test_citations_resolve_to_real_pages_of_the_registered_document(hits, index_dir):
    model = fake_llm(full_script(hits))
    note = note_module.analyse("q", model, index_dir=index_dir)
    pages = {h.page_number for h in hits}
    for citation in note.citations:
        page = int(citation.rsplit("p.", 1)[1].rstrip("]"))
        assert page in pages


# --- Disagreement is surfaced, not resolved (FR-14) --------------------------


CLAIM_A = (
    "The company reports consolidated covenant compliance certified against every "
    "outstanding debenture threshold during the quarter ended June 2026"
)
CLAIM_B = (
    "The company does not report consolidated covenant compliance certified against "
    "every outstanding debenture threshold during the quarter ended June 2026"
)


def test_contradicting_specialists_both_appear():
    findings = [
        agents.Finding(
            specialist=agents.CREDIT,
            claims=[agents.Claim(CLAIM_A, ["a"], ["[x p.1]"])],
            insufficient_evidence=False, note="",
        ),
        agents.Finding(
            specialist=agents.COMPLIANCE,
            claims=[agents.Claim(CLAIM_B, ["b"], ["[x p.2]"])],
            insufficient_evidence=False, note="",
        ),
    ]
    clashes = note_module.find_contradictions(findings)

    assert clashes, "opposing claims on the same subject were not flagged"
    assert set(clashes[0].specialists) == {agents.CREDIT, agents.COMPLIANCE}
    assert len(clashes[0].claims) == 2


def test_agreeing_specialists_are_not_flagged():
    findings = [
        agents.Finding(agents.CREDIT, [agents.Claim(CLAIM_A, ["a"], ["[x p.1]"])], False, ""),
        agents.Finding(agents.COMPLIANCE, [agents.Claim(CLAIM_A, ["b"], ["[x p.2]"])], False, ""),
    ]
    assert note_module.find_contradictions(findings) == []


def test_claims_that_merely_share_vocabulary_are_not_flagged():
    """Measured on a real run: at three shared words the detector flagged 11
    pairs out of 16 claims, almost all of them two specialists agreeing about
    different measures. Noise at that volume buries any real conflict."""
    findings = [
        agents.Finding(agents.CREDIT, [agents.Claim(
            "Consolidated gearing stands at 0.55x against the stated threshold", ["a"], ["[x p.1]"]
        )], False, ""),
        agents.Finding(agents.COMPLIANCE, [agents.Claim(
            "The Regulation 52 ratio set is not identical to the covenant table", ["b"], ["[x p.2]"]
        )], False, ""),
    ]
    assert note_module.find_contradictions(findings) == []


def test_at_most_a_handful_of_pairs_are_reported():
    many = [
        agents.Finding(agents.CREDIT, [
            agents.Claim(CLAIM_A + f" variant {i}", ["a"], ["[x p.1]"]) for i in range(6)
        ], False, ""),
        agents.Finding(agents.COMPLIANCE, [
            agents.Claim(CLAIM_B + f" variant {i}", ["b"], ["[x p.2]"]) for i in range(6)
        ], False, ""),
    ]
    assert len(note_module.find_contradictions(many)) <= note_module.MAX_REPORTED


def test_the_rendered_note_shows_both_sides_of_a_disagreement(hits, index_dir):
    model = fake_llm(full_script(hits))
    note = note_module.analyse("q", model, index_dir=index_dir)
    note.contradictions = [note_module.Contradiction(
        specialists=[agents.CREDIT, agents.COMPLIANCE],
        claims=["Covenants are met", "Covenants are not met"],
    )]
    markdown = note_module.render(note, model.budget)

    assert "Possible specialist disagreement" in markdown
    assert "Covenants are met" in markdown
    assert "Covenants are not met" in markdown
    assert "rather than resolved" in markdown
    # The section must not present a heuristic guess as a verified finding.
    assert "not verified" in markdown


# --- The note's shape (FR-15, FR-17, FR-18, AC-10) ---------------------------


def test_the_note_has_the_sections_a_rationale_note_needs(hits, index_dir):
    model = fake_llm(full_script(hits))
    markdown = note_module.render(
        note_module.analyse("q", model, index_dir=index_dir), model.budget
    )
    for heading in ("## Scope", "## Key findings", "## Supporting analysis",
                    "## Limitations", "## Routing", "## Sources", "## Run record"):
        assert heading in markdown, f"missing {heading}"


def test_the_note_carries_the_disclaimer(hits, index_dir):
    """AC-10. This is shown to a ratings professional; it must not read as a
    rating."""
    model = fake_llm(full_script(hits))
    markdown = note_module.render(
        note_module.analyse("q", model, index_dir=index_dir), model.budget
    )
    assert "not a credit rating" in markdown
    assert "not investment advice" in markdown
    assert "not been reviewed by a rating committee" in markdown


def test_the_note_reports_cost_iterations_and_specialists(hits, index_dir):
    """FR-18."""
    model = fake_llm(full_script(hits))
    note = note_module.analyse("q", model, index_dir=index_dir)
    markdown = note_module.render(note, model.budget)

    assert "Retrieval iterations:" in markdown
    assert "Model calls:" in markdown
    assert "Cost: $" in markdown
    assert f"${model.budget.spent_usd:.4f}" in markdown
    assert "Specialists invoked:" in markdown


def test_the_note_names_every_source_with_its_public_url(hits, index_dir):
    model = fake_llm(full_script(hits))
    markdown = note_module.render(
        note_module.analyse("q", model, index_dir=index_dir), model.budget
    )
    assert "https://" in markdown
    assert "apsez-q1fy27" in markdown


def test_the_note_records_non_selection_reasons(hits, index_dir):
    """FR-11 reaches the reader: a specialist that was not consulted says why."""
    model = fake_llm(full_script(hits))
    markdown = note_module.render(
        note_module.analyse("q", model, index_dir=index_dir), model.budget
    )
    assert "not invoked" in markdown
    for specialist_id in agents.SPECIALISTS:
        assert f"**{specialist_id}**" in markdown


# --- Degraded runs stay honest -----------------------------------------------


def test_a_run_with_no_supported_findings_says_so_rather_than_inventing(hits, index_dir):
    model = fake_llm([
        plan_reply("net gearing"),
        reflect_reply(True, "ok"),
        route_reply([agents.ESG]),
        finding_reply([], insufficient=True, note="no emissions disclosure in the excerpts"),
    ])
    note = note_module.analyse("What are the emissions?", model, index_dir=index_dir)
    markdown = note_module.render(note, model.budget)

    assert note.findings == []
    assert "_No finding survived citation checking._" in markdown
    assert "Insufficient evidence" in markdown
    assert "no emissions disclosure" in markdown


def test_stopping_early_is_disclosed_in_the_limitations(hits, index_dir):
    """A note resting on a truncated search must say so."""
    model = fake_llm([
        plan_reply("q1"),
        reflect_reply(False, "insufficient", missing=["x"], next_queries=["q1"]),
        route_reply([agents.CREDIT]),
        finding_reply([("a claim", [hits[0].chunk_id])]),
        synthesis_reply([("a finding", [hits[0].chunk_id])]),
    ])
    note = note_module.analyse("q", model, index_dir=index_dir)
    markdown = note_module.render(note, model.budget)

    assert any("no progress" in l or "Retrieval stopped" in l for l in note.limitations)
    assert "Retrieval stopped on" in markdown


def test_insufficient_specialists_are_listed_as_limitations(hits, index_dir):
    model = fake_llm([
        plan_reply("q"), reflect_reply(True, "ok"),
        route_reply([agents.ESG]),
        finding_reply([], insufficient=True, note="corpus carries no climate disclosure"),
    ])
    note = note_module.analyse("q", model, index_dir=index_dir)
    assert any("climate disclosure" in l for l in note.limitations)


# --- Provenance (NFR-7) -------------------------------------------------------


def test_the_written_trace_reconstructs_the_whole_run(tmp_path, hits, index_dir):
    model = fake_llm(full_script(hits))
    note = note_module.analyse("What is the headroom?", model, index_dir=index_dir)
    markdown_path, trace_path = note_module.write(note, model.budget, notes_dir=tmp_path)

    assert markdown_path.exists() and trace_path.exists()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))

    assert payload["question"] == "What is the headroom?"
    assert payload["trace"]["plan"]
    assert payload["trace"]["iterations"][0]["hit_ids"]
    assert payload["routing"]["selected"] == [agents.CREDIT]
    assert payload["budget"]["by_call"]
    assert payload["citations"]
    # Every cited page must be traceable back to a chunk the loop actually saw.
    seen = {i for it in payload["trace"]["iterations"] for i in it["hit_ids"]}
    for claim in payload["key_findings"]:
        assert set(claim["evidence_ids"]) <= seen


def test_the_markdown_and_trace_are_written_side_by_side(tmp_path, hits, index_dir):
    model = fake_llm(full_script(hits))
    note = note_module.analyse("q", model, index_dir=index_dir)
    markdown_path, trace_path = note_module.write(note, model.budget, notes_dir=tmp_path)
    assert markdown_path.stem == trace_path.stem.replace(".trace", "")


# --- CLI (FR-19, AC-4) --------------------------------------------------------


def test_the_cli_exposes_ingest_and_ask_separately():
    """AC-4 and FR-19: one command per note, and analysis does not re-ingest."""
    parser = cli.build_parser()
    ingest = parser.parse_args(["ingest"])
    assert ingest.command == "ingest" and ingest.func is cli.cmd_ingest

    ask = parser.parse_args(["ask", "What is the headroom?"])
    assert ask.command == "ask" and ask.func is cli.cmd_ask
    assert ask.question == "What is the headroom?"
    assert ask.ceiling == config.RUN_COST_CEILING_USD


def test_the_cli_accepts_a_spend_ceiling_and_a_table_filter():
    args = cli.build_parser().parse_args(["ask", "q", "--tables-only", "--ceiling", "0.10"])
    assert args.tables_only is True
    assert args.ceiling == 0.10


def test_the_cli_reports_a_missing_key_without_a_traceback(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(config, "ENV_FILE", config.ROOT / "does-not-exist.env")

    def boom(_args):
        raise config.MissingCredentialError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(cli, "cmd_ingest", boom)
    parser_args = cli.build_parser().parse_args(["ingest"])
    parser_args.func = boom

    monkeypatch.setattr(cli, "build_parser", lambda: _StubParser(parser_args))
    assert cli.main(["ingest"]) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


class _StubParser:
    def __init__(self, args):
        self._args = args

    def parse_args(self, argv=None):
        return self._args


def test_the_sources_table_marks_which_documents_were_actually_used(hits, index_dir):
    """A note citing one filing must not read as covering the whole corpus: the
    table lists every registered document, so it has to say which were drawn on."""
    model = fake_llm(full_script(hits))
    note = note_module.analyse("q", model, index_dir=index_dir)
    markdown = note_module.render(note, model.budget)

    assert "used here" in markdown
    assert "| cited |" in markdown
    assert "| not cited |" in markdown, (
        "agel-fy25 contributed nothing to this note and should be marked as such"
    )
