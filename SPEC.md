# Product specification — Financial-Document Intelligence Demo

> Status: draft, awaiting human approval. A coding agent must not implement product code until this specification is approved through Genesis.

## Problem

Credit and DCM professionals read long public financial documents — annual reports, bond offer documents, disclosures — to form a first view on an issuer. The evidence they need is scattered across prose, financial tables, and charts, and a single-shot vector search over a naively chunked PDF retrieves the wrong granularity: it finds paragraphs that mention leverage but not the maturity table that settles the question.

This project is a small, local, end-to-end demonstration that a multi-agent retrieval architecture handles that problem better than single-shot RAG, and that its output can be held to an auditable citation standard. It is a personal proof-of-work artifact, not a production system and not a rating tool. Its success is measured by whether a credit-ratings professional, reading the generated note beside the source PDFs, finds the claims accurate and the citations verifiable.

## Users

- **Primary — the author (Milap Vaghasia).** Runs the demo locally from a CLI, iterates on it, and demonstrates it live.
- **Audience — a credit-ratings / DCM professional at Adani Group.** Does not run the software. Reads a generated analysis note and judges whether the reasoning and citations hold up against the source documents. Non-technical with respect to the implementation; expert with respect to the content.
- **Affected third party — the issuers whose public documents are analysed.** They do not use the system, but the system makes assertions about them, which is why citation fidelity and public-source-only sourcing are hard requirements rather than preferences.

## Functional requirements

### Ingestion and page routing

- FR-1: The system ingests PDF documents from a local corpus directory and records, for every document, its title, issuer entity, document type, publication date, and the public source URL it was obtained from.
- FR-2: The system classifies every page of every ingested document into exactly one of three classes — text, table, or chart — using PyMuPDF layout statistics (text density, ruling-line count, image-area fraction), with no API call required for classification.
- FR-3: The system routes each page to an extraction method matching its class: prose extraction for text pages and structure-preserving table extraction for table pages. Pages classified as chart are recorded and skipped, not interpreted; see Constraints.
- FR-4: The system records the assigned class, the routing decision, and the extraction method used for every page, so that a routing decision can be inspected and disputed after the fact.

### Indexing

- FR-5: The system splits extracted content into retrievable chunks that preserve table integrity — a table is not split mid-row — and attaches to every chunk its source document identifier, page number, and page class.
- FR-6: The system embeds chunks with a locally executed sentence-transformers model and persists them in a local vector store (Chroma or FAISS) under the project directory, with no cloud storage and no network call at embedding time.
- FR-7: Re-running ingestion over an unchanged corpus reuses the existing index rather than re-embedding, and re-embeds only documents whose content hash has changed.

### Agentic retrieval

- FR-8: The system answers a query with an agentic retrieval loop that plans sub-queries, searches the index, reflects on whether retrieved evidence answers the question, iterates with revised queries when it does not, and synthesizes a final answer — rather than a single retrieval followed by one generation.
- FR-9: The retrieval loop terminates on exactly one of four conditions, whichever comes first, and records which one ended it: sufficiency; an iteration ceiling; a token-spend ceiling; or no progress, meaning reflection reports the evidence insufficient but proposes no query that was not already tried. The fourth was implemented at T6A and shipped undeclared (DECISION-975bc5c0); it is stated here so the specification matches the code. A fifth condition for an unclosable evidence gap was drafted and deliberately not built, so it is not claimed.
- FR-10: The system emits a machine-readable trace of each run: the plan, every sub-query issued, the chunks retrieved for each, every reflection verdict, and the specialists invoked.

### Multi-agent analysis

- FR-11: A router agent inspects the query and selects which specialists to invoke, and records a stated reason for each selection and each non-selection.
- FR-12: The system implements four narrow specialists — credit analysis, ESG/climate, compliance/disclosure, and peer/sector comparison — each with its own scope, prompt, and output shape, and each able to answer only from retrieved corpus evidence.
- FR-13: The router invokes at most two specialists for any single query.
- FR-14: A synthesis step merges specialist outputs into one coherent note, and surfaces rather than silently resolves any contradiction between two specialists.

### Output

- FR-15: The system writes a structured Markdown analysis note to disk, in the shape of a first-pass credit-rating rationale: issuer and scope, key findings, supporting analysis by theme, risks and watch items, and explicit limitations.
- FR-16: Every material factual claim in the note carries an inline citation naming the source document and page number.
- FR-17: The note states plainly that it is a demonstration artifact produced by an automated system from public documents, that it is not a credit rating, not investment advice, and not reviewed by a rating committee.
- FR-18: The note reports the run cost in tokens and dollars, the specialists invoked, and the number of retrieval iterations.
- FR-19: The CLI accepts an analyst question as an argument and runs ingestion and analysis as separate commands, so that analysis can run repeatedly against one ingested corpus.

## Non-functional requirements

- NFR-1: A single end-to-end analysis query costs under USD 0.25 in Anthropic API spend, measured from reported token usage and the published per-model rates. Exceeding the ceiling ends the run and reports the overrun rather than continuing.
- NFR-2: Ingestion and retrieval require no cloud infrastructure beyond the Anthropic API. The vector store, embedding model, corpus, and outputs all live on the local filesystem.
- NFR-3: Only publicly published documents are ingested, indexed, or quoted. No confidential, internal, or non-public company data enters the corpus, the index, or any prompt. Every corpus document records the public URL it came from.
- NFR-4: No API key, token, or credential is committed to the repository or written into any output artifact. Credentials are read from the environment.
- NFR-5: The page classifier is testable without network access or API spend, so that classification can be regression-tested offline.
- NFR-6: A single analysis query completes within 5 minutes on the author's machine.
- NFR-7: Every generated note is reproducible in provenance: the trace is sufficient to identify which chunks and pages produced which claim.
- NFR-8: Claude model identifiers are read from a single configuration point, not hardcoded at each call site, so the model can be changed in one edit.

## Constraints

- Runs locally on Windows 11, Python 3.11 or newer, built primarily via Claude Code.
- LLM calls use the Anthropic API through the official `anthropic` SDK. The Anthropic API provides no embeddings endpoint (KNOWLEDGE-740f4fdd), so embeddings are produced by a local sentence-transformers model.
- Corpus is two public documents totalling 401 pages, from two different Adani listed entities: the AGEL FY2024-25 Integrated Annual Report financial-reports extract (367 pages) and the APSEZ Q1 FY27 unaudited results filing (34 pages). One is an annual-report extract and one a quarterly filing, not two annual reports. Neither contains a chart page, and neither carries substantive ESG or climate disclosure.
- Vector store is Chroma or FAISS, local and file-backed.
- Solo project. The author is the only human reviewer and the only approver (DECISION-eeb0fd0d).
- Personal proof-of-work under time pressure: breadth of architecture demonstrated matters more than depth of any single component.

## Non-goals

- Not a credit rating, a rating methodology, or investment advice, and it will not emit a rating symbol or a numeric score resembling one.
- No web or graphical interface in this release. Streamlit is deferred, not planned.
- No cloud deployment, no hosted vector database, no containerisation, no CI pipeline.
- No fine-tuning and no custom model training.
- No ingestion of non-public, paywalled, or scraped-behind-login documents.
- No OCR of scanned image-only PDFs; the corpus is restricted to digitally generated PDFs with an extractable text layer.
- No interpretation of charts or figures. The classifier still identifies chart pages and records them, but no vision model reads them. Withdrawn because the corpus contains no chart page, so the path could be neither exercised nor verified.
- No real-time market, pricing, or news data. The corpus is static documents.
- No multi-user support, authentication, or persistence beyond local files.
- No financial-statement arithmetic validation; the system reports what documents state, it does not recompute or audit them.

## Acceptance criteria

- AC-1: Running ingestion over the two-document corpus produces a page-classification record for every page, and a manual review of a 20-page stratified sample shows at least 85 percent of pages assigned to the correct class.
- AC-2: The vector store persists to a local directory and is queryable in a fresh process without re-ingestion.
- AC-3: Re-running ingestion over the unchanged corpus completes without re-embedding any document, evidenced by the run log.
- AC-4: Each of three pre-written golden analyst questions produces a Markdown note, and each note is reachable from a single CLI command.
- AC-5: For each of the three golden questions, every material claim in the resulting note carries a citation, and manual verification against the source PDFs confirms each cited page actually supports the claim it is attached to, with zero unsupported claims tolerated.
- AC-6: For at least one golden question, the recorded trace shows the retrieval loop ran more than one iteration and revised its query after a reflection verdict — proving the loop is agentic and not single-shot.
- AC-7: The router's recorded reasoning shows different specialist selections across the three golden questions, proving routing is query-dependent rather than fixed.
- AC-8: No single golden-question run exceeds USD 0.25 in reported API spend.
- AC-9: At least one golden question is answerable only from content on a table page, and it is answered correctly — proving the routing layer earns its place. The chart half of this criterion is withdrawn because the corpus contains no chart page.
- AC-10: Every note carries the demonstration-artifact and not-a-rating disclaimer of FR-17.
- AC-11: A secrets scan of the repository at completion finds no API key or credential, and no corpus document lacks a recorded public source URL.
- AC-12: The page classifier has an offline test that runs with no network access and no API spend.
- AC-13: The agentic retrieval loop is compared against single-shot retrieval on the same questions, over the same index, with the same synthesis step, and the comparison is reported with its method, its per-question outcome and its cost. The criterion is that the comparison is run and reported honestly, not that the loop wins: FR-8 asserts an architecture, and an assertion carried through a whole project without measurement is the thing this criterion exists to prevent. If single-shot retrieval matches or beats the loop, that result stands and is reported.
- AC-14: A reader who has never seen the project can install it, ingest the corpus and produce a cited note by following the repository's own written instructions, without reading the source.

## Risks

- **Citation fabrication.** The synthesis step may attach a plausible page number to a claim the page does not support. This is the failure that would most damage the author's credibility with the audience. Mitigation: citations are carried structurally from retrieved chunk metadata rather than generated as free text, and AC-5 requires manual verification against source PDFs with zero tolerance.
- **Analysing the audience's own employer.** A note asserting weaknesses about an Adani entity is being shown to an Adani professional. Mitigation: public sources only, verifiable citations, explicit non-rating disclaimer, and framing the artifact as an architecture demonstration rather than a credit opinion. The author should read every note before showing it.
- **Cost ceiling versus architecture.** A four-specialist multi-agent loop under a USD 0.25 ceiling is tight. Mitigation: at most two specialists per query (FR-13), a cheaper model for routing and reflection, and a hard spend ceiling that ends the run (NFR-1). If the ceiling proves infeasible, the correct response is to raise it deliberately, not to quietly weaken the loop.
- **Retrieval quality from a small local embedding model.** A general-purpose MiniLM-class model may retrieve poorly on dense financial language. Mitigation: the agentic loop's reflect-and-retry compensates for weak single-shot retrieval; a finance-tuned embedding provider is the recorded upgrade path (DECISION-c49d609a).
- **Table extraction fidelity.** Financial tables in annual reports are visually complex, and a mangled table produces confidently wrong numbers. Mitigation: table-aware chunking (FR-5), and AC-9 forces at least one table-only question to be answered correctly.
- **Scanned or image-only pages.** If a sourced PDF lacks a text layer, extraction silently yields nothing. Mitigation: OCR is an explicit non-goal, so document selection must verify an extractable text layer before ingestion.
- **Peer specialist without peers.** Resolved: the corpus holds two distinct issuers, AGEL and APSEZ, so peer comparison has something to compare. It remains an uneven comparison — a full-year annual-report extract against a single quarter's filing — so peer findings must state which period each side is drawn from.
- **ESG specialist without ESG material.** The corpus carries no substantive emissions, capacity or transition disclosure: across 367 AGEL pages the word emission appears once and climate not at all, the frequent ESG hits being page-header navigation (KNOWLEDGE-4ef75831). The specialist will be built and routed but must report insufficient evidence rather than manufacture a finding, and no golden question is ESG-led. Adding the full AGEL Integrated Annual Report would close this and restore chart coverage at the same time.
- **Scope breadth versus time.** Six architectural layers in a personal project invites an unfinished demo. Mitigation: bounded Genesis tasks with executable gates, and a vertical slice that runs end to end before any layer is deepened.

## Open questions

- Resolved: the corpus is the two documents named under Constraints, both recorded in `corpus/manifest.json` with their public source URLs.
- Resolved: the three golden questions are recorded in DECISION-70a5ab26 — an APSEZ covenant-headroom question answerable only from tables, an AGEL auditor-and-disclosure question, and a cross-issuer financing-profile comparison. They satisfy AC-7 by selecting three different specialist sets and AC-9 by including a table-only question.
- Resolved: Chroma, with faiss-cpu as the recorded fallback (DECISION-f21324fe). Proven to install and persist across a process boundary on Windows at T1.
- Resolved as a starting point, not a measurement: Haiku 4.5 for routing, reflection and any per-page model call; Opus 5 for specialist analysis and synthesis (DECISION-39162611). AC-8 measures real per-run spend and the split is retuned there if it misses.
- Resolved: Python 3.12.5 via the `py` launcher, satisfying the >=3.11 constraint (ASSUMPTION-c7b5f9b9).
- Resolved: the APSEZ source URL is now a direct link to the filing PDF rather than the investor-downloads index page, so the corpus is reproducible from the manifest without depending on a page that rolls forward.
- Open: no `ANTHROPIC_API_KEY` is configured yet. Nothing before T5 needs one, but the agentic loop and specialists cannot run without it, and API access is billed separately from a Claude Pro subscription.
