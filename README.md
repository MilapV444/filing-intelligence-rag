# filing-intelligence-rag

A local pipeline that reads public financial filings and produces a cited,
first-pass credit rationale note.

```bash
python -m risk_analytics ingest
python -m risk_analytics ask "As at 30 June 2026, what are APSEZ's Net Gearing and DSCR, and how much covenant headroom remains?"
```

Output is a Markdown note plus a machine-readable trace of every decision that
produced it.

> **This is a demonstration artifact, not a rating tool.**
> It is not a credit rating, not investment advice, and not reviewed by a
> rating committee. Every note it writes says so.

---

## What it actually does

```
PDF ──► classify page ──► extract by class ──► chunk ──► embed ──► Chroma
          text/table/chart    prose | table       tables never     (local)
          from layout stats   structure           split mid-row
                                                       │
question ──► plan sub-queries ──► search ──► reflect ──┘
                    ▲                          │
                    └──── revise and retry ────┘
                                               │
                              router ──► ≤2 of 4 specialists ──► synthesis ──► note
```

Five things are worth knowing because they are where the design has opinions:

**Citations are built from stored metadata, never written by the model.** A
specialist names chunk IDs; the code resolves those IDs to document and page.
A claim citing an ID that was not retrieved is **dropped and counted**, not
printed with a caveat. This is the mechanism the whole thing rests on — a
fabricated page number in front of a credit professional is the failure that
ends the conversation.

**Page classification is free and offline.** Layout statistics from PyMuPDF,
no model call. Measured at **20/20 on a 20-page holdout sample labelled after
the thresholds were frozen** (`tests/labelled_pages.json`), alongside 19/20 on
the tuning split.

**Tables never split mid-row.** A row cut in half puts a figure beside the
wrong label, which is the quiet way to produce a confidently wrong citation.
Large tables split on row boundaries with the header repeated in each piece.

**The cost ceiling prices a call before authorising it.** An early run finished
at $0.43 against a $0.25 ceiling because the check only looked at money already
spent, then authorised one more expensive call. It now projects the cost of the
call it is about to make.

**Retrieval is balanced across documents.** The corpus is lopsided — 1,666
chunks for one issuer against 133 for the other — so a cross-issuer question
retrieved nothing at all from the smaller filing until both retrieval *and* the
evidence trim were balanced per document.

---

## Setup

Requires **Python 3.11+** and an **Anthropic API key with credits**. API access
is billed separately from a Claude Pro/Max subscription.

```bash
git clone https://github.com/MilapV444/filing-intelligence-rag.git
cd filing-intelligence-rag

# Windows line endings will otherwise rewrite every file on checkout and
# invalidate the project's recorded proof hashes.
git config core.autocrlf false

py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,embed]"
```

`[embed]` pulls PyTorch (~2 GB). It is separated because nothing before the
embedding step needs it.

### Credentials

Create `.env` in the project root — it is gitignored and never committed:

```
ANTHROPIC_API_KEY=sk-ant-...
```

An exported `ANTHROPIC_API_KEY` environment variable takes precedence over the
file. On Windows, note that `Set-Content -Encoding utf8` and Notepad both write
a byte-order mark; the parser reads `utf-8-sig` specifically so that a correct
file is not silently unreadable.

### Corpus

Source PDFs are **not committed** — they are public documents, but
redistributing them is not this project's business. `corpus/manifest.json` is
the committed record of what the corpus contains and where each document came
from, so the corpus is reproducible from public URLs.

Download each document listed in `corpus/manifest.json` from its `source_url`
into `sample_pdf/`, keeping the exact `filename` from the manifest. Then:

```bash
.venv/Scripts/python.exe -m risk_analytics ingest
```

Roughly two minutes for 401 pages. The embedding model (~90 MB) downloads once
on first use; after that, embedding runs offline.

---

## Usage

```bash
python -m risk_analytics ask "your question"

  --tables-only      restrict retrieval to table pages
  --doc DOC_ID       restrict to one manifest document (repeatable);
                     omitted, the issuer is inferred from the question
  --k N              chunks retrieved per sub-query (default 8)
  --ceiling USD      spend ceiling for this run (default 0.25)
```

Writes `notes/note-<timestamp>.md` and `notes/note-<timestamp>.trace.json`.
The trace holds the plan, every sub-query, every chunk retrieved, each
reflection verdict, the routing reasons, and per-call token cost.

Typical run: **$0.11–$0.21, 1–4 retrieval iterations, under two minutes.**

---

## Limitations

Stated plainly, because a demo that hides these is worth less than one that
doesn't.

- **The corpus is two documents and it is thin.** One annual-report financials
  extract (367pp) and one quarterly filing (34pp). Not two comparable annual
  reports.
- **It cannot answer every question asked of it.** Of three benchmark
  questions, one is answered fully; two correctly report insufficient evidence
  because the material genuinely is not in these filings. That is the intended
  failure mode, not a masked one — but it means the system refuses more often
  than it answers on this corpus.
- **The ESG specialist has almost nothing to read.** Across 367 AGEL pages,
  `emission` appears once and `climate` not at all. It is built and routed, and
  it correctly reports insufficient evidence.
- **Chart interpretation was cut.** Neither document contains a single chart
  page, so the vision path could be neither exercised nor verified. Chart pages
  are classified and skipped.
- **No evaluation of answer quality.** The 242 tests cover plumbing, citation
  integrity and cost — not whether the analysis is any good. That judgement is
  currently a human reading the note beside the PDF.
- **The contradiction detector is a lexical heuristic**, not a finding. It is
  labelled as unverified in the note because it cannot distinguish a genuine
  conflict from two correct statements about different measures.
- **Windows-developed.** Paths and the venv layout assume Windows; nothing is
  deliberately platform-specific, but it has not been run elsewhere.

---

## Development

```bash
.venv/Scripts/python.exe -m pytest tests/ -q     # 242 tests, ~100s, no API spend
```

Every test injects a scripted model client, so the suite costs nothing and is
deterministic. That is also its blind spot: the first live call ever made failed
because the SDK never received the key from `.env`, while 179 tests passed.
`tests/test_credentials.py` exists specifically to exercise the real client
construction path.

Acceptance assertions in `tests/test_acceptance.py` run against recorded traces
of real runs in `tests/acceptance/`, not fresh live calls — a gate that spends
money on every execution is a bad gate. It proves those runs happened and what
they produced, not that current source reproduces them.

The project was built specification-first under the
[Genesis](https://github.com/ayush488-glitch/genesis-kit) harness. `SPEC.md`
holds the requirements; `.genesis/` holds task state, recorded decisions and
gate evidence. `genesis brief .` restores full context.

---

## Layout

```
src/risk_analytics/
  classify.py    page classification from layout statistics
  extract.py     class-routed extraction
  indexing.py    chunking, local embedding, Chroma store
  retrieval.py   the plan/search/reflect/iterate loop
  agents.py      router and four specialists
  note.py        synthesis and Markdown rendering
  llm.py         Anthropic client with cost accounting
  cli.py         ingest / ask
```
