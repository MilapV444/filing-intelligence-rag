"""Chunking, embedding and persistence (FR-5, FR-6, FR-7, NFR-2, AC-2, AC-3)."""

import json
import subprocess
import sys

import pytest

from risk_analytics import config, extract, indexing

# --- Chunking: pure, no model, no store --------------------------------------


def table_of(rows: int, cols: int = 4) -> extract.Table:
    header = [f"col{c}" for c in range(cols)]
    body = [[f"r{r}c{c}" for c in range(cols)] for r in range(rows)]
    return extract.Table(rows=[header, *body])


def page_with(tables=(), text="", page_class="table", method=None) -> extract.ExtractedPage:
    return extract.ExtractedPage(
        doc_id="doc",
        page_number=7,
        page_class=page_class,
        method=method or (extract.TABLE_STRUCTURE if tables else extract.PROSE),
        text=text,
        tables=list(tables),
    )


def test_a_small_table_is_one_chunk():
    chunks = indexing.chunk_page(page_with(tables=[table_of(5)]))
    assert len(chunks) == 1
    assert chunks[0].kind == indexing.TABLE


def test_a_large_table_is_split_only_on_row_boundaries():
    """FR-5. A row cut in half puts a figure beside the wrong label, which is how
    a citation ends up plausible and wrong."""
    rows = 95
    chunks = indexing.chunk_page(page_with(tables=[table_of(rows)]))
    assert len(chunks) > 1

    seen = []
    for chunk in chunks:
        lines = chunk.text.split("\n")
        assert lines[0] == "col0 | col1 | col2 | col3", "header not repeated in every chunk"
        for line in lines[1:]:
            assert line.count("|") == 3, f"row lost cells: {line!r}"
            seen.append(line)

    assert len(seen) == rows, "rows were dropped or duplicated across the split"
    assert len(set(seen)) == rows


def test_every_table_chunk_repeats_the_header():
    """Without the header a chunk of figures has no column meaning, and a model
    reading it in isolation has to guess which period a number belongs to."""
    chunks = indexing.chunk_page(page_with(tables=[table_of(80)]))
    assert all(c.text.startswith("col0 |") for c in chunks)


def test_prose_is_chunked_within_the_window():
    text = "\n\n".join(f"Paragraph {i}. " + ("filler words " * 30) for i in range(12))
    chunks = indexing.chunk_page(page_with(text=text, page_class="text"))
    assert len(chunks) > 1
    assert all(len(c.text) <= indexing.MAX_CHUNK_CHARS * 1.2 for c in chunks)
    assert all(c.kind == indexing.PROSE for c in chunks)


def test_a_single_overlong_paragraph_is_still_split():
    chunks = indexing.chunk_page(page_with(text="word " * 2000, page_class="text"))
    assert len(chunks) > 1
    assert all(len(c.text) <= indexing.MAX_CHUNK_CHARS * 1.2 for c in chunks)


def test_chunks_carry_the_provenance_a_citation_needs():
    """FR-16 citations are built from this metadata, never from model output."""
    chunks = indexing.chunk_page(page_with(tables=[table_of(3)], text="Some prose here."))
    assert chunks
    for chunk in chunks:
        assert chunk.doc_id == "doc"
        assert chunk.page_number == 7
        assert chunk.page_class == "table"
        meta = chunk.metadata()
        assert meta["doc_id"] == "doc" and meta["page_number"] == 7


def test_chunk_ids_are_unique_and_deterministic():
    page = page_with(tables=[table_of(50)], text="prose " * 400)
    first = indexing.chunk_page(page)
    second = indexing.chunk_page(page)
    ids = [c.chunk_id for c in first]
    assert len(ids) == len(set(ids)), "duplicate ids would overwrite each other in the store"
    assert ids == [c.chunk_id for c in second], "ids must be stable across runs"


def test_a_skipped_chart_page_produces_no_chunks():
    page = extract.ExtractedPage(
        doc_id="doc", page_number=3, page_class="chart",
        method=extract.SKIPPED, text="", tables=[], note="skipped",
    )
    assert indexing.chunk_page(page) == []


def test_page_furniture_is_not_indexed():
    assert indexing.chunk_page(page_with(text="17", page_class="text")) == []


# --- Embedding and the persistent store --------------------------------------

@pytest.fixture(scope="session")
def embed_ready():
    """Fail rather than skip when the embedding dependency is absent.

    T5A is the embedding task, so a skipped embedding test is not proof of
    anything -- it would let the gate pass green over work that never ran.
    """
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pytest.fail(
            "sentence-transformers is not installed, so FR-6 cannot be verified. "
            "Install it with: .venv/Scripts/python.exe -m pip install -e \".[embed]\""
        )
    return True


@pytest.fixture(scope="module")
def indexed(tmp_path_factory, embed_ready):
    """Index the smaller document once for the whole module."""
    index_dir = tmp_path_factory.mktemp("index")
    report = indexing.ingest(["apsez-q1fy27"], index_dir=index_dir)
    return index_dir, report


def test_embeddings_are_the_configured_width(embed_ready):
    vectors = indexing.embed(["net debt to EBITDA", "scope 1 emissions"])
    assert len(vectors) == 2
    assert all(len(v) == config.EMBEDDING_DIM for v in vectors)


def test_embedding_needs_no_api_key(embed_ready, monkeypatch):
    """FR-6, NFR-2. The Anthropic API has no embeddings endpoint and this path
    must never acquire a dependency on one."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert indexing.embed(["a short sentence"])


def test_embedding_works_with_the_model_hub_offline(embed_ready, monkeypatch):
    """The model is downloaded once and cached. Embedding itself must not reach
    the network, or ingestion is not the offline step the design claims."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert indexing.embed(["liquidity and debt maturity profile"])


def test_ingest_populates_the_store(indexed):
    index_dir, report = indexed
    assert report.ingested == ["apsez-q1fy27"]
    assert report.chunks_added > 0
    assert indexing.collection(index_dir).count() == report.chunks_added


def test_reingesting_an_unchanged_document_does_no_work(indexed):
    """AC-3. Re-embedding 401 unchanged pages on every run would make iteration
    unusable."""
    index_dir, first = indexed
    again = indexing.ingest(["apsez-q1fy27"], index_dir=index_dir)
    assert again.skipped == ["apsez-q1fy27"]
    assert again.ingested == []
    assert again.chunks_added == 0
    assert indexing.collection(index_dir).count() == first.chunks_added


def test_force_reingests_without_duplicating_chunks(indexed):
    index_dir, first = indexed
    forced = indexing.ingest(["apsez-q1fy27"], force=True, index_dir=index_dir)
    assert forced.ingested == ["apsez-q1fy27"]
    assert indexing.collection(index_dir).count() == first.chunks_added, (
        "a forced re-ingest duplicated chunks instead of replacing them"
    )


def test_ingest_state_records_a_content_hash(indexed):
    index_dir, _ = indexed
    state = json.loads((index_dir / indexing.STATE_FILENAME).read_text(encoding="utf-8"))
    entry = state["apsez-q1fy27"]
    assert len(entry["content_hash"]) == 64
    assert entry["chunks"] > 0


def test_a_changed_document_is_re_embedded(indexed, tmp_path):
    """The fingerprint is over file bytes, so an edited document re-indexes even
    though its name and path never changed."""
    index_dir, _ = indexed
    state_path = index_dir / indexing.STATE_FILENAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["apsez-q1fy27"]["content_hash"] = "0" * 64
    state_path.write_text(json.dumps(state), encoding="utf-8")

    again = indexing.ingest(["apsez-q1fy27"], index_dir=index_dir)
    assert again.ingested == ["apsez-q1fy27"]


def test_search_returns_hits_with_a_checkable_citation(indexed):
    index_dir, _ = indexed
    hits = indexing.search("net gearing ratio and DSCR covenant", k=5, index_dir=index_dir)
    assert hits
    for hit in hits:
        assert hit.doc_id == "apsez-q1fy27"
        assert hit.page_number >= 1
        assert hit.citation() == f"[{hit.doc_id} p.{hit.page_number}]"
        assert hit.text.strip()


def test_search_can_be_restricted_to_table_content(indexed):
    """Golden question 1 is answerable only from tables; the retrieval loop needs
    to be able to say so."""
    index_dir, _ = indexed
    hits = indexing.search(
        "net gearing", k=5, where={"kind": indexing.TABLE}, index_dir=index_dir
    )
    assert hits
    assert all(h.kind == indexing.TABLE for h in hits)


def test_the_covenant_figures_are_retrievable(indexed):
    """End to end over the real filing: the numbers golden question 1 asks for
    must come back, attached to the page a reader can open."""
    index_dir, _ = indexed
    hits = indexing.search(
        "Net Gearing ratio total net debt tangible net worth DSCR",
        k=10, where={"kind": indexing.TABLE}, index_dir=index_dir,
    )
    blob = "\n".join(h.text for h in hits)
    assert "0.55" in blob, "the Net Gearing value was not retrievable"
    assert "5.42" in blob, "the DSCR value was not retrievable"


def test_a_fresh_process_can_query_without_reingesting(indexed):
    """AC-2. Persistence proven across a process boundary, not from a warm cache."""
    index_dir, report = indexed
    reader = (
        "import sys, json;"
        "sys.path.insert(0, 'src');"
        "from risk_analytics import indexing;"
        "from pathlib import Path;"
        f"c = indexing.collection(Path({str(index_dir)!r}));"
        "print(json.dumps({'count': c.count()}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", reader],
        capture_output=True, text=True, timeout=600, cwd=str(config.ROOT),
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["count"] == report.chunks_added


def test_model_loads_from_cache_without_contacting_the_hub(embed_ready, monkeypatch):
    """Regression: the loader used to check the hub on every load. Behind a
    TLS-inspecting proxy that check failed and retried five times with backoff,
    costing roughly thirty seconds per load before falling back to the cache.
    Forcing a reload here with no network allowed proves the cache path is the
    one taken."""
    monkeypatch.setattr(indexing, "_model", None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    vectors = indexing.embed(["total debt and cash equivalents"])
    assert len(vectors[0]) == config.EMBEDDING_DIM
