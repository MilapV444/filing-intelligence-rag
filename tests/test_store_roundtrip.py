"""Proves the chosen vector store actually installs and persists on this machine.

DECISION-f21324fe picked Chroma over FAISS because it persists vectors and
per-chunk metadata together. That decision is only sound if Chroma installs
cleanly on Windows and survives a process boundary, so this test exists to fail
loudly here at T-1 rather than at T-4 where the real indexing is built.

Embeddings are supplied explicitly, so nothing downloads an embedding model and
the test runs offline (NFR-2, NFR-5).
"""

import json
import subprocess
import sys
import textwrap

import pytest

chromadb = pytest.importorskip("chromadb", reason="chromadb is a declared project dependency")

COLLECTION = "roundtrip"

# Deliberately trivial 4-dimensional vectors: the point is persistence and
# metadata fidelity, not retrieval quality.
CHUNKS = [
    {"id": "c1", "text": "net debt to EBITDA", "vec": [1.0, 0.0, 0.0, 0.0],
     "meta": {"doc_id": "issuer-a-ar-fy25", "page": 42, "page_class": "table"}},
    {"id": "c2", "text": "scope 1 emissions", "vec": [0.0, 1.0, 0.0, 0.0],
     "meta": {"doc_id": "issuer-a-ar-fy25", "page": 118, "page_class": "chart"}},
    {"id": "c3", "text": "related party transactions", "vec": [0.0, 0.0, 1.0, 0.0],
     "meta": {"doc_id": "issuer-b-ar-fy25", "page": 7, "page_class": "text"}},
]


def _client(path):
    return chromadb.PersistentClient(
        path=str(path),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )


def test_chroma_persists_vectors_and_metadata_across_a_process_boundary(tmp_path):
    store = tmp_path / "index"

    client = _client(store)
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c["id"] for c in CHUNKS],
        embeddings=[c["vec"] for c in CHUNKS],
        documents=[c["text"] for c in CHUNKS],
        metadatas=[c["meta"] for c in CHUNKS],
    )
    assert collection.count() == len(CHUNKS)
    del collection, client

    # A separate interpreter, so this cannot pass on an in-memory cache.
    result_path = tmp_path / "result.json"
    reader = textwrap.dedent(
        f"""
        import json
        import chromadb
        client = chromadb.PersistentClient(
            path={str(store)!r},
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection({COLLECTION!r})
        hit = collection.query(query_embeddings=[[0.0, 1.0, 0.0, 0.0]], n_results=1)
        json.dump(
            {{
                "count": collection.count(),
                "id": hit["ids"][0][0],
                "document": hit["documents"][0][0],
                "metadata": hit["metadatas"][0][0],
            }},
            open({str(result_path)!r}, "w", encoding="utf-8"),
        )
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", reader], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, f"reader process failed:\n{proc.stdout}\n{proc.stderr}"

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["count"] == len(CHUNKS), "persisted collection lost rows across processes"
    assert result["id"] == "c2", "nearest neighbour was not returned"
    assert result["document"] == "scope 1 emissions"

    # FR-5 depends on this metadata surviving: without doc_id and page, a
    # retrieved chunk cannot produce a verifiable citation.
    assert result["metadata"]["doc_id"] == "issuer-a-ar-fy25"
    assert result["metadata"]["page"] == 118
    assert result["metadata"]["page_class"] == "chart"


def test_metadata_filtering_is_available(tmp_path):
    """T-4 needs to scope retrieval to one document or one page class. Chroma was
    chosen partly for this; confirm it works rather than assuming it."""
    client = _client(tmp_path / "index")
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c["id"] for c in CHUNKS],
        embeddings=[c["vec"] for c in CHUNKS],
        documents=[c["text"] for c in CHUNKS],
        metadatas=[c["meta"] for c in CHUNKS],
    )

    hit = collection.query(
        query_embeddings=[[1.0, 0.0, 0.0, 0.0]],
        n_results=1,
        where={"doc_id": "issuer-b-ar-fy25"},
    )
    assert hit["ids"][0] == ["c3"], "metadata filter did not constrain the result"
