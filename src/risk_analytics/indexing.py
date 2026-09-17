"""Chunking, local embedding and the persistent vector store (FR-5, FR-6, FR-7).

Three rules shape this module:

* A table is never split mid-row (FR-5). When a table is too large for one
  chunk it is split on row boundaries and the header row is repeated, so every
  chunk still says what its columns mean. A row cut in half puts a number next
  to the wrong label, which is the quiet way to produce a confidently wrong
  citation.
* Embeddings are computed locally (FR-6, NFR-2). Nothing leaves the machine at
  index time and no API key is required.
* Re-ingesting an unchanged document does no work (FR-7, AC-3). Documents are
  fingerprinted by the bytes of the PDF, not by mtime, which changes when a file
  is merely re-downloaded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import classify, config, extract, manifest

PROSE, TABLE = "prose", "table"

MAX_CHUNK_CHARS = 1200  # comfortably inside the embedding model's window
CHUNK_OVERLAP_CHARS = 150  # so a fact split across a boundary survives in one piece
MAX_TABLE_ROWS_PER_CHUNK = 30
MIN_CHUNK_CHARS = 20  # below this a chunk is page furniture, not content

COLLECTION = "corpus"
STATE_FILENAME = "ingested.json"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    page_number: int
    page_class: str
    kind: str
    text: str

    def metadata(self) -> dict:
        """Travels with the vector. FR-16 citations are built from these fields
        rather than from anything a model writes, which is what makes a citation
        checkable instead of plausible."""
        return {
            "doc_id": self.doc_id,
            "page_number": self.page_number,
            "page_class": self.page_class,
            "kind": self.kind,
        }


# --- Chunking (pure; no model, no store) -------------------------------------


def _split_prose(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > MAX_CHUNK_CHARS:
            # A single paragraph longer than the window: break on the last
            # sentence end inside it, falling back to a hard cut.
            window = para[:MAX_CHUNK_CHARS]
            cut = max(window.rfind(". "), window.rfind(".\n"))
            cut = cut + 1 if cut > MAX_CHUNK_CHARS // 2 else MAX_CHUNK_CHARS
            chunks.append(para[:cut].strip())
            para = para[max(0, cut - CHUNK_OVERLAP_CHARS):].strip()
        if len(current) + len(para) + 2 > MAX_CHUNK_CHARS and current:
            chunks.append(current.strip())
            current = current[-CHUNK_OVERLAP_CHARS:] if CHUNK_OVERLAP_CHARS else ""
        current = f"{current}\n\n{para}".strip()
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def _split_table(table: extract.Table) -> list[str]:
    """Split on row boundaries only, repeating the header (FR-5)."""
    rows = table.rows
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    if not body:
        return [extract.Table(rows=[header]).to_text()]

    pieces = []
    for start in range(0, len(body), MAX_TABLE_ROWS_PER_CHUNK):
        window = body[start : start + MAX_TABLE_ROWS_PER_CHUNK]
        pieces.append(extract.Table(rows=[header, *window]).to_text())
    return pieces


def chunk_page(page: extract.ExtractedPage) -> list[Chunk]:
    """Chunk one extracted page. Tables first, then the page's prose."""
    chunks: list[Chunk] = []

    def add(kind: str, text: str) -> None:
        if len(text.strip()) < MIN_CHUNK_CHARS:
            return
        ordinal = sum(1 for c in chunks if c.kind == kind)
        chunks.append(
            Chunk(
                chunk_id=f"{page.doc_id}:p{page.page_number}:{kind}{ordinal}",
                doc_id=page.doc_id,
                page_number=page.page_number,
                page_class=page.page_class,
                kind=kind,
                text=text.strip(),
            )
        )

    for table in page.tables:
        for piece in _split_table(table):
            add(TABLE, piece)

    # A chart page carries no text and no tables, so this yields nothing.
    if page.method != extract.SKIPPED:
        for piece in _split_prose(page.text):
            add(PROSE, piece)

    return chunks


def chunk_document(pages: list[extract.ExtractedPage]) -> list[Chunk]:
    return [chunk for page in pages for chunk in chunk_page(page)]


# --- Local embedding ----------------------------------------------------------

_model = None


def embedder():
    """Load the sentence-transformers model once, lazily.

    Importing it is expensive and pulls torch, so nothing that does not embed
    should pay for it -- classification and extraction never touch this.

    Loads from the local cache first. By default the library contacts the model
    hub on every load to check for updates, which is a network call NFR-2 does
    not want and which cost about thirty seconds per load behind a TLS-inspecting
    proxy: the check failed, retried five times with backoff, then fell back to
    the cache anyway. Trying the cache first makes the offline path the normal
    one and leaves the download for a genuinely cold machine.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        try:
            _model = SentenceTransformer(config.EMBEDDING_MODEL, local_files_only=True)
        except Exception:
            # Not cached yet: this is the one download, on first use only.
            _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = embedder().encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [list(map(float, v)) for v in vectors]


# --- Persistent store ---------------------------------------------------------


def client(index_dir: Path | None = None):
    import chromadb

    path = index_dir or config.INDEX_DIR
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path), settings=chromadb.config.Settings(anonymized_telemetry=False)
    )


def collection(index_dir: Path | None = None):
    return client(index_dir).get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def _state_path(index_dir: Path | None = None) -> Path:
    return (index_dir or config.INDEX_DIR) / STATE_FILENAME


def _load_state(index_dir: Path | None = None) -> dict:
    try:
        return json.loads(_state_path(index_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: dict, index_dir: Path | None = None) -> None:
    path = _state_path(index_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def content_hash(path: Path) -> str:
    """Fingerprint a document by its bytes. mtime changes on a re-download of an
    identical file and would force a pointless re-embed of 367 pages."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class IngestReport:
    ingested: list[str]
    skipped: list[str]
    chunks_added: int

    def to_dict(self) -> dict:
        return asdict(self)


def ingest(
    doc_ids: list[str] | None = None,
    force: bool = False,
    index_dir: Path | None = None,
) -> IngestReport:
    """Index the registered corpus, skipping documents that have not changed."""
    documents = manifest.by_id()
    targets = doc_ids or list(documents)
    store = collection(index_dir)
    state = _load_state(index_dir)

    ingested, skipped, added = [], [], 0
    for doc_id in targets:
        document = documents[doc_id]
        digest = content_hash(document.path())
        if not force and state.get(doc_id, {}).get("content_hash") == digest:
            skipped.append(doc_id)
            continue

        chunks = chunk_document(extract.extract_document(doc_id))
        # Replace this document's chunks wholesale so a shrunken document does
        # not leave orphans behind that would still be cited.
        existing = store.get(where={"doc_id": doc_id}, include=[])
        if existing["ids"]:
            store.delete(ids=existing["ids"])

        if chunks:
            store.add(
                ids=[c.chunk_id for c in chunks],
                embeddings=embed([c.text for c in chunks]),
                documents=[c.text for c in chunks],
                metadatas=[c.metadata() for c in chunks],
            )
        state[doc_id] = {
            "content_hash": digest,
            "chunks": len(chunks),
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        ingested.append(doc_id)
        added += len(chunks)

    _save_state(state, index_dir)
    return IngestReport(ingested=ingested, skipped=skipped, chunks_added=added)


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    page_number: int
    page_class: str
    kind: str
    text: str
    distance: float

    def citation(self) -> str:
        """The citation a note carries, built from stored metadata rather than
        from model output (FR-16)."""
        return f"[{self.doc_id} p.{self.page_number}]"


def search(
    query: str,
    k: int = 8,
    where: dict | None = None,
    index_dir: Path | None = None,
    doc_ids: list[str] | None = None,
) -> list[Hit]:
    """Search the store, optionally giving each document its own share of `k`.

    The corpus is lopsided: 1666 chunks for one issuer against 133 for the
    other. A single ranked query for a question spanning both returned nothing
    at all from the smaller document, and the specialists correctly reported
    that no comparison could be made. Retrieving per document and merging gives
    the smaller filing a floor rather than leaving it to out-rank twelve times
    its own volume.
    """
    if doc_ids and len(doc_ids) > 1:
        per_doc = max(2, k // len(doc_ids))
        merged: list[Hit] = []
        for doc_id in doc_ids:
            clause = {"doc_id": doc_id}
            scoped = {"$and": [where, clause]} if where else clause
            merged.extend(search(query, per_doc, scoped, index_dir))
        return sorted(merged, key=lambda h: h.distance)

    store = collection(index_dir)
    result = store.query(
        query_embeddings=embed([query]),
        n_results=k,
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append(
            Hit(
                chunk_id=chunk_id,
                doc_id=meta["doc_id"],
                page_number=int(meta["page_number"]),
                page_class=meta["page_class"],
                kind=meta["kind"],
                text=text,
                distance=float(distance),
            )
        )
    return hits
