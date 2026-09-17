"""The corpus manifest: what is in the corpus and where each document came from.

FR-1 requires title, issuer, document type, publication date and public source URL
per document. NFR-3 makes the source URL load-bearing rather than decorative: a
document with no verifiable public origin must not be ingestible at all, so a
manifest that omits or malforms one fails to load rather than loading with a gap.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from . import config

REQUIRED_FIELDS = ("doc_id", "title", "issuer", "doc_type", "published", "source_url", "filename")
PUBLIC_URL_SCHEMES = ("http://", "https://")


class ManifestError(ValueError):
    """The manifest is missing, malformed, or describes a document we must not ingest."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    issuer: str
    doc_type: str
    published: date
    source_url: str
    filename: str

    def path(self, documents_dir: Path | None = None) -> Path:
        return (documents_dir or config.DOCUMENTS_DIR) / self.filename


def _parse_entry(raw: object, position: int) -> Document:
    where = f"document at position {position}"
    if not isinstance(raw, dict):
        raise ManifestError(f"{where} is not an object")

    missing = [f for f in REQUIRED_FIELDS if not str(raw.get(f, "")).strip()]
    if missing:
        raise ManifestError(f"{where} is missing required field(s): {', '.join(missing)}")

    url = str(raw["source_url"]).strip()
    if not url.startswith(PUBLIC_URL_SCHEMES):
        # NFR-3. A local path or a bare filename here would mean a document whose
        # public origin nobody can check, which is exactly what must not be indexed.
        raise ManifestError(
            f"{where} has source_url {url!r}, which is not a public http(s) URL. "
            "Every corpus document must record the public page it was downloaded from."
        )

    try:
        published = date.fromisoformat(str(raw["published"]).strip())
    except ValueError:
        raise ManifestError(
            f"{where} has published {raw['published']!r}; expected an ISO date like 2025-03-31"
        ) from None

    return Document(
        doc_id=str(raw["doc_id"]).strip(),
        title=str(raw["title"]).strip(),
        issuer=str(raw["issuer"]).strip(),
        doc_type=str(raw["doc_type"]).strip(),
        published=published,
        source_url=url,
        filename=str(raw["filename"]).strip(),
    )


def parse(raw: object) -> list[Document]:
    """Validate already-decoded manifest data. Separate from `load` so the rules
    are testable without touching the filesystem (NFR-5 applies the same idea to
    the classifier)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("documents"), list):
        raise ManifestError("manifest must be an object with a 'documents' array")

    documents = [_parse_entry(entry, i) for i, entry in enumerate(raw["documents"])]

    seen: set[str] = set()
    for doc in documents:
        if doc.doc_id in seen:
            raise ManifestError(f"duplicate doc_id {doc.doc_id!r}; citations would be ambiguous")
        seen.add(doc.doc_id)
    return documents


def load(path: Path | None = None) -> list[Document]:
    """Read and validate the corpus manifest."""
    path = path or config.MANIFEST_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(f"no corpus manifest at {path}") from None
    except json.JSONDecodeError as exc:
        raise ManifestError(f"corpus manifest at {path} is not valid JSON: {exc}") from None
    return parse(raw)


def by_id(documents: list[Document] | None = None) -> dict[str, Document]:
    """Documents keyed by doc_id, which is what citations resolve against.

    Everything that opens a corpus PDF goes through this rather than naming a
    file: a path spelled out in code is a document with no recorded public
    origin, which is the thing NFR-3 exists to prevent.
    """
    return {d.doc_id: d for d in (documents if documents is not None else load())}


# "Adani Ports and Special Economic Zone" contributes "and" as a token unique to
# that issuer, so any question containing the word "and" matched both documents
# and scoping silently did nothing. Four characters keeps every real identifier
# (agel, fy25, zone, ports) and drops the conjunctions.
MIN_TOKEN_LENGTH = 4


def _tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= MIN_TOKEN_LENGTH
    }


def distinguishing_tokens(documents: list[Document] | None = None) -> dict[str, set[str]]:
    """Words that identify one corpus document and not the others.

    Tokens shared by every document are dropped, which removes "adani" and
    "limited" automatically rather than by a hand-maintained stopword list: a
    word common to the whole corpus cannot disambiguate within it.
    """
    documents = documents if documents is not None else load()
    per_doc = {
        d.doc_id: _tokens(f"{d.doc_id} {d.issuer} {d.doc_type}") for d in documents
    }
    if len(per_doc) < 2:
        return per_doc
    shared = set.intersection(*per_doc.values())
    return {doc_id: tokens - shared for doc_id, tokens in per_doc.items()}


def match_documents(question: str, documents: list[Document] | None = None) -> list[str]:
    """Which corpus documents a question is about, or all of them if unclear.

    A question naming one issuer retrieved from both by default, and chunks from
    the other issuer crowded the answer out of the top-ranked evidence: 38 of 50
    chunks in the first live run belonged to the document the question was not
    about. Scoping is a retrieval-quality fix, not an optimisation.
    """
    asked = _tokens(question)
    matched = [
        doc_id for doc_id, tokens in distinguishing_tokens(documents).items() if tokens & asked
    ]
    # No match, or every document matched: the question is not issuer-specific,
    # so do not narrow it and risk hiding the answer.
    return matched if 0 < len(matched) < len(distinguishing_tokens(documents)) else []


def missing_files(documents: list[Document], documents_dir: Path | None = None) -> list[str]:
    """Filenames listed in the manifest that are not on disk. Reported rather than
    raised: a manifest can legitimately be written before the PDFs are fetched."""
    return [d.filename for d in documents if not d.path(documents_dir).is_file()]
