"""Corpus registration (FR-1, NFR-3, AC-11).

The manifest is the only sanctioned way to reach a corpus document. A path
written into code is a document whose public origin nobody recorded, and that is
precisely what NFR-3 forbids -- so these tests check the registration itself,
not just that the loader parses.
"""

import re
import subprocess
from pathlib import Path

import pytest

from risk_analytics import manifest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DOC_IDS = {"agel-fy25", "apsez-q1fy27"}


@pytest.fixture(scope="module")
def documents():
    return manifest.load()


def test_both_corpus_documents_are_registered(documents):
    assert {d.doc_id for d in documents} == EXPECTED_DOC_IDS


def test_every_document_records_a_public_https_source(documents):
    """NFR-3 and AC-11. Without this the corpus is not reproducible from public
    sources and the provenance claim in every generated note is hollow."""
    for doc in documents:
        assert doc.source_url.startswith("https://"), f"{doc.doc_id}: {doc.source_url}"
        assert "adani" in doc.source_url, (
            f"{doc.doc_id} cites {doc.source_url}, which is not an issuer-controlled "
            "domain; provenance must point at the issuer, not a mirror"
        )


def test_the_corpus_spans_two_distinct_issuers(documents):
    """The peer specialist compares issuers. Two documents from one issuer would
    leave it nothing to compare (ASSUMPTION-762e7a43)."""
    assert len({d.issuer for d in documents}) == 2


def test_every_registered_document_is_present_on_disk(documents):
    missing = manifest.missing_files(documents)
    assert not missing, (
        f"registered but absent: {missing}. Fetch them from the recorded "
        "source_url into the documents directory."
    )


def test_documents_resolve_by_id_not_by_path(documents):
    """Pipeline code looks documents up by doc_id; nothing should be spelling out
    a filename."""
    index = manifest.by_id(documents)
    assert set(index) == EXPECTED_DOC_IDS
    for doc_id, doc in index.items():
        assert doc.path().is_file()
        assert doc.doc_id == doc_id


def test_registered_pdfs_have_an_extractable_text_layer(documents):
    """OCR is an explicit non-goal, so a scanned image-only PDF would extract
    silently empty and produce a confidently blank analysis."""
    pymupdf = pytest.importorskip("pymupdf")
    for doc in documents:
        with pymupdf.open(doc.path()) as pdf:
            sampled = [pdf[i].get_text("text").strip() for i in range(min(5, pdf.page_count))]
        assert sum(len(t) for t in sampled) > 500, (
            f"{doc.doc_id} yields almost no text in its first pages; it may be "
            "scanned, which this project does not handle"
        )


# --- AC-11: nothing secret, nothing unsourced, nothing redistributed ---------


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return [line for line in out.stdout.splitlines() if line.strip()]


def test_corpus_pdfs_are_never_committed():
    """They are public documents, but redistributing them through this
    repository is not this project's business. The manifest makes the corpus
    reproducible from the recorded URLs instead."""
    committable = [f for f in _tracked_files() if f.lower().endswith(".pdf")]
    assert not committable, f"PDFs would be committed: {committable}"


def test_no_credentials_anywhere_in_the_repository():
    """AC-11. The key prefix is assembled at runtime so this file does not itself
    contain the literal it searches for."""
    key_prefix = "sk-" + "ant-"
    patterns = [
        re.compile(re.escape(key_prefix) + r"[A-Za-z0-9_\-]{12,}"),
        re.compile(r"ANTHROPIC_API_KEY\s*[=:]\s*['\"][^'\"$%{]{8,}"),
    ]
    offenders = []
    for name in _tracked_files():
        path = ROOT / name
        if not path.is_file() or path.suffix.lower() in {".pdf", ".png", ".jpg"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in patterns:
            if pattern.search(text):
                offenders.append(f"{name} matches {pattern.pattern}")
    assert not offenders, "possible credential material: " + "; ".join(offenders)


def test_env_file_would_not_be_committed():
    assert ".env" in (ROOT / ".gitignore").read_text(encoding="utf-8")
