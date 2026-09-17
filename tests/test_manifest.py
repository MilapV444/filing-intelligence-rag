"""Checks on the corpus manifest, the gate that enforces public-sources-only (NFR-3)."""

import json

import pytest

from risk_analytics import manifest

VALID = {
    "doc_id": "example-ar-fy25",
    "title": "Annual Report 2024-25",
    "issuer": "Example Issuer Limited",
    "doc_type": "annual_report",
    "published": "2025-06-30",
    "source_url": "https://www.example.com/investors/ar-fy25.pdf",
    "filename": "example-ar-fy25.pdf",
}


def wrap(*entries):
    return {"documents": list(entries)}


def test_valid_entry_parses_into_a_document():
    (doc,) = manifest.parse(wrap(VALID))
    assert doc.doc_id == "example-ar-fy25"
    assert doc.issuer == "Example Issuer Limited"
    assert doc.published.year == 2025
    assert doc.source_url.startswith("https://")


def test_empty_manifest_is_valid():
    """The manifest is committed before the PDFs are chosen."""
    assert manifest.parse(wrap()) == []


@pytest.mark.parametrize("field", manifest.REQUIRED_FIELDS)
def test_every_required_field_is_required(field):
    entry = {k: v for k, v in VALID.items() if k != field}
    with pytest.raises(manifest.ManifestError, match=field):
        manifest.parse(wrap(entry))


@pytest.mark.parametrize("field", manifest.REQUIRED_FIELDS)
def test_blank_field_is_rejected_like_a_missing_one(field):
    with pytest.raises(manifest.ManifestError, match=field):
        manifest.parse(wrap({**VALID, field: "   "}))


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/Users/me/Downloads/report.pdf",  # a local copy, origin unverifiable
        "report.pdf",  # bare filename
        "/investors/ar-fy25.pdf",  # site-relative, no host
        "ftp://example.com/ar.pdf",  # not a public web page
    ],
)
def test_non_public_source_url_is_rejected(url):
    """NFR-3. A document whose public origin cannot be checked must not be
    ingestible at all, so this fails at load rather than warning."""
    with pytest.raises(manifest.ManifestError, match="public"):
        manifest.parse(wrap({**VALID, "source_url": url}))


@pytest.mark.parametrize("published", ["30-06-2025", "June 2025", "2025-13-01", "2025"])
def test_malformed_publication_date_is_rejected(published):
    with pytest.raises(manifest.ManifestError, match="ISO date"):
        manifest.parse(wrap({**VALID, "published": published}))


def test_duplicate_doc_id_is_rejected():
    """doc_id is what citations resolve against; two documents sharing one would
    make every citation against it ambiguous."""
    with pytest.raises(manifest.ManifestError, match="duplicate"):
        manifest.parse(wrap(VALID, {**VALID, "filename": "other.pdf"}))


@pytest.mark.parametrize("raw", [[], {}, {"documents": {}}, {"docs": []}, "documents"])
def test_structurally_wrong_manifest_is_rejected(raw):
    with pytest.raises(manifest.ManifestError, match="documents"):
        manifest.parse(raw)


def test_load_reads_and_validates_a_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(wrap(VALID)), encoding="utf-8")
    (doc,) = manifest.load(path)
    assert doc.doc_id == VALID["doc_id"]


def test_load_reports_a_missing_manifest_clearly(tmp_path):
    with pytest.raises(manifest.ManifestError, match="no corpus manifest"):
        manifest.load(tmp_path / "absent.json")


def test_load_reports_invalid_json_clearly(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(manifest.ManifestError, match="not valid JSON"):
        manifest.load(path)


def test_missing_files_are_reported_not_raised(tmp_path):
    """A manifest written before the PDFs are fetched is a normal state."""
    (doc,) = manifest.parse(wrap(VALID))
    assert manifest.missing_files([doc], tmp_path) == [VALID["filename"]]
    (tmp_path / VALID["filename"]).write_bytes(b"%PDF-1.7\n")
    assert manifest.missing_files([doc], tmp_path) == []


def test_committed_manifest_is_loadable():
    """The manifest actually in the repository must always parse, empty or not."""
    assert isinstance(manifest.load(), list)
