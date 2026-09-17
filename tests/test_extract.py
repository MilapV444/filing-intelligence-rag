"""Class-routed extraction tests (FR-3, FR-4).

Offline: no network, no API key, no model. The chart branch is asserted to make
no model call at all, which is the amended FR-3.
"""

import pytest

import synthetic
from risk_analytics import classify, extract, manifest

pymupdf = pytest.importorskip("pymupdf")


@pytest.fixture(scope="module")
def corpus():
    index = manifest.by_id()
    absent = manifest.missing_files(list(index.values()))
    if absent:
        pytest.fail(f"registered corpus documents not on disk: {absent}")
    return index


def page_of(corpus, doc_id, number):
    """Open one registered page. Documents resolve by manifest id, never path."""
    doc = pymupdf.open(corpus[doc_id].path())
    try:
        page = doc[number - 1]
        yield page, classify.classify_page(page, number, doc_id)
    finally:
        doc.close()


def extract_one(corpus, doc_id, number) -> extract.ExtractedPage:
    for page, record in page_of(corpus, doc_id, number):
        return extract.extract_page(page, record)


# --- Routing: the method matches the class -----------------------------------


def test_text_page_is_extracted_as_prose(corpus):
    page = extract_one(corpus, "agel-fy25", 1)  # Independent Auditor's Report
    assert page.page_class == classify.TEXT
    assert page.method == extract.PROSE
    assert "Independent Auditor" in page.text
    assert not page.tables


def test_table_page_is_extracted_with_its_structure(corpus):
    page = extract_one(corpus, "apsez-q1fy27", 34)  # covenant calculations
    assert page.page_class == classify.TABLE
    assert page.method == extract.TABLE_STRUCTURE
    assert page.tables


def test_table_cells_stay_in_their_own_row(corpus):
    """The value that answers golden question 1 must remain attached to the label
    that identifies it. Flattened into prose, `0.55` becomes a loose number on a
    page of loose numbers and any citation against it is guesswork."""
    page = extract_one(corpus, "apsez-q1fy27", 34)
    rows = [row for table in page.tables for row in table.rows]

    gearing = [r for r in rows if "Net Gearing (Total" in " | ".join(r)]
    assert gearing, "the Net Gearing row was lost"
    assert "0.55" in gearing[0], f"gearing value detached from its label: {gearing[0]}"

    dscr = [r for r in rows if "DSCR#" in " | ".join(r)]
    assert dscr, "the DSCR row was lost"
    assert "5.42" in dscr[0], f"DSCR value detached from its label: {dscr[0]}"


def test_rendered_table_keeps_one_row_per_line(corpus):
    """FR-5 forbids splitting a table mid-row. Rendering one row per line is what
    makes that enforceable by a chunker that splits on newlines."""
    page = extract_one(corpus, "apsez-q1fy27", 34)
    table = page.tables[0]
    lines = table.to_text().split("\n")
    assert len(lines) == len(table.rows)
    assert all("|" in line for line in lines if line.strip())


# --- The chart branch: recorded, never read ----------------------------------


def test_chart_page_is_skipped_and_never_read(tmp_path):
    """Amended FR-3. The corpus has no chart page, so this is exercised against a
    synthetic one -- the branch still has to behave."""
    path = tmp_path / "synthetic.pdf"
    expected = synthetic.build(path)
    chart_index = expected.index("chart") + 1

    doc = pymupdf.open(path)
    try:
        page = doc[chart_index - 1]
        record = classify.classify_page(page, chart_index, "synthetic")
        assert record.page_class == classify.CHART
        result = extract.extract_page(page, record)
    finally:
        doc.close()

    assert result.method == extract.SKIPPED
    assert result.text == ""
    assert not result.tables
    assert "skipped" in result.note
    assert result.is_empty


def test_extraction_makes_no_model_call(monkeypatch, corpus):
    """No API key present, and the module holds no client. If extraction ever
    starts needing one, it has stopped being the free deterministic layer the
    cost ceiling depends on."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not hasattr(extract, "anthropic")
    page = extract_one(corpus, "apsez-q1fy27", 34)
    assert page.method == extract.TABLE_STRUCTURE


# --- Honest fallback ----------------------------------------------------------


def test_unresolvable_table_falls_back_to_prose_and_says_so(corpus):
    """A page the classifier calls a table but whose grid cannot be resolved must
    announce the fallback. A silent one produces unaligned figures that read like
    a table and are wrong."""
    doc = pymupdf.open(corpus["agel-fy25"].path())
    try:
        page = doc[0]  # prose page, deliberately mislabelled to force the branch
        record = classify.classify_page(page, 1, "agel-fy25")
        forced = classify.PageRecord(
            doc_id=record.doc_id,
            page_number=record.page_number,
            page_class=classify.TABLE,
            extraction_method=classify.EXTRACTION_METHODS[classify.TABLE],
            reason="forced for test",
            features=record.features,
        )
        result = extract.extract_page(page, forced)
    finally:
        doc.close()

    assert result.page_class == classify.TABLE
    assert result.method == extract.PROSE
    assert "fell back to prose" in result.note
    assert "must not be read as a table" in result.note


# --- Whole-document coverage (FR-4) -------------------------------------------


@pytest.fixture(scope="module")
def apsez_pages(corpus):
    return extract.extract_document("apsez-q1fy27")


def test_every_page_records_the_method_that_actually_ran(apsez_pages):
    assert len(apsez_pages) == 34
    assert [p.page_number for p in apsez_pages] == list(range(1, 35))
    for page in apsez_pages:
        assert page.method in extract.METHODS
        assert page.page_class in classify.PAGE_CLASSES
        assert page.doc_id == "apsez-q1fy27"


def test_most_pages_yield_content(apsez_pages):
    """A filing where extraction produced nothing would mean a text-layer problem
    that silently empties the whole pipeline."""
    empty = [p.page_number for p in apsez_pages if p.is_empty]
    assert len(empty) <= 2, f"too many pages extracted empty: {empty}"


def test_method_counts_reconcile(apsez_pages):
    counts = extract.method_counts(apsez_pages)
    assert sum(counts.values()) == len(apsez_pages)
    assert counts[extract.SKIPPED] == 0, "no chart pages exist in this corpus"
    assert counts[extract.TABLE_STRUCTURE] > 0


def test_a_cell_containing_a_line_break_still_renders_as_one_row(corpus):
    """Regression: APSEZ p34 has a wrapped label that is one cell but two lines
    in the PDF. Rendering it verbatim produced more lines than rows, quietly
    breaking the invariant the chunker depends on."""
    page = extract_one(corpus, "apsez-q1fy27", 34)
    multiline = [
        row
        for table in page.tables
        for row in table.rows
        if any("\n" in (cell or "") for cell in row)
    ]
    assert multiline, "expected at least one wrapped cell on this page"
    for table in page.tables:
        assert len(table.to_text().split("\n")) == len(table.rows)
