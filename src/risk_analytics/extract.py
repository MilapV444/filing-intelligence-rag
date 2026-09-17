"""Class-routed extraction (FR-3, FR-4).

Each page is extracted by the method its class dictates:

    text   -> prose            running text, as laid out
    table  -> table_structure  rows and cells preserved
    chart  -> skipped          recorded, never read

Nothing here calls a model or touches the network. The chart branch is a
deliberate no-op: the corpus contains no chart page, so the vision path was
withdrawn from the specification rather than shipped untested.

Every page records the method that actually ran, which is not always the method
its class asked for -- a page classified `table` whose grid the extractor cannot
resolve falls back to prose and says so. FR-4 wants the decision inspectable
after the fact, and a silent fallback is the one that produces confidently wrong
numbers later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from . import classify, manifest

PROSE = "prose"
TABLE_STRUCTURE = "table_structure"
SKIPPED = "skipped"
METHODS = (PROSE, TABLE_STRUCTURE, SKIPPED)

# A "table" with one row and one column is a text box the detector latched onto,
# not a table. Extracting it as one loses the surrounding prose.
MIN_TABLE_ROWS = 2


@dataclass(frozen=True)
class Table:
    """One detected table. Rows stay in document order and cells stay separate;
    flattening either is what turns a balance sheet into unreadable prose."""

    rows: list[list[str]]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), max((len(r) for r in self.rows), default=0)

    def to_text(self) -> str:
        """Render for embedding and for a model to read. One row per line, cells
        pipe-separated, so a chunker splitting on newlines cannot cut a row in
        half (FR-5).

        Cells carry their own line breaks -- a wrapped label like "intangible
        asset under\\ndevelopment" is one cell in the PDF. Those are collapsed to
        spaces here, or one row would render as two lines and the invariant this
        method exists to provide would be silently false.
        """
        return "\n".join(
            " | ".join(" ".join((c or "").split()) for c in row) for row in self.rows
        )


@dataclass(frozen=True)
class ExtractedPage:
    doc_id: str
    page_number: int
    page_class: str
    method: str
    text: str
    tables: list[Table] = field(default_factory=list)
    note: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip() and not self.tables

    def to_dict(self) -> dict:
        return {**asdict(self), "tables": [t.rows for t in self.tables]}


def _prose(page) -> str:
    return page.get_text("text").strip()


def _tables(page) -> list[Table]:
    try:
        found = page.find_tables()
    except Exception:  # detector failures must not abort a 401-page ingest
        return []
    tables = []
    for table in found.tables:
        rows = [[(cell or "").strip() for cell in row] for row in table.extract()]
        rows = [r for r in rows if any(c for c in r)]
        if len(rows) >= MIN_TABLE_ROWS:
            tables.append(Table(rows=rows))
    return tables


def extract_page(page, record: classify.PageRecord) -> ExtractedPage:
    """Extract one page according to its recorded class."""
    common = dict(
        doc_id=record.doc_id, page_number=record.page_number, page_class=record.page_class
    )

    if record.page_class == classify.CHART:
        # FR-3 as amended: recorded and skipped. No model call is attempted.
        return ExtractedPage(
            **common,
            method=SKIPPED,
            text="",
            note="chart page recorded and skipped; figure interpretation is a non-goal",
        )

    if record.page_class == classify.TABLE:
        tables = _tables(page)
        if tables:
            return ExtractedPage(
                **common,
                method=TABLE_STRUCTURE,
                text=_prose(page),
                tables=tables,
                note=f"{len(tables)} table(s) resolved, shapes {[t.shape for t in tables]}",
            )
        return ExtractedPage(
            **common,
            method=PROSE,
            text=_prose(page),
            note=(
                "classified table but no grid could be resolved; fell back to prose. "
                "Figures on this page are unaligned and must not be read as a table."
            ),
        )

    return ExtractedPage(**common, method=PROSE, text=_prose(page))


def extract_document(doc_id: str, documents_dir: Path | None = None) -> list[ExtractedPage]:
    """Classify and extract every page of one registered document.

    The document is resolved through the manifest, never by path, so nothing is
    read that lacks a recorded public source (NFR-3).
    """
    document = manifest.by_id()[doc_id]
    path = document.path(documents_dir)
    with pymupdf.open(path) as doc:
        return [
            extract_page(page, classify.classify_page(page, number, doc_id))
            for number, page in enumerate(doc, start=1)
        ]


def method_counts(pages: list[ExtractedPage]) -> dict[str, int]:
    return {method: sum(p.method == method for p in pages) for method in METHODS}
