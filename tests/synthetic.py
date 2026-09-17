"""Synthetic PDF pages with known correct classes.

These stand in for real annual-report pages until the corpus is chosen. They are
built with PyMuPDF rather than committed as binary fixtures, so the test suite
carries no opaque files and each page's construction states plainly what it is
meant to represent.

What they prove: the classifier reads the layout signals it claims to read, and
the ordering of its rules is right. What they cannot prove: the thresholds are
correct for real documents. Annual-report pages are messier than anything
constructed here -- AC-1 measures accuracy on a real sample, and these fixtures
are not a substitute for it.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

PAGE = pymupdf.paper_rect("a4")

PROSE = (
    "The Company continued to focus on operational resilience during the year "
    "under review. Management assesses liquidity on a rolling basis and "
    "maintains committed facilities sufficient to meet near term obligations. "
    "The Board reviewed the risk register at each quarterly meeting and "
    "concluded that the existing mitigations remain appropriate to the scale "
    "and nature of the operations carried on by the group. "
)

FIGURE_ROWS = [
    "Revenue from operations 125430 118220 104880",
    "Other income 3420 2980 2610",
    "Total income 128850 121200 107490",
    "Finance costs 18640 19220 20110",
    "Depreciation and amortisation 22310 21040 19870",
    "Profit before tax 31280 26410 21330",
    "Total tax expense 8120 6890 5540",
    "Profit for the year 23160 19520 15790",
]


def _text_page(doc, body: str, fontsize: int = 10):
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    page.insert_textbox(
        pymupdf.Rect(50, 50, PAGE.width - 50, PAGE.height - 50),
        body,
        fontsize=fontsize,
        fontname="helv",
    )
    return page


def _swatch(width: int, height: int, colour=(40, 90, 160)):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height))
    pix.set_rect(pix.irect, colour)
    return pix


def prose_page(doc):
    """A page of running narrative. The commonest page in any annual report."""
    _text_page(doc, PROSE * 6)


def prose_with_logo_page(doc):
    """Prose carrying a small decorative image. Must not be pulled to `chart`
    just because a raster is present -- most report pages have one."""
    page = _text_page(doc, PROSE * 6)
    page.insert_image(pymupdf.Rect(460, 40, 540, 90), pixmap=_swatch(80, 50))


def ruled_table_page(doc):
    """A financial table drawn with an explicit grid."""
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    top, row_height, left, right = 120, 28, 50, PAGE.width - 50

    for i in range(len(FIGURE_ROWS) + 1):
        y = top + i * row_height
        page.draw_line(pymupdf.Point(left, y), pymupdf.Point(right, y), width=0.7)
    for x in (left, 300, 400, 495, right):
        page.draw_line(
            pymupdf.Point(x, top),
            pymupdf.Point(x, top + len(FIGURE_ROWS) * row_height),
            width=0.7,
        )
    for i, row in enumerate(FIGURE_ROWS):
        page.insert_textbox(
            pymupdf.Rect(left + 4, top + i * row_height + 6, right - 4, top + (i + 1) * row_height),
            row,
            fontsize=9,
            fontname="helv",
        )


def unruled_table_page(doc):
    """A financial table aligned by whitespace with no ruling lines at all.
    Common in printed annual reports, and invisible to a rules-only classifier."""
    _text_page(doc, "\n".join(FIGURE_ROWS * 3), fontsize=9)


def raster_chart_page(doc):
    """A chart embedded as an image, with a caption and nothing else."""
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    page.insert_image(pymupdf.Rect(80, 180, 520, 620), pixmap=_swatch(440, 440))
    page.insert_textbox(
        pymupdf.Rect(80, 630, 520, 700),
        "Scope 1 and Scope 2 emissions intensity",
        fontsize=10,
        fontname="helv",
    )


def vector_chart_page(doc):
    """A chart drawn as vectors rather than embedded as an image. Has axis lines,
    so it must not be mistaken for a table grid."""
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    base, left = 600, 90

    page.draw_line(pymupdf.Point(left, base), pymupdf.Point(500, base), width=1.0)
    page.draw_line(pymupdf.Point(left, base), pymupdf.Point(left, 200), width=1.0)

    for i, height in enumerate([120, 180, 150, 240, 210, 300, 275, 330]):
        x = left + 20 + i * 48
        page.draw_rect(
            pymupdf.Rect(x, base - height, x + 30, base),
            fill=(0.2, 0.4, 0.75),
            color=(0.2, 0.4, 0.75),
        )

    page.draw_bezier(
        pymupdf.Point(left + 20, 420),
        pymupdf.Point(220, 330),
        pymupdf.Point(380, 380),
        pymupdf.Point(500, 260),
        color=(0.8, 0.3, 0.1),
        width=1.5,
    )
    page.insert_textbox(
        pymupdf.Rect(left, 640, 500, 680), "FY21-FY25", fontsize=9, fontname="helv"
    )


def blank_page(doc):
    """A section divider. Real reports are full of them."""
    doc.new_page(width=PAGE.width, height=PAGE.height)


# Page builder -> the class it must be assigned. Order is the page order.
EXPECTED = [
    (prose_page, "text"),
    (prose_with_logo_page, "text"),
    (ruled_table_page, "table"),
    (unruled_table_page, "table"),
    (raster_chart_page, "chart"),
    (vector_chart_page, "chart"),
    (blank_page, "text"),
]


def build(path: Path) -> list[str]:
    """Write the synthetic document and return the expected class per page."""
    doc = pymupdf.open()
    for builder, _ in EXPECTED:
        builder(doc)
    doc.save(str(path))
    doc.close()
    return [expected for _, expected in EXPECTED]
