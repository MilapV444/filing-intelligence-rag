"""Page classification and extraction routing (FR-2, FR-4).

Every page is assigned exactly one of three classes from PyMuPDF layout
statistics alone. No API call, no network, no model (FR-2, NFR-5) -- which is
what lets the whole classifier be regression-tested offline (AC-12) and keeps
ingestion cost bounded.

The decision is deliberately split in two:

    page_features(page)  ->  PageFeatures     needs a PDF, no judgement
    classify(features)   ->  (class, reason)  pure judgement, no PDF

so the thresholds can be tested and tuned against recorded feature values
without carrying PDF fixtures around, and a misclassification can be argued
about in terms of the numbers that caused it.

Every record carries a human-readable `reason` because FR-4 requires the
routing decision to be inspectable and disputable after the fact. A classifier
that says "chart" without saying why cannot be corrected.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

TEXT, TABLE, CHART = "text", "table", "chart"
PAGE_CLASSES = (TEXT, TABLE, CHART)

# FR-3 routing target. T3 owns the implementations; T2 records the decision.
EXTRACTION_METHODS = {
    TEXT: "prose",
    TABLE: "table_structure",
    CHART: "vision",
}

# --- Thresholds --------------------------------------------------------------
# Named and gathered here because AC-1 requires at least 85 percent accuracy on a
# real sample, and reaching it means tuning these numbers against actual annual
# reports. Tuning is expected; scattering the constants through the code is not.

MIN_RULE_LENGTH = 40.0  # pt. Shorter axis-aligned segments are underlines and glyph
# decoration, not table rules. An A4 page is 595 x 842 pt.
RULE_TOLERANCE = 1.5  # pt of deviation still counted as axis-aligned.

# A single image this large is a page scan or a full-bleed background, not a
# figure on the page. Measured against APSEZ Q1 FY27, where every page carries a
# full-page raster behind real text: without this, image area is 1.0 everywhere
# and carries no information at all.
FULL_PAGE_IMAGE_RATIO = 0.9

CHART_MIN_FIGURE_AREA = 0.25  # page fraction covered by figure-sized imagery
CHART_MAX_CHARS = 1200  # a page with a big figure but this much prose is still prose
CHART_MIN_VECTOR_ITEMS = 25  # charts drawn as vectors rather than embedded images
SPARSE_TEXT_CHARS = 60  # below this a page carries no readable prose

# Financial statements are very often ruled horizontally and not vertically, so a
# grid is sufficient evidence of a table but never necessary. Measured against
# AGEL FY24-25: requiring vertical rules misread the Statement of Profit and Loss
# and most of the notes as prose.
TABLE_STRONG_GRID_RULES = 8  # h and v both this high is a table whatever the text
TABLE_MIN_HORIZONTAL_RULES = 3
TABLE_MIN_VERTICAL_RULES = 2
TABLE_GRID_MIN_DIGIT_RATIO = 0.03  # a light grid needs some figures to back it up;
# two-column prose pages draw a few rules and would otherwise read as tables
TABLE_RULED_ROWS = 8  # horizontal ruling alone, if the page carries figures
TABLE_RULED_ROWS_MIN_DIGIT_RATIO = 0.05
TABLE_MIN_DIGIT_RATIO = 0.18  # whitespace-aligned tables have no ruling lines at
TABLE_MIN_CHARS = 150  # all, so numeric density is the last-resort signal


@dataclass(frozen=True)
class PageFeatures:
    """Layout statistics for one page. Everything `classify` is allowed to see."""

    page_number: int  # 1-indexed, because citations are read by humans
    char_count: int
    word_count: int
    text_area_ratio: float
    horizontal_rules: int
    vertical_rules: int
    curve_items: int
    filled_shapes: int
    image_area_ratio: float  # all raster imagery, including page-sized backgrounds
    figure_image_ratio: float  # only imagery small enough to be a figure on the page
    digit_ratio: float


@dataclass(frozen=True)
class PageRecord:
    """The recorded classification and routing decision for one page (FR-4)."""

    doc_id: str
    page_number: int
    page_class: str
    extraction_method: str
    reason: str
    features: PageFeatures

    def to_dict(self) -> dict:
        return {**asdict(self), "features": asdict(self.features)}


def _area(box) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _count_rules(drawings) -> tuple[int, int, int, int]:
    """Count axis-aligned rules long enough to be table grid lines.

    Both explicit lines and very thin rectangles are counted: many PDF producers
    draw table rules as filled rectangles a fraction of a point tall, and missing
    those would make every ruled table in such a document look like prose.
    """
    horizontal = vertical = curves = filled = 0
    for path in drawings:
        if path.get("fill") is not None:
            filled += 1
        for item in path.get("items", ()):
            kind = item[0]
            if kind == "l":
                p0, p1 = item[1], item[2]
                dx, dy = abs(p1.x - p0.x), abs(p1.y - p0.y)
                if dy <= RULE_TOLERANCE and dx >= MIN_RULE_LENGTH:
                    horizontal += 1
                elif dx <= RULE_TOLERANCE and dy >= MIN_RULE_LENGTH:
                    vertical += 1
            elif kind == "re":
                rect = item[1]
                if rect.height <= RULE_TOLERANCE and rect.width >= MIN_RULE_LENGTH:
                    horizontal += 1
                elif rect.width <= RULE_TOLERANCE and rect.height >= MIN_RULE_LENGTH:
                    vertical += 1
            elif kind in ("c", "qu"):
                curves += 1
    return horizontal, vertical, curves, filled


def page_features(page, page_number: int) -> PageFeatures:
    """Measure one page. No judgement here -- see `classify`."""
    page_area = _area(page.rect) or 1.0

    text = page.get_text("text")
    words = page.get_text("words")
    text_area = sum(_area(w[:4]) for w in words)

    digits = sum(c.isdigit() for c in text)
    alnum = sum(c.isalnum() for c in text)

    horizontal, vertical, curves, filled = _count_rules(page.get_drawings())

    image_areas = [_area(info["bbox"]) for info in page.get_image_info()]
    image_area = sum(image_areas)
    # Ignore any single image large enough to be a scan or a full-bleed
    # background. Counting those makes every page of a scanned filing look like
    # a figure, which is exactly the false positive this avoids.
    figure_area = sum(a for a in image_areas if a / page_area < FULL_PAGE_IMAGE_RATIO)

    return PageFeatures(
        page_number=page_number,
        char_count=len(text.strip()),
        word_count=len(words),
        text_area_ratio=round(text_area / page_area, 4),
        horizontal_rules=horizontal,
        vertical_rules=vertical,
        curve_items=curves,
        filled_shapes=filled,
        image_area_ratio=round(min(image_area / page_area, 1.0), 4),
        figure_image_ratio=round(min(figure_area / page_area, 1.0), 4),
        digit_ratio=round(digits / alnum, 4) if alnum else 0.0,
    )


def classify(features: PageFeatures) -> tuple[str, str]:
    """Assign exactly one class, with the reason that decided it.

    Rules are ordered and the first match wins, so the reason names a single
    cause rather than a blend. Ordering matters: raster imagery is checked before
    ruling lines because a scanned or image-based chart often sits on a page that
    also has a border, and numeric density is checked last because it is the
    weakest signal.
    """
    f = features

    if f.figure_image_ratio >= CHART_MIN_FIGURE_AREA and f.char_count < CHART_MAX_CHARS:
        return CHART, (
            f"figure-sized imagery covers {f.figure_image_ratio:.0%} of the page "
            f"with only {f.char_count} characters of text"
        )

    if f.horizontal_rules >= TABLE_STRONG_GRID_RULES and f.vertical_rules >= TABLE_STRONG_GRID_RULES:
        return TABLE, (
            f"{f.horizontal_rules} horizontal and {f.vertical_rules} vertical rules "
            "form a dense grid"
        )

    if (
        f.horizontal_rules >= TABLE_MIN_HORIZONTAL_RULES
        and f.vertical_rules >= TABLE_MIN_VERTICAL_RULES
        and f.digit_ratio >= TABLE_GRID_MIN_DIGIT_RATIO
    ):
        return TABLE, (
            f"{f.horizontal_rules} horizontal and {f.vertical_rules} vertical rules "
            f"form a grid over {f.digit_ratio:.0%} digits"
        )

    if (
        f.horizontal_rules >= TABLE_RULED_ROWS
        and f.digit_ratio >= TABLE_RULED_ROWS_MIN_DIGIT_RATIO
    ):
        return TABLE, (
            f"{f.horizontal_rules} horizontal rules over {f.digit_ratio:.0%} digits: "
            "ruled rows of figures with no vertical grid, which is how most "
            "financial statements are set"
        )

    if f.digit_ratio >= TABLE_MIN_DIGIT_RATIO and f.char_count >= TABLE_MIN_CHARS:
        return TABLE, (
            f"{f.digit_ratio:.0%} of alphanumeric characters are digits over "
            f"{f.char_count} characters, which reads as a column of figures "
            "rather than prose"
        )

    if f.char_count < SPARSE_TEXT_CHARS and (
        f.curve_items > 0
        or f.filled_shapes >= CHART_MIN_VECTOR_ITEMS
        or f.figure_image_ratio > 0.05
    ):
        return CHART, (
            f"almost no text ({f.char_count} characters) beside "
            f"{f.curve_items} curves and {f.filled_shapes} filled shapes, "
            "which is a drawn figure rather than a page of prose"
        )

    if f.char_count < SPARSE_TEXT_CHARS:
        return TEXT, (
            f"blank or near-empty page ({f.char_count} characters); "
            "routed to prose extraction, which will yield nothing"
        )

    return TEXT, (
        f"{f.char_count} characters of running text with no grid "
        f"({f.horizontal_rules} horizontal rules) and "
        f"{f.digit_ratio:.0%} digits"
    )


def classify_page(page, page_number: int, doc_id: str) -> PageRecord:
    features = page_features(page, page_number)
    page_class, reason = classify(features)
    return PageRecord(
        doc_id=doc_id,
        page_number=page_number,
        page_class=page_class,
        extraction_method=EXTRACTION_METHODS[page_class],
        reason=reason,
        features=features,
    )


def classify_document(path: Path, doc_id: str) -> list[PageRecord]:
    """Classify every page of one PDF. FR-2 requires every page to be covered."""
    with pymupdf.open(path) as doc:
        return [classify_page(page, number, doc_id) for number, page in enumerate(doc, start=1)]


def write_page_records(records: list[PageRecord], path: Path) -> None:
    """Persist the routing decisions so they can be inspected and disputed (FR-4)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([r.to_dict() for r in records], indent=2),
        encoding="utf-8",
    )


def class_counts(records: list[PageRecord]) -> dict[str, int]:
    return {cls: sum(r.page_class == cls for r in records) for cls in PAGE_CLASSES}
