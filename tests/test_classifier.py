"""Page classification tests (FR-2, FR-4, NFR-5, AC-12).

Runs entirely offline: no network, no API key, no model. That is a requirement,
not a convenience -- AC-12 exists so the classifier can be regression-tested
freely while its thresholds are tuned against real documents.
"""

import json
from pathlib import Path

import pytest

import synthetic
from risk_analytics import classify, manifest

LABELS_PATH = Path(__file__).parent / "labelled_pages.json"
AC1_MIN_ACCURACY = 0.85


def features(**overrides) -> classify.PageFeatures:
    """A neutral page of prose, with the signal under test dialled in."""
    base = dict(
        page_number=1,
        char_count=2000,
        word_count=340,
        text_area_ratio=0.35,
        horizontal_rules=0,
        vertical_rules=0,
        curve_items=0,
        filled_shapes=0,
        image_area_ratio=0.0,
        figure_image_ratio=0.0,
        digit_ratio=0.02,
    )
    return classify.PageFeatures(**{**base, **overrides})


# --- The pure decision, with no PDF involved (AC-12) --------------------------


def test_running_prose_is_text():
    page_class, reason = classify.classify(features())
    assert page_class == classify.TEXT
    assert reason


def test_ruled_grid_is_a_table():
    page_class, _ = classify.classify(
        features(horizontal_rules=9, vertical_rules=5, char_count=300, digit_ratio=0.12)
    )
    assert page_class == classify.TABLE


def test_dense_grid_is_a_table_even_with_no_figures():
    """Entity and shareholding tables are fully ruled but carry names, not
    numbers. Requiring digits would misread them as prose."""
    page_class, _ = classify.classify(
        features(horizontal_rules=45, vertical_rules=60, digit_ratio=0.01)
    )
    assert page_class == classify.TABLE


def test_horizontal_ruling_alone_is_enough_for_a_table():
    """Measured against AGEL FY24-25: the Statement of Profit and Loss and most
    notes rule their rows and not their columns. Demanding vertical rules read
    the entire financial-statements section as prose."""
    page_class, reason = classify.classify(
        features(horizontal_rules=107, vertical_rules=1, digit_ratio=0.13, char_count=2700)
    )
    assert page_class == classify.TABLE
    assert "no vertical grid" in reason


def test_two_column_prose_with_a_few_rules_stays_text():
    """AGEL p31 is narrative set in two columns with three horizontal and two
    vertical decorative rules. Bare rule counts called it a table."""
    page_class, _ = classify.classify(
        features(horizontal_rules=3, vertical_rules=2, digit_ratio=0.0125, char_count=3701)
    )
    assert page_class == classify.TEXT


def test_numeric_density_finds_a_table_with_no_ruling_lines():
    """Printed annual reports align many tables by whitespace alone. A classifier
    that only looks for grid lines reads those as prose and hands a wall of
    figures to the prose extractor."""
    page_class, _ = classify.classify(features(digit_ratio=0.48, char_count=880))
    assert page_class == classify.TABLE


def test_page_dominated_by_figure_imagery_is_a_chart():
    page_class, _ = classify.classify(
        features(image_area_ratio=0.39, figure_image_ratio=0.39, char_count=39)
    )
    assert page_class == classify.CHART


def test_full_page_background_image_is_not_treated_as_a_figure():
    """Measured against APSEZ Q1 FY27: every page carries a page-sized raster
    behind real text, so raw image area is 1.0 throughout and says nothing. Two
    ordinary pages were called charts before page-sized images were excluded."""
    page_class, _ = classify.classify(
        features(image_area_ratio=1.0, figure_image_ratio=0.0, char_count=1071, digit_ratio=0.08)
    )
    assert page_class == classify.TEXT


def test_vector_drawing_with_almost_no_text_is_a_chart():
    page_class, _ = classify.classify(features(char_count=9, curve_items=1, filled_shapes=8))
    assert page_class == classify.CHART


def test_prose_page_with_a_small_logo_stays_text():
    """Nearly every report page carries a logo or a decorative rule. Treating a
    raster as sufficient evidence of a chart would misroute most of the corpus."""
    page_class, _ = classify.classify(features(image_area_ratio=0.08, figure_image_ratio=0.08))
    assert page_class == classify.TEXT


def test_large_image_under_heavy_prose_stays_text():
    """A full-bleed background behind a page of narrative is not a figure."""
    page_class, _ = classify.classify(
        features(image_area_ratio=0.6, figure_image_ratio=0.6, char_count=3000)
    )
    assert page_class == classify.TEXT


def test_imagery_outranks_ruling_lines():
    """Rule order is load-bearing: an image-based chart sitting inside a ruled
    border must be read as a chart, not as the table the border resembles."""
    page_class, _ = classify.classify(
        features(
            image_area_ratio=0.5,
            figure_image_ratio=0.5,
            char_count=40,
            horizontal_rules=6,
            vertical_rules=4,
        )
    )
    assert page_class == classify.CHART


def test_a_chart_axis_pair_does_not_reach_the_table_threshold():
    """A bar chart draws two axis lines. Two must not read as a grid."""
    page_class, _ = classify.classify(features(char_count=900, horizontal_rules=1, vertical_rules=1))
    assert page_class == classify.TEXT


def test_blank_page_is_text_and_says_so():
    page_class, reason = classify.classify(features(char_count=0, word_count=0, text_area_ratio=0.0))
    assert page_class == classify.TEXT
    assert "blank" in reason.lower()


def test_every_class_is_one_of_the_three_declared():
    """FR-2 allows exactly three classes; a fourth would break routing."""
    probes = [
        features(),
        features(horizontal_rules=9, vertical_rules=5, digit_ratio=0.1),
        features(image_area_ratio=0.5, figure_image_ratio=0.5, char_count=10),
        features(char_count=0),
        features(digit_ratio=0.9, char_count=400),
    ]
    for probe in probes:
        page_class, reason = classify.classify(probe)
        assert page_class in classify.PAGE_CLASSES
        assert reason.strip(), "every decision must carry a reason (FR-4)"


# --- Against real PDF pages ---------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("corpus") / "synthetic.pdf"
    expected = synthetic.build(path)
    return path, expected


def test_synthetic_pages_are_classified_correctly(synthetic_pdf):
    path, expected = synthetic_pdf
    records = classify.classify_document(path, "synthetic")
    got = [r.page_class for r in records]
    wrong = [
        f"page {i + 1}: expected {e}, got {g}"
        for i, (e, g) in enumerate(zip(expected, got))
        if e != g
    ]
    assert not wrong, "; ".join(wrong)


def test_every_page_is_classified(synthetic_pdf):
    """FR-2 says every page, so a document must produce exactly one record per
    page with no gaps in numbering."""
    path, expected = synthetic_pdf
    records = classify.classify_document(path, "synthetic")
    assert len(records) == len(expected)
    assert [r.page_number for r in records] == list(range(1, len(expected) + 1))


def test_each_record_carries_the_routing_decision(synthetic_pdf):
    """FR-4: class, routed extraction method, and the reason behind them."""
    path, _ = synthetic_pdf
    for record in classify.classify_document(path, "synthetic"):
        assert record.doc_id == "synthetic"
        assert record.page_class in classify.PAGE_CLASSES
        assert record.extraction_method == classify.EXTRACTION_METHODS[record.page_class]
        assert record.reason.strip()


def test_classification_needs_no_credentials_or_network(monkeypatch, synthetic_pdf):
    """FR-2 and NFR-5. If this ever starts needing a key, it has stopped being
    the cheap deterministic layer the design depends on."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path, expected = synthetic_pdf
    records = classify.classify_document(path, "synthetic")
    assert len(records) == len(expected)
    assert not hasattr(classify, "anthropic")


def test_records_round_trip_to_disk(tmp_path, synthetic_pdf):
    path, _ = synthetic_pdf
    records = classify.classify_document(path, "synthetic")
    out = tmp_path / "pages" / "synthetic.json"
    classify.write_page_records(records, out)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert len(loaded) == len(records)
    first = loaded[0]
    assert first["page_class"] == records[0].page_class
    assert first["extraction_method"] == records[0].extraction_method
    assert first["reason"] == records[0].reason
    # The measurements that caused the decision must survive, or a disputed
    # classification cannot be argued about after the fact (FR-4).
    assert first["features"]["char_count"] == records[0].features.char_count


def test_class_counts_cover_all_three_classes(synthetic_pdf):
    path, _ = synthetic_pdf
    counts = classify.class_counts(classify.classify_document(path, "synthetic"))
    assert set(counts) == set(classify.PAGE_CLASSES)
    assert sum(counts.values()) == 7
    assert counts["text"] == 3 and counts["table"] == 2 and counts["chart"] == 2


# --- AC-1: measured accuracy on the real corpus ------------------------------
# The synthetic pages above prove the classifier reads what it claims to read.
# They cannot prove the thresholds hold on real documents, which is what AC-1 is
# for. Labels in labelled_pages.json were assigned by rendering each page and
# reading it. `tune` was used while setting thresholds; `holdout` was labelled
# afterwards and never tuned against, so it is the honest measurement.


def _labels():
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def _corpus():
    """Resolve corpus documents through the manifest (FR-1, NFR-3): every page
    read here belongs to a document with a recorded public source."""
    index = manifest.by_id()
    absent = manifest.missing_files(list(index.values()))
    if absent:
        pytest.fail(
            f"AC-1 cannot be measured: {absent} registered in the manifest but not "
            "on disk. AC-1 is a claim about the real documents, so this fails "
            "rather than skipping."
        )
    return index


def _score(split):
    import pymupdf

    spec = _labels()
    index = _corpus()
    docs = {doc_id: pymupdf.open(doc.path()) for doc_id, doc in index.items()}
    try:
        wrong = []
        for item in spec[split]:
            page = docs[item["doc"]][item["page"] - 1]
            record = classify.classify_page(page, item["page"], item["doc"])
            if record.page_class != item["label"]:
                wrong.append(
                    f"{item['doc']} p{item['page']} ({item['note']}): "
                    f"expected {item['label']}, got {record.page_class} "
                    f"because {record.reason}"
                )
        return len(spec[split]), wrong
    finally:
        for doc in docs.values():
            doc.close()


def test_ac1_accuracy_on_unseen_holdout_pages():
    """AC-1: at least 85 percent correct on a stratified sample of real pages.
    This split was labelled after the thresholds were fixed."""
    total, wrong = _score("holdout")
    accuracy = (total - len(wrong)) / total
    assert accuracy >= AC1_MIN_ACCURACY, (
        f"holdout accuracy {accuracy:.0%} is below the AC-1 floor of "
        f"{AC1_MIN_ACCURACY:.0%}:\n  " + "\n  ".join(wrong)
    )


def test_ac1_accuracy_on_the_tuning_pages():
    """The tuning split is not independent evidence -- the thresholds were fitted
    against it. It is kept as a regression guard: a change that breaks pages
    already known to be handled should fail loudly."""
    total, wrong = _score("tune")
    accuracy = (total - len(wrong)) / total
    assert accuracy >= AC1_MIN_ACCURACY, (
        f"tuning-set accuracy regressed to {accuracy:.0%}:\n  " + "\n  ".join(wrong)
    )


def test_the_sample_is_stratified_across_documents_and_classes():
    """A sample drawn from one document, or made only of easy prose pages, would
    make the AC-1 number meaningless."""
    spec = _labels()
    for split in ("tune", "holdout"):
        items = spec[split]
        assert len(items) >= 20, f"{split} sample is too small to support AC-1"
        assert len({i["doc"] for i in items}) == 2, f"{split} covers only one document"
        labels = {i["label"] for i in items}
        assert {"text", "table"} <= labels, f"{split} does not cover text and tables"


def test_no_chart_pages_exist_in_the_current_corpus():
    """Documented, not aspirational. Neither document contains a single chart
    page: AGEL's financial-statements extract has no raster imagery at all, and
    APSEZ's largest figure covers 4 percent of a page. The chart branch of the
    classifier is therefore unexercised by real data, and AC-9 -- which needs a
    question answerable only from a chart page -- cannot be met by this corpus.

    If a document with charts is added, this test should start failing. That is
    the point: it is a tripwire, so the gap cannot be quietly forgotten."""
    found = []
    for doc_id, doc in _corpus().items():
        found += [
            (doc_id, r.page_number)
            for r in classify.classify_document(doc.path(), doc_id)
            if r.page_class == classify.CHART
        ]
    assert not found, (
        "chart pages now exist in the corpus: "
        f"{found}. Re-examine AC-9 and the vision extraction path, then update "
        "this test."
    )
