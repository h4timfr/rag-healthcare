"""Tests for degraded-extraction detection (src/ingestion/pdf_extractor.py).

Found while verifying a QA dataset premise against the real index: every
page of one of the three indexed documents extracts as a single run of
concatenated text ("Iconclude,therefore,thatthestandardofcarewasnotmet"),
averaging 53-86 characters per token against 4.4-7.0 for the other two.

That silently defeats FTS5 keyword matching, defeats the word-based chunker
(a whole page counts as ~40 "words", so it is never split), and makes the
model look like it is refusing to answer questions whose evidence is right
there. It is detected here so it is visible at upload time instead.

Detection only - nothing here repairs the text.
"""

import pymupdf
import pytest

from src.ingestion.pdf_extractor import (
    MAX_MEAN_WORD_LENGTH,
    degraded_pages,
    extract_pages,
    mean_word_length,
)

HEALTHY = (
    "The patient was seen for a two year well check. The mother reported that "
    "the child coughs when he runs and has done so for several weeks."
)
# The real shape of the failure: punctuation survives, spaces do not.
DEGRADED = (
    "Iconclude,therefore,thatthestandardofcarewasnotmetinthiscase."
    "3.DeviationFromTheStandardofCare:Failuretoimagetheentireabdominalaorta"
    "intheknownpresenceofthoracoproximalabdominalaorticaneurysm."
)


def test_mean_word_length_separates_the_two_populations():
    assert mean_word_length(HEALTHY) < MAX_MEAN_WORD_LENGTH
    assert mean_word_length(DEGRADED) > MAX_MEAN_WORD_LENGTH


def test_mean_word_length_of_no_text_is_zero():
    assert mean_word_length("") == 0.0
    assert mean_word_length("   ") == 0.0
    assert mean_word_length(None) == 0.0


def test_healthy_pages_are_not_flagged():
    pages = [{"page": 1, "text": HEALTHY}, {"page": 2, "text": HEALTHY}]
    assert degraded_pages(pages) == []


def test_degraded_page_is_flagged_by_number():
    pages = [
        {"page": 1, "text": HEALTHY},
        {"page": 2, "text": DEGRADED * 3},
        {"page": 3, "text": HEALTHY},
    ]
    assert degraded_pages(pages) == [2]


def test_short_pages_are_not_judged():
    """A heading or a stray line is too little text for the mean to mean
    anything - flagging it would train the user to ignore the warning."""
    pages = [{"page": 1, "text": "Confidential"}, {"page": 2, "text": "AppendixA"}]
    assert degraded_pages(pages) == []


def test_a_single_run_page_is_judged_not_skipped():
    """Regression: the length guard counts characters, not tokens. A page
    extracted as one 3000-character run is a single token, and a token-count
    guard would have skipped precisely the worst case."""
    pages = [{"page": 1, "text": "a" * 3000}]
    assert degraded_pages(pages) == [1]


def test_empty_page_list_is_not_flagged():
    assert degraded_pages([]) == []


def test_threshold_is_configurable_for_callers_that_want_to_be_stricter():
    pages = [{"page": 1, "text": HEALTHY * 2}]
    assert degraded_pages(pages) == []
    assert degraded_pages(pages, max_mean_word_length=3.0) == [1]


# --- end to end through the real extractor ------------------------------------


def _write_pdf(path, text, lines=12):
    """Write `text` as several lines. insert_text() does not wrap, so a
    single call is clipped at the page edge - a realistic page needs enough
    lines to carry a page's worth of characters."""
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(lines):
        page.insert_text((36, 72 + i * 18), text[:90])
    doc.save(str(path))
    doc.close()


def test_real_extraction_of_a_healthy_pdf_is_not_flagged(tmp_path):
    pdf = tmp_path / "healthy.pdf"
    _write_pdf(pdf, HEALTHY)

    pages = extract_pages(str(pdf))

    assert degraded_pages(pages) == []


def test_real_extraction_of_a_spaceless_pdf_is_flagged(tmp_path):
    pdf = tmp_path / "spaceless.pdf"
    _write_pdf(pdf, DEGRADED)

    pages = extract_pages(str(pdf))

    assert degraded_pages(pages) == [1]


def test_ingest_reports_degraded_pages(tmp_path, monkeypatch):
    """The flag has to survive all the way out of ingest_pdf() - that is what
    the upload path and the CLI surface to the user."""
    import numpy as np

    import scripts.ingest as ingest_script
    from src import config

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(
        ingest_script,
        "embed_texts",
        lambda texts: np.ones((len(texts), config.EMBEDDING_DIM), dtype="float32"),
    )

    pdf = tmp_path / "spaceless.pdf"
    _write_pdf(pdf, DEGRADED)

    result = ingest_script.ingest_pdf(str(pdf))

    assert result["degraded_pages"] == [1]
    # Still indexed: degraded text beats no document at all.
    assert result["chunk_count"] > 0
    assert result["skipped"] is False


def test_healthy_ingest_reports_no_degraded_pages(tmp_path, monkeypatch):
    import numpy as np

    import scripts.ingest as ingest_script
    from src import config

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(
        ingest_script,
        "embed_texts",
        lambda texts: np.ones((len(texts), config.EMBEDDING_DIM), dtype="float32"),
    )

    pdf = tmp_path / "healthy.pdf"
    _write_pdf(pdf, HEALTHY)

    assert ingest_script.ingest_pdf(str(pdf))["degraded_pages"] == []
