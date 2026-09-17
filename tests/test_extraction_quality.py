"""Tests for extraction quality detection and repair
(src/ingestion/pdf_extractor.py).

Two distinct failure modes are kept apart throughout, because they need
different answers:

  REPAIRABLE    The separators are present but are not whitespace. The real
                document that motivated this uses a font whose space glyph
                maps to character code 2 and whose hyphen maps to code 3,
                with no ToUnicode table to say so, so extraction returns
                U+0002 between every word. PyMuPDF's geometry-aware "words"
                mode finds the boundaries anyway, so this IS recoverable.

  UNREPAIRABLE The glyphs really are printed with no gaps. There is nothing
                to recover, and the detector must keep saying so rather than
                accepting a reconstruction that is just as bad.

The detector must not be tuned to let the first case pass; it must keep
failing the second. Both are asserted below, end to end through the real
extractor.
"""

import pymupdf
import pytest

from src.ingestion.pdf_extractor import (
    MAX_MEAN_WORD_LENGTH,
    control_char_ratio,
    degraded_pages,
    extract_pages,
    mean_word_length,
    quality_problems,
    repaired_pages,
    text_statistics,
    whitespace_ratio,
)

HEALTHY = (
    "The patient was seen for a routine checkup. The mother reported that "
    "the child coughs when he runs and has done so for several weeks."
)
# Unrepairable: no separators of any kind between the words.
DEGRADED = (
    "Iconclude,therefore,thatthestandardofcarewasnotmetinthiscase."
    "3.DeviationFromTheStandardofCare:Failuretoimagetheentireabdominalaorta"
    "intheknownpresenceofthoracoproximalabdominalaorticaneurysm."
)
# Repairable: the real defect - U+0002 where a space belongs, U+0003 where a
# hyphen belongs.
CONTROL_SEPARATED = (
    "The\x02patient\x02was\x02discharged\x02on\x02anti\x03arrythmic\x02medication.\x02"
    "She\x02presented\x02with\x02a\x02two\x03day\x02history\x02of\x02right\x02lower\x02"
    "quadrant\x02pain\x02and\x02was\x02admitted\x02for\x02observation."
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
    """Unrepairable: no gaps between the glyphs, so there is nothing for the
    fallback to recover. It must stay flagged."""
    pdf = tmp_path / "spaceless.pdf"
    _write_pdf(pdf, DEGRADED)

    pages = extract_pages(str(pdf))

    assert degraded_pages(pages) == [1]
    assert repaired_pages(pages) == []


# --- the repairable failure: separators present but not whitespace ------------


def test_control_separated_pdf_is_repaired_end_to_end(tmp_path):
    """The real defect. U+0002 between words defeats str.split(), but the
    glyphs are laid out with real gaps, so word segmentation recovers them."""
    pdf = tmp_path / "control_separated.pdf"
    _write_pdf(pdf, CONTROL_SEPARATED)

    pages = extract_pages(str(pdf))
    text = pages[0]["text"]

    assert repaired_pages(pages) == [1]
    assert degraded_pages(pages) == []
    assert "\x02" not in text and "\x03" not in text
    assert "patient" in text and "discharged" in text
    assert pages[0]["extraction"] == "words"


def test_repair_does_not_invent_or_lose_words(tmp_path):
    """Every word in the repaired text must be a word PyMuPDF actually found
    on the page - the fallback moves separators, it does not write text."""
    pdf = tmp_path / "control_separated.pdf"
    _write_pdf(pdf, CONTROL_SEPARATED)

    pages = extract_pages(str(pdf))
    repaired_words = set(pages[0]["text"].split())

    doc = pymupdf.open(str(pdf))
    pymupdf_words = {w[4] for w in doc[0].get_text("words")}
    doc.close()

    assert repaired_words == pymupdf_words


def test_healthy_pdf_text_is_untouched_by_the_fallback(tmp_path):
    """No regression: a healthy page never enters the fallback, and its text
    is exactly what normal extraction produced."""
    pdf = tmp_path / "healthy.pdf"
    _write_pdf(pdf, HEALTHY)

    pages = extract_pages(str(pdf))

    assert repaired_pages(pages) == []
    assert pages[0]["extraction"] == "text"

    doc = pymupdf.open(str(pdf))
    assert pages[0]["text"] == doc[0].get_text("text")
    doc.close()


def test_page_numbers_and_order_survive_repair(tmp_path):
    """Page boundaries are preserved: a mixed document keeps 1..N in order,
    and only the malformed page is repaired."""
    doc = pymupdf.open()
    for text in (HEALTHY, CONTROL_SEPARATED, HEALTHY):
        page = doc.new_page()
        for i in range(12):
            page.insert_text((36, 72 + i * 18), text[:90])
    pdf = tmp_path / "mixed.pdf"
    doc.save(str(pdf))
    doc.close()

    pages = extract_pages(str(pdf))

    assert [p["page"] for p in pages] == [1, 2, 3]
    assert repaired_pages(pages) == [2]
    assert degraded_pages(pages) == []


# --- the quality signals ------------------------------------------------------


def test_control_characters_are_their_own_signal():
    """The signal that generalises: control characters are never valid
    document text, whatever the mean word length happens to be."""
    assert "control_characters" in quality_problems(CONTROL_SEPARATED * 3)
    assert quality_problems(HEALTHY * 3) == []


def test_each_failure_mode_reports_its_own_problem_names():
    assert quality_problems(DEGRADED * 3) == ["long_words", "no_whitespace"]
    assert quality_problems("") == ["empty"]


def test_statistics_separate_the_populations():
    healthy = text_statistics(HEALTHY * 3)
    broken = text_statistics(CONTROL_SEPARATED * 3)

    assert healthy["control_char_ratio"] == 0.0
    assert broken["control_char_ratio"] > 0.0
    assert healthy["whitespace_ratio"] > 0.10
    assert broken["whitespace_ratio"] < 0.05


def test_whitespace_and_control_ratios_of_empty_text_are_zero():
    for fn in (whitespace_ratio, control_char_ratio):
        assert fn("") == 0.0
        assert fn(None) == 0.0


def test_image_only_page_is_not_silently_repaired(tmp_path):
    """A scanned page has no text to recover. OCR is deliberately not part of
    this pipeline, so the honest outcome is the existing explicit error, not
    an empty reconstruction presented as a repair."""
    from src.exceptions import EmptyDocumentError

    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 120))
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(pymupdf.Rect(50, 50, 250, 170), pixmap=pix)
    pdf = tmp_path / "scanned.pdf"
    doc.save(str(pdf))
    doc.close()

    with pytest.raises(EmptyDocumentError):
        extract_pages(str(pdf))


def test_pages_with_no_text_are_skipped_not_repaired(tmp_path):
    """A blank page among real ones is dropped as before; page numbers of the
    pages that DO carry text stay correct."""
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(12):
        page.insert_text((36, 72 + i * 18), HEALTHY[:90])
    doc.new_page()  # deliberately blank
    page = doc.new_page()
    for i in range(12):
        page.insert_text((36, 72 + i * 18), HEALTHY[:90])
    pdf = tmp_path / "with_blank.pdf"
    doc.save(str(pdf))
    doc.close()

    pages = extract_pages(str(pdf))

    assert [p["page"] for p in pages] == [1, 3]
    assert repaired_pages(pages) == []


def test_ingest_reports_repaired_pages(tmp_path, monkeypatch):
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

    pdf = tmp_path / "control_separated.pdf"
    _write_pdf(pdf, CONTROL_SEPARATED)

    result = ingest_script.ingest_pdf(str(pdf))

    assert result["repaired_pages"] == [1]
    assert result["degraded_pages"] == []
    # Repaired text chunks into real words, so the chunker can split it.
    assert result["chunk_count"] > 0


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
