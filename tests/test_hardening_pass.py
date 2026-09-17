"""Regression tests for the production-hardening pass.

Defects found by auditing the live system rather than the code:

  ingestion    SQLite committed a document's chunks BEFORE the FAISS vectors
               were written, so a failure in between left the document
               listed, keyword-searchable, invisible to dense retrieval, and
               permanently un-repairable: the content-hash dedup check then
               skipped every future re-upload of the same file.
  cleaning     control characters and compatibility Unicode reached the
               index, where they break FTS5 matching and word splitting.
"""

import numpy as np
import pymupdf
import pytest

import scripts.ingest as ingest_script
from src import config
from src.ingestion.cleaner import clean_text
from src.ingestion.pdf_extractor import extract_pages
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.metadata_store import MetadataStore


# --- ingestion integrity ------------------------------------------------------


def _make_pdf(path, text, pages=1):
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        for i in range(10):
            page.insert_text((40, 60 + i * 16), text[:80])
    doc.save(str(path))
    doc.close()


def _fake_embed_texts(texts):
    return np.ones((len(texts), config.EMBEDDING_DIM), dtype="float32")


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)
    return tmp_path


def test_failed_vector_write_leaves_no_half_indexed_document(isolated_index, monkeypatch):
    """The regression. A FAISS failure after the SQLite commit must undo the
    rows, not leave a document that looks indexed but cannot be retrieved."""
    pdf = isolated_index / "doc.pdf"
    _make_pdf(pdf, "Patient was seen in clinic for a routine review of medication.")

    def exploding_save(self):
        raise OSError("disk full")

    monkeypatch.setattr(FaissIndex, "save", exploding_save)

    with pytest.raises(OSError):
        ingest_script.ingest_pdf(str(pdf))

    store = MetadataStore(config.DB_PATH)
    assert store.count() == 0
    assert store.list_documents() == []


def test_document_can_be_reingested_after_a_failed_vector_write(isolated_index, monkeypatch):
    """The consequence that made the original bug unrecoverable: the
    content-hash dedup check saw chunk_count > 0 and skipped the retry.

    The failure is made to happen exactly once rather than undoing the
    patch afterwards - monkeypatch.undo() would also revert the fixture's
    config.DB_PATH redirect, sending the retry at the real production index.
    """
    pdf = isolated_index / "doc.pdf"
    _make_pdf(pdf, "Patient was seen in clinic for a routine review of medication.")

    real_save = FaissIndex.save
    calls = {"n": 0}

    def fail_first_save(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real_save(self)

    monkeypatch.setattr(FaissIndex, "save", fail_first_save)

    with pytest.raises(OSError):
        ingest_script.ingest_pdf(str(pdf))

    result = ingest_script.ingest_pdf(str(pdf))

    assert result["skipped"] is False
    assert result["chunk_count"] > 0
    store = MetadataStore(config.DB_PATH)
    index = FaissIndex(config.EMBEDDING_DIM, config.FAISS_INDEX_PATH)
    index.load()
    assert index.index.ntotal == store.count()


def test_successful_ingestion_reports_page_count(isolated_index):
    pdf = isolated_index / "doc.pdf"
    _make_pdf(pdf, "Routine clinic note about the patient's ongoing care.", pages=2)

    result = ingest_script.ingest_pdf(str(pdf))

    assert result["page_count"] == 2
    assert result["image_only_pages"] == []


# --- image-only page reporting ------------------------------------------------


def test_image_only_pages_are_reported_not_silently_dropped(tmp_path):
    """A scanned page yields no text and this pipeline has no OCR. It must
    be named, so a user who uploads a scan is told why it is not searchable
    instead of quietly getting answers from the other pages only."""
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(10):
        page.insert_text((40, 60 + i * 16), "Readable typed clinical text on page one.")
    image_page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 80))
    pix.set_rect(pix.irect, (210, 210, 210))
    image_page.insert_image(pymupdf.Rect(40, 40, 160, 120), pixmap=pix)
    pdf = tmp_path / "mixed.pdf"
    doc.save(str(pdf))
    doc.close()

    report: dict = {}
    pages = extract_pages(str(pdf), report=report)

    assert [p["page"] for p in pages] == [1]
    assert report["image_only_pages"] == [2]


def test_report_is_optional_and_extraction_is_unchanged_without_it(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(10):
        page.insert_text((40, 60 + i * 16), "Readable typed clinical text.")
    pdf = tmp_path / "plain.pdf"
    doc.save(str(pdf))
    doc.close()

    assert len(extract_pages(str(pdf))) == 1


# --- cleaning -----------------------------------------------------------------


def test_control_characters_are_stripped():
    assert clean_text("word\x02other\x03thing") == "wordotherthing"
    assert clean_text("keep\tthe\ttabs") == "keep the tabs"


def test_ligatures_and_smart_punctuation_are_normalised():
    """A search for "fibrillation" must match a PDF that renders it with an
    "fi" ligature, and a quoted phrase must match a curly-quoted source."""
    assert "fi" in clean_text("atrial ﬁbrillation")
    assert clean_text("the “standard” of care") == 'the "standard" of care'
    assert clean_text("non–responsive") == "non-responsive"


def test_zero_width_and_nbsp_characters_do_not_survive():
    assert clean_text("peri​umbilical") == "periumbilical"
    assert clean_text("8.3 cm") == "8.3 cm"


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_cleaning_does_not_disturb_ordinary_clinical_text():
    text = "The patient was discharged on warfarin, sotalol, and keflex.\nFollow-up in 12 weeks."
    assert clean_text(text) == text


# --- answer quality: internal prompt vocabulary must not reach the user -------


def test_no_evidence_placeholder_is_not_a_quotable_sentence():
    """It read "(No relevant evidence was retrieved for this document.)" -
    fluent English, and the most quotable line in the prompt whenever the
    question had no answer. Asked an unsupported question over three
    selected documents, the model returned exactly that sentence as its
    whole answer. A terse marker leaves nothing to copy."""
    from src.generation.prompt_builder import build_multi_document_context

    inventory = [{"document_id": "d1", "source": "a.pdf", "page_count": 1}]

    context = build_multi_document_context([], inventory)

    assert "EVIDENCE: none" in context
    assert "No relevant evidence was retrieved" not in context
    # Nothing in the block forms a sentence that could stand as an answer.
    assert "." not in context.split("EVIDENCE:")[1]


def test_multi_document_rules_forbid_echoing_the_placeholder():
    from src.generation.prompt_builder import build_prompt

    inventory = [
        {"document_id": "d1", "source": "a.pdf", "page_count": 1},
        {"document_id": "d2", "source": "b.pdf", "page_count": 1},
    ]
    prompt = build_prompt("What was the patient's blood type?", [], document_inventory=inventory)

    assert 'the phrase "no relevant evidence was retrieved"' in prompt
    assert "say so in your own words" in prompt


def test_cover_every_document_rule_applies_only_to_each_document_questions():
    """Applied to every multi-document question, "account for the documents
    with no evidence" turned an unsupported narrow question into a recital
    of the context's own placeholder."""
    from src.generation.prompt_builder import build_prompt

    inventory = [
        {"document_id": "d1", "source": "a.pdf", "page_count": 1},
        {"document_id": "d2", "source": "b.pdf", "page_count": 1},
    ]

    narrow = build_prompt("What was the patient's blood type?", [], document_inventory=inventory)
    each = build_prompt("Summarize each selected document.", [], document_inventory=inventory)

    assert "Cover every selected document" not in narrow
    assert "Cover every selected document" in each


def test_each_document_style_forbids_the_document_label_prefix():
    """The model began sections with "DOCUMENT: Pediatric Report.pdf",
    merging the context's heading format into its own answer."""
    from src.generation.prompt_builder import build_prompt

    inventory = [{"document_id": "d1", "source": "a.pdf", "page_count": 1}]
    prompt = build_prompt("Summarize each selected document.", [], document_inventory=inventory)

    assert 'never "DOCUMENT: Pediatric Report.pdf"' in prompt
    assert "belong to the material you are reading" in prompt


def test_upload_errors_name_the_users_file_not_a_temp_path(tmp_path, monkeypatch):
    """The UI showed the internal upload path from the exception text - an
    implementation detail the user never chose and cannot act on."""
    import app
    from src.exceptions import EmptyDocumentError

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))

    empty = pymupdf.open()
    empty.new_page()
    blank = tmp_path / "blank.pdf"
    empty.save(str(blank))
    empty.close()

    with pytest.raises(EmptyDocumentError) as excinfo:
        app.ingest_uploaded_pdf("My Clinic Report.pdf", blank.read_bytes())

    message = str(excinfo.value)
    assert "My Clinic Report.pdf" in message
    assert "rag_upload_" not in message
    assert "AppData" not in message
