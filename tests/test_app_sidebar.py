"""Headless Streamlit tests for the unified document library and scope UI.

The sidebar used to draw the document list TWICE: cards with a "Select"
button that set the ACTIVE document, and - under a separate "Analysis
scope" heading - a second list of checkboxes labelled "Selected documents".
One word meant two different things, and a card gave no hint whether it was
part of a multi-document selection.

There is now one list whose per-document control follows the scope. These
tests pin that contract, because it is the part of the UI a user has to
trust: it is the only thing telling them which documents an answer can
possibly have come from.
"""

from pathlib import Path

import numpy as np
import pymupdf
import pytest
from streamlit.testing.v1 import AppTest

import scripts.ingest as ingest_script
from src import config

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def _make_pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _fake_embed_texts(texts):
    rng = np.random.default_rng(0)
    return rng.normal(size=(len(texts), config.EMBEDDING_DIM)).astype("float32")


@pytest.fixture
def three_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)

    ids = []
    for name, text in (
        ("Pediatric Report.pdf", "Paediatric review of a chronic cough."),
        ("Cardio Report.pdf", "Cardiology review of an aortic aneurysm."),
        ("Method Review.pdf", "A proposed retrieval pipeline for documents."),
    ):
        path = tmp_path / name
        _make_pdf(path, text)
        ids.append(ingest_script.ingest_pdf(str(path))["document_id"])
    return ids


def _markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_library_is_rendered_once_not_twice(three_documents):
    """The regression: three documents must produce three cards, not three
    cards plus three checkboxes in a second list."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    body = _markdown(at)
    assert body.count('class="doc-card') == 3


def test_active_scope_offers_one_control_per_document(three_documents):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    assert at.session_state["analysis_scope"] == "active"
    # A button per document, and no selection checkboxes in this scope.
    assert not [c for c in at.checkbox if c.key and c.key.startswith("scope_select_")]
    assert [b for b in at.button if b.key and b.key.startswith("select_")]


def test_switching_scope_turns_the_same_list_into_checkboxes(three_documents):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    at.radio(key="analysis_scope").set_value("selected").run(timeout=30)

    assert not at.exception
    boxes = [c for c in at.checkbox if c.key and c.key.startswith("scope_select_")]
    assert len(boxes) == 3
    # Still one card per document - the list did not duplicate.
    assert _markdown(at).count('class="doc-card') == 3


def test_ticking_documents_builds_the_selection(three_documents):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.radio(key="analysis_scope").set_value("selected").run(timeout=30)

    first, second = three_documents[0], three_documents[1]
    at.checkbox(key=f"scope_select_{first}").set_value(True).run(timeout=30)
    at.checkbox(key=f"scope_select_{second}").set_value(True).run(timeout=30)

    assert not at.exception
    assert at.session_state["selected_document_ids"] == {first, second}


def test_unticking_removes_only_that_document(three_documents):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.radio(key="analysis_scope").set_value("selected").run(timeout=30)

    first, second = three_documents[0], three_documents[1]
    at.checkbox(key=f"scope_select_{first}").set_value(True).run(timeout=30)
    at.checkbox(key=f"scope_select_{second}").set_value(True).run(timeout=30)
    at.checkbox(key=f"scope_select_{first}").set_value(False).run(timeout=30)

    assert at.session_state["selected_document_ids"] == {second}


def test_scope_banner_names_every_document_in_scope(three_documents):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.radio(key="analysis_scope").set_value("selected").run(timeout=30)
    for doc_id in three_documents[:2]:
        at.checkbox(key=f"scope_select_{doc_id}").set_value(True).run(timeout=30)

    body = _markdown(at)
    assert "Analysing 2 documents" in body
    assert "Pediatric Report.pdf" in body
    assert "Cardio Report.pdf" in body


def test_scope_banner_is_grammatical_for_a_single_document(three_documents):
    """It read "Analysing 1 documents" in the running app."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.radio(key="analysis_scope").set_value("selected").run(timeout=30)
    at.checkbox(key=f"scope_select_{three_documents[0]}").set_value(True).run(timeout=30)

    body = _markdown(at)
    assert "Analysing 1 document" in body
    assert "1 documents" not in body


def test_active_scope_banner_names_the_active_document(three_documents):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    body = _markdown(at)
    assert "Analysing one document" in body
    assert 'class="scope-chip"' in body


def test_selected_scope_with_nothing_ticked_explains_what_to_do(three_documents):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.radio(key="analysis_scope").set_value("selected").run(timeout=30)

    body = _markdown(at)
    assert "Choose what to analyse" in body
    # Not the no-documents message - the library is not empty.
    assert "Start by adding a document" not in body


def test_in_scope_document_is_marked_on_its_card(three_documents):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.radio(key="analysis_scope").set_value("selected").run(timeout=30)
    at.checkbox(key=f"scope_select_{three_documents[0]}").set_value(True).run(timeout=30)

    body = _markdown(at)
    assert "in-scope" in body
    assert "in scope" in body  # the badge text


class _FakeUpload:
    def __init__(self, name, data):
        self.name, self._data, self.size = name, data, len(data)

    def getvalue(self):
        return self._data


def test_an_upload_appears_in_the_library_and_chat_in_the_same_run(tmp_path, monkeypatch):
    """The page read the document list before the sidebar ran the upload,
    so a freshly indexed document was missing from the library and the chat
    still said "Start by adding a document" until the user's next click."""
    import streamlit

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)
    pdf = tmp_path / "Fresh Upload.pdf"
    _make_pdf(pdf, "A freshly uploaded discharge summary.")
    monkeypatch.setattr(streamlit, "file_uploader", lambda *a, **k: [_FakeUpload(pdf.name, pdf.read_bytes())])

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    body = _markdown(at)
    assert body.count('class="doc-card') == 1
    assert "Fresh Upload.pdf" in body
    assert "Start by adding a document" not in body
