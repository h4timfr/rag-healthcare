"""Headless Streamlit smoke tests using AppTest: verifies the redesigned
app.py actually boots and renders without exceptions, in both the empty
(no documents) state and with a real ingested document, and that the theme
toggle changes session state as expected.
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
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)
    return tmp_path


def _all_markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_app_renders_empty_state_without_exception(isolated_storage):
    """With nothing indexed, the first screen must say what to do rather
    than sit blank - and must not claim a document is merely "not selected"
    when the library is empty."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    assert not at.exception
    assert any("RAG for Healthcare Documents" in t.value for t in at.title)

    body = _all_markdown(at)
    assert "Start by adding a document" in body
    assert "Add documents" in body
    # The other empty state - documents exist but none in scope - must not
    # be the one shown here.
    assert "Choose what to analyse" not in body


def test_app_renders_with_one_indexed_document(isolated_storage):
    pdf_path = isolated_storage / "doc.pdf"
    _make_pdf(pdf_path, "Sample healthcare content about patient monitoring.")
    ingest_script.ingest_pdf(str(pdf_path))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    assert not at.exception


def test_theme_defaults_to_dark_and_toggle_switches_to_light(isolated_storage):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    assert at.session_state["theme"] == "dark"

    at.toggle(key="theme_toggle").set_value(False).run(timeout=30)

    assert not at.exception
    assert at.session_state["theme"] == "light"
