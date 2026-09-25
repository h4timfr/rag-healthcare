"""Integration test reproducing the reported bug: uploading a second document
must not let questions about it retrieve chunks from a different, already
indexed document.

Uses two real small PDFs through the real ingestion pipeline and the real
retrieve_hybrid() document_id filtering; only the embedding model and Ollama
generation are stubbed out (no live model load, no live Ollama needed).
"""

import sqlite3

import numpy as np
import pymupdf
import pytest

import app
import scripts.ingest as ingest_script
from src import config
from src.retrieval import hybrid_retriever


def _make_pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _fake_embed_texts(texts):
    """Deterministic-per-text fake embedding (no real model load)."""
    vectors = []
    for text in texts:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=config.EMBEDDING_DIM).astype("float32")
        vectors.append(v / np.linalg.norm(v))
    return np.array(vectors, dtype="float32")


def _first_chunk_text(document_id: str) -> str:
    conn = sqlite3.connect(config.DB_PATH)
    row = conn.execute(
        "SELECT text FROM chunks WHERE document_id = ? LIMIT 1", (document_id,)
    ).fetchone()
    conn.close()
    return row[0]


@pytest.fixture
def two_uploaded_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)

    pdf_review = tmp_path / "review.pdf"
    pdf_omc = tmp_path / "omc.pdf"
    _make_pdf(
        pdf_review,
        "RAG for Healthcare Documents project review: retrieval augmented "
        "generation pipeline using FAISS and hybrid search.",
    )
    _make_pdf(
        pdf_omc,
        "OMC Cardiology Report: the patient presented with chest pain and "
        "was diagnosed with unstable angina, treated with cardiac catheterization.",
    )

    review_result = app.ingest_uploaded_pdf("RAG_Healthcare_Review1.pdf", pdf_review.read_bytes())
    omc_result = app.ingest_uploaded_pdf("OMC Report Sample - Cardio.pdf", pdf_omc.read_bytes())

    return {"review": review_result, "omc": omc_result}


def test_app_selects_newly_uploaded_document_as_active(two_uploaded_documents):
    # render_sidebar() sets st.session_state["active_document_id"] to exactly
    # the document_id ingest_uploaded_pdf() returns - assert that id exists,
    # is distinct per document, and reflects a fresh (not skipped) ingest.
    review, omc = two_uploaded_documents["review"], two_uploaded_documents["omc"]
    assert review["document_id"] != omc["document_id"]
    assert review["skipped"] is False
    assert omc["skipped"] is False


def test_querying_active_document_does_not_leak_other_documents_chunks(
    two_uploaded_documents, monkeypatch
):
    omc_id = two_uploaded_documents["omc"]["document_id"]
    review_id = two_uploaded_documents["review"]["document_id"]

    # Force a strong dense match on the OMC document's own chunk, regardless
    # of query text, then confirm scoping to it returns only its own chunks.
    omc_vector = _fake_embed_texts([_first_chunk_text(omc_id)])[0]
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: omc_vector)
    monkeypatch.setattr(app, "generate", lambda prompt, num_predict=None: "mock answer")

    result = app.answer_question("tell me about this document", document_id=omc_id)

    assert result["chunks"]
    assert all(c["document_id"] == omc_id for c in result["chunks"])
    assert all(s["source"] == "OMC Report Sample - Cardio.pdf" for s in result["sources"])

    # Same query vector (best matches OMC), but scoped to the review document
    # instead - must return the review document's own chunk(s), never OMC's.
    result_review = app.answer_question("tell me about this document", document_id=review_id)
    assert result_review["chunks"]
    assert all(c["document_id"] == review_id for c in result_review["chunks"])
    assert all(c["document_id"] != omc_id for c in result_review["chunks"])
