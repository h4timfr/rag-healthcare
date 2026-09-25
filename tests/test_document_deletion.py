"""Integration tests for real document deletion: two documents are ingested
through the real pipeline, one is deleted, and the test verifies the other
document still retrieves correctly, the deleted document cannot be retrieved
at all, FAISS/SQLite stay consistent, active-document fallback works, and
re-uploading the deleted PDF's content afterward is treated as fresh (not a
duplicate of a document that no longer exists).
"""

import sqlite3

import numpy as np
import pymupdf
import pytest

import app
import scripts.ingest as ingest_script
from src import config
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.metadata_store import MetadataStore


def _make_pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _fake_embed_texts(texts):
    vectors = []
    for text in texts:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=config.EMBEDDING_DIM).astype("float32")
        vectors.append(v / np.linalg.norm(v))
    return np.array(vectors, dtype="float32")


@pytest.fixture
def two_uploaded_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)

    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    _make_pdf(pdf_a, "Diabetes management guideline about insulin therapy and glucose monitoring.")
    _make_pdf(pdf_b, "Cardiology report about coronary arteries and cardiac catheterization.")

    result_a = app.ingest_uploaded_pdf("diabetes.pdf", pdf_a.read_bytes())
    result_b = app.ingest_uploaded_pdf("cardio.pdf", pdf_b.read_bytes())

    return {"a": result_a, "b": result_b, "pdf_a_bytes": pdf_a.read_bytes()}


def test_delete_one_of_multiple_documents_leaves_other_intact(two_uploaded_documents):
    doc_a_id = two_uploaded_documents["a"]["document_id"]
    doc_b_id = two_uploaded_documents["b"]["document_id"]

    app.delete_document(doc_a_id)

    documents = app.list_indexed_documents()
    remaining_ids = {d["document_id"] for d in documents}
    assert remaining_ids == {doc_b_id}


def test_remaining_document_still_retrieves_correctly(two_uploaded_documents, monkeypatch):
    doc_a_id = two_uploaded_documents["a"]["document_id"]
    doc_b_id = two_uploaded_documents["b"]["document_id"]

    app.delete_document(doc_a_id)

    conn = sqlite3.connect(config.DB_PATH)
    b_chunk_id = conn.execute(
        "SELECT chunk_id FROM chunks WHERE document_id = ? LIMIT 1", (doc_b_id,)
    ).fetchone()[0]
    conn.close()

    store = MetadataStore(config.DB_PATH)
    b_chunk_text = store.get_chunks([b_chunk_id])[b_chunk_id]["text"]

    b_vector = _fake_embed_texts([b_chunk_text])[0]
    from src.retrieval import hybrid_retriever
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: b_vector)
    monkeypatch.setattr(app, "generate", lambda prompt, num_predict=None: "mock answer")

    result = app.answer_question("tell me about this", document_id=doc_b_id)

    assert result["chunks"]
    assert all(c["document_id"] == doc_b_id for c in result["chunks"])


def test_deleted_document_cannot_be_retrieved(two_uploaded_documents):
    from src.retrieval.hybrid_retriever import retrieve_hybrid

    doc_a_id = two_uploaded_documents["a"]["document_id"]
    app.delete_document(doc_a_id)

    results = retrieve_hybrid("insulin glucose diabetes", top_k=5, document_id=doc_a_id)
    assert results == []


def test_faiss_and_sqlite_stay_consistent_after_deletion(two_uploaded_documents):
    doc_a_id = two_uploaded_documents["a"]["document_id"]

    app.delete_document(doc_a_id)

    store = MetadataStore(config.DB_PATH)
    index = FaissIndex(config.EMBEDDING_DIM, config.FAISS_INDEX_PATH)
    index.load()

    assert index.index.ntotal == store.count()


def test_active_document_id_resolves_away_from_deleted_document(two_uploaded_documents):
    doc_a_id = two_uploaded_documents["a"]["document_id"]
    doc_b_id = two_uploaded_documents["b"]["document_id"]

    app.delete_document(doc_a_id)
    remaining = app.list_indexed_documents()

    # Mirrors app._confirm_delete: if the deleted doc was active, request None.
    resolved = app.resolve_active_document_id(remaining, None)
    assert resolved == doc_b_id


def test_uploading_deleted_document_again_is_treated_as_fresh(two_uploaded_documents):
    doc_a_id = two_uploaded_documents["a"]["document_id"]
    pdf_a_bytes = two_uploaded_documents["pdf_a_bytes"]

    app.delete_document(doc_a_id)

    result = app.ingest_uploaded_pdf("diabetes.pdf", pdf_a_bytes)

    assert result["skipped"] is False
    assert result["document_id"] == doc_a_id  # same content hash, now indexed fresh

    store = MetadataStore(config.DB_PATH)
    assert store.document_chunk_count(doc_a_id) == result["chunk_count"]
