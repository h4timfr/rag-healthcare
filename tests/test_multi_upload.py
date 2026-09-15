"""Tests for app.py's multi-file upload batch helper and active-document
selection logic (no Streamlit runtime, no live Ollama) - see test_app_helpers.py
for the single-file ingest_uploaded_pdf() tests these build on.
"""

import pytest

import app


def test_ingest_uploaded_pdfs_processes_each_file_independently(monkeypatch):
    calls = []

    def fake_ingest_uploaded_pdf(filename, file_bytes):
        calls.append(filename)
        return {"document_id": f"id-{filename}", "chunk_count": 5, "skipped": False}

    monkeypatch.setattr(app, "ingest_uploaded_pdf", fake_ingest_uploaded_pdf)

    results = app.ingest_uploaded_pdfs(
        [("a.pdf", b"aaa"), ("b.pdf", b"bbb"), ("c.pdf", b"ccc")]
    )

    assert calls == ["a.pdf", "b.pdf", "c.pdf"]
    assert [r["filename"] for r in results] == ["a.pdf", "b.pdf", "c.pdf"]
    assert all(r["status"] == "indexed" for r in results)
    assert [r["document_id"] for r in results] == ["id-a.pdf", "id-b.pdf", "id-c.pdf"]


def test_ingest_uploaded_pdfs_marks_skipped_documents(monkeypatch):
    monkeypatch.setattr(
        app, "ingest_uploaded_pdf",
        lambda filename, file_bytes: {"document_id": "dup", "chunk_count": 4, "skipped": True},
    )
    results = app.ingest_uploaded_pdfs([("a.pdf", b"aaa")])
    assert results[0]["status"] == "skipped"
    assert results[0]["chunk_count"] == 4


def test_ingest_uploaded_pdfs_one_failure_does_not_abort_the_batch(monkeypatch):
    def fake_ingest(filename, file_bytes):
        if filename == "bad.pdf":
            raise app.RagError("could not extract text")
        return {"document_id": f"id-{filename}", "chunk_count": 2, "skipped": False}

    monkeypatch.setattr(app, "ingest_uploaded_pdf", fake_ingest)

    results = app.ingest_uploaded_pdfs(
        [("good1.pdf", b"1"), ("bad.pdf", b"2"), ("good2.pdf", b"3")]
    )

    assert [r["status"] for r in results] == ["indexed", "failed", "indexed"]
    assert results[1]["error"] == "could not extract text"
    assert results[1]["document_id"] is None
    # the good files either side of the bad one must still have succeeded
    assert results[0]["document_id"] == "id-good1.pdf"
    assert results[2]["document_id"] == "id-good2.pdf"


def test_ingest_uploaded_pdfs_survives_unexpected_non_rag_exception(monkeypatch):
    """A totally unexpected error (not a RagError) in one file must still be
    recorded as 'failed' rather than propagating and killing the whole batch."""

    def fake_ingest(filename, file_bytes):
        if filename == "corrupt.pdf":
            raise ValueError("unexpected internal error")
        return {"document_id": "ok", "chunk_count": 1, "skipped": False}

    monkeypatch.setattr(app, "ingest_uploaded_pdf", fake_ingest)

    results = app.ingest_uploaded_pdfs([("corrupt.pdf", b"x"), ("fine.pdf", b"y")])

    assert results[0]["status"] == "failed"
    assert results[0]["error"] == "unexpected internal error"
    assert results[1]["status"] == "indexed"


def test_ingest_uploaded_pdfs_duplicate_within_same_batch_is_skipped_not_duplicated(tmp_path, monkeypatch):
    """Real (unmocked) ingestion: uploading the identical bytes twice in one
    batch must index once and skip the second - proving the existing
    content-hash dedup in ingest_pdf() is checked fresh per file, in order,
    so within-batch duplicates are handled exactly like cross-session ones.
    """
    import numpy as np
    import pymupdf

    import scripts.ingest as ingest_script
    from src import config
    from src.retrieval.metadata_store import MetadataStore

    def fake_embed_texts(texts):
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(texts), config.EMBEDDING_DIM)).astype("float32")

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", fake_embed_texts)

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Duplicate content uploaded twice in one batch.")
    pdf_bytes_path = tmp_path / "source.pdf"
    doc.save(str(pdf_bytes_path))
    doc.close()
    pdf_bytes = pdf_bytes_path.read_bytes()

    results = app.ingest_uploaded_pdfs([("copy1.pdf", pdf_bytes), ("copy2.pdf", pdf_bytes)])

    assert results[0]["status"] == "indexed"
    assert results[1]["status"] == "skipped"
    assert results[0]["document_id"] == results[1]["document_id"]

    store = MetadataStore(config.DB_PATH)
    assert store.count() == results[0]["chunk_count"]  # not doubled


# --- resolve_active_document_id --------------------------------------------

def test_resolve_active_document_id_keeps_valid_requested_id():
    documents = [{"document_id": "a"}, {"document_id": "b"}]
    assert app.resolve_active_document_id(documents, "a") == "a"


def test_resolve_active_document_id_falls_back_to_most_recent_when_requested_is_gone():
    documents = [{"document_id": "a"}, {"document_id": "b"}]
    assert app.resolve_active_document_id(documents, "deleted-or-never-existed") == "b"


def test_resolve_active_document_id_falls_back_when_none_requested():
    documents = [{"document_id": "a"}, {"document_id": "b"}]
    assert app.resolve_active_document_id(documents, None) == "b"


def test_resolve_active_document_id_returns_none_when_no_documents():
    assert app.resolve_active_document_id([], "anything") is None


# --- friendly_error_message --------------------------------------------------

def test_friendly_error_message_passes_through_rag_error_text():
    assert app.friendly_error_message(app.EmptyQueryError("Query must not be empty")) == "Query must not be empty"


def test_friendly_error_message_adds_hint_for_ollama_unavailable():
    msg = app.friendly_error_message(app.OllamaUnavailableError("connection refused"))
    assert "Ollama" in msg
    assert "connection refused" in msg


def test_friendly_error_message_generic_fallback_for_unexpected_exception():
    msg = app.friendly_error_message(RuntimeError("some internal bug"))
    assert "some internal bug" not in msg  # never leak raw internals for unexpected errors
    assert "unexpected" in msg.lower()
