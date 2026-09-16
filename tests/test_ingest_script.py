"""Verifies scripts/ingest.py's duplicate-document handling: re-ingesting the
same PDF content must not create duplicate FAISS vectors / SQLite rows.
"""

import hashlib
import sqlite3

import numpy as np
import pymupdf
import pytest

import scripts.ingest as ingest_script
from src import config
from src.retrieval.metadata_store import MetadataStore


def _make_pdf(path, text="Hello world, this is a test document about healthcare."):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _fake_embed_texts(texts):
    rng = np.random.default_rng(0)
    return rng.normal(size=(len(texts), config.EMBEDDING_DIM)).astype("float32")


def test_reingesting_identical_pdf_is_skipped_not_duplicated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)

    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path)

    first = ingest_script.ingest_pdf(str(pdf_path))
    assert first["skipped"] is False
    assert first["chunk_count"] >= 1

    from src.retrieval.metadata_store import MetadataStore
    from src.retrieval.faiss_index import FaissIndex

    store = MetadataStore(config.DB_PATH)
    assert store.count() == first["chunk_count"]

    index = FaissIndex(config.EMBEDDING_DIM, config.FAISS_INDEX_PATH)
    index.load()
    assert index.index.ntotal == first["chunk_count"]

    # Re-ingest the exact same file.
    second = ingest_script.ingest_pdf(str(pdf_path))
    assert second["skipped"] is True
    assert second["document_id"] == first["document_id"]
    assert second["chunk_count"] == first["chunk_count"]

    # No new rows/vectors were added.
    assert store.count() == first["chunk_count"]
    index.load()
    assert index.index.ntotal == first["chunk_count"]


def test_ingesting_different_content_adds_separate_document(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)

    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    _make_pdf(pdf_a, text="Document A content about diabetes management.")
    _make_pdf(pdf_b, text="Document B content about cardiac care protocols.")

    result_a = ingest_script.ingest_pdf(str(pdf_a))
    result_b = ingest_script.ingest_pdf(str(pdf_b))

    assert result_a["document_id"] != result_b["document_id"]
    assert result_a["skipped"] is False
    assert result_b["skipped"] is False

    store = MetadataStore(config.DB_PATH)
    assert store.count() == result_a["chunk_count"] + result_b["chunk_count"]


def test_failed_partial_ingestion_does_not_get_skipped_on_retry(tmp_path, monkeypatch):
    """Regression test for the ingestion transactionality bug: if SQLite
    insertion fails partway through a document, the next ingest_pdf() call
    for that same file must NOT see a partial document and skip it - it
    must find zero chunks and actually re-ingest from scratch.
    """
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)

    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, text="Healthcare document about diabetes and hypertension management guidelines.")
    document_id = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]

    real_insert_chunks = MetadataStore.insert_chunks
    call_count = {"n": 0}

    def flaky_insert_chunks(self, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate a failure partway through the SQLite write phase
            # (e.g. a crash, disk error) on the FIRST ingestion attempt only.
            raise sqlite3.OperationalError("simulated failure during ingestion")
        return real_insert_chunks(self, *args, **kwargs)

    monkeypatch.setattr(MetadataStore, "insert_chunks", flaky_insert_chunks)

    with pytest.raises(sqlite3.OperationalError):
        ingest_script.ingest_pdf(str(pdf_path))

    store = MetadataStore(config.DB_PATH)
    assert store.document_chunk_count(document_id) == 0

    # Retry (insert_chunks now behaves normally again) - must actually
    # re-ingest, not be skipped as "already indexed".
    result = ingest_script.ingest_pdf(str(pdf_path))

    assert result["skipped"] is False
    assert result["document_id"] == document_id
    assert result["chunk_count"] > 0
    assert store.document_chunk_count(document_id) == result["chunk_count"]
