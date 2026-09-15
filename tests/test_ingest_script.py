"""Verifies scripts/ingest.py's duplicate-document handling: re-ingesting the
same PDF content must not create duplicate FAISS vectors / SQLite rows.
"""

import numpy as np
import pymupdf

import scripts.ingest as ingest_script
from src import config


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

    from src.retrieval.metadata_store import MetadataStore

    store = MetadataStore(config.DB_PATH)
    assert store.count() == result_a["chunk_count"] + result_b["chunk_count"]
