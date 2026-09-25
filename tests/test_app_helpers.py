"""Tests for app.py's UI-facing integration helpers (no Streamlit runtime, no live Ollama).

Importing app.py must not launch the page - main() is only invoked under
`streamlit run`, guarded by `if __name__ == "__main__"`, so plain `import app`
only defines the helper functions exercised below.
"""

import tempfile
from pathlib import Path

import pytest

import app
from src.exceptions import DatabaseNotFoundError, EmptyQueryError, InsufficientRetrievalError


def test_ingest_uploaded_pdf_writes_original_filename_and_cleans_up(monkeypatch):
    captured = {}

    def fake_ingest_pdf(path):
        p = Path(path)
        captured["path"] = p
        captured["name"] = p.name
        captured["bytes"] = p.read_bytes()
        captured["parent_exists_during_call"] = p.parent.exists()
        return {"document_id": "abc123", "chunk_count": 3, "skipped": False}

    monkeypatch.setattr(app.ingest_module, "ingest_pdf", fake_ingest_pdf)

    result = app.ingest_uploaded_pdf("My Report.pdf", b"%PDF-1.4 fake bytes")

    assert result == {"document_id": "abc123", "chunk_count": 3, "skipped": False}
    assert captured["name"] == "My Report.pdf"
    assert captured["bytes"] == b"%PDF-1.4 fake bytes"
    assert captured["parent_exists_during_call"] is True
    # temp dir must be removed after ingestion, whether it succeeded or not
    assert not captured["path"].parent.exists()


def test_ingest_uploaded_pdf_sanitizes_path_traversal_and_missing_extension(monkeypatch):
    captured = {}

    def fake_ingest_pdf(path):
        captured["name"] = Path(path).name
        return {"document_id": "x", "chunk_count": 1, "skipped": False}

    monkeypatch.setattr(app.ingest_module, "ingest_pdf", fake_ingest_pdf)

    app.ingest_uploaded_pdf("../../etc/passwd", b"data")

    assert captured["name"] == "passwd.pdf"
    assert "/" not in captured["name"] and "\\" not in captured["name"]


def test_ingest_uploaded_pdf_cleans_up_temp_dir_even_on_failure(monkeypatch):
    tmp_dirs = []

    def failing_ingest_pdf(path):
        tmp_dirs.append(Path(path).parent)
        raise RuntimeError("boom")

    monkeypatch.setattr(app.ingest_module, "ingest_pdf", failing_ingest_pdf)

    with pytest.raises(RuntimeError):
        app.ingest_uploaded_pdf("doc.pdf", b"data")

    assert not tmp_dirs[0].exists()


def test_list_indexed_documents_returns_empty_list_when_no_database(monkeypatch):
    class FakeStore:
        def __init__(self, path):
            pass

        def list_documents(self):
            raise DatabaseNotFoundError("no db yet")

    monkeypatch.setattr(app, "MetadataStore", FakeStore)
    assert app.list_indexed_documents() == []


def test_list_indexed_documents_returns_store_rows(monkeypatch):
    rows = [{"document_id": "d1", "source": "a.pdf", "chunk_count": 4}]

    class FakeStore:
        def __init__(self, path):
            pass

        def list_documents(self):
            return rows

    monkeypatch.setattr(app, "MetadataStore", FakeStore)
    assert app.list_indexed_documents() == rows


def test_answer_question_calls_hybrid_retrieval_and_returns_chunks(monkeypatch):
    calls = []

    def fake_retrieve_hybrid(query, top_k=5, document_id=None):
        calls.append((query, top_k, document_id))
        return [
            {"chunk_id": 1, "document_id": "d1", "source": "doc.pdf", "page": 3,
             "text": "evidence text", "score": 0.05, "rank": 1},
        ]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(app, "generate", lambda prompt, num_predict=None: "a generated answer")

    result = app.answer_question("what is X?", top_k=3)

    assert calls == [("what is X?", 3, None)]
    assert result["answer"] == "a generated answer"
    assert result["sources"] == [{"source": "doc.pdf", "page": 3, "score": 0.05}]
    assert result["chunks"][0]["text"] == "evidence text"


def test_answer_question_passes_document_id_through_to_hybrid_retrieval(monkeypatch):
    calls = []

    def fake_retrieve_hybrid(query, top_k=5, document_id=None):
        calls.append(document_id)
        return [
            {"chunk_id": 1, "document_id": document_id, "source": "doc.pdf", "page": 1,
             "text": "t", "score": 0.1, "rank": 1},
        ]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(app, "generate", lambda prompt, num_predict=None: "answer")

    app.answer_question("what is X?", document_id="docA")

    assert calls == ["docA"]


def test_answer_question_normalizes_empty_query_error():
    with pytest.raises(EmptyQueryError):
        app.answer_question("   ")


def test_answer_question_raises_on_no_hybrid_results(monkeypatch):
    monkeypatch.setattr(app, "retrieve_hybrid", lambda query, top_k=5, document_id=None: [])

    with pytest.raises(InsufficientRetrievalError):
        app.answer_question("no matches expected")


# --- group_sources -------------------------------------------------------


@pytest.mark.parametrize("data", [b"", b"%PDF-1.4 not really a pdf"])
def test_unopenable_upload_error_names_the_upload_not_the_temp_path(tmp_path, monkeypatch, data):
    """PyMuPDF quotes the file path in escaped form (doubled backslashes);
    the user must see the name they uploaded, never the temp directory."""
    monkeypatch.setattr(app.config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(app.config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))

    result = app.ingest_uploaded_pdfs([("report.pdf", data)])[0]

    assert result["status"] == "failed"
    assert "report.pdf" in result["error"]
    assert "rag_upload_" not in result["error"]
    assert tempfile.gettempdir().lower() not in result["error"].lower().replace("\\\\", "\\")


def test_completion_status_for_a_complete_answer_is_a_quiet_caption():
    assert app.completion_status(1500, truncated=False) == ("caption", "Complete · 1.5s")


def test_completion_status_for_a_cut_off_answer_warns_instead_of_saying_complete():
    kind, text = app.completion_status(1500, truncated=True)
    assert kind == "warning"
    assert "cut off" in text
    assert "Complete" not in text


def test_group_sources_dedupes_repeated_source_page_pairs():
    sources = [
        {"source": "file.pdf", "page": 1, "score": 0.1},
        {"source": "file.pdf", "page": 1, "score": 0.05},
        {"source": "file.pdf", "page": 3, "score": 0.2},
    ]
    assert app.group_sources(sources) == [{"source": "file.pdf", "pages": [1, 3]}]


def test_group_sources_groups_by_document_preserving_first_seen_order():
    sources = [
        {"source": "b.pdf", "page": 2, "score": 0.1},
        {"source": "a.pdf", "page": 1, "score": 0.1},
        {"source": "b.pdf", "page": 1, "score": 0.1},
    ]
    result = app.group_sources(sources)
    assert [g["source"] for g in result] == ["b.pdf", "a.pdf"]
    assert result[0]["pages"] == [1, 2]
    assert result[1]["pages"] == [1]


def test_group_sources_empty_list_returns_empty_list():
    assert app.group_sources([]) == []


# --- delete_document -------------------------------------------------------


def test_delete_document_removes_metadata_and_faiss_vectors(tmp_path, monkeypatch):
    import numpy as np

    from src import config
    from src.retrieval.faiss_index import FaissIndex
    from src.retrieval.metadata_store import MetadataStore

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))

    store = MetadataStore(config.DB_PATH)
    store.initialize()
    index = FaissIndex(config.EMBEDDING_DIM, config.FAISS_INDEX_PATH)
    index.load_or_create()

    chunk_id = store.insert_chunk(document_id="docA", source="a.pdf", page=1, text="hello")
    vec = np.random.default_rng(0).normal(size=(1, config.EMBEDDING_DIM)).astype("float32")
    index.add(vec, np.array([chunk_id], dtype="int64"))
    index.save()

    result = app.delete_document("docA")

    assert result == {"removed_chunks": 1}
    assert store.count() == 0

    reloaded = FaissIndex(config.EMBEDDING_DIM, config.FAISS_INDEX_PATH)
    reloaded.load()
    assert reloaded.index.ntotal == 0


def test_delete_document_with_no_chunks_does_not_touch_faiss(tmp_path, monkeypatch):
    from src import config
    from src.retrieval.metadata_store import MetadataStore

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))

    store = MetadataStore(config.DB_PATH)
    store.initialize()

    # No FAISS index file exists yet - delete_document must not try to load one.
    result = app.delete_document("does-not-exist")

    assert result == {"removed_chunks": 0}
    assert not (tmp_path / "faiss.index").exists()
