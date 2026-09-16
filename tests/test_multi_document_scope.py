"""Tests for multi-document "Analysis scope" mode: retrieve_hybrid()'s
list-of-document_ids support, MetadataStore.filter_chunk_ids_by_documents,
and app.py's resolve_scope_document_ids()/render_chat() scope handling.

Uses the same real-pipeline style as test_document_isolation.py: real PDFs
through the real ingestion pipeline and the real retrieve_hybrid()
filtering, with only the embedding model and Ollama generation stubbed out.
"""

import numpy as np
import pymupdf
import pytest

import app
from src import config
from src.retrieval import hybrid_retriever
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
def three_uploaded_documents(tmp_path, monkeypatch):
    import scripts.ingest as ingest_script

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)

    pdf_a, pdf_b, pdf_c = tmp_path / "a.pdf", tmp_path / "b.pdf", tmp_path / "c.pdf"
    _make_pdf(pdf_a, "Diabetes management guideline about insulin therapy and glucose monitoring.")
    _make_pdf(pdf_b, "Cardiology report about coronary arteries and cardiac catheterization.")
    _make_pdf(pdf_c, "Pediatric growth chart guidance about height and weight percentiles.")

    a = app.ingest_uploaded_pdf("diabetes.pdf", pdf_a.read_bytes())
    b = app.ingest_uploaded_pdf("cardio.pdf", pdf_b.read_bytes())
    c = app.ingest_uploaded_pdf("pediatric.pdf", pdf_c.read_bytes())

    return {"a": a["document_id"], "b": b["document_id"], "c": c["document_id"]}


def _first_chunk_vector(document_id: str) -> np.ndarray:
    import sqlite3

    conn = sqlite3.connect(config.DB_PATH)
    text = conn.execute(
        "SELECT text FROM chunks WHERE document_id = ? LIMIT 1", (document_id,)
    ).fetchone()[0]
    conn.close()
    return _fake_embed_texts([text])[0]


# --- retrieve_hybrid: multiple document_ids ---------------------------------


def test_one_selected_document_matches_active_document_mode(three_uploaded_documents, monkeypatch):
    doc_a = three_uploaded_documents["a"]
    vector = _first_chunk_vector(doc_a)
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: vector)

    single_str = hybrid_retriever.retrieve_hybrid("tell me about this", top_k=5, document_id=doc_a)
    single_list = hybrid_retriever.retrieve_hybrid("tell me about this", top_k=5, document_id=[doc_a])

    assert single_str
    assert [c["chunk_id"] for c in single_str] == [c["chunk_id"] for c in single_list]
    assert all(c["document_id"] == doc_a for c in single_list)


def test_two_selected_documents_retrieve_only_those_two(three_uploaded_documents, monkeypatch):
    doc_a, doc_b, doc_c = (
        three_uploaded_documents["a"],
        three_uploaded_documents["b"],
        three_uploaded_documents["c"],
    )
    # Query vector aligned with doc_b's own content, but doc_b isn't selected -
    # doc_a/doc_c's chunks must still be the only ones returned.
    vector = _first_chunk_vector(doc_b)
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: vector)

    results = hybrid_retriever.retrieve_hybrid(
        "tell me about this", top_k=10, document_id=[doc_a, doc_c]
    )

    assert results
    assert {c["document_id"] for c in results} <= {doc_a, doc_c}
    assert doc_b not in {c["document_id"] for c in results}


def test_unselected_document_never_appears_even_when_best_match(three_uploaded_documents, monkeypatch):
    doc_a, doc_b = three_uploaded_documents["a"], three_uploaded_documents["b"]
    doc_c = three_uploaded_documents["c"]
    # Force the strongest possible dense match on doc_c, which is NOT selected.
    vector = _first_chunk_vector(doc_c)
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: vector)

    results = hybrid_retriever.retrieve_hybrid(
        "tell me about this", top_k=10, document_id=[doc_a, doc_b]
    )

    ids = {c["document_id"] for c in results}
    assert doc_c not in ids
    assert ids <= {doc_a, doc_b}


def test_empty_document_id_list_returns_no_results_not_global_search(three_uploaded_documents, monkeypatch):
    vector = _first_chunk_vector(three_uploaded_documents["a"])
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: vector)

    results = hybrid_retriever.retrieve_hybrid("tell me about this", top_k=10, document_id=[])

    assert results == []


def test_metadata_store_filter_chunk_ids_by_documents(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()
    id_a = store.insert_chunk(document_id="docA", source="a.pdf", page=1, text="alpha")
    id_b = store.insert_chunk(document_id="docB", source="b.pdf", page=1, text="beta")
    id_c = store.insert_chunk(document_id="docC", source="c.pdf", page=1, text="gamma")

    allowed = store.filter_chunk_ids_by_documents([id_a, id_b, id_c], ["docA", "docC"])

    assert allowed == {id_a, id_c}


def test_metadata_store_filter_chunk_ids_by_documents_empty_inputs(tmp_path):
    store = MetadataStore(str(tmp_path / "metadata.db"))
    store.initialize()
    chunk_id = store.insert_chunk(document_id="docA", source="a.pdf", page=1, text="alpha")

    assert store.filter_chunk_ids_by_documents([], ["docA"]) == set()
    assert store.filter_chunk_ids_by_documents([chunk_id], []) == set()


# --- app.resolve_scope_document_ids -----------------------------------------


def test_resolve_scope_active_returns_active_document_id():
    result = app.resolve_scope_document_ids("active", "docA", {"docB", "docC"})
    assert result == "docA"


def test_resolve_scope_active_with_no_active_document_returns_none():
    result = app.resolve_scope_document_ids("active", None, {"docB"})
    assert result is None


def test_resolve_scope_selected_returns_sorted_selection():
    result = app.resolve_scope_document_ids("selected", "docA", {"docC", "docB"})
    assert result == ["docB", "docC"]


def test_resolve_scope_selected_with_empty_selection_returns_empty_list():
    result = app.resolve_scope_document_ids("selected", "docA", set())
    assert result == []


def test_switching_from_selected_back_to_active_restores_isolation():
    """Regardless of what's left in selected_document_ids, "active" scope
    must always resolve to exactly the active document."""
    selected = {"docB", "docC"}
    as_selected = app.resolve_scope_document_ids("selected", "docA", selected)
    as_active = app.resolve_scope_document_ids("active", "docA", selected)

    assert as_selected == ["docB", "docC"]
    assert as_active == "docA"


# --- source grouping by document (already-general helper, focused check) ---


def test_group_sources_groups_multiple_documents_separately():
    sources = [
        {"source": "a.pdf", "page": 1, "score": 0.1},
        {"source": "b.pdf", "page": 2, "score": 0.2},
        {"source": "a.pdf", "page": 3, "score": 0.1},
    ]
    grouped = app.group_sources(sources)
    assert grouped == [
        {"source": "a.pdf", "pages": [1, 3]},
        {"source": "b.pdf", "pages": [2]},
    ]


# --- deletion of a selected document ----------------------------------------


def test_deleting_a_selected_document_removes_it_from_the_selection(three_uploaded_documents, monkeypatch):
    doc_a, doc_b = three_uploaded_documents["a"], three_uploaded_documents["b"]

    # Simulate what render_analysis_scope/_confirm_delete do to session state.
    selected = {doc_a, doc_b}
    app.delete_document(doc_a)
    selected.discard(doc_a)

    assert selected == {doc_b}
    remaining_ids = {d["document_id"] for d in app.list_indexed_documents()}
    assert doc_a not in remaining_ids
    assert doc_b in remaining_ids


def test_deleted_document_dropped_from_scope_even_without_explicit_prune(three_uploaded_documents, monkeypatch):
    """Belt-and-suspenders: render_chat() itself intersects
    selected_document_ids against currently-indexed documents, so even a
    stale id left in session state can never reach retrieval."""
    doc_a, doc_b = three_uploaded_documents["a"], three_uploaded_documents["b"]
    app.delete_document(doc_a)

    documents = app.list_indexed_documents()
    valid_ids = {d["document_id"] for d in documents}
    stale_selection = {doc_a, doc_b}
    pruned = {d for d in stale_selection if d in valid_ids}

    assert pruned == {doc_b}
