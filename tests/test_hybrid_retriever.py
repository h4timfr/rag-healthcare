"""Unit tests for hybrid_retriever.py's document_id scoping.

Uses crafted (not real-model) vectors so dense ranking is deterministic:
embed_query is monkeypatched, FAISS candidates are exact by construction.
"""

import numpy as np
import pytest

from src import config
from src.retrieval import hybrid_retriever
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.metadata_store import MetadataStore

DIM = 4


def _vec(main_weight, main_idx=0, secondary_idx=None, secondary_weight=0.0):
    v = np.zeros(DIM, dtype="float32")
    v[main_idx] = main_weight
    if secondary_idx is not None:
        v[secondary_idx] = secondary_weight
    return (v / np.linalg.norm(v)).astype("float32")


@pytest.fixture
def scoped_corpus(tmp_path, monkeypatch):
    """9 'docA' chunks strongly aligned with the query direction, 2 'docB'
    chunks weakly aligned - so an unfiltered top-10 dense search would leave
    one of docB's 2 chunks out entirely (11 total chunks, default candidate_k=10).
    Neither docB chunk shares keywords with the query, so docB can only be
    found via dense retrieval - proving the over-retrieval fix actually matters.
    """
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(config, "EMBEDDING_DIM", DIM)

    store = MetadataStore(config.DB_PATH)
    store.initialize()
    index = FaissIndex(DIM, config.FAISS_INDEX_PATH)
    index.load_or_create()

    doc_a_vector = _vec(0.99, secondary_idx=1, secondary_weight=0.14)  # cos ~0.99 to query
    ids, vectors = [], []
    for i in range(9):
        cid = store.insert_chunk(
            document_id="docA", source="a.pdf", page=i + 1,
            text=f"Diabetes management passage {i} about insulin therapy.",
        )
        ids.append(cid)
        vectors.append(doc_a_vector)

    doc_b_vectors = [
        _vec(0.3, secondary_idx=2, secondary_weight=0.95),  # cos ~0.3 to query
        _vec(0.1, secondary_idx=3, secondary_weight=0.99),  # cos ~0.1 to query
    ]
    doc_b_ids = []
    for i, vec in enumerate(doc_b_vectors):
        cid = store.insert_chunk(
            document_id="docB", source="b.pdf", page=i + 1,
            text=f"Cardiac catheterization report section {i} about coronary arteries.",
        )
        doc_b_ids.append(cid)
        ids.append(cid)
        vectors.append(vec)

    index.add(np.array(vectors, dtype="float32"), np.array(ids, dtype="int64"))
    index.save()

    query_vector = _vec(1.0)  # pure alignment with docA's direction
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: query_vector)

    return {"doc_b_ids": set(doc_b_ids)}


def test_global_retrieval_without_document_id_is_unrestricted(scoped_corpus):
    results = hybrid_retriever.retrieve_hybrid(
        "insulin therapy blood glucose", top_k=5
    )
    assert results
    # Unfiltered top-5 is dominated by the strongly-aligned docA chunks.
    assert all(r["document_id"] == "docA" for r in results)


def test_document_scoped_retrieval_returns_only_that_document(scoped_corpus):
    doc_b_ids = scoped_corpus["doc_b_ids"]

    results = hybrid_retriever.retrieve_hybrid(
        "insulin therapy blood glucose", top_k=5, document_id="docB"
    )

    assert results
    assert all(r["document_id"] == "docB" for r in results)
    assert {r["chunk_id"] for r in results} == doc_b_ids


def test_document_scoping_over_retrieves_so_weak_matches_are_not_dropped(scoped_corpus):
    """With the default candidate_k=10 and 11 total chunks, an unfiltered dense
    search would already cut off one of docB's two (weakly-ranked) chunks
    before filtering ever runs. The fix must still surface both.
    """
    results = hybrid_retriever.retrieve_hybrid(
        "insulin therapy blood glucose", top_k=5, document_id="docB", candidate_k=10
    )
    assert len(results) == 2


def test_document_scoping_on_document_with_no_matches_returns_empty(scoped_corpus):
    # docA text/vectors don't relate to a cardiology-only query and vice versa
    # is already covered above; here document_id filters out an existing but
    # entirely non-matching set is exercised by requesting an unknown id.
    results = hybrid_retriever.retrieve_hybrid(
        "insulin therapy blood glucose", top_k=5, document_id="does-not-exist"
    )
    assert results == []
