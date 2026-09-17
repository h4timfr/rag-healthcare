"""Tests for document-balanced retrieval allocation (Phase 3).

The property being protected: an each-document or comparison question over
N selected documents should not be answered from whichever documents
happened to win the global ranking. The budget is spread first, then filled
by score.

Equally important is what must NOT change - single-document retrieval,
ordinary multi-document questions, document scoping - so those are asserted
here too, against the real retrieval stack over a real index, not a mock.
"""

import numpy as np
import pymupdf
import pytest

import app
from src import config
from src.generation.intent import needs_document_balance
from src.retrieval import hybrid_retriever
from src.retrieval.hybrid_retriever import _allocate_by_document, retrieve_hybrid
from src.retrieval.metadata_store import MetadataStore


# --- allocation logic in isolation -------------------------------------------


def test_allocation_gives_every_document_a_share():
    ranked = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Document A dominates the global ranking.
    doc_of = {1: "A", 2: "A", 3: "A", 4: "A", 5: "A", 6: "B", 7: "B", 8: "C", 9: "C"}

    taken = _allocate_by_document(ranked, doc_of, document_count=3, top_k=6)

    assert len(taken) == 6
    per_doc = {d: sum(1 for c in taken if doc_of[c] == d) for d in "ABC"}
    assert per_doc == {"A": 2, "B": 2, "C": 2}


def test_allocation_keeps_each_documents_best_chunks():
    ranked = [1, 2, 3, 4]
    doc_of = {1: "A", 2: "A", 3: "B", 4: "B"}

    taken = _allocate_by_document(ranked, doc_of, document_count=2, top_k=2)

    # One each, and the better-ranked one from each document.
    assert set(taken) == {1, 3}


def test_allocation_never_invents_evidence_for_a_document_with_none():
    """A document that matched nothing has no chunks in the fused ranking,
    so it contributes nothing and its budget goes to the others. Balancing
    must not be able to pull an irrelevant document into the context."""
    ranked = [1, 2, 3, 4, 5, 6]
    doc_of = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B"}  # nothing for C

    taken = _allocate_by_document(ranked, doc_of, document_count=3, top_k=6)

    assert len(taken) == 6
    assert set(taken) == {1, 2, 3, 4, 5, 6}
    assert all(doc_of[c] in ("A", "B") for c in taken)


def test_allocation_fills_remaining_budget_by_global_score():
    ranked = [1, 2, 3, 4, 5]
    doc_of = {1: "A", 2: "A", 3: "A", 4: "B", 5: "A"}

    taken = _allocate_by_document(ranked, doc_of, document_count=2, top_k=4)

    assert len(taken) == 4
    # Quota of 2 each: A gets 1,2 and B gets 4. The last slot goes to the
    # next best overall, which is 3.
    assert set(taken) == {1, 2, 3, 4}


def test_allocation_is_deterministic():
    ranked = [5, 3, 9, 1, 7, 2]
    doc_of = {5: "A", 3: "B", 9: "A", 1: "C", 7: "B", 2: "C"}

    first = _allocate_by_document(ranked, doc_of, document_count=3, top_k=4)
    second = _allocate_by_document(ranked, doc_of, document_count=3, top_k=4)

    assert first == second


def test_allocation_never_exceeds_top_k_even_with_more_documents_than_slots():
    ranked = list(range(1, 9))
    doc_of = {i: f"D{i}" for i in ranked}

    taken = _allocate_by_document(ranked, doc_of, document_count=8, top_k=3)

    assert len(taken) == 3
    assert taken == [1, 2, 3]


# --- against the real retrieval stack ----------------------------------------


def _make_pdf(path, pages: list[str]):
    doc = pymupdf.open()
    for text in pages:
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
def three_documents(tmp_path, monkeypatch):
    """Three documents of deliberately unequal size: `big` has many pages of
    text sharing the query's vocabulary, so it dominates an unbalanced
    ranking - which is exactly the situation balancing exists for."""
    import scripts.ingest as ingest_script

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)

    big = tmp_path / "big.pdf"
    small_a = tmp_path / "small_a.pdf"
    small_b = tmp_path / "small_b.pdf"

    _make_pdf(big, [f"patient treatment plan page {i} with follow up care notes" for i in range(1, 9)])
    _make_pdf(small_a, ["cardiology report on coronary arteries"])
    _make_pdf(small_b, ["pediatric growth chart percentiles"])

    ids = {
        "big": app.ingest_uploaded_pdf("big.pdf", big.read_bytes())["document_id"],
        "small_a": app.ingest_uploaded_pdf("small_a.pdf", small_a.read_bytes())["document_id"],
        "small_b": app.ingest_uploaded_pdf("small_b.pdf", small_b.read_bytes())["document_id"],
    }
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: _fake_embed_texts([q])[0])
    return ids


def _documents_covered(chunks) -> set:
    return {c["document_id"] for c in chunks}


def test_balanced_retrieval_covers_every_selected_document(three_documents):
    selected = sorted(three_documents.values())

    balanced = retrieve_hybrid(
        "summarize each selected document separately",
        top_k=config.TOP_K * 3,
        document_id=selected,
        balance_by_document=True,
    )

    assert _documents_covered(balanced) == set(selected)


def test_balancing_improves_or_matches_coverage_versus_unbalanced(three_documents):
    selected = sorted(three_documents.values())
    query = "patient treatment plan follow up care"

    unbalanced = retrieve_hybrid(query, top_k=6, document_id=selected)
    balanced = retrieve_hybrid(query, top_k=6, document_id=selected, balance_by_document=True)

    assert len(_documents_covered(balanced)) >= len(_documents_covered(unbalanced))
    # Same budget - balancing redistributes, it does not enlarge.
    assert len(balanced) == len(unbalanced)


def test_balanced_results_stay_sorted_by_fused_score(three_documents):
    selected = sorted(three_documents.values())

    balanced = retrieve_hybrid(
        "compare the documents", top_k=6, document_id=selected, balance_by_document=True
    )

    scores = [c["score"] for c in balanced]
    assert scores == sorted(scores, reverse=True)
    assert [c["rank"] for c in balanced] == list(range(1, len(balanced) + 1))


def test_balancing_never_leaks_an_unselected_document(three_documents):
    selected = sorted([three_documents["small_a"], three_documents["small_b"]])

    balanced = retrieve_hybrid(
        "summarize each document",
        top_k=config.TOP_K * 2,
        document_id=selected,
        balance_by_document=True,
    )

    assert _documents_covered(balanced) <= set(selected)
    assert three_documents["big"] not in _documents_covered(balanced)


def test_single_document_retrieval_is_byte_identical_with_the_flag(three_documents):
    """Balancing is a no-op for a single document: there is nothing to
    balance against, and single-document behaviour must not move."""
    doc = three_documents["big"]

    plain = retrieve_hybrid("treatment plan", top_k=5, document_id=doc)
    flagged = retrieve_hybrid("treatment plan", top_k=5, document_id=doc, balance_by_document=True)

    assert [c["chunk_id"] for c in plain] == [c["chunk_id"] for c in flagged]
    assert [c["score"] for c in plain] == [c["score"] for c in flagged]


def test_unbalanced_multi_document_retrieval_is_unchanged_by_default(three_documents):
    """The default path must be exactly what it was before balancing
    existed - the flag is opt-in per question."""
    selected = sorted(three_documents.values())

    a = retrieve_hybrid("coronary arteries", top_k=5, document_id=selected)
    b = retrieve_hybrid("coronary arteries", top_k=5, document_id=selected, balance_by_document=False)

    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]


def test_empty_selection_still_returns_nothing_when_balancing(three_documents):
    assert retrieve_hybrid("anything", document_id=[], balance_by_document=True) == []


def test_document_ids_for_chunks_maps_every_chunk(three_documents):
    store = MetadataStore(config.DB_PATH)
    chunks = retrieve_hybrid("treatment plan", top_k=5, document_id=three_documents["big"])
    chunk_ids = [c["chunk_id"] for c in chunks]

    mapping = store.document_ids_for_chunks(chunk_ids)

    assert set(mapping) == set(chunk_ids)
    assert set(mapping.values()) == {three_documents["big"]}


def test_document_ids_for_chunks_handles_empty_input():
    assert MetadataStore(config.DB_PATH).document_ids_for_chunks([]) == {}


# --- wiring: app.py only balances when the question asks for it ---------------


def _capture_retrieve(monkeypatch):
    captured = {}

    def fake_retrieve_hybrid(query, **kwargs):
        captured.update(kwargs)
        captured["query"] = query
        return [
            {
                "chunk_id": 1, "document_id": "d1", "source": "a.pdf", "page": 1,
                "text": "t", "score": 0.1, "rank": 1,
            }
        ]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)
    return captured


INVENTORY = [
    {"document_id": "d1", "source": "a.pdf", "page_count": 1},
    {"document_id": "d2", "source": "b.pdf", "page_count": 1},
]


def test_each_document_question_requests_balancing(monkeypatch):
    captured = _capture_retrieve(monkeypatch)

    app.retrieve_and_build_prompt(
        "Summarize each selected document separately.",
        document_id=["d1", "d2"],
        document_inventory=INVENTORY,
    )

    assert captured.get("balance_by_document") is True


def test_comparison_question_requests_balancing(monkeypatch):
    captured = _capture_retrieve(monkeypatch)

    app.retrieve_and_build_prompt(
        "Compare the selected documents.",
        document_id=["d1", "d2"],
        document_inventory=INVENTORY,
    )

    assert captured.get("balance_by_document") is True


def test_narrow_question_in_multi_document_scope_does_not_balance(monkeypatch):
    """A narrow question asked while several documents are selected should
    still get the best evidence overall, not an artificially spread one."""
    captured = _capture_retrieve(monkeypatch)

    app.retrieve_and_build_prompt(
        "What medication was prescribed?",
        document_id=["d1", "d2"],
        document_inventory=INVENTORY,
    )

    assert "balance_by_document" not in captured


def test_single_document_scope_never_requests_balancing(monkeypatch):
    """Active-document mode passes no inventory, so the kwarg is never even
    sent - which is also what keeps narrow test doubles of retrieve_hybrid
    working."""
    captured = _capture_retrieve(monkeypatch)

    app.retrieve_and_build_prompt(
        "Summarize each selected document separately.", document_id="d1"
    )

    assert "balance_by_document" not in captured


def test_intent_and_app_wiring_agree():
    assert needs_document_balance("Summarize each selected document separately.") is True
    assert needs_document_balance("What medication was prescribed?") is False
