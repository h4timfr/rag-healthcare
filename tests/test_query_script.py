"""Verifies the user-facing CLI query path invokes hybrid retrieval, not dense-only."""

import pytest

import scripts.query as query_script
from src.exceptions import EmptyQueryError, InsufficientRetrievalError
from src.retrieval.hybrid_retriever import retrieve_hybrid


def test_query_script_is_wired_to_hybrid_retriever_not_dense():
    # scripts.query must call the hybrid retriever, and must not import the
    # dense-only retrieve() at all (that stays reserved for evaluate.py).
    assert query_script.retrieve_hybrid is retrieve_hybrid
    assert not hasattr(query_script, "retrieve")


def test_answer_query_calls_hybrid_retriever(monkeypatch):
    calls = []

    def fake_retrieve_hybrid(query, top_k=5):
        calls.append((query, top_k))
        return [
            {"chunk_id": 1, "document_id": "d1", "source": "doc.pdf", "page": 2,
             "text": "some text", "score": 0.9, "rank": 1},
        ]

    monkeypatch.setattr(query_script, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(query_script, "generate", lambda prompt: "a generated answer")

    answer, sources = query_script.answer_query("what is X?", top_k=3)

    assert calls == [("what is X?", 3)]
    assert answer == "a generated answer"
    assert sources == [{"source": "doc.pdf", "page": 2, "score": 0.9}]


def test_answer_query_normalizes_empty_query_error():
    with pytest.raises(EmptyQueryError):
        query_script.answer_query("   ")


def test_answer_query_raises_on_no_hybrid_results(monkeypatch):
    monkeypatch.setattr(query_script, "retrieve_hybrid", lambda query, top_k=5: [])

    with pytest.raises(InsufficientRetrievalError):
        query_script.answer_query("no matches expected")
