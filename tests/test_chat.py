"""Verifies the interactive chat path invokes hybrid retrieval, not dense-only."""

import pytest

import chat
from src.exceptions import EmptyQueryError, InsufficientRetrievalError
from src.retrieval.hybrid_retriever import retrieve_hybrid


def test_chat_is_wired_to_hybrid_retriever_not_dense():
    assert chat.retrieve_hybrid is retrieve_hybrid
    assert not hasattr(chat, "retrieve")


def test_ask_calls_hybrid_retriever(monkeypatch, capsys):
    calls = []

    def fake_retrieve_hybrid(query, top_k=5):
        calls.append(query)
        return [
            {"chunk_id": 1, "document_id": "d1", "source": "doc.pdf", "page": 4,
             "text": "some text", "score": 0.7, "rank": 1},
        ]

    monkeypatch.setattr(chat, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(chat, "generate", lambda prompt: "a generated answer")

    answer = chat.ask("what is Y?", [])

    assert calls == ["what is Y?"]
    assert answer == "a generated answer"
    assert "doc.pdf" in capsys.readouterr().out


def test_ask_raises_on_empty_query():
    with pytest.raises(EmptyQueryError):
        chat.ask("   ", [])


def test_ask_raises_on_no_hybrid_results(monkeypatch):
    monkeypatch.setattr(chat, "retrieve_hybrid", lambda query, top_k=5: [])

    with pytest.raises(InsufficientRetrievalError):
        chat.ask("no matches expected", [])
