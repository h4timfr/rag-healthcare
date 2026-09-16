"""Tests for the timing instrumentation added to retrieve_hybrid() and
app.retrieve_and_build_prompt(), plus app._stream_with_ttft()'s
time-to-first-token capture. Retrieval behavior itself is unchanged and
already covered by test_hybrid_retriever.py/test_document_isolation.py -
these focus only on the new opt-in `timing` out-param.
"""

import numpy as np
import pytest

import app
from src import config
from src.retrieval import hybrid_retriever
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.metadata_store import MetadataStore

DIM = 4


@pytest.fixture
def small_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(config, "EMBEDDING_DIM", DIM)

    store = MetadataStore(config.DB_PATH)
    store.initialize()
    index = FaissIndex(DIM, config.FAISS_INDEX_PATH)
    index.load_or_create()

    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    chunk_id = store.insert_chunk(document_id="d1", source="a.pdf", page=1, text="diabetes insulin therapy")
    index.add(np.array([vec]), np.array([chunk_id], dtype="int64"))
    index.save()

    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: vec)


def test_retrieve_hybrid_timing_is_none_by_default(small_corpus):
    """Omitting `timing` must not error or change the return value shape."""
    results = hybrid_retriever.retrieve_hybrid("insulin therapy", top_k=5)
    assert results


def test_retrieve_hybrid_populates_timing_dict_with_expected_stages(small_corpus):
    timing: dict = {}

    hybrid_retriever.retrieve_hybrid("insulin therapy", top_k=5, timing=timing)

    for key in ("embedding_ms", "dense_ms", "keyword_ms", "fusion_ms"):
        assert key in timing
        assert isinstance(timing[key], float)
        assert timing[key] >= 0.0


# --- app.retrieve_and_build_prompt ------------------------------------------


def test_retrieve_and_build_prompt_populates_prompt_build_timing(monkeypatch):
    def fake_retrieve_hybrid(query, top_k=5, document_id=None, timing=None):
        if timing is not None:
            timing["embedding_ms"] = 1.0
            timing["dense_ms"] = 2.0
            timing["keyword_ms"] = 3.0
            timing["fusion_ms"] = 4.0
        return [{"chunk_id": 1, "document_id": "d1", "source": "a.pdf", "page": 1, "text": "t", "score": 0.1, "rank": 1}]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    timing: dict = {}
    chunks, prompt = app.retrieve_and_build_prompt("a question", timing=timing)

    assert chunks
    assert isinstance(prompt, str) and prompt
    assert timing["embedding_ms"] == 1.0
    assert "prompt_build_ms" in timing
    assert timing["prompt_build_ms"] >= 0.0


def test_retrieve_and_build_prompt_without_timing_does_not_pass_timing_kwarg(monkeypatch):
    """When the caller doesn't ask for timing, retrieve_hybrid must be called
    without a `timing` kwarg at all, so mocks with a narrower signature
    (no `timing` param) - like the ones in test_app_helpers.py - keep working.
    """
    calls = []

    def fake_retrieve_hybrid(query, top_k=5, document_id=None):  # no timing param
        calls.append((query, top_k, document_id))
        return [{"chunk_id": 1, "document_id": "d1", "source": "a.pdf", "page": 1, "text": "t", "score": 0.1, "rank": 1}]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    app.retrieve_and_build_prompt("a question")

    assert calls == [("a question", 5, None)]


def test_retrieve_and_build_prompt_scopes_to_document_id(monkeypatch):
    """Document isolation still holds through the new retrieve_and_build_prompt
    split: document_id must reach retrieve_hybrid unchanged."""
    calls = []

    def fake_retrieve_hybrid(query, top_k=5, document_id=None):
        calls.append(document_id)
        return [{"chunk_id": 1, "document_id": document_id, "source": "a.pdf", "page": 1, "text": "t", "score": 0.1, "rank": 1}]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    chunks, _ = app.retrieve_and_build_prompt("a question", document_id="docA")

    assert calls == ["docA"]
    assert all(c["document_id"] == "docA" for c in chunks)


def test_answer_question_still_works_after_refactor(monkeypatch):
    """answer_question() must keep its exact external contract (used by
    scripts/query.py, chat.py) now that it's built on retrieve_and_build_prompt."""
    monkeypatch.setattr(
        app, "retrieve_hybrid",
        lambda query, top_k=5, document_id=None: [
            {"chunk_id": 1, "document_id": "d1", "source": "a.pdf", "page": 2, "text": "t", "score": 0.1, "rank": 1}
        ],
    )
    monkeypatch.setattr(app, "generate", lambda prompt: "an answer")

    result = app.answer_question("a question")

    assert result["answer"] == "an answer"
    assert result["sources"] == [{"source": "a.pdf", "page": 2, "score": 0.1}]
    assert result["chunks"][0]["text"] == "t"


# --- app._stream_with_ttft ---------------------------------------------------


def test_stream_with_ttft_records_time_to_first_token(monkeypatch):
    def fake_generate_stream(prompt, result=None):
        if result is not None:
            result["answer"] = "hello world"
            result["timing"] = {"eval_ms": 1.0}
        yield "hello"
        yield " world"

    monkeypatch.setattr(app, "generate_stream", fake_generate_stream)

    timing: dict = {}
    gen_result: dict = {}
    tokens = list(app._stream_with_ttft("a prompt", gen_result, timing))

    assert tokens == ["hello", " world"]
    assert "time_to_first_token_ms" in timing
    assert timing["time_to_first_token_ms"] >= 0.0
    assert gen_result["answer"] == "hello world"


def test_stream_with_ttft_propagates_errors(monkeypatch):
    from src.exceptions import OllamaUnavailableError

    def failing_stream(prompt, result=None):
        raise OllamaUnavailableError("boom")
        yield  # pragma: no cover - unreachable, makes this a generator

    monkeypatch.setattr(app, "generate_stream", failing_stream)

    with pytest.raises(OllamaUnavailableError):
        list(app._stream_with_ttft("a prompt", {}, {}))
