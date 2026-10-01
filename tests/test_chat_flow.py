"""The chat turn lifecycle and the user-facing text around it.

A question is committed to the history as soon as it is sent and answered on
the next run with the input disabled, so it can't be lost to a second
submission or an interrupted run; every answer attempt ends in exactly one
committed turn - an answer, a no-evidence notice, or a retryable error.
"""

from pathlib import Path

import numpy as np
import pymupdf
import pytest
from streamlit.testing.v1 import AppTest

import app
import scripts.ingest as ingest_script
import src.embedding.embedder as embedder_module
import src.generation.ollama_client as ollama_module
import src.retrieval.hybrid_retriever as hybrid_module
from src import config
from src.exceptions import (
    InvalidPDFError,
    ModelUnavailableError,
    OllamaUnavailableError,
    RagError,
)

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


# --- pure helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    "count,expected",
    [(0, "0 chunks"), (1, "1 chunk"), (2, "2 chunks"), (11, "11 chunks")],
)
def test_pluralize(count, expected):
    assert app.pluralize(count, "chunk") == expected


def test_document_meta_is_grammatical_for_one_chunk_and_one_page():
    """The library card read "1 chunks · 1 pages"."""
    assert app.document_meta({"chunk_count": 1, "page_count": 1}) == "1 chunk · 1 page"
    assert app.document_meta({"chunk_count": 12, "page_count": 4}) == "12 chunks · 4 pages"
    assert app.document_meta({"chunk_count": 3, "page_count": None}) == "3 chunks"


@pytest.mark.parametrize(
    "exc,title_fragment",
    [
        (OllamaUnavailableError("Could not reach Ollama at http://127.0.0.1:9: HTTPConnectionPool"), "isn't available"),
        (OllamaUnavailableError("Ollama streaming response interrupted: reset"), "interrupted"),
        (OllamaUnavailableError("Ollama returned an empty response"), "empty answer"),
        (ModelUnavailableError("Model 'x' is not available"), "isn't installed"),
        (RuntimeError("KeyError deep in the stack"), "unexpected"),
    ],
)
def test_describe_error_answers_what_why_and_what_next(exc, title_fragment):
    err = app.describe_error(exc)
    assert title_fragment in err.title
    assert err.cause and err.action
    raw = str(exc)
    # The raw exception text is kept for the technical-details panel only.
    assert raw in err.detail
    assert raw not in err.title + err.cause + err.action


def test_plain_rag_error_text_is_already_user_facing_and_passes_through():
    err = app.describe_error(RagError("No chunks produced"))
    assert err.cause == "No chunks produced"


@pytest.mark.parametrize(
    "error_type,message,fragment",
    [
        ("InvalidPDFError", "PDF 'a.pdf' is encrypted/password-protected", "password-protected"),
        ("InvalidPDFError", "Could not open 'a.pdf' as a PDF: Failed to open file", "couldn't be opened"),
        ("EmptyDocumentError", "No extractable text in 'a.pdf': the file appears damaged or truncated.", "damaged"),
        ("EmptyDocumentError", "No extractable text in 'a.pdf' (likely empty or a scanned PDF)", "no readable text"),
        ("OSError", "[Errno 28] No space left on device", "couldn't be added"),
    ],
)
def test_upload_failures_are_described_in_plain_language(error_type, message, fragment):
    err = app.describe_upload_failure("a.pdf", error_type, message)
    assert fragment in err.title
    assert "Errno" not in err.title + err.cause + err.action
    assert err.detail == message


def test_ingest_uploaded_pdfs_records_the_failure_type(monkeypatch):
    def failing(path):
        raise InvalidPDFError("PDF 'x.pdf' is encrypted/password-protected")

    monkeypatch.setattr(app.ingest_module, "ingest_pdf", failing)
    [result] = app.ingest_uploaded_pdfs([("x.pdf", b"data")])
    assert result["error_type"] == "InvalidPDFError"


def test_summarize_upload_batch():
    results = [{"status": "indexed"}, {"status": "indexed"}, {"status": "skipped"}, {"status": "failed"}]
    assert app.summarize_upload_batch(results) == "2 documents added · 1 already in your library · 1 couldn't be added"
    assert app.summarize_upload_batch([{"status": "indexed"}]) == "1 document added"


def test_timing_rows_use_readable_labels_and_skip_unmeasured_stages():
    rows = dict(app.timing_rows({"embedding_ms": 20, "dense_ms": 7, "total_ms": 2400}))
    assert rows["Finding relevant passages"] == "27 ms"
    assert rows["Total"] == "2.4 s"
    assert not any("Until the answer" in label for label in rows)


def test_ollama_status_view():
    assert app.ollama_status_view({"ready": True, "detail": "Ready"})[0] is True
    assert app.ollama_status_view({"ready": False, "detail": "Ollama not reachable"})[1] == "Ollama isn't running"
    assert app.ollama_status_view({"ready": False, "detail": "Model 'x' not pulled"})[1] == "Model not installed"


# --- turn state machine (plain dict stands in for st.session_state) -------------


def test_a_queued_question_is_in_the_history_before_any_answer_exists():
    state = {"chat_history": []}
    app.queue_question(state, "What was prescribed?", {"document_id": "d1"})
    assert state["chat_history"] == [{"role": "user", "content": "What was prescribed?"}]
    assert state["pending_turn"]["history_len"] == 0


def test_a_failed_turn_can_be_retried_with_its_original_scope():
    state = {"chat_history": []}
    app.queue_question(state, "Q?", {"document_id": "d1"})
    pending = state["pending_turn"]
    app.finish_pending_turn(state, app.error_turn(app.describe_error(OllamaUnavailableError("down")), pending))

    assert "pending_turn" not in state
    assert app.can_retry(state)
    app.retry_last_turn(state)

    assert [t["role"] for t in state["chat_history"]] == ["user"]  # error turn dropped, question kept
    assert state["pending_turn"]["scope"] == {"document_id": "d1"}
    assert state["pending_turn"]["attempts"] == 0
    assert not app.can_retry(state)  # already queued


def test_error_turns_are_not_part_of_the_generation_context():
    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "status": "error", "content": "Ollama isn't available"},
        {"role": "assistant", "content": "an answer"},
    ]
    assert [t["content"] for t in app.conversation_context(history)] == ["a", "an answer"]


# --- the running app --------------------------------------------------------------


def _make_pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text)
    doc.save(str(path))
    doc.close()


def _fake_vectors(n):
    rng = np.random.default_rng(n)
    v = rng.standard_normal((n, config.EMBEDDING_DIM)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def chat_app(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_vectors_for_texts)
    monkeypatch.setattr(hybrid_module, "embed_query", lambda q: _fake_vectors(1)[0])
    monkeypatch.setattr(embedder_module, "warm_up", lambda: None)
    monkeypatch.setattr(ollama_module, "warm_up", lambda: None)
    monkeypatch.setattr(ollama_module, "get_status", lambda: {"ready": True, "detail": "Ready"})
    pdf = tmp_path / "note.pdf"
    _make_pdf(pdf, "The patient was prescribed amoxicillin for an ear infection. " * 20)
    ingest_script.ingest_pdf(str(pdf))
    return monkeypatch


def _fake_vectors_for_texts(texts):
    return _fake_vectors(len(texts))


def _stream(*pieces, fail_after=None):
    def fake_generate_stream(prompt, result=None, num_predict=None):
        for i, piece in enumerate(pieces):
            if fail_after is not None and i == fail_after:
                raise OllamaUnavailableError("Ollama streaming response interrupted: connection reset")
            yield piece
        if result is not None:
            result["answer"] = "".join(pieces).strip()
            result["timing"] = {"eval_ms": 5.0, "completion_tokens": 3}
            result["truncated"] = False

    return fake_generate_stream


def _text(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_a_question_is_answered_and_the_input_is_usable_again(chat_app):
    chat_app.setattr(ollama_module, "generate_stream", _stream("Amoxicillin", " was prescribed."))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    at.chat_input[0].set_value("What was prescribed?").run(timeout=30)

    assert not at.exception
    history = at.session_state["chat_history"]
    assert [t["role"] for t in history] == ["user", "assistant"]
    assert history[1]["status"] == "ok"
    assert history[1]["content"] == "Amoxicillin was prescribed."
    assert "pending_turn" not in at.session_state
    assert not at.chat_input[0].disabled
    assert 'class="source-row"' in _text(at)


def test_a_stream_that_fails_midway_leaves_a_retryable_error_not_a_partial_answer(chat_app):
    chat_app.setattr(ollama_module, "generate_stream", _stream("Amoxi", "cillin", fail_after=1))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    at.chat_input[0].set_value("What was prescribed?").run(timeout=30)

    history = at.session_state["chat_history"]
    assert [t["role"] for t in history] == ["user", "assistant"]
    assert history[1]["status"] == "error"
    assert "Amoxi" not in _text(at)  # the partial text was discarded
    assert "interrupted" in _text(at)
    assert "connection reset" not in _text(at)  # raw detail only in the collapsed panel

    chat_app.setattr(ollama_module, "generate_stream", _stream("Amoxicillin."))
    at.button(key="retry_1").click().run(timeout=30)

    history = at.session_state["chat_history"]
    assert [t["role"] for t in history] == ["user", "assistant"]
    assert history[1]["status"] == "ok"
    assert history[0]["content"] == "What was prescribed?"


def test_a_turn_interrupted_too_often_is_committed_as_an_error(chat_app):
    """Every rerun resumes the pending question; past the attempt limit it
    becomes a retryable error instead of being retried forever."""
    chat_app.setattr(ollama_module, "generate_stream", _stream("ok."))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.session_state["chat_history"] = [{"role": "user", "content": "Q?"}]
    at.session_state["pending_turn"] = {
        "query": "Q?",
        "scope": {
            "analysis_scope": "active",
            "document_id": at.session_state["active_document_id"],
            "document_inventory": None,
            "top_k": config.TOP_K,
            "scope_label": "note.pdf",
        },
        "history_len": 0,
        "attempts": MAX_ATTEMPTS_REACHED,
    }
    at.run(timeout=30)

    # Over the attempt limit: committed as an error rather than retried forever.
    history = at.session_state["chat_history"]
    assert history[-1]["status"] == "error"
    assert "pending_turn" not in at.session_state


MAX_ATTEMPTS_REACHED = app.MAX_GENERATION_ATTEMPTS


def test_no_matching_evidence_is_a_notice_not_an_error(chat_app):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    # AppTest re-executes app.py's imports on every run, so patching the
    # module it imports retrieval from takes effect on the next run.
    chat_app.setattr(hybrid_module, "retrieve_for_question", lambda *a, **k: [])

    at.chat_input[0].set_value("Unrelated question?").run(timeout=30)

    history = at.session_state["chat_history"]
    assert history[-1]["status"] == "no_evidence"
    assert "No matching passages" in _text(at)


def test_removing_a_cited_document_confirms_and_marks_the_old_sources(chat_app):
    chat_app.setattr(ollama_module, "generate_stream", _stream("Amoxicillin."))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.chat_input[0].set_value("What was prescribed?").run(timeout=30)
    doc_id = at.session_state["active_document_id"]

    at.button(key=f"remove_{doc_id}").click().run(timeout=30)
    at.button(key=f"confirm_{doc_id}").click().run(timeout=30)

    assert not at.exception
    assert any("Removed note.pdf" in t.value for t in at.toast)
    assert "removed from library" in _text(at)
    assert at.session_state["chat_history"][-1]["status"] == "ok"  # the answer itself is kept


def test_a_document_removed_before_its_answer_is_reported_not_crashed(chat_app):
    chat_app.setattr(ollama_module, "generate_stream", _stream("ok."))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.session_state["chat_history"] = [{"role": "user", "content": "Q?"}]
    at.session_state["pending_turn"] = {
        "query": "Q?",
        "scope": {
            "analysis_scope": "active",
            "document_id": "no-such-document",
            "document_inventory": None,
            "top_k": config.TOP_K,
            "scope_label": "gone.pdf",
        },
        "history_len": 0,
        "attempts": 0,
    }
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state["chat_history"][-1]["status"] == "error"
    assert "no longer in the library" in _text(at)
