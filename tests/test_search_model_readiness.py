"""The first question after start-up waits for the embedding model to load;
that wait is labelled and timed on its own, not as "Finding relevant passages".

Measured on the reference machine: the first question after the server
started showed "Finding relevant passages: 48030 ms" although retrieval took
tens of milliseconds. All of it was the one-time load of torch, transformers
and sentence-transformers, which the background warm-up had started on the
first page render and the question then waited for. A repeat of the same
question took 47 ms.
"""

import time

from streamlit.testing.v1 import AppTest

import app
import src.embedding.embedder as embedder_module
import src.generation.ollama_client as ollama_module

from test_chat_flow import APP_PATH, _stream, chat_app  # noqa: F401 - pytest fixture


def test_is_model_loaded_reflects_the_cached_model(monkeypatch):
    monkeypatch.setattr(embedder_module, "_model", None)
    assert embedder_module.is_model_loaded() is False
    monkeypatch.setattr(embedder_module, "_model", object())
    assert embedder_module.is_model_loaded() is True


def test_wait_is_free_once_the_model_is_loaded(monkeypatch):
    def must_not_load():
        raise AssertionError("get_model must not be called when the model is loaded")

    monkeypatch.setattr(app, "embedding_model_loaded", lambda: True)
    monkeypatch.setattr(app, "get_embedding_model", must_not_load)
    timing = {}
    app.wait_for_search_model(timing)
    assert timing == {}


def test_wait_times_the_model_load_separately(monkeypatch):
    calls = []

    def slow_load():
        calls.append(1)
        time.sleep(0.05)

    monkeypatch.setattr(app, "embedding_model_loaded", lambda: False)
    monkeypatch.setattr(app, "get_embedding_model", slow_load)
    timing = {}
    app.wait_for_search_model(timing)
    assert calls == [1]
    assert timing["model_load_ms"] >= 40


def test_timing_rows_report_model_start_apart_from_retrieval():
    rows = dict(
        app.timing_rows(
            {"model_load_ms": 48000, "embedding_ms": 20, "dense_ms": 7, "keyword_ms": 3, "fusion_ms": 2, "total_ms": 49200}
        )
    )
    assert rows["Starting the search model (first question only)"] == "48000 ms"
    assert rows["Finding relevant passages"] == "32 ms"


def test_timing_rows_omit_model_start_when_already_loaded():
    rows = dict(app.timing_rows({"embedding_ms": 20, "dense_ms": 7, "total_ms": 1200}))
    assert not any(label.startswith("Starting the search model") for label in rows)


def test_first_question_records_the_model_wait_in_the_app(chat_app):  # noqa: F811
    loaded = {"done": False}

    def fake_load():
        time.sleep(0.05)
        loaded["done"] = True

    chat_app.setattr(embedder_module, "is_model_loaded", lambda: loaded["done"])
    chat_app.setattr(embedder_module, "get_model", fake_load)
    chat_app.setattr(ollama_module, "generate_stream", _stream("Amoxicillin", " was prescribed."))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    at.chat_input[0].set_value("What was prescribed?").run(timeout=30)
    first = at.session_state["chat_history"][1]
    assert first["status"] == "ok"
    assert first["timing"]["model_load_ms"] >= 40

    at.chat_input[0].set_value("What was prescribed?").run(timeout=30)
    second = at.session_state["chat_history"][3]
    assert second["status"] == "ok"
    assert "model_load_ms" not in second["timing"]


def test_a_model_that_fails_to_load_is_a_retryable_error(chat_app):  # noqa: F811
    def broken_load():
        raise OSError("model files missing")

    chat_app.setattr(embedder_module, "is_model_loaded", lambda: False)
    chat_app.setattr(embedder_module, "get_model", broken_load)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)

    at.chat_input[0].set_value("What was prescribed?").run(timeout=30)
    assert not at.exception
    assert at.session_state["chat_history"][1]["status"] == "error"
