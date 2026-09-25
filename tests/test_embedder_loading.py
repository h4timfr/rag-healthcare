"""Embedding-model loading: offline-first, download only when not cached, and
loaded once even when several sessions ask for it at the same time.

A fake SentenceTransformer stands in for the real one - these tests are about
how the model is loaded, not what it computes.
"""

import sys
import threading
import time
import types

import pytest

from src.embedding import embedder


class _Recorder:
    def __init__(self, fail_offline=False, delay=0.0):
        self.calls = []
        self.fail_offline = fail_offline
        self.delay = delay
        self.lock = threading.Lock()

    def factory(self, name, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
        time.sleep(self.delay)
        if kwargs.get("local_files_only") and self.fail_offline:
            raise OSError("couldn't find them in the cached files")
        return object()


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    monkeypatch.setattr(embedder, "_model", None)

    def install(recorder):
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = recorder.factory
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        return recorder

    return install


def test_cached_model_loads_without_contacting_the_hub(fake_sentence_transformers):
    rec = fake_sentence_transformers(_Recorder())

    embedder.get_model()

    assert rec.calls == [{"local_files_only": True}]


def test_first_run_without_a_cached_model_falls_back_to_downloading(fake_sentence_transformers):
    rec = fake_sentence_transformers(_Recorder(fail_offline=True))

    model = embedder.get_model()

    assert model is not None
    assert rec.calls == [{"local_files_only": True}, {}]


def test_warm_up_loads_the_model_in_the_background(fake_sentence_transformers):
    rec = fake_sentence_transformers(_Recorder(delay=0.2))

    started = time.perf_counter()
    embedder.warm_up()
    returned_after = time.perf_counter() - started

    assert returned_after < 0.1  # never blocks the page render
    deadline = time.perf_counter() + 5
    while embedder._model is None and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert embedder._model is not None
    assert len(rec.calls) == 1


def test_warm_up_swallows_load_failures(fake_sentence_transformers, monkeypatch):
    def broken(name, **kwargs):
        raise RuntimeError("torch import failed")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = broken
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embedder.warm_up()  # must not raise into the caller
    time.sleep(0.2)
    assert embedder._model is None  # first real query will simply retry


def test_concurrent_first_calls_load_the_model_once(fake_sentence_transformers):
    rec = fake_sentence_transformers(_Recorder(delay=0.3))
    results = []

    threads = [threading.Thread(target=lambda: results.append(embedder.get_model())) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(rec.calls) == 1
    assert len({id(m) for m in results}) == 1
