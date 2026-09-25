"""Tests for src/generation/ollama_client.py's streaming, timed-generation,
and warm-up additions. generate() itself is unchanged and already covered
elsewhere; these focus on the new generate_stream()/generate_with_timing()/
warm_up() behavior: NDJSON stream parsing, final-answer assembly, timing
extraction, and error handling for a failed or interrupted stream.
"""

import json

import pytest
import requests

from src.exceptions import OllamaUnavailableError
from src.generation import ollama_client


class FakeStreamResponse:
    """Mimics requests.Response for a streaming /api/generate call."""

    def __init__(self, lines: list[dict], status_ok: bool = True, raise_iterating: Exception | None = None):
        self._lines = lines
        self.status_ok = status_ok
        self._raise_iterating = raise_iterating

    def raise_for_status(self):
        if not self.status_ok:
            raise requests.HTTPError("bad status")

    def iter_lines(self):
        for line in self._lines:
            yield json.dumps(line).encode("utf-8")
        if self._raise_iterating:
            raise self._raise_iterating


def _ok_lines(tokens: list[str], **done_fields) -> list[dict]:
    lines = [{"response": t, "done": False} for t in tokens]
    lines.append({"response": "", "done": True, **done_fields})
    return lines


@pytest.fixture(autouse=True)
def no_real_ollama_check(monkeypatch):
    """Every test here exercises generate_stream/generate_with_timing/warm_up
    in isolation - never a real Ollama server."""
    monkeypatch.setattr(ollama_client, "check_ollama_available", lambda: None)


def test_generate_stream_yields_tokens_in_order(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeStreamResponse(_ok_lines(["Hello", " ", "world"])),
    )

    tokens = list(ollama_client.generate_stream("a prompt"))

    assert tokens == ["Hello", " ", "world"]


def test_generate_stream_populates_result_with_answer_and_timing(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeStreamResponse(_ok_lines(
            ["Hello", " world"],
            total_duration=5_000_000,
            load_duration=1_000_000,
            prompt_eval_duration=500_000,
            eval_duration=3_000_000,
            prompt_eval_count=42,
            eval_count=7,
        )),
    )

    result: dict = {}
    list(ollama_client.generate_stream("a prompt", result=result))

    assert result["answer"] == "Hello world"
    assert result["timing"]["total_ms"] == pytest.approx(5.0)
    assert result["timing"]["load_ms"] == pytest.approx(1.0)
    assert result["timing"]["prompt_eval_ms"] == pytest.approx(0.5)
    assert result["timing"]["eval_ms"] == pytest.approx(3.0)
    assert result["timing"]["prompt_tokens"] == 42
    assert result["timing"]["completion_tokens"] == 7


@pytest.mark.parametrize(
    "done_fields, truncated",
    [
        ({"done_reason": "length"}, True),   # num_predict cut the answer off
        ({"done_reason": "stop"}, False),    # model finished on its own
        ({}, False),                         # older servers omit done_reason
    ],
)
def test_generate_stream_reports_whether_the_answer_was_cut_off(monkeypatch, done_fields, truncated):
    """A length-capped answer ends mid-sentence; the caller must be able to
    tell it apart from a complete one rather than presenting it as whole."""
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeStreamResponse(_ok_lines(["The values were 30.5 and"], **done_fields)),
    )

    result: dict = {}
    list(ollama_client.generate_stream("a prompt", result=result))

    assert result["truncated"] is truncated


def test_generate_stream_without_result_param_still_works(monkeypatch):
    """result is optional - omitting it must not error."""
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeStreamResponse(_ok_lines(["hi"])),
    )

    tokens = list(ollama_client.generate_stream("a prompt"))

    assert tokens == ["hi"]


def test_generate_stream_raises_on_empty_response(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeStreamResponse(_ok_lines([])),
    )

    with pytest.raises(OllamaUnavailableError):
        list(ollama_client.generate_stream("a prompt"))


def test_generate_stream_raises_when_request_fails(monkeypatch):
    def fake_post(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)

    with pytest.raises(OllamaUnavailableError):
        list(ollama_client.generate_stream("a prompt"))


def test_generate_stream_raises_on_interrupted_stream(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeStreamResponse(
            [{"response": "partial", "done": False}],
            raise_iterating=requests.exceptions.ChunkedEncodingError("connection dropped"),
        ),
    )

    with pytest.raises(OllamaUnavailableError):
        list(ollama_client.generate_stream("a prompt"))


def test_generate_stream_raises_on_bad_http_status(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeStreamResponse(_ok_lines(["hi"]), status_ok=False),
    )

    with pytest.raises(OllamaUnavailableError):
        list(ollama_client.generate_stream("a prompt"))


def test_generate_stream_sends_keep_alive_and_stream_true(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout, stream):
        captured["json"] = json
        captured["stream"] = stream
        return FakeStreamResponse(_ok_lines(["hi"]))

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)

    list(ollama_client.generate_stream("a prompt"))

    assert captured["json"]["stream"] is True
    assert captured["json"]["keep_alive"] == ollama_client.config.OLLAMA_KEEP_ALIVE
    assert captured["stream"] is True


def test_generate_stream_without_num_predict_uses_config_default(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout, stream):
        captured["json"] = json
        return FakeStreamResponse(_ok_lines(["hi"]))

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)

    list(ollama_client.generate_stream("a prompt"))

    assert captured["json"]["options"]["num_predict"] == ollama_client.config.OLLAMA_NUM_PREDICT


def test_generate_stream_num_predict_override_reaches_request(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout, stream):
        captured["json"] = json
        return FakeStreamResponse(_ok_lines(["hi"]))

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)

    list(ollama_client.generate_stream("a prompt", num_predict=768))

    assert captured["json"]["options"]["num_predict"] == 768
    # overriding num_predict must not touch any other option.
    assert captured["json"]["options"]["temperature"] == ollama_client.config.OLLAMA_TEMPERATURE


# --- generate_with_timing ---------------------------------------------------


class FakeResponse:
    def __init__(self, payload: dict, status_ok: bool = True):
        self._payload = payload
        self.status_ok = status_ok

    def raise_for_status(self):
        if not self.status_ok:
            raise requests.HTTPError("bad status")

    def json(self):
        return self._payload


def test_generate_with_timing_returns_answer_and_ms_breakdown(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeResponse({
            "response": "the answer",
            "total_duration": 10_000_000,
            "load_duration": 2_000_000,
            "prompt_eval_duration": 3_000_000,
            "eval_duration": 5_000_000,
            "prompt_eval_count": 10,
            "eval_count": 4,
        }),
    )

    answer, timing = ollama_client.generate_with_timing("a prompt")

    assert answer == "the answer"
    assert timing == {
        "load_ms": 2.0,
        "prompt_eval_ms": 3.0,
        "eval_ms": 5.0,
        "total_ms": 10.0,
        "prompt_tokens": 10,
        "completion_tokens": 4,
    }


def test_generate_with_timing_raises_on_empty_response(monkeypatch):
    monkeypatch.setattr(
        ollama_client.requests, "post",
        lambda *a, **k: FakeResponse({"response": "  "}),
    )

    with pytest.raises(OllamaUnavailableError):
        ollama_client.generate_with_timing("a prompt")


# --- warm_up -----------------------------------------------------------------


class _SyncThread:
    """Runs the target immediately instead of spawning a real thread, so the
    test can assert on the outgoing request deterministically."""

    def __init__(self, target, daemon=None):
        self._target = target

    def start(self):
        self._target()


def test_warm_up_sends_preload_request_with_no_prompt(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    monkeypatch.setattr(ollama_client.threading, "Thread", _SyncThread)

    ollama_client.warm_up()

    assert captured["json"] == {
        "model": ollama_client.config.OLLAMA_MODEL,
        "keep_alive": ollama_client.config.OLLAMA_KEEP_ALIVE,
    }
    assert "prompt" not in captured["json"]


def test_warm_up_swallows_request_errors(monkeypatch):
    def fake_post(*a, **k):
        raise requests.ConnectionError("ollama not running")

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    monkeypatch.setattr(ollama_client.threading, "Thread", _SyncThread)

    ollama_client.warm_up()  # must not raise
