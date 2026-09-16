"""Local generation via Ollama's HTTP API (no cloud calls)."""

import json
import threading

import requests

from .. import config
from ..exceptions import ModelUnavailableError, OllamaUnavailableError

_NS_PER_MS = 1_000_000


def check_ollama_available() -> None:
    """Verify the local Ollama server is reachable and the configured model is pulled."""
    try:
        resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {config.OLLAMA_HOST}: {e}"
        ) from e

    models = [m.get("name", "") for m in resp.json().get("models", [])]
    if not any(m == config.OLLAMA_MODEL or m.startswith(config.OLLAMA_MODEL) for m in models):
        raise ModelUnavailableError(
            f"Model '{config.OLLAMA_MODEL}' is not available in Ollama. "
            f"Available models: {models or 'none'}. Run: ollama pull {config.OLLAMA_MODEL}"
        )


def get_status() -> dict:
    """Non-raising, short-timeout readiness check for UI status display.

    Distinct from check_ollama_available(): that one is part of the
    generation path and intentionally raises with a 5s timeout. This one is
    for a sidebar indicator that must never block page rendering for long or
    interrupt the chat flow with an exception.
    """
    try:
        resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=2)
        resp.raise_for_status()
    except requests.RequestException:
        return {"ready": False, "detail": "Ollama not reachable"}

    models = [m.get("name", "") for m in resp.json().get("models", [])]
    if not any(m == config.OLLAMA_MODEL or m.startswith(config.OLLAMA_MODEL) for m in models):
        return {"ready": False, "detail": f"Model '{config.OLLAMA_MODEL}' not pulled"}

    return {"ready": True, "detail": f"Ready ({config.OLLAMA_MODEL})"}


def _generate_request(prompt: str, stream: bool):
    """Shared request body/options for every /api/generate call (blocking,
    timed, and streaming) - keeps the model/prompt/options identical across
    all three so none of them can silently drift apart from the others.
    """
    return requests.post(
        f"{config.OLLAMA_HOST}/api/generate",
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": config.OLLAMA_TEMPERATURE,
                "top_p": config.OLLAMA_TOP_P,
                "num_predict": config.OLLAMA_NUM_PREDICT,
            },
        },
        timeout=120,
        stream=stream,
    )


def _extract_timing(data: dict) -> dict:
    """Convert Ollama's nanosecond duration fields (present on the final
    response object, streamed or not) into a millisecond breakdown:
    load_ms is model-load/startup time (near zero once warm), prompt_eval_ms
    is prompt processing, eval_ms is pure token-generation time.
    """
    return {
        "load_ms": data.get("load_duration", 0) / _NS_PER_MS,
        "prompt_eval_ms": data.get("prompt_eval_duration", 0) / _NS_PER_MS,
        "eval_ms": data.get("eval_duration", 0) / _NS_PER_MS,
        "total_ms": data.get("total_duration", 0) / _NS_PER_MS,
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
    }


def generate(prompt: str) -> str:
    """Send a prompt to the local Ollama model and return the generated text."""
    check_ollama_available()
    try:
        resp = _generate_request(prompt, stream=False)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise OllamaUnavailableError(f"Ollama generation request failed: {e}") from e

    answer = resp.json().get("response", "").strip()
    if not answer:
        raise OllamaUnavailableError("Ollama returned an empty response")
    return answer


def generate_with_timing(prompt: str) -> tuple[str, dict]:
    """Same request/response as generate(), but also returns Ollama's own
    reported timing breakdown (see _extract_timing). Used by the profiling
    CLI (scripts/profile_query.py) - kept separate from generate() so every
    existing caller of generate() (including tests that monkeypatch it to
    return a plain string) is unaffected.
    """
    check_ollama_available()
    try:
        resp = _generate_request(prompt, stream=False)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise OllamaUnavailableError(f"Ollama generation request failed: {e}") from e

    data = resp.json()
    answer = data.get("response", "").strip()
    if not answer:
        raise OllamaUnavailableError("Ollama returned an empty response")
    return answer, _extract_timing(data)


def generate_stream(prompt: str, result: dict | None = None):
    """Stream generated text from Ollama's /api/generate (stream=True).

    Yields plain-text chunks as they arrive, so a caller like Streamlit's
    `st.write_stream()` can render tokens incrementally. If `result` is
    given, it's populated in place once the stream completes with
    `result["answer"]` (the full concatenated text, identical to what
    generate() would have returned for the same prompt) and
    `result["timing"]` (see _extract_timing) - mirrors retrieve_hybrid's
    `timing` out-param pattern used elsewhere in this codebase.

    Raises OllamaUnavailableError if the server can't be reached, the
    stream is interrupted/unparseable partway through, or the assembled
    answer ends up empty - the same failure contract as generate().
    """
    check_ollama_available()
    try:
        resp = _generate_request(prompt, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise OllamaUnavailableError(f"Ollama generation request failed: {e}") from e

    pieces: list[str] = []
    final_data: dict = {}
    try:
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("response", "")
            if token:
                pieces.append(token)
                yield token
            if data.get("done"):
                final_data = data
    except requests.RequestException as e:
        raise OllamaUnavailableError(f"Ollama streaming response interrupted: {e}") from e
    except ValueError as e:
        raise OllamaUnavailableError(f"Ollama returned an unparseable streaming response: {e}") from e

    full_answer = "".join(pieces).strip()
    if not full_answer:
        raise OllamaUnavailableError("Ollama returned an empty response")

    if result is not None:
        result["answer"] = full_answer
        result["timing"] = _extract_timing(final_data)


def warm_up() -> None:
    """Fire-and-forget request to make Ollama load the model into memory
    ahead of the user's first real question, so that query doesn't pay the
    full cold-load latency while they wait on a visible spinner/stream.

    Uses Ollama's documented preload pattern (a request with no "prompt"
    field triggers a model load with no generation). Runs in a daemon
    thread and swallows every error: this is a latency optimization, not a
    correctness requirement, and must never block the page or surface a
    failure - if it fails or Ollama isn't reachable, the first real query
    just proceeds at normal (unwarmed) speed, exactly as before this
    existed. Does not change `keep_alive`, so an idle model still unloads
    after config.OLLAMA_KEEP_ALIVE as before - this only avoids paying the
    load cost at an inconvenient moment, it does not keep anything loaded
    for longer than already configured.
    """

    def _run() -> None:
        try:
            requests.post(
                f"{config.OLLAMA_HOST}/api/generate",
                json={"model": config.OLLAMA_MODEL, "keep_alive": config.OLLAMA_KEEP_ALIVE},
                timeout=120,
            )
        except requests.RequestException:
            pass

    threading.Thread(target=_run, daemon=True).start()
