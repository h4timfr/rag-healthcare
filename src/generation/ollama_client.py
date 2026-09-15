"""Local generation via Ollama's HTTP API (no cloud calls)."""

import requests

from .. import config
from ..exceptions import ModelUnavailableError, OllamaUnavailableError


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


def generate(prompt: str) -> str:
    """Send a prompt to the local Ollama model and return the generated text."""
    check_ollama_available()
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={
    "model": config.OLLAMA_MODEL,
    "prompt": prompt,
    "stream": False,
    "keep_alive": config.OLLAMA_KEEP_ALIVE,
    "options": {
        "temperature": config.OLLAMA_TEMPERATURE,
        "top_p": config.OLLAMA_TOP_P,
        "num_predict": config.OLLAMA_NUM_PREDICT,
    },
},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise OllamaUnavailableError(f"Ollama generation request failed: {e}") from e

    answer = resp.json().get("response", "").strip()
    if not answer:
        raise OllamaUnavailableError("Ollama returned an empty response")
    return answer
