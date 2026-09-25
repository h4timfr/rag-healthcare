"""Embedding generation using sentence-transformers/all-MiniLM-L6-v2.

Embeddings are L2-normalized at generation time so that FAISS inner-product
search is equivalent to cosine similarity.
"""

import threading

import numpy as np

from ..config import EMBEDDING_MODEL_NAME

_model = None
_model_lock = threading.Lock()


def get_model():
    """Lazily import sentence-transformers and load/cache the model.

    The import itself (not just model loading) is deferred until first use:
    `sentence-transformers` pulls in torch, which is the single largest
    contributor to cold-start time. Modules that only need embed_texts/
    embed_query at call time (not import time) stay fast to import.

    Loads from the local cache without contacting the Hugging Face Hub. A
    default load checks the Hub for updates on every process start - measured
    at 6.2s versus 0.4s local-only - which is also a network call from a
    system meant to run offline. Only the very first run, before the model is
    cached, falls back to downloading it.

    Locked so two sessions' first queries don't both load the model.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                try:
                    _model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
                except OSError:
                    _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def warm_up() -> None:
    """Load the embedding model in a background daemon thread.

    Importing sentence-transformers (torch) takes ~16s cold on the reference
    machine, and without this it is paid inside the user's first query.
    Errors are swallowed: the first real embed call simply loads the model
    itself, exactly as it would have without warm-up.
    """

    def _run() -> None:
        try:
            get_model()
        except Exception:  # noqa: BLE001 - optimization only, never fatal
            pass

    threading.Thread(target=_run, daemon=True).start()


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts. Returns a float32 array of shape (n, EMBEDDING_DIM)."""
    if not texts:
        raise ValueError("texts must not be empty")
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string using the same model/normalization as documents."""
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    return embed_texts([query])[0]
