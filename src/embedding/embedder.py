"""Embedding generation using sentence-transformers/all-MiniLM-L6-v2.

Embeddings are L2-normalized at generation time so that FAISS inner-product
search is equivalent to cosine similarity.
"""

import numpy as np

from ..config import EMBEDDING_MODEL_NAME

_model = None


def get_model():
    """Lazily import sentence-transformers and load/cache the model.

    The import itself (not just model loading) is deferred until first use:
    `sentence-transformers` pulls in torch, which is the single largest
    contributor to cold-start time. Modules that only need embed_texts/
    embed_query at call time (not import time) stay fast to import.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


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
