"""Custom exceptions for the RAG engine.

All engine-raised errors derive from RagError so callers (CLI scripts, and later
the UI) can catch one base type and print a readable message.
"""


class RagError(Exception):
    """Base exception for all RAG engine errors."""


class InvalidPDFError(RagError):
    """Raised when a PDF is missing, unreadable, or encrypted."""


class PDFExtractionError(RagError):
    """Raised when text extraction fails on an otherwise valid PDF."""


class EmptyDocumentError(RagError):
    """Raised when a PDF contains no extractable text (e.g. scanned/image-only)."""


class IndexNotFoundError(RagError):
    """Raised when the FAISS index file does not exist yet."""


class DatabaseNotFoundError(RagError):
    """Raised when the SQLite metadata database does not exist yet."""


class EmptyQueryError(RagError):
    """Raised when a user query is empty or whitespace-only."""


class OllamaUnavailableError(RagError):
    """Raised when the local Ollama server cannot be reached or fails."""


class ModelUnavailableError(RagError):
    """Raised when the configured Ollama model is not pulled/available."""


class InsufficientRetrievalError(RagError):
    """Raised when retrieval finds no usable chunks for a query."""
