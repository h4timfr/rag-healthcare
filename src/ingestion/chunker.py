"""Word-based chunking with configurable size and overlap.

Chunking is word-based (not token-based) to avoid pulling in a separate
tokenizer dependency; chunk_size/overlap are tuned via src/config.py.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    page: int
    text: str


def chunk_page_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a single page's text into overlapping word-window chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(words):
        piece = words[start : start + chunk_size]
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def chunk_pages(pages: list[dict], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk a list of {"page": int, "text": str} entries, preserving page numbers."""
    result = []
    for p in pages:
        for piece in chunk_page_text(p["text"], chunk_size, overlap):
            result.append(Chunk(page=p["page"], text=piece))
    return result
