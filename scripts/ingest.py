#!/usr/bin/env python
"""CLI: ingest a PDF into the RAG engine (extract -> clean -> chunk -> embed -> FAISS + SQLite)."""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import config
from src.embedding.embedder import embed_texts
from src.exceptions import RagError
from src.ingestion.chunker import chunk_pages
from src.ingestion.cleaner import clean_text
from src.ingestion.pdf_extractor import extract_pages
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.metadata_store import MetadataStore


def ingest_pdf(pdf_path: str, db_path: str | None = None, faiss_index_path: str | None = None) -> dict:
    """Ingest a single PDF end-to-end.

    The document_id is a content hash, so re-ingesting a PDF whose bytes were
    already indexed is detected and skipped (no duplicate FAISS vectors/SQLite
    rows) rather than silently doubling up matching content on every re-upload.

    db_path/faiss_index_path default to the production config paths; pass
    alternatives to build a separate index (e.g. a research evaluation corpus)
    without touching the production demo index.

    Returns {"document_id": str, "chunk_count": int, "skipped": bool}.
    """
    db_path = db_path or config.DB_PATH
    faiss_index_path = faiss_index_path or config.FAISS_INDEX_PATH

    pdf_path = str(Path(pdf_path).resolve())
    raw_bytes = Path(pdf_path).read_bytes()
    document_id = hashlib.sha256(raw_bytes).hexdigest()[:16]

    store = MetadataStore(db_path)
    store.initialize()

    existing = store.document_chunk_count(document_id)
    if existing > 0:
        print(
            f"Skipped '{pdf_path}': already indexed as document_id={document_id} "
            f"({existing} chunks)"
        )
        return {"document_id": document_id, "chunk_count": existing, "skipped": True}

    pages = extract_pages(pdf_path)
    cleaned_pages = [{"page": p["page"], "text": clean_text(p["text"])} for p in pages]
    chunks = chunk_pages(cleaned_pages, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)

    if not chunks:
        raise RagError(f"No chunks produced from '{pdf_path}' after cleaning/chunking")

    index = FaissIndex(config.EMBEDDING_DIM, faiss_index_path)
    index.load_or_create()

    embeddings = embed_texts([c.text for c in chunks])

    # All of this document's chunks are inserted in one SQLite transaction
    # (see MetadataStore.insert_chunks): if anything fails partway through,
    # none of them are committed, so a retry never sees a partial document
    # and skip re-ingestion because of it.
    chunk_ids = store.insert_chunks(
        document_id=document_id,
        source=Path(pdf_path).name,
        chunks=[(chunk.page, chunk.text) for chunk in chunks],
    )

    index.add(embeddings, np.array(chunk_ids, dtype="int64"))
    index.save()

    print(f"Ingested '{pdf_path}': {len(chunks)} chunks (document_id={document_id})")
    return {"document_id": document_id, "chunk_count": len(chunks), "skipped": False}


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the RAG engine.")
    parser.add_argument("pdf", help="Path to the PDF file to ingest")
    args = parser.parse_args()

    try:
        ingest_pdf(args.pdf)
    except RagError as e:
        print(f"Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
