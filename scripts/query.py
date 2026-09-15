#!/usr/bin/env python
"""CLI: answer a query using the RAG engine (retrieve -> build prompt -> generate)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.exceptions import EmptyQueryError, InsufficientRetrievalError, RagError
from src.generation.ollama_client import generate
from src.generation.prompt_builder import build_prompt
from src.retrieval.hybrid_retriever import retrieve_hybrid


def answer_query(query: str, top_k: int = config.TOP_K):
    # retrieve_hybrid raises plain ValueError (not a RagError) on empty query
    # and returns [] rather than raising when nothing is found; normalize both
    # to the same RagError-based contract the dense path already had.
    try:
        chunks = retrieve_hybrid(query, top_k=top_k)
    except ValueError as e:
        raise EmptyQueryError(str(e)) from e

    if not chunks:
        raise InsufficientRetrievalError("No relevant chunks found for this query")

    prompt = build_prompt(query, chunks)
    answer = generate(prompt)
    sources = [{"source": c["source"], "page": c["page"], "score": c["score"]} for c in chunks]
    return answer, sources


def main():
    parser = argparse.ArgumentParser(description="Query the RAG engine.")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    try:
        answer, sources = answer_query(args.query, top_k=args.top_k)
    except RagError as e:
        print(f"Query failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("ANSWER:")
    print(answer)
    print("\nSOURCES:")
    for s in sources:
        print(f"  - {s['source']} (page {s['page']}, score {s['score']:.3f})")


if __name__ == "__main__":
    main()
