#!/usr/bin/env python
"""CLI: profile one real query end-to-end and print a stage-by-stage
latency breakdown (query embedding, dense retrieval, keyword retrieval,
hybrid fusion, prompt construction, Ollama model load, Ollama prompt eval,
Ollama generation, total end-to-end).

This is a measurement tool, not a permanent UI feature - see the equivalent
per-turn "Performance" expander in app.py for the same breakdown surfaced
(collapsed by default) in the Streamlit chat.

Usage:
    python scripts/profile_query.py "your question"
    python scripts/profile_query.py "your question" --document-id <id>
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.exceptions import EmptyQueryError, InsufficientRetrievalError, RagError
from src.generation.ollama_client import generate_with_timing
from src.generation.prompt_builder import build_prompt
from src.retrieval.hybrid_retriever import retrieve_hybrid


def profile_query(query: str, top_k: int = config.TOP_K, document_id: str | None = None) -> dict:
    """Run one query through the full pipeline, timing every stage.

    Returns a flat dict of stage -> milliseconds, plus prompt-size stats
    and the answer/sources, so callers (this CLI, or a future test) can
    inspect numbers without re-parsing printed text.
    """
    timing: dict = {}

    try:
        chunks = retrieve_hybrid(query, top_k=top_k, document_id=document_id, timing=timing)
    except ValueError as e:
        raise EmptyQueryError(str(e)) from e
    if not chunks:
        raise InsufficientRetrievalError("No relevant chunks found for this query")

    t0 = time.perf_counter()
    prompt = build_prompt(query, chunks)
    timing["prompt_build_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    answer, gen_timing = generate_with_timing(prompt)
    wall_generation_ms = (time.perf_counter() - t0) * 1000

    retrieval_ms = sum(
        timing.get(k, 0.0) for k in ("embedding_ms", "dense_ms", "keyword_ms", "fusion_ms")
    )
    total_ms = retrieval_ms + timing["prompt_build_ms"] + wall_generation_ms

    return {
        **timing,
        "generation": gen_timing,
        "wall_generation_ms": wall_generation_ms,
        "total_ms": total_ms,
        "prompt_chars": len(prompt),
        "prompt_words": len(prompt.split()),
        "chunk_count": len(chunks),
        "answer": answer,
        "sources": [{"source": c["source"], "page": c["page"]} for c in chunks],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one query's end-to-end latency.")
    parser.add_argument("query", help="Natural-language question")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--document-id", default=None, help="Scope retrieval to one document_id")
    args = parser.parse_args()

    try:
        result = profile_query(args.query, top_k=args.top_k, document_id=args.document_id)
    except RagError as e:
        print(f"Profiling failed: {e}", file=sys.stderr)
        sys.exit(1)

    gen = result["generation"]
    print("STAGE TIMINGS")
    print(f"  query embedding      : {result.get('embedding_ms', 0.0):8.1f} ms")
    print(f"  dense retrieval      : {result.get('dense_ms', 0.0):8.1f} ms")
    print(f"  keyword retrieval    : {result.get('keyword_ms', 0.0):8.1f} ms")
    print(f"  hybrid fusion        : {result.get('fusion_ms', 0.0):8.1f} ms")
    print(f"  prompt construction  : {result.get('prompt_build_ms', 0.0):8.1f} ms")
    print(f"  ollama model load    : {gen.get('load_ms', 0.0):8.1f} ms")
    print(f"  ollama prompt eval   : {gen.get('prompt_eval_ms', 0.0):8.1f} ms")
    print(f"  ollama generation    : {gen.get('eval_ms', 0.0):8.1f} ms")
    print(f"  ---------------------------------")
    print(f"  TOTAL END-TO-END     : {result['total_ms']:8.1f} ms")
    print()
    print("PROMPT SIZE")
    print(f"  chunks retrieved     : {result['chunk_count']}")
    print(f"  prompt characters    : {result['prompt_chars']}")
    print(f"  prompt words (approx): {result['prompt_words']}")
    if gen.get("prompt_tokens"):
        print(f"  prompt tokens (ollama): {gen['prompt_tokens']}")
    if gen.get("completion_tokens"):
        print(f"  completion tokens    : {gen['completion_tokens']}")
    print()
    print("ANSWER")
    print(f"  {result['answer']}")
    print()
    print("SOURCES")
    for s in result["sources"]:
        print(f"  - {s['source']} (page {s['page']})")


if __name__ == "__main__":
    main()
