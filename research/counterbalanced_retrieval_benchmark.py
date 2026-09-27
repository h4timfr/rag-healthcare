#!/usr/bin/env python
"""Phase 3 - counterbalanced, retrieval-only benchmark.

Corrects two confounds in the existing scripts/run_evaluation.py methodology
for a fair LATENCY comparison (Recall/MRR/nDCG are already known deterministic
- see the existing runs 1-3 std=0.000 - so order cannot bias them, only
latency):

1. Method order was fixed (always dense-then-hybrid, every question) -
   whichever method runs first pays any residual warm-up/cache cost the
   other doesn't. Here the per-question order alternates deterministically
   (even question index -> dense first, odd -> hybrid first), so a
   first-vs-second-call effect lands equally on both methods across the set.
2. Warm-up ran once, in whatever order `--methods` listed. Here both paths
   (embedding, dense search, keyword search) are warmed up BEFORE any timed
   question, symmetrically.

This produces THREE explicitly separate evaluation labels (do not conflate):

  A_global            dense vs hybrid, no document_id - matches the OLD
                       benchmark's scope, for direct comparability.
  B_scoped_hybrid      hybrid, scoped to the question's own document_id -
                       what production actually calls for a single-document
                       question (app.py's real default: top_k=5, scoped).
  B_scoped_dense       dense-only, scoped to the same document_id, built
                       from the same FAISS/SQLite building blocks
                       retrieve_hybrid() itself uses (metadata_store.
                       filter_chunk_ids_by_document) - NOT a change to
                       src/retrieval/retriever.py, which has no document_id
                       parameter and is not touched here. Exists only so a
                       scoped dense baseline exists to compare B_scoped_hybrid
                       against; production itself never calls dense-only.

"C. End-to-end production evaluation" (generation included) is intentionally
NOT re-implemented here - that is what scripts/run_product_qa.py already is;
re-running it is part of Phase 9's regression, not this retrieval-only
benchmark.

Repeated LATENCY_REPS times per question per method for a distribution
(mean/median/stdev/p95), not just a mean. Recall/MRR/nDCG/term-coverage are
computed once per (question, method) - each is a pure function of a fixed
index and a fixed query, and repeating them would only ever multiply an
already-known-zero variance (see the docstring above).

Term coverage here is evidence-based, not answer-based: whether each
expected_term appears in the retrieved chunks' own TEXT. This project's
existing term_coverage metric (src/evaluation/metrics.py) grades a
GENERATED ANSWER; grading retrieval against the same expected_terms
requires this retrieval-only counterpart instead, or a retrieval success
would be misclassified as failure for having no answer to grade.

Output: research/results/counterbalanced_retrieval_<UTC timestamp>.json -
a NEW, timestamped file every run. Never overwrites
research/results/final_metrics.csv or any other historical result.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.stats import wilcoxon  # noqa: E402

from src.evaluation.metrics import document_aware_page_recall, ndcg_at_k, normalize, reciprocal_rank_at_k  # noqa: E402
from src.retrieval.hybrid_retriever import retrieve_hybrid  # noqa: E402
from src.retrieval.metadata_store import MetadataStore  # noqa: E402
from src.retrieval.retriever import retrieve as dense_retrieve_global  # noqa: E402
from src.embedding.embedder import embed_query  # noqa: E402
from src.retrieval.faiss_index import FaissIndex  # noqa: E402
from src import config  # noqa: E402

RESEARCH_DB = str(ROOT / "data" / "research_corpus" / "metadata.db")
RESEARCH_FAISS = str(ROOT / "data" / "research_corpus" / "faiss.index")
TOP_K = 5
LATENCY_REPS = 5


def dense_retrieve_scoped(query: str, document_id: str, top_k: int = TOP_K) -> list[dict]:
    """Dense-only, scoped to one document - built from the same primitives
    retrieve_hybrid() itself uses, NOT a modification of retriever.py."""
    store = MetadataStore(config.DB_PATH)
    index = FaissIndex(config.EMBEDDING_DIM, config.FAISS_INDEX_PATH)
    index.load()
    candidate_k = max(top_k, store.count())
    query_vector = embed_query(query)
    scores, ids = index.search(query_vector, candidate_k)
    candidates = [(int(i), float(s)) for i, s in zip(ids, scores) if i != -1]
    allowed = store.filter_chunk_ids_by_document([c for c, _ in candidates], document_id)
    candidates = [(c, s) for c, s in candidates if c in allowed][:top_k]
    meta = store.get_chunks([c for c, _ in candidates])
    return [{**meta[c], "score": s, "rank": r} for r, (c, s) in enumerate(candidates, 1) if c in meta]


def evidence_term_coverage(chunks: list[dict], expected_terms: list[str]) -> float:
    if not expected_terms:
        return 1.0
    text = normalize(" ".join(c["text"] for c in chunks))
    return sum(normalize(t) in text for t in expected_terms) / len(expected_terms)


def score_chunks(chunks, item) -> dict:
    return {
        "recall": document_aware_page_recall(chunks, item["expected_pages"], item["source"]),
        "mrr": reciprocal_rank_at_k(chunks, item["expected_pages"], item["source"], k=TOP_K),
        "ndcg": ndcg_at_k(chunks, item["expected_pages"], item["source"], k=TOP_K),
        "term_coverage": evidence_term_coverage(chunks, item["expected_terms"]),
    }


def time_retrieval(fn, reps: int) -> tuple[list[dict], dict]:
    """Runs fn() `reps` times, returns (chunks from the LAST call, latency
    stats in ms). The chunks from every rep must be identical (retrieval is
    deterministic over an unchanged index) - callers verify this."""
    samples = []
    all_chunks = []
    for _ in range(reps):
        t0 = time.perf_counter()
        chunks = fn()
        samples.append((time.perf_counter() - t0) * 1000)
        all_chunks.append([c["chunk_id"] for c in chunks])
    deterministic = all(c == all_chunks[0] for c in all_chunks)
    stats = {
        "mean_ms": mean(samples),
        "median_ms": median(samples),
        "stdev_ms": stdev(samples) if len(samples) > 1 else 0.0,
        "p95_ms": sorted(samples)[int(0.95 * (len(samples) - 1))],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "deterministic_across_reps": deterministic,
    }
    return chunks, stats


def warm_up():
    """Both methods, symmetrically, before any timed measurement."""
    dense_retrieve_global("warm up query", top_k=1)
    retrieve_hybrid("warm up query", top_k=1)


def paired_test(a: list[float], b: list[float]) -> dict:
    diffs = [y - x for x, y in zip(a, b)]
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return {"mean_diff": 0.0, "n_nonzero": 0, "p_value": None}
    stat, p = wilcoxon(nonzero)
    return {"mean_diff": mean(diffs), "n_nonzero": len(nonzero), "statistic": float(stat), "p_value": float(p)}


def main():
    config.DB_PATH = RESEARCH_DB
    config.FAISS_INDEX_PATH = RESEARCH_FAISS
    items = json.loads((ROOT / "data" / "healthcare_eval_questions.json").read_text(encoding="utf-8"))

    print("Warming up dense and hybrid paths symmetrically before any timed question...")
    warm_up()

    per_question = []
    for i, item in enumerate(items):
        dense_first = (i % 2 == 0)  # counterbalanced: alternates by question index, deterministic
        order = ["dense", "hybrid"] if dense_first else ["hybrid", "dense"]

        row = {"id": item["id"], "category": item["category"], "order": order}

        for method in order:
            # --- A: global (no document_id) ---
            fn = (lambda m=method: dense_retrieve_global(item["question"], top_k=TOP_K)) if method == "dense" \
                else (lambda: retrieve_hybrid(item["question"], top_k=TOP_K))
            chunks, timing = time_retrieval(fn, LATENCY_REPS)
            row[f"A_global_{method}"] = {**score_chunks(chunks, item), "latency": timing}

            # --- B: production-scoped (this question's own document) ---
            fn = (lambda: dense_retrieve_scoped(item["question"], item["document_id"], top_k=TOP_K)) if method == "dense" \
                else (lambda: retrieve_hybrid(item["question"], top_k=TOP_K, document_id=item["document_id"]))
            chunks, timing = time_retrieval(fn, LATENCY_REPS)
            row[f"B_scoped_{method}"] = {**score_chunks(chunks, item), "latency": timing}

        per_question.append(row)
        print(f"[{i + 1}/{len(items)}] {item['id']} order={order}")

    summary = {}
    for label in ("A_global", "B_scoped"):
        summary[label] = {}
        for metric in ("recall", "mrr", "ndcg", "term_coverage"):
            d = [r[f"{label}_dense"][metric] for r in per_question]
            h = [r[f"{label}_hybrid"][metric] for r in per_question]
            summary[label][metric] = {
                "dense_mean": mean(d), "hybrid_mean": mean(h),
                "paired_wilcoxon": paired_test(d, h),
            }
        for method in ("dense", "hybrid"):
            lat = [r[f"{label}_{method}"]["latency"]["mean_ms"] for r in per_question]
            summary[label][f"{method}_latency_ms"] = {
                "mean": mean(lat), "median": median(lat), "stdev": stdev(lat), "p95": sorted(lat)[int(0.95 * (len(lat) - 1))],
            }

    non_deterministic = [
        (r["id"], label, method)
        for r in per_question
        for label in ("A_global", "B_scoped")
        for method in ("dense", "hybrid")
        if not r[f"{label}_{method}"]["latency"]["deterministic_across_reps"]
    ]

    payload = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "top_k": TOP_K,
            "latency_reps": LATENCY_REPS,
            "counterbalancing": "even question index -> dense measured first; odd -> hybrid first",
            "question_count": len(items),
            "non_deterministic_repeats": non_deterministic,
        },
        "summary": summary,
        "per_question": per_question,
    }

    out_dir = ROOT / "research" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"counterbalanced_retrieval_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nnon-deterministic repeats found: {len(non_deterministic)} {non_deterministic}")
    print("\n=== SUMMARY ===")
    for label in ("A_global", "B_scoped"):
        print(f"\n{label}:")
        for metric in ("recall", "mrr", "ndcg", "term_coverage"):
            s = summary[label][metric]
            print(f"  {metric:<14} dense={s['dense_mean']:.3f} hybrid={s['hybrid_mean']:.3f} "
                  f"wilcoxon_p={s['paired_wilcoxon']['p_value']}")
        for method in ("dense", "hybrid"):
            s = summary[label][f"{method}_latency_ms"]
            print(f"  {method}_latency_ms  mean={s['mean']:.2f} median={s['median']:.2f} "
                  f"stdev={s['stdev']:.2f} p95={s['p95']:.2f}")

    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
