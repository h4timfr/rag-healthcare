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

  C_production         app.retrieve_and_build_prompt() exactly as a
                       single-document chat turn calls it (scoped to the
                       question's document, default top_k): intent
                       classification, 5 -> 8 widening for explicit
                       three-component lists, per-part multi-part retrieval
                       and the presentation reorder are all included, so
                       metrics are computed on the list in the order the
                       model actually reads it. Recall is reported both on
                       its first 5 chunks (recall) and on everything it
                       returned (recall_all_returned). Latency here includes
                       prompt construction. Generation is NOT included.

Measurement order rotates over the three methods as a cyclic Latin square
(question index mod 3), so each method is measured first, second and third
equally often across the 45 questions.

Statistics: each paired comparison is a two-sided Wilcoxon signed-rank test
over the questions whose paired difference is nonzero (zero differences
dropped), with a matched-pairs rank-biserial correlation as effect size and
win/loss/tie counts. Holm-Bonferroni correction is applied across every
metric comparison in the run (latency is reported descriptively only).

End-to-end evaluation with generation is intentionally NOT re-implemented
here - that is scripts/run_product_qa.py.

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


def production_retrieve(query: str, document_id: str) -> list[dict]:
    import app

    chunks, _prompt = app.retrieve_and_build_prompt(query, top_k=config.TOP_K, document_id=document_id)
    return chunks


def warm_up():
    """Every method, before any timed measurement."""
    dense_retrieve_global("warm up query", top_k=1)
    retrieve_hybrid("warm up query", top_k=1)
    production_retrieve("warm up query", None)


def paired_test(a: list[float], b: list[float]) -> dict:
    """b - a. Zero differences are dropped before the Wilcoxon test (the
    'wilcox' zero_method). rank_biserial is (W+ - W-) / (W+ + W-) over the
    ranks of the nonzero |differences|: +1 means every nonzero difference
    favours b."""
    from scipy.stats import rankdata

    diffs = [y - x for x, y in zip(a, b)]
    nonzero = [d for d in diffs if d != 0]
    result = {
        "n_pairs": len(diffs),
        "mean_diff": mean(diffs),
        "median_diff": median(diffs),
        "wins_b": sum(d > 0 for d in diffs),
        "wins_a": sum(d < 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
        "n_nonzero": len(nonzero),
    }
    if not nonzero:
        return {**result, "statistic": None, "p_value": None, "rank_biserial": 0.0}
    ranks = rankdata([abs(d) for d in nonzero])
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    res = wilcoxon(nonzero)
    return {
        **result,
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "rank_biserial": float((w_plus - w_minus) / (w_plus + w_minus)),
    }


def holm(p_values: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for {name: p}; None entries skipped."""
    named = sorted(((p, n) for n, p in p_values.items() if p is not None))
    m = len(named)
    adjusted, running = {}, 0.0
    for i, (p, name) in enumerate(named):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def environment() -> dict:
    import platform
    from importlib.metadata import version

    import requests

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            p: version(p) for p in ("faiss-cpu", "sentence-transformers", "torch", "numpy", "scipy", "pymupdf")
        },
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "embedding_dim": config.EMBEDDING_DIM,
        "chunk_size_words": config.CHUNK_SIZE_WORDS,
        "chunk_overlap_words": config.CHUNK_OVERLAP_WORDS,
        "top_k": config.TOP_K,
        "ollama_model": config.OLLAMA_MODEL,
    }
    import inspect
    import re

    defaults = inspect.signature(retrieve_hybrid).parameters
    env["candidate_k"] = defaults["candidate_k"].default
    env["rrf_k"] = defaults["rrf_k"].default
    source = inspect.getsource(retrieve_hybrid)
    for name in ("dense_weight", "keyword_weight"):
        found = re.search(rf"{name} = ([0-9.]+)", source)
        env[name] = float(found.group(1)) if found else None
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - informational only
        pass
    try:
        env["ollama_version"] = requests.get(f"{config.OLLAMA_HOST}/api/version", timeout=3).json().get("version")
    except Exception:  # noqa: BLE001 - retrieval-only run; Ollama is not required
        env["ollama_version"] = None
    return env


METHODS = ("dense", "hybrid", "production")
METRICS = ("recall", "mrr", "ndcg", "term_coverage")
COMPARISONS = (
    ("A_global", "A_global_dense", "A_global_hybrid"),
    ("B_scoped", "B_scoped_dense", "B_scoped_hybrid"),
    ("C_vs_B", "B_scoped_hybrid", "C_production"),
)


def main():
    config.DB_PATH = RESEARCH_DB
    config.FAISS_INDEX_PATH = RESEARCH_FAISS
    items = json.loads((ROOT / "data" / "healthcare_eval_questions.json").read_text(encoding="utf-8"))
    started = time.perf_counter()

    print("Warming up every method before any timed question...")
    warm_up()

    per_question = []
    for i, item in enumerate(items):
        shift = i % len(METHODS)
        order = list(METHODS[shift:] + METHODS[:shift])

        row = {"id": item["id"], "category": item["category"], "order": order}
        q, doc = item["question"], item["document_id"]

        for method in order:
            if method == "production":
                chunks, timing = time_retrieval(lambda: production_retrieve(q, doc), LATENCY_REPS)
                row["C_production"] = {
                    **score_chunks(chunks[:TOP_K], item),
                    "recall_all_returned": document_aware_page_recall(chunks, item["expected_pages"], item["source"]),
                    "returned": len(chunks),
                    "latency": timing,
                }
                continue

            # --- A: global (no document_id) ---
            fn = (lambda: dense_retrieve_global(q, top_k=TOP_K)) if method == "dense" \
                else (lambda: retrieve_hybrid(q, top_k=TOP_K))
            chunks, timing = time_retrieval(fn, LATENCY_REPS)
            row[f"A_global_{method}"] = {**score_chunks(chunks, item), "latency": timing}

            # --- B: scoped to this question's own document ---
            fn = (lambda: dense_retrieve_scoped(q, doc, top_k=TOP_K)) if method == "dense" \
                else (lambda: retrieve_hybrid(q, top_k=TOP_K, document_id=doc))
            chunks, timing = time_retrieval(fn, LATENCY_REPS)
            row[f"B_scoped_{method}"] = {**score_chunks(chunks, item), "latency": timing}

        per_question.append(row)
        print(f"[{i + 1}/{len(items)}] {item['id']} order={order} elapsed={time.perf_counter() - started:.0f}s", flush=True)

    labels = ("A_global_dense", "A_global_hybrid", "B_scoped_dense", "B_scoped_hybrid", "C_production")
    summary = {"means": {}, "latency_ms": {}, "comparisons": {}}
    for label in labels:
        summary["means"][label] = {m: mean(r[label][m] for r in per_question) for m in METRICS}
        lat = [r[label]["latency"]["mean_ms"] for r in per_question]
        summary["latency_ms"][label] = {
            "mean": mean(lat), "median": median(lat), "stdev": stdev(lat), "p95": sorted(lat)[int(0.95 * (len(lat) - 1))],
        }
    summary["means"]["C_production"]["recall_all_returned"] = mean(r["C_production"]["recall_all_returned"] for r in per_question)
    summary["means"]["C_production"]["mean_chunks_returned"] = mean(r["C_production"]["returned"] for r in per_question)

    raw_p = {}
    for name, a, b in COMPARISONS:
        for metric in METRICS:
            test = paired_test([r[a][metric] for r in per_question], [r[b][metric] for r in per_question])
            summary["comparisons"][f"{name}:{metric}"] = {"a": a, "b": b, **test}
            raw_p[f"{name}:{metric}"] = test["p_value"]
    for key, adj in holm(raw_p).items():
        summary["comparisons"][key]["p_holm"] = adj

    categories = sorted({r["category"] for r in per_question})
    summary["per_category"] = {
        cat: {
            "n": sum(r["category"] == cat for r in per_question),
            **{
                label: {m: mean(r[label][m] for r in per_question if r["category"] == cat) for m in METRICS}
                for label in labels
            },
        }
        for cat in categories
    }

    non_deterministic = [
        (r["id"], label)
        for r in per_question
        for label in labels
        if not r[label]["latency"]["deterministic_across_reps"]
    ]

    payload = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.perf_counter() - started,
            "top_k": TOP_K,
            "latency_reps": LATENCY_REPS,
            "counterbalancing": "cyclic Latin square over (dense, hybrid, production) by question index mod 3",
            "statistics": "two-sided Wilcoxon signed-rank on nonzero paired differences; "
                          "matched-pairs rank-biserial effect size; Holm correction across all metric comparisons",
            "relevance": "document-aware page-level relevance from expected_pages (see research/chunk_evidence_audit.py "
                         "for the chunk-level audit of the same labels)",
            "question_count": len(items),
            "non_deterministic_repeats": non_deterministic,
            "environment": environment(),
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
    print("\n=== MEANS ===")
    for label in labels:
        s = summary["means"][label]
        lat = summary["latency_ms"][label]
        print(f"  {label:<16} " + " ".join(f"{m}={s[m]:.3f}" for m in METRICS)
              + f"  latency median={lat['median']:.1f}ms mean={lat['mean']:.1f}ms")
    print("\n=== PAIRED COMPARISONS (b - a) ===")
    for key, c in summary["comparisons"].items():
        p = "n/a" if c["p_value"] is None else f"{c['p_value']:.4f}"
        ph = "n/a" if c.get("p_holm") is None else f"{c['p_holm']:.4f}"
        print(f"  {key:<24} mean_diff={c['mean_diff']:+.3f} wins b/a/ties={c['wins_b']}/{c['wins_a']}/{c['ties']} "
              f"r_rb={c['rank_biserial']:+.2f} p={p} p_holm={ph}")

    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
