#!/usr/bin/env python
"""Clean, contamination-free evaluation harness.

Supersedes scripts/evaluate.py (kept unmodified as an audited historical
artifact - see its module docstring) for any evaluation whose numbers are
meant to be trusted. Differences, per the evaluation audit:

  - Uses src.generation.eval_prompt_builder (no scope/answer-key injection).
  - Uses src.evaluation.metrics (document-aware recall, no per-question
    aliases, errors scored and recorded rather than silently dropped).
  - Persists raw per-question results to JSON; the human-readable summary is
    generated FROM that JSON by write_summary(), never hand-typed.
  - Runs multiple methods interleaved per-question (not one full sequential
    pass per method), to avoid confounding latency comparisons with
    run-order/warm-up drift.

USAGE

  Reproduce the development/regression benchmark (self-referential; the
  corpus IS the system's own spec document - see README.md "Evaluation"):

    python scripts/run_evaluation.py \\
        --dataset data/eval_questions.json \\
        --source "RAG_Healthcare_Review1.pdf" \\
        --label dev-regression \\
        --output-json data/evaluation_results.json \\
        --output-summary data/evaluation_summary.txt

  Run the (future) publication healthcare corpus benchmark:

    python scripts/run_evaluation.py \\
        --dataset data/healthcare_eval_questions.json \\
        --label publication \\
        --output-json data/evaluation_results.publication.json \\
        --output-summary data/evaluation_summary.publication.txt

  Smoke-test on a handful of questions first with --limit N.

PRECONDITIONS
  - The FAISS/SQLite index at src.config.FAISS_INDEX_PATH / DB_PATH must
    already contain the document(s) the dataset's questions target.
  - Ollama must be running locally with the model configured in
    src/config.py (OLLAMA_MODEL) pulled.
  - Generation is not fully deterministic (temperature > 0, no fixed seed) -
    re-running will not reproduce byte-identical answers, only similar ones.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests

from src import config
from src.evaluation.aggregation import category_breakdown
from src.evaluation.metrics import aggregate, evaluate_question
from src.generation.eval_prompt_builder import build_eval_prompt
from src.generation.ollama_client import generate
from src.retrieval.hybrid_retriever import retrieve_hybrid
from src.retrieval.retriever import retrieve

METHOD_FNS = {"dense": retrieve, "hybrid": retrieve_hybrid}


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ollama_version() -> str | None:
    try:
        resp = requests.get(f"{config.OLLAMA_HOST}/api/version", timeout=5)
        resp.raise_for_status()
        return resp.json().get("version")
    except requests.RequestException:
        return None


def warm_up(methods: list[str]) -> None:
    print("Warming up embedding model / retrieval paths...")
    for name in methods:
        try:
            METHOD_FNS[name]("warmup query")
        except Exception as exc:
            print(f"  warm-up call for '{name}' raised {type(exc).__name__}: {exc}")
    print("Warm-up complete.\n")


def run_interleaved(
    dataset: list[dict],
    methods: list[str],
    default_source: str | None,
) -> dict[str, list[dict]]:
    """For each question in turn, run every requested method back-to-back
    before moving to the next question - avoids the confound of one full
    sequential dense pass followed by one full sequential hybrid pass.
    """
    results: dict[str, list[dict]] = {name: [] for name in methods}

    for i, item in enumerate(dataset, start=1):
        print(f"[{i}/{len(dataset)}] {item['id']}: {item['question']}")
        for name in methods:
            result = evaluate_question(
                item,
                retrieval_fn=METHOD_FNS[name],
                prompt_fn=build_eval_prompt,
                generate_fn=generate,
                default_source=default_source,
            )
            results[name].append(result)
            print(
                f"    {name:<7} failure_type={result['failure_type']:<22} "
                f"recall={result['retrieval_recall']:.2f} "
                f"coverage={result['term_coverage']:.2f}"
            )

    return results


def write_summary(results_json_path: Path, summary_txt_path: Path) -> None:
    """Generate the human-readable summary FROM the persisted raw JSON.

    Deliberately reads the file back from disk (rather than reusing in-memory
    data) so the summary is provably derived from what was actually
    persisted, not from a separate in-process code path.
    """
    with results_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    meta = payload["meta"]
    lines = []
    lines.append("RAG Healthcare Evaluation Summary")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Label:              {meta['label']}")
    lines.append(f"Generated by:       scripts/run_evaluation.py:write_summary() (code-generated, not hand-typed)")
    lines.append(f"Generated at (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Source raw file:    {results_json_path.name}")
    lines.append(f"Dataset:            {meta['dataset_path']}")
    lines.append(f"Dataset SHA-256:    {meta['dataset_sha256']}")
    lines.append(f"Question count:     {meta['question_count']}")
    lines.append(f"Evaluation prompt:  src.generation.eval_prompt_builder.build_eval_prompt "
                 f"(neutral - no scope/answer-key injection)")
    lines.append("")
    lines.append("Configuration:")
    for k, v in meta["config"].items():
        lines.append(f"  {k}: {v}")
    lines.append(f"  ollama_version: {meta.get('ollama_version')}")
    lines.append("")
    lines.append("NOTE: generation is not fully deterministic (temperature > 0, no fixed")
    lines.append("seed) and this reflects a single run - treat as one sample, not a")
    lines.append("statistically stable estimate. See README.md Evaluation section.")
    lines.append("")

    for method_name, method_results in payload["results"].items():
        summary = aggregate(method_results)
        lines.append("=" * 70)
        lines.append(f"{method_name.upper()} SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Attempted:                {summary['attempted']}")
        lines.append(f"Scored:                   {summary['scored']}  (errors are scored as 0, never excluded)")
        lines.append(f"Passes:                   {summary['passes']}")
        lines.append(f"Retrieval failures:       {summary['retrieval_failures']}  "
                      f"(content miss: {summary['retrieval_content_misses']}, "
                      f"error: {summary['retrieval_errors']})")
        lines.append(f"Generation failures:      {summary['generation_failures']}  "
                      f"(content miss: {summary['generation_content_misses']}, "
                      f"error: {summary['generation_errors']})")
        lines.append(f"Mean Retrieval Recall:    {summary['mean_retrieval_recall']:.3f}"
                      if summary['mean_retrieval_recall'] is not None else "Mean Retrieval Recall:    n/a")
        lines.append(f"Mean Term Coverage:       {summary['mean_term_coverage']:.3f}"
                      if summary['mean_term_coverage'] is not None else "Mean Term Coverage:       n/a")
        lines.append(f"Mean MRR@5:               {summary['mean_mrr_at_5']:.3f}"
                      if summary['mean_mrr_at_5'] is not None else "Mean MRR@5:               n/a")
        lines.append(f"Mean nDCG@5:              {summary['mean_ndcg_at_5']:.3f}"
                      if summary['mean_ndcg_at_5'] is not None else "Mean nDCG@5:              n/a")
        lines.append(f"Latency sample size:      {summary['timed_count']} of {summary['attempted']} attempted")
        if summary["mean_retrieval_ms"] is not None:
            lines.append(f"Mean Retrieval Latency:   {summary['mean_retrieval_ms']:.0f} ms")
            lines.append(f"Mean Generation Latency:  {summary['mean_generation_ms']:.0f} ms")
            lines.append(f"Mean End-to-End Latency:  {summary['mean_total_ms']:.0f} ms")
        lines.append("")

    if len(payload["results"]) >= 2:
        names = list(payload["results"].keys())
        lines.append("=" * 70)
        lines.append(f"COMPARISON ({' vs '.join(n.upper() for n in names)})")
        lines.append("=" * 70)
        summaries = {n: aggregate(payload["results"][n]) for n in names}
        lines.append(f"{'Metric':<28}" + "".join(f"{n.upper():>15}" for n in names))
        lines.append("-" * (28 + 15 * len(names)))
        for metric_key, metric_label in [
            ("mean_retrieval_recall", "Mean Retrieval Recall"),
            ("mean_term_coverage", "Mean Term Coverage"),
            ("mean_mrr_at_5", "Mean MRR@5"),
            ("mean_ndcg_at_5", "Mean nDCG@5"),
            ("mean_retrieval_ms", "Mean Retrieval (ms)"),
            ("mean_generation_ms", "Mean Generation (ms)"),
            ("mean_total_ms", "Mean End-to-End (ms)"),
        ]:
            row = f"{metric_label:<28}"
            for n in names:
                val = summaries[n][metric_key]
                row += f"{val:>15.3f}" if val is not None else f"{'n/a':>15}"
            lines.append(row)
        lines.append("")

    lines.append("=" * 70)
    lines.append("BY CATEGORY (mean recall/coverage per question category)")
    lines.append("=" * 70)
    method_names = list(payload["results"].keys())
    breakdowns = {m: category_breakdown([payload], m) for m in method_names}
    categories = sorted({c for b in breakdowns.values() for c in b})
    header = f"{'Category':<24}"
    for m in method_names:
        header += f"{m.upper() + ' N':>8}{m.upper() + ' Rec':>10}{m.upper() + ' Cov':>10}  "
    lines.append(header)
    lines.append("-" * len(header))
    for cat in categories:
        row = f"{cat:<24}"
        for m in method_names:
            b = breakdowns[m].get(cat, {"n": 0, "mean_recall": None, "mean_coverage": None})
            row += f"{b['n']:>8}"
            row += f"{b['mean_recall']:>10.2f}" if b["mean_recall"] is not None else f"{'n/a':>10}"
            row += f"{b['mean_coverage']:>10.2f}" if b["mean_coverage"] is not None else f"{'n/a':>10}"
            row += "  "
        lines.append(row)
    lines.append("")

    summary_txt_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clean, contamination-free evaluation harness.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--db-path", default=None,
        help="SQLite metadata DB to query (default: production config.DB_PATH). "
             "Use to point at a separate research corpus index without touching production data.",
    )
    parser.add_argument(
        "--faiss-path", default=None,
        help="FAISS index to query (default: production config.FAISS_INDEX_PATH).",
    )
    parser.add_argument("--source", default=None,
                         help="Default expected source/document for questions that don't carry their own 'source' field")
    parser.add_argument("--methods", default="dense,hybrid")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions (smoke test)")
    parser.add_argument("--label", required=True, help="Short label identifying this run, e.g. dev-regression")
    parser.add_argument("--output-json", type=Path, default=ROOT / "data" / "evaluation_results.json")
    parser.add_argument("--output-summary", type=Path, default=ROOT / "data" / "evaluation_summary.txt")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHOD_FNS:
            parser.error(f"Unknown method '{m}'. Choices: {list(METHOD_FNS)}")

    # retrieve()/retrieve_hybrid() both read config.DB_PATH/FAISS_INDEX_PATH
    # dynamically at call time (via `from .. import config`), so redirecting
    # these module attributes points retrieval at a different index (e.g. the
    # separate research corpus) without modifying retriever.py/hybrid_retriever.py
    # or touching the production index.
    if args.db_path:
        config.DB_PATH = args.db_path
    if args.faiss_path:
        config.FAISS_INDEX_PATH = args.faiss_path

    dataset = load_dataset(args.dataset)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    print(f"Dataset: {args.dataset} ({len(dataset)} questions, label='{args.label}')")
    warm_up(methods)

    start = time.time()
    results = run_interleaved(dataset, methods, args.source)
    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s")

    payload = {
        "meta": {
            "label": args.label,
            "dataset_path": str(args.dataset),
            "dataset_sha256": dataset_sha256(args.dataset),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "methods": methods,
            "question_count": len(dataset),
            "default_expected_source": args.source,
            "elapsed_seconds": elapsed,
            "db_path": config.DB_PATH,
            "faiss_index_path": config.FAISS_INDEX_PATH,
            "config": {
                "embedding_model": config.EMBEDDING_MODEL_NAME,
                "embedding_dim": config.EMBEDDING_DIM,
                "chunk_size_words": config.CHUNK_SIZE_WORDS,
                "chunk_overlap_words": config.CHUNK_OVERLAP_WORDS,
                "top_k": config.TOP_K,
                "hybrid_candidate_k": 10,
                "hybrid_rrf_k": 60,
                "hybrid_dense_weight": 2.0,
                "hybrid_keyword_weight": 1.0,
                "ollama_model": config.OLLAMA_MODEL,
                "ollama_temperature": config.OLLAMA_TEMPERATURE,
                "ollama_top_p": config.OLLAMA_TOP_P,
                "ollama_num_predict": config.OLLAMA_NUM_PREDICT,
            },
            "ollama_version": ollama_version(),
        },
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Raw results written to {args.output_json}")

    write_summary(args.output_json, args.output_summary)
    print(f"Summary written to {args.output_summary}")


if __name__ == "__main__":
    main()
