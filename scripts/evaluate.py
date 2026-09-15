"""LEGACY evaluation harness - kept unmodified as an audited historical artifact.

STATUS: superseded by scripts/run_evaluation.py. Do not cite numbers produced
by this script as research/publication evidence.

The evaluation audit (see project history) found this script's methodology
invalid for that purpose:
  - Its dataset (data/eval_questions.json) is entirely derived from the
    project's own specification document, not an independent healthcare
    corpus - development/regression signal only.
  - Its prompt path (src.generation.prompt_builder) contains question-specific
    scope/term injection that can hand a subset of questions their own
    expected answer as "retrieved evidence" - contamination, not measurement.
  - page_recall() below matches page number only, with no document/source
    check - unsafe now that the index holds more than one document.
  - A question that raises during evaluation is silently dropped from
    run_evaluation()'s results list rather than being scored as a failure,
    so its exception never affects the reported averages.
  - term_coverage()/matched_terms() include a hard-coded alias for exactly
    one known question's expected term.

scripts/run_evaluation.py (using src/evaluation/metrics.py and
src/generation/eval_prompt_builder.py) fixes all of the above: document-aware
recall, a neutral contamination-free prompt, errors scored and recorded
rather than dropped, no per-question aliases, and machine-readable raw
output. This file is left as-is, unexecuted-on-by-default, purely so the
exact code that was audited remains inspectable. See README.md "Evaluation"
for details and reproduction commands for the replacement.
"""

import json
import sys
import time
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.retriever import retrieve
from src.retrieval.hybrid_retriever import retrieve_hybrid
from src.generation.prompt_builder import build_prompt
from src.generation.ollama_client import generate

DATASET = ROOT / "data" / "eval_questions.json"


CATEGORY_RANGES = {
    "Problem / Objectives": range(1, 17),
    "RAG Concepts": range(17, 25),
    "Literature Survey": range(25, 31),
    "Methodology": range(31, 37),
    "Architecture": range(37, 43),
    "Workflow": range(43, 48),
    "Conclusion": range(48, 51),
}


def load_dataset() -> list[dict]:
    with DATASET.open("r", encoding="utf-8") as f:
        return json.load(f)

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())

def category_for_question(question_id: str) -> str:
    number = int(question_id[1:])

    for category, question_range in CATEGORY_RANGES.items():
        if number in question_range:
            return category

    return "Uncategorized"


def page_recall(chunks: list[dict], expected_pages: list[int]) -> float:
    retrieved_pages = {int(c["page"]) for c in chunks}
    expected = set(expected_pages)

    if not expected:
        return 1.0

    return len(retrieved_pages & expected) / len(expected)

def matched_terms(
    answer: str,
    expected_terms: list[str],
) -> tuple[list[str], list[str]]:
    answer_norm = normalize(answer)

    aliases = {
        "stores embeddings for similarity search": [
            "similarity search",
            "stores embeddings",
            "embedding storage",
        ],
    }

    matched = []
    missing = []

    for term in expected_terms:
        term_norm = normalize(term)

        if term_norm in answer_norm:
            matched.append(term)
            continue

        alias_match = any(
            normalize(alias) in answer_norm
            for alias in aliases.get(term.lower(), [])
        )

        if alias_match:
            matched.append(term)
        else:
            missing.append(term)

    return matched, missing

def term_coverage(
    answer: str,
    expected_terms: list[str],
) -> float:
    if not expected_terms:
        return 1.0

    answer_norm = normalize(answer)
    matched = 0

    for term in expected_terms:
        term_norm = normalize(term)

        if term_norm in answer_norm:
            matched += 1
            continue

        # Basic semantic-equivalent patterns for common RAG wording.
        aliases = {
            "stores embeddings for similarity search": [
                "similarity search",
                "stores embeddings",
                "embedding storage",
            ],
        }

        alias_match = any(
            normalize(alias) in answer_norm
            for alias in aliases.get(term.lower(), [])
        )

        if alias_match:
            matched += 1

    return matched / len(expected_terms)

def evaluate_question(item: dict, retrieval_fn) -> dict:
    question = item["question"]

    # Retrieval
    retrieval_start = time.perf_counter()
    chunks = retrieval_fn(question)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    expected_pages = item.get("expected_pages", [])
    retrieved_pages = sorted({int(c["page"]) for c in chunks})

    missing_pages = [
        page
        for page in expected_pages
        if page not in retrieved_pages
    ]

    # Generation
    prompt = build_prompt(question, chunks)

    generation_start = time.perf_counter()
    answer = generate(prompt)
    generation_ms = (time.perf_counter() - generation_start) * 1000

    matched, missing_terms = matched_terms(
        answer,
        item.get("expected_terms", []),
    )

    recall = page_recall(
        chunks,
        expected_pages,
    )

    coverage = term_coverage(
        answer,
        item.get("expected_terms", []),
    )

    # Failure classification
    if missing_pages:
        failure_type = "RETRIEVAL MISS"
    elif missing_terms:
        failure_type = "GENERATION MISS"
    else:
        failure_type = "PASS"

    total_ms = retrieval_ms + generation_ms

    return {
        "id": item["id"],
        "category": category_for_question(item["id"]),
        "question": question,

        "expected_pages": expected_pages,
        "retrieved_pages": retrieved_pages,
        "missing_pages": missing_pages,

        "expected_terms": item.get("expected_terms", []),
        "matched_terms": matched,
        "missing_terms": missing_terms,

        "retrieval_recall": recall,
        "term_coverage": coverage,

        "failure_type": failure_type,

        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "total_ms": total_ms,

        "answer": answer,
    }


def run_evaluation(
    dataset: list[dict],
    method_name: str,
    retrieval_fn,
) -> list[dict]:

    print(f"\n{'#' * 70}")
    print(f"{method_name.upper()} EVALUATION")
    print(f"{'#' * 70}")

    results = []

    for item in dataset:
        print(f"\n{'=' * 70}")
        print(f"{item['id']}: {item['question']}")
        print(f"{'=' * 70}")

        try:
            result = evaluate_question(
                item,
                retrieval_fn,
            )

            results.append(result)

            print(
                f"Category:          "
                f"{result['category']}"
            )

            print(
                f"Retrieval Recall:  "
                f"{result['retrieval_recall']:.2f}"
            )

            print(
                f"Term Coverage:     "
                f"{result['term_coverage']:.2f}"
            )

            print(
                f"Retrieval Time:    "
                f"{result['retrieval_ms']:.0f} ms"
            )

            print(
                f"Generation Time:   "
                f"{result['generation_ms']:.0f} ms"
            )

            print(
                f"Total Time:        "
                f"{result['total_ms']:.0f} ms"
            )

            if result["failure_type"] != "PASS":
                print(
                    f"FAILURE TYPE:      "
                    f"{result['failure_type']}"
                )

                print(
                    f"Expected pages:    "
                    f"{result['expected_pages']}"
                )

                print(
                    f"Retrieved pages:   "
                    f"{result['retrieved_pages']}"
                )

                print(
                    f"Missing pages:     "
                    f"{result['missing_pages']}"
                )

                print(
                    f"Expected terms:    "
                    f"{result['expected_terms']}"
                )

                print(
                    f"Matched terms:     "
                    f"{result['matched_terms']}"
                )

                print(
                    f"Missing terms:     "
                    f"{result['missing_terms']}"
                )

        except Exception as exc:
            print(f"ERROR: {exc}")

    return results


def summarize(results: list[dict]) -> dict:
    return {
        "questions": len(results),
        "recall": mean(
            r["retrieval_recall"]
            for r in results
        ),
        "coverage": mean(
            r["term_coverage"]
            for r in results
        ),
        "retrieval_ms": mean(
            r["retrieval_ms"]
            for r in results
        ),
        "generation_ms": mean(
            r["generation_ms"]
            for r in results
        ),
        "total_ms": mean(
            r["total_ms"]
            for r in results
        ),
    }


def print_summary(
    method_name: str,
    results: list[dict],
) -> dict:

    if not results:
        print(
            f"\nNo successful evaluations "
            f"for {method_name}."
        )
        return {}

    summary = summarize(results)

    print(f"\n{'=' * 70}")
    print(f"{method_name.upper()} SUMMARY")
    print(f"{'=' * 70}")

    print(
        f"Questions evaluated:   "
        f"{summary['questions']}"
    )

    print(
        f"Average Recall@K:      "
        f"{summary['recall']:.2f}"
    )

    print(
        f"Average Term Coverage: "
        f"{summary['coverage']:.2f}"
    )

    print(
        f"Average Retrieval:     "
        f"{summary['retrieval_ms']:.0f} ms"
    )

    print(
        f"Average Generation:    "
        f"{summary['generation_ms']:.0f} ms"
    )

    print(
        f"Average End-to-End:    "
        f"{summary['total_ms']:.0f} ms"
    )

    return summary


def print_failure_summary(
    method_name: str,
    results: list[dict],
) -> None:

    passes = sum(
        1
        for r in results
        if r["failure_type"] == "PASS"
    )

    retrieval_misses = sum(
        1
        for r in results
        if r["failure_type"] == "RETRIEVAL MISS"
    )

    generation_misses = sum(
        1
        for r in results
        if r["failure_type"] == "GENERATION MISS"
    )

    print(f"\n{'=' * 70}")
    print(f"{method_name.upper()} FAILURE ANALYSIS")
    print(f"{'=' * 70}")

    print(f"PASS:              {passes}")
    print(f"RETRIEVAL MISS:    {retrieval_misses}")
    print(f"GENERATION MISS:   {generation_misses}")

    print("\nFailed questions:")

    for result in results:
        if result["failure_type"] == "PASS":
            continue

        print(
            f"{result['id']} | "
            f"{result['failure_type']} | "
            f"Recall={result['retrieval_recall']:.2f} | "
            f"Coverage={result['term_coverage']:.2f}"
        )


def print_category_summary(
    method_name: str,
    results: list[dict],
) -> None:

    print(f"\n{'=' * 70}")
    print(
        f"{method_name.upper()} CATEGORY RESULTS"
    )
    print(f"{'=' * 70}")

    print(
        f"{'Category':<25}"
        f"{'N':>5}"
        f"{'Recall@K':>12}"
        f"{'Coverage':>12}"
    )

    print("-" * 54)

    for category in CATEGORY_RANGES:

        category_results = [
            r
            for r in results
            if r["category"] == category
        ]

        if not category_results:
            continue

        recall = mean(
            r["retrieval_recall"]
            for r in category_results
        )

        coverage = mean(
            r["term_coverage"]
            for r in category_results
        )

        print(
            f"{category:<25}"
            f"{len(category_results):>5}"
            f"{recall:>12.2f}"
            f"{coverage:>12.2f}"
        )


def print_comparison(
    dense_summary: dict,
    hybrid_summary: dict,
) -> None:

    print(f"\n{'=' * 70}")
    print("DENSE VS HYBRID COMPARISON")
    print(f"{'=' * 70}")

    print(
        f"{'Metric':<25}"
        f"{'Dense':>15}"
        f"{'Hybrid':>15}"
    )

    print("-" * 55)

    print(
        f"{'Recall@K':<25}"
        f"{dense_summary['recall']:>15.2f}"
        f"{hybrid_summary['recall']:>15.2f}"
    )

    print(
        f"{'Term Coverage':<25}"
        f"{dense_summary['coverage']:>15.2f}"
        f"{hybrid_summary['coverage']:>15.2f}"
    )

    print(
        f"{'Retrieval Time (ms)':<25}"
        f"{dense_summary['retrieval_ms']:>15.0f}"
        f"{hybrid_summary['retrieval_ms']:>15.0f}"
    )

    print(
        f"{'Generation Time (ms)':<25}"
        f"{dense_summary['generation_ms']:>15.0f}"
        f"{hybrid_summary['generation_ms']:>15.0f}"
    )

    print(
        f"{'End-to-End Time (ms)':<25}"
        f"{dense_summary['total_ms']:>15.0f}"
        f"{hybrid_summary['total_ms']:>15.0f}"
    )


def print_category_comparison(
    dense_results: list[dict],
    hybrid_results: list[dict],
) -> None:

    print(f"\n{'=' * 70}")
    print("DENSE VS HYBRID BY CATEGORY")
    print(f"{'=' * 70}")

    print(
        f"{'Category':<25}"
        f"{'Dense R':>10}"
        f"{'Hybrid R':>10}"
        f"{'Dense C':>10}"
        f"{'Hybrid C':>10}"
    )

    print("-" * 65)

    for category in CATEGORY_RANGES:

        dense_category = [
            r
            for r in dense_results
            if r["category"] == category
        ]

        hybrid_category = [
            r
            for r in hybrid_results
            if r["category"] == category
        ]

        if not dense_category or not hybrid_category:
            continue

        dense_recall = mean(
            r["retrieval_recall"]
            for r in dense_category
        )

        hybrid_recall = mean(
            r["retrieval_recall"]
            for r in hybrid_category
        )

        dense_coverage = mean(
            r["term_coverage"]
            for r in dense_category
        )

        hybrid_coverage = mean(
            r["term_coverage"]
            for r in hybrid_category
        )

        print(
            f"{category:<25}"
            f"{dense_recall:>10.2f}"
            f"{hybrid_recall:>10.2f}"
            f"{dense_coverage:>10.2f}"
            f"{hybrid_coverage:>10.2f}"
        )


def main() -> None:
    dataset = load_dataset()

    print(
        f"Evaluation questions loaded: "
        f"{len(dataset)}"
    )

    print("\nWarming up embedding model...")

    retrieve("warmup query")
    retrieve_hybrid("warmup query")

    print("Warm-up complete.\n")

    dense_results = run_evaluation(
        dataset,
        "Dense Retrieval",
        retrieve,
    )

    hybrid_results = run_evaluation(
        dataset,
        "Hybrid Retrieval",
        retrieve_hybrid,
    )

    dense_summary = print_summary(
        "Dense Retrieval",
        dense_results,
    )

    hybrid_summary = print_summary(
        "Hybrid Retrieval",
        hybrid_results,
    )

    if not dense_summary or not hybrid_summary:
        return

    print_failure_summary(
        "Dense Retrieval",
        dense_results,
    )

    print_failure_summary(
        "Hybrid Retrieval",
        hybrid_results,
    )

    print_category_summary(
        "Dense Retrieval",
        dense_results,
    )

    print_category_summary(
        "Hybrid Retrieval",
        hybrid_results,
    )

    print_comparison(
        dense_summary,
        hybrid_summary,
    )

    print_category_comparison(
        dense_results,
        hybrid_results,
    )


if __name__ == "__main__":
    main()