#!/usr/bin/env python
"""CLI: run the production-only answer-quality QA harness
(src/evaluation/product_qa.py) against the app's real RAG pipeline and the
existing production-indexed documents, then print a baseline report.

This is separate from the publication/research benchmark
(scripts/run_evaluation.py) - it exercises app.py's actual single/multi-
document/follow-up behavior with real generated text, not retrieval-only
metrics over a large corpus.

Usage:
    python scripts/run_product_qa.py
    python scripts/run_product_qa.py --dataset data/product_qa.json --output output/product_qa_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.evaluation.product_qa import (
    aggregate,
    load_dataset,
    resolve_document_ids,
    run_case,
    worst_failures,
)
from src.exceptions import RagError


def _fmt_pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def print_report(dataset: dict, summary: dict, case_results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("PRODUCT QA BASELINE REPORT")
    print("=" * 78)

    print(f"\nQA cases: {summary['n_cases']}   QA turns: {summary['n_turns']}")
    print("\nCategory counts:")
    for cat, count in summary["category_counts"].items():
        print(f"  {cat:16s} {count}")

    o = summary["overall"]
    print("\nOVERALL METRICS")
    print(f"  pass rate                 : {o['passed']}/{o['n_turns']} ({_fmt_pct(o['pass_rate'])})")
    print(f"  answer generated ok       : {_fmt_pct(o['generated_ok_rate'])}")
    print(f"  avg required-fact coverage: {_fmt_pct(o['avg_fact_coverage'])}")
    print(f"  avg required-doc coverage : {_fmt_pct(o['avg_document_coverage'])}")
    print(f"  source attribution present: {_fmt_pct(o['source_attribution_rate'])}")
    print(f"  validation clean          : {_fmt_pct(o['validation_ok_rate'])}")
    print(f"  latency avg/p50/p95 (ms)  : {o['avg_latency_ms']:.0f} / {o['p50_latency_ms']:.0f} / {o['p95_latency_ms']:.0f}")

    issue_counts = summary["validation_issue_counts"]
    print("\nVALIDATION ISSUES (deterministic post-generation checks)")
    if not issue_counts:
        print("  (none)")
    for issue, count in issue_counts.items():
        print(f"  {issue:28s} {count}")

    print("\nPER-CATEGORY METRICS")
    header = (
        f"  {'category':16s} {'n':>3s} {'pass':>6s} {'facts':>6s} {'docs':>6s} "
        f"{'srcs':>6s} {'valid':>6s} {'p50ms':>7s} {'p95ms':>7s}"
    )
    print(header)
    for cat, m in summary["by_category"].items():
        print(
            f"  {cat:16s} {m['n_turns']:3d} {_fmt_pct(m['pass_rate']):>6s} "
            f"{_fmt_pct(m['avg_fact_coverage']):>6s} {_fmt_pct(m['avg_document_coverage']):>6s} "
            f"{_fmt_pct(m['source_attribution_rate']):>6s} {_fmt_pct(m['validation_ok_rate']):>6s} "
            f"{m['p50_latency_ms']:7.0f} {m['p95_latency_ms']:7.0f}"
        )

    failures = worst_failures(case_results, limit=10)
    print(f"\nWORST FAILING EXAMPLES ({len(failures)} shown, sorted worst-first)")
    if not failures:
        print("  (none - every turn passed its automated checks)")
    for f in failures:
        print(f"\n  [{f['id']}] ({f['category']}) turn {f['turn_index']}  score={f['score']:.2f}")
        print(f"    Q: {f['question']}")
        if f["error"]:
            print(f"    ERROR: {f['error']}")
        else:
            snippet = (f["answer"] or "")[:220].replace("\n", " ")
            print(f"    A: {snippet}{'...' if len(f['answer'] or '') > 220 else ''}")
        m = f["metrics"]
        reasons = []
        if not m["generated_ok"]:
            reasons.append("generation failed")
        if m.get("fact_coverage") is not None and m["fact_coverage"] < 0.5:
            reasons.append(f"low fact coverage ({_fmt_pct(m['fact_coverage'])})")
        if m.get("document_coverage") is not None and m["document_coverage"] < 0.5:
            reasons.append(f"low document coverage ({_fmt_pct(m['document_coverage'])})")
        if m.get("forbidden_term_hit"):
            reasons.append(f"forbidden term(s) present: {m['forbidden_terms_found']}")
        if m.get("structure_ok") is False:
            reasons.append(f"missing document section(s) - named only: {m.get('documents_named_in_answer')}")
        if m.get("refusal_correct") is False:
            reasons.append("did not refuse/flag insufficient evidence for an unsupported question")
        print(f"    Failure pattern: {'; '.join(reasons) if reasons else 'below pass threshold'}")
        issues = m.get("validation_issues") or []
        print(f"    Validator: {', '.join(issues) if issues else 'no issues detected'}")

    print("\n" + "=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the production answer-quality QA harness.")
    parser.add_argument("--dataset", default=str(config.PROJECT_ROOT / "data" / "product_qa.json"))
    parser.add_argument("--output", default=str(config.PROJECT_ROOT / "output" / "product_qa_results.json"))
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    sources = sorted({s for case in dataset["cases"] for s in case["selected_document_sources"]})

    try:
        id_by_source = resolve_document_ids(sources)
    except RagError as e:
        print(f"Cannot run product QA: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(dataset['cases'])} QA cases against the production RAG pipeline...")
    started = time.time()
    case_results = []
    for case in dataset["cases"]:
        print(f"  {case['id']:5s} [{case['category']:14s}] ...", end=" ", flush=True)
        result = run_case(case, id_by_source)
        n_pass = sum(1 for t in result["turns"] if t["metrics"]["passed"])
        n_turns = len(result["turns"])
        print(f"{n_pass}/{n_turns} turns passed")
        case_results.append(result)
    wall_clock_s = time.time() - started

    summary = aggregate(case_results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"dataset_path": args.dataset, "wall_clock_s": wall_clock_s, "summary": summary, "cases": case_results},
            f,
            indent=2,
            default=str,
        )

    print_report(dataset, summary, case_results)
    print(f"\nWall clock: {wall_clock_s:.1f}s")
    print(f"Raw results (full prompts/answers/chunks, for manual inspection) saved to: {output_path}")


if __name__ == "__main__":
    main()
