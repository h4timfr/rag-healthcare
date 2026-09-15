#!/usr/bin/env python
"""Aggregate repeated evaluation_results.json runs from scripts/run_evaluation.py.

For a latency or significance claim to mean anything, run the same
evaluation multiple times (generation is not deterministic - see README) and
aggregate here rather than trusting a single run's numbers.

Usage:
    python scripts/aggregate_runs.py run1.json run2.json run3.json \\
        --output-json data/evaluation_aggregate.json \\
        --output-summary data/evaluation_aggregate_summary.txt

Works with a single run too (n_runs=1: means are reported, std is None).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.aggregation import build_aggregate_report

METRIC_LABELS = [
    ("retrieval_recall", "Recall@K"),
    ("term_coverage", "Term Coverage"),
    ("mrr_at_5", "MRR@5"),
    ("ndcg_at_5", "nDCG@5"),
    ("retrieval_ms", "Retrieval latency (ms)"),
    ("generation_ms", "Generation latency (ms)"),
    ("total_ms", "End-to-end latency (ms)"),
]


def write_summary(report: dict, path: Path) -> None:
    lines = ["Repeated-Run Evaluation Aggregate", "=" * 70, ""]
    lines.append(f"Runs aggregated: {report['n_runs']}")
    for f in report["run_files"]:
        lines.append(f"  - {f}")
    lines.append("")

    if report["compatibility_warnings"]:
        lines.append("COMPATIBILITY WARNINGS - these runs may not be directly comparable:")
        for w in report["compatibility_warnings"]:
            lines.append(f"  ! {w}")
    else:
        lines.append("All runs share the same dataset hash, method set, and configuration.")
    lines.append("")

    for method, summary in report["per_method"].items():
        lines.append("=" * 70)
        lines.append(
            f"{method.upper()} - mean +/- std across {summary['n_runs']} run(s), "
            f"{summary['n_questions_per_run']} questions/run"
        )
        lines.append("=" * 70)
        for key, label in METRIC_LABELS:
            s = summary[key]
            if s["mean"] is None:
                lines.append(f"  {label:<26} n/a")
            elif s["n"] == 1:
                lines.append(f"  {label:<26} {s['mean']:.3f}  (n=1 run, no std)")
            else:
                lines.append(f"  {label:<26} {s['mean']:.3f} +/- {s['std']:.3f}  (n={s['n']} runs)")
        lines.append("")

    if "paired_dense_vs_hybrid" in report:
        lines.append("=" * 70)
        lines.append("PAIRED DENSE vs HYBRID (per question, pooled across all runs)")
        lines.append("=" * 70)
        lines.append(
            "Caveat: repeated runs of the SAME question are not fully independent\n"
            "observations (retrieval is deterministic given a fixed index; only\n"
            "generation varies run to run), so N here is an upper bound on genuine\n"
            "statistical power, not N independent question samples. Treat this as\n"
            "exploratory evidence, not a definitive significance claim."
        )
        lines.append("")
        for label, data in report["paired_dense_vs_hybrid"].items():
            w = data["wilcoxon"]
            lines.append(f"{label}:")
            if data["mean_diff_hybrid_minus_dense"] is not None:
                lines.append(f"  mean(hybrid - dense) = {data['mean_diff_hybrid_minus_dense']:+.4f}")
            lines.append(f"  N pairs = {data['n_pairs']}, N nonzero = {w['n_nonzero']}")
            if w["p_value"] is not None:
                lines.append(f"  Wilcoxon signed-rank: statistic={w['statistic']:.3f}, p={w['p_value']:.4f}")
                if w["p_value"] >= 0.05:
                    lines.append("  -> not significant at alpha=0.05")
                else:
                    lines.append("  -> significant at alpha=0.05 (see caveat above before citing this)")
            if w.get("note"):
                lines.append(f"  NOTE: {w['note']}")
            lines.append("")

    lines.append("=" * 70)
    lines.append("CATEGORY BREAKDOWN (mean recall/coverage, pooled across all runs)")
    lines.append("=" * 70)
    categories = sorted({c for breakdown in report["category_breakdown"].values() for c in breakdown})
    methods = list(report["category_breakdown"].keys())
    header = f"{'Category':<24}"
    for m in methods:
        header += f"{m.upper() + ' N':>8}{m.upper() + ' Rec':>10}{m.upper() + ' Cov':>10}  "
    lines.append(header)
    lines.append("-" * len(header))
    for cat in categories:
        row = f"{cat:<24}"
        for m in methods:
            b = report["category_breakdown"][m].get(cat, {"n": 0, "mean_recall": None, "mean_coverage": None})
            row += f"{b['n']:>8}"
            row += f"{b['mean_recall']:>10.2f}" if b["mean_recall"] is not None else f"{'n/a':>10}"
            row += f"{b['mean_coverage']:>10.2f}" if b["mean_coverage"] is not None else f"{'n/a':>10}"
            row += "  "
        lines.append(row)
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated evaluation_results.json runs.")
    parser.add_argument("runs", nargs="+", help="Paths to one or more evaluation_results*.json files")
    parser.add_argument("--output-json", type=Path, default=ROOT / "data" / "evaluation_aggregate.json")
    parser.add_argument("--output-summary", type=Path, default=ROOT / "data" / "evaluation_aggregate_summary.txt")
    args = parser.parse_args()

    report = build_aggregate_report(args.runs)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_summary(report, args.output_summary)

    print(f"Aggregate JSON: {args.output_json}")
    print(f"Aggregate summary: {args.output_summary}")
    if report["compatibility_warnings"]:
        print("\nWARNING: input runs are not fully compatible - see the summary for details.")


if __name__ == "__main__":
    main()
