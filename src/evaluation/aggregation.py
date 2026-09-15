"""Aggregation across repeated evaluation_results.json runs.

Computes mean +/- std per method across runs, a paired Dense-vs-Hybrid
significance test on per-question differences (Wilcoxon signed-rank), and a
category-by-method breakdown - all from data scripts/run_evaluation.py
already persists, so nothing here recomputes or redefines a metric.

Statistical note: paired_differences() pools per-question (dense, hybrid)
pairs across every run given. Repeated runs of the SAME question are not
fully independent observations (retrieval is deterministic given a fixed
index, so only the generation side varies run to run), so the resulting N is
an upper bound on genuine statistical power, not N independent question
samples. wilcoxon_test() and callers must not present this as more powerful
than it is - see the "do not overstate significance" note surfaced in the
CLI summary.
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

METRIC_KEYS = [
    "retrieval_recall", "term_coverage", "mrr_at_5", "ndcg_at_5",
    "retrieval_ms", "generation_ms", "total_ms",
]

# Metrics eligible for the paired Dense-vs-Hybrid significance test - all are
# per-question scores in a fixed [0, 1] range, unlike latency.
PAIRED_METRICS = [
    ("retrieval_recall", "recall"),
    ("term_coverage", "coverage"),
    ("mrr_at_5", "mrr_at_5"),
    ("ndcg_at_5", "ndcg_at_5"),
]


def load_run(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_compatible(runs: list[dict]) -> list[str]:
    """Return human-readable warnings for any dataset/config mismatch across
    runs. An empty list means all runs share the same dataset hash, method
    set, and configuration and are safe to average together.
    """
    warnings = []
    if len(runs) < 2:
        return warnings

    first = runs[0]["meta"]
    for i, run in enumerate(runs[1:], start=2):
        m = run["meta"]
        if m.get("dataset_sha256") != first.get("dataset_sha256"):
            warnings.append(
                f"Run {i}: dataset_sha256 differs from run 1 "
                f"({m.get('dataset_sha256')} vs {first.get('dataset_sha256')})"
            )
        if set(m.get("methods", [])) != set(first.get("methods", [])):
            warnings.append(f"Run {i}: methods differ ({m.get('methods')} vs {first.get('methods')})")

        config_a, config_b = first.get("config", {}), m.get("config", {})
        if config_a != config_b:
            diffs = {
                k: (config_a.get(k), config_b.get(k))
                for k in set(config_a) | set(config_b)
                if config_a.get(k) != config_b.get(k)
            }
            warnings.append(f"Run {i}: config differs from run 1: {diffs}")

    return warnings


def mean_std(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": None, "n": 1}
    return {"mean": mean(values), "std": stdev(values), "n": len(values)}


def per_run_metric_means(runs: list[dict], method: str, metric_key: str) -> list[float]:
    """One value per run: that run's own mean over its questions for metric_key."""
    out = []
    for run in runs:
        values = [q[metric_key] for q in run["results"].get(method, []) if q.get(metric_key) is not None]
        if values:
            out.append(mean(values))
    return out


def repeated_run_summary(runs: list[dict], method: str) -> dict:
    n_questions = len(runs[0]["results"].get(method, [])) if runs else 0
    return {
        "n_runs": len(runs),
        "n_questions_per_run": n_questions,
        **{key: mean_std(per_run_metric_means(runs, method, key)) for key in METRIC_KEYS},
    }


def paired_differences(runs: list[dict], metric_key: str) -> list[tuple]:
    """(question_id, dense_value, hybrid_value, hybrid - dense) for every
    (run, question) where both methods produced a value for metric_key.
    """
    pairs = []
    for run in runs:
        dense_by_id = {q["id"]: q for q in run["results"].get("dense", [])}
        hybrid_by_id = {q["id"]: q for q in run["results"].get("hybrid", [])}
        for qid in dense_by_id.keys() & hybrid_by_id.keys():
            d = dense_by_id[qid].get(metric_key)
            h = hybrid_by_id[qid].get(metric_key)
            if d is not None and h is not None:
                pairs.append((qid, d, h, h - d))
    return pairs


def wilcoxon_test(diffs: list[float]) -> dict:
    """Wilcoxon signed-rank test on paired differences. Ties (zero
    differences) are dropped per the standard Wilcoxon convention and
    reported explicitly via n vs n_nonzero.
    """
    n = len(diffs)
    non_zero = [d for d in diffs if d != 0]
    n_nonzero = len(non_zero)

    if n == 0:
        return {
            "statistic": None, "p_value": None, "n": n, "n_nonzero": n_nonzero,
            "note": "no paired data available for this metric (e.g. older runs "
                    "predating it) - test not applicable",
        }

    if n_nonzero < 1:
        return {
            "statistic": None, "p_value": None, "n": n, "n_nonzero": n_nonzero,
            "note": "all paired differences are zero - test not applicable",
        }

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {
            "statistic": None, "p_value": None, "n": n, "n_nonzero": n_nonzero,
            "note": "scipy not installed - Wilcoxon test not computed",
        }

    note = None
    try:
        if n_nonzero < 10:
            note = f"n_nonzero={n_nonzero} is small - p-value from the exact method may still be unreliable"
            statistic, p_value = wilcoxon(non_zero, method="exact")
        else:
            statistic, p_value = wilcoxon(non_zero)
    except ValueError as exc:
        return {
            "statistic": None, "p_value": None, "n": n, "n_nonzero": n_nonzero,
            "note": f"Wilcoxon test could not be computed: {exc}",
        }

    return {
        "statistic": float(statistic), "p_value": float(p_value),
        "n": n, "n_nonzero": n_nonzero, "note": note,
    }


def category_breakdown(runs: list[dict], method: str) -> dict:
    """mean recall/coverage per question category, pooled across all runs."""
    buckets = defaultdict(lambda: {"recall": [], "coverage": []})
    for run in runs:
        for q in run["results"].get(method, []):
            category = q.get("category", "Uncategorized")
            buckets[category]["recall"].append(q["retrieval_recall"])
            buckets[category]["coverage"].append(q["term_coverage"])

    return {
        category: {
            "n": len(v["recall"]),
            "mean_recall": mean(v["recall"]) if v["recall"] else None,
            "mean_coverage": mean(v["coverage"]) if v["coverage"] else None,
        }
        for category, v in buckets.items()
    }


def build_aggregate_report(run_paths: list[str]) -> dict:
    """Full aggregation report: compatibility warnings, per-method mean+/-std
    across runs, paired Dense-vs-Hybrid Wilcoxon test, category breakdown,
    and the untouched raw per-question results of every input run.
    """
    runs = [load_run(p) for p in run_paths]
    compatibility_warnings = check_compatible(runs)
    methods = sorted(set().union(*(set(r["meta"].get("methods", [])) for r in runs))) if runs else []

    report = {
        "run_files": [str(p) for p in run_paths],
        "n_runs": len(runs),
        "compatibility_warnings": compatibility_warnings,
        "methods": methods,
        "per_method": {m: repeated_run_summary(runs, m) for m in methods},
        "category_breakdown": {m: category_breakdown(runs, m) for m in methods},
    }

    if "dense" in methods and "hybrid" in methods:
        paired = {}
        for metric_key, label in PAIRED_METRICS:
            pairs = paired_differences(runs, metric_key)
            diffs = [p[3] for p in pairs]
            paired[label] = {
                "mean_diff_hybrid_minus_dense": mean(diffs) if diffs else None,
                "n_pairs": len(diffs),
                "wilcoxon": wilcoxon_test(diffs),
            }
        report["paired_dense_vs_hybrid"] = paired

    # Preserve every input run's full per-question results, untouched.
    report["runs"] = [
        {"label": r["meta"].get("label"), "meta": r["meta"], "results": r["results"]}
        for r in runs
    ]

    return report
