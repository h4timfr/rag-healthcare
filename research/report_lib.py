"""Pure data-loading and table-building functions for the research report.

Reads only already-persisted evaluation result files (data/evaluation_results
.publication_run*.json, data/evaluation_aggregate.publication.json,
data/healthcare_eval_questions.json, data/evaluation_corpus.json) - never
runs Ollama, never runs retrieval, never invents a value not present in one
of those files.

Data sourcing, made explicit so nothing is silently blended:

- Recall@5, Term Coverage, and all three latencies: read verbatim from the
  EXISTING data/evaluation_aggregate.publication.json (repeated-run mean +/-
  std across runs 1-3, and the paired Dense-vs-Hybrid Wilcoxon test for
  recall/coverage already computed there). Nothing here recomputes them.

- MRR@5 / nDCG@5: runs 1-3 predate these metrics (see the evaluation audit
  history) and evaluation_aggregate.publication.json was never regenerated
  to include them - it has no data for them at all. Run 4 is the only run
  that measured them. Their mean/paired-Wilcoxon values here come from
  calling the SAME existing, unmodified statistics function
  (src.evaluation.aggregation.build_aggregate_report) on run 4 alone - i.e.
  reusing the existing statistics engine on the one run that has the data,
  not writing new statistical logic. Reported as n=1 (single run, no std,
  no repeated-run averaging) throughout - never blended with runs 1-3.

- Category/document/per-question breakdowns: these were never computed by
  any existing tool (the existing category_breakdown() only knows about
  recall/coverage, predating MRR/nDCG). They are computed here via plain
  pandas groupby/mean over run 4's already-computed per-question values
  (retrieval_recall, term_coverage, mrr_at_5, ndcg_at_5) - grouping and
  averaging numbers that evaluate_question() already produced, not
  redefining how any of them are calculated. Run 4 is used as the single
  source for these tables (the only run with complete metric coverage), not
  a mix of runs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.aggregation import build_aggregate_report  # noqa: E402

DATA_DIR = ROOT / "data"
QUESTIONS_PATH = DATA_DIR / "healthcare_eval_questions.json"
CORPUS_MANIFEST_PATH = DATA_DIR / "evaluation_corpus.json"
AGGREGATE_PATH = DATA_DIR / "evaluation_aggregate.publication.json"
RUN_PATHS = {i: DATA_DIR / f"evaluation_results.publication_run{i}.json" for i in (1, 2, 3, 4)}
PRIMARY_RUN = 4  # the only run measuring MRR@5/nDCG@5; used for all per-question breakdowns

METRIC_ROWS = [
    ("retrieval_recall", "mrr_at_5", "ndcg_at_5", "term_coverage"),
]


def load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_question_metadata() -> pd.DataFrame:
    questions = load_json(QUESTIONS_PATH)
    return pd.DataFrame(questions)


def load_document_manifest() -> pd.DataFrame:
    manifest = load_json(CORPUS_MANIFEST_PATH)
    df = pd.DataFrame(manifest)
    # 'source' in results/questions is the filename, except OMC (filename is
    # null there) where it's the title itself - this reproduces that key.
    df["source_key"] = df["filename"].fillna(df["title"])
    return df


def run_to_long_dataframe(run_number: int) -> pd.DataFrame:
    """One row per (question_id, method) from evaluation_results.publication_run<N>.json."""
    run = load_json(RUN_PATHS[run_number])
    rows = []
    for method, results in run["results"].items():
        for r in results:
            row = dict(r)
            row["method"] = method
            row["run_label"] = run["meta"]["label"]
            rows.append(row)
    return pd.DataFrame(rows)


def build_primary_dataframe() -> pd.DataFrame:
    """Run 4's per-question results joined with the document manifest's
    human-readable title/type, for category/document/per-question tables."""
    df = run_to_long_dataframe(PRIMARY_RUN)
    manifest = load_document_manifest()
    df = df.merge(
        manifest[["source_key", "title", "document_type"]],
        left_on="expected_source",
        right_on="source_key",
        how="left",
    )
    return df


def load_aggregate() -> dict:
    """The EXISTING repeated-run aggregate (runs 1-3), read verbatim."""
    return load_json(AGGREGATE_PATH)


def compute_run4_only_aggregate() -> dict:
    """Run the EXISTING, unmodified aggregation function on run 4 alone, to
    get its own mean/paired-Wilcoxon stats (including mrr_at_5/ndcg_at_5,
    which the 3-run aggregate has no data for at all). No new statistics
    code - this calls the same src.evaluation.aggregation module used
    throughout the evaluation framework.
    """
    return build_aggregate_report([str(RUN_PATHS[PRIMARY_RUN])])


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_final_metrics_table(aggregate: dict, run4_aggregate: dict) -> pd.DataFrame:
    rows = []

    def agg_row(metric_key, label, unit, source):
        d = aggregate["per_method"]["dense"][metric_key]
        h = aggregate["per_method"]["hybrid"][metric_key]
        rows.append(
            {
                "metric": label,
                "unit": unit,
                "dense_mean": d["mean"],
                "dense_std": d["std"],
                "dense_n_runs": d["n"],
                "hybrid_mean": h["mean"],
                "hybrid_std": h["std"],
                "hybrid_n_runs": h["n"],
                "abs_diff_hybrid_minus_dense": (
                    None if d["mean"] is None or h["mean"] is None else h["mean"] - d["mean"]
                ),
                "source": source,
            }
        )

    three_run_source = "data/evaluation_aggregate.publication.json (repeated-run mean/std, runs 1-3, n=3)"
    run4_source = "data/evaluation_results.publication_run4.json (single run, n=1, no repeated-run std)"

    agg_row("retrieval_recall", "Recall@5", "", three_run_source)
    agg_row("term_coverage", "Term Coverage", "", three_run_source)

    def run4_row(metric_key, label, unit):
        d = run4_aggregate["per_method"]["dense"][metric_key]
        h = run4_aggregate["per_method"]["hybrid"][metric_key]
        rows.append(
            {
                "metric": label,
                "unit": unit,
                "dense_mean": d["mean"],
                "dense_std": d["std"],
                "dense_n_runs": d["n"],
                "hybrid_mean": h["mean"],
                "hybrid_std": h["std"],
                "hybrid_n_runs": h["n"],
                "abs_diff_hybrid_minus_dense": (
                    None if d["mean"] is None or h["mean"] is None else h["mean"] - d["mean"]
                ),
                "source": run4_source,
            }
        )

    run4_row("mrr_at_5", "MRR@5", "")
    run4_row("ndcg_at_5", "nDCG@5", "")

    agg_row("retrieval_ms", "Retrieval Latency", "ms", three_run_source)
    agg_row("generation_ms", "Generation Latency", "ms", three_run_source)
    agg_row("total_ms", "End-to-End Latency", "ms", three_run_source)

    return pd.DataFrame(rows)


def build_category_table(primary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        primary_df.groupby(["category", "method"])
        .agg(
            n_questions=("id", "count"),
            recall_at_5=("retrieval_recall", "mean"),
            mrr_at_5=("mrr_at_5", "mean"),
            ndcg_at_5=("ndcg_at_5", "mean"),
            term_coverage=("term_coverage", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["category", "method"]).reset_index(drop=True)


def build_document_table(primary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        primary_df.groupby(["title", "expected_source", "document_type", "method"])
        .agg(
            n_questions=("id", "count"),
            recall_at_5=("retrieval_recall", "mean"),
            mrr_at_5=("mrr_at_5", "mean"),
            ndcg_at_5=("ndcg_at_5", "mean"),
            term_coverage=("term_coverage", "mean"),
        )
        .reset_index()
        .rename(columns={"title": "document", "expected_source": "source_file"})
    )
    return grouped.sort_values(["document", "method"]).reset_index(drop=True)


def build_question_comparison_table(primary_df: pd.DataFrame) -> pd.DataFrame:
    meta = primary_df.drop_duplicates("id").set_index("id")[
        ["category", "title", "expected_source"]
    ].rename(columns={"title": "document", "expected_source": "source_file"})

    pivot = primary_df.pivot(index="id", columns="method", values=["mrr_at_5", "ndcg_at_5", "retrieval_recall"])
    pivot.columns = [f"{metric}_{method}" for metric, method in pivot.columns]

    result = meta.join(pivot).reset_index()

    result["mrr_winner"] = np.select(
        [result["mrr_at_5_hybrid"] > result["mrr_at_5_dense"], result["mrr_at_5_dense"] > result["mrr_at_5_hybrid"]],
        ["hybrid", "dense"],
        default="tie",
    )
    result["ndcg_winner"] = np.select(
        [result["ndcg_at_5_hybrid"] > result["ndcg_at_5_dense"], result["ndcg_at_5_dense"] > result["ndcg_at_5_hybrid"]],
        ["hybrid", "dense"],
        default="tie",
    )

    result = result.rename(
        columns={
            "mrr_at_5_dense": "dense_mrr_at_5",
            "mrr_at_5_hybrid": "hybrid_mrr_at_5",
            "ndcg_at_5_dense": "dense_ndcg_at_5",
            "ndcg_at_5_hybrid": "hybrid_ndcg_at_5",
            "retrieval_recall_dense": "dense_recall_at_5",
            "retrieval_recall_hybrid": "hybrid_recall_at_5",
        }
    )
    ordered_cols = [
        "id", "category", "document", "source_file",
        "dense_recall_at_5", "hybrid_recall_at_5",
        "dense_mrr_at_5", "hybrid_mrr_at_5", "mrr_winner",
        "dense_ndcg_at_5", "hybrid_ndcg_at_5", "ndcg_winner",
    ]
    result = result[ordered_cols].sort_values("id").reset_index(drop=True)
    return result


def build_statistical_tests_table(aggregate: dict, run4_aggregate: dict) -> pd.DataFrame:
    rows = []

    def add_row(paired_dict, key, label, source):
        p = paired_dict[key]
        w = p["wilcoxon"]
        significant = bool(w["p_value"] is not None and w["p_value"] < 0.05)
        rows.append(
            {
                "metric": label,
                "source": source,
                "n_pairs": p["n_pairs"],
                "n_nonzero": w["n_nonzero"],
                "mean_diff_hybrid_minus_dense": p["mean_diff_hybrid_minus_dense"],
                "wilcoxon_statistic": w["statistic"],
                "p_value": w["p_value"],
                "significant_at_0.05": significant,
                "note": w["note"],
            }
        )

    three_run_paired = aggregate["paired_dense_vs_hybrid"]
    add_row(
        three_run_paired, "recall", "Recall@5",
        "runs 1-3 pooled (3 runs x 45 questions = 135 pairs; repeated runs of "
        "the same question are not fully independent - see caveat)",
    )
    add_row(
        three_run_paired, "coverage", "Term Coverage",
        "runs 1-3 pooled (3 runs x 45 questions = 135 pairs; repeated runs of "
        "the same question are not fully independent - see caveat)",
    )

    run4_paired = run4_aggregate["paired_dense_vs_hybrid"]
    add_row(
        run4_paired, "mrr_at_5", "MRR@5",
        "run 4 only (45 distinct questions, single run - not independently repeated)",
    )
    add_row(
        run4_paired, "ndcg_at_5", "nDCG@5",
        "run 4 only (45 distinct questions, single run - not independently repeated)",
    )
    add_row(
        run4_paired, "recall", "Recall@5 (run 4 only, cross-check vs runs 1-3)",
        "run 4 only (45 distinct questions, single run - not independently repeated)",
    )
    add_row(
        run4_paired, "coverage", "Term Coverage (run 4 only, cross-check vs runs 1-3)",
        "run 4 only (45 distinct questions, single run - not independently repeated)",
    )

    return pd.DataFrame(rows)
