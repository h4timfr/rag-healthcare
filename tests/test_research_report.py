"""Tests for research/report_lib.py - the offline reporting utility that
turns already-persisted evaluation result files into tables. No retrieval,
generation, or Ollama call happens anywhere in this module or its tests.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research import report_lib as rl


# --- loading -----------------------------------------------------------

def test_run_paths_point_at_real_existing_files():
    for path in rl.RUN_PATHS.values():
        assert path.exists(), f"{path} does not exist"


def test_load_aggregate_covers_runs_1_to_3_only():
    aggregate = rl.load_aggregate()
    assert aggregate["n_runs"] == 3
    labels = [r["label"] for r in aggregate["runs"]]
    assert labels == ["publication_run1", "publication_run2", "publication_run3"]


def test_run4_has_ranking_metrics_runs_1_to_3_do_not():
    run4 = rl.load_json(rl.RUN_PATHS[4])
    assert "mrr_at_5" in run4["results"]["dense"][0]
    assert "ndcg_at_5" in run4["results"]["dense"][0]

    for i in (1, 2, 3):
        run = rl.load_json(rl.RUN_PATHS[i])
        assert "mrr_at_5" not in run["results"]["dense"][0], (
            f"run{i} unexpectedly has mrr_at_5 - it should predate ranking metrics"
        )


def test_all_run_dataset_hashes_match():
    hashes = {i: rl.load_json(rl.RUN_PATHS[i])["meta"]["dataset_sha256"] for i in (1, 2, 3, 4)}
    assert len(set(hashes.values())) == 1, f"dataset hashes differ across runs: {hashes}"


# --- build_primary_dataframe --------------------------------------------

def test_build_primary_dataframe_has_90_rows_45_questions_x_2_methods():
    df = rl.build_primary_dataframe()
    assert len(df) == 90
    assert set(df["method"].unique()) == {"dense", "hybrid"}
    assert df["id"].nunique() == 45


def test_build_primary_dataframe_documents_all_resolved():
    """Every row's expected_source must join to a real document title - a
    failed join (unmapped source) would leave 'title' as NaN."""
    df = rl.build_primary_dataframe()
    assert df["title"].isna().sum() == 0
    assert df["title"].nunique() == 5


# --- build_final_metrics_table -------------------------------------------

def test_final_metrics_recall_and_coverage_come_from_three_run_aggregate():
    aggregate = rl.load_aggregate()
    run4_aggregate = rl.compute_run4_only_aggregate()
    table = rl.build_final_metrics_table(aggregate, run4_aggregate)

    recall_row = table[table["metric"] == "Recall@5"].iloc[0]
    assert recall_row["dense_n_runs"] == 3
    assert recall_row["dense_mean"] == pytest.approx(aggregate["per_method"]["dense"]["retrieval_recall"]["mean"])
    assert "evaluation_aggregate.publication.json" in recall_row["source"]


def test_final_metrics_mrr_and_ndcg_come_from_run4_only_with_no_std():
    aggregate = rl.load_aggregate()
    run4_aggregate = rl.compute_run4_only_aggregate()
    table = rl.build_final_metrics_table(aggregate, run4_aggregate)

    for metric in ("MRR@5", "nDCG@5"):
        row = table[table["metric"] == metric].iloc[0]
        assert row["dense_n_runs"] == 1
        assert row["hybrid_n_runs"] == 1
        assert row["dense_std"] is None or row["dense_std"] != row["dense_std"]  # None or NaN
        assert "publication_run4.json" in row["source"]


def test_final_metrics_has_all_seven_required_metrics():
    aggregate = rl.load_aggregate()
    run4_aggregate = rl.compute_run4_only_aggregate()
    table = rl.build_final_metrics_table(aggregate, run4_aggregate)
    expected = {
        "Recall@5", "MRR@5", "nDCG@5", "Term Coverage",
        "Retrieval Latency", "Generation Latency", "End-to-End Latency",
    }
    assert set(table["metric"]) == expected


# --- build_category_table -------------------------------------------------

def test_category_table_covers_all_five_categories_both_methods():
    df = rl.build_primary_dataframe()
    table = rl.build_category_table(df)
    assert set(table["category"].unique()) == {
        "factual_extraction", "numeric_detail", "section_retrieval",
        "terminology_semantic", "multi_part",
    }
    assert len(table) == 10  # 5 categories x 2 methods
    assert table["n_questions"].sum() == 90


def test_category_table_values_are_bounded_zero_to_one():
    df = rl.build_primary_dataframe()
    table = rl.build_category_table(df)
    for col in ("recall_at_5", "mrr_at_5", "ndcg_at_5", "term_coverage"):
        assert (table[col] >= 0).all() and (table[col] <= 1).all()


# --- build_document_table -------------------------------------------------

def test_document_table_covers_all_five_documents_both_methods():
    df = rl.build_primary_dataframe()
    table = rl.build_document_table(df)
    assert table["document"].nunique() == 5
    assert len(table) == 10  # 5 documents x 2 methods


def test_document_table_question_counts_sum_to_ninety():
    df = rl.build_primary_dataframe()
    table = rl.build_document_table(df)
    assert table["n_questions"].sum() == 90


# --- build_question_comparison_table --------------------------------------

def test_question_comparison_has_45_rows_one_per_question():
    df = rl.build_primary_dataframe()
    table = rl.build_question_comparison_table(df)
    assert len(table) == 45
    assert table["id"].nunique() == 45


def test_question_comparison_winner_logic_is_consistent_with_raw_scores():
    df = rl.build_primary_dataframe()
    table = rl.build_question_comparison_table(df)
    for _, row in table.iterrows():
        if row["dense_mrr_at_5"] > row["hybrid_mrr_at_5"]:
            assert row["mrr_winner"] == "dense"
        elif row["hybrid_mrr_at_5"] > row["dense_mrr_at_5"]:
            assert row["mrr_winner"] == "hybrid"
        else:
            assert row["mrr_winner"] == "tie"


def test_question_comparison_required_columns_present():
    df = rl.build_primary_dataframe()
    table = rl.build_question_comparison_table(df)
    required = {
        "id", "category", "document",
        "dense_mrr_at_5", "hybrid_mrr_at_5", "mrr_winner",
        "dense_ndcg_at_5", "hybrid_ndcg_at_5", "ndcg_winner",
    }
    assert required.issubset(table.columns)


# --- build_statistical_tests_table -----------------------------------------

def test_statistical_tests_recall_and_coverage_use_three_run_aggregate():
    aggregate = rl.load_aggregate()
    run4_aggregate = rl.compute_run4_only_aggregate()
    table = rl.build_statistical_tests_table(aggregate, run4_aggregate)

    recall_row = table[table["metric"] == "Recall@5"].iloc[0]
    assert recall_row["n_pairs"] == aggregate["paired_dense_vs_hybrid"]["recall"]["n_pairs"]
    assert "runs 1-3" in recall_row["source"]


def test_statistical_tests_mrr_and_ndcg_use_run4_only():
    aggregate = rl.load_aggregate()
    run4_aggregate = rl.compute_run4_only_aggregate()
    table = rl.build_statistical_tests_table(aggregate, run4_aggregate)

    for metric in ("MRR@5", "nDCG@5"):
        row = table[table["metric"] == metric].iloc[0]
        assert row["n_pairs"] == 45  # single run x 45 questions
        assert "run 4 only" in row["source"]


def test_statistical_tests_significance_flag_never_true_for_none_pvalue():
    aggregate = rl.load_aggregate()
    run4_aggregate = rl.compute_run4_only_aggregate()
    table = rl.build_statistical_tests_table(aggregate, run4_aggregate)
    none_pvalue_rows = table[table["p_value"].isna()]
    assert not none_pvalue_rows["significant_at_0.05"].any()


def test_statistical_tests_never_invents_a_pvalue_below_alpha_without_scipy_result():
    """Every 'significant' row must have an actual numeric p-value backing it."""
    aggregate = rl.load_aggregate()
    run4_aggregate = rl.compute_run4_only_aggregate()
    table = rl.build_statistical_tests_table(aggregate, run4_aggregate)
    significant_rows = table[table["significant_at_0.05"]]
    assert (significant_rows["p_value"] < 0.05).all()
    assert significant_rows["p_value"].notna().all()
