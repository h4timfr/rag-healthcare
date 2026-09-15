"""Tests for src/evaluation/aggregation.py: mean/std across repeated runs,
dataset/config compatibility checks, paired Dense-vs-Hybrid differences,
the Wilcoxon signed-rank test, category breakdown, and the end-to-end report.
"""

import json

import pytest

from src.evaluation.aggregation import (
    build_aggregate_report,
    category_breakdown,
    check_compatible,
    mean_std,
    paired_differences,
    repeated_run_summary,
    wilcoxon_test,
)


def _question(qid, recall, coverage, category="factual_extraction", ms=100.0):
    return {
        "id": qid,
        "category": category,
        "retrieval_recall": recall,
        "term_coverage": coverage,
        "retrieval_ms": ms,
        "generation_ms": ms * 10,
        "total_ms": ms * 11,
    }


def _make_run(label, dataset_hash="abc123", dense_recall=0.8, hybrid_recall=0.9, ids=("Q1", "Q2", "Q3"), config=None):
    dense = [_question(qid, dense_recall, 0.70) for qid in ids]
    hybrid = [_question(qid, hybrid_recall, 0.80) for qid in ids]
    return {
        "meta": {
            "label": label,
            "dataset_sha256": dataset_hash,
            "methods": ["dense", "hybrid"],
            "config": config or {"top_k": 5},
        },
        "results": {"dense": dense, "hybrid": hybrid},
    }


# --- mean_std ---------------------------------------------------------

def test_mean_std_empty():
    assert mean_std([]) == {"mean": None, "std": None, "n": 0}


def test_mean_std_single_value_has_no_std():
    result = mean_std([0.5])
    assert result["mean"] == 0.5
    assert result["std"] is None
    assert result["n"] == 1


def test_mean_std_multiple_values():
    result = mean_std([0.8, 0.9, 1.0])
    assert result["mean"] == pytest.approx(0.9)
    assert result["n"] == 3
    assert result["std"] is not None and result["std"] > 0


def test_mean_std_ignores_none():
    result = mean_std([0.5, None, 0.7])
    assert result["n"] == 2


# --- check_compatible ---------------------------------------------------

def test_check_compatible_no_warnings_for_identical_runs():
    runs = [_make_run("run1"), _make_run("run2")]
    assert check_compatible(runs) == []


def test_check_compatible_flags_dataset_hash_mismatch():
    runs = [_make_run("run1", dataset_hash="abc"), _make_run("run2", dataset_hash="different")]
    warnings = check_compatible(runs)
    assert any("dataset_sha256" in w for w in warnings)


def test_check_compatible_flags_config_mismatch():
    runs = [
        _make_run("run1", config={"top_k": 5}),
        _make_run("run2", config={"top_k": 10}),
    ]
    warnings = check_compatible(runs)
    assert any("config differs" in w for w in warnings)


def test_check_compatible_single_run_has_no_warnings():
    assert check_compatible([_make_run("only")]) == []


# --- repeated_run_summary ------------------------------------------------

def test_repeated_run_summary_averages_across_runs():
    runs = [
        _make_run("r1", dense_recall=0.8),
        _make_run("r2", dense_recall=1.0),
    ]
    summary = repeated_run_summary(runs, "dense")
    assert summary["n_runs"] == 2
    assert summary["n_questions_per_run"] == 3
    assert summary["retrieval_recall"]["mean"] == pytest.approx(0.9)
    assert summary["retrieval_recall"]["n"] == 2
    assert summary["retrieval_recall"]["std"] is not None


# --- paired_differences ---------------------------------------------------

def test_paired_differences_matches_by_question_id():
    runs = [_make_run("r1", dense_recall=0.6, hybrid_recall=0.9, ids=("Q1", "Q2"))]
    pairs = paired_differences(runs, "retrieval_recall")
    assert len(pairs) == 2
    for qid, dense_val, hybrid_val, diff in pairs:
        assert dense_val == pytest.approx(0.6)
        assert hybrid_val == pytest.approx(0.9)
        assert diff == pytest.approx(0.3)


def test_paired_differences_pools_across_multiple_runs():
    runs = [_make_run("r1", ids=("Q1", "Q2")), _make_run("r2", ids=("Q1", "Q2"))]
    pairs = paired_differences(runs, "retrieval_recall")
    assert len(pairs) == 4  # 2 questions x 2 runs


def test_paired_differences_skips_questions_missing_from_one_method():
    run = _make_run("r1", ids=("Q1", "Q2"))
    del run["results"]["hybrid"][1]  # Q2 missing from hybrid
    pairs = paired_differences([run], "retrieval_recall")
    assert len(pairs) == 1
    assert pairs[0][0] == "Q1"


# --- wilcoxon_test ---------------------------------------------------------

def test_wilcoxon_test_all_zero_differences_not_applicable():
    result = wilcoxon_test([0.0, 0.0, 0.0])
    assert result["p_value"] is None
    assert "not applicable" in result["note"]


def test_wilcoxon_test_reports_n_and_n_nonzero():
    diffs = [0.1, 0.2, 0.0, 0.15, -0.05]
    result = wilcoxon_test(diffs)
    assert result["n"] == 5
    assert result["n_nonzero"] == 4  # the 0.0 is dropped


def test_wilcoxon_test_returns_statistic_and_pvalue_for_enough_data():
    diffs = [0.30, 0.35, 0.40, 0.32, 0.38, 0.31, 0.36, 0.33, 0.29, 0.34, 0.37, 0.28]
    result = wilcoxon_test(diffs)
    assert result["statistic"] is not None
    assert result["p_value"] is not None
    assert 0.0 <= result["p_value"] <= 1.0
    # a strongly one-sided, consistent difference should read as significant
    assert result["p_value"] < 0.05


def test_wilcoxon_test_small_n_gets_a_reliability_note():
    diffs = [0.1, 0.2, 0.15]
    result = wilcoxon_test(diffs)
    assert result["note"] is not None and "small" in result["note"]


# --- category_breakdown ---------------------------------------------------

def test_category_breakdown_groups_by_category():
    run = {
        "results": {
            "dense": [
                _question("Q1", 1.0, 1.0, category="numeric_detail"),
                _question("Q2", 0.0, 0.0, category="numeric_detail"),
                _question("Q3", 1.0, 1.0, category="terminology_semantic"),
            ]
        }
    }
    breakdown = category_breakdown([run], "dense")
    assert breakdown["numeric_detail"]["n"] == 2
    assert breakdown["numeric_detail"]["mean_recall"] == pytest.approx(0.5)
    assert breakdown["terminology_semantic"]["n"] == 1
    assert breakdown["terminology_semantic"]["mean_recall"] == pytest.approx(1.0)


# --- build_aggregate_report (end-to-end, via files) ------------------------

def test_build_aggregate_report_end_to_end(tmp_path):
    run1 = _make_run("r1", dense_recall=0.8, hybrid_recall=0.9)
    run2 = _make_run("r2", dense_recall=0.9, hybrid_recall=1.0)

    path1 = tmp_path / "run1.json"
    path2 = tmp_path / "run2.json"
    path1.write_text(json.dumps(run1), encoding="utf-8")
    path2.write_text(json.dumps(run2), encoding="utf-8")

    report = build_aggregate_report([str(path1), str(path2)])

    assert report["n_runs"] == 2
    assert report["compatibility_warnings"] == []
    assert set(report["methods"]) == {"dense", "hybrid"}
    assert report["per_method"]["dense"]["retrieval_recall"]["mean"] == pytest.approx(0.85)
    assert "paired_dense_vs_hybrid" in report
    assert report["paired_dense_vs_hybrid"]["recall"]["n_pairs"] == 6  # 3 questions x 2 runs
    # raw per-question results must be preserved, not just the summary
    assert len(report["runs"]) == 2
    assert report["runs"][0]["results"]["dense"][0]["id"] == "Q1"


def test_build_aggregate_report_flags_incompatible_runs(tmp_path):
    run1 = _make_run("r1", dataset_hash="hash-a")
    run2 = _make_run("r2", dataset_hash="hash-b")
    path1 = tmp_path / "run1.json"
    path2 = tmp_path / "run2.json"
    path1.write_text(json.dumps(run1), encoding="utf-8")
    path2.write_text(json.dumps(run2), encoding="utf-8")

    report = build_aggregate_report([str(path1), str(path2)])
    assert report["compatibility_warnings"]
