"""Tests for scripts/rescore_product_qa.py.

Re-scoring exists so a grader change and a generation change can be told
apart: replaying saved answers through the current grader isolates the
grader's contribution. The properties that matter are that it never calls
the model, and that it grades a saved run exactly as a live run would.
"""

import json

import pytest

from scripts.rescore_product_qa import rescore
from src.evaluation import product_qa


def _run(answer: str, required_facts: list[str], category: str = "factual") -> dict:
    return {
        "cases": [
            {
                "id": "Q01",
                "category": category,
                "selected_document_sources": ["a.pdf"],
                "turns": [
                    {
                        "spec": {
                            "question": "What was the chief complaint?",
                            "required_facts": required_facts,
                            "required_sources": ["a.pdf"],
                            "forbidden_terms": [],
                        },
                        "result": {
                            "generated_ok": True,
                            "answer": answer,
                            "sources": [{"source": "a.pdf", "page": 1}],
                            "latency_ms": 1234.0,
                        },
                        "metrics": {"passed": False},  # stale, must be recomputed
                    }
                ],
            }
        ]
    }


def test_rescore_recomputes_metrics_and_ignores_the_stored_ones():
    """The stored metrics say failed; the current grader says passed, because
    "syncopal" now satisfies "syncope"."""
    summary, cases = rescore(_run("She had a syncopal episode.", ["syncope"]))

    assert cases[0]["turns"][0]["metrics"]["passed"] is True
    assert summary["overall"]["passed"] == 1
    assert summary["overall"]["n_turns"] == 1


def test_rescore_agrees_with_a_live_evaluate_turn():
    run = _run("She had a syncopal episode.", ["syncope"])
    turn = run["cases"][0]["turns"][0]

    _, cases = rescore(run)

    assert cases[0]["turns"][0]["metrics"] == product_qa.evaluate_turn(
        turn["spec"], turn["result"], "factual"
    )


def test_rescore_never_invokes_the_model(monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("re-scoring must not call the model")

    monkeypatch.setattr(product_qa, "generate", explode)
    monkeypatch.setattr(product_qa, "run_turn", explode)

    summary, _ = rescore(_run("She had a syncopal episode.", ["syncope"]))

    assert summary["overall"]["n_turns"] == 1


def test_rescore_preserves_case_identity_and_stored_answers():
    run = _run("An answer.", ["answer"])

    _, cases = rescore(run)

    assert cases[0]["id"] == "Q01"
    assert cases[0]["category"] == "factual"
    assert cases[0]["turns"][0]["result"]["answer"] == "An answer."


def test_rescore_output_is_serialisable():
    summary, cases = rescore(_run("An answer.", ["answer"]))
    json.dumps({"summary": summary, "cases": cases})


def test_rescore_handles_a_failing_turn():
    summary, cases = rescore(_run("Something unrelated.", ["syncope"]))

    assert cases[0]["turns"][0]["metrics"]["passed"] is False
    assert summary["overall"]["passed"] == 0


def test_rescore_handles_an_empty_run():
    summary, cases = rescore({"cases": []})
    assert cases == []
    assert summary["n_cases"] == 0


@pytest.mark.parametrize("category", product_qa.REQUIRED_CATEGORIES)
def test_rescore_accepts_every_category(category):
    """Each category takes a different grading branch; none may raise."""
    summary, _ = rescore(_run("The documents do not contain that.", [], category=category))
    assert summary["overall"]["n_turns"] == 1
