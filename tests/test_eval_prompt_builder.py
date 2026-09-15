"""Confirms the evaluation prompt path is free of the production
scope/answer-key injection identified in the evaluation audit."""

import inspect

from src.generation import eval_prompt_builder


def test_eval_prompt_builder_has_no_scope_injection_functions():
    names = {n for n, _ in inspect.getmembers(eval_prompt_builder)}
    assert "detect_target_scope" not in names
    assert "get_scope_terms" not in names
    assert "extract_matching_items" not in names


def test_build_eval_prompt_contains_only_raw_chunk_text():
    chunks = [
        {"source": "doc.pdf", "page": 4, "text": "Retrieval latency and end-to-end response time are tracked."},
    ]
    prompt = eval_prompt_builder.build_eval_prompt(
        "What are the system-level evaluation metrics?", chunks
    )
    assert "[1] Source: doc.pdf, page 4" in prompt
    assert "Retrieval latency and end-to-end response time are tracked." in prompt
    # No scope-detection artifact should appear - the neutral builder has no
    # concept of "TARGET SCOPE" instructions.
    assert "TARGET SCOPE" not in prompt


def test_build_eval_prompt_signature_takes_no_ground_truth():
    sig = inspect.signature(eval_prompt_builder.build_eval_prompt)
    assert list(sig.parameters) == ["query", "chunks"]
