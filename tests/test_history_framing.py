"""Conversation history must be framed as reference context, not evidence.

Production failure this guards against: in a multi-turn chat about a
clinical write-up, "What medication did she take for hypertension?" was
answered "No medication is mentioned" although the top EVIDENCE passage
read "began on unknown medication. Stopped after 6 months because of
drowsiness." Retrieval and the evidence block were correct. The cause was the
RECENT CONVERSATION section: it sits between the evidence and the question,
under a bare header, and llama3.2:3b followed an earlier wrong answer in it
instead of the evidence (10 of 10 runs on the reconstructed prompt; 0 of 10
with HISTORY_HEADER). These tests pin the framing; they do not call a model.
"""

from src.generation.prompt_builder import (
    HISTORY_HEADER,
    build_multi_document_prompt,
    build_production_prompt,
    build_prompt,
    format_conversation_history,
)

EVIDENCE_TEXT = (
    "Medical History - 1998: Diagnosed with hypertension and began on unknown "
    "medication. Stopped after 6 months because of drowsiness."
)
CHUNKS = [
    {
        "chunk_id": 152,
        "document_id": "d1",
        "source": "UMNwriteup.pdf",
        "page": 2,
        "text": EVIDENCE_TEXT,
        "score": 0.05,
        "rank": 1,
    }
]
INVENTORY_1 = [{"document_id": "d1", "source": "UMNwriteup.pdf", "page_count": 6}]
QUESTION = "What medication did she take for hypertension?"
WRONG_PRIOR_ANSWER = "She was taking a medication that would treat her high BP as well, secondary to volume overload."
HISTORY = [
    {"role": "user", "content": "What treatment was she taking for it?"},
    {"role": "assistant", "content": WRONG_PRIOR_ANSWER},
]


def _both_production_prompts(history):
    return {
        "single-document": build_production_prompt(QUESTION, CHUNKS, history=history),
        "multi-document": build_multi_document_prompt(QUESTION, CHUNKS, INVENTORY_1, history=history),
    }


def test_header_says_earlier_answers_are_not_evidence():
    assert HISTORY_HEADER.startswith("RECENT CONVERSATION")
    assert "not evidence" in HISTORY_HEADER
    assert "answer from the EVIDENCE" in HISTORY_HEADER
    # it must still license the one legitimate use of history
    assert "what the question refers to" in HISTORY_HEADER


def test_both_production_templates_use_the_header_exactly_once():
    for name, prompt in _both_production_prompts(HISTORY).items():
        assert prompt.count(HISTORY_HEADER) == 1, name
        assert "RECENT CONVERSATION:\n" not in prompt, name


def test_prior_answers_sit_under_the_header_between_evidence_and_question():
    for name, prompt in _both_production_prompts(HISTORY).items():
        evidence_idx = prompt.index(EVIDENCE_TEXT)
        header_idx = prompt.index(HISTORY_HEADER)
        prior_idx = prompt.index(WRONG_PRIOR_ANSWER)
        question_idx = prompt.index("USER QUESTION:")
        assert evidence_idx < header_idx < prior_idx < question_idx, name


def test_history_turns_are_rendered_unchanged_under_the_header():
    expected = format_conversation_history(HISTORY)
    for name, prompt in _both_production_prompts(HISTORY).items():
        assert f"{HISTORY_HEADER}\n{expected}\n" in prompt, name


def test_no_history_means_no_header():
    for history in (None, []):
        for name, prompt in _both_production_prompts(history).items():
            assert HISTORY_HEADER not in prompt, name
            assert "RECENT CONVERSATION" not in prompt, name


def test_production_entry_point_routes_to_the_framed_header():
    single = build_prompt(QUESTION, CHUNKS, history=HISTORY, disable_benchmark_scope=True)
    multi = build_prompt(QUESTION, CHUNKS, history=HISTORY, document_inventory=INVENTORY_1)
    assert HISTORY_HEADER in single
    assert HISTORY_HEADER in multi


def test_legacy_context_prompt_is_unchanged():
    # The non-production CONTEXT-based path (reached history-free by
    # scripts/evaluate.py) says nothing about an "EVIDENCE above", so it keeps
    # its original header; the research harness never passes history.
    prompt = build_prompt(QUESTION, CHUNKS, history=HISTORY)
    assert "RECENT CONVERSATION:\n" in prompt
    assert HISTORY_HEADER not in prompt
