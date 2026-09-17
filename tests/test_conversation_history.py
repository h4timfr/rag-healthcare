"""Regression tests for conversation history (Phase 6).

Two invariants, in tension with each other, both of which have been broken
in this project before:

  retrieval   sees only the current question. A transcript embedded into the
              search query matches whatever the conversation mentioned
              earlier instead of what is being asked now.
  generation  sees recent history, so "what about the second document?" can
              be resolved - but bounded, so a long session does not grow the
              prompt without limit.
"""

import app
from src.generation.prompt_builder import (
    MAX_HISTORY_CHARS_PER_TURN,
    MAX_HISTORY_TURNS,
    build_prompt,
    format_conversation_history,
)


def _chunk(rank=1):
    return {
        "chunk_id": rank, "document_id": "d1", "source": "a.pdf", "page": 1,
        "text": "evidence text", "score": 0.5, "rank": rank,
    }


def _history(n_exchanges: int, content: str = "text") -> list[dict]:
    history = []
    for i in range(n_exchanges):
        history.append({"role": "user", "content": f"question {i} {content}"})
        history.append({"role": "assistant", "content": f"answer {i} {content}"})
    return history


# --- retrieval must never see the transcript ---------------------------------


def test_retrieval_query_is_the_current_question_only(monkeypatch):
    seen = {}

    def fake_retrieve_hybrid(query, **kwargs):
        seen["query"] = query
        return [_chunk()]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    history = [
        {"role": "user", "content": "earlier question about aneurysm repair"},
        {"role": "assistant", "content": "earlier answer about metformin"},
    ]
    app.retrieve_and_build_prompt("what about the second document?", history=history)

    assert seen["query"] == "what about the second document?"
    assert "aneurysm" not in seen["query"]
    assert "metformin" not in seen["query"]


def test_history_reaches_generation_but_not_retrieval(monkeypatch):
    seen = {}

    def fake_retrieve_hybrid(query, **kwargs):
        seen["query"] = query
        seen["kwargs"] = kwargs
        return [_chunk()]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    history = [{"role": "user", "content": "distinctive earlier phrase"}]
    _, prompt = app.retrieve_and_build_prompt("current question", history=history)

    assert "distinctive earlier phrase" not in seen["query"]
    assert "distinctive earlier phrase" in prompt
    assert "history" not in seen["kwargs"]


# --- the recent-history window ------------------------------------------------


def test_only_the_most_recent_turns_are_kept():
    history = _history(5)  # 10 entries

    rendered = format_conversation_history(history)

    assert rendered.count("\n") == MAX_HISTORY_TURNS - 1
    assert "question 4" in rendered
    assert "question 0" not in rendered


def test_long_turns_are_truncated_not_dropped():
    long_answer = "x" * (MAX_HISTORY_CHARS_PER_TURN + 500)
    history = [{"role": "assistant", "content": long_answer}]

    rendered = format_conversation_history(history)

    assert rendered.endswith("[...]")
    assert len(rendered) < MAX_HISTORY_CHARS_PER_TURN + 100


def test_truncation_keeps_the_head_where_documents_are_named():
    """A per-document answer names its documents up front; a follow-up asking
    "what about the second document?" resolves against that."""
    answer = "Pediatric Report.pdf: a routine checkup. " + ("filler " * 500)
    history = [{"role": "assistant", "content": answer}]

    rendered = format_conversation_history(history)

    assert "Pediatric Report.pdf" in rendered


def test_a_realistic_multi_document_answer_is_not_truncated():
    """The longest answer this app produced in QA was a ~1.2k-char
    three-document summary. The cap must sit above real answers."""
    realistic = "y" * 1342
    history = [{"role": "assistant", "content": realistic}]

    assert "[...]" not in format_conversation_history(history)


def test_history_stays_bounded_as_a_conversation_grows():
    """Prompt size must be a function of the window, not of how long the
    user has been chatting. Fixed-width entries so the only variable is
    how many of them there are."""
    def fixed(n_exchanges):
        return [
            {"role": role, "content": "z" * 200}
            for _ in range(n_exchanges)
            for role in ("user", "assistant")
        ]

    short = build_prompt("q", [_chunk()], history=fixed(2), disable_benchmark_scope=True)
    long = build_prompt("q", [_chunk()], history=fixed(40), disable_benchmark_scope=True)

    assert len(long) == len(short)


def test_no_history_omits_the_section_entirely():
    assert format_conversation_history(None) == ""
    assert format_conversation_history([]) == ""
    assert "RECENT CONVERSATION" not in build_prompt("q", [_chunk()], disable_benchmark_scope=True)


def test_non_string_content_does_not_crash_rendering():
    assert "42" in format_conversation_history([{"role": "user", "content": 42}])


# --- placement ----------------------------------------------------------------


def test_history_never_lands_after_the_generation_cue():
    history = [{"role": "assistant", "content": "earlier answer text"}]

    for inventory in (None, [{"document_id": "d1", "source": "a.pdf", "page_count": 1}]):
        prompt = build_prompt(
            "current question", [_chunk()],
            document_inventory=inventory, history=history, disable_benchmark_scope=True,
        )
        answer_idx = prompt.rindex("ANSWER:")
        assert "earlier answer text" in prompt[:answer_idx]
        assert "earlier answer text" not in prompt[answer_idx:]
