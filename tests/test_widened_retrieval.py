"""Tests for the Task 13 single-document retrieval-depth widening: a
FACTUAL question whose answer must cover several distinct components
(intent.requests_multiple_components() - the same signal build_answer_style()
already uses for the evidence-complete ANSWER STYLE) is retrieved at
app.WIDENED_FACTUAL_TOP_K instead of config.TOP_K. Every other intent, and
any caller that already asked for a specific top_k, is left unchanged.

These tests exercise the real production call site (app.retrieve_and_build_prompt)
with retrieve_hybrid replaced by a spy that records the top_k it was called
with, exactly matching the monkeypatch convention already used in
tests/test_context_ordering.py.
"""

import app
from src import config


def _chunk(chunk_id, rank, document_id="d1", page=1, source="a.pdf"):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source": source,
        "page": page,
        "text": f"chunk {chunk_id} text",
        "score": 1.0 / rank,
        "rank": rank,
    }


def _spy_retrieve_hybrid(monkeypatch, calls):
    def fake(query, top_k=5, document_id=None, timing=None, **kwargs):
        calls.append(top_k)
        return [_chunk(i, i) for i in range(1, top_k + 1)]

    monkeypatch.setattr(app, "retrieve_hybrid", fake)
    return fake


# --- A: ordinary factual question -> K=5 --------------------------------------


def test_ordinary_factual_question_stays_at_default_top_k(monkeypatch):
    calls = []
    _spy_retrieve_hybrid(monkeypatch, calls)

    app.retrieve_and_build_prompt("What was the patient's chief complaint?", document_id="d1")

    assert calls == [config.TOP_K]
    assert config.TOP_K == 5


# --- B: factual 3+ component question -> K=8 -----------------------------------


def test_factual_three_component_question_widens_to_eight(monkeypatch):
    calls = []
    _spy_retrieve_hybrid(monkeypatch, calls)

    app.retrieve_and_build_prompt(
        "What were the main findings regarding examinations, laboratory tests, and treatments?",
        document_id="d1",
    )

    assert calls == [app.WIDENED_FACTUAL_TOP_K]
    assert app.WIDENED_FACTUAL_TOP_K == 8


def test_widening_only_applies_when_caller_left_top_k_at_the_default(monkeypatch):
    """A caller that already asked for a specific top_k (a diagnostic
    script, or the narrowed-to-one-document multi-document path above this
    code) is left alone - this rule only steps in for the ordinary default
    case, never overriding an explicit choice."""
    calls = []
    _spy_retrieve_hybrid(monkeypatch, calls)

    app.retrieve_and_build_prompt(
        "What were the main findings regarding examinations, laboratory tests, and treatments?",
        top_k=11,
        document_id="d1",
    )

    assert calls == [11]


# --- C: each/every <noun> question -> remains K=5 (not implemented yet) -------


def test_each_every_question_remains_at_default_top_k(monkeypatch):
    """Task 13 deliberately does not implement the each/every -> K=11 tier
    yet - this question must still retrieve at the ordinary default."""
    calls = []
    _spy_retrieve_hybrid(monkeypatch, calls)

    app.retrieve_and_build_prompt(
        "What were the findings for each diagnostic category?", document_id="d1"
    )

    assert calls == [config.TOP_K]


# --- D-G: every other intent is unchanged --------------------------------------


def test_summary_intent_is_unchanged(monkeypatch):
    calls = []
    _spy_retrieve_hybrid(monkeypatch, calls)

    app.retrieve_and_build_prompt("Summarize this document.", document_id="d1")

    assert calls == [config.TOP_K]


def test_each_document_intent_is_unchanged(monkeypatch):
    """Classified purely from the query text, independent of whether
    document_inventory happens to be set - the widening rule is gated on
    FACTUAL specifically, so EACH_DOCUMENT never qualifies either way."""
    calls = []
    _spy_retrieve_hybrid(monkeypatch, calls)

    app.retrieve_and_build_prompt("Summarize each selected document separately.", document_id="d1")

    assert calls == [config.TOP_K]


def test_comparison_intent_is_unchanged(monkeypatch):
    """This query also independently decomposes into 2 parts (decompose_question
    finds "and what differs" regardless of classify_intent's COMPARISON
    label), so retrieve_for_question makes one retrieve_hybrid() call per
    part - a pre-existing, unrelated mechanism. What matters here is that
    NEITHER call was widened."""
    calls = []
    _spy_retrieve_hybrid(monkeypatch, calls)

    app.retrieve_and_build_prompt(
        "What do they have in common and what differs between them?", document_id="d1"
    )

    assert calls and all(c == config.TOP_K for c in calls)


def test_follow_up_intent_is_unchanged(monkeypatch):
    calls = []
    _spy_retrieve_hybrid(monkeypatch, calls)

    history = [
        {"role": "user", "content": "Summarize each selected document."},
        {"role": "assistant", "content": "Pediatric Report.pdf: ..."},
    ]
    app.retrieve_and_build_prompt(
        "What about the medications in that report?", document_id="d1", history=history
    )

    assert calls == [config.TOP_K]


# --- H: multi-part intent -------------------------------------------------------


def test_genuinely_multi_part_question_is_unchanged():
    """A question classify_intent() itself resolves to MULTI_PART (two
    standalone parts joined by "and") never reaches the FACTUAL-gated
    widening rule, regardless of requests_multiple_components() - MULTI_PART
    wins classify_intent()'s priority over FACTUAL, so the gate's
    `classify_intent(...) == FACTUAL` check is never true for it."""
    from src.generation.intent import MULTI_PART, classify_intent

    query = "What test was ordered, and what medications were prescribed?"
    assert classify_intent(query) == MULTI_PART


def test_multi_part_shaped_question_is_unchanged_at_the_call_site(monkeypatch):
    """Genuinely multi-part: retrieve_for_question retrieves one part at a
    time (a pre-existing, unrelated mechanism - see decompose_question),
    so more than one retrieve_hybrid() call is expected. What matters here
    is that none of them were widened."""
    calls = []
    _spy_retrieve_hybrid(monkeypatch, calls)

    app.retrieve_and_build_prompt(
        "What test was ordered, and what medications were prescribed?", document_id="d1"
    )

    assert calls and all(c == config.TOP_K for c in calls)


def test_factual_multi_component_question_is_not_reclassified_as_multi_part(monkeypatch):
    """The distinguishing case the task's condition #8 describes: a question
    that IS multi-component in the requests_multiple_components() sense but
    classify_intent() resolves to FACTUAL (not MULTI_PART, since it has no
    "and <interrogative>" second-question shape) DOES widen - this is the
    same query as test B, re-asserted here to make the FACTUAL-vs-MULTI_PART
    distinction explicit."""
    from src.generation.intent import FACTUAL, classify_intent

    query = "What were the main findings regarding examinations, laboratory tests, and treatments?"
    assert classify_intent(query) == FACTUAL

    calls = []
    _spy_retrieve_hybrid(monkeypatch, calls)
    app.retrieve_and_build_prompt(query, document_id="d1")
    assert calls == [app.WIDENED_FACTUAL_TOP_K]
