"""Regression tests for the independent production-quality audit fixes:

1. multi-document retrieval depth (TOP_K * number of selected documents)
2. a larger generation budget for multi-document synthesis only
3. chat.py's conversation history placed before "ANSWER:", never after
4. app.py/chat.py using consistent history semantics (display + generation
   context only, never embedded into the retrieval query)
5. the old project-benchmark's target-scope detection kept out of the
   production answer path (isolated behind disable_benchmark_scope=False,
   the untouched default only scripts/evaluate.py still relies on)
"""

import numpy as np
import pymupdf
import pytest

import app
import chat
from src import config
from src.generation.prompt_builder import build_multi_document_prompt, build_prompt


def _chunk(document_id, source, page, text, rank, score=0.1):
    return {
        "chunk_id": rank,
        "document_id": document_id,
        "source": source,
        "page": page,
        "text": text,
        "score": score,
        "rank": rank,
    }


# --- 1. multi-document retrieval depth --------------------------------------


def test_resolve_retrieval_top_k_selected_narrow_question_is_plain_top_k():
    """Changed deliberately: this used to widen to TOP_K * document count for
    every question in selected scope. See the regression tests in
    test_multi_document_budget.py - that widening produced a 15-chunk prompt
    in which the model denied a fact held in its own highest-ranked chunk."""
    assert app.resolve_retrieval_top_k("selected", {"d1", "d2", "d3"}) == config.TOP_K


def test_resolve_retrieval_top_k_selected_widens_only_when_balanced():
    assert app.resolve_retrieval_top_k("selected", {"d1", "d2", "d3"}, balanced=True) == (
        config.BALANCED_CHUNKS_PER_DOCUMENT * 3
    )


def test_resolve_retrieval_top_k_selected_one_document_is_plain_top_k():
    assert app.resolve_retrieval_top_k("selected", {"d1"}) == config.TOP_K


def test_resolve_retrieval_top_k_selected_empty_selection_falls_back_to_top_k():
    assert app.resolve_retrieval_top_k("selected", set()) == config.TOP_K


def test_resolve_retrieval_top_k_active_scope_is_always_plain_top_k():
    """Single-document (active) behavior must never scale with whatever is
    left over in selected_document_ids from a previous "selected" session."""
    assert app.resolve_retrieval_top_k("active", {"d1", "d2", "d3"}) == config.TOP_K
    assert app.resolve_retrieval_top_k("active", set()) == config.TOP_K


def test_retrieve_and_build_prompt_forwards_the_resolved_top_k_to_retrieval(monkeypatch):
    """A narrow question keeps the plain budget even with three documents
    selected; only an each-document/comparison question widens it."""
    captured = {}

    def fake_retrieve_hybrid(query, top_k=5, document_id=None, **kwargs):
        captured["top_k"] = top_k
        return [_chunk("d1", "a.pdf", 1, "t", 1)]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    inventory = [
        {"document_id": "d1", "source": "a.pdf", "page_count": 1},
        {"document_id": "d2", "source": "b.pdf", "page_count": 1},
        {"document_id": "d3", "source": "c.pdf", "page_count": 1},
    ]

    app.retrieve_and_build_prompt(
        "What medication was prescribed?",
        top_k=app.resolve_retrieval_top_k("selected", {"d1", "d2", "d3"}),
        document_id=["d1", "d2", "d3"],
        document_inventory=inventory,
    )
    assert captured["top_k"] == config.TOP_K

    app.retrieve_and_build_prompt(
        "Summarize each selected document separately.",
        top_k=config.TOP_K,
        document_id=["d1", "d2", "d3"],
        document_inventory=inventory,
    )
    assert captured["top_k"] == config.BALANCED_CHUNKS_PER_DOCUMENT * 3


# --- 2. multi-document generation budget ------------------------------------


def test_multi_document_stream_uses_larger_num_predict_budget(monkeypatch):
    captured = {}

    def fake_generate_stream(prompt, result=None, num_predict=None):
        captured["num_predict"] = num_predict
        if result is not None:
            result["answer"] = "ok"
            result["timing"] = {}
        yield "ok"

    monkeypatch.setattr(app, "generate_stream", fake_generate_stream)

    list(app._stream_with_ttft("prompt", {}, {}, num_predict=config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT))

    assert captured["num_predict"] == config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT
    assert config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT > config.OLLAMA_NUM_PREDICT


def test_single_document_stream_omits_num_predict_override(monkeypatch):
    """Single-document answers must keep the existing default - not every
    answer should get a bigger budget, only genuine multi-document ones."""
    captured = {}

    def fake_generate_stream(prompt, result=None, num_predict=None):
        captured["num_predict"] = num_predict
        if result is not None:
            result["answer"] = "ok"
            result["timing"] = {}
        yield "ok"

    monkeypatch.setattr(app, "generate_stream", fake_generate_stream)

    list(app._stream_with_ttft("prompt", {}, {}))  # no num_predict passed

    assert captured["num_predict"] is None


# --- resolve_num_predict: multi-part single-document questions need the ------
# --- larger budget too, not only multi-document answers ----------------------
#
# Live testing hit this directly: "What happened during the second
# admission? What about the first one?" was measured hitting the default
# 256-token cap mid sentence, part way through its second (correctly
# retrieved and correctly structured) section.


def test_multi_document_scope_gets_the_larger_budget():
    assert app.resolve_num_predict("anything", None, [{"document_id": "d1", "source": "a.pdf"}]) == (
        config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT
    )


def test_multi_part_single_document_question_gets_the_larger_budget():
    query = "What happened during the second admission? What about the first one?"
    assert app.resolve_num_predict(query, None, None) == config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT


def test_standalone_elliptical_follow_up_keeps_the_default_budget():
    """"What about the first?" alone resolves to exactly ONE clean
    sub-question ("What happened during the first admission?") - it is a
    single-part question after resolution, not two, so it needs the normal
    single-answer budget, not the multi-part one."""
    history = [{"role": "user", "content": "What happened during the second admission?"}]
    assert app.resolve_num_predict("What about the first?", history, None) is None


def test_multi_part_tag_question_in_the_same_message_gets_the_larger_budget():
    history = [{"role": "user", "content": "unrelated earlier turn"}]
    query = "What happened during the second admission? What about the first one?"
    assert app.resolve_num_predict(query, history, None) == config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT


def test_ordinary_single_part_single_document_question_keeps_the_default():
    assert app.resolve_num_predict("What medication was prescribed?", None, None) is None


def test_ordinal_document_reference_keeps_the_default():
    """"the second document" is a document reference, not a multi-part
    question - must not trigger the larger budget on ordinal-word content
    alone."""
    assert app.resolve_num_predict("What about the second document?", None, None) is None


# --- 3. chat.py history placement (never after ANSWER:) ----------------------


def test_build_prompt_places_history_before_answer_cue():
    chunks = [_chunk("d1", "a.pdf", 1, "hello", 1)]
    history = [
        {"role": "user", "content": "earlier question text"},
        {"role": "assistant", "content": "earlier answer text"},
    ]

    prompt = build_prompt("current question", chunks, history=history)

    answer_idx = prompt.rindex("ANSWER:")
    history_idx = prompt.index("earlier question text")
    assert history_idx < answer_idx
    # And nothing about the conversation appears after the generation cue.
    assert "earlier question text" not in prompt[answer_idx:]
    assert "earlier answer text" not in prompt[answer_idx:]


def test_build_prompt_history_section_between_context_and_user_question():
    chunks = [_chunk("d1", "a.pdf", 1, "hello", 1)]
    history = [{"role": "user", "content": "prior turn"}]

    prompt = build_prompt("current question", chunks, history=history)

    context_idx = prompt.index("CONTEXT:")
    history_idx = prompt.index("RECENT CONVERSATION:")
    question_idx = prompt.index("USER QUESTION:")
    answer_idx = prompt.rindex("ANSWER:")
    assert context_idx < history_idx < question_idx < answer_idx


def test_build_prompt_without_history_omits_conversation_section():
    chunks = [_chunk("d1", "a.pdf", 1, "hello", 1)]
    prompt = build_prompt("current question", chunks)
    assert "RECENT CONVERSATION" not in prompt


def test_build_multi_document_prompt_also_places_history_before_answer_cue():
    chunks = [_chunk("d1", "a.pdf", 1, "hello", 1)]
    inventory = [{"document_id": "d1", "source": "a.pdf", "page_count": 1}]
    history = [{"role": "user", "content": "earlier multi-doc question"}]

    prompt = build_multi_document_prompt("compare them", chunks, inventory, history=history)

    answer_idx = prompt.rindex("ANSWER:")
    history_idx = prompt.index("earlier multi-doc question")
    assert history_idx < answer_idx


def test_chat_ask_never_appends_history_after_answer_cue(monkeypatch):
    """Direct regression test for the reported bug: chat.ask() used to
    string-append 'RECENT CONVERSATION' onto the end of an already-built
    prompt, landing it after the trailing 'ANSWER:' cue."""
    captured = {}

    def fake_retrieve_hybrid(query, **kwargs):
        return [_chunk("d1", "a.pdf", 1, "t", 1)]

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return "an answer"

    monkeypatch.setattr(chat, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(chat, "generate", fake_generate)

    history = [
        {"role": "user", "content": "what about the previous patient"},
        {"role": "assistant", "content": "previous patient details"},
    ]
    chat.ask("a follow-up question", history)

    prompt = captured["prompt"]
    answer_idx = prompt.rindex("ANSWER:")
    assert prompt[answer_idx:].strip() == "ANSWER:"
    assert "what about the previous patient" in prompt[:answer_idx]


# --- 4. history never contaminates the retrieval query -----------------------


def test_chat_ask_retrieval_query_is_clean_not_history_embedded(monkeypatch):
    captured = {}

    def fake_retrieve_hybrid(query, **kwargs):
        captured["query"] = query
        return [_chunk("d1", "a.pdf", 1, "t", 1)]

    monkeypatch.setattr(chat, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(chat, "generate", lambda prompt: "an answer")

    history = [
        {"role": "user", "content": "unrelated earlier topic about diabetes"},
        {"role": "assistant", "content": "diabetes details"},
    ]
    chat.ask("current clean question", history)

    assert captured["query"] == "current clean question"
    assert "unrelated earlier topic" not in captured["query"]
    assert "Conversation context" not in captured["query"]


def test_app_retrieve_and_build_prompt_never_embeds_history_into_query(monkeypatch):
    captured = {}

    def fake_retrieve_hybrid(query, top_k=5, document_id=None):
        captured["query"] = query
        return [_chunk("d1", "a.pdf", 1, "t", 1)]

    monkeypatch.setattr(app, "retrieve_hybrid", fake_retrieve_hybrid)

    history = [{"role": "user", "content": "an earlier unrelated topic"}]
    app.retrieve_and_build_prompt("clean question", history=history)

    assert captured["query"] == "clean question"
    assert "earlier unrelated topic" not in captured["query"]


# --- 5. benchmark-specific target-scope isolation -----------------------------


def test_benchmark_scope_disabled_by_default_in_production_helper(monkeypatch):
    """retrieve_and_build_prompt() (the Streamlit app's real production
    path) must always disable the old benchmark scope detection."""
    captured = {}
    real_build_prompt = app.build_prompt

    def spy_build_prompt(query, chunks, document_inventory=None, history=None, disable_benchmark_scope=False, **kwargs):
        captured["disable_benchmark_scope"] = disable_benchmark_scope
        return real_build_prompt(
            query, chunks, document_inventory=document_inventory,
            history=history, disable_benchmark_scope=disable_benchmark_scope,
        )

    monkeypatch.setattr(app, "retrieve_hybrid", lambda query, top_k=5, document_id=None: [_chunk("d1", "a.pdf", 1, "t", 1)])
    monkeypatch.setattr(app, "build_prompt", spy_build_prompt)

    app.retrieve_and_build_prompt("a question")

    assert captured["disable_benchmark_scope"] is True


def test_chat_ask_disables_benchmark_scope(monkeypatch):
    captured = {}
    real_build_prompt = chat.build_prompt

    def spy_build_prompt(query, chunks, document_inventory=None, history=None, disable_benchmark_scope=False, **kwargs):
        captured["disable_benchmark_scope"] = disable_benchmark_scope
        return real_build_prompt(query, chunks, history=history, disable_benchmark_scope=disable_benchmark_scope)

    monkeypatch.setattr(chat, "retrieve_hybrid", lambda query, **kwargs: [_chunk("d1", "a.pdf", 1, "t", 1)])
    monkeypatch.setattr(chat, "build_prompt", spy_build_prompt)
    monkeypatch.setattr(chat, "generate", lambda prompt: "answer")

    chat.ask("a question", [])

    assert captured["disable_benchmark_scope"] is True


def test_disable_benchmark_scope_bypasses_target_scope_detection_for_a_normal_query():
    """A genuine production healthcare query that happens to contain one of
    the old benchmark's magic phrases ("document collection") must not have
    its context silently narrowed/rewritten when the production flag is set."""
    long_evidence = (
        "Our clinic's health document collection includes detailed patient "
        "vitals, medication history, and follow-up notes that are not "
        "related to any research benchmark terminology."
    )
    chunks = [_chunk("d1", "a.pdf", 1, long_evidence, 1)]
    query = "Tell me about our document collection of patient files."

    prompt_production = build_prompt(query, chunks, disable_benchmark_scope=True)
    prompt_legacy = build_prompt(query, chunks, disable_benchmark_scope=False)

    # The legacy (evaluate.py-only) path still narrows this query.
    assert "TARGET SCOPE: DOCUMENT COLLECTION" in prompt_legacy

    # The production path must not - full evidence text survives untouched,
    # and no benchmark scope instruction is injected.
    assert "TARGET SCOPE" not in prompt_production
    assert long_evidence in prompt_production


def test_disable_benchmark_scope_default_preserves_legacy_behavior():
    """The default (False) must be unchanged from before this fix, since
    scripts/evaluate.py (legacy, untouched) calls build_prompt() with no
    extra arguments and must keep getting the exact same behavior."""
    chunks = [_chunk("d1", "a.pdf", 1, "some traditional llm and rag discussion", 1)]
    query = "Compare the traditional LLM approach with RAG."

    prompt_default = build_prompt(query, chunks)
    prompt_explicit_false = build_prompt(query, chunks, disable_benchmark_scope=False)

    assert prompt_default == prompt_explicit_false
    assert "TARGET SCOPE: TRADITIONAL LLM VS RAG" in prompt_default


# --- single-document behavior is otherwise unchanged --------------------------


def test_single_document_prompt_unaffected_by_new_parameters_when_unused():
    chunks = [_chunk("d1", "a.pdf", 1, "hello world", 1)]
    prompt = build_prompt("what is this about?", chunks)
    assert "MULTI-DOCUMENT RULES" not in prompt
    assert "RECENT CONVERSATION" not in prompt
    assert "hello world" in prompt


def test_answer_question_unaffected_by_top_k_widening_logic(monkeypatch):
    """answer_question() (used by tests/CLI reuse) must keep its exact
    single-document contract regardless of the new multi-document top_k
    widening, which only ever happens inside app.py's render_chat wiring."""
    monkeypatch.setattr(
        app, "retrieve_hybrid",
        lambda query, top_k=5, document_id=None: [_chunk("d1", "a.pdf", 2, "t", 1)],
    )
    monkeypatch.setattr(app, "generate", lambda prompt, num_predict=None: "an answer")

    result = app.answer_question("a question")

    assert result["answer"] == "an answer"
    assert result["sources"] == [{"source": "a.pdf", "page": 2, "score": 0.1}]


# --- end-to-end: widened top_k actually surfaces more than TOP_K chunks -------


def _make_pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _fake_embed_texts(texts):
    vectors = []
    for text in texts:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=config.EMBEDDING_DIM).astype("float32")
        vectors.append(v / np.linalg.norm(v))
    return np.array(vectors, dtype="float32")


def test_widened_top_k_surfaces_more_than_plain_top_k_across_documents(tmp_path, monkeypatch):
    import scripts.ingest as ingest_script
    from src.retrieval import hybrid_retriever

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", str(tmp_path / "faiss.index"))
    monkeypatch.setattr(ingest_script, "embed_texts", _fake_embed_texts)

    # Three documents, each with several distinct short "pages" so there are
    # more than TOP_K chunks available across all three combined.
    texts = {
        "a.pdf": "Diabetes management overview. " * 1 + "Insulin therapy details page one.",
        "b.pdf": "Cardiology overview. " * 1 + "Coronary artery findings page one.",
        "c.pdf": "Pediatric growth overview. " * 1 + "Height and weight percentile notes.",
    }
    doc_ids = []
    for name, text in texts.items():
        path = tmp_path / name
        _make_pdf(path, text)
        result = app.ingest_uploaded_pdf(name, path.read_bytes())
        doc_ids.append(result["document_id"])

    # Force a neutral, non-discriminating query vector so ranking doesn't
    # collapse everything onto a single document.
    monkeypatch.setattr(hybrid_retriever, "embed_query", lambda q: np.ones(config.EMBEDDING_DIM, dtype="float32") / np.sqrt(config.EMBEDDING_DIM))

    plain_top_k = config.TOP_K
    widened_top_k = app.resolve_retrieval_top_k("selected", set(doc_ids), balanced=True)
    assert widened_top_k == config.BALANCED_CHUNKS_PER_DOCUMENT * 3

    from src.retrieval.hybrid_retriever import retrieve_hybrid

    plain_results = retrieve_hybrid("overview", top_k=plain_top_k, document_id=doc_ids)
    widened_results = retrieve_hybrid("overview", top_k=widened_top_k, document_id=doc_ids)

    assert len(widened_results) >= len(plain_results)
