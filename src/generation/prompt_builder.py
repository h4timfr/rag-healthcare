import re

SYSTEM_INSTRUCTION = """You are a healthcare document assistant.

Answer the user's question using ONLY the retrieved context below.

Rules:
- Use only information supported by the supplied context.
- Preserve the source's terminology, categories, organization, and distinctions exactly.
- Do not merge separate items or categories from the source.
- Do not move an item from one category to another.
- Do not add facts, metrics, interpretations, or examples that are not supported by the context.
- If the context is insufficient, say so explicitly.
- When listing multiple items, keep each source item separate.
- For healthcare content, do not provide medical diagnosis or treatment directives.
- If the question asks about a specific section or category, answer only that section unless the user explicitly asks for comparison with another section.
- When the source presents a list, reproduce every distinct item from that list without omitting, merging, or reclassifying items.
- When the question asks for a specific category, subsection, or evaluation group, first identify the exact matching section in the retrieved evidence.
- Do not answer with a neighboring or similarly named section unless it matches the question explicitly.
- When the question asks for one specific item, category, subsection, or list entry, answer ONLY that requested item or category.
- Do not include neighboring items from the same source list.
- Do not infer additional items.
- Do not invent paraphrased variants of source items.
- For comparison questions, preserve the distinction between the compared categories exactly as presented by the source.

Citation rules:
- Cite factual claims using the source numbers provided in the context.
- Use citations like [1], [2], [3].
- Cite only sources that directly support the claim.
"""


def detect_target_scope(query: str) -> str | None:
    """Detect a narrow requested scope from common question patterns."""
    q = query.lower()

    if "traditional llm" in q and "rag" in q:
        return "TRADITIONAL LLM VS RAG"

    if (
        "source presentation" in q
        or "source / citation" in q
        or "citation display" in q
    ):
        return "SOURCE PRESENTATION"

    if "document collection" in q or "document collections" in q:
        return "DOCUMENT COLLECTION"

    if "retrieval improvements" in q:
        return "RETRIEVAL IMPROVEMENTS"

    if "generation metrics" in q:
        return "GENERATION"

    if "retrieval metrics" in q:
        return "RETRIEVAL"

    if "system-level" in q or "system level" in q:
        return "SYSTEM"

    if "future scope" in q:
        return "FUTURE SCOPE"

    return None


def get_scope_terms(scope: str | None) -> list[str]:
    scope_terms = {
        "SOURCE PRESENTATION": [
            "Stronger source / citation display",
        ],
        "DOCUMENT COLLECTION": [
            "Larger healthcare document collections",
        ],
        "RETRIEVAL IMPROVEMENTS": [
            "Improve chunking strategy",
            "Hybrid keyword + semantic retrieval",
            "Reranking of retrieved chunks",
        ],
        "GENERATION": [
            "Answer relevance",
            "Faithfulness / groundedness",
            "Completeness",
        ],
        "RETRIEVAL": [
            "Precision@k",
            "Recall@k",
            "Retrieval relevance",
        ],
        "SYSTEM": [
            "Retrieval latency",
            "End-to-end response time",
        ],
    }

    return scope_terms.get(scope, [])


def extract_matching_items(
    text: str,
    target_terms: list[str],
) -> list[str]:
    if not target_terms:
        return []

    matches = []

    for term in target_terms:
        if re.search(
            re.escape(term),
            text,
            flags=re.IGNORECASE,
        ):
            matches.append(term)

    return matches


def build_context(
    chunks: list[dict],
    target_scope: str | None = None,
) -> str:
    """
    Format retrieved chunks as numbered evidence blocks.
    Narrow list/category questions get exact-item extraction.
    Comparison questions preserve the relevant source context.
    """

    blocks = []
    target_terms = get_scope_terms(target_scope)

    for i, chunk in enumerate(chunks, start=1):
        text = chunk["text"]

        if target_scope == "TRADITIONAL LLM VS RAG":
            blocks.append(
                f"[{i}] Source: {chunk['source']}, page {chunk['page']}\n"
                f"{text}"
            )
            continue

        if target_terms:
            matching_items = extract_matching_items(
                text,
                target_terms,
            )

            if matching_items:
                text = "\n".join(
                    f"• {item}"
                    for item in matching_items
                )
            else:
                continue

        blocks.append(
            f"[{i}] Source: {chunk['source']}, page {chunk['page']}\n"
            f"{text}"
        )

    if (
        target_scope
        and target_scope != "TRADITIONAL LLM VS RAG"
        and not blocks
    ):
        for i, chunk in enumerate(chunks, start=1):
            blocks.append(
                f"[{i}] Source: {chunk['source']}, page {chunk['page']}\n"
                f"{chunk['text']}"
            )

    return "\n\n".join(blocks)


def format_conversation_history(history: list[dict] | None, max_turns: int = 4) -> str:
    """Render the last `max_turns` {"role", "content"} entries as plain text
    for the prompt's RECENT CONVERSATION section. Returns "" for no/empty
    history, so callers can skip the section entirely rather than emit an
    empty header. This is the single place both app.py and chat.py go
    through to format history for generation, so the two stay consistent.

    This text is for the generation prompt only - it is never sent to
    retrieve_hybrid(). Retrieval always receives just the current, clean
    query string; embedding a multi-turn transcript into the retrieval
    query would blur the dense/keyword match against whatever the
    conversation happened to mention earlier, not just the current question.
    """
    if not history:
        return ""
    recent = history[-max_turns:]
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)


def build_prompt(
    query: str,
    chunks: list[dict],
    document_inventory: list[dict] | None = None,
    history: list[dict] | None = None,
    disable_benchmark_scope: bool = False,
) -> str:
    """Assemble system instruction, focused evidence, and query.

    `document_inventory` is optional and only ever passed by the Streamlit
    app's multi-document "Selected documents" scope (see
    app.retrieve_and_build_prompt / build_multi_document_prompt below).

    `history` (optional) is prior {"role", "content"} conversation turns,
    rendered via format_conversation_history() into a "RECENT CONVERSATION"
    section placed BEFORE "USER QUESTION:"/"ANSWER:" - never after, so it
    never buries the model's actual generation cue behind more text (a
    former chat.py bug). Omitted (the default), the prompt is unchanged.

    `disable_benchmark_scope` (default False, preserving this function's
    original behavior for any caller that doesn't pass it) skips
    detect_target_scope()/get_scope_terms()/extract_matching_items()
    entirely when True. Those exist only to narrow answers for this
    project's own historical benchmark questions (e.g. a query containing
    "source presentation" or "document collection") and have no business
    silently reshaping a real healthcare query's answer. Every production
    entry point (app.py, chat.py, scripts/query.py, scripts/profile_query.py)
    passes True; scripts/evaluate.py (legacy, kept unmodified, its output
    already documented as not trustworthy) is the only remaining caller that
    still gets the old auto-detected behavior via this default - the
    research/evaluation harness itself (scripts/run_evaluation.py) never
    calls this function at all, it uses the separate eval_prompt_builder.py.
    """
    if document_inventory:
        return build_multi_document_prompt(query, chunks, document_inventory, history=history)

    target_scope = None if disable_benchmark_scope else detect_target_scope(query)
    context = build_context(
        chunks,
        target_scope,
    )

    scope_instruction = ""

    if target_scope == "TRADITIONAL LLM VS RAG":
        scope_instruction = (
            "\n\nTARGET SCOPE: TRADITIONAL LLM VS RAG\n"
            "Compare the two approaches using only the distinctions "
            "explicitly stated in the retrieved context. "
            "Do not attribute RAG behavior to the traditional LLM."
        )

    elif target_scope:
        scope_instruction = (
            f"\n\nTARGET SCOPE: {target_scope}\n"
            "Use ONLY the explicitly extracted evidence for this scope. "
            "Do not include any other items from the source."
        )

    history_text = format_conversation_history(history)
    history_section = f"\n\nRECENT CONVERSATION:\n{history_text}" if history_text else ""

    return (
        f"{SYSTEM_INSTRUCTION}"
        f"{scope_instruction}\n\n"
        f"CONTEXT:\n{context}"
        f"{history_section}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"ANSWER:"
    )


# ---------------------------------------------------------------------------
# Multi-document synthesis (Streamlit "Selected documents" analysis scope).
#
# Kept fully separate from the single-document path above: it never touches
# detect_target_scope()/build_context()'s narrow single-topic extraction
# logic, and is only ever reached via build_prompt(document_inventory=...).
# ---------------------------------------------------------------------------

MULTI_DOCUMENT_INTENT_PATTERNS = (
    re.compile(r"\ball\b"),
    re.compile(r"\beach\b"),
    re.compile(r"\bevery\b"),
    re.compile(r"\bcompar"),  # compare, comparison, comparing
    re.compile(r"\bdiffer"),  # differ, differs, difference
    re.compile(r"in common"),
)


def detect_multi_document_intent(query: str) -> bool:
    """Whether the question is asking to cover/compare multiple documents
    (vs. one narrow question that merely happens to be asked while several
    documents are selected) - used to decide whether to instruct the model
    to structure its answer with one section per document.
    """
    q = query.lower()
    return any(pattern.search(q) for pattern in MULTI_DOCUMENT_INTENT_PATTERNS)


def build_multi_document_context(chunks: list[dict], document_inventory: list[dict]) -> str:
    """Group retrieved chunks by document, in document_inventory's order.

    A selected document with zero retrieved chunks gets an explicit
    "no evidence retrieved" placeholder instead of being silently omitted
    (which could read as "nothing to say") or left for the model to guess
    at (which is exactly the invented-content failure mode this exists to
    prevent - selected does not mean evidence exists).

    Headers are kept plain (no "===" fencing) deliberately: llama3.2:3b is
    small enough that heavily-symbolic delimiters get echoed back into the
    answer verbatim instead of being read as structure to summarize from -
    see build_multi_document_prompt()'s ANSWER FORMAT template, which is the
    main defense against that, but plainer input formatting helps too.

    Citation numbers ([1], [2], ...) are each chunk's own `rank` from
    retrieve_hybrid (its position in the original fused-ranking result),
    not a fresh per-group counter - so a citation means the same chunk here
    as it does in the Streamlit evidence panel, regardless of how chunks
    get visually regrouped by document in either place.
    """
    blocks = []
    for doc in document_inventory:
        blocks.append(f"Document: {doc['source']}")

        doc_chunks = [c for c in chunks if c.get("document_id") == doc.get("document_id")]
        if not doc_chunks:
            blocks.append("(No relevant evidence was retrieved for this document.)")
            continue

        for chunk in doc_chunks:
            blocks.append(f"[{chunk['rank']}] (page {chunk['page']}) {chunk['text']}")

    return "\n\n".join(blocks)


def build_multi_document_prompt(
    query: str,
    chunks: list[dict],
    document_inventory: list[dict],
    history: list[dict] | None = None,
) -> str:
    """build_prompt()'s multi-document counterpart to the single-document
    SYSTEM_INSTRUCTION + build_context() path: same base rules and citation
    convention, plus explicit multi-document grounding rules and a
    document-grouped context. `history` is placed the same way as in
    build_prompt() - before USER QUESTION/ANSWER, never after.
    """
    context = build_multi_document_context(chunks, document_inventory)
    document_list = ", ".join(doc["source"] for doc in document_inventory)

    rules = (
        "\n\nMULTI-DOCUMENT RULES (multiple documents are selected for this question):\n"
        f"- Selected documents: {document_list}\n"
        "- The CONTEXT below is reference material for you to read, organized into one "
        "\"Document: ...\" section per selected document - it is not a format to copy.\n"
        "- Write your answer in your own full sentences. Never copy a \"Document: ...\" "
        "label, the parenthetical no-evidence note, or a bare citation/page number "
        "straight into your answer with no sentence around it.\n"
        "- Every citation like [1] must sit inside a sentence stating a specific fact "
        "from that evidence - never output a citation or page number by itself.\n"
        "- A document being selected does NOT mean you have evidence about it - base "
        "what you say about each document only on that document's own section above, "
        "never on other documents' evidence or on general knowledge.\n"
        "- If a document's section has no evidence, say in one sentence that no "
        "relevant evidence was retrieved for it - do not guess, invent, or omit it.\n"
        "- Do not merely list document filenames - summarize the actual retrieved "
        "evidence for each document that has any.\n"
        "- Keep each citation ([1], [2], ...) attached to the document section it "
        "came from; do not attribute a citation to the wrong document."
    )

    if detect_multi_document_intent(query):
        answer_template = "\n".join(
            f"{doc['source']}: <1-3 sentences summarizing that document's own evidence "
            "above, or a sentence saying no relevant evidence was retrieved if its "
            "section says so>"
            for doc in document_inventory
        )
        rules += (
            "\n- The question asks about all/each/every document, or for a comparison. "
            "ANSWER FORMAT: write exactly one paragraph per selected document, in this "
            "order, starting each paragraph with the document's name as a heading "
            "followed by a colon, then your own sentences (never leave the "
            "\"<...>\" placeholder text itself in your answer):\n"
            f"{answer_template}\n"
        )

    history_text = format_conversation_history(history)
    history_section = f"\n\nRECENT CONVERSATION:\n{history_text}" if history_text else ""

    return (
        f"{SYSTEM_INSTRUCTION}"
        f"{rules}\n\n"
        f"CONTEXT:\n{context}"
        f"{history_section}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"ANSWER:"
    )