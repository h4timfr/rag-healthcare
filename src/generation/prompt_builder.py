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


def build_prompt(
    query: str,
    chunks: list[dict],
) -> str:
    """Assemble system instruction, focused evidence, and query."""

    target_scope = detect_target_scope(query)
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

    return (
        f"{SYSTEM_INSTRUCTION}"
        f"{scope_instruction}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"ANSWER:"
    )