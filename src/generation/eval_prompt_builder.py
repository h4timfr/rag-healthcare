"""Neutral, contamination-free prompt builder used ONLY by the evaluation harness.

src/generation/prompt_builder.py (the production prompt used by app.py, chat.py,
and scripts/query.py) contains question-specific scope detection that can
replace retrieved context with a hard-coded bullet list matching some
benchmark questions' own expected_terms - see the evaluation audit. That
logic must never run during evaluation, so this module is deliberately
independent of it: no detect_target_scope(), no get_scope_terms(), no
extract_matching_items(), no per-question special-casing. It never receives
expected_pages/expected_terms/ground truth of any kind - only the query and
the chunks retrieval actually returned.

Kept separate from prompt_builder.py rather than modifying it, per the task's
instruction not to change existing production prompt behavior unless needed.
"""

EVAL_SYSTEM_INSTRUCTION = """You are a document question-answering assistant.

Answer the user's question using ONLY the retrieved context below.

Rules:
- Use only information supported by the supplied context.
- If the context does not contain enough information to answer, say so explicitly.
- Do not invent facts, figures, or details that are not present in the context.
- Cite the source document and page number for factual claims using the
  numbered references shown in the context, e.g. [1], [2].
- Do not provide medical diagnosis or treatment directives."""


def build_eval_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered evidence blocks. No filtering,
    reordering, or content substitution of any kind."""
    return "\n\n".join(
        f"[{i}] Source: {chunk['source']}, page {chunk['page']}\n{chunk['text']}"
        for i, chunk in enumerate(chunks, start=1)
    )


def build_eval_prompt(query: str, chunks: list[dict]) -> str:
    """Assemble the neutral system instruction, raw retrieved context, and query."""
    context = build_eval_context(chunks)
    return (
        f"{EVAL_SYSTEM_INSTRUCTION}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"ANSWER:"
    )
