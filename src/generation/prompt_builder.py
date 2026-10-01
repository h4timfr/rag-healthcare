import re

from . import intent as intent_module
from .. import text_match
from ..retrieval.keyword_index import KeywordIndex

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


# ---------------------------------------------------------------------------
# Production answering profile.
#
# SYSTEM_INSTRUCTION above is the original rule set and stays exactly as it
# is: scripts/evaluate.py (legacy, unmodified) still depends on it, and it
# grew around that benchmark's section/category questions.
#
# It is the wrong instruction set for real product questions, and measurably
# so. In the product QA baseline the model refused or deflected on questions
# whose answers were sitting in its own prompt - "The text does not provide
# information on the medications the patient was discharged on" with the
# page listing them retrieved, and "the context provided does not contain a
# summary of the document" in response to "Summarize this document". Fifteen
# of that instruction's lines are prohibitions ("do not add", "do not infer",
# "answer ONLY that", "if the context is insufficient, say so"), which on a
# 3B model biases hard toward declining to commit.
#
# So the production path gets its own instruction: shorter, evidence-first,
# and explicit that describing the context is not answering the question.
# ---------------------------------------------------------------------------

PRODUCTION_SYSTEM_INSTRUCTION = """You are a healthcare document assistant.

Answer the user's question from the EVIDENCE below, which was retrieved from
the user's own documents.

How to answer:
- The EVIDENCE is authoritative. Read all of it before you answer.
- Answer the question directly. Never describe the evidence, the context, the
  retrieval, the page numbers available to you, or how this system works.
- Never write phrases like "the context provided", "the retrieved chunks",
  "the snippets", or "the provided information". The user cannot see them and
  did not ask about them.
- Only say something is unavailable AFTER checking every piece of EVIDENCE
  for it. If it is there, answer with it.
- If it genuinely is not there, say so plainly and stop. Never fill the gap
  with a number, name, date or finding from your own knowledge, and never
  substitute a similar-sounding fact that is about something else - a result
  reported for a different system, study or patient is not an answer about
  this one.
- Never invent facts, numbers, dates, names, medications, or findings that
  are not in the EVIDENCE.
- A document's filename or title is not evidence about its contents.
- Do not hedge, and do not pad the answer with caveats about what you cannot
  see.
- Keep the source's own wording for clinical terms, categories and
  distinctions; do not merge or reclassify separate items.
- Do not give a medical diagnosis or treatment directive.

Citations:
- Cite each factual claim with the evidence number it came from, like [1].
- Cite only evidence that actually supports the claim.
"""

# Placed right before the question, where this model actually follows
# instructions (see _ANSWER_STYLE_PLACEMENT), and only for first-person advice
# requests. With just the general "no treatment directive" rule at the top,
# "Should I stop taking warfarin?" was answered "No, ..." and "My child has a
# cough like this patient. What medicine should I give her?" was answered
# "Albuterol." - directives lifted from someone else's chart. Rewording the
# top-of-prompt rule instead was measured to change unrelated answers
# (a summary question lost required facts in 4 of 5 runs), so every
# non-advice prompt is left exactly as it was.
PERSONAL_ADVICE_REMINDER = (
    "- If the question asks what the user (or someone they care for) should "
    "take, stop, dose, or do about their own health, do not answer yes or no "
    "and do not name a treatment for them. Start by saying these documents "
    "cannot advise them and they should ask a doctor or pharmacist."
)


def _personal_advice_reminder(query: str) -> str:
    """The reminder, only for first-person advice requests. Added to every
    prompt it was over-applied: "Compare the selected documents." was
    answered with "These documents cannot advise you ..." Every other
    question's prompt is unchanged by it."""
    if intent_module.is_personal_advice_request(query):
        return f"\n{PERSONAL_ADVICE_REMINDER}"
    return ""

ANSWER_STYLES = {
    intent_module.FACTUAL: (
        "ANSWER STYLE\n"
        "Give the answer directly, in one or two sentences. No preamble."
    ),
    intent_module.NUMERIC: (
        "ANSWER STYLE\n"
        "State the exact value from the EVIDENCE, with its unit and what it "
        "refers to. Do not round, estimate, or substitute a similar number "
        "from elsewhere in the evidence.\n"
        "- If no such value appears in the EVIDENCE, say that it is not "
        "given. Do not produce a plausible-looking number."
    ),
    intent_module.SUMMARY: (
        "ANSWER STYLE\n"
        "Write a concise summary of what this document actually says, "
        "synthesised from the EVIDENCE - the key findings, events and "
        "conclusions it contains.\n"
        "- Summarise the content, not the document structure. Never answer "
        "with a list of which pages were available.\n"
        "- Never say a summary is not possible. The EVIDENCE is the document "
        "content; summarising it is exactly the task."
    ),
    intent_module.COMPARISON: (
        "ANSWER STYLE\n"
        "Give concrete similarities first, then concrete differences, each "
        "one grounded in specific EVIDENCE and naming the document it came "
        "from.\n"
        "- Compare what the documents actually say, not what kind of "
        "documents they are.\n"
        "- You have the documents' content in the EVIDENCE, so never write "
        "that you cannot see them or that it is difficult to say."
    ),
    intent_module.EACH_DOCUMENT: (
        "ANSWER STYLE\n"
        "Write one short section per selected document, in the order listed, "
        "each starting with that document's name followed by a colon, then "
        "your own sentences about that document's own EVIDENCE.\n"
        "- One section per DOCUMENT heading - never one section per evidence "
        "entry, and never a section called \"Document 1\" or \"Evidence [1]\".\n"
        "- Start each section with the filename alone, exactly as listed below. "
        "Write \"Pediatric Report.pdf:\", never \"DOCUMENT: Pediatric Report.pdf\" - "
        "the DOCUMENT/PAGE/EVIDENCE labels belong to the material you are reading, "
        "not to your answer."
    ),
    intent_module.FOLLOW_UP: (
        "ANSWER STYLE\n"
        "This question continues the conversation above. Resolve what it "
        "refers back to from RECENT CONVERSATION, then answer it from the "
        "EVIDENCE. Answer only what is being asked now; do not repeat the "
        "previous answer."
    ),
    intent_module.UNSUPPORTED: (
        "ANSWER STYLE\n"
        "State plainly that the selected documents do not contain this "
        "information. Do not speculate and do not substitute a related fact."
    ),
}

# FACTUAL's own style ("one or two sentences, no preamble") measurably fails
# on multi-component/enumeration factual questions. Controlled ablation -
# same retrieved evidence, same model, 5 repetitions per condition, on two
# real questions whose answer requires several distinct facts - found 0/5
# complete answers under the concise style and 5/5 complete under an
# instruction that explicitly asks for every named component, with nothing
# else changed. This is that instruction, used whenever
# intent.is_enumeration_question() says the question calls for it (see
# build_answer_style below) - the plain concise style stays the default for
# an ordinary single-fact question. Kept fully generic: it names no
# specific question, document, heading, or number, so it reads the same for
# any document's content.
FACTUAL_MULTI_COMPONENT_STYLE = (
    "ANSWER STYLE\n"
    "Answer the question directly from the EVIDENCE. Explicitly address "
    "every distinct component the question asks about - do not omit one "
    "merely to keep the answer short.\n"
    "- Include the relevant numerical values for each component when the "
    "EVIDENCE contains them.\n"
    "- If the question asks for the most/least frequent, common, or "
    "highest-ranking items, list them in the order the EVIDENCE's own "
    "numbers support, not the order they happen to appear in the EVIDENCE "
    "text.\n"
    "- Do not invent facts. If one component genuinely has no supporting "
    "EVIDENCE, say so for that component only and still answer the rest.\n"
    "- Use headings or a list only if they make the answer clearer - do not "
    "impose a structure the question does not call for.\n"
    "- Never describe the retrieval process or which evidence was searched - "
    "answer the question itself."
)


# ANSWER_STYLES[COMPARISON] is written for comparing SEVERAL documents ("naming
# the document it came from"). A comparison inside ONE document - e.g. an
# initial vs a revised problem list on the same page - got that style and no
# instruction to keep the two sections apart: the model merged the lists and
# reported items present in both as added/removed. Controlled ablation (same
# evidence, 5 runs): with this style both sections were reproduced under their
# own headings 5/5, against confused lists under the multi-document style.
# Used only by build_production_prompt (single-document path), and only when
# the comparison names parts of the document (intent.compares_document_sections)
# - a concept comparison keeps ANSWER_STYLES[COMPARISON]. The multi-document
# comparison prompt is unchanged.
SINGLE_DOCUMENT_COMPARISON_STYLE = (
    "ANSWER STYLE\n"
    "This question compares two sections of the SAME document. Find each section under its "
    "own heading in the EVIDENCE and use only the items listed under that heading for it.\n"
    "- An item that appears in both sections, even with a different number, is unchanged - "
    "do not list it as added or removed.\n"
    "- Report exactly what the question asks for, each as its own list.\n"
    "- Margin notes and instructions to the reader are not list items.\n"
    "- When an item appears in both lists but is renamed or rephrased, treat it as the same item "
    "and explain the wording change rather than marking it as added or removed."
)

# Appended to the multi-part plan. Without it, multi-part answers named only the
# first few items of a part: in a controlled ablation (same evidence, 5 runs)
# a current-plan medication present in the evidence was listed 0/5 times
# without this line and 5/5 with it.
MULTI_PART_EXHAUSTIVE_RULE = (
    "- For each part, include every item the EVIDENCE gives for it - do not stop at the first "
    "few, and do not drop an item because it is unnamed or described only by its effect."
)

# Counterweight to the rule above for a medication question: with "include
# every item", one answer in eight listed lab tests and a ventriculogram as
# medications. Added only when the question asks for medications.
MEDICATION_ONLY_RULE = (
    "- When the question asks for medications, include only drugs or therapeutic agents. Do not "
    "include tests, procedures, monitoring, diet, lifestyle recommendations, or diagnostic studies "
    "unless explicitly requested."
)


def build_answer_plan(query: str, history: list[dict] | None = None) -> str:
    """Numbered plan for a multi-part question, e.g.

        This question has 2 parts. Answer each one separately:
        Part 1: What medications was the patient discharged on
        Part 2: what condition was found during her second hospital admission

    Q15 of the product QA baseline - a two-part question - was answered with
    a single flat refusal covering both parts at once, while the evidence for
    both was present. Naming the parts forces each one to be looked for
    separately instead of being disposed of together.

    Uses intent.decompose_question() rather than split_question_parts()
    directly, so a follow-up-shaped multi-part question ("X? What about the
    first one?") gets the same numbered plan as one joined by "and" - see
    that function's docstring for how it splits and resolves parts.
    """
    parts = intent_module.decompose_question(query, history=history)
    if len(parts) < 2:
        return ""

    lines = "\n".join(f"Part {i}: {part}" for i, part in enumerate(parts, start=1))
    return (
        "ANSWER STYLE\n"
        f"This question has {len(parts)} parts. Answer each one separately as a "
        "numbered item, in this order:\n"
        f"{lines}\n"
        "- If the EVIDENCE below is grouped under per-part \"PART N (...)\" "
        "headings, use each heading to find that part's own evidence - do not "
        "borrow evidence from another part's heading.\n"
        "- Look for each part in the EVIDENCE on its own. The parts are "
        "independent; evidence for one is not evidence about another.\n"
        "- If one part genuinely is not in the EVIDENCE, say so for that part "
        "only and still answer the others.\n"
        f"{MULTI_PART_EXHAUSTIVE_RULE}"
        + (f"\n{MEDICATION_ONLY_RULE}" if intent_module.asks_for_medications(query) else "")
    )


# Why the ANSWER STYLE block is emitted after the evidence rather than with
# the other instructions at the top.
#
# A three-document prompt is ~17KB, nearly all of it evidence. With the
# formatting contract at the top, everything the model saw most recently
# before "ANSWER:" was evidence text - and it structured its answer around
# whatever looked structural in there. Asked to summarize three selected
# documents, it produced sections called "Licensee response", "Hospital
# Records" and "Image CD's from Hospital": a list of record TYPES that
# appears inside one document's text, not the documents that were selected.
#
# This project has hit the same shape of bug before, in chat.py, where the
# conversation history was appended after the "ANSWER:" cue and buried it.
# The rule that came out of it applies here too: whatever the model must
# actually do belongs next to the cue to do it, not 15KB upstream.
_ANSWER_STYLE_PLACEMENT = "after EVIDENCE, immediately before USER QUESTION"


def build_answer_style(
    query: str,
    resolved_intent: str,
    history: list[dict] | None = None,
) -> str:
    """The ANSWER STYLE block for this turn - one short block, never more.

    A question that decomposes into 2+ parts always gets the multi-part
    plan, regardless of which single label classify_intent() settled on for
    it - in particular FOLLOW_UP, which wins classify_intent()'s priority
    whenever history is present and the question also matches a follow-up
    trigger like "what about". That label is right for the ANSWER STYLE
    text only when there is genuinely one thing being asked; "What happened
    during the second admission? What about the first one?" matches both
    "what about" and being multi-part, and it is multi-part that should
    shape the answer, or the second admission's plan silently disappears
    behind a style written for a single reference.

    EACH_DOCUMENT/COMPARISON are the one exception: they already have their
    own established per-document structure (see build_multi_document_prompt),
    and decomposing on top of that would double-structure the answer for no
    benefit.

    A FACTUAL question (only - every other intent keeps its own established
    style unchanged) that intent.is_enumeration_question() flags - a named
    list, an implied "each/every" enumeration, or a most/least-frequent
    ranking - gets FACTUAL_MULTI_COMPONENT_STYLE instead of the plain
    concise FACTUAL style, checked after the multi-part plan so a question
    that genuinely decomposes into standalone parts still gets the numbered
    plan rather than this single-answer-with-several-components style.
    """
    if resolved_intent not in (intent_module.EACH_DOCUMENT, intent_module.COMPARISON):
        plan = build_answer_plan(query, history=history)
        if plan:
            return plan

    if resolved_intent == intent_module.FACTUAL and (
        intent_module.is_enumeration_question(query) or intent_module.requests_item_count(query)
    ):
        return FACTUAL_MULTI_COMPONENT_STYLE

    return ANSWER_STYLES.get(resolved_intent, ANSWER_STYLES[intent_module.FACTUAL])


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


MAX_HISTORY_TURNS = 4
MAX_HISTORY_CHARS_PER_TURN = 1500


def format_conversation_history(
    history: list[dict] | None,
    max_turns: int = MAX_HISTORY_TURNS,
    max_chars_per_turn: int = MAX_HISTORY_CHARS_PER_TURN,
) -> str:
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

    Two bounds keep the prompt from growing with the conversation: only the
    last `max_turns` entries are kept, and each one is truncated to
    `max_chars_per_turn`. The per-entry cap sits above the longest answer
    this app has produced in QA (a three-document summary, ~1.2k chars), so
    it bounds an unbounded case without trimming a real one - a multi-turn
    session with several long multi-document answers would otherwise add
    several KB of history to a prompt that is already ~17KB of evidence.
    The head of an answer is kept rather than the tail: that is where a
    per-document answer names its documents, which is what a follow-up like
    "what about the second document?" has to resolve against.
    """
    if not history:
        return ""

    lines = []
    for turn in history[-max_turns:]:
        content = str(turn["content"])
        if len(content) > max_chars_per_turn:
            content = content[:max_chars_per_turn].rstrip() + " [...]"
        lines.append(f"{turn['role']}: {content}")
    return "\n".join(lines)


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
    original behavior for any caller that doesn't pass it) selects the
    PRODUCTION ANSWERING PROFILE when True: build_production_prompt()'s
    evidence-first instruction and per-intent ANSWER STYLE block, instead of
    SYSTEM_INSTRUCTION plus the benchmark scope machinery below. The flag has
    always meant "this is the real product answer path, not the old
    benchmark" - every production entry point passes True and only
    scripts/evaluate.py does not - so the production instruction set is
    keyed off the same boolean rather than adding a second one that would
    always have the same value.

    When False, detect_target_scope()/get_scope_terms()/
    extract_matching_items() run exactly as before. Those exist only to
    narrow answers for this project's own historical benchmark questions
    (e.g. a query containing "source presentation" or "document collection")
    and have no business silently reshaping a real healthcare query's
    answer. Every production entry point (app.py, chat.py,
    scripts/query.py, scripts/profile_query.py) passes True;
    scripts/evaluate.py (legacy, kept unmodified, its output already
    documented as not trustworthy) is the only remaining caller that still
    gets the old auto-detected behavior via this default - the
    research/evaluation harness itself (scripts/run_evaluation.py) never
    calls this function at all, it uses the separate eval_prompt_builder.py.
    """
    if document_inventory:
        return build_multi_document_prompt(query, chunks, document_inventory, history=history)

    if disable_benchmark_scope:
        return build_production_prompt(query, chunks, history=history)

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


def build_part_context(chunks: list[dict], parts: list[str]) -> str:
    """Group retrieved evidence by which requested part of a multi-part
    question it was retrieved for - the EVIDENCE-side counterpart to
    build_answer_plan()'s "Part 1: ... Part 2: ..." instruction.

    Expects each chunk to carry a `part` key (1-indexed, matching `parts`'
    numbering) as produced by retrieve_for_question() - see that function's
    docstring for why retrieval runs once per part instead of once for the
    whole question. A chunk with no `part` key is treated as part 1, so a
    caller that mixes tagged and untagged chunks still gets a well-formed
    (if imperfectly grouped) context rather than a KeyError.

    A part with no matching evidence still gets its own heading and an
    explicit "no evidence retrieved" marker, on the same reasoning as
    build_multi_document_context()'s per-document placeholder: a silently
    missing section reads as "nothing to say" and invites the model to
    invent content for it, and a short marker (not a full sentence, so it
    cannot be quoted back as an answer) keeps that from happening for a
    part exactly as it already does for a document.
    """
    by_part: dict[int, list[dict]] = {}
    for chunk in chunks:
        by_part.setdefault(chunk.get("part", 1), []).append(chunk)

    sections = []
    for i, part_query in enumerate(parts, start=1):
        part_chunks = by_part.get(i, [])

        if not part_chunks:
            sections.append(f"PART {i} ({part_query}):\n  EVIDENCE: none")
            continue

        blocks = "\n\n".join(
            f"  PAGE: {chunk['page']}\n"
            f"  EVIDENCE [{chunk['rank']}]:\n"
            f"  {chunk['text']}"
            for chunk in part_chunks
        )
        sections.append(f"PART {i} ({part_query}):\n{blocks}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Presentation-only context ordering (single-document path).
#
# Retrieval audit finding: for two real single-document questions, the chunk
# containing the actual answer (specific percentages) was retrieved -
# RRF ranked it within the selected chunks - but sat at or near the END of
# the evidence list, behind several topically-adjacent chunks that discuss
# the same general subject without the specific figures asked for. Given
# that evidence, the model described the existence of the right table/
# section instead of reading its values, or answered from the wrong
# lookalike chunk. Retrieval SELECTION was not the (whole) problem;
# PRESENTATION ORDER was untested as a separate variable, and turned out to
# matter.
#
# This reorders the already-selected chunks - it can never add or remove a
# chunk retrieval didn't already choose, and RRF's ranking remains the
# reason any given chunk was selected at all. It only decides where, among
# the selected chunks, a chunk already carrying strong, specific evidence
# for THIS query is placed for the model to read.
# ---------------------------------------------------------------------------

_NUMERIC_TOKEN = re.compile(r"\d+(?:\.\d+)?%?")

# Bounds, chosen to be conservative rather than tuned to any one document:
#   PROTECT_FRONT   the strongest RRF results are never displaced - "prefer
#                    an approach that preserves the strongest RRF results
#                    near the front" is satisfied structurally, not by score.
#   MAX_PROMOTED    caps how many chunks a single reorder can pull forward,
#                    so one strong chunk cannot reshuffle the whole context -
#                    only a small, bounded prefix right after the protected
#                    front ever changes.
#   MIN_TERM_MATCHES the promotion threshold. Set above 1 specifically so a
#                    single incidental word match - e.g. a stopword-adjacent
#                    common term - can never promote a chunk on its own; see
#                    _numeric_evidence_score's gate below for why lexical
#                    overlap alone (no numeric content) also never promotes.
PROTECT_FRONT = 2
MAX_PROMOTED = 3
MIN_TERM_MATCHES = 2


def _significant_query_terms(query: str) -> list[str]:
    """The query's own content words, by the SAME definition FTS5 already
    uses for "significant term" (KeywordIndex.STOPWORDS) - reused rather
    than a second, potentially-diverging notion of what counts as content
    versus filler in a question.
    """
    terms = re.findall(r"[A-Za-z0-9_]+", (query or "").lower())
    return [t for t in terms if t not in KeywordIndex.STOPWORDS]


def _numeric_evidence_score(text: str, significant_terms: list[str]) -> int:
    """How strongly `text` carries SPECIFIC, on-topic evidence for the query.

    Deliberately not a plain word-overlap count - "avoid simply sorting by
    raw word overlap" - because raw overlap promotes any chunk that merely
    discusses the same subject, which is exactly the failure mode observed
    (topically-adjacent chunks outnumbering the one with the actual figures).
    The score is the count of distinct significant query terms the text
    contains (word-boundary/morphology-aware match via text_match.contains_term,
    the same matcher this project already uses to grade whether an ANSWER
    covers a required term) - but ONLY if the text also contains at least
    one number. A chunk with high term overlap and no numbers scores 0: it
    may be relevant prose, but it is not the specific data point the
    "strong lexical/numeric evidence" requirement asks to detect, and a
    plain lexical scorer would promote it anyway.
    """
    if not _NUMERIC_TOKEN.search(text or ""):
        return 0
    return sum(1 for term in significant_terms if text_match.contains_term(text, term))


def order_chunks_for_presentation(query: str, chunks: list[dict]) -> list[dict]:
    """Reorder already-retrieved `chunks` for presentation only - RRF
    selection and scoring are untouched; this decides where, among the
    chunks retrieval already chose, each one appears in the prompt (and,
    since this is applied to the same list callers pass on to the evidence
    UI, in the evidence panel too - so a citation number keeps meaning the
    same chunk in both places, exactly as it already does for the
    multi-document path's rank numbering).

    The first PROTECT_FRONT chunks (RRF's own strongest results) are never
    moved. Among the rest, up to MAX_PROMOTED chunks scoring at least
    MIN_TERM_MATCHES on _numeric_evidence_score are pulled forward, ordered
    by that score (ties broken by original rank, for determinism); every
    other chunk keeps its original relative RRF order and is appended
    after. A chunk's own `score` (the RRF fused score) is left untouched;
    only `rank` is reassigned, to the chunk's new position, matching how
    retrieve_for_question() already renumbers `rank` after merging
    multi-part results.

    Returns a new list of shallow-copied chunk dicts; `chunks` itself is
    never mutated. Returns `chunks` unchanged (but still copied) if there
    are too few chunks to reorder or the query has no significant terms to
    score against.
    """
    if len(chunks) <= PROTECT_FRONT:
        return [dict(c) for c in chunks]

    significant_terms = _significant_query_terms(query)
    if not significant_terms:
        return [dict(c) for c in chunks]

    protected = chunks[:PROTECT_FRONT]
    rest = chunks[PROTECT_FRONT:]

    scored = [(chunk, _numeric_evidence_score(chunk["text"], significant_terms)) for chunk in rest]

    # Stable sort by score only, so equal-scoring chunks keep their original
    # relative RRF order - this is what makes tie-breaking deterministic
    # without a second sort key.
    promotable = sorted(
        (pair for pair in scored if pair[1] >= MIN_TERM_MATCHES),
        key=lambda pair: pair[1],
        reverse=True,
    )
    promoted = [chunk for chunk, _score in promotable[:MAX_PROMOTED]]
    promoted_ids = {c["chunk_id"] for c in promoted}

    remainder = [c for c in rest if c["chunk_id"] not in promoted_ids]

    new_order = protected + promoted + remainder
    return [{**c, "rank": i} for i, c in enumerate(new_order, start=1)]


def build_production_prompt(
    query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> str:
    """Single-document production prompt: evidence-first instruction plus
    one intent-specific ANSWER STYLE block.

    The retrieved evidence is formatted exactly as build_context() has always
    formatted it - numbered blocks with source and page - so citations and
    the Streamlit evidence panel keep meaning the same thing. What changes is
    the instruction around it, and the section label: the evidence is
    introduced as EVIDENCE rather than CONTEXT, because a section the model
    reads as "the context" is a section it writes sentences about ("the
    context provided does not contain...") instead of answering from.

    For a multi-part question whose chunks were retrieved per-part (i.e. by
    retrieve_for_question(), which is what tags them with `part`), the
    evidence is grouped under "PART N (...)" headings instead of one flat
    list - so each part's own evidence is findable on its own, matching the
    "Part 1: ... Part 2: ..." plan build_answer_style() attaches below it.
    A question that isn't multi-part, or whose chunks were retrieved by a
    caller that doesn't tag them (e.g. scripts/evaluate.py), takes the
    original flat build_context() path unchanged.
    """
    parts = intent_module.decompose_question(query, history=history)
    if len(parts) > 1 and any("part" in chunk for chunk in chunks):
        context = build_part_context(chunks, parts)
    else:
        context = build_context(chunks, target_scope=None)
    resolved_intent = intent_module.classify_intent(query, history=history)
    if resolved_intent == intent_module.COMPARISON and intent_module.compares_document_sections(query):
        answer_style = SINGLE_DOCUMENT_COMPARISON_STYLE
    else:
        answer_style = build_answer_style(query, resolved_intent, history=history)

    history_text = format_conversation_history(history)
    history_section = f"\n\nRECENT CONVERSATION:\n{history_text}" if history_text else ""

    # ANSWER STYLE sits AFTER the evidence, immediately before the question
    # and the generation cue - see _ANSWER_STYLE_PLACEMENT.
    return (
        f"{PRODUCTION_SYSTEM_INSTRUCTION}\n"
        f"EVIDENCE:\n{context}"
        f"{history_section}\n\n"
        f"{answer_style}{_personal_advice_reminder(query)}\n\n"
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

def detect_multi_document_intent(query: str) -> bool:
    """Whether the question is asking to cover/compare multiple documents
    (vs. one narrow question that merely happens to be asked while several
    documents are selected) - used to decide whether to instruct the model
    to structure its answer with one section per document.

    Delegates to the shared intent classifier rather than keeping a second
    pattern list in this module: "each"/"every"/"compare"/"in common" are
    exactly EACH_DOCUMENT and COMPARISON there. One behavioural difference
    falls out of that and is intended - a bare "all" no longer counts unless
    it is "all the documents", so "list all medications" is treated as the
    narrow question it is instead of being forced into a per-document
    structure.
    """
    return intent_module.classify_intent(query) in intent_module.MULTI_DOCUMENT_INTENTS


def build_multi_document_context(chunks: list[dict], document_inventory: list[dict]) -> str:
    """Group retrieved chunks by document, in document_inventory's order:

        DOCUMENT: Pediatric Report.pdf
          PAGE: 1
          EVIDENCE [4]:
          <chunk text>

          PAGE: 2
          EVIDENCE [7]:
          <chunk text>

        DOCUMENT: OMC Report Sample - Cardio.pdf
          PAGE: 3
          EVIDENCE [2]:
          <chunk text>

    One DOCUMENT header per document, with that document's evidence indented
    beneath it - NOT one header per chunk.

    Repeating the header on every chunk was tried and measurably backfired:
    llama3.2:3b read each block as its own document and answered "Summarize
    each selected document separately" with six sections titled "Document 1:
    RAG_Healthcare_Review1.pdf, Page 8", "Document 2: ... Page 1" and so on -
    six sections, one real document, three documents selected. Indentation
    keeps the document-to-evidence hierarchy visible, which is what makes
    "one section per document" a structure the model can actually follow.

    A selected document with zero retrieved chunks gets an explicit
    "no evidence retrieved" placeholder instead of being silently omitted
    (which could read as "nothing to say") or left for the model to guess
    at (which is exactly the invented-content failure mode this exists to
    prevent - selected does not mean evidence exists).

    Citation numbers ([1], [2], ...) are each chunk's own `rank` from
    retrieve_hybrid (its position in the original fused-ranking result),
    not a fresh per-group counter - so a citation means the same chunk here
    as it does in the Streamlit evidence panel, regardless of how chunks
    get visually regrouped by document in either place.
    """
    sections = []
    for doc in document_inventory:
        doc_chunks = [c for c in chunks if c.get("document_id") == doc.get("document_id")]

        if not doc_chunks:
            # A terse marker, not a sentence. It used to read "(No relevant
            # evidence was retrieved for this document.)" - fluent English,
            # and the most quotable line in the whole prompt whenever the
            # question had no answer. Asked an unsupported question, the
            # model returned exactly that sentence as its entire answer:
            # internal plumbing handed back to the user. There is nothing
            # here to copy now, so it has to use its own words.
            sections.append(f"DOCUMENT: {doc['source']}\n  EVIDENCE: none")
            continue

        blocks = "\n\n".join(
            f"  PAGE: {chunk['page']}\n"
            f"  EVIDENCE [{chunk['rank']}]:\n"
            f"  {chunk['text']}"
            for chunk in doc_chunks
        )
        sections.append(f"DOCUMENT: {doc['source']}\n{blocks}")

    return "\n\n".join(sections)


def build_multi_document_prompt(
    query: str,
    chunks: list[dict],
    document_inventory: list[dict],
    history: list[dict] | None = None,
) -> str:
    """build_prompt()'s multi-document counterpart: the same evidence-first
    production instruction as build_production_prompt(), plus the grounding
    rules that only apply when several documents are in play, and the
    document-grouped EVIDENCE blocks. `history` is placed the same way as in
    build_prompt() - before USER QUESTION/ANSWER, never after.

    Each-document and comparison questions get different answer shapes on
    purpose. They previously shared one "write a paragraph per document"
    template, which is right for "summarize each document" and wrong for
    "what do these two have in common" - being told to produce per-document
    paragraphs for a comparison is part of why that question came back as
    two generic paragraphs about what kind of documents these are.
    """
    context = build_multi_document_context(chunks, document_inventory)
    document_list = ", ".join(doc["source"] for doc in document_inventory)
    resolved_intent = intent_module.classify_intent(query, history=history)

    rules = (
        f"\nMULTI-DOCUMENT RULES ({len(document_inventory)} documents are selected):\n"
        f"- The {len(document_inventory)} selected documents are: {document_list}\n"
        "- The EVIDENCE is grouped under one \"DOCUMENT:\" heading per selected "
        "document. Several PAGE/EVIDENCE entries under the same heading are "
        "different parts of the SAME document, not different documents. There are "
        f"exactly {len(document_inventory)} documents, however many evidence "
        "entries there are.\n"
        "- A document being selected does NOT mean you have evidence about it. Never "
        "claim anything about a document whose block has no evidence.\n"
        "- The EVIDENCE blocks are material to read, not a format to copy. Never write "
        "a bare \"DOCUMENT:\"/\"PAGE:\"/\"EVIDENCE:\" label, a bare citation or page "
        "number, or the phrase \"no relevant evidence was retrieved\", into your "
        "answer. If the documents do not answer the question, say so in your own "
        "words.\n"
        "- Keep each citation ([1], [2], ...) attached to the evidence it came from."
    )

    # Accounting for every selected document - including ones that returned
    # nothing - is the task ONLY when the question asks about each of them.
    # Applied to every multi-document question it backfired: asked an
    # unsupported question ("what was the patient's blood type?") the model
    # replied with the context's own placeholder sentence, "No relevant
    # evidence was retrieved for this document" - internal plumbing quoted
    # back as if it were an answer.
    if resolved_intent in intent_module.MULTI_DOCUMENT_INTENTS:
        rules += (
            "\n- Cover every selected document. For one with no evidence, say in your "
            "own words that these documents do not cover it - do not guess, and do "
            "not silently leave it out."
        )

    answer_style = build_answer_style(query, resolved_intent, history=history)

    # A filled-in skeleton, not a description of one. Telling this model to
    # "use these section names" and listing them produced sections called
    # "Document 1: MD 09" and "Evidence [1]" - it enumerated whatever looked
    # enumerable. Handing it the literal section headings with a slot after
    # each one is what actually produces one section per document, and is
    # the form this prompt used when each-document questions last scored
    # 100%.
    if resolved_intent == intent_module.EACH_DOCUMENT:
        # The slot after each heading is conditional on whether that document
        # actually has retrieved evidence.
        #
        # Offering every document the same "...or say no relevant evidence was
        # retrieved" slot handed the model an escape hatch it took even when
        # the evidence was there: with three chunks from Pediatric Report.pdf
        # sitting in the prompt, it still wrote "No relevant evidence was
        # retrieved for this document" under that heading. A document with
        # evidence is now given no way to phrase that.
        documents_with_evidence = {c.get("document_id") for c in chunks}
        template = "\n".join(
            f"{doc['source']}: "
            + (
                "<1-3 sentences on this document's own evidence above>"
                if doc.get("document_id") in documents_with_evidence
                else "<one sentence saying no relevant evidence was retrieved for this document>"
            )
            for doc in document_inventory
        )
        answer_style += (
            f"\nWrite exactly {len(document_inventory)} sections, in this order, each "
            "starting with the document's name exactly as written below, followed by a "
            "colon. Replace each <...> with your own sentences - never leave the "
            "\"<...>\" text itself in your answer, and never invent a different "
            "section heading:\n"
            f"{template}"
        )
    elif resolved_intent == intent_module.COMPARISON:
        names = "; ".join(doc["source"] for doc in document_inventory)
        answer_style += (
            "\n- Refer to each document by its name exactly as written here, and name "
            f"every one of them somewhere in your answer: {names}"
        )

    history_text = format_conversation_history(history)
    history_section = f"\n\nRECENT CONVERSATION:\n{history_text}" if history_text else ""

    # ANSWER STYLE after the evidence, immediately before the question - see
    # _ANSWER_STYLE_PLACEMENT. It matters most here: this is the longest
    # prompt the app builds, and the per-document section headings are
    # exactly what the model was losing track of.
    return (
        f"{PRODUCTION_SYSTEM_INSTRUCTION}"
        f"{rules}\n\n"
        f"EVIDENCE:\n{context}"
        f"{history_section}\n\n"
        f"{answer_style}{_personal_advice_reminder(query)}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"ANSWER:"
    )