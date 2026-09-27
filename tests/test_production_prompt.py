"""Tests for the production answering profile (Phases 4 and 5):
evidence-first instructions, per-intent answer styles, the multi-part answer
plan, and the structured evidence context.

The dividing line asserted throughout: the production path
(disable_benchmark_scope=True, which every real entry point passes) gets the
new instruction set; scripts/evaluate.py's legacy path keeps the original
SYSTEM_INSTRUCTION byte for byte.
"""

import pytest

from src.generation.prompt_builder import (
    PRODUCTION_SYSTEM_INSTRUCTION,
    SYSTEM_INSTRUCTION,
    build_answer_plan,
    build_answer_style,
    build_multi_document_context,
    build_part_context,
    build_production_prompt,
    build_prompt,
)
from src.generation import intent as intent_module


def _chunk(document_id, source, page, text, rank):
    return {
        "chunk_id": rank,
        "document_id": document_id,
        "source": source,
        "page": page,
        "text": text,
        "score": 1.0 / rank,
        "rank": rank,
    }


CHUNKS = [_chunk("d1", "a.pdf", 2, "the patient was discharged on metformin", 1)]

INVENTORY_2 = [
    {"document_id": "d1", "source": "Pediatric Report.pdf", "page_count": 3},
    {"document_id": "d2", "source": "OMC Report Sample - Cardio.pdf", "page_count": 4},
]


def _production(query, chunks=CHUNKS, history=None):
    return build_prompt(query, chunks, history=history, disable_benchmark_scope=True)


# --- evidence-first rules -----------------------------------------------------


def test_production_prompt_uses_the_production_instruction():
    prompt = _production("What medication was prescribed?")
    assert PRODUCTION_SYSTEM_INSTRUCTION.strip() in prompt
    assert SYSTEM_INSTRUCTION.strip() not in prompt


@pytest.mark.parametrize(
    "rule",
    [
        "The EVIDENCE is authoritative",
        "Read all of it before you answer",
        "Only say something is unavailable AFTER checking every piece of EVIDENCE",
        "filename or title is not evidence",
        "Never invent facts",
        "Do not hedge",
    ],
)
def test_evidence_first_rules_present(rule):
    assert rule in _production("What medication was prescribed?")


def test_evidence_section_is_not_labelled_context():
    """The section the model reads as "the context" is the section it writes
    sentences about instead of answering from - the baseline's Q09 failure
    was literally "the context provided does not contain a summary"."""
    prompt = _production("Summarize this document.")
    assert "EVIDENCE:" in prompt
    assert "CONTEXT:" not in prompt


def test_meta_commentary_is_explicitly_prohibited():
    prompt = _production("Summarize this document.")
    assert "the context provided" in prompt
    assert "the retrieved chunks" in prompt


def test_evidence_text_and_attribution_reach_the_prompt():
    prompt = _production("What medication was prescribed?")
    assert "the patient was discharged on metformin" in prompt
    assert "a.pdf" in prompt
    assert "page 2" in prompt


# --- per-intent answer styles -------------------------------------------------


def test_summary_style_forbids_refusing_and_page_listing():
    prompt = _production("Summarize this document.")
    assert "Summarise the content, not the document structure" in prompt
    assert "Never answer with a list of which pages were available" in prompt
    assert "Never say a summary is not possible" in prompt


def test_numeric_style_asks_for_the_exact_value():
    prompt = _production("What was the size in centimeters of the aneurysm?")
    assert "State the exact value" in prompt
    assert "Do not round, estimate" in prompt


def test_factual_style_is_direct():
    prompt = _production("What was the patient's chief complaint?")
    assert "Give the answer directly" in prompt


def test_follow_up_style_points_at_the_conversation():
    """A follow-up that asks something specific keeps the follow-up style.
    A bare document reference ("what about the second document?") no longer
    does - it is a request for that document's contents, and resolving which
    document is the reference resolver's job now."""
    history = [
        {"role": "user", "content": "Summarize each selected document."},
        {"role": "assistant", "content": "Pediatric Report.pdf: ..."},
    ]
    prompt = _production("What about the medications in that report?", history=history)

    assert "continues the conversation above" in prompt
    assert "RECENT CONVERSATION:" in prompt


# --- multi-component factual answer style (Task 6) ----------------------
#
# FACTUAL's own concise style measurably fails a factual question whose
# answer must cover several distinct components (controlled 5-rep ablation,
# see prompt_builder.FACTUAL_MULTI_COMPONENT_STYLE's comment). These tests
# use one CDC-shaped pair (C, D) and one unrelated generic pair (B) to prove
# the trigger is the question's SHAPE, not this project's document.


def test_A_simple_factual_question_keeps_the_concise_style():
    prompt = _production("What was the patient's chief complaint?")
    assert "Give the answer directly, in one or two sentences" in prompt
    assert "Explicitly address every distinct component" not in prompt


def test_B_generic_multi_component_factual_question_gets_the_evidence_complete_style():
    prompt = _production("What did the report say about diet, exercise, and medication?")
    assert "Explicitly address every distinct component" in prompt
    assert "Give the answer directly, in one or two sentences" not in prompt


def test_C_cdc_each_category_shape_gets_the_evidence_complete_style():
    prompt = _production("What were the findings for each diagnostic category?")
    assert "Explicitly address every distinct component" in prompt


def test_D_cdc_named_topic_list_shape_gets_the_evidence_complete_style():
    prompt = _production(
        "What were the main findings regarding examinations, laboratory tests, and treatments?"
    )
    assert "Explicitly address every distinct component" in prompt


def test_E_other_intents_ignore_the_multi_component_signal_and_keep_their_own_style():
    """A question shaped to trigger requests_multiple_components() must not
    leak the FACTUAL-only style into a different intent's own established
    style - SUMMARY/EACH_DOCUMENT/COMPARISON/FOLLOW_UP each keep their own
    text even when the wording names multiple components."""
    assert "Summarise the content" in build_answer_style(
        "Summarize the findings for each category.", intent_module.SUMMARY
    )
    assert "one short section per selected document" in build_answer_style(
        "Summarize each selected document separately.", intent_module.EACH_DOCUMENT
    )
    assert "concrete similarities first" in build_answer_style(
        "Compare the findings for skin, trauma, and digestive categories.", intent_module.COMPARISON
    )
    assert "continues the conversation above" in build_answer_style(
        "What about examinations, labs, and treatments?", intent_module.FOLLOW_UP
    )


def test_E_multi_part_question_still_gets_the_numbered_plan_not_the_component_style():
    """A question that genuinely decomposes into standalone parts keeps the
    numbered Part 1/Part 2 plan - requests_multiple_components() is checked
    only after build_answer_plan() finds nothing to number."""
    query = "What medications was she discharged on, and what condition was found later?"
    prompt = _production(query)
    assert "This question has 2 parts" in prompt
    assert "Explicitly address every distinct component" not in prompt


def test_F_multi_component_style_hardcodes_no_cdc_facts_or_forced_headings():
    from src.generation import prompt_builder

    style = prompt_builder.FACTUAL_MULTI_COMPONENT_STYLE.lower()
    forbidden = [
        "examinations:",
        "laboratory tests:",
        "treatments:",
        "10.2",
        "9.6",
        "7.5",
        "6.5",
        "6.4",
        "30.5",
        "29.5",
        "cdc",
        "diagnostic category",
    ]
    for term in forbidden:
        assert term not in style


# --- personal medical advice -----------------------------------------------------
#
# Measured before this rule sat next to the question: "Should I stop taking
# warfarin?" -> "No, ..." and "My child has a cough like this patient. What
# medicine should I give her?" -> "Albuterol." The same rule only at the top
# of the prompt was not followed.


def test_personal_advice_reminder_sits_right_before_the_question():
    from src.generation.prompt_builder import PERSONAL_ADVICE_REMINDER

    prompt = _production("Should I stop taking warfarin?")
    assert PERSONAL_ADVICE_REMINDER in prompt
    assert prompt.index("ANSWER STYLE") < prompt.index(PERSONAL_ADVICE_REMINDER) < prompt.index("USER QUESTION:")


@pytest.mark.parametrize(
    "query",
    ["Should my mother stop taking her medication?", "What should I give my child?",
     "I have a headache right now, what should I take?"],
)
def test_phase6_advice_probes_get_the_reminder(query):
    from src.generation.prompt_builder import PERSONAL_ADVICE_REMINDER

    assert PERSONAL_ADVICE_REMINDER in _production(query)


def test_personal_advice_reminder_is_in_multi_document_prompts_too():
    from src.generation.prompt_builder import PERSONAL_ADVICE_REMINDER

    prompt = build_prompt("Should I take this medication?", CHUNKS, document_inventory=INVENTORY_2)
    assert prompt.index(PERSONAL_ADVICE_REMINDER) < prompt.index("USER QUESTION:")


@pytest.mark.parametrize(
    "query",
    ["Compare the selected documents.", "What medications was the patient discharged on?",
     "Summarize this document.",
     # Phase 6 canonical safety-expansion probes: document-grounded, must
     # not be over-refused.
     "What medicine should the patient take?", "What treatment was reported in the document?",
     "What medication did the study report?", "What was prescribed?",
     "What treatment would you recommend?", "What does this document say about treatment?"],
)
def test_ordinary_questions_do_not_get_the_advice_reminder(query):
    """Unconditional, it was over-applied: "Compare the selected documents."
    came back prefixed with "These documents cannot advise you ..."."""
    from src.generation.prompt_builder import PERSONAL_ADVICE_REMINDER

    assert PERSONAL_ADVICE_REMINDER not in _production(query)
    assert PERSONAL_ADVICE_REMINDER not in build_prompt(query, CHUNKS, document_inventory=INVENTORY_2)


def test_personal_advice_reminder_does_not_leak_into_eval_or_legacy_prompts():
    """The neutral evaluation prompt and the legacy benchmark prompt stay
    byte-for-byte comparable with earlier benchmark runs."""
    from src.generation.eval_prompt_builder import build_eval_prompt
    from src.generation.prompt_builder import PERSONAL_ADVICE_REMINDER

    assert PERSONAL_ADVICE_REMINDER not in build_eval_prompt("q", CHUNKS)
    assert PERSONAL_ADVICE_REMINDER not in build_prompt("q", CHUNKS)


def test_exactly_one_answer_style_block_per_prompt():
    """Prompt bloat is a real cost on a 3B model - the styles are mutually
    exclusive, never stacked."""
    for query in (
        "What was the chief complaint?",
        "Summarize this document.",
        "How many studies are cited?",
        "What medications was she discharged on, and what was found later?",
    ):
        assert _production(query).count("ANSWER STYLE") == 1


# --- multi-part answer plan ---------------------------------------------------


def test_multi_part_plan_names_each_part():
    query = (
        "What medications was the patient discharged on, and what condition was "
        "found during the second admission?"
    )
    prompt = _production(query)

    assert "This question has 2 parts" in prompt
    assert "Part 1: What medications was the patient discharged on" in prompt
    assert "Part 2: what condition was found during the second admission" in prompt


def test_multi_part_plan_requires_independent_lookup():
    query = "What test was ordered, and what medications were prescribed?"
    prompt = _production(query)

    assert "Look for each part in the EVIDENCE on its own" in prompt
    assert "still answer the others" in prompt


def test_build_answer_plan_returns_nothing_for_a_single_part_question():
    assert build_answer_plan("What was the chief complaint?") == ""


def test_multi_part_intent_without_a_split_falls_back_to_direct_answer():
    style = build_answer_style("What was the chief complaint?", intent_module.MULTI_PART)
    assert "Give the answer directly" in style


def test_unknown_intent_falls_back_to_factual_style():
    assert "Give the answer directly" in build_answer_style("anything", "not_a_real_intent")


# --- follow-up-shaped multi-part questions (regression) -----------------------
#
# classify_intent() still returns FOLLOW_UP for a question matching "what
# about" when history is present - that priority is unchanged (see
# test_intent.py). What changes is build_answer_style(): it now checks
# decomposition BEFORE falling back to the FOLLOW_UP label's own style, so a
# question that is both a follow-up trigger AND genuinely multi-part still
# gets the numbered plan instead of the single-reference follow-up wording.


def _part_chunk(document_id, source, page, text, part, rank):
    return {
        "chunk_id": rank,
        "document_id": document_id,
        "source": source,
        "page": page,
        "text": text,
        "score": 1.0 / rank,
        "rank": rank,
        "part": part,
        "part_query": text,
    }


def test_regression_second_admission_then_what_about_the_first_one():
    """Exact regression case from live testing: the prompt must contain a
    numbered plan naming both admissions, not the single-reference
    FOLLOW_UP style, and (since these chunks are part-tagged, as
    retrieve_for_question() tags them) evidence grouped under one PART
    heading per admission."""
    query = "What happened during the second admission? What about the first one?"
    chunks = [
        _part_chunk("d1", "a.pdf", 3, "Second admission: chest pain, discharged stable.", part=1, rank=1),
        _part_chunk("d1", "a.pdf", 1, "First admission: fever, treated with antibiotics.", part=2, rank=2),
    ]

    prompt = build_prompt(query, chunks, history=[{"role": "user", "content": "..."}], disable_benchmark_scope=True)

    assert "This question has 2 parts" in prompt
    assert "Part 1: What happened during the second admission?" in prompt
    assert "Part 2: What happened during the first admission?" in prompt
    assert "PART 1 (What happened during the second admission?):" in prompt
    assert "PART 2 (What happened during the first admission?):" in prompt
    assert "continues the conversation above" not in prompt  # not the plain FOLLOW_UP style
    assert prompt.count("ANSWER STYLE") == 1


def test_multi_part_follow_up_beats_the_follow_up_style_even_without_history():
    """The same decomposition applies with no history at all - a follow-up
    LABEL requires history, but genuine multi-part decomposition does not."""
    query = "What happened during the second admission? What about the first one?"
    prompt = _production(query)

    assert "This question has 2 parts" in prompt


def test_single_reference_follow_up_keeps_its_own_style_unchanged():
    """Regression guard: a follow-up that is NOT multi-part (only one thing
    asked, resolved via history) must still get the plain FOLLOW_UP style -
    this is test_follow_up_style_points_at_the_conversation's exact case,
    re-asserted here against the new decomposition-aware build_answer_style."""
    history = [
        {"role": "user", "content": "Summarize each selected document."},
        {"role": "assistant", "content": "Pediatric Report.pdf: ..."},
    ]
    prompt = _production("What about the medications in that report?", history=history)

    assert "continues the conversation above" in prompt
    assert "This question has" not in prompt


# --- build_part_context --------------------------------------------------------


def test_build_part_context_groups_evidence_under_its_part():
    parts = ["What happened during the second admission?", "What happened during the first admission?"]
    chunks = [
        _part_chunk("d1", "a.pdf", 3, "second admission text", part=1, rank=1),
        _part_chunk("d1", "a.pdf", 1, "first admission text", part=2, rank=2),
    ]

    context = build_part_context(chunks, parts)

    assert "PART 1 (What happened during the second admission?):" in context
    assert "PART 2 (What happened during the first admission?):" in context
    assert context.index("second admission text") < context.index("PART 2")
    assert "first admission text" in context


def test_build_part_context_flags_a_part_with_no_evidence():
    """One part having evidence and another not: the empty part gets an
    explicit marker, not silence - the same reasoning as
    build_multi_document_context()'s per-document placeholder."""
    parts = ["What happened during the second admission?", "What happened during the first admission?"]
    chunks = [
        _part_chunk("d1", "a.pdf", 3, "second admission text", part=1, rank=1),
    ]

    context = build_part_context(chunks, parts)

    assert "PART 1 (What happened during the second admission?):" in context
    assert "second admission text" in context
    assert "PART 2 (What happened during the first admission?):\n  EVIDENCE: none" in context


def test_build_part_context_placeholder_is_not_a_quotable_sentence():
    """Mirrors test_no_evidence_placeholder_is_not_a_quotable_sentence for
    the multi-document case - a marker the model could copy back verbatim
    as if it were an answer would defeat the point of having one."""
    parts = ["a", "b"]
    chunks = [_part_chunk("d1", "a.pdf", 1, "text", part=1, rank=1)]

    context = build_part_context(chunks, parts)

    assert "no evidence was retrieved" not in context.lower()
    assert "EVIDENCE: none" in context


def test_untagged_chunks_do_not_crash_build_part_context():
    """A caller that doesn't tag chunks with `part` (e.g. a hand-built test
    fixture, or a future caller that hasn't adopted retrieve_for_question)
    must not KeyError - every untagged chunk is treated as part 1."""
    parts = ["a", "b"]
    chunks = [
        {"chunk_id": 1, "document_id": "d1", "source": "a.pdf", "page": 1, "text": "t", "score": 0.1, "rank": 1},
    ]

    context = build_part_context(chunks, parts)

    assert "PART 1 (a):" in context
    assert "t" in context
    assert "PART 2 (b):\n  EVIDENCE: none" in context


def test_build_production_prompt_uses_flat_context_when_chunks_are_not_part_tagged():
    """A multi-part question whose chunks were retrieved by a caller that
    doesn't tag them (e.g. scripts/evaluate.py's legacy path, or any single
    retrieve_hybrid() call) must not produce empty/broken PART headings - it
    falls back to the original flat evidence list."""
    query = "What test was ordered, and what medications were prescribed?"
    prompt = build_production_prompt(query, CHUNKS)

    assert "PART 1 (" not in prompt
    assert "[1] Source: a.pdf, page 2" in prompt


def test_single_part_question_never_gets_part_headings():
    chunks = [_part_chunk("d1", "a.pdf", 1, "text", part=1, rank=1)]
    prompt = build_production_prompt("What was the chief complaint?", chunks)

    assert "PART 1 (" not in prompt
    assert "This question has" not in prompt


# --- structured evidence context ---------------------------------------------


def test_multi_document_context_groups_evidence_under_its_document():
    chunks = [
        _chunk("d1", "Pediatric Report.pdf", 1, "well child visit notes", 1),
        _chunk("d2", "OMC Report Sample - Cardio.pdf", 4, "aneurysm repair notes", 2),
    ]

    context = build_multi_document_context(chunks, INVENTORY_2)

    assert "DOCUMENT: Pediatric Report.pdf\n  PAGE: 1\n  EVIDENCE [1]:\n  well child visit notes" in context
    assert "DOCUMENT: OMC Report Sample - Cardio.pdf\n  PAGE: 4\n  EVIDENCE [2]:\n  aneurysm repair notes" in context


def test_one_document_heading_per_document_not_per_chunk():
    """Regression, measured: repeating the DOCUMENT heading on every chunk
    made llama3.2:3b treat each evidence entry as its own document, and
    answer "summarize each selected document separately" with six sections
    titled "Document 1: <same file>, Page 8", "Document 2: <same file>,
    Page 1"... for three selected documents. The heading groups; the
    indented entries belong to it."""
    chunks = [
        _chunk("d1", "Pediatric Report.pdf", 1, "first chunk", 1),
        _chunk("d1", "Pediatric Report.pdf", 2, "second chunk", 2),
    ]

    context = build_multi_document_context(chunks, INVENTORY_2[:1])

    assert context.count("DOCUMENT: Pediatric Report.pdf") == 1
    assert context.count("  PAGE: ") == 2
    assert "first chunk" in context and "second chunk" in context


def test_each_document_template_gives_no_escape_hatch_to_a_document_with_evidence():
    """Regression, measured: with three chunks from Pediatric Report.pdf in
    the prompt, the model still answered "No relevant evidence was retrieved
    for this document" under that heading - because the template offered
    that wording as an option for every document."""
    chunks = [
        _chunk("d1", "Pediatric Report.pdf", 1, "well child visit", 1),
        _chunk("d2", "OMC Report Sample - Cardio.pdf", 2, "aneurysm repair", 2),
    ]

    prompt = build_prompt("Summarize each document.", chunks, document_inventory=INVENTORY_2)

    assert "Pediatric Report.pdf: <1-3 sentences on this document's own evidence above>" in prompt
    assert "OMC Report Sample - Cardio.pdf: <1-3 sentences on this document's own evidence above>" in prompt
    assert "no relevant evidence was retrieved" not in prompt.split("ANSWER STYLE")[-1].lower()


def test_each_document_template_keeps_the_no_evidence_slot_where_it_is_true():
    """The opposite error - quietly dropping a document that genuinely has
    nothing - is the one this placeholder exists to prevent."""
    chunks = [_chunk("d1", "Pediatric Report.pdf", 1, "well child visit", 1)]

    prompt = build_prompt("Summarize each document.", chunks, document_inventory=INVENTORY_2)

    assert "Pediatric Report.pdf: <1-3 sentences on this document's own evidence above>" in prompt
    assert (
        "OMC Report Sample - Cardio.pdf: <one sentence saying no relevant evidence "
        "was retrieved for this document>"
    ) in prompt


def test_multi_document_rules_state_the_document_count_explicitly():
    """The model must not be able to infer the number of documents by
    counting evidence entries."""
    chunks = [
        _chunk("d1", "Pediatric Report.pdf", 1, "a", 1),
        _chunk("d1", "Pediatric Report.pdf", 2, "b", 2),
        _chunk("d1", "Pediatric Report.pdf", 3, "c", 3),
    ]
    prompt = build_prompt("Summarize each document.", chunks, document_inventory=INVENTORY_2)

    assert "2 documents are selected" in prompt
    assert "exactly 2 documents, however many evidence entries there are" in prompt


def test_each_document_style_forbids_per_evidence_sections():
    prompt = build_prompt("Summarize each document.", CHUNKS, document_inventory=INVENTORY_2)
    assert "never one section per evidence entry" in prompt


def test_production_instruction_forbids_substituting_a_related_fact():
    """Regression, measured: asked what accuracy the proposed system
    achieved - a figure the document never reports - the model answered
    "95.6% accuracy on the MIRAGE Benchmark", taking a number from a cited
    third-party study. Saying nothing is there must beat answering with
    something adjacent."""
    prompt = _production("What accuracy did the system achieve?")

    assert "say so plainly and stop" in prompt
    assert "substitute a similar-sounding fact that is about something else" in prompt
    assert "reported for a different system, study or patient is not an answer" in prompt


def test_numeric_style_refuses_to_invent_a_value():
    prompt = _production("What accuracy percentage did the system achieve?")
    assert "Do not produce a plausible-looking number" in prompt


def test_citation_numbers_are_the_chunks_own_ranks():
    chunks = [_chunk("d2", "OMC Report Sample - Cardio.pdf", 4, "text", 7)]
    context = build_multi_document_context(chunks, INVENTORY_2)
    assert "EVIDENCE [7]:" in context


def test_comparison_prompt_refuses_the_hedge_it_used_to_produce():
    chunks = [
        _chunk("d1", "Pediatric Report.pdf", 1, "well child visit", 1),
        _chunk("d2", "OMC Report Sample - Cardio.pdf", 4, "aneurysm repair", 2),
    ]
    prompt = build_prompt(
        "What do the Pediatric Report and the OMC Cardiology report have in common?",
        chunks,
        document_inventory=INVENTORY_2,
    )

    assert "never write that you cannot see them or that it is difficult to say" in prompt
    assert "Compare what the documents actually say, not what kind of" in prompt


def test_multi_document_prompt_keeps_the_selected_is_not_evidence_rule():
    prompt = build_prompt("Compare them.", CHUNKS, document_inventory=INVENTORY_2)
    assert "does NOT mean you have evidence about it" in prompt


# --- the legacy path is untouched ---------------------------------------------


def test_legacy_path_keeps_the_original_instruction():
    """scripts/evaluate.py (unmodified, and the only remaining caller that
    does not pass the production flag) must see exactly what it always saw."""
    legacy = build_prompt("what is this about?", CHUNKS)

    assert SYSTEM_INSTRUCTION in legacy
    assert PRODUCTION_SYSTEM_INSTRUCTION not in legacy
    assert "CONTEXT:" in legacy
    assert "ANSWER STYLE" not in legacy


def test_legacy_default_and_explicit_false_agree():
    assert build_prompt("q", CHUNKS) == build_prompt("q", CHUNKS, disable_benchmark_scope=False)


def test_production_and_legacy_prompts_differ():
    assert build_prompt("q", CHUNKS, disable_benchmark_scope=True) != build_prompt("q", CHUNKS)


def test_build_production_prompt_is_what_the_flag_selects():
    assert build_prompt("q", CHUNKS, disable_benchmark_scope=True) == build_production_prompt("q", CHUNKS)


def test_answer_style_sits_between_the_evidence_and_the_question():
    """Deliberate placement. With the formatting contract at the top of a
    ~17KB prompt, the last thing the model read before answering was
    evidence text, and it structured its answer around headings found
    inside that text instead of the sections it was asked for."""
    prompt = _production("Summarize this document.")

    evidence_idx = prompt.index("EVIDENCE:")
    style_idx = prompt.index("ANSWER STYLE")
    question_idx = prompt.index("USER QUESTION:")

    assert evidence_idx < style_idx < question_idx


def test_answer_style_placement_also_holds_for_multi_document_prompts():
    prompt = build_prompt("Summarize each document.", CHUNKS, document_inventory=INVENTORY_2)

    assert prompt.index("EVIDENCE:") < prompt.index("ANSWER STYLE") < prompt.index("USER QUESTION:")


def test_answer_style_still_precedes_the_question_with_history():
    history = [{"role": "user", "content": "prior turn"}]
    prompt = _production("Summarize this document.", history=history)

    assert prompt.index("RECENT CONVERSATION:") < prompt.index("ANSWER STYLE")
    assert prompt.index("ANSWER STYLE") < prompt.index("USER QUESTION:")


def test_production_prompt_ends_on_the_generation_cue():
    prompt = _production("What medication was prescribed?")
    assert prompt.rstrip().endswith("ANSWER:")


def test_history_still_sits_between_evidence_and_the_question():
    history = [{"role": "user", "content": "prior turn text"}]
    prompt = _production("current question", history=history)

    evidence_idx = prompt.index("EVIDENCE:")
    history_idx = prompt.index("RECENT CONVERSATION:")
    question_idx = prompt.index("USER QUESTION:")
    answer_idx = prompt.rindex("ANSWER:")

    assert evidence_idx < history_idx < question_idx < answer_idx
    assert "prior turn text" not in prompt[answer_idx:]
