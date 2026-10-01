"""Streamlit dashboard for the RAG for Healthcare Documents pipeline.

This file is UI glue only: it saves an uploaded PDF, hands it to the existing
ingestion pipeline, and calls the existing hybrid retriever / prompt builder /
Ollama client for querying. No extraction, chunking, embedding, FAISS, FTS5,
prompt, or generation logic is reimplemented here.
"""

import re
import shutil
import sys
import tempfile
import time
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from scripts import ingest as ingest_module
from src import config
from src.embedding.embedder import warm_up as warm_up_embedder
from src.exceptions import (
    DatabaseNotFoundError,
    EmptyQueryError,
    IndexNotFoundError,
    InsufficientRetrievalError,
    ModelUnavailableError,
    OllamaUnavailableError,
    RagError,
)
from src.generation.intent import (
    FACTUAL,
    _COMPONENT_LIST,
    classify_intent,
    decompose_question,
    is_enumeration_question,
    needs_document_balance,
    requests_multiple_components,
)
from src.generation.ollama_client import (
    generate,
    generate_stream,
    get_status as get_ollama_status,
    warm_up as warm_up_ollama,
)
from src.generation.prompt_builder import build_prompt, order_chunks_for_presentation
from src.retrieval.document_reference import resolve_document_reference
from src.retrieval.faiss_index import FaissIndex, cross_process_index_lock
from src.retrieval.hybrid_retriever import retrieve_for_question, retrieve_hybrid
from src.retrieval.index_consistency import check_consistency, repair
from src.retrieval.metadata_store import MetadataStore
from src.ui.icons import (
    ICON,
    assistant_avatar_svg,
    logo_svg,
    material,
    svg_data_uri,
    user_avatar_svg,
)

AUTHOR = "Hatim Maula"

TAGLINE = (
    "Ask questions about your healthcare PDFs. Answers are drawn only from the "
    "documents in scope, with every source passage shown - and everything runs "
    "on this machine."
)
SAFETY_NOTICE = (
    "This tool provides information grounded in uploaded documents and is not "
    "a substitute for professional medical advice."
)

# ---------------------------------------------------------------------------
# Pure helper functions (no Streamlit calls) - these are what tests exercise.
# ---------------------------------------------------------------------------


def ingest_uploaded_pdf(filename: str, file_bytes: bytes) -> dict:
    """Run uploaded PDF bytes through the existing ingestion pipeline.

    The bytes are written to a temporary file under the original filename (so
    `source` stays traceable to the user's document name) and removed again
    once ingestion finishes. Nothing is written into the project's data/
    directories by this function - ingest_pdf() owns the FAISS/SQLite writes.
    """
    safe_name = Path(filename).name
    # Characters Windows forbids in a filename (a macOS/Linux browser can
    # still send them) plus control characters.
    safe_name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", safe_name).strip(" .") or "upload.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"

    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    try:
        tmp_path = Path(tmp_dir) / safe_name
        tmp_path.write_bytes(file_bytes)
        try:
            return ingest_module.ingest_pdf(str(tmp_path))
        except RagError as e:
            # Ingestion errors name the file by its temporary path, which is
            # an internal detail the user never chose and cannot act on.
            # Re-raise the same error type with the name they uploaded.
            # PyMuPDF's own messages quote the path in escaped form
            # (backslashes doubled), so both spellings are replaced.
            message = str(e)
            for spelling in (str(tmp_path).replace("\\", "\\\\"), str(tmp_path)):
                message = message.replace(spelling, safe_name)
            raise type(e)(message) from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def ingest_uploaded_pdfs(files: list[tuple[str, bytes]]) -> list[dict]:
    """Ingest multiple uploaded PDFs, one at a time, through the same
    single-file pipeline above. One file failing does not abort the batch -
    each file's outcome (indexed / skipped / failed) is recorded
    independently, so the caller can show a per-document summary.

    A duplicate WITHIN this same batch (identical bytes uploaded twice) is
    handled correctly for free: ingest_pdf()'s content-hash dedup check runs
    fresh for each file in order, so the first copy indexes and the second
    is detected as already-present and marked "skipped", exactly like a
    duplicate uploaded in a separate session.

    Returns one dict per file, in input order:
    {"filename": str, "status": "indexed"|"skipped"|"failed",
     "document_id": str|None, "chunk_count": int|None, "error": str|None}
    """
    results = []
    for filename, file_bytes in files:
        try:
            result = ingest_uploaded_pdf(filename, file_bytes)
        except Exception as e:
            # Broad on purpose: the whole point of batch upload is that one
            # bad/unexpected file (corrupt PDF, surprise error) must not
            # abort the rest of the batch or crash the page.
            results.append(
                {
                    "filename": filename,
                    "status": "failed",
                    "document_id": None,
                    "chunk_count": None,
                    "degraded_pages": [],
                    "repaired_pages": [],
                    "image_only_pages": [],
                    "page_count": None,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
        else:
            results.append(
                {
                    "filename": filename,
                    "status": "skipped" if result["skipped"] else "indexed",
                    "document_id": result["document_id"],
                    "chunk_count": result["chunk_count"],
                    "degraded_pages": result.get("degraded_pages", []),
                    "repaired_pages": result.get("repaired_pages", []),
                    "image_only_pages": result.get("image_only_pages", []),
                    "page_count": result.get("page_count"),
                    "error": None,
                    "error_type": None,
                }
            )
    return results


def resolve_active_document_id(documents: list[dict], requested_id: str | None) -> str | None:
    """Pick which document_id should be active: the requested one if it's
    still indexed, otherwise the most recently indexed document, or None if
    nothing is indexed at all.
    """
    if not documents:
        return None
    ids = [d["document_id"] for d in documents]
    if requested_id in ids:
        return requested_id
    return ids[-1]


def list_indexed_documents() -> list[dict]:
    """Read-only summary of what's currently indexed, for the sidebar status panel."""
    store = MetadataStore(config.DB_PATH)
    try:
        return store.list_documents()
    except DatabaseNotFoundError:
        return []


def delete_document(document_id: str) -> dict:
    """Permanently remove a document: its SQLite/FTS rows and its FAISS vectors.

    SQLite rows are deleted (and committed) first. The FAISS index is only
    mutated and re-saved after that succeeds, so a crash/failure during the
    FAISS step can never leave metadata pointing at chunks that don't exist.
    In that narrow window it can leave harmless orphaned FAISS vectors, but
    those are never served: hybrid_retriever already skips any chunk_id that
    has no matching SQLite row (see MetadataStore.get_chunks usage there).

    Held under cross_process_index_lock() so a concurrent upload's
    load..save - in this process or another (e.g. the ingest CLI) - cannot
    interleave with this one and have either overwrite the other's index.

    Returns {"removed_chunks": int}.
    """
    store = MetadataStore(config.DB_PATH)
    with cross_process_index_lock(config.FAISS_INDEX_PATH):
        chunk_ids = store.delete_document(document_id)

        if chunk_ids:
            faiss_index = FaissIndex(config.EMBEDDING_DIM, config.FAISS_INDEX_PATH)
            faiss_index.load()
            faiss_index.remove_ids(chunk_ids)
            faiss_index.save()

    return {"removed_chunks": len(chunk_ids)}


def completion_status(total_ms: float, truncated: bool) -> tuple[str, str]:
    """("caption" | "warning", text) shown under a finished answer.

    A truncated answer (generation hit num_predict) ends mid-sentence; saying
    "Complete" under it would present a partial answer as a whole one.
    """
    seconds = f"{total_ms / 1000:.1f}s"
    if truncated:
        return (
            "warning",
            f"This answer reached the length limit and was cut off ({seconds}). "
            "It may be incomplete - a narrower question is more likely to get a complete answer.",
        )
    return "caption", f"Complete · {seconds}"


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    """'1 chunk', '2 chunks', '0 chunks' - the count is always included."""
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def document_meta(doc: dict) -> str:
    """'12 chunks · 4 pages' for a library card; pages omitted when unknown."""
    bits = [pluralize(doc.get("chunk_count") or 0, "chunk")]
    if doc.get("page_count"):
        bits.append(pluralize(doc["page_count"], "page"))
    return " · ".join(bits)


def group_sources(sources: list[dict]) -> list[dict]:
    """Collapse repeated (source, page) hits from multiple chunks into one
    entry per document with a sorted, de-duplicated page list, e.g.
    [{"source": "Pediatric Report.pdf", "pages": [1, 2, 3]}], preserving the
    order documents first appear in.
    """
    pages_by_source: dict[str, set[int]] = {}
    order: list[str] = []
    for s in sources:
        name = s["source"]
        if name not in pages_by_source:
            pages_by_source[name] = set()
            order.append(name)
        pages_by_source[name].add(s["page"])
    return [{"source": name, "pages": sorted(pages_by_source[name])} for name in order]


def build_document_inventory(documents: list[dict], selected_document_ids) -> list[dict]:
    """The {document_id, source, page_count} inventory for the multi-document
    prompt - built ONLY from documents whose id is in selected_document_ids,
    in `documents`' own order, so an unselected (or since-deleted) document
    can never reach the generation context. Returns [] if nothing selected.
    """
    return [
        {"document_id": d["document_id"], "source": d["source"], "page_count": d.get("page_count")}
        for d in documents
        if d["document_id"] in selected_document_ids
    ]


def resolve_retrieval_top_k(
    analysis_scope: str, selected_document_ids, balanced: bool = False
) -> int:
    """How many chunks retrieve_hybrid() should keep from its fused ranking
    for this turn.

    Only a question that must cover several documents gets a widened budget,
    and only up to config.MAX_CONTEXT_CHUNKS. Everything else - including a
    narrow question asked while several documents are selected - gets exactly
    config.TOP_K, the same budget it would get about a single document.

    This used to widen to TOP_K * (number of selected documents) for EVERY
    question asked in "Selected documents" scope. That was a workaround from
    before retrieval could balance its allocation, and it caused a real
    production failure: "What medications was the patient discharged on, and
    what condition was found during the second hospital admission?" asked
    over three selected documents produced a 15-chunk, 20KB prompt, and the
    model answered that the evidence did not mention the discharge
    medications - while the sentence naming them sat in the HIGHEST-ranked
    chunk of that same prompt. The identical question over the same index
    answered correctly at 5, 7, 9 and 12 chunks.

    The widening also had it backwards. An each-document or comparison
    question, which genuinely needs breadth, was getting the same treatment
    as a narrow factual question, which needs focus - and after balanced
    allocation arrived, the narrow question was the one left with the larger
    budget. Breadth now comes from _allocate_by_document() choosing WHICH
    chunks survive, not from asking for more of them.

    FAISS, FTS5, RRF weights and candidate_k remain untouched: this only
    decides how many results are kept from the end of the existing fused
    ranking.
    """
    if analysis_scope == "selected" and selected_document_ids and balanced:
        return min(
            config.BALANCED_CHUNKS_PER_DOCUMENT * len(selected_document_ids),
            config.MAX_CONTEXT_CHUNKS,
        )
    return config.TOP_K


# Widened retrieval depth for a single-document FACTUAL question whose
# answer must cover several distinct, EXPLICITLY NAMED components (intent.
# requests_multiple_components() - the same signal build_answer_style()
# already uses to pick the evidence-complete ANSWER STYLE, see
# prompt_builder.py). Controlled 45-question research-corpus evaluation
# (baseline K=5 vs this rule) found zero regressions on any metric, since
# retrieve_hybrid()'s candidate pool and RRF fusion are independent of
# top_k - widening only appends more already-ranked candidates, it never
# changes or reorders the first 5.
#
# requests_multiple_components() itself is True for either of two shapes -
# an explicit "X, Y, and Z" list, or a bare "each/every <noun>" - because
# both need the same ANSWER STYLE. They do NOT get the same retrieval
# treatment here: only the explicit-list shape (intent._COMPONENT_LIST) is
# widened below. The each/every shape is deliberately left at config.TOP_K
# for now - evidence for widening it is thinner (a single real document's
# worth of measurement, see Task 8-11) and is being held back pending
# further validation, not implemented alongside this narrower, better-
# supported change.
WIDENED_FACTUAL_TOP_K = 8


def resolve_num_predict(
    query: str,
    history: list[dict] | None,
    document_inventory: list[dict] | None,
) -> int | None:
    """The Ollama `num_predict` override for this turn, or None for the
    single-document, single-part default (config.OLLAMA_NUM_PREDICT).

    config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT was originally sized only for
    multi-document answers, but a multi-part question over a SINGLE
    document needs the same larger budget just as much: "What happened
    during the second admission? What about the first one?" was measured
    hitting the default 256-token cap mid sentence, part way through its
    second section - the model never got to finish answering the part
    retrieve_for_question() exists to guarantee evidence for in the first
    place. Reused rather than adding a second, identically-sized constant.

    Uses the same intent.decompose_question() check retrieve_and_build_prompt()
    itself uses for the multi-part retrieval decision, so this can never
    disagree with whether the prompt actually ended up structured as
    multiple parts.

    Also widened for a single-document FACTUAL enumeration question
    (intent.is_enumeration_question() - a named list, an implied "each/
    every" enumeration, or a most/least-frequent ranking; the same signal
    build_answer_style() uses for FACTUAL_MULTI_COMPONENT_STYLE, see
    prompt_builder.py). Measured need: "What were the main findings
    regarding examinations, laboratory tests, and treatments?" is not
    multi-part (decompose_question finds no "and <interrogative>" second
    question) and has no document_inventory, so it kept the smaller
    default budget even once retrieval supplied the right evidence - the
    answer was cut off mid-sentence before reaching its third named
    component. Gated on FACTUAL specifically (not every factual question)
    so an ordinary single-fact question keeps the smaller default budget.
    """
    multi_part = len(decompose_question(query, history=history)) > 1
    enumeration = (
        classify_intent(query, history=history) == FACTUAL
        and is_enumeration_question(query)
    )
    if document_inventory or multi_part or enumeration:
        return config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT
    return None


def retrieve_and_build_prompt(
    query: str,
    top_k: int = config.TOP_K,
    document_id: str | None = None,
    timing: dict | None = None,
    document_inventory: list[dict] | None = None,
    history: list[dict] | None = None,
) -> tuple[list[dict], str]:
    """Hybrid retrieval -> prompt construction (no generation).

    Split out of answer_question() so the Streamlit UI can run this part
    (fast, ~tens of ms) under a brief spinner and then stream the slow part
    (generation) separately, while CLI callers that don't need streaming
    (scripts/query.py, chat.py) keep using answer_question() unchanged.

    If `timing` is given, it's populated in place with retrieve_hybrid's own
    `embedding_ms`/`dense_ms`/`keyword_ms`/`fusion_ms` breakdown plus this
    function's own `prompt_build_ms`.

    `document_inventory` (optional) marks this as a multi-document turn, and
    is forwarded to build_prompt() - see its docstring. It is also the
    precondition for the two multi-document behaviours below, document
    reference scoping and balanced allocation; scoring, the FAISS/FTS5
    searches and the RRF fusion are unaffected by either.

    If the question names one or more of the selected documents, retrieval
    and the prompt are narrowed to exactly those. The narrowing is always a
    subset of `document_inventory`, so selected-document isolation holds,
    and active-document scope never reaches it at all.

    For an each-document or comparison question over several selected
    documents, retrieval is asked to balance its allocation across them and
    `top_k` is replaced by config.BALANCED_CHUNKS_PER_DOCUMENT per selected
    document. Every other question - including any single-document question -
    keeps the caller's `top_k` and the unmodified ranking.

    `history` (optional) is prior conversation turns, forwarded to
    build_prompt() for GENERATION context, and also to retrieve_for_question()
    below for the single-document path. Conversation TEXT itself is never
    embedded or searched either way - retrieve_for_question() only ever uses
    history to resolve an elliptical reference ("what about the first?")
    into a standalone question BEFORE retrieval runs (see
    intent.decompose_question), so what reaches FAISS/FTS5 is always a
    clean information need, never a blended transcript.

    A single-document question that asks about more than one part - joined
    by "and", or phrased as a follow-up tag ("X? What about Y?", "What about
    the first?") - is retrieved one part at a time and merged, instead of as
    one embedding of the whole compound question; see
    retrieve_for_question()'s docstring. This is deliberately scoped to the
    single-document path (document_inventory falsy here) only - multi-document
    balanced allocation above already gives a comparison/each-document
    question its own breadth guarantee along the document axis, and layering
    per-part retrieval on top of that was judged more complexity than the
    reported failure (which was always single-document) warranted.

    Always disables prompt_builder's benchmark-specific target-scope
    detection (see build_prompt's docstring): this is the app's real
    production answer path, not the old project benchmark.
    """
    # timing is only passed through when requested, so existing callers/mocks
    # of retrieve_hybrid() with a narrower signature (no `timing` kwarg) keep
    # working unchanged - see tests/test_app_helpers.py's fake_retrieve_hybrid.
    retrieve_kwargs = {"top_k": top_k, "document_id": document_id}
    if timing is not None:
        retrieve_kwargs["timing"] = timing

    # This function owns the whole multi-document budget decision, and
    # replaces the caller's `top_k` for it: the budget is a property of the
    # question and the retrieval strategy, not of the call site, and
    # deciding it in one place is what keeps the Streamlit app and the
    # product QA harness identical without either having to mirror the
    # other. Single-document retrieval never reaches any of this -
    # document_inventory is None in active-document scope.
    if document_inventory:
        # A question that names a document ("the OMC report", "the third
        # document") is scoped to it before retrieval runs. Filenames appear
        # in no chunk text, so lexical matching cannot do this - see
        # src/retrieval/document_reference.py. The resolver only ever
        # returns entries from this inventory, so scoping can never widen
        # the selection or resurrect a deleted document, and an ambiguous
        # reference resolves to nothing and leaves the selection alone.
        reference = resolve_document_reference(query, document_inventory, history=history)
        if reference:
            document_inventory = list(reference.documents)
            retrieve_kwargs["document_id"] = reference.document_ids
            if len(document_inventory) == 1:
                # Narrowed to exactly one document, so this is no longer a
                # multi-document turn: drop to the single-document prompt,
                # the same one active-document scope uses.
                #
                # Keeping the multi-document prompt here actively misleads
                # the model. Asked "what about the third document?" with the
                # evidence correctly narrowed to that document, it read
                # rules headed "1 documents are selected" alongside a rule
                # about documents whose evidence was not retrieved, and
                # answered that the third document was a different file
                # whose evidence block was empty. The single-document prompt
                # has no such machinery and answers it correctly.
                document_inventory = None
                retrieve_kwargs["top_k"] = config.TOP_K

    if document_inventory:
        balanced = needs_document_balance(query, history=history) and len(document_inventory) > 1
        if balanced:
            # Same conditional-kwarg reason as `timing` above: narrow test
            # doubles of retrieve_hybrid must keep working.
            retrieve_kwargs["balance_by_document"] = True
        retrieve_kwargs["top_k"] = resolve_retrieval_top_k(
            "selected", document_inventory, balanced=balanced
        )

    try:
        if document_inventory:
            # Multi-document retrieval (balanced allocation, document-scoped
            # candidate filtering) is untouched by this - see
            # retrieve_for_question()'s docstring for why per-part retrieval
            # is deliberately scoped to the single-document path only.
            chunks = retrieve_hybrid(query, **retrieve_kwargs)
        else:
            single_document_top_k = retrieve_kwargs["top_k"]
            # Only step in when the caller left top_k at its ordinary
            # default - a caller that already asked for a specific top_k
            # (a diagnostic script, e.g.) is left alone, and this can never
            # apply to a genuinely MULTI_PART-classified question, since
            # that label wins classify_intent()'s priority over FACTUAL and
            # this rule is gated on FACTUAL specifically. requests_multiple_
            # components() alone is not enough to decide the WIDTH - see
            # WIDENED_FACTUAL_TOP_K's comment - so the explicit-list shape
            # is also checked directly, and the each/every shape is left
            # untouched even though it also satisfies requests_multiple_
            # components().
            if (
                single_document_top_k == config.TOP_K
                and classify_intent(query, history=history) == FACTUAL
                and requests_multiple_components(query)
                and _COMPONENT_LIST.search(query.lower())
            ):
                single_document_top_k = WIDENED_FACTUAL_TOP_K
            chunks = retrieve_for_question(
                query,
                history=history,
                top_k=single_document_top_k,
                document_id=retrieve_kwargs.get("document_id"),
                timing=timing,
                retrieve_fn=retrieve_hybrid,
            )
    except ValueError as e:
        raise EmptyQueryError(str(e)) from e

    if not chunks:
        raise InsufficientRetrievalError("No relevant chunks found for this query")

    # Presentation-only reorder (see order_chunks_for_presentation's
    # docstring for the full rationale): RRF has already done the actual
    # retrieval selection above, unmodified - this only decides where,
    # among the chunks it selected, each one is placed for the model to
    # read. Scoped to the single-document, single-part case on purpose -
    # document_inventory means the each-document/comparison balancing path
    # below owns ordering instead, and a `part`-tagged chunk (multi-part
    # retrieval) has its own PART-grouped structure in build_part_context()
    # that this reorder isn't designed for.
    if not document_inventory and not any("part" in c for c in chunks):
        chunks = order_chunks_for_presentation(query, chunks)

    t0 = time.perf_counter()
    prompt = build_prompt(
        query,
        chunks,
        document_inventory=document_inventory,
        history=history,
        disable_benchmark_scope=True,
    )
    if timing is not None:
        timing["prompt_build_ms"] = (time.perf_counter() - t0) * 1000

    return chunks, prompt


def resolve_scope_document_ids(
    analysis_scope: str,
    active_document_id: str | None,
    selected_document_ids,
) -> str | list[str] | None:
    """Translate the sidebar's "Analysis scope" selection into the
    document_id argument for retrieve_and_build_prompt()/retrieve_hybrid():

    - "active": the single active document (str, or None if none is active)
      - unchanged single-document isolation behavior, regardless of
        whatever is left over in selected_document_ids from a previous
        "selected" session, so switching back to Active document always
        restores isolation.
    - "selected": a sorted list of the chosen document_ids - empty if
      nothing is selected. Callers must treat an empty list as "cannot
      query yet", not as "no filter": retrieve_hybrid() returns no results
      for an empty list rather than silently searching everything.
    """
    if analysis_scope == "selected":
        return sorted(selected_document_ids)
    return active_document_id


def answer_question(query: str, top_k: int = config.TOP_K, document_id: str | None = None) -> dict:
    """Hybrid retrieval -> prompt construction -> Ollama generation.

    Mirrors scripts/query.py's answer_query(), extended to also return the
    full retrieved chunks (with text) so the UI can show the evidence, and to
    accept an optional document_id so retrieval can be scoped to the active
    document instead of the whole corpus.

    Uses resolve_num_predict() for the generation budget - the same function
    the Streamlit chat turn uses - so a multi-part or enumeration question
    gets the larger budget here too. Before this, this entry point never
    passed num_predict at all: a single-document FACTUAL enumeration
    question could have retrieval supply every needed chunk and still have
    its answer cut off mid-sentence on the default budget.
    """
    chunks, prompt = retrieve_and_build_prompt(query, top_k=top_k, document_id=document_id)
    num_predict = resolve_num_predict(query, history=None, document_inventory=None)
    answer = generate(prompt, num_predict=num_predict)
    sources = [{"source": c["source"], "page": c["page"], "score": c["score"]} for c in chunks]
    return {"answer": answer, "sources": sources, "chunks": chunks}


@dataclass(frozen=True)
class UserFacingError:
    """What happened, why it might have happened, and what to do about it.
    `detail` is the raw exception text, shown only inside a collapsed
    "Technical details" panel, never in the message itself."""

    title: str
    cause: str
    action: str
    detail: str = ""


def describe_error(exc: Exception) -> UserFacingError:
    """Map an answer-path failure to a message a user can act on."""
    detail = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, ModelUnavailableError):
        return UserFacingError(
            "The language model isn't installed",
            f"Ollama is running, but the configured model ({config.OLLAMA_MODEL}) hasn't been downloaded.",
            f"Run `ollama pull {config.OLLAMA_MODEL}` in a terminal, then select Retry.",
            detail,
        )
    if isinstance(exc, OllamaUnavailableError):
        text = str(exc).lower()
        if "interrupted" in text or "unparseable" in text:
            return UserFacingError(
                "The answer was interrupted",
                "The connection to Ollama dropped while the answer was being written, "
                "so the partial answer was discarded rather than shown as if it were complete.",
                "Select Retry. If it happens again, check that Ollama is still running.",
                detail,
            )
        if "empty response" in text:
            return UserFacingError(
                "The model returned an empty answer",
                "Ollama responded, but produced no text for this question.",
                "Select Retry, or rephrase the question.",
                detail,
            )
        return UserFacingError(
            "Ollama isn't available",
            "The local Ollama service didn't respond, so no answer could be generated.",
            "Start Ollama (open the Ollama app or run `ollama serve`) and make sure the "
            "configured model is installed, then select Retry.",
            detail,
        )
    if isinstance(exc, EmptyQueryError):
        return UserFacingError("The question is empty", str(exc), "Type a question and send it again.", detail)
    if isinstance(exc, (DatabaseNotFoundError, IndexNotFoundError)):
        return UserFacingError(
            "The document index isn't available",
            "The search index for your library could not be found on disk.",
            "Upload the document again to rebuild its index.",
            detail,
        )
    if isinstance(exc, RagError):
        return UserFacingError(
            "Couldn't answer that question", str(exc), "Select Retry, or rephrase the question.", detail
        )
    return UserFacingError(
        "Something unexpected went wrong",
        "An internal error stopped this answer before it finished.",
        "Select Retry. If it keeps happening, the terminal running the app has the details.",
        detail,
    )


def describe_upload_failure(filename: str, error_type: str | None, message: str) -> UserFacingError:
    """Map an ingestion failure (see ingest_uploaded_pdfs) to plain language."""
    text = (message or "").lower()
    if error_type == "InvalidPDFError" and "encrypted" in text:
        return UserFacingError(
            f"{filename} is password-protected",
            "Encrypted PDFs can't be read.",
            "Save an unprotected copy and upload that instead.",
            message,
        )
    if error_type == "InvalidPDFError":
        return UserFacingError(
            f"{filename} couldn't be opened as a PDF",
            "The file may be damaged, incomplete, or not actually a PDF.",
            "Check that it opens in a PDF viewer, then upload it again.",
            message,
        )
    if error_type == "EmptyDocumentError" and "damaged or truncated" in text:
        return UserFacingError(
            f"{filename} appears to be damaged",
            "No text could be recovered from it - it may have been cut off while downloading.",
            "Download or export it again, then upload the new copy.",
            message,
        )
    if error_type == "EmptyDocumentError":
        return UserFacingError(
            f"{filename} has no readable text",
            "It looks like a scanned or image-only PDF, and this app doesn't run OCR.",
            "Use a text-based PDF, such as one exported from a word processor.",
            message,
        )
    return UserFacingError(
        f"{filename} couldn't be added",
        "An unexpected error occurred while it was being indexed. Nothing was added to the library.",
        "Select Try again. If it keeps failing, the terminal running the app has the details.",
        message,
    )


# ---------------------------------------------------------------------------
# Chat turn state. A question is recorded in the history the moment it is
# submitted and the answer is produced on the following run, so the input can
# be shown disabled while it is generated, and a run that is interrupted
# (e.g. by another widget) resumes the same question instead of losing it.
# ---------------------------------------------------------------------------

MAX_QUERY_CHARS = 2000
MAX_GENERATION_ATTEMPTS = 3


def queue_question(state: MutableMapping, query: str, scope: dict) -> None:
    """Record the user's question and mark it as awaiting an answer."""
    history = state.setdefault("chat_history", [])
    history.append({"role": "user", "content": query})
    state["pending_turn"] = {"query": query, "scope": scope, "history_len": len(history) - 1, "attempts": 0}


def finish_pending_turn(state: MutableMapping, assistant_turn: dict) -> None:
    state.setdefault("chat_history", []).append(assistant_turn)
    state.pop("pending_turn", None)


def error_turn(err: UserFacingError, pending: dict) -> dict:
    """An assistant turn recording a failure. Kept out of the generation
    context (see conversation_context) - an error is not part of the
    conversation the model should continue."""
    return {
        "role": "assistant",
        "status": "error",
        "content": err.title,
        "error": asdict(err),
        "retry": {k: pending[k] for k in ("query", "scope", "history_len")},
    }


def can_retry(state: MutableMapping) -> bool:
    history = state.get("chat_history") or []
    return bool(history) and history[-1].get("status") == "error" and "pending_turn" not in state


def retry_last_turn(state: MutableMapping) -> None:
    """Drop the trailing error turn and queue its question again, against
    the same scope it was originally asked in."""
    if not can_retry(state):
        return
    failed = state["chat_history"].pop()
    state["pending_turn"] = {**failed["retry"], "attempts": 0}


def conversation_context(history: list[dict]) -> list[dict]:
    """Prior turns that are part of the conversation (errors excluded)."""
    return [t for t in history if t.get("status") != "error"]


# ---------------------------------------------------------------------------
# Theming - session-scoped CSS variables, no config.toml / restart required.
# ---------------------------------------------------------------------------

THEMES = {
    "dark": {
        "bg": "#0e1117",
        "bg-secondary": "#161b22",
        "card-bg": "#1c2128",
        "border": "#30363d",
        "text": "#e6edf3",
        "text-secondary": "#9198a1",
        "accent": "#4f8cff",
        "accent-soft": "rgba(79, 140, 255, 0.15)",
        "accent-text": "#8fb8ff",
        "on-accent": "#ffffff",
        "user-bg": "#1f2937",
        "assistant-bg": "#161b22",
        "danger": "#f85149",
        "danger-soft": "rgba(248, 81, 73, 0.12)",
        "warning": "#d29922",
        "warning-soft": "rgba(210, 153, 34, 0.12)",
        "success": "#3fb950",
        "passage-bg": "#0d1117",
    },
    "light": {
        "bg": "#ffffff",
        "bg-secondary": "#f6f8fa",
        "card-bg": "#ffffff",
        "border": "#d0d7de",
        "text": "#1f2328",
        "text-secondary": "#59636e",
        "accent": "#0969da",
        "accent-soft": "rgba(9, 105, 218, 0.10)",
        "accent-text": "#0550ae",
        "on-accent": "#ffffff",
        "user-bg": "#f0f4f9",
        "assistant-bg": "#ffffff",
        "danger": "#cf222e",
        "danger-soft": "rgba(207, 34, 46, 0.08)",
        "warning": "#9a6700",
        "warning-soft": "rgba(154, 103, 0, 0.08)",
        "success": "#1a7f37",
        "passage-bg": "#f6f8fa",
    },
}

BASE_CSS = """
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stSidebar"] { background: var(--bg-secondary); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text); }

/* Streamlit's own native widgets (buttons, uploader, header, chat input,
   expander) paint fixed colors baked from its own auto light/dark detection,
   not CSS variables - override just the ones that would otherwise clash
   with our chosen theme and break contrast. Primary-styled buttons (e.g.
   "Confirm delete") keep Streamlit's own accent styling untouched. */
[data-testid="stHeader"],
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {
    background: var(--bg) !important;
}
[data-testid^="stBaseButton-secondary"] {
    background: var(--card-bg) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--card-bg) !important;
    border-color: var(--border) !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }
[data-testid="stFileChip"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stExpander"] {
    background: var(--card-bg) !important;
    border-color: var(--border) !important;
}
[data-testid="stExpander"] summary:hover { background: var(--accent-soft) !important; }

/* Radio, checkbox and toggle take their colors from Streamlit's base theme,
   which follows the OS, not this app's theme switch - an unselected radio
   rendered as a solid dark disc in light mode. Structure (Streamlit 1.6x):
   label > span > input, then the drawn control. */
[data-testid="stRadioOption"] > div > div > div:first-child {
    background: transparent !important;
    box-shadow: inset 0 0 0 2px var(--text-secondary) !important;
}
[data-testid="stRadioOption"] > div > div > div:first-child > div { background: transparent !important; }
[data-testid="stRadioOption"]:has(input:checked) > div > div > div:first-child {
    box-shadow: inset 0 0 0 2px var(--accent) !important;
}
[data-testid="stRadioOption"]:has(input:checked) > div > div > div:first-child > div {
    background: var(--accent) !important;
}
[data-testid="stCheckbox"] label:has(input:not([role="switch"])) > span + div {
    background: var(--card-bg) !important;
    border: 1.5px solid var(--text-secondary) !important;
}
[data-testid="stCheckbox"] label:has(input:not([role="switch"]):checked) > span + div {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
}
[data-testid="stCheckbox"] label:has(input[role="switch"]) > span + div { background: var(--border) !important; }
[data-testid="stCheckbox"] label:has(input[role="switch"]:checked) > span + div { background: var(--accent) !important; }
[data-testid="stCheckbox"] label:has(input[role="switch"]) > span + div > div { background: #ffffff !important; }
[data-testid="stCheckbox"] label:has(input:focus-visible) > span + div,
[data-testid="stRadioOption"]:has(input:focus-visible) > div > div > div:first-child {
    outline: 2px solid var(--accent); outline-offset: 2px;
}

/* Buttons: secondary (above) for everyday actions, primary in the accent
   color, and the one destructive confirmation in the danger color. */
[data-testid^="stBaseButton-primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: var(--on-accent) !important;
}
[class*="st-key-confirm_"] [data-testid^="stBaseButton-primary"] {
    background: var(--danger) !important;
    border-color: var(--danger) !important;
}
[data-testid^="stBaseButton"]:disabled { opacity: 0.55; cursor: not-allowed; }
[data-testid^="stBaseButton"] p { font-weight: 500; }
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div:first-child {
    background: var(--card-bg) !important;
    border-color: var(--border) !important;
}
/* The override above also replaced Streamlit's focus highlight, leaving the
   main input with no visible keyboard focus. */
[data-testid="stChatInput"]:focus-within > div:first-child {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}
[data-testid="stChatInputTextArea"] { color: var(--text) !important; background: transparent !important; }
[data-testid="stChatInputTextArea"]::placeholder { color: var(--text-secondary) !important; opacity: 1 !important; }

/* Page header and readable main column. */
[data-testid="stMainBlockContainer"],
[data-testid="stBottomBlockContainer"] { max-width: 960px; }
[data-testid="stMainBlockContainer"] { padding-top: 2.4rem; }
h1 { font-size: 1.6rem !important; font-weight: 700 !important; letter-spacing: -0.01em; padding-bottom: 0.2rem !important; }
.app-subtitle { color: var(--text-secondary); font-size: 0.95rem; line-height: 1.5; margin: 0 0 1.2rem 0; max-width: 720px; }

/* Icon glyphs in custom HTML: the same Material Symbols Rounded font
   Streamlit uses for its own buttons and expanders (see src/ui/icons.py). */
.ms {
    font-family: "Material Symbols Rounded"; font-weight: normal; font-style: normal;
    font-size: 1.15em; line-height: 1; letter-spacing: normal; text-transform: none;
    display: inline-block; white-space: nowrap; direction: ltr; flex: none;
    font-feature-settings: "liga"; -webkit-font-smoothing: antialiased;
    font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 20;
    user-select: none;
}

/* Title mark: the project logo before the page title. */
h1::before {
    content: ""; display: inline-block; flex: none;
    width: 1.55em; height: 1.55em; margin-right: 0.55em; vertical-align: -0.38em;
    background: var(--logo) center / contain no-repeat;
}

/* Sidebar sections: one small, quiet label style instead of large headings. */
.section-label {
    display: flex; align-items: center; gap: 0.4rem;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--text-secondary) !important; margin: 0.2rem 0 0.4rem 0;
}
.section-label .ms { font-size: 1.05rem; }
.section-label .count { font-weight: 600; }

.brand-footer {
    display: flex; align-items: center; gap: 0.55rem;
    margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid var(--border);
}
.brand-footer img { flex: none; border-radius: 5px; }
.brand-footer-title { font-size: 0.78rem; font-weight: 600; }
.brand-footer-author { font-size: 0.72rem; color: var(--text-secondary) !important; }
.hint { color: var(--text-secondary) !important; font-size: 0.78rem; line-height: 1.45; margin-top: 0.3rem; }

.doc-card {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.55rem 0.75rem;
    margin-top: 0.6rem;
    margin-bottom: 0.3rem;
    background: var(--card-bg);
}
.doc-card.in-scope { border-color: var(--accent); background: var(--accent-soft); box-shadow: inset 3px 0 0 var(--accent); }
/* Long filenames truncate to one line with the full name on hover, so a
   30-character report name cannot push the sidebar controls out of shape. */
.doc-name { display: flex; align-items: center; gap: 0.4rem; min-width: 0; font-weight: 600; font-size: 0.92rem; }
.doc-name .doc-icon { color: var(--text-secondary) !important; font-size: 1.1rem; }
.doc-card.in-scope .doc-icon { color: var(--accent-text) !important; }
.doc-name-text { min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.doc-meta {
    color: var(--text-secondary); font-size: 0.78rem; margin-top: 0.15rem;
    display: flex; align-items: center; gap: 0.4rem;
}
.doc-badge {
    color: var(--accent-text) !important; border: 1px solid var(--accent);
    border-radius: 999px; padding: 0 0.45rem;
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
}
.delete-confirm {
    border: 1px solid var(--danger); background: var(--danger-soft);
    border-radius: 8px; padding: 0.55rem 0.7rem; margin: 0.2rem 0 0.4rem 0;
    font-size: 0.85rem; line-height: 1.45;
}
.delete-confirm strong { overflow-wrap: anywhere; }

/* Scope banner - what this turn can actually see. */
.scope-banner {
    border: 1px solid var(--border); border-left: 3px solid var(--accent);
    border-radius: 8px; background: var(--bg-secondary);
    padding: 0.6rem 0.8rem; margin-bottom: 1rem;
}
.scope-headline {
    display: flex; align-items: center; gap: 0.4rem;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.03em;
    text-transform: uppercase; color: var(--text-secondary);
}
.scope-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.4rem; }
.scope-chip {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.1rem 0.45rem;
    font-size: 0.82rem; color: var(--text);
    max-width: 100%; overflow-wrap: anywhere;
}

.empty-state {
    border: 1px dashed var(--border); border-radius: 12px;
    padding: 1.4rem 1.6rem; background: var(--bg-secondary);
    max-width: 680px;
}
.empty-state h3 { margin: 0 0 0.5rem 0; font-size: 1.02rem; color: var(--text); }
.empty-state p, .empty-state li {
    margin: 0 0 0.5rem 0; color: var(--text-secondary);
    font-size: 0.9rem; line-height: 1.55;
}
.empty-state ul { margin: 0.2rem 0 0 1.1rem; padding: 0; }
.empty-state p:last-child, .empty-state li:last-child { margin-bottom: 0; }

/* Keep the sidebar narrow enough to leave the chat usable on a laptop. */
[data-testid="stSidebar"] { min-width: 300px; max-width: 340px; }

.status-row { display: flex; align-items: center; gap: 0.45rem; font-size: 0.82rem; padding: 0.1rem 0; }
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; flex: none; }
.status-ok { background: var(--success); }
.status-bad { background: var(--danger); }

.safety-note { color: var(--text-secondary); font-size: 0.85rem; line-height: 1.45; }

/* Chat. Each message starts with a small role label whose class also marks
   the message for styling - independent of which avatar is shown. */
[data-testid="stChatMessage"] { padding: 0.9rem 1rem; margin-bottom: 0.5rem; gap: 0.75rem; }
[data-testid="stChatMessage"]:has(.msg-user) {
    background: var(--user-bg);
    border-radius: 12px;
}
[data-testid="stChatMessage"]:has(.msg-assistant) {
    background: var(--assistant-bg);
    border-radius: 12px;
    border: 1px solid var(--border);
}
.msg-head {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--text-secondary) !important; margin-bottom: 0.15rem;
}
.user-text { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.55; }
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {
    color: var(--text) !important;
}
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"],
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p {
    color: var(--text-secondary) !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { line-height: 1.6; }
/* Secondary text inside messages: these must out-rank the chat's own
   "[data-testid=stChatMessage] span/p { color: text }" rule above. */
[data-testid="stChatMessage"] .source-pages,
[data-testid="stChatMessage"] .passage-page,
[data-testid="stChatMessage"] .sources-label,
[data-testid="stChatMessage"] .stage,
[data-testid="stChatMessage"] .msg-head,
[data-testid="stChatMessage"] .timing-table td:last-child { color: var(--text-secondary) !important; }
[data-testid="stChatMessage"] .passage-num { color: var(--accent-text) !important; }
[data-testid="stChatMessage"] .source-removed { color: var(--warning) !important; }

/* Live progress line while a turn is being answered. */
.stage { display: flex; align-items: center; gap: 0.55rem; color: var(--text-secondary) !important; font-size: 0.9rem; }
.stage-spinner {
    width: 14px; height: 14px; flex: none; border-radius: 50%;
    border: 2px solid var(--border); border-top-color: var(--accent);
    animation: stage-spin 0.9s linear infinite;
}
@keyframes stage-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .stage-spinner { animation: none; } }

/* Sources: one row per document with its pages. */
.sources-label {
    display: flex; align-items: center; gap: 0.4rem;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--text-secondary) !important; margin: 0.2rem 0 0.35rem 0;
}
.sources-label .ms { font-size: 1.05rem; }
.source-list { display: flex; flex-direction: column; gap: 0.3rem; }
.source-row {
    display: flex; flex-wrap: wrap; align-items: center; gap: 0.15rem 0.5rem;
    border: 1px solid var(--border); border-radius: 6px; background: var(--bg-secondary);
    padding: 0.3rem 0.6rem; font-size: 0.85rem;
}
.source-name { font-weight: 600; overflow-wrap: anywhere; }
.source-pages { color: var(--text-secondary) !important; }
.source-removed { color: var(--warning) !important; font-size: 0.75rem; font-weight: 600; }

/* Evidence passages: metadata header, then the verbatim retrieved text. */
.passage { border: 1px solid var(--border); border-radius: 8px; margin: 0 0 0.6rem 0; overflow: hidden; }
.passage-head {
    display: flex; flex-wrap: wrap; gap: 0.15rem 0.5rem; align-items: baseline;
    background: var(--bg-secondary); border-bottom: 1px solid var(--border);
    padding: 0.35rem 0.65rem; font-size: 0.8rem;
}
.passage-num {
    font-weight: 700; color: var(--accent-text) !important;
    font-variant-numeric: tabular-nums;
}
.passage-source { font-weight: 600; overflow-wrap: anywhere; }
.passage-page { color: var(--text-secondary) !important; }
.passage-text {
    background: var(--passage-bg); padding: 0.6rem 0.75rem;
    font-size: 0.86rem; line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere;
}
.evidence-group { font-size: 0.85rem; font-weight: 700; margin: 0.4rem 0 0.5rem 0; overflow-wrap: anywhere; }

/* Timing details: a small, quiet table. */
.timing-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.timing-table td { padding: 0.25rem 0.2rem; border-bottom: 1px solid var(--border); }
.timing-table td:last-child { text-align: right; font-variant-numeric: tabular-nums; color: var(--text-secondary) !important; }
.timing-table tr.total td { font-weight: 700; border-bottom: none; color: var(--text) !important; }

/* Notices: no evidence (informational) and errors. */
.notice {
    border: 1px solid var(--border); border-left: 3px solid var(--text-secondary);
    border-radius: 8px; padding: 0.7rem 0.9rem; background: var(--bg-secondary);
    font-size: 0.88rem; line-height: 1.55;
}
.notice.error { border-color: var(--danger); border-left-color: var(--danger); background: var(--danger-soft); }
.notice.warning { border-color: var(--warning); border-left-color: var(--warning); background: var(--warning-soft); }
.notice-title { display: flex; align-items: flex-start; gap: 0.45rem; font-weight: 700; margin-bottom: 0.2rem; overflow-wrap: anywhere; }
.notice-title .notice-icon { font-size: 1.15rem; margin-top: 0.05rem; }
.stApp .notice .notice-icon { color: var(--text-secondary) !important; }
.stApp .notice.error .notice-icon { color: var(--danger) !important; }
.stApp .notice.warning .notice-icon { color: var(--warning) !important; }
.stApp .source-row .source-icon { color: var(--text-secondary) !important; font-size: 1rem; }
.stApp .scope-headline .ms { font-size: 1rem; }
.notice p { margin: 0.15rem 0 0 0; color: var(--text) !important; }
.notice code { font-size: 0.8rem; }

.upload-result { font-size: 0.84rem; line-height: 1.45; margin: 0.15rem 0; overflow-wrap: anywhere; }
.upload-result .ok { color: var(--success) !important; font-weight: 700; }
.upload-result .skip { color: var(--text-secondary) !important; font-weight: 700; }

@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding-left: 0.9rem; padding-right: 0.9rem; padding-top: 1.6rem; }
    h1 { font-size: 1.35rem !important; }
    [data-testid="stChatMessage"] { padding: 0.7rem 0.75rem; }
}
"""


def inject_theme_css() -> None:
    theme = st.session_state.get("theme", "dark")
    palette = THEMES[theme]
    css_vars = "\n".join(f"--{key}: {value};" for key, value in palette.items())
    css_vars += f'\n--logo: url("{svg_data_uri(logo_svg(64))}");'
    st.markdown(f"<style>:root {{{css_vars}}}\n{BASE_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Streamlit page (only runs under `streamlit run app.py`, not on plain import)
# ---------------------------------------------------------------------------


def _upload_key(uploaded_file) -> tuple:
    """Identifies one uploaded file across Streamlit reruns (the widget keeps
    returning the same files every rerun until the user changes the
    selection), so already-processed files aren't re-ingested each rerun."""
    return (uploaded_file.name, uploaded_file.size)


def summarize_upload_batch(results: list[dict]) -> str:
    """'2 added · 1 already in your library · 1 failed' (zero counts omitted)."""
    added = sum(r["status"] == "indexed" for r in results)
    skipped = sum(r["status"] == "skipped" for r in results)
    failed = sum(r["status"] == "failed" for r in results)
    parts = []
    if added:
        parts.append(f"{pluralize(added, 'document')} added")
    if skipped:
        parts.append(f"{skipped} already in your library")
    if failed:
        parts.append(f"{failed} couldn't be added")
    return " · ".join(parts) or "Nothing to add"


def _upload_widget_key(key: tuple) -> str:
    return "upload_" + re.sub(r"[^A-Za-z0-9]+", "_", f"{key[0]}_{key[1]}")


def _retry_upload(key: tuple) -> None:
    st.session_state.setdefault("processed_upload_keys", set()).discard(key)
    st.session_state.setdefault("upload_failures", {}).pop(key, None)


def _dismiss_upload_failure(key: tuple) -> None:
    st.session_state.setdefault("upload_failures", {}).pop(key, None)


def _render_upload_result(result: dict) -> None:
    name = escape(result["filename"])
    if result["status"] == "indexed":
        pages = f" from {pluralize(result['page_count'], 'page')}" if result.get("page_count") else ""
        st.markdown(
            f'<div class="upload-result"><span class="ok">✓</span> <strong>{name}</strong> added - '
            f"{pluralize(result['chunk_count'], 'chunk')}{pages}</div>",
            unsafe_allow_html=True,
        )
        if result.get("image_only_pages"):
            pages_list = ", ".join(str(p) for p in result["image_only_pages"])
            st.warning(
                f"Page(s) {pages_list} of **{result['filename']}** contain only images, so their "
                "text can't be read and they won't appear in answers. This app doesn't run OCR."
            )
        if result.get("degraded_pages"):
            pages_list = ", ".join(str(p) for p in result["degraded_pages"])
            st.warning(
                f"Page(s) {pages_list} of **{result['filename']}** have a damaged text layer that "
                "couldn't be repaired, so answers about those pages may be unreliable."
            )
    elif result["status"] == "skipped":
        st.markdown(
            f'<div class="upload-result"><span class="skip">•</span> <strong>{name}</strong> is '
            "already in your library (identical file), so it wasn't added again.</div>",
            unsafe_allow_html=True,
        )


def _render_upload_failures(current_keys: set) -> None:
    for key, failure in list(st.session_state.get("upload_failures", {}).items()):
        err = UserFacingError(**failure)
        render_notice(err.title, [err.cause, err.action], kind="error")
        cols = st.columns(2)
        with cols[0]:
            st.button(
                "Try again",
                icon=ICON["retry"],
                key=f"{_upload_widget_key(key)}_retry",
                use_container_width=True,
                disabled=key not in current_keys,
                help=None if key in current_keys else "Select the file again to retry it.",
                on_click=_retry_upload,
                args=(key,),
            )
        with cols[1]:
            st.button(
                "Dismiss",
                icon=ICON["dismiss"],
                key=f"{_upload_widget_key(key)}_dismiss",
                use_container_width=True,
                on_click=_dismiss_upload_failure,
                args=(key,),
            )


def render_upload_area(busy: bool = False) -> bool:
    """Returns whether any file was processed this run, so the caller knows
    the document list it read earlier is stale. `busy` (an answer is being
    generated) disables the uploader so a new upload can't interrupt it."""
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Add one or more healthcare PDFs to the searchable library.",
        disabled=busy,
    )
    st.markdown(
        '<div class="hint">PDF only. Text-based PDFs work best - scanned pages can\'t be read. '
        "Files are processed on this machine.</div>",
        unsafe_allow_html=True,
    )
    uploaded_files = uploaded_files or []
    current_keys = {_upload_key(f) for f in uploaded_files}

    processed_keys = st.session_state.setdefault("processed_upload_keys", set())
    failures = st.session_state.setdefault("upload_failures", {})
    new_files = [f for f in uploaded_files if _upload_key(f) not in processed_keys]
    if not new_files:
        _render_upload_failures(current_keys)
        return False

    with st.status(f"Adding {pluralize(len(new_files), 'document')}…", expanded=True) as status:
        batch_results = []
        for uploaded_file in new_files:
            progress = st.empty()
            progress.markdown(
                f'<div class="upload-result"><strong>{escape(uploaded_file.name)}</strong> - '
                "reading, splitting and indexing the text…</div>",
                unsafe_allow_html=True,
            )
            result = ingest_uploaded_pdfs([(uploaded_file.name, uploaded_file.getvalue())])[0]
            batch_results.append(result)
            key = _upload_key(uploaded_file)
            processed_keys.add(key)
            progress.empty()
            if result["status"] == "failed":
                failures[key] = asdict(
                    describe_upload_failure(uploaded_file.name, result.get("error_type"), result["error"] or "")
                )
            else:
                failures.pop(key, None)
                _render_upload_result(result)

        has_warnings = any(r.get("image_only_pages") or r.get("degraded_pages") for r in batch_results)
        status.update(
            label=summarize_upload_batch(batch_results),
            state="error" if all(r["status"] == "failed" for r in batch_results) else "complete",
            expanded=has_warnings,
        )
    _render_upload_failures(current_keys)

    # The most recently successfully-processed document (indexed or already
    # present) becomes active, unless the user already had a still-valid
    # active document selected before this upload.
    current_active = st.session_state.get("active_document_id")
    still_valid = current_active in {r["document_id"] for r in batch_results if r["document_id"]} or any(
        d["document_id"] == current_active for d in list_indexed_documents()
    )
    if not still_valid:
        for result in reversed(batch_results):
            if result["document_id"] is not None:
                st.session_state["active_document_id"] = result["document_id"]
                break
    return True


def render_document_library(documents: list[dict], active_id: str | None, busy: bool = False) -> None:
    """One document list, showing each document's state for the CURRENT
    analysis scope.

    There used to be two lists: cards with a "Select" button that set the
    ACTIVE document, and - further down, under "Analysis scope" - a second
    list of checkboxes labelled "Selected documents". The same word meant
    two different things in one sidebar, the same documents were drawn
    twice, and a document's card gave no hint whether it was part of a
    multi-document selection. The scope now decides what the one list does:
    a radio in Active mode, checkboxes in Selected mode.
    """
    if not documents:
        st.markdown(
            '<div class="hint">No documents yet. Add a PDF above and it will appear here.</div>',
            unsafe_allow_html=True,
        )
        return

    scope = st.session_state.get("analysis_scope", "active")
    selected = st.session_state.setdefault("selected_document_ids", set())
    pending_delete = st.session_state.get("pending_delete_id")

    for doc in documents:
        doc_id = doc["document_id"]
        in_scope = doc_id == active_id if scope == "active" else doc_id in selected
        badge = "<span class='doc-badge'>in scope</span>" if in_scope else ""

        st.markdown(
            f'<div class="doc-card{" in-scope" if in_scope else ""}">'
            f'<div class="doc-name" title="{escape(doc["source"])}">{material("description", "doc-icon")}'
            f'<span class="doc-name-text">{escape(doc["source"])}</span></div>'
            f'<div class="doc-meta">{escape(document_meta(doc))}{badge}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # [9, 5], not [3, 1]: at the default 300px sidebar the row has 243px of
        # column space; "Remove" needs ~81px and "Use this document" ~151px
        # (text + button padding, measured). [3, 1] gave Remove 61px and
        # clipped the destructive button's label to "Re...".
        col_scope, col_remove = st.columns([9, 5])
        with col_scope:
            if scope == "active":
                st.button(
                    "In scope" if in_scope else "Use this document",
                    icon=ICON["selected"] if in_scope else None,
                    key=f"select_{doc_id}",
                    use_container_width=True,
                    disabled=in_scope or busy,
                    on_click=_set_active_document,
                    args=(doc_id,),
                )
            else:
                st.checkbox(
                    "Include in analysis",
                    value=in_scope,
                    key=f"scope_select_{doc_id}",
                    disabled=busy,
                    on_change=_toggle_selected_document,
                    args=(doc_id,),
                )
        with col_remove:
            st.button(
                "Remove",
                icon=ICON["remove"],
                key=f"remove_{doc_id}",
                use_container_width=True,
                disabled=busy,
                help=f"Remove {doc['source']} from the library",
                on_click=_request_delete,
                args=(doc_id,),
            )

        if pending_delete == doc_id and not busy:
            st.markdown(
                f'<div class="delete-confirm">Remove <strong>{escape(doc["source"])}</strong> from the '
                "library? Its text and search index are deleted from this machine. Earlier answers "
                "stay in the conversation. This can't be undone.</div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                st.button(
                    "Yes, remove",
                    icon=ICON["remove_confirm"],
                    key=f"confirm_{doc_id}",
                    type="primary",
                    use_container_width=True,
                    on_click=_confirm_delete,
                    args=(doc_id, active_id),
                )
            with c2:
                st.button(
                    "Cancel",
                    icon=ICON["cancel"],
                    key=f"cancel_{doc_id}",
                    use_container_width=True,
                    on_click=_cancel_delete,
                )


def _set_active_document(doc_id: str) -> None:
    st.session_state["active_document_id"] = doc_id
    st.session_state.pop("pending_delete_id", None)


def _toggle_selected_document(doc_id: str) -> None:
    """Mirror one document's checkbox into the selected-document set."""
    selected = st.session_state.setdefault("selected_document_ids", set())
    if st.session_state.get(f"scope_select_{doc_id}"):
        selected.add(doc_id)
    else:
        selected.discard(doc_id)


def _request_delete(doc_id: str) -> None:
    st.session_state["pending_delete_id"] = doc_id


def _cancel_delete() -> None:
    st.session_state.pop("pending_delete_id", None)


def _confirm_delete(doc_id: str, active_id: str | None) -> None:
    name = next((d["source"] for d in list_indexed_documents() if d["document_id"] == doc_id), "The document")
    st.session_state.pop("pending_delete_id", None)
    try:
        delete_document(doc_id)
    except Exception as exc:  # noqa: BLE001 - reported to the user; the store is left as it was
        st.session_state["flash"] = (
            "error",
            f"{name} couldn't be removed ({type(exc).__name__}). Nothing was changed - try again.",
        )
        return
    st.session_state.setdefault("selected_document_ids", set()).discard(doc_id)
    remaining = list_indexed_documents()
    fallback_request = active_id if active_id != doc_id else None
    st.session_state["active_document_id"] = resolve_active_document_id(remaining, fallback_request)
    st.session_state["flash"] = ("success", f"Removed {name} from the library.")


def ollama_status_view(status: dict) -> tuple[bool, str, str]:
    """(ready, label, hint) for the sidebar status row, from get_status()."""
    if status.get("ready"):
        return True, f"Ready · {config.OLLAMA_MODEL}", ""
    if "reachable" in status.get("detail", ""):
        return False, "Ollama isn't running", "Start Ollama, then select Check again."
    return False, "Model not installed", f"Run `ollama pull {config.OLLAMA_MODEL}`, then select Check again."


def _recheck_ollama() -> None:
    _cached_ollama_status.clear()


def render_status() -> None:
    ready, label, hint = ollama_status_view(_cached_ollama_status())
    dot_class = "status-ok" if ready else "status-bad"
    st.markdown(
        f'<div class="status-row"><span class="status-dot {dot_class}"></span>{escape(label)}</div>',
        unsafe_allow_html=True,
    )
    if not ready:
        st.caption(hint)
        st.button("Check again", icon=ICON["retry"], key="recheck_ollama", on_click=_recheck_ollama)


@st.cache_data(ttl=15, show_spinner=False)
def _cached_ollama_status() -> dict:
    return get_ollama_status()


def render_theme_toggle(busy: bool = False) -> None:
    theme = st.session_state.get("theme", "dark")
    is_dark = st.toggle("Dark mode", value=(theme == "dark"), key="theme_toggle", disabled=busy)
    st.session_state["theme"] = "dark" if is_dark else "light"


def render_analysis_scope(documents: list[dict], busy: bool = False) -> None:
    """"Analysis scope" selector: Active document (default, single-document
    isolation, unchanged) or Selected documents (an explicit, opt-in subset
    the user checks off - never "everything" implicitly).
    """
    if not documents:
        return

    st.radio(
        "Analysis scope",
        options=["active", "selected"],
        format_func=lambda v: (
            "One document at a time" if v == "active" else "Several documents together"
        ),
        key="analysis_scope",
        disabled=busy,
        help=(
            "One document: questions are answered from a single document, fully "
            "isolated from the rest of the library. Several documents: compare "
            "or summarise across the documents you tick below."
        ),
    )


def render_section_label(text: str, icon: str, count: int | None = None) -> None:
    suffix = f'<span class="count">({count})</span>' if count else ""
    st.markdown(
        f'<div class="section-label">{material(icon)}<span>{escape(text)}</span>{suffix}</div>',
        unsafe_allow_html=True,
    )


def render_brand_footer() -> None:
    """A quiet project/author mark at the foot of the sidebar - the product
    title stays the header's job."""
    st.markdown(
        f'<div class="brand-footer"><img src="{svg_data_uri(logo_svg(20))}" width="20" height="20" alt="">'
        f'<div><div class="brand-footer-title">RAG for Healthcare Documents</div>'
        f'<div class="brand-footer-author">Research prototype · {escape(AUTHOR)}</div></div></div>',
        unsafe_allow_html=True,
    )


def _clear_chat() -> None:
    st.session_state["chat_history"] = []


def render_sidebar(documents: list[dict], active_id: str | None) -> tuple[list[dict], str | None]:
    """Returns the document list and active document as they stand after
    any upload this run, for the chat area rendered after the sidebar.

    While an answer is being generated every control that could change what
    it is about (upload, scope, library, clear) is disabled, so it can't be
    interrupted or have its documents removed underneath it."""
    busy = "pending_turn" in st.session_state
    render_theme_toggle(busy)
    render_status()

    st.divider()
    render_section_label("Add documents", "upload_file")
    if render_upload_area(busy):
        documents = list_indexed_documents()
        active_id = resolve_active_document_id(documents, st.session_state.get("active_document_id"))
        st.session_state["active_document_id"] = active_id

    st.divider()
    # Scope comes before the library because it decides what the library's
    # per-document control means (pick one, or tick several).
    render_analysis_scope(documents, busy)

    render_section_label("Library", "folder_open", len(documents))
    render_document_library(documents, active_id, busy)

    st.divider()
    st.button(
        "Clear conversation",
        icon=ICON["clear"],
        use_container_width=True,
        disabled=busy or not st.session_state.get("chat_history"),
        on_click=_clear_chat,
    )

    with st.expander("About this app", icon=ICON["about"]):
        st.markdown(f'<p class="safety-note">{escape(SAFETY_NOTICE)}</p>', unsafe_allow_html=True)
        st.caption(
            "Answers are generated locally by "
            f"{config.OLLAMA_MODEL} via Ollama from passages found with hybrid search "
            "(FAISS vectors + SQLite FTS5 keywords, fused with reciprocal rank fusion). "
            "Documents, the index and the conversation stay on this machine."
        )
        st.caption(f"Built by {AUTHOR}.")
    render_brand_footer()
    return documents, active_id


def render_sources(sources: list[dict], current_names: set | None = None) -> None:
    """One row per document with the pages the answer drew on. A document
    removed from the library since is marked, so an old answer's sources
    stay truthful about what can still be opened."""
    if not sources:
        return
    rows = []
    for group in group_sources(sources):
        pages = group["pages"]
        word = "Page" if len(pages) == 1 else "Pages"
        removed = (
            '<span class="source-removed">removed from library</span>'
            if current_names is not None and group["source"] not in current_names
            else ""
        )
        rows.append(
            f'<div class="source-row">{material("description", "source-icon")}'
            f'<span class="source-name">{escape(group["source"])}</span>'
            f'<span class="source-pages">{word} {", ".join(str(p) for p in pages)}</span>{removed}</div>'
        )
    st.markdown(
        f'<div class="sources-label">{material("menu_book")}<span>Sources</span></div>'
        f'<div class="source-list">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def timing_rows(timing: dict) -> list[tuple[str, str]]:
    """Readable (label, value) rows for the timing panel; stages that
    weren't measured are left out rather than shown as 0 ms."""
    rows = []
    retrieval_keys = ("embedding_ms", "dense_ms", "keyword_ms", "fusion_ms")
    if any(k in timing for k in retrieval_keys):
        retrieval_ms = sum(timing.get(k, 0.0) for k in retrieval_keys)
        rows.append(("Finding relevant passages", f"{retrieval_ms:.0f} ms"))
    if "prompt_build_ms" in timing:
        rows.append(("Preparing the question for the model", f"{timing['prompt_build_ms']:.1f} ms"))
    if timing.get("time_to_first_token_ms") is not None:
        rows.append(("Until the answer started", f"{timing['time_to_first_token_ms']:.0f} ms"))
    gen = timing.get("generation") or {}
    if gen.get("load_ms", 0) >= 50:
        rows.append(("Loading the model", f"{gen['load_ms']:.0f} ms"))
    if gen.get("eval_ms"):
        tokens = f" ({gen['completion_tokens']} tokens)" if gen.get("completion_tokens") else ""
        rows.append((f"Writing the answer{tokens}", f"{gen['eval_ms']:.0f} ms"))
    if timing.get("total_ms") is not None:
        rows.append(("Total", f"{timing['total_ms'] / 1000:.1f} s"))
    return rows


def render_timing(timing: dict) -> None:
    """Collapsed by default - secondary to the answer."""
    rows = timing_rows(timing)
    if not rows:
        return
    with st.expander("Timing", icon=ICON["timing"]):
        body = "".join(
            f'<tr class="{"total" if label == "Total" else ""}"><td>{escape(label)}</td><td>{escape(value)}</td></tr>'
            for label, value in rows
        )
        st.markdown(f'<table class="timing-table">{body}</table>', unsafe_allow_html=True)


def _render_evidence_chunks(chunks: list[dict]) -> None:
    for chunk in chunks:
        st.markdown(
            '<div class="passage"><div class="passage-head">'
            f'<span class="passage-num">[{chunk["rank"]}]</span>'
            f'<span class="passage-source">{escape(chunk["source"])}</span>'
            f'<span class="passage-page">Page {chunk["page"]}</span></div>'
            f'<div class="passage-text">{escape(chunk["text"])}</div></div>',
            unsafe_allow_html=True,
        )


def render_evidence(chunks: list[dict]) -> None:
    """Single-document mode (active-document scope, or "selected" with all
    evidence from one document) always has exactly one distinct source, so
    this renders identically to before: one flat, rank-ordered list.
    Multi-document answers (more than one distinct source among the
    retrieved chunks) instead group evidence by document, each chunk still
    labeled with its citation number ([N] = its rank from retrieve_hybrid,
    matching the number the model was given in the prompt - see
    prompt_builder.build_multi_document_context), so a citation in the
    answer always points at the same chunk here regardless of display order.
    """
    with st.expander(f"Evidence · {pluralize(len(chunks), 'passage')} given to the model", icon=ICON["evidence"]):
        st.caption("The exact text retrieved from your documents. Numbers match the model's evidence labels.")
        distinct_sources = list(dict.fromkeys(c["source"] for c in chunks))
        if len(distinct_sources) <= 1:
            _render_evidence_chunks(chunks)
            return

        for source in distinct_sources:
            st.markdown(f'<div class="evidence-group">{escape(source)}</div>', unsafe_allow_html=True)
            _render_evidence_chunks([c for c in chunks if c["source"] == source])


def _stream_with_ttft(
    prompt: str,
    gen_result: dict,
    timing: dict,
    num_predict: int | None = None,
    on_first_token=None,
):
    """Wrap generate_stream() to also record time-to-first-token into
    `timing` (UI-only instrumentation; generate_stream itself stays
    display-agnostic and reusable from non-Streamlit callers).

    `on_first_token` (optional), if given, is called once, the moment the
    first token arrives - before it is yielded. render_chat() uses this to
    clear its "Generating answer..." status the instant real output starts,
    instead of leaving it on screen next to the streamed text or guessing at
    when to remove it.
    """
    start = time.perf_counter()
    first_token = True
    for token in generate_stream(prompt, result=gen_result, num_predict=num_predict):
        if first_token:
            timing["time_to_first_token_ms"] = (time.perf_counter() - start) * 1000
            first_token = False
            if on_first_token is not None:
                on_first_token()
        yield token


def render_empty_state(has_documents: bool) -> None:
    """Two genuinely different situations, which used to share one message
    that said "No document selected yet" even when the library was empty."""
    if not has_documents:
        st.markdown(
            '<div class="empty-state">'
            "<h3>Start by adding a document</h3>"
            "<p>Upload a healthcare PDF using <strong>Add documents</strong> in the "
            "sidebar. It is parsed, split into passages and indexed on this machine - "
            "nothing is uploaded anywhere.</p>"
            "<p>Once it is indexed you can ask questions about it. Each answer lists "
            "the pages it drew on and shows the exact passages it was given.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="empty-state">'
        "<h3>Choose what to analyse</h3>"
        "<p>Your library has documents, but none are in scope yet. Pick one under "
        "<strong>Library</strong> in the sidebar, or switch to "
        "<em>Several documents together</em> and tick the ones you want to "
        "compare.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def shorten(text: str, limit: int = 48) -> str:
    """Middle-truncate a long filename for single-line UI text, keeping the
    end (often a version or date) visible: 'Discharge summary wi…FINAL.pdf'."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head].rstrip()}…{text[-(limit - head - 1):].lstrip()}"


def render_conversation_start() -> None:
    """Scope is ready but nothing has been asked yet. The scope banner above
    already names the documents in full, so the heading doesn't repeat them."""
    st.markdown(
        '<div class="empty-state">'
        "<h3>Ask your first question</h3>"
        "<ul>"
        "<li>Answers are written only from passages found in the documents in scope.</li>"
        "<li>Under each answer you can see which pages it used and read the exact passages.</li>"
        "<li>Specific questions work best - a name, a value, a date, a section.</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_notice(title: str, lines: list[str], kind: str = "info") -> None:
    """A calm bordered notice; `kind` is 'info', 'warning' or 'error'. Text
    is escaped except `backticks`, which become inline code."""
    def fmt(text: str) -> str:
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", escape(text))

    glyph = {"info": "info", "warning": "warning", "error": "error"}.get(kind, "info")
    body = "".join(f"<p>{fmt(line)}</p>" for line in lines if line)
    st.markdown(
        f'<div class="notice {kind}"><div class="notice-title">{material(glyph, "notice-icon")}'
        f"<span>{fmt(title)}</span></div>{body}</div>",
        unsafe_allow_html=True,
    )


def render_no_evidence(scope_label: str) -> None:
    """Shown when retrieval finds nothing. Deliberately not an error: the
    system has no evidence either way, which is not the same as the fact
    being false, and saying so is the honest thing for a clinical tool."""
    render_notice(
        f"No matching passages in {scope_label}",
        [
            "This doesn't mean the answer is no - only that nothing in the documents "
            "in scope matched the question.",
            "Try rephrasing it with words the document is likely to use, or widen the scope in the sidebar.",
        ],
    )


def render_error_turn(turn: dict, retry_enabled: bool, key: str) -> None:
    err = turn["error"]
    render_notice(err["title"], [err["cause"], err["action"]], kind="error")
    if retry_enabled:
        st.button(
            "Retry", icon=ICON["retry"], key=key, type="primary", on_click=retry_last_turn, args=(st.session_state,)
        )
    if err.get("detail"):
        with st.expander("Technical details", icon=ICON["details"]):
            st.code(err["detail"], language=None)


def render_scope_banner(headline: str, names: list[str]) -> None:
    """A compact, always-visible statement of exactly which documents this
    turn can draw on - the single most important thing to be unambiguous
    about in a tool whose whole promise is document isolation."""
    chips = "".join(f'<span class="scope-chip">{escape(n)}</span>' for n in names)
    st.markdown(
        f'<div class="scope-banner"><div class="scope-headline">{material("visibility")}<span>{escape(headline)}</span></div>'
        f'<div class="scope-chips">{chips}</div></div>',
        unsafe_allow_html=True,
    )


def render_chat(
    documents: list[dict],
    active_document_id: str | None,
    analysis_scope: str,
    selected_document_ids,
) -> None:
    # Defensive intersection against what's actually indexed right now, so a
    # deleted or otherwise stale id can never linger in the query scope -
    # deletion already prunes this set too (see _confirm_delete), this is
    # belt-and-suspenders for "unselected/deleted document never appears".
    valid_ids = {d["document_id"] for d in documents}
    selected_document_ids = {d for d in selected_document_ids if d in valid_ids}

    scope_document_id = resolve_scope_document_ids(analysis_scope, active_document_id, selected_document_ids)
    # None in active-document mode: build_prompt() then takes its original
    # single-document path, completely unchanged (see its docstring).
    document_inventory = (
        build_document_inventory(documents, selected_document_ids) if analysis_scope == "selected" else None
    )

    if analysis_scope == "selected":
        scope_ready = bool(scope_document_id)
        names = [d["source"] for d in documents if d["document_id"] in selected_document_ids]
        scope_label = f"{len(names)} documents" if len(names) != 1 else names[0]
        if scope_ready:
            noun = "document" if len(names) == 1 else "documents"
            render_scope_banner(f"Analysing {len(names)} {noun}", names)
        placeholder = (
            f"Ask about {shorten(names[0])}…"
            if len(names) == 1
            else f"Ask about these {len(names)} documents…"
        ) if scope_ready else "Tick at least one document in the sidebar…"
    else:
        scope_ready = scope_document_id is not None
        active_doc = next((d for d in documents if d["document_id"] == scope_document_id), None)
        scope_label = active_doc["source"] if active_doc else "the selected document"
        if scope_ready and active_doc:
            render_scope_banner("Analysing one document", [active_doc["source"]])
        placeholder = f"Ask about {shorten(active_doc['source'])}…" if active_doc else "Ask a question…"

    history = st.session_state["chat_history"]
    pending = st.session_state.get("pending_turn")
    current_names = {d["source"] for d in documents}
    for i, turn in enumerate(history):
        render_turn(turn, current_names, retry_key=f"retry_{i}", is_last=i == len(history) - 1)

    if pending is not None:
        st.chat_input("Writing an answer…", key="chat_input", disabled=True)
        run_pending_turn(pending, documents)
        # Re-render from the committed history, which also re-enables the input.
        st.rerun()
        return

    if not scope_ready:
        render_empty_state(has_documents=bool(documents))
        return

    if not history:
        render_conversation_start()

    query = st.chat_input(placeholder, key="chat_input", max_chars=MAX_QUERY_CHARS)
    if not query or not query.strip():
        return

    queue_question(
        st.session_state,
        query.strip(),
        {
            "analysis_scope": analysis_scope,
            "document_id": scope_document_id,
            "document_inventory": document_inventory,
            "top_k": resolve_retrieval_top_k(analysis_scope, selected_document_ids),
            "scope_label": scope_label,
        },
    )
    st.rerun()


def render_turn(turn: dict, current_names: set, retry_key: str, is_last: bool) -> None:
    """One committed turn, exactly as it looked when it finished."""
    if turn["role"] == "user":
        with st.chat_message("user", avatar=user_avatar_svg()):
            st.markdown(
                f'<div class="msg-head msg-user">You</div><div class="user-text">{escape(turn["content"])}</div>',
                unsafe_allow_html=True,
            )
        return

    with st.chat_message("assistant", avatar=assistant_avatar_svg()):
        st.markdown('<div class="msg-head msg-assistant">Answer</div>', unsafe_allow_html=True)
        status = turn.get("status", "ok")
        if status == "error":
            render_error_turn(turn, retry_enabled=is_last and can_retry(st.session_state), key=retry_key)
            return
        if status == "no_evidence":
            render_no_evidence(turn.get("scope_label", "the documents in scope"))
            return
        st.markdown(turn["content"])
        turn_timing = turn.get("timing") or {}
        if turn_timing.get("total_ms") is not None:
            render_completion_status(turn_timing["total_ms"], turn.get("truncated", False), turn.get("chunks") or [])
        render_sources(turn.get("sources") or [], current_names)
        if turn.get("chunks"):
            render_evidence(turn["chunks"])
        if turn_timing:
            render_timing(turn_timing)


def render_completion_status(total_ms: float, truncated: bool, chunks: list[dict] | None = None) -> None:
    kind, text = completion_status(total_ms, truncated)
    if kind == "warning":
        render_notice("This answer may be incomplete", [text], kind="warning")
        return
    if chunks:
        n_docs = len({c["source"] for c in chunks})
        text += f" · based on {pluralize(len(chunks), 'passage')} from {pluralize(n_docs, 'document')}"
    st.caption(text)


def _stage(placeholder, text: str) -> None:
    placeholder.markdown(
        f'<div class="stage"><span class="stage-spinner"></span>{escape(text)}</div>', unsafe_allow_html=True
    )


def run_pending_turn(pending: dict, documents: list[dict]) -> None:
    """Answer the queued question, streaming it into an assistant message,
    then commit exactly one turn - the answer, a no-evidence notice, or an
    error the user can retry. A partially streamed answer that then fails
    is discarded rather than kept as if it were complete."""
    pending["attempts"] = pending.get("attempts", 0) + 1
    scope = pending["scope"]
    query = pending["query"]

    if pending["attempts"] > MAX_GENERATION_ATTEMPTS:
        err = UserFacingError(
            "The answer was interrupted",
            "Generating this answer was interrupted several times before it could finish.",
            "Select Retry and avoid using the sidebar until the answer has finished.",
        )
        finish_pending_turn(st.session_state, error_turn(err, pending))
        return

    scope_ids = scope["document_id"] if isinstance(scope["document_id"], list) else [scope["document_id"]]
    indexed_ids = {d["document_id"] for d in documents}
    if any(doc_id not in indexed_ids for doc_id in scope_ids):
        err = UserFacingError(
            "A document in scope is no longer in the library",
            "It was removed before this answer could be generated.",
            "Choose a document in the sidebar and ask the question again.",
        )
        finish_pending_turn(st.session_state, error_turn(err, pending))
        return

    prior_history = conversation_context(st.session_state["chat_history"][: pending["history_len"]])
    num_predict = resolve_num_predict(query, prior_history, scope["document_inventory"])

    with st.chat_message("assistant", avatar=assistant_avatar_svg()):
        st.markdown('<div class="msg-head msg-assistant">Answer</div>', unsafe_allow_html=True)
        timing: dict = {}
        turn_start = time.perf_counter()

        # Truthful about which stage is running: retrieval takes tens of
        # milliseconds, the real wait is generation. Cleared the instant the
        # first token arrives (on_first_token below).
        stage = st.empty()
        _stage(stage, "Searching the documents in scope…")
        try:
            chunks, prompt = retrieve_and_build_prompt(
                query,
                top_k=scope["top_k"],
                document_id=scope["document_id"],
                timing=timing,
                document_inventory=scope["document_inventory"],
                history=prior_history,
            )
        except InsufficientRetrievalError:
            # Not an error: nothing matched, which is an honest answer.
            stage.empty()
            finish_pending_turn(
                st.session_state,
                {
                    "role": "assistant",
                    "status": "no_evidence",
                    "scope_label": scope["scope_label"],
                    "content": f"No relevant evidence was found in {scope['scope_label']} for that question.",
                },
            )
            return
        except Exception as e:  # noqa: BLE001 - every failure becomes a retryable error turn
            stage.empty()
            finish_pending_turn(st.session_state, error_turn(describe_error(e), pending))
            return

        _stage(stage, f"Writing the answer from {pluralize(len(chunks), 'passage')}…")
        gen_result: dict = {}
        try:
            st.write_stream(
                _stream_with_ttft(
                    prompt, gen_result, timing, num_predict=num_predict, on_first_token=stage.empty
                )
            )
        except Exception as e:  # noqa: BLE001
            stage.empty()
            # A generation attempt is first-hand evidence of Ollama's state;
            # don't let the sidebar keep showing a cached status that disagrees.
            _cached_ollama_status.clear()
            finish_pending_turn(st.session_state, error_turn(describe_error(e), pending))
            return

        _cached_ollama_status.clear()
        timing["generation"] = gen_result.get("timing", {})
        timing["total_ms"] = (time.perf_counter() - turn_start) * 1000

    finish_pending_turn(
        st.session_state,
        {
            "role": "assistant",
            "status": "ok",
            "content": gen_result.get("answer", ""),
            "sources": [{"source": c["source"], "page": c["page"], "score": c["score"]} for c in chunks],
            "chunks": chunks,
            "timing": timing,
            "truncated": gen_result.get("truncated", False),
        },
    )


def ensure_index_consistency():
    """Reconcile SQLite/FAISS (see src/retrieval/index_consistency.py)
    before anything else touches either store - the known failure this
    guards is a process kill between an ingest's SQLite commit and its
    FAISS save, which leaves a document's chunks searchable by keyword but
    invisible to dense retrieval.

    Cheap when already consistent - an id-set diff plus an O(n) norm check
    over at most a few hundred vectors, this project's own observed scale -
    so it runs on every call (a Streamlit rerun included) rather than being
    cached: simpler than adding a "checked once" flag, and the caching
    would save time nobody would notice at this scale.

    Returns the post-repair report if something remains wrong (only a
    dimension mismatch or an unrepaired corruption can leave this non-None -
    both are deliberately never auto-repaired, see repair()'s docstring), or
    None if the index was already consistent or repair() fixed everything.
    """
    report = check_consistency()
    if report.is_consistent:
        return None

    print(
        "Index consistency check found an issue - repairing: "
        f"dense_invisible={len(report.dense_invisible_chunk_ids)} "
        f"orphan={len(report.orphan_faiss_ids)} duplicate={len(report.duplicate_faiss_ids)} "
        f"dimension_mismatch={report.dimension_mismatch} corrupted={report.faiss_corrupted}"
    )
    repaired = repair()
    if repaired.is_consistent:
        print("Index consistency repaired.")
        return None

    print(
        "Index consistency issue could not be auto-repaired - "
        f"dimension_mismatch={repaired.dimension_mismatch} "
        f"unnormalized={len(repaired.unnormalized_chunk_ids)}"
    )
    return repaired


def main() -> None:
    st.set_page_config(page_title="RAG for Healthcare Documents", page_icon=logo_svg(64), layout="wide")

    consistency_issue = ensure_index_consistency()
    if consistency_issue is not None:
        if consistency_issue.dimension_mismatch:
            st.error(
                "The document index's vectors do not match the configured embedding "
                "model (dimension mismatch) and could not be automatically repaired. "
                "Dense retrieval is unavailable until the index is rebuilt."
            )
        else:
            st.warning(
                "The document index could not be fully repaired on startup. Dense "
                "retrieval may be incomplete until the index is rebuilt."
            )

    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("active_document_id", None)
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("analysis_scope", "active")
    st.session_state.setdefault("selected_document_ids", set())

    # The toggle widget (rendered later, inside the sidebar) is the source of
    # truth once it exists. Resolving "theme" from it here - before the CSS
    # is injected - avoids the CSS lagging one rerun behind a toggle click.
    if "theme_toggle" in st.session_state:
        st.session_state["theme"] = "dark" if st.session_state["theme_toggle"] else "light"

    inject_theme_css()

    st.title("RAG for Healthcare Documents")
    st.markdown(f'<p class="app-subtitle">{escape(TAGLINE)}</p>', unsafe_allow_html=True)

    flash = st.session_state.pop("flash", None)
    if flash:
        kind, text = flash
        st.toast(text, icon=ICON["removed"] if kind == "success" else ICON["failed"])

    documents = list_indexed_documents()
    active_id = resolve_active_document_id(documents, st.session_state.get("active_document_id"))
    st.session_state["active_document_id"] = active_id

    # Fire-and-forget model warm-up, once per session, as soon as there's a
    # document to chat about - gives Ollama a head start loading the model
    # while the user is still reading/selecting, so their first real
    # question is less likely to pay the full cold-load latency. Does not
    # change keep_alive, so an idle model still unloads as configured.
    if active_id and not st.session_state.get("model_warmed"):
        warm_up_ollama()
        # The embedding model too: its torch import (~16s cold) otherwise
        # lands inside the first query's "Retrieving evidence" step.
        warm_up_embedder()
        st.session_state["model_warmed"] = True

    with st.sidebar:
        documents, active_id = render_sidebar(documents, active_id)

    render_chat(
        documents,
        active_id,
        st.session_state.get("analysis_scope", "active"),
        st.session_state.get("selected_document_ids", set()),
    )


if __name__ == "__main__":
    main()
