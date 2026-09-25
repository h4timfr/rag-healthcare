"""Streamlit dashboard for the RAG for Healthcare Documents pipeline.

This file is UI glue only: it saves an uploaded PDF, hands it to the existing
ingestion pipeline, and calls the existing hybrid retriever / prompt builder /
Ollama client for querying. No extraction, chunking, embedding, FAISS, FTS5,
prompt, or generation logic is reimplemented here.
"""

import shutil
import sys
import tempfile
import time
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from scripts import ingest as ingest_module
from src import config
from src.exceptions import (
    DatabaseNotFoundError,
    EmptyQueryError,
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
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.hybrid_retriever import retrieve_for_question, retrieve_hybrid
from src.retrieval.metadata_store import MetadataStore

TAGLINE = "A retrieval-augmented question-answering prototype over healthcare PDFs"
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
    safe_name = Path(filename).name or "upload.pdf"
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
            raise type(e)(str(e).replace(str(tmp_path), safe_name)) from e
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

    Returns {"removed_chunks": int}.
    """
    store = MetadataStore(config.DB_PATH)
    chunk_ids = store.delete_document(document_id)

    if chunk_ids:
        faiss_index = FaissIndex(config.EMBEDDING_DIM, config.FAISS_INDEX_PATH)
        faiss_index.load()
        faiss_index.remove_ids(chunk_ids)
        faiss_index.save()

    return {"removed_chunks": len(chunk_ids)}


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


def friendly_error_message(exc: Exception) -> str:
    """A short, non-technical message for the chat UI. RagError subclasses
    already carry human-readable text; this adds actionable hints for the
    most common operational failures and a safe fallback for anything else
    (so an unexpected bug shows a plain message instead of a raw traceback).
    """
    if isinstance(exc, OllamaUnavailableError):
        return f"Could not reach Ollama - is it running locally? ({exc})"
    if isinstance(exc, ModelUnavailableError):
        return f"The configured model isn't available in Ollama yet. ({exc})"
    if isinstance(exc, RagError):
        return str(exc)
    return "Something unexpected went wrong answering that question. Please try again."


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
        "user-bg": "#1f2937",
        "assistant-bg": "#161b22",
        "danger": "#f85149",
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
        "user-bg": "#f0f4f9",
        "assistant-bg": "#f6f8fa",
        "danger": "#cf222e",
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
[data-testid="stExpander"] {
    background: var(--card-bg) !important;
    border-color: var(--border) !important;
}
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div:first-child {
    background: var(--card-bg) !important;
    border-color: var(--border) !important;
}
[data-testid="stChatInputTextArea"] { color: var(--text) !important; background: transparent !important; }
[data-testid="stChatInputTextArea"]::placeholder { color: var(--text-secondary) !important; opacity: 1 !important; }

.app-subtitle { color: var(--text-secondary); font-size: 0.95rem; margin-top: -0.6rem; }

.doc-card {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
    margin-top: 0.6rem;
    margin-bottom: 0.3rem;
    background: var(--card-bg);
}
.doc-card.in-scope { border-color: var(--accent); background: var(--accent-soft); }
/* Long filenames truncate to one line with the full name on hover, so a
   30-character report name cannot push the sidebar controls out of shape. */
.doc-name {
    font-weight: 600; font-size: 0.92rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.doc-meta {
    color: var(--text-secondary); font-size: 0.78rem; margin-top: 0.15rem;
    display: flex; align-items: center; gap: 0.4rem;
}
.doc-badge {
    color: var(--accent); border: 1px solid var(--accent);
    border-radius: 999px; padding: 0 0.4rem;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.02em;
}

/* Scope banner - what this turn can actually see. */
.scope-banner {
    border: 1px solid var(--border); border-left: 3px solid var(--accent);
    border-radius: 8px; background: var(--bg-secondary);
    padding: 0.6rem 0.8rem; margin-bottom: 1rem;
}
.scope-headline {
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
    padding: 1.6rem 1.8rem; background: var(--bg-secondary);
    max-width: 640px;
}
.empty-state h3 { margin: 0 0 0.6rem 0; font-size: 1.05rem; color: var(--text); }
.empty-state p {
    margin: 0 0 0.6rem 0; color: var(--text-secondary);
    font-size: 0.9rem; line-height: 1.55;
}
.empty-state p:last-child { margin-bottom: 0; }

/* Keep the sidebar narrow enough to leave the chat usable on a laptop. */
[data-testid="stSidebar"] { min-width: 300px; max-width: 340px; }

.status-badge { font-size: 0.8rem; padding: 0.15rem 0; }
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.4rem; }
.status-ok { background: #3fb950; }
.status-bad { background: var(--danger); }

.safety-note { color: var(--text-secondary); font-size: 0.85rem; line-height: 1.4; }

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--user-bg);
    border-radius: 12px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--assistant-bg);
    border-radius: 12px;
    border: 1px solid var(--border);
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {
    color: var(--text) !important;
}
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {
    color: var(--text-secondary) !important;
}
"""


def inject_theme_css() -> None:
    theme = st.session_state.get("theme", "dark")
    palette = THEMES[theme]
    css_vars = "\n".join(f"--{key}: {value};" for key, value in palette.items())
    st.markdown(f"<style>:root {{{css_vars}}}\n{BASE_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Streamlit page (only runs under `streamlit run app.py`, not on plain import)
# ---------------------------------------------------------------------------


def _upload_key(uploaded_file) -> tuple:
    """Identifies one uploaded file across Streamlit reruns (the widget keeps
    returning the same files every rerun until the user changes the
    selection), so already-processed files aren't re-ingested each rerun."""
    return (uploaded_file.name, uploaded_file.size)


def render_upload_area() -> None:
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Add one or more healthcare PDFs to the searchable library.",
    )
    if not uploaded_files:
        return

    processed_keys = st.session_state.setdefault("processed_upload_keys", set())
    new_files = [f for f in uploaded_files if _upload_key(f) not in processed_keys]
    if not new_files:
        return

    with st.status(f"Indexing {len(new_files)} document(s)...", expanded=True) as status:
        batch_results = []
        for uploaded_file in new_files:
            st.write(f"**{uploaded_file.name}**")
            progress = st.empty()
            progress.caption("uploading -> extracting -> chunking -> embedding -> indexing...")

            result = ingest_uploaded_pdfs([(uploaded_file.name, uploaded_file.getvalue())])[0]
            batch_results.append(result)
            processed_keys.add(_upload_key(uploaded_file))

            if result["status"] == "indexed":
                pages = result.get("page_count")
                detail = f"Indexed - {result['chunk_count']} chunks"
                if pages:
                    detail += f" from {pages} pages"
                progress.caption(detail)
                if result.get("repaired_pages"):
                    st.caption(
                        f"Page(s) {result['repaired_pages']} had malformed word "
                        "boundaries and were re-extracted from glyph geometry."
                    )
                if result.get("image_only_pages"):
                    st.warning(
                        f"Page(s) {result['image_only_pages']} of **{uploaded_file.name}** "
                        "contain only images, so their text could not be read and they "
                        "are not searchable. This app does not run OCR; a text-based PDF "
                        "will give much better answers."
                    )
                if result.get("degraded_pages"):
                    st.warning(
                        f"Page(s) {result['degraded_pages']} of **{uploaded_file.name}** "
                        "extracted without usable word boundaries and could not be "
                        "repaired, so keyword search and answer quality will be poor for "
                        "this document. This is a property of the PDF's text layer, not "
                        "of your question."
                    )
            elif result["status"] == "skipped":
                progress.caption(
                    f"Already in your library - {result['chunk_count']} chunks (not re-indexed)."
                )
            else:
                progress.caption("Could not be indexed.")
                st.error(f"**{uploaded_file.name}** - {result['error']}")

        indexed = sum(1 for r in batch_results if r["status"] == "indexed")
        skipped = sum(1 for r in batch_results if r["status"] == "skipped")
        failed = sum(1 for r in batch_results if r["status"] == "failed")
        status.update(
            label=f"Done: {indexed} indexed, {skipped} skipped, {failed} failed",
            state="error" if failed else "complete",
            expanded=failed > 0,
        )

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


def render_document_library(documents: list[dict], active_id: str | None) -> None:
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
        st.caption("No documents yet - upload a PDF above to begin.")
        return

    scope = st.session_state.get("analysis_scope", "active")
    selected = st.session_state.setdefault("selected_document_ids", set())
    pending_delete = st.session_state.get("pending_delete_id")

    for doc in documents:
        doc_id = doc["document_id"]
        in_scope = doc_id == active_id if scope == "active" else doc_id in selected

        meta_bits = [f"{doc['chunk_count']} chunks"]
        if doc.get("page_count"):
            meta_bits.append(f"{doc['page_count']} pages")

        st.markdown(
            f'<div class="doc-card{" in-scope" if in_scope else ""}">'
            f'<div class="doc-name" title="{escape(doc["source"])}">{escape(doc["source"])}</div>'
            f'<div class="doc-meta">{" · ".join(meta_bits)}'
            f'{"<span class=\'doc-badge\'>in scope</span>" if in_scope else ""}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        col_scope, col_remove = st.columns([3, 1])
        with col_scope:
            if scope == "active":
                st.button(
                    "In scope" if in_scope else "Use this document",
                    key=f"select_{doc_id}",
                    use_container_width=True,
                    disabled=in_scope,
                    on_click=_set_active_document,
                    args=(doc_id,),
                )
            else:
                st.checkbox(
                    "Include in analysis",
                    value=in_scope,
                    key=f"scope_select_{doc_id}",
                    on_change=_toggle_selected_document,
                    args=(doc_id,),
                )
        with col_remove:
            st.button(
                "Remove",
                key=f"remove_{doc_id}",
                use_container_width=True,
                help=f"Delete {doc['source']} from the library",
                on_click=_request_delete,
                args=(doc_id,),
            )

        if pending_delete == doc_id:
            st.warning(f"Delete **{doc['source']}** permanently? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                st.button(
                    "Confirm delete",
                    key=f"confirm_{doc_id}",
                    type="primary",
                    use_container_width=True,
                    on_click=_confirm_delete,
                    args=(doc_id, active_id),
                )
            with c2:
                st.button(
                    "Cancel",
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
    delete_document(doc_id)
    st.session_state.pop("pending_delete_id", None)
    st.session_state.setdefault("selected_document_ids", set()).discard(doc_id)
    remaining = list_indexed_documents()
    fallback_request = active_id if active_id != doc_id else None
    st.session_state["active_document_id"] = resolve_active_document_id(remaining, fallback_request)


def render_status() -> None:
    status = _cached_ollama_status()
    dot_class = "status-ok" if status["ready"] else "status-bad"
    st.markdown(
        f'<div class="status-badge"><span class="status-dot {dot_class}"></span>{escape(status["detail"])}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=15, show_spinner=False)
def _cached_ollama_status() -> dict:
    return get_ollama_status()


def render_theme_toggle() -> None:
    theme = st.session_state.get("theme", "dark")
    is_dark = st.toggle("Dark mode", value=(theme == "dark"), key="theme_toggle")
    st.session_state["theme"] = "dark" if is_dark else "light"


def render_analysis_scope(documents: list[dict]) -> None:
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
        help=(
            "One document: questions are answered from a single document, fully "
            "isolated from the rest of the library. Several documents: compare "
            "or summarise across the documents you tick below."
        ),
    )


def render_sidebar(documents: list[dict], active_id: str | None) -> None:
    render_theme_toggle()
    render_status()

    st.divider()
    st.subheader("Add documents")
    render_upload_area()

    st.divider()
    # Scope comes before the library because it decides what the library's
    # per-document control means (pick one, or tick several).
    render_analysis_scope(documents)

    header = "Library" if not documents else f"Library ({len(documents)})"
    st.subheader(header)
    render_document_library(documents, active_id)

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state["chat_history"] = []

    with st.expander("About / System information"):
        st.markdown(f'<p class="safety-note">{escape(SAFETY_NOTICE)}</p>', unsafe_allow_html=True)
        st.caption(
            "Hybrid FAISS (dense) + SQLite FTS5 (keyword) retrieval, fused with "
            "reciprocal rank fusion. Answers are generated locally by "
            f"{config.OLLAMA_MODEL} via Ollama - no data leaves this machine."
        )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("**Sources**")
    for group in group_sources(sources):
        pages = group["pages"]
        word = "page" if len(pages) == 1 else "pages"
        st.caption(f"• {group['source']} — {word} {', '.join(str(p) for p in pages)}")


def render_timing(timing: dict) -> None:
    """Per-turn, collapsed-by-default performance breakdown - opt-in detail,
    not a persistent log: nothing here renders unless the user expands it.
    """
    with st.expander("⏱️ Performance"):
        retrieval_ms = sum(
            timing.get(k, 0.0)
            for k in ("embedding_ms", "dense_ms", "keyword_ms", "fusion_ms")
        )
        st.caption(f"Retrieval: {retrieval_ms:.0f} ms total")
        st.caption(
            f"  • query embedding: {timing.get('embedding_ms', 0.0):.1f} ms  ·  "
            f"dense (FAISS): {timing.get('dense_ms', 0.0):.1f} ms  ·  "
            f"keyword (FTS5): {timing.get('keyword_ms', 0.0):.1f} ms  ·  "
            f"fusion: {timing.get('fusion_ms', 0.0):.1f} ms"
        )
        st.caption(f"Prompt construction: {timing.get('prompt_build_ms', 0.0):.1f} ms")

        gen = timing.get("generation", {})
        if gen:
            st.caption(
                f"Ollama - model load: {gen.get('load_ms', 0.0):.0f} ms  ·  "
                f"prompt eval: {gen.get('prompt_eval_ms', 0.0):.0f} ms  ·  "
                f"generation: {gen.get('eval_ms', 0.0):.0f} ms"
            )
            if gen.get("completion_tokens"):
                st.caption(f"  ({gen['completion_tokens']} tokens generated)")

        if timing.get("time_to_first_token_ms") is not None:
            st.caption(f"Time to first token: {timing['time_to_first_token_ms']:.0f} ms")

        st.caption(f"**Total end-to-end: {timing.get('total_ms', 0.0):.0f} ms**")


def _render_evidence_chunks(chunks: list[dict]) -> None:
    for i, chunk in enumerate(chunks, start=1):
        st.markdown(f"**[{chunk['rank']}] {chunk['source']} - page {chunk['page']}**")
        st.text(chunk["text"])
        if i < len(chunks):
            st.divider()


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
    with st.expander(f"Retrieved evidence ({len(chunks)} chunks)"):
        distinct_sources = list(dict.fromkeys(c["source"] for c in chunks))
        if len(distinct_sources) <= 1:
            _render_evidence_chunks(chunks)
            return

        for i, source in enumerate(distinct_sources):
            if i > 0:
                st.divider()
            st.markdown(f"**{source}**")
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
            "sidebar. It is parsed, chunked and indexed locally on this machine - "
            "nothing is uploaded anywhere.</p>"
            "<p>Once it is indexed you can ask questions about it, and every answer "
            "will cite the pages it came from.</p>"
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


def render_no_evidence(scope_label: str) -> None:
    """Shown when retrieval finds nothing. Deliberately not an error: the
    system has no evidence either way, which is not the same as the fact
    being false, and saying so is the honest thing for a clinical tool."""
    st.warning(
        f"**No relevant evidence found in {scope_label}.**\n\n"
        "This does not mean the answer is no - only that nothing in the "
        "documents currently in scope matched the question. Try rephrasing it, "
        "or widen the scope in the sidebar."
    )


def render_scope_banner(headline: str, names: list[str]) -> None:
    """A compact, always-visible statement of exactly which documents this
    turn can draw on - the single most important thing to be unambiguous
    about in a tool whose whole promise is document isolation."""
    chips = "".join(f'<span class="scope-chip">{escape(n)}</span>' for n in names)
    st.markdown(
        f'<div class="scope-banner"><div class="scope-headline">{escape(headline)}</div>'
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
            f"Ask about {names[0]}..."
            if len(names) == 1
            else f"Ask about these {len(names)} documents..."
        ) if scope_ready else "Tick at least one document in the sidebar..."
    else:
        scope_ready = scope_document_id is not None
        active_doc = next((d for d in documents if d["document_id"] == scope_document_id), None)
        scope_label = active_doc["source"] if active_doc else "the selected document"
        if scope_ready and active_doc:
            render_scope_banner("Analysing one document", [active_doc["source"]])
        placeholder = (
            f"Ask about {active_doc['source']}..." if active_doc else "Ask a question..."
        )

    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            turn_timing = turn.get("timing") or {}
            if turn["role"] == "assistant" and turn_timing.get("total_ms") is not None:
                # Mirrors the live "Complete · X.Xs" caption shown right after
                # this turn finished generating, so it stays visible after a
                # later turn causes a rerun - not just for the instant it
                # first appeared.
                st.caption(f"Complete · {turn_timing['total_ms'] / 1000:.1f}s")
            if turn["role"] == "assistant" and turn.get("sources"):
                render_sources(turn["sources"])
            if turn["role"] == "assistant" and turn.get("chunks"):
                render_evidence(turn["chunks"])
            if turn["role"] == "assistant" and turn_timing:
                render_timing(turn_timing)

    if not scope_ready:
        render_empty_state(has_documents=bool(documents))
        return

    query = st.chat_input(placeholder)
    if not query:
        return

    # Snapshot prior turns for generation context BEFORE appending this new
    # user turn - history is for display and (below) generation context
    # only; retrieve_and_build_prompt() below is only ever given the clean
    # `query` string, never this history, so retrieval matching is never
    # blurred by earlier conversation.
    prior_history = list(st.session_state["chat_history"])

    st.session_state["chat_history"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    top_k = resolve_retrieval_top_k(analysis_scope, selected_document_ids)
    num_predict = resolve_num_predict(query, prior_history, document_inventory)

    with st.chat_message("assistant"):
        timing: dict = {}
        turn_start = time.perf_counter()

        # A single status line, truthful about which stage is actually
        # running. Retrieval (FAISS + FTS5) is ~40-50ms - the real wait on
        # this hardware is Ollama generation, which used to run silently
        # under a spinner still reading "Searching your documents...". The
        # message is updated in place, then cleared the instant the first
        # real token streams in (see on_first_token below), so nothing
        # lingers above the answer once there is something to look at.
        stage = st.empty()
        stage.markdown("🔎 *Retrieving relevant evidence...*")
        try:
            chunks, prompt = retrieve_and_build_prompt(
                query,
                top_k=top_k,
                document_id=scope_document_id,
                timing=timing,
                document_inventory=document_inventory,
                history=prior_history,
            )
        except InsufficientRetrievalError:
            # Not an error state: nothing matched, which is a real and
            # honest answer for a clinical tool, not a failure.
            stage.empty()
            render_no_evidence(scope_label)
            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": f"No relevant evidence was found in {scope_label} for that question.",
                }
            )
            return
        except Exception as e:
            stage.empty()
            message = friendly_error_message(e)
            st.error(message)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": f"Error: {message}"}
            )
            return

        stage.markdown("🧠 *Generating answer...*")

        gen_result: dict = {}
        try:
            answer = st.write_stream(
                _stream_with_ttft(
                    prompt, gen_result, timing, num_predict=num_predict, on_first_token=stage.empty
                )
            )
        except Exception as e:
            stage.empty()
            message = friendly_error_message(e)
            st.error(message)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": f"Error: {message}"}
            )
            return

        # Prefer the out-param's stripped text (matches generate()'s
        # normalization exactly); st.write_stream's return value is an
        # equivalent fallback if generation completed but somehow didn't
        # populate the out-param.
        answer = gen_result.get("answer", answer)
        timing["generation"] = gen_result.get("timing", {})
        timing["total_ms"] = (time.perf_counter() - turn_start) * 1000

        st.caption(f"Complete · {timing['total_ms'] / 1000:.1f}s")

        sources = [{"source": c["source"], "page": c["page"], "score": c["score"]} for c in chunks]
        render_sources(sources)
        render_evidence(chunks)
        render_timing(timing)

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "chunks": chunks,
            "timing": timing,
        }
    )


def main() -> None:
    st.set_page_config(page_title="RAG for Healthcare Documents", page_icon="🩺", layout="wide")

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

    st.title("🩺 RAG for Healthcare Documents")
    st.markdown(f'<p class="app-subtitle">{escape(TAGLINE)}</p>', unsafe_allow_html=True)

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
        st.session_state["model_warmed"] = True

    with st.sidebar:
        render_sidebar(documents, active_id)

    render_chat(
        documents,
        active_id,
        st.session_state.get("analysis_scope", "active"),
        st.session_state.get("selected_document_ids", set()),
    )


if __name__ == "__main__":
    main()
