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
from src.generation.intent import needs_document_balance
from src.generation.ollama_client import (
    generate,
    generate_stream,
    get_status as get_ollama_status,
    warm_up as warm_up_ollama,
)
from src.generation.prompt_builder import build_prompt
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.hybrid_retriever import retrieve_hybrid
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
        return ingest_module.ingest_pdf(str(tmp_path))
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


def resolve_retrieval_top_k(analysis_scope: str, selected_document_ids) -> int:
    """How many chunks retrieve_hybrid() should keep from its fused ranking
    for this turn.

    Active-document scope always keeps exactly config.TOP_K - single-document
    retrieval behavior is unchanged. "Selected documents" scope widens it to
    TOP_K * (number of selected documents), so a multi-document
    summary/comparison has a real chance of surfacing evidence from every
    selected document instead of a flat top-TOP_K list being dominated by
    whichever one or two documents rank highest overall. FAISS, FTS5, RRF
    weights, and candidate_k are all untouched by this - it only changes how
    many results are kept from the very end of the existing fused ranking.
    """
    if analysis_scope == "selected" and selected_document_ids:
        return config.TOP_K * len(selected_document_ids)
    return config.TOP_K


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

    `document_inventory` (optional) is forwarded to build_prompt() as-is -
    see its docstring. It also marks this as a multi-document turn, which is
    the precondition for document-balanced retrieval below; scoring, the
    FAISS/FTS5 searches and the RRF fusion are unaffected either way.

    For an each-document or comparison question over several selected
    documents, retrieval is asked to balance its allocation across them and
    `top_k` is replaced by config.BALANCED_CHUNKS_PER_DOCUMENT per selected
    document. Every other question - including any single-document question -
    keeps the caller's `top_k` and the unmodified ranking.

    `history` (optional) is prior conversation turns, forwarded to
    build_prompt() for GENERATION context only - retrieve_hybrid() above
    only ever sees the clean `query` string, never history, so conversation
    context can never blur retrieval matching.

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

    # Document-balanced allocation applies only to a multi-document scope
    # asked an each-document/comparison question. Same conditional-kwarg
    # reason as `timing` above, and single-document retrieval never reaches
    # it at all (document_inventory is None in active-document scope).
    #
    # This also replaces the caller's `top_k` for that case, deliberately:
    # the balanced budget is a property of the retrieval strategy, not of
    # the call site, and deciding it here is what keeps the Streamlit app
    # and the product QA harness identical without either having to
    # remember to mirror the other.
    if document_inventory and needs_document_balance(query, history=history):
        retrieve_kwargs["balance_by_document"] = True
        retrieve_kwargs["top_k"] = config.BALANCED_CHUNKS_PER_DOCUMENT * len(document_inventory)

    try:
        chunks = retrieve_hybrid(query, **retrieve_kwargs)
    except ValueError as e:
        raise EmptyQueryError(str(e)) from e

    if not chunks:
        raise InsufficientRetrievalError("No relevant chunks found for this query")

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
    """
    chunks, prompt = retrieve_and_build_prompt(query, top_k=top_k, document_id=document_id)
    answer = generate(prompt)
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
.doc-card.active { border-color: var(--accent); background: var(--accent-soft); }
.doc-name { font-weight: 600; font-size: 0.92rem; overflow-wrap: anywhere; }
.doc-meta { color: var(--text-secondary); font-size: 0.78rem; margin-top: 0.1rem; }

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
                progress.caption(f"complete - {result['chunk_count']} chunks indexed.")
                if result.get("degraded_pages"):
                    st.warning(
                        f"Page(s) {result['degraded_pages']} of **{uploaded_file.name}** "
                        "extracted without word boundaries, so keyword search and answer "
                        "quality will be poor for this document. This is a property of "
                        "the PDF's text layer, not of your question."
                    )
            elif result["status"] == "skipped":
                progress.caption(f"skipped - already indexed ({result['chunk_count']} chunks).")
            else:
                progress.caption(f"failed - {result['error']}")

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
    if not documents:
        st.caption("No documents indexed yet. Upload a PDF above to begin.")
        return

    pending_delete = st.session_state.get("pending_delete_id")

    for doc in documents:
        doc_id = doc["document_id"]
        is_active = doc_id == active_id
        pages = doc.get("page_count")
        meta_bits = [f"{doc['chunk_count']} chunks"]
        if pages:
            meta_bits.append(f"{pages} pages")

        st.markdown(
            f'<div class="doc-card{" active" if is_active else ""}">'
            f'<div class="doc-name">{"● " if is_active else ""}{escape(doc["source"])}</div>'
            f'<div class="doc-meta">{" · ".join(meta_bits)}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        col_select, col_remove = st.columns(2)
        with col_select:
            st.button(
                "Active" if is_active else "Select",
                key=f"select_{doc_id}",
                use_container_width=True,
                disabled=is_active,
                on_click=_set_active_document,
                args=(doc_id,),
            )
        with col_remove:
            st.button(
                "Remove",
                key=f"remove_{doc_id}",
                use_container_width=True,
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

    st.divider()
    st.subheader("Analysis scope")
    st.radio(
        "Analysis scope",
        options=["active", "selected"],
        format_func=lambda v: "Active document" if v == "active" else "Selected documents",
        key="analysis_scope",
        label_visibility="collapsed",
    )

    if st.session_state["analysis_scope"] != "selected":
        return

    selected = st.session_state.setdefault("selected_document_ids", set())
    for doc in documents:
        doc_id = doc["document_id"]
        checked = st.checkbox(doc["source"], value=doc_id in selected, key=f"scope_select_{doc_id}")
        if checked:
            selected.add(doc_id)
        else:
            selected.discard(doc_id)

    if selected:
        st.caption(f"**{len(selected)} document(s) selected**")
    else:
        st.warning("Select at least one document to ask a question.")


def render_sidebar(documents: list[dict], active_id: str | None) -> None:
    render_theme_toggle()
    render_status()
    st.caption(f"{len(documents)} document(s) indexed")

    st.divider()
    st.subheader("Upload")
    render_upload_area()

    st.divider()
    st.subheader("Documents")
    render_document_library(documents, active_id)

    render_analysis_scope(documents)

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


def _stream_with_ttft(prompt: str, gen_result: dict, timing: dict, num_predict: int | None = None):
    """Wrap generate_stream() to also record time-to-first-token into
    `timing` (UI-only instrumentation; generate_stream itself stays
    display-agnostic and reusable from non-Streamlit callers).
    """
    start = time.perf_counter()
    first_token = True
    for token in generate_stream(prompt, result=gen_result, num_predict=num_predict):
        if first_token:
            timing["time_to_first_token_ms"] = (time.perf_counter() - start) * 1000
            first_token = False
        yield token


def render_empty_state() -> None:
    st.info(
        "**No document selected yet.**\n\n"
        "Upload a healthcare PDF from the sidebar to build a searchable "
        "library, then select it to start asking questions."
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
        if scope_ready:
            names = ", ".join(d["source"] for d in documents if d["document_id"] in selected_document_ids)
            st.info(f"Chatting with **{len(scope_document_id)} selected documents**: {names}")
        placeholder = (
            f"Ask a question about the {len(scope_document_id)} selected documents..."
            if scope_ready
            else "Select at least one document to begin..."
        )
    else:
        scope_ready = scope_document_id is not None
        if scope_ready:
            active_doc = next((d for d in documents if d["document_id"] == scope_document_id), None)
            if active_doc:
                st.info(f"Chatting with: **{active_doc['source']}**")
        placeholder = "Ask a question about the active document..."

    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn["role"] == "assistant" and turn.get("sources"):
                render_sources(turn["sources"])
            if turn["role"] == "assistant" and turn.get("chunks"):
                render_evidence(turn["chunks"])
            if turn["role"] == "assistant" and turn.get("timing"):
                render_timing(turn["timing"])

    if not scope_ready:
        if analysis_scope == "selected" and documents:
            st.info("Select at least one document in the sidebar to start asking questions.")
        else:
            render_empty_state()
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
    num_predict = config.OLLAMA_NUM_PREDICT_MULTI_DOCUMENT if document_inventory else None

    with st.chat_message("assistant"):
        timing: dict = {}
        turn_start = time.perf_counter()

        with st.spinner("Retrieving evidence..."):
            try:
                chunks, prompt = retrieve_and_build_prompt(
                    query,
                    top_k=top_k,
                    document_id=scope_document_id,
                    timing=timing,
                    document_inventory=document_inventory,
                    history=prior_history,
                )
            except Exception as e:
                message = friendly_error_message(e)
                st.error(message)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": f"Error: {message}"}
                )
                return

        gen_result: dict = {}
        try:
            answer = st.write_stream(_stream_with_ttft(prompt, gen_result, timing, num_predict=num_predict))
        except Exception as e:
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
