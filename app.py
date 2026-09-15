"""Streamlit dashboard for the RAG for Healthcare Documents pipeline.

This file is UI glue only: it saves an uploaded PDF, hands it to the existing
ingestion pipeline, and calls the existing hybrid retriever / prompt builder /
Ollama client for querying. No extraction, chunking, embedding, FAISS, FTS5,
prompt, or generation logic is reimplemented here.
"""

import shutil
import sys
import tempfile
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
from src.generation.ollama_client import generate, get_status as get_ollama_status
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


def answer_question(query: str, top_k: int = config.TOP_K, document_id: str | None = None) -> dict:
    """Hybrid retrieval -> prompt construction -> Ollama generation.

    Mirrors scripts/query.py's answer_query(), extended to also return the
    full retrieved chunks (with text) so the UI can show the evidence, and to
    accept an optional document_id so retrieval can be scoped to the active
    document instead of the whole corpus.
    """
    try:
        chunks = retrieve_hybrid(query, top_k=top_k, document_id=document_id)
    except ValueError as e:
        raise EmptyQueryError(str(e)) from e

    if not chunks:
        raise InsufficientRetrievalError("No relevant chunks found for this query")

    prompt = build_prompt(query, chunks)
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

    if len(documents) > 1:
        st.divider()
        st.checkbox(
            "Ask across all documents",
            key="cross_document_mode",
            help="When off (default), questions are answered using only the active document.",
        )

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


def render_evidence(chunks: list[dict]) -> None:
    with st.expander(f"Retrieved evidence ({len(chunks)} chunks)"):
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(f"**[{i}] {chunk['source']} - page {chunk['page']}**")
            st.text(chunk["text"])
            if i < len(chunks):
                st.divider()


def render_empty_state() -> None:
    st.info(
        "**No document selected yet.**\n\n"
        "Upload a healthcare PDF from the sidebar to build a searchable "
        "library, then select it to start asking questions."
    )


def render_chat(active_document_id: str | None, documents: list[dict], cross_document: bool) -> None:
    if cross_document:
        st.info(f"Chatting across **all {len(documents)} documents**")
    elif active_document_id:
        active_doc = next((d for d in documents if d["document_id"] == active_document_id), None)
        if active_doc:
            st.info(f"Chatting with: **{active_doc['source']}**")

    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn["role"] == "assistant" and turn.get("sources"):
                render_sources(turn["sources"])
            if turn["role"] == "assistant" and turn.get("chunks"):
                render_evidence(turn["chunks"])

    if not active_document_id and not cross_document:
        render_empty_state()
        return

    query = st.chat_input("Ask a question about the active document...")
    if not query:
        return

    st.session_state["chat_history"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating an answer..."):
            try:
                result = answer_question(
                    query, document_id=None if cross_document else active_document_id
                )
            except Exception as e:
                message = friendly_error_message(e)
                st.error(message)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": f"Error: {message}"}
                )
                return

        st.write(result["answer"])
        render_sources(result["sources"])
        render_evidence(result["chunks"])

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "chunks": result["chunks"],
        }
    )


def main() -> None:
    st.set_page_config(page_title="RAG for Healthcare Documents", page_icon="🩺", layout="wide")

    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("active_document_id", None)
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("cross_document_mode", False)

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

    with st.sidebar:
        render_sidebar(documents, active_id)

    cross_document = bool(documents) and len(documents) > 1 and st.session_state.get("cross_document_mode", False)
    render_chat(active_id, documents, cross_document)


if __name__ == "__main__":
    main()
