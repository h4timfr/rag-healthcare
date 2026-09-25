"""Central configuration for the RAG engine.

All tunable parameters (chunking, retrieval top-k, model names, storage paths)
live here so behavior can be adjusted without touching module internals.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Storage locations ---
PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
DB_PATH = str(PROJECT_ROOT / "data" / "metadata.db")
FAISS_INDEX_PATH = str(PROJECT_ROOT / "data" / "faiss.index")

# --- Embedding model ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Chunking (word-based, not token-based, to avoid an extra tokenizer dependency) ---
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40

# --- Retrieval ---
TOP_K = 5

# Per-document chunk budget for each-document/comparison questions, where
# retrieval balances its allocation across the selected documents (see
# hybrid_retriever._allocate_by_document).
#
# Lower than TOP_K on purpose. The unbalanced multi-document budget is
# TOP_K * number_of_documents, and it has to be that large because nothing
# guarantees the documents are represented - it buys coverage with volume,
# and often still misses. Balanced allocation guarantees the coverage
# directly, so the same answer needs fewer chunks: for three documents this
# is 9 chunks instead of 15, which measurably shortens a prompt that is
# otherwise dominated by whichever document happens to have the longest
# chunks.
BALANCED_CHUNKS_PER_DOCUMENT = 3

# Hard ceiling on how many chunks any single answer is built from.
#
# llama3.2:3b stops reliably using evidence it has been given somewhere
# above this. Measured on a two-part question whose answer sat in the
# HIGHEST-ranked chunk of the prompt: with three documents selected it was
# answered correctly at 5, 7, 9 and 12 chunks, and failed at 13, 14 and 15
# (18.5KB and up), claiming the evidence did not contain a fact that was in
# its own context. The failure was deterministic across repeated runs.
#
# 10 leaves margin below that boundary, sits above the single-document
# TOP_K, and does not change the current balanced allocation (3 documents x
# 3 = 9). It exists mainly so the budget cannot grow without limit as more
# documents are selected - ten selected documents must not ask for fifty
# chunks.
MAX_CONTEXT_CHUNKS = 10

# --- Ollama generation ---
# 127.0.0.1, not "localhost": Ollama listens on IPv4 127.0.0.1 by default,
# while Windows resolves "localhost" to IPv6 ::1 first. Each request then
# waits for the IPv6 attempt to fail before falling back - measured at
# ~2000 ms per request versus ~30 ms direct, twice per answer (availability
# check + generation) and once per page render (sidebar status).
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"

OLLAMA_TEMPERATURE = 0.1
OLLAMA_TOP_P = 0.9
OLLAMA_NUM_PREDICT = 256
# Multi-document summaries/comparisons cover several documents' evidence in
# one answer and were getting cut off at the single-document budget above -
# used only for the Streamlit "Selected documents" scope (see app.py); every
# single-document answer keeps the default OLLAMA_NUM_PREDICT unchanged.
OLLAMA_NUM_PREDICT_MULTI_DOCUMENT = 768
OLLAMA_KEEP_ALIVE = "10m"