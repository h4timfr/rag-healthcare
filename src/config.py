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

# --- Ollama generation ---
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

OLLAMA_TEMPERATURE = 0.1
OLLAMA_TOP_P = 0.9
OLLAMA_NUM_PREDICT = 256
OLLAMA_KEEP_ALIVE = "10m"