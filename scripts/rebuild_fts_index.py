#!/usr/bin/env python
"""CLI: rebuild the FTS5 keyword index from the existing chunks table.

Run this once after changing chunks_fts's own definition in
src/retrieval/metadata_store.py (e.g. its tokenizer) - a database that was
initialized under the old definition keeps the old chunks_fts table
otherwise, since initialize() uses CREATE VIRTUAL TABLE IF NOT EXISTS and
will not touch a table that already exists.

Does not touch the chunks table or the FAISS index - only chunks_fts is
dropped and repopulated, from data that is already ingested.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.retrieval.metadata_store import MetadataStore


def main():
    parser = argparse.ArgumentParser(description="Rebuild the FTS5 keyword index from the chunks table.")
    parser.add_argument("--db", default=config.DB_PATH, help="Path to the metadata SQLite database.")
    args = parser.parse_args()

    store = MetadataStore(args.db)
    before = store.count()
    store.rebuild_fts_index()
    after = store.count()

    print(f"Rebuilt chunks_fts at '{args.db}': {after} chunks indexed (chunks table unchanged, {before} -> {after}).")


if __name__ == "__main__":
    main()
