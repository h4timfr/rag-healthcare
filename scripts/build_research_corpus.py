#!/usr/bin/env python
"""Builds the separate research healthcare corpus (Phase 2 of the evaluation
audit remediation) into its own FAISS index / SQLite DB at
data/research_corpus/{metadata.db,faiss.index} - entirely separate from the
production data/faiss.index + data/metadata.db used by the live Streamlit app
and the dev-regression benchmark, so the two never contaminate each other.

Ingests, via the existing (unmodified) ingestion pipeline:
  - The freshly downloaded public-domain documents under
    data/research_corpus/raw/ (see data/evaluation_corpus.json for
    provenance of each, written by this script).
  - "OMC Report Sample - Cardio.pdf"'s existing chunk text, migrated from the
    production index. Its original uploaded PDF file is not retained on disk
    by design (app.py deletes the temp upload after ingestion), so instead of
    re-extracting from a PDF, this script reads the already-extracted chunk
    text straight from production data/metadata.db and re-embeds it with the
    same embed_texts() used for every other document (MiniLM-L6-v2 is
    deterministic at inference, so this reproduces the same vectors that
    would result from re-ingesting the original file).

Usage:
    python scripts/build_research_corpus.py
"""

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.ingest import ingest_pdf
from src import config
from src.embedding.embedder import embed_texts
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.metadata_store import MetadataStore

RESEARCH_DIR = ROOT / "data" / "research_corpus"
RAW_DIR = RESEARCH_DIR / "raw"
RESEARCH_DB_PATH = str(RESEARCH_DIR / "metadata.db")
RESEARCH_FAISS_PATH = str(RESEARCH_DIR / "faiss.index")
MANIFEST_PATH = ROOT / "data" / "evaluation_corpus.json"

# (filename, title, document_type, publisher, source_url)
NEW_DOCUMENTS = [
    (
        "cdc_diabetes_statistics_2020.pdf",
        "National Diabetes Statistics Report, 2020",
        "public-health statistics report",
        "Centers for Disease Control and Prevention (CDC) - U.S. government work, public domain",
        "https://stacks.cdc.gov/view/cdc/85309/cdc_85309_DS1.pdf",
    ),
    (
        "nhlbi_dash_blood_pressure_guide.pdf",
        "Your Guide to Lowering Your Blood Pressure With DASH",
        "patient/clinical guideline",
        "National Institutes of Health, National Heart, Lung, and Blood Institute (NHLBI) - "
        "U.S. government work, public domain",
        "https://www.nhlbi.nih.gov/files/docs/public/heart/new_dash.pdf",
    ),
    (
        "cdc_dental_infection_control_2003.pdf",
        "CDC's Core Infection Prevention and Control Practices for Safe Healthcare Delivery in All Settings",
        "clinical/infection-control guideline",
        "Centers for Disease Control and Prevention (CDC) - U.S. government work, public domain",
        "https://stacks.cdc.gov/view/cdc/134781/cdc_134781_DS1.pdf",
    ),
    (
        "cdc_ckd_fact_sheet_2014.pdf",
        "National Chronic Kidney Disease Fact Sheet, 2014",
        "public-health fact sheet",
        "Centers for Disease Control and Prevention (CDC) - U.S. government work, public domain",
        "https://stacks.cdc.gov/view/cdc/21850/cdc_21850_DS1.pdf",
    ),
]

OMC_SOURCE_NAME = "OMC Report Sample - Cardio.pdf"


def migrate_omc_from_production() -> dict:
    """Copy OMC's already-extracted chunk text from the production DB into
    the research index, re-embedding with the same embed_texts() used
    everywhere else (see module docstring for why re-embedding, not FAISS
    vector reconstruction).
    """
    prod_conn = sqlite3.connect(config.DB_PATH)
    prod_conn.row_factory = sqlite3.Row
    rows = prod_conn.execute(
        "SELECT document_id, source, page, text FROM chunks WHERE source = ? ORDER BY chunk_id",
        (OMC_SOURCE_NAME,),
    ).fetchall()
    prod_conn.close()

    if not rows:
        raise RuntimeError(
            f"No chunks found for source='{OMC_SOURCE_NAME}' in production DB "
            f"({config.DB_PATH}). Was it removed from the production index?"
        )

    document_id = rows[0]["document_id"]

    research_store = MetadataStore(RESEARCH_DB_PATH)
    research_store.initialize()

    if research_store.document_chunk_count(document_id) > 0:
        print(f"Skipped migrating '{OMC_SOURCE_NAME}': already present in research corpus")
        return {"document_id": document_id, "chunk_count": research_store.document_chunk_count(document_id),
                 "skipped": True, "page_count": max(r["page"] for r in rows)}

    texts = [r["text"] for r in rows]
    vectors = embed_texts(texts)

    research_index = FaissIndex(config.EMBEDDING_DIM, RESEARCH_FAISS_PATH)
    research_index.load_or_create()

    chunk_ids = [
        research_store.insert_chunk(
            document_id=document_id, source=r["source"], page=r["page"], text=r["text"]
        )
        for r in rows
    ]
    research_index.add(vectors, np.array(chunk_ids, dtype="int64"))
    research_index.save()

    print(f"Migrated '{OMC_SOURCE_NAME}': {len(rows)} chunks (document_id={document_id})")
    return {
        "document_id": document_id,
        "chunk_count": len(rows),
        "skipped": False,
        "page_count": max(r["page"] for r in rows),
    }


def main() -> None:
    manifest = []

    for filename, title, doc_type, publisher, url in NEW_DOCUMENTS:
        pdf_path = RAW_DIR / filename
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"Expected downloaded file not found: {pdf_path}. "
                f"Download it from {url} first."
            )

        raw_bytes = pdf_path.read_bytes()
        expected_document_id = hashlib.sha256(raw_bytes).hexdigest()[:16]

        result = ingest_pdf(str(pdf_path), db_path=RESEARCH_DB_PATH, faiss_index_path=RESEARCH_FAISS_PATH)
        assert result["document_id"] == expected_document_id

        import pymupdf
        page_count = pymupdf.open(str(pdf_path)).page_count

        manifest.append(
            {
                "filename": filename,
                "source_url": url,
                "title": title,
                "document_type": doc_type,
                "publisher": publisher,
                "document_id": result["document_id"],
                "page_count": page_count,
                "chunk_count": result["chunk_count"],
            }
        )

    omc_result = migrate_omc_from_production()
    manifest.append(
        {
            "filename": None,
            "source_url": None,
            "title": OMC_SOURCE_NAME,
            "document_type": "clinical/medical report",
            "publisher": "user-provided sample document, previously ingested into the production "
                         "demo index; original PDF file not retained on disk (app.py discards the "
                         "temp upload after ingestion), so its content was migrated from the "
                         "production index's already-extracted chunk text rather than re-extracted "
                         "from a file",
            "document_id": omc_result["document_id"],
            "page_count": omc_result["page_count"],
            "chunk_count": omc_result["chunk_count"],
        }
    )

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {MANIFEST_PATH}")
    print(f"Research index: {RESEARCH_DB_PATH}, {RESEARCH_FAISS_PATH}")
    total_chunks = sum(m["chunk_count"] for m in manifest)
    print(f"Total documents: {len(manifest)}, total chunks: {total_chunks}")


if __name__ == "__main__":
    main()
