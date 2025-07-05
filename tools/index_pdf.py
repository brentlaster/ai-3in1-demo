#!/usr/bin/env python3
# index_pdfs.py — line-level PDF indexing into a fresh ChromaDB
#
# Workflow
# ──────────────────────────────────────────────────────────────────────
#   1. Remove any existing ./chroma_db so each run starts with a clean DB.
#   2. Find every *.pdf inside ./data.
#   3. Extract non-blank lines from each page using pdfplumber.
#   4. Embed every line with the “all-MiniLM-L6-v2” Sentence-BERT model.
#   5. Store the embeddings, the raw line text, and metadata (path + line
#      index) in a persistent Chroma collection called “codebase”.
#
# After the script finishes you will have a brand-new vector database at
# ./chroma_db that can be queried by the companion search script.

import shutil
import re
from pathlib import Path
from typing import List

import pdfplumber                              # pip install pdfplumber
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ───────────────────── user-configurable constants ──────────────────────
PDF_DIR          = Path("./data")          # Folder containing the PDFs
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"      # Local or HF Hub model name
CHROMA_PATH      = Path("./chroma_db")     # Vector DB folder (will be wiped)
COLLECTION_NAME  = "codebase"              # Logical collection name

# ───────────────────────── regex helper ─────────────────────────────────
# Splits on any newline while trimming leading/trailing whitespace from
# each resulting fragment. Empty strings are discarded.
LINE_RE = re.compile(r"[^\S\r\n]*\r?\n[^\S\r\n]*")

def extract_lines(path: Path) -> List[str]:
    """
    Extract every non-blank line from all pages in a PDF.
    Returns:
        List[str]: ordered list of text lines (page order preserved).
    """
    lines: List[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            for line in LINE_RE.split(txt):
                cleaned = line.strip()
                if cleaned:
                    lines.append(cleaned)
    return lines

def reset_chroma(db_path: Path) -> None:
    """
    Delete the existing ChromaDB folder so each run starts from scratch.
    """
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── main routine ───────────────────────────────
def index_pdfs() -> None:
    """
    Index every PDF in PDF_DIR into a fresh ChromaDB collection.
    """
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR.resolve()}")
        return

    print(f"Embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 1. Fresh database every run
    reset_chroma(CHROMA_PATH)

    # 2. Connect to Chroma (persistent on disk)
    client = PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    coll = client.get_or_create_collection(COLLECTION_NAME)

    # 3. Walk through each PDF and embed every line
    for pdf_path in pdf_files:
        print(f"-> Indexing {pdf_path.name}")
        try:
            lines = extract_lines(pdf_path)
        except Exception as e:
            print(f"[WARN] Could not read {pdf_path}: {e}")
            continue

        for idx, line in enumerate(lines):
            # Convert the line into a vector
            emb = embed_model.encode(line).tolist()

            # Store vector + metadata + raw text
            coll.add(
                ids=[f"{pdf_path}-{idx}"],
                embeddings=[emb],
                metadatas=[{"path": str(pdf_path), "chunk_index": idx}],
                documents=[line],
            )

    print("Indexing complete — new DB stored in ./chroma_db")

# ─────────────────────────── entry point ────────────────────────────────
if __name__ == "__main__":
    index_pdfs()
