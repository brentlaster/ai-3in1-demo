#!/usr/bin/env python3
# index_py_files_st.py
#
# Build a fresh ChromaDB index of every Python source file in
#   workspaces/ai-3in1
# using Sentence-Transformers **all-MiniLM-L6-v2** embeddings
# (the same model used in the PDF indexer).
#
# Why this version?
# -----------------
# • A single embedding model for both PDFs and Python code keeps every
#   chunk in the *same* vector space—simplifies downstream search logic.
# • MiniLM runs quickly on CPU; no Ollama server or HTTP round-trip needed.
#
# Prerequisites
# -------------
#   pip install sentence-transformers chromadb tiktoken
#
# Result
# ------
# After running:
#   ./chroma_db     → newly created vector database on disk
#   collection      → logical name “codebase”
#   one vector      → one code chunk (≤500 GPT-3.5 tokens, line-aligned)

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, List

from sentence_transformers import SentenceTransformer
from tiktoken import encoding_for_model
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ------------------------------------------------------------------- #
# Configuration                                                       #
# ------------------------------------------------------------------- #

ROOT_DIR         = Path(".")   # repository root to scan
CHROMA_PATH      = Path("./chroma_db")          # vector DB location (fresh)
COLLECTION_NAME  = "codebase"                   # logical collection name
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"           # SBERT model
MAX_TOKENS       = 500                          # max GPT-3.5 tokens per chunk

# prune these folders to avoid noise / huge file counts
SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", "node_modules",
    ".venv", "venv", "env", "py_env", "site-packages",
}

# ------------------------------------------------------------------- #
# Helper: chunk Python code                                           #
# ------------------------------------------------------------------- #

def chunk_python_code(code: str, max_tokens: int = MAX_TOKENS) -> Iterable[str]:
    """
    Yield successive code chunks that respect line boundaries.

    * Never splits a physical line of code.
    * A chunk ends when:
        - Adding the next line would exceed `max_tokens`, OR
        - A completely blank line appears and the current chunk
          already has content (blank lines often separate blocks).
    """
    enc = encoding_for_model("gpt-3.5-turbo")

    current_lines: List[str] = []
    token_count = 0

    for line in code.splitlines():
        line_tokens = len(enc.encode(line + "\n"))

        # Hard limit on token budget
        if current_lines and token_count + line_tokens > max_tokens:
            yield "\n".join(current_lines)
            current_lines, token_count = [], 0

        # Optional soft break at blank lines
        if not line.strip() and current_lines:
            yield "\n".join(current_lines)
            current_lines, token_count = [], 0
            continue  # skip the blank line itself

        current_lines.append(line)
        token_count += line_tokens

    if current_lines:
        yield "\n".join(current_lines)

# ------------------------------------------------------------------- #
# Helper: wipe the on-disk ChromaDB                                   #
# ------------------------------------------------------------------- #

def reset_chroma(db_path: Path) -> None:
    """Delete any existing DB folder so each run starts clean."""
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------- #
# Main routine                                                        #
# ------------------------------------------------------------------- #

def index_python_sources() -> None:
    """
    Recursively embed every .py file under ROOT_DIR into a fresh ChromaDB.
    """
    if not ROOT_DIR.exists():
        print(f"[ERROR] {ROOT_DIR.resolve()} does not exist.")
        return

    print(f"Embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 1. Fresh database every run
    reset_chroma(CHROMA_PATH)

    # 2. Connect to persistent Chroma store
    client = PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    collection = client.get_or_create_collection(COLLECTION_NAME)

    file_counter = 0  # summary counter

    # 3. Walk directory tree, pruning unwanted folders
    for root, dirs, files in os.walk(ROOT_DIR):
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for name in files:
            if not name.endswith(".py"):
                continue

            file_path = Path(root) / name

            # Read file content
            try:
                code_text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as err:
                print(f"[WARN] Could not read {file_path}: {err}")
                continue

            # Chunk and embed
            for idx, chunk in enumerate(chunk_python_code(code_text)):
                vector = embed_model.encode(chunk).tolist()

                collection.add(
                    ids=[f"{file_path}-{idx}"],
                    embeddings=[vector],
                    documents=[chunk],
                    metadatas=[{"path": str(file_path), "chunk_index": idx}],
                )

            file_counter += 1
            print(f"Indexed {file_path}")

    # 4. Summary
    print(
        f"Indexing complete: {file_counter} Python files processed.\n"
        "Fresh vector DB saved to ./chroma_db"
    )

# ------------------------------------------------------------------- #
# Entry point                                                         #
# ------------------------------------------------------------------- #

if __name__ == "__main__":
    index_python_sources()
