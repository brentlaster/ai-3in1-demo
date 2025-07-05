#!/usr/bin/env python3
# search.py — colourised, similarity-aware search against a ChromaDB collection
#
# This script:
#   1. Connects to a *persistent* Chroma vector database stored in ./chroma_db.
#   2. Embeds any user-supplied query with the same SBERT model used for indexing.
#   3. Retrieves the top-k most similar chunks from the “codebase” collection.
#   4. Computes cosine similarity between the query embedding and each returned
#      chunk embedding (Chroma already performs an ANN search, but we recompute
#      the exact cosine score here for transparency).
#   5. Displays the highest-scoring chunk in **green**, the other results in
#      **blue**, and each similarity score in **red**.
#
# Requirements:
#   pip install sentence-transformers chromadb numpy
#   A populated ./chroma_db created with the complementary indexing script.

import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ─────────────────────────── ANSI colour codes ────────────────────────────
GREEN = "\033[92m"   # best match
BLUE  = "\033[94m"   # other matches
RED   = "\033[91m"   # similarity score
RESET = "\033[0m"    # reset to default terminal colour

# ────────────────────────── ChromaDB connection ──────────────────────────
# PersistentClient keeps both settings and data on disk so multiple runs of
# this script (or other scripts) share the same collection and embeddings.
db_client = PersistentClient(
    path="./chroma_db",            # folder containing the Chroma database
    settings=Settings(),           # default engine settings
    tenant=DEFAULT_TENANT,         # default tenant & DB names are fine
    database=DEFAULT_DATABASE,
)

# The *same* embedding model used during indexing must be used here, or the
# query vectors will live in a different embedding space.
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ───────────────────────── cosine-similarity helper ───────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors."""
    # Add a small epsilon (1e-10) to avoid division by zero.
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# ────────────────────────────── search core ──────────────────────────────
def search(query: str, top_k: int = 5) -> None:
    """
    Embed the query, retrieve the top_k closest chunks, print them with colour.

    Args:
        query:  Natural-language query string entered by the user.
        top_k:  Number of hits to display.  Default is 5.
    """
    coll = db_client.get_or_create_collection(name="codebase")

    # Quick check so we fail early if the DB hasn’t been populated yet.
    docs = coll.get()
    total_chunks = len(docs.get("documents", []))
    if total_chunks == 0:
        print("Collection is empty — nothing to search.")
        return
    print(f"Collection contains {total_chunks} chunks.")

    # Produce an embedding for the query in the same vector space.
    query_embedding = embed_model.encode(query)

    # Retrieve top_k nearest neighbours (ANN). We also request each chunk’s
    # raw embedding so we can compute exact cosine similarity for display.
    results = coll.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "embeddings"],
    )

    retrieved_docs   = results["documents"][0]
    retrieved_meta   = results["metadatas"][0]
    retrieved_embeds = results["embeddings"][0]

    if not retrieved_docs:
        print("No matches found.")
        return

    # Compute cosine similarity between the query and every returned chunk.
    similarities = [
        cosine_sim(query_embedding, np.array(emb))
        for emb in retrieved_embeds
    ]
    best_idx = int(np.argmax(similarities))  # index of highest score

    # Pretty-print the results with colours.
    for i, (doc, meta, sim) in enumerate(zip(retrieved_docs,
                                             retrieved_meta,
                                             similarities)):
        colour = GREEN if i == best_idx else BLUE
        print(
            f"{colour}{doc}{RESET}  "
            f"{RED}{sim:.4f}{RESET}  "
            f"({meta['path']}  chunk {meta['chunk_index']})\n"
        )

# ─────────────────────────── interactive loop ────────────────────────────
if __name__ == "__main__":
    print("Enter your search query (type 'exit' to quit):")
    while True:
        user_input = input("🔍 Search: ").strip()
        if user_input.lower() == "exit":
            print("Exiting search.")
            break
        elif user_input:
            search(user_input)
        else:
            print("Please enter a valid query.")
