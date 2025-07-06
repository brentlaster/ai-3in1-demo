#!/usr/bin/env python3
# search.py — colourised, similarity-aware search for a ChromaDB collection
#
#  This is the *query‐side* companion to the two indexing scripts:
#     • index_pdfs.py     – embeds every non-blank PDF line
#     • index_py_files_st.py – embeds every chunk of Python code
#
#  The scripts all share **one** Chroma collection called “codebase”, so the
#  vectors live in the *same* semantic space (MiniLM-L6-v2).  That means the
#  user can type *any* question and retrieve the most relevant text line or
#  code chunk across docs + code.
#
#  ————————————————————————————————————————————————————————————————
#  HOW IT WORKS
#  ————————————————————————————————————————————————————————————————
#   1. Load the same Sentence-Transformers model used for indexing.
#   2. Embed the user’s free-text query into a 384-d vector.
#   3. Ask Chroma for the TOP_K nearest neighbours (Approximate NN).
#   4. Re-compute the **exact** cosine similarity for each hit so we can
#      display the real score (Chroma itself only guarantees an ANN rank).
#   5. Print the best hit in green, the others in blue, and each cosine
#      score in red (ANSI escape sequences).
#
#  Quick demo   ----------------------------------------------------
#   $ python search.py
#   🔍 Search: paris office marketing
#     → top RAG lines about the Paris office, highest score in green.
#
#  Dependencies
#  ---------------------------------------------------------------
#   pip install sentence-transformers chromadb numpy
#   (make sure ./chroma_db exists and is populated first)
#

# ─────────────────────────────────────────────────────────────────────────
#  Standard + 3rd-party imports
# ─────────────────────────────────────────────────────────────────────────
import numpy as np                                     # fast vector math
from sentence_transformers import SentenceTransformer  # local embedding
from chromadb import PersistentClient                  # disk-backed store
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ─────────────────────────────────────────────────────────────────────────
#  1.  ANSI colour codes — purely cosmetic
#      (works on most terminals; ignored by e.g. Windows cmd.exe <10)
# ─────────────────────────────────────────────────────────────────────────
GREEN = "\033[92m"   # colour for the single best match
BLUE  = "\033[94m"   # colour for the “runner-up” matches
RED   = "\033[91m"   # colour for the numeric similarity
RESET = "\033[0m"    # reset to default terminal colours

# ─────────────────────────────────────────────────────────────────────────
#  2.  Connect to the *persistent* Chroma database
#      • path='./chroma_db' must exist (created by the indexers)
#      • DEFAULT_TENANT / DATABASE are fine for local dev
# ─────────────────────────────────────────────────────────────────────────
db_client = PersistentClient(
    path="./chroma_db",
    settings=Settings(),             # default index + search engine params
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Use the *exact same* embedding model as the indexers, otherwise the
# cosine similarity between query-vector and stored-vector is meaningless.
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────────────────────────────────
#  3.  Tiny cosine-similarity helper (manual recompute for transparency)
# ─────────────────────────────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity ∈ [-1, 1]; higher = more similar.
    Small epsilon avoids division-by-zero when vectors accidentally
    have zero norm (shouldn’t happen with MiniLM embeddings).
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# ─────────────────────────────────────────────────────────────────────────
#  4.  Core search function: embed → nearest neighbours → print
# ─────────────────────────────────────────────────────────────────────────
def search(query: str, top_k: int = 5) -> None:
    """
    Search the “codebase” collection for the `top_k` most similar chunks.

    Parameters
    ----------
    query  : str  — natural language (or code) search string.
    top_k  : int  — how many results to show (default 5).
    """
    # Retrieve (or lazily create) the Chroma collection object.
    coll = db_client.get_or_create_collection(name="codebase")

    # If the DB is empty the user probably forgot to run the indexer.
    docs = coll.get()                             # fetch minimal metadata
    total_chunks = len(docs.get("documents", []))
    if total_chunks == 0:
        print("Collection is empty — nothing to search.")
        return
    print(f"Collection contains {total_chunks} chunks.")

    # Embed the query once — MiniLM returns a numpy array we use later.
    query_vec = embed_model.encode(query)

    # ANN search (Approximate Nearest Neighbour) inside Chroma
    results = coll.query(
        query_embeddings=[query_vec.tolist()],    # list[vector]
        n_results=top_k,
        include=["documents", "metadatas", "embeddings"],  # need raw vectors
    )

    retrieved_docs   = results["documents"][0]    # list[str]
    retrieved_meta   = results["metadatas"][0]    # list[dict]
    retrieved_embeds = results["embeddings"][0]   # list[list[float]]

    if not retrieved_docs:
        print("No matches found.")
        return

    # Compute *exact* cosine sim for human-readable scores
    similarities = [
        cosine_sim(query_vec, np.array(emb))
        for emb in retrieved_embeds
    ]
    best_idx = int(np.argmax(similarities))       # index of top score

    # Pretty-print each hit: chunk text + similarity + source path
    for i, (doc, meta, sim) in enumerate(
        zip(retrieved_docs, retrieved_meta, similarities)
    ):
        colour = GREEN if i == best_idx else BLUE
        print(
            f"{colour}{doc}{RESET}  "
            f"{RED}{sim:.4f}{RESET}  "
            f"({meta['path']}  chunk {meta['chunk_index']})\n"
        )

# ─────────────────────────────────────────────────────────────────────────
#  5.  Simple command-line REPL
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Enter your search query (type 'exit' to quit):")
    while True:
        user_input = input("🔍 Search: ").strip()
        if user_input.lower() == "exit":
            print("Exiting search.")
            break
        if user_input:
            search(user_input)
        else:
            print("Please enter a valid query.")
