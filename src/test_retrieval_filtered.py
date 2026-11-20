
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DENSE_DIR = ROOT / "data" / "indexes" / "fixed_size_dense_filtered_v1"
BM25_DIR = ROOT / "data" / "indexes" / "fixed_size_bm25_filtered_v1"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


# --------------------------------------------------
# 2. TOKENIZATION FOR BM25
# --------------------------------------------------

def tokenize(text):
    """Convert text into lowercase scientific word tokens."""
    return re.findall(r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b", str(text).lower())


# --------------------------------------------------
# 3. LOAD DENSE INDEX
# --------------------------------------------------

def load_dense_index():
    print("\nLoading dense retrieval index...")

    index_path = DENSE_DIR / "faiss.index"
    metadata_path = DENSE_DIR / "chunk_metadata.csv"

    index = faiss.read_index(str(index_path))
    metadata = pd.read_csv(metadata_path)

    if index.ntotal != len(metadata):
        raise ValueError("Dense index and metadata row counts do not match.")

    print("Dense index loaded successfully.")
    print("Vectors:", index.ntotal)

    return index, metadata


# --------------------------------------------------
# 4. LOAD BM25 INDEX
# --------------------------------------------------

def load_bm25_index():
    print("\nLoading BM25 retrieval index...")

    index_path = BM25_DIR / "bm25_index.pkl"
    metadata_path = BM25_DIR / "chunk_metadata.csv"

    with open(index_path, "rb") as file:
        bm25 = pickle.load(file)

    metadata = pd.read_csv(metadata_path)

    if bm25.corpus_size != len(metadata):
        raise ValueError("BM25 index and metadata row counts do not match.")

    print("BM25 index loaded successfully.")
    print("Documents:", bm25.corpus_size)

    return bm25, metadata


# --------------------------------------------------
# 5. CHECK METADATA ALIGNMENT
# --------------------------------------------------

def validate_alignment(dense_metadata, bm25_metadata):

    if not dense_metadata["chunk_id"].equals(
        bm25_metadata["chunk_id"]
    ):
        raise ValueError(
            "Dense and BM25 chunk metadata are not aligned."
        )

    print("\nMetadata alignment verified successfully.")


# --------------------------------------------------
# 6. DENSE RETRIEVAL
# --------------------------------------------------

def dense_search(query, index, metadata, model):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        TOP_K
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        row = metadata.iloc[int(idx)]

        results.append({
            "score": float(score),
            "article_id": row.get("article_id", "N/A"),
            "chunk_id": row.get("chunk_id", "N/A"),
            "title": row.get("title", "N/A"),
            "page": row.get("page_label", row.get("page", "N/A")),
            "text": row.get("text", "")
        })

    return results


# --------------------------------------------------
# 7. BM25 RETRIEVAL
# --------------------------------------------------

def bm25_search(query, bm25, metadata):

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    results = []

    for idx in top_indices:

        row = metadata.iloc[int(idx)]

        results.append({
            "score": float(scores[idx]),
            "article_id": row.get("article_id", "N/A"),
            "chunk_id": row.get("chunk_id", "N/A"),
            "title": row.get("title", "N/A"),
            "page": row.get("page_label", row.get("page", "N/A")),
            "text": row.get("text", "")
        })

    return results


# --------------------------------------------------
# 8. DISPLAY SEARCH RESULTS
# --------------------------------------------------

def display_results(method, results):

    print("\n" + "=" * 80)
    print(method)
    print("=" * 80)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Article ID: {result['article_id']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Title: {result['title']}")
        print(f"Page: {result['page']}")

        print("\nPassage:")
        print(str(result["text"])[:800])
        print("-" * 80)


# --------------------------------------------------
# 9. MAIN PROGRAM
# --------------------------------------------------

def main():

    print("=" * 80)
    print("RESEARCH PAPER RETRIEVAL TEST")
    print("=" * 80)

    dense_index, dense_metadata = load_dense_index()
    bm25_index, bm25_metadata = load_bm25_index()

    validate_alignment(dense_metadata, bm25_metadata)

    print("\nLoading sentence-transformer model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully.")

    query = (
        "What deep learning models are used "
        "for plant disease detection?"
    )

    print("\nSearch question:")
    print(query)

    dense_results = dense_search(
        query, dense_index, dense_metadata, model
    )

    bm25_results = bm25_search(
        query, bm25_index, bm25_metadata
    )

    display_results("DENSE RETRIEVAL RESULTS", dense_results)
    display_results("BM25 RETRIEVAL RESULTS", bm25_results)

    print("\nRETRIEVAL TEST COMPLETED.")


if __name__ == "__main__":
    main()
