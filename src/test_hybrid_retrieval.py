
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DENSE_DIR = ROOT / "data" / "indexes" / "fixed_size_dense"
BM25_DIR = ROOT / "data" / "indexes" / "fixed_size_bm25"

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CANDIDATE_K = 20
TOP_K = 10
RRF_K = 60


# --------------------------------------------------
# 2. TOKENIZATION
# --------------------------------------------------

def tokenize(text):
    return re.findall(
        r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b",
        str(text).lower()
    )


# --------------------------------------------------
# 3. LOAD INDEXES
# --------------------------------------------------

def load_indexes():

    print("\nLoading dense index...")

    dense_index = faiss.read_index(
        str(DENSE_DIR / "faiss.index")
    )

    dense_metadata = pd.read_csv(
        DENSE_DIR / "chunk_metadata.csv"
    )

    print("Dense vectors:", dense_index.ntotal)

    print("\nLoading BM25 index...")

    with open(BM25_DIR / "bm25_index.pkl", "rb") as file:
        bm25_index = pickle.load(file)

    bm25_metadata = pd.read_csv(
        BM25_DIR / "chunk_metadata.csv"
    )

    print("BM25 documents:", bm25_index.corpus_size)

    if not dense_metadata["chunk_id"].equals(
        bm25_metadata["chunk_id"]
    ):
        raise ValueError("Dense and BM25 metadata are not aligned.")

    if dense_index.ntotal != len(dense_metadata):
        raise ValueError("Dense index size mismatch.")

    if bm25_index.corpus_size != len(bm25_metadata):
        raise ValueError("BM25 index size mismatch.")

    print("\nBoth indexes loaded and validated.")

    return dense_index, bm25_index, dense_metadata


# --------------------------------------------------
# 4. DENSE RETRIEVAL
# --------------------------------------------------

def dense_search(query, index, metadata, model):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        CANDIDATE_K
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):

        if idx < 0:
            continue

        row = metadata.iloc[int(idx)]

        results.append({
            "chunk_id": row["chunk_id"],
            "rank": rank,
            "dense_score": float(score)
        })

    return results


# --------------------------------------------------
# 5. BM25 RETRIEVAL
# --------------------------------------------------

def bm25_search(query, bm25, metadata):

    scores = bm25.get_scores(tokenize(query))

    top_indices = np.argsort(scores)[::-1][:CANDIDATE_K]

    results = []

    for rank, idx in enumerate(top_indices, start=1):

        row = metadata.iloc[int(idx)]

        results.append({
            "chunk_id": row["chunk_id"],
            "rank": rank,
            "bm25_score": float(scores[idx])
        })

    return results


# --------------------------------------------------
# 6. RECIPROCAL RANK FUSION
# --------------------------------------------------

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    metadata
):

    rrf_scores = {}

    dense_ranks = {}
    bm25_ranks = {}

    # Add dense retrieval rankings
    for result in dense_results:

        chunk_id = result["chunk_id"]
        rank = result["rank"]

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

        dense_ranks[chunk_id] = rank

    # Add BM25 retrieval rankings
    for result in bm25_results:

        chunk_id = result["chunk_id"]
        rank = result["rank"]

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

        bm25_ranks[chunk_id] = rank

    # Sort by combined RRF score
    ranked_chunks = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    metadata_lookup = metadata.set_index("chunk_id")

    final_results = []

    for chunk_id, score in ranked_chunks[:TOP_K]:

        row = metadata_lookup.loc[chunk_id]

        final_results.append({
            "rrf_score": score,
            "chunk_id": chunk_id,
            "dense_rank": dense_ranks.get(chunk_id, "Not retrieved"),
            "bm25_rank": bm25_ranks.get(chunk_id, "Not retrieved"),
            "article_id": row.get("article_id", "N/A"),
            "title": row.get("title", "N/A"),
            "page": row.get(
                "page_label",
                row.get("page", "N/A")
            ),
            "text": row.get("text", "")
        })

    return final_results


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

def display_results(results):

    print("\n" + "=" * 85)
    print("HYBRID RETRIEVAL RESULTS - RRF")
    print("=" * 85)

    for rank, result in enumerate(results, start=1):

        print(f"\nFinal Rank: {rank}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Article ID: {result['article_id']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Title: {result['title']}")
        print(f"Page: {result['page']}")
        print(f"Dense Rank: {result['dense_rank']}")
        print(f"BM25 Rank: {result['bm25_rank']}")

        print("\nPassage:")
        print(str(result["text"])[:800])

        print("-" * 85)


# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------

def save_results(results):

    output_file = OUTPUT_DIR / "hybrid_rrf_test.csv"

    pd.DataFrame(results).to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nResults saved to:")
    print(output_file)


# --------------------------------------------------
# 9. MAIN PROGRAM
# --------------------------------------------------

def main():

    print("=" * 85)
    print("HYBRID RETRIEVAL USING RECIPROCAL RANK FUSION")
    print("=" * 85)

    dense_index, bm25_index, metadata = load_indexes()

    print("\nLoading sentence-transformer model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully.")

    query = (
        "What deep learning models are used "
        "for plant disease detection?"
    )

    print("\nQuestion:", query)

    dense_results = dense_search(
        query, dense_index, metadata, model
    )

    bm25_results = bm25_search(
        query, bm25_index, metadata
    )

    print("\nDense candidates:", len(dense_results))
    print("BM25 candidates:", len(bm25_results))

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        metadata
    )

    display_results(hybrid_results)

    save_results(hybrid_results)

    print("\nHYBRID RETRIEVAL TEST COMPLETED.")


if __name__ == "__main__":
    main()
