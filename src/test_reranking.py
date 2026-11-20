
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DENSE_DIR = ROOT / "data" / "indexes" / "fixed_size_dense"
BM25_DIR = ROOT / "data" / "indexes" / "fixed_size_bm25"

OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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
# 3. LOAD INDEXES AND METADATA
# --------------------------------------------------

def load_indexes():

    print("\nLoading dense index...")

    dense_index = faiss.read_index(
        str(DENSE_DIR / "faiss.index")
    )

    metadata = pd.read_csv(
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

    if not metadata["chunk_id"].equals(
        bm25_metadata["chunk_id"]
    ):
        raise ValueError("Metadata alignment failed.")

    if dense_index.ntotal != len(metadata):
        raise ValueError("Dense index size mismatch.")

    if bm25_index.corpus_size != len(bm25_metadata):
        raise ValueError("BM25 index size mismatch.")

    if metadata["chunk_id"].duplicated().any():
        raise ValueError("Duplicate chunk IDs found.")

    print("Index and metadata validation passed.")

    return dense_index, bm25_index, metadata


# --------------------------------------------------
# 4. DENSE RETRIEVAL
# --------------------------------------------------

def dense_search(query, index, metadata, model):

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores, indices = index.search(
        embedding.astype("float32"),
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

    indices = np.argsort(scores)[::-1][:CANDIDATE_K]

    results = []

    for rank, idx in enumerate(indices, start=1):

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

    for result in dense_results:

        chunk_id = result["chunk_id"]
        rank = result["rank"]

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

        dense_ranks[chunk_id] = rank

    for result in bm25_results:

        chunk_id = result["chunk_id"]
        rank = result["rank"]

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

        bm25_ranks[chunk_id] = rank

    ranked_chunks = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    metadata_lookup = metadata.set_index("chunk_id")

    candidates = []

    for chunk_id, score in ranked_chunks:

        row = metadata_lookup.loc[chunk_id]

        candidates.append({
            "chunk_id": chunk_id,
            "rrf_score": score,
            "dense_rank": dense_ranks.get(
                chunk_id, "Not retrieved"
            ),
            "bm25_rank": bm25_ranks.get(
                chunk_id, "Not retrieved"
            ),
            "article_id": row.get("article_id", "N/A"),
            "title": row.get("title", "N/A"),
            "page": row.get(
                "page_label",
                row.get("page", "N/A")
            ),
            "text": str(row.get("text", ""))
        })

    return candidates


# --------------------------------------------------
# 7. CROSS-ENCODER RERANKING
# --------------------------------------------------

def rerank_results(query, candidates, reranker):

    print("\nApplying Cross-Encoder reranking...")
    print("Candidate passages:", len(candidates))

    pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs,
        batch_size=8,
        show_progress_bar=True
    )

    for candidate, score in zip(candidates, scores):
        candidate["cross_encoder_score"] = float(score)

    reranked = sorted(
        candidates,
        key=lambda item: item["cross_encoder_score"],
        reverse=True
    )

    return reranked[:TOP_K]


# --------------------------------------------------
# 8. DISPLAY FINAL RESULTS
# --------------------------------------------------

def display_results(results):

    print("\n" + "=" * 85)
    print("FINAL CROSS-ENCODER RERANKED RESULTS")
    print("=" * 85)

    for rank, result in enumerate(results, start=1):

        print(f"\nFinal Rank: {rank}")
        print(
            f"Cross-Encoder Score: "
            f"{result['cross_encoder_score']:.4f}"
        )
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Article ID: {result['article_id']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Title: {result['title']}")
        print(f"Page: {result['page']}")
        print(f"Dense Rank: {result['dense_rank']}")
        print(f"BM25 Rank: {result['bm25_rank']}")

        print("\nPassage:")
        print(result["text"][:1000])

        print("-" * 85)


# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

def save_results(results):

    output_file = OUTPUT_DIR / "cross_encoder_reranked.csv"

    pd.DataFrame(results).to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nResults saved to:")
    print(output_file)


# --------------------------------------------------
# 10. MAIN PROGRAM
# --------------------------------------------------

def main():

    print("=" * 85)
    print("HYBRID RETRIEVAL WITH CROSS-ENCODER RERANKING")
    print("=" * 85)

    dense_index, bm25_index, metadata = load_indexes()

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    print("\nLoading Cross-Encoder model...")

    reranker = CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
        device="cpu"
    )

    print("Cross-Encoder loaded.")

    query = (
        "What deep learning models are used "
        "for plant disease detection?"
    )

    print("\nQuestion:", query)

    dense_results = dense_search(
        query,
        dense_index,
        metadata,
        embedding_model
    )

    bm25_results = bm25_search(
        query,
        bm25_index,
        metadata
    )

    print("\nDense candidates:", len(dense_results))
    print("BM25 candidates:", len(bm25_results))

    candidates = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        metadata
    )

    print("Unique hybrid candidates:", len(candidates))

    final_results = rerank_results(
        query,
        candidates,
        reranker
    )

    display_results(final_results)

    save_results(final_results)

    print("\nCROSS-ENCODER RERANKING COMPLETED.")


if __name__ == "__main__":
    main()
