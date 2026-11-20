
import json
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

QUESTION_FILE = ROOT / "data" / "evaluation" / "question_bank_80.csv"

OUTPUT_DIR = ROOT / "data" / "evaluation" / "rag_evidence_80_current_index"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "retrieved_evidence_top10.csv"
PARTIAL_FILE = OUTPUT_DIR / "retrieved_evidence_top10.partial.csv"

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
# 3. LOAD AND VALIDATE INDEXES
# --------------------------------------------------

def load_indexes():

    print("\nLoading Dense index...")

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

    if not dense_metadata["chunk_id"].equals(
        bm25_metadata["chunk_id"]
    ):
        raise ValueError("Dense and BM25 metadata are not aligned.")

    if dense_index.ntotal != len(dense_metadata):
        raise ValueError("Dense index size mismatch.")

    if bm25_index.corpus_size != len(bm25_metadata):
        raise ValueError("BM25 index size mismatch.")

    if dense_metadata["chunk_id"].duplicated().any():
        raise ValueError("Duplicate chunk IDs found.")

    if dense_metadata["text"].isna().any():
        raise ValueError("Missing passage text found.")

    print("Index validation passed.")

    return dense_index, bm25_index, dense_metadata


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

        results.append({
            "chunk_id": metadata.iloc[int(idx)]["chunk_id"],
            "rank": rank
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

        results.append({
            "chunk_id": metadata.iloc[int(idx)]["chunk_id"],
            "rank": rank
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

    for result in dense_results:

        chunk_id = result["chunk_id"]
        rank = result["rank"]

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

    for result in bm25_results:

        chunk_id = result["chunk_id"]
        rank = result["rank"]

        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (RRF_K + rank)
        )

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
            "article_id": row["article_id"],
            "title": row["title"],
            "page": row["page"],
            "text": str(row["text"])
        })

    return candidates


# --------------------------------------------------
# 7. CROSS-ENCODER RERANKING
# --------------------------------------------------

def rerank_results(query, candidates, reranker):

    pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False
    )

    for candidate, score in zip(candidates, scores):
        candidate["cross_encoder_score"] = float(score)

    return sorted(
        candidates,
        key=lambda item: item["cross_encoder_score"],
        reverse=True
    )[:TOP_K]


# --------------------------------------------------
# 8. MAIN PROGRAM
# --------------------------------------------------

def main():

    if OUTPUT_FILE.exists() or PARTIAL_FILE.exists():
        raise FileExistsError(
            "Output already exists. No existing results were overwritten. "
            "Inspect the output folder before running again."
        )

    questions = pd.read_csv(QUESTION_FILE)

    if len(questions) != 80:
        raise ValueError(
            f"Expected 80 questions, found {len(questions)}."
        )

    if questions["question_id"].nunique() != 80:
        raise ValueError("Question IDs are not unique.")

    print("=" * 75)
    print("PREPARING RAG EVIDENCE FOR 80 QUESTIONS")
    print("=" * 75)

    dense_index, bm25_index, metadata = load_indexes()

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )

    print("Loading Cross-Encoder...")

    reranker = CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
        device="cpu"
    )

    print("\nStarting retrieval...\n")

    all_results = []

    for number, question in enumerate(
        questions.to_dict("records"), start=1
    ):

        question_id = question["question_id"]
        query = question["question"]

        print(
            f"[{number}/80] {question_id}: "
            f"{query[:90]}",
            flush=True
        )

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

        candidates = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            metadata
        )

        final_results = rerank_results(
            query,
            candidates,
            reranker
        )

        for rank, result in enumerate(
            final_results, start=1
        ):

            all_results.append({
                "question_id": question_id,
                "question": query,
                "category": question["category"],
                "answerability": question["answerability"],
                "method": "Hybrid_CrossEncoder_CurrentIndex",
                "rank": rank,
                "chunk_id": result["chunk_id"],
                "article_id": result["article_id"],
                "title": result["title"],
                "page": result["page"],
                "rrf_score": result["rrf_score"],
                "cross_encoder_score": result["cross_encoder_score"],
                "retrieved_text": result["text"]
            })

        # Save progress after each question.
        pd.DataFrame(all_results).to_csv(
            PARTIAL_FILE,
            index=False,
            encoding="utf-8-sig"
        )

    results = pd.DataFrame(all_results)

    if results["question_id"].nunique() != 80:
        raise ValueError("Not all 80 questions were retrieved.")

    if len(results) != 800:
        raise ValueError(
            f"Expected 800 passages, found {len(results)}."
        )

    PARTIAL_FILE.replace(OUTPUT_FILE)

    run_info = {
        "experiment": "RAG evidence preparation for 80 questions",
        "questions": 80,
        "retrieved_passages": len(results),
        "indexed_chunks": int(dense_index.ntotal),
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
        "rrf_k": RRF_K,
        "retrieval_method": "Dense + BM25 + RRF + Cross-Encoder",
        "device": "cpu",
        "source_question_bank": str(QUESTION_FILE),
        "source_dense_index": str(DENSE_DIR),
        "source_bm25_index": str(BM25_DIR)
    }

    with open(
        OUTPUT_DIR / "run_info.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(run_info, file, indent=4)

    print("\n" + "=" * 75)
    print("RETRIEVAL COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print("Questions:", results["question_id"].nunique())
    print("Retrieved passages:", len(results))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
