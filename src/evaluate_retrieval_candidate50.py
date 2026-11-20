
from pathlib import Path
import argparse
import json
import pickle
import re
import time

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

QUESTION_FILE = (
    ROOT / "data" / "evaluation" / "question_bank_80.csv"
)

INDEX_ROOT = ROOT / "data" / "indexes"
EVALUATION_ROOT = ROOT / "data" / "evaluation"

STRATEGIES = {
    "fixed_size": {
        "qrels": EVALUATION_ROOT / "qrels_fixed_size_audited.csv",
        "dense": INDEX_ROOT / "fixed_size_dense_filtered_v1",
        "bm25": INDEX_ROOT / "fixed_size_bm25_filtered_v1",
        "output": (
            EVALUATION_ROOT
            / "retrieval_results_fixed_size_candidate50"
        )
    }
}

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

TOP_K = 10
CANDIDATE_K = 50
RRF_K = 60


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def tokenize(text):
    return re.findall(
        r"[a-z0-9]+(?:[._+-][a-z0-9]+)*",
        str(text).lower()
    )


def find_column(frame, possible_names, description):
    for name in possible_names:
        if name in frame.columns:
            return name

    raise ValueError(
        f"Missing expected column in {description}. "
        f"Available columns: {list(frame.columns)}"
    )


def recall_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)

    if not relevant:
        return np.nan

    found = set(retrieved_ids[:k]) & relevant

    return len(found) / len(relevant)


def ndcg_at_k(retrieved_ids, relevant_grades, k):
    if not relevant_grades:
        return np.nan

    gains = [
        (2 ** relevant_grades.get(chunk_id, 0)) - 1
        for chunk_id in retrieved_ids[:k]
    ]

    dcg = sum(
        gain / np.log2(rank + 2)
        for rank, gain in enumerate(gains)
    )

    ideal_grades = sorted(
        relevant_grades.values(),
        reverse=True
    )[:k]

    idcg = sum(
        ((2 ** grade) - 1) / np.log2(rank + 2)
        for rank, grade in enumerate(ideal_grades)
    )

    return dcg / idcg if idcg else np.nan


def reciprocal_rank_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(
        retrieved_ids[:k], start=1
    ):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def reciprocal_rank_fusion(dense_ids, bm25_ids):
    scores = {}

    for ranked_list in (dense_ids, bm25_ids):
        for rank, chunk_id in enumerate(
            ranked_list, start=1
        ):
            scores[chunk_id] = (
                scores.get(chunk_id, 0)
                + 1.0 / (RRF_K + rank)
            )

    return sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True
    )


# --------------------------------------------------
# 3. LOAD DATA AND INDEXES
# --------------------------------------------------

def load_inputs(strategy_config):

    dense_dir = strategy_config["dense"]
    bm25_dir = strategy_config["bm25"]
    qrels_file = strategy_config["qrels"]

    required_files = [
        QUESTION_FILE,
        qrels_file,
        dense_dir / "faiss.index",
        dense_dir / "chunk_metadata.csv",
        dense_dir / "index_config.json",
        bm25_dir / "bm25_index.pkl",
        bm25_dir / "chunk_metadata.csv"
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(missing)
        )

    questions = pd.read_csv(
        QUESTION_FILE,
        encoding="utf-8-sig"
    ).fillna("")

    qrels = pd.read_csv(
        qrels_file,
        encoding="utf-8-sig"
    ).fillna("")

    dense_meta = pd.read_csv(
        dense_dir / "chunk_metadata.csv",
        encoding="utf-8-sig"
    )

    bm25_meta = pd.read_csv(
        bm25_dir / "chunk_metadata.csv",
        encoding="utf-8-sig"
    )

    # Identify columns

    qid_col = find_column(
        qrels,
        ["query_id", "question_id"],
        "qrels"
    )

    qrels_chunk_col = find_column(
        qrels,
        ["chunk_id"],
        "qrels"
    )

    grade_col = find_column(
        qrels,
        ["relevance_grade", "relevance", "grade"],
        "qrels"
    )

    question_id_col = find_column(
        questions,
        ["question_id", "query_id"],
        "question bank"
    )

    question_text_col = find_column(
        questions,
        ["question", "query"],
        "question bank"
    )

    answerability_col = find_column(
        questions,
        ["answerability"],
        "question bank"
    )

    chunk_id_col = find_column(
        dense_meta,
        ["chunk_id"],
        "Dense metadata"
    )

    text_col = find_column(
        dense_meta,
        ["text"],
        "Dense metadata"
    )

    bm25_chunk_col = find_column(
        bm25_meta,
        ["chunk_id"],
        "BM25 metadata"
    )

    # Validate metadata order

    dense_meta[chunk_id_col] = (
        dense_meta[chunk_id_col].astype(str)
    )

    bm25_meta[bm25_chunk_col] = (
        bm25_meta[bm25_chunk_col].astype(str)
    )

    if (
        dense_meta[chunk_id_col].tolist()
        != bm25_meta[bm25_chunk_col].tolist()
    ):
        raise ValueError(
            "Dense and BM25 metadata order does not match."
        )

    # Normalize question and qrels fields

    questions[question_id_col] = (
        questions[question_id_col].astype(str)
    )

    questions[answerability_col] = (
        questions[answerability_col]
        .astype(str)
        .str.strip()
    )

    qrels[qid_col] = qrels[qid_col].astype(str)

    qrels[qrels_chunk_col] = (
        qrels[qrels_chunk_col].astype(str)
    )

    qrels[grade_col] = pd.to_numeric(
        qrels[grade_col],
        errors="coerce"
    )

    if qrels[grade_col].isna().any():
        raise ValueError(
            "Invalid or missing relevance grades found."
        )

    if not qrels[grade_col].isin([0, 1, 2]).all():
        raise ValueError(
            "Unexpected relevance grades. "
            "Expected only 0, 1 and 2."
        )

    qrels[grade_col] = qrels[grade_col].astype(int)

    # Keep answerable questions only

    answerable = questions[
        questions[answerability_col].str.lower()
        == "answerable"
    ].copy()

    if answerable.empty:
        raise ValueError(
            "No answerable questions found."
        )

    # Group relevance judgments

    qrels_by_query = {}

    for query_id, group in qrels.groupby(qid_col):
        qrels_by_query[str(query_id)] = (
            group.groupby(qrels_chunk_col)[grade_col]
            .max()
            .to_dict()
        )

    # Ensure all answerable questions have qrels

    missing_qrels = [
        qid
        for qid in answerable[question_id_col]
        if qid not in qrels_by_query
        or not qrels_by_query[qid]
    ]

    if missing_qrels:
        raise ValueError(
            "Missing relevance judgments for: "
            + ", ".join(missing_qrels)
        )

    # Load FAISS index

    index = faiss.read_index(
        str(dense_dir / "faiss.index")
    )

    if index.ntotal != len(dense_meta):
        raise ValueError(
            "FAISS vector count does not match metadata."
        )

    # Load BM25 index

    with open(
        bm25_dir / "bm25_index.pkl",
        "rb"
    ) as file:
        bm25 = pickle.load(file)

    if len(bm25.doc_len) != len(bm25_meta):
        raise ValueError(
            "BM25 document count does not match metadata."
        )

    return {
        "questions": answerable,
        "qrels": qrels_by_query,
        "question_id_col": question_id_col,
        "question_text_col": question_text_col,
        "metadata": dense_meta,
        "chunk_id_col": chunk_id_col,
        "text_col": text_col,
        "index": index,
        "bm25": bm25,
        "qrels_rows": len(qrels),
        "qrels_grade_distribution": (
            qrels[grade_col].value_counts().to_dict()
        )
    }


# --------------------------------------------------
# 4. EVALUATION
# --------------------------------------------------

def evaluate(strategy_name, strategy_config):

    output_dir = strategy_config["output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading evaluation data and indexes...")

    data = load_inputs(strategy_config)

    questions = data["questions"]
    metadata = data["metadata"]

    index = data["index"]
    bm25 = data["bm25"]
    qrels = data["qrels"]

    chunk_ids = (
        metadata[data["chunk_id_col"]]
        .astype(str)
        .tolist()
    )

    chunk_texts = (
        metadata[data["text_col"]]
        .fillna("")
        .astype(str)
        .tolist()
    )

    chunk_positions = {
        chunk_id: position
        for position, chunk_id in enumerate(chunk_ids)
    }

    question_records = []

    for _, row in questions.iterrows():
        question_records.append({
            "query_id": str(
                row[data["question_id_col"]]
            ),
            "question": str(
                row[data["question_text_col"]]
            ),
            "category": str(
                row.get("category", "")
            )
        })

    # Validate qrels chunk IDs

    unknown_chunks = {
        chunk_id
        for grades in qrels.values()
        for chunk_id in grades
        if chunk_id not in chunk_positions
    }

    if unknown_chunks:
        raise ValueError(
            "Some judged chunk IDs are missing "
            "from the index: "
            + ", ".join(sorted(unknown_chunks)[:10])
        )

    print("Strategy:", strategy_name)
    print("Answerable questions:", len(question_records))
    print("Qrels judgments:", data["qrels_rows"])
    print(
        "Qrels grade distribution:",
        data["qrels_grade_distribution"]
    )
    print("Indexed chunks:", len(chunk_ids))
    print("Candidate pool:", CANDIDATE_K)
    print("Final retrieval depth:", TOP_K)

    if CANDIDATE_K > len(chunk_ids):
        raise ValueError(
            "Candidate pool exceeds the number of indexed chunks."
        )

    # Load models

    print("\nLoading embedding model...")

    embedder = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu"
    )

    print("Loading Cross-Encoder model...")

    reranker = CrossEncoder(
        RERANKER_MODEL_NAME,
        device="cpu"
    )

    methods = [
        "BM25",
        "Dense",
        "Hybrid_RRF",
        "Hybrid_CrossEncoder"
    ]

    rankings = {
        method: {}
        for method in methods
    }

    timing_records = []
    detailed_records = []

    # Process questions

    for number, record in enumerate(
        question_records, start=1
    ):

        query_id = record["query_id"]
        query = record["question"]

        print(
            f"\n[{number}/{len(question_records)}] "
            f"Evaluating {query_id}"
        )

        # BM25 retrieval

        start = time.perf_counter()

        bm25_scores = np.asarray(
            bm25.get_scores(tokenize(query))
        )

        bm25_positions = np.argsort(
            -bm25_scores,
            kind="stable"
        )[:CANDIDATE_K]

        bm25_ids = [
            chunk_ids[int(position)]
            for position in bm25_positions
        ]

        bm25_time = time.perf_counter() - start

        # Dense retrieval

        start = time.perf_counter()

        query_vector = embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        dense_scores, dense_positions = index.search(
            query_vector,
            CANDIDATE_K
        )

        dense_ids = [
            chunk_ids[int(position)]
            for position in dense_positions[0]
            if int(position) >= 0
        ]

        dense_time = time.perf_counter() - start

        # Hybrid RRF

        start = time.perf_counter()

        hybrid_ids = reciprocal_rank_fusion(
            dense_ids,
            bm25_ids
        )[:CANDIDATE_K]

        hybrid_time = time.perf_counter() - start

        # Cross-Encoder reranking

        start = time.perf_counter()

        pairs = [
            (
                query,
                chunk_texts[chunk_positions[chunk_id]]
            )
            for chunk_id in hybrid_ids
        ]

        cross_scores = reranker.predict(
            pairs,
            show_progress_bar=False
        )

        reranked_positions = np.argsort(
            -np.asarray(cross_scores),
            kind="stable"
        )

        reranked_ids = [
            hybrid_ids[int(position)]
            for position in reranked_positions
        ]

        reranker_time = time.perf_counter() - start

        # Store top 10 results

        rankings["BM25"][query_id] = bm25_ids[:TOP_K]

        rankings["Dense"][query_id] = dense_ids[:TOP_K]

        rankings["Hybrid_RRF"][query_id] = (
            hybrid_ids[:TOP_K]
        )

        rankings["Hybrid_CrossEncoder"][query_id] = (
            reranked_ids[:TOP_K]
        )

        # Record latency

        timing_records.extend([
            {
                "question_id": query_id,
                "method": "BM25",
                "latency_seconds": bm25_time
            },
            {
                "question_id": query_id,
                "method": "Dense",
                "latency_seconds": dense_time
            },
            {
                "question_id": query_id,
                "method": "Hybrid_RRF",
                "latency_seconds": (
                    bm25_time + dense_time + hybrid_time
                )
            },
            {
                "question_id": query_id,
                "method": "Hybrid_CrossEncoder",
                "latency_seconds": (
                    bm25_time + dense_time
                    + hybrid_time + reranker_time
                )
            }
        ])

        # Save retrieved passage details

        for method in methods:
            for rank, chunk_id in enumerate(
                rankings[method][query_id],
                start=1
            ):

                position = chunk_positions[chunk_id]
                meta = metadata.iloc[position]

                detailed_records.append({
                    "question_id": query_id,
                    "category": record["category"],
                    "method": method,
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "article_id": meta.get(
                        "article_id", ""
                    ),
                    "title": meta.get("title", ""),
                    "page": meta.get("page", ""),
                    "retrieved_text": chunk_texts[position]
                })

    # --------------------------------------------------
    # 5. CALCULATE METRICS
    # --------------------------------------------------

    metric_records = []

    for method in methods:
        for record in question_records:

            query_id = record["query_id"]

            retrieved = rankings[method][query_id]
            relevant_grades = qrels[query_id]

            relevant_ids = [
                chunk_id
                for chunk_id, grade in relevant_grades.items()
                if grade > 0
            ]

            metric_records.append({
                "question_id": query_id,
                "category": record["category"],
                "method": method,

                "Recall@5": recall_at_k(
                    retrieved,
                    relevant_ids,
                    5
                ),

                "nDCG@10": ndcg_at_k(
                    retrieved,
                    relevant_grades,
                    10
                ),

                "MRR@10": reciprocal_rank_at_k(
                    retrieved,
                    relevant_ids,
                    10
                )
            })

    metrics = pd.DataFrame(metric_records)
    timings = pd.DataFrame(timing_records)

    per_question = metrics.merge(
        timings,
        on=["question_id", "method"],
        how="left"
    )

    summary = (
        per_question.groupby("method", as_index=False)
        .agg(
            questions_evaluated=("question_id", "nunique"),
            mean_recall_at_5=("Recall@5", "mean"),
            mean_ndcg_at_10=("nDCG@10", "mean"),
            mean_mrr_at_10=("MRR@10", "mean"),
            mean_latency_seconds=("latency_seconds", "mean"),
            median_latency_seconds=("latency_seconds", "median")
        )
    )

    # --------------------------------------------------
    # 6. SAVE RESULTS
    # --------------------------------------------------

    per_question.to_csv(
        output_dir / "retrieval_metrics_per_question.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        output_dir / "retrieval_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(detailed_records).to_csv(
        output_dir / "retrieved_chunks_top10.csv",
        index=False,
        encoding="utf-8-sig"
    )

    run_info = {
        "chunking_strategy": strategy_name,
        "answerable_questions_evaluated": len(question_records),
        "unanswerable_questions_excluded": (
            80 - len(question_records)
        ),
        "qrels_file": str(strategy_config["qrels"]),
        "qrels_judgment_count": data["qrels_rows"],
        "qrels_grade_distribution": {
            str(key): int(value)
            for key, value in (
                data["qrels_grade_distribution"].items()
            )
        },
        "indexed_chunks": len(chunk_ids),
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
        "metrics": [
            "Recall@5",
            "nDCG@10",
            "MRR@10"
        ],
        "note": (
            "Candidate-pool-50 experiment based on the "
            "fixed-size chunking strategy. Only qrels "
            "grades 1 and 2 count as relevant for Recall "
            "and MRR. Grade 0 contributes no gain to nDCG. "
            "Unjudged chunks are treated as non-relevant. "
            "This run should be compared with v11 using "
            "the same qrels and evaluation settings."
        )
    }

    with open(
        output_dir / "evaluation_run_info.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            run_info,
            file,
            indent=4
        )

    # --------------------------------------------------
    # 7. DISPLAY SUMMARY
    # --------------------------------------------------

    print("\n==========================================")
    print("       RETRIEVAL EVALUATION SUMMARY")
    print("==========================================")

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nResults saved in:")
    print(output_dir)

    print("\nEvaluation completed successfully.")


# --------------------------------------------------
# 8. COMMAND-LINE SELECTION
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate crop disease retrieval strategies."
    )

    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        required=True,
        help="Select fixed_size."
    )

    args = parser.parse_args()

    selected_config = STRATEGIES[args.strategy]

    print("=" * 60)
    print("CROP DISEASE RETRIEVAL EVALUATION")
    print("=" * 60)

    evaluate(
        args.strategy,
        selected_config
    )
