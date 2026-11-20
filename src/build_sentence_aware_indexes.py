
import json
import csv
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# 1. PATHS AND CONFIGURATION

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "phase4_chunking_comparison_filtered_v1"
    / "sentence_aware_chunks.csv"
)

DENSE_DIR = ROOT / "data" / "indexes" / "sentence_aware_dense_filtered_v1"
BM25_DIR = ROOT / "data" / "indexes" / "sentence_aware_bm25_filtered_v1"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


# 2. LOAD AND VALIDATE CHUNKS

def load_chunks():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = ["article_id", "chunk_id", "title", "page", "text"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].copy()

    if df["chunk_id"].duplicated().any():
        raise ValueError("Duplicate chunk IDs found.")

    df.reset_index(drop=True, inplace=True)

    print("Valid chunks:", len(df))
    print("Unique articles:", df["article_id"].nunique())

    return df


# 3. BUILD DENSE INDEX

def build_dense(df):

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("Generating embeddings...")

    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    DENSE_DIR.mkdir(parents=True)

    faiss.write_index(index, str(DENSE_DIR / "faiss.index"))

    save_metadata(df, DENSE_DIR, utf8_sig=True)

    config = {
        "model_name": MODEL_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
        "similarity_metric": "cosine_similarity",
        "index_type": "FAISS IndexFlatIP",
        "normalization": True,
        "device": "cpu",
        "total_vectors": int(index.ntotal),
        "total_chunks": len(df),
        "unique_articles": int(df["article_id"].nunique()),
        "chunking_strategy": "sentence_aware",
        "input_file": str(INPUT_FILE)
    }

    with open(DENSE_DIR / "index_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("Dense index completed:", index.ntotal)


# 4. BUILD BM25 INDEX

def tokenize(text):

    return re.findall(
        r"[a-z0-9]+(?:[._+-][a-z0-9]+)*",
        str(text).lower()
    )


def build_bm25(df):

    print("\nTokenizing sentence-aware chunks...")

    corpus = [tokenize(text) for text in df["text"]]

    if any(not tokens for tokens in corpus):
        raise ValueError("Empty tokenized documents found.")

    bm25 = BM25Okapi(corpus)

    BM25_DIR.mkdir(parents=True)

    with open(BM25_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_metadata(df, BM25_DIR, utf8_sig=False)

    config = {
        "index_type": "BM25Okapi",
        "tokenization": "lowercase regex tokenizer",
        "total_documents": len(df),
        "unique_articles": int(df["article_id"].nunique()),
        "chunking_strategy": "sentence_aware",
        "input_file": str(INPUT_FILE)
    }

    with open(BM25_DIR / "index_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("BM25 index completed:", len(corpus))


# 5. SAVE CONSISTENT METADATA

def save_metadata(df, output_dir, utf8_sig):

    columns = [
        "article_id", "chunk_id", "title", "authors",
        "year", "venue", "doi_or_url", "page",
        "section_title", "word_count", "chunking_strategy",
        "corpus_role", "extraction_method", "text"
    ]

    columns = [c for c in columns if c in df.columns]

    df[columns].to_csv(
        output_dir / "chunk_metadata.csv",
        index=False,
        encoding="utf-8-sig" if utf8_sig else "utf-8",
        quoting=csv.QUOTE_ALL if utf8_sig else csv.QUOTE_MINIMAL,
        escapechar="\\" if utf8_sig else None
    )


# 6. MAIN EXECUTION

def main():

    print("=" * 55)
    print("SENTENCE-AWARE INDEX BUILDING")
    print("=" * 55)

    # Prevent accidental overwriting
    if DENSE_DIR.exists() or BM25_DIR.exists():
        raise FileExistsError(
            "An output index folder already exists. "
            "No existing index was overwritten."
        )

    df = load_chunks()

    build_dense(df)
    build_bm25(df)

    print("\nBoth sentence-aware indexes built successfully.")
    print("Dense:", DENSE_DIR)
    print("BM25:", BM25_DIR)


if __name__ == "__main__":
    main()
