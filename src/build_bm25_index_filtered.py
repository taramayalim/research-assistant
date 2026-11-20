
import json
import pickle
import re
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4_chunking_comparison_filtered_v1"
    / "section_aware_chunks.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "section_aware_bm25_filtered_v1"
)

INDEX_FILE = OUTPUT_DIR / "bm25_index.pkl"
METADATA_FILE = OUTPUT_DIR / "chunk_metadata.csv"
CONFIG_FILE = OUTPUT_DIR / "index_config.json"


# ==========================================
# 2. TEXT TOKENIZATION
# ==========================================

def tokenize(text):
    """
    Convert text into lowercase word tokens.
    Preserve scientific terms, numbers and
    model names such as ResNet50 and YOLOv8.
    """

    text = str(text).lower()

    tokens = re.findall(
        r"[a-z0-9]+(?:[._+-][a-z0-9]+)*",
        text
    )

    return tokens


# ==========================================
# 3. LOAD CHUNK DATA
# ==========================================

def load_chunks():

    print("\nLoading section-aware chunk dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Chunk dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "article_id",
        "chunk_id",
        "title",
        "page",
        "text"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[df["text"] != ""].copy()

    if df["chunk_id"].duplicated().any():
        raise ValueError(
            "Duplicate chunk IDs found."
        )

    df.reset_index(drop=True, inplace=True)

    print("Total rows loaded:", len(df))
    print("Valid chunks:", len(df))
    print("Unique articles:", df["article_id"].nunique())
    print("Chunking strategy: section_aware")

    return df


# ==========================================
# 4. BUILD BM25 INDEX
# ==========================================

def build_bm25(df):

    print("\nTokenizing research passages...")

    tokenized_corpus = [
        tokenize(text)
        for text in df["text"]
    ]

    empty_documents = sum(
        len(tokens) == 0
        for tokens in tokenized_corpus
    )

    if empty_documents:
        raise ValueError(
            f"{empty_documents} empty tokenized documents found."
        )

    print("Tokenization completed.")
    print("Total documents:", len(tokenized_corpus))

    print("\nBuilding BM25 index...")

    bm25 = BM25Okapi(tokenized_corpus)

    print("BM25 index created successfully.")

    return bm25


# ==========================================
# 5. SAVE INDEX AND METADATA
# ==========================================

def save_outputs(bm25, df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSaving BM25 index...")

    with open(INDEX_FILE, "wb") as file:
        pickle.dump(
            bm25,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    metadata_columns = [
        "article_id",
        "chunk_id",
        "title",
        "authors",
        "year",
        "venue",
        "doi_or_url",
        "page",
        "section_title",
        "word_count",
        "chunking_strategy",
        "corpus_role",
        "extraction_method",
        "text"
    ]

    metadata_columns = [
        col for col in metadata_columns
        if col in df.columns
    ]

    metadata = df[metadata_columns].copy()

    metadata.to_csv(
        METADATA_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    config = {
        "index_type": "BM25Okapi",
        "tokenization": "lowercase regex tokenizer",
        "total_documents": len(df),
        "unique_articles": int(
            df["article_id"].nunique()
        ),
        "chunking_strategy": "section_aware",
        "input_file": str(INPUT_FILE)
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            config,
            file,
            indent=4
        )

    print("\nSaved files:")
    print("BM25 index:", INDEX_FILE)
    print("Metadata:", METADATA_FILE)
    print("Configuration:", CONFIG_FILE)


# ==========================================
# 6. MAIN
# ==========================================

def main():

    print("=" * 55)
    print("SECTION-AWARE BM25 RETRIEVAL INDEX BUILDING")
    print("=" * 55)

    df = load_chunks()

    bm25 = build_bm25(df)

    save_outputs(bm25, df)

    print("\n" + "=" * 55)
    print("SECTION-AWARE BM25 INDEX BUILD COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()
