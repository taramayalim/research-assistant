
import json
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


# =====================================================
# 1. PROJECT PATHS AND CONFIGURATION
# =====================================================

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
    / "section_aware_dense_filtered_v1"
)

INDEX_FILE = OUTPUT_DIR / "faiss.index"
METADATA_FILE = OUTPUT_DIR / "chunk_metadata.csv"
CONFIG_FILE = OUTPUT_DIR / "index_config.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32


# =====================================================
# 2. LOAD CHUNK DATA
# =====================================================

def load_chunks():

    print("\nLoading section-aware chunk dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("Total rows loaded:", len(df))

    required_columns = [
        "article_id",
        "chunk_id",
        "title",
        "page",
        "text"
    ]

    missing = [
        col for col in required_columns
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
            "Duplicate chunk IDs found. Resolve them before indexing."
        )

    df.reset_index(drop=True, inplace=True)

    print("Valid chunks:", len(df))
    print("Unique articles:", df["article_id"].nunique())
    print("Chunking strategy: section_aware")

    return df


# =====================================================
# 3. GENERATE SEMANTIC EMBEDDINGS
# =====================================================

def generate_embeddings(df):

    print("\nLoading sentence embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    print("Model loaded successfully.")
    print("\nGenerating embeddings...")

    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )

    print("\nEmbedding generation completed.")
    print("Embedding shape:", embeddings.shape)

    return embeddings


# =====================================================
# 4. BUILD FAISS DENSE VECTOR INDEX
# =====================================================

def build_faiss_index(embeddings):

    print("\nBuilding FAISS index...")

    dimension = embeddings.shape[1]

    # Normalized vectors with inner product
    # provide cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("FAISS index created.")
    print("Vector dimension:", dimension)
    print("Indexed vectors:", index.ntotal)

    return index


# =====================================================
# 5. SAVE INDEX AND CHUNK METADATA
# =====================================================

def save_outputs(index, df, embeddings):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSaving FAISS index...")

    faiss.write_index(
        index,
        str(INDEX_FILE)
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
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        escapechar="\\"
    )

    config = {
        "model_name": MODEL_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
        "similarity_metric": "cosine_similarity",
        "index_type": "FAISS IndexFlatIP",
        "normalization": True,
        "device": "cpu",
        "total_vectors": int(index.ntotal),
        "total_chunks": int(len(metadata)),
        "unique_articles": int(metadata["article_id"].nunique()),
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

    print("\nAll outputs saved successfully.")

    print("FAISS index:", INDEX_FILE)
    print("Metadata:", METADATA_FILE)
    print("Configuration:", CONFIG_FILE)


# =====================================================
# 6. MAIN EXECUTION
# =====================================================

def main():

    print("=" * 60)
    print("SECTION-AWARE DENSE RETRIEVAL INDEX BUILDING")
    print("=" * 60)

    df = load_chunks()

    embeddings = generate_embeddings(df)

    index = build_faiss_index(embeddings)

    save_outputs(
        index,
        df,
        embeddings
    )

    print("\n" + "=" * 60)
    print("SECTION-AWARE DENSE INDEX BUILD COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
