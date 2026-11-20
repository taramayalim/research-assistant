
from pathlib import Path
import pickle
import re
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent

INDEX_DIR = ROOT / "data" / "indexes"
QUESTION = "What is the initial custom CNN accuracy reported in the study?"

# Load metadata
dense_meta = pd.read_csv(
    INDEX_DIR / "fixed_size_dense" / "chunk_metadata.csv"
)

bm25_meta = pd.read_csv(
    INDEX_DIR / "fixed_size_bm25" / "chunk_metadata.csv"
)

# Load BM25 index
with open(
    INDEX_DIR / "fixed_size_bm25" / "bm25_index.pkl",
    "rb"
) as f:
    bm25 = pickle.load(f)

# Load dense index and embedding model
dense_index = faiss.read_index(
    str(INDEX_DIR / "fixed_size_dense" / "faiss.index")
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# BM25 retrieval
def tokenize(text):
    return re.findall(
        r"[a-z0-9]+(?:[._+-][a-z0-9]+)*",
        str(text).lower()
    )

bm25_scores = bm25.get_scores(tokenize(QUESTION))
bm25_top = bm25_scores.argsort()[::-1][:10]

# Dense retrieval
query_embedding = model.encode(
    [QUESTION],
    normalize_embeddings=True
)

dense_scores, dense_ids = dense_index.search(
    query_embedding.astype("float32"), 10
)

print("\n========== BM25 TOP 10 ==========")

for rank, idx in enumerate(bm25_top, 1):
    row = bm25_meta.iloc[idx]
    print(
        rank,
        row["article_id"],
        row["chunk_id"],
        "Score:", round(float(bm25_scores[idx]), 4)
    )

print("\n========== DENSE TOP 10 ==========")

for rank, idx in enumerate(dense_ids[0], 1):
    row = dense_meta.iloc[idx]
    print(
        rank,
        row["article_id"],
        row["chunk_id"],
        "Score:", round(float(dense_scores[0][rank-1]), 4)
    )

print("\n========== TARGET CHUNK ==========")

target = "A006_FIXED_page_0001_chunk_0002"

for name, metadata in [
    ("BM25", bm25_meta),
    ("Dense", dense_meta)
]:
    matches = metadata.index[metadata["chunk_id"] == target].tolist()
    print(name, "target index:", matches)
