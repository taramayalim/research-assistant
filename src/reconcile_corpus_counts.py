import pandas as pd
from pathlib import Path

metadata = pd.read_csv("data/inventory/article_metadata.csv")
screening = pd.read_csv("data/screening/article_screening.csv")
dense = pd.read_csv("data/indexes/fixed_size_dense/chunk_metadata.csv")
bm25 = pd.read_csv("data/indexes/fixed_size_bm25/chunk_metadata.csv")

def unique_ids(df):
    return set(df["article_id"].dropna().astype(str).str.strip())

print("========== 1. CORPUS ROLE COUNTS ==========")
print(metadata["corpus_role"].value_counts(dropna=False).to_string())

print("\n========== 2. SCREENING STATUS BY RELEVANCE ==========")
print(pd.crosstab(
    screening["relevance"],
    screening["screening_status"],
    margins=True
).to_string())

print("\n========== 3. INDEXED ARTICLE COUNTS ==========")
for name, df in [("Dense", dense), ("BM25", bm25)]:
    ids = unique_ids(df)
    print(f"{name}: {len(ids)} unique articles, {len(df)} chunks")

print("\n========== 4. SCREENED RELEVANT BUT NOT INDEXED ==========")
relevant = screening[
    screening["relevance"].astype(str).str.strip().str.lower() == "relevant"
]
indexed = unique_ids(dense)
not_indexed = relevant[~relevant["article_id"].astype(str).str.strip().isin(indexed)]

print("Relevant papers:", len(relevant))
print("Relevant papers not in dense index:", len(not_indexed))
print(not_indexed[
    ["article_id", "filename", "screening_status",
     "duplicate_status", "exclusion_reason"]
].to_string(index=False))

print("\n========== 5. INDEXED BUT NOT SCREENED AS RELEVANT ==========")
screened_ids = unique_ids(relevant)
extra_indexed = dense[
    ~dense["article_id"].astype(str).str.strip().isin(screened_ids)
]
print("Chunks:", len(extra_indexed))
print("Article IDs:", sorted(unique_ids(extra_indexed)))

print("\n========== 6. METADATA AND SCREENING RECONCILIATION ==========")
print("Metadata records:", len(metadata))
print("Unique metadata IDs:", metadata["article_id"].nunique())
print("Screening records:", len(screening))
print("Unique screening IDs:", screening["article_id"].nunique())
print("Dense/BM25 article ID sets match:",
      unique_ids(dense) == unique_ids(bm25))
