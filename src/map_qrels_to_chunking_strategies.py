import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

CHUNK_DIR = (
    ROOT / "data" / "processed"
    / "phase4_chunking_comparison_filtered_v1"
)

QRELS_FILE = ROOT / "data" / "evaluation" / "qrels.csv"

OUTPUT_DIR = ROOT / "data" / "evaluation" / "chunking_mapping"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "fixed": "fixed_size_chunks.csv",
    "sentence": "sentence_aware_chunks.csv",
    "section": "section_aware_chunks.csv"
}


def tokenize(text):
    return re.findall(
        r"[a-z0-9]+",
        str(text).lower()
    )


def calculate_scores(source_text, candidate_text):
    source_tokens = tokenize(source_text)
    candidate_tokens = tokenize(candidate_text)

    if not source_tokens or not candidate_tokens:
        return 0.0, 0.0, 0.0

    source_counts = Counter(source_tokens)
    candidate_counts = Counter(candidate_tokens)

    overlap = sum(
        (source_counts & candidate_counts).values()
    )

    coverage = overlap / len(source_tokens)

    union = sum(
        (source_counts | candidate_counts).values()
    )

    jaccard = overlap / union if union else 0.0

    sequence_ratio = SequenceMatcher(
        None,
        " ".join(source_tokens),
        " ".join(candidate_tokens)
    ).ratio()

    return coverage, jaccard, sequence_ratio


print("Loading qrels and chunk datasets...")

qrels = pd.read_csv(QRELS_FILE).fillna("")

chunks = {}

for strategy, filename in FILES.items():
    df = pd.read_csv(CHUNK_DIR / filename).fillna("")
    df["article_id"] = df["article_id"].astype(str)
    df["page"] = df["page"].astype(str)
    df["chunk_id"] = df["chunk_id"].astype(str)
    chunks[strategy] = df

fixed = chunks["fixed"].set_index("chunk_id", drop=False)

records = []
missing_originals = []

print("Generating candidate mappings...")

for _, judgment in qrels.iterrows():

    query_id = str(judgment["query_id"])
    original_id = str(judgment["chunk_id"])
    grade = judgment["relevance_grade"]

    if original_id not in fixed.index:
        missing_originals.append(original_id)
        continue

    source = fixed.loc[original_id]

    if isinstance(source, pd.DataFrame):
        source = source.iloc[0]

    article_id = str(source["article_id"])
    page = str(source["page"])
    source_text = str(source["text"])

    for strategy in ["sentence", "section"]:

        candidates = chunks[strategy]

        candidates = candidates.loc[
            (candidates["article_id"] == article_id)
            & (candidates["page"] == page)
        ]

        scored = []

        for _, candidate in candidates.iterrows():

            coverage, jaccard, sequence_ratio = (
                calculate_scores(
                    source_text,
                    candidate["text"]
                )
            )

            scored.append({
                "query_id": query_id,
                "original_chunk_id": original_id,
                "original_grade": grade,
                "strategy": strategy,
                "article_id": article_id,
                "page": page,
                "candidate_chunk_id": candidate["chunk_id"],
                "source_coverage": round(coverage, 4),
                "token_jaccard": round(jaccard, 4),
                "sequence_ratio": round(sequence_ratio, 4),
                "source_text": source_text,
                "candidate_text": candidate["text"]
            })

        scored.sort(
            key=lambda x: (
                x["source_coverage"],
                x["token_jaccard"],
                x["sequence_ratio"]
            ),
            reverse=True
        )

        for rank, item in enumerate(scored[:3], start=1):
            item["candidate_rank"] = rank
            records.append(item)


if missing_originals:
    print("\nERROR: Original chunks not found:")
    for item in missing_originals:
        print(item)
    raise SystemExit(1)


output = pd.DataFrame(records)

output_file = OUTPUT_DIR / "qrels_mapping_candidates.csv"

output.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nMapping candidate generation complete.")
print("Original judgments:", len(qrels))
print("Candidate records:", len(output))
print("Output:", output_file)

print("\nCandidates per strategy:")
print(output.groupby("strategy").size().to_string())

print("\nBest candidate coverage summary:")
print(
    output[output["candidate_rank"] == 1]
    .groupby("strategy")["source_coverage"]
    .agg(["count", "mean", "min", "max"])
    .round(4)
    .to_string()
)
