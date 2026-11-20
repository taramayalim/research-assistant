
import csv
from io import StringIO
from pathlib import Path


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4_chunking_comparison_normalized"
    / "fixed_size_chunks.csv"
)

TARGET_ARTICLES = [
    "A023", "A006", "A019", "A095",
    "A148", "A072", "A113", "A138"
]

KEYWORDS = [
    "proposed",
    "dataset",
    "accuracy",
    "precision",
    "recall",
    "f1-score",
    "convolutional neural network",
    "deep learning",
    "classification",
    "model",
    "plant disease"
]

MAX_PASSAGES_PER_ARTICLE = 3
MAX_TEXT_LENGTH = 1200


# ============================================================
# 2. LOAD AND CLEAN CSV
# ============================================================

def load_chunks(file_path):
    """Load chunk records while handling NUL characters."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        content = file.read().replace("\x00", "")

    reader = csv.DictReader(StringIO(content))

    required_columns = {
        "article_id",
        "chunk_id",
        "title",
        "page",
        "word_count",
        "section_title",
        "text"
    }

    missing_columns = required_columns - set(reader.fieldnames or [])

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing_columns)}"
        )

    return list(reader)


# ============================================================
# 3. FIND RELEVANT PASSAGES
# ============================================================

def find_matching_chunks(article_chunks):
    """Find chunks containing research-related keywords."""

    matching_chunks = []

    for row in article_chunks:
        text = (row.get("text") or "").lower()

        if any(keyword in text for keyword in KEYWORDS):
            matching_chunks.append(row)

    return matching_chunks


# ============================================================
# 4. DISPLAY ARTICLE AND PASSAGE DETAILS
# ============================================================

def display_article(article_id, rows):

    article_chunks = [
        row for row in rows
        if row["article_id"] == article_id
    ]

    print("\n" + "=" * 90)
    print(f"ARTICLE ID: {article_id}")

    if not article_chunks:
        print("No chunks found.")
        return

    print(f"TITLE: {article_chunks[0]['title']}")
    print(f"TOTAL CHUNKS: {len(article_chunks)}")

    matching_chunks = find_matching_chunks(article_chunks)

    print(f"MATCHING CHUNKS: {len(matching_chunks)}")

    if not matching_chunks:
        print("No matching passages found.")
        return

    for row in matching_chunks[:MAX_PASSAGES_PER_ARTICLE]:

        print("\nCHUNK ID:", row["chunk_id"])
        print("PAGE:", row["page"])
        print("WORD COUNT:", row["word_count"])
        print("SECTION:", row["section_title"] or "Not identified")

        print("\nRESEARCH PASSAGE:")
        print((row.get("text") or "")[:MAX_TEXT_LENGTH])

        print("-" * 90)


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

def main():

    print("Loading research chunks...")

    rows = load_chunks(CHUNK_FILE)

    print(f"Total chunks loaded: {len(rows)}")
    print("\nInspecting selected research articles...")

    for article_id in TARGET_ARTICLES:
        display_article(article_id, rows)

    print("\nInspection completed successfully.")


if __name__ == "__main__":
    main()
