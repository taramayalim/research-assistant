
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

PROCESSED = (
    ROOT / "data" / "processed" / "phase3_current_corpus"
)

FILES = [
    PROCESSED / "chunks_fixed.csv",
    PROCESSED / "chunks_sentence.csv",
]

REQUIRED_FIELDS = [
    "article_id", "chunk_id", "title", "authors", "year",
    "venue", "doi_or_url", "source", "topical_category",
    "corpus_role", "ingestion_status", "access_license",
    "screening_status", "page_number",
    "chunking_strategy", "chunk_text",
]


def validate_file(path, expected_strategy):

    print(f"\nChecking: {path.name}")
    print("=" * 55)

    if not path.exists():
        print("ERROR: File not found.")
        return

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []

        missing_fields = [
            field for field in REQUIRED_FIELDS
            if field not in fields
        ]

        rows = list(reader)

    chunk_ids = [
        row.get("chunk_id", "").strip()
        for row in rows
    ]

    duplicate_ids = (
        len(chunk_ids) - len(set(chunk_ids))
    )

    missing_ids = sum(not value for value in chunk_ids)

    missing_text = sum(
        not row.get("chunk_text", "").strip()
        for row in rows
    )

    missing_article_ids = sum(
        not row.get("article_id", "").strip()
        for row in rows
    )

    missing_pages = sum(
        not row.get("page_number", "").strip()
        for row in rows
    )

    articles = {
        row.get("article_id", "").strip()
        for row in rows
        if row.get("article_id", "").strip()
    }

    strategies = Counter(
        row.get("chunking_strategy", "").strip()
        for row in rows
    )

    invalid_pages = 0
    invalid_word_counts = 0
    invalid_screening = 0
    metadata_missing = 0

    for row in rows:

        try:
            if int(row.get("page_number", "")) < 1:
                invalid_pages += 1
        except (ValueError, TypeError):
            invalid_pages += 1

        if len(row.get("chunk_text", "").split()) == 0:
            invalid_word_counts += 1

        if row.get("screening_status", "").strip() != "Included - provisional":
            invalid_screening += 1

        if not all(
            row.get(field, "").strip()
            for field in [
                "title", "authors", "year",
                "venue", "doi_or_url"
            ]
        ):
            metadata_missing += 1

    wrong_strategy = sum(
        count for strategy, count in strategies.items()
        if strategy != expected_strategy
    )

    print("Total chunks:", len(rows))
    print("Distinct articles:", len(articles))
    print("Missing required columns:", missing_fields)
    print("Duplicate chunk IDs:", duplicate_ids)
    print("Missing chunk IDs:", missing_ids)
    print("Missing article IDs:", missing_article_ids)
    print("Missing page numbers:", missing_pages)
    print("Invalid page numbers:", invalid_pages)
    print("Empty chunk texts:", missing_text)
    print("Invalid screening records:", invalid_screening)
    print("Rows with incomplete bibliographic metadata:", metadata_missing)
    print("Unexpected chunking strategy rows:", wrong_strategy)

    print("\nChunking strategy counts:", dict(strategies))

    if rows:
        print("\nSample chunk:")
        for field in [
            "article_id", "chunk_id", "title",
            "page_number", "chunking_strategy"
        ]:
            print(f"{field}: {rows[0].get(field, '')}")

        print(
            "Text preview:",
            rows[0].get("chunk_text", "")[:250]
        )

    problems = (
        missing_fields or duplicate_ids or missing_ids
        or missing_text or missing_article_ids
        or missing_pages or invalid_pages
        or invalid_word_counts or invalid_screening
        or wrong_strategy
    )

    if problems:
        print("\nRESULT: Issues found. Review before proceeding.")
    else:
        print("\nRESULT: Structural validation passed.")

    if metadata_missing:
        print(
            "NOTE: Some bibliographic metadata is incomplete; "
            "review these records before final indexing."
        )


if __name__ == "__main__":

    validate_file(
        FILES[0], "fixed"
    )

    validate_file(
        FILES[1], "sentence"
    )
