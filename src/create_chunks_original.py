
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "data" / "inventory" / "article_inventory.csv"
EXTRACTED = ROOT / "data" / "extracted"
OUTPUT = ROOT / "data" / "processed"

CHUNK_WORDS = 250
OVERLAP_WORDS = 50

OUTPUT.mkdir(parents=True, exist_ok=True)

OUTPUT_FIELDS = [
    "article_id", "chunk_id", "title", "authors", "year", "venue",
    "doi_or_stable_url", "doi", "stable_url", "source",
    "topical_category", "ingestion_status", "access_license",
    "pdf_filename", "paper_type", "plant_crop", "disease",
    "detection_methods", "dataset", "key_findings", "limitations",
    "screening_status", "page_number", "chunking_strategy", "chunk_text",
]


def clean_text(value):
    """Convert values to safe UTF-8 text without losing normal Unicode."""
    if value is None:
        return ""

    text = str(value)

    # Replace invalid Unicode surrogate characters.
    text = text.encode("utf-8", errors="replace").decode("utf-8")

    # Remove null bytes and disallowed control characters.
    text = "".join(
        char for char in text
        if char in "\n\r\t" or ord(char) >= 32
    )

    return text.strip()


def get_value(row, *names):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return clean_text(value)
    return ""


def is_excluded(row):
    screening = get_value(row, "screening_status").lower()
    eligibility = get_value(row, "eligibility").lower()
    duplicate = get_value(row, "duplicate_status").lower()
    ingestion = get_value(row, "ingestion_status").lower()

    exclusion_terms = ("exclude", "ineligible")
    duplicate_terms = {"duplicate", "excluded_duplicate"}

    return (
        any(term in screening for term in exclusion_terms)
        or any(term in eligibility for term in exclusion_terms)
        or duplicate in duplicate_terms
        or ingestion in {"excluded", "failed"}
    )


def build_metadata(row, article_id, extracted_data):
    doi = get_value(row, "doi")
    url = get_value(row, "stable_url", "paper_url")

    return {
        "article_id": clean_text(article_id),
        "title": get_value(row, "title") or clean_text(
            extracted_data.get("title", "")
        ),
        "authors": get_value(row, "authors"),
        "year": get_value(row, "year", "publication_year"),
        "venue": get_value(row, "venue", "journal_conference"),
        "doi_or_stable_url": doi or url,
        "doi": doi,
        "stable_url": url,
        "source": get_value(row, "source"),
        "topical_category": get_value(row, "topical_category"),
        "ingestion_status": get_value(row, "ingestion_status"),
        "access_license": get_value(row, "access_license"),
        "pdf_filename": get_value(row, "filename", "pdf_filename"),
        "paper_type": get_value(row, "paper_type"),
        "plant_crop": get_value(row, "plant_crop", "crop"),
        "disease": get_value(row, "disease", "diseases"),
        "detection_methods": get_value(row, "detection_methods"),
        "dataset": get_value(row, "dataset"),
        "key_findings": get_value(row, "key_findings"),
        "limitations": get_value(row, "limitations"),
        "screening_status": get_value(row, "screening_status"),
    }


def fixed_chunks(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = clean_text(text).split()

    if not words:
        return []

    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError(
            "Chunk size must be positive and overlap must be "
            "non-negative and smaller than chunk size."
        )

    chunks = []
    step = size - overlap

    for start in range(0, len(words), step):
        part = words[start:start + size]
        if not part:
            break

        chunks.append(" ".join(part))

        if start + size >= len(words):
            break

    return chunks


def sentence_chunks(text, size=CHUNK_WORDS):
    text = re.sub(r"\s+", " ", clean_text(text)).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    current_words = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        word_count = len(sentence.split())

        if word_count > size:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0

            chunks.extend(fixed_chunks(sentence, size, 0))
            continue

        if current and current_words + word_count > size:
            chunks.append(" ".join(current))

            previous_sentence = current[-1]
            current = [previous_sentence]
            current_words = len(previous_sentence.split())

            if current_words + word_count > size:
                current = []
                current_words = 0

        current.append(sentence)
        current_words += word_count

    if current:
        chunks.append(" ".join(current))

    return chunks


def write_csv_safely(path, rows):
    """Write robust UTF-8 CSV, handling quotes and special characters."""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_FIELDS,
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            doublequote=True,
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            safe_row = {
                field: clean_text(row.get(field, ""))
                for field in OUTPUT_FIELDS
            }

            try:
                writer.writerow(safe_row)
            except (csv.Error, TypeError, ValueError) as error:
                print(f"\nCSV writing failed: {path}")
                print(f"Row number: {index}")
                print(f"Chunk ID: {safe_row.get('chunk_id', 'Unknown')}")
                print(f"Error: {error}")
                raise


def main():
    if not INVENTORY.exists():
        raise FileNotFoundError(f"Inventory not found: {INVENTORY}")

    if not EXTRACTED.exists():
        raise FileNotFoundError(
            f"Extracted JSON folder not found: {EXTRACTED}"
        )

    with INVENTORY.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        inventory_rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "article_id" not in fieldnames:
        raise ValueError("Inventory must contain an article_id column.")

    metadata_by_id = {
        get_value(row, "article_id"): row
        for row in inventory_rows
        if get_value(row, "article_id")
    }

    outputs = {
        "fixed": OUTPUT / "chunks_fixed.csv",
        "sentence": OUTPUT / "chunks_sentence.csv",
    }

    records = {"fixed": [], "sentence": []}
    skipped = []
    processed_articles = set()

    json_files = sorted(EXTRACTED.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {EXTRACTED}")

    for json_path in json_files:
        try:
            data = json.loads(
                json_path.read_text(encoding="utf-8")
            )

            article_id = clean_text(
                data.get("article_id", json_path.stem)
            )

            row = metadata_by_id.get(article_id)

            if row is None:
                skipped.append((article_id, "No inventory record"))
                continue

            if is_excluded(row):
                skipped.append(
                    (article_id, "Excluded, failed, or duplicate")
                )
                continue

            pages = data.get("pages", [])

            if not isinstance(pages, list):
                skipped.append((article_id, "Invalid pages structure"))
                continue

            article_has_text = False
            article_chunk_counts = {"fixed": 0, "sentence": 0}
            metadata = build_metadata(row, article_id, data)

            for index, page in enumerate(pages, start=1):
                if not isinstance(page, dict):
                    continue

                page_text = clean_text(page.get("text", ""))

                if not page_text:
                    continue

                article_has_text = True
                page_number = page.get("page_number", index)

                strategies = {
                    "fixed": fixed_chunks(page_text),
                    "sentence": sentence_chunks(page_text),
                }

                for strategy, chunk_list in strategies.items():
                    for number, chunk_text in enumerate(
                        chunk_list, start=1
                    ):
                        record = metadata.copy()
                        record.update({
                            "chunk_id": (
                                f"{article_id}_{strategy.upper()}_"
                                f"P{page_number}_C{number:04d}"
                            ),
                            "page_number": page_number,
                            "chunking_strategy": strategy,
                            "chunk_text": clean_text(chunk_text),
                        })

                        records[strategy].append(record)
                        article_chunk_counts[strategy] += 1

            if article_has_text:
                processed_articles.add(article_id)

                for strategy, count in article_chunk_counts.items():
                    if count == 0:
                        skipped.append((
                            article_id,
                            f"Text found but no {strategy} chunks",
                        ))
            else:
                skipped.append((article_id, "No usable page text"))

        except Exception as error:
            skipped.append((
                json_path.stem,
                f"{type(error).__name__}: {error}",
            ))

    for strategy, output_path in outputs.items():
        write_csv_safely(output_path, records[strategy])

    print("\nCHUNKING SUMMARY")
    print("----------------")
    print("Inventory records:", len(inventory_rows))
    print("JSON files found:", len(json_files))
    print("Articles with usable text:", len(processed_articles))
    print("Fixed-size chunks:", len(records["fixed"]))
    print("Sentence-aware chunks:", len(records["sentence"]))

    print("\nCreated files:")
    for output_path in outputs.values():
        print(output_path)

    if skipped:
        print("\nSkipped items or warnings:", len(skipped))
        for article_id, reason in skipped:
            print(f"{article_id}: {reason}")
    else:
        print("\nNo skipped items or warnings.")

    print(
        "\nNext: validate chunk counts and metadata before "
        "building retrieval indexes."
    )


if __name__ == "__main__":
    main()
