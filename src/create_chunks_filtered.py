
import csv
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_ROOT / "data" / "inventory" / "article_metadata.csv"
)

SCREENING_FILE = (
    PROJECT_ROOT / "data" / "screening" / "article_screening.csv"
)

INPUT_FOLDER = PROJECT_ROOT / "data" / "cleaned_filtered_v1"

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4_chunking_comparison_filtered_v1"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

FIXED_CHUNK_SIZE = 250
FIXED_OVERLAP = 50

SENTENCE_CHUNK_SIZE = 250
SENTENCE_OVERLAP = 50

SECTION_MIN_WORDS = 50
SECTION_MAX_WORDS = 350


# ============================================================
# 2. GENERAL HELPER FUNCTIONS
# ============================================================

def normalize_value(value):
    """Convert values to clean strings."""

    if value is None:
        return ""

    return str(value).strip()


def normalize_text(text):
    """Normalize extracted text while preserving content."""

    text = unicodedata.normalize("NFC", text)

    # Normalize special spaces.
    text = re.sub(
        r"[\u00A0\u2007\u202F\u2009\u200A]",
        " ",
        text
    )

    # Remove zero-width characters and BOM.
    text = re.sub(
        r"[\u200B\u200C\u200D\uFEFF]",
        "",
        text
    )

    # Remove soft hyphens.
    text = text.replace("\u00AD", "")

    # Join words split by line-break hyphenation.
    text = re.sub(
        r"(\w+)-\s*\n\s*(\w+)",
        r"\1\2",
        text
    )

    # Normalize horizontal whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def count_words(text):
    """Count words in a text passage."""

    return len(text.split())


def load_csv(file_path):
    """Load a CSV file as a list of dictionaries."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        return list(csv.DictReader(file))


def format_page_label(page_number):
    """Create a safe page label for chunk IDs."""

    page_text = str(page_number).strip()

    if page_text.isdigit():
        return page_text.zfill(4)

    safe_text = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        page_text
    )

    return safe_text or "unknown"


# ============================================================
# 3. LOAD METADATA AND SCREENING INFORMATION
# ============================================================

def load_metadata():
    """Create an article ID to metadata lookup."""

    records = load_csv(METADATA_FILE)

    metadata_lookup = {}

    for record in records:

        article_id = normalize_value(
            record.get("article_id")
        )

        if article_id:
            metadata_lookup[article_id] = record

    return metadata_lookup


def load_screening():
    """Create an article ID to screening lookup."""

    records = load_csv(SCREENING_FILE)

    screening_lookup = {}

    for record in records:

        article_id = normalize_value(
            record.get("article_id")
        )

        if article_id:
            screening_lookup[article_id] = record

    return screening_lookup


# ============================================================
# 4. PAPER ELIGIBILITY
# ============================================================

def is_eligible(screening):
    """
    Select provisionally included and relevant papers.

    Duplicate verification is handled separately.
    Papers awaiting manual duplicate verification remain
    in the working chunking dataset.
    """

    status = normalize_value(
        screening.get("screening_status")
    ).lower()

    relevance = normalize_value(
        screening.get("relevance")
    ).lower()

    return (
        status.startswith("included")
        and relevance == "relevant"
    )


# ============================================================
# 5. EXTRACT PAGE-LEVEL TEXT FROM JSON
# ============================================================

def extract_pages(json_data):
    """
    Extract page numbers and text from the cleaned JSON.

    Supports list-based and dictionary-based page structures.
    """

    pages = []

    page_data = json_data.get("pages", [])

    if isinstance(page_data, list):

        for index, page in enumerate(page_data, start=1):

            if isinstance(page, dict):

                page_number = (
                    page.get("page")
                    or page.get("page_number")
                    or page.get("page_num")
                    or index
                )

                page_text = (
                    page.get("text")
                    or page.get("content")
                    or ""
                )

            elif isinstance(page, str):

                page_number = index
                page_text = page

            else:
                continue

            page_text = normalize_text(
                normalize_value(page_text)
            )

            if page_text:
                pages.append({
                    "page": page_number,
                    "text": page_text
                })

    elif isinstance(page_data, dict):

        for page_number, page_content in page_data.items():

            if isinstance(page_content, dict):

                page_text = (
                    page_content.get("text")
                    or page_content.get("content")
                    or ""
                )

            else:
                page_text = page_content

            page_text = normalize_text(
                normalize_value(page_text)
            )

            if page_text:
                pages.append({
                    "page": page_number,
                    "text": page_text
                })

    # Alternative structure: text stored directly.
    if not pages and isinstance(
        json_data.get("text"), str
    ):

        text = normalize_text(json_data["text"])

        if text:
            pages.append({
                "page": 1,
                "text": text
            })

    return pages


def get_extraction_method(json_data):
    """Retrieve the extraction method recorded in the JSON."""

    return (
        json_data.get("extraction_method")
        or json_data.get("method")
        or "Not recorded"
    )


# ============================================================
# 6. CHUNK RECORD CREATION
# ============================================================

def create_record(
    article_id,
    metadata,
    screening,
    extraction_method,
    page_number,
    strategy,
    chunk_number,
    text,
    section_title=""
):
    """Create a chunk record with article and page provenance."""

    page_label = format_page_label(page_number)

    return {
        "article_id": article_id,

        "chunk_id": (
            f"{article_id}_{strategy.upper()}_"
            f"page_{page_label}_"
            f"chunk_{chunk_number:04d}"
        ),

        "title": metadata.get("title", ""),
        "authors": metadata.get("authors", ""),
        "year": metadata.get("year", ""),
        "venue": metadata.get("venue", ""),
        "doi_or_url": metadata.get("doi_or_url", ""),
        "source": metadata.get("source", ""),
        "topical_category": metadata.get("topical_category", ""),
        "corpus_role": metadata.get("corpus_role", ""),
        "ingestion_status": metadata.get("ingestion_status", ""),
        "access_license": metadata.get("access_license", ""),

        "metadata_verification_status": metadata.get(
            "metadata_verification_status", ""
        ),

        "screening_status": screening.get(
            "screening_status", ""
        ),

        "relevance": screening.get("relevance", ""),
        "publication_type": screening.get(
            "publication_type", ""
        ),

        "duplicate_status": screening.get(
            "duplicate_status", ""
        ),

        "extraction_method": extraction_method,
        "page": page_number,
        "chunking_strategy": strategy,
        "section_title": section_title,
        "word_count": count_words(text),
        "text": text
    }


# ============================================================
# 7. STRATEGY 1: FIXED-SIZE CHUNKING
# ============================================================

def fixed_size_chunking(
    article_id,
    metadata,
    screening,
    extraction_method,
    page_number,
    text
):
    """Create fixed-size word chunks with overlap."""

    words = text.split()

    chunks = []

    start = 0
    chunk_number = 1

    while start < len(words):

        end = start + FIXED_CHUNK_SIZE

        chunk_text = " ".join(words[start:end])

        if chunk_text.strip():

            chunks.append(
                create_record(
                    article_id,
                    metadata,
                    screening,
                    extraction_method,
                    page_number,
                    "fixed",
                    chunk_number,
                    chunk_text
                )
            )

        if end >= len(words):
            break

        start = end - FIXED_OVERLAP
        chunk_number += 1

    return chunks


# ============================================================
# 8. STRATEGY 2: SENTENCE-AWARE CHUNKING
# ============================================================

def split_sentences(text):
    """Split text into sentence-like units."""

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9(\[])",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def sentence_aware_chunking(
    article_id,
    metadata,
    screening,
    extraction_method,
    page_number,
    text
):
    """Create chunks while preserving sentence boundaries."""

    sentences = split_sentences(text)

    chunks = []

    current_sentences = []
    current_word_count = 0

    chunk_number = 1

    for sentence in sentences:

        sentence_word_count = count_words(sentence)

        if (
            current_sentences
            and current_word_count + sentence_word_count
            > SENTENCE_CHUNK_SIZE
        ):

            chunk_text = " ".join(current_sentences)

            chunks.append(
                create_record(
                    article_id,
                    metadata,
                    screening,
                    extraction_method,
                    page_number,
                    "sentence",
                    chunk_number,
                    chunk_text
                )
            )

            # Retain previous sentences for overlap.
            overlap_sentences = []
            overlap_count = 0

            for previous_sentence in reversed(
                current_sentences
            ):

                previous_count = count_words(
                    previous_sentence
                )

                if (
                    overlap_count + previous_count
                    > SENTENCE_OVERLAP
                ):
                    break

                overlap_sentences.insert(
                    0,
                    previous_sentence
                )

                overlap_count += previous_count

            current_sentences = overlap_sentences
            current_word_count = overlap_count

            chunk_number += 1

        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    if current_sentences:

        chunk_text = " ".join(current_sentences)

        chunks.append(
            create_record(
                article_id,
                metadata,
                screening,
                extraction_method,
                page_number,
                "sentence",
                chunk_number,
                chunk_text
            )
        )

    return chunks


# ============================================================
# 9. STRATEGY 3: SECTION-AWARE CHUNKING
# ============================================================

SECTION_KEYWORDS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "literature review",
    "materials and methods",
    "methodology",
    "proposed methodology",
    "proposed method",
    "experimental setup",
    "experiments",
    "results",
    "results and discussion",
    "discussion",
    "conclusion",
    "conclusions",
    "future work",
    "references"
]


def is_section_heading(line):
    """Identify likely academic section headings."""

    cleaned = line.strip()

    if not cleaned:
        return False

    if len(cleaned) > 100:
        return False

    normalized = cleaned.lower().strip(" .:-")

    if normalized in SECTION_KEYWORDS:
        return True

    # Detect numbered headings such as 1. Introduction.
    if re.match(
        r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z &,-]{1,80}$",
        cleaned
    ):
        return True

    # Detect short uppercase headings.
    letters = re.sub(r"[^A-Za-z]", "", cleaned)

    if (
        len(letters) >= 4
        and cleaned.isupper()
        and len(cleaned.split()) <= 10
    ):
        return True

    return False


def split_into_sections(text):
    """Divide page text into likely academic sections."""

    lines = text.splitlines()

    sections = []

    current_title = "Unclassified"
    current_lines = []

    for line in lines:

        cleaned_line = line.strip()

        if is_section_heading(cleaned_line):

            if current_lines:

                section_text = "\n".join(
                    current_lines
                ).strip()

                if section_text:
                    sections.append({
                        "title": current_title,
                        "text": section_text
                    })

            current_title = cleaned_line
            current_lines = []

        else:
            current_lines.append(line)

    if current_lines:

        section_text = "\n".join(
            current_lines
        ).strip()

        if section_text:
            sections.append({
                "title": current_title,
                "text": section_text
            })

    if not sections and text.strip():

        sections.append({
            "title": "Unclassified",
            "text": text.strip()
        })

    return sections


def split_long_section(text):
    """Split long sections using sentence boundaries."""

    sentences = split_sentences(text)

    chunks = []

    current_sentences = []
    current_words = 0

    for sentence in sentences:

        sentence_words = count_words(sentence)

        if (
            current_sentences
            and current_words + sentence_words
            > SECTION_MAX_WORDS
        ):

            chunks.append(
                " ".join(current_sentences)
            )

            current_sentences = []
            current_words = 0

        current_sentences.append(sentence)
        current_words += sentence_words

    if current_sentences:

        chunks.append(
            " ".join(current_sentences)
        )

    return chunks


def section_aware_chunking(
    article_id,
    metadata,
    screening,
    extraction_method,
    page_number,
    text
):
    """Create chunks based on detected academic sections."""

    sections = split_into_sections(text)

    # Merge short sections with the following section
    # when their combined size remains reasonable.
    merged_sections = []

    for section in sections:

        if (
            merged_sections
            and count_words(
                merged_sections[-1]["text"]
            ) < SECTION_MIN_WORDS
            and (
                count_words(
                    merged_sections[-1]["text"]
                )
                + count_words(section["text"])
                <= SECTION_MAX_WORDS
            )
        ):

            merged_sections[-1]["text"] += (
                "\n" + section["text"]
            )

            merged_sections[-1]["title"] += (
                " / " + section["title"]
            )

        else:
            merged_sections.append(section.copy())

    chunks = []
    chunk_number = 1

    for section in merged_sections:

        section_text = section["text"]
        section_title = section["title"]

        if count_words(section_text) <= SECTION_MAX_WORDS:

            section_chunks = [section_text]

        else:

            section_chunks = split_long_section(
                section_text
            )

        for chunk_text in section_chunks:

            if not chunk_text.strip():
                continue

            chunks.append(
                create_record(
                    article_id,
                    metadata,
                    screening,
                    extraction_method,
                    page_number,
                    "section",
                    chunk_number,
                    chunk_text,
                    section_title
                )
            )

            chunk_number += 1

    return chunks


# ============================================================
# 10. SAVE CHUNKS - UPDATED CSV EXPORT
# ============================================================

def save_chunks(chunks, output_file):
    """
    Save chunk records safely to CSV.

    QUOTE_ALL and escapechar help handle special characters
    and quotation marks in extracted academic text.
    """

    if not chunks:
        print(
            f"No chunks generated for {output_file.name}"
        )
        return

    fieldnames = list(chunks[0].keys())

    with open(
        output_file,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            lineterminator="\n"
        )

        writer.writeheader()
        writer.writerows(chunks)

    print(
        f"Saved: {output_file.name} "
        f"({len(chunks)} chunks)"
    )


# ============================================================
# 11. MAIN CHUNKING PROCESS
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("THREE-STRATEGY CHUNKING")
    print("=" * 60)

    metadata_lookup = load_metadata()
    screening_lookup = load_screening()

    print(
        f"Metadata records: {len(metadata_lookup)}"
    )

    print(
        f"Screening records: {len(screening_lookup)}"
    )

    eligible_ids = {
        article_id
        for article_id, screening in screening_lookup.items()
        if is_eligible(screening)
        and article_id in metadata_lookup
    }

    print(
        f"Eligible provisional papers: {len(eligible_ids)}"
    )

    json_files = sorted(
        INPUT_FOLDER.glob("*.json")
    )

    print(
        f"JSON files found: {len(json_files)}"
    )

    fixed_chunks = []
    sentence_chunks = []
    section_chunks = []

    processed_articles = set()
    extraction_methods = Counter()

    skipped_items = []

    for json_file in json_files:

        article_id = json_file.stem

        if article_id not in eligible_ids:
            continue

        metadata = metadata_lookup[article_id]
        screening = screening_lookup[article_id]

        try:

            with open(
                json_file,
                "r",
                encoding="utf-8"
            ) as file:

                json_data = json.load(file)

            pages = extract_pages(json_data)

            if not pages:

                skipped_items.append(
                    f"{article_id}: No page text found"
                )

                continue

            extraction_method = get_extraction_method(
                json_data
            )

            extraction_methods[
                extraction_method
            ] += 1

            article_generated_chunks = 0

            for page_data in pages:

                page_number = page_data["page"]
                page_text = page_data["text"]

                fixed_result = fixed_size_chunking(
                    article_id,
                    metadata,
                    screening,
                    extraction_method,
                    page_number,
                    page_text
                )

                sentence_result = sentence_aware_chunking(
                    article_id,
                    metadata,
                    screening,
                    extraction_method,
                    page_number,
                    page_text
                )

                section_result = section_aware_chunking(
                    article_id,
                    metadata,
                    screening,
                    extraction_method,
                    page_number,
                    page_text
                )

                fixed_chunks.extend(fixed_result)
                sentence_chunks.extend(sentence_result)
                section_chunks.extend(section_result)

                article_generated_chunks += (
                    len(fixed_result)
                    + len(sentence_result)
                    + len(section_result)
                )

            if article_generated_chunks > 0:

                processed_articles.add(article_id)

            else:

                skipped_items.append(
                    f"{article_id}: No chunks generated"
                )

        except Exception as error:

            skipped_items.append(
                f"{article_id}: {str(error)}"
            )

    # --------------------------------------------------------
    # SAVE OUTPUT FILES
    # --------------------------------------------------------

    save_chunks(
        fixed_chunks,
        OUTPUT_FOLDER / "fixed_size_chunks.csv"
    )

    save_chunks(
        sentence_chunks,
        OUTPUT_FOLDER / "sentence_aware_chunks.csv"
    )

    save_chunks(
        section_chunks,
        OUTPUT_FOLDER / "section_aware_chunks.csv"
    )

    # --------------------------------------------------------
    # GENERATE SUMMARY
    # --------------------------------------------------------

    summary = {
        "eligible_provisional_papers": len(eligible_ids),

        "json_files_found": len(json_files),

        "articles_processed": len(processed_articles),

        "articles_without_chunks": len(
            eligible_ids - processed_articles
        ),

        "extraction_methods": dict(extraction_methods),

        "fixed_size_chunks": len(fixed_chunks),

        "sentence_aware_chunks": len(sentence_chunks),

        "section_aware_chunks": len(section_chunks),

        "skipped_items": skipped_items
    }

    summary_file = OUTPUT_FOLDER / "chunking_summary.json"

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # PRINT FINAL REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("THREE-STRATEGY CHUNKING SUMMARY")
    print("=" * 60)

    print(
        f"Eligible provisional papers: {len(eligible_ids)}"
    )

    print(
        f"JSON files found: {len(json_files)}"
    )

    print(
        f"Articles processed: {len(processed_articles)}"
    )

    print("\nExtraction methods:")

    for method, count in extraction_methods.items():

        print(f"{method}: {count}")

    print("\nGenerated chunks:")

    print(
        f"Fixed-size chunks: {len(fixed_chunks)}"
    )

    print(
        f"Sentence-aware chunks: {len(sentence_chunks)}"
    )

    print(
        f"Section-aware chunks: {len(section_chunks)}"
    )

    print(
        f"\nOutput folder: {OUTPUT_FOLDER}"
    )

    print(
        f"Skipped items or warnings: {len(skipped_items)}"
    )

    print(
        "Eligible papers without generated chunks: "
        f"{len(eligible_ids - processed_articles)}"
    )

    if skipped_items:

        print("\nSkipped items:")

        for item in skipped_items:
            print(f"- {item}")

    print("\nChunking process completed.")


# ============================================================
# 12. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
