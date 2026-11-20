
import csv
import json
import logging
import re
from pathlib import Path

from pypdf import PdfReader

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "Articles"
INVENTORY_FILE = (
    BASE_DIR / "data" / "inventory" / "pdf_file_inventory.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "extracted"
OCR_FILE = OUTPUT_DIR / "A023_ocr.txt"

REPORT_DIR = BASE_DIR / "data" / "reports"
REPORT_FILE = REPORT_DIR / "extraction_report.csv"

LOG_DIR = BASE_DIR / "logs" / "ingestion"
LOG_FILE = LOG_DIR / "extraction.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOGGING
# --------------------------------------------------

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)


# --------------------------------------------------
# 3. NORMAL PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_pdf(pdf_path):
    """Extract text from each PDF page."""

    reader = PdfReader(str(pdf_path), strict=False)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:
            logging.warning(
                "%s, page %s: %s",
                pdf_path.name,
                page_number,
                error,
            )
            text = ""

        pages.append({
            "page_number": page_number,
            "text": text.strip(),
        })

    return pages


# --------------------------------------------------
# 4. A023 OCR TEXT EXTRACTION
# --------------------------------------------------

def extract_a023_ocr():
    """Read A023's existing OCR text and separate its pages."""

    if not OCR_FILE.exists():
        raise FileNotFoundError(
            f"A023 OCR file not found: {OCR_FILE}"
        )

    content = OCR_FILE.read_text(encoding="utf-8")

    marker_pattern = re.compile(
        r"(?m)^={5,}\r?\n"
        r"ARTICLE ID:\s*A023\s*\|\s*PAGE:\s*(\d+)\r?\n"
        r"={5,}\s*$"
    )

    matches = list(marker_pattern.finditer(content))

    if not matches:
        raise ValueError("No A023 OCR page markers were found.")

    pages = []

    for index, match in enumerate(matches):
        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(content)

        page_text = content[start:end].strip()

        pages.append({
            "page_number": int(match.group(1)),
            "text": page_text,
        })

    # Check that pages are numbered consecutively.
    page_numbers = [page["page_number"] for page in pages]

    if page_numbers != list(range(1, len(pages) + 1)):
        raise ValueError(
            f"Unexpected A023 page numbering: {page_numbers}"
        )

    return pages


# --------------------------------------------------
# 5. SAVE EXTRACTION AS JSON
# --------------------------------------------------

def save_json(article_id, filename, pages, method):
    """Save one article's extracted text in JSON format."""

    record = {
        "article_id": article_id,
        "filename": filename,
        "extraction_method": method,
        "page_count": len(pages),
        "pages": pages,
    }

    output_file = OUTPUT_DIR / f"{article_id}.json"

    output_file.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_file


# --------------------------------------------------
# 6. MAIN EXTRACTION PROCESS
# --------------------------------------------------

def main():

    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Inventory not found: {INVENTORY_FILE}"
        )

    with INVENTORY_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        inventory = list(csv.DictReader(file))

    if not inventory:
        raise ValueError("The PDF inventory is empty.")

    required_columns = {"article_id", "filename"}

    if not required_columns.issubset(inventory[0].keys()):
        raise ValueError(
            "Inventory must contain article_id and filename."
        )

    # Check for duplicate article IDs.
    article_ids = [
        row["article_id"].strip() for row in inventory
    ]

    if len(article_ids) != len(set(article_ids)):
        raise ValueError("Duplicate article IDs found in inventory.")

    results = []
    success_count = 0
    failed_count = 0
    empty_count = 0
    skipped_count = 0

    print(f"Starting extraction for {len(inventory)} papers...\n")

    for index, article in enumerate(inventory, start=1):

        article_id = article["article_id"].strip()
        filename = article["filename"].strip()

        status = ""
        method = ""
        page_count = 0
        pages_with_text = 0
        extracted_characters = 0
        error_message = ""

        output_file = OUTPUT_DIR / f"{article_id}.json"

        try:

            if not article_id or not filename:
                raise ValueError("Missing article ID or filename.")

            # Never overwrite an existing JSON file.
            if output_file.exists():
                status = "skipped_existing_output"
                skipped_count += 1

            else:

                # A023 uses its recovered OCR text.
                if article_id == "A023":
                    pages = extract_a023_ocr()
                    method = "OCR - Tesseract"

                else:
                    pdf_path = PDF_DIR / filename

                    if not pdf_path.is_file():
                        raise FileNotFoundError(
                            f"PDF not found: {pdf_path}"
                        )

                    pages = extract_pdf(pdf_path)
                    method = "pypdf"

                page_count = len(pages)

                pages_with_text = sum(
                    bool(page["text"]) for page in pages
                )

                extracted_characters = sum(
                    len(page["text"]) for page in pages
                )

                if extracted_characters == 0:
                    status = "no_extractable_text"
                    empty_count += 1

                elif pages_with_text < page_count:
                    status = "extracted_with_empty_pages"

                else:
                    status = "extracted"

                save_json(
                    article_id,
                    filename,
                    pages,
                    method,
                )

                if status == "extracted":
                    success_count += 1
                elif status == "extracted_with_empty_pages":
                    success_count += 1

            print(
                f"[{index}/{len(inventory)}] "
                f"{article_id}: {status}"
            )

        except Exception as error:

            status = "failed"
            error_message = str(error)
            failed_count += 1

            logging.exception(
                "Extraction failed for %s", article_id
            )

            print(
                f"[{index}/{len(inventory)}] "
                f"{article_id}: FAILED - {error}"
            )

        results.append({
            "article_id": article_id,
            "filename": filename,
            "extraction_method": method,
            "page_count": page_count,
            "pages_with_text": pages_with_text,
            "extracted_characters": extracted_characters,
            "ingestion_status": status,
            "error_message": error_message,
        })

    # --------------------------------------------------
    # 7. SAVE SEPARATE EXTRACTION REPORT
    # --------------------------------------------------

    fieldnames = [
        "article_id",
        "filename",
        "extraction_method",
        "page_count",
        "pages_with_text",
        "extracted_characters",
        "ingestion_status",
        "error_message",
    ]

    with REPORT_FILE.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(results)

    # --------------------------------------------------
    # 8. FINAL SUMMARY
    # --------------------------------------------------

    json_count = len(
        list(OUTPUT_DIR.glob("A*.json"))
    )

    print("\n========== EXTRACTION SUMMARY ==========")
    print(f"Inventory records: {len(inventory)}")
    print(f"Successfully extracted: {success_count}")
    print(f"No extractable text: {empty_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped existing JSON: {skipped_count}")
    print(f"JSON files currently present: {json_count}")
    print(f"Extraction report: {REPORT_FILE}")
    print(f"Error log: {LOG_FILE}")
    print("========================================")


if __name__ == "__main__":
    main()
