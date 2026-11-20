import csv
import re
from pathlib import Path

from pypdf import PdfReader

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
ARTICLES_DIR = PROJECT_DIR / "Articles"
INVENTORY_FILE = PROJECT_DIR / "data" / "inventory" / "pdf_file_inventory.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "screening" / "pdf_inspection_report.csv"

# DOI pattern
DOI_PATTERN = re.compile(
    r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b',
    re.IGNORECASE
)

def clean_text(value):
    """Convert metadata values into clean strings."""
    if value is None:
        return ""
    return str(value).strip()

def inspect_pdf(article):
    """Inspect one PDF and return its preliminary information."""

    pdf_path = ARTICLES_DIR / article["filename"]

    result = {
        "article_id": article["article_id"],
        "filename": article["filename"],
        "file_size_bytes": article.get("file_size_bytes", ""),
        "pdf_title": "",
        "pdf_author": "",
        "page_count": 0,
        "pages_with_text": 0,
        "extracted_characters": 0,
        "doi": "",
        "text_preview": "",
        "inspection_status": "Pending",
        "error_message": ""
    }

    try:
        if not pdf_path.exists():
            raise FileNotFoundError("PDF file not found in Articles folder.")

        reader = PdfReader(str(pdf_path), strict=False)

        if reader.is_encrypted:
            decrypt_result = reader.decrypt("")

            if decrypt_result == 0:
                raise ValueError("PDF is encrypted and cannot be opened with an empty password.")

        result["page_count"] = len(reader.pages)

        metadata = reader.metadata

        if metadata:
            result["pdf_title"] = clean_text(metadata.title)
            result["pdf_author"] = clean_text(metadata.author)

        extracted_pages = []
        preview_pages = []

        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()

            if page_text:
                result["pages_with_text"] += 1
                result["extracted_characters"] += len(page_text)
                extracted_pages.append(page_text)

                if page_number < 3:
                    preview_pages.append(page_text)

        full_text = "\n".join(extracted_pages)

        doi_matches = DOI_PATTERN.findall(full_text)

        if doi_matches:
            result["doi"] = doi_matches[0].rstrip(".,;")

        result["text_preview"] = " ".join(
            "\n".join(preview_pages).split()
        )[:1800]

        if result["extracted_characters"] == 0:
            result["inspection_status"] = "No extractable text"
        else:
            result["inspection_status"] = "Inspected"

    except Exception as error:
        result["inspection_status"] = "Error"
        result["error_message"] = str(error)

    return result


def main():
    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {INVENTORY_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(INVENTORY_FILE, "r", encoding="utf-8-sig", newline="") as file:
        inventory = list(csv.DictReader(file))

    results = []

    print(f"Starting PDF inspection for {len(inventory)} files...\n")

    for number, article in enumerate(inventory, start=1):
        result = inspect_pdf(article)
        results.append(result)

        print(
            f"[{number}/{len(inventory)}] "
            f"{result['article_id']} - {result['inspection_status']}"
        )

    fieldnames = list(results[0].keys()) if results else []

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    inspected = sum(
        row["inspection_status"] == "Inspected" for row in results
    )

    no_text = sum(
        row["inspection_status"] == "No extractable text" for row in results
    )

    errors = sum(
        row["inspection_status"] == "Error" for row in results
    )

    print("\n========== INSPECTION SUMMARY ==========")
    print(f"Total PDFs: {len(results)}")
    print(f"Inspected with extractable text: {inspected}")
    print(f"No extractable text: {no_text}")
    print(f"Errors: {errors}")
    print(f"Report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
