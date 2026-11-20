
import json
import logging
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "Articles"
INVENTORY_FILE = BASE_DIR / "data" / "inventory" / "article_inventory.csv"
OUTPUT_DIR = BASE_DIR / "data" / "extracted"
LOG_DIR = BASE_DIR / "logs" / "ingestion"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log extraction errors
logging.basicConfig(
    filename=LOG_DIR / "extraction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

def extract_pdf(pdf_path):
    """Extract text from each page of a PDF."""
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


def main():
    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Inventory not found: {INVENTORY_FILE}"
        )

    inventory = pd.read_csv(INVENTORY_FILE, dtype=str).fillna("")

    required_columns = {"article_id", "filename"}
    if not required_columns.issubset(inventory.columns):
        raise ValueError(
            "Inventory must contain article_id and filename columns."
        )

    for column in [
        "ingestion_status",
        "page_count",
        "pages_with_text",
        "extracted_characters",
    ]:
        if column not in inventory.columns:
            inventory[column] = ""

    success_count = 0
    failed_count = 0

    for index, article in inventory.iterrows():
        article_id = article["article_id"].strip()
        filename = article["filename"].strip()
        pdf_path = PDF_DIR / filename

        if not article_id or not filename:
            inventory.at[index, "ingestion_status"] = "failed_missing_metadata"
            logging.error("Missing article ID or filename at row %s", index + 1)
            failed_count += 1
            continue

        if not pdf_path.is_file():
            inventory.at[index, "ingestion_status"] = "file_not_found"
            logging.error("PDF not found: %s", pdf_path)
            failed_count += 1
            continue

        try:
            pages = extract_pdf(pdf_path)
            total_text = "\n".join(page["text"] for page in pages)
            pages_with_text = sum(bool(page["text"]) for page in pages)

            record = {
                "article_id": article_id,
                "filename": filename,
                "page_count": len(pages),
                "pages": pages,
            }

            output_file = OUTPUT_DIR / f"{article_id}.json"
            output_file.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            if not total_text.strip():
                status = "no_extractable_text"
            elif pages_with_text < len(pages):
                status = "extracted_with_empty_pages"
            else:
                status = "extracted"

            inventory.at[index, "ingestion_status"] = status
            inventory.at[index, "page_count"] = str(len(pages))
            inventory.at[index, "pages_with_text"] = str(pages_with_text)
            inventory.at[index, "extracted_characters"] = str(len(total_text))

            if status == "no_extractable_text":
                logging.warning("No extractable text: %s", filename)
            else:
                success_count += 1

            print(
                f"[{index + 1}/{len(inventory)}] "
                f"{article_id}: {status} "
                f"({len(pages)} pages)"
            )

        except Exception as error:
            inventory.at[index, "ingestion_status"] = "failed"
            logging.exception("Failed to extract %s", filename)
            failed_count += 1
            print(f"ERROR: {filename}: {error}")

    # Save the updated inventory
    inventory.to_csv(INVENTORY_FILE, index=False, encoding="utf-8-sig")

    print("\nExtraction finished.")
    print(f"Inventory records: {len(inventory)}")
    print(f"Processed successfully: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Extracted JSON files: {OUTPUT_DIR}")
    print(f"Updated inventory: {INVENTORY_FILE}")
    print(f"Error log: {LOG_DIR / 'extraction.log'}")


if __name__ == "__main__":
    main()
