
import csv
import re
from pathlib import Path

import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_FOLDER = PROJECT_ROOT / "Articles"
METADATA_FILE = PROJECT_ROOT / "data" / "inventory" / "article_metadata.csv"

# Articles and keywords needed to verify Q001-Q008
TARGETS = {
    "A006": [
        "30,945", "30945", "eight plant",
        "35 disease", "95.62"
    ],
    "A019": [
        "ResNet-152", "ResNet152", "Inception-v3",
        "InceptionV3", "97.81", "97.48"
    ],
    "A072": [
        "9,127", "9127", "VGG16", "VGG19",
        "0.97", "97%"
    ]
}


def find_matches(text, keywords):
    matches = []

    for keyword in keywords:
        pattern = re.escape(keyword)
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(keyword)

    return matches


def main():
    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        metadata = list(csv.DictReader(file))

    print("\nORIGINAL PDF EVIDENCE EXTRACTION")
    print("=" * 90)

    for article_id, keywords in TARGETS.items():

        records = [
            row for row in metadata
            if row["article_id"].strip() == article_id
        ]

        if not records:
            print(f"\n{article_id}: Metadata not found.")
            continue

        filename = records[0]["filename"]
        pdf_path = PDF_FOLDER / filename

        print(f"\n\nARTICLE ID: {article_id}")
        print(f"TITLE: {records[0]['title']}")
        print(f"PDF: {filename}")
        print("-" * 90)

        if not pdf_path.exists():
            print(f"PDF NOT FOUND: {pdf_path}")
            continue

        try:
            document = fitz.open(pdf_path)
            found_pages = 0

            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text")
                matches = find_matches(page_text, keywords)

                if matches:
                    found_pages += 1

                    print(f"\nPAGE: {page_number}")
                    print(f"KEYWORDS FOUND: {', '.join(matches)}")
                    print("-" * 70)

                    # Display extracted page text for verification
                    print(page_text[:6000])

                    if len(page_text) > 6000:
                        print("\n[Remaining page text omitted]")

            if found_pages == 0:
                print(
                    "No matching keywords found in extracted PDF text. "
                    "The PDF may require OCR or different search terms."
                )

            document.close()

        except Exception as error:
            print(f"ERROR reading PDF: {error}")

    print("\n\nEVIDENCE EXTRACTION COMPLETED.")


if __name__ == "__main__":
    main()
