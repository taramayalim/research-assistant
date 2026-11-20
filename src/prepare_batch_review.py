
import csv
import json
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = ROOT / "data/inventory/article_metadata.csv"
SCREENING_FILE = ROOT / "data/screening/article_screening.csv"
EXTRACTED_DIR = ROOT / "data/extracted"
OUTPUT_FILE = ROOT / "data/inventory/research_corpus_review.xlsx"


# --------------------------------------------------
# 2. READ CSV FILES
# --------------------------------------------------

def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


# --------------------------------------------------
# 3. EXTRACT ABSTRACT
# --------------------------------------------------

def extract_abstract(text):

    patterns = [
        r"\bABSTRACT\b\s*[:\-—]?\s*(.*?)(?=\n\s*(?:INDEX\s+TERMS|KEYWORDS)\b)",
        r"\bAbstract\b\s*[:\-—]?\s*(.*?)(?=\n\s*(?:Keywords|Index\s+Terms)\b)"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:
            return " ".join(match.group(1).split())[:4000]

    return "Abstract not automatically identified - review PDF"


# --------------------------------------------------
# 4. GET EXTRACTED PDF CONTENT
# --------------------------------------------------

def get_pdf_content(article_id):

    path = EXTRACTED_DIR / f"{article_id}.json"

    if not path.exists():
        return "", "JSON file missing"

    try:
        with open(path, "r", encoding="utf-8") as file:
            document = json.load(file)

        pages = document.get("pages", [])

        first_page = (
            pages[0].get("text", "")
            if pages else ""
        )

        abstract = extract_abstract(first_page)

        preview = " ".join(first_page.split())[:1200]

        return abstract, preview

    except Exception as error:
        return "", f"Reading error: {error}"


# --------------------------------------------------
# 5. CLEAN TEXT FOR EXCEL
# --------------------------------------------------

def clean_excel_value(value):

    if isinstance(value, str):
        value = ILLEGAL_CHARACTERS_RE.sub("", value)

        # Excel cells support a maximum of 32,767 characters.
        value = value[:32767]

    return value


# --------------------------------------------------
# 6. LOAD METADATA AND SCREENING RECORDS
# --------------------------------------------------

metadata = read_csv(METADATA_FILE)
screening = read_csv(SCREENING_FILE)

screening_lookup = {
    row["article_id"]: row
    for row in screening
}


# --------------------------------------------------
# 7. CREATE WORKBOOK
# --------------------------------------------------

workbook = Workbook()

review_sheet = workbook.active
review_sheet.title = "Batch Review"


headers = [
    "article_id",
    "filename",
    "title",
    "authors",
    "year",
    "venue",
    "doi_or_url",
    "topical_category",
    "access_license",
    "metadata_verification_status",
    "abstract",
    "first_page_preview",
    "relevance",
    "publication_type",
    "full_text_status",
    "duplicate_status",
    "screening_status",
    "exclusion_reason",
    "reviewer_notes"
]

review_sheet.append(headers)


# --------------------------------------------------
# 8. ADD ARTICLE RECORDS
# --------------------------------------------------

for item in metadata:

    article_id = item["article_id"]

    abstract, preview = get_pdf_content(article_id)

    screening_item = screening_lookup.get(article_id, {})

    row = [
        item.get("article_id", ""),
        item.get("filename", ""),
        item.get("title", ""),
        item.get("authors", ""),
        item.get("year", ""),
        item.get("venue", ""),
        item.get("doi_or_url", ""),
        item.get("topical_category", ""),
        item.get("access_license", ""),
        item.get("metadata_verification_status", ""),
        abstract,
        preview,
        screening_item.get("relevance", "Pending"),
        screening_item.get("publication_type", ""),
        screening_item.get("full_text_status", ""),
        screening_item.get("duplicate_status", "Not checked"),
        screening_item.get("screening_status", "Pending"),
        screening_item.get("exclusion_reason", ""),
        screening_item.get("reviewer_notes", "")
    ]

    # Remove characters that Excel cannot store.
    clean_row = [
        clean_excel_value(value)
        for value in row
    ]

    review_sheet.append(clean_row)


# --------------------------------------------------
# 9. FORMAT REVIEW WORKSHEET
# --------------------------------------------------

review_sheet.freeze_panes = "C2"
review_sheet.auto_filter.ref = review_sheet.dimensions

for cell in review_sheet[1]:

    cell.font = Font(
        bold=True,
        color="FFFFFF"
    )

    cell.fill = PatternFill(
        "solid",
        fgColor="1F4E78"
    )

    cell.alignment = Alignment(
        wrap_text=True,
        vertical="center"
    )


widths = {
    1: 12, 2: 35, 3: 45, 4: 35, 5: 10,
    6: 40, 7: 35, 8: 40, 9: 22, 10: 25,
    11: 65, 12: 65, 13: 20, 14: 32, 15: 25,
    16: 20, 17: 25, 18: 40, 19: 60
}

for column, width in widths.items():

    review_sheet.column_dimensions[
        get_column_letter(column)
    ].width = width


for row in review_sheet.iter_rows(min_row=2):

    for cell in row:

        cell.alignment = Alignment(
            wrap_text=True,
            vertical="top"
        )

review_sheet.row_dimensions[1].height = 35


# --------------------------------------------------
# 10. ADD SCREENING GUIDE
# --------------------------------------------------

screening_sheet = workbook.create_sheet("Screening Guide")

guide = [
    ["Field", "Meaning"],
    ["Relevant", "Directly investigates plant or crop disease detection."],
    ["Partially relevant", "Closely related supporting research, but not direct disease detection."],
    ["Irrelevant", "Does not fit the approved research scope."],
    ["Included - provisional", "Suitable for inclusion, pending final verification."],
    ["Excluded", "Does not meet the corpus inclusion criteria."],
    ["Pending", "Not yet reviewed."],
    ["Not checked", "Duplicate status has not yet been audited."],
    ["Important", "Review every paper before approving its final screening decision."]
]

for row in guide:

    screening_sheet.append([
        clean_excel_value(value)
        for value in row
    ])


for cell in screening_sheet[1]:

    cell.font = Font(
        bold=True,
        color="FFFFFF"
    )

    cell.fill = PatternFill(
        "solid",
        fgColor="1F4E78"
    )


screening_sheet.column_dimensions["A"].width = 30
screening_sheet.column_dimensions["B"].width = 90

for row in screening_sheet.iter_rows():

    for cell in row:

        cell.alignment = Alignment(
            wrap_text=True,
            vertical="top"
        )


# --------------------------------------------------
# 11. SAVE WORKBOOK
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

workbook.save(OUTPUT_FILE)


# --------------------------------------------------
# 12. DISPLAY RESULTS
# --------------------------------------------------

print("\nBATCH REVIEW WORKBOOK CREATED")
print("--------------------------------")
print(f"Metadata records: {len(metadata)}")
print(f"Screening records: {len(screening)}")
print(f"Workbook: {OUTPUT_FILE}")
print(f"File exists: {OUTPUT_FILE.exists()}")
print(f"Review worksheet records: {review_sheet.max_row - 1}")
print(f"Columns: {review_sheet.max_column}")
