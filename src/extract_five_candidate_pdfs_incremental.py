import json
import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "Articles" / "candidate_additions"
OUTPUT_DIR = ROOT / "data" / "evaluation" / "candidate_addition_extraction"

CANDIDATES = [
    ("A160", "*hybrid*Framework*plant*leaf*disease*.pdf"),
    ("A161", "*Multitask*Vision*Transformer*Plant*Disease*.pdf"),
    ("A162", "*Advancing*plant*leaf*disease*.pdf"),
    ("A163", "*Enhancing*plant*disease*.pdf"),
    ("A164", "*Visualizing*Plant*Disease*Distribution*.pdf"),
]

METADATA_PATH = ROOT / "data" / "inventory" / "article_metadata.csv"
SCREENING_PATH = ROOT / "data" / "screening" / "article_screening.csv"
ACTIVE_EXTRACTED = ROOT / "data" / "extracted_v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check that the proposed IDs are not already registered or extracted.
import pandas as pd

metadata = pd.read_csv(METADATA_PATH, dtype={"article_id": str})
screening = pd.read_csv(SCREENING_PATH, dtype={"article_id": str})

for article_id, _ in CANDIDATES:
    if article_id in set(metadata["article_id"].dropna()):
        raise RuntimeError(f"{article_id} already exists in metadata.")
    if article_id in set(screening["article_id"].dropna()):
        raise RuntimeError(f"{article_id} already exists in screening.")
    if (ACTIVE_EXTRACTED / f"{article_id}.json").exists():
        raise RuntimeError(f"{article_id}.json already exists in active extraction.")
    if (OUTPUT_DIR / f"{article_id}.json").exists():
        raise RuntimeError(
            f"{article_id}.json already exists in staging. "
            "Inspect it before rerunning."
        )

manifest = []

for article_id, pattern in CANDIDATES:
    matches = list(SOURCE_DIR.glob(pattern))

    if len(matches) != 1:
        raise RuntimeError(
            f"{article_id}: expected exactly one PDF for {pattern}, "
            f"but found {len(matches)}: {matches}"
        )

    pdf_path = matches[0]

    with pymupdf.open(pdf_path) as doc:
        pages = []

        for page_number, page in enumerate(doc, start=1):
            pages.append({
                "page_number": page_number,
                "text": page.get_text("text")
            })

        total_chars = sum(len(page["text"]) for page in pages)

        if len(pages) == 0 or total_chars < 1000:
            raise RuntimeError(
                f"{article_id}: extraction looks incomplete "
                f"({len(pages)} pages, {total_chars} characters)."
            )

        record = {
            "article_id": article_id,
            "filename": pdf_path.name,
            "extraction_method": "PyMuPDF",
            "page_count": len(pages),
            "pages": pages
        }

    output_path = OUTPUT_DIR / f"{article_id}.json"
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    manifest.append({
        "article_id": article_id,
        "filename": pdf_path.name,
        "page_count": record["page_count"],
        "characters_extracted": total_chars,
        "output_file": str(output_path.relative_to(ROOT))
    })

    print(
        f"{article_id}: {pdf_path.name} | "
        f"{record['page_count']} pages | "
        f"{total_chars:,} characters"
    )

manifest_path = OUTPUT_DIR / "extraction_manifest.csv"
pd.DataFrame(manifest).to_csv(manifest_path, index=False, encoding="utf-8-sig")

print("\nExtraction completed.")
print(f"Extracted papers: {len(manifest)}")
print(f"Staging folder: {OUTPUT_DIR}")
print(f"Manifest: {manifest_path}")
