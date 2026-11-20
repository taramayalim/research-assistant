import json
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "data" / "extracted"
PDF_DIR = ROOT / "Articles"
OUTPUT_DIR = ROOT / "data" / "extracted_v2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

processed = 0
preserved_ocr = 0
errors = []

for json_path in sorted(SOURCE_DIR.glob("A*.json")):
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))

        article_id = data["article_id"]
        filename = data["filename"]
        method = str(data.get("extraction_method", "")).lower()

        output_path = OUTPUT_DIR / json_path.name

        # Preserve existing OCR text instead of replacing it.
        if "ocr" in method or "tesseract" in method:
            output_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            preserved_ocr += 1
            print(f"{article_id}: Existing OCR preserved")
            continue

        pdf_path = PDF_DIR / filename

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = pymupdf.open(pdf_path)

        pages = []
        for page_number, page in enumerate(doc, start=1):
            pages.append({
                "page_number": page_number,
                "text": page.get_text("text")
            })

        result = {
            "article_id": article_id,
            "filename": filename,
            "extraction_method": "PyMuPDF",
            "page_count": len(doc),
            "pages": pages
        }

        doc.close()

        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        processed += 1
        print(f"{article_id}: PyMuPDF extraction completed")

    except Exception as error:
        errors.append({
            "file": json_path.name,
            "error": str(error)
        })
        print(f"ERROR - {json_path.name}: {error}")

print("\n========== EXTRACTION SUMMARY ==========")
print(f"PyMuPDF extracted: {processed}")
print(f"Existing OCR preserved: {preserved_ocr}")
print(f"Errors: {len(errors)}")

if errors:
    print("\nFiles requiring attention:")
    for item in errors:
        print(item["file"], "-", item["error"])
