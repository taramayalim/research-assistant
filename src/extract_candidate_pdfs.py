import pandas as pd
import fitz
from pathlib import Path

ids = ["A040", "A049", "A055", "A081", "A101", "A114"]

metadata = pd.read_csv("data/inventory/article_metadata.csv")
output_dir = Path("data/evaluation/candidate_pdf_extracts")
output_dir.mkdir(parents=True, exist_ok=True)

for article_id in ids:
    row = metadata.loc[metadata["article_id"] == article_id].iloc[0]
    pdf_path = Path("Articles") / row["filename"]

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        continue

    doc = fitz.open(pdf_path)
    output = []

    for page_number, page in enumerate(doc, start=1):
        output.append(f"\n{'='*25} PAGE {page_number} {'='*25}\n")
        output.append(page.get_text())

    text_path = output_dir / f"{article_id}_original.txt"
    text_path.write_text("".join(output), encoding="utf-8")

    print(f"{article_id}: {len(doc)} pages extracted -> {text_path}")
