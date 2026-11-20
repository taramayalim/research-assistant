
from pathlib import Path
import re

folder = Path("data/evaluation/candidate_pdf_extracts")

keywords = [
    "dataset", "images", "samples", "classes",
    "accuracy", "precision", "recall", "f1-score",
    "proposed model", "architecture", "parameters",
    "inference time", "mobile net", "mobilenet",
    "training", "testing", "performance"
]

for file_path in sorted(folder.glob("*_original.txt")):
    print("\n" + "=" * 90)
    print(f"ARTICLE: {file_path.stem}")
    print("=" * 90)

    text = file_path.read_text(encoding="utf-8")
    pages = re.split(r"={10,}\s*PAGE\s+(\d+)\s*={10,}", text)

    for i in range(1, len(pages), 2):
        page_number = pages[i]
        page_text = pages[i + 1]

        lines = page_text.splitlines()
        matches = []

        for line_no, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in keywords):
                start = max(0, line_no - 1)
                end = min(len(lines), line_no + 3)
                snippet = " ".join(
                    part.strip() for part in lines[start:end]
                    if part.strip()
                )

                if snippet and snippet not in matches:
                    matches.append(snippet)

        if matches:
            print(f"\n--- PAGE {page_number} ---")
            for match in matches[:8]:
                print("-", match[:500])
