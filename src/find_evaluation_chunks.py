
import csv
from pathlib import Path

CHUNK_FILE = Path(
    "data/processed/phase4_chunking_comparison_normalized/"
    "fixed_size_chunks.csv"
)

TARGETS = {
    "A006": ["30,945", "eight plant types", "35 disease classes", "95.62"],
    "A072": ["9,127", "CNN", "VGG16", "VGG19", "0.97"],
    "A019": ["ResNet152", "InceptionV3", "dense layer"],
}

with open(CHUNK_FILE, "r", encoding="utf-8-sig", newline="") as f:
    content = f.read().replace("\x00", "")

from io import StringIO
reader = csv.DictReader(StringIO(content))

print("\n========== EVALUATION CHUNK IDENTIFICATION ==========\n")

for row in reader:
    article_id = row.get("article_id", "").strip()

    if article_id not in TARGETS:
        continue

    chunk_text = row.get("text", "")
    matched = [
        keyword
        for keyword in TARGETS[article_id]
        if keyword.lower() in chunk_text.lower()
    ]

    if matched:
        print(f"Article ID : {article_id}")
        print(f"Chunk ID   : {row.get('chunk_id')}")
        print(f"Page       : {row.get('page')}")
        print(f"Keywords   : {', '.join(matched)}")
        print("-" * 65)

print("\nChunk identification completed.")
