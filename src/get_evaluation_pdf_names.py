
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "data" / "inventory" / "article_metadata.csv"

TARGET_IDS = ["A006", "A019", "A072"]

with open(METADATA_FILE, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    metadata = list(reader)

print("\nORIGINAL PDF FILES")
print("=" * 80)

for article_id in TARGET_IDS:
    matches = [
        row for row in metadata
        if row["article_id"] == article_id
    ]

    if matches:
        print(f"\nArticle ID: {article_id}")
        print(f"Title: {matches[0]['title']}")
        print(f"Filename: {matches[0]['filename']}")
    else:
        print(f"\n{article_id}: Metadata not found")
