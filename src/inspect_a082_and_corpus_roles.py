import pandas as pd
import json
from pathlib import Path

metadata = pd.read_csv("data/inventory/article_metadata.csv")
screening = pd.read_csv("data/screening/article_screening.csv")

print("========== A082 SCREENING AND METADATA ==========")
print(metadata[metadata.article_id == "A082"].to_string(index=False))
print(screening[screening.article_id == "A082"].to_string(index=False))

print("\n========== A082 EXTRACTED TEXT ==========")
p = Path("data/extracted_v2/A082.json")

if p.exists():
    data = json.loads(p.read_text(encoding="utf-8"))
    print("JSON keys:", list(data.keys()))

    def show_text(obj, depth=0):
        if depth > 3:
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and len(value.strip()) > 100:
                    print(f"\n[{key}]")
                    print(value[:2500])
                elif isinstance(value, (dict, list)):
                    show_text(value, depth + 1)
        elif isinstance(obj, list):
            for item in obj[:3]:
                show_text(item, depth + 1)

    show_text(data)
else:
    print("A082 extracted JSON not found.")

print("\n========== SUPPORTING PAPERS ==========")
supporting = metadata[metadata["corpus_role"].astype(str).str.strip() == "Supporting"]

print(supporting[
    ["article_id", "filename", "title", "topical_category",
     "metadata_verification_status"]
].to_string(index=False))

print("\n========== CORE PAPERS ==========")
core = metadata[metadata["corpus_role"].astype(str).str.strip() == "Core"]
print("Core count:", len(core))
print(core[["article_id", "title"]].to_string(index=False))
