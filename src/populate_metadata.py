import csv
import json
import re
import shutil
import time
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "data" / "inventory" / "article_inventory.csv"
EXTRACTED = ROOT / "data" / "extracted"
BACKUP = INVENTORY.with_name("article_inventory_before_metadata_recovery.csv")

DOI_PATTERN = re.compile(
    r'10\.\d{4,9}/[^\s<>"\']+',
    re.IGNORECASE
)

def get_text(data):
    parts = []
    for page in data.get("pages", []):
        if isinstance(page, dict):
            parts.append(page.get("text", ""))
    return "\n".join(parts)

def find_doi(text):
    match = DOI_PATTERN.search(text)
    if not match:
        return ""

    doi = match.group(0).rstrip(".,;:)]}")
    doi = re.sub(r"[\s]+", "", doi)
    return doi

def get_year(message):
    for key in ("published-print", "published-online", "issued"):
        date_parts = message.get(key, {}).get("date-parts", [])
        if date_parts and date_parts[0]:
            return str(date_parts[0][0])
    return ""

def lookup_crossref(doi):
    url = "https://api.crossref.org/works/" + quote(doi, safe="")
    response = requests.get(
        url,
        headers={"User-Agent": "AcademicMetadataRecovery/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()["message"]

def authors_text(authors):
    names = []
    for author in authors or []:
        name = " ".join(
            part for part in [
                author.get("given", ""),
                author.get("family", "")
            ] if part
        ).strip()

        if name:
            names.append(name)

    return "; ".join(names)

def main():
    if not INVENTORY.exists():
        raise FileNotFoundError(f"Inventory not found: {INVENTORY}")

    # Preserve the original inventory before making any changes.
    if not BACKUP.exists():
        shutil.copy2(INVENTORY, BACKUP)
        print(f"Backup created: {BACKUP}")

    with INVENTORY.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        rows = list(reader)

    if not fields or "article_id" not in fields:
        raise ValueError("Inventory must contain article_id.")

    stats = {
        "records": len(rows),
        "doi_found": 0,
        "crossref_matches": 0,
        "metadata_filled": 0,
        "lookup_failures": 0,
    }

    for index, row in enumerate(rows, start=1):
        article_id = row.get("article_id", "").strip()
        json_path = EXTRACTED / f"{article_id}.json"

        if not json_path.exists():
            continue

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            text = get_text(data)

            doi = (row.get("doi") or "").strip() or find_doi(text)

            if not doi:
                continue

            stats["doi_found"] += 1

            # Keep the DOI found in the PDF even if Crossref is unavailable.
            if not (row.get("doi") or "").strip():
                row["doi"] = doi
                stats["metadata_filled"] += 1

            try:
                message = lookup_crossref(doi)
                stats["crossref_matches"] += 1

                title_values = message.get("title", [])
                title = title_values[0].strip() if title_values else ""
                authors = authors_text(message.get("author", []))
                year = get_year(message)

                venue_values = message.get("container-title", [])
                venue = venue_values[0].strip() if venue_values else ""

                candidates = {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "venue": venue,
                    "stable_url": f"https://doi.org/{doi}",
                }

                for field, value in candidates.items():
                    if field in row and value and not row[field].strip():
                        row[field] = value
                        stats["metadata_filled"] += 1

                if "doi_or_stable_url" in row and not row["doi_or_stable_url"].strip():
                    row["doi_or_stable_url"] = f"https://doi.org/{doi}"

            except Exception as error:
                stats["lookup_failures"] += 1
                print(f"Crossref lookup failed for {article_id}: {error}")

            if index % 10 == 0:
                print(f"Checked {index}/{len(rows)} inventory records...")

            time.sleep(0.15)

        except Exception as error:
            print(f"Could not process {article_id}: {error}")

    with INVENTORY.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nMETADATA RECOVERY SUMMARY")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\nUpdated inventory: {INVENTORY}")
    print(f"Backup: {BACKUP}")
    print("Access licences were not inferred or changed.")

if __name__ == "__main__":
    main()
