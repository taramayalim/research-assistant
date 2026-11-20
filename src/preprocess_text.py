import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "data" / "extracted_v2"
OUTPUT_DIR = ROOT / "data" / "cleaned"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    # Join words split by a hyphen at a line break.
    text = re.sub(
        r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=[a-z])",
        "",
        text
    )

    # Remove trailing spaces before line breaks.
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Reduce excessive horizontal spaces.
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Preserve paragraphs but remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


processed = 0
total_pages = 0
total_hyphenations = 0
errors = []

for file in sorted(SOURCE_DIR.glob("A*.json")):
    try:
        data = json.loads(file.read_text(encoding="utf-8"))

        for page in data.get("pages", []):
            original = page.get("text", "")

            matches = len(re.findall(
                r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=[a-z])",
                original
            ))

            page["text"] = clean_text(original)

            total_hyphenations += matches
            total_pages += 1

        output_file = OUTPUT_DIR / file.name
        output_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        processed += 1

    except Exception as error:
        errors.append(f"{file.name}: {error}")

print("\n========== PREPROCESSING SUMMARY ==========")
print("Articles processed:", processed)
print("Pages processed:", total_pages)
print("Line-break hyphenations corrected:", total_hyphenations)
print("Errors:", len(errors))

for error in errors:
    print(error)
