import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "evaluation" / "candidate_addition_extraction"
OUTPUT = ROOT / "data" / "evaluation" / "candidate_addition_review"

OUTPUT.mkdir(parents=True, exist_ok=True)

for article_id in ["A160", "A161", "A162", "A163", "A164"]:
    path = SOURCE / f"{article_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    pages = data["pages"]
    full_text = "\n".join(p["text"] for p in pages)

    abstract_match = re.search(
        r"\babstract\b(.*?)(?=\b(?:keywords|index terms|1\.?\s+introduction|introduction)\b)",
        full_text, re.I | re.S
    )

    conclusion_matches = list(re.finditer(
        r"\b(?:conclusion|conclusions|concluding remarks)\b",
        full_text, re.I
    ))

    abstract = abstract_match.group(1).strip() if abstract_match else (
        "ABSTRACT NOT AUTOMATICALLY LOCATED"
    )

    conclusion = "CONCLUSION NOT AUTOMATICALLY LOCATED"
    if conclusion_matches:
        start = conclusion_matches[-1].start()
        conclusion = full_text[start:start + 5000]

    headings = []
    for line in full_text.splitlines():
        line = line.strip()
        if (
            len(line) < 120
            and re.match(
                r"^(?:\d+(?:\.\d+)*\.?\s+)?"
                r"(?:introduction|related work|literature review|"
                r"materials and methods|methodology|proposed method|"
                r"results|discussion|experiments|conclusion|"
                r"future work|references)\b",
                line, re.I
            )
        ):
            headings.append(line)

    doi_count = len(re.findall(r"10\.\d{4,9}/[^\s]+", full_text))
    reference_count = len(re.findall(r"\breferences\b", full_text, re.I))

    report = [
        f"ARTICLE ID: {article_id}",
        f"FILENAME: {data['filename']}",
        f"PAGE COUNT: {data['page_count']}",
        f"TOTAL CHARACTERS: {len(full_text)}",
        "\n" + "=" * 75,
        "OPENING PAGES (first 2 pages)",
        "=" * 75,
    ]

    for page in pages[:2]:
        report.append(
            f"\n--- PAGE {page['page_number']} ---\n"
            + page["text"][:7000]
        )

    report.extend([
        "\n" + "=" * 75,
        "ABSTRACT",
        "=" * 75,
        abstract[:7000],
        "\n" + "=" * 75,
        "DETECTED SECTION HEADINGS",
        "=" * 75,
        "\n".join(dict.fromkeys(headings)) or "No headings detected",
        "\n" + "=" * 75,
        "CONCLUSION",
        "=" * 75,
        conclusion,
        "\n" + "=" * 75,
        "REFERENCES INDICATORS",
        "=" * 75,
        f"Occurrences of 'References': {reference_count}",
        f"DOI occurrences: {doi_count}",
    ])

    output_path = OUTPUT / f"{article_id}_review.txt"
    output_path.write_text("\n".join(report), encoding="utf-8")

    print(f"{article_id}: review report created -> {output_path}")

print("\nAll five review reports created.")
