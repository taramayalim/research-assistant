
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

categories = {
    "Factual": 20,
    "Conceptual": 20,
    "Cross-paper comparison": 20,
    "Unanswerable": 20
}

fields = [
    "question_id",
    "category",
    "question",
    "answerability",
    "gold_answer",
    "supporting_article_ids",
    "supporting_pages",
    "relevant_chunk_ids",
    "verification_notes"
]

output_file = OUTPUT_DIR / "question_bank_80.csv"

with open(output_file, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    question_number = 1

    for category, count in categories.items():
        for _ in range(count):
            writer.writerow({
                "question_id": f"Q{question_number:03d}",
                "category": category,
                "question": "",
                "answerability": (
                    "Unanswerable"
                    if category == "Unanswerable"
                    else "Answerable"
                ),
                "gold_answer": "",
                "supporting_article_ids": "",
                "supporting_pages": "",
                "relevant_chunk_ids": "",
                "verification_notes": ""
            })

            question_number += 1

print("Evaluation template created successfully.")
print(f"Total questions: {sum(categories.values())}")
print(f"Saved to: {output_file}")
