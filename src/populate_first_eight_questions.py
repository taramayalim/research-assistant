
import csv
from pathlib import Path

file_path = Path("data/evaluation/question_bank_80.csv")

questions = [
    {
        "question_id": "Q001",
        "question": "How many images were included in the dataset used by Rahman et al.?",
        "gold_answer": "30,945 images",
        "article": "A006",
        "pages": "1",
        "chunk": "A006_FIXED_page_0001_chunk_0001",
        "notes": "Verified from the abstract of the original research paper."
    },
    {
        "question_id": "Q002",
        "question": "How many plant types were represented in the dataset used by Rahman et al.?",
        "gold_answer": "Eight plant types",
        "article": "A006",
        "pages": "1",
        "chunk": "A006_FIXED_page_0001_chunk_0001",
        "notes": "Verified from the abstract of the original research paper."
    },
    {
        "question_id": "Q003",
        "question": "How many disease classes were included in the dataset used by Rahman et al.?",
        "gold_answer": "35 disease classes",
        "article": "A006",
        "pages": "1",
        "chunk": "A006_FIXED_page_0001_chunk_0001",
        "notes": "Verified from the abstract of the original research paper."
    },
    {
        "question_id": "Q004",
        "question": "What accuracy did the initial custom CNN achieve in the leaf classification task?",
        "gold_answer": "95.62%",
        "article": "A006",
        "pages": "1",
        "chunk": "A006_FIXED_page_0001_chunk_0001",
        "notes": "Verified from the abstract; refers to initial leaf classification."
    },
    {
        "question_id": "Q005",
        "question": "Which three deep learning models were compared by Shivaprasad and Wadhawan?",
        "gold_answer": "CNN, VGG16, and VGG19",
        "article": "A072",
        "pages": "1",
        "chunk": "A072_FIXED_page_0001_chunk_0001",
        "notes": "Verified from the abstract of the original research paper."
    },
    {
        "question_id": "Q006",
        "question": "How many plant images were used in the study by Shivaprasad and Wadhawan?",
        "gold_answer": "9,127 images",
        "article": "A072",
        "pages": "1",
        "chunk": "A072_FIXED_page_0001_chunk_0001",
        "notes": "Verified from the abstract of the original research paper."
    },
    {
        "question_id": "Q007",
        "question": "What accuracy did the CNN model achieve in the study by Shivaprasad and Wadhawan?",
        "gold_answer": "97% (0.97)",
        "article": "A072",
        "pages": "1",
        "chunk": "A072_FIXED_page_0001_chunk_0001",
        "notes": "Verified from the abstract of the original research paper."
    },
    {
        "question_id": "Q008",
        "question": "Which two pretrained architectures were modified by adding a dense layer in the multi-crop disease detection study?",
        "gold_answer": "ResNet152 and InceptionV3",
        "article": "A019",
        "pages": "2",
        "chunk": "A019_FIXED_page_0002_chunk_0003",
        "notes": "Verified from the original paper, pages 2-3."
    }
]

with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

required = {
    "question_id", "category", "question", "answerability",
    "gold_answer", "supporting_article_ids", "supporting_pages",
    "relevant_chunk_ids", "verification_notes"
}

if not fieldnames or not required.issubset(fieldnames):
    raise ValueError(
        f"Unexpected CSV headers. Found: {fieldnames}"
    )

updates = {q["question_id"]: q for q in questions}

for row in rows:
    qid = row["question_id"]

    if qid not in updates:
        continue

    q = updates[qid]
    row["category"] = "Factual"
    row["question"] = q["question"]
    row["answerability"] = "Answerable"
    row["gold_answer"] = q["gold_answer"]
    row["supporting_article_ids"] = q["article"]
    row["supporting_pages"] = q["pages"]
    row["relevant_chunk_ids"] = q["chunk"]
    row["verification_notes"] = q["notes"]

with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("First eight factual questions populated successfully.")
print("Updated questions: Q001-Q008")
print("Remaining questions: Q009-Q080 unchanged.")
print(f"Saved to: {file_path}")
