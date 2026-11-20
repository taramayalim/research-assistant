
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# File paths
QUESTION_FILE = ROOT / "data" / "evaluation" / "question_bank_80.csv"
QRELS_FILE = ROOT / "data" / "evaluation" / "qrels.csv"

# Correct Q001 details
QUESTION_ID = "Q001"
CORRECT_CHUNK = "A006_FIXED_page_0009_chunk_0005"

# 1. Correct the question bank
questions = pd.read_csv(QUESTION_FILE, dtype=str).fillna("")

mask = questions["question_id"] == QUESTION_ID

if mask.sum() != 1:
    raise ValueError(f"Expected exactly one Q001, found {mask.sum()}")

questions.loc[mask, "question"] = (
    "What accuracy did the custom CNN achieve after 20 epochs, "
    "as reported in Table 5?"
)
questions.loc[mask, "gold_answer"] = "95.62%"
questions.loc[mask, "supporting_article_ids"] = "A006"
questions.loc[mask, "supporting_pages"] = "9"
questions.loc[mask, "relevant_chunk_ids"] = CORRECT_CHUNK

questions.to_csv(QUESTION_FILE, index=False, encoding="utf-8-sig")

# 2. Correct Q001 in qrels
qrels = pd.read_csv(QRELS_FILE, dtype=str).fillna("")

# Remove old Q001 relevance entries
qrels = qrels[qrels["query_id"] != QUESTION_ID].copy()

# Add the verified supporting chunk
new_row = pd.DataFrame([{
    "query_id": QUESTION_ID,
    "chunk_id": CORRECT_CHUNK,
    "relevance_grade": 1
}])

qrels = pd.concat([qrels, new_row], ignore_index=True)

qrels.to_csv(QRELS_FILE, index=False, encoding="utf-8-sig")

print("Q001 correction completed.")
print("\nUpdated question:")
print(questions.loc[mask, [
    "question_id",
    "question",
    "gold_answer",
    "supporting_article_ids",
    "supporting_pages",
    "relevant_chunk_ids"
]].to_string(index=False))

print("\nUpdated Q001 qrels:")
print(qrels[qrels["query_id"] == QUESTION_ID].to_string(index=False))
