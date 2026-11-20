
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

QUESTION_FILE = ROOT / "data/evaluation/question_bank_80.csv"
ANSWER_FILE = ROOT / "data/evaluation/rag_generation_80/qwen3_8b_answers.jsonl"

OUTPUT_FILE = (
    ROOT
    / "data/evaluation/rag_answer_evaluation/qwen3_8b_manual_evaluation.csv"
)

# Load verified questions and generated answers
questions = pd.read_csv(QUESTION_FILE)

with ANSWER_FILE.open("r", encoding="utf-8") as file:
    answers = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

answers_df = pd.DataFrame(answers)

# Match answers with their verified questions
evaluation = questions.merge(
    answers_df,
    on="question_id",
    how="left",
    suffixes=("_gold", "_generated"),
    validate="one_to_one"
)

# Check completeness
if len(evaluation) != 80:
    raise ValueError("Expected exactly 80 evaluation records.")

if evaluation["answer"].isna().any():
    raise ValueError("Some generated answers are missing.")

# Add manual evaluation columns
evaluation["factual_correctness"] = ""
evaluation["completeness"] = ""
evaluation["citation_accuracy"] = ""
evaluation["groundedness"] = ""
evaluation["relevance"] = ""
evaluation["abstention_correct"] = ""
evaluation["hallucination"] = ""
evaluation["overall_score"] = ""
evaluation["reviewer_comments"] = ""

# Save separately; do not modify original files
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
evaluation.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("Evaluation dataset created successfully.")
print("Total questions:", len(evaluation))
print("Missing generated answers:", evaluation["answer"].isna().sum())
print("Saved to:", OUTPUT_FILE)
print("\nAnswerability distribution:")
print(evaluation["answerability"].value_counts())
