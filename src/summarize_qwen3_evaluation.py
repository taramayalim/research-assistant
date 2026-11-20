import pandas as pd
from pathlib import Path

path = Path("data/evaluation/rag_answer_evaluation/qwen3_8b_manual_evaluation.csv")
df = pd.read_csv(path)

score_cols = [
    "factual_correctness", "completeness", "citation_accuracy",
    "groundedness", "relevance", "overall_score"
]

if len(df) != 80 or df["question_id"].nunique() != 80:
    raise ValueError("Evaluation must contain exactly 80 unique questions.")

if df["overall_score"].isna().any():
    raise ValueError("Some overall scores are missing.")

# Convert scores to numeric for reliable calculations.
for col in score_cols + ["abstention_correct", "hallucination"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n========== 1. OVERALL PERFORMANCE ==========")
for col in score_cols:
    mean = df[col].mean()
    print(f"{col:25s}: {mean:.3f}/2 ({mean/2*100:.2f}%)")

print("\nOverall score distribution:")
print(df["overall_score"].value_counts().sort_index())

print("\n========== 2. PERFORMANCE BY CATEGORY ==========")
category_summary = df.groupby("category")[score_cols].mean().round(3)
print(category_summary.to_string())

print("\n========== 3. ANSWERABLE VS UNANSWERABLE ==========")
df["answerability_group"] = df["answerability"].astype(str).str.strip().str.lower()

for group, subset in df.groupby("answerability_group"):
    print(f"\n{group.upper()} — {len(subset)} questions")
    for col in score_cols:
        mean = subset[col].mean()
        print(f"{col:25s}: {mean:.3f}/2 ({mean/2*100:.2f}%)")

print("\n========== 4. HALLUCINATION ANALYSIS ==========")
hallucinated = df[df["hallucination"] > 0]
clear_hallucinations = df[df["hallucination"] == 2]

print("Any hallucination/unsupported addition:",
      len(hallucinated), "/", len(df))
print("Clear hallucinations:",
      len(clear_hallucinations), "/", len(df))
print("\nHallucination score distribution:")
print(df["hallucination"].value_counts().sort_index())

print("\nQuestions with hallucination or unsupported additions:")
print(hallucinated[["question_id", "answerability", "hallucination",
                    "overall_score", "reviewer_comments"]].to_string(index=False))

print("\n========== 5. UNANSWERABLE QUESTION ABSTENTION ==========")
unanswerable = df[df["answerability_group"].str.contains("unanswerable")]

clean = unanswerable[
    (unanswerable["abstention_correct"] == 2) &
    (unanswerable["hallucination"] == 0)
]
partial = unanswerable[
    (unanswerable["abstention_correct"] == 2) &
    (unanswerable["hallucination"] > 0)
]
incorrect = unanswerable[unanswerable["abstention_correct"] == 0]

print("Unanswerable questions:", len(unanswerable))
print(f"Clean abstentions: {len(clean)}/{len(unanswerable)}")
print(f"Partial abstentions: {len(partial)}/{len(unanswerable)}")
print(f"Incorrect answers: {len(incorrect)}/{len(unanswerable)}")

print("\n========== 6. ANSWERABLE ABSTENTION RECORDS ==========")
answerable = df[~df.index.isin(unanswerable.index)]
answerable_abstentions = answerable[answerable["abstention_correct"].notna()]

print("Answerable questions with an abstention score:",
      len(answerable_abstentions))
if len(answerable_abstentions):
    print(answerable_abstentions[
        ["question_id", "abstention_correct", "overall_score",
         "reviewer_comments"]
    ].to_string(index=False))

print("\n========== 7. MISSING VALUES ==========")
print(df[score_cols + ["abstention_correct", "hallucination"]].isna().sum())

# Save a separate summary; original manual scores remain unchanged.
summary_rows = []
for group_name, subset in [
    ("All questions", df),
    ("Answerable", answerable),
    ("Unanswerable", unanswerable)
]:
    for col in score_cols:
        summary_rows.append({
            "group": group_name,
            "metric": col,
            "mean_score_out_of_2": subset[col].mean(),
            "percentage_of_maximum": subset[col].mean() / 2 * 100
        })

summary_path = Path("data/evaluation/rag_answer_evaluation/qwen3_8b_metric_summary.csv")
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

print("\nSummary saved to:", summary_path)
