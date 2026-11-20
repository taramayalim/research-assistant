import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

path = Path("data/evaluation/rag_answer_evaluation/qwen3_8b_manual_evaluation.csv")

if not path.exists():
    raise FileNotFoundError(f"Evaluation file not found: {path}")

df = pd.read_csv(path)

if len(df) != 80 or df["question_id"].nunique() != 80:
    raise ValueError("Expected exactly 80 unique questions. No changes made.")

scores = {
    "Q001": (2,2,2,2,2,None,0,2,"Correct answer and citation."),
    "Q002": (0,0,1,2,2,0,0,0,"Appropriate abstention given retrieved evidence, but the answerable fact was not retrieved."),
    "Q003": (0,0,0,1,2,None,1,0,"Answered using A060 instead of the target Rahman study A006."),
    "Q004": (0,0,None,2,2,0,0,0,"Abstained because the required A006 evidence was not retrieved."),
    "Q005": (2,2,2,2,2,None,0,2,"Correct models and citation."),
    "Q006": (2,2,2,2,2,None,0,2,"Correct image count and citation."),
    "Q007": (2,2,2,2,2,None,0,2,"Correct CNN accuracy and citation."),
    "Q008": (2,2,2,2,2,None,0,2,"Correct architectures and citation."),
    "Q009": (0,0,0,1,2,None,1,0,"Used the 96.03% result from A001 instead of the 10.15% result in A114."),
    "Q010": (2,2,2,2,2,None,0,2,"Correct result and citation."),
    "Q011": (2,2,2,2,2,None,0,2,"Correct result and citation."),
    "Q012": (2,2,2,2,2,None,0,2,"Correct number of disease classes and citation."),
    "Q013": (2,2,2,2,2,None,0,2,"Correct number of crop types and citation."),
    "Q014": (0,0,None,2,2,0,0,0,"Abstained despite the relevant A114 evidence being retrieved."),
    "Q015": (2,2,2,2,2,None,0,2,"Correct parameter count and breakdown."),
    "Q016": (2,2,2,2,2,None,0,2,"Correct accuracy and parameter count."),
    "Q017": (0,0,0,1,2,None,1,0,"Used the 97.84% result from A097 instead of the 95.38% result in A114."),
    "Q018": (2,2,2,2,2,None,0,2,"Correct accuracy and citation."),
    "Q019": (2,2,2,2,2,None,0,2,"Correct accuracy and parameter count."),
    "Q020": (0,0,0,0,2,None,2,0,"Provided techniques not matching the A114 future-research recommendation and cited A037.")
}

score_columns = [
    "factual_correctness", "completeness", "citation_accuracy",
    "groundedness", "relevance", "abstention_correct",
    "hallucination", "overall_score"
]

required = ["question_id"] + score_columns + ["reviewer_comments"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Create a separate backup without overwriting an existing backup.
backup = path.with_name(
    f"{path.stem}_backup_before_q001_q020_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}"
)
shutil.copy2(path, backup)

for qid, values in scores.items():
    mask = df["question_id"].eq(qid)
    if mask.sum() != 1:
        raise ValueError(f"{qid} does not occur exactly once.")

    for column, value in zip(score_columns, values[:8]):
        df.loc[mask, column] = value

    df.loc[mask, "reviewer_comments"] = values[8]

df.to_csv(path, index=False, encoding="utf-8")

check = pd.read_csv(path)
reviewed = check[check["question_id"].isin(scores)]

print("Q001-Q020 scoring saved successfully.")
print("Updated file:", path)
print("Backup:", backup)
print("Questions scored:", len(reviewed))
print("Overall score distribution:")
print(reviewed["overall_score"].value_counts(dropna=False).sort_index())
print("\nScoring completeness:")
print(reviewed[score_columns].notna().sum())
