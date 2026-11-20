import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

path = Path("data/evaluation/rag_answer_evaluation/qwen3_8b_manual_evaluation.csv")
df = pd.read_csv(path)

if len(df) != 80 or df["question_id"].nunique() != 80:
    raise ValueError("Expected exactly 80 unique questions.")

# Score order:
# factual, completeness, citation, groundedness, relevance,
# abstention_correct, hallucination, overall_score, reviewer_comments

scores = {
    "Q061": (2,2,2,2,2,2,0,2,"Correctly abstained; the requested BTN deployment cost is not reported in the retrieved evidence."),
    "Q062": (2,2,2,2,2,2,0,2,"Correctly abstained; the exact Random Forest hyperspectral energy consumption is unavailable."),
    "Q063": (2,2,2,2,2,2,0,2,"Correctly abstained; the requested duration of operational farm testing is not established."),
    "Q064": (2,2,2,2,2,2,0,2,"Correctly abstained; no corpus-wide average improvement attributable exclusively to augmentation is established."),
    "Q065": (2,2,2,2,2,2,0,2,"Correctly abstained; energy consumption per 1,000 images is not reported."),
    "Q066": (2,2,2,2,2,2,0,2,"Correctly abstained; no farmer user-evaluation percentage is provided."),
    "Q067": (2,2,2,2,2,2,0,2,"Correctly abstained; exact Android inference latency for ResNet101 is unavailable."),
    "Q068": (1,2,1,1,2,2,1,1,"Mostly abstained, but added an unsupported estimate about battery consumption from the available evidence."),
    "Q069": (0,0,0,0,1,0,2,0,"Incorrectly supplied a 12.6% crop-yield increase that does not establish the requested field-deployment outcome."),
    "Q070": (2,2,2,2,2,2,0,2,"Correctly abstained; the number of farmers in the specified controlled field trial is not established."),
    "Q071": (2,2,2,2,2,2,0,2,"Correctly abstained; no measured crop-yield gain on Bhutanese farms is reported."),
    "Q072": (2,2,2,2,2,2,0,2,"Correctly abstained; nighttime accuracy below 10 lux is not reported."),
    "Q073": (0,0,0,0,1,0,2,0,"Incorrectly presented a field-testing pesticide reduction as a result of commercial deployment."),
    "Q074": (2,2,2,2,2,2,0,2,"Correctly abstained; the requested number of wrong diagnoses per hectare is not reported."),
    "Q075": (1,2,1,1,2,2,1,1,"Recognised that exact Android latency was unavailable but extrapolated from a different model."),
    "Q076": (2,2,2,2,2,2,0,2,"Correctly abstained; PlantNet peak Android RAM usage is not reported."),
    "Q077": (2,2,2,2,2,2,0,2,"Correctly abstained; no same-dataset statistical significance comparison is provided."),
    "Q078": (2,2,2,2,2,2,0,2,"Correctly abstained; the two studies do not provide a direct comparison on the same unseen field test."),
    "Q079": (2,2,2,2,2,2,0,2,"Correctly abstained; the requested combined training carbon footprint is not reported."),
    "Q080": (2,2,2,2,2,2,0,2,"Correctly abstained; no measured crop losses prevented by actual YOLOv8 field deployment are reported.")
}

columns = [
    "factual_correctness", "completeness", "citation_accuracy",
    "groundedness", "relevance", "abstention_correct",
    "hallucination", "overall_score"
]

for col in columns + ["reviewer_comments"]:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

for qid in scores:
    mask = df["question_id"].eq(qid)
    if mask.sum() != 1:
        raise ValueError(f"{qid} must occur exactly once.")
    if df.loc[mask, "overall_score"].notna().any():
        raise ValueError(f"{qid} is already scored. No changes made.")

backup = path.with_name(
    f"{path.stem}_backup_before_q061_q080_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}"
)
shutil.copy2(path, backup)

df["reviewer_comments"] = df["reviewer_comments"].astype("string")

for qid, values in scores.items():
    mask = df["question_id"].eq(qid)
    for col, value in zip(columns, values[:8]):
        df.loc[mask, col] = value
    df.loc[mask, "reviewer_comments"] = values[8]

df.to_csv(path, index=False, encoding="utf-8")

check = pd.read_csv(path)
reviewed = check[check["question_id"].isin(scores)]

print("Q061-Q080 scores saved successfully.")
print("Questions scored:", len(reviewed))
print("Backup:", backup)
print("Overall score distribution:")
print(reviewed["overall_score"].value_counts().sort_index())
print("Scoring completeness:")
print(reviewed[columns].notna().sum())
