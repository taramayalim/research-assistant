import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

path = Path("data/evaluation/rag_answer_evaluation/qwen3_8b_manual_evaluation.csv")

if not path.exists():
    raise FileNotFoundError(path)

df = pd.read_csv(path)

if len(df) != 80 or df["question_id"].nunique() != 80:
    raise ValueError("Expected exactly 80 unique questions.")

scores = {
    "Q021": (2,2,1,2,2,None,0,2,"Correct general explanation; cited papers differ from the gold-answer source."),
    "Q022": (2,2,2,2,2,None,0,2,"Correct purpose of self-supervised learning."),
    "Q023": (2,2,2,2,2,None,0,2,"Correctly identifies BYOL, MIM and contrastive learning."),
    "Q024": (1,1,1,1,2,None,1,1,"Reverses the PlantDoc and PlantVillage sequence in the expected explanation."),
    "Q025": (2,2,2,2,2,None,0,2,"Correct explanation of MixUp and CutMix."),
    "Q026": (2,2,2,2,2,None,0,2,"Correct explanation of hierarchical shifted-window attention."),
    "Q027": (2,1,2,2,2,None,0,2,"Identifies class imbalance but omits two example counts and the total of 56 classes."),
    "Q028": (2,2,2,2,2,None,0,2,"Correct explanation of augmentation benefits."),
    "Q029": (2,2,2,2,2,None,0,2,"Correct explanation of augmentation and environmental robustness."),
    "Q030": (2,2,1,2,2,None,0,2,"Correct explanation; cited sources differ from the gold-answer source."),
    "Q031": (2,2,2,2,2,None,0,2,"Correct explanation of Grad-CAM and its purpose."),
    "Q032": (2,2,1,2,2,None,0,2,"Correct LIME explanation; cited page differs from the gold-answer source."),
    "Q033": (2,2,1,2,2,None,0,2,"Correct explanation of MobileNetV2 suitability; one supporting citation differs from the target study."),
    "Q034": (1,2,2,1,2,None,1,1,"Core explanation is relevant, but interpretability and error-reduction claims are not established by the cited evidence."),
    "Q035": (2,2,1,2,2,None,0,2,"Relevant limitations discussed, but citations differ from the gold-answer source."),
    "Q036": (2,2,2,2,2,None,0,2,"Correct explanation of MixUp and CutMix."),
    "Q037": (2,2,1,2,2,None,0,2,"Correct generalisation explanation; cited studies differ from the gold-answer source."),
    "Q038": (2,2,1,2,2,None,0,2,"Correct definitions of precision, recall and F1-score; citations differ from the gold-answer source."),
    "Q039": (1,2,1,1,2,None,1,1,"Correct harmonic-mean explanation, but overstates standard F1; macro-F1 is needed for equal class weighting."),
    "Q040": (2,2,1,2,2,None,0,2,"Correct confusion-matrix explanation; citations differ from the gold-answer source.")
}

score_columns = [
    "factual_correctness", "completeness", "citation_accuracy",
    "groundedness", "relevance", "abstention_correct",
    "hallucination", "overall_score"
]

for col in score_columns + ["reviewer_comments"]:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

for qid in scores:
    if (df["question_id"] == qid).sum() != 1:
        raise ValueError(f"{qid} must occur exactly once.")
    row = df.loc[df["question_id"] == qid]
    if row["overall_score"].notna().any():
        raise ValueError(f"{qid} already has an overall score. No changes made.")

backup = path.with_name(
    f"{path.stem}_backup_before_q021_q040_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}"
)
shutil.copy2(path, backup)

df["reviewer_comments"] = df["reviewer_comments"].astype("string")

for qid, values in scores.items():
    mask = df["question_id"].eq(qid)
    for col, value in zip(score_columns, values[:8]):
        df.loc[mask, col] = value
    df.loc[mask, "reviewer_comments"] = values[8]

df.to_csv(path, index=False, encoding="utf-8")

check = pd.read_csv(path)
reviewed = check[check["question_id"].isin(scores)]

print("Q021-Q040 scores saved successfully.")
print("Questions scored:", len(reviewed))
print("Backup:", backup)
print("Overall score distribution:")
print(reviewed["overall_score"].value_counts().sort_index())
print("Scoring completeness:")
print(reviewed[score_columns].notna().sum())
