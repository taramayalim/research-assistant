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
    "Q041": (1,1,1,1,2,None,1,1,"Comparison is incomplete and overstates PlantNet's superiority; the studies use different experimental settings."),
    "Q042": (0,1,0,0,1,None,2,0,"Incorrectly guesses InceptionV3 or DenseNet for PlantNet instead of MobileNet V1."),
    "Q043": (2,2,1,1,2,None,1,2,"Main comparison is correct, but the interpretability claim about the self-supervised framework is unsupported."),
    "Q044": (1,1,1,1,2,None,1,1,"Correct Mob-Res description, but discusses modified ResNet50 from A076 instead of A019."),
    "Q045": (1,1,1,1,2,None,1,1,"Uses unrelated A068 and A108 findings instead of comparing A006 and A072."),
    "Q046": (2,2,1,2,2,None,0,2,"Correct conceptual comparison; A007 is not cited, but the answer is supported by relevant evidence."),
    "Q047": (1,1,1,1,2,None,1,1,"Correct Mob-Res explanation, but speculates about PlantNet XAI rather than identifying its stated future work."),
    "Q048": (1,1,2,2,2,None,0,1,"Correctly explains Swin Transformer but abstains on Mob-Res despite the question requiring both."),
    "Q049": (1,1,2,2,2,None,0,1,"Correct Mob-Res details, but PlantNet comparison is missing due to abstention."),
    "Q050": (2,2,1,2,2,None,0,2,"Correct comparison of labelled and unlabelled data; cited sources differ from the gold-answer sources."),
    "Q051": (1,1,1,1,2,None,1,1,"Explains augmentation and transfer learning generally but omits MixUp and CutMix specificity and adds unsupported diffusion claims."),
    "Q052": (1,1,1,2,2,None,0,1,"Correct Mob-Res XAI methods, but abstains on PlantNet's future-work recommendations."),
    "Q053": (2,1,2,2,2,None,0,2,"Correct CNN versus Swin comparison, but omits the specific CNN architectures in A006."),
    "Q054": (1,1,1,1,2,None,1,1,"Mob-Res is broadly correct, but PlantNet architecture and performance are incorrectly described."),
    "Q055": (0,0,1,2,1,None,0,0,"Abstains although the question is answerable; the required A006 and A114 details were not retrieved."),
    "Q056": (1,1,1,1,2,None,1,1,"Uses unrelated A088 and A104 studies instead of the specified A002 and A158."),
    "Q057": (1,1,1,1,2,None,1,1,"Misrepresents A007 and fails to discuss the A013 self-supervised framework."),
    "Q058": (0,0,1,2,1,None,0,0,"Abstains on an answerable comparison; relevant A027 and A013 evidence is missing."),
    "Q059": (1,1,1,1,2,None,1,1,"Uses A068 and A007 rather than comparing the specified A006 and A072 studies."),
    "Q060": (0,0,1,1,1,None,2,0,"Incorrectly attributes ensemble learning to A013 and a CNN-XGBoost hybrid to A027.")
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
        raise ValueError(f"{qid} already scored. No changes made.")

backup = path.with_name(
    f"{path.stem}_backup_before_q041_q060_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}"
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

print("Q041-Q060 scores saved successfully.")
print("Questions scored:", len(reviewed))
print("Backup:", backup)
print("Overall score distribution:")
print(reviewed["overall_score"].value_counts().sort_index())
print("Scoring completeness:")
print(reviewed[columns].notna().sum())
