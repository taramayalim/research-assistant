
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RESULT_FILE = (
    ROOT / "data" / "evaluation" /
    "retrieval_results" / "retrieved_chunks_top10.csv"
)

QRELS_FILE = ROOT / "data" / "evaluation" / "qrels.csv"

# Load evaluation results
results = pd.read_csv(RESULT_FILE, dtype=str)
qrels = pd.read_csv(QRELS_FILE, dtype=str)

results["rank"] = pd.to_numeric(results["rank"])
qrels["relevance_grade"] = pd.to_numeric(
    qrels["relevance_grade"]
)

# Keep only relevant chunks
relevant = qrels[qrels["relevance_grade"] > 0].copy()

methods = results["method"].unique()

summary = []
missed_questions = []

for method in methods:
    method_results = results[results["method"] == method]

    question_ids = sorted(method_results["question_id"].unique())

    hit5 = 0
    hit10 = 0
    relevant_total = 0
    relevant_found5 = 0
    relevant_found10 = 0

    for qid in question_ids:
        expected = set(
            relevant.loc[
                relevant["query_id"] == qid, "chunk_id"
            ]
        )

        retrieved = method_results[
            method_results["question_id"] == qid
        ].sort_values("rank")

        top5 = set(retrieved[retrieved["rank"] <= 5]["chunk_id"])
        top10 = set(retrieved["chunk_id"])

        found5 = expected.intersection(top5)
        found10 = expected.intersection(top10)

        relevant_total += len(expected)
        relevant_found5 += len(found5)
        relevant_found10 += len(found10)

        if found5:
            hit5 += 1

        if found10:
            hit10 += 1

        if not found10:
            missed_questions.append({
                "method": method,
                "question_id": qid,
                "relevant_chunks": "; ".join(sorted(expected))
            })

    summary.append({
        "method": method,
        "questions": len(question_ids),
        "questions_hit_at_5": hit5,
        "questions_hit_at_10": hit10,
        "question_hit_rate_at_5": round(hit5 / len(question_ids), 4),
        "question_hit_rate_at_10": round(hit10 / len(question_ids), 4),
        "relevant_chunks": relevant_total,
        "relevant_chunks_found_at_5": relevant_found5,
        "relevant_chunks_found_at_10": relevant_found10,
        "chunk_recall_at_5": round(
            relevant_found5 / relevant_total, 4
        ) if relevant_total else 0,
        "chunk_recall_at_10": round(
            relevant_found10 / relevant_total, 4
        ) if relevant_total else 0
    })

# Save diagnostic reports
output_dir = ROOT / "data" / "evaluation" / "retrieval_diagnostics"
output_dir.mkdir(parents=True, exist_ok=True)

summary_df = pd.DataFrame(summary)
missed_df = pd.DataFrame(missed_questions)

summary_df.to_csv(
    output_dir / "retrieval_failure_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

missed_df.to_csv(
    output_dir / "questions_missed_at_top10.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========== RETRIEVAL FAILURE SUMMARY ==========")
print(summary_df.to_string(index=False))

print("\n========== QUESTIONS MISSED IN TOP 10 ==========")
if not missed_df.empty:
    print(missed_df.to_string(index=False))
else:
    print("No missed questions.")

print("\nReports saved in:")
print(output_dir)
