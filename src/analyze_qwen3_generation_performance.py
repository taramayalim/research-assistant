import json
import pandas as pd
from pathlib import Path

path = Path("data/evaluation/rag_generation_80/qwen3_8b_answers.jsonl")

records = []
with path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

df = pd.DataFrame(records)

print("========== 1. DATA VALIDATION ==========")
print("Total records:", len(df))
print("Unique questions:", df["question_id"].nunique())
print("Models:", df["model"].unique())
print("Missing values:")
print(df[["latency_seconds", "total_duration_ns",
          "load_duration_ns", "prompt_tokens",
          "generated_tokens"]].isna().sum())

if len(df) != 80 or df["question_id"].nunique() != 80:
    raise ValueError("Expected exactly 80 unique generation records.")

print("\n========== 2. GENERATION LATENCY ==========")
for col in ["latency_seconds"]:
    print(f"Mean:   {df[col].mean():.2f} seconds")
    print(f"Median: {df[col].median():.2f} seconds")
    print(f"Minimum: {df[col].min():.2f} seconds")
    print(f"Maximum: {df[col].max():.2f} seconds")
    print(f"Std dev: {df[col].std():.2f} seconds")

print("\nLatency percentiles:")
print(df["latency_seconds"].quantile([0.25, 0.50, 0.75, 0.90]).round(2))

print("\n========== 3. TOKEN STATISTICS ==========")
for col in ["prompt_tokens", "generated_tokens"]:
    print(f"\n{col}")
    print(df[col].describe().round(2))

print("\n========== 4. MODEL LOADING ==========")
df["load_seconds"] = df["load_duration_ns"] / 1_000_000_000
print(df["load_seconds"].describe().round(2))

print("\n========== 5. GENERATION SPEED ==========")
df["tokens_per_second"] = (
    df["generated_tokens"] / df["latency_seconds"]
)
print("Average generated tokens per second:",
      round(df["tokens_per_second"].mean(), 3))
print("Median generated tokens per second:",
      round(df["tokens_per_second"].median(), 3))

print("\n========== 6. SLOWEST 10 QUESTIONS ==========")
print(
    df.nlargest(10, "latency_seconds")[
        ["question_id", "latency_seconds",
         "prompt_tokens", "generated_tokens"]
    ].to_string(index=False)
)

print("\n========== 7. FASTEST 10 QUESTIONS ==========")
print(
    df.nsmallest(10, "latency_seconds")[
        ["question_id", "latency_seconds",
         "prompt_tokens", "generated_tokens"]
    ].to_string(index=False)
)

output = Path("data/evaluation/rag_answer_evaluation/qwen3_8b_generation_performance.csv")

summary = pd.DataFrame({
    "metric": [
        "mean_latency_seconds",
        "median_latency_seconds",
        "min_latency_seconds",
        "max_latency_seconds",
        "mean_prompt_tokens",
        "mean_generated_tokens",
        "mean_model_load_seconds",
        "mean_tokens_per_second"
    ],
    "value": [
        df["latency_seconds"].mean(),
        df["latency_seconds"].median(),
        df["latency_seconds"].min(),
        df["latency_seconds"].max(),
        df["prompt_tokens"].mean(),
        df["generated_tokens"].mean(),
        df["load_seconds"].mean(),
        df["tokens_per_second"].mean()
    ]
})

summary.to_csv(output, index=False)

print("\nPerformance summary saved to:", output)
