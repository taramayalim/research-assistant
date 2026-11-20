
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "retrieval_results"
PLOT_DIR = PROJECT_ROOT / "data" / "evaluation" / "retrieval_plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load retrieval summary
summary_file = RESULTS_DIR / "retrieval_metrics_summary.csv"
df = pd.read_csv(summary_file)

print("Retrieval Evaluation Summary:")
print(df.to_string(index=False))

# Plot 1: Recall@5
plt.figure(figsize=(9, 5))
plt.bar(df["method"], df["mean_recall_at_5"])
plt.title("Retrieval Method Comparison - Recall@5")
plt.xlabel("Retrieval Method")
plt.ylabel("Mean Recall@5")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(PLOT_DIR / "recall_at_5_comparison.png", dpi=300)
plt.show()
plt.close()

# Plot 2: nDCG@10
plt.figure(figsize=(9, 5))
plt.bar(df["method"], df["mean_ndcg_at_10"])
plt.title("Retrieval Method Comparison - nDCG@10")
plt.xlabel("Retrieval Method")
plt.ylabel("Mean nDCG@10")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(PLOT_DIR / "ndcg_at_10_comparison.png", dpi=300)
plt.show()
plt.close()

# Plot 3: Mean Retrieval Latency
plt.figure(figsize=(9, 5))
plt.bar(df["method"], df["mean_latency_seconds"])
plt.title("Retrieval Method Comparison - Mean Latency")
plt.xlabel("Retrieval Method")
plt.ylabel("Mean Latency (seconds)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(PLOT_DIR / "retrieval_latency_comparison.png", dpi=300)
plt.show()
plt.close()

print("\nAll three plots generated successfully!")
print(f"Saved in: {PLOT_DIR}")
