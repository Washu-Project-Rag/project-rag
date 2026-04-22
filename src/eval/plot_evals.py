import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # goes up from src/eval → project-rag

csv_path = BASE_DIR / "eval" / "eval_scores.csv"

df = pd.read_csv(csv_path)
# print(df.head())

grouped_df = df.groupby("k").mean(numeric_only=True).reset_index()

k = grouped_df["k"].tolist()

correctness_scores = grouped_df["correctness_score"].tolist()
faithfulness_scores = grouped_df["faithfulness_score"].tolist()
context_scores = grouped_df["context_precision_recall_score"].tolist()

x_axis = np.arange(len(k))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x_axis - width, correctness_scores, width=width, label="Correctness")
plt.bar(x_axis, faithfulness_scores, width=width, label="Faithfulness")
plt.bar(x_axis + width, context_scores, width=width, label="Context Precision/Recall")

plt.xlabel("Top K Retrieved Chunks")
plt.ylabel("Average Score")
plt.title("Evaluation Metrics vs. Top K Retrieved Chunks")
plt.xticks(x_axis, k)
plt.ylim(0, 1)
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.grid(axis="y", linestyle="--", alpha=0.7)

for i in range(len(k)):
    plt.text(x_axis[i] - width, correctness_scores[i] + 0.01, f"{correctness_scores[i]:.2f}", ha='center', fontsize=8)
    plt.text(x_axis[i], faithfulness_scores[i] + 0.01, f"{faithfulness_scores[i]:.2f}", ha='center', fontsize=8)
    plt.text(x_axis[i] + width, context_scores[i] + 0.01, f"{context_scores[i]:.2f}", ha='center', fontsize=8)


plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
# plt.savefig("eval/eval_plots.png")
plt.show()