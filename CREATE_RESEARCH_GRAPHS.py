import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = os.path.join(
    "results",
    "evaluation_results_chunked.csv"
)


df = pd.read_csv(INPUT_PATH)


metrics = [
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "BERTScore"
]


bart_scores = [
    df["BART_ROUGE1"].mean(),
    df["BART_ROUGE2"].mean(),
    df["BART_ROUGEL"].mean(),
    df["BART_BERTScore"].mean()
]


t5_scores = [
    df["T5_ROUGE1"].mean(),
    df["T5_ROUGE2"].mean(),
    df["T5_ROUGEL"].mean(),
    df["T5_BERTScore"].mean()
]


# ============================================================
# ALL METRICS
# ============================================================

x = range(len(metrics))

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    bart_scores,
    width=width,
    label="BART"
)

plt.bar(
    [i + width / 2 for i in x],
    t5_scores,
    width=width,
    label="T5"
)

plt.xticks(
    list(x),
    metrics
)

plt.ylabel("Average Score")

plt.xlabel("Evaluation Metric")

plt.title(
    "BART vs T5 Automatic Evaluation"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/automatic_metric_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# ROUGE ONLY
# ============================================================

rouge_metrics = [
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L"
]


bart_rouge = [
    df["BART_ROUGE1"].mean(),
    df["BART_ROUGE2"].mean(),
    df["BART_ROUGEL"].mean()
]


t5_rouge = [
    df["T5_ROUGE1"].mean(),
    df["T5_ROUGE2"].mean(),
    df["T5_ROUGEL"].mean()
]


x = range(len(rouge_metrics))


plt.figure(figsize=(9, 6))

plt.bar(
    [i - width / 2 for i in x],
    bart_rouge,
    width=width,
    label="BART"
)

plt.bar(
    [i + width / 2 for i in x],
    t5_rouge,
    width=width,
    label="T5"
)

plt.xticks(
    list(x),
    rouge_metrics
)

plt.ylabel("Average Score")

plt.title(
    "ROUGE Score Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/rouge_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# BERTSCORE
# ============================================================

plt.figure(figsize=(7, 6))

plt.bar(
    ["BART", "T5"],
    [
        df["BART_BERTScore"].mean(),
        df["T5_BERTScore"].mean()
    ]
)

plt.ylabel("Average BERTScore")

plt.title(
    "BERTScore Comparison"
)

plt.tight_layout()

plt.savefig(
    "results/bertscore_comparison.png",
    dpi=300
)

plt.close()


print("Research graphs created successfully.")

print("\nAverage scores:")

for name, score in zip(
    metrics,
    bart_scores
):

    print(
        f"BART {name}: {score:.4f}"
    )


for name, score in zip(
    metrics,
    t5_scores
):

    print(
        f"T5 {name}: {score:.4f}"
    )