import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = os.path.join(
    "results",
    "factuality_analysis_completed.csv"
)

SUMMARY_OUTPUT = os.path.join(
    "results",
    "factuality_summary.csv"
)

COMPARISON_OUTPUT = os.path.join(
    "results",
    "factuality_model_comparison.csv"
)

GRAPH_OUTPUT = os.path.join(
    "results",
    "factuality_errors.png"
)

RATE_GRAPH_OUTPUT = os.path.join(
    "results",
    "factuality_rates.png"
)


print("Loading factuality analysis...")

df = pd.read_csv(INPUT_PATH)

print("Number of papers:", len(df))


categories = [
    "Correct",
    "Missing information",
    "Incorrect",
    "Repetition",
    "Hallucination"
]


def count_categories(column):

    counts = {}

    for category in categories:

        counts[category] = sum(
            df[column]
            .fillna("")
            .str.lower()
            .str.contains(
                category.lower(),
                regex=False
            )
        )

    return counts


bart_counts = count_categories(
    "BART_error_type"
)

t5_counts = count_categories(
    "T5_error_type"
)


summary_df = pd.DataFrame({
    "Error Type": categories,

    "BART": [
        bart_counts[x]
        for x in categories
    ],

    "T5": [
        t5_counts[x]
        for x in categories
    ]
})


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


print("\nFactuality Summary")
print(summary_df)


number_of_papers = len(df)


bart_hallucination_rate = (
    bart_counts["Hallucination"]
    / number_of_papers
    * 100
)

t5_hallucination_rate = (
    t5_counts["Hallucination"]
    / number_of_papers
    * 100
)


bart_correct_rate = (
    bart_counts["Correct"]
    / number_of_papers
    * 100
)

t5_correct_rate = (
    t5_counts["Correct"]
    / number_of_papers
    * 100
)


comparison_df = pd.DataFrame({

    "Metric": [
        "Correct rate",
        "Hallucination rate"
    ],

    "BART": [
        bart_correct_rate,
        bart_hallucination_rate
    ],

    "T5": [
        t5_correct_rate,
        t5_hallucination_rate
    ]
})


comparison_df.to_csv(
    COMPARISON_OUTPUT,
    index=False
)


print("\n--------------------------------")
print("Factuality Results")
print("--------------------------------")

print(
    f"BART correct rate: "
    f"{bart_correct_rate:.2f}%"
)

print(
    f"T5 correct rate: "
    f"{t5_correct_rate:.2f}%"
)

print(
    f"BART hallucination rate: "
    f"{bart_hallucination_rate:.2f}%"
)

print(
    f"T5 hallucination rate: "
    f"{t5_hallucination_rate:.2f}%"
)


# ============================================================
# GRAPH 1 — ERROR TYPES
# ============================================================

x = range(len(categories))

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    summary_df["BART"],
    width=width,
    label="BART"
)

plt.bar(
    [i + width / 2 for i in x],
    summary_df["T5"],
    width=width,
    label="T5"
)

plt.xticks(
    list(x),
    categories,
    rotation=20
)

plt.xlabel("Factuality / Error Type")

plt.ylabel("Number of Papers")

plt.title(
    "Human Factuality Error Analysis"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    GRAPH_OUTPUT,
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2 — CORRECT / HALLUCINATION RATE
# ============================================================

metrics = [
    "Correct rate",
    "Hallucination rate"
]

x = range(len(metrics))

plt.figure(figsize=(8, 6))

plt.bar(
    [i - width / 2 for i in x],
    comparison_df["BART"],
    width=width,
    label="BART"
)

plt.bar(
    [i + width / 2 for i in x],
    comparison_df["T5"],
    width=width,
    label="T5"
)

plt.xticks(
    list(x),
    metrics
)

plt.ylabel("Percentage (%)")

plt.title(
    "BART vs T5 Factuality Rates"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RATE_GRAPH_OUTPUT,
    dpi=300
)

plt.close()


print("\nGraphs created successfully!")

print(
    "Saved:",
    GRAPH_OUTPUT
)

print(
    "Saved:",
    RATE_GRAPH_OUTPUT
)