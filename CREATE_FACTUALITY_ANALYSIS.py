import json
import os
import pandas as pd


DATASET_PATH = os.path.join(
    "data",
    "scitldr",
    "test.jsonl"
)

RESULTS_PATH = os.path.join(
    "results",
    "evaluation_results_chunked.csv"
)

OUTPUT_PATH = os.path.join(
    "results",
    "factuality_analysis.csv"
)


# Load SciTLDR
print("Loading SciTLDR dataset...")

dataset = []

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as file:

    for line in file:
        dataset.append(json.loads(line))


# Load BART/T5 results
print("Loading model results...")

results_df = pd.read_csv(RESULTS_PATH)


# Prepare rows for human inspection
analysis_rows = []

for _, result in results_df.iterrows():

    paper_id = result["paper_id"]

    matching_papers = [
        item
        for item in dataset
        if item.get("paper_id") == paper_id
    ]

    if not matching_papers:
        continue

    paper = matching_papers[0]

    source = " ".join(paper["source"])

    targets = paper.get("target", [])

    reference = targets[0] if targets else ""


    analysis_rows.append({

        "paper_id": paper_id,

        "title": result["title"],

        "source_text": source,

        "reference_summary": reference,

        "BART_summary": result.get(
            "BART_summary",
            ""
        ),

        "T5_summary": result.get(
            "T5_summary",
            ""
        ),

        "BART_error_type": "",

        "T5_error_type": "",

        "BART_factuality_notes": "",

        "T5_factuality_notes": ""
    })


# Save file
analysis_df = pd.DataFrame(analysis_rows)

analysis_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nFactuality analysis file created!")

print(
    "Saved to:",
    OUTPUT_PATH
)

print(
    "Number of papers:",
    len(analysis_df)
)