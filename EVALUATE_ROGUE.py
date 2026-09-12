import json
import os

import pandas as pd
import matplotlib.pyplot as plt

from transformers import pipeline
from rouge_score import rouge_scorer
from bert_score import score as bertscore


# ==================================================
# SETTINGS
# ==================================================

DATASET_PATH = os.path.join(
    "data",
    "scitldr",
    "test.jsonl"
)

RESULTS_FOLDER = "results"

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)


# Number of papers to evaluate
NUMBER_OF_PAPERS = 20


# ==================================================
# LOAD DATASET
# ==================================================

def load_scitldr(path):

    examples = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            examples.append(
                json.loads(line)
            )

    return examples


# ==================================================
# LOAD MODELS
# ==================================================

print("Loading DistilBART...")

bart_summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)


print("Loading T5...")

t5_summarizer = pipeline(
    "summarization",
    model="t5-small"
)


# ==================================================
# GET TOKENIZERS
# ==================================================

bart_tokenizer = bart_summarizer.tokenizer

t5_tokenizer = t5_summarizer.tokenizer


# ==================================================
# LOAD ROUGE
# ==================================================

rouge = rouge_scorer.RougeScorer(
    [
        "rouge1",
        "rouge2",
        "rougeL"
    ],
    use_stemmer=True
)


# ==================================================
# TOKEN-BASED CHUNKING
# ==================================================

def create_chunks(
    text,
    tokenizer,
    chunk_size
):

    # Convert text into tokens
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []

    # Split tokens into smaller pieces
    for start in range(
        0,
        len(tokens),
        chunk_size
    ):

        chunk_tokens = tokens[
            start:start + chunk_size
        ]

        # Convert tokens back into text
        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        )

        if chunk_text.strip():

            chunks.append(
                chunk_text
            )

    return chunks


# ==================================================
# BART CHUNK SUMMARIZATION
# ==================================================

def summarize_with_bart(text):

    # DistilBART supports approximately
    # 1024 input tokens.
    #
    # We use 700 to leave some safety space.

    chunks = create_chunks(
        text,
        bart_tokenizer,
        chunk_size=700
    )

    print(
        f"      BART chunks: {len(chunks)}"
    )

    chunk_summaries = []

    for chunk_number, chunk in enumerate(chunks):

        print(
            f"      BART chunk "
            f"{chunk_number + 1}/{len(chunks)}"
        )

        try:

            result = bart_summarizer(
                chunk,
                max_length=120,
                min_length=20,
                do_sample=False
            )

            summary = result[0][
                "summary_text"
            ]

            chunk_summaries.append(
                summary
            )

        except Exception as e:

            print(
                f"      BART chunk error: {e}"
            )

    # Nothing was generated
    if not chunk_summaries:

        return ""

    # Combine summaries
    combined_summary = " ".join(
        chunk_summaries
    )

    # ------------------------------------------------
    # FINAL SUMMARIZATION
    # ------------------------------------------------

    # Check how many tokens the combined
    # summaries contain.

    combined_tokens = bart_tokenizer.encode(
        combined_summary,
        add_special_tokens=False
    )

    # If it is small enough, summarize directly
    if len(combined_tokens) <= 700:

        try:

            final_result = bart_summarizer(
                combined_summary,
                max_length=120,
                min_length=20,
                do_sample=False
            )

            return final_result[0][
                "summary_text"
            ]

        except Exception as e:

            print(
                f"      BART final error: {e}"
            )

            return combined_summary

    # ------------------------------------------------
    # SECOND LEVEL CHUNKING
    # ------------------------------------------------

    print(
        "      BART combined summary is long."
    )

    second_level_chunks = create_chunks(
        combined_summary,
        bart_tokenizer,
        chunk_size=700
    )

    second_level_summaries = []

    for chunk_number, chunk in enumerate(
        second_level_chunks
    ):

        print(
            f"      BART second-level chunk "
            f"{chunk_number + 1}/"
            f"{len(second_level_chunks)}"
        )

        try:

            result = bart_summarizer(
                chunk,
                max_length=120,
                min_length=20,
                do_sample=False
            )

            second_level_summaries.append(
                result[0]["summary_text"]
            )

        except Exception as e:

            print(
                f"      BART second-level error: {e}"
            )

    final_text = " ".join(
        second_level_summaries
    )

    # Final compression
    try:

        final_result = bart_summarizer(
            final_text,
            max_length=120,
            min_length=20,
            do_sample=False
        )

        return final_result[0][
            "summary_text"
        ]

    except Exception as e:

        print(
            f"      BART final compression error: {e}"
        )

        return final_text


# ==================================================
# T5 CHUNK SUMMARIZATION
# ==================================================

def summarize_with_t5(text):

    # T5-small has a smaller practical
    # input limit than DistilBART.
    #
    # We use 400 tokens.

    chunks = create_chunks(
        text,
        t5_tokenizer,
        chunk_size=400
    )

    print(
        f"      T5 chunks: {len(chunks)}"
    )

    chunk_summaries = []

    for chunk_number, chunk in enumerate(chunks):

        print(
            f"      T5 chunk "
            f"{chunk_number + 1}/{len(chunks)}"
        )

        try:

            result = t5_summarizer(
                "summarize: " + chunk,
                max_length=120,
                min_length=20,
                do_sample=False
            )

            summary = result[0][
                "summary_text"
            ]

            chunk_summaries.append(
                summary
            )

        except Exception as e:

            print(
                f"      T5 chunk error: {e}"
            )

    # Nothing was generated
    if not chunk_summaries:

        return ""

    # Combine summaries
    combined_summary = " ".join(
        chunk_summaries
    )

    # ------------------------------------------------
    # FINAL SUMMARIZATION
    # ------------------------------------------------

    combined_tokens = t5_tokenizer.encode(
        combined_summary,
        add_special_tokens=False
    )

    # If small enough, summarize directly
    if len(combined_tokens) <= 400:

        try:

            final_result = t5_summarizer(
                "summarize: " + combined_summary,
                max_length=120,
                min_length=20,
                do_sample=False
            )

            return final_result[0][
                "summary_text"
            ]

        except Exception as e:

            print(
                f"      T5 final error: {e}"
            )

            return combined_summary

    # ------------------------------------------------
    # SECOND LEVEL CHUNKING
    # ------------------------------------------------

    print(
        "      T5 combined summary is long."
    )

    second_level_chunks = create_chunks(
        combined_summary,
        t5_tokenizer,
        chunk_size=400
    )

    second_level_summaries = []

    for chunk_number, chunk in enumerate(
        second_level_chunks
    ):

        print(
            f"      T5 second-level chunk "
            f"{chunk_number + 1}/"
            f"{len(second_level_chunks)}"
        )

        try:

            result = t5_summarizer(
                "summarize: " + chunk,
                max_length=120,
                min_length=20,
                do_sample=False
            )

            second_level_summaries.append(
                result[0]["summary_text"]
            )

        except Exception as e:

            print(
                f"      T5 second-level error: {e}"
            )

    final_text = " ".join(
        second_level_summaries
    )

    # Final compression
    try:

        final_result = t5_summarizer(
            "summarize: " + final_text,
            max_length=120,
            min_length=20,
            do_sample=False
        )

        return final_result[0][
            "summary_text"
        ]

    except Exception as e:

        print(
            f"      T5 final compression error: {e}"
        )

        return final_text


# ==================================================
# LOAD DATA
# ==================================================

print(
    "\nLoading SciTLDR test dataset..."
)

dataset = load_scitldr(
    DATASET_PATH
)

print(
    f"Loaded {len(dataset)} papers."
)


# ==================================================
# SELECT PAPERS
# ==================================================

dataset = dataset[
    :NUMBER_OF_PAPERS
]

print(
    f"Evaluating first "
    f"{len(dataset)} papers."
)


# ==================================================
# RESULTS
# ==================================================

results = []


# ==================================================
# PROCESS PAPERS
# ==================================================

for index, item in enumerate(dataset):

    print("\n")
    print("=" * 60)

    print(
        f"Processing paper "
        f"{index + 1} / {len(dataset)}"
    )

    print("=" * 60)


    # ------------------------------------------------
    # SOURCE TEXT
    # ------------------------------------------------

    source = " ".join(
        item["source"]
    )

    print(
        f"Source characters: "
        f"{len(source)}"
    )


    # ------------------------------------------------
    # REFERENCE SUMMARY
    # ------------------------------------------------

    targets = item.get(
        "target",
        []
    )

    if not targets:

        print(
            "No reference summary found."
        )

        continue


    # Use first reference summary
    reference_summary = targets[0]


    # ------------------------------------------------
    # BART
    # ------------------------------------------------

    print("\nRunning DistilBART...")

    bart_summary = summarize_with_bart(
        source
    )

    print(
        "\nBART final summary:"
    )

    print(
        bart_summary
    )


    # ------------------------------------------------
    # T5
    # ------------------------------------------------

    print("\nRunning T5...")

    t5_summary = summarize_with_t5(
        source
    )

    print(
        "\nT5 final summary:"
    )

    print(
        t5_summary
    )


    # ------------------------------------------------
    # ROUGE
    # ------------------------------------------------

    bart_rouge = rouge.score(
        reference_summary,
        bart_summary
    )

    t5_rouge = rouge.score(
        reference_summary,
        t5_summary
    )


    # ------------------------------------------------
    # BERTSCORE
    # ------------------------------------------------

    try:

        bart_p, bart_r, bart_f1 = bertscore(
            [bart_summary],
            [reference_summary],
            lang="en",
            verbose=False
        )

        t5_p, t5_r, t5_f1 = bertscore(
            [t5_summary],
            [reference_summary],
            lang="en",
            verbose=False
        )

        bart_bert_f1 = (
            bart_f1.item()
        )

        t5_bert_f1 = (
            t5_f1.item()
        )

    except Exception as e:

        print(
            f"BERTScore error: {e}"
        )

        bart_bert_f1 = None
        t5_bert_f1 = None


    # ------------------------------------------------
    # SAVE RESULT
    # ------------------------------------------------

    results.append({

        "paper_id":
            item.get(
                "paper_id",
                ""
            ),

        "title":
            item.get(
                "title",
                ""
            ),

        "BART_ROUGE1":
            bart_rouge[
                "rouge1"
            ].fmeasure,

        "BART_ROUGE2":
            bart_rouge[
                "rouge2"
            ].fmeasure,

        "BART_ROUGEL":
            bart_rouge[
                "rougeL"
            ].fmeasure,

        "BART_BERTScore":
            bart_bert_f1,

        "T5_ROUGE1":
            t5_rouge[
                "rouge1"
            ].fmeasure,

        "T5_ROUGE2":
            t5_rouge[
                "rouge2"
            ].fmeasure,

        "T5_ROUGEL":
            t5_rouge[
                "rougeL"
            ].fmeasure,

        "T5_BERTScore":
            t5_bert_f1,
            
        "BART_summary":
            bart_summary,

        "T5_summary":
            t5_summary
    })


# ==================================================
# CREATE DATAFRAME
# ==================================================

results_df = pd.DataFrame(
    results
)


# ==================================================
# SAVE CSV
# ==================================================

csv_path = os.path.join(
    RESULTS_FOLDER,
    "evaluation_results_chunked.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# ==================================================
# PRINT INDIVIDUAL RESULTS
# ==================================================

print("\n")
print("=" * 60)

print(
    "CHUNKED MODEL RESULTS"
)

print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ==================================================
# AVERAGE SCORES
# ==================================================

score_columns = [

    "BART_ROUGE1",
    "BART_ROUGE2",
    "BART_ROUGEL",
    "BART_BERTScore",

    "T5_ROUGE1",
    "T5_ROUGE2",
    "T5_ROUGEL",
    "T5_BERTScore"
]


averages = results_df[
    score_columns
].mean()


print("\n")
print("=" * 60)

print(
    "AVERAGE MODEL SCORES"
)

print("=" * 60)


print(
    averages
)


# ==================================================
# MODEL AVERAGE
# ==================================================

bart_average = averages[
    [
        "BART_ROUGE1",
        "BART_ROUGE2",
        "BART_ROUGEL",
        "BART_BERTScore"
    ]
].mean()


t5_average = averages[
    [
        "T5_ROUGE1",
        "T5_ROUGE2",
        "T5_ROUGEL",
        "T5_BERTScore"
    ]
].mean()


print("\n")

print(
    f"Average BART score: "
    f"{bart_average:.4f}"
)

print(
    f"Average T5 score: "
    f"{t5_average:.4f}"
)


# ==================================================
# WINNER
# ==================================================

if bart_average > t5_average:

    print(
        "\nOverall winner: DistilBART"
    )

elif t5_average > bart_average:

    print(
        "\nOverall winner: T5"
    )

else:

    print(
        "\nOverall result: Tie"
    )


# ==================================================
# GRAPH
# ==================================================

models = [
    "DistilBART",
    "T5"
]

average_scores = [
    bart_average,
    t5_average
]


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    models,
    average_scores
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Average Evaluation Score"
)

plt.title(
    "Chunked DistilBART vs T5"
)

plt.tight_layout()


plot_path = os.path.join(
    RESULTS_FOLDER,
    "chunked_evaluation_plot.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.show()


# ==================================================
# FINAL MESSAGE
# ==================================================

print("\n")
print(
    f"Results saved to: "
    f"{csv_path}"
)

print(
    f"Graph saved to: "
    f"{plot_path}"
)