import os
import re

import streamlit as st
import pandas as pd
import pymupdf
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📄",
    layout="wide"
)


st.title("📄 Research Paper Summarizer")
st.write(
    "Compare BART and T5 summaries and automatically check "
    "their factual consistency with the source paper."
)


# ============================================================
# MODEL NAMES
# ============================================================

BART_MODEL = "sshleifer/distilbart-cnn-12-6"
T5_MODEL = "t5-small"

# NLI model for factuality checking
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


# ============================================================
# LOAD SUMMARIZATION MODELS
# ============================================================

@st.cache_resource
def load_bart():

    tokenizer = AutoTokenizer.from_pretrained(
        BART_MODEL
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        BART_MODEL
    )

    return tokenizer, model


@st.cache_resource
def load_t5():

    tokenizer = AutoTokenizer.from_pretrained(
        T5_MODEL
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        T5_MODEL
    )

    return tokenizer, model


# ============================================================
# LOAD NLI MODEL
# ============================================================

@st.cache_resource
def load_nli():

    tokenizer = AutoTokenizer.from_pretrained(
        NLI_MODEL
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL
    )

    return tokenizer, model


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file):

    pdf_bytes = uploaded_file.read()

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page in document:

        text = page.get_text()

        if text:
            pages.append(text)

    document.close()

    return "\n".join(pages)


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(text, chunk_size=250):

    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ============================================================
# BART SUMMARY
# ============================================================

def summarize_bart(
    text,
    tokenizer,
    model
):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():

        summary_ids = model.generate(
            **inputs,
            max_length=180,
            min_length=50,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


# ============================================================
# T5 SUMMARY
# ============================================================

def summarize_t5(
    text,
    tokenizer,
    model
):

    prompt = "summarize: " + text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        summary_ids = model.generate(
            **inputs,
            max_length=180,
            min_length=50,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


# ============================================================
# HIERARCHICAL SUMMARIZATION
# ============================================================

def summarize_document(
    text,
    model_type,
    tokenizer,
    model,
    chunk_size=250
):

    chunks = chunk_text(
        text,
        chunk_size
    )

    chunk_summaries = []

    for i, chunk in enumerate(chunks):

        if model_type == "BART":

            summary = summarize_bart(
                chunk,
                tokenizer,
                model
            )

        else:

            summary = summarize_t5(
                chunk,
                tokenizer,
                model
            )

        chunk_summaries.append(summary)

    combined_summary = " ".join(
        chunk_summaries
    )

    # Final compression if many chunk summaries exist
    if len(combined_summary.split()) > 250:

        if model_type == "BART":

            final_summary = summarize_bart(
                combined_summary,
                tokenizer,
                model
            )

        else:

            final_summary = summarize_t5(
                combined_summary,
                tokenizer,
                model
            )

        return final_summary

    return combined_summary


# ============================================================
# SPLIT SUMMARY INTO SENTENCES
# ============================================================

def split_into_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# SIMPLE RELEVANCE SEARCH
# ============================================================

def find_relevant_source_sentences(
    claim,
    source_text,
    top_k=5
):

    source_sentences = re.split(
        r'(?<=[.!?])\s+',
        source_text
    )

    source_sentences = [
        s.strip()
        for s in source_sentences
        if len(s.strip()) > 20
    ]

    claim_words = set(
        re.findall(
            r'\b[a-zA-Z]{3,}\b',
            claim.lower()
        )
    )

    scored = []

    for sentence in source_sentences:

        sentence_words = set(
            re.findall(
                r'\b[a-zA-Z]{3,}\b',
                sentence.lower()
            )
        )

        overlap = len(
            claim_words.intersection(
                sentence_words
            )
        )

        scored.append(
            (
                overlap,
                sentence
            )
        )

    scored.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    return [
        sentence
        for score, sentence
        in scored[:top_k]
        if score > 0
    ]


# ============================================================
# NLI PREDICTION
# ============================================================

def nli_prediction(
    premise,
    hypothesis,
    tokenizer,
    model
):

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    predicted_id = torch.argmax(
        probabilities
    ).item()

    label = model.config.id2label[
        predicted_id
    ].lower()

    confidence = probabilities[
        predicted_id
    ].item()

    return label, confidence


# ============================================================
# FACTUALITY CHECK
# ============================================================

def check_factuality(
    summary,
    source_text,
    tokenizer,
    model
):

    sentences = split_into_sentences(
        summary
    )

    results = []

    for claim in sentences:

        relevant_sources = (
            find_relevant_source_sentences(
                claim,
                source_text,
                top_k=5
            )
        )

        if not relevant_sources:

            results.append({
                "claim": claim,
                "status": "Uncertain",
                "confidence": 0.0,
                "source": ""
            })

            continue

        best_label = None
        best_confidence = 0.0
        best_source = ""

        for source_sentence in relevant_sources:

            label, confidence = nli_prediction(
                source_sentence,
                claim,
                tokenizer,
                model
            )

            if confidence > best_confidence:

                best_confidence = confidence
                best_label = label
                best_source = source_sentence

        # Convert model labels into user-friendly labels

        if "entail" in best_label:

            status = "Supported"

        elif "contrad" in best_label:

            status = "Contradicted"

        else:

            status = "Uncertain"

        results.append({
            "claim": claim,
            "status": status,
            "confidence": best_confidence,
            "source": best_source
        })

    return results


# ============================================================
# FACTUALITY SUMMARY
# ============================================================

def calculate_factuality_score(results):

    if not results:
        return 0

    supported = sum(
        1
        for result in results
        if result["status"] == "Supported"
    )

    return (
        supported
        / len(results)
        * 100
    )


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a research paper PDF",
    type=["pdf"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    with st.spinner(
        "Extracting paper text..."
    ):

        paper_text = extract_text_from_pdf(
            uploaded_file
        )

    if not paper_text.strip():

        st.error(
            "No readable text was found in this PDF."
        )

        st.stop()

    word_count = len(
        paper_text.split()
    )

    chunks = chunk_text(
        paper_text,
        250
    )

    st.success(
        f"Paper loaded successfully — "
        f"{word_count:,} words detected."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Word count",
            f"{word_count:,}"
        )

    with col2:

        st.metric(
            "Estimated chunks",
            len(chunks)
        )


    st.divider()


    # ========================================================
    # LOAD MODELS
    # ========================================================

    with st.spinner(
        "Loading summarization models..."
    ):

        bart_tokenizer, bart_model = load_bart()

        t5_tokenizer, t5_model = load_t5()


    # ========================================================
    # GENERATE SUMMARIES
    # ========================================================

    if st.button(
        "🚀 Generate Summaries",
        type="primary"
    ):

        with st.spinner(
            "Generating BART summary..."
        ):

            bart_summary = summarize_document(
                paper_text,
                "BART",
                bart_tokenizer,
                bart_model
            )


        with st.spinner(
            "Generating T5 summary..."
        ):

            t5_summary = summarize_document(
                paper_text,
                "T5",
                t5_tokenizer,
                t5_model
            )


        st.session_state[
            "bart_summary"
        ] = bart_summary

        st.session_state[
            "t5_summary"
        ] = t5_summary

        st.success(
            "Both summaries generated successfully!"
        )


    # ========================================================
    # DISPLAY SUMMARIES
    # ========================================================

    if (
        "bart_summary"
        in st.session_state
    ):

        bart_summary = st.session_state[
            "bart_summary"
        ]

        t5_summary = st.session_state[
            "t5_summary"
        ]

        st.header(
            "📝 Generated Summaries"
        )

        tab1, tab2 = st.tabs(
            [
                "BART",
                "T5"
            ]
        )


        with tab1:

            st.subheader(
                "DistilBART Summary"
            )

            st.write(
                bart_summary
            )

            st.download_button(
                "Download BART Summary",
                bart_summary,
                file_name="bart_summary.txt"
            )


        with tab2:

            st.subheader(
                "T5 Summary"
            )

            st.write(
                t5_summary
            )

            st.download_button(
                "Download T5 Summary",
                t5_summary,
                file_name="t5_summary.txt"
            )


        # ====================================================
        # SUMMARY COMPARISON
        # ====================================================

        st.header(
            "📊 Summary Comparison"
        )

        bart_words = len(
            bart_summary.split()
        )

        t5_words = len(
            t5_summary.split()
        )

        comparison = pd.DataFrame({
            "Model": [
                "BART",
                "T5"
            ],
            "Summary Words": [
                bart_words,
                t5_words
            ],
            "Compression Ratio": [
                round(
                    bart_words / word_count,
                    3
                ),
                round(
                    t5_words / word_count,
                    3
                )
            ]
        })

        st.dataframe(
            comparison,
            use_container_width=True
        )


        # ====================================================
        # FACTUALITY
        # ====================================================

        st.header(
            "🔍 Automatic Factuality Analysis"
        )

        st.info(
            "The application checks whether summary "
            "statements are supported by the uploaded paper. "
            "Results are automatically generated — no CSV "
            "editing is required."
        )


        with st.spinner(
            "Loading factuality model..."
        ):

            nli_tokenizer, nli_model = load_nli()


        # ----------------------------------------------------
        # BART FACTUALITY
        # ----------------------------------------------------

        with st.spinner(
            "Checking BART factuality..."
        ):

            bart_factuality = check_factuality(
                bart_summary,
                paper_text,
                nli_tokenizer,
                nli_model
            )


        # ----------------------------------------------------
        # T5 FACTUALITY
        # ----------------------------------------------------

        with st.spinner(
            "Checking T5 factuality..."
        ):

            t5_factuality = check_factuality(
                t5_summary,
                paper_text,
                nli_tokenizer,
                nli_model
            )


        bart_score = calculate_factuality_score(
            bart_factuality
        )

        t5_score = calculate_factuality_score(
            t5_factuality
        )


        # ====================================================
        # FACTUALITY SCORE DISPLAY
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "BART Factuality",
                f"{bart_score:.1f}%"
            )

        with col2:

            st.metric(
                "T5 Factuality",
                f"{t5_score:.1f}%"
            )


        # ====================================================
        # DETAILED CLAIM ANALYSIS
        # ====================================================

        st.subheader(
            "BART Claim Analysis"
        )

        bart_df = pd.DataFrame(
            bart_factuality
        )

        if not bart_df.empty:

            st.dataframe(
                bart_df[
                    [
                        "claim",
                        "status",
                        "confidence",
                        "source"
                    ]
                ],
                use_container_width=True
            )


        st.subheader(
            "T5 Claim Analysis"
        )

        t5_df = pd.DataFrame(
            t5_factuality
        )

        if not t5_df.empty:

            st.dataframe(
                t5_df[
                    [
                        "claim",
                        "status",
                        "confidence",
                        "source"
                    ]
                ],
                use_container_width=True
            )


        # ====================================================
        # FINAL MODEL COMPARISON
        # ====================================================

        st.header(
            "🏆 Model Comparison"
        )

        if bart_score > t5_score:

            factuality_winner = "BART"

        elif t5_score > bart_score:

            factuality_winner = "T5"

        else:

            factuality_winner = "Tie"


        st.write(
            f"### Factuality winner: **{factuality_winner}**"
        )

        st.caption(
            "Factuality scores are based on automated "
            "NLI/entailment analysis and should be interpreted "
            "as an automated estimate rather than a perfect "
            "human factuality judgment."
        )