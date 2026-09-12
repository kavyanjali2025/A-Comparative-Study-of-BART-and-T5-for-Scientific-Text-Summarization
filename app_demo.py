import streamlit as st
import fitz
from transformers import pipeline

st.title("📄 Research Paper Summarizer")

st.write(
    "Upload a research paper and generate an AI-powered summary."
)

uploaded_file = st.file_uploader(
    "Upload your research paper",
    type=["pdf"]
)

if uploaded_file is not None:

    # Open PDF
    document = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    # Extract text
    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    st.success("PDF uploaded successfully!")

    # Basic information
    words = text.split()

    st.write("Number of words:", len(words))

    # Load AI model
    with st.spinner("Loading AI model..."):

        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6"
        )

    # Generate summary
    if st.button("Generate Summary"):

        with st.spinner("Generating summary..."):

            # Temporary limit
            text_for_model = " ".join(words[:500])

            result = summarizer(
                text_for_model,
                max_length=200,
                min_length=50,
                do_sample=False
            )

            summary = result[0]["summary_text"]

        st.subheader("🤖 AI Generated Summary")

        st.write(summary)

        st.download_button(
            "Download Summary",
            summary,
            file_name="research_summary.txt"
        )