from transformers import pipeline

# Load T5 summarization model
summarizer = pipeline(
    "summarization",
    model="t5-small"
)

text = """
Artificial intelligence is becoming increasingly important in modern
technology. Machine learning allows computers to learn patterns from
data and make predictions. Deep learning uses neural networks with
multiple layers to solve complex problems. These technologies are
being used in healthcare, finance, transportation and scientific
research.
"""

result = summarizer(
    "summarize: " + text,
    max_length=80,
    min_length=20,
    do_sample=False
)

print("\nT5 Summary:")
print(result[0]["summary_text"])