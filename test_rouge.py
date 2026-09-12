from rouge_score import rouge_scorer

reference = "Machine learning helps computers learn from data."

generated = "Machine learning allows computers to learn from data."

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

scores = scorer.score(reference, generated)

print("ROUGE-1:", scores["rouge1"].fmeasure)
print("ROUGE-2:", scores["rouge2"].fmeasure)
print("ROUGE-L:", scores["rougeL"].fmeasure)