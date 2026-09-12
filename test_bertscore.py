from bert_score import score

reference = [
    "The model detects fraudulent transactions."
]

generated = [
    "The system identifies fake financial transactions."
]

P, R, F1 = score(
    generated,
    reference,
    lang="en"
)

print("Precision:", P.mean().item())
print("Recall:", R.mean().item())
print("F1:", F1.mean().item())