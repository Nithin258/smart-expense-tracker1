"""Measure the categoriser on descriptions it was NOT trained on."""
from smart_expense_tracker import build_classifier, predict_category

HELD_OUT = [
    ("sandwich at cafe", "Food"), ("lunch with friends", "Food"), ("dosa", "Food"),
    ("bus pass", "Transport"), ("uber to railway station", "Transport"), ("petrol for bike", "Transport"),
    ("semester exam fee", "Education"), ("python book", "Education"), ("udemy data science course", "Education"),
    ("netflix subscription", "Entertainment"), ("movie with friends", "Entertainment"),
    ("electricity bill", "Bills"), ("jio recharge", "Bills"), ("room rent", "Bills"),
    ("medicine from pharmacy", "Other"), ("gift for friend", "Other"),
]

if __name__ == "__main__":
    model = build_classifier()
    correct = 0
    for text, truth in HELD_OUT:
        pred, conf = predict_category(model, text)
        ok = pred == truth
        correct += ok
        print(f"{'OK ' if ok else 'ERR'} {text:<30} -> {pred:<14} ({conf:.0%})  expected {truth}")
    print(f"\nAccuracy on {len(HELD_OUT)} held-out examples: {correct / len(HELD_OUT):.0%}")
