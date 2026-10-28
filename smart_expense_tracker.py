"""Smart Expense Tracker - AI-powered upgrade of the Module 02 project.

New AI features
1. Auto-categorisation: a TF-IDF + Logistic Regression text classifier predicts the
   category from the expense description (e.g. "uber to college" -> Transport).
2. Anomaly detection: flags expenses that are unusually large for their category.
3. Forecast: predicts next month's spending with a linear trend.

Run:  python smart_expense_tracker.py
"""
import json
import os
from collections import defaultdict
from datetime import date

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expenses.json")
CATEGORIES = ["Food", "Transport", "Education", "Entertainment", "Bills", "Other"]

# Small labelled seed set used to train the classifier.
TRAINING_DATA = {
    "Food": ["lunch", "dinner", "breakfast", "pizza", "biryani", "coffee", "tea", "snacks",
             "groceries", "vegetables", "canteen meal", "restaurant", "swiggy order",
             "zomato order", "ice cream", "juice", "bakery", "milk", "burger", "tiffin", "meals",
             "fruits", "chai", "pani puri", "friends lunch", "cafe", "sandwich combo", "idli", "noodles",
             "dinner with friends", "biryani with friends", "snacks with friends", "coffee with friends"],
    "Transport": ["bus ticket", "auto fare", "uber ride", "ola cab", "metro card recharge",
                  "train ticket", "petrol", "fuel", "bike service", "parking fee",
                  "cab to college", "taxi", "flight ticket", "toll", "bus fare", "bus pass renewal",
                  "uber to airport", "rapido bike taxi", "cycle repair", "metro", "auto to station", "ticket to hometown"],
    "Education": ["college fees", "exam fee", "textbook", "notebook", "stationery", "pen and files",
                  "online course", "udemy course", "coursera subscription", "printout",
                  "project report binding", "lab record", "certification exam", "semester fee",
                  "reference book", "tuition", "workshop registration", "library fine", "data science book"],
    "Entertainment": ["movie ticket", "netflix", "spotify", "game purchase", "concert",
                      "cricket match tickets", "outing with friends", "bowling", "youtube premium",
                      "amazon prime video", "party", "gaming recharge", "hotstar subscription", "movie night",
                      "theme park", "ott subscription", "pubg skins", "trip with friends"],
    "Bills": ["electricity bill", "mobile recharge", "wifi bill", "water bill", "rent",
              "gas cylinder", "dth recharge", "internet bill", "phone bill", "hostel rent", "airtel recharge", "room rent advance",
              "postpaid bill", "maintenance charges", "broadband", "pg rent", "electric bill"],
    "Other": ["gift", "medicine", "clothes", "haircut", "donation", "repair", "laundry",
              "shoes", "birthday present", "pharmacy", "doctor consultation", "tablets", "tailor",
              "wedding gift", "shopping", "tshirt"],
}


def build_classifier():
    texts, labels = [], []
    for category, phrases in TRAINING_DATA.items():
        for p in phrases:
            texts.append(p)
            labels.append(category)
    features = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), lowercase=True),
    )
    model = make_pipeline(features, LogisticRegression(C=20, max_iter=1000))
    model.fit(texts, labels)
    return model


def predict_category(model, description, min_confidence=0.40, min_margin=0.10):
    """Return (category, confidence). Falls back to 'Other' if unsure."""
    text = description.strip()
    if not text:
        return "Other", 0.0
    probs = model.predict_proba([text])[0]
    order = np.argsort(probs)[::-1]
    category, conf = str(model.classes_[order[0]]), float(probs[order[0]])
    margin = conf - float(probs[order[1]])
    # Unsure (low confidence, or two categories nearly tied) -> safe fallback.
    if conf < min_confidence or margin < min_margin:
        return "Other", conf
    return category, conf


# ---------- storage / core logic ----------
def load_expenses(path=DATA_FILE):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_expenses(expenses, path=DATA_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(expenses, f, indent=2)


def add_expense(expenses, amount, category, description="", when=None):
    if amount <= 0:
        raise ValueError("Amount must be greater than zero.")
    if category not in CATEGORIES:
        category = "Other"
    expense = {
        "id": max((e["id"] for e in expenses), default=0) + 1,
        "date": when or date.today().isoformat(),
        "amount": round(float(amount), 2),
        "category": category,
        "description": description.strip(),
    }
    expenses.append(expense)
    return expense


def delete_expense(expenses, expense_id):
    for i, e in enumerate(expenses):
        if e["id"] == expense_id:
            return expenses.pop(i)
    return None


def total_by_category(expenses):
    totals = defaultdict(float)
    for e in expenses:
        totals[e["category"]] += e["amount"]
    return dict(totals)


# ---------- AI insights ----------
def find_anomalies(expenses, z_threshold=2.0, min_samples=4):
    """Flag expenses far above the mean of their category (z-score)."""
    by_cat = defaultdict(list)
    for e in expenses:
        by_cat[e["category"]].append(e)
    flagged = []
    for items in by_cat.values():
        if len(items) < min_samples:
            continue
        amounts = np.array([i["amount"] for i in items])
        std = amounts.std()
        if std == 0:
            continue
        for item in items:
            z = (item["amount"] - amounts.mean()) / std
            if z >= z_threshold:
                flagged.append((item, round(float(z), 2)))
    return flagged


def monthly_totals(expenses):
    totals = defaultdict(float)
    for e in expenses:
        totals[e["date"][:7]] += e["amount"]
    return dict(sorted(totals.items()))


def forecast_next_month(expenses):
    """Linear-trend forecast of next month's total. Needs >= 3 months of data."""
    totals = monthly_totals(expenses)
    if len(totals) < 3:
        return None
    y = np.array(list(totals.values()))
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    return max(0.0, float(slope * len(y) + intercept))


# ---------- display ----------
def print_expenses(expenses):
    if not expenses:
        print("\nNo expenses recorded yet.")
        return
    print(f"\n{'ID':<4}{'Date':<12}{'Category':<15}{'Amount':>10}  Description")
    print("-" * 60)
    for e in expenses:
        print(f"{e['id']:<4}{e['date']:<12}{e['category']:<15}{e['amount']:>10.2f}  {e['description']}")
    print("-" * 60)
    print(f"{'Total':<31}{sum(e['amount'] for e in expenses):>10.2f}")


def print_insights(expenses):
    if not expenses:
        print("\nNo data for insights yet.")
        return
    totals = total_by_category(expenses)
    grand = sum(totals.values())
    print("\nSpending by category")
    print("-" * 40)
    for cat, amt in sorted(totals.items(), key=lambda x: -x[1]):
        print(f"{cat:<15}{amt:>10.2f}  ({amt / grand * 100:.1f}%)")

    anomalies = find_anomalies(expenses)
    print("\nUnusual expenses")
    if anomalies:
        for e, z in anomalies:
            print(f"  ! #{e['id']} {e['description']} ({e['category']}): {e['amount']:.2f}  (z={z})")
    else:
        print("  None found.")

    forecast = forecast_next_month(expenses)
    print("\nForecast")
    if forecast is None:
        print("  Need at least 3 months of data to forecast.")
    else:
        print(f"  Expected spending next month: about {forecast:.2f}")


def read_amount():
    while True:
        try:
            value = float(input("Amount: "))
            if value > 0:
                return value
            print("Amount must be greater than zero.")
        except ValueError:
            print("Please enter a valid number.")


def main():
    model = build_classifier()
    expenses = load_expenses()
    menu = ("\n=== Smart Expense Tracker (AI) ===\n1. Add expense (auto-category)\n"
            "2. View expenses\n3. Delete expense\n4. AI insights\n5. Exit")
    while True:
        print(menu)
        choice = input("Select an option: ").strip()
        if choice == "1":
            desc = input("Description (e.g. 'uber to college'): ")
            amount = read_amount()
            category, conf = predict_category(model, desc)
            print(f"AI suggests: {category} (confidence {conf:.0%})")
            override = input(f"Press Enter to accept, or type one of {CATEGORIES}: ").strip().title()
            if override in CATEGORIES:
                category = override
            add_expense(expenses, amount, category, desc)
            save_expenses(expenses)
            print("Expense added.")
        elif choice == "2":
            print_expenses(expenses)
        elif choice == "3":
            print_expenses(expenses)
            raw = input("Enter ID to delete: ").strip()
            if raw.isdigit() and delete_expense(expenses, int(raw)):
                save_expenses(expenses)
                print("Expense deleted.")
            else:
                print("ID not found.")
        elif choice == "4":
            print_insights(expenses)
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid option, try again.")


if __name__ == "__main__":
    main()
