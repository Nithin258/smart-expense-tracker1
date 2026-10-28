import unittest

from smart_expense_tracker import (
    add_expense, build_classifier, delete_expense, find_anomalies,
    forecast_next_month, monthly_totals, predict_category, total_by_category,
)


class CoreTests(unittest.TestCase):
    def test_add_delete_totals(self):
        data = []
        add_expense(data, 100, "Food", "lunch")
        add_expense(data, 50.567, "Bogus", "x")
        self.assertEqual(data[1]["category"], "Other")
        self.assertEqual(data[1]["amount"], 50.57)
        self.assertEqual(total_by_category(data)["Food"], 100)
        self.assertIsNotNone(delete_expense(data, 1))
        self.assertIsNone(delete_expense(data, 99))

    def test_invalid_amount(self):
        with self.assertRaises(ValueError):
            add_expense([], -5, "Food")


class AITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_classifier()

    def test_clear_cases(self):
        cases = {"uber to railway station": "Transport", "electricity bill": "Bills",
                 "netflix subscription": "Entertainment", "python book": "Education"}
        for text, expected in cases.items():
            self.assertEqual(predict_category(self.model, text)[0], expected, text)

    def test_empty_description_is_other(self):
        self.assertEqual(predict_category(self.model, "   ")[0], "Other")

    def test_gibberish_falls_back_to_other(self):
        self.assertEqual(predict_category(self.model, "zzqx")[0], "Other")

    def test_anomaly_detection(self):
        data = []
        for amt in (100, 110, 95, 105, 102, 98, 900):
            add_expense(data, amt, "Food", "meal", when="2026-01-10")
        flagged = find_anomalies(data)
        self.assertEqual([e["amount"] for e, _ in flagged], [900.0])

    def test_no_anomaly_with_few_samples(self):
        data = []
        add_expense(data, 10, "Food")
        add_expense(data, 999, "Food")
        self.assertEqual(find_anomalies(data), [])

    def test_forecast_follows_linear_trend(self):
        data = []
        for month, amt in (("2026-01", 1000), ("2026-02", 1100), ("2026-03", 1200)):
            add_expense(data, amt, "Bills", "x", when=f"{month}-05")
        self.assertAlmostEqual(forecast_next_month(data), 1300, places=3)
        self.assertEqual(len(monthly_totals(data)), 3)

    def test_forecast_needs_three_months(self):
        data = []
        add_expense(data, 100, "Food", when="2026-01-01")
        self.assertIsNone(forecast_next_month(data))


if __name__ == "__main__":
    unittest.main()
