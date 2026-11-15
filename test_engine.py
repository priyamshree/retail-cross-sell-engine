import csv
import json
import tempfile
import unittest
from pathlib import Path

from app import app
from engine import CatalogueEngine
from pipeline import build

ROOT = Path(__file__).resolve().parent

CATALOG = [
    {"code": "AA", "name": "Alpha mug"},
    {"code": "BB", "name": "Beta jug"},
    {"code": "CC", "name": "Gamma bowl"},
    {"code": "DD", "name": "Delta tin"},
    {"code": "EE", "name": "Echo lamp"},
    {"code": "POST", "name": "POSTAGE"},
    {"code": "REG", "name": "REGENCY TEACUP AND SAUCER"},
    {"code": "CAKE", "name": "REGENCY CAKESTAND 3 TIER"},
    {"code": "OWL", "name": "JUMBO BAG OWLS"},
]
RULES = [
    {"antecedents": ["AA"], "consequents": ["BB"], "support": 0.015, "confidence": 0.30, "lift": 40},
    {"antecedents": ["AA"], "consequents": ["CC"], "support": 0.04, "confidence": 0.85, "lift": 6},
    {"antecedents": ["AA"], "consequents": ["DD"], "support": 0.03, "confidence": 0.55, "lift": 8},
    {"antecedents": ["AA", "CC"], "consequents": ["DD"], "support": 0.02, "confidence": 0.70, "lift": 10},
    {"antecedents": ["AA", "BB"], "consequents": ["CC"], "support": 0.02, "confidence": 0.90, "lift": 30},
    {"antecedents": ["AA", "BB"], "consequents": ["EE"], "support": 0.02, "confidence": 0.80, "lift": 12},
    {"antecedents": ["AA"], "consequents": ["POST"], "support": 0.05, "confidence": 0.95, "lift": 15},
    {"antecedents": ["REG"], "consequents": ["CAKE"], "support": 0.02, "confidence": 0.6, "lift": 5},
]


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = CatalogueEngine(CATALOG, RULES)

    def test_direct_rules_exclude_unsatisfied_conditions(self):
        product = self.engine.product("AA")
        direct = [item["code"] for item in product["direct"]]
        self.assertEqual(direct, ["CC", "DD", "BB"])
        self.assertNotIn("EE", direct)
        self.assertNotIn("POST", direct)
        self.assertTrue(all(item["applicable"] for item in product["direct"]))
        self.assertIn("recommended because", product["direct"][0]["explanation"])
        self.assertIn("confidence", product["direct"][0]["explanation"])
        tight = self.engine.product("AA", floors={"min_support": 0, "min_confidence": 0.8, "min_lift": 0})
        self.assertEqual([item["code"] for item in tight["direct"]], ["CC"])
        conditional_targets = {
            item["code"]
            for row in product["conditional"]
            for item in row["recommend"]
        }
        self.assertIn("EE", conditional_targets)
        for row in product["conditional"]:
            self.assertFalse(row["applicable"])
            self.assertTrue(row["also_requires"])

    def test_basket_ranks_aggregated_evidence_above_raw_lift(self):
        result = self.engine.recommend(["AA", "CC"])
        codes = [item["code"] for item in result["recommendations"]]
        self.assertEqual(codes[0], "DD")
        delta = result["recommendations"][0]
        self.assertEqual(delta["rule_count"], 2)
        self.assertGreater(delta["score"], result["recommendations"][1]["score"])
        self.assertNotIn("CC", codes)
        self.assertTrue(all(item["applicable"] for item in result["recommendations"]))

    def test_partial_basket_does_not_fire_multi_item_rule(self):
        result = self.engine.recommend(["AA"])
        codes = [item["code"] for item in result["recommendations"]]
        self.assertNotIn("EE", codes)
        missing = {row["missing"]["code"] for row in result["almost"]}
        self.assertIn("BB", missing)
        echo = next(row for row in result["almost"] if any(item["code"] == "EE" for item in row["recommend"]))
        self.assertFalse(echo["applicable"])

    def test_unknown_code_and_uncovered_product(self):
        result = self.engine.recommend(["AA", "NOPE"])
        self.assertEqual(result["unknown"], ["NOPE"])
        self.assertIsNone(self.engine.product("NOPE"))
        owl = self.engine.product("OWL")
        self.assertFalse(owl["covered"])
        self.assertFalse(owl["has_direct"])
        self.assertEqual(owl["direct"], [])

    def test_tfidf_prefers_description_overlap(self):
        found = self.engine.similar("regency teacup")
        self.assertEqual(found["matches"][0]["code"], "REG")
        self.assertGreater(found["matches"][0]["similarity"], found["matches"][-1]["similarity"])
        self.assertTrue(any(item["code"] == "CAKE" for item in found["suggestions"]))

    def test_real_rules_do_not_treat_bundle_as_single_product(self):
        engine = CatalogueEngine(
            json.loads((ROOT / "catalog.json").read_text(encoding="utf-8")),
            json.loads((ROOT / "rules.json").read_text(encoding="utf-8")),
        )
        product = engine.product("22697")
        direct_codes = {item["code"] for item in product["direct"]}
        # 22697 + 22423 -> 22698 is a real multi-item rule and must not be direct.
        bundled = [
            row for row in product["conditional"]
            if any(item["code"] == "22423" for item in row["also_requires"])
        ]
        self.assertTrue(bundled)
        self.assertTrue(all(not row["applicable"] for row in product["conditional"]))
        for item in product["direct"]:
            self.assertEqual(item["because"], [product["name"]])
        self.assertIn("22698", direct_codes)
        stats = engine.stats()
        self.assertEqual(stats["catalog_size"], 400)
        self.assertEqual(stats["rule_count"], 500)
        self.assertEqual(stats["products_in_rules"], 122)


class PipelineTests(unittest.TestCase):
    def test_mines_expected_pair_and_drops_postage_and_cancellations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tx.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice"],
                )
                writer.writeheader()
                lines = []
                for invoice in range(1, 7):
                    lines.append((str(invoice), "MILK", "FRESH MILK", 2, 1.2))
                    lines.append((str(invoice), "BREAD", "SLICED BREAD", 1, 0.8))
                lines.append(("7", "MILK", "FRESH MILK", 1, 1.2))
                lines.append(("8", "MILK", "FRESH MILK", 1, 1.2))
                lines.append(("9", "EGGS", "FARM EGGS", 1, 1.5))
                lines.append(("10", "EGGS", "FARM EGGS", 1, 1.5))
                lines.append(("11", "POST", "POSTAGE", 1, 5))
                lines.append(("C12", "MILK", "FRESH MILK", -1, 1.2))
                lines.append(("13", "MILK", "FRESH MILK", 0, 1.2))
                for invoice, stock, name, qty, price in lines:
                    writer.writerow({
                        "InvoiceNo": invoice,
                        "StockCode": stock,
                        "Description": name,
                        "Quantity": qty,
                        "UnitPrice": price,
                    })
            result = build(path, min_support=0.4, min_confidence=0.5, min_lift=1.0, top_n=20, max_size=3)
        pairs = {
            (tuple(rule["antecedents"]), tuple(rule["consequents"]))
            for rule in result["rules"]
        }
        self.assertIn((("BREAD",), ("MILK",)), pairs)
        self.assertIn((("MILK",), ("BREAD",)), pairs)
        self.assertTrue(all("POST" not in rule["antecedents"] + rule["consequents"] for rule in result["rules"]))
        self.assertEqual(result["invoices"], 10)
        names = {item["code"]: item["name"] for item in result["catalog"]}
        self.assertEqual(names["MILK"], "FRESH MILK")

    def test_mines_a_three_item_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tx.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice"],
                )
                writer.writeheader()
                for invoice in range(1, 7):
                    for stock, name in (("CUP", "TEACUP"), ("SAUCER", "SAUCER"), ("PLATE", "PLATE")):
                        writer.writerow({
                            "InvoiceNo": str(invoice),
                            "StockCode": stock,
                            "Description": name,
                            "Quantity": 1,
                            "UnitPrice": 1,
                        })
                for invoice in range(7, 11):
                    writer.writerow({
                        "InvoiceNo": str(invoice),
                        "StockCode": "CUP",
                        "Description": "TEACUP",
                        "Quantity": 1,
                        "UnitPrice": 1,
                    })
            result = build(path, min_support=0.5, min_confidence=0.6, min_lift=1.0, top_n=50, max_size=3)
        pairs = {
            (tuple(rule["antecedents"]), tuple(rule["consequents"]))
            for rule in result["rules"]
        }
        self.assertIn((("CUP", "SAUCER"), ("PLATE",)), pairs)


class AnalyticsTests(unittest.TestCase):
    def test_analytics_does_not_invent_transactions_and_scenarios_get_stricter(self):
        engine = CatalogueEngine(
            json.loads((ROOT / "catalog.json").read_text(encoding="utf-8")),
            json.loads((ROOT / "rules.json").read_text(encoding="utf-8")),
        )
        report = engine.analytics()
        self.assertNotIn("total_transactions", report["kpis"])
        self.assertIn("not in this file", report["source"])
        self.assertEqual(report["kpis"]["rule_count"], 500)
        self.assertEqual(sum(bucket["count"] for bucket in report["lift_histogram"]), 500)
        self.assertTrue(report["network"]["nodes"])
        self.assertTrue(report["scatter"])
        compared = engine.what_if(["22697"], limit=3)
        kept = [scenario["rules_kept"] for scenario in compared["scenarios"]]
        self.assertEqual(
            [scenario["name"] for scenario in compared["scenarios"]],
            ["Conservative", "Balanced", "Exploratory"],
        )
        self.assertLessEqual(kept[0], kept[1])
        self.assertLessEqual(kept[1], kept[2])
        self.assertTrue(compared["scenarios"][2]["recommendations"])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_empty_basket_and_bad_threshold_are_rejected(self):
        empty = self.client.post("/api/recommend", json={"codes": []})
        self.assertEqual(empty.status_code, 400)
        bad = self.client.post("/api/recommend", json={"codes": ["22697"], "min_confidence": 2})
        self.assertEqual(bad.status_code, 400)
        missing = self.client.post("/api/recommend", data=b"not-json", content_type="application/json")
        self.assertEqual(missing.status_code, 400)

    def test_recommend_explanation_and_analytics_route(self):
        response = self.client.post("/api/recommend", json={"codes": ["22697"], "limit": 2})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(len(body["recommendations"]), 2)
        self.assertIn("lift", body["recommendations"][0]["explanation"])
        analytics = self.client.get("/api/analytics")
        self.assertEqual(analytics.status_code, 200)
        self.assertEqual(analytics.get_json()["kpis"]["catalog_size"], 400)


if __name__ == "__main__":
    unittest.main()
