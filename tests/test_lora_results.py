import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class LoraResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / "results" / "lora_results.json").read_text(encoding="utf-8"))

    def test_protocol_and_data_identity(self):
        protocol = self.report["protocol"]
        self.assertEqual(protocol["pairs"], 2000)
        self.assertEqual(protocol["train"] + protocol["validation"] + protocol["test"], 2000)
        linear = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(
            self.report["data_manifest"]["sha256_canonical_jsonl"],
            linear["data_manifest"]["sha256_canonical_jsonl"],
        )

    def test_run_summary_is_recomputable(self):
        self.assertEqual(len(self.report["runs"]), 6)
        for summary in self.report["summary"]:
            subset = [row for row in self.report["runs"] if row["method"] == summary["method"]]
            self.assertEqual(len(subset), 3)
            self.assertAlmostEqual(summary["accuracy_mean"], float(np.mean([row["test_accuracy"] for row in subset])))
            self.assertAlmostEqual(summary["pair_loss_mean"], float(np.mean([row["test_pair_loss"] for row in subset])))

    def test_gradient_geometry_invariants(self):
        rows = self.report["geometry"]
        reference = next(row for row in rows if row["beta"] == 1.0)
        self.assertAlmostEqual(reference["angle_to_beta_1_deg"], 0.0, places=10)
        for row in rows:
            self.assertGreater(row["normalized_ess"], 0.0)
            self.assertLessEqual(row["normalized_ess"], 1.0)
            self.assertGreaterEqual(row["saturated_weight_share"], 0.0)
            self.assertLessEqual(row["saturated_weight_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
