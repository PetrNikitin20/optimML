import json
import math
import statistics
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "factorial_pilot"
RESULTS = sorted(RESULT_DIR.glob("[0-9][0-9][0-9]_*.json"))
SUMMARY = RESULT_DIR / "pairwise_3B_ultrafeedback_noise0.0_summary.json"


class FactorialPilotResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = [json.loads(path.read_text(encoding="utf-8")) for path in RESULTS]
        cls.summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    def test_three_seed_series_identity(self):
        self.assertEqual(len(self.results), 3)
        self.assertEqual({item["run_index"] for item in self.results}, {0, 1, 2})
        self.assertEqual({item["config"]["seed"] for item in self.results}, {11, 29, 47})
        self.assertEqual(len({item["data_sha256"] for item in self.results}), 3)
        for item in self.results:
            with self.subTest(seed=item["config"]["seed"]):
                self.assertEqual(item["execution_profile"], "pilot")
                self.assertEqual(item["train"]["execution_profile"], "pilot")
                self.assertEqual(item["config"]["model_id"], "Qwen/Qwen2.5-3B-Instruct")
                self.assertEqual(item["config"]["dataset"], "ultrafeedback")
                self.assertEqual(item["config"]["loss"], "pairwise")
                self.assertEqual(item["config"]["noise"], 0.0)

    def test_pilot_sample_and_step_accounting(self):
        for item in self.results:
            with self.subTest(seed=item["config"]["seed"]):
                self.assertEqual(item["config"]["train_pairs"], 256)
                self.assertEqual(item["evaluation"]["n_eval_pairs"], 64)
                self.assertEqual(item["generation"]["n_generation_prompts"], 8)
                self.assertEqual(item["train"]["optimizer_steps"], 10)
                self.assertEqual(item["train"]["examples_seen"], 40)

    def test_all_reported_numeric_metrics_are_finite(self):
        for item in self.results:
            groups = [
                item["gradient_preflight"],
                item["evaluation"],
                item["curvature"],
                item["generation"],
            ]
            for group in groups:
                for key, value in group.items():
                    if isinstance(value, (int, float)):
                        with self.subTest(seed=item["config"]["seed"], metric=key):
                            self.assertTrue(math.isfinite(value))
            self.assertTrue(math.isfinite(item["kl_to_sft"]))

    def test_cost_metrics_are_positive(self):
        for item in self.results:
            for metric in ["train_seconds", "examples_per_second", "peak_gpu_memory_gb"]:
                with self.subTest(seed=item["config"]["seed"], metric=metric):
                    self.assertGreater(item["train"][metric], 0)

    def test_summary_declares_pilot_scope(self):
        self.assertEqual(self.summary["n_seeds"], 3)
        self.assertEqual(self.summary["seeds"], [11, 29, 47])
        self.assertEqual(self.summary["ddof"], 1)
        self.assertEqual(self.summary["scope"], "pilot_not_inferential")
        self.assertEqual(self.summary["metrics"]["likelihood_ranking_accuracy"]["n"], 3)

    def test_summary_accuracy_is_recomputable(self):
        values = [item["evaluation"]["likelihood_ranking_accuracy"] for item in self.results]
        metric = self.summary["metrics"]["likelihood_ranking_accuracy"]
        self.assertAlmostEqual(metric["mean"], statistics.mean(values))
        self.assertAlmostEqual(metric["sd"], statistics.stdev(values))


if __name__ == "__main__":
    unittest.main()
