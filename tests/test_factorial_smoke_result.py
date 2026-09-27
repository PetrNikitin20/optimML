import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "factorial_smoke"
    / "000_pairwise_3B_ultrafeedback_noise0.0_seed11.json"
)


class FactorialSmokeResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_result_is_explicitly_non_inferential_smoke_profile(self):
        self.assertEqual(self.result["execution_profile"], "smoke")
        self.assertEqual(self.result["train"]["execution_profile"], "smoke")
        self.assertEqual(self.result["run_index"], 0)
        self.assertEqual(self.result["config"]["max_steps"], 2)

    def test_real_dataset_and_model_identity_are_fixed(self):
        self.assertEqual(self.result["config"]["dataset"], "ultrafeedback")
        self.assertEqual(self.result["config"]["model_id"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(
            self.result["data_sha256"],
            "fe45487eceec7507df796dcff7ace11fe944c34473ab0afffeb98b5193848196",
        )

    def test_requested_metrics_are_finite(self):
        metric_paths = [
            ("evaluation", "likelihood_ranking_accuracy"),
            ("evaluation", "brier"),
            ("evaluation", "ece_15"),
            ("evaluation", "c_mean"),
            ("evaluation", "d_mean"),
            ("curvature", "empirical_fisher_top_eigenvalue"),
            ("curvature", "hessian_top_eigenvalue_power"),
            ("generation", "generation_win_rate_vs_sft"),
            ("train", "train_seconds"),
            ("train", "peak_gpu_memory_gb"),
        ]
        for group, metric in metric_paths:
            with self.subTest(metric=metric):
                self.assertTrue(math.isfinite(self.result[group][metric]))
        self.assertTrue(math.isfinite(self.result["kl_to_sft"]))

    def test_measured_sample_counts_and_compute_accounting(self):
        self.assertEqual(self.result["evaluation"]["n_eval_pairs"], 16)
        self.assertEqual(self.result["generation"]["n_generation_prompts"], 2)
        self.assertEqual(self.result["train"]["optimizer_steps"], 2)
        self.assertEqual(self.result["train"]["examples_seen"], 4)
        self.assertGreater(self.result["train"]["examples_per_second"], 0)


if __name__ == "__main__":
    unittest.main()
