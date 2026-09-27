import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "factorial_pilot"
    / "000_pairwise_3B_ultrafeedback_noise0.0_seed11.json"
)


class FactorialPilotResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_pilot_identity_and_data_hash(self):
        self.assertEqual(self.result["execution_profile"], "pilot")
        self.assertEqual(self.result["train"]["execution_profile"], "pilot")
        self.assertEqual(self.result["config"]["model_id"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(self.result["config"]["dataset"], "ultrafeedback")
        self.assertEqual(
            self.result["data_sha256"],
            "e839ebee337d556915513bd098665507cf4cac094c3940777f3598de1c68902a",
        )

    def test_pilot_sample_and_step_accounting(self):
        self.assertEqual(self.result["config"]["train_pairs"], 256)
        self.assertEqual(self.result["evaluation"]["n_eval_pairs"], 64)
        self.assertEqual(self.result["generation"]["n_generation_prompts"], 8)
        self.assertEqual(self.result["train"]["optimizer_steps"], 10)
        self.assertEqual(self.result["train"]["examples_seen"], 40)

    def test_all_reported_numeric_metrics_are_finite(self):
        groups = [
            self.result["gradient_preflight"],
            self.result["evaluation"],
            self.result["curvature"],
            self.result["generation"],
        ]
        for group in groups:
            for key, value in group.items():
                if isinstance(value, (int, float)):
                    with self.subTest(metric=key):
                        self.assertTrue(math.isfinite(value))
        self.assertTrue(math.isfinite(self.result["kl_to_sft"]))

    def test_cost_metrics_are_positive(self):
        for metric in ["train_seconds", "examples_per_second", "peak_gpu_memory_gb"]:
            with self.subTest(metric=metric):
                self.assertGreater(self.result["train"][metric], 0)


if __name__ == "__main__":
    unittest.main()
