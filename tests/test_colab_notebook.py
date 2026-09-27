import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab" / "factorial_preference_study.ipynb"


class ColabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_notebook_structure_and_gpu_metadata(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        self.assertGreaterEqual(len(self.notebook["cells"]), 10)

    def test_complete_factorial_grid(self):
        for token in [
            'LOSSES = ["pairwise", "pointwise"]',
            'DATASETS = ["ultrafeedback", "reddit_tldr", "helpsteer2", "hh_rlhf"]',
            'NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3]',
            'SEEDS = [11, 29, 47]',
            "assert len(MANIFEST) == 288",
        ]:
            self.assertIn(token, self.source)

    def test_requested_metrics_are_computed(self):
        for metric in [
            "likelihood_ranking_accuracy",
            "generation_win_rate_vs_sft",
            "brier",
            "ece_15",
            "c_abs_mean",
            "d_abs_mean",
            "kl_to_sft",
            "empirical_fisher_top_eigenvalue",
            "hessian_top_eigenvalue_power",
            "policy_response_length_tokens",
            "peak_gpu_memory_gb",
        ]:
            self.assertIn(metric, self.source)

    def test_rational_execution_profiles_and_resume(self):
        for token in [
            'EXECUTION_PROFILE = "pilot"',
            'if profile == "smoke"',
            'if profile == "pilot"',
            'if profile == "full"',
            "validate_trainable_gradients",
            "No LoRA gradients were produced",
            "save_training_checkpoint",
            "restore_training_checkpoint",
            "eta_min=",
            'OUTPUT_ROOT / "runs" / "full"',
        ]:
            self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
