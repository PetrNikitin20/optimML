import csv
import hashlib
import json
import math
import statistics
import unittest
from pathlib import Path
from tools.verify_protocol_v2 import recompute_rows, ece

DIRECTORY = Path(__file__).resolve().parents[1] / "results/factorial_v2/runs/pilot"
KEY = "014_pairwise_3B_reddit_tldr_noise0.0_seed47"


class ProtocolV2ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((DIRECTORY / f"{KEY}.json").read_text(encoding="utf-8"))
        with (DIRECTORY / f"{KEY}_rows.csv").open(encoding="utf-8", newline="") as stream:
            cls.rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]

    def test_real_pilot_identity_and_cost(self):
        self.assertEqual(self.result["protocol_version"], "v2")
        self.assertEqual(self.result["status"], "complete")
        self.assertEqual(self.result["config"]["seed"], 47)
        self.assertEqual(self.result["config"]["dataset"], "reddit_tldr")
        self.assertEqual(self.result["source_revision"], "1037290")
        self.assertEqual(self.result["train"]["optimizer_steps"], 10)
        self.assertEqual(self.result["train"]["examples_seen"], 40)
        self.assertGreater(self.result["train"]["train_seconds"], 0)
        self.assertEqual(self.result["train"]["total_parameters"], 3115872256)
        self.assertEqual(len(self.rows), 64)

    def test_public_artifact_hashes(self):
        report = json.loads((DIRECTORY / f"{KEY}_verification.json").read_text(encoding="utf-8"))
        self.assertTrue(report["raw_data_hash_verified"])
        self.assertTrue(report["train_holdout_prompt_disjoint_verified"])
        self.assertEqual(report["scope"], "single_seed_pilot_not_inferential")
        for name, expected in report["public_artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest(), expected)

    def test_metrics_recomputed_from_all_rows(self):
        measured = recompute_rows(self.rows, self.result["config"]["beta"])
        for name, computed in measured.items():
            self.assertAlmostEqual(self.result["evaluation"][name], computed, places=10)
        confidence = [max(r["presentation_probability"], 1 - r["presentation_probability"]) for r in self.rows]
        correct = [float(float(r["presentation_probability"] >= 0.5) == r["presentation_target"]) for r in self.rows]
        diagnostic = json.loads((DIRECTORY / f"{KEY}_calibration_diagnostics.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(diagnostic["confidence_ece_15"], ece(confidence, correct), places=10)

    def test_judge_scores_and_lengths_are_recomputable(self):
        rows = json.loads((DIRECTORY / f"{KEY}_judge_scores.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 8)
        self.assertTrue(all("prompt" not in row and "response" not in row for row in rows))
        self.assertEqual(statistics.mean(float(r["policy_reward"] > r["reference_reward"]) for r in rows), self.result["generation"]["generation_win_rate_vs_sft"])
        self.assertAlmostEqual(statistics.mean(r["policy_reward"] - r["reference_reward"] for r in rows), self.result["generation"]["mean_reward_delta"], places=10)
        self.assertEqual(statistics.mean(r["policy_tokens"] for r in rows), self.result["generation"]["policy_response_length_tokens"])
        self.assertEqual(statistics.mean(r["reference_tokens"] for r in rows), self.result["generation"]["sft_response_length_tokens"])

    def test_finite_difference_hessian_is_explicitly_unstable(self):
        obj = json.loads((DIRECTORY / f"{KEY}_hvp_sensitivity.json").read_text(encoding="utf-8"))
        self.assertFalse(obj["epsilon_consistency_pass"])
        self.assertTrue(obj["weights_exactly_restored"])
        self.assertEqual(obj["repeat_gradient_relative_difference"], 0)
        self.assertEqual(len(obj["probes"]), 3)

    def test_exact_hessian_is_restricted_and_numerically_checked(self):
        obj = json.loads((DIRECTORY / f"{KEY}_projected_hessian_exact.json").read_text(encoding="utf-8"))
        self.assertIn("not_full_model_Hessian", obj["scope"])
        self.assertEqual(obj["n_parameters"], 32768)
        self.assertEqual(obj["n_eval_pairs_used"], 1)
        self.assertFalse(obj["weights_modified"])
        self.assertEqual(obj["repeat_hvp_relative_difference"], 0)
        self.assertLess(obj["symmetry_relative_error"], 1e-3)
        self.assertLess(obj["relative_residual"], 0.01)
        self.assertTrue(math.isfinite(obj["rayleigh"]))
        self.assertEqual(len(obj["history"]), 8)


if __name__ == "__main__":
    unittest.main()
