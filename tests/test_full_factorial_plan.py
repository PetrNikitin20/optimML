import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

ROOT = Path(__file__).resolve().parents[1]


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


plan = module(ROOT / "tools/plan_full_factorial.py")
preflight = module(ROOT / "colab/full_design_preflight.py")


class FullFactorialTests(unittest.TestCase):
    def test_manifest_has_matched_seeds_in_every_cell(self):
        configs = plan.manifest_from_source(ROOT / "colab/factorial_preference_colab.py")
        self.assertEqual(len(configs), 288)
        cells = {}
        for config in configs:
            key = tuple(config[k] for k in ["loss", "model_label", "dataset", "noise"])
            cells.setdefault(key, set()).add(config["seed"])
        self.assertEqual(len(cells), 96)
        self.assertTrue(all(seeds == {11, 29, 47} for seeds in cells.values()))

    def test_real_pilot_does_not_count_as_full(self):
        configs = plan.manifest_from_source(ROOT / "colab/factorial_preference_colab.py")
        _, status = plan.coverage(configs, ROOT / "results/factorial_v2")
        self.assertEqual(status["full_budget_runs"], 0)
        self.assertGreaterEqual(len(status["ignored_non_full_records"]), 1)
        self.assertFalse(status["publication_ready"])

    def test_incomplete_budget_rejected_even_if_renamed_full(self):
        configs = plan.manifest_from_source(ROOT / "colab/factorial_preference_colab.py")
        record = json.loads((ROOT / "results/factorial_v2/runs/pilot/014_pairwise_3B_reddit_tldr_noise0.0_seed47.json").read_text())
        record["execution_profile"] = "full"
        self.assertIn("mismatch:config", plan.full_training_errors(record, 14, configs[14]))
        self.assertIn("incomplete_optimizer_budget", plan.full_training_errors(record, 14, configs[14]))

    def test_missing_results_are_not_imputed(self):
        configs = plan.manifest_from_source(ROOT / "colab/factorial_preference_colab.py")
        with TemporaryDirectory() as tmp:
            cells, status = plan.coverage(configs, Path(tmp))
        self.assertEqual(status["missing_or_conflicting_runs"], 288)
        self.assertTrue(all(x["coverage_status"] == "missing" for x in cells))

    def test_finite_difference_or_projected_hessian_is_not_full_validation(self):
        errors = plan.publication_errors({"curvature": {"hessian_validated": True,
                                          "hessian_scope": "last_MLP_projection_only"}})
        self.assertIn("full_scope_Hessian_not_validated", errors)

    def test_cardinality_uses_unique_prompts_not_comparisons(self):
        rows = [{"prompt": "Hello", "chosen": "yes", "rejected": "no"}] * 100
        result = preflight.cardinality(rows, 3, 2)
        self.assertEqual(result["unique_exact_prompts"], 1)
        self.assertFalse(result["exact_cardinality_pass"])

    def test_basic_normalization_detects_superficial_duplicates(self):
        rows = [{"prompt": x, "chosen": "yes", "rejected": "no"} for x in [" HELLO  world", "hello world"]]
        result = preflight.cardinality(rows, 1, 1)
        self.assertTrue(result["exact_cardinality_pass"])
        self.assertFalse(result["normalized_cardinality_pass"])

    def test_corrected_split_is_normalized_disjoint_and_order_independent(self):
        rows = [{"prompt": f" Question {i} ", "chosen": "yes", "rejected": "no"} for i in range(20)]
        rows += [{"prompt": f"question {i}", "chosen": "other", "rejected": "bad"} for i in range(20)]
        train, holdout = preflight.prepare_confirmatory_split(rows, 8, 6)
        self.assertEqual((train, holdout), preflight.prepare_confirmatory_split(list(reversed(rows)), 8, 6))
        self.assertFalse({preflight.normalized_prompt(x["prompt"]) for x in train}
                         & {preflight.normalized_prompt(x["prompt"]) for x in holdout})
        self.assertEqual(len(train), 8)
        self.assertEqual(len(holdout), 6)

    def test_corrected_split_filters_whitespace_and_equal_responses(self):
        rows = [{"prompt": "valid", "chosen": "yes", "rejected": "no"},
                {"prompt": "invalid", "chosen": " ", "rejected": "no"},
                {"prompt": "same", "chosen": "yes", "rejected": "yes"}]
        with self.assertRaises(ValueError):
            preflight.prepare_confirmatory_split(rows, 1, 1)

    def test_real_corrected_preflight_evidence_is_hashed_and_not_training(self):
        import hashlib
        directory = ROOT / "results/factorial_full_plan_20261001"
        for name, digest in [("original", "fe9a669d33031d17ed55779b5c8310a151f30ba577449865a200544a3409969a"),
                             ("corrected", "1fa1b91a700cd13ae9284386764144ddfc031f222d522ad8fee3bb026ecbfe00")]:
            payload = (directory / ("preflight_" + name + ".json")).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
        report = json.loads((directory / "preflight_corrected.json").read_text())
        self.assertFalse(report["full_design_ready"])
        self.assertEqual(len(report["datasets"]), 4)
        for data in report["datasets"].values():
            split = data["proposed_v3_split"]
            self.assertEqual((split["train_pairs"], split["eval_pairs"]), (4000, 800))
            self.assertEqual(split["normalized_train_eval_prompt_overlap"], 0)
            self.assertFalse(split["semantic_near_duplicate_audit_complete"])


if __name__ == "__main__":
    unittest.main()
