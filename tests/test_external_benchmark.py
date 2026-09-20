import csv
import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimml.external_stats import paired_change


class ExternalBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "results" / "external_benchmark_results.json"
        cls.report = json.loads(cls.path.read_text(encoding="utf-8"))
        with (ROOT / "results" / "external_benchmark_rows.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_stratified_rewardbench_sample(self):
        sample = self.report["rewardbench_sample"]
        counts = Counter(row["subset"] for row in sample)
        self.assertEqual(len(sample), 92)
        self.assertEqual(len(counts), 23)
        self.assertEqual(set(counts.values()), {4})
        self.assertEqual(len({row["id"] for row in sample}), 92)

    def test_row_level_outputs_cover_all_evaluations(self):
        self.assertEqual(len(self.rows), 92 * 3)
        methods = Counter((row["model"], row["phase"]) for row in self.rows)
        self.assertEqual(set(methods.values()), {92})

    def test_report_hash_and_dataset_hashes(self):
        verification = json.loads(
            (ROOT / "results" / "external_benchmark_verification.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), verification["sha256"])
        self.assertEqual(
            self.report["rewardbench_manifest"]["sha256_canonical_jsonl"],
            verification["rewardbench_sha256"],
        )
        self.assertEqual(
            self.report["hh_manifest"]["sha256_canonical_jsonl"],
            verification["hh_sha256"],
        )

    def test_paired_change_is_recomputable(self):
        before = [row | {"correct": row["correct"] == "True"} for row in self.rows if row["model"] == "tinyllama-1.1b-lora" and row["phase"] == "before"]
        after = [row | {"correct": row["correct"] == "True"} for row in self.rows if row["model"] == "tinyllama-1.1b-lora" and row["phase"] == "after"]
        recomputed = paired_change(before, after, seed=20260919)
        stored = self.report["tinyllama_rewardbench_paired_change"]
        for key in (
            "accuracy_change",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "exact_sign_test_two_sided_p",
        ):
            self.assertAlmostEqual(recomputed[key], stored[key])
        self.assertEqual(recomputed["improved_examples"], 8)
        self.assertEqual(recomputed["harmed_examples"], 5)

    def test_adapters_are_present(self):
        expected = [
            ROOT / "results" / "checkpoints" / "bert_tiny_lora" / "adapter_model.safetensors",
            ROOT / "results" / "checkpoints" / "tinyllama_1p1b_lora" / "adapter_model.safetensors",
        ]
        for path in expected:
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
