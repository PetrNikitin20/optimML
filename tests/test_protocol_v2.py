import ast
import json
import random
import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "colab/factorial_preference_colab.py").read_text(encoding="utf-8")


def isolated_functions():
    tree = ast.parse("\n".join(line for line in SOURCE.splitlines() if not line.startswith("!")))
    names = {"messages_to_text", "prepare_pairs"}
    module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    scope = {"Any": Any, "StudyConfig": SimpleNamespace, "random": random, "hashlib": hashlib, "json": json}
    exec(compile(module, "notebook_functions", "exec"), scope)
    return scope


class ProtocolV2Tests(unittest.TestCase):
    def test_summary_does_not_leak_policy_metadata(self):
        parse = isolated_functions()["messages_to_text"]
        self.assertEqual(parse({"text": "actual answer", "policy": "secret label", "note": "annotation"}), "actual answer")
        self.assertEqual(parse({"policy": "not an answer"}), "")
        self.assertEqual(parse(None), "")
        self.assertEqual(parse([{"role": "assistant", "content": "answer"}]), "answer")

    def test_fixed_disjoint_split_across_seeds(self):
        scope = isolated_functions()
        rows = [{"prompt": f"post{i}", "chosen": "good", "rejected": "bad"} for i in range(20)]
        rows += [dict(row, rejected="other") for row in rows]
        scope["LOADERS"] = {"reddit_tldr": lambda: list(rows)}
        results = [scope["prepare_pairs"](SimpleNamespace(dataset="reddit_tldr", seed=s, noise=0.0, train_pairs=8, eval_pairs=6)) for s in [11, 29, 47]]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])
        train, test, _ = results[0]
        self.assertFalse({x["prompt"] for x in train} & {x["prompt"] for x in test})
        self.assertEqual(len({x["prompt"] for x in train + test}), 14)

    def test_noise_only_changes_training_labels(self):
        scope = isolated_functions()
        scope["LOADERS"] = {"test": lambda: [{"prompt": str(i), "chosen": "good", "rejected": "bad"} for i in range(20)]}
        base = dict(dataset="test", seed=11, train_pairs=8, eval_pairs=6)
        clean, holdout, _ = scope["prepare_pairs"](SimpleNamespace(**base, noise=0.0))
        noisy, noisy_holdout, _ = scope["prepare_pairs"](SimpleNamespace(**base, noise=1.0))
        self.assertEqual(holdout, noisy_holdout)
        for left, right in zip(clean, noisy):
            self.assertEqual(left["chosen"], right["rejected"])

    def test_persistence_and_deterministic_curvature(self):
        self.assertIn('PROTOCOL_VERSION = "v2"', SOURCE)
        self.assertIn('param.copy_(base)', SOURCE)
        self.assertIn('model.eval()  # Keep autograd', SOURCE)
        self.assertIn('result = finish_run()', SOURCE)
        self.assertIn('overwriting is disabled', SOURCE)
        self.assertIn('oriented_prob', SOURCE)
        self.assertIn('reference_ratio_ranking_accuracy', SOURCE)


if __name__ == "__main__":
    unittest.main()
