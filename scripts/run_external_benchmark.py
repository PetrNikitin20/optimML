from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from optimml.large_model_experiment import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the external RewardBench and 1.1B CPU pilot")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--rewardbench-per-subset", type=int, default=4)
    parser.add_argument("--large-train-pairs", type=int, default=24)
    parser.add_argument("--large-hh-test-pairs", type=int, default=48)
    args = parser.parse_args()
    report = run(ROOT / "results", ROOT / "data" / "cache", args.seed, args.rewardbench_per_subset, args.large_train_pairs, args.large_hh_test_pairs)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
