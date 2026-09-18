from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimml.experiment import run


def main():
    parser = argparse.ArgumentParser(description="Run the real-data HH-RLHF preference experiment")
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    report = run(ROOT / "results", ROOT / "data" / "cache", args.pairs, args.dimension, args.seed)
    print(json.dumps({"best_pairwise": report["best_pairwise"], "best_pointwise": report["best_pointwise"], "gradient_checks": report["gradient_checks"]}, indent=2))


if __name__ == "__main__":
    main()
