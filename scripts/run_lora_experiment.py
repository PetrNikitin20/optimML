from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from optimml.lora_experiment import run_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CPU-reproducible LoRA reward-model experiment")
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    report = run_lora(
        ROOT / "results",
        ROOT / "data" / "cache",
        count=args.pairs,
        sample_seed=args.seed,
        training_seeds=tuple(args.training_seeds),
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps({"summary": report["summary"], "geometry": report["geometry"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
