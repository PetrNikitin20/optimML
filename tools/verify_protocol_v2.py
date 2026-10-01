"""Recompute exported v2 pilot metrics without GPU or training libraries.

Usage: python tools/verify_protocol_v2.py PATH_TO_PROTOCOL_V2_DIRECTORY
Only checks quantities recoverable from raw records, not Hessian convergence.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def ece(probabilities, targets, bins=15):
    value = 0.0
    for index in range(bins):
        items = [(p, y) for p, y in zip(probabilities, targets)
                 if p >= index / bins and (p < (index + 1) / bins if index < bins - 1 else p <= 1)]
        if items:
            value += len(items) / len(probabilities) * abs(statistics.mean(p for p, _ in items) - statistics.mean(y for _, y in items))
    return value


def recompute_rows(records, beta):
    n = len(records)
    d = [r["d"] for r in records]
    c = [r["c"] for r in records]
    probability = [1 / (1 + math.exp(-beta * value)) for value in d]
    targets = [float(i % 2 == 0) for i in range(n)]
    oriented = [p if y else 1 - p for p, y in zip(probability, targets)]
    for row, pi, yi in zip(records, oriented, targets):
        if not math.isclose(row["a"] - row["b"], row["d"], abs_tol=1e-6):
            raise ValueError("Margin identity failed")
        if not math.isclose((row["a"] + row["b"]) / 2, row["c"], abs_tol=1e-6):
            raise ValueError("Common-shift identity failed")
        if row["presentation_target"] != yi or not math.isclose(row["presentation_probability"], pi, abs_tol=1e-12):
            raise ValueError("Calibration orientation failed")
    return {
        "likelihood_ranking_accuracy": statistics.mean(float(r["policy_chosen_logp"] > r["policy_rejected_logp"]) for r in records),
        "reference_ratio_ranking_accuracy": statistics.mean(float(v > 0) for v in d),
        "sft_likelihood_ranking_accuracy": statistics.mean(float(r["sft_chosen_logp"] > r["sft_rejected_logp"]) for r in records),
        "brier": statistics.mean((p - y) ** 2 for p, y in zip(oriented, targets)),
        "ece_15": ece(oriented, targets),
        "c_mean": statistics.mean(c), "c_abs_mean": statistics.mean(abs(v) for v in c),
        "c_sd": statistics.stdev(c), "d_mean": statistics.mean(d),
        "d_abs_mean": statistics.mean(abs(v) for v in d), "d_sd": statistics.stdev(d),
        "n_eval_pairs": n,
    }


def verify_run(result_file: Path, artifact_root: Path):
    obj = json.loads(result_file.read_text(encoding="utf-8"))
    if obj.get("protocol_version") != "v2" or obj.get("status") != "complete":
        raise ValueError("Not a completed v2 run")
    for name, expected in obj["artifact_sha256"].items():
        relative = Path(name).relative_to("optimML_factorial/protocol_v2")
        file = artifact_root / relative
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {file}")
    key = obj["run_key"]
    directory = result_file.parent
    pair_data = json.loads((directory / f"{key}_pairs.json").read_text(encoding="utf-8"))
    train, test = pair_data["train"], pair_data["eval"]
    if {r["prompt"] for r in train} & {r["prompt"] for r in test}:
        raise ValueError("Train/holdout prompt leakage")
    canonical = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in train + test)
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != obj["data_sha256"]:
        raise ValueError("Selected data hash mismatch")
    with (directory / f"{key}_rows.csv").open(encoding="utf-8", newline="") as stream:
        records = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    metrics = recompute_rows(records, obj["config"]["beta"])
    for name, measured in metrics.items():
        if not math.isclose(obj["evaluation"][name], measured, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"Evaluation metric mismatch: {name}")
    generations = json.loads((directory / f"{key}_generations.json").read_text(encoding="utf-8"))
    deltas = [r["policy_reward"] - r["sft_reward"] for r in generations]
    for name, computed in {
        "generation_win_rate_vs_sft": statistics.mean(float(v > 0) for v in deltas),
        "generation_tie_rate": statistics.mean(float(v == 0) for v in deltas),
        "mean_reward_delta": statistics.mean(deltas),
        "n_generation_prompts": len(deltas),
    }.items():
        if not math.isclose(obj["generation"][name], computed, abs_tol=1e-10):
            raise ValueError(f"Generation metric mismatch: {name}")
    cfg = obj["config"]
    if obj["train"]["examples_seen"] != cfg["max_steps"] * cfg["batch_size"] * cfg["grad_accum"]:
        raise ValueError("Training sample accounting mismatch")
    return {"run_key": key, "status": "verified_against_raw_records", "recomputed_evaluation": metrics,
            "limits": "KL and curvature are recorded measurements, not independently recomputed by this verifier; pilot is not inferential evidence."}


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve(strict=True)
    reports = []
    for file in sorted((root / "runs/pilot").glob("[0-9][0-9][0-9]_*.json")):
        obj = json.loads(file.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and obj.get("status") == "complete" and obj.get("run_key") == file.stem:
            reports.append(verify_run(file, root))
    if not reports:
        raise SystemExit("No completed v2 runs found")
    print(json.dumps(reports, ensure_ascii=False, indent=2))
