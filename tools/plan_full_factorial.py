"""Export all 288 full-budget cells and audit coverage without importing torch.

This is a planning/coverage tool, NOT a training runner. Missing observations
remain missing. Pilot, smoke, mismatched, duplicate and corrupted records never
count towards the full experiment. No statistical results are manufactured.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def manifest_from_source(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse("\n".join(x for x in text.splitlines() if not x.startswith("!")))
    names = {"MODELS", "LOSSES", "DATASETS", "NOISE_LEVELS", "SEEDS"}
    nodes = [x for x in tree.body if (
        isinstance(x, ast.ClassDef) and x.name == "StudyConfig") or (
        isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in x.targets)) or (
        isinstance(x, ast.FunctionDef) and x.name == "build_manifest")]
    scope = {"dataclass": dataclass}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "registered_manifest", "exec"), scope)
    configs = [asdict(x) for x in scope["build_manifest"]()]
    if len(configs) != 288 or len({json.dumps(x, sort_keys=True) for x in configs}) != 288:
        raise ValueError("Expected 288 unique registered configurations")
    return configs


def run_key(index: int, config: dict) -> str:
    return (f'{index:03d}_{config["loss"]}_{config["model_label"]}_{config["dataset"]}_'
            f'noise{config["noise"]:.1f}_seed{config["seed"]}')


def full_training_errors(record: dict, index: int, config: dict) -> list[str]:
    errors = []
    for key, value in {"status": "complete", "stage": "complete", "protocol_version": "v2",
                       "execution_profile": "full", "run_index": index,
                       "run_key": run_key(index, config), "config": config}.items():
        if record.get(key) != value:
            errors.append("mismatch:" + key)
    train = record.get("train", {})
    examples = config["max_steps"] * config["grad_accum"] * config["batch_size"]
    if train.get("optimizer_steps") != config["max_steps"]:
        errors.append("incomplete_optimizer_budget")
    if train.get("examples_seen") != examples:
        errors.append("incomplete_example_budget")
    if record.get("evaluation", {}).get("n_eval_pairs") != config["eval_pairs"]:
        errors.append("incomplete_holdout_budget")
    if record.get("generation", {}).get("n_generation_prompts") != config["generation_prompts"]:
        errors.append("incomplete_generation_budget")
    if not record.get("source_sha256") or not record.get("data_sha256"):
        errors.append("missing_provenance_hash")
    return errors


def publication_errors(record: dict) -> list[str]:
    errors = []
    metrics = {
        "evaluation": ["likelihood_ranking_accuracy", "reference_ratio_ranking_accuracy", "brier",
                       "ece_15", "c_mean", "c_abs_mean", "d_mean", "d_abs_mean"],
        "generation": ["generation_win_rate_vs_sft", "policy_response_length_tokens",
                       "sft_response_length_tokens"],
        "train": ["train_seconds", "peak_gpu_memory_gb"],
        "curvature": ["empirical_fisher_top_eigenvalue", "empirical_fisher_trace"],
    }
    for section, fields in metrics.items():
        for key in fields:
            value = record.get(section, {}).get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                errors.append("missing_or_nonfinite:" + section + "." + key)
    value = record.get("kl_to_sft")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        errors.append("missing_or_nonfinite:kl_to_sft")
    curvature = record.get("curvature", {})
    if curvature.get("hessian_validated") is not True or curvature.get("hessian_scope") != "all_trainable_LoRA_parameters":
        errors.append("full_scope_Hessian_not_validated")
    if not record.get("artifact_sha256"):
        errors.append("missing_raw_artifact_hashes")
    return errors


def coverage(configs: list[dict], results: Path) -> tuple[list[dict], dict]:
    candidates, ignored, rejected = {}, [], []
    for path in sorted(results.rglob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError) as error:
            rejected.append({"file": str(path), "reason": type(error).__name__})
            continue
        if not isinstance(obj, dict) or obj.get("status") != "complete" or path.stem != obj.get("run_key"):
            continue
        if obj.get("execution_profile") != "full":
            ignored.append({"file": str(path), "reason": "not_full_profile"})
            continue
        index = obj.get("run_index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(configs):
            rejected.append({"file": str(path), "reason": "invalid_run_index"})
            continue
        errors = full_training_errors(obj, index, configs[index])
        if errors:
            rejected.append({"file": str(path), "reason": errors})
            continue
        candidates.setdefault(index, []).append((path, obj))
    cells = []
    for index, config in enumerate(configs):
        matches = candidates.get(index, [])
        state, issues, filename = "missing", [], ""
        if len(matches) > 1:
            state, issues = "duplicate_conflict", [str(p) for p, _ in matches]
        elif matches:
            filename = str(matches[0][0])
            issues = publication_errors(matches[0][1])
            state = "full_budget_needs_validation" if issues else "full_budget_metrics_present"
        cells.append({"run_index": index, "run_key": run_key(index, config),
                      **config, "coverage_status": state, "result_file": filename,
                      "validation_issues": issues})
    n_training = sum(c["coverage_status"].startswith("full_budget_") for c in cells)
    n_metrics = sum(c["coverage_status"] == "full_budget_metrics_present" for c in cells)
    return cells, {"expected_runs": len(configs), "full_budget_runs": n_training,
                   "full_budget_metrics_present": n_metrics,
                   "missing_or_conflicting_runs": len(configs) - n_training,
                   "ignored_non_full_records": ignored, "rejected_records": rejected,
                   "publication_ready": False,
                   "publication_note": "Coverage is not independent verification, uncertainty analysis or scientific protocol approval."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "colab/factorial_preference_colab.py")
    parser.add_argument("--results", type=Path, default=ROOT / "results/factorial_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory to preserve earlier coverage evidence")
    configs = manifest_from_source(args.source)
    cells, status = coverage(configs, args.results)
    status.update(checked_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  purpose="planning_and_coverage_not_measured_results")
    args.output.mkdir(parents=True)
    (args.output / "coverage.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "manifest_full.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")
    with (args.output / "cells.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
