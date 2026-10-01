"""Read-only scientific/data preflight. Does not load model weights or train.

Run with --source pointing to factorial_preference_colab.py. Dataset revisions
are resolved once and pinned for this diagnostic, not silently registered as
the final confirmatory protocol. Only counts/hashes, never texts, are exported.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import random
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET_IDS = {
    "ultrafeedback": "HuggingFaceH4/ultrafeedback_binarized",
    "reddit_tldr": "openai/summarize_from_feedback",
    "helpsteer2": "nvidia/HelpSteer2",
    "hh_rlhf": "Anthropic/hh-rlhf",
}


def normalized_prompt(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def cardinality(rows: list[dict], train: int, evaluation: int) -> dict:
    valid = [x for x in rows if all(isinstance(x.get(k), str) and x[k].strip()
             for k in ("prompt", "chosen", "rejected")) and x["chosen"] != x["rejected"]]
    exact = {x["prompt"] for x in valid}
    normalized = {normalized_prompt(x["prompt"]) for x in valid}
    return {"loaded_comparisons": len(rows), "valid_comparisons": len(valid),
            "unique_exact_prompts": len(exact), "unique_normalized_prompts": len(normalized),
            "required_prompts": train + evaluation,
            "exact_cardinality_pass": len(exact) >= train + evaluation,
            "normalized_cardinality_pass": len(normalized) >= train + evaluation,
            "normalization": "NFKC_casefold_whitespace_only_not_semantic_deduplication"}


def prepare_confirmatory_split(rows: list[dict], train: int, evaluation: int) -> tuple[list, list]:
    """Proposed v3 partition. Preserve texts; deduplicate normalized prompt keys.

    Deterministic lexicographic selection is label-independent in its ordering
    of the two responses. It does not select by perceived response quality.
    This is basic normalization, NOT a semantic near-duplicate audit.
    """
    valid = [dict(x) for x in rows if all(isinstance(x.get(k), str) and x[k].strip()
             for k in ("prompt", "chosen", "rejected")) and x["chosen"] != x["rejected"]]
    unique = {}
    for row in sorted(valid, key=lambda x: (x["prompt"], tuple(sorted([x["chosen"], x["rejected"]])))):
        unique.setdefault(normalized_prompt(row["prompt"]), row)
    ordered = [unique[key] for key in sorted(unique)]
    random.Random(20261001).shuffle(ordered)
    if len(ordered) < train + evaluation:
        raise ValueError("Insufficient unique normalized prompts for the full budget")
    training, holdout = ordered[:train], ordered[train:train + evaluation]
    if {normalized_prompt(x["prompt"]) for x in training} & {normalized_prompt(x["prompt"]) for x in holdout}:
        raise AssertionError("Normalized prompt leakage")
    return training, holdout


def inspect_data(source: bytes, selected: list[str], train: int, evaluation: int,
                 export_dir: Path | None = None, revision_report: dict | None = None) -> dict:
    from datasets import get_dataset_split_names, load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    revisions, provenance = {}, {}
    if revision_report:
        for label in selected:
            prov = revision_report["datasets"][label]["provenance"]
            revision = prov.get("revision", prov.get("dataset_revision"))
            if not revision:
                raise ValueError("No pinned revision for " + label)
            revisions[DATASET_IDS[label]] = revision

    def pin(name, *args, **kwargs):
        if name != "parquet":
            if name not in revisions:
                revisions[name] = api.dataset_info(name).sha
            kwargs["revision"] = revisions[name]
            provenance[name] = {"revision": revisions[name]}
        return load_dataset(name, *args, **kwargs)

    def splits(name, *args, **kwargs):
        if name not in revisions:
            revisions[name] = api.dataset_info(name).sha
        kwargs["revision"] = revisions[name]
        return get_dataset_split_names(name, *args, **kwargs)

    names = {"messages_to_text", "assistant_message"} | {"load_" + x for x in DATASET_IDS}
    cleaned = "\n".join(x for x in source.decode("utf-8").splitlines() if not x.startswith("!"))
    tree = ast.parse(cleaned)
    module = ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef)
                             and x.name in names], type_ignores=[])
    scope = {"Any": Any, "load_dataset": pin, "get_dataset_split_names": splits,
             "save_stage": lambda stage, **values: provenance.update({"reddit_tldr": values})}
    exec(compile(module, "isolated_dataset_loaders", "exec"), scope)
    output = {}
    for label in selected:
        print("PREFLIGHT_LOADING", label, flush=True)
        try:
            rows = scope["load_" + label]()
            report = cardinality(rows, train, evaluation)
            # Reproduce exactly the current fixed split; inspect overlap after
            # basic normalization in addition to exact-string separation.
            unique = {}
            for row in rows:
                unique.setdefault(row["prompt"], row)
            ordered = [unique[key] for key in sorted(unique)]
            random.Random(20261001).shuffle(ordered)
            tr, ev = ordered[:train], ordered[train:train + evaluation]
            report["normalized_train_eval_prompt_overlap"] = len(
                {normalized_prompt(x["prompt"]) for x in tr}
                & {normalized_prompt(x["prompt"]) for x in ev})
            report["selected_clean_pairs_sha256"] = hashlib.sha256("\n".join(
                json.dumps(x, sort_keys=True, ensure_ascii=False) for x in tr + ev
            ).encode("utf-8")).hexdigest()
            report["provenance"] = dict(provenance.get(DATASET_IDS[label],
                                           provenance.get(label, {})))
            if export_dir is not None:
                training, holdout = prepare_confirmatory_split(rows, train, evaluation)
                canonical = "\n".join(json.dumps(x, sort_keys=True, ensure_ascii=False)
                                       for x in training + holdout)
                export_path = export_dir / (label + "_clean_pairs.json")
                if export_path.exists():
                    raise FileExistsError("Prepared data must not be overwritten")
                export_path.write_text(json.dumps({"data_protocol": "proposed_v3_normalized_prompt_split",
                    "train": training, "eval": holdout}, ensure_ascii=False), encoding="utf-8")
                report["proposed_v3_split"] = {
                    "train_pairs": len(training), "eval_pairs": len(holdout),
                    "normalized_train_eval_prompt_overlap": 0,
                    "data_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                    "export_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
                    "export_file": export_path.name,
                    "scope": "prepared_real_data_not_training_or_evaluation_results",
                    "semantic_near_duplicate_audit_complete": False}
            output[label] = report
            print("PREFLIGHT_DATA", label, json.dumps(report), flush=True)
        except Exception as error:
            output[label] = {"error_type": type(error).__name__, "error": str(error),
                             "exact_cardinality_pass": False}
            print("PREFLIGHT_ERROR", label, json.dumps(output[label]), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_IDS), default=list(DATASET_IDS))
    parser.add_argument("--train-pairs", type=int, default=4000)
    parser.add_argument("--eval-pairs", type=int, default=800)
    parser.add_argument("--export-pairs-dir", type=Path)
    parser.add_argument("--revision-report", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preflight evidence must not be overwritten")
    source = args.source.read_bytes()
    if args.export_pairs_dir:
        if args.export_pairs_dir.exists():
            raise FileExistsError("Choose a new prepared-data directory")
        args.export_pairs_dir.mkdir(parents=True)
    revision_report = json.loads(args.revision_report.read_text()) if args.revision_report else None
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                                       "--format=csv,noheader"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu = None
    result = {"kind": "data_and_resource_preflight_not_training_results",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(source).hexdigest(),
              "python": platform.python_version(), "gpu": gpu,
              "colab_drive_mounted": os.path.ismount("/content/drive"),
              "datasets": inspect_data(source, args.datasets, args.train_pairs, args.eval_pairs,
                                       args.export_pairs_dir, revision_report)}
    result["all_requested_data_checks_pass"] = all(
        x.get("exact_cardinality_pass", False) and x.get("normalized_cardinality_pass", False)
        and x.get("normalized_train_eval_prompt_overlap") == 0 for x in result["datasets"].values())
    result["full_design_ready"] = False
    result["additional_gates"] = ["persistent_storage", "allocated_compute_budget",
        "registered_context_and_judging_protocol", "validated_full_scope_curvature",
        "pinned_model_and_data_revisions", "near_duplicate_audit"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PREFLIGHT_REPORT", json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
