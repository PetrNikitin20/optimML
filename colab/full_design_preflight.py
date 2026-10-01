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


def inspect_data(source: bytes, selected: list[str], train: int, evaluation: int) -> dict:
    from datasets import get_dataset_split_names, load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    revisions, provenance = {}, {}

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
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preflight evidence must not be overwritten")
    source = args.source.read_bytes()
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
              "datasets": inspect_data(source, args.datasets, args.train_pairs, args.eval_pairs)}
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
