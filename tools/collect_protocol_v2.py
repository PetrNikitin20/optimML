"""Publish numeric records from an independently verified local Colab export.

Raw public-dataset text and generated responses remain in the local backup.
The upstream dataset revision and deterministic sampler allow reconstruction.
Usage: python tools/collect_protocol_v2.py PATH_TO_EXTRACTED_PROTOCOL_V2
"""
from __future__ import annotations

import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path
from verify_protocol_v2 import verify_run

ROOT = Path(__file__).resolve().parents[1]


def collect(raw_root: Path):
    destination = ROOT / "results/factorial_v2/runs/pilot"
    source_dir = raw_root / "runs/pilot"
    for file in sorted(source_dir.glob("[0-9][0-9][0-9]_*.json")):
        obj = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(obj, dict) or obj.get("status") != "complete" or obj.get("run_key") != file.stem:
            continue
        report = verify_run(file, raw_root)
        key = obj["run_key"]
        if any(character in key for character in "/\\:"):
            raise ValueError("Unsafe run key")
        raw_lengths = json.loads((source_dir / f"{key}_generation_lengths.json").read_text(encoding="utf-8"))
        if statistics.mean(raw_lengths["policy"]) != obj["generation"]["policy_response_length_tokens"] or statistics.mean(raw_lengths["sft"]) != obj["generation"]["sft_response_length_tokens"]:
            raise ValueError("Response length metric mismatch")
        raw_generations = json.loads((source_dir / f"{key}_generations.json").read_text(encoding="utf-8"))
        judge_scores = [{"prompt_index": index, "policy_reward": row["policy_reward"],
                         "reference_reward": row["sft_reward"],
                         "policy_tokens": raw_lengths["policy"][index],
                         "reference_tokens": raw_lengths["sft"][index]}
                        for index, row in enumerate(raw_generations)]
        artifacts = [file, source_dir / f"{key}_rows.csv", source_dir / f"{key}_rows.parquet", source_dir / f"{key}_generation_lengths.json"]
        for suffix in ("hvp_sensitivity", "calibration_diagnostics", "projected_hessian_exact"):
            candidate = source_dir / f"{key}_{suffix}.json"
            if candidate.exists():
                artifacts.append(candidate)
        destination.mkdir(parents=True, exist_ok=True)
        output_paths = [destination / path.name for path in artifacts]
        score_path = destination / f"{key}_judge_scores.json"
        report_path = destination / f"{key}_verification.json"
        if any(path.exists() for path in output_paths + [score_path, report_path]):
            raise FileExistsError("Will not overwrite a collected run")
        for source, target in zip(artifacts, output_paths):
            shutil.copyfile(source, target)
        score_path.write_text(json.dumps(judge_scores, indent=2), encoding="utf-8")
        report.update(raw_data_hash_verified=True, train_holdout_prompt_disjoint_verified=True,
                      response_lengths_verified=True,
                      raw_text_distribution="kept_in_local_backup_not_redistributed_in_git",
                      scope="single_seed_pilot_not_inferential",
                      public_artifact_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output_paths + [score_path]})
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Collected and verified:", key)


if __name__ == "__main__":
    collect(Path(sys.argv[1]).resolve(strict=True))
