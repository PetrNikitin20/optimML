from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optimml.external_stats import accuracy_interval, paired_change


def main() -> None:
    results = ROOT / "results"
    report_path = results / "external_benchmark_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with (results / "external_benchmark_rows.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["id"] = int(row["id"])
        row["correct"] = row["correct"].lower() == "true"
    bert = [row for row in rows if row["model"] == "bert-tiny-lora"]
    before = [row for row in rows if row["model"] == "tinyllama-1.1b-lora" and row["phase"] == "before"]
    after = [row for row in rows if row["model"] == "tinyllama-1.1b-lora" and row["phase"] == "after"]
    seed = int(report["protocol"]["seed"])
    report["bert_rewardbench_accuracy_interval"] = accuracy_interval(bert, seed + 1)
    report["tinyllama_rewardbench_before_accuracy_interval"] = accuracy_interval(before, seed + 2)
    report["tinyllama_rewardbench_after_accuracy_interval"] = accuracy_interval(after, seed + 3)
    report["tinyllama_rewardbench_paired_change"] = paired_change(before, after, seed + 4)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    verification = json.loads((results / "external_benchmark_verification.json").read_text(encoding="utf-8"))
    verification["sha256"] = digest
    (results / "external_benchmark_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps({"paired_change": report["tinyllama_rewardbench_paired_change"], "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
