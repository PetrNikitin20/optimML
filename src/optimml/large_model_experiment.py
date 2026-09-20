from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import fetch_pairs
from .experiment import make_split
from .external_stats import accuracy_interval, paired_change
from .lora_experiment import EncodedPairs, RewardModel, evaluate, tokenize, train_once
from .rewardbench import CATEGORY_BY_SUBSET, as_dialogue, fetch_all, stratified_sample

LARGE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _score_bert(model: RewardModel, rows: list[dict], max_length: int = 192) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    tokenizer.truncation_side = "left"
    output = []
    model.eval()
    for row in rows:
        texts = [as_dialogue(row["prompt"], row["chosen"]), as_dialogue(row["prompt"], row["rejected"])]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        with torch.no_grad():
            score = model(batch["input_ids"], batch["attention_mask"]).double().cpu().numpy()
        output.append({"id": row["id"], "subset": row["subset"], "chosen_score": float(score[0]), "rejected_score": float(score[1]), "correct": bool(score[0] > score[1])})
    return output


def _large_model(seed: int):
    _seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(LARGE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    base = AutoModelForSequenceClassification.from_pretrained(
        LARGE_MODEL_ID,
        num_labels=1,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    return tokenizer, get_peft_model(base, config)


def _encode_pair(tokenizer, chosen: str, rejected: str, max_length: int = 64):
    common = dict(return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    return tokenizer(chosen, **common), tokenizer(rejected, **common)


def _large_scores(model, chosen_batch, rejected_batch):
    chosen = model(**chosen_batch).logits.float().squeeze()
    rejected = model(**rejected_batch).logits.float().squeeze()
    return chosen, rejected


def _evaluate_large(model, tokenizer, pairs: list[dict], max_length: int = 64) -> list[dict]:
    model.eval()
    output = []
    for number, row in enumerate(pairs, 1):
        if "prompt" in row:
            chosen_text = as_dialogue(row["prompt"], row["chosen"])
            rejected_text = as_dialogue(row["prompt"], row["rejected"])
        else:
            chosen_text, rejected_text = row["chosen"], row["rejected"]
        chosen_batch, rejected_batch = _encode_pair(tokenizer, chosen_text, rejected_text, max_length)
        with torch.no_grad():
            chosen_score, rejected_score = _large_scores(model, chosen_batch, rejected_batch)
        output.append(
            {
                "id": row.get("id", number - 1),
                "subset": row.get("subset", "HH-RLHF"),
                "chosen_score": float(chosen_score),
                "rejected_score": float(rejected_score),
                "correct": bool(chosen_score > rejected_score),
            }
        )
        print(f"TinyLlama evaluation: {number}/{len(pairs)}", flush=True)
    return output


def _train_large(model, tokenizer, rows: list[dict], beta: float = 0.3, max_length: int = 64) -> dict:
    encoder_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (head_parameters if "score" in name or "modules_to_save" in name else encoder_parameters).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": encoder_parameters, "lr": 2e-4}, {"params": head_parameters, "lr": 1e-3}],
        weight_decay=0.01,
    )
    started = time.perf_counter()
    losses = []
    model.train()
    for step, row in enumerate(rows, 1):
        chosen, rejected = _encode_pair(tokenizer, row["chosen"], row["rejected"], max_length)
        optimizer.zero_grad(set_to_none=True)
        chosen_score, rejected_score = _large_scores(model, chosen, rejected)
        loss = F.softplus(-beta * (chosen_score - rejected_score))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"TinyLlama training: {step}/{len(rows)} loss={losses[-1]:.6f}", flush=True)
    return {"steps": len(rows), "loss_first": losses[0], "loss_last": losses[-1], "loss_mean": float(np.mean(losses)), "elapsed_seconds": time.perf_counter() - started}


def _summarize(records: list[dict]) -> list[dict]:
    summary = []
    for category in ("Chat", "Chat Hard", "Safety", "Reasoning"):
        subset = [row for row in records if CATEGORY_BY_SUBSET.get(row["subset"]) == category]
        summary.append({"category": category, "count": len(subset), "accuracy": float(np.mean([row["correct"] for row in subset]))})
    summary.append({"category": "Macro categories", "count": len(records), "accuracy": float(np.mean([row["accuracy"] for row in summary]))})
    summary.append({"category": "Micro examples", "count": len(records), "accuracy": float(np.mean([row["correct"] for row in records]))})
    return summary


def run(output_dir: Path, cache_dir: Path, seed: int = 20260919, rewardbench_per_subset: int = 4, large_train_pairs: int = 24, large_hh_test_pairs: int = 48) -> dict:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    output_dir.mkdir(parents=True, exist_ok=True)
    hh_rows, hh_manifest = fetch_pairs(2000, cache_dir, sample_seed=20260918)
    split = make_split(2000, 20260918)
    rb_rows, rb_manifest = fetch_all(cache_dir)
    rb_sample = stratified_sample(rb_rows, rewardbench_per_subset, seed)

    # Reproduce one registered BERT-tiny LoRA checkpoint, then evaluate it out of domain.
    _, chosen, rejected = tokenize(hh_rows)
    bert_model, bert_run = train_once(chosen, rejected, split, "lora", 47, 0.3, 4, 16)
    bert_rb = _score_bert(bert_model, rb_sample)
    bert_summary = _summarize(bert_rb)
    bert_checkpoint = output_dir / "checkpoints" / "bert_tiny_lora"
    bert_checkpoint.mkdir(parents=True, exist_ok=True)
    bert_model.encoder.save_pretrained(bert_checkpoint, safe_serialization=True)
    save_file(
        {"weight": bert_model.score.weight.detach().cpu().contiguous(), "bias": bert_model.score.bias.detach().cpu().contiguous()},
        str(bert_checkpoint / "reward_head.safetensors"),
        metadata={"base_model": "prajjwal1/bert-tiny"},
    )
    del bert_model

    # CPU-budgeted 1.1B proof of concept; indices come only from the training partition.
    rng = np.random.default_rng(seed)
    train_indices = rng.choice(split.train, size=large_train_pairs, replace=False)
    hh_test_indices = rng.choice(split.test, size=large_hh_test_pairs, replace=False)
    large_train = [hh_rows[int(index)] for index in train_indices]
    large_hh_test = [hh_rows[int(index)] for index in hh_test_indices]
    tokenizer, large_model = _large_model(seed)
    trainable = sum(parameter.numel() for parameter in large_model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in large_model.parameters())
    before_hh = _evaluate_large(large_model, tokenizer, large_hh_test)
    before_rb = _evaluate_large(large_model, tokenizer, rb_sample)
    training = _train_large(large_model, tokenizer, large_train)
    after_hh = _evaluate_large(large_model, tokenizer, large_hh_test)
    after_rb = _evaluate_large(large_model, tokenizer, rb_sample)
    large_checkpoint = output_dir / "checkpoints" / "tinyllama_1p1b_lora"
    large_checkpoint.mkdir(parents=True, exist_ok=True)
    large_model.save_pretrained(large_checkpoint, safe_serialization=True)
    large_summary_before = _summarize(before_rb)
    large_summary_after = _summarize(after_rb)

    rows = []
    for model_name, phase, records in (
        ("bert-tiny-lora", "after", bert_rb),
        ("tinyllama-1.1b-lora", "before", before_rb),
        ("tinyllama-1.1b-lora", "after", after_rb),
    ):
        for row in records:
            rows.append({"model": model_name, "phase": phase, **row, "category": CATEGORY_BY_SUBSET[row["subset"]]})

    report = {
        "protocol": {
            "seed": seed,
            "rewardbench_per_subset": rewardbench_per_subset,
            "rewardbench_examples": len(rb_sample),
            "rewardbench_note": "Stratified diagnostic subset; not the official RewardBench leaderboard aggregate",
            "large_model_id": LARGE_MODEL_ID,
            "large_model_parameters": total,
            "large_model_trainable_parameters": trainable,
            "large_model_trainable_share": trainable / total,
            "large_train_pairs": large_train_pairs,
            "large_hh_test_pairs": large_hh_test_pairs,
            "large_max_length": 64,
            "large_lora_rank": 4,
            "large_lora_alpha": 8,
            "large_beta": 0.3,
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__},
        "hh_manifest": hh_manifest,
        "rewardbench_manifest": rb_manifest,
        "rewardbench_sample": [{"id": row["id"], "subset": row["subset"]} for row in rb_sample],
        "bert_registered_run": bert_run.__dict__,
        "bert_rewardbench": bert_summary,
        "bert_rewardbench_accuracy_interval": accuracy_interval(bert_rb, seed + 1),
        "tinyllama_training": training,
        "tinyllama_hh_before_accuracy": float(np.mean([row["correct"] for row in before_hh])),
        "tinyllama_hh_after_accuracy": float(np.mean([row["correct"] for row in after_hh])),
        "tinyllama_rewardbench_before": large_summary_before,
        "tinyllama_rewardbench_after": large_summary_after,
        "tinyllama_rewardbench_before_accuracy_interval": accuracy_interval(before_rb, seed + 2),
        "tinyllama_rewardbench_after_accuracy_interval": accuracy_interval(after_rb, seed + 3),
        "tinyllama_rewardbench_paired_change": paired_change(before_rb, after_rb, seed + 4),
    }
    result_path = output_dir / "external_benchmark_results.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    (output_dir / "external_benchmark_verification.json").write_text(json.dumps({"sha256": digest, "rewardbench_sha256": rb_manifest["sha256_canonical_jsonl"], "hh_sha256": hh_manifest["sha256_canonical_jsonl"]}, indent=2), encoding="utf-8")
    with (output_dir / "external_benchmark_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return report
