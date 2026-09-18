from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from .data import fetch_pairs
from .experiment import make_split

MODEL_ID = "prajjwal1/bert-tiny"
MAX_LENGTH = 192
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class EncodedPairs(Dataset):
    def __init__(self, chosen: dict[str, torch.Tensor], rejected: dict[str, torch.Tensor], indices: np.ndarray):
        self.chosen = chosen
        self.rejected = rejected
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = self.indices[item]
        result: dict[str, torch.Tensor] = {}
        for key, value in self.chosen.items():
            result[f"chosen_{key}"] = value[index]
        for key, value in self.rejected.items():
            result[f"rejected_{key}"] = value[index]
        return result


class RewardModel(nn.Module):
    def __init__(self, method: str):
        super().__init__()
        base = AutoModel.from_pretrained(MODEL_ID)
        if method == "lora":
            config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=["query", "value"],
                bias="none",
            )
            self.encoder = get_peft_model(base, config)
        elif method == "head_only":
            for parameter in base.parameters():
                parameter.requires_grad = False
            self.encoder = base
        else:
            raise ValueError(f"Unknown method: {method}")
        self.score = nn.Linear(base.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.score(output.last_hidden_state[:, 0]).squeeze(-1)

    def trainable_parameters(self) -> tuple[int, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return trainable, total


@dataclass
class RunResult:
    method: str
    seed: int
    beta: float
    validation_accuracy: float
    test_accuracy: float
    test_pair_loss: float
    test_brier: float
    test_margin: float
    trainable_parameters: int
    total_parameters: int
    elapsed_seconds: float
    selected_epoch: int


def tokenize(rows: list[dict[str, str]]):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.truncation_side = "left"
    common = dict(padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    chosen = tokenizer([row["chosen"] for row in rows], **common)
    rejected = tokenizer([row["rejected"] for row in rows], **common)
    return tokenizer, chosen, rejected


def scores(model: RewardModel, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
    rejected = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
    return chosen, rejected


def evaluate(model: RewardModel, loader: DataLoader, beta: float) -> dict[str, float]:
    model.eval()
    margins: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            chosen, rejected = scores(model, batch)
            margins.append((chosen - rejected).cpu())
    margin = torch.cat(margins).double()
    probability = torch.sigmoid(beta * margin)
    return {
        "accuracy": float((margin > 0).double().mean()),
        "pair_loss": float(F.softplus(-beta * margin).mean()),
        "brier": float(((probability - 1.0) ** 2).mean()),
        "mean_margin": float(margin.mean()),
    }


def train_once(
    chosen: dict[str, torch.Tensor],
    rejected: dict[str, torch.Tensor],
    split,
    method: str,
    seed: int,
    beta: float,
    epochs: int,
    batch_size: int,
) -> tuple[RewardModel, RunResult]:
    set_seed(seed)
    model = RewardModel(method)
    trainable, total = model.trainable_parameters()
    train_loader = DataLoader(
        EncodedPairs(chosen, rejected, split.train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(EncodedPairs(chosen, rejected, split.validation), batch_size=batch_size * 2)
    test_loader = DataLoader(EncodedPairs(chosen, rejected, split.test), batch_size=batch_size * 2)
    encoder_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("score.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": 5e-4},
            {"params": model.score.parameters(), "lr": 1e-3},
        ],
        weight_decay=0.01,
    )
    started = time.perf_counter()
    best_state = None
    best_validation_loss = math.inf
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            chosen_score, rejected_score = scores(model, batch)
            loss = F.softplus(-beta * (chosen_score - rejected_score)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
        validation = evaluate(model, validation_loader, beta)
        if validation["pair_loss"] < best_validation_loss:
            best_validation_loss = validation["pair_loss"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    validation = evaluate(model, validation_loader, beta)
    test = evaluate(model, test_loader, beta)
    return model, RunResult(
        method=method,
        seed=seed,
        beta=beta,
        validation_accuracy=validation["accuracy"],
        test_accuracy=test["accuracy"],
        test_pair_loss=test["pair_loss"],
        test_brier=test["brier"],
        test_margin=test["mean_margin"],
        trainable_parameters=trainable,
        total_parameters=total,
        elapsed_seconds=elapsed,
        selected_epoch=best_epoch,
    )


def gradient_vector(model: RewardModel, batch: dict[str, torch.Tensor], beta: float) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.zero_grad(set_to_none=True)
    chosen_score, rejected_score = scores(model, batch)
    margins = chosen_score - rejected_score
    loss = F.softplus(-beta * margins).mean()
    loss.backward()
    pieces = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            pieces.append((parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1))
    gradient = torch.cat(pieces).detach().double().cpu().numpy()
    weights = torch.sigmoid(-beta * margins.detach()).double().cpu().numpy()
    return gradient, weights


def angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    cosine = float(np.dot(left, right) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def lora_geometry(model: RewardModel, dataset: EncodedPairs, batch_size: int) -> list[dict[str, float]]:
    batch = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False)))
    betas = [0.1, 0.3, 1.0, 3.0, 10.0]
    vectors: dict[float, np.ndarray] = {}
    weights: dict[float, np.ndarray] = {}
    for beta in betas:
        vectors[beta], weights[beta] = gradient_vector(model, batch, beta)
    reference = vectors[1.0]
    rows = []
    for beta in betas:
        weight = weights[beta]
        ess = float(weight.sum() ** 2 / np.square(weight).sum())
        rows.append(
            {
                "beta": beta,
                "angle_to_beta_1_deg": angle_degrees(vectors[beta], reference),
                "gradient_norm": float(np.linalg.norm(vectors[beta])),
                "normalized_ess": ess / weight.size,
                "saturated_weight_share": float(((weight < 0.05) | (weight > 0.95)).mean()),
            }
        )
    model.zero_grad(set_to_none=True)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_lora(
    output_dir: Path,
    cache_dir: Path,
    count: int = 2000,
    sample_seed: int = 20260918,
    training_seeds: tuple[int, ...] = (11, 29, 47),
    beta: float = 0.3,
    epochs: int = 4,
    batch_size: int = 16,
) -> dict:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = fetch_pairs(count, cache_dir, sample_seed=sample_seed)
    tokenizer, chosen, rejected = tokenize(rows)
    split = make_split(count, sample_seed)

    run_rows: list[dict] = []
    lora_models: list[tuple[float, RewardModel, int]] = []
    for method in ("head_only", "lora"):
        for seed in training_seeds:
            print(f"training method={method} seed={seed}", flush=True)
            model, result = train_once(chosen, rejected, split, method, seed, beta, epochs, batch_size)
            row = result.__dict__.copy()
            run_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if method == "lora":
                lora_models.append((result.validation_accuracy, model, seed))
            else:
                del model

    summaries = []
    for method in ("head_only", "lora"):
        subset = [row for row in run_rows if row["method"] == method]
        summaries.append(
            {
                "method": method,
                "runs": len(subset),
                "accuracy_mean": float(np.mean([row["test_accuracy"] for row in subset])),
                "accuracy_sd": float(np.std([row["test_accuracy"] for row in subset], ddof=1)),
                "pair_loss_mean": float(np.mean([row["test_pair_loss"] for row in subset])),
                "brier_mean": float(np.mean([row["test_brier"] for row in subset])),
                "trainable_parameters": subset[0]["trainable_parameters"],
                "trainable_share": subset[0]["trainable_parameters"] / subset[0]["total_parameters"],
                "elapsed_seconds_mean": float(np.mean([row["elapsed_seconds"] for row in subset])),
                "selected_epoch_mean": float(np.mean([row["selected_epoch"] for row in subset])),
            }
        )

    _, geometry_model, geometry_seed = max(lora_models, key=lambda item: item[0])
    geometry = lora_geometry(
        geometry_model,
        EncodedPairs(chosen, rejected, split.validation),
        batch_size=min(64, len(split.validation)),
    )
    del geometry_model

    config = AutoModel.from_pretrained(MODEL_ID).config
    report = {
        "protocol": {
            "model_id": MODEL_ID,
            "model_commit": getattr(config, "_commit_hash", None),
            "tokenizer_class": tokenizer.__class__.__name__,
            "max_length": MAX_LENGTH,
            "truncation_side": "left",
            "pairs": count,
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
            "sample_seed": sample_seed,
            "training_seeds": list(training_seeds),
            "beta": beta,
            "epochs": epochs,
            "batch_size_pairs": batch_size,
            "optimizer": "AdamW",
            "encoder_learning_rate": 5e-4,
            "head_learning_rate": 1e-3,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_targets": ["query", "value"],
            "geometry_seed": geometry_seed,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "data_manifest": manifest,
        "runs": run_rows,
        "summary": summaries,
        "geometry": geometry,
    }
    report_path = output_dir / "lora_results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    (output_dir / "lora_verification.json").write_text(
        json.dumps({"lora_results_sha256": digest, "model_id": MODEL_ID, "data_sha256": manifest["sha256_canonical_jsonl"]}, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "lora_runs.csv", run_rows)
    write_csv(output_dir / "lora_summary.csv", summaries)
    write_csv(output_dir / "lora_geometry.csv", geometry)
    return report
