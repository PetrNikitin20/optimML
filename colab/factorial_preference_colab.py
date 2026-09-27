# %% [markdown]
# # Real-data factorial study of preference losses
#
# This notebook executes one reproducible cell of the factorial design
# `loss x model size x dataset x label noise x seed`. Every completed run is
# written as JSON so interrupted Colab sessions can resume without inventing or
# reconstructing results. Run the manifest cell first, then assign a `RUN_INDEX`
# to each Colab session.

# %%
!pip -q install "transformers>=4.48,<5" "datasets>=3.2,<4" "peft>=0.14,<1" "accelerate>=1.2,<2" "bitsandbytes>=0.45,<1" "scikit-learn>=1.5,<2" "pynvml>=12,<13" "pandas>=2.2,<3" "pyarrow>=18,<20"

# %%
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, get_dataset_split_names, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/optimML_factorial")
except ImportError:
    OUTPUT_ROOT = Path("./optimML_factorial")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "runs").mkdir(exist_ok=True)
(OUTPUT_ROOT / "manifests").mkdir(exist_ok=True)


@dataclass(frozen=True)
class StudyConfig:
    loss: str
    model_label: str
    model_id: str
    dataset: str
    noise: float
    seed: int
    train_pairs: int = 4000
    eval_pairs: int = 800
    max_length: int = 256
    max_steps: int = 400
    batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 2e-4
    beta: float = 0.1
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    curvature_batches: int = 6
    hessian_power_iterations: int = 5
    generation_prompts: int = 32
    checkpoint_every: int = 25
    log_every: int = 1


# Exact requested size classes. The checkpoints are all from the Qwen family,
# but the 8B checkpoint belongs to Qwen3. Treat model checkpoint, rather than
# parameter count alone, as the experimental factor in inferential models.
MODELS = {
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "8B": "Qwen/Qwen3-8B",
    "14B": "Qwen/Qwen2.5-14B-Instruct",
}
LOSSES = ["pairwise", "pointwise"]
DATASETS = ["ultrafeedback", "reddit_tldr", "helpsteer2", "hh_rlhf"]
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3]
SEEDS = [11, 29, 47]


def build_manifest() -> list[StudyConfig]:
    return [
        StudyConfig(loss, label, model_id, dataset, noise, seed)
        for loss in LOSSES
        for label, model_id in MODELS.items()
        for dataset in DATASETS
        for noise in NOISE_LEVELS
        for seed in SEEDS
    ]


MANIFEST = build_manifest()
assert len(MANIFEST) == 288
manifest_path = OUTPUT_ROOT / "manifests" / "factorial_manifest.json"
manifest_path.write_text(
    json.dumps([asdict(x) for x in MANIFEST], ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"Manifest: {len(MANIFEST)} runs -> {manifest_path}")

# %% [markdown]
# ## Select one run
#
# Change only `RUN_INDEX`. Completed indices are skipped automatically.

# %%
RUN_INDEX = 0  # @param {type:"integer"}
EXECUTION_PROFILE = "pilot"  # @param ["smoke", "pilot", "full"]


def apply_execution_profile(config: StudyConfig, profile: str) -> StudyConfig:
    if profile == "full":
        return config
    if profile == "pilot":
        return replace(
            config,
            train_pairs=256,
            eval_pairs=64,
            max_length=192,
            max_steps=10,
            grad_accum=4,
            curvature_batches=2,
            hessian_power_iterations=2,
            generation_prompts=8,
            checkpoint_every=5,
            log_every=1,
        )
    if profile == "smoke":
        return replace(
            config,
            train_pairs=64,
            eval_pairs=16,
            max_length=128,
            max_steps=2,
            grad_accum=2,
            curvature_batches=1,
            hessian_power_iterations=1,
            generation_prompts=2,
            checkpoint_every=1,
            log_every=1,
        )
    raise ValueError(f"Unknown EXECUTION_PROFILE={profile!r}")


cfg = apply_execution_profile(MANIFEST[RUN_INDEX], EXECUTION_PROFILE)
run_key = (
    f"{RUN_INDEX:03d}_{cfg.loss}_{cfg.model_label}_{cfg.dataset}_"
    f"noise{cfg.noise:.1f}_seed{cfg.seed}"
)
RUN_DIR = OUTPUT_ROOT / "runs" / EXECUTION_PROFILE
ADAPTER_DIR = OUTPUT_ROOT / "adapters" / EXECUTION_PROFILE
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints" / EXECUTION_PROFILE
for directory in (RUN_DIR, ADAPTER_DIR, CHECKPOINT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
result_path = RUN_DIR / f"{run_key}.json"
checkpoint_path = CHECKPOINT_DIR / f"{run_key}.pt"
checkpoint_adapter_dir = CHECKPOINT_DIR / f"{run_key}_adapter"
print(asdict(cfg))
print("Execution profile:", EXECUTION_PROFILE)
print("Result:", result_path)
if result_path.exists():
    print("This run is already complete. Choose another RUN_INDEX.")


def require_gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the 3B/8B/14B factorial study")
    props = torch.cuda.get_device_properties(0)
    info = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": round(props.total_memory / 2**30, 3),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    print(info)
    return info


SYSTEM_INFO = require_gpu()

# %%
def messages_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("content", item.get("text", ""))))
            else:
                parts.append(str(item))
        return "\n".join(x for x in parts if x)
    return str(value)


def assistant_message(value: Any) -> str:
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict) and item.get("role") == "assistant":
                return str(item.get("content", item.get("text", "")))
    return messages_to_text(value)


def load_ultrafeedback() -> list[dict[str, str]]:
    name = "HuggingFaceH4/ultrafeedback_binarized"
    splits = get_dataset_split_names(name)
    split = "train_prefs" if "train_prefs" in splits else "train"
    ds = load_dataset(name, split=split)
    rows = []
    for x in ds:
        prompt = messages_to_text(x.get("prompt", ""))
        chosen = assistant_message(x["chosen"])
        rejected = assistant_message(x["rejected"])
        if prompt and chosen and rejected and chosen != rejected:
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return rows


def load_reddit_tldr() -> list[dict[str, str]]:
    ds = load_dataset("openai/summarize_from_feedback", "comparisons", split="train")
    rows = []
    for x in ds:
        info = x.get("info", {})
        prompt = str(info.get("post", info.get("article", info.get("title", ""))))
        summaries = x.get("summaries", [])
        choice = int(x.get("choice", 0))
        if len(summaries) < 2 or choice not in (0, 1):
            continue
        texts = [messages_to_text(s) for s in summaries[:2]]
        if prompt and texts[0] and texts[1] and texts[0] != texts[1]:
            rows.append(
                {
                    "prompt": "Summarize the following Reddit post:\n" + prompt,
                    "chosen": texts[choice],
                    "rejected": texts[1 - choice],
                }
            )
    return rows


def load_helpsteer2() -> list[dict[str, str]]:
    ds = load_dataset("nvidia/HelpSteer2", data_dir="preference", split="train")
    rows = []
    for x in ds:
        strength = int(x["preference_strength"])
        if strength == 0:
            continue
        response_1, response_2 = str(x["response_1"]), str(x["response_2"])
        chosen, rejected = (response_1, response_2) if strength < 0 else (response_2, response_1)
        rows.append({"prompt": str(x["prompt"]), "chosen": chosen, "rejected": rejected})
    return rows


def load_hh_rlhf() -> list[dict[str, str]]:
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    rows = []
    for x in ds:
        chosen, rejected = str(x["chosen"]), str(x["rejected"])
        marker = "\n\nAssistant:"
        prompt = chosen.rsplit(marker, 1)[0] + marker if marker in chosen else "Conversation:\n"
        chosen_answer = chosen[len(prompt):] if chosen.startswith(prompt) else chosen
        rejected_answer = rejected[len(prompt):] if rejected.startswith(prompt) else rejected
        if chosen_answer and rejected_answer and chosen_answer != rejected_answer:
            rows.append(
                {"prompt": prompt, "chosen": chosen_answer, "rejected": rejected_answer}
            )
    return rows


LOADERS = {
    "ultrafeedback": load_ultrafeedback,
    "reddit_tldr": load_reddit_tldr,
    "helpsteer2": load_helpsteer2,
    "hh_rlhf": load_hh_rlhf,
}


def prepare_pairs(config: StudyConfig) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    rows = LOADERS[config.dataset]()
    rng = random.Random(config.seed)
    rng.shuffle(rows)
    required = config.train_pairs + config.eval_pairs
    if len(rows) < required:
        raise ValueError(f"{config.dataset}: only {len(rows)} valid pairs; need {required}")
    train = [dict(x) for x in rows[: config.train_pairs]]
    test = [dict(x) for x in rows[config.train_pairs : required]]
    flip_rng = random.Random(config.seed + 1_000_003)
    flips = 0
    for row in train:
        if flip_rng.random() < config.noise:
            row["chosen"], row["rejected"] = row["rejected"], row["chosen"]
            flips += 1
    canonical = "\n".join(
        json.dumps(x, sort_keys=True, ensure_ascii=False) for x in (train + test)
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    print(f"pairs train={len(train)} eval={len(test)} flips={flips} sha256={digest}")
    return train, test, digest


train_pairs, eval_pairs, data_hash = prepare_pairs(cfg)

# %%
def load_policy(config: StudyConfig):
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    try:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


model, tokenizer = load_policy(cfg)


def encode_response(prompt: str, response: str, max_length: int) -> dict[str, torch.Tensor]:
    try:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prefix = prompt.rstrip() + "\n\nAssistant: "
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]
    response_ids = response_ids[: max(1, max_length - 1)]
    prefix_budget = max_length - len(response_ids)
    prefix_ids = prefix_ids[-max(1, prefix_budget) :]
    input_ids = prefix_ids + response_ids
    labels = [-100] * len(prefix_ids) + response_ids
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_pairs(rows: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    encoded = []
    for row in rows:
        encoded.append(encode_response(row["prompt"], row["chosen"], cfg.max_length))
        encoded.append(encode_response(row["prompt"], row["rejected"], cfg.max_length))
    max_len = max(len(x["input_ids"]) for x in encoded)
    ids, labels, masks = [], [], []
    for x in encoded:
        pad = max_len - len(x["input_ids"])
        ids.append(F.pad(x["input_ids"], (0, pad), value=tokenizer.pad_token_id))
        labels.append(F.pad(x["labels"], (0, pad), value=-100))
        masks.append(torch.cat([torch.ones(len(x["input_ids"])), torch.zeros(pad)]))
    return {
        "input_ids": torch.stack(ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(masks).long(),
    }


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = shifted_logits.log_softmax(-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def forward_logps(batch: dict[str, torch.Tensor], reference: bool = False) -> torch.Tensor:
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in batch.items()}
    context = model.disable_adapter() if reference else nullcontext()
    with context:
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return sequence_logps(out.logits, inputs["labels"])


def preference_loss(batch: dict[str, torch.Tensor], kind: str) -> tuple[torch.Tensor, dict[str, float]]:
    policy = forward_logps(batch, reference=False)
    with torch.no_grad():
        reference = forward_logps(batch, reference=True)
    policy_c, policy_r = policy[0::2], policy[1::2]
    ref_c, ref_r = reference[0::2], reference[1::2]
    score_c, score_r = policy_c - ref_c, policy_r - ref_r
    if kind == "pairwise":
        loss = -F.logsigmoid(cfg.beta * ((score_c - score_r))).mean()
    elif kind == "pointwise":
        loss = 0.5 * (
            F.softplus(-cfg.beta * score_c).mean()
            + F.softplus(cfg.beta * score_r).mean()
        )
    else:
        raise ValueError(kind)
    return loss, {
        "score_chosen": float(score_c.mean().detach()),
        "score_rejected": float(score_r.mean().detach()),
    }


def validate_trainable_gradients() -> dict[str, float]:
    """Fail fast before a long run if checkpointing detached the LoRA graph."""
    model.train()
    model.zero_grad(set_to_none=True)
    loss, _ = preference_loss(collate_pairs([train_pairs[0]]), cfg.loss)
    loss.backward()
    grads = [p.grad.detach().float() for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads:
        raise RuntimeError("No LoRA gradients were produced; refusing to start the run")
    grad_norm = float(torch.sqrt(sum(g.square().sum() for g in grads)).cpu())
    finite = bool(math.isfinite(grad_norm) and grad_norm > 0)
    model.zero_grad(set_to_none=True)
    if not finite:
        raise RuntimeError(f"Invalid LoRA gradient norm: {grad_norm}")
    check = {"loss": float(loss.detach().cpu()), "grad_norm": grad_norm}
    print("Gradient preflight:", check)
    return check


GRADIENT_PREFLIGHT = validate_trainable_gradients()

# %%
def deterministic_training_batches(config: StudyConfig, start_micro_step: int):
    """Yield a restart-stable stream; checkpoints occur only at optimizer boundaries."""
    n = len(train_pairs)
    micro_step = start_micro_step
    cached_epoch = None
    order: list[int] = []
    while True:
        rows = []
        for batch_offset in range(config.batch_size):
            sample_number = micro_step * config.batch_size + batch_offset
            epoch, position = divmod(sample_number, n)
            if epoch != cached_epoch:
                order = list(range(n))
                random.Random(config.seed + epoch).shuffle(order)
                cached_epoch = epoch
            rows.append(train_pairs[order[position]])
        yield collate_pairs(rows)
        micro_step += 1


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer) -> None:
    device = next(model.parameters()).device
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_training_checkpoint(
    optimizer: torch.optim.Optimizer,
    steps: int,
    examples: int,
    losses: list[float],
    elapsed_seconds: float,
) -> None:
    checkpoint_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_adapter_dir)
    state = {
        "optimizer_steps": steps,
        "examples_seen": examples,
        "losses": losses,
        "elapsed_seconds": elapsed_seconds,
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(checkpoint_path)
    print(f"checkpoint step={steps} -> {checkpoint_path}")


def restore_training_checkpoint(optimizer: torch.optim.Optimizer) -> dict[str, Any] | None:
    adapter_file = checkpoint_adapter_dir / "adapter_model.safetensors"
    if not (checkpoint_path.exists() and adapter_file.exists()):
        return None
    device = next(model.parameters()).device
    adapter_state = load_safetensors(str(adapter_file), device=str(device))
    set_peft_model_state_dict(model, adapter_state)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    move_optimizer_state_to_device(optimizer)
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    print(f"Resumed from step={state['optimizer_steps']} at {checkpoint_path}")
    return state


def train_one(config: StudyConfig) -> dict[str, Any]:
    set_seed(config.seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    steps = 0
    examples = 0
    losses = []
    prior_elapsed = 0.0
    restored = restore_training_checkpoint(optimizer)
    if restored is not None:
        steps = int(restored["optimizer_steps"])
        examples = int(restored["examples_seen"])
        losses = list(restored["losses"])
        prior_elapsed = float(restored.get("elapsed_seconds", 0.0))
    micro_steps = steps * config.grad_accum
    batches = deterministic_training_batches(config, micro_steps)
    while steps < config.max_steps:
        loss = None
        for _ in range(config.grad_accum):
            loss, _ = preference_loss(next(batches), config.loss)
            (loss / config.grad_accum).backward()
            examples += config.batch_size
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
        losses.append(float(loss.detach()))
        elapsed = prior_elapsed + time.perf_counter() - started
        seconds_per_step = elapsed / max(steps, 1)
        eta_seconds = seconds_per_step * (config.max_steps - steps)
        if steps % config.log_every == 0:
            window = losses[-min(25, len(losses)) :]
            print(
                f"step={steps}/{config.max_steps} loss={np.mean(window):.6f} "
                f"elapsed_min={elapsed / 60:.1f} eta_min={eta_seconds / 60:.1f}"
            )
        if steps % config.checkpoint_every == 0 or steps == config.max_steps:
            save_training_checkpoint(optimizer, steps, examples, losses, elapsed)
    elapsed = prior_elapsed + time.perf_counter() - started
    adapter_dir = ADAPTER_DIR / run_key
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return {
        "train_loss_last_25": float(np.mean(losses[-25:])),
        "train_seconds": elapsed,
        "optimizer_steps": steps,
        "examples_seen": examples,
        "examples_per_second": examples / elapsed,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "adapter_path": str(adapter_dir),
        "checkpoint_path": str(checkpoint_path),
        "execution_profile": EXECUTION_PROFILE,
    }


train_metrics = train_one(cfg)
print(train_metrics)

# %%
def calibration_error(prob: np.ndarray, target: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(prob)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (prob >= left) & (prob < right if right < 1 else prob <= right)
        if mask.any():
            ece += mask.mean() * abs(prob[mask].mean() - target[mask].mean())
    return float(ece)


@torch.no_grad()
def evaluate_pairs(rows: list[dict[str, str]]) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    records = []
    loader = DataLoader(rows, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_pairs)
    for batch in loader:
        policy = forward_logps(batch, reference=False)
        reference = forward_logps(batch, reference=True)
        pc, pr = policy[0::2], policy[1::2]
        rc, rr = reference[0::2], reference[1::2]
        a, b = pc - rc, pr - rr
        for ai, bi, pci, pri, rci, rri in zip(a, b, pc, pr, rc, rr):
            d = float((ai - bi).cpu())
            c = float(((ai + bi) / 2).cpu())
            records.append(
                {
                    "a": float(ai.cpu()), "b": float(bi.cpu()), "c": c, "d": d,
                    "policy_chosen_logp": float(pci.cpu()), "policy_rejected_logp": float(pri.cpu()),
                    "sft_chosen_logp": float(rci.cpu()), "sft_rejected_logp": float(rri.cpu()),
                }
            )
    frame = pd.DataFrame(records)
    prob = 1.0 / (1.0 + np.exp(-cfg.beta * frame["d"].to_numpy()))
    target = np.ones(len(prob), dtype=np.float64)
    metrics = {
        "likelihood_ranking_accuracy": float((frame["d"] > 0).mean()),
        "brier": float(brier_score_loss(target, prob)),
        "ece_15": calibration_error(prob, target, bins=15),
        "c_mean": float(frame["c"].mean()),
        "c_abs_mean": float(frame["c"].abs().mean()),
        "c_sd": float(frame["c"].std(ddof=1)),
        "d_mean": float(frame["d"].mean()),
        "d_abs_mean": float(frame["d"].abs().mean()),
        "d_sd": float(frame["d"].std(ddof=1)),
        "n_eval_pairs": int(len(frame)),
    }
    return metrics, frame


eval_metrics, row_metrics = evaluate_pairs(eval_pairs)
row_path = RUN_DIR / f"{run_key}_rows.parquet"
row_metrics.to_parquet(row_path, index=False)
print(eval_metrics)

# %%
def trainable_parameters() -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def flatten_tensors(tensors: Iterable[torch.Tensor | None], params: list[torch.nn.Parameter]) -> torch.Tensor:
    chunks = []
    for grad, param in zip(tensors, params):
        chunks.append(torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1))
    return torch.cat(chunks)


def curvature_estimates(rows: list[dict[str, str]]) -> dict[str, float]:
    model.train()
    params = trainable_parameters()
    sample = rows[: cfg.curvature_batches]
    batches = [collate_pairs([x]) for x in sample]
    grad_rows = []
    for batch in batches:
        model.zero_grad(set_to_none=True)
        loss, _ = preference_loss(batch, cfg.loss)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        grad_rows.append(flatten_tensors(grads, params).float().cpu())
    G = torch.stack(grad_rows)
    singular = torch.linalg.svdvals(G / math.sqrt(len(G)))
    fisher_top = float(singular[0].square())
    fisher_trace = float(G.square().sum() / len(G))
    del G, grad_rows, singular

    vector = [torch.randn_like(p) for p in params]
    norm = torch.sqrt(sum((v.float().square().sum() for v in vector))).clamp_min(1e-12)
    vector = [v / norm.to(v.dtype) for v in vector]
    eigenvalue = torch.tensor(float("nan"), device=params[0].device)
    batch = batches[0]
    for _ in range(cfg.hessian_power_iterations):
        model.zero_grad(set_to_none=True)
        loss, _ = preference_loss(batch, cfg.loss)
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        dot = sum(
            (g * v).sum() for g, v in zip(grads, vector) if g is not None
        )
        hv = torch.autograd.grad(dot, params, allow_unused=True)
        flat_hv = flatten_tensors(hv, params)
        hv_norm = flat_hv.float().norm().clamp_min(1e-12)
        eigenvalue = sum(
            (v * (h if h is not None else torch.zeros_like(p))).sum()
            for v, h, p in zip(vector, hv, params)
        )
        vector = [
            (h if h is not None else torch.zeros_like(p)) / hv_norm.to(p.dtype)
            for h, p in zip(hv, params)
        ]
    model.eval()
    return {
        "empirical_fisher_top_eigenvalue": fisher_top,
        "empirical_fisher_trace": fisher_trace,
        "hessian_top_eigenvalue_power": float(eigenvalue.detach().float().cpu()),
        "curvature_batches": len(batches),
    }


curvature_metrics = curvature_estimates(eval_pairs)
print(curvature_metrics)

# %%
@torch.no_grad()
def categorical_kl(rows: list[dict[str, str]], limit: int = 24) -> float:
    model.eval()
    values = []
    for row in rows[:limit]:
        batch = collate_pairs([row])
        device = next(model.parameters()).device
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        policy_logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        with model.disable_adapter():
            ref_logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        valid = labels[:, 1:].ne(-100)
        p_log = policy_logits.log_softmax(-1)
        q_log = ref_logits.log_softmax(-1)
        p = p_log.exp()
        token_kl = (p * (p_log - q_log)).sum(-1)
        values.extend(token_kl[valid].detach().cpu().tolist())
    return float(np.mean(values))


kl_metric = categorical_kl(eval_pairs)
print("KL(policy || SFT):", kl_metric)


def generate_texts(prompts: list[str], adapter_enabled: bool) -> tuple[list[str], list[int]]:
    model.eval()
    texts, lengths = [], []
    context = nullcontext() if adapter_enabled else model.disable_adapter()
    with context, torch.no_grad():
        for prompt in prompts:
            try:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = prompt.rstrip() + "\n\nAssistant: "
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=cfg.max_length).to(next(model.parameters()).device)
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_ids = output[0, inputs["input_ids"].shape[1]:]
            texts.append(tokenizer.decode(new_ids, skip_special_tokens=True))
            lengths.append(int(len(new_ids)))
    return texts, lengths


def judge_win_rate(prompts: list[str], policy_texts: list[str], sft_texts: list[str]) -> dict[str, float]:
    judge_id = "OpenAssistant/reward-model-deberta-v3-large-v2"
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_id)
    judge = AutoModelForSequenceClassification.from_pretrained(judge_id).to("cuda").eval()
    policy_scores, sft_scores = [], []
    with torch.no_grad():
        for prompt, policy_text, sft_text in zip(prompts, policy_texts, sft_texts):
            for response, target in [(policy_text, policy_scores), (sft_text, sft_scores)]:
                encoded = judge_tokenizer(
                    prompt, response, return_tensors="pt", truncation=True, max_length=512
                ).to("cuda")
                target.append(float(judge(**encoded).logits.squeeze().float().cpu()))
    policy_arr, sft_arr = np.asarray(policy_scores), np.asarray(sft_scores)
    del judge
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "judge_model": judge_id,
        "generation_win_rate_vs_sft": float((policy_arr > sft_arr).mean()),
        "generation_tie_rate": float((policy_arr == sft_arr).mean()),
        "mean_reward_delta": float((policy_arr - sft_arr).mean()),
    }


generation_sample = eval_pairs[: cfg.generation_prompts]
generation_prompts = [x["prompt"] for x in generation_sample]
policy_texts, policy_lengths = generate_texts(generation_prompts, adapter_enabled=True)
sft_texts, sft_lengths = generate_texts(generation_prompts, adapter_enabled=False)
generation_metrics = judge_win_rate(generation_prompts, policy_texts, sft_texts)
generation_metrics.update(
    {
        "policy_response_length_tokens": float(np.mean(policy_lengths)),
        "sft_response_length_tokens": float(np.mean(sft_lengths)),
        "n_generation_prompts": len(generation_prompts),
    }
)
print(generation_metrics)

# %%
result = {
    "schema_version": 1,
    "execution_profile": EXECUTION_PROFILE,
    "run_index": RUN_INDEX,
    "run_key": run_key,
    "config": asdict(cfg),
    "system": SYSTEM_INFO,
    "data_sha256": data_hash,
    "gradient_preflight": GRADIENT_PREFLIGHT,
    "train": train_metrics,
    "evaluation": eval_metrics,
    "kl_to_sft": kl_metric,
    "curvature": curvature_metrics,
    "generation": generation_metrics,
    "row_metrics_path": str(row_path),
    "completed_utc": pd.Timestamp.utcnow().isoformat(),
}
tmp_path = result_path.with_suffix(".json.tmp")
tmp_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
tmp_path.replace(result_path)
print("Saved verified run:", result_path)

# %% [markdown]
# ## Aggregate completed runs
#
# This cell reads only completed JSON files. Missing factorial cells remain
# explicit and are never imputed.

# %%
def flatten_result(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    row = {"result_file": str(path), "run_index": obj["run_index"], **obj["config"]}
    row.update({f"train_{k}": v for k, v in obj["train"].items() if isinstance(v, (int, float))})
    row.update(obj["evaluation"])
    row["kl_to_sft"] = obj["kl_to_sft"]
    row.update(obj["curvature"])
    row.update({k: v for k, v in obj["generation"].items() if isinstance(v, (int, float))})
    return row


completed = pd.DataFrame(
    [flatten_result(p) for p in sorted((OUTPUT_ROOT / "runs" / "full").glob("[0-9][0-9][0-9]_*.json"))]
)
aggregate_path = OUTPUT_ROOT / "factorial_results.csv"
completed.to_csv(aggregate_path, index=False)
print(f"Completed {len(completed)} / {len(MANIFEST)} runs")
print("Aggregate:", aggregate_path)
display(completed.head())
