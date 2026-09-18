from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import fetch_pairs
from .features import encode_pairs
from .losses import (
    angle_degrees,
    effective_sample_size,
    expected_calibration_error,
    pairwise_loss_grad,
    pointwise_loss_grad,
    sigmoid,
)


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def make_split(count: int, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    n_train = int(0.70 * count)
    n_validation = int(0.15 * count)
    return Split(order[:n_train], order[n_train : n_train + n_validation], order[n_train + n_validation :])


def train_pairwise(x: np.ndarray, beta: float, seed: int, steps: int = 800, batch_size: int = 128):
    rng = np.random.default_rng(seed)
    theta = np.zeros(x.shape[1], dtype=np.float64)
    first = np.zeros_like(theta)
    second = np.zeros_like(theta)
    lr, l2 = 0.03, 1e-3
    for step in range(1, steps + 1):
        idx = rng.integers(0, x.shape[0], size=min(batch_size, x.shape[0]))
        _, grad, _, _ = pairwise_loss_grad(theta, x[idx], beta, l2)
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * (grad * grad)
        first_hat = first / (1.0 - 0.9**step)
        second_hat = second / (1.0 - 0.999**step)
        theta -= lr * first_hat / (np.sqrt(second_hat) + 1e-8)
    return theta


def train_pointwise(chosen: np.ndarray, rejected: np.ndarray, beta: float, seed: int, steps: int = 800, batch_size: int = 128):
    rng = np.random.default_rng(seed)
    theta = np.zeros(chosen.shape[1], dtype=np.float64)
    first = np.zeros_like(theta)
    second = np.zeros_like(theta)
    lr, l2 = 0.03, 1e-3
    for step in range(1, steps + 1):
        idx = rng.integers(0, chosen.shape[0], size=min(batch_size, chosen.shape[0]))
        _, grad, _, _ = pointwise_loss_grad(theta, chosen[idx], rejected[idx], beta, l2)
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * (grad * grad)
        first_hat = first / (1.0 - 0.9**step)
        second_hat = second / (1.0 - 0.999**step)
        theta -= lr * first_hat / (np.sqrt(second_hat) + 1e-8)
    return theta


def metrics(theta: np.ndarray, differences: np.ndarray, beta: float) -> dict[str, float]:
    margins = differences @ theta
    probabilities = sigmoid(beta * margins)
    outcomes = np.ones_like(probabilities)
    return {
        "accuracy": float((margins > 0).mean()),
        "pair_loss": float(np.logaddexp(0.0, -beta * margins).mean()),
        "brier": float(np.mean((probabilities - outcomes) ** 2)),
        "ece": expected_calibration_error(probabilities, outcomes),
        "mean_margin": float(margins.mean()),
    }


def gradient_check(chosen: np.ndarray, rejected: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(17)
    d = min(24, chosen.shape[1])
    c = chosen[:8, :d]
    r = rejected[:8, :d]
    x = c - r
    theta = rng.normal(0, 0.1, size=d)
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    eps = 1e-6

    pair_loss, pair_grad, _, _ = pairwise_loss_grad(theta, x, beta=1.3, l2=1e-3)
    pair_plus = pairwise_loss_grad(theta + eps * direction, x, 1.3, 1e-3)[0]
    pair_minus = pairwise_loss_grad(theta - eps * direction, x, 1.3, 1e-3)[0]
    pair_fd = (pair_plus - pair_minus) / (2 * eps)
    pair_error = abs(pair_fd - float(pair_grad @ direction))

    point_loss, point_grad, _, _ = pointwise_loss_grad(theta, c, r, beta=0.7, l2=1e-3)
    point_plus = pointwise_loss_grad(theta + eps * direction, c, r, 0.7, 1e-3)[0]
    point_minus = pointwise_loss_grad(theta - eps * direction, c, r, 0.7, 1e-3)[0]
    point_fd = (point_plus - point_minus) / (2 * eps)
    point_error = abs(point_fd - float(point_grad @ direction))
    if pair_error > 1e-7 or point_error > 1e-7:
        raise AssertionError(f"Gradient check failed: {pair_error=}, {point_error=}")
    return {
        "pair_loss": pair_loss,
        "point_loss": point_loss,
        "pair_directional_error": pair_error,
        "point_directional_error": point_error,
    }


def bootstrap_accuracy(margins: np.ndarray, seed: int, replicates: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(replicates)
    correct = margins > 0
    for i in range(replicates):
        values[i] = correct[rng.integers(0, correct.size, size=correct.size)].mean()
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def write_plot(rows: list[dict], output: Path):
    width, height = 1600, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular_path = Path(r"C:\Windows\Fonts\times.ttf")
    bold_path = Path(r"C:\Windows\Fonts\timesbd.ttf")
    font = ImageFont.truetype(str(bold_path), 36) if bold_path.exists() else ImageFont.load_default(size=28)
    small = ImageFont.truetype(str(regular_path), 28) if regular_path.exists() else ImageFont.load_default(size=22)
    tiny = ImageFont.truetype(str(regular_path), 23) if regular_path.exists() else ImageFont.load_default(size=18)
    margin = 130
    draw.text((margin, 22), "Геометрия пакетного градиента на HH-RLHF", fill="black", font=font)
    betas = [row["beta"] for row in rows]
    log_betas = np.log10(betas)
    xmin, xmax = min(log_betas), max(log_betas)
    x0, x1 = margin, width - 80

    def panel(top, bottom, values, low, high, color, label):
        draw.line((x0, bottom, x1, bottom), fill="black", width=3)
        draw.line((x0, bottom, x0, top), fill="black", width=3)
        points = []
        for row, xb, value in zip(rows, log_betas, values):
            x = x0 + (xb - xmin) / (xmax - xmin) * (x1 - x0)
            y = bottom - (value - low) / (high - low) * (bottom - top)
            points.append((x, y))
            draw.text((x - 18, bottom + 8), f"{row['beta']:g}", fill="black", font=tiny)
        draw.line(points, fill=color, width=6)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
        draw.text((x0 + 20, top + 10), label, fill=color, font=tiny)
        draw.text((x0 - 65, top - 5), f"{high:g}", fill="black", font=tiny)
        draw.text((x0 - 65, bottom - 24), f"{low:g}", fill="black", font=tiny)

    angles = [row["angle_to_beta_1_deg"] for row in rows]
    ess = [row["normalized_ess"] for row in rows]
    panel(105, 405, angles, 0.0, 50.0, "#1f4e79", "Угол к градиенту при β=1, градусы")
    panel(505, 805, ess, 0.5, 1.0, "#a61c00", "Нормированный ESS")
    draw.text((width // 2 - 25, 850), "β", fill="black", font=small)
    image.save(output, dpi=(300, 300))


def run(output_dir: Path, cache_dir: Path, count: int = 2000, dimension: int = 4096, seed: int = 20260918):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = fetch_pairs(count, cache_dir, sample_seed=seed)
    chosen, rejected = encode_pairs(rows, dimension)
    differences = chosen - rejected
    split = make_split(count, seed)
    if len(set(split.train) & set(split.test)) or len(set(split.validation) & set(split.test)):
        raise AssertionError("Split leakage detected")
    checks = gradient_check(chosen[split.train], rejected[split.train])

    betas = [0.1, 0.3, 1.0, 3.0, 10.0]
    seeds = [11, 29, 47, 83, 101]
    geometry_theta = train_pairwise(differences[split.train], 1.0, seed=seed, steps=1000)
    reference_grad = pairwise_loss_grad(geometry_theta, differences[split.validation], 1.0)[1]
    geometry = []
    for beta in betas:
        _, grad, margins, weights = pairwise_loss_grad(geometry_theta, differences[split.validation], beta)
        geometry.append(
            {
                "beta": beta,
                "angle_to_beta_1_deg": angle_degrees(grad, reference_grad),
                "gradient_norm": float(np.linalg.norm(grad)),
                "normalized_ess": effective_sample_size(weights) / weights.size,
                "saturated_weight_share": float(((weights < 0.05) | (weights > 0.95)).mean()),
                "mean_margin": float(margins.mean()),
            }
        )

    results = []
    for beta in betas:
        for run_seed in seeds:
            pair_theta = train_pairwise(differences[split.train], beta, run_seed)
            point_theta = train_pointwise(chosen[split.train], rejected[split.train], beta, run_seed)
            for method, theta in (("pairwise", pair_theta), ("pointwise", point_theta)):
                row = {"method": method, "beta": beta, "seed": run_seed}
                row.update(metrics(theta, differences[split.test], beta))
                lo, hi = bootstrap_accuracy(differences[split.test] @ theta, run_seed)
                row["accuracy_ci_low"] = lo
                row["accuracy_ci_high"] = hi
                results.append(row)

    summary = []
    for method in ("pairwise", "pointwise"):
        for beta in betas:
            subset = [r for r in results if r["method"] == method and r["beta"] == beta]
            summary.append(
                {
                    "method": method,
                    "beta": beta,
                    "accuracy_mean": float(np.mean([r["accuracy"] for r in subset])),
                    "accuracy_sd": float(np.std([r["accuracy"] for r in subset], ddof=1)),
                    "pair_loss_mean": float(np.mean([r["pair_loss"] for r in subset])),
                    "brier_mean": float(np.mean([r["brier"] for r in subset])),
                    "ece_mean": float(np.mean([r["ece"] for r in subset])),
                    "margin_mean": float(np.mean([r["mean_margin"] for r in subset])),
                }
            )

    best_pair = max((r for r in summary if r["method"] == "pairwise"), key=lambda r: r["accuracy_mean"])
    best_point = max((r for r in summary if r["method"] == "pointwise"), key=lambda r: r["accuracy_mean"])
    report = {
        "protocol": {"count": count, "dimension": dimension, "seed": seed, "train": len(split.train), "validation": len(split.validation), "test": len(split.test), "training_seeds": seeds, "betas": betas},
        "data_manifest": manifest,
        "gradient_checks": checks,
        "geometry": geometry,
        "runs": results,
        "summary": summary,
        "best_pairwise": best_pair,
        "best_pointwise": best_point,
    }
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["results_sha256"] = hashlib.sha256(json_path.read_bytes()).hexdigest()
    (output_dir / "verification.json").write_text(json.dumps(report["gradient_checks"] | {"results_sha256": report["results_sha256"]}, indent=2), encoding="utf-8")
    for name, data in (("geometry.csv", geometry), ("summary.csv", summary), ("runs.csv", results)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    write_plot(geometry, output_dir / "gradient_geometry.png")
    return report
