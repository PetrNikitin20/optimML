from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def pairwise_loss_grad(theta: np.ndarray, differences: np.ndarray, beta: float, l2: float = 0.0):
    margins = differences @ theta
    weights = sigmoid(-beta * margins)
    loss = float(np.logaddexp(0.0, -beta * margins).mean() + 0.5 * l2 * np.dot(theta, theta))
    grad = -(beta / differences.shape[0]) * (differences.T @ weights) + l2 * theta
    return loss, grad, margins, weights


def pointwise_loss_grad(
    theta: np.ndarray, chosen: np.ndarray, rejected: np.ndarray, beta: float, l2: float = 0.0
):
    chosen_scores = chosen @ theta
    rejected_scores = rejected @ theta
    chosen_weights = sigmoid(-beta * chosen_scores)
    rejected_weights = sigmoid(beta * rejected_scores)
    loss_terms = np.logaddexp(0.0, -beta * chosen_scores) + np.logaddexp(0.0, beta * rejected_scores)
    loss = float(loss_terms.mean() + 0.5 * l2 * np.dot(theta, theta))
    grad = (beta / chosen.shape[0]) * (
        -(chosen.T @ chosen_weights) + rejected.T @ rejected_weights
    ) + l2 * theta
    return loss, grad, chosen_scores, rejected_scores


def effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.dot(weights, weights))
    return float(weights.sum() ** 2 / denominator) if denominator > 0 else 0.0


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def expected_calibration_error(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = probabilities.size
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lo) & (probabilities < hi if hi < 1.0 else probabilities <= hi)
        if mask.any():
            ece += mask.mean() * abs(float(probabilities[mask].mean() - outcomes[mask].mean()))
    return float(ece)
