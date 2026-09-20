from __future__ import annotations

import math

import numpy as np


def accuracy_interval(records: list[dict], seed: int, replicates: int = 10000) -> dict[str, float]:
    values = np.asarray([bool(row["correct"]) for row in records], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"accuracy": float(values.mean()), "bootstrap_ci_low": float(low), "bootstrap_ci_high": float(high), "n": len(values)}


def paired_change(before: list[dict], after: list[dict], seed: int, replicates: int = 10000) -> dict[str, float | int]:
    before_map = {(row["subset"], int(row["id"])): bool(row["correct"]) for row in before}
    after_map = {(row["subset"], int(row["id"])): bool(row["correct"]) for row in after}
    if before_map.keys() != after_map.keys():
        raise ValueError("Before/after records are not paired on the same examples")
    keys = sorted(before_map)
    delta = np.asarray([int(after_map[key]) - int(before_map[key]) for key in keys], dtype=np.float64)
    improved = int((delta == 1).sum())
    harmed = int((delta == -1).sum())
    discordant = improved + harmed
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(0, min(improved, harmed) + 1)) / (2**discordant)
        exact_two_sided_p = min(1.0, 2.0 * tail)
    else:
        exact_two_sided_p = 1.0
    rng = np.random.default_rng(seed)
    changes = delta[rng.integers(0, len(delta), size=(replicates, len(delta)))].mean(axis=1)
    low, high = np.quantile(changes, [0.025, 0.975])
    return {
        "n": len(delta),
        "accuracy_change": float(delta.mean()),
        "bootstrap_ci_low": float(low),
        "bootstrap_ci_high": float(high),
        "improved_examples": improved,
        "harmed_examples": harmed,
        "unchanged_examples": int((delta == 0).sum()),
        "exact_sign_test_two_sided_p": float(exact_two_sided_p),
    }
