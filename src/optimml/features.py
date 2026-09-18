from __future__ import annotations

import hashlib
import re

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9']+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _index_and_sign(feature: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8, person=b"optimML").digest()
    value = int.from_bytes(digest, "little", signed=False)
    return value % dimension, 1.0 if ((value >> 63) & 1) == 0 else -1.0


def hashed_text_vector(text: str, dimension: int = 4096) -> np.ndarray:
    words = tokens(text)
    vector = np.zeros(dimension, dtype=np.float32)
    features = [f"u:{word}" for word in words]
    features.extend(f"b:{a}_{b}" for a, b in zip(words, words[1:]))
    for feature in features:
        index, sign = _index_and_sign(feature, dimension)
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def encode_pairs(rows: list[dict[str, str]], dimension: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    chosen = np.stack([hashed_text_vector(row["chosen"], dimension) for row in rows])
    rejected = np.stack([hashed_text_vector(row["rejected"], dimension) for row in rows])
    return chosen.astype(np.float64), rejected.astype(np.float64)
