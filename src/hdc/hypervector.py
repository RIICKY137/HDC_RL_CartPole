from __future__ import annotations

import numpy as np


def random_hypervector(dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice([-1.0, 1.0], size=dim).astype(np.float32)


def bind(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left * right


def bundle(vectors: list[np.ndarray] | np.ndarray) -> np.ndarray:
    stacked = np.stack(vectors, axis=0) if isinstance(vectors, list) else vectors
    summed = np.sum(stacked, axis=0)
    norm = np.linalg.norm(summed)
    if norm < 1e-8:
        return summed.astype(np.float32)
    return (summed / norm).astype(np.float32)
