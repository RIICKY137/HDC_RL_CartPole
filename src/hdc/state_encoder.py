from __future__ import annotations

import numpy as np

from .config import HDCEncoderConfig
from .hypervector import bind, bundle, random_hypervector


class CartPoleHDCEncoder:
    """Encode CartPole observations into fixed-size hypervectors."""

    def __init__(self, config: HDCEncoderConfig) -> None:
        if len(config.bounds) != config.state_dim:
            raise ValueError("bounds length must match state_dim")

        self.config = config
        self.hdc_dim = config.hdc_dim
        self.n_bins = config.n_bins
        self.state_dim = config.state_dim
        self.bounds = config.bounds
        self._rng = np.random.default_rng(config.seed)

        self.position_hvs = [
            random_hypervector(config.hdc_dim, self._rng) for _ in range(config.state_dim)
        ]
        self.level_hvs = [
            [random_hypervector(config.hdc_dim, self._rng) for _ in range(config.n_bins)]
            for _ in range(config.state_dim)
        ]

    def discretize(self, value: float, dim_idx: int) -> int:
        low, high = self.bounds[dim_idx]
        clipped = float(np.clip(value, low, high))
        if high <= low:
            return 0
        bin_idx = int((clipped - low) / (high - low) * (self.n_bins - 1))
        return min(max(bin_idx, 0), self.n_bins - 1)

    def encode(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.state_dim:
            raise ValueError(f"expected state dim {self.state_dim}, got {state.shape[0]}")

        components = []
        for dim_idx in range(self.state_dim):
            bin_idx = self.discretize(float(state[dim_idx]), dim_idx)
            components.append(bind(self.position_hvs[dim_idx], self.level_hvs[dim_idx][bin_idx]))
        return bundle(components)

    def encode_batch(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            return self.encode(states)

        encoded = np.empty((states.shape[0], self.hdc_dim), dtype=np.float32)
        for index, state in enumerate(states):
            encoded[index] = self.encode(state)
        return encoded
