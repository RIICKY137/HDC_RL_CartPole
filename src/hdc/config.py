from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HDCEncoderConfig:
    hdc_dim: int = 8192  # 2^13, closest power of 2 to 10_000
    n_bins: int = 10
    state_dim: int = 4
    bounds: tuple[tuple[float, float], ...] = (
        (-2.4, 2.4),
        (-3.0, 3.0),
        (-0.418, 0.418),
        (-4.0, 4.0),
    )
    seed: int = 7
