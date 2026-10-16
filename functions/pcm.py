"""Step 3 - natural-binary PCM (MSB first) <-> bit streams."""
from __future__ import annotations

import numpy as np


def pcm_encode(idx: np.ndarray, n_bits: int) -> np.ndarray:
    """(..., N) integer code words -> (..., N*n_bits) bits, MSB first."""
    shifts = np.arange(n_bits - 1, -1, -1)
    bits = (np.asarray(idx)[..., None] >> shifts) & 1
    return bits.reshape(*bits.shape[:-2], -1).astype(np.uint8)


def pcm_decode(bits: np.ndarray, n_bits: int) -> np.ndarray:
    """(..., N*n_bits) bits -> (..., N) integer code words."""
    bits = np.asarray(bits).astype(np.int64)
    words = bits.reshape(*bits.shape[:-1], -1, n_bits)
    weights = 1 << np.arange(n_bits - 1, -1, -1)
    return (words * weights).sum(-1)
