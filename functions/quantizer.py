"""Step 2b - uniform mid-rise quantizer (L = 2**n_bits levels over [-vmax, +vmax])."""
from __future__ import annotations

import numpy as np


def step_size(n_bits: int, vmax: float = 1.0) -> float:
    return 2.0 * vmax / (2 ** n_bits)


def quantize(x: np.ndarray, n_bits: int, vmax: float = 1.0):
    """Return (index, quantized value, step).  Index 0 ... 2**n_bits - 1."""
    levels = 2 ** n_bits
    delta = step_size(n_bits, vmax)
    idx = np.clip(np.floor((np.asarray(x) + vmax) / delta), 0, levels - 1).astype(np.int64)
    return idx, dequantize(idx, n_bits, vmax), delta


def dequantize(idx: np.ndarray, n_bits: int, vmax: float = 1.0) -> np.ndarray:
    """Mid-point reconstruction level of every code word."""
    return -vmax + (np.asarray(idx) + 0.5) * step_size(n_bits, vmax)
