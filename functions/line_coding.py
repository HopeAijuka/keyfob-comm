"""Step 4 - line coding.  A *chip* is one line-code symbol (0 or 1).

NRZ-L      : 1 chip per bit, chip = bit.
Manchester : 2 chips per bit, 1 -> (1, 0)   0 -> (0, 1)   (a transition in every bit).
"""
from __future__ import annotations

import numpy as np


def nrz_encode(bits: np.ndarray) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint8).copy()


def manchester_encode(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    chips = np.empty(bits.shape[:-1] + (2 * bits.shape[-1],), dtype=np.uint8)
    chips[..., 0::2] = bits
    chips[..., 1::2] = 1 - bits
    return chips


def line_encode(bits: np.ndarray, scheme: str) -> np.ndarray:
    if scheme == "nrz":
        return nrz_encode(bits)
    if scheme == "manchester":
        return manchester_encode(bits)
    raise ValueError(f"unknown line code {scheme!r}")


def nrz_decode(stats: np.ndarray, threshold: float) -> np.ndarray:
    return (stats > threshold).astype(np.uint8)


def manchester_decode(stats: np.ndarray) -> np.ndarray:
    """Compare the two halves of every bit - no threshold needed."""
    return (stats[..., 0::2] > stats[..., 1::2]).astype(np.uint8)


def chips_to_levels(chips: np.ndarray, spc: int) -> np.ndarray:
    """Rectangular baseband waveform of a chip sequence (for plotting)."""
    return np.repeat(np.asarray(chips, dtype=float), spc, axis=-1)
