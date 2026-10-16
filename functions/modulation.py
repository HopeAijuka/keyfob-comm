"""Step 5 - digital modulation of the chip stream onto the carrier.

OOK  (On-Off Keying, the 'ASK' used by real key fobs):  s(t) = A * chip(t) * cos(2 pi fc t)
BPSK (textbook baseline from the guide)              :  s(t) = A * (2 chip(t) - 1) * cos(2 pi fc t)
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .line_coding import chips_to_levels


def carrier(n: int, cfg: Config, quadrature: bool = False) -> np.ndarray:
    """cos(2 pi fc t)  (or sin(2 pi fc t) for the quadrature branch used by the channel)."""
    arg = 2 * np.pi * cfg.fc * np.arange(n) / cfg.fs_rf
    return np.sin(arg) if quadrature else np.cos(arg)


def ook_modulate(chips: np.ndarray, spc: int, cfg: Config, amplitude: float = 1.0,
                 quadrature: bool = False) -> np.ndarray:
    env = chips_to_levels(chips, spc)
    return amplitude * env * carrier(env.shape[-1], cfg, quadrature)


def bpsk_modulate(chips: np.ndarray, spc: int, cfg: Config, amplitude: float = 1.0,
                  quadrature: bool = False) -> np.ndarray:
    env = 2.0 * chips_to_levels(chips, spc) - 1.0
    return amplitude * env * carrier(env.shape[-1], cfg, quadrature)


def modulate(chips: np.ndarray, spc: int, cfg: Config, scheme: str = "ook",
             quadrature: bool = False) -> np.ndarray:
    """In-phase burst (the real transmitted signal).  quadrature=True returns the sin-carrier
    twin  env(t) sin(2 pi fc t), which lets the channel rotate the carrier phase *exactly*
    (Re{h (x_I + j x_Q)}) without Hilbert-transform artefacts."""
    if scheme == "ook":
        return ook_modulate(chips, spc, cfg, quadrature=quadrature)
    if scheme == "bpsk":
        return bpsk_modulate(chips, spc, cfg, quadrature=quadrature)
    raise ValueError(f"unknown modulation {scheme!r}")
