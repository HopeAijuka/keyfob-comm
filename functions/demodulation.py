"""Step 7a - demodulation: down-conversion, chip integration and timing recovery."""
from __future__ import annotations

import numpy as np

from .config import Config


def downconvert_integrate(rx: np.ndarray, cfg: Config, spc: int) -> np.ndarray:
    """Mix with exp(-j 2 pi fc t) and run a *sliding* integrate-and-dump of length `spc`.

    y[n] = sum_{k<spc} rx[n+k] e^{-j w (n+k)}.  Because `spc` covers an integer number of
    carrier cycles the double-frequency term cancels exactly, so |y[n]|^2 is the energy
    that a non-coherent (phase-independent) receiver would measure in the chip starting
    at sample n.
    """
    n = np.arange(rx.shape[-1])
    z = rx * np.exp(-2j * np.pi * cfg.fc * n / cfg.fs_rf)
    c = np.concatenate([np.zeros(rx.shape[:-1] + (1,), dtype=complex), np.cumsum(z, axis=-1)], axis=-1)
    return c[..., spc:] - c[..., :-spc]


def chip_energy(y: np.ndarray) -> np.ndarray:
    return y.real ** 2 + y.imag ** 2


def sync_metric(energy: np.ndarray, template_chips: np.ndarray, spc: int, n_search: int) -> np.ndarray:
    """Correlate the chip-energy stream with the known preamble+sync pattern.

    The template is +1 / -1 (zero mean for Manchester), so the noise floor cancels.
    Returns an array (..., n_search): one correlation value per candidate start sample.
    """
    tpl = 2.0 * np.asarray(template_chips, dtype=float) - 1.0
    offsets = np.arange(n_search)[:, None] + np.arange(tpl.size)[None, :] * spc
    return (energy[..., offsets] * tpl).sum(-1)


def estimate_timing(energy: np.ndarray, template_chips: np.ndarray, spc: int, n_search: int):
    metric = sync_metric(energy, template_chips, spc, n_search)
    return np.argmax(metric, axis=-1), metric


def gather_chip_stats(energy: np.ndarray, t0: np.ndarray, n_chips: int, spc: int) -> np.ndarray:
    """Pick the energy of every chip once the start sample t0 of the burst is known."""
    t0 = np.atleast_1d(t0)
    idx = t0[:, None] + np.arange(n_chips)[None, :] * spc
    return np.take_along_axis(energy, idx, axis=-1)


def coherent_chip_stats(rx: np.ndarray, cfg: Config, spc: int, n_chips: int) -> np.ndarray:
    """Coherent correlator (known phase and timing, burst starting at sample 0): used for BPSK."""
    n = np.arange(n_chips * spc)
    ref = np.cos(2 * np.pi * cfg.fc * n / cfg.fs_rf)
    return (rx[..., : n_chips * spc] * ref).reshape(*rx.shape[:-1], n_chips, spc).sum(-1)
