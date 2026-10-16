"""Step 1 - message source and Step 2a - sampling."""
from __future__ import annotations

import numpy as np

from .config import Config


def analog_message(cfg: Config):
    """'Analog' message, represented on a fine 64 kHz grid (8 x Fs)."""
    fs_fine = cfg.fs_audio * cfg.analog_oversample
    n_fine = cfg.n_samples * cfg.analog_oversample
    t = np.arange(n_fine) / fs_fine
    return t, cfg.amplitude * np.sin(2 * np.pi * cfg.fm * t)


def sample(x_fine: np.ndarray, cfg: Config):
    """Ideal sampler: keep every `analog_oversample`-th point -> Fs = 8 kHz."""
    x = x_fine[:: cfg.analog_oversample]
    t = np.arange(x.size) / cfg.fs_audio
    return t, x
