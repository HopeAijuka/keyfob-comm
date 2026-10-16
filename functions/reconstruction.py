"""Step 7c - PCM decoding is in pcm.py / quantizer.py; here: the D/A converter + reconstruction filter."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import Config


def dac_reconstruct(x_hat: np.ndarray, cfg: Config, cutoff_hz: float = 3400.0, order: int = 6):
    """Zero-order-hold D/A (x8 oversampled) followed by a zero-phase Butterworth low-pass.

    The half-sample delay of the hold is compensated so that the result can be overlaid
    on the original analog message.  Returns the signal on the 64 kHz display grid.
    """
    up = cfg.analog_oversample
    zoh = np.repeat(np.asarray(x_hat, dtype=float), up)
    zoh = np.concatenate([zoh[up // 2:], np.full(up // 2, zoh[-1])])
    sos = butter(order, cutoff_hz, btype="low", fs=cfg.fs_audio * up, output="sos")
    return sosfiltfilt(sos, zoh)
