"""Glue: put bursts on the air (random start, fading, AWGN) and run the receiver."""
from __future__ import annotations

import numpy as np

from .channel import apply_flat_fading, awgn, random_phase_gain, rayleigh_gain, rician_gain
from .config import Config
from .receiver import receive
from .transmitter import place_in_record


def draw_gain(kind: str, n: int, rng: np.random.Generator, k_db: float = 6.0) -> np.ndarray:
    if kind == "awgn":
        return random_phase_gain(n, rng)
    if kind == "rayleigh":
        return rayleigh_gain(n, rng)
    if kind == "rician":
        return rician_gain(n, k_db, rng)
    raise ValueError(kind)


def over_the_air(burst: np.ndarray, burst_q: np.ndarray, cfg: Config, snr_db,
                 rng: np.random.Generator, n: int = 1, fading: str = "awgn",
                 leads: np.ndarray | None = None, h: np.ndarray | None = None):
    """Send `n` copies of `burst` through the channel (burst_q = its quadrature twin).

    snr_db is the *mean* simulation SNR  P_burst / sigma^2  (like MATLAB awgn 'measured');
    with fading the instantaneous SNR is |h|^2 times larger or smaller.
    Returns (records, leads, gains).
    """
    if leads is None:
        leads = rng.integers(cfg.lead_min, cfg.lead_max + 1, size=n)
    if h is None:
        h = draw_gain(fading, n, rng)
    p_ref = np.mean(burst ** 2)
    rec = apply_flat_fading(place_in_record(burst, leads, cfg),
                            place_in_record(burst_q, leads, cfg), h)
    return awgn(rec, snr_db, p_ref, rng), np.atleast_1d(leads), h


def run_link(burst: np.ndarray, burst_q: np.ndarray, cfg: Config, snr_db: float,
             rng: np.random.Generator, n: int = 1, fading: str = "awgn",
             expected_id: int | None = None, leads: np.ndarray | None = None):
    rx, leads, h = over_the_air(burst, burst_q, cfg, snr_db, rng, n, fading, leads)
    res, genie = receive(rx, cfg, expected_id, t0_true=leads)
    return rx, leads, h, res, genie
