"""Step 6 - channel: path loss, flat fading and AWGN.

All functions work on arrays of shape (n_packets, n_samples) (or 1-D arrays).
"""
from __future__ import annotations

import numpy as np

from .config import Config

C_LIGHT = 299_792_458.0


def awgn(x: np.ndarray, snr_db: float, ref_power, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise so that  (reference signal power) / (noise power) = SNR.

    Equivalent to MATLAB's  awgn(x, snr, 'measured')  when `ref_power` is the measured
    power of the transmitted burst.  The noise occupies 0 ... fs/2 of the simulation.
    """
    ref_power = np.asarray(ref_power, dtype=float)
    sigma = np.sqrt(ref_power / 10.0 ** (snr_db / 10.0))
    sigma = sigma[..., None] if sigma.ndim else sigma
    return x + sigma * rng.standard_normal(x.shape)


def apply_flat_fading(x_i: np.ndarray, x_q: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Flat (frequency-non-selective) fading: y = Re{ h (x_I + j x_Q) } = Re(h) x_I - Im(h) x_Q.

    x_I = env(t) cos(wt) is the transmitted burst, x_Q = env(t) sin(wt) its quadrature twin.
    |h| scales the amplitude, arg(h) rotates the carrier phase (unknown to the receiver).
    """
    h = np.asarray(h, dtype=complex)
    return h.real[..., None] * x_i - h.imag[..., None] * x_q


def random_phase_gain(n: int, rng: np.random.Generator) -> np.ndarray:
    """AWGN-only channel: unit gain with an unknown random carrier phase."""
    return np.exp(1j * rng.uniform(0, 2 * np.pi, n))


def rayleigh_gain(n: int, rng: np.random.Generator) -> np.ndarray:
    """NLOS block fading: h ~ CN(0, 1)  ->  E|h|^2 = 1."""
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)


def rician_gain(n: int, k_db: float, rng: np.random.Generator) -> np.ndarray:
    """LOS block fading with Rician factor K (dB):  E|h|^2 = 1."""
    k = 10.0 ** (k_db / 10.0)
    scatter = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)
    h = np.sqrt(k / (k + 1.0)) + np.sqrt(1.0 / (k + 1.0)) * scatter
    return h * np.exp(1j * rng.uniform(0, 2 * np.pi, n))


# ----------------------------------------------------------------------- link budget
def path_loss_db(distance_m, cfg: Config) -> np.ndarray:
    """Log-distance model: free-space loss at 1 m + 10 n log10(d) + extra shadowing loss."""
    d = np.asarray(distance_m, dtype=float)
    pl0 = 20 * np.log10(4 * np.pi * 1.0 * cfg.f_rf_real / C_LIGHT)
    return pl0 + 10 * cfg.path_loss_exp * np.log10(d) + cfg.extra_loss_db


def noise_floor_dbm(cfg: Config) -> float:
    """kTB + noise figure, measured over the RF bandwidth B = 4 Rb."""
    return -174.0 + 10 * np.log10(cfg.bandwidth_hz) + cfg.noise_figure_db


def snr_inband_db(distance_m, cfg: Config) -> np.ndarray:
    """Mean SNR in the signal bandwidth B at a given distance."""
    return cfg.tx_power_dbm - path_loss_db(distance_m, cfg) - noise_floor_dbm(cfg)


def inband_to_sim_snr_db(snr_b_db, cfg: Config):
    """Convert in-band SNR (noise in B) to the simulation SNR (noise in 0 ... fs/2)."""
    return np.asarray(snr_b_db) - 10 * np.log10(cfg.sim_bandwidth_hz / cfg.bandwidth_hz)


def sim_snr_to_ebn0_db(snr_sim_db, cfg: Config):
    """Eb/N0 = SNR_sim * fs / (2 Rb)."""
    return np.asarray(snr_sim_db) + 10 * np.log10(cfg.fs_rf / (2 * cfg.bit_rate))
