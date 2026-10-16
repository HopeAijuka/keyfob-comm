"""Performance metrics and theory (BER, MSE, SQNR, Shannon)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc

from .config import Config


# ------------------------------------------------------------------------ signal quality
def mse(x: np.ndarray, y: np.ndarray, axis=-1):
    return np.mean((np.asarray(x) - np.asarray(y)) ** 2, axis=axis)


def snr_db(signal: np.ndarray, error_power) -> float:
    return 10 * np.log10(np.mean(np.asarray(signal) ** 2) / error_power)


def sqnr_theory_db(n_bits: int) -> float:
    """Full-scale sine:  SQNR = 6.02 n + 1.76 dB."""
    return 6.02 * n_bits + 1.76


def q_func(x):
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2.0))


# ------------------------------------------------------------------------ BER theory
def ber_ook_noncoherent(ebn0_db):
    """Non-coherent OOK with Manchester (energy comparison of the two half-bits):
    Pb = 1/2 exp(-Eb / 2N0)  - with Eb the *average* energy per bit."""
    return 0.5 * np.exp(-0.5 * 10 ** (np.asarray(ebn0_db) / 10))


def ber_bpsk_coherent(ebn0_db):
    return q_func(np.sqrt(2 * 10 ** (np.asarray(ebn0_db) / 10)))


def required_ebn0_db(ber_func, target: float) -> float:
    return brentq(lambda e: ber_func(e) - target, -5, 30)


def psr_theory(ber, cfg: Config, n_crc_bits: int | None = None):
    """Packet success probability ~ (1-p)^(bits protected by CRC) * P(sync word has <= tol errors)."""
    from scipy.stats import binom
    if n_crc_bits is None:
        n_crc_bits = 32 + 8 + 32 + cfg.n_payload_bits + 16
    ber = np.asarray(ber, dtype=float)
    return (1 - ber) ** n_crc_bits * binom.cdf(cfg.sync_tol, 16, ber)


def mse_from_ber(ber, q_err: np.ndarray, pcm_bits: np.ndarray, n_bits: int, vmax: float = 1.0):
    """Exact expected MSE when every PCM bit is flipped independently with probability p.

    q_err = x - x_q (quantization error of every sample); a flip of the bit with weight w
    changes the decoded value by +/- w*delta.  Uses E[e] and E[e^2] of the error e.
    """
    delta = 2.0 * vmax / 2 ** n_bits
    w = (2.0 ** np.arange(n_bits - 1, -1, -1)) * delta                  # (n_bits,)
    b = pcm_bits.reshape(-1, n_bits).astype(float)                      # (N, n_bits)
    step = (1 - 2 * b) * w                                              # signed jump of every bit
    s1 = step.sum(-1)
    s2 = (w ** 2).sum()
    out = []
    for p in np.atleast_1d(ber):
        e1 = p * s1
        e2 = p * s2 + p ** 2 * (s1 ** 2 - s2)
        out.append(np.mean(q_err ** 2 - 2 * q_err * e1 + e2))
    return np.array(out)


# ------------------------------------------------------------------------ Shannon
def shannon_capacity(bandwidth_hz: float, snr_lin):
    return bandwidth_hz * np.log2(1 + np.asarray(snr_lin))


def shannon_min_ebn0_db(eta):
    """Minimum Eb/N0 for reliable communication at spectral efficiency eta = Rb/B [bit/s/Hz]."""
    eta = np.asarray(eta, dtype=float)
    return 10 * np.log10((2 ** eta - 1) / eta)


def shannon_min_snr_db(rate_bps: float, bandwidth_hz: float) -> float:
    """SNR (in the band B) at which C(B, SNR) = rate."""
    return 10 * np.log10(2 ** (rate_bps / bandwidth_hz) - 1)
