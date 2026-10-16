"""Transmit side helpers: source -> PCM -> packet -> line code -> modulated burst."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .framing import assemble_packet
from .line_coding import line_encode
from .modulation import modulate
from .pcm import pcm_encode
from .quantizer import quantize
from .source import analog_message, sample


@dataclass
class TxChain:
    """All intermediate signals of the transmitter (kept for plotting)."""
    t_fine: np.ndarray
    x_fine: np.ndarray
    t_s: np.ndarray
    x_s: np.ndarray
    q_idx: np.ndarray
    x_q: np.ndarray
    delta: float
    pcm_bits: np.ndarray
    packet_bits: np.ndarray
    chips: np.ndarray
    burst: np.ndarray
    burst_q: np.ndarray       # quadrature twin (used by the channel for phase rotation)
    command: int
    hop: int
    counter: int


def build_pcm(cfg: Config):
    """Steps 1-3 of the guide: source -> sampling -> quantization -> PCM bits."""
    t_fine, x_fine = analog_message(cfg)
    t_s, x_s = sample(x_fine, cfg)
    q_idx, x_q, delta = quantize(x_s, cfg.n_bits_pcm, cfg.amplitude)
    return t_fine, x_fine, t_s, x_s, q_idx, x_q, delta, pcm_encode(q_idx, cfg.n_bits_pcm)


def transmit(cfg: Config, fob, command: int, line: str = "manchester", mod: str = "ook") -> TxChain:
    """One button press: run the complete transmitter."""
    t_fine, x_fine, t_s, x_s, q_idx, x_q, delta, pcm_bits = build_pcm(cfg)
    cmd, hop, counter = fob.press(command)
    packet = assemble_packet(cfg.preamble, cfg.sync_word, fob.device_id, cmd, hop, pcm_bits)
    chips = line_encode(packet, line)
    burst = modulate(chips, cfg.spc(line), cfg, mod)
    burst_q = modulate(chips, cfg.spc(line), cfg, mod, quadrature=True)
    return TxChain(t_fine, x_fine, t_s, x_s, q_idx, x_q, delta, pcm_bits, packet, chips,
                   burst, burst_q, cmd, hop, counter)


def record_length(cfg: Config, n_burst: int) -> int:
    return cfg.lead_max + n_burst + cfg.spc("manchester")


def place_in_record(burst: np.ndarray, leads: np.ndarray, cfg: Config) -> np.ndarray:
    """Embed the burst into an idle (silent) time record at the given start samples."""
    leads = np.atleast_1d(leads)
    rec = np.zeros((leads.size, record_length(cfg, burst.size)))
    for i, lead in enumerate(leads):
        rec[i, lead: lead + burst.size] = burst
    return rec
