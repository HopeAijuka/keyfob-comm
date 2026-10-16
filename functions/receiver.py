"""Step 7b - receiver: timing recovery, demodulation, line decoding, frame parsing, checks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .demodulation import (chip_energy, downconvert_integrate, estimate_timing,
                           gather_chip_stats)
from .framing import frame_layout, int_to_bits, parse_packet
from .line_coding import line_encode, manchester_decode


@dataclass
class RxResult:
    """Everything the receiver learned about a batch of recorded signals."""
    t0: np.ndarray            # estimated start sample of each burst
    chip_stats: np.ndarray    # energy of every chip (n, n_chips)
    bits: np.ndarray          # decoded packet bits (n, 3320)
    fields: dict              # parsed fields + integrity checks
    sync_ok: np.ndarray       # sync word matched (<= sync_tol bit errors)
    crc_ok: np.ndarray        # CRC-16 matched
    id_ok: np.ndarray         # device ID is the one the car was paired with
    accepted: np.ndarray      # link-layer success = sync_ok & crc_ok & id_ok
    metric: np.ndarray | None = None   # sync correlation (for plotting)


def header_template(cfg: Config) -> np.ndarray:
    """Chips of [preamble + sync word] - the pattern the receiver searches for."""
    bits = np.concatenate([int_to_bits(cfg.preamble, 16), int_to_bits(cfg.sync_word, 16)])
    return line_encode(bits, "manchester")


def front_end(rx: np.ndarray, cfg: Config) -> np.ndarray:
    """Down-convert and return the sliding chip-energy stream (..., n)."""
    spc = cfg.spc("manchester")
    return chip_energy(downconvert_integrate(np.atleast_2d(rx), cfg, spc))


def decode_at(energy: np.ndarray, t0: np.ndarray, cfg: Config, expected_id: int) -> RxResult:
    spc = cfg.spc("manchester")
    n_bits_total = frame_layout(cfg.n_payload_bits)["total"][1]
    stats = gather_chip_stats(energy, t0, 2 * n_bits_total, spc)
    bits = manchester_decode(stats)
    fields = parse_packet(bits, cfg.n_payload_bits, cfg.sync_word)
    sync_ok = fields["sync_errors"] <= cfg.sync_tol
    crc_ok = np.asarray(fields["crc_ok"])
    id_ok = fields["device_id"] == expected_id
    return RxResult(np.atleast_1d(t0), stats, bits, fields, sync_ok, crc_ok, id_ok,
                    sync_ok & crc_ok & id_ok)


def receive(rx: np.ndarray, cfg: Config, expected_id: int | None = None,
            t0_true: np.ndarray | None = None):
    """Full receiver.  Returns (result with estimated timing, result with genie timing or None)."""
    expected_id = cfg.device_id if expected_id is None else expected_id
    spc = cfg.spc("manchester")
    energy = front_end(rx, cfg)
    t0, metric = estimate_timing(energy, header_template(cfg), spc, cfg.lead_max + 1)
    res = decode_at(energy, t0, cfg, expected_id)
    res.metric = metric
    genie = decode_at(energy, np.asarray(t0_true), cfg, expected_id) if t0_true is not None else None
    return res, genie
