"""Packet framing:  [Preamble][Sync][Device ID][Command][Rolling code][PCM payload][CRC-16].

Field widths (bits): 16 | 16 | 32 | 8 | 32 | 3200 | 16  = 3320 bits in total.
"""
from __future__ import annotations

import numpy as np

FIELDS = (("preamble", 16), ("sync", 16), ("device_id", 32), ("command", 8),
          ("hop", 32), ("payload", None), ("crc", 16))


def frame_layout(n_payload_bits: int) -> dict:
    """{field: (start, stop)} bit positions inside the packet."""
    out, pos = {}, 0
    for name, n in FIELDS:
        n = n_payload_bits if n is None else n
        out[name] = (pos, pos + n)
        pos += n
    out["total"] = (0, pos)
    return out


def int_to_bits(value: int, n_bits: int) -> np.ndarray:
    return np.array([(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)], dtype=np.uint8)


def bits_to_int(bits: np.ndarray) -> np.ndarray:
    """(..., n) bits (MSB first, n <= 62) -> (...) int64."""
    bits = np.asarray(bits).astype(np.int64)
    weights = 1 << np.arange(bits.shape[-1] - 1, -1, -1, dtype=np.int64)
    return (bits * weights).sum(-1)


# --------------------------------------------------------------------------- CRC-16/CCITT-FALSE
def _make_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.uint32)
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
        table[byte] = crc & 0xFFFF
    return table


_CRC_TABLE = _make_table()


def crc16_ccitt(data: np.ndarray) -> np.ndarray:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over the last axis of a uint8 array.

    Works on a whole batch of packets at once.  Check value of b"123456789" is 0x29B1.
    """
    data = np.asarray(data, dtype=np.uint32)
    crc = np.full(data.shape[:-1], 0xFFFF, dtype=np.uint32)
    for i in range(data.shape[-1]):
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[((crc >> 8) ^ data[..., i]) & 0xFF]
    return crc


def _crc_input_bits(bits: np.ndarray, lay: dict) -> np.ndarray:
    return bits[..., lay["device_id"][0]: lay["payload"][1]]


def assemble_packet(preamble: int, sync_word: int, device_id: int, command: int,
                    hop: int, payload_bits: np.ndarray) -> np.ndarray:
    """Build the bit array of one packet (adds the CRC)."""
    payload_bits = np.asarray(payload_bits, dtype=np.uint8)
    lay = frame_layout(payload_bits.size)
    body = np.concatenate([int_to_bits(device_id, 32), int_to_bits(command, 8),
                           int_to_bits(hop, 32), payload_bits])
    crc = int(crc16_ccitt(np.packbits(body)))
    return np.concatenate([int_to_bits(preamble, 16), int_to_bits(sync_word, 16),
                           body, int_to_bits(crc, 16)]).astype(np.uint8)


def parse_packet(bits: np.ndarray, n_payload_bits: int, sync_word: int) -> dict:
    """Split decoded bits (..., N) into fields and compute the integrity checks."""
    bits = np.asarray(bits, dtype=np.uint8)
    lay = frame_layout(n_payload_bits)
    f = {k: bits[..., a:b] for k, (a, b) in lay.items() if k != "total"}
    crc_calc = crc16_ccitt(np.packbits(_crc_input_bits(bits, lay), axis=-1))
    crc_rx = bits_to_int(f["crc"])
    return {
        "device_id": bits_to_int(f["device_id"]),
        "command": bits_to_int(f["command"]),
        "hop": bits_to_int(f["hop"]),
        "payload": f["payload"],
        "crc_rx": crc_rx,
        "crc_calc": crc_calc.astype(np.int64),
        "crc_ok": crc_rx == crc_calc,
        "sync_errors": (f["sync"] != int_to_bits(sync_word, 16)).sum(-1),
    }
