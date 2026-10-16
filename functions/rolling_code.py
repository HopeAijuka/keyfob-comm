"""Rolling-code (hopping-code) security layer, modelled on Microchip KeeLoq.

* ``keeloq_encrypt`` / ``keeloq_decrypt`` : 32-bit block, 64-bit key, 528-round NLFSR cipher.
* ``KeyFob``          : transmitter side - increments a counter on every button press.
* ``VehicleReceiver`` : car side - decrypts the hop code, checks it and rejects replayed codes.

This is a *teaching simulation*; do not use it to protect anything real.
"""
from __future__ import annotations

from dataclasses import dataclass

KEELOQ_NLF = 0x3A5C742E
MASK32 = 0xFFFFFFFF

# command codes (must fit in 4 bits because they are protected inside the hop code)
CMD_LOCK, CMD_UNLOCK, CMD_TRUNK, CMD_PANIC = 0x1, 0x2, 0x4, 0x8
CMD_NAMES = {CMD_LOCK: "LOCK", CMD_UNLOCK: "UNLOCK", CMD_TRUNK: "TRUNK", CMD_PANIC: "PANIC"}


def _bit(x: int, n: int) -> int:
    return (x >> n) & 1


def _nlf(x: int, a: int, b: int, c: int, d: int, e: int) -> int:
    idx = _bit(x, a) + 2 * _bit(x, b) + 4 * _bit(x, c) + 8 * _bit(x, d) + 16 * _bit(x, e)
    return _bit(KEELOQ_NLF, idx)


def keeloq_encrypt(block: int, key: int) -> int:
    x = block & MASK32
    for r in range(528):
        fb = _bit(x, 0) ^ _bit(x, 16) ^ _nlf(x, 1, 9, 20, 26, 31) ^ _bit(key, r & 63)
        x = (x >> 1) | (fb << 31)
    return x


def keeloq_decrypt(block: int, key: int) -> int:
    x = block & MASK32
    for r in range(528):
        fb = _bit(x, 31) ^ _bit(x, 15) ^ _nlf(x, 0, 8, 19, 25, 30) ^ _bit(key, (15 - r) & 63)
        x = ((x << 1) & MASK32) | fb
    return x


def make_plaintext(command: int, device_id: int, counter: int) -> int:
    """[ 4-bit command | 12-bit discrimination (low bits of the serial) | 16-bit counter ]"""
    return ((command & 0xF) << 28) | ((device_id & 0xFFF) << 16) | (counter & 0xFFFF)


def split_plaintext(p: int):
    return (p >> 28) & 0xF, (p >> 16) & 0xFFF, p & 0xFFFF


class KeyFob:
    """Transmitter: every press advances the counter, so no two packets are alike."""

    def __init__(self, device_id: int, key: int, counter: int = 0):
        self.device_id, self.key, self.counter = device_id, key, counter

    def press(self, command: int):
        """Returns (command, hop_code, counter_used)."""
        self.counter = (self.counter + 1) & 0xFFFF
        hop = keeloq_encrypt(make_plaintext(command, self.device_id, self.counter), self.key)
        return command, hop, self.counter

    def press_out_of_range(self, times: int = 1):
        """Button pressed while the car could not hear the fob (counter still advances)."""
        for _ in range(times):
            self.press(CMD_UNLOCK)


@dataclass
class Decision:
    accepted: bool
    reason: str
    counter: int | None = None


class VehicleReceiver:
    """Receiver: stores the last accepted counter of every learned fob."""

    def __init__(self, window: int = 16):
        self.window = window
        self._fobs: dict[int, list[int]] = {}
        self._pending: dict[int, int] = {}     # first counter seen outside the window (re-sync)

    def learn(self, device_id: int, key: int, counter: int = 0) -> None:
        self._fobs[device_id] = [key, counter]

    def last_counter(self, device_id: int) -> int:
        return self._fobs[device_id][1]

    def process(self, device_id: int, command: int, hop: int) -> Decision:
        if device_id not in self._fobs:
            return Decision(False, "unknown device ID")
        key, last = self._fobs[device_id]
        cmd_h, disc, cnt = split_plaintext(keeloq_decrypt(hop, key))
        if disc != (device_id & 0xFFF) or cmd_h != (command & 0xF):
            return Decision(False, "hop code failed authentication (forged / wrong key)")
        diff = (cnt - last) & 0xFFFF
        if diff == 0 or diff >= 0x8000:
            return Decision(False, "replay: counter already used", cnt)
        if diff > self.window:
            # KeeLoq-style re-synchronisation: accept only after *two consecutive* counters
            if self._pending.get(device_id) == (cnt - 1) & 0xFFFF:
                self._fobs[device_id][1] = cnt
                self._pending.pop(device_id)
                return Decision(True, "re-synchronised (two consecutive counters)", cnt)
            self._pending[device_id] = cnt
            return Decision(False, "counter outside look-ahead window - press again to re-sync", cnt)
        self._fobs[device_id][1] = cnt
        self._pending.pop(device_id, None)
        return Decision(True, "valid rolling code", cnt)
