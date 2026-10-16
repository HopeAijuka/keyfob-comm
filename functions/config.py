"""Global configuration of the simulated key-fob link.

Every number used anywhere in the project lives here, so that the whole
experiment can be re-run with different parameters by editing a single place.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ source / PCM
    fs_audio: float = 8000.0        # sampling rate of the message (guide: Fs = 8000 Hz)
    fm: float = 300.0               # message frequency (guide: fm = 300 Hz)
    duration: float = 0.05          # message length in seconds (15 full periods)
    amplitude: float = 1.0          # message amplitude (full scale of the quantizer)
    n_bits_pcm: int = 8             # bits per PCM sample (guide: n_bits = 8)
    analog_oversample: int = 8      # "analog" reference grid = 8 x Fs = 64 kHz (for plots)

    # ------------------------------------------------------------------ key-fob packet
    bit_rate: float = 9600.0        # on-air bit rate Rb [bit/s]
    preamble: int = 0xAAAA          # 16-bit preamble (1010...), clock / AGC training
    sync_word: int = 0x2DD4         # 16-bit sync word (start-of-frame marker)
    sync_tol: int = 2               # bit errors tolerated in the sync word
    device_id: int = 0xA5C39E17     # 32-bit serial number of the fob
    other_device_id: int = 0x5E71B2C4   # a *different* fob (used in the security demo)
    fob_key: int = 0x6B1D4F9AC27E3085   # 64-bit secret shared by fob and car (simulation only)
    rc_window: int = 16             # rolling-code look-ahead window in the car

    # ------------------------------------------------------------------ simulated RF
    fs_rf: float = 307_200.0        # simulation sample rate of the pass-band signal [Hz]
    fc: float = 38_400.0            # *scaled* carrier [Hz] (real fobs: 315 / 433.92 MHz)
    lead_min: int = 128             # random idle time before the burst [samples]
    lead_max: int = 256

    # ------------------------------------------------------------------ link budget (illustrative)
    f_rf_real: float = 433.92e6     # real carrier used to compute the path loss [Hz]
    tx_power_dbm: float = -15.0     # EIRP of a small loop antenna inside a pocket-sized fob
    path_loss_exp: float = 3.5      # obstructed urban / parking-lot environment
    extra_loss_db: float = 15.0     # body + car-body shadowing
    noise_figure_db: float = 10.0   # cheap receiver front-end

    seed: int = 2026

    # ------------------------------------------------------------------ derived values
    @property
    def n_samples(self) -> int:
        return int(round(self.duration * self.fs_audio))

    @property
    def n_payload_bits(self) -> int:
        return self.n_samples * self.n_bits_pcm

    @staticmethod
    def chips_per_bit(line: str) -> int:
        return {"nrz": 1, "manchester": 2}[line]

    def spc(self, line: str = "manchester") -> int:
        """Samples per chip (a chip is one line-code symbol)."""
        chip_rate = self.bit_rate * self.chips_per_bit(line)
        spc = self.fs_rf / chip_rate
        if abs(spc - round(spc)) > 1e-9:
            raise ValueError("fs_rf must be an integer multiple of the chip rate")
        return int(round(spc))

    @property
    def bandwidth_hz(self) -> float:
        """Null-to-null RF bandwidth of Manchester + OOK = 4 Rb."""
        return 4.0 * self.bit_rate

    @property
    def sim_bandwidth_hz(self) -> float:
        """Noise bandwidth of the simulation (0 ... fs_rf/2), like MATLAB's awgn()."""
        return self.fs_rf / 2.0
