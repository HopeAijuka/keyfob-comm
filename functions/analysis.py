"""Monte-Carlo experiments: SNR sweep, BPSK baseline, distance sweep, security demo."""
from __future__ import annotations

import numpy as np

from . import channel as ch
from .config import Config
from .demodulation import coherent_chip_stats
from .framing import assemble_packet
from .line_coding import line_encode
from .link import over_the_air, run_link
from .metrics import mse
from .modulation import modulate
from .pcm import pcm_decode
from .quantizer import dequantize
from .receiver import receive
from .rolling_code import (CMD_LOCK, CMD_UNLOCK, KeyFob, VehicleReceiver)
from .transmitter import TxChain


def decode_payload(payload_bits: np.ndarray, cfg: Config) -> np.ndarray:
    """PCM decoder + de-quantizer: payload bits -> reconstructed samples."""
    return dequantize(pcm_decode(payload_bits, cfg.n_bits_pcm), cfg.n_bits_pcm, cfg.amplitude)


# ------------------------------------------------------------------------------ SNR sweep
def snr_sweep(cfg: Config, tx: TxChain, snr_list, n_packets: int, rng, chunk: int = 50,
              fading: str = "awgn", verbose: bool = True) -> dict:
    p_x = np.mean(tx.x_s ** 2)
    keys = ["ber_est", "ber_genie", "sync_ok", "crc_ok", "psr", "mse_raw", "mse_delivered",
            "mse_accepted", "undetected", "timing_ok"]
    out = {k: [] for k in keys}
    out["snr_db"] = list(snr_list)
    for snr in snr_list:
        acc = dict(err_est=0, err_gen=0, bits=0, sync=0, crc=0, psr=0, mse_raw=0.0, mse_del=0.0,
                   mse_acc=0.0, n_acc=0, undet=0, timing=0, n=0)
        done = 0
        while done < n_packets:
            n = min(chunk, n_packets - done)
            _, leads, _, res, gen = run_link(tx.burst, tx.burst_q, cfg, snr, rng, n, fading)
            m = mse(tx.x_s, decode_payload(res.fields["payload"], cfg))        # per packet
            acc["err_est"] += (res.bits != tx.packet_bits).sum()
            acc["err_gen"] += (gen.bits != tx.packet_bits).sum()
            acc["bits"] += res.bits.size
            acc["sync"] += res.sync_ok.sum()
            acc["crc"] += res.crc_ok.sum()
            acc["psr"] += res.accepted.sum()
            acc["mse_raw"] += m.sum()
            acc["mse_del"] += np.where(res.accepted, m, p_x).sum()        # muted output = silence
            acc["mse_acc"] += m[res.accepted].sum()
            acc["n_acc"] += res.accepted.sum()
            acc["undet"] += (res.accepted & (res.fields["payload"] != tx.pcm_bits).any(-1)).sum()
            acc["timing"] += (np.abs(res.t0 - leads) <= 1).sum()
            done += n
        N = n_packets
        out["ber_est"].append(acc["err_est"] / acc["bits"])
        out["ber_genie"].append(acc["err_gen"] / acc["bits"])
        out["sync_ok"].append(acc["sync"] / N)
        out["crc_ok"].append(acc["crc"] / N)
        out["psr"].append(acc["psr"] / N)
        out["mse_raw"].append(acc["mse_raw"] / N)
        out["mse_delivered"].append(acc["mse_del"] / N)
        out["mse_accepted"].append(acc["mse_acc"] / acc["n_acc"] if acc["n_acc"] else np.nan)
        out["undetected"].append(acc["undet"] / N)
        out["timing_ok"].append(acc["timing"] / N)
        if verbose:
            print(f"  SNR {snr:+5.1f} dB  BER {out['ber_genie'][-1]:.2e}  PSR {out['psr'][-1]:.3f}  "
                  f"MSE_raw {out['mse_raw'][-1]:.3e}  MSE_deliv {out['mse_delivered'][-1]:.3e}", flush=True)
    out = {k: np.array(v) for k, v in out.items()}
    out["n_packets"] = n_packets
    out["bits_per_point"] = int(n_packets * tx.packet_bits.size)
    return out


# ------------------------------------------------------------------------------ BPSK baseline
def bpsk_baseline(cfg: Config, snr_list, n_packets: int, rng, chunk: int = 50) -> dict:
    """Textbook reference: NRZ-L + coherent BPSK, genie timing and phase."""
    nb = 3320
    spc = cfg.spc("nrz")
    errs, total = [], 0
    for snr in snr_list:
        e = 0
        done = 0
        while done < n_packets:
            n = min(chunk, n_packets - done)
            bits = rng.integers(0, 2, size=(n, nb)).astype(np.uint8)
            tx = modulate(line_encode(bits, "nrz"), spc, cfg, "bpsk")
            rx = ch.awgn(tx, snr, np.mean(tx ** 2, axis=-1), rng)
            dec = (coherent_chip_stats(rx, cfg, spc, nb) > 0).astype(np.uint8)
            e += (dec != bits).sum()
            done += n
        errs.append(e / (n_packets * nb))
    return {"snr_db": np.array(list(snr_list)), "ber": np.array(errs), "bits_per_point": n_packets * nb}


# ------------------------------------------------------------------------------ distance sweep
def distance_sweep(cfg: Config, tx: TxChain, distances, kinds, n_packets: int, rng,
                   chunk: int = 50, k_db: float = 6.0, verbose: bool = True) -> dict:
    out = {"distance_m": np.array(distances, dtype=float)}
    snr_b = ch.snr_inband_db(out["distance_m"], cfg)
    snr_sim = ch.inband_to_sim_snr_db(snr_b, cfg)
    out["snr_inband_db"], out["snr_sim_db"] = snr_b, snr_sim
    for kind in kinds:
        psr, ber = [], []
        for d, s in zip(distances, snr_sim):
            ok, e, done = 0, 0, 0
            while done < n_packets:
                n = min(chunk, n_packets - done)
                h = ch.rician_gain(n, k_db, rng) if kind == "rician" else None
                rx, leads, h, = over_the_air(tx.burst, tx.burst_q, cfg, s, rng, n, kind, h=h)
                res, gen = receive(rx, cfg, t0_true=leads)
                ok += res.accepted.sum()
                e += (gen.bits != tx.packet_bits).sum()
                done += n
            psr.append(ok / n_packets)
            ber.append(e / (n_packets * tx.packet_bits.size))
            if verbose:
                print(f"  {kind:8s} d={d:5.1f} m  SNR_B {ch.snr_inband_db(d, cfg):6.1f} dB  PSR {psr[-1]:.3f}", flush=True)
        out[f"psr_{kind}"] = np.array(psr)
        out[f"ber_{kind}"] = np.array(ber)
    return out


# ------------------------------------------------------------------------------ security demo
def security_demo(cfg: Config, tx_template: TxChain, rng, good_snr: float = 8.0, bad_snr: float = -6.0):
    """Run the rolling-code story through the *complete* waveform chain (modulation, channel,
    demodulation).  Returns a list of event dictionaries."""
    fob = KeyFob(cfg.device_id, cfg.fob_key)
    other = KeyFob(cfg.other_device_id, 0x1122334455667788)
    car = VehicleReceiver(cfg.rc_window)
    car.learn(cfg.device_id, cfg.fob_key, 0)
    events = []
    spc = cfg.spc("manchester")

    def to_burst(packet_bits):
        chips = line_encode(packet_bits, "manchester")
        return (modulate(chips, spc, cfg, "ook"), modulate(chips, spc, cfg, "ook", quadrature=True))

    def deliver(rx, label, who, note, counter=None):
        res, _ = receive(rx, cfg)
        link = bool(res.accepted[0])
        if link:
            dec = car.process(int(res.fields["device_id"][0]), int(res.fields["command"][0]),
                              int(res.fields["hop"][0]))
            accepted, reason = dec.accepted, dec.reason
            if dec.counter is not None:
                counter = dec.counter
        else:
            accepted = False
            if res.sync_ok[0] and res.crc_ok[0] and not res.id_ok[0]:
                reason = "ignored: device ID not paired with this car"
            else:
                why = [name for name, ok in (("sync word", res.sync_ok[0]), ("CRC", res.crc_ok[0])) if not ok]
                reason = "dropped at link layer: " + " + ".join(why) + " check failed"
        events.append(dict(label=label, who=who, note=note, link_ok=link, accepted=accepted,
                           reason=reason, counter=counter, car_counter=car.last_counter(cfg.device_id),
                           cmd=int(res.fields["command"][0]) if link else None))
        return res

    def press(f, command, snr, label, who, note):
        t = _transmit_with(cfg, tx_template, f, command)
        rx, _, _ = over_the_air(t.burst, t.burst_q, cfg, snr, rng, 1, "awgn")
        deliver(rx, label, who, note, counter=t.counter)
        return t, rx

    # 1. normal press, and the attacker records it
    t1, rx1 = press(fob, CMD_UNLOCK, good_snr, "1", "owner", "UNLOCK pressed next to the car")
    # 2. replay of that recording
    replay = ch.awgn(rx1 * 1.0, good_snr, np.mean(t1.burst ** 2), rng)
    deliver(replay, "2", "attacker", "recorded UNLOCK signal re-transmitted (replay attack)", counter=t1.counter)
    # 3. fob pressed 3 times out of range, then in range
    fob.press_out_of_range(3)
    press(fob, CMD_LOCK, good_snr, "3", "owner", "3 presses out of range, then LOCK in range")
    # 4. an unpaired fob
    press(other, CMD_UNLOCK, good_snr, "4", "stranger", "unpaired fob of another car")
    # 5. forged packet: correct ID (sent in clear) + valid CRC, but random hop code
    forged_bits = assemble_packet(cfg.preamble, cfg.sync_word, cfg.device_id, CMD_UNLOCK,
                                  int(rng.integers(0, 2 ** 32)), tx_template.pcm_bits)
    fb, fq = to_burst(forged_bits)
    rx, _, _ = over_the_air(fb, fq, cfg, good_snr, rng, 1, "awgn")
    deliver(rx, "5", "attacker", "forged packet: sniffed ID, valid CRC, guessed hop code")
    # 6. press in a very noisy environment
    press(fob, CMD_UNLOCK, bad_snr, "6", "owner", f"UNLOCK pressed at SNR = {bad_snr:.0f} dB (too noisy)")
    # 7. press again at good SNR
    press(fob, CMD_UNLOCK, good_snr, "7", "owner", "user presses again, link is good")
    # 8/9. far outside the window, then re-synchronise
    fob.press_out_of_range(20)
    press(fob, CMD_UNLOCK, good_snr, "8", "owner", "20 blind presses, then 1st press in range")
    press(fob, CMD_UNLOCK, good_snr, "9", "owner", "2nd consecutive press -> re-synchronisation")
    return events


def _transmit_with(cfg: Config, template: TxChain, fob: KeyFob, command: int) -> TxChain:
    """Like transmitter.transmit but re-using the PCM payload of `template`."""
    cmd, hop, counter = fob.press(command)
    packet = assemble_packet(cfg.preamble, cfg.sync_word, fob.device_id, cmd, hop, template.pcm_bits)
    chips = line_encode(packet, "manchester")
    spc = cfg.spc("manchester")
    return TxChain(template.t_fine, template.x_fine, template.t_s, template.x_s, template.q_idx,
                   template.x_q, template.delta, template.pcm_bits, packet, chips,
                   modulate(chips, spc, cfg, "ook"), modulate(chips, spc, cfg, "ook", quadrature=True),
                   cmd, hop, counter)
