#!/usr/bin/env python3
"""
Practical 8 mini-project  -  End-to-end communication system simulation
=======================================================================
Automotive key-fob (remote keyless entry) link carrying PCM data.

    Source -> Sampling -> Quantization/PCM -> Packet framing (+rolling code, CRC)
           -> Line coding (Manchester) -> Digital modulation (OOK)
           -> Channel (path loss, fading, AWGN) -> Demodulation (sync + non-coherent)
           -> Decoding (Manchester, frame parser, rolling-code check, PCM decoder)
           -> Reconstructed signal  (+ MSE / BER / PSR analysis, Shannon discussion)

Run:      python main.py            (full run, ~3-5 min)
          python main.py --quick    (fewer Monte-Carlo packets, ~1 min)

Outputs:  figures/*.png   results/*.csv|json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from functions import channel as ch
from functions import metrics as mt
from functions.analysis import (bpsk_baseline, decode_payload, distance_sweep, security_demo,
                                snr_sweep)
from functions.config import Config
from functions.link import over_the_air, run_link
from functions.plots_chain import (fig_block_diagram, fig_channel, fig_demod, fig_eye, fig_frame_parser,
                                   fig_line_coding, fig_modulation, fig_packet, fig_pcm,
                                   fig_quantization, fig_reconstruction, fig_source_sampling)
from functions.plots_results import (fig_ber, fig_distance, fig_mse, fig_psr, fig_rolling_flow,
                                     fig_security_timeline, fig_shannon, shannon_numbers)
from functions.receiver import front_end
from functions.reconstruction import dac_reconstruct
from functions.rolling_code import CMD_UNLOCK, KeyFob
from functions.transmitter import transmit

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
RES = ROOT / "results"


def log(msg=""):
    print(msg, flush=True)


def to_py(o):
    """Make numpy objects JSON-serialisable."""
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_csv(path, table: dict):
    keys = [k for k, v in table.items() if isinstance(v, np.ndarray) and v.ndim == 1]
    n = len(table[keys[0]])
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for i in range(n):
            f.write(",".join(f"{table[k][i]:.8g}" for k in keys) + "\n")


def pick_run(cfg, tx, snr, rng, lead, want_accept=None, tries=40, **kw):
    """Run one packet; optionally retry until it is accepted / dropped (for illustration)."""
    for _ in range(tries):
        out = run_link(tx.burst, tx.burst_q, cfg, snr, rng, n=1, leads=np.array([lead]), **kw)
        res = out[3]
        if want_accept is None or bool(res.accepted[0]) == want_accept:
            return out
    return out


def run_demo(cfg, tx, rng):
    """One representative packet through the whole chain + the data needed by the stage plots."""
    lead = 200
    snr_main, snr_low = 2.0, -3.0
    d = dict(lead=lead, snr_main=snr_main, snr_low=snr_low)
    d["rx_view"] = {s: over_the_air(tx.burst, tx.burst_q, cfg, s, rng, 1, h=np.array([1 + 0j]),
                                    leads=np.array([lead]))[0] for s in (10.0, snr_main, snr_low)}
    rx, _, _, res, genie = pick_run(cfg, tx, snr_main, rng, lead, want_accept=True)
    rx_l, _, _, res_l, _ = pick_run(cfg, tx, snr_low, rng, lead, want_accept=False)
    d.update(rx=rx, res=res, genie=genie, energy=front_end(rx, cfg), res_low=res_l, energy_low=front_end(rx_l, cfg))
    # frame-parser illustration: one clean and one corrupted packet
    cases = [dict(snr=snr_main, res=res, bits=res.bits[0])]
    r2 = pick_run(cfg, tx, -2.0, rng, lead, want_accept=False)[3]
    cases.append(dict(snr=-2.0, res=r2, bits=r2.bits[0]))
    d["parser_cases"] = cases
    # reconstruction at four channel SNRs
    p_x = np.mean(tx.x_s ** 2)
    rc = []
    for s in (2.0, -1.0, -3.0, -5.0):
        r = pick_run(cfg, tx, s, rng, lead)[3]
        xh = decode_payload(r.fields["payload"][0], cfg)
        m = float(mt.mse(tx.x_s, xh))
        rc.append(dict(snr=s, accepted=bool(r.accepted[0]), mse=m, snr_rec=10 * np.log10(p_x / m),
                       n_err=int((r.fields["payload"][0] != tx.pcm_bits).sum()), recon=dac_reconstruct(xh, cfg), xhat=xh))
    d["recon_cases"] = rc
    return d


def rayleigh_psr_theory(cfg, snr_sim_db):
    """Average of the AWGN packet success probability over |h|^2 ~ Exp(1)."""
    x = np.logspace(-5, 1.8, 6000)
    pdf = np.exp(-x)
    out = []
    for s in np.atleast_1d(snr_sim_db):
        ber = mt.ber_ook_noncoherent(ch.sim_snr_to_ebn0_db(s + 10 * np.log10(x), cfg))
        out.append(np.trapezoid(mt.psr_theory(ber, cfg) * pdf, x))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer Monte-Carlo packets (fast preview)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = Config() if args.seed is None else Config(seed=args.seed)
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    seeds = np.random.SeedSequence(cfg.seed).spawn(5)
    r_demo, r_sweep, r_bpsk, r_dist, r_sec = [np.random.default_rng(s) for s in seeds]
    n_sweep, n_bpsk, n_dist = (60, 20, 40) if args.quick else (400, 100, 200)
    t_start = time.time()

    # ------------------------------------------------------------------ 1-5: transmitter
    log("[1/7] Transmitter: source -> sampling -> PCM -> packet -> Manchester -> OOK")
    fob = KeyFob(cfg.device_id, cfg.fob_key)
    tx = transmit(cfg, fob, CMD_UNLOCK)
    p_x = float(np.mean(tx.x_s ** 2))
    mse_q = float(np.mean((tx.x_s - tx.x_q) ** 2))
    log(f"      {tx.x_s.size} samples, {tx.pcm_bits.size} PCM bits, packet = {tx.packet_bits.size} bits, "
        f"burst = {tx.burst.size / cfg.fs_rf * 1e3:.1f} ms @ {cfg.fs_rf / 1e3:.1f} kHz simulation rate")
    log(f"      quantization: step = {tx.delta:.5f}, MSE = {mse_q:.3e}, SQNR = {10*np.log10(p_x/mse_q):.2f} dB "
        f"(theory 6.02n+1.76 = {mt.sqnr_theory_db(cfg.n_bits_pcm):.2f} dB)")

    # ------------------------------------------------------------------ demo packet + stage figures
    log("[2/7] Stage-by-stage figures of one packet (SNR = +2 dB)")
    d = run_demo(cfg, tx, r_demo)
    fig_block_diagram(FIG / "01_block_diagram.png")
    fig_source_sampling(cfg, tx, FIG / "02_source_sampling.png")
    fig_quantization(cfg, tx, FIG / "03_quantization.png")
    fig_pcm(cfg, tx, FIG / "04_pcm_encoding.png")
    fig_packet(cfg, tx, FIG / "05_packet_structure.png")
    fig_line_coding(cfg, tx, FIG / "06_line_coding.png")
    fig_modulation(cfg, tx, FIG / "07_ook_modulation.png")
    fig_channel(cfg, tx, d, FIG / "08_channel_received.png")
    fig_demod(cfg, tx, d, FIG / "09_demodulation.png")
    fig_eye(cfg, tx, d, FIG / "10_eye_diagram.png")
    fig_frame_parser(cfg, tx, d, FIG / "11_frame_parser.png")
    fig_reconstruction(cfg, tx, d, FIG / "12_reconstruction.png")
    demo_bit_errors = int((d["res"].bits[0] != tx.packet_bits).sum())
    log(f"      demo packet: {demo_bit_errors} bit errors, accepted = {bool(d['res'].accepted[0])}, "
        f"reconstruction MSE = {float(mt.mse(tx.x_s, decode_payload(d['res'].fields['payload'][0], cfg))):.3e}")

    # ------------------------------------------------------------------ sweeps
    snr_list = list(np.r_[np.arange(-8.0, 7.0, 1.0), [8.0, 10.0, 15.0, 20.0]])
    log(f"[3/7] Monte-Carlo: channel SNR sweep ({n_sweep} packets x {len(snr_list)} SNR values)")
    sweep = snr_sweep(cfg, tx, snr_list, n_sweep, r_sweep)
    log(f"[4/7] BPSK + NRZ reference curve ({n_bpsk} packets per point)")
    bpsk = bpsk_baseline(cfg, list(np.arange(-12.0, -2.0, 1.0)), n_bpsk, r_bpsk)
    distances = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70]
    log(f"[5/7] Distance sweep: path loss + fading ({n_dist} packets x {len(distances)} distances x 3 channels)")
    dist = distance_sweep(cfg, tx, distances, ["awgn", "rician", "rayleigh"], n_dist, r_dist, verbose=False)
    dist["psr_rayleigh_theory"] = lambda dd: rayleigh_psr_theory(cfg, ch.inband_to_sim_snr_db(ch.snr_inband_db(dd, cfg), cfg))
    for k in ("awgn", "rician", "rayleigh"):
        log(f"      PSR({k}) at 20/40/60 m = " + " / ".join(f"{dist['psr_' + k][distances.index(x)]:.2f}" for x in (20, 40, 60)))

    # ------------------------------------------------------------------ security demo
    log("[6/7] Rolling-code security demo (complete waveform chain)")
    events = security_demo(cfg, tx, r_sec)
    for e in events:
        log(f"      {e['label']} {e['who']:9s} -> {'ACCEPT' if e['accepted'] else 'REJECT':6s} {e['reason']}")

    # ------------------------------------------------------------------ figures of results
    log("[7/7] Result figures, Shannon analysis, files")
    fig_ber(cfg, sweep, bpsk, FIG / "13_ber_vs_snr.png")
    fig_mse(cfg, sweep, tx, FIG / "14_mse_vs_snr.png")
    fig_psr(cfg, sweep, FIG / "15_psr_vs_snr.png")
    sh = shannon_numbers(cfg)
    _, snr90_sim_b = fig_shannon(cfg, sweep, sh, FIG / "16_shannon.png")
    fig_rolling_flow(FIG / "17_rolling_code_flow.png")
    fig_security_timeline(events, FIG / "18_security_demo.png")
    fig_distance(cfg, dist, sweep, FIG / "19_distance_range.png")

    # --- derived numbers for the report
    psr, snr = sweep["psr"], sweep["snr_db"]
    snr_psr90 = float(np.interp(0.9, psr, snr))
    snr_psr50 = float(np.interp(0.5, psr, snr))
    q_err = tx.x_s - tx.x_q
    from scipy.optimize import brentq
    p_star = brentq(lambda p: mt.mse_from_ber(p, q_err, tx.pcm_bits, cfg.n_bits_pcm)[0] - 2 * mse_q, 1e-9, 0.2)
    snr_star_sim = float(10 * np.log10(-np.log(2 * p_star) / 8.0))        # p = 0.5 exp(-8 SNR)  (Manchester OOK, spc = 16)
    ebn0_star = float(ch.sim_snr_to_ebn0_db(snr_star_sim, cfg))
    metrics = dict(
        config=dict(fs_audio=cfg.fs_audio, fm=cfg.fm, n_bits=cfg.n_bits_pcm, n_samples=cfg.n_samples, bit_rate=cfg.bit_rate,
                    fs_rf=cfg.fs_rf, fc=cfg.fc, packet_bits=int(tx.packet_bits.size), payload_bits=cfg.n_payload_bits,
                    airtime_ms=tx.packet_bits.size / cfg.bit_rate * 1e3, n_sweep=n_sweep, n_dist=n_dist, seed=cfg.seed),
        source=dict(signal_power=p_x, delta=tx.delta, mse_quant=mse_q, sqnr_db=10 * np.log10(p_x / mse_q),
                    sqnr_theory_db=mt.sqnr_theory_db(cfg.n_bits_pcm)),
        demo=dict(snr_db=d["snr_main"], bit_errors=demo_bit_errors, accepted=bool(d["res"].accepted[0]),
                  recon=[{k: v for k, v in c.items() if k not in ("recon", "xhat")} for c in d["recon_cases"]]),
        sweep_summary=dict(snr_psr90_db=snr_psr90, snr_psr50_db=snr_psr50,
                           ebn0_psr90_db=float(ch.sim_snr_to_ebn0_db(snr_psr90, cfg)),
                           inband_psr90_db=snr_psr90 + sh["offset_b_to_sim"],
                           p_star=p_star, snr_star_sim_db=snr_star_sim, ebn0_star_db=ebn0_star,
                           inband_star_db=snr_star_sim + sh["offset_b_to_sim"],
                           undetected_max=float(sweep["undetected"].max())),
        shannon=sh,
        distance=dict(distance_m=distances, snr_inband_db=dist["snr_inband_db"],
                      psr_awgn=dist["psr_awgn"], psr_rician=dist["psr_rician"], psr_rayleigh=dist["psr_rayleigh"],
                      noise_floor_dbm=ch.noise_floor_dbm(cfg)),
        security=events,
    )
    for name, kind in (("awgn", "AWGN"), ("rician", "Rician"), ("rayleigh", "Rayleigh")):
        p = dist["psr_" + name]
        metrics["distance"]["range90_" + name] = float(np.interp(0.9, p[::-1], np.array(distances, dtype=float)[::-1])) if p.min() < 0.9 <= p.max() else None
    (RES / "metrics.json").write_text(json.dumps(to_py(metrics), indent=2))
    save_csv(RES / "sweep_snr.csv", {k: v for k, v in sweep.items() if isinstance(v, np.ndarray)})
    save_csv(RES / "ber_bpsk.csv", bpsk)
    save_csv(RES / "sweep_distance.csv", {k: v for k, v in dist.items() if isinstance(v, np.ndarray)})

    # ------------------------------------------------------------------ summary
    log("\n================= SUMMARY =================")
    log(f"SQNR (8-bit PCM)                : {metrics['source']['sqnr_db']:.2f} dB   (MSE floor {mse_q:.2e})")
    log(f"90 % packet success at          : SNR = {snr_psr90:+.1f} dB  (Eb/N0 = {metrics['sweep_summary']['ebn0_psr90_db']:.1f} dB)")
    log(f"Channel MSE = quantization MSE  : BER = {p_star:.2e}  -> SNR = {snr_star_sim:+.1f} dB")
    log(f"Shannon limit for Rb in B       : SNR_B = {sh['shannon_snr_b_db']:.1f} dB   (OOK gap {sh['gap_ook_db']:.1f} dB)")
    log(f"Range (90 % PSR)                : " + ", ".join(f"{k}: {metrics['distance']['range90_' + k]}" for k in ("awgn", "rician", "rayleigh")))
    log(f"Figures in {FIG}, results in {RES}   (total {time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
