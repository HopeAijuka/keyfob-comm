"""Performance figures: BER, MSE, PSR, Shannon, rolling-code and distance results."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch

from . import channel as ch
from . import metrics as mt
from .config import Config
from .plots_common import C, plt, save


def _ebn0(snr_sim, cfg):
    return ch.sim_snr_to_ebn0_db(np.asarray(snr_sim, dtype=float), cfg)


def _top_axis(ax, cfg, label="Eb/N0 [dB]"):
    off = 10 * np.log10(cfg.fs_rf / (2 * cfg.bit_rate))
    sec = ax.secondary_xaxis("top", functions=(lambda x: x + off, lambda x: x - off))
    sec.set_xlabel(label)
    return sec


# ----------------------------------------------------------------------------- BER
def fig_ber(cfg: Config, sweep, bpsk, path):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    e_th = np.linspace(0, 16, 300)
    ax.semilogy(e_th, mt.ber_ook_noncoherent(e_th), color=C["red"], lw=1.6, label="OOK + Manchester, non-coherent - theory ½·exp(−Eb/2N0)")
    ax.semilogy(e_th, mt.ber_bpsk_coherent(e_th), color=C["blue"], lw=1.6, label="BPSK + NRZ, coherent - theory Q(√(2Eb/N0))")
    n_bits = sweep["bits_per_point"]
    floor = 1.0 / n_bits
    e = _ebn0(sweep["snr_db"], cfg)
    b = sweep["ber_genie"]
    ax.semilogy(e[b > 0], b[b > 0], "o", color=C["red"], mfc="white", ms=6, label="OOK simulation (full chain)")
    ax.semilogy(e[b == 0], np.full((b == 0).sum(), floor), "v", color=C["red"], mfc="white", ms=5,
                label=f"no error in {n_bits/1e3:.0f}·10³ bits (upper bound)")
    eb = _ebn0(bpsk["snr_db"], cfg)
    bb = bpsk["ber"]
    ax.semilogy(eb[bb > 0], bb[bb > 0], "s", color=C["blue"], mfc="white", ms=6, label="BPSK simulation")
    ax.set(xlim=(0, 16), ylim=(floor / 2, 0.6), xlabel="Eb/N0 [dB]", ylabel="bit error rate",
           title="Bit error rate: simulation vs theory")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    off = 10 * np.log10(cfg.fs_rf / (2 * cfg.bit_rate))
    sec = ax.secondary_xaxis("top", functions=(lambda x: x - off, lambda x: x + off))
    sec.set_xlabel("channel SNR as used in awgn() [dB]")
    return save(fig, path)


# ----------------------------------------------------------------------------- MSE
def fig_mse(cfg: Config, sweep, tx, path):
    snr = sweep["snr_db"]
    p_x = np.mean(tx.x_s ** 2)
    mse_q = np.mean((tx.x_s - tx.x_q) ** 2)
    fine = np.linspace(snr.min(), snr.max(), 200)
    ber_th = mt.ber_ook_noncoherent(_ebn0(fine, cfg))
    mse_th = mt.mse_from_ber(ber_th, tx.x_s - tx.x_q, tx.pcm_bits, cfg.n_bits_pcm, cfg.amplitude)
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.0))
    a = ax[0]
    a.semilogy(fine, mse_th, color=C["gray"], lw=1.6, ls="--", label="theory: MSE(BER_th(SNR))")
    a.semilogy(snr, sweep["mse_raw"], "o-", color=C["red"], ms=5, lw=1.2, label="raw decoded payload (guide's MSE)")
    a.semilogy(snr, sweep["mse_delivered"], "s-", color=C["blue"], ms=5, lw=1.2, label="delivered (CRC fail → silence)")
    a.axhline(mse_q, color=C["green"], lw=1.4, ls=":", label=f"quantization floor Δ²/12 = {mse_q:.2e}")
    a.axhline(p_x, color=C["purple"], lw=1.2, ls=":", label=f"muted output = signal power = {p_x:.2f}")
    a.set(xlabel="channel SNR [dB]", ylabel="MSE  mean((x[n] − x̂[n])²)", title="(a) Reconstruction MSE vs channel SNR",
          ylim=(mse_q / 3, 3))
    a.grid(True, which="both", alpha=0.25)
    a.legend(loc="upper right", fontsize=8)
    _top_axis(a, cfg)
    b = ax[1]
    to_db = lambda m: 10 * np.log10(p_x / np.asarray(m))
    b.plot(fine, to_db(mse_th), color=C["gray"], lw=1.6, ls="--", label="theory")
    b.plot(snr, to_db(sweep["mse_raw"]), "o-", color=C["red"], ms=5, lw=1.2, label="raw decoded payload")
    b.plot(snr, to_db(sweep["mse_delivered"]), "s-", color=C["blue"], ms=5, lw=1.2, label="delivered (muted on CRC fail)")
    b.axhline(to_db(mse_q), color=C["green"], ls=":", lw=1.4, label=f"quantization limit {to_db(mse_q):.1f} dB")
    b.set(xlabel="channel SNR [dB]", ylabel="reconstruction SNR  10·log10(Px / MSE)  [dB]",
          title="(b) Reconstruction SNR vs channel SNR")
    b.legend(loc="center right", fontsize=8)
    _top_axis(b, cfg)
    return save(fig, path)


# ----------------------------------------------------------------------------- PSR
def fig_psr(cfg: Config, sweep, path):
    snr = sweep["snr_db"]
    fine = np.linspace(snr.min(), snr.max(), 300)
    th = mt.psr_theory(mt.ber_ook_noncoherent(_ebn0(fine, cfg)), cfg)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(fine, th, color=C["gray"], ls="--", lw=1.6, label="theory (1−p)^3288 · P(sync ok)")
    ax.plot(snr, sweep["psr"], "o-", color=C["blue"], ms=5, label="packet success rate (sync + CRC + ID)")
    ax.plot(snr, sweep["sync_ok"], "^-", color=C["orange"], ms=4, lw=1, label="sync word found")
    ax.plot(snr, sweep["timing_ok"], ".:", color=C["green"], lw=1, label="timing error ≤ 1 sample")
    ax.set(xlabel="channel SNR [dB]", ylabel="probability", ylim=(-0.03, 1.05),
           title="Packet success rate vs channel SNR")
    ax.legend(loc="center right", fontsize=8)
    _top_axis(ax, cfg)
    return save(fig, path)


# ----------------------------------------------------------------------------- Shannon
def shannon_numbers(cfg: Config) -> dict:
    B, Rb = cfg.bandwidth_hz, cfg.bit_rate
    r_src = cfg.fs_audio * cfg.n_bits_pcm
    ook_req = mt.required_ebn0_db(mt.ber_ook_noncoherent, 1e-5)
    bpsk_req = mt.required_ebn0_db(mt.ber_bpsk_coherent, 1e-5)
    eta_ook, eta_bpsk = Rb / B, Rb / (2 * Rb)
    return dict(
        B=B, Rb=Rb, R_src=r_src, eta_ook=eta_ook, eta_bpsk=eta_bpsk,
        shannon_snr_b_db=mt.shannon_min_snr_db(Rb, B),
        shannon_snr_b_src_db=mt.shannon_min_snr_db(r_src, B),
        shannon_ebn0_ook_db=float(mt.shannon_min_ebn0_db(eta_ook)),
        shannon_ebn0_bpsk_db=float(mt.shannon_min_ebn0_db(eta_bpsk)),
        ook_req_ebn0_db=ook_req, bpsk_req_ebn0_db=bpsk_req,
        gap_ook_db=ook_req - float(mt.shannon_min_ebn0_db(eta_ook)),
        gap_bpsk_db=bpsk_req - float(mt.shannon_min_ebn0_db(eta_bpsk)),
        ultimate_db=10 * np.log10(np.log(2)),
        offset_b_to_sim=float(10 * np.log10(cfg.sim_bandwidth_hz / B)),
    )


def fig_shannon(cfg: Config, sweep, sh, path):
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.0))
    a = ax[0]
    snr_b = np.linspace(-15, 25, 400)
    cap = mt.shannon_capacity(sh["B"], 10 ** (snr_b / 10)) / 1e3
    a.semilogy(snr_b, cap, color=C["dark"], lw=2, label=f"Shannon capacity C = B·log2(1+SNR),  B = {sh['B']/1e3:.1f} kHz")
    a.axhline(sh["Rb"] / 1e3, color=C["red"], ls="--", label=f"fob bit rate Rb = {sh['Rb']/1e3:.1f} kbit/s")
    a.axhline(sh["R_src"] / 1e3, color=C["purple"], ls="--", label=f"real-time PCM source = {sh['R_src']/1e3:.0f} kbit/s")
    a.axvline(sh["shannon_snr_b_db"], color=C["red"], ls=":", lw=1)
    a.axvline(sh["shannon_snr_b_src_db"], color=C["purple"], ls=":", lw=1)
    # region where the real link loses packets
    snr = sweep["snr_db"] + sh["offset_b_to_sim"]
    psr = sweep["psr"]
    snr90 = float(np.interp(0.9, psr, snr)) if psr.max() >= 0.9 else np.nan
    a.axvspan(snr.min() - 1, snr90, color=C["orange"], alpha=0.18, label="our uncoded OOK link: packets lost")
    a.axvline(snr90, color=C["orange"], lw=1.2)
    a.annotate(f"{sh['shannon_snr_b_db']:.1f} dB", (sh["shannon_snr_b_db"], 400), xytext=(-5, 0), textcoords="offset points",
               color=C["red"], ha="right", fontsize=8.5)
    a.annotate(f"{sh['shannon_snr_b_src_db']:.1f} dB", (sh["shannon_snr_b_src_db"], 400), xytext=(5, 0),
               textcoords="offset points", color=C["purple"], ha="left", fontsize=8.5)
    a.annotate(f"PSR = 90 %\nat {snr90:.1f} dB", (snr90, 220.0), xytext=(6, 0), textcoords="offset points", color=C["orange"], fontsize=8.5)
    a.set(xlabel="in-band SNR  (noise power in B)  [dB]", ylabel="rate [kbit/s]", xlim=(-15, 25), ylim=(1, 600),
          title="(a) Capacity vs the rates in this project")
    a.legend(loc="lower right", fontsize=8)
    a.grid(True, which="both", alpha=0.25)
    b = ax[1]
    eta = np.logspace(-2, 1, 300)
    b.semilogx(eta, mt.shannon_min_ebn0_db(eta), color=C["dark"], lw=2, label="Shannon limit (min Eb/N0)")
    b.axhline(sh["ultimate_db"], color=C["gray"], ls=":", label=f"ultimate limit {sh['ultimate_db']:.2f} dB")
    b.plot(sh["eta_ook"], sh["ook_req_ebn0_db"], "o", color=C["red"], ms=9, label=f"OOK-Manchester @ BER 1e-5: {sh['ook_req_ebn0_db']:.1f} dB")
    b.plot(sh["eta_bpsk"], sh["bpsk_req_ebn0_db"], "s", color=C["blue"], ms=9, label=f"BPSK-NRZ @ BER 1e-5: {sh['bpsk_req_ebn0_db']:.1f} dB")
    b.plot([sh["eta_ook"]] * 2, [sh["shannon_ebn0_ook_db"], sh["ook_req_ebn0_db"]], color=C["red"], lw=1.2)
    b.plot([sh["eta_bpsk"]] * 2, [sh["shannon_ebn0_bpsk_db"], sh["bpsk_req_ebn0_db"]], color=C["blue"], lw=1.2)
    b.text(sh["eta_ook"] * 0.8, (sh["shannon_ebn0_ook_db"] + sh["ook_req_ebn0_db"]) / 2, f"gap\n{sh['gap_ook_db']:.1f} dB",
           color=C["red"], ha="right", va="center", fontsize=9)
    b.text(sh["eta_bpsk"] * 1.1, (sh["shannon_ebn0_bpsk_db"] + sh["bpsk_req_ebn0_db"]) / 2, f"gap {sh['gap_bpsk_db']:.1f} dB",
           color=C["blue"], ha="left", va="center", fontsize=9)
    b.set(xlabel="spectral efficiency  η = Rb / B  [bit/s/Hz]", ylabel="Eb/N0 [dB]", ylim=(-3, 20),
          title="(b) Distance of the modulations from the Shannon limit")
    b.legend(loc="upper left", fontsize=8)
    b.grid(True, which="both", alpha=0.25)
    return save(fig, path), snr90


# ----------------------------------------------------------------------------- rolling code
def fig_rolling_flow(path):
    fig, ax = plt.subplots(figsize=(12.6, 5.6), constrained_layout=False)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, t, fc, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec="#374151", lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.25)

    def arr(p0, p1, label=None, col="#374151", off=(0, 0.12)):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3))
        if label:
            ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], label, ha="center", fontsize=8.5, color=col)

    box(0.2, 2.4, 1.7, 0.9, "Packet received\nfrom demodulator", "#dbeafe")
    box(2.5, 2.4, 1.9, 0.9, "Sync word, CRC-16\nand device ID OK?", "#fef9c3")
    box(5.0, 2.4, 1.9, 0.9, "Decrypt hop code\nwith the fob key", "#e0e7ff")
    box(7.5, 2.4, 1.9, 0.9, "Command + 12-bit\ndiscrimination match?", "#fef9c3", fs=8.4)
    box(10.0, 2.4, 2.4, 0.9, "Δ = counter − last\n(mod 2¹⁶)", "#e0e7ff")
    arr((1.9, 2.85), (2.5, 2.85))
    arr((4.4, 2.85), (5.0, 2.85), "yes")
    arr((6.9, 2.85), (7.5, 2.85))
    arr((9.4, 2.85), (10.0, 2.85), "yes")
    box(2.5, 0.5, 1.9, 0.9, "DROP\n(noise / other car)", "#fecaca")
    arr((3.45, 2.4), (3.45, 1.4), "no", col="#b91c1c", off=(0.25, 0))
    box(7.5, 0.5, 1.7, 0.9, "REJECT\nforged / wrong key", "#fecaca", fs=8.5)
    arr((8.45, 2.4), (8.45, 1.4), "no", col="#b91c1c", off=(0.25, 0))
    box(9.4, 4.2, 3.0, 0.8, "0 < Δ ≤ 16  →  ACCEPT\nupdate last counter, act on command", "#bbf7d0")
    arr((11.2, 3.3), (11.2, 4.2), col="#15803d")
    box(9.4, 0.4, 1.4, 1.0, "Δ = 0 or Δ ≥ 2¹⁵\nREJECT: replay", "#fecaca", fs=8.2)
    arr((10.05, 2.4), (10.05, 1.4), col="#b91c1c")
    box(10.95, 0.25, 1.55, 1.15, "16 < Δ < 2¹⁵\nstore counter; 2nd\nconsecutive press\n→ re-sync, ACCEPT", "#fed7aa", fs=7.6)
    arr((11.75, 2.4), (11.75, 1.4), col="#c2410c")
    ax.text(6.3, 5.3, "Rolling-code state machine in the car", fontsize=12, fontweight="bold", ha="center")
    return save(fig, path)


def fig_security_timeline(events, path):
    import textwrap
    W = 15.2
    fig, ax = plt.subplots(figsize=(W, 0.66 * len(events) + 1.4), constrained_layout=False)
    ax.axis("off")
    ax.set_xlim(0, W)
    ax.set_ylim(0, len(events) + 1)
    heads = [("#", 0.15), ("who", 0.6), ("what happens", 1.8), ("counter", 7.3), ("car last", 8.3),
             ("link", 9.3), ("decision", 10.0), ("reason", 11.1)]
    for t, x in heads:
        ax.text(x, len(events) + 0.5, t, fontweight="bold", fontsize=9.5, va="center")
    for i, e in enumerate(events):
        y = len(events) - i - 0.5
        ok = e["accepted"]
        ax.add_patch(plt.Rectangle((0.05, y - 0.42), W - 0.1, 0.84, fc="#dcfce7" if ok else "#fee2e2", ec="none", alpha=0.8))
        ax.text(0.15, y, e["label"], va="center", fontsize=9)
        ax.text(0.6, y, e["who"], va="center", fontsize=9)
        ax.text(1.8, y, "\n".join(textwrap.wrap(e["note"], 58)), va="center", fontsize=8.6)
        ax.text(7.3, y, "-" if e["counter"] is None else str(e["counter"]), va="center", fontsize=9)
        ax.text(8.3, y, str(e["car_counter"]), va="center", fontsize=9)
        ax.text(9.3, y, "OK" if e["link_ok"] else "FAIL", va="center", fontsize=9)
        ax.text(10.0, y, "ACCEPT" if ok else "REJECT", va="center", fontsize=9, fontweight="bold",
                color="#166534" if ok else "#991b1b")
        ax.text(11.1, y, "\n".join(textwrap.wrap(e["reason"], 40)), va="center", fontsize=8)
    return save(fig, path)


# ----------------------------------------------------------------------------- distance
def fig_distance(cfg: Config, dist, sweep, path):
    d = dist["distance_m"]
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.9), gridspec_kw={"width_ratios": [1, 1.25]})
    dd = np.linspace(d.min(), d.max(), 300)
    a = ax[0]
    a.plot(dd, ch.snr_inband_db(dd, cfg), color=C["dark"], lw=1.8)
    snr_ok = float(np.interp(0.9, sweep["psr"], sweep["snr_db"])) - float(ch.inband_to_sim_snr_db(0.0, cfg))
    a.axhline(snr_ok, color=C["orange"], ls="--", label=f"AWGN: 90 % PSR needs {snr_ok:.1f} dB")
    a.set(xlabel="distance fob - car [m]", ylabel="mean in-band SNR [dB]", title="(a) Link budget")
    a.text(0.97, 0.93, f"Pt = {cfg.tx_power_dbm:.0f} dBm EIRP, n = {cfg.path_loss_exp}\nextra loss {cfg.extra_loss_db:.0f} dB, NF {cfg.noise_figure_db:.0f} dB\n"
           f"noise floor {ch.noise_floor_dbm(cfg):.1f} dBm", transform=a.transAxes, ha="right", va="top", fontsize=8.5)
    a.legend(loc="lower left", fontsize=8.5)
    b = ax[1]
    b.plot(d, dist["psr_awgn"], "o-", color=C["dark"], label="path loss + AWGN only")
    b.plot(d, dist["psr_rician"], "s-", color=C["green"], label="+ Rician fading (K = 6 dB, LOS)")
    b.plot(d, dist["psr_rayleigh"], "^-", color=C["red"], label="+ Rayleigh fading (NLOS, parking garage)")
    if "psr_rayleigh_theory" in dist:
        b.plot(dd, dist["psr_rayleigh_theory"](dd), "--", color=C["red"], lw=1, alpha=0.8, label="Rayleigh theory")
    b.set(xlabel="distance fob - car [m]", ylabel="packet success rate", ylim=(-0.03, 1.05),
          title="(b) Range of the key fob")
    b.legend(loc="lower left", fontsize=8.5)
    return save(fig, path)
