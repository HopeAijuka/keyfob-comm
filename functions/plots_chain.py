"""Stage-by-stage figures of the communication chain (one function per figure)."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.signal import welch

from .config import Config
from .framing import frame_layout
from .line_coding import chips_to_levels, manchester_encode, nrz_encode
from .plots_common import C, plt, save


# ----------------------------------------------------------------------------- 1. block diagram
def _box(ax, x, y, w, h, text, fc, ec="#374151", fs=7.8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#111827", linespacing=1.25)


def _arrow(ax, p0, p1, style="-|>", color="#374151", ls="-"):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle=style, color=color, lw=1.4, ls=ls, shrinkA=0, shrinkB=0))


def fig_block_diagram(path):
    fig, ax = plt.subplots(figsize=(13.2, 6.3), constrained_layout=False)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6.9)
    ax.axis("off")
    bw, bh, gap, x0 = 1.4, 1.0, 0.2, 0.3
    xs = [x0 + i * (bw + gap) for i in range(8)]
    y1, y2 = 4.55, 2.35
    tx = ["Message source\nsin(2π·300·t)", "Sampler\nFs = 8 kHz", "Quantizer\n8 bit, 256 levels",
          "PCM encoder\nnatural binary", "Packet framing\nhop code + CRC",
          "Line coder\nManchester", "Modulator\nOOK on carrier fc"]
    for i, t in enumerate(tx):
        _box(ax, xs[i], y1, bw, bh, t, "#dbeafe")
        if i:
            _arrow(ax, (xs[i - 1] + bw, y1 + bh / 2), (xs[i], y1 + bh / 2))
    _box(ax, xs[7], y1, bw, bh, "CHANNEL\nloss, fading\n+ AWGN", "#fde7c8", bold=False)
    _arrow(ax, (xs[6] + bw, y1 + bh / 2), (xs[7], y1 + bh / 2))
    _box(ax, xs[4], 5.85, bw, 0.7, "Key fob counter\n(KeeLoq-style)", "#e0e7ff", fs=7.8)
    _arrow(ax, (xs[4] + bw / 2, 5.85), (xs[4] + bw / 2, y1 + bh))
    # receiver row: slot 7 -> slot 0 (right to left)
    rx = ["Down-convert\n+ chip integrate", "Timing sync\npreamble+sync", "Manchester\ndecoder",
          "Frame parser\nCRC · ID checks", "Rolling-code\ncheck (car)", "PCM decoder\n+ de-quantizer",
          "D/A + low-pass\nfilter"]
    for k, t in enumerate(rx):
        _box(ax, xs[7 - k], y2, bw, bh, t, "#dcfce7")
    _arrow(ax, (xs[7] + bw / 2, y1), (xs[7] + bw / 2, y2 + bh))
    for k in range(1, 7):
        _arrow(ax, (xs[7 - k + 1], y2 + bh / 2), (xs[7 - k] + bw, y2 + bh / 2))
    _box(ax, xs[0], y2, bw, bh, "Reconstructed\nsignal x̂(t)", "#fef3c7", bold=True)
    _arrow(ax, (xs[1], y2 + bh / 2), (xs[0] + bw, y2 + bh / 2))
    _box(ax, xs[3], 0.85, bw, 0.7, "Vehicle memory\nID, last counter", "#e0e7ff", fs=7.8)
    _arrow(ax, (xs[3] + bw / 2, 1.55), (xs[3] + bw / 2, y2), style="<|-|>")
    _box(ax, xs[0], 0.25, 4.4, 0.85, "PERFORMANCE ANALYSIS: x[n] vs x̂[n]\n"
         "BER · PSR · MSE · SNR  vs  channel SNR", "#fce7f3", fs=8.6)
    _arrow(ax, (xs[0] + bw / 2, y2), (xs[0] + bw / 2, 1.1), ls="--", color="#9d174d")
    ax.text(0.3, 6.55, "TRANSMITTER  (key fob)", fontsize=11, fontweight="bold", color="#1e40af")
    ax.text(0.3, 3.75, "RECEIVER  (car)", fontsize=11, fontweight="bold", color="#166534")
    return save(fig, path)


# ----------------------------------------------------------------------------- 2. source + sampling
def fig_source_sampling(cfg: Config, tx, path):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    m, ms = tx.t_fine < 0.010, tx.t_s < 0.010
    ax[0].plot(tx.t_fine[m] * 1e3, tx.x_fine[m], color=C["blue"], lw=1.6, label="analog message x(t)")
    ax[0].vlines(tx.t_s[ms] * 1e3, 0, tx.x_s[ms], color=C["orange"], lw=1)
    ax[0].plot(tx.t_s[ms] * 1e3, tx.x_s[ms], "o", color=C["orange"], ms=4.5, label=f"samples x[n], Fs = {cfg.fs_audio/1e3:.0f} kHz")
    ax[0].set(xlabel="time [ms]", ylabel="amplitude", title="(a) Source and sampling (first 10 ms)")
    ax[0].legend(loc="upper right", ncol=1, framealpha=0.9)
    x = tx.x_s * np.hanning(tx.x_s.size)
    f = np.fft.rfftfreq(x.size, 1 / cfg.fs_audio)
    sp = 20 * np.log10(np.abs(np.fft.rfft(x)) / np.abs(np.fft.rfft(x)).max() + 1e-12)
    ax[1].plot(f, sp, color=C["blue"])
    ax[1].axvline(cfg.fm, color=C["red"], ls="--", lw=1)
    ax[1].text(cfg.fm + 80, -8, f"fm = {cfg.fm:.0f} Hz", color=C["red"])
    ax[1].axvline(cfg.fs_audio / 2, color=C["gray"], ls=":", lw=1.2)
    ax[1].text(cfg.fs_audio / 2 - 80, -60, "Nyquist\n4 kHz", ha="right", color=C["gray"])
    ax[1].set(xlabel="frequency [Hz]", ylabel="magnitude [dB]", ylim=(-110, 5),
              title="(b) Spectrum of the samples")
    return save(fig, path)


# ----------------------------------------------------------------------------- 3. quantization
def fig_quantization(cfg: Config, tx, path):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
    lo, hi = 0.0003, 0.0014
    m = (tx.t_fine >= lo) & (tx.t_fine <= hi)
    ms = (tx.t_s >= lo) & (tx.t_s <= hi)
    ax[0].plot(tx.t_fine[m] * 1e3, tx.x_fine[m], color=C["blue"], lw=1.4, label="x(t)")
    ax[0].plot(tx.t_s[ms] * 1e3, tx.x_s[ms], "o", color=C["orange"], label="sample x[n]")
    ax[0].plot(tx.t_s[ms] * 1e3, tx.x_q[ms], "s", color=C["red"], label="quantized x_q[n]")
    ax[0].vlines(tx.t_s[ms] * 1e3, tx.x_s[ms], tx.x_q[ms], color=C["red"], lw=1)
    ylo, yhi = 0.93, 1.006
    levels = -1 + (np.arange(2 ** cfg.n_bits_pcm) + 0.5) * tx.delta
    for lv in levels[(levels > ylo) & (levels < yhi)]:
        ax[0].axhline(lv, color=C["gray"], lw=0.6, ls=":")
    ax[0].set(xlabel="time [ms]", ylabel="amplitude", ylim=(ylo, yhi),
              title="(a) Zoom near the first peak: dotted = 256 levels")
    ax[0].legend(loc="lower center", ncol=3, fontsize=8, framealpha=0.95)
    err = tx.x_s - tx.x_q
    ax[1].plot(np.arange(err.size), err * 1e3, color=C["purple"], lw=0.9)
    for s in (+1, -1):
        ax[1].axhline(s * tx.delta / 2 * 1e3, color=C["red"], ls="--", lw=1)
    ax[1].set(xlabel="sample index n", ylabel="error x[n] − x_q[n]  [10⁻³]",
              title=f"(b) Quantization error, |e| ≤ Δ/2 = {tx.delta/2*1e3:.2f}·10⁻³")
    return save(fig, path)


# ----------------------------------------------------------------------------- 4. PCM
def fig_pcm(cfg: Config, tx, path):
    n_show = 40
    bits = tx.pcm_bits.reshape(-1, cfg.n_bits_pcm)[:n_show]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax[0].imshow(bits.T, cmap="Blues", aspect="auto", interpolation="nearest", vmin=0, vmax=1.25)
    ax[0].set_yticks(range(cfg.n_bits_pcm))
    ax[0].set_yticklabels([f"b{cfg.n_bits_pcm-1-i}" + (" (MSB)" if i == 0 else " (LSB)" if i == cfg.n_bits_pcm - 1 else "")
                           for i in range(cfg.n_bits_pcm)])
    ax[0].set(xlabel="sample index n", title=f"(a) PCM bit matrix of the first {n_show} samples (dark = 1)")
    ax[0].grid(False)
    ax[1].axis("off")
    ax[1].grid(False)
    rows = [["n", "x[n]", "level", "PCM word"]]
    for n in range(8):
        rows.append([str(n), f"{tx.x_s[n]:+.4f}", f"{int(tx.q_idx[n]):3d}", "".join(map(str, bits[n]))])
    tb = ax[1].table(cellText=rows, loc="center", cellLoc="center", colWidths=[0.12, 0.3, 0.22, 0.36])
    tb.auto_set_font_size(False)
    tb.set_fontsize(9.5)
    tb.scale(1, 1.5)
    for j in range(4):
        tb[0, j].set_facecolor("#dbeafe")
        tb[0, j].set_text_props(fontweight="bold")
    ax[1].set_title("(b) Code words (natural binary, MSB first)")
    return save(fig, path)


# ----------------------------------------------------------------------------- 5. packet
def fig_packet(cfg: Config, tx, path):
    lay = frame_layout(cfg.n_payload_bits)
    names = ["preamble", "sync", "device_id", "command", "hop", "payload", "crc"]
    titles = ["Preamble", "Sync word", "Device ID", "Command", "Rolling\ncode", "PCM payload", "CRC-16"]
    bits = tx.packet_bits
    def hexof(a, b):
        v = int("".join(map(str, bits[a:b])), 2)
        return f"0x{v:0{(b - a + 3) // 4}X}"
    contents = [hexof(*lay["preamble"]), hexof(*lay["sync"]), hexof(*lay["device_id"]),
                hexof(*lay["command"]), hexof(*lay["hop"]), f"{cfg.n_samples} samples × {cfg.n_bits_pcm} bit",
                hexof(*lay["crc"])]
    widths = [1.2, 1.2, 1.5, 1.1, 1.5, 3.0, 1.2]
    cols = ["#e5e7eb", "#fde68a", "#bfdbfe", "#bbf7d0", "#ddd6fe", "#fbcfe8", "#fecaca"]
    fig, ax = plt.subplots(figsize=(12.5, 3.3), constrained_layout=False)
    ax.axis("off")
    ax.set_xlim(0, sum(widths) + 0.6)
    ax.set_ylim(-1.8, 2.2)
    x = 0.3
    for n, t, c, w, col in zip(names, titles, contents, widths, cols):
        a, b = lay[n]
        ax.add_patch(plt.Rectangle((x, 0), w, 1.2, fc=col, ec="#374151", lw=1.3))
        ax.text(x + w / 2, 0.82, t, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x + w / 2, 0.35, c, ha="center", va="center", fontsize=8.6, family="monospace")
        ax.text(x + w / 2, -0.35, f"{b - a} bit", ha="center", va="center", fontsize=9, color=C["gray"])
        x += w
    ax.annotate("", xy=(0.3, 1.75), xytext=(x, 1.75), arrowprops=dict(arrowstyle="<->", color="#374151"))
    ax.text((x + 0.3) / 2, 1.95, f"{lay['total'][1]} bits  =  {lay['total'][1]/cfg.bit_rate*1e3:.0f} ms on air at {cfg.bit_rate/1e3:.1f} kbit/s",
            ha="center", fontsize=10)
    xc = 0.3 + sum(widths[:2]) + 0.05
    ax.annotate("", xy=(0.3 + sum(widths[:2]) + sum(widths[2:6]), -0.95), xytext=(0.3 + sum(widths[:2]), -0.95),
                arrowprops=dict(arrowstyle="<->", color=C["red"]))
    ax.text(0.3 + sum(widths[:2]) + sum(widths[2:6]) / 2, -1.35, "protected by the CRC-16 (ID … payload)",
            ha="center", color=C["red"], fontsize=9.5)
    return save(fig, path)


# ----------------------------------------------------------------------------- 6. line coding
def fig_line_coding(cfg: Config, tx, path, n_bits=24):
    bits = tx.packet_bits[:n_bits]
    spc = 20
    fig, ax = plt.subplots(3, 1, figsize=(11.5, 6.0), sharex=True)
    t = np.arange(n_bits * spc) / spc
    ax[0].step(np.r_[t, n_bits], np.r_[np.repeat(bits, spc), bits[-1]], where="post", color=C["dark"])
    for k, b in enumerate(bits):
        ax[0].text(k + 0.5, 1.25, str(b), ha="center", fontsize=9)
    ax[0].set(ylim=(-0.3, 1.6), yticks=[0, 1], title="Packet bits (preamble 0xAAAA, first byte of the sync word 0x2D)")
    nrz = chips_to_levels(nrz_encode(bits), spc)
    ax[1].step(np.r_[t, n_bits], np.r_[nrz, nrz[-1]], where="post", color=C["blue"], lw=1.8)
    ax[1].set(ylim=(-0.3, 1.4), yticks=[0, 1], title="NRZ-L  (1 chip per bit)  - long runs of equal bits give no transitions")
    man = chips_to_levels(manchester_encode(bits), spc // 2)
    ax[2].step(np.r_[t, n_bits], np.r_[man, man[-1]], where="post", color=C["red"], lw=1.8)
    ax[2].set(ylim=(-0.3, 1.4), yticks=[0, 1], xlabel="bit period (1 bit = 104 µs at 9.6 kbit/s)",
              title="Manchester  (1 → 10, 0 → 01)  - a transition in every bit: self-clocking, DC-free, 50 % duty")
    for a in ax:
        for k in range(n_bits + 1):
            a.axvline(k, color=C["gray"], lw=0.4, ls=":")
        a.axvspan(0, 16, color="#fde68a", alpha=0.25)
        a.axvspan(16, n_bits, color="#bfdbfe", alpha=0.25)
    ax[0].text(8, -0.15, "preamble", ha="center", fontsize=9)
    ax[0].text(20, -0.15, "sync word", ha="center", fontsize=9)
    return save(fig, path)


# ----------------------------------------------------------------------------- 7. modulation
def fig_modulation(cfg: Config, tx, path, n_bits=6):
    spc = cfg.spc("manchester")
    n = n_bits * 2 * spc
    t = np.arange(n) / cfg.fs_rf * 1e3
    fig = plt.figure(figsize=(12.4, 5.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    a0, a1, axp = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0], sharex=None), fig.add_subplot(gs[:, 1])
    chips = chips_to_levels(tx.chips[: n_bits * 2], spc)
    a0.plot(t, chips, color=C["red"], lw=1.8)
    a0.set(ylabel="chip level", ylim=(-0.2, 1.3), xlim=(0, t[-1]), title="(a) Manchester chips of the first 6 preamble bits")
    a1.plot(t, tx.burst[:n], color=C["blue"], lw=1)
    a1.plot(t, chips, color=C["red"], lw=1, ls="--", alpha=0.7)
    a1.set(xlabel="time [ms]", ylabel="amplitude", ylim=(-1.3, 1.3), xlim=(0, t[-1]),
           title=f"(b) OOK: carrier on for a 1-chip, off for a 0-chip (fc = {cfg.fc/1e3:.1f} kHz)")
    f, p = welch(tx.burst, fs=cfg.fs_rf, nperseg=2048)
    pdb = 10 * np.log10(p / p.max() + 1e-12)
    axp.plot(f / 1e3, pdb, color=C["blue"], lw=1)
    axp.axvspan((cfg.fc - cfg.bandwidth_hz / 2) / 1e3, (cfg.fc + cfg.bandwidth_hz / 2) / 1e3,
                color=C["orange"], alpha=0.2, label=f"B = 4Rb = {cfg.bandwidth_hz/1e3:.1f} kHz")
    axp.axvline(cfg.fc / 1e3, color=C["red"], ls="--", lw=1, label="carrier fc")
    axp.set(xlabel="frequency [kHz]", ylabel="PSD [dB, normalised]", xlim=(0, cfg.fs_rf / 2e3),
            ylim=(-70, 3), title="(c) Spectrum of the whole burst")
    axp.legend(loc="upper right")
    return save(fig, path)


# ----------------------------------------------------------------------------- 8. channel
def fig_channel(cfg: Config, tx, d, path, n_bits=8):
    spc = cfg.spc("manchester")
    n = n_bits * 2 * spc
    lead = d["lead"]
    snrs = list(d["rx_view"].keys())
    fig, ax = plt.subplots(len(snrs), 1, figsize=(11.5, 2.5 * len(snrs)), sharex=True)
    t = np.arange(n) / cfg.fs_rf * 1e3
    for a, s in zip(np.atleast_1d(ax), snrs):
        a.plot(t, d["rx_view"][s][0, lead: lead + n], color=C["gray"], lw=0.8, label="received r(t)")
        a.plot(t, tx.burst[:n], color=C["red"], lw=0.9, alpha=0.85, label="transmitted s(t)")
        a.set(ylabel="amplitude", title=f"SNR = {s:+.0f} dB   (Eb/N0 = {s + 10*np.log10(cfg.fs_rf/(2*cfg.bit_rate)):.1f} dB)")
        a.set_ylim(-3.2, 3.2)
    np.atleast_1d(ax)[0].legend(loc="upper right", ncol=2)
    np.atleast_1d(ax)[-1].set_xlabel("time [ms]   (first 8 preamble bits; carrier phase shown at 0° for comparison)")
    return save(fig, path)


# ----------------------------------------------------------------------------- 9. demodulation
def fig_demod(cfg: Config, tx, d, path):
    spc = cfg.spc("manchester")
    lead, e, metric = d["lead"], d["energy"][0], d["res"].metric[0]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.0))
    n0, n1 = lead - 64, lead + 20 * 2 * spc
    ax[0, 0].plot(np.arange(n0, n1), e[n0:n1], color=C["blue"], lw=1)
    for k in range(0, 20 * 2 + 1):
        ax[0, 0].axvline(lead + k * spc, color=C["gray"], lw=0.4, ls=":")
    ax[0, 0].set(xlabel="sample index", ylabel="|y[n]|²  (chip energy)",
                 title="(a) Sliding integrate-and-dump output (preamble region)")
    ax[0, 1].plot(np.arange(metric.size), metric / metric.max(), color=C["purple"])
    ax[0, 1].axvline(lead, color=C["red"], ls="--", lw=1.2, label=f"true start = {lead}")
    ax[0, 1].plot(d["res"].t0[0], 1, "o", color=C["green"], label=f"estimated start = {d['res'].t0[0]}")
    ax[0, 1].set(xlabel="candidate start sample", ylabel="correlation (normalised)",
                 title="(b) Timing recovery: preamble + sync correlation")
    ax[0, 1].legend(loc="upper left")
    for a, (s, stats, bits) in zip(ax[1], [(d["snr_main"], d["res"].chip_stats[0], tx.packet_bits),
                                           (d["snr_low"], d["res_low"].chip_stats[0], tx.packet_bits)]):
        e1, e2 = stats[0::2], stats[1::2]
        a.scatter(e1[bits == 0], e2[bits == 0], s=7, color=C["blue"], label="bit 0 sent", alpha=0.7)
        a.scatter(e1[bits == 1], e2[bits == 1], s=7, color=C["red"], label="bit 1 sent", alpha=0.7)
        lim = max(e1.max(), e2.max()) * 1.05
        a.plot([0, lim], [0, lim], "k--", lw=1)
        a.text(lim * 0.62, lim * 0.08, "decide 1  (E1 > E2)", fontsize=9)
        a.text(lim * 0.02, lim * 0.9, "decide 0  (E1 < E2)", fontsize=9)
        a.set(xlabel="E1 = energy of 1st half-bit", ylabel="E2 = energy of 2nd half-bit", xlim=(0, lim), ylim=(0, lim),
              title=f"({'c' if a is ax[1, 0] else 'd'}) Decision statistics, SNR = {s:+.0f} dB")
        a.legend(loc="center right", markerscale=2)
    return save(fig, path)


# ----------------------------------------------------------------------------- 10. eye
def fig_eye(cfg: Config, tx, d, path):
    spc = cfg.spc("manchester")
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
    for a, key, s in zip(ax, ("energy", "energy_low"), (d["snr_main"], d["snr_low"])):
        e = np.sqrt(d[key][0])
        t0 = d["res"].t0[0] if key == "energy" else d["res_low"].t0[0]
        idx = t0 + np.arange(0, 600) [:, None] * (2 * spc) + np.arange(-spc // 2, 2 * spc + spc // 2)[None, :]
        tr = e[idx]
        x = np.arange(-spc // 2, 2 * spc + spc // 2) / spc
        for k in range(tr.shape[0]):
            a.plot(x, tr[k], color=C["blue"], alpha=0.07, lw=0.8)
        a.axvline(0, color=C["red"], ls="--", lw=1)
        a.axvline(1, color=C["red"], ls="--", lw=1)
        a.text(0.02, a.get_ylim()[1] * 0.02 + 0.5, "sample\nhalf-bit 1", color=C["red"], fontsize=8)
        a.text(1.02, a.get_ylim()[1] * 0.02 + 0.5, "sample\nhalf-bit 2", color=C["red"], fontsize=8)
        a.set(xlabel="time [chips]", title=f"Eye diagram of the envelope, SNR = {s:+.0f} dB")
    ax[0].set_ylabel("|y(t)|  (envelope)")
    return save(fig, path)


# ----------------------------------------------------------------------------- 11. frame parser
def fig_frame_parser(cfg: Config, tx, d, path):
    lay = frame_layout(cfg.n_payload_bits)
    fig, ax = plt.subplots(2, 1, figsize=(11.5, 5.4), gridspec_kw={"height_ratios": [1.3, 1]})
    names = ["preamble", "sync", "device_id", "command", "hop", "payload", "crc"]
    cols = ["#9ca3af", "#f59e0b", "#3b82f6", "#22c55e", "#8b5cf6", "#ec4899", "#ef4444"]
    for (nm, col) in zip(names, cols):
        a, b = lay[nm]
        ax[0].axvspan(a, b, color=col, alpha=0.18)
        if nm in ("payload", "crc"):
            ax[0].text((a + b) / 2 if nm == "payload" else b + 15, 1.18, nm.upper() if nm == "crc" else "PCM payload (3200 bit)",
                       ha="center" if nm == "payload" else "left", fontsize=8.5)
    ax[0].text(5, 1.18, "header (preamble, sync, ID, command, hop = 104 bit)", ha="left", fontsize=8.5)
    ax[0].text(5, 1.34, "", fontsize=1)
    cases = d["parser_cases"]
    for row, case in enumerate(cases):
        pos = np.flatnonzero(case["bits"] != tx.packet_bits)
        ax[0].vlines(pos, row * 0.5, row * 0.5 + 0.35, color=C["red"] if row else C["green"], lw=1.2)
        ax[0].text(lay["total"][1] + 20, row * 0.5 + 0.15,
                   f"SNR {case['snr']:+.0f} dB: {pos.size} bit errors", va="center", fontsize=9)
    ax[0].set(xlim=(0, lay["total"][1] + 700), ylim=(-0.1, 1.5), yticks=[],
              xlabel="bit position in the packet", title="(a) Where the bit errors fall inside the packet")
    ax[1].axis("off")
    ax[1].grid(False)
    rows = [["SNR", "sync word", "CRC-16", "device ID", "link-layer decision"]]
    for case in cases:
        r = case["res"]
        rows.append([f"{case['snr']:+.0f} dB", "OK" if r.sync_ok[0] else "FAIL", "OK" if r.crc_ok[0] else "FAIL",
                     "OK" if r.id_ok[0] else "FAIL", "ACCEPT → rolling-code check" if r.accepted[0] else "DROP packet"])
    tb = ax[1].table(cellText=rows, loc="center", cellLoc="center", colWidths=[0.12, 0.16, 0.16, 0.16, 0.32])
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.7)
    for j in range(5):
        tb[0, j].set_facecolor("#dbeafe")
        tb[0, j].set_text_props(fontweight="bold")
    for i, case in enumerate(cases, start=1):
        for j in range(1, 5):
            txt = tb[i, j].get_text().get_text()
            tb[i, j].set_facecolor("#dcfce7" if txt in ("OK",) or txt.startswith("ACCEPT") else "#fee2e2")
    ax[1].set_title("(b) Receiver checks")
    return save(fig, path)


# ----------------------------------------------------------------------------- 12. reconstruction
def fig_reconstruction(cfg: Config, tx, d, path):
    cases = d["recon_cases"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True, sharey=True)
    for a, c in zip(ax.ravel(), cases):
        a.plot(tx.t_fine * 1e3, tx.x_fine, color=C["blue"], lw=1.4, label="original x(t)")
        a.plot(tx.t_fine * 1e3, c["recon"], color=C["red"], lw=1.0, label="reconstructed x̂(t)")
        status = "packet ACCEPTED" if c["accepted"] else "packet DROPPED (CRC)"
        a.set_title(f"SNR = {c['snr']:+.0f} dB  -  {c['n_err']} bit errors - {status}\n"
                    f"MSE = {c['mse']:.2e}, reconstruction SNR = {c['snr_rec']:.1f} dB", fontsize=10)
    ax[0, 0].legend(loc="lower right", ncol=2, fontsize=8)
    for a in ax[1]:
        a.set_xlabel("time [ms]")
    for a in ax[:, 0]:
        a.set_ylabel("amplitude")
    ax[0, 0].set_ylim(-1.55, 1.55)
    return save(fig, path)
