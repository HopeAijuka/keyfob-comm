#!/usr/bin/env python3
"""Fills report/report.tex with the numbers of results/metrics.json and compiles report.pdf."""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "results/metrics.json").read_text())
S = np.genfromtxt(ROOT / "results/sweep_snr.csv", delimiter=",", names=True)

cfg, src, sw, sh, dist = M["config"], M["source"], M["sweep_summary"], M["shannon"], M["distance"]
off_eb = 10 * np.log10(cfg["fs_rf"] / (2 * cfg["bit_rate"]))
cap_at_psr90 = sh["B"] * np.log2(1 + 10 ** (sw["inband_psr90_db"] / 10)) / 1e3
bits_pt = cfg["n_sweep"] * cfg["packet_bits"]

def sci(x, d=2):
    m, e = f"{x:.{d}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"

rows = []
for snr in (-6, -4, -2, -1, 0, 1, 2, 3, 5):
    i = int(np.argmin(np.abs(S["snr_db"] - snr)))
    ber = "$0$ (none seen)" if S["ber_genie"][i] == 0 else sci(S["ber_genie"][i])
    mraw, mdel = S["mse_raw"][i], S["mse_delivered"][i]
    rows.append(f"{snr:+d} & {snr + off_eb:.1f} & {ber} & {S['psr'][i]:.3f} & {sci(mraw)} & {10*np.log10(src['signal_power']/mraw):.1f} & {sci(mdel)} \\\\")
table_sweep = "\n".join(rows)

ev_rows = []
for e in M["security"]:
    cnt = "--" if e["counter"] is None else str(e["counter"])
    dec = "\\textbf{ACCEPT}" if e["accepted"] else "REJECT"
    arrow = "$\\rightarrow$"
    reason = e["reason"].replace("->", arrow).replace("_", "\\_")
    note = e["note"].replace("->", arrow)
    ev_rows.append(f"{e['label']} & {e['who']} & {note} & {cnt} & {e['car_counter']} & {dec} & {reason} \\\\")
table_events = "\n".join(ev_rows)

dr = []
for k, d in enumerate(dist["distance_m"]):
    if int(d) in (10, 20, 30, 40, 50, 60):
        dr.append(f"{int(d)} & {dist['snr_inband_db'][k]:.1f} & {dist['psr_awgn'][k]:.2f} & {dist['psr_rician'][k]:.2f} & {dist['psr_rayleigh'][k]:.2f} \\\\")
table_dist = "\n".join(dr)

def r90(k):
    v = dist.get("range90_" + k)
    return f"{v:.0f}" if v else "n/a"

V = {
 "N_SWEEP": cfg["n_sweep"], "N_DIST": cfg["n_dist"], "BITS_PT": f"{bits_pt/1e6:.2f}",
 "SQNR": f"{src['sqnr_db']:.2f}", "SQNR_TH": f"{src['sqnr_theory_db']:.2f}", "MSEQ": sci(src["mse_quant"]),
 "DELTA": f"{src['delta']:.5f}", "AIRTIME": f"{cfg['airtime_ms']:.0f}",
 "PSR90": f"{sw['snr_psr90_db']:+.1f}", "PSR90_EB": f"{sw['ebn0_psr90_db']:.1f}", "PSR90_B": f"{sw['inband_psr90_db']:.1f}",
 "PSR50": f"{sw['snr_psr50_db']:+.1f}",
 "PSTAR": sci(sw["p_star"], 1), "SNRSTAR": f"{sw['snr_star_sim_db']:+.1f}", "EBSTAR": f"{sw['ebn0_star_db']:.1f}", "BSTAR": f"{sw['inband_star_db']:.1f}",
 "SH_SNR": f"{sh['shannon_snr_b_db']:.1f}", "SH_SNR_SRC": f"{sh['shannon_snr_b_src_db']:.1f}",
 "SH_EB_OOK": f"{sh['shannon_ebn0_ook_db']:.2f}", "SH_EB_BPSK": f"{sh['shannon_ebn0_bpsk_db']:.2f}",
 "REQ_OOK": f"{sh['ook_req_ebn0_db']:.1f}", "REQ_BPSK": f"{sh['bpsk_req_ebn0_db']:.1f}",
 "GAP_OOK": f"{sh['gap_ook_db']:.1f}", "GAP_BPSK": f"{sh['gap_bpsk_db']:.1f}",
 "GAP_SNR": f"{sw['inband_psr90_db'] - sh['shannon_snr_b_db']:.1f}", "CAP90": f"{cap_at_psr90:.0f}",
 "R_RAYLEIGH": r90("rayleigh"), "R_RICIAN": r90("rician"), "R_AWGN": r90("awgn"),
 "NF": f"{dist['noise_floor_dbm']:.1f}",
 "TABLE_SWEEP": table_sweep, "TABLE_EVENTS": table_events, "TABLE_DIST": table_dist,
 "OFF_EB": f"{off_eb:.2f}",
}

tex = (ROOT / "report/report_template.tex").read_text()
for k, v in V.items():
    tex = tex.replace(f"@{k}@", str(v))
left = [w for w in tex.split("@") if w.isupper() and len(w) > 2 and "_" in w or w in V]
(ROOT / "report/report.tex").write_text(tex)
import re
miss = re.findall(r"@[A-Z_0-9]+@", tex)
if miss:
    sys.exit(f"unfilled placeholders: {set(miss)}")
for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], cwd=ROOT / "report", capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-3000:]); sys.exit("pdflatex failed")
shutil.copy(ROOT / "report/report.pdf", ROOT / "report.pdf")
print("report.pdf written")
