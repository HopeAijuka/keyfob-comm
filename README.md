# End-to-End Communication System Simulation - Automotive Key-Fob Link

**Practical 8 mini-project. Language: Python 3 (3.10 or newer) - not MATLAB.**

A complete simulated communication chain, in the setting of a remote keyless entry (RKE) key fob:

```
Source -> Sampling -> Quantization / PCM -> Packet framing (device ID, rolling code, CRC-16)
       -> Line coding (Manchester) -> Digital modulation (OOK) -> Channel (path loss, fading, AWGN)
       -> Demodulation (timing sync + non-coherent detector) -> Decoding (frame parser, rolling-code
          check, PCM decoder) -> Reconstructed signal  (+ BER / PSR / MSE analysis, Shannon discussion)
```

The message is the guideline's test signal (300 Hz sine, `Fs = 8 kHz`, 8-bit PCM, 50 ms). Its PCM bits are the payload field of a
key-fob packet, so that every block of the guideline chain is present *and* the system behaves like a real fob (OOK, Manchester, preamble/sync,
rolling code, CRC). See `report.pdf` for the full explanation.

## How to run

```bash
pip install -r requirements.txt          # numpy, scipy, matplotlib
python main.py                           # full run, about 2-3 minutes
python main.py --quick                   # fast preview (fewer Monte-Carlo packets)
python -m unittest discover -s tests -v  # 12 unit tests
```

`main.py` is the single entry point: it runs the whole chain, produces all 19 figures in `figures/` and the numeric results in `results/`
(`metrics.json` and CSV tables). Results are reproducible (seed 2026 in `functions/config.py`).

To rebuild the written report (needs a LaTeX installation): `python report/build_report.py`. The executed notebook version is `report.ipynb`
(`python report/make_notebook.py` regenerates it, or simply *Run all* in Jupyter after `python main.py`).

## Repository structure

```
main.py                 entry point - runs source -> ... -> reconstruction and all analyses
functions/              building blocks (one module per block)
    config.py           every parameter of the experiment
    source.py           message signal + sampler
    quantizer.py        uniform mid-rise 8-bit quantizer
    pcm.py              natural-binary PCM encoder / decoder
    framing.py          packet format, CRC-16/CCITT, frame parser
    rolling_code.py     KeeLoq-style hop code, key fob and vehicle receiver
    line_coding.py      NRZ-L and Manchester
    modulation.py       OOK (and BPSK reference)
    channel.py          AWGN, flat fading, path loss / link budget
    demodulation.py     down-conversion, integrate-and-dump, timing recovery
    receiver.py         full receiver (sync -> detect -> parse -> checks)
    reconstruction.py   D/A (zero-order hold) + low-pass filter
    metrics.py          MSE, SQNR, BER/PSR theory, Shannon capacity
    analysis.py         Monte-Carlo sweeps and the security demo
    plots_*.py          figure code
figures/                saved plots of every stage and every result (01 ... 19)
results/                metrics.json, sweep_snr.csv, sweep_distance.csv, ber_bpsk.csv
tests/                  unit tests
report.pdf              project report (block diagram, plots at every stage, MSE vs SNR, Shannon)
report.ipynb            the report as an executed notebook
report/                 LaTeX source and scripts that build report.pdf / report.ipynb
```

## Key parameters

| Item | Value |
|---|---|
| Message | 300 Hz sine, 50 ms, `Fs = 8 kHz` (400 samples), 8-bit PCM (3200 bits) |
| Packet | preamble 16 + sync 16 + ID 32 + command 8 + rolling code 32 + payload 3200 + CRC 16 = 3320 bits (346 ms) |
| Bit rate / line code | 9.6 kbit/s, Manchester (19.2 kchip/s) |
| Modulation | OOK on a *scaled* carrier `fc = 38.4 kHz` (real fobs: 315 / 433.92 MHz), simulation rate 307.2 kHz |
| Channel SNR | defined like MATLAB `awgn(x, SNR, 'measured')`; `Eb/N0 = SNR + 12.04 dB`; in-band SNR (B = 4 Rb) = `SNR + 6.02 dB` |

## Main results (400 packets per SNR point)

* Simulated BER follows theory (`1/2 exp(-Eb/2N0)` for OOK-Manchester, `Q(sqrt(2Eb/N0))` for the BPSK reference).
* Reconstruction MSE sits at the 8-bit quantization floor (6.3e-6, 49.0 dB) for channel SNR >= +3 dB, then collapses within a few dB
  (channel MSE equals quantization MSE at BER 4.7e-6, SNR = +1.6 dB, Eb/N0 = 13.6 dB).
* 90 % packet success needs SNR = +0.9 dB (in-band 6.9 dB), about 14 dB above the Shannon limit (-7.2 dB) for the same rate and bandwidth:
  the channel *capacity* is not the bottleneck, uncoded OOK is; at high SNR the 8-bit quantizer is.
* Rolling code: replayed, forged and unpaired packets are rejected in a full waveform simulation.
* 90 % range with the illustrative link budget: about 40 m (AWGN), 30 m (Rician), 22 m (Rayleigh).

## Honest notes

* The carrier is scaled down for simulation speed; error performance depends on Eb/N0, not on fc.
* The hop-code cipher is *KeeLoq-style* (written from the algorithm structure, not validated against official test vectors) and is for teaching only.
* Link-budget numbers (transmit power, path-loss exponent, shadowing, noise figure) are illustrative assumptions.
