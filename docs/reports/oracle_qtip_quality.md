# Oracle — QTIP byte-cut QUALITY (L1.5 reframe, axis-2 deep byte-cut)

**Model:** `models/qwen2.5-3b-instruct-q4_k_m.gguf`  **Lane:** CPU NumPy  **Target store:** ~3.0 bits (~3.06 eff, ~98 B/256-blk)
**Date:** 2026-05-31

> **Scope = DIRECTION-ONLY local proxy.** The DECISIVE quality verdict (recon-RMSE vs **f16** + logit-cosine/KL/argmax on code) is **Colab** — QTIP must be fit **from f16**, never requant-from-Q4_K (a recorded kill), and the f16 Qwen + a forward pass are not on this machine. Numbers below are grounded in REAL Qwen weights but are a first cut, not the gate.

> **QTIP quality is BRACKETED** [lower, upper] = RHT + optimal {k, k+1}-bit scalar quantizer, both at the SAME ~k-bit byte cut. The trellis coding gain (≤~1 bit, QTIP/TCQ literature) lives inside this interval; the proxy models RHT exactly and refuses to invent a single trellis number ('a wrong trellis sim is worse than none').

## (i) RHT-Gaussianization headroom on real weights

How much outlier tax the incoherence rotation removes — QTIP's core lever. Excess-kurtosis and max/mean are distribution-SHAPE stats (robust to Q4 dequant noise; not requant-from-Q4). Gaussian ref: exkurt 0, max/mean ~6 per 256-block.

| tensor | shape | disk | raw exkurt | RHT exkurt | raw max/mean | RHT max/mean |
|---|---|---|---|---|---|---|
| blk.0.attn_q.weight | 2048x2048 | Q4_K | 2.52 | 1.12 | 26.1 | 14.8 |
| blk.0.ffn_gate.weight | 11008x2048 | Q4_K | 0.57 | 0.23 | 27.8 | 13.9 |
| blk.0.ffn_down.weight | 2048x11008 | Q6_K | 0.37 | 0.09 | 22.1 | 8.0 |
| blk.17.attn_output.weight | 2048x2048 | Q4_K | 2.08 | 0.98 | 51.0 | 29.7 |
| blk.17.ffn_up.weight | 11008x2048 | Q4_K | 0.51 | 0.19 | 28.5 | 21.8 |
| blk.35.attn_q.weight | 2048x2048 | Q4_K | 0.47 | 0.27 | 11.3 | 8.1 |
| blk.35.ffn_down.weight | 2048x11008 | Q6_K | 2.43 | 0.09 | 74.6 | 11.0 |

**Median reduction:** excess-kurtosis −0.34, max/mean −13.9. RHT materially Gaussianizes (the precondition for a low-bit quantizer to hit its rate-distortion target), but real weights stay somewhat heavier-tailed than ideal Gaussian after a single 256-RHT.

## (ii) Q4_K vs QTIP RMSE — clean bootstrap of the real marginal

Fresh i.i.d. resample of each tensor's real weight values (clean f32 -> Q4_K pays real quant error incl. the outlier tax; **not** the forbidden requant-from-already-Q4). QTIP[lower,upper] at ~98 B/256-blk vs Q4_K_M (NumPy, 4.5 bits, 144 B).

| tensor | Q4_K_M rmse | QTIP lower | QTIP upper | lower≤Q4K | upper≤Q4K |
|---|---|---|---|---|---|
| blk.0.attn_q.weight | 0.0852 | 0.1830 | 0.0957 | no | no |
| blk.0.ffn_gate.weight | 0.0804 | 0.1855 | 0.0980 | no | no |
| blk.0.ffn_down.weight | 0.0797 | 0.1865 | 0.0983 | no | no |
| blk.17.attn_output.weight | 0.0839 | 0.1844 | 0.0965 | no | no |
| blk.17.ffn_up.weight | 0.0815 | 0.1854 | 0.0973 | no | no |
| blk.35.attn_q.weight | 0.0808 | 0.1878 | 0.0991 | no | no |
| blk.35.ffn_down.weight | 0.0803 | 0.1825 | 0.0955 | no | no |

**Median:** Q4_K_M 0.0808 vs QTIP [0.1854, 0.0973] at −32% bytes. QTIP-lower ≤ Q4_K on **0/7** tensors; QTIP-upper ≤ Q4_K on **0/7**.

**Bits-equivalent gap:** to MATCH Q4_K_M weight-RMSE at this byte budget, QTIP must extract **~1.20 bits** of combined RHT-whitening + trellis coding gain over a 3-bit scalar (each ~1 bit ≈ 6 dB ≈ ×½ RMSE). That is at the **upper edge** of the TCQ envelope (~0.5–1 bit typical, ~1.2 deep) — the modeled +1-bit upper bound still falls ~20% short. So matching Q4_K_M on RMSE is possible only if the real trellis lands near its best-case gain AND RHT whitens more than the single 256-rotation here.

## Direction read (NOT the verdict)

- **Cautionary direction.** Even the UPPER bound (full ~1-bit trellis gain) trails Q4_K_M on 7/7 tensors. RHT + ~3-bit does not cover the 1.5-bit deficit on this proxy. The Colab f16 gate must clear a real margin or QTIP's quality Type-1 fires (dead_levers entry; closes the byte-cut axis with the §5.0/§6 speed gate).
- **Caveats (why direction-only):** (1) bootstrap destroys per-channel spatial structure; (2) RMSE ≠ logit quality — the decisive metric is logit-cosine/KL/argmax on code (Colab); (3) the QTIP source here is a resample of Q4_K-dequant values, not true f16 — the f16 source is the only fair quantizer input (kill-respect #2); (4) the Q4_K_M baseline is a faithful NumPy reimplementation (gguf has no K-quant quantizer), a touch below llama.cpp's importance-weighted optimum.

## How the DECISIVE (Colab) gate fires — `--colab` runbook

On Colab, with f16 Qwen2.5-3B + a code corpus:
1. Export per-tensor f16 weights -> `weights.npz`; run `oracle_qtip_quality.py --colab weights.npz` for **recon-RMSE vs f16** (QTIP-3.0/3.25 vs real Q4_K_M; **GO floor: QTIP ≤ Q4_K_M at fewer bytes**). Swap in the REAL QTIP quantizer for the codec there.
2. Forward-pass f16 / Q4_K_M / QTIP on held-out **code** tokens; export next-token logits -> three `.npy`; run `--colab ... --logits f16 q4k qtip` for **logit-cosine / KL / argmax** (GO: QTIP ≥ Q4_K_M cosine & argmax, ≤ KL).
3. GO on both -> §5.2 on-GPU decode-cost oracle (trellis BW-bound on M3?). NO-GO on either -> quality Type-1 kill; byte-cut axis closes.

_Wall: 8.8s. Peak RSS: 1.59 GB. Run `--selftest` (must pass) for these numbers to be trustworthy._
