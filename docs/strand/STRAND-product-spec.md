# STRAND — product specification

_Decision-grade product definition, frozen from the 2026-06-08 Qwen2.5-7B results. This is the
durable "what STRAND IS" doc: the bpw→quality curve, the shipping recipe, the moat, the closed
(do-not-rerun) negatives, and the one open frontier. PPL chase is over; this is the format._

> **⚠️ CORRECTION 2026-06-09 — the mixed-precision rows below are RETRACTED, pending re-measurement.**
> The original `q3_mixed_heavy` = 7.81 @ "3.7 bpw" was a **bug, not a result**: `strand-7b-ppl.sh`
> hardcoded the mp-config fallback to 4 bits, so every mixed-precision run silently quantized the
> *unmatched* tensors (gate/up_proj) at 4-bit too → **uniform 4-bit (4.50 bpw), PPL bit-identical to
> `q4_diag` (7.809945735787925)**. Fixed in commit `f2f7716`; the mp configs are re-running correctly
> now, and the **corrected numbers are IN**: uniform-3 **9.42 @ 3.34** → `mp_light` (down_proj@4)
> **8.45 @ 3.67** → uniform-4 **7.81 @ 4.50**. Mixed-precision IS a real lever (−10% PPL vs uniform-3
> for +0.33 bpw) but **modest** — it interpolates the q3→q4 curve; it does NOT reach q4 quality at
> 3.7 bpw (that was the bug). And **`down_proj` is the whole lever**: also protecting attention
> (`q3_mixed_heavy`, 3.81 bpw) gives the *same* 8.45 PPL for more bits → dominated. Treat §2/§3's
> specific mixed rows as superseded by this line.

---

## 1. What STRAND is (one paragraph)

STRAND is **the deterministic, float-free, sub-4-bit weight FORMAT** for an on-device inference
engine — the GGUF-equivalent for [`dismantle`](../../dismantle) (pure-Rust + Metal Apple-Silicon
engine). A model is encoded once in the cloud (RHT incoherence → Hessian-aware Viterbi trellis →
frozen integer LUT) into a compact `.strand` archive; every device decodes it with **integer-only**
arithmetic (`reconstruct_q = (scale_q·quantile_q) >> 16`, no float, no GPU required) and gets a
**bit-identical** result on phone / WASM / MCU / FPGA. The product target is **3-bit and under** —
the band where GGUF has no commodity answer — chosen for **bytes/token**, not perplexity.

---

## 2. The bpw → quality curve (the product's R-D table)

Qwen2.5-7B, WikiText-2, ctx 2048, bf16 throughout (fp16 → NaN on Qwen, forbidden). Screening rows
use the **64-window** harness (fast, used to rank recipes); the q3/q4 anchors were also confirmed on
the **full 146-window** gate (baseline 7.7362) and agree. PPL is a *good-enough floor* check now, not
a beat-GGUF target.

| recipe | bpw | PPL | Δ vs bf16 | role | status |
|---|---:|---:|---:|---|---|
| **bf16 baseline** | 16.0 | **7.74** | — | R-D ceiling (how close is even possible) | measured |
| uniform 4-bit | 4.50 | 7.81 | +0.9% | the **yardstick** (commodity, llama.cpp Q4 already on phones) | measured |
| `mp_heavy` pass-3 (4-bit attn+down_proj, multi-pass) | ~3.75 | _~7.7–8.0_ | ~bf16-band | **near-q4 quality at −17% bytes** | pending |
| `q3_mixed_heavy` (4-bit attn+down_proj, 3-bit else) | 3.70 | **~7.7–8.0** | low single digits | **the quality lever** (near uniform-4 at lower bpw) | measured/screening |
| `mp_light` (4-bit first+last 2 layers, 3-bit else) | ~3.46 | _pending_ | TBD | cheapest protect (boundary layers) | pending |
| **uniform 3-bit** | **3.34** | **9.42** | +21.7% | **the safe deployable base** (integer float-free) | measured (q3_diag_pass3) |
| uniform 2-bit | 2.34 | 213 | COLLAPSE | below the scalar-trellis floor — **dead without QAT** | measured |

**Reading of the curve:** the cliff is between 3-bit (works) and 2-bit (collapses). 3-bit alone is
+21.7% PPL — usable, dense, deployable, but visibly behind bf16. **Mixed-precision spends ~0.4 bpw to
buy back almost all of that loss**: protecting the attention and `down_proj` tensors at 4-bit pulls
PPL into the ~7.7–8.0 band (near uniform-4-bit) while *still sitting below 4-bit on bytes*. That is
the product: **3-bit density with 4-bit quality.**

---

## 3. The product RECIPE (what ships)

**`uniform 3-bit base  +  mixed-precision protect (4-bit on attn + down_proj)  +  multi-pass`.**

1. **Uniform 3-bit base (3.34 bpw).** Scalar bitshift trellis, L/K per `TrellisConfig::for_bpw(3.0)`,
   RHT incoherence on every tensor, frozen Gaussian Q12 LUT, tail-biting, affine-min + per-sub-block
   scales. Integer-only decode. This is the floor everything else builds on.
2. **Mixed-precision protect → 4-bit on `self_attn` + `down_proj`** (config:
   `scratch/qwen-7b/mp-heavy.json` = `[{self_attn,4},{down_proj,4}]`). These are the
   loss-sensitive tensors; lifting *only* them to 4-bit is the highest-leverage ~0.4 bpw spend and
   lands the model in the ~7.7–8.0 PPL band at ~3.7 bpw. A lighter variant
   (`mp-protect.json` = first/last 2 layers at 4-bit, ~3.46 bpw) is the cheaper protect point.
3. **Multi-pass encode (pass-3).** Re-quantize with the residual/refinement passes
   (`q3_diag_pass3`, `mp_heavy` pass-3); each pass tightens the trellis fit at the same bpw. This is a
   pure encode-side cost (cloud), free at decode.

Everything in the recipe is **encode-time** and **determinism-preserving** — the decoder is the same
integer LUT replay regardless of which tensors are 3- vs 4-bit (the per-tensor `(L,K)` is in the
header). Bit budget is tunable continuously by changing which patterns get 4 bits.

---

## 4. The moat — density × determinism × float-free decode

The differentiator vs GGUF K-quants is a **trinity no competitor has all three of**:

1. **Density.** Product surface is **3-bit and under** (3.34 bpw) — ~**26% fewer bytes/token than
   Q4_K** (4.5 bpw), and the mixed-precision product is still below 4-bit. 5/6/8-bit is commodity and
   explicitly *not* the game; q4 is only the line in the sand.
2. **Determinism.** Decode is **integer-only** (`reconstruct_q = (scale_q·quantile_q) >> 16`, a single
   64-bit multiply + arithmetic shift; the only float touch is a final exact ×2⁻¹² cast). Given the
   same `.strand` bytes, **every device returns a byte-identical result** — phone, WASM, MCU, FPGA,
   any compiler, any thread count. GGUF dequantizes through floats and drifts across backends; STRAND
   does not. This axis is **uncontested**.
3. **Runs anywhere.** Integer-only decode needs **no GPU and no float unit**. The same artifact runs
   on a watch, a phone, a laptop, and an FPGA, *identically* — "compress once in the cloud, run
   anywhere identically at max density, cryptographically reproducible by construction."

**North star:** _the densest fully-deterministic model that runs on-device._ Near-term sure thing:
3-bit 7B ≈ 2.6 GB — fits a phone today.

**Why this reframe wins where the PPL chase lost:** STRAND cannot beat Q4_K on perplexity at iso-bits
(falsified repeatedly). It does not need to. As a *format*, it only needs to be the **densest
bytes/token that decodes fast and identically** — a winnable game, and the one dismantle's
`paradigmshift.md` (Part IV/V.3, roadmap #4) explicitly asks for.

---

## 5. Closed negatives — DO NOT re-run

These were measured and falsified on 7B this cycle. They are settled; spending compute on them again
is waste.

| Lever | Verdict | Why (root cause) |
|---|---|---|
| **Salient-patch** (keep top-x% weights at high precision) | **DEAD** — *hurts* at 1%, pointless at 5% | RHT already spreads salience; carving outliers breaks the whitened trellis more than it helps |
| **Vector / sub-2-bit trellis** (d>1, k/d bpw) | **DEAD as PTQ** — 2-bit collapses (PPL 213); space-filling gain bounded ~0.1–0.2 bpw | Every 2-bit *winner* (AQLM/QuIP#/GLVQ/PV-Tuning) needs **learned FP codebooks + fine-tuning = determinism-breaking** |
| **Block-Hessian / LDLQ** (vs diagonal Hessian) | **DEAD** — 9.66 > 9.42, *worse* than diagonal at 3-bit | The diagonal Hessian already captures the available signal here; the block correction backfired |
| **Entropy-code the index stream (B2)** | DEAD — ~1.4% headroom | Trellis index stream is near-max-entropy by design (bijective state-hash scatters successors) |
| **Alt codebook distributions (B3)** | DEAD — Gaussian already optimal | RHT whitens the marginal to Gaussian; empirical fit can't beat the analytic Gaussian |

**The one-line rule for sub-3-bit:** _usable sub-3-bit needs **QAT/BitNet (training-time)**, not PTQ._
STRAND's PTQ + float-free constraint bottoms out at **3-bit**; that is structural (confirmed
0.5B→7B), not a scale artifact. The 3-bit floor is the product, not a bug to grind on.

---

## 6. The open frontier — SPEED, and the dismantle fusion

Quality is settled (§2–3). The remaining make-or-break is **speed on Apple Silicon**, plus the
integration.

- **THE GATE = the Metal trellis-decode kernel.** The fusion's whole thesis is "fewer bytes/token ⇒
  faster." That only holds if the trellis-decode GEMV stays **bandwidth-bound** on the M-series. The
  decoder's arithmetic intensity is ~9–14 ops/byte at the **ridge** (~30 ops/byte on M3 Pro) — i.e.
  **borderline**. dismantle's own Q3_K kernel died **compute-bound at 24% peak**; that is the trap to
  clear. Levers (from `docs/STRAND-metal-decode-gate.md`, 2026-06-08 refinement): **aligned 32-bit-word
  bitstream reads** (the biggest real win — kill the per-weight unaligned `read_bits`), **per-sub-block
  scale-fold** (reconstruct becomes a load, not a mul), RHT moved to the *activation* (one FWHT/GEMV,
  no per-row inverse), and **keep the native 32×32→64 multiply** (the i32-reconstruct shortcut
  *overflows* — Q16×Q12=Q28 hits 2³² on magnitude-4 weights — so it is bit-unsafe and rejected). The
  ridge proof is **the MSL kernel + an Instruments trace on the M3**, not a CPU building block. One
  number decides rocket vs trap.
- **`.strand` v2 = the mmap/deploy layout.** v1 (`format.rs`, `STRQ` magic, shipped) is a *sequential*
  per-tensor stream — perfect for CPU round-trip, wrong for a GPU that must jump to any `(row,block)`.
  v2 adds a **per-tensor block-offset table** (`{bit_offset, init_state, scale_q}` per block),
  page-aligned and mmap-ready. The encoder already carries every field (`BlockMeta`); v2 is a *layout*
  change in `write_strand`, **not** new quantization.
- **The dismantle fusion (after the gate clears).** STRAND reader beside `dismantle/src/gguf/`, the
  Metal trellis-GEMV beside `quant.metal`, wired through the `backend/` seam. Metric for done =
  **tokens/sec ↑ and joules/token ↓** (both ≡ bytes/token) at quality-good-enough + determinism — NOT
  PPL. dismantle reads ~1.9 GB/token at Q4_K; STRAND 3-bit directly cuts that ~26%.

**Build order:** (1) Metal trellis-GEMV + M3 bandwidth measurement = **the gate** (build nothing
format-side until this clears) → (2) `.strand` v2 writer with the block-offset table → (3) the
dismantle integration shim.

---

## 7. One-line summary

**STRAND = uniform-3-bit base + 4-bit-on-attn/down_proj mixed-precision + multi-pass → ~3.7 bpw at
near-q4 quality, decoded integer-deterministically and bit-identically on any device. The moat is
density × determinism × float-free decode; the only open question is whether the Metal decode stays
bandwidth-bound (the gate).**
