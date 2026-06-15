# STRAND density moat — the sub-4-bit, on-device roadmap

_Strategic pivot, 2026-06-08._

## The pivot

- **Product surface = 3-bit-and-under.** 5/6/8-bit is **commodity** — GGUF and everyone else do near-lossless fine; there is no moat there. Drop 5-bit entirely.
- **q4 = the yardstick, not a product.** It's the incumbent that already runs on phones (llama.cpp 4-bit). We densify *past* it; we report it only as the line in the sand.
- **One high-bit point (q6 or q8) = the R-D ceiling**, kept purely for the rate-distortion science (how close to bf16 is even possible).

## The moat — a trinity nobody else has all three of

1. **Density** — the fewest bits/weight that still works (push below 4-bit into 2 and sub-2).
2. **Determinism** — float-free *integer* decode ⇒ **bit-identical on every device** (GGUF dequantizes through floats and drifts).
3. **Runs anywhere** — integer-only decode needs no GPU and no float unit: phone, browser (WASM), microcontroller, FPGA.

**North star:** _the densest fully-deterministic model that runs on-device._
- **Near-term sure thing:** 3-bit 7B = **2.6 GB**, fits a phone *today* (~8.2 PPL). The only missing piece is a packed integer kernel.
- **Moonshot:** sub-2-bit *usable* (7B < 2 GB) via the vector trellis.

## How we reach sub-2-bit — the bpw map

`payload_bpw = k / d`  (k = bits/symbol, d = vec_dim). So:

| k | d | bpw | 7B size |
|---|---|-----|---------|
| 4 | 2 | 2.0 | 1.75 GB |
| 3 | 2 | 1.5 | 1.31 GB |
| 3 | 3 | 1.0 | 0.88 GB |
| 2 | 2 | 1.0 | 0.88 GB |

Sub-2-bit = **small k, larger d**. Scalar (d=1) collapses at 2-bit (PPL 213); the **vector trellis (B1)** is the only determinism-preserving way down.

## Build A — vector-trellis density (the capability)

**Status:** built (`learned_codebook.rs` Lloyd/k-means trainer + `encode_tensor_with_lut_vec`) but **unstable** — 18.5% mean rel-RMS and a **47% outlier on `layers.25.o_proj`** (Lloyd diverges on some tensors).

**Critical-path fix — per-tensor non-regression guard:** for each tensor, encode with the *learned* LUT and with the *broadcast-scalar* LUT, measure both rel-RMS, and **keep the lower one**. This makes the vector path **≤ scalar everywhere, and better where it genuinely wins** — turning a sometimes-catastrophic lever into a safe, monotone one. Insertion point: the `--learned-codebook` path in `bin/quantize-model.rs` (wrap the encode in compare-and-pick); a flag records which LUT won per tensor.

**Then:** sweep d=2,3,4 × k=2,3,4 → map the sub-2-bit density/quality frontier. The cloud A100 is the **packing factory** — Lloyd training is the expensive *encode*; the result is a tiny deterministic blob.

## Build B — integer decode kernel (the deployment)

Decode is already float-free integer (phone-perfect). **Missing:** a *fast* packed kernel (decode trellis-coded weights → int matmul) so the dense model actually *runs* fast on-device. New crate `strand-decode-kernel` (portable SIMD + optional GPU). Benchmark: tokens/sec on-device vs llama.cpp 4-bit. **This is what converts every density gain into something on a phone.**

## Build order

1. **[critical] Build-A fallback** → usable, safe vector trellis (unlocks sub-2-bit).
2. **Density Colab** — sub-4-bit + sub-2-bit vector sweep, with density metrics (GB, "fits on X") on A100.
3. **Build-B kernel skeleton** + throughput bench.
4. **Scale-validate** (70B + lm-eval-harness) — *after* the levers, to prove them (validation, not capability).

## Honest gradient

- **3-bit-on-phone:** buildable **now** — kernel is the sole blocker, no research required.
- **sub-2-bit usable:** an **open problem** (2-bit collapses; vector trellis is bounded ~0.1–0.2 bpw). Credible angle, not guaranteed.

## The bigger vision

A **universal deterministic compression substrate for intelligence**: compress not just weights but **KV-cache + activations**, alongside the **media (image/video/audio/text)** the model consumes — one integer-deterministic layer. Compress once in the cloud, run *anywhere*, *identically*, at maximum density, cryptographically verifiable because it's reproducible by construction. 70B on a laptop · 7B on a phone · 1B on a watch — same artifact, same behavior, no float anywhere.

---

## ⭐ Philosophy reframe (2026-06-08): STRAND is the FORMAT for an inference engine

The standalone 3-bit-PPL chase is dead (9.17 floor, confirmed by `q3_diag_pass3 = 9.42`).
The reframe: **STRAND is not a "best quantizer" — it is the densest deterministic trellis
FORMAT + fast decode, the GGUF-equivalent for the `dismantle` inference engine** (same author;
dismantle's `paradigmshift.md` Part IV / V.3 / roadmap-#4 literally calls for a "custom trellis
format co-designed with a fast decode kernel" = STRAND, a QTIP-class trellis).

**The metric is no longer PPL.** It is dismantle's two north stars — **tokens/sec ↑ and
joules/token ↓**, both = **bytes/token** — plus **quality-good-enough** at the chosen bpw, plus
**determinism** (STRAND's unique, uncontested axis). STRAND 3-bit (~3.34 bpw) = ~26% fewer bytes
than Q4_K (4.5 bpw); 2-bit ≈ half. dismantle reads ~1.9 GB/token at Q4_K ⇒ STRAND directly cuts
its bandwidth ⇒ tps + J/tok win — **IF** the Metal trellis decode stays bandwidth-bound (the gate;
dismantle's own Q3_K kernel died compute-bound at 24% peak — the trap to clear). **This reframe
RESCUES STRAND:** it doesn't need to beat GGUF on PPL (it can't), only to be the densest
bytes/token that decodes fast — a winnable game.

### "STRAND finished" = ready to integrate (completion checklist)
1. ✅ **Encode** (`quantize-model`) — trellis quant works (RHT + Viterbi + frozen-LUT).
2. ⏳ **The `.strand` FORMAT** — mmap-ready, page-aligned archive: trellis weights + LUT + scales
   + metadata; a `tools/strand_bake`-style writer + a reader dismantle can mmap.
3. ⏳ **The decode kernel** — CPU reference DONE (`strand-decode-kernel`); **the Metal trellis-GEMV
   + batch-1 bandwidth validation = THE GATE** (decides the whole fusion).
4. ⏳ **Quality map (bpw → usable)** — the local + cloud runs provide it; picks the format's bpw
   target (3-bit = the safe point; sub-2-bit if salient/vector clear).
5. ⏳ **Integration shim into dismantle** — STRAND reader beside `src/gguf/`, Metal kernel beside
   `quant.metal`, wired via the `backend/` seam. (Only after the gate clears.)

### The current runs are STILL VALID under this reframe
The local + cloud runs map **bpw→quality**, which the format choice REQUIRES (the lowest usable
bpw). PPL is now a "good-enough floor" check, not a "beat-GGUF" target — nothing to stop or redo.
The cloud master already folds in the kernel bench (the decode-speed axis, now central). The data
is exactly what the format needs.

### #1 PRIORITY = the Metal trellis-decode kernel validation (= dismantle roadmap #4)
Port `strand-decode-kernel`'s decode to a Metal GEMV, measure batch-1 bandwidth-boundedness on
M-series. **One number decides if the fusion is a rocket (denser→faster) or the Q3_K trap
(denser→slower).** Build nothing format-side until this clears.
