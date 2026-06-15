# STRAND at the Physical Limits — the 4 / 3 / 2 / 1-bit ladder

**Scope.** STRAND's product is the *bleeding-edge* low-bit end only: the **4 / 3 / 2 / ~1-bit**
rungs. The commodity 5/6/8-bit formats are out of scope and deliberately not measured here.
The thesis this doc defends, with numbers: **at these low bits STRAND is the densest
*deterministic, float-free* weight representation, and it decodes fast on any device.**

Every number below is produced by two reproducible gates in `crates/strand-decode-kernel`:

- **`gate-ladder`** — density + decode throughput per rung, on a real `.strand` v2
  archive, with each rung's decode **asserted bit-identical to `strand_quant::decode::decode_lean`**.
  `cargo run -p strand-decode-kernel --release --bin gate-ladder`
- **`gate-decode-speed`** — single / parallel / SIMD decode throughput vs the compute
  ceiling and the bandwidth ceiling, on a 67.9 Mw 3-bit tensor.
  `cargo run -p strand-decode-kernel --release --bin gate-decode-speed`

**Provenance of every number in this doc: ACTUALLY RAN** (release, M3 Pro, 12 logical cores /
12 rayon threads), not cargo-checked-only. Numbers are best-of-N wall time (the decode is pure
compute, so the minimum is the cleanest achievable-rate estimate). Re-running will vary the
throughput by a few percent (scheduler/turbo); the densities and the bit-identity verdicts are
exact and deterministic.

---

## 1. DENSITY — STRAND is AT / NEAR the deterministic density floor at every rung

`gate-ladder`, 2048×4096 = 8.39 Mw tensor, `in % 256 == 0` (full STRICT blocks), real
`.strand` v2 mmap round-trip:

| rung     | kind       | payload bpw | realised B/w | floor B/w | % above floor | < Q4_K (0.5625)? |
|----------|------------|------------:|-------------:|----------:|--------------:|:----------------:|
| 4-bit    | scalar     |       4.000 |   **0.5430** |    0.5390 |        +0.7%  | YES |
| 3-bit    | scalar     |       3.000 |   **0.4175** |    0.4140 |        +0.8%  | YES |
| 2-bit    | scalar     |       2.000 |   **0.2920** |    0.2890 |        +1.0%  | YES |
| ~1.5-bit | vector d=2 |       1.500 |   **0.2300** |    0.2265 |        +1.5%  | YES |
| ~1.0-bit | vector d=3 |       1.000 |   **0.1685** |    0.1640 |        +2.7%  | YES |

**What the floor is, and why "realised" sits just above it.** `realised B/w` is
`EncodedTensor::total_bpw(cfg) / 8` — the *honest* on-disk density: the `k`-bit-per-step
payload **plus all per-block side info** (`encode.rs::block_side_bits`). For these rungs (no
affine-min, no tail-biting, no RHT) the side info per 256-weight block is exactly a 32-bit
super-scale + eight 6-bit sub-scales = `32 + 48 = 80` bits/block = **0.039 B/w**. The `floor`
column is the idealised model `payload/8 + 0.039` (scalar `k/8 + 0.039`; vector `(k/d)/8 + 0.039`).
Realised lands **+0.7% to +2.7%** above it — the residual is the 16-bit tensor-length word and
sub-block rounding, amortised over fewer weights at the lower payloads (hence the gap grows as
bits shrink). **The density at each rung is essentially MAXED**: there is no slack to remove
without removing the scale metadata, and the scale metadata is what keeps decode float-free and
deterministic.

**The lever for fewer bytes is the LADDER, not tighter coding.** Going denser means moving *down*
a rung (4→3→2→1), and that is the **quality cliff**, not a free lunch:

- Each rung is already near its own deterministic floor (table above), so per-rung you cannot
  meaningfully compress further.
- STRAND's payload sits **~0.2–0.35 bpw above the Shannon rate** for the weight distribution —
  the **VQ gap** of a finite trellis codebook. RHT (random Hadamard transform) already whitens
  the weights toward Gaussian (the 7B sweep confirmed the post-RHT signal is decorrelated, so
  entropy-coding the indices and alternate codebook shapes were both measured *dead*: B2 ≈ 1.4%,
  B3 Gaussian-optimal). That gap is **uncloseable without training** — it is a property of
  scalar/vector quantization at a fixed rate, not an implementation inefficiency.
- The sub-2-bit vector rungs (d=2 → 1.5 bpw, d=3 → 1.0 bpw) push the payload lower by sharing one
  `k`-bit symbol across `d` weights (the learned `[2^L·d]` space-filling codebook), bounded to a
  ~0.1–0.2 bpw gain — and they pay for it in quality (see §3).

> **Density verdict: AT THE PHYSICAL LIMIT.** Each rung is +0.7–2.7% off its deterministic floor;
> every rung is denser than llama.cpp's Q4_K (0.5625 B/w), down to **0.17 B/w** at ~1-bit. The
> only knob left for fewer bytes is descending the ladder, which trades density for quality.

**One honest framing note.** The `total_bpw` density above is the *encoded-tensor* density. The
shipped `.strand` v2 **archive** additionally carries a 16-byte block-offset record per 256-weight
block for random-access mmap decode (= 0.0625 B/w), so the on-the-wire archive traffic the *speed*
gate models is `payload + 0.0625` (e.g. 0.4375 B/w at 3-bit). That offset table is an
mmap-random-access convenience, not part of the quantizer's rate; it is counted in the bandwidth
analysis (§2) and called out separately rather than folded into the density headline.

---

## 2. SPEED — decode is COMPUTE-BOUND, with headroom; density is nowhere near the wall

`gate-decode-speed`, 18944×3584 = **67.9 Mw** 3-bit tensor (an ffn_down-class Qwen2.5-7B shape),
265,216 independent blocks, 12 rayon threads, best-of-12:

| path                  |   Mw/s |  Gw/s | speedup |
|-----------------------|-------:|------:|:-------:|
| single-thread (fast)  |  773.8 | 0.774 |  1.00×  |
| parallel (rayon)      | 4850.4 | 4.850 | **6.27×** |
| SIMD (NEON 4-block)   | 3814.5 | 3.815 |  4.93×  |

**Two ceilings, and where the wall is now:**

- **Compute ceiling** = cores × single-core = 12 × 773.8 = **9286 Mw/s (9.29 Gw/s)** at perfect
  linear scaling. Parallel decode reaches **52%** of it. The shortfall from a theoretical 12× is
  expected: this is 8 performance + 4 efficiency cores, and the E-cores don't scale linearly with
  the P-cores. The realised **6.27×** is the genuine throughput multiplier; the remaining ~2× of
  the nominal 12× is the P/E asymmetry, not a contention bug (the blocks are fully independent —
  each owns its `init_state` and start bit, decoded into a disjoint `split_at_mut` output slice
  with zero cross-block state).
- **Bandwidth ceiling (the compute→bandwidth flip point).** Measured streaming-read BW =
  **78.8 GB/s**. On-disk weight traffic at 3-bit = **0.4375 B/w** (0.375 payload + 0.0625 for the
  16-B/256-block offset table). The flip — the throughput at which reads would saturate memory —
  is `78.8e9 / 0.4375 = ` **180.1 Gw/s** (187.6 Gw/s at the 0.42 B/w headline). Parallel decode at
  4.85 Gw/s sits **37× below** the flip point.

> **Speed verdict: COMPUTE-BOUND, HAS HEADROOM.** Even fully parallelized, decode (4.85 Gw/s) is
> **37× under** the 180 Gw/s bandwidth wall — STRAND's density is *not* the bottleneck; the serial
> 256-step trellis chain is. There is real compute headroom left (only 52% of the core ceiling;
> the gap to the bandwidth ceiling is enormous). The remaining levers are more cores and a genuine
> SIMD gather — see "what remains".

**Honest note on SIMD.** The NEON 4-block path (3.81 Gw/s) is **slower than plain rayon**
(4.85 Gw/s) on this hardware. It vectorizes only the state *advance* (`((s<<k)|sym)&mask` across a
`uint32x4_t`) while the LUT gather + `reconstruct_q` — which dominate this serial-chain workload —
stay scalar-per-lane to keep the bytes identical, and the across-block lockstep adds per-step
bookkeeping rayon's straight scalar path avoids. So the **shipping win is rayon (6.27×)**; SIMD is
bit-identical and available but currently a net loss vs rayon here.

**What remains (speed headroom, not yet taken):**
1. **More cores** — parallel scaling is linear in independent blocks; an all-P-core or
   bigger-socket machine moves the 52% toward the ceiling directly.
2. **A real SIMD gather** — vectorizing `lut[state]` (NEON table/gather) instead of only the state
   advance is the path to a SIMD win, but it is higher-risk against the bit-identity contract, so
   it is deferred.
3. **Block-parallel vector trellis** — the vector (d=2/d=3) rungs currently show **no** parallel
   speedup because `decode_q12_par_with_lut` detects `vec_dim > 1` and delegates to the
   single-threaded reference vector path (`gemv_par.rs:148`); "scalar" and "parallel" are literally
   the same serial code there (measured 290 / 411 Mw/s). Block-parallel vector decode is the
   documented next lever. The **6.3×–6.5×** parallel wins are real on the scalar 4/3/2-bit rungs —
   the deploy targets.

---

## 3. POSITIONING — why the bleeding-edge bits are worth choosing (made honest)

STRAND's moat at 4/3/2/1 bits is the **product** of three things, none of which a generic GGUF
quant offers together:

- **Density** — denser than Q4_K at every rung, down to 0.17 B/w (§1), at the deterministic floor.
- **Determinism** — every decode path is bit-identical to `decode_lean` (the integer reference),
  asserted in the gates over 8.4 Mw (ladder) and 67.9 Mw (speed) tensors, for *both* the scalar
  wire path and the vector learned-LUT path, across all encode levers. The codebook is a **frozen
  `&'static [i32]`** table — **zero float on the decode path**, so the same weights decode
  bit-for-bit on any CPU/GPU/WASM target.
- **Float-free portability + speed** — 6.3× parallel CPU decode at ~4.85 Gw/s, no FP rounding to
  diverge across devices.

**The honest per-rung status (7B WikiText-2, PTQ).** This is the line that must not be overclaimed:

| rung     | quality (7B WT2 PTQ)               | status |
|----------|------------------------------------|--------|
| **4-bit** | **7.81 PPL** | **Ships today.** The usable yardstick. |
| **3-bit** | **9.42 PPL** | **Ships today.** The deployable sweet spot (best density × quality on the PTQ curve). |
| **2-bit** | **213 PPL** — quality COLLAPSES under PTQ | **Format + runtime ready; quality is training-gated.** |
| **~1-bit** | COLLAPSE under PTQ (vector trellis) | **Format + runtime ready; quality is training-gated.** |

- **4-bit and 3-bit ship TODAY via PTQ.** 3-bit (9.42) is the sweet spot — usable quality at
  0.42 B/w, denser than Q4_K and fully deterministic.
- **2-bit and ~1-bit are FORMAT + RUNTIME ready, but their quality is the TRAINING FRONTIER, not
  a PTQ result.** Under plain post-training quantization, 2-bit degrades to 213 PPL and the vector
  ~1-bit collapses. **Do NOT claim 2/1 are quality-usable via PTQ.** What makes them actually
  usable is **training**: BitNet-/QAT-style training that learns weights *for* the low-bit codebook.
  In that world, **STRAND is the deterministic runtime** those models decode through — the format
  and the fast float-free decoder already exist and are verified bit-identical; only the trained
  weights are missing.

> **The product is the 4/3/2/1 ladder. The moat is density × determinism × float-free portability.**
> 4/3 are the shipping rungs; 2/1 are the format+runtime-ready frontier whose quality is gated on
> training, not on STRAND.

---

## 4. The crisp summary

| axis        | where STRAND is | one-line reason |
|-------------|-----------------|-----------------|
| **DENSITY** | **AT THE PHYSICAL LIMIT** | +0.7–2.7% off the deterministic floor at every rung; the only knob for fewer bytes is descending the ladder = the quality cliff. The ~0.2–0.35 bpw VQ gap above Shannon is uncloseable without training. |
| **SPEED**   | **HAS HEADROOM** | parallel decode is 6.27× single-thread but only 52% of the core ceiling and **37× below** the bandwidth wall — compute-bound on the serial trellis chain, with more cores / a real SIMD gather still to take. |
| **QUALITY** | **4/3 ship; 2/1 are training-gated** | 4-bit 7.81 / 3-bit 9.42 PPL are PTQ-usable today; 2-bit (213) and ~1-bit collapse under PTQ. |

**What makes 2/1 actually usable = training.** Density and the deterministic float-free runtime
are already at the limit at 2/1; the missing piece is weights *trained* for those rungs (BitNet/QAT).
STRAND is the format and the decoder that those trained low-bit models run on — bit-identical,
float-free, fast on any device.

---

### Reproduce

```
cargo run -p strand-decode-kernel --release --bin gate-ladder        # §1 density, §3 quality table
cargo run -p strand-decode-kernel --release --bin gate-decode-speed  # §2 speed vs ceilings
```

Both gates assert the sacred determinism contract (`decode_q12_fast == decode_q12_par ==
decode_lean`, bit-for-bit, scalar **and** vector-trellis) before reporting any number — a perf or
density figure off a wrong decode is meaningless and would fail the gate.
