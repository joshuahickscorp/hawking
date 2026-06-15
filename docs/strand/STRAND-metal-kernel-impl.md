# STRAND Metal kernel — implementation spec (decode_lean + strand_trellis_gemv + M3 gate)

_Companion to `STRAND-metal-decode-gate.md` (the ridge analysis) and `STRAND-density-roadmap.md`
(the strategy). This is the **build sheet**: the bit-exact lean CPU decode, the MSL GEMV kernel,
and the one M3 measurement that decides bandwidth-vs-compute. Source is scaffolded in phase 2 — this
doc is the contract those files must satisfy._

Grounding (read before editing the spec):
- `crates/strand-quant/src/decode.rs` — `decode_tensor_fixed` / `decode_tensor_fixed_with_lut`
  (scalar path, lines 112–183), `reconstruct_q` (37–40), `eff_scale_q` (59–62), `eff_min_q` (80–88).
- `crates/strand-quant/src/trellis.rs` — `read_bits` (197–211, **LSB-first**), `next_state`
  (167–169), `state_mask`/`num_inputs`/`num_states`.
- `crates/strand-quant/src/encode.rs` — `SUB_BLOCK = 32` (77), `BlockMeta` (90–117),
  `unpack_sub_scales` (140–159), `n_sub_blocks` (163–165).
- `crates/strand-quant/src/rht.rs` — `rht_forward_inplace` (141–156), `HADAMARD_BLOCK = 256`,
  per-tensor seed = FNV-1a(name) (`bin/quantize-model.rs:439 rht_seed_for`).
- dismantle `crates/dismantle-core/shaders/quant.metal:85 gemm_q4_k_m_fused` — the GEMV template:
  one threadgroup/output-row, 256 threads, buffers `(0)=w (1)=x (2)=y (3)=rows (4)=cols`,
  `threadgroup(0)=shmem[256]`, shmem tree-reduction. **Mirror its dispatch shape exactly** so the
  dismantle harness (`backend/` seam) can swap it in with no host change.

The default geometry this spec targets: **3-bit deploy point** `k=3, L=k+4=7` (`for_bpw(3.0)`),
`block_len=256`, `SUB_BLOCK=32` ⇒ 8 sub-blocks/block, scalar `d=1` (vector trellis is a later
lever; its kernel diff is noted in §B.7). 7-bit register ⇒ **128-entry LUT** (512 B as i32) — fits
in threadgroup memory trivially.

---

## A. `decode_lean` — bit-EXACT lean CPU decode (correctness scaffold, NOT the ridge proof)

Per the gate's 2026-06-08 refinement: `decode_lean` is **not** the bandwidth proof (a CPU build
can't show the GPU memory-access wins). It exists to (1) lock the *exact arithmetic* the kernel
must reproduce into a Rust function with a `proptest`/round-trip identity test against
`decode_tensor_fixed`, and (2) pre-compute the **scale-folded LUT** the kernel uses, in tested
integer Rust, so the MSL only ports a verified recipe. It must be **byte-identical** to
`decode_tensor_fixed` (scalar path) on every input — that is the determinism contract.

### A.1 Signature & placement

```rust
// crates/strand-quant/src/decode.rs, beside decode_tensor_fixed_with_lut.
// Scalar (d==1) only; vector path stays on decode_tensor_fixed_with_lut_vec for now.
pub fn decode_lean(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<i32> {
    decode_lean_with_lut(enc, cfg, codebook_lut(cfg.l_bits))
}
pub fn decode_lean_with_lut(enc: &EncodedTensor, cfg: &TrellisConfig, lut: &[i32]) -> Vec<i32>;
```

### A.2 The three lean transforms — and exactly why each stays bit-exact

The reference inner loop (decode.rs:169–180) per weight is: `read_bits` (unaligned) → shift-in →
`lut[state]` → `eff[i/32]` → `reconstruct_q` (i64 mul, >>16) → `+offs[i/32]` → push. `decode_lean`
changes **how** three of those are computed, never **what** they evaluate to:

1. **Aligned 32-bit-word symbol reads** (the biggest lever). The bitstream is a contiguous
   LSB-first `k`-bit symbol stream (`push_bits`/`read_bits`, trellis.rs:177/197) with **no
   per-block byte padding** — block *b*'s first symbol starts at bit `Σ_{b'<b} n_{b'}·k`, which is
   only byte-aligned by luck. So `decode_lean` keeps **one running 64-bit shift accumulator** per
   tensor: refill from the next `u32` (LE) when `< k` bits remain, pop the low `k` bits as the
   symbol, drop them. Reading whole `u32`s and masking is **the identical bit selection** as
   `read_bits` looping LSB-first (LSB-first pop of a LE word = same bit order), so the symbol
   sequence is bit-identical. Edge: tensors whose total bit length isn't a multiple of 32 — pad the
   in-memory `bits` view to a `u32` boundary with zero bytes for the refill (the stream never reads
   past `total·k` bits; trailing zero pad is never consumed). **Match `read_bits`'s
   read-past-end-as-0 by zero-padding, never by UB.**

2. **Per-sub-block scale-fold** (turns the i64 reconstruct into a load **when it amortizes**).
   `reconstruct_q(es, q) = (es·q) >> 16` where `es = eff_scale_q(blk.scale_q, mult)` is constant
   across all 32 weights of a sub-block (decode.rs:177, `eff[i/SUB_BLOCK]`). The default geometry
   has `2^L = 128 > SUB_BLOCK = 32`, so folding `es` into a full 128-entry LUT per sub-block is a
   **net loss** (build 128 entries, use ≤32) — so for the **default 3-bit point, do NOT fold;**
   keep the explicit i64 mul. Fold **only** when `SUB_BLOCK ≥ 2^L` (small-L / large-SUB_BLOCK
   regimes, or a future merged sub-block): precompute `lut_folded[s] = (es · lut[s]) >> 16` once per
   sub-block, then reconstruct is `lut_folded[state]`. The fold is bit-exact **iff** it uses the
   same `(es·i32_lut_entry) >> 16` (i64 product, arithmetic `>>16`) — i.e. call `reconstruct_q`
   itself to build the folded entry, never a float scale. Gate it on `cfg` so the identity test
   covers both branches.

3. **Native 64-bit reconstruct — KEEP IT, do not use i32.** The gate's original "i32 reconstruct"
   lever is **wrong and dropped**: `scale_q` (Q16) × `quantile_q` (Q12) = Q28; a magnitude-~4 weight
   gives `2^18 × 2^14 = 2^32`, which **overflows i32**. `decode_lean` keeps
   `reconstruct_q` verbatim (i64 product, arithmetic `>> 16`). On the GPU this maps to a native
   `mulhi/mul` 32×32→64 (≈2–3 ALU ops), not a 4–5-op emulated i64 — cheap enough that it is **not**
   the tax. (Affine-min `eff_min_q`, decode.rs:80, likewise stays integer and is added post-shift,
   unchanged; for the 3-bit deploy point `has_affine_min` is **false** so the offset path is absent.)

What `decode_lean` does **NOT** touch (must stay identical): the tail-biting start-state pre-scan
(decode.rs:155–166) — it reads the same trailing `k`-bit symbols to recover `start_state`; the
`state = (state<<k | sym) & mask` advance; the `lut[state]` index; the per-weight push order. Same
in, same out.

### A.3 The identity test (the only thing that makes A worth shipping)

```rust
#[test] fn decode_lean_is_bit_identical() {
    for bpw in [3.0, 2.0, 4.0] {            // k=3 (no fold), k=2 (fold path if SUB_BLOCK≥2^L), k=4
        let cfg = TrellisConfig::for_bpw(bpw);
        for seed in 0..64u64 {              // vary lengths incl. short final block & sub-block tail
            let n = 1 + (seed as usize * 37) % 2048;
            let w: Vec<f32> = (0..n).map(|i| ((i as f32+seed as f32)*0.0137).sin()).collect();
            // exercise tail-biting ON and OFF, and affine-min ON and OFF, via encode opts.
            let enc = encode_tensor(&w, &cfg);
            assert_eq!(decode_lean(&enc, &cfg), decode_tensor_fixed(&enc, &cfg));
        }
    }
}
```
Must cover: short final block (`n*k < L` ⇒ stored `init_state`, no tail-bite), the sub-block tail
(`n` not a multiple of 32), and a `u32`-unaligned total bit length (the refill edge). A single
counterexample = the determinism guarantee is broken; treat a failure as a release blocker.

---

## B. `strand_trellis_gemv` — the MSL kernel (the actual ridge experiment)

Models `gemm_q4_k_m_fused` (quant.metal:85): **one threadgroup per output row**, threads
cooperatively dot the row and tree-reduce in shmem. Two structural changes vs Q4_K: **(1) RHT moves
to the activation** (no per-row inverse-RHT), **(2) parallelize over independent 256-blocks** (the
trellis is sequential *within* a block but blocks carry their own `init_state`).

### B.1 RHT on the activation — once per GEMV, host-side or a tiny pre-kernel

The encoder quantizes each weight **row in RHT space** (`rht_forward_rows_inplace`, per-input-channel
sign-flip + FWHT over `HADAMARD_BLOCK=256` segments that divide `in_features`, restarted at every
row, seed = FNV-1a(tensor name)). Because the Hadamard is orthogonal,
`y[o] = ⟨W_weightspace[o], x⟩ = ⟨W_rhtspace[o], RHT(x)⟩`. So:

- Compute `x_rht = rht_forward(x, RhtConfig::from_seed(rht_seed))` **once per GEMV**, before the
  kernel. The kernel then dots **decoded RHT-space weights** straight against `x_rht` — **no
  per-row inverse-RHT**. This is the QTIP/QuIP# trick.
- **Determinism caveat (flag for hardening):** the FWHT/sign-flip on `x` is **float** and runs
  per-token at inference, not part of the frozen integer decode. The *weights* stay bit-identical
  across devices; the activation transform is ordinary float GEMV preprocessing (same status as the
  `q·1/4096·x` MAC). Keep `rht_forward` numerically pinned to the encoder's segmentation: **same
  256-wide block, same row-restart, same seed**, or `y` silently corrupts (the rotation must match
  the `diag(H)` the encoder indexed). The seed travels in the `.strand` v2 header per tensor
  (`rht_seed`); the host reads it and builds the matching `RhtConfig`. A mismatched block size is
  the single most likely integration bug — assert `in_features % 256 == 0` for every Qwen2.5-7B
  tensor (all are) and fall back to the flat path only when it isn't.

### B.2 The v2 random-access layout the kernel reads (buffer 0 + buffer 5)

`.strand` v1 (format.rs:89 `write_strand`) is a **sequential** per-tensor stream — fine for CPU,
useless for a GPU that must jump to any `(row, block)`. v2 adds, per tensor, a **page-aligned
block-offset table** so the kernel seeks in O(1). The fields all already exist on `BlockMeta`
(`scale_q`, `init_state`, `n`, `sub_scales`, `mins`) — v2 is a **layout transpose, not new
quantization**. Per `(row, block)` entry the kernel needs:

| field | bits | source | use |
|---|---|---|---|
| `bit_offset` | u32 (or u64 per tensor base + u32 delta) | prefix-sum of `n·k` | seek into buffer 0 |
| `init_state` | u32 (low `L` bits used) | `BlockMeta.init_state` | start of the shift register |
| `scale_q` | i32 (Q16) | `BlockMeta.scale_q` | super-scale for the block |
| `sub_scales` | 8×6 bit = 48 bit (pack to 6 B, or pre-expand to 8×i32 eff) | `BlockMeta.sub_scales` | per-32 eff-scale |
| `n` | u16 | `BlockMeta.n` | weights in block (last may be <256) |

Tail-biting note: when `enc.tail_biting`, `init_state` is *not* on v1's wire (the decoder rescans
trailing symbols). For the GPU, **bake `init_state` into the v2 table unconditionally** (the encoder
already recorded it in `BlockMeta.init_state`, decode.rs:113–114) so the kernel never does the
pre-scan — that removes a whole sequential pass per block. v2 is the deploy format; v1 stays the
round-trip reference.

Recommended v2 packing for the kernel: pre-expand each block's 8 sub-scales to **8 × i32 effective
scales** (`eff_scale_q(scale_q, mult)`) at bake time, so the kernel does zero sub-scale unpacking on
the hot path — it indexes `eff[(j>>5)]` directly. (This is the GPU analog of A.2-lever-2's fold,
done host-side at bake, always worth it because it's amortized over every token.)

### B.3 Buffer bindings (dispatch-compatible with `gemm_q4_k_m_fused`)

```
kernel void strand_trellis_gemv(
    device   const uchar*  w_bits   [[buffer(0)]],  // tensor's k-bit symbol stream (contiguous, LSB-first)
    device   const float*  x_rht    [[buffer(1)]],  // RHT(activation), length = cols (= in_features)
    device         float*  y        [[buffer(2)]],  // output, length = rows  (= out_features)
    constant       uint&   rows     [[buffer(3)]],
    constant       uint&   cols     [[buffer(4)]],  // multiple of 256
    device   const BlockEntry* tbl  [[buffer(5)]],  // v2 (row,block) table, row-major, blocks_per_row stride
    constant       uint&   k_bits   [[buffer(6)]],  // 3 at the deploy point
    constant       uint&   l_bits   [[buffer(7)]],  // 7  ⇒ 128-entry LUT
    device   const int*    lut_q12  [[buffer(8)]],  // frozen Q12 codebook, 2^L entries (i32)
    threadgroup    int*    sh_lut   [[threadgroup(0)]],  // 2^L ints  (128 → 512 B)
    threadgroup    float*  sh_red   [[threadgroup(1)]],  // 256 floats for the reduction
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],          // = output row
    uint tgs [[threads_per_threadgroup]]);              // = 256
// BlockEntry = { uint bit_offset; uint init_state; int  scale_q;
//               int eff[8]; ushort n; ushort _pad; }   // 16+? bytes; eff pre-expanded per B.2
```

`bit_offset` is the **absolute** bit position of the block's first symbol within `w_bits` for this
tensor (so buffer 0 stays one flat per-tensor blob, no per-row base math). `eff[]` is the 8
pre-expanded effective i32 scales. Drop `lut_q12`/`l_bits` buffers if the LUT is folded into a
constant; keeping them as buffers lets B3 (`--dist`) ship a custom codebook without a recompile.

### B.4 Kernel body (the lean inner loop)

```
if (gid >= rows) return;
uint bpr = cols / 256u;                       // blocks per row
// 1) cooperatively stage the 2^L Q12 LUT into threadgroup memory (once/TG)
for (uint s = tid; s < (1u<<l_bits); s += tgs) sh_lut[s] = lut_q12[s];
threadgroup_barrier(mem_flags::mem_threadgroup);

uint mask  = (1u<<l_bits) - 1u;               // state mask
uint imask = (1u<<k_bits) - 1u;               // symbol mask
float partial = 0.0f;

// 2) parallelize over the row's independent 256-blocks (one thread = one block here;
//    bpr ≈ 14–74 for 7B, good occupancy. For small bpr, see B.6 split-within-block.)
for (uint b = tid; b < bpr; b += tgs) {
    device const BlockEntry* e = &tbl[(uint64_t)gid*bpr + b];
    uint  state = e->init_state & mask;        // baked, no tail-bite pre-scan
    uint  bitpos = e->bit_offset;              // ABSOLUTE bit offset in w_bits
    uint  col0  = b * 256u;                    // this block's first input channel
    uint  n     = e->n;
    // running 32-bit-word accumulator (aligned reads; A.2-lever-1)
    uint  word_idx = bitpos >> 5;
    uint  bit_in_w = bitpos & 31u;
    uint  acc      = (load_u32_le(w_bits, word_idx) >> bit_in_w);
    uint  have     = 32u - bit_in_w;
    for (uint j = 0; j < n; ++j) {
        if (have < k_bits) {                   // refill from next LE word
            uint nxt = load_u32_le(w_bits, ++word_idx);
            acc |= nxt << have;                // low bits already consumed; OR in the rest
            have += 32u;                       // (acc is 32-bit; only low k bits are read below)
        }
        uint sym = acc & imask;                // LSB-first pop  == read_bits(...)
        acc >>= k_bits; have -= k_bits;
        state = ((state << k_bits) | sym) & mask;   // == next_state()
        int  q  = sh_lut[state];                     // 1 shmem load (Q12)
        int  es = e->eff[j >> 5];                    // pre-expanded eff-scale (no unpack)
        // native 32x32->64 reconstruct (NOT i32 — overflows): (es*q) >> 16, signed
        int  w  = (int)( ((long)es * (long)q) >> 16 );
        partial += (float)w * (1.0f/4096.0f) * x_rht[col0 + j];   // Q12 → real, then MAC
    }
}

// 3) shmem tree-reduce partial across all 256 threads → y[gid]  (identical to Q4_K template)
sh_red[tid] = partial; threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint s = tgs>>1; s>0; s>>=1) { if (tid<s) sh_red[tid]+=sh_red[tid+s];
                                    threadgroup_barrier(mem_flags::mem_threadgroup); }
if (tid==0) y[gid] = sh_red[0];
```

Notes that keep it bit-faithful to `decode_lean`:
- `load_u32_le(p, idx)` = little-endian assemble of `p[4*idx..4*idx+4]` (or `device const uint*`
  reinterpret **only if** buffer 0 is guaranteed 4-byte aligned — the v2 writer must align each
  tensor's `w_bits` to 4 B; if not, assemble from bytes). The LSB-first pop above reproduces
  `read_bits` exactly.
- The Q12→real `*(1/4096)` and the `x_rht` MAC are float and mirror the CPU reference `matvec`
  (lib.rs:45–53). Cross-device `y` is float-reduction-order-dependent (acceptable — same status as
  every GGUF GEMV); the **integer weight reconstruction is bit-identical**, which is the moat.
- `acc` as a single `uint` caps `have` at 32; since `k≤8`, after a pop `have≥0` and a refill keeps
  the consumed window inside 32 bits. (If a future `k` makes `32-bit_in_w + 32` overflow the OR,
  widen `acc` to `ulong` — trivial; flagged for the hardening pass.)

### B.5 Threadgroup memory budget
- `sh_lut`: `2^L` ints. L=7 → 512 B. (Even L=14 = 64 Ki ints = 256 KB would blow the 32 KB TG
  limit — so for large-L research configs, fall back to reading `lut_q12` from device with the L1
  cache instead of staging. At the **3-bit deploy point this never triggers**.)
- `sh_red`: 256 floats = 1 KB. Total ≈ 1.5 KB ≪ 32 KB. Plenty of headroom for B.6.

### B.6 Occupancy fallback when `blocks_per_row` is small
`bpr = cols/256`. For `cols=2048` that's only 8 blocks < 256 threads ⇒ 248 idle threads. Two-level
split: assign `g = 256/bpr` threads per block (lane `= tid % g`), each lane strides the block's `n`
weights by `g` — **but the trellis state chain is sequential**, so a lane can't start mid-block
without the state. Resolution: keep **one thread per block** for state-walking, but have it **emit
decoded Q12 weights to shmem**, then a second pass does the MAC with all 256 threads over the
materialized row tile (decode-then-dot, like Q4_K's predec variants). This trades a shmem round-trip
for full-width MAC occupancy; **measure both** (B.6a = thread-per-block fused, B.6b = predec-to-shmem)
in §C — the gate decision (bandwidth-bound?) is about the **weight-byte traffic**, which is identical
between them, so prefer whichever clocks faster but neither changes the gate verdict.

### B.7 Vector-trellis variant (later lever, do not block on it)
For `d>1` (B1 vector trellis) the only kernel diffs: one `k`-bit symbol advances the state and emits
`d` weights from a **state-major** LUT (`sh_lut[state*d .. +d]`, decode.rs:202 path); `n_steps =
ceil(n/d)` symbols per block; the column index advances by `d` per step. Same RHT-on-activation, same
v2 table (+ a `d` field), same reduction. Out of scope for the 3-bit gate; noted so the kernel struct
leaves room (`ushort d` in `BlockEntry`).

---

## C. M3 measurement plan — the bandwidth-vs-compute gate (pass/fail)

**The whole fusion rides on this one experiment.** dismantle's Q3_K kernel died compute-bound at 24%
of peak; we must show STRAND's trellis-GEMV stays **bandwidth-bound** so fewer bytes/token actually
converts to higher tps + lower J/token. Run on the actual M3 (the dev machine), not modeled.

### C.1 What to build to measure
- A minimal Metal host harness (Rust `metal` crate or a tiny Swift/ObjC shim beside dismantle's
  existing kernel-bench path) that: bakes one tensor to v2, uploads buffers 0/5/8, runs
  `strand_trellis_gemv` for a **batch-1** GEMV (the token-decode regime), and times with GPU
  timestamp counters (`MTLCounterSampleBuffer`, `.timestamp` at encoder boundaries) — **not**
  wall-clock (excludes dispatch/CPU). Mirror it for `gemm_q4_k_m_fused` on the same shapes for a
  head-to-head.
- Shapes: the real Qwen2.5-7B GEMV rows — `o_proj`/`down_proj` (`cols=4864`/`18944`-class) and
  `q/k/v/gate/up`. Sweep at least one wide (`cols≥11008`) and one narrow (`cols≈2048`) to exercise
  B.6. Batch sizes `{1, 4, 16}` (1 is the gate; 4/16 check the cross-over to compute-bound where it's
  *expected and fine*).

### C.2 What to instrument (the numbers that decide it)
Per kernel run capture:
1. **Achieved bandwidth** `BW = bytes_read / time`. `bytes_read` ≈ weight bytes (the v2 `w_bits` +
   table for the tensor) + `x_rht` (cols·4) + `y` (rows·4); weights dominate. At 3-bit,
   weight bytes ≈ `rows·cols·(k/8) + table` ≈ `rows·cols·0.375` + side-info. Report `BW` as a **% of
   the measured M3 peak** (run a pure `memcpy`/streaming-load microbench first to get *this* M3's
   real STREAM bandwidth, ~100–150 GB/s on M3 Pro; don't trust the datasheet number).
2. **Achieved compute** `OPS = work_ops / time`, where `work_ops` = per-weight op count from the
   inner loop (count them from the emitted MSL: symbol pop, shift, shmem load, 32×32→64 mul, shift,
   2 float ops ≈ **9–11 ops/weight**) × `rows·cols`. Report as **% of measured int/FP32 peak**
   (microbench a compute-only kernel for the real ALU peak).
3. **Arithmetic intensity** `I = work_ops / bytes_read` (ops/byte) — the x-axis of the roofline.
4. Occupancy / threadgroup residency from a single GPU **frame capture** (Xcode Instruments
   "Metal System Trace" / the shader profiler) on the one representative wide tensor — to see if
   we're barrier-bound or memory-latency-bound and to confirm which B.6 variant wins.

### C.3 The ridge math (where the line is)
`ridge I* = (achieved-or-peak compute) / (achieved-or-peak bandwidth)` (ops/byte). From the gate
doc, the modeled M3 Pro ridge is **~30 ops/byte** (≈7 TFLOP/s ÷ ~150 GB/s; int ≤ FP32 on the unified
ALUs). **Recompute `I*` from C.2's *measured* peaks** — do not ship the modeled 30. Then:
- The roofline ceiling at our intensity is `min( peak_BW · I , peak_compute )`.
- If `I < I*` → the `min` is `peak_BW · I` → **bandwidth-bound** (the desired side): the kernel's
  speed scales with bytes/token, so STRAND's ~26% fewer bytes than Q4_K → ~26% faster + lower J. ✅
- If `I ≥ I*` → **compute-bound** (the Q3_K trap): byte savings don't convert; the decode ALU work
  is the wall. ✗

Our inner loop is ~9–11 ops/weight at ~0.375–0.45 byte/weight ⇒ `I ≈ 20–29 ops/byte` — **at/under
the modeled ridge, but borderline**. That borderline is exactly why this must be measured, not
argued.

### C.4 Pass / fail criteria (commit these before running, no goalpost-moving)
- **PASS (the fusion is a rocket):** at **batch 1**, `strand_trellis_gemv` achieves **≥ 60% of
  measured peak bandwidth** AND its measured `I < I*` (bandwidth side of the roofline) AND its
  **tokens/sec beats `gemm_q4_k_m_fused` on the same shapes by ≥ (4.5/3.34 − 1) ≈ 25%** (the
  byte-ratio Q4_K→STRAND-3bit). Bandwidth-bound + the byte win realized ⇒ build the dismantle shim.
- **MARGINAL (recoverable):** bandwidth-bound (`I<I*`, ≥60% peak BW) but the tps win is **<25%**
  (decode overhead eats some of the byte savings). Action: apply the lean levers harder — bigger
  aligned reads (pop 2 symbols/iter), B.6b predec staging, fold the LUT into constant memory — and
  re-measure. Still a win, just smaller; ship after one optimization pass.
- **FAIL (the Q3_K trap):** at batch 1, `I ≥ I*` OR achieved BW **< 40% of peak** with compute at
  ≥ its ceiling ⇒ **compute-bound**. The byte savings don't convert. Do **not** build the shim on
  this; the decode ALU work (most likely the symbol-pop / state-walk serialization, or barrier
  overhead from B.6) is the wall. Escalate: the trellis sequential chain may be fundamentally too
  ALU-heavy for Apple's GEMV regime, and the density moat stands on **determinism + on-device-fits**
  alone, not speed. (This is the honest downside the roadmap already prices in.)

### C.5 Control / sanity checks (so a pass is real)
- **Correctness gate first:** the kernel's decoded weights (dump `w` before the MAC for one block)
  must equal `decode_lean`'s for that block, bit-for-bit — run this before trusting any timing. A
  fast wrong kernel is worse than no kernel.
- **Peak microbenches on THIS M3** (not datasheet): a streaming-load kernel for real BW, a
  fused-multiply loop for real ALU peak — both feed C.3's `I*`.
- **Head-to-head isolation:** identical host timing harness for STRAND vs Q4_K (same warm-up, same
  timestamp method, same shapes), so the ≥25% claim is apples-to-apples.

---

## Build order & open risks for the hardening pass
1. `decode_lean` + `decode_lean_is_bit_identical` test (§A) — pure CPU, cheap, locks the arithmetic.
2. `.strand` v2 writer: the page-aligned `(row,block)` table with **pre-expanded eff-scales** and
   **baked `init_state`** (§B.2), `w_bits` 4-byte aligned per tensor (§B.4). Layout-only; the encoder
   already has every field.
3. `strand_trellis_gemv.metal` (§B) + the M3 harness (§C). **This is the one real GPU experiment** —
   everything before it is scaffolding.

**Risks / TODO flagged for post-training hardening:**
- **RHT-on-activation float determinism** (§B.1): the per-token FWHT is float and *not* covered by
  the integer-decode guarantee. Pin block=256 / row-restart / seed to the encoder or `y` corrupts.
  Decide if "weights bit-identical, activation float" is the determinism claim we make publicly.
- **`acc` width** (§B.4): single `uint` accumulator assumes `k≤8` and the OR-window stays ≤32 bits;
  widen to `ulong` if a research config breaks that. Add a debug assert.
- **Large-L LUT** (§B.5): L>~12 overflows 32 KB TG memory — staging fallback to device LUT needed
  for research configs (never at the 3-bit deploy point).
- **Tail-biting**: v2 must bake `init_state` unconditionally (§B.2) so the kernel skips the
  sequential pre-scan; verify the baker writes `BlockMeta.init_state` even when `enc.tail_biting`.
- **Buffer-0 alignment**: if the v2 writer can't 4-byte-align `w_bits`, the kernel must assemble
  `u32`s from bytes (slower) — make alignment a writer invariant, not a kernel branch.
