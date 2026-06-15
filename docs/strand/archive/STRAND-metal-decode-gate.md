# STRAND Metal decode gate — bandwidth-vs-compute, on paper before MSL

_The make-or-break for the STRAND → dismantle fusion's **speed** win. Worked out from the
actual decode loop (`decode.rs:169–180`) so we don't blind-write a Metal kernel into the
Q3_K trap._

## The gate, in one number

The fusion's speed win needs the trellis-decode GEMV to be **bandwidth-bound** on Apple
(fewer bytes/token ⇒ faster). If it's **compute-bound**, the byte savings don't convert —
exactly dismantle's dead Q3_K kernel ("compute-bound at 24 % peak"). The decider is the
decode's **arithmetic intensity (ops/byte)** vs the GPU's **ridge point**:

- **M3 Pro ridge** ≈ peak compute ÷ bandwidth ≈ ~7 TFLOP/s ÷ ~150 GB/s ≈ **~30–47 ops/byte**
  (int throughput ≤ FP32 on the unified ALUs, so use the conservative **~30**).

## Current decode = borderline / at the ridge (risky)

Per-weight inner loop, every weight:

| step | code | ~ops |
|---|---|---|
| read symbol | `read_bits(bits, cursor, k) & mask` (unaligned) | ~5 |
| advance state | `state = (state << k \| sym) & mask` | ~3 |
| LUT | `q = lut[state]` (cached) | 1 |
| sub-scale | `es = eff[i / SUB_BLOCK]` | ~2 |
| **reconstruct** | `(scale_q i64 * quantile_q i64) >> 16` | **~5 (a 64-bit multiply)** |
| store | `out.push(... )` | 1 |

≈ **~14–16 ops/weight, including an i64 multiply** (~4 ops on a 32-bit-native GPU).
At 3-bit the payload+side-info is ~**0.45 bytes/weight** (matches the 0.49 B/w `--packed-out`
smoke). Intensity ≈ **14 / 0.45 ≈ ~31 ops/byte** — **at the low edge of the ridge.** A straight
port could land either side; the i64 mul makes it lean *compute-bound*. Too risky to bet on.

## Lean decode = comfortably bandwidth-bound (clears the gate)

Three **lossless** simplifications for the GPU path — each must stay **bit-exact** so the
deterministic guarantee holds:

1. **i32 reconstruct, not i64.** `scale_q` (Q16) × `quantile_q` (Q12) = Q28; for the valid
   weight-magnitude range the product is `< 2^31`, so an **i32** multiply is exact. Guard the
   rare large-scale block (fall back to i64 there only). **i32 mul ≈ 1 op vs i64 ≈ 4 → −3 ops.**
2. **Aligned/vectorized bitstream reads.** Unpack `k`-bit symbols from 32-bit words held in
   registers (process a word's worth of symbols at once) instead of a per-weight unaligned
   `read_bits`. **−2 ops.**
3. **Per-block scaled LUT when it amortizes.** If `SUB_BLOCK ≥ 2^L`, pre-fold `eff_scale` into a
   per-sub-block scaled LUT (one pass) so reconstruct is a single load; otherwise the i32 mul
   from (1) is already cheap. **0 to −1 op.**

Lean total ≈ **~9 ops/weight** → **9 / 0.45 ≈ ~20 ops/byte < ridge → BANDWIDTH-bound.**
**The speed win is recoverable** — and it's the *lean* decode the Metal kernel must implement,
not the rich CPU decode.

| variant | ops/weight | ops/byte | vs M3 ridge (~30) | verdict |
|---|---:|---:|---|---|
| current (i64) | ~14 | ~31 | at the edge | risky (Q3_K trap) |
| **lean (i32 + aligned)** | **~9** | **~20** | **under** | **bandwidth-bound** ✅ |
| HYB-style (fold + 2 KB LUT) | ~5 | ~11 | well under | bandwidth-bound ✅✅ |

## Plan (gate, de-risked, no blind MSL)

1. Add `decode_lean` (i32 + aligned, scale-folded where it amortizes) in `decode.rs`; a test
   proves it's **bit-identical** to `decode_tensor_fixed` → determinism preserved.
2. Re-count its measured ops/weight; confirm intensity **< ridge** on paper.
3. Port `decode_lean` to a Metal GEMV kernel (modeled on dismantle's `shaders/quant.metal`
   Q4_K GEMV); apply the RHT to the **activation** once per GEMV (`y[o] = ⟨W_rht[o], RHT(x)⟩`,
   orthogonality of the Hadamard) so no per-row inverse-RHT.
4. Measure batch-1 decode bandwidth on the M3. Bandwidth-bound confirmed ⇒ speed thesis holds
   ⇒ build the dismantle shim.

**Bottom line:** the gate is **clearable**, but only via the lean decode. The current decode is
on the knife's edge; the i64 multiply is the single biggest GPU tax, and killing it (i32 +
aligned reads) drops intensity from ~31 → ~20 ops/byte — from *risky* to *bandwidth-bound*.
And density + determinism hold regardless of how this lands.

---

## Metal GEMV kernel design (modeled on dismantle's `gemm_q4_k_m_fused`)

dismantle's Q4_K GEMV (`shaders/quant.metal:85`) is the template: **one threadgroup per output
row**, buffers `(0)=weights (1)=x (2)=y (3)=rows (4)=cols`, `threadgroup(0)=shmem`, threads split
the row's dot product and reduce in shmem. The STRAND kernel mirrors it with two changes.

**1. RHT moves to the activation (free incoherence).** Precompute `x_rht = RHT(x)` **once per
GEMV** (a tiny separate kernel / host FWHT). Because the Hadamard is orthogonal,
`y[o] = ⟨W_weightspace[o], x⟩ = ⟨W_rht[o], x_rht⟩` — so the kernel dots the **decoded RHT-space
weights** directly against `x_rht`, with **no per-row inverse-RHT**. (This is how QTIP/QuIP# do it.)

**2. Parallelize over BLOCKS, not weights.** The trellis state chain is sequential *within* a
256-block, but each block carries its own `init_state`, so **blocks are independent**. A row of
`in_features` weights has `in_features/256` ≈ 14–74 blocks → split them across the threadgroup's
threads for good occupancy.

```
kernel strand_trellis_gemv(buf0 = trellis bytes, buf1 = x_rht, buf2 = y,
                           buf3 = rows, buf4 = cols, buf5 = per-(row,block) offset table,
                           tg0 = shmem[ LUT(2^L) + reduction ]):
  row = threadgroup_position_in_grid
  cooperatively load the 2^L Q12 LUT into shmem (once per threadgroup)
  partial = 0
  for each block b assigned to this thread (from buf5[row]):
     seek to b.bit_offset; state = b.init_state; es = b.scale_q (i32)
     for each weight j in block (aligned 32-bit word reads of k-bit symbols):
        sym   = next k bits                       # aligned, register-resident
        state = (state << k | sym) & mask         # bitshift register
        q     = shmem_LUT[state]                  # 1 shmem load
        w     = (es * q) >> 16                    # i32 reconstruct — NO i64
        partial += w * x_rht[block_col + j]       # the MAC
  shmem-reduce partial across threads -> y[row]
```

Inner loop ≈ **~9 ops/weight** (the lean count) → bandwidth-bound per the table above.

## Format v2 implication (the GPU needs random access)

`.strand` v1 (shipped) is a **sequential** per-tensor stream — perfect for CPU decode, but a GPU
kernel must jump to *any* `(row, block)`. So **v2 adds a per-tensor block-offset table**
(`{bit_offset, init_state, scale_q}` per block), **page-aligned and mmap-ready** — exactly the
"bake the layout as the zero-cost default" archive dismantle's `paradigmshift.md` Part IV asks for.
v1 stays the reference/round-trip format; v2 is the deploy/mmap format. The encoder already has all
the fields (`BlockMeta`); v2 is a layout change in `format::write_strand`, not new quantization.

**Build order when memory frees:** (1) `decode_lean` + bit-identity test (the ridge proof in code),
(2) `.strand` v2 writer with the block-offset table, (3) the Metal kernel above + the M3 bandwidth
measurement. Steps 1–2 are pure CPU/Rust (cheap); step 3 is the one real GPU experiment.

## ⚠️ Refinement (2026-06-08): the i32 lever is wrong; the real levers are reads + fold

Pressure-testing `decode_lean` before building it: the proposed **i32 reconstruct is NOT
bit-identical.** `scale_q` (Q16) × `quantile_q` (Q12) = Q28; a magnitude-~4 weight gives
`2^18 × 2^14 = 2^32`, which **overflows i32**. The 64-bit product is genuinely required.

But this *helps* the gate, because the i64 cost was overestimated: **Apple/most GPUs do a
`32×32→64` multiply natively (mul.lo + mul.hi ≈ 2–3 ops)**, not ~4–5. So `reconstruct_q` is
~3 ops, and the *real* per-weight tax is the **unaligned `read_bits` (~5 ops)** and the
**per-sub-block scale lookup**. Revised levers, in order:

1. **Aligned/vectorized bitstream reads** — unpack `k`-bit symbols from 32-bit words in
   registers (the biggest lever; the current per-weight `read_bits` is the costly part).
2. **Scale-fold per sub-block** — one scaled-LUT pass so reconstruct is a load, not a mul.
3. ~~i32 reconstruct~~ — **dropped (overflows); keep the native 32×32→64**.

Net op-count is similar to before (~9–11 ops/weight with reads+fold), so the conclusion holds —
**but the gate is genuinely *borderline at the ridge*, and these are GPU *memory-access* wins
that a CPU `decode_lean` cannot demonstrate.** ⇒ the decisive step is the **MSL kernel measured on
the M3**, not a CPU building block. `decode_lean` is demoted to "nice correctness scaffold," not
the ridge proof; the ridge proof is the kernel + Instruments trace.
