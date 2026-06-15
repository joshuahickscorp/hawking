# A10 — access-order weight-layout repack: assessment + DESIGN (2026-05-31)

**Decision: NO-CHANGE / HALT-WITH-DESIGN. Type-1 kill on the bit-identical
formulation; the non-bit-identical formulation is also slower AND fails A10's
hard gate.** No source changes landed (exploratory kernel/wrapper/test built,
measured, reverted — tree clean at HEAD).

## The question A10 was asked

A4: `gemm_q4_k_v4_predec_pair` (fused FFN gate+up, 46.6% of decode) runs at
~56-59% of peak BW. A5: the per-thread Q4_K weight read is **stride-32**
(`w_q4[bo+16 + pi*32 + simd_lane]`, `pi`∈0..3) — coalesced *across* the
simdgroup (32 lanes → 32 contiguous bytes = 1 transaction/pi) but scattered
*per thread* (4 bytes spread over 128). A5's note: an access-order repack could
make each thread's 4 bytes contiguous, enabling a `uint`/`uint4` load — "but
only if paired with a matching host repack, which is A10's scope." A10: build
the repack + matching kernel; is there a **bit-identical** win?

## What was built and measured (then reverted)

- **Repack** `repack_q4_k_pair_access_order`: permute the 128-byte qs plane so
  the 4 bytes lane L consumes (orig `16+pi*32+L`) land contiguous at
  `16 + L*4 + pi`. Header (0..16, d/dmin + packed 6-bit scales) untouched. The
  predec scale TABLE is untouched (separate buffer). The simdgroup still reads
  128 contiguous bytes — lane k → bytes `[16+4k .. 19+4k]`.
- **Kernel** `gemm_q4_k_v4_predec_pair_ao`: same dsg/dmg/dsu/dmu, same xl[8],
  same `pi`=0..3 / k0=2pi / k1=2pi+1 nibble selection, same FMA order. Two read
  variants tested.
- **Microbench**: dominant shape 11008×2048 (FFN gate+up), 200 iters/30 warmup,
  `nice -n 19 taskpolicy -b`, vs the production `_pair` wrapper.

### Result 1 — bit-identical (scalar `uchar` loads from the contiguous layout)

Reading the 4 contiguous bytes as 4 separate `uchar` is **bit-identical** to
`_pair` (parity test green over 512 rows, exact `to_bits()` match). But it is
**consistently SLOWER**:

| shape | stride-32 `_pair` | contig scalar | Δ |
|---|---|---|---|
| 11008×2048 (FFN g+u) | 58.0 GB/s | 49.7 GB/s | **−16.8%** |
| 2048×2048 (attn) | 13.6 GB/s | 12.9 GB/s | −5.5% |

**Why:** the stride-32 original is *already* the optimally-coalesced layout. For
a fixed `pi`, the 32 lanes read 32 contiguous bytes = one wide transaction. The
per-thread-contiguous repack destroys that: within one `pi`-iteration the 32
lanes now touch bytes spread across the whole 128-byte plane (lane k at
`16+4k+pi`), so the hardware issues more, narrower transactions. Reordering for
per-thread contiguity trades away the simdgroup coalescing the HW already gives.

### Result 2 — vectorized (one `uint` 4-byte load/thread)

The widened load is **NOT bit-identical**: ~1 ULP drift (e.g. −71.67029 vs
−71.67032, bits 3264173872 vs 3264173876). The Metal compiler re-contracts the
FMA chain differently when the 4 bytes arrive as one hoisted `uint` vs four
in-loop `uchar`. So it **fails A10's hard bit-identical gate** on the spot. And
it is *also* slower / no faster: a tight 5-run sweep on 11008×2048 gave
{−32.2%, +15.8%, +27.6%, −22.6%, +19.5%} — i.e. dominated by Claude.app GPU
contamination (A4 already documented Claude.app owning the GPU intervals), with
no consistent win. Achieved BW on the clean-looking runs (48-49 GB/s) was below
the 58-64 GB/s baseline.

## Verdict — Type-1 kill (bit-identical form)

The bit-identical access-order repack **dies on a measured property of the Apple
GPU memory model**: the stride-32 simdgroup-coalesced layout is already the
efficient one, and per-thread contiguity (the only thing a repack buys) makes
the simdgroup access *less* coalesced, costing BW. No implementation cleverness
changes this — it is the same fact A5 recorded (loads already coalesced),
confirmed here with a built+measured repack rather than inference.

- **Type-1 or Type-2:** **Type-1.** Death is a hardware memory-model fact
  (simdgroup coalescing already optimal; per-thread reorder is strictly worse).
- **Reframe considered:** the vectorized `uint`/`uint4` load on the repacked
  layout (the only formulation that could use the contiguity). It (a) breaks
  bit-identity via FMA re-contraction → fails the A10 gate, and (b) shows no
  reliable speedup (48 GB/s < 58 GB/s baseline on clean runs). Not a live
  Type-2: there is no formulation that gets a wider per-thread transaction
  while keeping both bit-identity AND the simdgroup coalescing.
- **Why the reframe also dies:** same root cause — any layout that makes
  per-thread access contiguous de-coalesces the simdgroup, and the only way to
  recover throughput (vectorized load) changes the compiler's FMA contraction
  (non-bit-identical) without even paying back in BW.

## What a future attended session could still try (NOT A10's bit-identical scope)

These are **separate, non-bit-identical** levers — each needs its own quality
gate (rel-L2 / token-drift), like A6.5's f16-scales did. They are NOT
resurrections of this Type-1 kill; they attack the BW gap from different angles
and would need their own oracle:

1. **4-bit-packed weight + on-the-fly recompute is already minimal.** The qs
   plane is 128 B for 256 weights = the Q4_K floor. No repack shrinks weight
   bytes; only the *scale* bytes are compressible, and A6.5 already did that
   (f16 scales, +6-9%, shipped opt-in). The remaining BW is the irreducible
   weight read at the layout the HW prefers.
2. **Lower-precision weights (Q3_K / Q2_K)** cut weight bytes — but A8 already
   measured Q3_K GEMV is 22-43% *slower* (Type-2, not BW-bound; named oracle =
   re-bench after a Q3_K GEMV is rewritten to the predec/2r standard). Footprint
   lever only, until that rewrite.
3. **Activation-side (W4A8)** is the other byte axis; already characterized
   (held, sub-additive — see `composition_decision_matrix`).

## Files
- **No source changes.** Exploratory `gemm_q4_k_v4_predec_pair_ao` kernel +
  `gemv_q4_k_v4_predec_pair_ao_pinned_tcb` wrapper + `q4k_ao_repack_bench.rs`
  (repack + bit-identical parity + microbench) were built, validated, measured,
  and reverted. Tree is clean at HEAD.
- This note. Kill recorded in `reports/dead_levers.md`.

## Is the repack worth an attended session?

**No.** The bit-identical repack is a confirmed Type-1 kill (measurably slower,
not just neutral). The only formulation that could exploit per-thread
contiguity (vectorized load) breaks bit-identity AND shows no BW gain. The
dominant `_pair` kernel's ~56-59%-of-peak is the Apple-GPU efficiency for this
weight shape at the coalesced layout it already uses; the live BW levers are
scale-byte volume (A6.5, done) and lower weight precision (A8, footprint-only
today), not access-order. Do not re-test this kill.
