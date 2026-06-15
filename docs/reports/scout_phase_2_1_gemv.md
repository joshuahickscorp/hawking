> ⛔ **SUPERSEDED 2026-06-01 by `scout_phase_2_gemv_reconciliation.md`.** The A5
> (uint4 unpack) and A6 (occupancy) levers this spec scopes are a **recorded
> Type-1 kill** in `reports/dead_levers.md` ("Decode-kernel micro-opt: vectorized
> uint4 unpack (A5) + threadgroup/occupancy tuning (A6)" + "A10 access-order
> layout"): the predec GEMV loads are already simdgroup-coalesced, `_pair` is
> oversubscribed (~76 TGs/core), and the kernel is at the Apple-GPU memory-model
> optimum for batch-1 (M=1) decode. The reconciliation agent confirmed the fresh
> 0.2 "52% peak / 1.55× gap" finding does **NOT** resurrect these — the residual
> gap is **bytes (QTIP) + dispatch count**, not kernel micro-opt. **Do NOT execute
> A5-a/A5-b/A6/A10.** The only live GEMV lever is the already-built f16-scales
> (Phase 1.2). The bit-identical `_pair`/`_2r` parity tests this spec flags
> (the "Parity gate GAP") are still worth adding as **regression coverage** for
> the production kernels — keep that part; drop the A5/A6 kernel work.

# Phase 2.1 spec — GEMV uint4 unpack (A5) + occupancy (A6) (scout a9631fd8, 2026-06-01)

> Full transcript in agent a9631fd8. Actionable distillation + strategic caution.

## Layout facts (corrected)
- Q4_K block = **144 B** (16 B header + **128 B nibble plane**), `quant.metal:2243`.
  Nibble plane starts at `bo+16`, always **16 B-aligned** (144 = 9×16). Clean
  `uint4` target.
- "192 vs 160" = *effective decode footprint*: 128 B nibble + 64 B f32-scale (192)
  vs 128 + 32 B f16-scale (160). f16-scales cuts the SCALE stream; A5 cuts the
  NIBBLE-load instruction count. **Disjoint → they compose.**

## The inefficiency (precise)
Inner loop (`_pair` quant.metal:2259-2267): per `pi`∈0..3, each thread does a
1-byte `uchar` load → **4 separate scalar loads/thread (8 for gate+up)**. Across
the 32-lane simdgroup at fixed `pi`, lanes cover a contiguous 32 B span → **already
coalesced**. The waste is per-thread load-issue/addr-gen overhead, not coalescing.
Each thread's 4 bytes are at **stride-32** (`16+L, 48+L, 80+L, 112+L`).

## A5 — two ordered, measurement-gated attempts
- **A5-a (bit-identical, try FIRST):** replace the 4 scalar loads with a register
  gather `uchar4 qg = uchar4(wp[0],wp[32],wp[64],wp[96])` then unpack 8 nibble-pairs
  from regs. **Preserves exact (pi,lane)→(k0,k1) map + FMA order → byte-identical.**
  ⚠️ **Whether the Metal compiler fuses the strided gather into fewer transactions
  is NOT guaranteed** (Apple GPUs often keep it 4 loads). **Verify with an Xcode
  Metal GPU capture / the `q4k_predec_f16s_bench.rs` GB/s readout BEFORE shipping —
  do not assume.**
- **A5-b (atol 1e-3, FALLBACK only if A5-a shows no win):** re-tile the plane as
  `device const uint4*`, one true contiguous 16 B load/thread → **changes reduction
  grouping → NOT bit-identical** (gate drops to atol 1e-3 / rel-L2 <1e-2). This is
  the form that physically widens the transaction. Gate behind a new
  `DISMANTLE_QWEN_PREDEC_UINT4=1` until the clean bench confirms.
- ⚠️ A5-b on `_2r` (q/o/ffn_down/LM-head): verify every tensor offset is ≥16 B
  aligned first. `_pair` (ffn only) is safe.

## Parity gate GAP (must fix first)
No bit-identical test exists for `_pair` or `_2r` vs `v3_8r` today (only f16s-
relative). **Add `q4k_v4_predec_pair_bit_identical_to_v3_8r` + `_2r` version**,
cloning `tests/q4k_predec_parity.rs` (`.to_bits()` byte-equality, `make_q4k_bytes`/
`make_x` generators). This is the A5-a gate.

## A6 occupancy sweep
Params: TG width {128,192,256,320 threads = 4/6/8/10 simdgroups}, rows/simdgroup
{1,2,3,4}, accumulator-chain count (a `_pair_2r` = 4 chains is the new register-
pressure test; `_4r` single-output already exists default-off `mod.rs:1326` — the
precedent). Bench: extend `tests/q4k_predec_f16s_bench.rs` (`time_dispatch` :64,
the 11008×2048 shape :154) to a **two-output `_pair`** variant; compare RELATIVE to
the harness's own `_pair` baseline (microbench abs µs ≠ the in-engine 281 µs).
**Keep-rule (the _4r lesson):** ship a variant ONLY if it beats `_pair` in (a)
microbench µs AND (b) end-to-end paired dec_tps (`clean_bench.sh`/`bench_diff.sh`).
No shipping on register-pressure speculation. Gate each behind an env flag, flip
default only after the clean bench.

## f16-scales interaction
Compose (disjoint byte streams). Apply the SAME A5 diff to BOTH `_pair` AND
`_pair_f16s` (nibble-load lines are identical: 2259-2267 vs 2330-2338), then
`_2r`/`_2r_f16s`. Order: f32 `_pair` first (cleanest bit-identical gate), replicate
to `_pair_f16s` (re-run f16s relative gate — unchanged), then the 2r pair. No new
flag for A5-a (ship into all four); A5-b needs the UINT4 flag.

## MMA/simdgroup_matrix for decode = DEAD
Confirmed in dead-lever record + `qwen_dense.rs:4890-4895`: MMA needs N≥2 (a real
M×N tile); decode batch-1 has N=1 → 7/8 of every `simdgroup_multiply_accumulate`
wasted + staging-barrier overhead. MMA is prefill/batched only
(`DISMANTLE_QWEN_Q4K_MMA`). **Decode GEMV stays predec scalar-reduction; A5/A6 are
the levers.**

## ⚠️ STRATEGIC CAUTION (orchestrator)
The loop is already coalesced AND MMA is dead at N=1 → the "wider loads close the
1.55× gap" thesis is NOT a slam dunk. A5-a may yield little; A5-b is quality-risky.
**Part of llama's 1.55× may live in dispatch count / residency-sets, not the nibble
load.** ⇒ Phase 2 plan: (1) **cheap A5-a spike first** — add the parity test, do the
`uchar4` rewrite on `_pair`, bench; decide a-vs-b on data. (2) Run **dispatch-fusion
(2.2)** as a parallel high-value track (324/616 trivial dispatches). Don't sink deep
effort into A5-b before confirming where the gap actually is.

## Files
- quant.metal inner loops: `_pair` 2259-2267, `_pair_f16s` 2330-2338, `_2r`
  2403-2411, `_2r_f16s` 2472-2480; new A6 `_pair_Nr` near 2208.
- kernels/mod.rs wrappers + env gates (`_4r` precedent 1321-1328).
- qwen_dense.rs selection: ffn `_pair`/`_pair_f16s` 4269/4291, `predec_f16scales_active`
  3459, MMA-decode exclusion 4890-4896.
- tests/q4k_predec_parity.rs (clone for the bit-identical gate); q4k_predec_f16s_bench.rs (A6 bench).
