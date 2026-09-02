# Prefill: the measured state, and what is left

Everything below is measured on this machine. No estimate is presented as a
result. Date: 2026-09-02.

## The defect

`crates/hawking-core/src/model/qwen38_hybrid_decode.rs:7452`, `generate_greedy`:

```rust
// "Every prompt token is stepped once"
for (i, &token) in prompt.iter().enumerate() {
    let (sampled, timing) = session.step(token)?;
}
```

`session.step` is the SINGLE-TOKEN decode path. There was no prefill: the prompt
was decoded one token at a time. Every kernel the resident bound was
matrix-VECTOR — `gk_matvec_binary`, `q80_binary_group_matvec_simd_bytes`,
`qwen_q2f_group64_matvec_geo_tpr64_tg128`, and siblings. Not one GEMM.

Two production receipts confirm it independently:

| prompt | completion | wall | tok/s |
|---:|---:|---:|---:|
| 4,920 | 131 | 249.2s | 20.3 |
| 5,558 | 73 | 272.0s | 20.7 |

A prompt token cost exactly what a generated token cost. That only happens with
no prefill.

## Consequences

- 12,456-token sovereign ultragoal: rejected outright at an 8,192 window.
- 15,561-token needle probe: **did not finish prefill in 1800s** and timed out.
  Prefill is worse than linear because the 16 full-attention layers are
  quadratic in sequence length.
- Extrapolated at a flat 20.5 tok/s: 131K prompt = ~1.8 h, 262K = ~3.6 h,
  ~1.01M = ~14 h, PER REQUEST. And the transport is stateless, so that cost is
  paid on every single turn — nothing is cached between requests.

## What was built

Branch `grok/prefill-gemm-20260901-232724`, commit `8391a0ef8`.

| file | lines | what |
|---|---:|---|
| `shaders/qwen38_prefill.metal` | +800 | simdgroup_matrix MMA kernels |
| `src/model/qwen38_hybrid_prefill.rs` | +771 | chunked prefill (<=64 tokens) |
| `src/model/qwen38_hybrid_decode.rs` | +137 | wiring; decode path untouched |
| `tests/qwen38_prefill_gemm_parity.rs` | +248 | GEMM vs CPU oracle |
| `tests/qwen38_batched_prefill_greedy.rs` | +121 | greedy identity + dispatch gate |

Gated by `HAWKING_QWEN38_BATCH_PREFILL` (0 = old path, for bisecting).
Chunk size `HAWKING_QWEN38_PREFILL_CHUNK`, clamped 1..=64.

Two shader-signature fixes were needed before it would compile at all; the Rust
compiled but Metal compiles at RUNTIME and failed. `qwen38_prefill_gqa_rope_cache`
mixed `uint3 tgp [[threadgroup_position_in_grid]]` with
`uint tid [[thread_position_in_threadgroup]]` and `uint tg_size
[[threads_per_threadgroup]]`; Metal requires every position-class input to be
all-scalar or all-same-width-vector.

## Measured result, same harness, same hardware

| prompt | old per-token | batched | speedup | dispatches old -> new |
|---:|---:|---:|---:|---|
| 422 | 42.7 tok/s | **78.3** | 1.83x | 244,760 -> **5,505** |
| 843 | 41.1 tok/s | **76.4** | 1.86x | 488,940 -> **11,007** |
| 3,373 | — | 67.9 | — | 41,661 |

Decode: 39.6 -> 38.9 tok/s. No regression.
Correctness: **identical generated text, both paths, 48 tokens.** Verified
directly, since the in-tree greedy test asserts a dispatch threshold BEFORE the
token-id comparison and never reached it.

## The open issue

**Dispatches fell 44x. Throughput rose 1.85x.** Those two facts together prove
dispatch overhead was never the bottleneck. The GEMM kernels are dispatching but
not achieving GEMM efficiency.

The artifact census names the most likely reason:

```
tensors=755   affine=192   q4=210   f32=353
```

Prefill GEMM kernels were written for the **affine-q2 (192)** and **q4 (210)**
tensors — 402 of 755. The remaining **353 tensors are f32 and have no prefill
GEMM at all**; they still go through matvec. Nearly half the model is on the old
path inside the new path.

The in-tree test agrees and currently FAILS, correctly:

```
prefill dispatches did not drop (seq=2900, bat=789); GEMM path not engaged
```

It wants a 4x drop and sees 3.67x. That failure is accurate and should stay
failing until the f32 gap is closed.

## Where the ceiling actually is

Prefill is compute-bound: FLOPs ~ 2 x params x tokens.
27e9 params -> 54e9 FLOP/token. M3 Ultra ~54 TFLOPS FP16 -> **1,000 tok/s at
100% utilisation**; 300-500 at a realistic 30-50%.

Published references, 27B-class, 4-bit, same chip class:

| stack | prompt tok/s |
|---|---:|
| MLX, M3 Ultra 60-core (this SKU class) | 310 @1k, 334 @4k |
| oMLX, M3 Ultra 80-core | 397 @1k, 426 @4k |
| mlx-serve, ANE split, 16k | 414 GPU-only, 498 both ANEs |
| llama.cpp Metal, Qwen3.5-27B, M4 Max | **40.3** |

That llama.cpp number is the warning: they fused the recurrence and left the
projections unbatched, and got 40. The recurrence is ~1% of FLOPs until very
long T. **The GEMM on projections/FFN is the lever; the DeltaNet chunkwise scan
is not.**

100x (2,050 tok/s) is above the 100%-utilisation roof for a DENSE 27B on this
chip. The 1900-2890 tok/s figures circulating for M3 Ultra are MoE models with
~3B active parameters — different arithmetic, not a better kernel. This is worth
re-testing rather than assuming, because we own the kernels and the packing and
are not bound by a stock runtime's choices; but nothing measured so far
approaches any wall. We are at 78 against a conservative 300.

## Next, in order

1. **GEMM for the 353 f32 tensors.** Largest identified gap. Should move
   1.85x toward the 250-350 band on its own.
2. **Make the in-tree dispatch gate pass** rather than relaxing it.
3. **Chunk-size sweep for the projections.** Free parameter, untuned.
   NOT for the DeltaNet scan: C=64 is forced — flash-linear-attention
   hard-requires {16,32,64}, the fused kkt+solve_tril path exists only for 64,
   and a CxC fp32 tile is 16 KiB at 64 but 64 KiB at 128, over Metal's 32 KiB
   threadgroup limit.
4. **Metal flash-attention over the 16 full layers.** Fixes the quadratic term
   that made the 15.5K needle time out.
5. **Then, and only then, revisit the ceiling** with a measurement instead of a
   spec sheet.

## Adjacent, cheaper, not yet done

**HCLI costs 2x before the model sees anything.** 20.5 tok/s measured through
HCLI receipts vs **42.7 tok/s** driving the same old path directly. Pure Python
overhead, and the cheapest speedup available anywhere in this stack.

## Not reachable by configuration

YaRN to ~1.01M is NOT a config change here. Our artifact's rope config matches
upstream native exactly (`rope_type: default`, `partial_rotary_factor 0.25`,
`mrope_section [11,11,10]`, theta 1e7, 262144), and the vendor recipe is just
`rope_type: yarn` + `factor: 4.0` + `original_max_position_embeddings: 262144`.
But the Rust hardcodes `QWEN38_GQA_ROTARY_DIM = 64` and `QWEN38_ROPE_THETA` and
**never reads `rope_type`**. YaRN exists in the DeepSeek-V4 path, not qwen38.
(Partial rotary is correct: 0.25 x 256 = 64 matches the constant.)

KV is not the constraint at any window we care about: 16 of 64 layers keep KV at
65,536 B/token, so 131K = 8 GB and 262K = 16 GB on a 96 GB machine.
