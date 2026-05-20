# MLX reference drop — for L7.D / L7.E inner-block work

**Fetched:** 2026-05-20, autonomous drop. Unblocks the architecture step in [phase_l7d_plan.md](../plans/phase_l7d_plan.md).

## Files

| File | Source | Why we want it |
|---|---|---|
| `quantized.metal` | `ml-explore/mlx@main:mlx/backend/metal/kernels/quantized.metal` (158 lines) | Top-level kernel-instantiation macros. Maps kernel names → templated implementations in `quantized.h`. Light reading; the action is in `quantized.h`. |
| `quantized.h` | `ml-explore/mlx@main:mlx/backend/metal/kernels/quantized.h` (2603 lines) | The actual qmv / qmm template kernels — `qmv_fast_impl` (L750), `qmv_impl` (L817), `qmv_quad_impl` (L693). These hold the inner-block patterns L7.D needs to port. |
| `mlx__nn__quantized.py` | `ml-explore/mlx@main:python/mlx/nn/layers/quantized.py` (426 lines) | High-level QuantizedLinear wrapper. Useful for understanding the group-size / bits parameterization and how MLX's affine quantization differs from llama.cpp's Q4_K_M. |
| `mlx_lm__deepseek_v2.py` | `ml-explore/mlx-lm@main:mlx_lm/models/deepseek_v2.py` (501 lines) | DeepSeek-V2 MoE model definition in MLX-LM. Shows how the expert dispatch / routing layout maps to MLX's `quantized_matmul` calls. Cross-reference for L7.D's MoE plumbing. |

## Caveat — MLX quantization ≠ Q4_K_M

MLX uses **affine quantization**: per-group `scale + bias` (group sizes 32, 64, 128; 2/3/4/5/6/8 bits). Symmetric "amount of data per group" is much simpler than Q4_K_M's hierarchical super-block scheme (256-element super-blocks, 8 sub-blocks per super-block, scale-of-scale + min-of-min, paired-nibble encoding into 144 bytes/super-block).

So the MLX kernels are **not drop-in** for the dismantle Q4_K_M pipeline. What's portable:

- **Decode loop structure** — how the per-thread nibble-reading + register tiling is laid out.
- **Threadgroup geometry choices** — `qmv_fast_impl` is the row-batched analog of our `v3_8r` / `v3_xtg_sumy`. Compare row counts and shmem usage.
- **simd-shuffle patterns** — whether MLX uses cross-lane shuffles for nibble distribution (cf. our paired-nibble pattern at `quant.metal:478`).
- **fp16 vs fp32 accumulator choice** — Apple's MLX team made a specific call here; worth seeing what they picked.

## What L7.D's next step is

1. Read `qmv_fast_impl` (quantized.h:749-815). That's the closest MLX analog to dismantle's `v3_xtg_sumy`. Identify the specific divergence from our paired-nibble + sumy pattern.
2. Update [phase_l7d_plan.md](../plans/phase_l7d_plan.md)'s "What changes vs the existing" section with the actual MLX inner-block, replacing the hypothesis.
3. Decide whether to port the divergence as-is (e.g. simd-shuffle nibble broadcast) or whether the affine-vs-K_M quant-scheme gap rules it out.
4. If port-worthy: implement the new kernel and parity test per the existing [phase_l7d_plan.md](../plans/phase_l7d_plan.md) sequence.
5. If not: write a closeout doc noting MLX patterns are quant-scheme-specific and don't apply to Q4_K_M, then move to L7.E or L5 Lever B.

## License

These files are Apache 2.0 (MLX upstream). Not redistributed publicly; held in `reports/` which is force-added for the dismantle haul archive (gitignored from normal commits per `post_prune_operating_model` memory). If this directory ever needs to be public, redistribute the originals from MLX upstream rather than republishing this copy.
