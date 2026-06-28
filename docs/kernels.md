# Metal kernels

This document maps every kernel in `crates/hawking-core/shaders/`
to its purpose, the phase it lands in, and the wedge it implements
(if any).

(Auto-generated table lands here as Phase 1+ kernels ship. Until
then, see the header comments in each `.metal` source file.)

## Conventions

- All kernels are in `metal_stdlib` namespace.
- All matmul-shaped kernels accept `simdgroup_matrix` 8×8 fp16 mma
  tiles where the dimensions allow.
- Threadgroup memory budget per kernel is documented at the top of
  the kernel function.
- Each kernel's *occupancy* (active threadgroups per GPU core) is
  recorded once Phase 1+ tuning lands.

## Files

| File | Kernels | Phase | Wedge |
|---|---|---|---|
| `moe.metal` | gate, dispatch, grouped GEMM, gather | 1, 2 | 1, 2 |
| `attn.metal` | MHA, MLA compress/decompress, KV append | 0–3 | — |
| `quant.metal` | Q4_K_M / Q5_K_M / Q8_0 dequant + fused-dequant GEMM | 0, 1 | 2 |
| `sample.metal` | top-K, top-P, temperature, constraint mask | 2.5 | 3 |
| `common.metal` | rmsnorm, swiglu, rope, embed | 0 | — |
