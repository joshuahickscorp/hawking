# GLM-5.2 breakthrough baseline: matched truth on REAL artifacts

Receipt: `GLM52_BREAKTHROUGH_BASELINE.json` (schema `hawking.glm52.breakthrough_baseline.v1`).
Harness: `tools/condense/gravity_breakthrough_baseline.py`, every comparison through
`gravity_bench_lab` (refuses unmatched specs), every tensor through `gravity_real_fixtures`
(read-only, 2 h safety age). Machine: Mac15,14 / M3 Ultra / 60 GPU cores, Metal device
`Apple M3 Ultra`, maxThreadgroupMemoryLength 32768. Total GPU time for the sealed run: 5.09 s.

**activation_source = SYNTHETIC** on every number below. Teacher capsules were reported, not
read. **The router is ABSENT** (`model.layers.3.mlp.gate.weight` not on any shard), so expert
selection here is a FIXED LIST, not a routing decision.

## 1. Provenance (real tensors)

Shard `model-00075-of-00282.gravity` (`/Users/scammermike/Desktop/GLM52-Gravity-SubBit`),
205 tensors, header `production_rung=R0`, header `packed_bpw=0.8762916457915831`.
Layer 3, expert 0, all three projections from the SAME expert and the SAME shard.

| projection | shape | sha256 | descriptor bpw |
|---|---|---|---|
| gate_proj | [2048, 6144] | `6c8911891ba0e40f2517d176476ea49ae084de48d36f831a1c0e7986ceb6d77a` | 0.8763427734375 |
| up_proj | [2048, 6144] | `f0ee1b60b7fb9c03015582904125933d3690f24567cd2981455c47f0fe799a78` | 0.8763427734375 |
| down_proj | [6144, 2048] | `1de14b2f26fc3abc6c235ef0182338f968ff0b32841df8e913cdca210c70bc40` | 0.8763427734375 |

## 2. Geometry vs header — MATCH

D=8, subspaces S=1, sub=8, k=128, codebook [128, 8], rotate=False, rung R0.
nchunk 768 at gate/up, 256 at down. `matches_R0_header: true` for all three.

## 3. Parity vs `gravity_forge.pq_execute` — and a correction

Gate = 2e-3 (the module's own tolerance), NOT 1e-6. Achieved, on real artifacts:

| tensor | relative L2 | max abs error | cosine | relative max gap |
|---|---|---|---|---|
| gate_proj | 1.414e-06 | 6.318e-06 | 0.99999988 | 2.289e-06 |
| up_proj | 1.414e-06 | 4.888e-06 | 1.0000000 | 2.156e-06 |
| down_proj | 7.996e-07 | 2.086e-06 | 1.0000001 | 1.271e-06 |

The expected 2.1e-4 **did not appear on real artifacts, and the mechanism explains why**.
`gravity_metal` casts the codebook to fp16; a REAL codebook came off disk as fp16 already, so
the cast is a no-op — measured: `codebook_fp16_roundtrip_lossless = true`, max delta exactly
0.0, on all three tensors. Control run in the same session, synthetic weights at the same rung
geometry (512x256, what `gravity_metal.selftest` uses): codebook round-trip lossless = **false**,
relative max gap **2.216e-04**. So 2.1e-4 is a property of SYNTHETIC packs, not of the kernel on
production artifacts. The residual ~1e-6 on real artifacts is fp32 accumulation order.

This is not a revival of the retracted `metal_parity_1.4e-6` claim: that claim was a
torch-fp32 figure mislabelled as the Metal figure. This is the Metal figure, measured here, on
named real tensors, with the fp16 mechanism tested directly.

## 4. Matched baselines — raw samples, 8 warmup / 60 timed reps

Spec (identical on all four): batch 1, input_seed 20260722, fp32 in/out,
`sync_boundary=per_call_host_sync`, `dependency_shape=independent_calls`,
`pack_in_timed_region=false`, `unpack_in_timed_region=true`. The dense weight is reconstructed
from the artifact ONCE, outside the timed region; only the matvec is timed.

**gate/up geometry [2048, 6144]** (ms)

| variant | min | median | p95 | max | CV | contended |
|---|---|---|---|---|---|---|
| (a) dense fp16 torch MPS | 0.2826 | 0.3327 | 0.6202 | 0.9030 | 0.416 | yes |
| (b) torch/MPS compact (`decode_matvec_mps`) | 2.4635 | 2.8150 | 6.8869 | 8.7988 | 0.448 | yes |
| (c) gravity_metal, 1 call incl. waitUntilCompleted | 0.8703 | 0.9846 | 2.0925 | 5.2242 | 0.671 | yes |
| (d) gravity_metal, 32 dispatches / 1 command buffer (per dispatch, wall) | 0.7569 | 0.9019 | 1.0973 | 1.1787 | 0.134 | no |
| (d) same, per-dispatch GPU time | 0.7419 | 0.8906 | 1.0683 | 1.1581 | 0.130 | no |
| (e) `gravity_forge.pq_execute` CPU | 11.4070 | 11.6866 | 12.8056 | 13.6037 | 0.040 | no |

**down geometry [6144, 2048]** (ms)

| variant | min | median | p95 | max | CV | contended |
|---|---|---|---|---|---|---|
| (a) dense fp16 torch MPS | 0.1919 | 0.3260 | 0.4930 | 0.7674 | 0.294 | yes |
| (b) torch/MPS compact | 2.5495 | 2.9875 | 7.5896 | 7.8131 | 0.469 | yes |
| (c) gravity_metal, 1 call | 0.4482 | 0.5241 | 1.2564 | 2.3691 | 0.707 | yes |
| (d) 32 dispatches / 1 command buffer (per dispatch, wall) | 0.2153 | 0.3134 | 0.4148 | 0.4350 | 0.201 | yes |
| (d) same, per-dispatch GPU time | 0.2040 | 0.3017 | 0.4047 | 0.4243 | 0.206 | yes |
| (e) CPU authority | 11.2873 | 11.7348 | 13.2734 | 14.6405 | 0.056 | no |

(d) uses `newCommandQueueWithMaxCommandBufferCount_(1024)` and an `objc.autorelease_pool()` per
rep — the 64-in-flight deadlock never armed. (d) is a DIFFERENT BenchSpec (`sync_boundary=
per_command_buffer_host_sync`), so the harness forms no speedup against it, deliberately.

Note (d) vs (c): per-dispatch **GPU** time does not fall when dispatches are batched
(0.891 vs 0.740 ms at gate/up; 0.302 vs 0.210 ms at down — batched is slightly *higher*,
back-to-back steady state). The entire batching win is command-buffer removal from the wall:
1.18x at gate/up, 1.58x at down.

## 5. Decomposition of the single-call wall (median, % of median wall)

| geometry | wall | gpu_execution | command_buffer (residual) | host_encode | x upload |
|---|---|---|---|---|---|
| gate/up [2048,6144] | 1.0671 ms | 0.7397 ms — **69.3%** | 0.2593 ms — **24.3%** | 0.0177 ms — 1.66% | 0.0048 ms — 0.45% |
| down [6144,2048] | 0.4959 ms | 0.2101 ms — **42.4%** | 0.2536 ms — **51.1%** | 0.0126 ms — 2.54% | 0.0027 ms — 0.54% |

`gpu_execution = GPUEndTime - GPUStartTime`. `command_buffer` is the residual
(commit + scheduling + waitUntilCompleted + readback), reported as a residual rather than
asserted. Driver `kernelEndTime - kernelStartTime` median: 0.0254 ms both geometries.
The ~0.25 ms command-buffer residual is geometry-independent and is the whole story at `down`.

## 6. Roofline, corrected byte model

Bytes from `gravity_metal.matvec_bytes` (executed stream: 8-bit uploaded indices, codebook
re-read per threadgroup, staged x re-read per threadgroup), not the old scalar. Roofs measured
on this machine: 736 GB/s, 17,703 GFLOP/s. FLOPs = 2·rows·cols = 25,165,824.

| geometry | executed total B | logical artifact B | dense bf16 B | TGs | stage_x |
|---|---|---|---|---|---|
| gate/up | 1,794,048 (1.135 executed BPW) | 1,378,304 (0.8763 BPW) | 25,165,824 | 8 | true |
| down | 1,843,200 (1.156 executed BPW) | 1,378,304 (0.8763 BPW) | 25,165,824 | 24 | true |

Billed against the batched per-dispatch **GPU** time (the only command-buffer-free reading):

| geometry | GB/s | % of 736 GB/s | GFLOP/s | % of 17,703 GFLOP/s |
|---|---|---|---|---|
| gate/up | 2.014 | **0.274%** | 28.26 | **0.160%** |
| down | 6.109 | **0.830%** | 83.41 | **0.471%** |

Billed against the single-call wall: gate/up 1.822 GB/s (0.248%) / 25.56 GFLOP/s (0.144%);
down 3.517 GB/s (0.478%) / 48.02 GFLOP/s (0.271%).
Dense fp16 for contrast: 75.7 GB/s (10.3%) at gate/up, 77.2 GB/s (10.5%) at down.

## 7. Honest matched speedup, (c) vs (a)

| geometry | dense fp16 median | custom 1-call median | **speedup (median)** | ratio at min | verdict |
|---|---|---|---|---|---|
| gate/up [2048,6144] | 0.3327 ms | 0.9846 ms | **0.3379x** | 0.3247x | CUSTOM_SLOWER_THAN_DENSE |
| down [6144,2048] | 0.3260 ms | 0.5241 ms | **0.6221x** | 0.4281x | CUSTOM_SLOWER_THAN_DENSE |

Also matched, same specs: torch/MPS compact 0.118x (gate/up) and 0.109x (down); CPU authority
0.028x and 0.028x.

**The current kernel is 3.0x slower than dense fp16 at gate/up and 1.6x slower at down.**
That is where this campaign starts. The 35.9x headline is refuted and is seeded by name and by
value in `gravity_bench_lab.REFUTED_CLAIMS`, so it cannot be republished through this harness.

Caveats, stated: the box carries a live GLM-5.2 campaign, and (a)–(c) are flagged
`is_contended` (CV > 0.15) at both geometries. min and median agree on the verdict at every
geometry, and the CPU authority (CV 0.04–0.06) shows the contention is on the GPU path. p95 and
max are load, not hardware.
