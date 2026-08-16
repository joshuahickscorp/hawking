# SUPERWAVE STATE — femtosecond + generalized kernel (2026-08-16)

HEAD ca1f29bec | M3 Ultra, 60 GPU cores, 96 GB unified, 819 GB/s published peak
Measured bandwidth control band: 560-647 GB/s (68-79% of peak). Unity is NOT achievable.

## Law
fs_per_weight = 152.6252 * BPW / efficiency   (receipts/ascent-2026-08-16/FS_PER_WEIGHT_LAW.json)
Q80 today: 156,970 fs at 0.135% efficiency. Floor 212.53 fs at 1.392467 BPW.
Gap decomposition: efficiency ~740x, density ~2.1x. EFFICIENCY IS THE DOMINANT TERM.
Sub-100 fs needs BPW < 0.448 (low band) to 0.5176 (high band), NOT the 0.6552 the
ledger nominally states -- that figure assumes efficiency=1.0.
(receipts/ascent-2026-08-16/G013_FS_EFFICIENCY_CLOSURE.json)

## Artifacts on disk (workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-candidates)
mixed-1p5-v1      1.4444457 BPW   13.7 GB   <- Q80 sealing vehicle (G020)
mixed-1p5-ne4-v1  1.3257127 BPW   12.6 GB
mixed-sub655-v1   0.6462039 BPW    6.2 GB   <- densest coherent pack
uniform-q4-group64-v1  DE-AUTHORISED by user steer S003. Tensors deleted. DO NOT REBUILD.

## Measured token budgets
Q80 (mixed): 1.249 s/token. Q80 (best any vehicle): 225 ms/token = 4.43 tok/s.
  facets: host expert-table bind 54-79 ms | 98 command buffers, wait-minus-gpu 36 ms
          GPU 125 ms at 15 GB/s (1.8% of peak) | 1155 dispatches/token
DSV4F: 1.109 s median body (paired, 3.286 s base, 2.96x, hc_sha bit-identical)
  facets: host.expert_slab_io 523 ms | metal_gpu 410 ms | host_exclusive 696 ms
          137 command buffers, 1857 dispatches
Gate for G001/G002: <= 20,000,000 ns/token (50 TPS). G017: all three models. G020: 100 TPS.

## Kernel fork (the G023 target)
39 .metal shaders in crates/hawking-core/shaders/
  24 qwen-family (qwen*, q80*) | 4 dsv4f-family (dsv4f*, deepseek_v4*)
Two fully bespoke paths. No shared kernel abstraction. Hot path kernels:
  qwen80_device_expert_table.metal, qwen80_routed_expert_wave.metal,
  qwen80_shared_expert_wave.metal, q80_mixed_decode.metal, q80_matvec_geometry_gen.metal
  dsv4f_native_token_graph.metal, dsv4f_activation_x_batch.metal

## Rules
- batch=1 decode: intensity ~1 flop/byte, zero reuse. No kernel trick escapes the
  bandwidth floor. Occupancy and bytes-moved are the only levers.
- GPU timing authority = completed MTLCommandBuffer GPUStartTime/GPUEndTime ONLY.
  Never a CPU-wait proxy.
- Paired alternating reps, >=3 each, full spread. A single run is page-cache confounded.
- hc_sha must stay bit-identical. 0 fallbacks.

## Lanes already running (DO NOT DUPLICATE)
auto-dsv4f-host-exclusive-not-this-*  and  auto-dsv4f-level-host-exclusive-not-*
  both attacking DSV4F host_exclusive / expert slab I/O.

## G023 STRUCTURE — TWO AXES, NOT ONE FAMILY (receipts/ascent-2026-08-16/G023_KERNEL_AXES.json)
Grouping by vendor name is the WRONG cut. Verified from QWEN38_REUSE_MATRIX.json (G014):
  AXIS 1 deltanet/linear attention: Q80 + Qwen3.8. "SAME recurrence as Q80 Gated
    DeltaNet; port is parameterization + projection-layout adapter." EXCLUDES DSV4F.
  AXIS 2 MoE expert path: Q80 + DSV4F. EXCLUDES Qwen3.8 -- it is DENSE, no num_experts.
Q80 is the only model on BOTH axes: it is the integration point.
Qwen3.8: 64 layers (48 linear + 16 full, rule (layer+1)%4==0), hidden 5120,
  intermediate 17408, dense SwiGLU every layer, multimodal (SKIP vision for bring-up).
  Start from qwen80_hybrid_token_graph.rs, NOT qwen_dense.rs.
  Do NOT port: moe_table_build, expert residency, device top-k, expert address
  tables, first-touch upload, routed/shared expert waves.
  qwen38-27b/uniform-q4-v1 (13.65 GB) is ~4.3 BPW: bring-up vehicle ONLY, it cannot
  satisfy G016's 2.0 target or 3.0 hard limit.

## CORRECTION 2026-08-16 — THE DENSITY CAMPAIGN HAS BEEN COMPRESSING THE WRONG 9%
(receipts/ascent-2026-08-16/G013_FS_EFFICIENCY_CLOSURE_V2.json — supersedes v1, which was wrong)
Per-token bytes moved, VERIFIED against QWEN80_TOKEN_NS_LEDGER to within 0.03%/class:
  attention      818,151,424   73.0%
  lm_head        165,329,552   14.7%   <- attention + lm_head = 86-88% of traffic
  routed experts 100,915,200    9.0%   <- the entire density campaign compresses THIS
  router          26,742,448    2.4%
  shared expert    10,091,520    0.9%
STORAGE BPW IS NOT ACTIVE BPW. At batch=1 only 10 of 512 experts are read, so
mixed-sub655's 0.6462 storage figure is really 2.518 ACTIVE BPW; mixed-1p5 is 4.980.
MEASURED no-model control in the real decode shape (98 CBs, 10-of-512 gather, unique
bytes once) = 320-411 GB/s. NOT 560-647, NOT 819 peak.
Consequence: compressing unused experts cannot move fs/weight OR token time.
Attention is the only mass whose compression changes anything.
