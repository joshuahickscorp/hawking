## HAWKING ASCENT — standing laws for every lane

Repo: /Users/scammermike/Downloads/hawking . Build dir is `workspace/ops/build/rust`
(set by .cargo/config.toml). NEVER use `target/` or `target-parallel/` — stale
binaries there still run and have produced false results before.

Build: `cargo build --profile release-fast -p hawking-core --example <name>`
Binaries: `workspace/ops/build/rust/release-fast/examples/<name>`

### Resource discipline (MANDATORY)
This machine runs several lanes at once. Any command that touches the GPU or
allocates more than ~8 GiB MUST be wrapped:

    ./tools/gpu_lane_lock.sh <your-lane-name> <your command...>

It is a mutex; it blocks until free (90 min cap). Compiling, reading, static
analysis and unit tests do NOT need it. Never bypass it — an unlocked benchmark
run silently corrupts another lane's timing.

### Measurement law
- A single Metal run is page-cache confounded. Any timing claim needs >= 3
  alternating paired reps (A,B,A,B,A,B) and you must report the full spread,
  not just the median.
- GPU time means `MTLCommandBuffer.GPUEndTime - GPUStartTime` after wait.
  A CPU wall-clock wait is NOT GPU time; never report it as such.
- Label every number DIRTY_ENGINEERING (other lanes running), CLEAN_CANDIDATE,
  or BASE_TRUE. Do not launder a dirty number into a clean claim.
- Report ns/token, not just tok/s.

### Correctness law
- Bit-identity or a stated numeric-equivalence gate is required for every
  optimization. "Looks close enough" is a rejected result.
- 0 fallbacks. If a fast path silently falls back, that run is invalid.
- Never weaken an existing gate, assertion, or seal to make something pass.
  If a gate blocks you, report it as a finding — do not edit it away.

### Negative science — do NOT re-pay for these
- Q80 cross-expert shared-basis: REFUTED (experts mutually orthogonal, cos 0.004).
- Q80 "simply bandwidth-bound": REFUTED. Measured 0.79% of the 700-800 GB/s
  ceiling with ~51% GPU idle. It is dispatch/host bound, not bandwidth bound.
- DSV4F route-ID readback serializer hypothesis: REFUTED.
- Shader compile as the primary current wall: REFUTED / deprioritized.
- Single-family Q80 representation: INSUFFICIENT. gate_proj/up_proj/down_proj
  each prefer a different codec family; down_proj inverts the ranking and needs
  post-SwiGLU X, not the layer hidden.
- Q30 static <=1.5 coherence: FAILED. Do not copy the Q30 approach.
- Immutable-identity recomputation (SHA, st_dev, geometry parse, manifest scan)
  per token has repeatedly been the real latency. Suspect it early.
- Giant JSON indexes are a real iteration wall (1.38 GB capture-result.json).
  Do not add one.

### Reporting
End your final message with:

    LANE: <name>
    STATUS: SHIPPED | PARTIAL | BLOCKED
    BASELINE_NS_PER_TOKEN: <n> (label)
    RESULT_NS_PER_TOKEN: <n> (label)
    REPS: <the actual paired numbers>
    CORRECTNESS: <bit-identical | numeric gate + measured drift | N/A>
    FILES: <paths touched>
    RECEIPT: <path to json receipt you wrote under receipts/ascent-2026-08-16/>
    NEXT_BOTTLENECK: <what is now the top cost, with its measured ns>

Commit your work on your branch before finishing. Uncommitted lanes have been
lost here before.

---
# LANE: qwen38-native-bringup
## Class: COMPILE / CPU, GPU under the lock for generation. NOW P1 per the 2026-08-16 amendment.

## Why this is now high priority
Resource priority shifted to the two Qwen-family models. They share the DeltaNet
hybrid architecture, so work transfers BOTH ways: Q80's kernels serve Qwen3.8, and
Qwen3.8 as a developer unit works on Q80. DSV4F is architecturally separate (MLA,
FP4 experts) and is now theory-only.

## What already exists (all merged, all on main)
- **Model downloaded and verified**: `workspace/campaign/records/runs/qwen38-27b/bf16`
  11/11 shards, 54.71 GB, 1184 tensors, index complete.
- **Census**: `receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json`
  64 layers, hidden 5120, intermediate 17408, head_dim 256, vocab 248320, untied
  LM head, DENSE (no MoE), 48 linear_attention + 16 full_attention on a 4-interval
  schedule, attn_output_gate true, rms_norm_eps 1e-6.
- **Reuse matrix**: `receipts/ascent-2026-08-16/QWEN38_REUSE_MATRIX.json`
  Recommends starting at Q4 (~13.45 GB) if that packer retargets cheaply. 3 BPW
  allowed, 2.0 is the TARGET. Do NOT open 2-bit yet.

## The decisive reuse fact - do not re-derive it
Qwen3.8's `linear_attention` is **the same gated-DeltaNet recurrence Q80 already
runs**. Identical tensor names: `A_log`, `conv1d.weight`, `dt_bias`, `norm.weight`,
`out_proj.weight`. The ONLY difference is projection fusion:

    Q80      in_proj_ba (b,a fused)      in_proj_qkvz (q,k,v,z fused)
    Qwen3.8  in_proj_a, in_proj_b        in_proj_qkv, in_proj_z

**Fuse at PACK TIME** into Q80's layout and the existing kernels apply directly.
The reuse-matrix lane named this as the next step. Reuse:
    crates/hawking-core/shaders/qwen_next.metal
    crates/hawking-core/src/model/qwen80_hybrid_token_graph.rs
    crates/hawking-core/src/model/qwen80_48_layer_execution_schedule.rs
Full attention is GQA 24:4 at head_dim 256 with q_norm/k_norm - Q80's GQA is a
parametric fit. All 64 MLPs are plain dense SwiGLU.

## Skip vision - proven, not assumed
`vision_tower.*` is a separate root holding 333 of 1184 tensors; `language_model.*`
is self-contained at 851. Load language_model only.

## Also inherit the matvec win
The matvec campaign is RESOLVED and its kernels are on main. Q4 organ went
209250 -> 6709 ns (31.2x) via threadgroup geometry (`named` / binary tg256), now
~1.8x of MLX incremental. Register pressure was never the limiter and widening
(rowblock 4-8, fp16 acc, X-tile) regressed. Use the shipped geometry; do not
re-sweep it.

Qwen3.8's MLP organs are 5120x17408 = ~50 MiB, far LARGER than Q80's 0.59 MiB
experts. MLX reaches ~182 GB/s on exactly that shape, so for Qwen3.8 the ceiling is
real and directly achievable - any shortfall is our kernel. Study MLX `qmv_fast`.

## Order of work
1. Loader + tensor mapping for `language_model.*`, with pack-time in_proj fusion.
2. Bind the hybrid schedule: 48 linear + 16 full on the 4-interval pattern.
3. Complete native token forward. Reference/oracle execution is allowed ONLY for
   correctness bring-up - native is the product.
4. Coherent generation. Report the text verbatim.
5. First ns/token with paired reps.

## Density
Target <=2.0 BPW complete physical (3.0 hard limit). At 2.0 the resident footprint
is 5.9 GB and the DRAM floor is 7.207 ms/token, ceiling 138.8 tok/s. At 3.0 the
ceiling drops to 92.5 - so 100 TPS is IMPOSSIBLE at 3.0 and possible at 2.0. Start
wherever generates fastest; do not burn the lane forcing 2-bit.

## Scope discipline
Steer S002 §44: no sub-bit campaign, no kernel tournament, no TG3 chase. This unit
exists to RUN and then take routine work off Claude. If you find yourself designing
a research program, stop and report.
