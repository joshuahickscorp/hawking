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
# LANE: qwen38-reuse-matrix
## Class: STATIC / CPU / COMPILE. Must NOT contaminate Q80/DSV4F GPU benchmarks.
## Steer S002 §4-5. Census is DONE - this lane produces the REUSE_MATRIX and the port plan.

## The census (merged: receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json)

Qwen3.8-27B, `model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration`.
Downloading to `workspace/campaign/records/runs/qwen38-27b/bf16` (54.74 GB, 11 shards).

    64 layers, hidden 5120, intermediate 17408, head_dim 256
    DENSE - no num_experts. NOT MoE.
    HYBRID attention: 48 linear_attention + 16 full_attention, full_attention_interval 4
    linear:  conv_kernel_dim 4, key/value head dim 128, 16 key heads, 48 value heads
    full:    24 heads / 4 KV heads (GQA 6:1), head_dim 256
    attn_output_gate: true, hidden_act silu, rms_norm_eps 1e-6
    vocab 248320, tie_word_embeddings FALSE, max_position 262144
    MULTIMODAL: vision_config + video preprocessors present

**It is NOT Q30-compatible.** Do not reuse Q30 paths on the assumption that
parameter counts are similar - the census already refuted that.

## The two findings that should drive your plan
1. **Dense, not MoE.** The entire Q80 expert machinery - `moe_table_build`, expert
   residency, device top-k, expert address tables, first-touch upload - is NOT
   NEEDED. Every wall that dominated the Q80 campaign is absent here. Do not port
   any of it.
2. **Hybrid linear/full attention on a 4-interval schedule** is structurally the
   same family as Q80's DeltaNet(36)+GQA(12). `qwen80_hybrid_token_graph.rs`,
   `qwen80_48_layer_execution_schedule.rs` and the DeltaNet/GQA kernels are
   PARAMETRIC_REUSE candidates. This is a far better starting point than
   `qwen_dense.rs`.

## Deliverable 1 — REUSE_MATRIX
For every subsystem needed to run this model, classify:

    EXACT_REUSE | PARAMETRIC_REUSE | ADAPTER_REQUIRED | NEW_IMPLEMENTATION_REQUIRED

citing the existing Hawking file path for anything reusable. Cover at minimum:
RMSNorm, linear (deltanet-style) attention, full GQA attention, the hybrid layer
schedule, QKV/O projections, attn output gate, SwiGLU MLP, embeddings, untied LM
head, tokenizer/vocab, KV cache, linear-attention recurrent state, Metal buffer
layouts, sampling, the runtime session/server, and the benchmark harness.

Existing surface to search (non-exhaustive):
    crates/hawking-core/src/model/qwen80_hybrid_token_graph.rs
    crates/hawking-core/src/model/qwen80_48_layer_execution_schedule.rs
    crates/hawking-core/src/model/qwen80_complete_runtime.rs
    crates/hawking-core/src/model/qwen30_complete_runtime.rs
    crates/hawking-core/src/model/qwen_dense.rs, qwen_moe.rs
    crates/hawking-core/src/model/qwen_complete_binary/
    crates/hawking-core/shaders/qwen*.metal

**Do not build anything already present.** For each NEW_IMPLEMENTATION_REQUIRED
entry, say specifically why no existing component fits.

## Deliverable 2 — compare the linear attention against Q80 DeltaNet, concretely
This is the highest-value comparison in the lane. Q80's DeltaNet and Qwen3.8's
`linear_attention` may be the same recurrence with different head geometry, or
genuinely different. Read both, and state which - with the recurrence written out
- rather than assuming from the name. If they match, the port is mostly a
parameterization and you should say so loudly.

## Deliverable 3 — bring-up plan, ordered, with the vision decision
Per steer §6 the path is: census -> reuse existing runtime -> conservative
representation -> complete native token -> coherent generation -> perf cleanup ->
HCLI -> tools -> AgentOS -> Grok.

**Recommend explicitly whether to skip the vision tower.** This is a DEVELOPER
model; a text-only path likely saves both bytes and bring-up work. Check whether
the text path stands alone (are the vision weights in separate tensors? does the
text forward reference them?) and say so with evidence.

## Representation budget (steer S002 §7 + user amendment)
**3 BPW is allowed - double the conventional gravity allowance.** The <=1.5
tournament law does NOT apply to this unit. The law is: the smallest representation
that is coherent, stable, sufficiently capable and fast enough to be useful. Do not
burn time forcing 2-bit if 3-bit runs today. 27B at 3 BPW is ~10.1 GB resident.

## Scope discipline - this is a hard rule
Steer §44: no sub-bit campaign, no kernel tournament, no TG3 chase, no giant
capture campaign. This unit exists to RUN and then take work off Claude. If you
find yourself designing a research program, stop and report instead.

## Resource discipline
STATIC/CPU/COMPILE only in this lane. The GPU belongs to Q80/DSV4F right now.
Do not run model inference or any GPU benchmark. Reading files, reading configs,
static analysis and `cargo build` are all fine.
