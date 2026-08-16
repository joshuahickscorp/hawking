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
# LANE: q80-ns-ledger
## Class: COMPILE / LIGHT_CONTROL, one GPU run under the lock.
## Mandate: EVERY latency in nanoseconds. No stage may report zero because nothing measures it.

## The problem
The Q80 baseline prints:

    stage_secs embed=0.0000 deltanet=3.3269 gqa=1.1335 moe_norm_router=0.0000
      moe_shared=0.0000 moe_table_build=9.0777 moe_routed=0.0000 moe_combine=1.6958
      terminal=0.0000 q4_matvec=0.0000 host_expert_bind=0.0000

**Seven of eleven stages report 0.0000.** They are not free. Either nothing writes
to them, or their cost is being absorbed into a neighbouring stage. A stage that
reads zero because it is never populated is a MEASUREMENT BUG, and it hides
whatever it actually costs.

The same applies to `Qwen80ActivationClassTimes`
(`crates/hawking-core/src/model/qwen80_uniform_q4_hybrid_decode.rs:752`):
shared_swiglu, shared_mlp_sandwich, deltanet_conv, deltanet_recurrent,
gqa_input_layernorm, gqa_norm_rope, other_host_activation, metal_matvec_sync.

## Deliverable
A complete per-token ledger in NANOSECONDS where every stage is either measured or
explicitly marked `not_applicable` with a reason. For each stage:

    stage, substage, calls/token, ns/call, ns/token, % of complete token
    resource class: CPU | GPU | RAM/DRAM | IO | synchronization
    serial vs overlappable
    removable vs physically necessary
    confidence, measurement method, measured commit

Plus per token: TOTAL_TOKEN_NS, TOTAL_GPU_BUSY_NS, TOTAL_GPU_IDLE_NS,
TOTAL_GPU_GAP_NS, TOTAL_CPU_CRITICAL_NS, TOTAL_DISPATCHES, TOTAL_COMMAND_BUFFERS,
TOTAL_SYNC_POINTS, TOTAL_READBACKS, TOTAL_BUFFER_CREATIONS, TOTAL_BUFFER_REBINDS,
DRAM_BYTES_PER_TOKEN, TEMP_BYTES_PER_TOKEN.

Reuse `crates/hawking-core/src/model/qwen80_token_ns_ledger.rs` (env
`HAWKING_QWEN80_TOKEN_NS_LEDGER`) - it already has byte accounting via
`theoretical_weight_bytes_per_token`. Extend it; do not start a parallel ledger.

## THE RULE THAT MATTERS MOST
**The stage ns must sum to the complete token, and you must show that they do.**
Report `sum(stages)` against measured token wall and name the residual explicitly.
In the current baseline the stages sum to ~15.23 s against a 15.6 s run - that
0.37 s gap is unattributed and is exactly the kind of place a real cost hides.

Per-stage optimization without a whole-token identity produces local wins that do
not show up in tok/s. This ledger is the guard against that.

## Do not
- Do not optimize anything. Other lanes own the stages. You make the token legible.
- Do not create a giant JSON index; a 1.38 GB receipt was a real iteration wall.
