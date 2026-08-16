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

# LANE: q80-deltanet-gqa
## Class: GPU_EXCLUSIVE for benchmarks, COMPILE otherwise

## The target

From the 2026-08-16 baseline (15.6 s total wall for prefill + 12 tokens):

    deltanet = 3.3269 s   (21% of the run)
    gqa      = 1.1335 s   ( 7%)
    moe_combine = 1.6958 s (11%)

Q80 is a hybrid: 36 DeltaNet layers + 12 GQA layers. DeltaNet at 3.33 s is the
second-largest single class after the MoE table build, and unlike the table build
it is real per-token work rather than bookkeeping — so the win here has to come
from *how* it executes, not from deleting it.

## What to establish first

1. Split `deltanet_secs` into its real substages. `Qwen80ActivationClassTimes`
   already has `deltanet_conv_secs` and `deltanet_recurrent_secs`
   (`qwen80_uniform_q4_hybrid_decode.rs:752`) — populate and report them if they
   are currently zero.
2. For DeltaNet and GQA separately report, per token:
   dispatches, command buffers, GPU busy ns, GPU gap ns (next_kernel_start minus
   previous_kernel_end), host ns, bytes read, bytes written.
3. State plainly whether each is GPU-idle-because-no-work-is-ready (fix: batching,
   fusion, removing CPU dependencies) or GPU-busy-but-under-occupied (fix:
   threadgroup geometry, simdgroup mapping, tiling, register pressure). These have
   opposite fixes and guessing wrong wastes the lane.

## Known context
- The default matvec kernel was recently switched to a simdgroup variant
  (commit a47f8259). Check whether DeltaNet/GQA actually use it or still run the
  serial kernel.
- Measured system-wide: ~0.79% of the 700-800 GB/s bandwidth ceiling with ~51% GPU
  idle. Assume gaps and occupancy, not arithmetic, until your own numbers say
  otherwise.
- `moe_combine` at 1.70 s is in scope for you too if the split shows it is
  dispatch/sync rather than real reduction work.

## Correctness gate
Generated token ids MUST stay exactly:
    [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914]
DeltaNet is stateful — a silent state reset between tokens still produces
plausible text. There are state-contract tests
(`qwen80_fixture_advance_hybrid_state`); run them and report.

## Do not
- Do not touch `ensure_selected_expert_table` or the activation fallback path.
  Other lanes own those.
