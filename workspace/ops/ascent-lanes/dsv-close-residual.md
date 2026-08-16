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
# LANE: dsv-close-residual
## Class: COMPILE / LIGHT_CONTROL, one GPU run under the lock.
## INSTRUMENTATION ONLY. DSV4F is theory-only per the 2026-08-16 amendment - this
## lane closes its ACCOUNTING, it does not optimize it.

## The open number
`receipts/ascent-2026-08-16/TOKEN_NS_DSV4F.json`:

    closure.identity_holds        true
    closure.residual_ns           275,026,398
    closure.residual_fraction     0.2686        <- 26.86% of the token
    closure.residual_limit        0.05
    closure.residual_within_limit FALSE

So the identity balances only because a quarter of the token is swept into an
unnamed residual. Q80's equivalent is 0.73% and names its residual
(`inter_phase_gap_plus_seal_jitter`). DSV4F's is 37x looser and anonymous.

## Your job — attribute it, do not optimize it
Classify those 275 ms into measured categories. The standing list:

    CPU feed
    wait / synchronization
    buffer lifecycle (creation, bind, rebind)
    state movement
    hidden IO
    address construction
    Rust/Python boundary
    other, explicitly named

Then report the split with ns per class, and drive `residual_fraction` under the
0.05 limit or state exactly which class resists attribution and why.

## Two traps this campaign already hit here - do not repeat them
1. **A parallel sum is not token latency.** `reader.path_resolve` reads 1318 ms and
   `verify_ns` 2505 ms against a ~1038 ms token because they are summed across
   threads. `dsv-admission-identity` cut that tax 2.9x and the token did not move at
   all. Separate SERIAL from OVERLAPPABLE and only count the critical path.
2. `DSV4F_HOST_WALL_BASELINE.json` labels its parallel sums correctly and is the
   model to copy. `DSV4F_TOKEN_NS_LEDGER.json` and `dsv-host-wall-rep6.json` do not
   and were flagged for it.

## Scope discipline
This is measurement. Do NOT start DSV4F optimization work, do not open the resident
<=1.5 path, do not re-attack host IO. If attribution reveals a large removable
class, REPORT it as a finding for later rather than fixing it - DSV4F is theory-only
while Q80 seals and Qwen3.8 comes up.

## Reference
    crates/hawking-core/src/gravity_deepseek_v4_token_ns_ledger.rs
    crates/hawking-core/src/token_ns/   (joint schema + reconciler, merged)
    receipts/ascent-2026-08-16/DSV4F_HOST_WALL_BASELINE.json  (the authority)
