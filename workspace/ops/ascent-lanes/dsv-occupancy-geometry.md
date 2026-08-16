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
# LANE: dsv-occupancy-geometry
## Class: GPU_EXCLUSIVE for benchmarks, COMPILE otherwise.

## The finding you are acting on
`dsv-mla` (branch `grok/dsv-mla-20260816-003211`, commit `1983ec3b`) established
that DSV4F attention is **OCCUPANCY-limited, not bandwidth-limited**: realized
traffic is **24-36 GB/s against a ~750 GB/s ceiling**.

The thread geometry it reported is the smoking gun:

    RMSNorm            1 thread        <- one thread
    WQ-A               1024 threads
    WKV                512 threads
    sparse attention   64 threads

A one-thread dispatch on a GPU with thousands of lanes is pure launch latency.

It already banked one win (simdgroup KV QAT, 2.9 ms -> 107 us/layer, attention GPU
198 -> 128 ms) and one NEGATIVE you must not re-pay for: **collapsing the 17
attention dispatches into one serial encoder was bit-identical, took encoders
731 -> 43, and did NOT move attention GPU at all.** Encoder gaps are not the
attention wall. Do not retry that.

## Targets, in order
1. **RMSNorm on 1 thread.** Parallelize across the hidden dimension with a
   simdgroup/threadgroup reduction. Cheapest available occupancy win.
2. **wo_a 0.67-1.08 ms and wo_b 0.64-1.01 ms per layer** = 29-44 ms/token combined,
   the largest remaining isolated kernels.
3. wq_a / wkv on 1024 / 512-thread grids - size the grid to the machine, not to the
   tensor's convenience.

For each kernel report before/after: threads, threadgroups, simdgroup geometry,
occupancy proxy, realized GB/s, register pressure and threadgroup memory if
observable, and wall ns. State whether it became bandwidth-bound; that is the
success condition, since bandwidth is the physical floor.

## The trap
A fused or widened kernel can be SLOWER through register pressure or spills.
Compare complete-token wall and occupancy, never dispatch count alone. A widening
that regresses the token is a rejected result and real negative science - report it.

## Correctness gate
`hc_sha == c94da765c4bbf795b598d96209cd80821e5a81ab97a8712586f54b8c8b612597`,
greedy token 5, logit 16.7818546295166 (delta <= 0.05), 0 fallbacks.
RMSNorm changes reduction ORDER, so state your numeric-equivalence gate explicitly
and report measured drift.

## Coordination
`dsv-integrate-graph` is currently merging cb-collapse + dsv-expert + dsv-mla onto
main. **Base your work on `grok/dsv-mla-20260816-003211`**, not on main, and say so,
so your change composes with the KV-QAT win rather than conflicting with it.
