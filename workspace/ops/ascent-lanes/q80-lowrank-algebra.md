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
# LANE: q80-lowrank-algebra
## Class: GPU_EXCLUSIVE for benchmarks, COMPILE otherwise.
## Doctrine: REPRESENTATION -> ALGEBRA. Named one of the highest-value research classes.

## The principle
Never reconstruct a structured representation when the algebra can consume it directly.

For a low-rank organ W = U @ V, do NOT do:

    reconstruct W  (dense, rank-limited)  then  W @ x

Do:

    U @ (V @ x)

Cost changes from `d_out * d_in` to `r * (d_in + d_out)`. For down_proj at
hgravs01_r160_b3 that is a large asymptotic reduction AND it removes the dense
temporary entirely - satisfying the zero-temporary principle at the same time.

## The target
Q80 down_proj uses `hgravs01_r160_b3`, activation-weighted low-rank, scored on the
post-SwiGLU intermediate. It is already the most expensive decode organ.

Merged decode-throughput result (receipts/ascent-2026-08-16/q80-decode-throughput.json):

    gate  60.5 -> 6.875 us
    up    80.0 -> 17.25 us
    down 261.3 -> 13.959 us
    per expert 400 -> 38.08 us; routed 48x10 projection 192.9 -> 18.28 ms
    up_proj is now ~45% of the remaining 38 us; achieved 15 GB/s - still
    decode/occupancy bound, not DRAM bound.

## What to establish
1. Does the CURRENT down_proj kernel reconstruct W, or does it already apply
   factors? Read `crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs`
   and `shaders/q80_mixed_decode.metal`. Report which, with the line. If it already
   does U @ (V @ x), say so plainly and the lane is a cheap confirmed negative -
   that is a fine outcome, do not manufacture work.
2. If it reconstructs: implement factored application and measure. Report FLOPs
   before/after from the real shapes, plus measured us/organ with >=3 paired reps.
3. Apply the same question to every structured organ: base+residual, additive
   codebooks, sparse corrections. For `up` (binary + rice_q1 sparse residual @2%),
   check whether the residual is applied as a sparse correction to a dense
   reconstruction or fused into the accumulation. Fusing it is the same principle.

## Hard rule
NEVER materialize a dense weight tensor. packed -> registers/simdgroup -> decode ->
multiply -> accumulate. A version that is faster by reconstructing is REJECTED.

## Correctness
Keep the shipped numeric gates at least as tight: gate 1.81e-5, up 1.10e-5,
down 1.14e-5 at tolerance 2e-5, rice indices bit-identical, 0 fallbacks. Grade
against the ARTIFACT oracle, never the BF16 parent. Report measured drift.
Reassociating a matrix product changes floating-point order - state your
numeric-equivalence gate explicitly.

## Report
Per-organ us before/after with the full spread, recomputed 48x10 routed projection,
FLOP counts from real shapes, temporary bytes eliminated, and honest distance from
the 8 ms expert budget.
