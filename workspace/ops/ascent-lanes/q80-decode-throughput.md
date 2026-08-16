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

# LANE: q80-decode-throughput
## Class: GPU_EXCLUSIVE for benchmarks, COMPILE otherwise.

## Where this comes from

The `q80-decode-kernels` lane SHIPPED working Metal kernels for all three mixed
codecs, numeric-gated against the artifact oracle (0 fallbacks, rice indices
bit-identical). Merged on main. Measured GPU time per organ, 6 interleaved reps
after 2 warmups:

    gate (binary_group)                      61 us
    up   (binary + rice_q1, expand + CSR)    80 us
    down (hgravs01_r160_b3, two-stage)      259 us   <- 65% of the cost

Its own projection: `48 layers x 10 routed x (61+80+259) us` = **192 ms/token**
for these three organs alone.

## The problem, stated plainly

The Q80 20 ms token budget allots **<= 8 ms to expert work**. The mixed
representation currently needs **192 ms**. That is 24x over budget, and it means
the <=1.5 BPW representation is presently *unaffordable to execute* even though it
is affordable to store.

This is the exact tension the doctrine names: representation and execution format
must be optimized JOINTLY. A slightly larger representation that decodes far
cheaper can beat a smaller one. The smallest artifact is not automatically the best
artifact. Density that cannot be executed in budget is not density, it is storage.

## The arithmetic that says this is fixable

One expert projection is `QWEN80_MOE_INTERMEDIATE x QWEN80_HIDDEN`. At ~1.23 BPW
the packed payload is on the order of a few hundred KB. Moving ~240 KB in 400 us
is roughly **0.6 GB/s** against a machine ceiling near 800 GB/s.

**Decode is therefore compute-bound, not bandwidth-bound** — by about three orders
of magnitude. The serial decode is the wall, not the bytes. Verify this arithmetic
yourself with the real tensor shapes and packed sizes before acting on it; if it is
wrong, say so, because the whole lane rests on it.

## What to do

1. **Confirm the regime.** Per organ: packed bytes read, achieved GB/s, arithmetic
   ops per weight, occupancy proxy, register pressure, threadgroup memory. State
   for each whether it is bandwidth-bound, occupancy-bound, or serial-decode-bound.
2. **Attack down_proj first** — 259 us is 65% of the cost. It is described as a
   serial 3-bit factor decode across two dispatches. Two dispatches per organ is
   also 2x the dispatch count. Target: single dispatch, parallel factor decode,
   decode in registers/simdgroup, consume immediately.
3. Then gate and up. For `up`, the lane already established that serial per-token
   rice decode is REJECTED (15.597 ms measured) and that expand-once + CSR apply is
   the right shape at 78-82 us. Do not undo that; make the CSR apply faster.
4. **Consider amortization honestly.** The 48x10 projection assumes every routed
   expert decodes fresh every token. Check whether expert reuse across tokens makes
   a decoded cache viable — but note the trap: caching DECODED dense weights
   restores the memory footprint the representation exists to remove. If you
   propose caching, state its byte cost explicitly against the residency budget.

## The rule you must not break
NEVER reconstruct a full dense tensor. packed bytes -> decode in
registers/simdgroup -> immediately consume. A kernel that reconstructs is rejected
even if it is faster, because it defeats the representation.

## Correctness
Keep the existing numeric gates and drifts at least as tight as shipped:
gate 4.77e-6, up 4.05e-6, down 1.14e-5, tolerance 2e-5, rice indices bit-identical.
Report measured drift for every kernel you touch. Regressing the gate is a failure.

## Report
Per-organ us before and after with paired reps, the achieved GB/s, the recomputed
`48 x 10` projection, and an honest statement of how far the routed-expert total
still is from the 8 ms budget.
