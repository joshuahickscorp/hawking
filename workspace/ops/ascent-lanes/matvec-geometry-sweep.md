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
# LANE: matvec-geometry-sweep
## Class: GPU_EXCLUSIVE. Use ./tools/gpu_lane_lock.sh.
## Steer S002 §27 - microkernel autogeneration. Brute-force the geometry space.

## The target
Q80 packed gate matvec runs 2.47 GB/s; Q4 gate 2.65; DSV4F FP4 reduced 0.93.
The same box streams 560-647 GB/s on a row probe. We believe the cause is launch
geometry and work decomposition, not bandwidth.

Do not hand-design variants one at a time. **Generate and sweep.**

## Sweep these axes, Tier-1 reject fast
    threads per threadgroup      32, 64, 128, 256, 512, 1024
    simdgroups per threadgroup   1, 2, 4, 8, 16
    rows (outputs) per threadgroup
    work items per thread        1, 2, 4, 8 (grouped rows / split-K)
    vector load width            scalar, 2x, 4x (packed uint/uint2/uint4)
    reduction strategy           per-thread serial | simdgroup cooperative
                                 (simd_shuffle_down / simd_sum) | threadgroup tree
    unroll factor                1, 2, 4, 8
    accumulation type            fp16 vs fp32

For each surviving candidate report: achieved GB/s, % of the 560-647 GB/s control,
kernel ns, threadgroup occupancy, register pressure and threadgroup memory if
observable.

## The prior worth testing first
`q80-component-simdgroup` found DeltaNet/GQA running **one thread per row** - a
serial matvec - and wiring the existing simdgroup kernel gave a 3x mixer win
(111 -> 36 ms). Check whether the PACKED matvecs have the same defect before
sweeping blindly. If they do, the fix may be a known pattern rather than a search.

## Hard rules
- NEVER materialize a dense weight tensor. packed -> registers/simdgroup -> decode
  -> multiply -> accumulate.
- Keep the shipped numeric gates: gate 1.81e-5, up 1.10e-5, down 1.14e-5 at tol
  2e-5, rice indices bit-identical, 0 fallbacks. Grade against the ARTIFACT oracle.
- A widening that raises register pressure can be net SLOWER. Compare complete
  token wall and occupancy, not GB/s alone. Report regressions - they are real
  results and they bound the search.

## Do not re-pay
Dispatch/encoder/topology collapse is REFUTED on both models (Q80 fuse regressed
516 vs 307 ms; DSV4F 731 -> 43 encoders moved attention GPU by nothing). DRAM row
interleaving LOST on Q4 and binary. Expert co-occurrence layout is weak (1.037x).
Switching-activity alpha is already ~0.5 (random). Occupancy is the untested axis.

## Coordination
`matvec-occupancy-230x` is running and owns diagnosing WHY. You own the empirical
sweep of WHAT geometry wins. If your sweep finds the answer first, say so; if its
diagnosis lands first, use it to prune your axes.
