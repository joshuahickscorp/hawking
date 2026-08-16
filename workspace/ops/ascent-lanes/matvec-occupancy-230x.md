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
# LANE: matvec-occupancy-230x
## Class: GPU_EXCLUSIVE. Use ./tools/gpu_lane_lock.sh.
## THE LARGEST IDENTIFIED HEADROOM IN THE CAMPAIGN. Treat as top priority.

## The measurement (merged, receipts/ascent-2026-08-16/dram-row-locality.json)

A control probe on the SAME box in the SAME run:

    dram_row_probe sequential   560.25 GB/s
    dram_row_probe conflict     646.71 GB/s

The actual packed matvec kernels, same box, same run:

    q80_q4_gate_matvec            2.65 GB/s
    q80_mixed_binary_gate         2.47 GB/s
    dsv4f_fp4_reduced             0.93 GB/s

**That is a ~230x gap against a measured CONTROL, not a datasheet ceiling.** The
memory system is demonstrably capable of 560-647 GB/s on this machine. The kernels
are therefore NOT memory-bound. They are under-occupied by roughly two orders of
magnitude.

If packed matvec reached even 100 GB/s - still 6x below the control - the Q80
routed decode projection falls from 18.28 ms to roughly 0.46 ms per token.

## Why this is credible and not a measurement artifact
Four independent results already point the same way:
- Q80 measured ~1% of the bandwidth ceiling with ~51% GPU idle.
- DSV4F MLA realized 24-36 GB/s of ~750 and was diagnosed occupancy-limited.
- The Q80 component matvec was found running SERIAL, one thread per row; wiring
  simdgroup gave a 3x mixer win (111 -> 36 ms).
- Topology collapse FAILED on both models (Q80 fuse regressed 516 vs 307 ms; DSV4F
  731 -> 43 encoders moved attention GPU by nothing) while every occupancy fix won.

The consistent story: this codebase's kernels are dispatch- and occupancy-starved,
not bandwidth-starved.

## Your job
1. **Reproduce the control and the gap yourself** before optimizing. Report the
   probe GB/s and each kernel's GB/s from your own run. If the gap does not
   reproduce, say so immediately - that is the single most valuable thing you could
   find, and it would invalidate this lane.
2. **Find why.** For each packed matvec kernel report: threads, threadgroups,
   threads-per-threadgroup, simdgroups-per-threadgroup, occupancy proxy, register
   pressure and threadgroup memory if observable, bytes read per thread, and the
   arithmetic-to-load ratio. Name the limiter with evidence: launch geometry,
   serialization, register spill, insufficient parallel work, or memory access
   pattern.
3. **Fix the largest one.** The control shows what the hardware does when fed
   properly - use it as the target shape. Likely candidates, in rough order:
   too few threads in flight; one-thread-per-row serialization surviving somewhere;
   scalar loads where vector loads are legal; no simdgroup cooperation on the
   reduction; threadgroup size not a multiple of the execution width.
4. Report achieved GB/s before/after per kernel, plus the recomputed 48x10 routed
   projection and the honest remaining distance from the 8 ms expert budget.

## Hard rules
- NEVER materialize a dense weight tensor. packed -> registers/simdgroup ->
  decode -> multiply -> accumulate.
- Keep the shipped numeric gates at least as tight: gate 1.81e-5, up 1.10e-5,
  down 1.14e-5 at tolerance 2e-5, rice indices bit-identical, 0 fallbacks.
  Grade against the ARTIFACT oracle, never the BF16 parent.
- A widening that raises register pressure can be net SLOWER. Compare complete
  token wall and occupancy, never GB/s alone. Report regressions as real results.

## Do not re-pay for these
- Dispatch/encoder/topology collapse: REFUTED on both models (see above).
- DRAM row interleaving: Q4 and binary both LOST; only FP4 gained; live wall
  unchanged.
- Expert routing co-occurrence layout: WEAK, 1.037x.
- Switching-activity permutation: real but not the current wall; alpha is already
  ~0.5 (random).
