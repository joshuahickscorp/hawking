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
# LANE: q80-recalibrate-capability-bar
## Class: GPU_EXCLUSIVE + MEMORY_HEAVY. Use ./tools/gpu_lane_lock.sh.
## Resolves a direct contradiction between two merged receipts.

## The contradiction
`q80-subbit-capability-curve` (merged) concludes NO_GO for sub-0.655 BPW: 0 of 588
recipes clear all organs at the **0.8604** organ-cosine bar, with the incumbent's
own holdout cosines at gate 0.8755, up 0.8665, **down 0.7684**.

`q80-mixed-generate` (merged) ran the **same 1.44445 BPW artifact** and it
**GENERATED COHERENT TEXT**: "Here's a function that reverses a string (i.e",
numeric drift 3.58e-7, no dense W.

So down_proj sits at 0.7684 - well under the 0.8604 bar - in an artifact that
demonstrably works. **The bar predicted failure where the model succeeds.**

Generation is ground truth. The bar is a screen. When they disagree, the screen is
wrong.

## Your job
1. **Re-derive the capability bar against generation, not against residual
   identity.** The 0.8604 figure came from a D23 residual-identity break-even. Find
   the organ-cosine level that actually predicts coherent generation, by generating
   at several deliberately-degraded points and recording where output breaks.
   Report the empirical cliff with the generated text at each point.
2. **Re-score the 588-recipe curve against the corrected bar.** The subbit lane's
   data is preserved at `receipts/ascent-2026-08-16/q80-subbit-capability-curve.SUMMARY.json`
   and its operator at `lab/operators/q80_subbit_capability_curve.py`. Do not re-run
   the sweep if the stored per-recipe organ cosines are sufficient to re-rank - just
   re-threshold. Report how many of the 588 clear the corrected bar.
3. **Then answer the real question**: does any recipe reach BPW <= 0.6552 while
   still generating coherently? Verify the top candidates by ACTUALLY GENERATING,
   not by cosine.

## What to keep from the previous lane
Its methodology was good and should be preserved: 110 pairs scored against a
16-sample NULL DISTRIBUTION rather than a single null. That fixed the earlier
`separated_from_null=FALSE` defect. Keep it.

Its framing also survives: bits are reachable, capability is the binding
constraint. What is unproven is WHERE the cliff sits.

## Honesty
If the corrected bar still rules out sub-0.655, say so - that is a clean NO_GO and
it retires an expensive research direction, which is valuable. But it must be a
NO_GO against a bar that generation validates, not one generation has already
contradicted.

Do NOT lower the bar to manufacture a pass. Derive it, from measured generation.
