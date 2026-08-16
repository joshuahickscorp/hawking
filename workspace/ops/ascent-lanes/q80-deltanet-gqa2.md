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

# WHY THIS LANE SURVIVES THE ARTIFACT CHANGE — read this first

Q80's uniform-Q4 vehicle (4.259241 BPW) has been **abandoned as a target**. Q80 is
being re-gravitied to a <=1.5 complete-physical-BPW artifact in parallel lanes.

**Your lane is artifact-INDEPENDENT.** The cost you are attacking is a property of
the token graph, the dispatch topology, the state handling or the host/device
split — not of how expert weights are encoded. It will still be there, essentially
unchanged, on the <=1.5 artifact.

So: use the Q4 catalog purely as a **test harness and correctness reference**. It
is a convenient runnable vehicle, nothing more.

**The design rule that follows from this:** do not hard-code anything to the
uniform-Q4 representation. Where you touch a path that knows about weight
encoding, keep the mechanism generic so the <=1.5 artifact inherits it for free.
If you cannot avoid a Q4-specific assumption, isolate it behind a seam and say so
in your report. A win that has to be rebuilt for the real artifact is a half win.

Q80 measured baseline on the Q4 harness (2026-08-16, DIRTY, 12 new tokens):
    steady_state 2.479023 tok/s = 403 ms/token; prefill 11.16 s
    stage_secs over the 15.6 s run:
        moe_table_build 9.0777 (58%)   deltanet 3.3269 (21%)
        moe_combine 1.6958 (11%)       gqa 1.1335 (7%)
    fallback_count=1637 (vec=265 act=1344 sample=28)
    table_builds=1344  table_dispatches=6720
    generated ids: [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914]

Target is a 20 ms complete token (50 TPS). Everything above is ~20x too slow.

---

# LANE: q80-deltanet-gqa2
## Class: GPU_EXCLUSIVE for benchmarks, COMPILE otherwise.

## Target: 29% of the token that has nothing to do with expert encoding

    deltanet    3.3269 s  (21%)
    gqa         1.1335 s  ( 7%)
    moe_combine 1.6958 s  (11%)   <- in scope if the split shows dispatch/sync,
                                     not real reduction work

Q80 is a hybrid: 36 DeltaNet layers + 12 GQA layers. None of this cost is a
function of how expert weights are encoded, so **all of it survives onto the
<=1.5 artifact**. That makes it some of the most durable work available right now.

## Establish before optimizing
1. Split `deltanet_secs` into substages. `Qwen80ActivationClassTimes`
   (`qwen80_uniform_q4_hybrid_decode.rs:752`) already declares
   `deltanet_conv_secs` and `deltanet_recurrent_secs` — in the baseline these
   reported ~0, which means either they are genuinely small or nothing populates
   them. Determine which; a stage that reads zero because it is never written is
   a measurement bug worth fixing on its own.
2. Per token, for DeltaNet and GQA separately: dispatches, command buffers,
   GPU busy ns, GPU gap ns (`next_kernel_start - previous_kernel_end`), host ns,
   bytes read, bytes written.
3. Classify each, explicitly:
       GPU idle because no work is ready  -> batching, fusion, device-side
                                             scheduling, removing CPU dependencies
       GPU busy but under-occupied        -> threadgroup geometry, simdgroup
                                             mapping, tiling, register pressure,
                                             threadgroup memory
   These have OPPOSITE fixes. Guessing wrong wastes the lane.

## Context
- The default matvec kernel was recently switched to a simdgroup variant
  (commit a47f8259). Verify whether DeltaNet/GQA actually reach it or still run
  the serial path.
- System-wide measurement: ~0.79% of the 700-800 GB/s bandwidth ceiling with ~51%
  GPU idle. Assume gaps and occupancy dominate until your numbers say otherwise.
- A fusion that raises register pressure can be net slower. Compare complete token
  wall and an occupancy proxy, never dispatch count alone. A negative result here
  is real science — report it.

## Correctness gate
Generated ids must stay exactly the 12 listed above. DeltaNet is stateful and a
silent state reset still produces plausible-looking text, so also run the state
contract tests (`qwen80_fixture_advance_hybrid_state`) and report.
