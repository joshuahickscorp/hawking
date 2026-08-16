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
# LANE: q80-component-simdgroup
## Class: GPU_EXCLUSIVE for benchmarks, COMPILE otherwise.

## The finding — a kernel upgrade that was never wired through

`q80-deltanet-gqa2` (merged, `ae19c053`) reports:

> DeltaNet/GQA still use `qwen_uniform_q4_group64_matvec` (serial, **one thread per
> row**). Commit a47f8259 switched the **expert-table** kernel to simdgroup; the
> component dispatcher never followed. The simdgroup component kernel is documented
> not bit-identical, so it was not made the default (12-id gate).

So DeltaNet and GQA — 36 + 12 layers, ~29% of the Q80 token — run a **serial
one-thread-per-row matvec** while a simdgroup kernel already exists and is the
default elsewhere. This is the same class of defect `dsv-mla` found on DSV4F, where
RMSNorm dispatched on a single thread and MLA realized 24-36 GB/s of a ~750 GB/s
ceiling.

## Your job
1. Measure the component matvec's realized GB/s and occupancy on the serial path.
   Establish the regime before changing anything.
2. Wire the simdgroup component kernel through the component dispatcher.
3. **Resolve the bit-identity problem — this is the actual work.** The simdgroup
   kernel is documented as not bit-identical, which is why it is not the default.
   Determine WHY: almost certainly a parallel reduction changing floating-point
   summation order. Then either
     (a) make it bit-identical (fixed reduction order / deterministic tree), or
     (b) prove NUMERIC_EQUIVALENCE with a quantitative gate: report per-layer
         hidden drift, logit drift, and confirm the 12 generated ids are unchanged.
   Option (a) is strongly preferred. Do not simply flip the default and note the
   drift in passing.

## Correctness gate — non-negotiable
Generated ids must be exactly
    [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914]
Run enough repetitions to show it is stable, not lucky. `q80-deltanet-gqa2` matched
18/18; match that standard. Also run the `qwen80_fixture_advance_hybrid_state`
state-contract tests — DeltaNet is stateful and a silent state reset still produces
plausible text.

## Negative science — do NOT re-pay for these
`q80-deltanet-gqa2` already tested and REJECTED, all bit-identical, all failing to
win complete-token wall:
  - overlap (no-wait suffix/mixer): not robust; extra CBs 146 vs 98 tax prefill
  - fuse to 51 CBs: REGRESSED on the cleanest pair (516 ms vs 307 ms)
  - serial encoder: failed to win complete-token wall
All three remain behind env flags with defaults on the historical topology.
**Topology collapse is not the Q80 lever.** Occupancy is what is untested — that
is your lane. The same pattern held on DSV4F: collapsing 731 encoders to 43 was
bit-identical and moved attention GPU by nothing.
