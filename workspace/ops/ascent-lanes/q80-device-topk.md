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
# LANE: q80-device-topk
## Class: GPU_EXCLUSIVE for benchmarks, COMPILE otherwise.

## The wall

`q80-deltanet-gqa2` (merged, `ae19c053`) names the next Q80 bottleneck:

    host top-10 + 512-entry expert address-table rewrite
    ~157,820,565 ns/token of HOST work, with the GPU IDLE
    (moe_table_build 5.92 s over 28 forwards)

Its own conclusion: *"Device-side top-k is what would actually keep mixer/suffix on
the queue."*

This is the doctrine's canonical bad shape:

    GPU router -> CPU readback -> CPU top-k -> CPU expert lookup
                -> CPU address construction -> GPU experts

and the target shape:

    GPU router -> GPU top-k -> GPU expert ids -> GPU worklist / address
                indirection -> GPU expert compute

## Your job
Move top-k and address-table construction to the device so the host is not on the
critical path between the router and the experts.

1. Measure first: split the ~158 ms into router readback, top-k, expert lookup,
   address construction, and table write. Report ns for each.
2. Implement device-side top-k over the router logits, producing expert IDs in
   device memory.
3. Build the worklist / address indirection on device. `dsv-expert` did the
   analogous thing for DSV4F (branch `grok/dsv-expert-20260816-003212`): a
   six-entry gpuAddress table the worklist kernel indexes, with `useResources`.
   Read it — the pattern transfers.
4. Report GPU idle time before and after. The success condition is that the GPU
   stops waiting for the host between router and experts, not merely that a host
   timer got smaller.

## Important context
`q80-runtime-residency` found that even with experts resident, first-touch still
pays `catalog read + SHA-256 + MTLBuffer copy` (90-245 ms/token late-token). The
`q80-expert-first-touch` lane owns that. **Coordinate: you own routing and address
construction, they own payload upload.** Say in your report where the boundary
landed.

## Correctness gate
Generated ids exactly
    [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914]
Top-k ties and ordering are the classic device-side divergence: if the device
tie-breaks differently from the host, routing silently changes and the model still
produces plausible text. **Prove the device route IDs match the host route IDs
exactly**, per layer per token, not just that output looks right.
