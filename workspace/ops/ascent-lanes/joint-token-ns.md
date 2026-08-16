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
# LANE: joint-token-ns
## Class: STATIC_ANALYSIS / LIGHT_CONTROL. Read-mostly, no GPU needed.
## Mandate: prevent per-stage optimization from over-specializing into local optima.

## Why you exist

Eleven lanes are each attacking one stage of one model. That is efficient and it is
also the classic failure mode: **every lane optimizes its own stage, each reports a
win, and tok/s barely moves** — because the wins were in overlapped time, or the
cost moved into a neighbouring stage, or the stage was never on the critical path.

You own the WHOLE TOKEN for BOTH models, and nothing else.

## Deliverable 1 — one schema, both models

Q80 and DSV4F currently keep separate ledgers with different shapes:
    crates/hawking-core/src/model/qwen80_token_ns_ledger.rs
    crates/hawking-core/src/gravity_deepseek_v4_token_ns_ledger.rs

Define ONE `TOKEN_NS` schema that both emit, so a stage in one is comparable to a
stage in the other and cross-model transfer becomes visible rather than anecdotal.
Everything in NANOSECONDS. Do not rewrite either runtime; add a common emit/adapter
layer and keep both existing ledgers working.

Required per stage: `stage, substage, calls_per_token, ns_per_call, ns_per_token,
pct_of_token, resource_class {CPU|GPU|DRAM|IO|SYNC}, serial_or_overlappable,
removable_or_necessary, confidence, method, commit`.

Required per token: `TOTAL_TOKEN_NS, TOTAL_GPU_BUSY_NS, TOTAL_GPU_IDLE_NS,
TOTAL_GPU_GAP_NS, TOTAL_CPU_CRITICAL_NS, TOTAL_DISPATCHES, TOTAL_COMMAND_BUFFERS,
TOTAL_SYNC_POINTS, TOTAL_READBACKS, TOTAL_BUFFER_CREATIONS, TOTAL_BUFFER_REBINDS,
DRAM_BYTES_PER_TOKEN, TEMP_BYTES_PER_TOKEN`.

## Deliverable 2 — the closure identity, enforced

    sum(stage_ns) + residual_ns == TOTAL_TOKEN_NS

Emit `residual_ns` explicitly and fail loudly when it exceeds a stated fraction.
An unattributed residual is where real cost hides. The Q80 baseline already shows
one: stages sum to ~15.23 s against a 15.6 s run.

Equally important, and the reason this lane exists:

    sum(ns saved by all lanes)  is NOT  (token_before - token_after)

Build a small reconciler that takes the per-lane claimed savings and the measured
whole-token delta and reports the DISCREPANCY. A lane whose win does not appear in
the whole token has either optimized overlapped time, moved the cost sideways, or
measured something off the critical path. Name which, per lane.

## Deliverable 3 — critical path, not just totals

Separate SERIAL from OVERLAPPABLE time. On DSV4F specifically, `dsv-host-wall`
reported parallel sums far larger than the token itself (`path_resolve` 1318 ms,
`verify` 2505 ms against a 1038 ms token) because they are summed across threads.
**A parallel sum is not token latency.** State, for each model, what the critical
path actually is, and flag any receipt in `receipts/ascent-2026-08-16/` that
presents a parallel sum where a critical-path number belongs.

## Inputs (all on main)
    receipts/ascent-2026-08-16/ASCENT_STATE.json          (ledger of record)
    receipts/ascent-2026-08-16/DSV4F_HOST_WALL_BASELINE.json   (DSV4F authority)
    receipts/ascent-2026-08-16/Q80_BASELINE_2026_08_16.json    (Q80 authority)
    receipts/ascent-2026-08-16/CROSS_ADVERSARIAL_FINDINGS.json (known defects)
    receipts/ascent-2026-08-16/dsv-cb-collapse.json, dsv_expert.json,
      DSV_MLA_2026_08_16.json, q80-decode-kernels.json, q80-runtime-residency.json

## Honesty
If the reconciliation shows the lane wins do not add up to the measured token
improvement, **say so plainly with numbers**. That is the single most valuable thing
you can produce, and it is exactly what per-stage lanes cannot see about themselves.

## Do not
- Do not optimize. Do not modify runtime hot paths. You make the token legible and
  the claims reconcilable.
