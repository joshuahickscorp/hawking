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

# LANE: dsv-cb-collapse
## Class: GPU_EXCLUSIVE for benchmarks, COMPILE otherwise

## The target: ~137 command buffers per token

For a 20 ms token budget, 137 command buffers means ~146 microseconds of budget per
CB before any useful work happens. That is structurally suspicious and must be
**audited, not cosmetically reduced**. Fewer CBs is not automatically the goal —
more useful work per submission is.

Counters live at `gravity_deepseek_v4_native_token_graph.rs:117`, incremented at
:678 (after a metal wait) and :2435 (moe_combine runloop). The increment at :678
sitting right after a wait is worth looking at hard: a wait per command buffer is a
full pipeline drain, and 137 drains per token would dominate everything else.

## What to produce

1. A complete census of the 137: for each command buffer, what dispatches it
   carries, why it is a separate submission, and whether anything waits on it.
   Group them — there will be a small number of repeating patterns across 43 layers.
2. Per token: `TOTAL_DISPATCHES`, `TOTAL_COMMAND_BUFFERS`, `TOTAL_SYNC_POINTS`,
   `TOTAL_READBACKS`, `TOTAL_BUFFER_CREATIONS`, `TOTAL_BUFFER_REBINDS`.
3. GPU gap accounting: `gap_ns = next_kernel_start - previous_kernel_end`, summed
   per token, and separately the gap *between command buffers*. State what
   fraction of the token is GPU-idle.
4. Then collapse what the census proves is collapsible.

## Candidate fusion/grouping surfaces (from the standing doctrine)
    norm + projection            router + top-k preparation
    expert gather/worklist + projection
    activation + quant/dequant   packed decode + matvec
    expert accumulation + residual
    state update + normalization

## The trap to avoid
A fused kernel can be SLOWER if it raises register pressure or spills. Every
fusion you land must compare **complete token wall** and an occupancy proxy, not
dispatch count alone. A fusion that halves CB count and slows the token is a
rejected result — report it as negative science, which is still a real deliverable.

## Correctness gate
Expected `hc_sha` preserved on the verified route, 0 fallbacks. Report both.

## Do not
- Do not restructure the reader or the host IO path; `dsv-host-wall` owns it.
- Cite `dsv-host-wall`'s baseline number rather than publishing a competing one,
  unless yours disagrees — in which case say so explicitly.
