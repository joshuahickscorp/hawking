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

# LANE: dsv-expert
## Class: GPU_EXCLUSIVE for benchmarks, COMPILE otherwise.

## Surface
    crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs
        execute_layer :1205,  preload_moe_io :1710 (overlapped expert streaming)
        prefetch threads via std::thread::scope :2013
        command_buffers counter :117, incremented :678 and :2435 (moe_combine)
    shaders/dsv4f_activation_x_batch.metal:101  (simdgroup act_quant, batched)

## Target: expert execution

DSV4F routes top-6 plus a shared expert. The standing doctrine says sparse MoE
execution underfills the GPU when mapped as one GEMV per expert:

    BAD:        expert 1 GEMV -> expert 2 GEMV -> ... -> expert N GEMV
    PREFERRED:  device route -> compact active-expert worklist
                -> grouped/batched expert compute -> fused accumulation

Establish which shape DSV4F currently has, then measure per token:
    dispatches, useful work per dispatch, GPU busy vs gap ns
    bytes read/written, temporary bytes, buffer creations, rebinds
    occupancy proxy per expert kernel

**Tiny GPU work is poison** — many small dispatches leave the device idle between
launches. But measure the whole-token wall, not a microbenchmark: a grouping that
wins in isolation can lose in situ.

## Host expert gather is the specific anti-pattern to hunt
The preferred shape keeps routing on device end to end:

    GPU router -> GPU top-k -> GPU expert ids -> GPU worklist/address indirection
    -> GPU expert compute

The rejected shape bounces through the host for top-k, expert lookup or address
construction. Note that a **route-ID readback serializer hypothesis was already
REFUTED** on DSV4F — do not re-pay for that one. Look instead at payload staging,
address construction, and whether `preload_moe_io` overlap actually covers the
critical path or merely moves it.

Consider Metal argument buffers, device-side address tables and indirect command
mechanisms where measurement supports them.

## Transfer note
Q80 is building a general residency + address-indirection mechanism in the
`q80-runtime-residency` lane, and a packed-matvec decode pattern in
`q80-decode-kernels`. If your findings point the same way, say so — cross-model
transfer is explicitly valuable here.

## Correctness gate
Expected `hc_sha` preserved on the verified route, 0 fallbacks. Report both.
