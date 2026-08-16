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

# LANE: dsv-mla
## Class: GPU_EXCLUSIVE for benchmarks, COMPILE otherwise.

## Surface
    crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs
        43-layer loop :1085,  execute_layer :1205
    crates/hawking-core/src/gravity_deepseek_v4_token_ns_ledger.rs
    crates/hawking-core/examples/gravity_deepseek_v4_native_token_graph.rs:73

## Target: MLA / attention inside the GPU body

Expected DSV4F state to verify, not trust: ~1.299 s/token total, host-exclusive
~641 ms, Metal GPU ~401 ms, ~137 command buffers/token, 43 layers, 0 fallbacks.
The `dsv-host-wall` lane is the baseline authority — cite its number rather than
publishing a competing one, unless yours disagrees, in which case say so loudly.

For a 20 ms complete token the entire GPU body must land in roughly the 10-15 ms
class. **Any single GPU component above 20 ms is automatically a structural
blocker**, not a tuning target — say so plainly if you find one.

Your slice is MLA and sparse attention. Produce, per token:
    invocations, threads, threadgroups, simdgroup geometry
    bytes read / written, approximate arithmetic ops
    wall ns, ns/invocation
    occupancy proxy, memory-stall proxy, execution-stall proxy
    register pressure and threadgroup memory if observable
    dispatch gaps and command-buffer gaps attributable to this region

Then state whether MLA is bandwidth-limited, occupancy-limited, or gap-limited,
and attack accordingly. Do not assume; the system-wide measurement is ~0.79% of
the bandwidth ceiling with ~51% GPU idle, which points at gaps and occupancy
rather than arithmetic.

## MLA-specific angle worth checking
MLA keeps a compressed latent KV state. Persistent state should stay
device-addressable across tokens. Track:
    state bytes read/token, written/token, copies/token, sync/token
and look for read -> temp -> copy-back patterns where ping-pong or in-place
semantics would be safe. Rebuilding, rebinding or reallocating per-token state
that is reused across tokens is the standing anti-pattern here.

## Correctness gate
Expected `hc_sha` preserved on the verified route, 0 fallbacks. Report both.
Attention state is stateful — a silent reset still yields plausible output.

## Do not
- Do not restructure the reader/host IO (dsv-host-wall) or command-buffer topology
  (dsv-cb-collapse). Coordinate, do not overlap.
