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

# LANE: dsv-host-wall
## Class: GPU_EXCLUSIVE for benchmarks, IO_HEAVY, COMPILE otherwise
## You are the BASELINE AUTHORITY for DSV4F. Other DSV lanes will cite your number.

## Surface
- Native token graph: `crates/hawking-core/src/gravity_deepseek_v4_native_token_graph.rs`
  - 43-layer loop at :1085, `execute_layer` at :1205
  - counters struct at :117, `command_buffers` increments at :678 and :2435
  - `preload_moe_io` overlapped streaming at :1710
  - prefetch threads via `std::thread::scope` at :2013
- Reader / 148 GiB artifact: `crates/hawking-core/src/gravity_deepseek_v4.rs`
  - `admit()` :456, `read_verified_range()` :1190 (unbounded host alloc),
    `read_verified_range_view()` :1204, `read_verified_full_view()` :1289
- Benchmark: `crates/hawking-core/examples/gravity_deepseek_v4_native_token_graph.rs:73`
- Existing ledger: `crates/hawking-core/src/gravity_deepseek_v4_token_ns_ledger.rs`

## Expected state to VERIFY (not to trust)
    ~1.299 s/token, ~0.77 tok/s
    host-exclusive ~641 ms, Metal GPU ~401 ms
    ~137 command buffers/token
    43-layer full token graph, 0 fallbacks, hc_sha preserved on the verified route

**Job zero: reproduce this and publish the authoritative current number**, with
paired reps, under the GPU lock. If reality differs from the above, reality wins
and you say so loudly.

## The target: the ~641 ms host-exclusive wall

Decompose host-exclusive time into these classes and report ns/token for each:

    file/source reads          mmap / page-fault behaviour
    memcpy                     address lookup
    table construction         route preparation
    buffer lifecycle           state movement
    synchronization            validation / identity

For every class that survives, answer these in your report — one line each, no
hand-waving:

    Can it disappear entirely under resident packed execution?
    Can it move to admission (paid once)?
    Can it move device-side?
    Can it be cached?
    Can it overlap with GPU work?

## Strong prior from this codebase's history
Immutable-identity recomputation has been the real latency **three separate times**
here: a SHA per token, a clone-tree-on-open, and an `st_dev`-keyed admission check
that alone cost 28 s of startup. `read_verified_range` / `read_verified_*` names
suggest per-call verification. **Measure how often verification actually runs per
token and what it costs.** The correct shape is: cold artifact proof once, sealed
session proof, token path trusts the seal. Integrity must stay strong; only the
repetition disappears.

Also check: `read_verified_range()` at :1190 is flagged as an unbounded host
allocation. Count allocations and bytes per token.

## Do not
- Do not polish streamed-source infrastructure that a resident <=1.5 BPW model
  would delete outright. If a stage only exists because the model is streamed,
  say so and leave it — the `dsv-resident-gravity` lane is attacking that.
- Do not touch command-buffer topology; `dsv-cb-collapse` owns that.

## Correctness gate
The verified route's expected `hc_sha` must be preserved, and fallbacks must stay
at 0. Report both explicitly.
