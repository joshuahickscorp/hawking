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
# LANE: q80-integrate-graph
## Class: GPU_EXCLUSIVE for verification, COMPILE otherwise.
## INTEGRATION lane. The science is done. Do not redo it, do not re-benchmark from scratch.

## Situation
Three Q80 lanes independently modified the same decode path. `q80-deltanet-gqa2` is
already merged to main; two are not, and they conflict with it and each other.

    ON MAIN   grok/q80-deltanet-gqa2 (ae19c053, merged 49b452f1)
        adds OverlapMode enum + env-gated overlap/fuse/serial paths.
        DEFAULTS STAY ON THE HISTORICAL TOPOLOGY - all three attacks were
        bit-identical and all FAILED to win complete-token wall (fuse regressed
        516 vs 307 ms). Keep them opt-in. Do not promote any of them.

    NOT MERGED  grok/q80-expert-first-touch-20260816-010044 (5f78cb35, ahead=2)
        kills catalog-read + SHA-256 + MTLBuffer copy on expert first-touch;
        env HAWKING_Q80_EXPERT_NOCOPY (default on),
            HAWKING_Q80_EXPERT_PREFETCH (default on, prev-token next-layer prefault)
        MEASURED C 518.7/556.7/339.2 ms -> D 208.6/383.0/188.2 ms
        bit-identical, ids match, mmap == rehash verified
        touches: metal/mod.rs, qwen80_uniform_q4_hybrid_decode.rs,
                 qwen80_device_expert_table.rs, the greedy example

    NOT MERGED  grok/q80-ns-ledger-20260816-010044 (ahead=1)
        populates the seven stages that reported 0.0000 and enforces the closure
        identity. Measurement only; production CB shape unchanged.
        MEASURED moe_table_build 246283892 ns = 57.4% of a 428985708 ns wall
        touches: qwen80_token_ns_ledger.rs, qwen80_uniform_q4_hybrid_decode.rs,
                 metal/mod.rs, the greedy example

## Known conflicts (I attempted the merge; these are the real ones)
    metal/mod.rs      1 conflict, additive: HEAD re-exports vs lane's device
                      memory-ceiling helpers (max_buffer_length via objc, NOT the
                      metal-rs 0.29 feature-set helper). Take both.
    qwen80_uniform_q4_hybrid_decode.rs
      C1 @~97   additive: env-flag doc comments. Take both.
      C2 @~132  additive: OverlapMode enum vs the lane's prefetch flag. Take both.
      C3 @~4033 THE REAL ONE: 4 lines (HEAD's OverlapMode::Off branch calling
                forward_token_device_serial_waits) versus 172 lines (the lane's
                no-copy pack path). These must COMPOSE: the no-copy expert path
                has to work under the default historical topology, with
                OverlapMode's opt-in branches still reachable.

## Requirements
1. All three land. Defaults: historical topology ON (per deltanet-gqa2), expert
   no-copy ON, expert prefetch ON, full ns ledger populated.
2. Every env escape still works, since they are the A/B harness:
       HAWKING_Q80_EXPERT_NOCOPY=0, HAWKING_Q80_EXPERT_PREFETCH=0,
       plus deltanet-gqa2's overlap/fuse/serial flags
   Verify all build and that defaults are correct.
3. **NO WHOLESALE FILE COPY.** main carries deltanet-gqa2 and joint-token-ns work
   that a blind overwrite would silently revert.

## Proof required
    generated ids EXACTLY [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914]
    on at least 6 runs (deltanet-gqa2 matched 18/18; do not regress the standard)
    qwen80_fixture_advance_hybrid_state state-contract tests pass
    the expert first-touch win SURVIVES: paired reps at or better than
      D 208.6/383.0/188.2 ms against C 518.7/556.7/339.2 ms
    the ns ledger still populates all stages and its closure identity holds
A merge that compiles but loses the win, or regresses the ids, is a failed lane.
Report measured numbers, not "should be equivalent".

## Commit
You are on `gate` (unsandboxed). Commit normally, then verify with `git log` that
it landed on your branch. Several lanes here hit Seatbelt/macl denials writing to
.git, finished ahead=0, and nearly lost their work.
