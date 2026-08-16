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
# LANE: dsv-attn-weight-io
## Class: GPU_EXCLUSIVE for benchmarks, MEMORY_HEAVY.

## The wall you are attacking — newly exposed, not previously visible

`dsv-expert-cache` (branch `grok/dsv-expert-cache-20260816-003822`, `072427c8`)
made expert payloads device-resident behind an 8 GiB cache. On a 100%-hit path:

    expert_slab_io   A (cache off): 776.4 / 612.1 / 1044.0 ms
                     B (100% hit):   24.5 /  29.5 /   27.3 MICROSECONDS

bit-identical `hc_sha c94da765`, cached==fresh memcmp 8/8 over 106,954,752 bytes.

**Killing that wall exposed the next one, which it had been hiding:**

    host.attn_weight_io_prefetch   ~379-533 ms, median 481,955,669 ns/token

That is now the largest single DSV4F cost. This is the standing law in action:
any >20% improvement invalidates the old bottleneck ranking, so the ranking is
rebuilt rather than the old plan continued.

## Your job
1. Decompose `attn_weight_io_prefetch` into ns classes: file read, mmap/page
   fault, memcpy, host->device fill, buffer bind, synchronization.
2. Ask the standing questions of each: can it disappear under residency, move to
   admission, move device-side, be cached, or overlap with GPU work?
3. The obvious hypothesis, which you must TEST rather than assume: attention
   weights are per-layer and REUSED EVERY TOKEN, unlike experts which are routed.
   If so they are a far better residency candidate than experts were — 43 layers of
   fixed attention weights, no routing, perfect hit rate by construction.
   **Measure their total byte footprint first** and check it against the real
   budget before designing anything.
4. Note the hard constraint another lane hit: a **384 MiB Metal weight bound**
   blocked pinning all 256 experts per layer. Establish whether attention weights
   fit inside it, or how they must be split.

## Interaction you must respect
`dsv-expert-cache` is NOT yet merged (it skews with an in-flight integration).
**Base your work on `grok/dsv-expert-cache-20260816-003822`** so the 482 ms is
actually exposed in your build; on main it is still hidden behind expert_slab_io
and you will not be able to measure it. Say explicitly which base you used.

## Measurement honesty
The expert-cache B numbers are a **repeated-BOS, 100%-hit best case**. Do not
inherit that framing. Report your win on a prompt with varied routing as well, and
state the hit rate. A number that only holds when every token routes identically
is a prompt-shaped result.

## Correctness gate
`hc_sha == c94da765c4bbf795b598d96209cd80821e5a81ab97a8712586f54b8c8b612597`,
0 fallbacks, and prove cached/pinned attention payloads are byte-identical to
freshly-read ones (the expert-cache lane did this with a memcmp; do the same).
