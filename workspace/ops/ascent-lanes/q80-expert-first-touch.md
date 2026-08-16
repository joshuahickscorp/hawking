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
# LANE: q80-expert-first-touch
## Class: GPU_EXCLUSIVE for benchmarks. HIGH VALUE, well-localized.

## The finding you are acting on
`q80-runtime-residency` (branch `grok/q80-runtime-residency-20260816-003204`,
commit `7ced3732`, receipt `receipts/ascent-2026-08-16/q80-runtime-residency.json`)
made experts device-resident. Result was honest and negative on wall time: C (old)
370/574/295 ms vs D (persistent 8 GiB) 461/387/356 ms — spreads overlap, no clean
speedup. All 12 runs bit-identical.

Its real finding: once an expert is resident the bind is microseconds, but
**`expert_address_table_bind` is still 90-245 ms/token on late tokens because new
`(layer, expert)` pairs keep appearing and each miss pays
`catalog read + SHA-256 + MTLBuffer copy`.**

**A SHA-256 per expert first-touch is immutable-identity work paid in the token
loop.** On this machine that exact pattern has been the top latency finding three
separate times: a per-token SHA, a clone-tree-on-open, and an `st_dev`-keyed
admission check that alone cost 28 s of startup and, once dropped, took startup
13.5 s -> 2.3 s and a warm repack 4606 s -> 95 s (48x).

## Do this
1. Attribute the 90-245 ms across its three parts separately: catalog read, SHA-256,
   MTLBuffer copy. Report ns for each. Do not assume the SHA dominates - measure.
2. Apply the standing law:
       cold artifact proof -> sealed session proof -> token path trusts the seal
   The catalog is immutable within a session. Verify once at admission, then never
   re-derive per expert first-touch. **Integrity must stay strong - move the check,
   do not delete it.** State exactly what is now verified once and what still
   guards it.
3. For the remaining copy: the catalog is mmap-able and expert chunks are
   contiguous. Consider a no-copy Metal bind of the mmap window instead of a
   copy - this is exactly what `dsv-expert` did for DSV4F
   (branch `grok/dsv-expert-20260816-003212`): pin the mmap'd window as an
   MTLBuffer, no `write_at` packing. Read that lane before designing yours.
   Note the constraint it hit: a 384 MiB Metal weight bound.
4. Also consider prefetching the next layer's likely experts during the current
   layer's GPU work, if routing makes that predictable. Measure, do not assume.

## Correctness gate
Generated ids must stay exactly
    [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431, 288, 264, 914]
Report the ids you got. A wrong expert payload is the obvious failure mode of any
caching or seal change - prove cached and freshly-read payloads are identical.
