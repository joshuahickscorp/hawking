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

# LANE: dsv-expert-cache
## Class: GPU_EXCLUSIVE for benchmarks, MEMORY_HEAVY. Use ./tools/gpu_lane_lock.sh.

## The measured target — this is now DSV4F's single largest cost

From `receipts/ascent-2026-08-16/DSV4F_HOST_WALL_BASELINE.json` (the baseline
AUTHORITY, 6 reps, cold discarded, GPU lock held for the whole series):

    body            1037.8 ms/token median   (min 1024.1, max 1067.7)
    wall            1205 ms/token median
    host_exclusive   564.9 ms/token median
    metal GPU        399.0 ms/token          (true GPU timestamps)
    **host.expert_slab_io  415.1 ms/token**  <- YOUR TARGET, 40% of the body

`expert_slab_io` is the streamed top-6 expert read+fill, and it is **GPU-idle
time** — the device waits while the host pulls expert payloads off disk.

Ignore any older DSV4F figure near 3.3 s/token; that came from a stale
`DSV4F_TOKEN_NS_LEDGER.json baseline_warm` entry and is superseded.

## The opportunity

The `dsv-resident-gravity` lane established that a <=1.5 BPW DSV4F would be
**50.83 GiB** and therefore resident-class on this 96 GB box. But that path needs
a determined teacher-X capture estimated at **~20 hours** of streamed forward, so
it is not reachable today.

**You do not need it.** DSV4F routes top-6 of 256 experts per layer. Expert
selection is skewed in practice, so a bounded RAM-resident LRU expert cache can
absorb most of the traffic **on the existing artifact, with no re-fit and no
representation change at all**. Q80 already does exactly this — see
`expert_cache: HashMap<(usize, u32), Qwen80ExpertGpuTriplet>` in
`crates/hawking-core/src/model/qwen80_uniform_q4_hybrid_decode.rs:971` and its
`upload_qwen80_expert_triplet` miss path. Read it; DSV4F should get the analogous
mechanism.

## Do this

1. **Measure the routing distribution first.** Over a few hundred tokens, record
   the (layer, expert) access sequence and compute the hit rate a cache of size
   N GiB would have achieved, for several N. This is a cheap offline calculation
   from a trace and it tells you the ceiling before you build anything. If the
   distribution is near-uniform the whole idea is refuted — report that and stop.
2. Size the cache against real headroom. Measured now: 96 GB total, ~27-40 GiB
   free with lanes running. Do not design for 96.
3. Implement the cache with an eviction policy the trace justifies. Keep payloads
   device-addressable where possible so a hit avoids both the disk read AND the
   host->device fill.
4. Report hit rate and the resulting `expert_slab_io` ns/token, paired reps.

## Honesty requirement
Report the hit rate, not just the speedup. A cache that only wins on a repetitive
prompt is a prompt-shaped result — say so, and test on a prompt with varied
routing. State how the win would change under a longer or more diverse context.

## Correctness gate
Expected `hc_sha` preserved on the verified route (baseline `c94da765`), 0
fallbacks. A cache that returns a stale or wrong expert payload is the obvious
failure mode: prove identity of cached vs freshly-read payloads.

## Do not
- Do not touch command-buffer topology (`dsv-cb-collapse`) or MLA (`dsv-mla`).
- Do not attempt the <=1.5 pack; that is gated behind the capture.
