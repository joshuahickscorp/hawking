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
# LANE: dram-row-locality
## Class: GPU_EXCLUSIVE for benchmarks, CPU_HEAVY for layout analysis.
## The third electrical lever, and the only one that pays in BOTH energy and latency.

## The physics

DRAM is not flat. A read is:

    ACTIVATE row -> sense amplifiers charge -> READ column -> PRECHARGE

If the next access hits the SAME open row, only the column read happens: fast and
cheap. If it lands in a different row of the same bank, the controller must
precharge and activate again. **Row activate/precharge dominates DRAM energy - it
is roughly an order of magnitude more energy than the column read it enables - and
it costs real latency (tRP + tRCD) that no amount of bandwidth hides.**

So two streams that move the SAME number of bytes can differ substantially in both
energy and achieved bandwidth, purely by access order. Peak bandwidth figures
(819 GB/s here) assume near-perfect row locality; scattered access does not reach
them. This is a large part of why measured bandwidth on this box has been far below
ceiling - DSV4F MLA realized 24-36 GB/s of ~750, and after the occupancy fix
92 GB/s; Q80 measured ~1% of ceiling.

**Hawking controls physical layout at pack time and access order in the kernel.**
That is the whole lever, and it is free at runtime: layout is decided once.

## What to do

1. **Characterise current access order.** For Q80's mixed artifact and DSV4F's
   expert chunks, derive the address sequence the kernel actually issues for one
   token. Report: sequential run lengths, stride distribution, distinct 4 KiB /
   16 KiB pages touched, and estimated distinct DRAM rows per token. You cannot
   read Apple's memory-controller counters, so build this from the address stream,
   and say plainly that it is a derived model, not a hardware counter.
2. **Quantify the gap.** Compare achieved GB/s against 819 for each major stream.
   A stream at 1-12% of ceiling with a scattered address pattern is the signature
   you are hunting. Rank streams by (bytes moved) x (scatter).
3. **Relayout in execution order.** The standing rule is that physical layout
   should follow the order the token graph consumes tensors. Check whether it
   actually does. For MoE the interesting case is that expert selection is
   dynamic - so co-locate what is fetched TOGETHER (an expert's gate/up/down
   triplet, and its scales beside its codes) rather than what is numbered
   consecutively.
4. **Measure.** Paired reps, whole-token wall, plus realized GB/s per stream.

## Bounded, evidence-gated extension
Routing-cooccurrence-informed expert locality - placing experts that are frequently
co-routed near each other - is permitted ONLY if cheap evidence supports it. The
`q80-capture-coverage` lane measured the routing distribution and found it heavily
skewed (p10 = 34 rows, 221 never-routed pairs of 24576), which is exactly the
skew that makes co-location plausible. Use that existing evidence; do NOT run a
new expensive capture for this.

## Interaction
`dsv-expert-cache` (grok/dsv-expert-cache-20260816-003822) makes hot experts
resident, which removes their DRAM traffic entirely. Your win is therefore on the
COLD/streamed remainder and on Q80. Do not double-count a saving that residency
already took - state explicitly which streams you are claiming.

## Correctness
Layout changes must be provably value-preserving: identical decoded tensors,
identical BPW, identical generated ids (Q80: the 12-id list) and identical
`hc_sha c94da765...` (DSV4F). A relayout that changes a value is a bug, not a
tradeoff.

## Honesty
Report the wall-clock effect even if it is zero. "Better locality, same speed"
means the stream was not row-limited, which is itself a real finding that redirects
effort. Do not present a derived energy figure as measured.
