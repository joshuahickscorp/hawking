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
# LANE: q80-mixed-generate
## Class: GPU_EXCLUSIVE + MEMORY_HEAVY. Use ./tools/gpu_lane_lock.sh.
## HIGHEST VALUE LANE IN THE CAMPAIGN. This is the density gate.

## What now exists on main

**The <=1.5 BPW Q80 artifact is PACKED** (merged a57a7d0a, receipt
`receipts/ascent-2026-08-16/Q80_PACK.json`):

    root   workspace/campaign/records/ascension-sandbox/physical/qwen80/
             quality-candidates/mixed-1p5-v1
    74,391 tensors / 50 segments / 13.4 GiB
    all_required_execute_bytes  14,385,668,506
    complete_physical_bpw       1.44445   (threshold 1.5, margin 0.05555)
    Rust open confirms          1.4444456847927971
    per-organ physical BPW: gate 1.126923, up 1.291849, down 1.285870,
                            non-expert 8.250601
    format doc: docs/QWEN80_MIXED_1P5_PACKED_FORMAT.md
    catalog:    crates/hawking-core/src/model/qwen80_mixed_catalog.rs
                (Qwen80MixedStreamingCatalog::open)
    inspector:  crates/hawking-core/examples/ascension_qwen80_mixed_catalog_inspect.rs

**The mixed decode kernels also exist** (merged b5bc4afa + decode-throughput):

    crates/hawking-core/shaders/q80_mixed_decode.metal
    crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs
    crates/hawking-core/examples/ascension_qwen80_mixed_decode_kernel_parity.rs
    measured: gate 6.875 us, up 17.25 us, down 13.959 us per organ
    numeric gates: gate 1.81e-5, up 1.10e-5, down 1.14e-5 at tol 2e-5,
                   rice indices bit-identical, no dense W

**They are not wired together.** The pack lane's closing line: *"native mixed decode
kernel does not exist (kernel lane). Until that kernel runs, generation cannot be
measured; no ns/token for this artifact."*

## Your job — the gate

Wire the decode kernels to the mixed catalog and **generate text**.

Everything else in this campaign is instrumentation. Organ cosine is a screen.
**Generation is the gate.** Until this artifact produces coherent output there is
no <=1.5 claim, regardless of how good the BPW is.

1. Bind `Qwen80MixedStreamingCatalog` into the hybrid decode session so routed
   expert organs are served by the mixed codecs (gate binary_group, up
   binary+rice_q1, down hgravs01_r160_b3) and non-experts by their 8-bit form.
   Reuse the existing Q4 hybrid graph structure - only the expert weight service
   changes. Do NOT rebuild the token graph.
2. **No dense reconstruction.** packed bytes -> registers/simdgroup -> decode ->
   multiply -> accumulate. A path that materializes a dense W is REJECTED even if
   it is faster; it defeats the artifact's purpose and reintroduces the memory
   footprint. Report explicitly that no dense weight tensor is materialized.
3. Generate with the standard harness prompt and report the actual text and ids:
       --prompt "Write a function that reverses a string." --max-new-tokens 12
   The Q4 vehicle produced [8420, 594, 264, 4285, 729, 304, 13027, 429, 17431,
   288, 264, 914] = "Here's a simple function in Python that reverses a string".
   **Do NOT expect the mixed artifact to match those ids** - it is a different
   representation, not a bit-exact requantization. What matters is whether the
   output is COHERENT ENGLISH that answers the prompt.
4. Report a first ns/token for the artifact, with paired reps, plus its stage
   split. This is the artifact that matters; the 4.259 BPW vehicle is abandoned.

## The honest possible outcomes, all valuable
    COHERENT     - text is sensible. The <=1.5 path is ALIVE. Report the text.
    DEGRADED     - grammatical but wrong/repetitive/drifting. Report verbatim
                   samples and which organs you suspect.
    INCOHERENT   - gibberish. This is the Q30 failure mode repeating. Report it
                   immediately and plainly; it redirects the entire campaign.

**Do not dress up a degraded result as success, and do not hide an incoherent
one.** A clear incoherent verdict is worth more than an ambiguous optimistic one.
Print raw generated text verbatim, not a summary of it.

## Context that bears on the risk
The coherence probe was INCONCLUSIVE, not green: mixed rel-L2 grew 1.2772x/layer
against a null of 1.0061x, extrapolating to 16211x at layer 48 - but that was
`geo^44` from only a 4-LAYER span, `separated_from_null` was FALSE, and 395/2048
organs had clamped rank at that time. The packed artifact does NOT clamp rank.
Capture is route-starved (p10 = 34 rows vs 2048 dims, 221 never-routed pairs, no
on-disk post-SwiGLU X), so down_proj is the organ most likely to be wrong.
`q80-coherence-deep` is attacking the measurement in parallel. **You are the
ground truth that outranks all of it** - generation settles what drift curves only
predict.

## Correctness
Grade execution against what the ARTIFACT encodes (the CPU/numpy oracle), never
against the BF16 parent. A kernel that faithfully executes a bad representation is
a CORRECT kernel - that failure belongs to Gravity. Keeping these separate is what
previously exonerated the runtime and correctly redirected effort to the foundry.
0 fallbacks. Report fallback_count.

## Commit
You are on `gate` (unsandboxed). Commit normally and verify with `git log` that it
landed. Several lanes here hit Seatbelt/macl denials, finished ahead=0, and nearly
lost their work.
