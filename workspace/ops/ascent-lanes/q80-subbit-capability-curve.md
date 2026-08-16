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
# LANE: q80-subbit-capability-curve
## Class: CPU_HEAVY / MEMORY_HEAVY, GPU under the lock for any generation.
## Answers: where does capability actually break as BPW falls?

## The law you are serving (receipts/ascent-2026-08-16/FS_PER_WEIGHT_LAW.json)

    fs_per_weight = 152.6252 * BPW / efficiency        (model-independent)

Sub-100 fs/weight therefore requires **BPW < 0.6552** AND near-unity bandwidth
efficiency. Both are mandatory. A separate lane (`matvec-occupancy-230x`) owns the
efficiency half - packed matvecs currently run 2.5 GB/s against a measured 560-647
GB/s control on the same box.

**You own the BPW half.** The question is not "can we make bits smaller" - it is
**where does capability actually break**, measured, on the real Q80 pipeline.

## Current state
The packed <=1.5 artifact exists and is admitted:
    workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-candidates/mixed-1p5-v1
    complete_physical_bpw 1.44445, 13.4 GiB, 74,391 tensors
    per-organ physical: gate 1.126923, up 1.291849, down 1.285870, non-expert 8.250601
    packer: lab/operators/q80_mixed_representation_pack.py
    format: docs/QWEN80_MIXED_1P5_PACKED_FORMAT.md
Its coherence is NOT yet established - `q80-mixed-generate` is running that gate now.

## The deliverable: a capability-vs-BPW curve, not a single point
Sweep complete physical BPW downward and measure where capability degrades. At
minimum: ~1.44 (the existing artifact), ~1.0, ~0.8, ~0.655 (the sub-100 fs
threshold), and one point below it. For each point report:

    complete_physical_bpw   (computed from real packed bytes, not design)
    per-organ BPW
    output-space organ cosine against the D23 bar (0.8604)
    per-layer drift growth ratio, full depth or tiled - NOT a 4-layer extrapolation
    separation from a matched-magnitude null DISTRIBUTION (not one sample)
    generated text, verbatim, if it can be generated

**The curve is the product.** A single number at one BPW is worth much less than
knowing the shape and where the cliff is.

## Where the bits must come from
Non-expert is currently 8.250601 BPW and 2,439,063,174 bytes across 663 tensors -
about 17% of the artifact for 0.9% of the tensors. Under the identity
`complete = 0.97032*expert + 0.02968*nonexpert` the non-expert term contributes
~0.245 BPW of the 1.44. Dropping it to 4-bit buys roughly 0.12; the rest must come
from the routed experts, which are already at 1.235 mixed. So sub-0.655 complete
means routed experts near ~0.5 BPW. State this arithmetic yourself and correct me
if the identity coefficients have moved.

## Precedent and its caveat
Activation-aware sub-bit reached **0.167 BPW at 0.755 cosine on REAL activations**
for GLM - and every prior sub-bit negative was a Gaussian-proxy artifact, so the
technique is real. But 0.755 cosine is LOW, and **Q30 at static <=1.5 FAILED
coherence outright**. So the honest prior is: bits are reachable, capability is
the binding constraint. Do not repeat Q30's approach.

## Known input defect - state its effect on every point you report
The Q80 capture is route-starved: p10 = 34 rows against 2048 dims, p50 = 258, 221
never-routed (layer,expert) pairs, and NO on-disk post-SwiGLU X for down_proj. An
extension capture is checkpointed at layer 16/48. Any fit you make inherits this.
Report rows-per-fit per organ alongside each curve point, and say plainly which
points are underdetermined.

## Honesty
A curve showing the cliff sits ABOVE 0.655 BPW is a complete and valuable answer:
it would mean sub-100 fs/weight is unreachable for Q80 at preserved capability on
this architecture, and the campaign should stop pursuing it. Report that plainly if
it is what the evidence says. Do not manufacture a reachable-looking point.
