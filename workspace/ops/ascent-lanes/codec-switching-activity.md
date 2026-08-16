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
# LANE: codec-switching-activity
## Class: CPU_HEAVY analysis + one GPU verification run under the lock.
## This is the "manipulate the electricity" lane. Read the physics section first.

## The physics you are exploiting

Dynamic power in CMOS is

    P = alpha * C * V^2 * f

`C`, `V` and `f` belong to the hardware. **`alpha`, the activity factor - the
fraction of wires that TOGGLE per cycle - belongs to whoever chooses the bit
patterns.** Hawking chooses the bit patterns, because Hawking owns the codec.

Every weight fetched crosses a wide bus. A wire that holds its value costs
approximately nothing; a wire that flips charges and discharges its line
capacitance. So the energy of moving a packed weight stream is, to first order,
proportional to the **total Hamming distance between consecutively transmitted
words**, not to the number of words.

This is established hardware practice - bus-invert coding, Gray coding,
low-transition encoding - but it is essentially unexploited in LLM weight codecs,
because most codecs are chosen by RMSE alone and are agnostic to the bit patterns
they emit.

## THE KEY INSIGHT — this is why the lane is worth running

For any codebook or cluster-index codec, **the assignment of binary codes to
centroids is arbitrary**. Permuting which code means which centroid:

    costs EXACTLY ZERO bits
    changes the model NOT AT ALL (decode is a lookup; permute the table too)
    changes every bit pattern that crosses the bus

So there exists a free variable - a permutation - that is invisible to BPW,
invisible to accuracy, and directly multiplies the activity factor. Choosing it to
minimise expected Hamming distance between successively fetched codes is free
energy reduction, and possibly free latency reduction if bus contention or DRAM
row behaviour is affected.

## What to do

1. **Measure the current activity factor.** Take Q80's packed streams (the mixed
   codecs: gate binary_group, up binary+rice_q1, down hgravs01_r160_b3 - see
   `crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs` and
   `shaders/q80_mixed_decode.metal`). In the order the KERNEL ACTUALLY FETCHES
   THEM, compute:
       total bit transitions, mean Hamming distance between consecutive words,
       transitions per byte, and alpha versus the 0.5 random-data baseline
   Fetch order matters more than storage order - derive it from the kernel, not
   from the file layout.
2. **Find the permutation.** For the codebook/index-bearing components, solve or
   approximate the assignment that minimises expected consecutive Hamming
   distance under the empirical transition frequencies. This is a
   minimum-Hamming-distance assignment problem; a greedy or simulated-annealing
   solution is fine, optimality is not required. Report the alpha reduction.
3. **Prove it is free.** Show BPW is byte-for-byte unchanged and that decoded
   values are BIT-IDENTICAL after permuting both the codes and the lookup table.
   If it is not exactly free, it is a different and much weaker result - say so.
4. **Then look for the same free variable elsewhere**: sign/magnitude versus two's
   complement for residuals, bit-plane ordering, Gray-coding the quantisation
   levels so adjacent levels differ in one bit, and the packing order of the
   group-scale bytes.

## Honesty about what you can and cannot measure
You almost certainly CANNOT measure the energy delta directly - `powermetrics`
needs root and is unavailable in this session. Therefore:
  - Report the **transition-count reduction** as the primary measured result. It
    is a real, exactly-computable quantity, not an estimate.
  - Treat energy as DERIVED and label it so. Do not claim measured joules.
  - Also measure the WALL-CLOCK effect and report it honestly even if it is zero.
    Reduced switching may not show up in latency at all; the honest outcome may be
    "same speed, lower switching activity", which is still a real result on the
    energy axis and should be reported as exactly that.

## Do not
- Do not change any weight VALUE. This lane must be provably lossless: identical
  decoded tensors, identical BPW, identical generated ids.
- Do not fabricate an energy number from a datasheet and present it as measured.
