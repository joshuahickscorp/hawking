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
# LANE: matvec-mlx-reference
## Class: GPU_EXCLUSIVE (short bursts). Use ./tools/gpu_lane_lock.sh.
## FAST LANE. The single sharpest question in the campaign. Report early, report short.

## The question
Our packed matvecs measure ~2.5 GB/s. A DRAM row probe on the SAME box in the SAME
run sustains 560-647 GB/s (receipts/ascent-2026-08-16/dram-row-locality.json).
That is a ~230x gap and we believe it is occupancy, not bandwidth.

**A control that streams memory is not the same as a control that does a matvec.**
A row probe proves the memory system is capable; it does not prove that a *quantized
matvec* can approach it. This lane closes that gap in the argument.

## What to do — smallest thing that settles it
Apple's MLX is on this machine's Python (the Qwen3.8 source is literally an MLX
repo). MLX has heavily optimized quantized matvec kernels for exactly this shape.

1. Measure **MLX quantized matvec GB/s** on this box at shapes matching our organs:
   Q80 expert projection 512x2048 (and 2048x512), Qwen3.8 MLP 5120x17408,
   at 4-bit and whatever low-bit MLX supports. Batch=1 decode shape, since that is
   our regime (arithmetic intensity ~1, every weight read exactly once).
2. Report GB/s for each, next to our 2.47-2.65 GB/s and the 560-647 GB/s row probe.
3. State the verdict plainly:
     - If MLX hits ~100-400 GB/s -> our kernels are the problem, the gap is real
       and addressable, and MLX's geometry is the reference to study. Say what
       geometry it uses (threadgroup size, simdgroup usage, vector width, work per
       thread) from its source or its dispatch.
     - If MLX also lands in the single-digit GB/s -> the 230x framing is WRONG,
       batch-1 quantized matvec is fundamentally latency/dependency-bound on this
       hardware, and the whole occupancy campaign needs rethinking. **This
       falsification would be the most valuable result in the campaign** - report
       it immediately and loudly.

## Constraints
- Do NOT rewrite our kernels in this lane. Measure and report only.
- Keep it SHORT. This is a reference measurement, not a research program. If MLX
  is not importable or the shapes do not map, say so in one line and stop rather
  than building a harness.
- Use `~/.grok-vision/bin/python` if the system python lacks mlx; check both.
- Label everything DIRTY_ENGINEERING; other lanes are running.

## Why this matters
Three models now depend on this single answer. Q80, DSV4F and Qwen3.8 all need to
escape ~0.4% of control efficiency to reach 50 TPS. If the ceiling is real and
reachable, all three get faster together. If it is not, all three targets need
rethinking. Speed matters more than completeness here.
