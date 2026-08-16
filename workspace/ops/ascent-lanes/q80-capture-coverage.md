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

# LANE: q80-capture-coverage
## Class: CPU_HEAVY / MEMORY_HEAVY / IO_HEAVY. Lock anything GPU or >8 GiB.
## No GPU contention with the runtime lanes — you are the density substrate.

## Why this lane exists

Q80's <=1.5 BPW representation is fit against captured activations. This codebase
has been destroyed twice by capture defects, and both failures were INVISIBLE in
the scores:

1. **Underdetermined fits.** A prior Q80 campaign fit with a median of **92 rows
   against 2048 dimensions**. Every fit was underdetermined and every resulting
   score was meaningless. Compounding it, `rank = min(budget, n_fit_rows)` plus a
   per-layer capture budget capped N <= 85, so the rank that actually worked
   (r192) was unreachable BY CONSTRUCTION. Also measured: 42.8% of 6144 expert
   pairs were never routed at all in 3929 tokens.
2. **Calibration on a broken baseline.** X was once captured from a 0.7966
   gibberish baseline, so every score ranked the wrong trajectory. AWQ/GPTQ
   calibrate against full precision precisely to avoid this.

The current mixed representation cites a 25258-token capture. **Your job is to
prove that capture is sufficient, or to fix it.**

## What to produce

1. **Coverage audit** of the existing capture:
   - rows-per-fit versus fitted dimension, per organ (gate/up/down), reported as a
     distribution — min, median, max — not just a mean.
   - expert routing coverage: what fraction of (layer, expert) pairs are observed,
     and the occupancy histogram. Never-routed experts cannot be fit at all.
   - confirm the source is the **BF16 model** at
     `workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next`, not any
     quantized or degraded artifact. State the provenance you actually verified.
   - confirm down_proj capture is **post-SwiGLU intermediate X**, not the layer
     hidden. This is the known-correct target and a prior blocker.
2. **A verdict:** is any organ underdetermined? Name it, with numbers.
3. **If under-covered, extend the capture.** Use a per-expert reservoir so rare
   experts still accumulate rows, rather than a flat per-layer budget that starves
   them. Flush hidden states per layer rather than holding all layers, which is
   what previously forced the tiny N.

## Efficiency constraints
- Do NOT write a giant JSON index. A 1.38 GB `capture-result.json` was a real
  iteration wall; a CSR/mmap index cut a census from 1112 s to 0.052 s. Use the
  compact form.
- Capture is memory-heavy. Stream and flush; do not hold the model plus all
  activations resident.

## Do not
- Do not re-fit codecs or change the representation. The probe and pack lanes own
  those. You establish whether their inputs are trustworthy.
