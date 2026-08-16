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

# Q80 REGRAVITY — shared context for all three density lanes

## The order from the user, verbatim in effect
Stop using the uniform-Q4 vehicle. Q80 must be re-gravitied to a **<=1.5 complete
physical BPW artifact that generates coherent text**. The Q4 4.259-BPW artifact is
abandoned as a target; it may only be used as a correctness reference.

## Source weights ARE on device (confirmed 2026-08-16)
    workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next
    40 safetensors shards, 148 GB, BF16
This is the calibration and parity authority. All captures and all fits come from
THIS, never from a degraded or quantized baseline.

## The current density state — receipts/QWEN80_MIXED_REPRESENTATION_UNDER_1_5.json
    status: SCREEN_PASSED_NOT_YET_PACKED_OR_GENERATED
    identity: complete_bpw = 0.97032*expert_bpw + 0.02968*nonexpert_bpw
    organ_cosine_bar: 0.8604  (D23 residual-identity break-even, from Q80's own
                               25258-token capture)

    gate_proj: binary_group                              expert_bpw 1.1269  cos 0.8586-0.8932
    up_proj:   binary + rice_q1_rms sparse resid @2%     expert_bpw 1.2918  cos 0.86416-0.86524
               (8.24 bits/outlier)
    down_proj: hgravs01_r160_b3 activation-weighted      expert_bpw 1.27    cos 0.8862-0.8978
               low-rank, scored on POST-SwiGLU intermediate

    mixed_expert_bpw = 1.22957
    complete_bpw: nonexpert_8bit=1.43051  6bit=1.37115  4bit=1.31179
    margin at 8-bit non-expert = 0.06949 below the 1.5 ceiling
    sensitive_3pct_untouched = true

**The three open gaps, stated by the receipt itself:**
    artifact_packed: false
    decode_kernel_exists: false
    coherence_generation_tested: false

Each of the three lanes closes exactly one. Do not drift into another lane's gap.

## Existing implementations — read before writing anything new
    lab/operators/q80_residual_encoding_sweep.py         (rice_q1 residual)
    lab/operators/q80_representation_frontier_sweep.py   (binary_group, frontier)
    lab/operators/hgravs01_adapter.py                    (low-rank hgravs01)
    lab/operators/doctor6/prescribe.py, doctor6/rungs.py
    crates/.../qwen_complete_binary/activation_weighted_svd.rs
    crates/hawking-core/examples/ascension_qwen30_hgravs01_packed_matvec_parity.rs
      ^ this is the Q30 precedent for a packed matvec parity harness. Read it.

## THE RISK THAT KILLS THIS PLAN — take it seriously
An organ cosine of ~0.86 is a **screen, not a guarantee**. This codebase has a
directly relevant failure: **Q30 at static <=1.5 BPW FAILED coherence.** Related
measured facts from this machine:
- the GLM residual stream is EXPANSIVE, 1.4-2.4x per layer, so per-organ error
  does not stay put — it compounds with depth;
- a functional-student arc was CLOSED after the student diverged by layer 4-8 in
  all 40 layers;
- raw activation cosine is a deceptive metric here (measured null baseline 0.898 —
  i.e. cosine 0.898 can mean NOTHING).

Note that the 0.8604 bar sits BELOW that 0.898 null. That is not automatically
fatal — the bar is a residual-identity break-even, a different quantity — but it
means per-organ cosine cannot be the thing that certifies this artifact.
**Generation is the gate.** Nothing else counts.

## Standing negative science — do not re-pay
- Cross-expert shared basis: REFUTED, experts mutually orthogonal (cos 0.004).
- Single-family representation: INSUFFICIENT — that is why this is per-component.
- down_proj must be fit on POST-SwiGLU X, never the layer hidden. It also INVERTS
  the family ranking (low-rank beats binary there).
- Fits with fewer captured rows than the tensor dimension are UNDERDETERMINED and
  their scores are meaningless. A previous Q80 run had a median of 92 rows against
  2048 dims and every score was garbage. Watch for `rank = min(budget, n_fit_rows)`
  style caps silently starving rank.
- Never calibrate on a degraded baseline. A prior campaign captured X from a 0.7966
  gibberish baseline and every score ranked the wrong trajectory.
- Never evaluate compression on synthetic/Gaussian activations — every sub-bit
  negative from that era was an artifact of the proxy.
- Do not create a giant JSON index (a 1.38 GB capture-result.json was a real wall).

---

# LANE: q80-coherence-probe
## Class: GPU + MEMORY_HEAVY for capture. Use ./tools/gpu_lane_lock.sh.
## You are the CHEAPEST REFUTATION of the whole plan. Run first, report fast.

## Why you exist
Packing a 40 GiB artifact and *then* discovering it is incoherent costs a day.
Your job is to find that out in the next hour, on a few layers, before the pack
lane commits. **A refutation from you is a maximally successful lane.**

## What to do

1. **Teacher-forced layer-drift probe.** Take the BF16 source model. For a real
   prompt, run the true forward pass and record the hidden state entering each
   layer. Then apply the three mixed codecs to the routed-expert weights of a
   contiguous span of layers and re-run, feeding the TRUE hidden in at the span
   start. Measure, per layer:
       cosine(h_mixed, h_true)
       relative L2 error
       the layer-over-layer growth ratio of that error
   The decisive question is the **growth ratio**, not the absolute cosine. If
   error is multiplied by >1.0 per layer, depth 48 destroys the model and 0.86
   per-organ cosine is irrelevant. Report the ratio explicitly.

2. **Establish the null.** Compute the same drift for a random/shuffled
   perturbation of matched magnitude. If the mixed codec's drift is not clearly
   separated from the null, the metric is not measuring what we think — this
   codebase has already been burned by a 0.898 null cosine.

3. **Logit-level check.** At the end of the span, report top-1 agreement, top-5
   overlap, and KL divergence against the BF16 logits. Token behaviour is what
   matters; hidden cosine is a proxy for it.

4. **Full-depth extrapolation.** From the measured per-layer growth, state what
   the drift will be at 48 layers, and therefore whether coherent generation is
   plausible. Show the arithmetic.

5. **Capture sufficiency audit.** The fits behind the receipt used a
   25258-token capture. Report rows-per-fit against the fitted dimension for
   each of gate/up/down. Any organ where rows < dimension is UNDERDETERMINED and
   its cosine number in the receipt cannot be trusted — say so plainly.

## The deliverable
A verdict, with numbers:
    GO          — drift is sub-multiplicative or bounded; packing is justified
    GO_WITH_FIX — needs a specific change (protect more layers, raise non-expert
                  bits, raise rank on a named organ, protect the sensitive 3%
                  more aggressively). Name the change and the BPW it costs, and
                  check it still fits under 1.5 using the identity.
    NO_GO       — with the growth ratio that proves it, and the cheapest
                  alternative representation you can defend.

There is real headroom to spend if you need it: dropping non-expert from 8-bit to
6-bit buys 1.43051 -> 1.37115, and to 4-bit buys 1.31179. Spending that headroom
on coherence is legitimate; exceeding 1.5 is not.

## Do not
- Do not pack a full artifact. The pack lane owns that.
- Do not write Metal kernels. The kernel lane owns that.
