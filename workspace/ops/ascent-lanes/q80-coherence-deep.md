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

# LANE: q80-coherence-deep
## Class: GPU_EXCLUSIVE + MEMORY_HEAVY. Use ./tools/gpu_lane_lock.sh.
## THIS LANE DECIDES WHETHER THE <=1.5 Q80 PATH IS ALIVE. Highest priority open question.

## What the first probe found

`receipts/ascent-2026-08-16/Q80_COHERENCE_LAYER_DRIFT_PROBE.json` (merged on main):

    mixed_geo_growth_per_layer      1.277152     <- MULTIPLICATIVE
    null_geo_growth_per_layer       1.006081
    true_residual_geo_growth        1.060697
    extrapolated_rel_l2_at_48       16211.29     = 0.342884 * 1.277152^44
    separated_from_null             FALSE        <- !!
    mixed_vs_null_span_end_ratio    2.0643
    logits: mixed_top1 8420 AGREES with true; KL 3.35e-4; top5 overlap 0.6

    reconstruction:
      mean_gate_bpw                 1.126923
      mean_up_bpw                   1.292079
      mean_down_bpw_fitted          1.152004     <- better than the 1.27 assumed
      mixed_expert_bpw_measured     1.190335
      complete_bpw_8bit_nonexpert   1.392467
      complete_bpw_4bit_nonexpert   1.273735
      hgravs_rank_clamped           395 of 2048 organs
      down_cold_left_bf16           9 organs

It self-labels `GO_WITH_FIX`. **Treat that as unproven, not as a green light.**

## Three defects that make the verdict inconclusive — fix all three

1. **The 16211x number is an extrapolation from a 4-layer span.** `geo^44` amplifies
   any error in the growth estimate enormously. A growth of 1.277 gives 16211; 1.05
   gives 8.6; 1.0 gives 0.34. The conclusion is entirely controlled by a quantity
   measured over 4 layers. **This is the load-bearing weakness.**
2. **`separated_from_null = FALSE`.** The probe could not distinguish mixed-codec
   drift from a random perturbation of matched magnitude (ratio only 2.06). If the
   measurement cannot separate signal from noise, neither a GO nor a NO_GO follows
   from it. This machine has been burned before: raw activation cosine has a
   measured null baseline of 0.898.
3. **395 of 2048 organs had hgravs rank CLAMPED**, and 9 fell back to BF16. Rank
   clamping is the known underdetermined-fit failure: a prior campaign had
   `rank = min(budget, n_fit_rows)` silently cap rank so the r192 that actually
   worked was unreachable BY CONSTRUCTION. A clamped fit's score is not the codec's
   score.

## What to do — in priority order

1. **Stop extrapolating. Measure the full depth.** Run the teacher-forced drift
   probe across ALL 48 layers, or as deep as memory allows, reporting per-layer
   rel-L2 and the growth ratio as a curve. Growth is very often not constant with
   depth; a ratio measured on layers 0-4 may not hold at 20-48. If full depth does
   not fit in memory, tile it and carry the true hidden in at each tile boundary,
   and say exactly what you did.
2. **Fix the rank clamp**, then re-fit the 395 clamped organs. Report rows-per-fit
   against dimension for every organ. Never let rank be capped by row count without
   flagging it. If capture rows are the binding constraint, say so — the
   `q80-capture-coverage` lane is extending capture and you should use its output
   if available.
3. **Establish null separation properly.** Use several matched-magnitude null
   perturbations to get a distribution, not one sample, and report the separation
   as a margin. If mixed still does not separate from null, the honest conclusion is
   that this metric cannot certify the representation, and you should say that
   plainly rather than reporting a verdict it cannot support.
4. **Then run the actual gate: autoregressive generation.** Apply the mixed codecs
   to all routed experts and generate. Coherent text is the ONLY thing that settles
   this. Everything above is instrumentation to predict the answer cheaply; if you
   can afford to just run it, run it.

## Spend the headroom if you need it
Measured complete BPW is 1.392467 at 8-bit non-expert and 1.273735 at 4-bit — so
there is real room under the 1.5 ceiling. Legitimate places to spend it: raise rank
on the clamped organs, protect more of the sensitive tail, keep more layers at
higher precision (especially early layers, where error compounds through the most
remaining depth). Report the BPW cost of whatever you spend, and verify against the
identity `complete_bpw = 0.97032*expert + 0.02968*nonexpert`.

## Report honestly
    GO          — full-depth drift bounded and null-separated, or generation coherent
    GO_WITH_FIX — a specific named change, its BPW cost, and evidence it works
    NO_GO       — with the full-depth curve that proves it, plus the cheapest
                  alternative representation you can defend
A well-evidenced NO_GO is a maximally successful lane. Do not manufacture optimism;
this decides whether a large amount of downstream work is worth doing.

## Commit note
Earlier lanes hit Seatbelt denials writing to `.git` and finished with their work
UNCOMMITTED, which nearly lost it. You are running `gate` (unsandboxed) so commit
normally — and verify with `git log` that your commit actually landed on your
branch before you finish.
