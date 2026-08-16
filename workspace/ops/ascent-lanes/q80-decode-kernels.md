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

# LANE: q80-decode-kernels
## Class: GPU_EXCLUSIVE for benchmarks/parity, COMPILE otherwise.

## Your gap
    decode_kernel_exists: false

Three packed formats need Metal kernels that execute them:
    gate_proj  binary_group
    up_proj    binary + rice_q1_rms sparse residual @2% outliers (8.24 bits/outlier)
    down_proj  hgravs01_r160_b3 activation-weighted low-rank

## The law these kernels must obey
NEVER reconstruct a dense tensor. The forbidden shape is:

    packed -> full reconstruction -> large dense temporary -> GPU matvec

The required shape is:

    packed bytes -> kernel reads them directly -> decode in registers/simdgroup
                 -> immediately consume the decoded value in the matvec

Reconstructing dense would defeat the entire point: it restores the RAM footprint,
the DRAM traffic, the temporaries and the dispatches that the representation exists
to remove. A kernel that reconstructs is a rejected result even if it is fast.

## Precedent to read first
    crates/hawking-core/examples/ascension_qwen30_hgravs01_packed_matvec_parity.rs
    crates/.../qwen_complete_binary/activation_weighted_svd.rs
Q30 already has a packed hgravs01 matvec with a parity harness. Follow that
pattern rather than inventing a new one; hgravs01_r160_b3 for down_proj is the
same family.

## Deliverable per codec
1. A Metal kernel that consumes the packed bytes directly.
2. A **numpy/CPU oracle** and a parity harness proving the kernel matches it.
   This codebase's rule, learned the hard way: grade execution against what the
   ARTIFACT encodes (the oracle), never against the BF16 parent. A kernel that
   faithfully executes a bad representation is a CORRECT kernel — that failure
   belongs to Gravity, not to you. Keeping these separate is what previously
   exonerated the runtime and correctly redirected effort to the foundry.
3. Per-kernel cost, on the doctrine's terms: decode ops/weight, decode ns/token,
   DRAM bytes/token, temporary bytes, dispatches, register pressure if observable,
   and whether it fuses with the surrounding op.

## Coordination
The pack lane defines the exact packed byte format. Get it from them and follow it
literally. If the format is not yet fixed, define the format you need, document it
precisely, and say that the pack lane must match it — but do not both invent
formats independently.

## Correctness
Bit-exact against your oracle where the arithmetic is deterministic. Where a legal
parallel reduction reorders floating point, define a numeric-equivalence gate with
a quantitative bound and report the measured drift. "Looks close" is rejected.
