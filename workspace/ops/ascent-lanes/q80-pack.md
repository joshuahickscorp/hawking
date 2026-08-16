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

# LANE: q80-pack
## Class: CPU_HEAVY / MEMORY_HEAVY / IO_HEAVY. Lock for anything >8 GiB or GPU.

## Your gap
    artifact_packed: false

Produce the first complete Q80 artifact at <=1.5 complete physical BPW, packed
from the BF16 source using the exact per-component recipe in the receipt.

## Requirements

1. **Follow the identity, and prove it on real bytes.**
       complete_bpw = 0.97032*expert_bpw + 0.02968*nonexpert_bpw
   The BPW you report must be computed from the **actual bytes on disk**, not from
   the design. Complete physical means everything: codes, scales, codebooks, rank
   factors, residual indices, outlier payloads, per-group metadata, padding, and
   any side table. If a byte is required to execute the model, it counts.
   A design-BPW that disagrees with the on-disk BPW is a bug — report both.

2. **Start at 8-bit non-expert (1.43051).** If the coherence-probe lane returns
   GO_WITH_FIX, apply its named change instead and recompute. Do not silently
   pick a different point on the ladder.

3. **Keep the sensitive 3% untouched**, as the receipt records. If you change what
   is protected, that is a representation decision — report it loudly with its
   BPW cost.

4. **Manifest and seal.** Follow whatever manifest/seal shape the existing
   uniform-Q4 catalog uses so the runtime can admit it
   (`Qwen80UniformQ4StreamingCatalog` and its manifest are the reference for
   structure only — the codecs differ). Per-tensor content digests, so the
   artifact's identity is verifiable once at admission and never re-derived per
   token.

5. **Layout in execution order.** Physical layout should follow the order the
   token graph consumes tensors. This is nearly free at pack time and expensive
   to retrofit.

## Hard constraints
- Never mutate the protected source at `.../runs/qwen-80b/Qwen3-Coder-Next`.
  It is read-only input. Write only under a new candidate directory beside the
  existing `quality-candidates/uniform-q4-group64-v1`.
- Disk floor is 15 GiB hard; there is ~199 GiB free now, and a full pack is on the
  order of 15-20 GiB, but intermediates can dwarf the output. Stream, do not
  materialize the whole model. Check free space before and during.
- Do not create a giant JSON index. Use a compact/CSR/mmap side format.

## Report
On-disk complete BPW to 5 decimals, total bytes, per-organ byte breakdown, and the
path to the artifact. State explicitly that packing alone is NOT a <=1.5 claim —
generation on a native kernel is the gate.

## Do not
- Do not write Metal decode kernels; the kernel lane owns that. Coordinate only on
  the packed byte format, which you define and must document precisely for them.
