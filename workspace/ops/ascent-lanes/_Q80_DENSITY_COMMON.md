
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
