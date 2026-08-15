# Q30 activation-aware family probe — verdict

**Status:** `EARNED_ACTIVATION_AWARE_FAMILY_PROBE_COMPLETE_LOCAL_ONLY_OR_NEGATIVE`  
**Ceiling:** component BPW ≤ 1.5  
**Objective:** output cosine on **real** held-out routed activations, always reported against the **constant-mean null**  
**Primary cohort:** high-hit L0 experts (≥200 routed tokens in the sealed three-prompt HCLI capture)

## One-line answer

**No family tested here is a credible path to a coherent full-model Q30 artifact at ≤1.5 BPW.**  
Activation-aware fitting **does invert family ranking** vs raw-weight low-rank on real activations, but the surplus over the constant-mean null is thin, weight recovery stays poor under the ceiling, and this three-prompt capture is dominated by a **null ≈ 0.94 trap**.

## What was measured

| Asset | Provenance |
|---|---|
| Activations | Existing sealed current-HCLI L0 route+hidden capture (embedding + L0 attn/postnorm/router only; no server, no lease) |
| Hidden source | Device L0 post-attention RMSNorm = router input = expert input for gate/up |
| down_proj inputs | SwiGLU intermediate from true BF16 gate/up on those hiddens |
| Weights | Positioned BF16 reads of `model.layers.0.mlp.experts.{e}.{gate,up,down}_proj.weight` |
| Experts | 104, 1, 45 (high-hit) + 127 (prior-measurement continuity, 48 hits) |
| Families | (1) raw-weight randomized low-rank+q, (2) activation-PCA low-rank+q, (3) activation-weighted binary+top-column residual, (4) activation-weighted SVD (output Frobenius) low-rank+q |

No gate was weakened. No full-model pack. Component BPW is per-tensor billed rate, not complete-model BPW.

## Family ranking on high-hit experts (under ceiling)

| family | mean out-cos | mean wt-cos | mean null | mean surplus | frac beats null | best surplus | operator coh? |
|---|---:|---:|---:|---:|---:|---:|---|
| `activation_weighted_svd_low_rank_q` | 0.981 | 0.464 | **0.942** | **+0.039** | **1.00** | +0.094 | no |
| `activation_pca_low_rank_q` | 0.962 | 0.385 | **0.942** | +0.019 | 0.72 | +0.104 | no |
| `activation_weighted_binary_residual` | 0.875 | 0.552 | **0.942** | −0.067 | 0.11 | +0.020 | no |
| `raw_weight_low_rank_q` (incumbent) | 0.787 | **0.681** | **0.942** | **−0.155** | **0.00** | −0.002 | no |

### Ranking inversion (hypothesis confirmed in the narrow sense)

- Every activation-aware family beats raw-weight low-rank on **surplus over null**.
- Raw-weight low-rank has **higher weight cosine** but **never beats the null** on this capture.
- This matches the GLM-5.2 lesson: weight-space cosine is the wrong objective; real activations reverse the ranking.

### Null trap (hypothesis also burned here)

- Mean constant-mean null on high-hit under-ceiling rows: **0.942**.
- Absolute output cosine without null subtraction is **inadmissible** (same class of failure as the prior 0.898 constant-mean null).
- Best high-hit surplus under 1.5 BPW: **+0.104** (`activation_pca` / r192_b4 / gate / expert 1) with weight cosine only **0.381** → labelled **distribution-local only**.

## Matched budget slice (r192_b4 ≈ 1.46 BPW, experts 1/104/45)

Pattern at the best under-ceiling low-rank budget:

| family | typical wt-cos | typical out-cos | typical surplus | beats null? |
|---|---:|---:|---:|---|
| raw-weight low-rank | 0.67–0.82 | 0.84–0.89 | −0.12 … −0.00 | almost never |
| activation-PCA low-rank | 0.36–0.63 | 0.93–0.995 | −0.03 … +0.10 | usually |
| activation-weighted SVD | 0.59–0.63 | 0.99 | +0.03 … +0.09 | always |
| activation-weighted binary+residual | 0.03–0.79 | 0.85–0.92 | −0.11 … +0.02 | rare |

## Coherence-grade and BPW reachability

**Definition used (surplus-first, not raw cosine):**

- output_cos ≥ 0.90 **and** surplus ≥ 0.10 **and** beats null  
- “operator recovery” additionally requires weight_cos ≥ 0.50  
- high-hit experts (≥200 tokens) are primary; expert 127 (48 tokens) is footnote only

| question | answer |
|---|---|
| Any family coherent under ≤1.5 BPW on high-hit experts with operator recovery? | **No** |
| Any local surplus-first pass under ≤1.5 on high-hit? | **Yes, thin** — a few activation-PCA rows, all distribution-local (wt-cos ≪ 0.5) |
| Exact BPW where joint surplus+operator becomes reachable on high-hit experts? | **Above the entire tested grid** (anchors through rank-640 / ~4.9 BPW). At wt-cos 0.985 / 4.87 BPW, surplus is still only **+0.022** because null ≈ 0.94 |
| Low-hit footnote (expert 127, n_hold=12) | Joint surplus+operator appears near **1.46 BPW** for binary-residual / activation-weighted SVD — **not primary**; small-N activations inflate surplus |

## Verdict on the campaign question

1. **Family ranking:** activation-aware methods win on real activations; the live search’s raw-weight low-rank family loses on the correct objective (surplus over null).
2. **≤1.5 BPW coherence:** **not earned** as a promotion path. Surpluses are small; operator recovery under the ceiling fails; absolute output cosine is null-dominated on this capture.
3. **Cannot quote a clean “BPW where it becomes reachable” for high-hit experts** from this capture: even ~5 BPW raw-weight reconstruction barely beats the mean output. That is a property of **near-constant expert outputs on three prompts**, not a proof that 5 BPW is enough or that 10 BPW is required.
4. **Next measurement that would change the answer:** a broader real-activation set (many prompts / layers / tokens) where the constant-mean null falls well below ~0.9. Until then, any under-ceiling “high output cosine” claim is a null trap.

## Claim boundary

- No server started, no lease, no full-model pack, no gate weakened.
- CPU/numpy only; positioned source-shard reads.
- Component BPW ≠ complete physical BPW.
- Not a runtime admission or capability claim.
- Negative / mixed results are the deliverable.

## Artifacts

- Probe: `lab/operators/q30_activation_aware_family_probe.py`
- Full table JSON: `Q30_ACTIVATION_AWARE_FAMILY_PROBE.json`
- Full table MD: `Q30_ACTIVATION_AWARE_FAMILY_PROBE.md`
- Capture used:  
  `.../quality-candidates/gate-up-residual-v1/current-hcli-route-capture/runs/74c918d5…_8bd3bfb3…`
