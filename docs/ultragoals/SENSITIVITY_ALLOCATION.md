# Sensitivity allocation (N051)

S026 §26 (protected-island allocator) and §3 (marginal value, Pareto cost vector).
Feeds N041's whole-model information allocator so bits can move **below** the
uniform per-organ floors at the **same** 2.5969 complete-EBPW budget.

**This is a proxy. It ranks. It does not certify. It does not claim a new
whole-model floor.** Composition failures at 1.25 (binary) and 1.85 (ternary)
on the MLP remain closed for those families; starving a channel to those
densities is a ranking, not a reopen.

Canonical receipt: `receipts/headless/SENSITIVITY_ALLOCATION.json`

The one-shot generator has been retired from active HEAD. The sealed receipt
and this bounded interpretation are the retained signal; the generator is
recoverable from Git if the measurement must be independently reproduced.

---

## Proxy

Activation-aware second-order (diagonal Fisher / activation-weighted magnitude)
on real captured activations, streamed from the parent BF16 tensors on CPU.
No GPU, no cargo, no Metal, no second 27B decode.

For a linear map `Y = X Wᵀ`:

```
F_jj = E_t[x_{t j}²]          # diagonal Fisher / Hessian on input channel j
VoI_ij = ½ F_jj W_ij²         # expected MSE if weight ij is deleted
capability_VoI = VoI · q_mult(layer)
```

`q_mult` is the residual-injection depth weight cited from
`tools/headless/global_allocator.py` (gravity_error_chain), not re-derived.
L0 = 1, L63 ≈ 16.1. Embed uses L0; lm_head uses L63.

Embedding is a gather, not `Y = X Wᵀ`:

```
VoI_token = ½ p(token) ‖row_token‖²
```

`p(token)` is the capture frequency (reconstructed prompt ids, Laplace +1).
Rows **unseen in the capture are locked** at the N041 embed floor (S026 §109:
the calibration long tail is not evidence of disposable information).

The MSE-per-bit model the allocator consumes is the RTN grouped-quant proxy

```
remaining_mse = capability_VoI / 4^{bpw}
```

Each extra bit quarters remaining MSE under that model. It is **not** a
composition certificate. Cosine is not used: `0.01·W` has 10⁻⁴ the VoI of `W`.

### Activation source (cited, not re-run)

`capture_diverse2` (`post_attn_norm`, 11,269 real tokens, BF16 parent MLX
full-model forward). Not Gaussian. Not a llama-server teacher.

| tensor | site | status |
|---|---|---|
| MLP gate/up | `post_attn_norm` of that layer, all 11,269 tokens | MEASURED |
| mixer in-proj (qkv/z/q/k/v) | same capture (true site is `input_layernorm`) | PROXY_SITE |
| MLP down_proj | SwiGLU intermediate reconstructed on a 256-token fit subsample from streamed parent gate/up | PROXY_SUBSAMPLE |
| DeltaNet out_proj | `v·silu(z)` from streamed in_proj, same subsample; **not** the recurrent S mix | PROXY_SUBSAMPLE |
| GQA o_proj | `repeat(v)·sigmoid(q_gate)` from streamed q/v | PROXY_SUBSAMPLE |
| lm_head | last-layer `post_attn_norm` as a proxy for final hidden (last MLP + `model.norm` were not captured) | PROXY_SITE |
| A_log / dt_bias / conv | no captured recurrent state | WEIGHT_ONLY, locked at leftover f32 (S024 §32) |
| embed unseen rows | not in the 1,575 unique capture tokens | UNMEASURED_IN_CAPTURE, locked (S026 §109) |

Visual tower and MTP heads are skipped: they are not in the N041 26,895,998,464
language-parameter closure.

---

## Classes (S026 §26)

Per-organ rank cuts on depth-weighted VoI per parameter of unlocked channels:

| class | rank |
|---|---|
| disposable | ≤ p05 |
| cheap | p05–p25 |
| ordinary | p25–p75 |
| sensitive | p75–p95 |
| critical | > p95 |

Global GEMV-only cuts are also stored so the vocab tail cannot dominate
percentiles. Recurrent leftovers are forced critical and locked.

---

## Equal-bit reallocation (the N041 consumer)

Start every unlocked channel-class bucket at 1.25 bpw and spend the N041
uniform-per-organ GEMV bit budget greedily on the highest
`Δremaining_mse / Δbits`. Discrete ladder:
`1.25, 1.85, 2.25, 3.125, 3.25, 4.125, 4.25`.
Locked leftovers stay at 32 bpw (N040). Locked unseen embed stays at 3.125.

Total GEMV bits are held at the uniform mix. The complete EBPW stays the
cited 2.5969. **Only the distribution moves.**

`n041_consumer` in the receipt is the machine-readable payload: per-bucket
`n_params`, `voi`, `uniform_bpw`, `recommended_bpw`, plus the marginal
capability-gain-per-bit curve.

Under this proxy, greedy cuts remaining MSE by ~78% vs uniform at the same
bits (`assignment.relative_mse_drop`). That number is a proxy, not a
capability claim.

Channel-grain Spearman of activation-aware VoI vs weight-only `‖row‖²` is
~0.68 (the activation term is doing work). Tensor-grain Spearman is ~0.99
(same-site tensors share similar `E[x²]`, so totals track `‖W‖_F`).

---

## Findings (from the sealed receipt; re-run the generator to refresh)

Least-sensitive **organs** (depth-weighted VoI per parameter, lowest first):
embedding, MLP, GQA, DeltaNet, output (lm_head).

Recommended organ bpw at equal bits vs the N041 uniform floors:

| organ | N041 uniform | recommended (incl. locked) | direction |
|---|---|---|---|
| embedding | 3.125 | 3.113 | slightly below (seen rows starved; unseen locked) |
| mlp | 2.25 | 2.304 | slightly above |
| gqa | 3.127 | 2.928 | below uniform |
| deltanet | 3.261 | 2.914 | below uniform |
| output | 3.125 | 4.250 | above (lm_head takes the 4.25 cap) |

**Cheapest large GEMVs to compress further** (lowest VoI/param, n ≥ 1e6):
early MLP `down_proj` (L1, L3, L4, L2, L5, L0, L6, …) and L0 DeltaNet
`out_proj`. These classify disposable under the proxy.

**Most sensitive large GEMVs** (spend bits here first): late DeltaNet
`out_proj` (L62, L60, L61, L57, L58), L63 `lm_head`, late GQA `v_proj`.
Depth weighting (`q_mult`) is the dominant reason late mixer outputs outrank
early MLP.

N040 measured GQA 3.13 and DeltaNet 3.26 as **composition** floors. This map
wanting to take those organs below those floors is exactly why the proxy
must not certify: N041 still has to screen any below-uniform move on the
composition ladder.

f16 islands (16 bpw) are not on the greedy ladder. Under `VoI/4^{bpw}` the
step 4.25 → 16 almost never pays. Hypersensitive channels are **named**
(top-k per tensor, `most_sensitive_regions`) as island *candidates* for a
composition metric; the MSE proxy does not buy them.

---

## What this does not do

- Does not claim a new complete EBPW. N041 remains 2.5969.
- Does not reopen the closed 2.25 MLP composition floor.
- Does not run a native decode or a GPU benchmark (N042 owns that).
- Does not delete unseen embed rows (N045 tokenizer gravity / S026 §109).
- Does not score recurrent DeltaNet state (no captured S; leftovers stay f32).
