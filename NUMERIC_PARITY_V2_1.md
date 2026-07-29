# Numeric Parity Contract V2.1 — condition-aware

**Status: CANONICAL, user-authorized 2026-07-26.** Extends V2 from transcendentals to
**reductions**, and replaces scalar relative error with condition-aware metrics.

**Oracle correction (2026-07-28):** the authority is an independent **FP64 reference
computation**. Lifting an f32 backend's own logits to f64 and scoring against that lift
is circular and banned. Score **both** host and device f32 arms against the same f64
reference. The printed field formerly mislabelled `mean_rel` is
`max_meaningful_rel` — a max over single elements, not a mean.

## Why V2.1 exists

V2 was applied to `lm_head` using a single scalar metric: max relative error across the logit
vector, bound 1e-5. It reported 3.23e-3 and the lane was rejected.

**That rejection was wrong.** The test vector spanned ~1e-32 to ~5e18. A *relative* error on a
denormal-scale element is not a measurement — any absolute perturbation produces an enormous
relative figure. The well-scaled elements agreed to ~1.4e-7, which is exactly what
reduction-order divergence should look like.

The gate was insufficiently conditioned, and it was chosen by the controller immediately after
two lanes had failed — the state of mind in which a too-strict metric feels like rigour.
A metric must be justified by the numerics of the operation, not by the mood of the reviewer.

## The FP64 authority

Parity is judged against an **FP64 reference computation**, not against the f32 host path.
The f32 host is one implementation with its own accumulation order; making it the oracle
bakes its rounding into the contract. FP64 gives a neutral ground truth both backends can be
scored against.

**Banned authorities**

- `ref64 = host_f32.iter().map(|&v| v as f64).collect()` then `score_pair(host, device, ref64, …)`
- peer GPU mode, mode-vs-peer, or any other f32 backend as the continuous oracle
- any construction where the host arm is tautologically zero-error by construction

**Required authority**

- Op-local: f64 evaluation of that op (e.g. `matvec_bf16_f64_authority`,
  `silu_mul_f64_authority`, dense matvec in f64).
- Full fixture forward: independent f64 forward of the same weights/tokens
  (`gravity_glm_device_only_mlp` harness for `glm52-tiny-R0`).

If a domain has no valid original-input f64 authority, mark it
`AUTHORITY_UNAVAILABLE` and refuse qualification — do not substitute a host lift.

## Condition-aware hybrid metrics

Applied together; a result passes only if all **hard** gates pass. Diagnostics are always
printed and never silently dropped.

| metric | role |
|---|---|
| **absolute error** | near zero, where relative error is undefined in practice — **hard** |
| **relative L2 over the complete vector** | headline continuous agreement — **hard** |
| **cosine similarity** | direction / full-vector shape — **hard** |
| **KL on softmax** | when the vector is a pre-softmax distribution (logits) — **hard** if `require_kl` |
| **exact top-k and greedy decisions** | **hard, no tolerance, ever** |
| **max meaningful-scale relative** (`max_meaningful_rel`) | single-element max on \|ref\| ≥ cutoff — **hard for op-local**, **diagnostic for full multi-layer forward** |
| **ULP distribution** (median, p95, p99, max) | reported as a distribution — **diagnostic** |
| **diagnostic max scalar rel (all elements)** | V2 metric, for re-runs — **diagnostic** |

The discrete row is unchanged from V2 and is the real decision contract. Continuous
arithmetic may drift within bounds; a decision may not.

### Why full-forward demotes `max_meaningful_rel`

Measured on `glm52-tiny-R0` (prompt tokens from `ref_glm.json`), host f32 vs independent
f64 forward:

| field | host vs f64 |
|---|---|
| `rel_l2` | ≈ 9.1e-7 (under 1e-5) |
| `cos` | 1.000000000 |
| `kl` | ≈ 2.5e-13 |
| argmax | 986 / 986 |
| `topk_ok` | true |
| `max_meaningful_rel` | ≈ 1.7e-2 (above 1e-5) |
| ULP med / max | ≈ 11 / 2.5e5 |

A 1e-5 hard bound on the **single-element max** therefore rejects the host baseline itself
while every full-vector and discrete gate is clean. That is the same class of pathology V2.1
was written to cure: one element carrying a large relative figure that does not move the
distribution or the decision.

- **Op-local** kernels (one matvec, one silu, one decoder step): keep
  `max_meaningful_rel ≤ 1e-5` as a hard gate (`Bounds::logits()` /
  `Bounds::continuous_only()`).
- **Full multi-layer fixture forward**: use `Bounds::full_forward_logits()` —
  hard gates are rel_l2, cosine, KL, abs-near-zero, greedy, top-k;
  `max_meaningful_rel` and ULP tails are **reported**, not gated.

This is correct measurement, not a relaxation for a candidate. A wrong expert, wrong
routing, dropped layer, or argmax flip still fails discrete and/or rel_l2 under
full-forward bounds (see deliberate-break tests in `gravity_glm_device_only_mlp`).

### Label contract

`format_score_line` prints `max_meaningful_rel=…`. The historical label `mean_rel` was a
lie (the value was always a max). Failure strings use `max_meaningful_rel`. Sealed receipts
that embed the old `mean_rel=` string are left alone; new logs must not reintroduce it.

## What the gate admits and rejects

**Admits (full-forward)** when, against an independent f64 reference:

- `rel_l2 ≤ 1e-5`
- `cosine ≥ 1 − 1e-7`
- `kl ≤ 1e-6` (logits)
- abs-near-zero within bound on the near-zero subset
- greedy argmax and top-k exact
- (max_meaningful_rel and ULP may be large; they are printed)

**Rejects** when any hard gate fails — including:

- wrong expert / wrong routing / dropped layer (discrete and usually rel_l2)
- argmax or top-k flip (no tolerance)
- full-vector drift (`rel_l2` / cos / KL)
- op-local max_meaningful_rel breach under op-local bounds

**Does not promote.** A V2.1 pass is a measurement. Promotion, default flips, and TG claims
require separate authorization.

## Reclassification

The `head-to-device` result is reclassified from `REJECTED` to
**`NUMERIC_GATE_INSUFFICIENTLY_CONDITIONED`**. It is not a permanent rejection and the lane's
kernel is not condemned. The diagnostic suite is rerun under V2.1, and rejection follows only
if **hard** continuous metrics, full-vector parity, or **discrete decisions** fail.

### Prior rejection re-score (oracle correction; no promotion)

| artifact | old figure | class under corrected oracle |
|---|---|---|
| device-only MLP | `mean_rel = 6.194e-3` (actually max_meaningful_rel) vs host lift | **Circular oracle + mislabel.** Host vs true f64 shows max_meaningful_rel ≈ 1.7e-2 with clean rel_l2/cos/kl/argmax. Continuous rejection on that single field alone is **not sound** under full-forward policy. Device-vs-f64 live numbers require Metal (unmeasured in sandbox). **Do not promote.** |
| residual-router C2 | `meaningful_rel = 4.853e-3` | Same single-element class for the parity field. **Physics kill survives:** waits 102 > baseline 39. Parity-only meaningful_rel rejection is the same class as above if authority was host-relative. |
| discrete / rel_l2 / wait / CB kills | various | **Sound** when the failure is discrete mismatch, rel_l2 breach against true f64, or independent physics. |

## Timing hygiene

The synchronization ledger must be rerun against the **current 2,518 ms post-memoization
path**. The earlier 6,446 ms figure predates verification memoization and must not be
attributed to the current runtime. Any per-synchronization cost derived from it is void — the
controller's "~5.5 ms per synchronization" figure used that stale wall time and is withdrawn
pending re-measurement.

## Isolation, unchanged

The default resident path stays untouched. All head and expert-wave work remains isolated and
additive until regression, numerical, and complete-token gates pass.
