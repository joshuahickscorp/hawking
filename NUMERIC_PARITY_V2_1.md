# Numeric Parity Contract V2.1 — condition-aware

**Status: CANONICAL, user-authorized 2026-07-26.** Extends V2 from transcendentals to
**reductions**, and replaces scalar relative error with condition-aware metrics.

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

## Condition-aware hybrid metrics

Applied together; a result passes only if all applicable ones pass.

| metric | where it applies |
|---|---|
| **absolute error** | near zero, where relative error is undefined in practice |
| **relative L2 over the complete vector** | overall agreement, immune to a single tiny element |
| **ULP distribution** | reported as a distribution — median and tail — not a max |
| **KL divergence / cosine** | where the vector is a distribution or a direction |
| **exact top-k and greedy decisions** | **no tolerance, ever** |

The last row is unchanged from V2 and is the real contract. Continuous arithmetic may drift
within bounds; a decision may not.

## Reclassification

The `head-to-device` result is reclassified from `REJECTED` to
**`NUMERIC_GATE_INSUFFICIENTLY_CONDITIONED`**. It is not a permanent rejection and the lane's
kernel is not condemned. The diagnostic suite is rerun under V2.1, and rejection follows only
if **meaningful-scale logits**, **full-vector parity**, or **discrete decisions** fail.

## Timing hygiene

The synchronization ledger must be rerun against the **current 2,518 ms post-memoization
path**. The earlier 6,446 ms figure predates verification memoization and must not be
attributed to the current runtime. Any per-synchronization cost derived from it is void — the
controller's "~5.5 ms per synchronization" figure used that stale wall time and is withdrawn
pending re-measurement.

## Isolation, unchanged

The default resident path stays untouched. All head and expert-wave work remains isolated and
additive until regression, numerical, and complete-token gates pass.
