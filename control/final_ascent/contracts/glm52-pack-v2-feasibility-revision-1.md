# Revision 1: route-population uncertainty and truthful fake down basis

The first implementation under `glm52-pack-v2-feasibility.md` is not eligible for
integration yet.  Preserve its good work, but correct the following two material
issues without starting any source traversal.

## 1. The rank-64 whole-population total is not conservative

The generated receipt assigns all 19,456 routed experts to the rank-64
high-traffic class and reports:

- `total_bytes=76,751,084,032`
- `complete_bpw≈0.815`
- `within_target_bpw=true`

The same receipt admits that no full-model traffic map exists and that the sealed
low-traffic diagnostic needed rank 128 to clear its `0.91` per-tensor floor.
Therefore the current 0.815 number is an optimistic/lower-bound scenario, not an
authorization-deciding conservative total.

Revise the model and receipt to expose at least:

1. `all_routed_rank64_lower_bound_ledger`
   - target-local basis identities;
   - all routed experts at rank 64;
   - explicitly `authorizing=false`;
   - explicitly not called conservative.

2. `all_routed_rank128_uncertainty_bound_ledger`
   - target-local basis identities;
   - all routed experts at rank 128 while retaining the preregistered ranks for
     shared MLP, router, and q-a;
   - this is the only ledger that may decide the top-level
     `within_target_bpw`;
   - explain that it is a byte-feasibility uncertainty bound, not proof that rank
     128 is quality-sufficient for every routed expert.

3. `route_population_sensitivity`
   - compute exact totals for a deterministic sweep of routed experts at rank 128
     versus rank 64 (at minimum 0%, 25%, 50%, 75%, 100%);
   - report the exact maximum number and fraction of rank-128 routed experts that
     can fit under `49/50` BPW;
   - count one expert as its gate/up/down triplet with the shared hidden basis and
     separate real-SwiGLU-input basis;
   - make selection deterministic (for example sorted `(layer, expert)` identities)
     and label it arithmetic sensitivity, not a traffic classification.

The top-level `within_target_bpw` must equal the all-rank-128 uncertainty-bound
result.  It is expected to be false; compute it rather than hard-coding it.  Add
explicit fields:

- `full_route_population_classified=false`
- `route_population_evidence_sufficient_for_rank_assignment=false`
- `rank64_population_fit_is_lower_bound_only=true`
- `full_traversal_authorized=false`

Do not change any quality floor or target BPW to make the result fit.

Classify routed tensors neutrally in the static census (for example
`routed_gate/up/down`), not as `high_traffic_*`.  Traffic is not present in sealed
source headers.

Keep the existing transfer-sharing scenario non-authorizing.  It must not affect
any top-level decision.

## 2. The fake down basis is mislabeled

`fake_gate_up_down_roundtrip()` currently builds `B_z` from unrelated random
rows, then labels it `real_swiglu_input`.  That does not demonstrate the required
program.

Revise the fake proof so that:

1. deterministic fake pre-router rows `X` and synthetic `topk_indices` are created;
2. `select_route_rows(X, topk, expert_id)` produces the expert-specific rows used
   to build each hidden basis;
3. actual matching fake `W_gate` and `W_up` form
   `Z = swiglu_intermediate(X_route, W_gate, W_up)`;
4. `B_z = build_uncentered_basis(Z, rank)` is the down input basis;
5. the serialized down metadata still says `real_swiglu_input`;
6. the proof returns deterministic hashes or numeric witnesses binding `X_route`,
   `Z`, and `B_z`, plus route counts;
7. a negative test proves an unrelated random basis cannot be substituted while
   preserving the witness.

Random numbers are acceptable for deterministic fake fixtures.  They are not
acceptable as a direct replacement for the SwiGLU intermediate while claiming
that down used `Z`.

## Receipt and report corrections

- Regenerate JSON and Markdown deterministically.
- The Markdown headline must not call the rank-64 result conservative.
- The next safe action must say that route-population measurement is required
  before any full traversal and that a passing rank-mixture budget alone would
  still not prove representation capability.
- Update remaining uncertainties and non-claims accordingly.
- Preserve every safety fence as false.

## Tests

Add tests that fail against the first implementation and prove:

- neutral static routed classification;
- top-level decision equals the all-rank-128 uncertainty bound;
- rank-64 lower bound is non-authorizing;
- exact 0/25/50/75/100% monotonic sensitivity and threshold count;
- no rank reduction occurs to force the uncertainty bound under budget;
- fake hidden bases use selected route rows;
- fake down basis is derived from actual `swiglu_intermediate`;
- a substituted unrelated basis fails the witness;
- deterministic hashes/receipt;
- all fences remain false.

Run the focused v2 tests, v1 activation-aware tests/selftest, basis-pilot tests,
and `py_compile`.  Do not rerun the unrelated legacy `glm52_pack` tests that were
already shown to spend more than 20 minutes in a pre-existing atomic-pack path;
record that prior bounded timeout as an inherited verification gap.
