# Contract: GLM-5.2 route-population census from retained top-k metadata

## Purpose

Resolve the rank-64 versus rank-128 population uncertainty exposed by
`GLM52_V2_PROGRAM_FEASIBILITY.json` as far as the retained teacher route metadata
allows, without fetching or reading any parent weight body and without claiming
representation capability.

This is a route-metadata census, not a compression run.

## Allowed inputs

- the 33 retained capsule `.npz` files under
  `/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity/source_fetch/teacher/capsules`;
- only `layer_NN/topk_indices.npy` members from those archives;
- sealed pilot/census receipts and source header JSON in the repository;
- existing small code/tests.

Do not load hidden-state, logits, expert-output, or other large capsule members.
Use `zipfile` plus the exact top-k member stream (or another demonstrably
member-selective path) so the tool never decompresses whole 0.8–10 GB archives.
Do not hash whole capsule files in this task; bind their already sealed hashes
where available and hash each loaded top-k array/member.

## Safety

Bind every fence false:

- `RAMANUJAN_RESEARCH_AUTHORIZED=false`
- `HIDE_KERNEL_TURN=false`
- `ODYSSEY_LAUNCH_AUTHORIZED=false`
- `full_parent_traversal_started=false`
- `full_traversal_authorized=false`
- `capable_artifact_claimed=false`
- `MOP_touched=false`

Do not modify capsules, source artifacts, prior refused artifacts, runtime,
HIDE, Odyssey, or `/Users/scammermike/Downloads/ramanujan`.

## Required coverage proof

1. Enumerate all `layer_NN/topk_indices.npy` members across the 33 capsule files.
2. Normalize the known capture shape `[16,256,8]` to `[4096,8]`; also accept an
   already flattened `[4096,8]` form.  Reject every other shape.
3. Validate integer dtype, expert IDs in `[0,255]`, 4,096 rows, top-k width 8,
   and no duplicate expert ID within a row.
4. Prove the exact layer coverage and duplication:
   - dense layers 0–2 may be reported but are outside the routed-expert census;
   - routed MoE layers are 3–78 inclusive;
   - expected retained coverage is 3–77, with layer 78 missing;
   - overlapping single-layer and range capsules must have byte-identical
     normalized top-k arrays.  A conflict fails closed.
5. Select one canonical member per layer deterministically after duplicate
   agreement, and bind its capsule filename, member name, normalized array hash,
   dtype, original shape, and normalized shape.

## Route counts

For every covered `(layer, expert_id)` pair, compute:

- `route_count`: number of the 4,096 rows whose top-8 contains the expert;
- `route_fraction=route_count/4096`;
- zero-route flag.

Because per-row uniqueness is required, the total route counts per layer must
equal `4096*8=32768`.  Reconcile all covered layers exactly.

Publish:

- all 19,200 covered expert records (75 layers × 256);
- global and per-layer histograms/quantiles;
- counts at exact thresholds and in preregistered evidence bands;
- layer 78’s 256 experts explicitly as `UNOBSERVED`, never imputed.

## Evidence bands (do not call them quality labels)

The sealed five-shard basis pilot supplies only these route-count anchors:

- promotion-grade rank-64 routed examples: minimum observed route count `2577`;
- low-traffic diagnostic rank-128 example: route count `205`;
- below 205: no bounded tensor-quality exemplar;
- missing layer 78: no route evidence.

Classify covered pairs only as arithmetic/evidence bands:

- `PROMOTION_PANEL_ROUTE_RANGE`: `route_count >= 2577`;
- `BETWEEN_PILOT_ANCHORS`: `205 <= route_count < 2577`;
- `BELOW_LOW_TRAFFIC_ANCHOR`: `1 <= route_count < 205`;
- `ZERO_ROUTE`: `route_count == 0`;
- uncovered layer 78: `UNOBSERVED`.

These names must not imply that rank 64 or rank 128 is proven for every member.
Do not invent a smooth quality-versus-route-count law from five tensors.

## Byte implications

Reuse exact v2 ledger arithmetic, without altering its floors or target:

1. `anchor_assignment_scenario`:
   - `PROMOTION_PANEL_ROUTE_RANGE` experts at rank 64;
   - `BETWEEN_PILOT_ANCHORS` experts at rank 128;
   - `BELOW_LOW_TRAFFIC_ANCHOR`, `ZERO_ROUTE`, and `UNOBSERVED` unresolved.
   - Report the known-rank encoded bytes and the unresolved expert count; this
     scenario is never authorizing because it is incomplete.

2. `rank128_for_all_nonpromotion_bound`:
   - promotion-range covered experts at rank 64;
   - every other expert, including layer 78, at rank 128.
   - Compute exact total bytes/BPW through the v2 target-local ledger.
   - This remains a byte uncertainty bound, not quality proof.

3. `native_for_unresolved_bound`:
   - promotion-range covered experts at rank 64;
   - between-anchor covered experts at rank 128;
   - below-anchor, zero-route, and unobserved expert gate/up/down triplets billed
     at sealed native BF16 payload width.
   - Compute exact total bytes/BPW and expose the component reconciliation.

4. Compare all scenarios with the already sealed maximum of 6,583 rank-128
   expert triplets under `49/50` BPW.

Top-level fields must remain:

- `route_population_evidence_sufficient_for_rank_assignment=false`;
- `within_target_bpw_for_proven_complete_assignment=false`;
- `full_traversal_authorized=false`.

Even if a byte scenario fits, do not promote: route count is not tensor quality.

## Deliverables

- `tools/condense/glm52_route_population_census.py`
- `tools/condense/tests/test_glm52_route_population_census.py`
- `GLM52_ROUTE_POPULATION_CENSUS.json`
- `GLM52_ROUTE_POPULATION_CENSUS.md`

The receipt must include source/code/test hashes, deterministic canonical member
selection, exact coverage and duplicate checks, per-expert records, evidence-band
summaries, byte scenarios, uncertainties, next safe action, and all safety fences.

The Markdown should be concise; the JSON may retain the 19,200 expert rows.

## Tests

Use small fake `.npz` archives to prove:

- member-selective loading never asks for non-top-k members;
- accepted shapes normalize identically;
- invalid shape/dtype/range/per-row duplicates fail closed;
- duplicate layer copies must match and conflicts fail;
- deterministic canonical member selection;
- exact per-layer route conservation;
- evidence-band boundaries at 0, 1, 204, 205, 2576, and 2577;
- missing layer 78 remains unobserved;
- all three byte scenarios reconcile and never change ranks to fit;
- top-level authorization/evidence fields remain false;
- deterministic receipt generation.

Run focused tests, `py_compile`, v2 feasibility tests, and v2 selftest.  Report
exact pass counts and the measured census distribution.

## Next safe action semantics

The output may select representative experts for a later bounded real-weight
pilot, but it must not rehydrate them.  The next pilot must cover multiple
below-anchor and between-anchor experts across early/middle/late layers and must
test a representation designed for zero/rare routes.  No full traversal is
authorized by this census.
