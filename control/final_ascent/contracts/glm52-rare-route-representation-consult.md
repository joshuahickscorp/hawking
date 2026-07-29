# Read-only challenge: GLM-5.2 zero/rare-route representation pilot

## Purpose

Design the smallest decisive real-weight pilot that can determine whether the
sealed GLM-5.2 activation-aware v2 program has a byte-feasible representation
for zero- and rare-route experts. This is a source-body-free design review.
Do not implement, fetch, rehydrate, delete, or modify anything.

The current per-expert program cannot be promoted:

- the route census observed 19,200 experts in layers 3–77 and left layer 78's
  256 experts UNOBSERVED;
- 329 observed experts have zero routes and 16,607 have 1–204 routes;
- only 133 experts are in the sealed promotion route-count range;
- assigning rank 128 to every non-promotion expert is 1.299189 BPW, above the
  exact 49/50 target;
- the exact maximum rank-128 population under target is 6,583/19,456.

Challenge this diagnosis and propose a representation experiment, not a
capability claim.

## Read-only evidence

Inspect:

- `GLM52_ROUTE_POPULATION_CENSUS.json`
- `GLM52_ROUTE_POPULATION_CENSUS.md`
- `GLM52_V2_PROGRAM_FEASIBILITY.json`
- `GLM52_BASIS_PILOT_RECEIPT.json`
- `GLM52_BASIS_PILOT_CONTROLLER_RESEAL.json`
- `GLM52_REHYDRATION_RECEIPT.json`
- `GLM52_RESOURCE_RESERVE_POLICY.json`
- `tools/condense/glm52_activation_aware_pack_v2.py`
- `tools/condense/glm52_basis_pilot.py`
- `/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity/source_fetch/GLM52_OFFICIAL_TENSOR_INDEX.json`

You may inspect file metadata and the retained capsule top-k/pre-router evidence
already authorized by the census and basis-pilot receipts. Do not decompress a
whole capsule. Do not read any safetensors body or incomplete download body.
Do not use the network.

## Required scientific challenge

Evaluate at least these candidate ABI families:

1. one uncentered hidden-input basis per layer, with an expert-local real-SwiGLU
   down-input basis;
2. route-cohort-shared hidden bases, with expert-local real-SwiGLU down bases;
3. route-cohort-shared hidden and down bases;
4. a clearly billed native or parameter-cluster fallback for zero-route experts;
5. at least one independently proposed alternative.

For each family:

- derive exact whole-program bytes/BPW from the sealed v2 ledger rather than
  describing it qualitatively;
- distinguish coefficient, basis, native, header, and other physical bytes;
- state the maximum number of shared bases/groups that still fits 49/50 BPW;
- identify what is measurable from the retained calibration capture;
- identify what is intrinsically unobservable for zero-route and layer-78
  experts;
- reject any Gaussian proxy, centered-only fit, output-side down promotion, or
  all-row score mislabeled as route-conditioned evidence.

The protocol must preserve uncentered mean retention and must derive every
promotional down-projection input from real `Z =
silu(X @ W_gate.T) * (X @ W_up.T)`.

For one-route experts, the single routed row may be a holdout only if the fitted
basis is derived from peers and excludes that expert/row. A one-row self-fit is
not evidence. Zero-route experts have no route-conditioned quality score in this
capture; counterfactual contextual scores must be labeled diagnostic and cannot
promote a population.

## Required source-locality analysis

Independently derive a bounded panel from the official tensor index and route
census. Prefer exactly three source shards, one early, one middle, and one late,
where each selected shard contains complete gate/up/down triplets for:

- at least one promotion-range expert;
- at least one between-anchor expert;
- at least two below-anchor experts, including a count near 1 and a count near
  204 where available;
- at least one zero-route expert.

Reject experts whose triplet straddles source shards. Report exact tensor names,
route counts, source shard names, sealed source sizes/hashes, and why the panel
is representative. It is acceptable to replace the census's original
24-expert suggestion with a shard-local panel if the evidence-band and depth
coverage is stronger and the source set is smaller.

Plan serialized rehydration: at most one source shard body resident for the
measurement at a time, with a fresh 75,000,000,000-byte free-disk hard gate
before each acquisition and a hash-bound release receipt after each shard.
Do not perform the acquisition or release in this task. Treat the existing
incomplete cache bodies as occupied disk, not verified source.

## Pilot decision semantics

Preregister:

- deterministic fit/holdout identities;
- no target-expert basis leakage for one-route experts;
- exact rank and byte comparisons;
- per-organ and per-evidence-band floors;
- promotion versus diagnostic panels;
- failure conditions and the smallest next action for each outcome.

Do not lower the sealed absolute floors to make a candidate pass. Explain
whether the pilot can only falsify candidates, can select a next candidate, or
could legitimately authorize a wider bounded pilot. It cannot authorize a full
282-shard traversal from three shards.

## Safety fences

Bind all false:

- `RAMANUJAN_RESEARCH_AUTHORIZED=false`
- `HIDE_KERNEL_TURN=false`
- `ODYSSEY_LAUNCH_AUTHORIZED=false`
- `full_parent_traversal_started=false`
- `full_traversal_authorized=false`
- `capable_artifact_claimed=false`
- `MOP_touched=false`

Do not inspect or modify MOP, Ramanujan, runtime, HIDE, Odyssey, launchd, prior
negative controls, repository files, capsules, source bodies, or remote state.

## Required report

Return:

1. independent diagnosis and strongest competing explanation;
2. exact byte table for every candidate family;
3. exact three-shard panel and resource envelope;
4. preregistered measurement protocol and floors;
5. claims the capture cannot support;
6. recommended smallest implementation task;
7. all fences false.

End with:

`COMPLETION REPORT`

followed by `status`, `files_changed`, `commands_run`, `measured_evidence`,
`remaining_uncertainty`, and `next_safe_action`.
