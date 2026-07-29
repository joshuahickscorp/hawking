# Implement the bounded GLM-5.2 rare-route shared-basis pilot

## Goal

Implement, fake-test, and source-body-free preflight the smallest decisive
real-weight pilot for a byte-feasible representation of zero- and rare-route
GLM-5.2 experts.

This implementation task must not fetch, rehydrate, read, release, or modify a
real safetensors body. The supervising controller will review and integrate the
code before it separately authorizes the real serialized lifecycle.

No production default changes. No full traversal.

## Sealed diagnosis

Bind:

- exact weight denominator: `753329940480`;
- target: `49/50` BPW;
- exact target byte ceiling: `92282917708`;
- target-local all-rank-128 v2 total: `122653547008` bytes, fails;
- layer-shared hidden plus expert-local rank-128 down total:
  `92166481408` bytes, passes with only `116436300` bytes headroom;
- layer-shared hidden and down rank-128 total: `82000818688` bytes;
- maximum total rank-128 shared hidden basis identities with expert-local down:
  `150` (151 fails);
- maximum total equal hidden/down shared basis pairs at rank 128: `4977`;
- `329` observed zero-route experts and layer 78's `256` unobserved experts
  cannot receive a route-conditioned score from the retained capture.

Recompute every number through the live v2 ledger. Do not copy an unexplained
constant into a receipt.

## Exact real panel

The later real run is restricted to these three official source shards and
these 15 target experts. Every target gate/up/down triplet is wholly resident
in its named shard.

### Early

Shard `model-00035-of-00282.safetensors`

- size `5359985464`
- sha256 `bde40fe925bdee14397ed505ceb2a9f69c45f2439cee7019599c1b0dfaf53158`
- L19 E134: route 3454, promotion band
- L19 E146: route 207, between-anchor
- L19 E116: route 1, below-anchor
- L18 E99: route 202, below-anchor
- L19 E105: route 0, zero-route

### Middle

Shard `model-00086-of-00282.safetensors`

- size `5366406840`
- sha256 `9fa52ba7e2831b6c1ac35f0b6d4940b24de0a9c897281f7cf8187ae8698ef014`
- L31 E92: route 3280, promotion band
- L31 E83: route 476, between-anchor
- L31 E64: route 1, below-anchor
- L31 E84: route 203, below-anchor
- L31 E54: route 0, zero-route

### Late

Shard `model-00264-of-00282.safetensors`

- size `5360347320`
- sha256 `d695bc81d74a3e317ed396c45f1f50ba4b295f8583cc2bf123bc64e7eb782203`
- L76 E172: route 3223, promotion band
- L76 E133: route 252, between-anchor
- L76 E181: route 1, below-anchor
- L76 E159: route 198, below-anchor
- L76 E151: route 0, zero-route

Aggregate official source size: `16086739624`.

Derive and verify this panel from:

- `GLM52_ROUTE_POPULATION_CENSUS.json`;
- `GLM52_REHYDRATION_RECEIPT.json`;
- `/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity/source_fetch/GLM52_OFFICIAL_TENSOR_INDEX.json`.

Do not trust the prose list alone.

## Deployment-honest shared fit

A shared candidate must build exactly one physical shared basis for each
declared identity. It is forbidden to build a different leave-one-target-out
basis for every target and bill it once.

For each panel layer:

1. identify every non-panel expert whose complete gate/up/down triplet is
   resident in that shard;
2. use only real retained `pre_router_hidden` and sealed `topk_indices`;
3. exclude all panel target experts from shared-basis fitting;
4. exclude the union of all panel routed holdout rows from every shared fit;
5. build one uncentered hidden basis for the layer;
6. for the shared-down arm, build one uncentered 2,048-wide down basis by
   concatenating real peer intermediates
   `Z = silu(X @ W_gate.T) * (X @ W_up.T)`;
7. a peer may contribute only its real routed fit rows; require at least 32;
8. cap peer contribution deterministically so one high-traffic peer cannot
   dominate, and publish the cap and exact row identities;
9. store and hash the single basis identity used by every target in the layer.

No panel target weight or target routed row may enter a shared fit. This is a
stronger transfer test and prevents target leakage for count-1 experts.

For a zero-route target, score the same shared basis only on a deterministic
counterfactual subset of real contextual layer activations held out from the
shared fit. Label every such score `COUNTERFACTUAL_DIAGNOSTIC`; it can never
promote.

Layer 78 remains `UNOBSERVED_NOT_TESTED`.

## Measurement arms

Use identical target holdout rows across every constructible arm.

1. `TARGET_LOCAL_H_Z_ORACLE_R128`
   - target-local uncentered hidden and real-SwiGLU down bases;
   - only constructible when a leakage-free target fit has at least 32 rows;
   - control only, not the population candidate.

2. `LAYER_H_LOCAL_Z_R128`
   - the one deployment-honest shared hidden basis plus target-local down;
   - byte-feasible but `NOT_CONSTRUCTIBLE` for count 0 or 1;
   - this fact must remain visible.

3. `LAYER_H_LAYER_PEER_Z_R128`
   - one deployment-honest hidden basis and one deployment-honest peer-SwiGLU
     basis for the layer;
   - the decisive rare-route treatment;
   - constructible for count 1; zero scores remain diagnostic.

4. `LAYER_H64_Z64_PLUS_LOCAL_RESIDUAL_R32`
   - layer-shared rank-64 primary plus target-local activation-aware rank-32
     residual;
   - exact additive byte billing;
   - constructible only with at least 32 leakage-free target fit rows;
   - exploratory fallback, never silently substituted for arm 3.

5. Negative controls:
   - centered-only fit;
   - production output-side down;
   - all-row target-local score;
   - all must be labeled non-promotional and cannot select an arm.

No Gaussian or synthetic proxy may select a representation.

## Target fit and holdout

Use a fixed preregistered seed and publish it.

- route count >= 32: deterministic 80/20 split of exact target route rows;
- route count 2–31: target-local arms fail closed; shared arm uses every
  deterministic target holdout row and no target fit;
- route count 1: the sole routed row is holdout; no self-fit;
- route count 0: no route-conditioned holdout exists; use only the separate
  counterfactual diagnostic set described above.

The shared layer fit must be identical across targets in that layer and exclude
the union of all target holdouts.

For gate/up, score true versus reconstructed outputs on `X_hold`.
For down, derive true target `Z_hold` from the target's real gate/up weights and
score input-side down outputs on exactly that `Z_hold`.

Publish fit/holdout indices and SHA-256 witnesses, target and peer row counts,
basis hashes/identities, target-exclusion proof, and no-Gaussian proof.

## Floors and verdicts

Retain the sealed v2 floors:

- promotion panel at rank 64: minimum cosine `0.85`, median `0.96`;
- between-anchor and below-anchor candidate tensors at rank 128: every
  gate/up/down cosine at least `0.91`.

For the rank-128 shared treatment, also report promotion-band results at rank
128, but do not rewrite the sealed rank-64 promotion result.

Count-1 peer-transfer must reach `0.91` per tensor to remain a candidate, but
three single rows are not population proof.

Zero-route scores have no promotion floor. They remain diagnostic regardless of
their magnitude.

The final receipt may set:

- `shared_rare_route_candidate_survives_bounded_panel=true` only if the
  deployment-honest arm 3 clears every route-conditioned per-tensor floor and
  all integrity gates;
- `wider_bounded_pilot_authorized=true` only under the same condition;

but must always keep:

- `full_parent_traversal_started=false`;
- `full_traversal_authorized=false`;
- `capable_artifact_claimed=false`.

If shared arms fail, say so and stop. Never reduce rank/floors to fit.

## Source lifecycle CLI

Implement a single fail-closed tool with these commands:

```text
preflight
acquire --shard 35 --confirm ACQUIRE_EXACT_RARE_ROUTE_SHARD_00035
measure --shard 35
release --shard 35 --confirm RELEASE_EXACT_RARE_ROUTE_SHARD_00035
aggregate
selftest
```

Equivalent exact phrases apply to 86 and 264.

The implementation task may run only `preflight`, `selftest`, fake-only tests,
and compilation. It must not run real `acquire`, `measure`, `release`, or
`aggregate`.

### Acquisition requirements

- accept only one of 35, 86, or 264;
- before any write, require `free_bytes - sealed_shard_size >= 75000000000`;
- the 75-billion-byte floor is hard-coded and cannot be lowered by environment;
- refuse if any other `model-*.safetensors` body is resident;
- download only the exact immutable
  `zai-org/GLM-5.2@b4734de4facf877f85769a911abafc5283eab3d9`;
- use the isolated pilot cache, never MOP's cache;
- verify size and full SHA-256 before publishing the body;
- quarantine a mismatch;
- append a lifecycle ledger.

Existing incomplete cache files are occupied disk, not verified source.
Do not delete or count them as source.

### Measurement partial

Write one canonical sealed partial per shard:

- `GLM52_RARE_ROUTE_PILOT_PARTIAL_00035.json`;
- `GLM52_RARE_ROUTE_PILOT_PARTIAL_00086.json`;
- `GLM52_RARE_ROUTE_PILOT_PARTIAL_00264.json`.

Each binds source size/hash, exact capsule members and sealed member hashes,
code/test/preflight hashes, target/peer tensors, row identities, basis
identities/hashes, every score, floor result, non-claims, resource sample, all
false fences, and a canonical receipt SHA-256.

Load capsule members selectively. Do not decompress a whole capsule and do not
load unrelated hidden/logit/output members.

### Release requirements

Release must rerun a complete gate in the same process:

- exact target is a regular non-symlink file below the exact pilot root;
- no other source body is resident;
- size/full hash match;
- matching sealed measurement partial exists and validates;
- all required targets were measured;
- no Gaussian, no leakage, zero not promoted, all fences false;
- process scan plus `lsof` prove there is no other consumer;
- exact confirmation phrase matches.

Delete only the one explicit verified source path. No glob or recursive delete.
Retain lifecycle ledger, logs, cache, incomplete files, `hf_home`, and pilot
root. Publish a canonical per-shard release receipt with before/after free bytes
and prove the body absent.

### Aggregate

Require all three valid partials and all three valid release receipts, with all
three bodies absent. Emit:

- `GLM52_RARE_ROUTE_BASIS_PILOT_RECEIPT.json`;
- `GLM52_RARE_ROUTE_BASIS_PILOT_RECEIPT.md`.

Aggregate exact panel floors, byte ledgers, constructibility failures,
diagnostic-only zero results, remaining uncertainty, verdicts, and all safety
fences. Never hand-edit generated receipts.

## Preflight deliverables

- `tools/condense/glm52_rare_route_basis_pilot.py`
- `tools/condense/tests/test_glm52_rare_route_basis_pilot.py`
- `GLM52_RARE_ROUTE_PILOT_PREFLIGHT.json`
- `GLM52_RARE_ROUTE_PILOT_PREFLIGHT.md`

The preflight must be source-body-free, deterministic, and include exact
candidate ledgers, the derived panel, lifecycle plan, scientific protocol,
code/test/input hashes, non-claims, and all fences.

## Fake-only tests

Use temporary small fake worlds only. Cover at least:

- exact panel derivation and a triplet-straddle refusal;
- v2 ledger component reconciliation and exact F1/F3 totals;
- max-shared-basis boundaries 150/151 and 4977/4978;
- deployment-honest one-basis identity shared by all targets;
- all panel targets and holdouts excluded from shared fit;
- route-count 1 self-fit forbidden;
- zero-route never promotional;
- target-local and local-down arms not constructible below 32 fit rows;
- real SwiGLU down derivation and input-side scoring;
- Gaussian/output-side/all-row/centered controls never promote;
- absolute floor aggregation;
- hard 75-billion-byte acquisition gate;
- wrong shard or confirmation refusal;
- one-body-at-a-time refusal;
- size/hash mismatch quarantine;
- real-source commands disabled in fake tests;
- partial canonical seal and tamper detection;
- release requires a valid complete partial and reruns its gate;
- symlink/path escape/extra body/live consumer/no-process-probe refusal;
- explicit one-file release preserves ledgers/logs/caches/incomplete files;
- aggregate refuses missing/tampered partials or release receipts;
- aggregate always leaves full traversal and all fences false;
- deterministic preflight generation.

Run fake-only tests, `selftest`, `py_compile`, existing v2 tests/selftest, route
census tests/selftest, and `git diff --check`.

Commit only the four preflight deliverables. Exclude `.serena`, temporary
partials, lifecycle receipts, and source files.

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
negative controls, teacher capsules, real source bodies, or remote state in the
implementation task.

## Required report

Return exact files and commit, test counts, deterministic receipt hash, byte
tables, panel derivation, source-body non-mutation proof, remaining
uncertainties, and the exact next supervising command. A skipped test is not
evidence.
