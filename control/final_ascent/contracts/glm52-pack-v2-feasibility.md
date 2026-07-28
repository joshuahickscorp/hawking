# Contract: GLM-5.2 activation-aware pack v2 feasibility

## Purpose

Build an opt-in, source-body-free feasibility and fake-data implementation for the
next GLM-5.2 representation program.  This task must correct the scientific defects
proven by Generation B without starting a full parent traversal or claiming a
capable model.

The old v1 packer and all existing receipts are historical evidence.  Preserve its
defaults and behavior.  Prefer a separate v2 module and tests over invasive edits.

## Inputs that may be read

- `GLM52_BASIS_PILOT_RECEIPT.json`
- `GLM52_BASIS_PILOT_CONTROLLER_RESEAL.json`
- `GLM52_GENERATION_B_CAPABILITY_VERDICT.json`
- `reports/condense/glm52_generation_b/GLM52_SOURCE_SHARD_HEADERS.json`
- `tools/condense/glm52_activation_aware_pack.py`
- `tools/condense/glm52_basis_pilot.py`
- existing unit tests and campaign status

The 282 source weight bodies must not be fetched, rehydrated, or traversed.  The
five pilot bodies have already been released.  Reading retained JSON headers,
receipts, and small fake fixtures is allowed.

## Required safety state

All generated output and code paths must bind:

- `RAMANUJAN_RESEARCH_AUTHORIZED=false`
- `HIDE_KERNEL_TURN=false`
- `ODYSSEY_LAUNCH_AUTHORIZED=false`
- `full_parent_traversal_started=false`
- `full_traversal_authorized=false`
- `capable_artifact_claimed=false`
- `MOP_touched=false`

Do not change the production disk floor, do not delete artifacts, do not modify
teacher capsules, and do not create `/Users/scammermike/Downloads/ramanujan`.

Any real traversal entry point must fail closed by default.  This contract does not
authorize implementing or exercising an escape hatch.

## Scientific requirements

1. **Retain the mean.**  V2 uses an uncentered SVD basis.  The pilot proved
   uncentered and explicit-mean equivalent within `1e-4`; choose uncentered because
   it represents the mean direction without a second encoding path.  Centered-only
   fitting is forbidden for v2 promotion.

2. **Route-conditioned routed experts.**  For
   `model.layers.L.mlp.experts.E.{gate,up}_proj.weight`, select real
   `pre_router_hidden` rows where `topk_indices` contains expert `E`.  Gate and up
   for one `(layer, expert)` share one hidden-input basis identity.  Empty or
   undersampled route selections fail closed; they never silently fall back to all
   rows.

3. **Real SwiGLU inputs for down projection.**  For routed and shared
   `down_proj`, form held-out and fitting inputs with the actual matching gate/up
   weights:

   `Z = silu(X @ W_gate.T) * (X @ W_up.T)`.

   Fit and score down projection on `Z` as an input-side projection.  Gaussian
   probes and production output-side down projection are forbidden.

4. **Real inputs for every promotional score.**  Shared MLP uses all real capsule
   rows and its actual gate/up weights.  Router and supported attention input
   projections use real capsule inputs.  Classes lacking the required real
   intermediate (notably `attention.o_proj` in current capsules), global
   embeddings, and `lm_head` remain explicitly unvalidated/native in the
   feasibility census.  Do not invent activations.

5. **Absolute floors, not relative admission.**  `beats_null` is diagnostic only.
   An allocator/program selector must reject a point below its preregistered
   organ-class absolute floor even if it beats its null.  It must fail the byte
   budget rather than lower a rank or quality floor.  Native fallback is allowed
   only when billed exactly at source payload width.

6. **Preregister the heterogeneous candidate, without promotion claims.**  Encode
   the current bounded candidate:

   - high-traffic routed gate/up/down: uncentered rank 64; panel floor
     minimum cosine `0.85`, median `0.96`;
   - low-traffic routed diagnostics: rank 128; per-tensor floor `0.91`;
   - shared MLP gate/up/down: rank 256; per-tensor floor `0.91` and panel median
     `0.93`;
   - router control: rank 128; per-tensor floor `0.99`;
   - supported attention input control (`q_a_proj` only at present): rank 128;
     per-tensor floor `0.91`;
   - every other class: native/unvalidated until a later bounded real-input pilot.

   These values are a feasibility candidate derived from the sealed five-shard
   receipt.  They are not whole-model capability thresholds and may not authorize
   a traversal.

7. **No unsupported transfer claim.**  Teacher capsules cover only a subset of
   layers.  The feasibility ledger may report both:

   - conservative target-local basis billing (one unique routed hidden basis and
     one unique down-input basis for every target `(layer, expert)`), and
   - a separately labelled transfer-sharing scenario.

   Only the conservative target-local total may decide `within_target_bpw`.
   Cross-layer transfer must remain unvalidated and must not reduce the
   authorization-deciding total.

## Exact byte-accounting requirements

Build the census from all 59,585 tensor entries and shapes in
`GLM52_SOURCE_SHARD_HEADERS.json`; do not extrapolate from the five-shard panel.
Reconcile the census to:

- 59,585 unique tensor names;
- 753,329,940,480 original weights;
- 1,506,659,919,872 source payload bytes.

Bill each physical object exactly once:

- float16 coefficient matrices;
- float16 basis matrices keyed by a serializable basis identity;
- tensor headers/metadata/alignment/packaging if the proposed ABI uses them;
- native tensors at their recorded source payload bytes.

For a routed expert at rank `r`, gate/up share the hidden basis, while down owns a
distinct real-SwiGLU-input basis.  Do not bill a basis once per tensor and do not
bill it once per layer when the identity is per expert.  The ledger must expose
unique basis identities, reference counts, component totals, total bytes, exact
complete BPW, and an itemization reconciliation check.

The target is `49/50` BPW.  If the conservative program is above it, report
`within_target_bpw=false`; do not weaken floors or hide native islands.

## Runtime/serialization requirements

Implement a small v2 fake codec/ABI proof whose serialized metadata identifies at
least:

- format version;
- organ class;
- layer;
- expert id when applicable;
- projection side;
- basis kind (`uncentered_hidden`, `real_swiglu_input`, or another explicit real
  input kind);
- basis identity and rank;
- activation provenance and route-conditioning status.

Round-trip fake gate/up/down tensors through shared-basis encoding/decoding and
verify numeric reconstruction.  Demonstrate that gate/up share the intended basis,
that different experts do not alias, and that down uses the separate SwiGLU basis.
The fake implementation must be deterministic under a fixed seed.

## Deliverables

- `tools/condense/glm52_activation_aware_pack_v2.py`
- `tools/condense/tests/test_glm52_activation_aware_pack_v2.py`
- `GLM52_V2_PROGRAM_FEASIBILITY.json`, generated deterministically from the sealed
  headers and pilot receipt
- `GLM52_V2_PROGRAM_FEASIBILITY.md`, a concise human-readable interpretation
- any minimal additional fake fixture only if unavoidable

The generated JSON must include source hashes, code/test hashes where practical,
the preregistered program, pilot checks, exact census, both conservative and
clearly non-authorizing transfer-scenario ledgers, unsupported classes, remaining
uncertainties, and every safety fence above.

## Required tests

At minimum cover:

- uncentered basis retains a nonzero mean direction and differs from centered;
- deterministic route row selection from `topk_indices`;
- empty/undersampled expert routes fail closed;
- real SwiGLU `Z` construction matches a direct reference;
- no Gaussian/proxy selection path exists in v2;
- absolute organ floors override `beats_null`;
- budget failure never reduces a floor;
- basis identities/refcounts and exact-once billing;
- gate/up basis sharing, expert non-aliasing, separate down basis;
- native source-width billing;
- full 59,585-tensor census reconciliation;
- deterministic feasibility receipt generation;
- all authorization fences false;
- v1 defaults/CLI selftest remain passing.

Run focused tests, v1 packer selftest/regression tests, and `py_compile`.  Report
commands and exact pass counts.

## Non-claims

This task does not prove representation quality on uncovered layers, low-traffic
experts, globals, or attention output projections.  It does not authorize a full
traversal, capability gate, HIDE kernel turn, Odyssey launch, Math-Frozen, or
Ramanujan research.
