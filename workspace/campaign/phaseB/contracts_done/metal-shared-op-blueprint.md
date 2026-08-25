# Task contract — metal-shared-op-blueprint

## Objective
Write an implementation-ready DESIGN BLUEPRINT for adding a "shared-operator" MLP execution kind to
the Hawking Qwen3.8 hybrid decoder. Deliverable is one markdown report. Do NOT modify the decoder or
any Rust/Metal source. The shared operator is ONE SwiGLU operator (gate G, up U, down D matrices)
reused across ALL 64 layers plus a tiny per-layer FiLM code (gamma, beta vectors). Per layer:
  inter = silu(x @ G.T) * (x @ U.T);  inter = inter * gamma_L + beta_L;  y = inter @ D.T
Intermediate width m ~= 6144-8192 (native is 17408). Physics win: G/U/D (~38 MB, group-64 quantized)
load ONCE and stay resident, reused for all 64 layers, so MLP DRAM traffic/token drops from
~64*108.6 MB to ~38 MB + 64 tiny codes.

## Context
Local Hawking NOS. Decoder crates/hawking-core/src/model/qwen38_hybrid_decode.rs (5950 lines) runs a
mixed-catalog artifact. MLP weights dispatch through a group-64 GEMV Metal path with per-tensor
native kinds (HGRAVU01 uniform q4, HGRAVB01 binary, HGRAVR02 residual, HGRAVS01 low-rank) bound from
a mixed catalog. Shaders in crates/hawking-core/shaders. Current per-layer MLP does gate/up/down
GEMVs at intermediate=17408. We add a path binding ONE shared operator (m~=6144) + per-layer FiLM
codes executing the SwiGLU above.

## Accepted architecture
- Shared operator is a NEW mixed-catalog MLP kind (e.g. MixedMlpNativeKind::SharedOp), selectable
  per-artifact, not replacing existing kinds.
- Weights group-64 quantized like other kinds; reuse the existing group-64 GEMV kernel where possible.
- FiLM codes per-layer (gamma[m], beta[m]) elementwise on the intermediate.
- Residency: shared G/U/D buffers allocated once, reused every layer; only FiLM code + input differ.

## Non-goals
- MLP only; no attention or DeltaNet design.
- No on-disk pack/catalog format redesign beyond minimal naming additions.
- No speculative optimization beyond what the residency win requires.

## Read these sources
- READ crates/hawking-core/src/model/qwen38_hybrid_decode.rs for the per-layer MLP dispatch, the
  mixed-catalog binding (load_mixed, MixedMlpNativeKind, mixed_mlp_native_kind_from_lane,
  assert_mixed_mlp_native_kinds, is_mixed_mlp_gemv_name), and the group-64 GEMV dispatch.
- READ crates/hawking-core/shaders for the group-64 matvec kernel (e.g. geo_tpr64): which kernel the
  MLP GEMV uses and its buffer/threadgroup signature.

## Deliverable
- Write workspace/campaign/metal_shared_op_blueprint.md with the full blueprint.

## Do not touch
- Never modify any file under crates. Never modify any .rs or .metal file.

## Acceptance criterion
Done when workspace/campaign/metal_shared_op_blueprint.md exists, is non-empty, cites exact per-layer
MLP dispatch function+line numbers and the real group-64 GEMV kernel signature, and covers all six
completion items. Must pass the verify check below. A blueprint that does not cite the real dispatch
site and real kernel signature is a fail.

## Completion criteria
- [ ] Exact call site(s) where per-layer MLP GEMVs dispatch, cited by function+line.
- [ ] How MixedMlpNativeKind::SharedOp threads through catalog lane to kind mapping to role check to
      GEMV dispatch, naming every function needing a new match arm.
- [ ] Metal kernel plan: reuse group-64 matvec for x@G.T, x@U.T, inter@D.T; where silu*gate and FiLM
      (inter*gamma+beta) elementwise steps run (new tiny kernel vs fold in), buffer bindings.
- [ ] Residency plan: how G/U/D allocate once and reuse across 64 layers; what currently forces
      per-layer weight binding and what changes so shared buffers persist.
- [ ] One-layer-swap test plan: replace only layer L MLP with the shared op in a real decode and diff
      logits/coherence vs the q3 patient using the existing greedy example runner.
- [ ] Estimated active MLP bytes/token and dispatch count, shared-op path vs current per-layer path.

## Verify
- VERIFY: test -s workspace/campaign/metal_shared_op_blueprint.md

## Required evidence
Quote exact function names, line numbers, signatures. Paste real grep/read output, not summaries.

## Limits
- Max turns: 30. No commits to any branch other than the one you are on. No push, merge, deploy.
  External network: not permitted.

## Required completion report
End with: ## SUMMARY / ## FILES CHANGED / ## DECISIONS MADE / ## ASSUMPTIONS /
## DEVIATIONS FROM CONTRACT / ## TESTS RUN / ## KNOWN LIMITATIONS / ## REMAINING WORK / ## CONFIDENCE
