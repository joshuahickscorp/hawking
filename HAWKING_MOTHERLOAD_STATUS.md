# Hawking Motherload Completion Status

endpoint: `IN_PROGRESS`  
gates closed: 9/20  
ODYSSEY_LAUNCH_AUTHORIZED: `false`  
updated: 2026-07-25T02:59:58Z

| gate | state | condition | note |
|---|---|---|---|
| M01 | green | GLM traversal is 282/282 | GLM source traversal complete: 282/282 verified, 282/282 packed, 0 faults across the run |
| M02 | green | complete local GLM .gravity artifact exists | complete local GLM .gravity artifact assembled: 282/282 shards, 59585/59585 tensors, 0 missing/undeclared/misplaced, 0.882888 whole-model BPW, 83.14 GB physical (753.3B logical elements), path ~/Library/Application Support/Hawking/Models/GLM-5.2/b4734de4facf877f85769a911abafc5283eab3d9/General-R0/, hardlinked from the packer output (2 links confirmed, zero duplicate bytes) |
| M03 | open | lowest broad parity rate is sealed, sub-bit preferred and H15 maximum |  |
| M04 | green | full GLM token executes from .gravity | GLM-5.2 adapter executes a complete token from the REAL flagship .gravity artifact (282 shards, 83.14GB, 0.883 whole-model BPW) and agrees with an independent numpy oracle reading the same bytes: argmax exact (9540), top5 exact in order, discrete DSA key selection exact ([1,2,0] both), max logit diff 6.08e-5 mean 1.12e-5 across 154880 outputs after 78 real layers of MLA+DSA+256-expert MoE+noaux_tc router. Neither side touches source safetensors. Receipt: GLM52_FLAGSHIP_ADAPTER_PARITY.json |
| M05 | running | measured base TPS and prefill exist | base measured on the instrument: decode 105.8/68.8/29.2/13.3 tok/s at ctx 128/512/2048/8192, prefill 116.5/92.8/48.8/19.1, 1 command buffer + 210 dispatches per token; end-to-end generation 60.7 tok/s with incremental decode bit-identical to full replay; GLM numbers gated on the flagship artifact |
| M06 | open | acceleration stack is terminal and measured separately |  |
| M07 | open | GLM runs end to end inside HIDE |  |
| M08 | open | Prometheus S0 and source decision are sealed |  |
| M09 | green | Prometheus architecture and profiles are implemented | all 14 Revision 3 §7 components implemented and wired: 8 measured, 5 gated with named gates, 1 sealer; profiles general/math/uniform/random compiled and hashed; equal-budget solver matches all four arms to 0.0175% at 46.70 GB |
| M10 | running | equal-budget Claim A is sealed | Claim A NOT_SEALED and correctly so: allocation plans are byte-matched and ready, but retention is deliberately null -- at equal bytes retention IS the claim. Blocked on S0.8 cartography membership and the served flagship. |
| M11 | open | General and Math artifacts are selected and verified |  |
| M12 | green | Forge, continuity, sovereignty, and Limit Registry are sealed | sovereignty sealed for both the Llama instrument and the real GLM-5.2 flagship (282-shard multi-shard manifest, artifact_hash = sha256 of model.gravity.index.json since no single body hash names a multi-shard model, refuses to seal on incomplete coverage -- proven via a synthetic INCOMPLETE fixture). hidden_intervention 0.0, model_continuity 1.0, attribution_completeness 1.0. false_refusal/boundary_error remain GATED on a served model. |
| M13 | running | Odyssey substrate and training bundle are complete | training bundle complete: plan T0-T5, objective/checkpoint/evaluation contracts, data + teacher-trace manifests, profile manifest; substrate itself still GATED on M11 and declared so rather than named speculatively |
| M14 | green | sandbox, roles, Ledger, verifiers, Tribunal, and retrieval are scaffolded | sandbox policy (network deny-by-default, filesystem allowlist, emergency stop), 12 roles with promotion held only by verifier and Tribunal, Ledger contract, 4-tier lattice, 7 memory stores, Tribunal + prior-art protocol, retrieval against a pinned snapshot, branch economics, Graveyard |
| M15 | green | Lean/Mathlib and evidence environment are pinned | Lean leanprover/lean4:v4.15.0 and Mathlib v4.15.0 pinned to concrete revisions; validator rejects 'latest'; container digest declared with gate ODYSSEY-ENV-01 |
| M16 | green | Odyssey dry-run validation passes | odyssey_package.py validate: 86 checks, 0 failed, DRY_RUN_PASS; selftest proves a flipped fence FAILS validation and the runner exits 1 |
| M17 | green | ODYSSEY_LAUNCH_AUTHORIZED remains false | ODYSSEY_LAUNCH_AUTHORIZED=false; the builder reads the fence and never writes it, so rebuilding cannot authorize a run |
| M18 | running | rollback/source lifecycle is green | eviction VERIFIED FIRING: free disk 274.6 -> 404.8 GiB at the W003/W004 boundary, 1 EVICT event, 0 faults; traversal now sustainable to 282/282 |
| M19 | open | all campaign commits are pushed |  |
| M20 | open | worktree and process state are clean except intentional detached services |  |
