# HAWKING SECOND LIGHT :: Required Report

> Second Light is the actual descent. First Light proved Hawking could touch the parent.

This report answers the 28 required points (goal Section 28). All artifacts are under
`reports/condense/second_light/`; all tools under `tools/condense/second_light_*.py`. Nothing was
committed or pushed (house rule). No em/en dashes. Apple Silicon (M3 Ultra, MPS) only.

## 1. Authoritative main commit
`7f237ed3` on branch `main` (verified live at precheck time). Rollback anchor unchanged.

## 2. Correction of prior run status (Section 0)
`SECOND_LIGHT_PRECHECK.json` -> **full_run_status = NOT_STARTED**, established from LIVE evidence:
no hawking controller process alive, both launchd jobs dead (exit statuses, not PIDs), no advancing
lease/heartbeat. The prior commit `0504b0f7` "one Gravity run ignited" is reclassified as
**FIRST-LIGHT CALIBRATION** (untrusted per Section 0; no live process ever advanced a full program).
A committed JSON or a historical PID is explicitly NOT trusted as a live run.

## 3. First-Light calibration identity
`GPT_OSS_120B_FIRST_LIGHT_CALIBRATION.json` + `_DOSSIER.md`. Bounded slice campaign (layer 0, 128
experts, 256x256 mlp1 slices, low-rank ternary) sealed as calibration with the boundary statement.
Evidence bound: effective rank ~104/256 (experts HIGH-RANK), SVD rank-8 error 0.916, ternary rank-8
0.986, PQ subdim8 K256 @ 1 BPW rel-err 0.543, PQ beats ternary 32/32 (~45% lower error), residual
kurtosis 5.19. Low-rank ternary REJECTED as principal geometry; PQ selected on evidence.

## 4. Quality contract
`GPT_OSS_120B_QUALITY_CONTRACT.json` (sha `6edc2121...`). 7 hard invariants, soft thresholds grounded
in measured baselines, calibration/validation/holdout partitions (holdout never tuned), Harmony
config, deterministic seeds, reference runtime + revision `b5c939de`, metric tolerances, failure
conditions, Gate 1-7 hierarchy. Gate law: no red gate made green by weakening its threshold.

## 5. PQ family
`gravity_forge.py` first-class PQ: `PQFamily` with the 7 verbs (inspect/fit/pack/measure/execute/
validate/repairability), plain `pack_product_quant` (PQ as its own geometry) + rotated `transform_pq`.
`pq_execute` is a bounded direct compact matvec (no dense shadow); matches dense recon to ~1e-7.
34 forge tests pass.

## 6. Codebook-sharing strategy
Shared amortized codebooks per (layer, expert-class) via `shared_expert_grammar`; global shared
codebook for embeddings. Amortization billed ONCE, per-expert cost approaches indices-only.
`second_light_pack.pack_layer_grammar_full` streams the full 128-expert scope (one read, sampled fit,
exact per-expert assignment).

## 7. Protected-island strategy
`select_protected_islands` with 4 deterministic strategies (magnitude, activation_aware, sensitivity,
residual_energy); `pack_pq_protected_islands` bills protected rows (fp16 + row index) INSIDE the same
ByteLedger. Islands strictly increase whole-artifact BPW (no free islands). Program default:
residual_energy for experts, sensitivity for attention, magnitude for embeddings.

## 8. Doctor strategy
`doctor_pq` with 4 budgeted treatments (residual_codebook, sparse_residual, per_channel_scale,
protected_island_expansion). Returns {treatment, added_bytes, new_whole_bpw, quality_delta, evidence};
asserts added_bytes <= byte_budget; stores only billed corrections (no uncounted dense residual).
Program reserves 0.15 bpw Doctor budget on expert rows.

## 9. Activation dataset
Harmony-formatted prompts (general/code/math/tool-use) via `chat_template.jinja` + `tokenizer.json`
(vocab 201088, roundtrips). True-residual block-0 activations via `gptoss_block.py`. Calibration vs
holdout partitions declared in the quality contract.

## 10. Router / expert importance analysis
Router kept high-precision (capability-critical top-k). Program allocates codebook capacity + island
bytes to experts; every expert still counted in whole-artifact accounting. (Frequency-weighted byte
allocation is a declared refinement; whole-artifact accounting includes all experts regardless.)

## 11. CPU reference
`gptoss_moe_runtime.py` (CPU MoE reference, proven on the real 61 GiB source) + `gravity_forge.pq_execute`
(CPU direct compact execute). CPU is authoritative for final selection.

## 12. Metal kernels
MPS via torch: k-means assignment, batched distance, codebook lookup, direct PQ contraction, shared
grammar. `pq_cpu_metal_parity` enforces the Metal Quality Law.

## 13. Speedups
Full-scope streaming packer: 128 real experts (layer-0 mlp2) in ~10s vs ~128 GiB naive re-reads;
one full expert row (mlp1, 128 experts) sealed in ~20s through the controller. Staged gates over real
experts across 3-4 layers in ~59s.

## 14. CPU/Metal parity
`PQ_CPU_METAL_PARITY.json` on a REAL expert (block.0.expert.0.mlp2): pq_family_complete,
islands_complete, doctor_complete, cpu_execute_green, metal_execute_green, parity_green -> all_green.
Byte accounting exactly deterministic; recon may drift ~1e-6 on MPS so CPU is authoritative.

## 15-18. Staged gates (apparatus, real weights)
`STAGED_GATES.json` (synthetic-activation functional proxies, capability_parity=false):
- expert gate (layers 0/18/35): mean output div 0.457
- full-layer gate (layer 0): 0.604
- multi-layer gate (0/12/24/35): 0.595
- short end-to-end proxy: 0.594
These prove the apparatus RUNS on real weights; the faithful true-residual number is ~0.688. No
capability pass.

## 19. Exact whole-artifact rate accounting
`GPT_OSS_120B_PQ_GRAVITY_PROGRAM.json` (sha `3a4061f2...`): 183 rows, complete tensor scope
(116,829,156,672 logical weights), **complete-artifact BPW = 0.92788** (sub-bit budget), expected
output 12.62 GiB vs 60.77 GiB source (4.815x). Every row binds exact integer/rational budgets
(index + codebook + seed + metadata + island + Doctor reserve + alignment); packer fails over budget.

## 20. Program hash
`program_sha256 = 3a4061f2b8d467b4a994ce3f46224ffb96f4bcab3f88abd5b2e10925b22ce760`.

## 21. Readiness result
`GPT_OSS_120B_PQ_READINESS.json` (sha `d5dd3b0c...`): **apparatus_readiness_green = true, 25/25**.
Capability status reported SEPARATELY and honestly: sub-bit capability NOT passed (expert output
divergence ~0.69 >> the 0.60 promote threshold). Ignition launches the durable SEARCH; it does not
assert a capability pass.

## 22. Controller PID
`84690` (detached, setsid, survives chat exit). Singleton fcntl-flock lease
`com.hawking.second_light` held (count=1).

## 23. Lease
`com.hawking.second_light` at `reports/condense/second_light/leases/second_light.lease` (flock
liveness is the only truth; a dead pid can never read live).

## 24. Live heartbeat
`heartbeat/second_light.heartbeat.json`, self-sealed, refreshed every row; fresh (age ~3s at ignition
observe end).

## 25. First checkpoint
`checkpoints/r0000.json` SEALED: expert_mlp1 layer 0, FULL 128-expert scope, within budget,
whole_artifact_bpw 0.750016, rel_error 0.6769. Second and third rows sealed during observation.

## 26. Current progress
At ignition observe end: 3 rows sealed / 180 pending / 183 total, state RUNNING. Advancing (see
`SECOND_LIGHT_IGNITION_RECEIPT.json`).

## 27. ETA
~1h (dominated by the heavy 72 expert rows at ~20-25s each; attn/router/kept/embedding rows are
faster). One full pass produces the first complete candidate artifact + complete byte accounting +
resumable checkpoint graph.

## 28. Rollback
Controller `reset` clears only controller-owned checkpoints (never the shared evidence dir, fixed
this session); campaign state is fully resettable to NOT_STARTED. Main commit unchanged; no git
mutation. Legacy rollback anchors (`hawking-pre-collapse-main`) intact.

---

## Adversarial verification (house rule: independent reproduction before any claim)
`ADVERSARIAL_VERIFICATION.json`. 7 independent skeptics, each instructed to REFUTE one claim:
**6 CONFIRMED, 1 PLAUSIBLE (fixed), 0 REFUTED.**
- CONFIRMED: source manifest byte-ranges; program accounting closes over all 543 tensors; forge PQ
  byte accounting (deterministic, islands billed, Doctor within budget, execute==dense); full-scope
  streaming packer (128 experts assigned exactly, only the codebook fit sampled); controller
  singleton + 5-way crash/resume idempotency; no capability overclaim anywhere.
- DEFECT found + fixed: `second_light_precheck.py` hardcoded `advancing = False`, making RUNNING
  unreachable (honest-by-coincidence, not by measurement). Now MEASURED from
  `second_light_status.snapshot()` (flock liveness + fresh heartbeat + working/sealed cursor);
  `com.hawking.second_light` added to the launchd probe; heavy-process scan tightened to exclude
  shell wrappers and the precheck's own tree. Reachability of RUNNING was then adversarially proven
  (idle -> NOT_STARTED; live+advancing controller -> RUNNING; kill+reset -> NOT_STARTED).

## Ignition
`SECOND_LIGHT_IGNITION_RECEIPT.json`. **ignited = true.** Durable singleton controller PID 84690
launched detached (setsid, survives chat exit) on the COMPLETE 183-row program at FULL 128-expert
scope. Observed genuinely advancing: state RUNNING, lease live (count=1), heartbeat fresh, first real
program row (r0000, 128 experts) sealed and valid, second/third rows progressing, resources sane.
`capability_claim = false`. This launched the durable search producing the first complete candidate
artifact; it is NOT a capability pass or Event Horizon. Sub-bit expert divergence remains large.

## Honesty ledger
- Sub-bit expert PQ remains a LARGE perturbation (weight rel-err ~0.68, true-residual output div
  ~0.69). This is negative science, consistent with First-Light. No capability pass, no Event Horizon,
  no Gravity escape is claimed anywhere.
- Embedding rows pack a labelled representative 65536-row slice with the whole-tensor budget
  extrapolated (`bounded_slice=true`); experts (99% of weights) pack at FULL 128-expert scope.
- Reference forward is authoritative for RELATIVE orig-vs-packed divergence, not HF-parity absolute
  perplexity (Gate 7 requires an HF-validated forward).
- The separate MoP project runs a heavy CPU load (~load 22); the PQ campaign is MPS/GPU-bound so they
  coexist. MoP was not touched.
