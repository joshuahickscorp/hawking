# Condensation Doctor v5 — Canonical Capability-First Specification

**Version:** Doctor v5.0 / Training Ladder v5.0  
**Date:** 2026-07-12  
**Machine envelope:** Apple M3 Ultra, 96 GiB unified memory, 1 TB storage  
**Objective:** maximize verified capability and quality per exact physical artifact byte  
**Speed:** explicitly deferred to a later native-runtime proof ladder  
**Current verdict:** **v5 static package hardened after adversarial Audit C; no execution, evidence,
or measured dominance**

Doctor v5 is Hawking's fail-closed system for discovering, restoring, elevating, and proving the
quality of extremely condensed models. It is not a claim that Hawking already has the best codec,
the best quantized model, a high-fidelity sub-bit model, or a 671B/1.6T model that runs on this
Studio. It defines exactly what would have to be built and measured before any such statement is
permitted.

The version number measures the maturity of the research and proof contract. It does not measure
experimental success. A v5 artifact can fail. A complete negative result is valid v5 evidence. An
unmeasured v5 plan is not evidence.

---

## 1. Canonical local system

These files jointly define Doctor v5. This document is the human-readable canonical specification;
the Python surfaces are stricter machine validators.

| Artifact | Canonical role | Current boundary |
|---|---|---|
| [Three research expansion passes](DOCTOR_V5_RESEARCH_PASSES.md) | v3 representation foundry, audit A, v4 capability lab, audit B, and v5 adversarial synthesis rationale. | Design record; no execution. |
| [Doctor-v5 contract](../../tools/condense/doctor_v5_contract.py) | Program, artifact, observation, evidence-state, exact-resume, and dominance schemas. | Deterministic, stdlib-only validator; loads no model and launches nothing. |
| [Doctor-v5 campaign compiler](../../tools/condense/doctor_v5.py) | Sourced mechanism registry, equal-byte controls, failure-route search space, deterministic candidate sampling, and planned-program materialization. | Every mechanism executor is unwired; every candidate is unlaunchable. |
| [Quality Battery v5](../../tools/condense/quality_battery_v5.py) | Capability suites, data firewall, matched test-time-compute policy, statistical units, and dominance-expiry policy. | Manifest compiler, not an evaluator. |
| [Training Ladder v5](../../tools/condense/training_ladder_v5.py) | Complete model × physical-rate × claim-scope × stage plan. | Planner only; no process, model, or live-state access. |
| [Doctor-v5 root compiler](../../tools/condense/doctor_v5_root.py) | Builds the single immutable package identity over reports, implementation sources, and specifications. | Identity only; cannot grant execution or evidence. |
| [Doctor-v5 static auditor](../../tools/condense/doctor_v5_audit.py) | Cross-validates all v5 schemas, reports, counts, claim firewalls, materialization paths, documents, and no-launch source surfaces. | Static integration proof only; reads no live state and authorizes no execution. |
| [Compiled Doctor-v5 campaign](../../reports/condense/doctor_v5_campaign.json) | Hash-bound snapshot of mechanisms, controls, explicit candidates, and projected search size. | Planning artifact; zero wired or launchable entries. |
| [First materialized v5 program](../../reports/condense/doctor_v5_first_program.json) | Example candidate lowered into the strict program schema. | Planned mode with unwired operators; not executable evidence. |
| [Compiled quality manifest](../../reports/condense/quality_battery_v5.json) | Hashable 21-suite quality-battery plan. | Contains no benchmark data or model output. |
| [Compiled training ladder](../../reports/condense/training_ladder_v5.json) | Hashable 32-model, ten-rate, four-scope plan. | Contains no runnable executor or launch permission. |
| [Doctor-v5 root manifest](../../reports/condense/doctor_v5_root.json) | Canonical identity of the frozen v5 design package. | Necessary for later executable validation; never sufficient for it. |
| [Doctor-v5 audit receipt](../../reports/condense/doctor_v5_audit.json) | Static adversarial-integration verdict bound to the root manifest and compiled reports. | May pass static integrity only; `execution_authorized=false`, `evidence_complete=false`, and `dominance_proven=false`. |
| [Condensation Doctor v2](CONDENSATION_DOCTOR_V2.md) | Historical design lineage and prior mechanism vocabulary. | Superseded where v5 is stricter. |

The v5 schemas are:

- `hawking.doctor_v5_program.v5`;
- `hawking.doctor_v5_artifact.v5`;
- `hawking.doctor_v5_observation.v5`;
- `hawking.doctor_v5_dominance.v5`;
- `hawking.doctor_v5_campaign.v5`;
- `hawking.quality_battery.v5`; and
- `hawking.training_ladder.v5`; and
- `hawking.doctor_v5_root.v5`.

### 1.1 Actual current state

| Surface | What exists | What does not exist |
|---|---|---|
| Contract | Fail-closed program, real-file artifact, observation, and dominance validation hardened against Audit C attacks. | No source-bound executable adapters, trusted evidence verifier configuration, or experimental observations. |
| Campaign | 191 sourced mechanisms, including 74 explicitly marked Hawking hypotheses; 21 direct competitors; 32,768 deterministic explicit candidates; and 10,240 exact per-lane controls sampled from a projected 2,386,972,468,454,400-program combinatorial space. | No v5 campaign has permission to load or mutate a model; projected programs are not jobs or evidence. |
| Ladder | 32 models × 10 rates × 4 scopes = **1,280 research lanes**; 11 stages = **14,080 referenced stage cells**. | No lane is launched; exact integer source tensor counts are still required before any physical-bpw evidence. |
| Battery | 21 planned suites covering all 12 capability domains. | No private items, model outputs, or scores are embedded in the manifest. |
| Execution | F0–F8 are present; all 191 executors are `wired=false`; all 32,768 candidates are `launchable=false`; all 32 exact parameter manifests are unresolved; greenlight is false. | No Doctor-v5 training, repair, packing, sealed evaluation, or independent reproduction. |
| Claim | “Unproven; no unbeatable/dominant claim permitted.” | No frontier-champion or uniform-quality-dominance receipt. |

Legacy Studio downloads or older Doctor/ladder processes are outside this specification. Their
existence cannot be counted as v5 work without a v5 program identity and evidence receipt.

---

## 2. Non-negotiable v5 contract

Every v5 program must satisfy all of the following:

1. **Quality first.** Speed cannot select a winner, break a quality tie, or appear in a v5 quality
   observation.
2. **Four isolated claim scopes.** Codec fidelity, parent restoration, standalone elevation, and
   augmented-system capability never borrow evidence from one another.
3. **Physical bytes are authoritative.** Nominal payload bits are secondary; actual model-specific
   file bytes decide the budget.
4. **Complete ownership.** Every source tensor is assigned to a packed artifact component; dense
   parent fallback is forbidden.
5. **Disease before treatment.** Preserved-but-noisy signal and computation collapse are separate
   failure routes.
6. **Capability vector, not a scalar proxy.** Perplexity, KL, CKA, and weight error are diagnostics;
   every full claim covers all protected domains.
7. **Equal-byte and causal controls remain.** Zero treatment, untreated same-rate, scalar, smaller
   higher-bit, BF16 same-treatment, and best public same-byte controls cannot be optimized away.
8. **Information provenance is explicit.** Parent, stronger teacher, truth oracle, retrieval, tools,
   verifiers, auxiliary models, and persistent state are separately authorized and billed.
9. **Data are firewalled.** Search/selection data are separate from shadow, frozen, sealed, and
   independent-replication data.
10. **Uncertainty can block promotion.** An underpowered or route-undetermined result is
    inconclusive, not a win.
11. **Exact resume precedes mutating work.** Optimizer, RNG, sampler, source-shard offset, partial
    output hashes, and best-checkpoint identity must survive power loss or relocation.
12. **Negative results persist.** Failed seeds, rates, mechanisms, and scale transfers remain in the
    frontier ledger.
13. **No scale interpolation.** Evidence at 7B–32B does not become evidence at 70B, 671B, or 1.6T.
14. **Independent reproduction is required.** Hawking cannot self-certify a dominance headline.

Evidence progresses only through:

`planned → feasibility → tensor oracle → block oracle → shard oracle → full-model quality →
replicated quality → sealed final → independent reproduction`.

The observation evidence state is separately one of `PLANNED`, `RUNNING`, `PROVISIONAL`, `PROVEN`,
`INVALID`, `UNREPRODUCED`, or `REVOKED`. A dominance receipt requires independent-reproduction proof
state and `PROVEN` evidence state.

---

## 3. Four canonical claim scopes

The scope is frozen before operators, data, teacher authority, and evaluation are chosen. A result
cannot migrate after the fact.

Teacher authority is an exact-key, hash-bound contract rather than a permission Boolean. It names
teacher identity and revision, role, provenance, output/cache and split manifests, and training-only
lifetime, plus an externally trusted authorization receipt. Unknown teacher fields are rejected.
Restoration and augmentation require the stronger-teacher slot to be absent; elevation requires a
concrete authorized stronger teacher that is absent at inference.

| Scope | Allowed training information and operators | Inference boundary | Valid claim | Explicitly forbidden |
|---|---|---|---|---|
| `codec_fidelity` | Exact parent for representation/reconstruction only; diagnose, transform, represent, reconstruct, codec training, package, evaluate. | Standalone packed codec. | Quality attributable to the representation itself. | Doctor repair, hardening, stronger teacher, retrieval, tools, verifier, external state. |
| `restorative_training` | Exact identity parent and truth-verified parent behavior; static/conditional repair, codec-native QAT, hardening, parent-good replay. | Standalone artifact; zero external runtime bytes. | Damage restored toward parent parity. | Stronger teacher; calling beyond-parent gain “restoration”; external inference. |
| `capability_elevation` | Exact parent plus provenance-bound stronger teacher and truth oracle; repair/training/hardening. | Standalone artifact; teacher absent at inference. | Verified capability added to the artifact itself. | Hiding teacher uplift inside a restoration score; external inference. |
| `augmented_system` | Exact-parent/truth-verified core treatment plus retrieval, tools, verifiers, auxiliary models, or persistent external state. | Entire declared system, including every external byte and call. | Capability of the fully billed system. | Stronger training teacher (use a separate `capability_elevation` lineage); using augmented gains to support any standalone-model claim. |

Mandatory causal comparisons include:

- `codec_fidelity` versus parent, untreated same-rate, scalar equal-byte, and public same-byte codec;
- `restorative_training` versus codec-only, zero repair, exact-parent control, and BF16 receiving the
  same data and optimization;
- `capability_elevation` versus parent-only restoration, teacher-free training, and BF16 receiving
  the same stronger-teacher treatment; and
- `augmented_system` versus its closed-book artifact and the parent/competitors receiving the same
  augmentation budget.

Test-time compute is matched within scope. The frozen tuple includes maximum input, output, and
reasoning tokens; samples; temperature; timeout; verifier retries; retrieval calls; tool calls; and
external-model calls, plus the battery's identity/runtime, sampling, cache/speculation, resource,
and failure-policy fields. Core scopes require zero retrieval, tool, and external-model calls.
Augmented limits are separately preregistered and fully billed. These are ceilings, not entitlements
to spend more compute on a weak candidate.

---

## 4. Program, artifact, and observation ABI

### 4.1 Program identity

A Doctor program is a typed acyclic operator graph. Its identity binds:

- exact experiment and candidate hashes;
- parent revision, config, tokenizer, and chat-template hashes;
- an exact integer tensor parameter count from a hashed source manifest (rounded catalogue billions
  remain scheduling estimates only), plus separately declared active parameters;
- physical-bpw, physical-file-byte, and resident-byte ceilings;
- claim scope and diagnosed failure class;
- operator ids, mechanism sources, implementation state, dependencies, adapter ids, and source hashes;
- every data-split manifest, teacher cache, seed, and test-time-compute limit;
- exact-resume state; and
- evaluation suite, prompts, scorers, comparator registry, and test-time-compute protocol.

Operator kinds are `diagnose`, `transform`, `represent`, `reconstruct`, `repair_static`,
`repair_conditional`, `train`, `harden`, `augment_external`, `package`, and `evaluate`. A research or
unimplemented operator cannot be wired. A wired operator requires executable mode, a source hash,
and an audited adapter id.

Executable validation crosses no trust boundary by accepting self-stamped JSON. The caller must
supply role-separated trusted identities for the package root, user greenlight, exact parameter
manifest, teacher authority, adapter allowlist, and resource-admission receipt. A canonical
`program_spec_sha256` covers every semantic program field while excluding only the receipt wrappers
that would create a circular hash. Every authorization receipt binds that spec; changing a seed,
teacher, parameter count, operator, byte ceiling, or compute limit invalidates the old receipts. The
same object fails executable validation when external trust is absent or mismatched. A package hash
establishes identity; it does not establish user authority.

Every program, artifact, observation, receipt, operator, split row, accounting object, and nested
budget uses an exact allowed-key set. Unknown fields are rejected even when inserted before signing
and included in every hash. Campaign provenance has one explicit exact-schema metadata object; no
generic extension dictionary can smuggle adapter-visible authority.

The parameter-manifest receipt binds the exact integer tensor count and ownership classification to
the parent revision/config, source shard/file manifest, counting implementation, and program spec.
A positive integer or rounded catalogue estimate alone never establishes the physical-bpw
denominator.

### 4.2 Exact-resume boundary

Before any representation, reconstruction, repair, or training mutation, the checkpoint must contain:

- program and candidate identity;
- operator cursor and operator-local state;
- optimizer, scheduler, gradient-accumulation phase, and microstep;
- RNG state for every backend;
- sampler cursor and curriculum state;
- source shard and byte offset;
- teacher-cache and failure-replay identity;
- partial-output hashes and best-checkpoint identity; and
- resume-command identity.

Writes use an atomic replacement, fsync the file and parent directory, and validate before resume.
A resumed replay must be identity-equivalent. Wall-clock time is unbounded; losing weeks of work is
not acceptable.

### 4.3 Artifact contract

An artifact is valid only when:

- it is bound to the exact program;
- packed and decoder semantics are hashed;
- all tensor ownership is complete;
- no dense parent fallback exists;
- each physical file has a path, component, SHA-256, and actual length;
- the actual file sum equals both `all_in_physical_bytes` and the component-ledger sum;
- `all_in_bpw` is derived exactly from bytes and exact source parameter count;
- the physical ceiling is met; and
- decoded, peak, expected-active, worst-active, and context-specific bytes are present.

The byte ledger names exactly:

`base`, `pass_through`, `scales`, `codebooks`, `indices`, `corrections`, `routers`, `state`,
`metadata`, `alignment`, `tokenizer`, `retrieval_index`, `auxiliary_models`, and
`persistent_external_state`, plus `decoder_runtime`, `runtime_dependencies`, and `context_state`.

Non-augmented scopes require the last three external-system components to be zero. MoE artifacts
add total installed, expected active-expert, and worst active-expert bytes.

A completed observation must carry an externally trusted `artifact_validation_receipt` proving that
strict `verify_files=true` validation ran. The receipt binds the exact program and artifact, exact
integer parameter count, all-in bytes and derived bpw, aggregate file/component manifests, and
validator source identity. A self-stamped receipt or a receipt absent from the caller's
role-separated trust context is not evidence.

### 4.4 Observation contract

A full observation binds the artifact manifest and records:

- success, complete-negative, retryable failure, terminal failure, or invalidation;
- proof and evidence state;
- all twelve capability domains with candidate, parent, delta, lower/upper confidence bounds,
  non-inferiority margin, and sample count;
- every same-budget competitor's per-domain lower bounds and macro lower bound;
- data-firewall, parent-protocol-parity, and matched-test-time-compute receipts;
- one-time sealed-final consumption;
- exact external runtime bytes, which must be zero outside `augmented_system`; and
- a claim snapshot bound to the artifact, quality battery, and competitor registry.

Evidence-grade observations additionally bind raw per-item and independence-cluster outputs, every
training seed and calibration draw, preregistration, Holm-corrected test receipt, matched-compute
measurements, one-time sealed-service custody, and independent reproduction. `PROVEN` requires
caller-supplied, role-separated trusted identities for the evidence verifier, sealed service, and
independent owner. Self-stamped owner or signature-shaped hashes are rejected when that trust
context is absent.

Changing an artifact, benchmark, scorer, prompt protocol, comparator, or runtime semantics expires
the claim rather than silently updating it.

---

## 5. Diagnostic router: degradation is not collapse

The Doctor must not treat “quantization damage” as one disease.

### 5.1 Required probes

- weight-error spectrum, spectral decay, coherence, kurtosis, and outlier topology;
- activation energy, Hessian/Fisher sketches, and calibration distribution shift;
- early/middle/late hidden-state survival, CKA/geometry, logit KL/JS, top-k and tail mass;
- attention, KV-sensitive, router, shared-expert, and routed-expert traces;
- token entropy, margin-conditioned token flips, prompt sensitivity, and long-context accumulation;
- parent-correct/candidate-wrong capability clusters;
- activation, weight, block, expert, and causal-island patching;
- linear/low-rank readout oracles restricted to diagnostic data; and
- evaluator, chat-template, scorer, contamination, and teacher-regurgitation checks.

### 5.2 Hard route

| Diagnosis | Evidence signature | Admissible first action | Prohibited shortcut |
|---|---|---|---|
| `no_material_damage` | All protected differences lie inside preregistered noise and margins. | Retain zero treatment and proceed to packed/full evaluation. | Adding repair because unused bytes remain. |
| `signal_degradation` | Parent-relevant causal features survive but are noisy, rotated, biased, or conditionally obscured; a budgeted correction oracle recovers them. | Precondition, refine representation, apply static/sparse/low-rank repair, exact-parent distillation, or jointly trained conditional repair. | Assuming local recovery scales to full-model capability. |
| `computation_collapse` | Required rank, circuit, attention behavior, router decision, expert computation, or reasoning trace is absent; budgeted compensation cannot recover it. | Representation reset, structural reconstruction, protected precision island, codec-native QAT, binary/codebook relearning, then optional repair. | Compensation-only treatment, retrieval-backed standalone claim, or perplexity-only promotion. |
| `mixed_failure` | Some components preserve signal while others collapse. | Structural treatment for collapsed components plus independently ablated repair for degraded ones. | One global format or one global failure story. |
| `undetermined` | Probes disagree or lack power. | Preserve both routes and acquire discriminating evidence. | Choosing the cheaper route as if diagnosed. |

A full-precision activation patch can inject missing computation, so patch recovery alone does not
prove that a small repair is reachable. The prospective test is whether the **budgeted** treatment
predicted by the diagnosis succeeds on held-out cases. Computation collapse categorically forbids a
compensation-only Doctor program.

---

## 6. Representation, repair, and training foundry

Doctor v5 does not choose one quantizer. It compiles sourced operator programs under an exact byte
and claim-scope contract. External mechanisms are priors and competitors, never imported evidence.

### 6.1 Representation foundry

Mandatory families include:

- round-to-nearest, GPTQ, AWQ, HQQ, OmniQuant, SpQR, and SqueezeLLM scalar/sparse controls;
- incoherence/lattice, additive, vector, and product-codebook representations;
- learned rotations, Hadamard/Kronecker transforms, activation scaling, channel reorder, whitening,
  outlier splitting, and residual-subspace transforms;
- nested/progressive representations and physical prefixes;
- binary/ternary bases, low-rank binary factors, binary patterns, magnitude envelopes, structured
  binary regions, and pattern codebooks;
- entropy-coded indices, factors, codebooks, scales, and tiles;
- mixed/critical-layer precision and sparse exceptions;
- lexical-boundary representations for embeddings and LM heads; and
- MoE shared bases, cross-expert dictionaries, expert-delta codes, and router protection.

Hawking-original hypotheses remain explicitly unimplemented until proven: shared-parameter grammar,
cross-layer dictionary, binary latent program, rate-distortion trellis, cross-expert grammar,
entropy-coded compute tiles, and on-demand weight synthesis. Their presence in the registry is not an
implementation claim.

Every physical rate retains four meta-controls: untreated same-rate, scalar equal-byte, smaller
higher-bit equal-byte, and best public same-byte. At or below two bits, both progressive inheritance
and fresh representation reset are mandatory. The 0.10-bpw point is a destructive stress control
unless it independently earns capability evidence.

### 6.2 Structural reconstruction

Reconstruction choices include:

- representation reset;
- block-output and shard-transactional matching;
- alternating quantized-base/low-rank fitting;
- residual SVD, additive residual codebooks, and sparse-plus-low-rank correction;
- progressive binary-factor reconstruction;
- module-adaptive residual strength;
- causal capability islands and cross-expert dictionaries; and
- repairability-aware fitting that shapes error into a reachable correction subspace.

The last mechanism generalizes the principle in
[ProjQ](https://arxiv.org/abs/2606.00494): the best condensed error is not necessarily the smallest
weight error but the error most efficiently correctable under the declared artifact budget. This is
a medium-term Hawking proposal, not current measured evidence.

### 6.3 Doctor repair foundry

Static arms include zero repair, bias, variable-rank residuals, sparse exceptions, codebook residuals,
exact-parent distribution repair, capability islands, cross-expert repair, and capability-specific
skill banks. Conditional arms include token/error-syndrome gates, uncertainty-triggered precision,
progressive slices, hot/cold expert repair, causal-island activation, and static/random/zero-gate
controls.

Every repair must answer four causal questions:

1. Did treatment-on fix the targeted held-out failure?
2. Did treatment-off remove the improvement?
3. Did an equal-byte random or static control do as well?
4. Did any protected capability regress beyond its margin?

All correction, router, gate, cache, and optional-bank bytes count. Dynamic paths report installed,
mean, p95, and worst active bytes. Zero treatment remains eligible. Residual strength may be zero or
negative; “more healing” is not assumed better.

### 6.4 Training foundry

Training arms include:

- no-training control;
- codec-native QAT through exact packed/decoder semantics;
- progressive rate curricula and nested-prefix objectives;
- exact-parent CE/KL/JS, top-k/tail, hidden geometry, attention/router, and capability losses;
- active parent-correct/candidate-wrong failure mining;
- multi-capability minimax/CVaR objectives and gradient-conflict projection;
- parent-good replay to prevent forgetting;
- verified reasoning trajectories, mistakes, correction, and backtracking; and
- stronger-teacher elevation only inside `capability_elevation`.

Teacher outputs are split-bound. Early stopping uses selection data only. The BF16 same-treatment
control separates compression recovery from general training uplift. Answer-correct but invalid
reasoning traces are rejected when an executable or formal verifier exists.

### 6.5 Failure foundry and hardening

The active failure foundry mines verified parent/student gaps and oracle-preserving adversarial or
metamorphic mutations. Mutation families are split before generation. The generator never sees
selection, frozen, sealed, or independent data. Hardening covers prompt invariance, calibration,
selective risk, long context, multilingual behavior, structured output, instruction following,
safety/security, and tool choice. A hardening improvement cannot cross claim scopes.

---

## 7. Training Ladder v5

### 7.1 Axes

- **32 source models**, pinned by id and normalized manifest hash;
- **ten physical ceilings:** `4.0, 3.0, 2.0, 1.0, 0.8, 0.55, 0.50, 0.33, 0.25, 0.10 bpw`;
- **four claim scopes** from §3; and
- **eleven proof-work stages** below.

This produces 1,280 research lanes and 14,080 referenced stage cells. A lane is a plan, not a job.

### 7.2 Stages

| Stage | Name | Promotion meaning |
|---|---|---|
| L0 | Evidence quarantine | Parent/config/tokenizer/template, data vaults, contamination scans, margins, and comparator identity are frozen. |
| L1 | Mechanistic disease atlas | Failure is classified with confidence; evaluator artifacts are separated; collapse forbids compensation-only treatment. |
| L2 | Equal-byte representation tournament | All required representation and zero-treatment controls are present; actual metadata-inclusive byte projections pass. |
| L3 | Codec-native reconstruction | Training semantics equal packed semantics; artifact fits; exact-resume replay is identity-valid. |
| L4 | Identity and uplift restoration | Distribution, geometry, and capability objectives obey claim-scope teacher authority; worst domain is reported. |
| L5 | Active failure foundry | Verified failures and held-out mutation families are provenance-safe; forgetting and diversity are measured. |
| L6 | Targeted Doctor synthesis | Smallest causal static/conditional treatment wins; zero treatment remains; all repair bytes are billed. |
| L7 | Verified reasoning restoration | Executable/formal verification, correction/backtracking, on-policy controls, and parent-good replay pass. |
| L8 | Augmentation plane/scope firewall | Standalone lanes prove zero external dependency; augmented lane bills every external mechanism. |
| L9 | Robustness, alignment, calibration hardening | Metamorphic, safety, instruction, and selective-risk gates pass without protected-domain regression. |
| L10 | Sealed champion audit | Five or more independent seeds, Holm-corrected intervals, same-budget competitors, and independent reproduction agree. |

### 7.3 Resource classes

- **Resident parent research:** up to roughly 16B, still subject to measured peak-wave fit.
- **Streamed single-host research:** above 16B through roughly 235B, one bounded block/shard window and
  one heavy-work lease.
- **Frontier out-of-core research:** above 235B, transactional multi-pass shards and deterministic
  global reductions.

Parallel early-tier work is allowed only when each cell has an independent atomic checkpoint and the
measured combined peak stays inside the memory envelope with normal pressure and zero swap. Streamed
and frontier classes have a future parallel cap of one. The ladder itself cannot acquire the lease or
launch a process.

---

## 8. Whole-artifact and residency accounting

For exact source parameter count \(N\):

\[
b_{physical}=\frac{8\,B_{all-in}}{N}.
\]

`B_all-in` is the sum of actual standalone shipped file lengths—not estimated tensor payload. It
includes quantized payload, pass-through tensors, embeddings, head, norms, scales/zero points,
codebooks, dictionaries, indices, masks, outliers, transforms, residuals, adapters, Doctor modules,
routers, drafters, verifiers, tokenizer, chat template, metadata, state, and decoder-required
alignment/padding. Retrieval indexes, auxiliary models, and persistent external state are zero for
standalone scopes and included for `augmented_system`.

No bytes may be amortized across hypothetical deployments. No remote or post-install unbilled
download is allowed. No dense parent may be fetched after load.

Every receipt reports:

- actual archive and installed bytes;
- nominal payload and physical whole-artifact bpw;
- decoded steady-state and peak resident bytes by context;
- expected and worst active bytes;
- bytes read per token at mean, p95, and worst case;
- total and active MoE bytes;
- KV, activation, workspace, generated-table, and cache bytes;
- creation peak RSS, scratch, and checkpoint bytes; and
- external runtime bytes/calls where authorized.

The shared Studio policy reserves approximately 78 GB of the 96 GiB unified memory for weights so
the OS, ChatGPT/Codex control plane, KV, activations, and workspaces retain headroom. This is a
planning envelope, not proof of fit. Decimal GB and binary GiB are kept distinct. Admission requires
measured normal memory pressure, zero swap, and worst-context peak below the active resource limit.
OOM, timeout, unloadable artifact, or hidden paging counts as failure.

Payload-only arithmetic explains but does not solve the terminal challenge:

| Model size | 0.80 bpw | 0.50 bpw | 0.33 bpw | 0.25 bpw | 0.10 bpw |
|---:|---:|---:|---:|---:|---:|
| 120B | 12.0 GB | 7.5 GB | 4.95 GB | 3.75 GB | 1.5 GB |
| 405B | 40.5 GB | 25.3 GB | 16.7 GB | 12.7 GB | 5.1 GB |
| 671B | 67.1 GB | 41.9 GB | 27.7 GB | 21.0 GB | 8.4 GB |
| 1.6T | 160 GB | 100 GB | 66 GB | 50 GB | 20 GB |

These figures exclude every overhead above. No cited work demonstrates broad modern-model quality at
the whole-artifact rate a 1.6T resident model would require.

---

## 9. Quality Battery v5

The battery covers twelve protected domains:

`language_modeling`, `knowledge`, `reasoning`, `mathematics`, `science`, `coding`,
`instruction_following`, `long_context`, `multilingual`, `tool_use`, `calibration`, and
`safety_security`.

### 9.1 Suites and roles

| Suite | Role | Primary purpose and statistical cluster |
|---|---|---|
| Fresh document NLL | Sealed primary | Post-cutoff document NLL, parent KL/JS, calibration, rare-token/tail loss; cluster by document. |
| [LiveBench](https://arxiv.org/abs/2406.19314) frozen snapshot | Public development | Broad ground-truth development diagnostic; public exposure prevents it from carrying sealed proof alone. |
| [LiveCodeBench](https://arxiv.org/abs/2403.07974) post-cutoff | Frozen primary | Code execution with identical sandbox, tests, samples, and token budget; cluster by task. |
| [ARC-AGI-2](https://arxiv.org/abs/2505.11831) | Frozen primary | Exact grid reasoning under matched attempts/compute. |
| [IFEval](https://arxiv.org/abs/2311.07911) | Public development | Verifiable instruction-following checks. |
| [BFCL](https://gorilla.cs.berkeley.edu/leaderboard) v4 frozen | Frozen primary | AST, live, multiturn, and hallucination tool-use checks; cluster by interaction. |
| [RULER](https://arxiv.org/abs/2404.06654) curve | Frozen primary | Objective long-context curve at 4k through maximum context. |
| [NoLiMa](https://arxiv.org/abs/2502.05167) curve | Frozen primary | Long-context latent association by context. |
| [LongBench v2](https://arxiv.org/abs/2412.15204) | Secondary diagnostic | Public long-context reasoning/coding transfer diagnostic. |
| [BigCodeBench](https://arxiv.org/abs/2406.15877) | Frozen primary | Instruction-rich code execution; cluster by task. |
| [EvalPlus](https://arxiv.org/abs/2305.01210) | Public development | Augmented code tests under matched samples. |
| [MMLU-ProX](https://arxiv.org/abs/2503.10497) | Frozen primary | Per-language/per-domain knowledge, reasoning, and science. |
| [MMLU-Redux](https://arxiv.org/abs/2406.04127) | Secondary diagnostic | Audited MMLU items; original MMLU is not primary evidence. |
| [GPQA Diamond](https://arxiv.org/abs/2311.12022) | Secondary diagnostic | Exposed science/reasoning diagnostic with prompt variants. |
| [MATH-RoB](https://arxiv.org/abs/2503.04550) | Transfer primary | Robustness transformations with exact answers; cluster base item and mutation family. |
| [LGMT](https://arxiv.org/abs/2605.23965) metamorphics | Transfer primary | Logic-grounded invariance across reasoning, instructions, and safety. |
| Fresh procedural reasoning | Sealed primary | Seed-committed generators with programmatic oracle; generator families held out from Doctor. |
| Fresh multilingual constraints | Sealed primary | Post-cutoff objective constraints plus bilingual blind audit. |
| [Quantization security metamorphics](https://arxiv.org/abs/2605.15152) | Sealed primary | Harmful-compliance, over-refusal, outlier injection, and confidence shifts. |
| Sealed capability tail | Sealed primary | Private post-freeze coverage of all domains; aggregate-only promotion feedback. |
| Independent replication battery | Independent primary | Separate owner, prompts, and environment under the same preregistered metrics. |

SWE-bench Verified, original MMLU, and WikiText-only perplexity are forbidden as primary v5 proof:
the first has a contract-recorded contamination/task-validity concern, the second has known item
quality and saturation issues, and the third cannot establish capability preservation. They may be
diagnostics only when their limitations are explicit.

### 9.2 Data firewall

Ten mutually hash-distinct vaults are required:

`calibration`, `reconstruction_train`, `repair_train`, `treatment_search`, `selection`,
`public_validation`, `shadow`, `frozen_final`, `sealed_final`, and `independent_replication`.

Every vault receives exact-hash, normalized n-gram/MinHash, semantic-neighbor,
paraphrase/translation/mutation, teacher-query-log, and retrieval-index exclusion checks. Teacher
caches are split-bound. Shadow, frozen, sealed, and independent vaults are optimizer-inaccessible.
The sealed set is consumed once per lineage, and the promotion service returns no item labels.

### 9.3 Evaluator red team

- Candidate identities are blinded and presentation order is counterbalanced.
- Parent, candidate, and competitor use identical tokenizer, template, prompts, stopping, sampling,
  context, output limits, attempts, tools, and verifier/retrieval budgets.
- OOM, timeout, parser failure, and invalid output count as failure and are never dropped.
- Objective or executable scorers override judge preference.
- The distillation teacher cannot be the judge.
- Non-objective tasks use multiple unrelated judges plus human adjudication of disagreement.
- Prompt-format sensitivity, verbosity/order bias, refusal versus inability, and invalid-but-correct
  traces are reported.
- Quality-versus-output-token, sample, and external-call curves accompany any test-time-compute claim.

---

## 10. Statistics and dominance

### 10.1 Statistical contract

- Preregister metrics, directions, margins, superiority deltas, exclusions, stopping, and comparator
  set before frozen-final evaluation.
- Use at least **five independent quantization/training seeds** and **five calibration draws**.
- Every protected-domain metric must contain at least **five independent evaluation clusters**;
  correlated tokens, samples, prompt variants, or mutations inside one cluster do not increase `n`.
- Pre-power each primary domain from paired discordance; an underpowered margin is inconclusive.
- Use exact paired McNemar for paired binary outcomes.
- Use paired stratified/hierarchical cluster bootstrap for continuous/generative outcomes.
- Bootstrap language modeling by document, code by task, generation by item and seed, and transfer by
  model family—not by correlated tokens.
- Report 95% intervals and control familywise alpha ≤0.05 with Holm or predeclared closed testing.
  Every parent non-inferiority and competitor-superiority inference must pass both its
  direction-consistent simultaneous interval rule and its preregistered adjusted-alpha rule;
  positive bounds with non-significant adjusted p-values cannot promote a claim.
- Primary summaries are worst-domain normalized retention and capability CVaR; macro average is
  secondary only.

### 10.2 Exact verdicts

For domain \(k\), let \(LCB_k\) be the simultaneous lower confidence bound on candidate-minus-parent
or candidate-minus-competitor performance, oriented so positive is better.

**Parent non-inferiority** requires, for every protected domain:

\[
LCB_k \ge -m_k,
\]

where \(m_k\) is the preregistered non-inferiority margin, and the matching Holm-adjusted one-sided
non-inferiority test must satisfy the preregistered familywise alpha.

**Frontier champion** requires:

1. parent non-inferiority in all domains;
2. positive macro lower bound against every applicable same-budget competitor;
3. no competitor-domain lower bound below that domain's negative margin;
4. at least one primary domain above its preregistered superiority delta against each competitor;
5. every relied-upon parent/competitor inference passes its Holm-adjusted alpha rule;
6. exact data, artifact, protocol, and test-time-compute validity; and
7. independent reproduction with `PROVEN` evidence.

**Uniform quality dominance** is stronger: every protected-domain lower bound is positive against
every required same-budget competitor. The machine `passed` field is true only for this stronger
condition. A frontier champion that is not uniformly dominant must say exactly that.

Any failure emits: **unproven; no unbeatable/dominant claim permitted**.

Claim snapshots expire when the competitor registry, benchmark, artifact hash, prompts/scorers,
runtime semantics, or contamination state changes. Dominance is therefore scoped and dated, never a
permanent universal assertion.

---

## 11. Required competitors

Every applicable comparison uses the same exact parent, no greater actual physical bytes, the same
data and teacher authority inside the claim scope, identical prompts/scorers/test-time compute, the
same augmentation boundary, and a real packed artifact.

The machine registry currently freezes 21 direct implementations across 16 families. Each
program carries the exact rate/model/scope-applicable implementation ids, and a completed
observation must contain exactly that comparator set. One arbitrary comparator, one family
representative, or a reported paper number can never satisfy coverage.

### 11.1 Mandatory controls

- exact full-precision parent;
- untreated same-rate representation;
- zero repair and zero conditional gate;
- scalar equal-byte PTQ;
- smaller higher-bit model at equal or lower bytes;
- best public same-parent same-byte artifact;
- codec-native QAT;
- representation-reset and progressive-inherited arms below two bits;
- BF16 parent with the same treatment outside codec scope; and
- Hawking's prior source-bound champion.

### 11.2 Public mechanism envelope

- preconditioning/transforms: [SmoothQuant](https://arxiv.org/abs/2211.10438),
  [Hadamard/incoherence via QuIP#](https://arxiv.org/abs/2402.04396),
  [learned rotations](https://arxiv.org/abs/2405.16406),
  [orthogonal-Kronecker transformation](https://arxiv.org/abs/2605.00422), and
  [bidirectional channel reordering](https://arxiv.org/abs/2602.17698);
- scalar/mixed PTQ: [GPTQ](https://arxiv.org/abs/2210.17323),
  [AWQ](https://arxiv.org/abs/2306.00978), [OmniQuant](https://arxiv.org/abs/2308.13137),
  [SpQR](https://arxiv.org/abs/2306.03078), [HQQ](https://arxiv.org/abs/2309.15531), and
  [SqueezeLLM](https://arxiv.org/abs/2306.07629);
- lattice/additive/vector: [QuIP#](https://arxiv.org/abs/2402.04396),
  [AQLM](https://arxiv.org/abs/2401.06118), [VPTQ](https://arxiv.org/abs/2409.17066), and
  [TesseraQ](https://arxiv.org/abs/2410.19103);
- QAT/progressive: [ParetoQ](https://arxiv.org/abs/2502.02631),
  [LC-QAT](https://arxiv.org/abs/2606.10531), [MatQuant](https://arxiv.org/abs/2502.06786),
  [MatGPTQ](https://arxiv.org/abs/2602.03537), and [BitStack](https://arxiv.org/abs/2410.23918);
- one/sub-bit: [OneBit](https://arxiv.org/abs/2402.11295),
  [BitNet b1.58](https://arxiv.org/abs/2402.17764), [BiLLM](https://arxiv.org/abs/2402.04291),
  [PB-LLM](https://arxiv.org/abs/2310.00034),
  [BitDistiller](https://aclanthology.org/2024.acl-long.7/),
  [STBLLM](https://arxiv.org/abs/2408.01803), [BTC-LLM](https://arxiv.org/abs/2506.12040),
  [DBF](https://arxiv.org/abs/2505.11076),
  [multi-envelope DBF](https://arxiv.org/abs/2512.24545),
  [LittleBit](https://arxiv.org/abs/2506.13771),
  [LittleBit-2](https://arxiv.org/abs/2603.00042),
  [NanoQuant](https://arxiv.org/abs/2602.06694), and
  [BWLA](https://arxiv.org/abs/2605.00422);
- mixed/structured: [ScaleBITS](https://arxiv.org/abs/2602.17698),
  [CCQ](https://arxiv.org/abs/2507.07145),
  [Bit-by-Bit](https://arxiv.org/abs/2604.07888),
  [Sparse-BitNet](https://arxiv.org/abs/2603.05168),
  [lossless Shannon-bound compression](https://arxiv.org/abs/2606.15789), and the
  [QBB binary-bases paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/05b69cc4c8ff6e24c5de1ecd27223d37-Paper-Conference.pdf);
- repair/conditional: [MARR](https://arxiv.org/abs/2605.17997),
  [SPEAR](https://arxiv.org/abs/2606.11244),
  [ProjQ](https://arxiv.org/abs/2606.00494),
  [intrinsic low-rank quantization repair](https://arxiv.org/abs/2606.01412), and
  [targeted reasoning repair](https://arxiv.org/abs/2501.03035);
- MoE: [QMoE](https://arxiv.org/abs/2310.16795),
  [MoEQuant](https://arxiv.org/abs/2505.03804),
  [KBVQ-MoE](https://arxiv.org/abs/2602.11184), and
  [AlphaQ](https://arxiv.org/abs/2606.04980), with
  [671B-class quantization evaluation](https://arxiv.org/abs/2505.02390) as a scale control; and
- global allocation/search priors: [BAQ](https://arxiv.org/abs/2506.05664) and
  [AMQ](https://arxiv.org/abs/2509.12019).

Reported paper numbers do not satisfy this envelope. Hawking must reproduce the applicable method
under the v5 byte, data, protocol, and statistical contract or label the comparison unreproduced and
narrow the claim.

---

## 12. Proposal horizons and compatibility matrix

Complexity notation: \(P\) source weights, \(P_a\) active weights/token, \(T\) training/calibration
tokens, \(r\) trial rank, \(I\) iterations, \(G\) tensor groups, \(K\) candidate encodings, \(E\)
experts, and \(N\) evaluation items. “Bandwidth reduction” refers to a hypothesis about model-weight
traffic or installed bytes, not a measured result. Speed remains deferred in every row.

### 12.1 Immediate implementation

| Proposal | Theory / complexity | Expected bandwidth and latency | Difficulty | Existing GPU | Apple Silicon | Future specialized hardware | Quantization, speculation, distributed, future-architecture interaction |
|---|---|---|---|---|---|---|---|
| Fail-closed program/artifact/observation control plane | DAG validation \(O(V+E)\); hashing \(O(bytes)\). | No direct reduction; prevents invalid work and hidden bytes. No latency claim. | Medium | Backend-neutral control plane. | Works now as stdlib planning/validation. | Natural firmware/compiler contract. | Binds every codec, drafter, shard, state format, and architecture adapter. |
| Whole-artifact accountant and fit solver | \(O(files+tensor\ groups)\), plus file hashing. | No direct reduction; makes advertised reductions real. | Medium | Generic. | Immediate; required for 96-GiB admission. | Can be enforced by loaders. | Counts quant metadata, speculative models, communication state, MoE total/active bytes. |
| Disease atlas and hard failure router | Statistics \(O(P)\); spectral/curvature probes roughly \(O(IPr+TP_a)\). | No direct reduction; avoids spending traffic on the wrong treatment. | High | CUDA tracing/scaling practical. | CPU/Metal feasible on early tiers; large tiers streamed. | Telemetry/near-memory probes favorable. | Routes all quantizers; determines whether drafter/repair is legal; traces shard deterministically; semantic probes adapt to MoE/SSM/multimodal. |
| Equal-byte representation tournament | Search is combinatorial; explicit controls plus deterministic sampling rather than exhaustive execution. | Candidate-dependent; no reduction until packed bytes exist. | High | Most public controls have GPU paths. | Offline CPU/Metal oracles first; native decoders incomplete. | Compiler-guided format selection natural. | Preserves smaller/higher-bit and reset controls; cells distribute independently; future architectures use semantic tensor groups. |
| Codec-native reconstruction and exact resume | Streamed work \(O(P)\) per pass; training \(O(TP_a)\). | No guaranteed serving reduction; prevents proxy/packed mismatch. | High | Research/QAT practical. | Required bounded-window path; no full-parent residency assumption. | Native packed training could fuse. | Essential for nested prefixes/speculation identity and transactional distributed shards. |
| Static repair tournament | Low-rank fit roughly \(O(IPr)\); sparse selection \(O(P\log k)\). | Adds bytes; wins only if capability gained per byte exceeds base allocation. Latency unknown; sparse gathers may hurt. | Medium-high | Generic GEMV/gather. | CPU oracle feasible; Metal ABI/fusion needed later. | Fused epilogue and sparse engines favorable. | Codec-agnostic; target/drafter semantics hash-bound; tensor repairs shard; applies to new block types. |
| Quality battery, firewall, and statistics | Evaluation \(O(N\cdot test\ time\ compute)\); paired inference and resampling. | No model-traffic reduction; blocks false wins. | High operational rigor | Backend-neutral. | Runs after valid Apple artifact path. | Hardware-independent proof layer. | Equal budgets across quant/spec/tool/distributed systems; architecture-independent domains. |

### 12.2 Medium-term research

| Proposal | Theory / complexity | Expected bandwidth and latency | Difficulty | Existing GPU | Apple Silicon | Future specialized hardware | Quantization, speculation, distributed, future-architecture interaction |
|---|---|---|---|---|---|---|---|
| Entropy-shaped binary latent codec | Alternating binary/low-rank/ITQ fit roughly \(O(IPr)\); codebook clustering nonconvex; rANS \(O(P)\). | Physical payload may enter sub-bit regimes if factor entropy is low; metadata/lexical floor can erase gains. Lookup/entropy decode may increase latency. | Very high | CUDA research feasible. | No codec-native Metal implementation yet. | Bitwise, lookup, near-memory, and entropy units favorable. | A base representation, not post-PTQ polish; nested slices may draft; blocks/experts distribute with deterministic dictionaries; can adapt to non-transformer matrices. |
| Progressive semantic bitplanes | \(J\)-prefix QAT roughly \(O(JTP_a)\), partly shared. | One installed artifact may expose several rates; actual traffic follows selected prefix. Latency/acceptance unknown. | Very high | GPU multi-objective QAT feasible. | Packing/decoder missing. | Hierarchical memory and bit-serial compute favorable. | Every prefix is a trained quantizer; low prefix is only a speculative-drafter hypothesis; distributed checkpoints must preserve prefix identity; useful for adaptive future architectures. |
| Repairability-aware condensation | Alternating representation/projector fit \(O(I(Pr+TP_a))\), nonconvex. | Same base bytes may need fewer correction bytes; empirical only. Conditional repair may lower mean but worsen p95. | Very high | Research implementation feasible. | Oracle first; Metal path absent. | Joint codec/repair hardware attractive. | Co-optimizes quantization and Doctor; speculation distribution may change; correction bases must merge deterministically; applies to MoE/SSM. |
| Capability banks and syndrome routing | Multi-task training \(O(TP_a)\); gating adds classification/decision cost. | Installed bank grows; good gates may reduce mean active correction bytes. p95/miss latency can rise. | Very high | Generic training and gathers. | mmap/prefetch possible; native route/fusion unproven. | Conditional compute/cache hardware favorable. | Must be jointly trained with codec; speculative acceptance becomes part of objective; banks shard by capability; future event-driven models natural. |
| Lexical-boundary Doctor | Product/code fitting approximately \(O(IMCv)\) or task training \(O(TP_a)\). | Removes embedding/head floor if successful; vocabulary lookup indirection may cost latency. | High | Generic embedding/codebook kernels. | Feasible but no measured packed path. | Embedding/associative memory units favorable. | Preserves tied quantized weights and probability tails critical to speculation; vocab partitions distribute; extends to multimodal token tables. |
| Hierarchical MoE Doctor | Shared-basis fitting \(O(IPr)\); calibration/training \(O(TP_a)\); global allocation combinatorial. | Shared bases may cut installed bytes and active expert traffic; cold misses can worsen tail latency. | Very high | CUDA MoE references exist. | Out-of-core expert runtime and Metal kernels absent. | Expert-cache/near-memory fabrics favorable. | Joint router/expert quantization; verifier/drafter must match routing; expert shards distribute naturally; future sparse architectures align. |
| Active failure and verified reasoning foundry | Generation/verification \(O(N\cdot TTC)\); training \(O(TP_a)\). | No direct compression; aims to spend repair bytes on capability-critical failures. | Very high | Accelerator training favorable. | Small-tier/local oracle possible; large training likely external CUDA later. | Verifier and data-generation engines favorable. | Training source determines claim scope; runtime verifier moves to augmented scope; datasets shard with strict provenance; architecture-neutral. |
| Global capability/byte allocator | Multiple-choice knapsack is pseudo-polynomial; Lagrangian/VOI approximation around \(O(GK\log G)\). | Can reduce low-value base/correction traffic; fragmented kernels may erase latency gains. | High | Planner generic; evaluation costly. | Planner immediate, deployment constraints later. | Compiler/hardware co-design ideal. | Joint bits, ranks, codebooks, exceptions, speculative-critical islands, experts, state, and future block types. |

### 12.3 Long-term paradigm shifts

| Proposal | Theory / complexity | Expected bandwidth and latency | Difficulty | Existing GPU | Apple Silicon | Future specialized hardware | Quantization, speculation, distributed, future-architecture interaction |
|---|---|---|---|---|---|---|---|
| Shared parameter grammar / on-demand weight synthesis | Learned program complexity unknown; synthesis roughly proportional to generated active tiles rather than installed dense \(P\). | Could replace stored weights with compact generative state; synthesis/cache misses may dominate latency. | Extreme | Prototype kernels/training possible, no established LLM result. | Unified memory helps caching but not synthesis cost. | Generative/near-memory tile engines ideal. | Replaces conventional quantization; synthesized prefix may draft; grammar shards/distributes; naturally targets future architectures. |
| Nested target/drafter from one artifact | Training roughly multi-prefix \(O(JTP_a)\); verification cost model required. | Avoids a second independent model and may reuse state; accepted-token latency unmeasured. | Extreme | Batched verifier research feasible. | Blocked on exact packed batched-target parity. | Bitplane-aware speculative hardware favorable. | Quantized prefixes co-trained; distributed verification and rollback identity required; extends to recurrent/persistent state. |
| [Retrieval-first](https://arxiv.org/abs/2112.04426) persistent capability plane | Retrieval approximately \(O(\log M)\) plus generation, depending on index. | May reduce core parameter bytes but adds index traffic and tail latency. | Extreme | Generic retrieval/tool stack. | Local SSD/unified memory possible; must bill page faults and index bytes. | Associative memory/storage compute favorable. | Only `augmented_system`; speculation must include retrieval-conditioned target; indexes distribute; future memory-centric models align. |
| Inference OS for locality/event-driven execution | Scheduling/placement problem is online and architecture-dependent. | Goal is move only causally required state; queue/cache misses define p95. | Extreme | Requires runtime/compiler changes. | Apple unified memory is attractive but coherence is not free. | Near-memory fabrics, event engines, neural schedulers favorable. | Manages codec slices, speculative branches, experts, KV/persistent state, distributed placement, and future graph/SSM execution. |
| Native sparse/ternary/sub-bit architecture | Training \(O(TP_a)\); representation learned from initialization rather than conversion. | Potentially fundamental installed/active traffic reduction; actual latency depends on sparse/bitwise utilization. | Extreme | Research training possible; commodity kernels imperfect. | No production Metal path. | Purpose-built sparse/bit-serial hardware favorable. | Quantization becomes architecture; speculation can share native prefixes; model/expert training distributes; future event-driven networks natural. |
| CUDA/distributed semantic backend | Operator-dependent; communication at least proportional to moved shards/state. | Multi-device bandwidth can unlock scale but network traffic and synchronization may dominate latency. | Very high | Natural research path. | Separate from Apple proof. | Fabric-attached memory and collective accelerators favorable. | Same semantic program, distinct source hashes, communication ledger, quant formats, speculation/KV identity, and architecture adapters. |

No bandwidth or latency value in these matrices is a result. Each row states an expected direction and
failure mode so a later runtime ladder can falsify it.

---

## 13. Apple-first proof path; CUDA later

### 13.1 Apple path

Apple Silicon is the first deployment proof because that is the actual Studio target:

1. CPU/bfloat16 oracles on resident early tiers;
2. bounded-window diagnosis, representation, reconstruction, and training for larger parents;
3. exact packed artifact and all-in byte ledger;
4. CPU reference decode and packed round trip;
5. Metal semantic parity with no dense fallback;
6. resident unified-memory load with normal pressure, zero swap, and context-specific peak receipt;
7. full Quality Battery v5; and only later
8. bytes moved, latency, energy, KV, and speculative goodput in a separate runtime campaign.

Unified memory enables flexible placement but does not make movement free. CPU/GPU coherence, page
faults, caches, synchronization, and generated-table expansion must be measured.

### 13.2 CUDA and distributed path

CUDA may later accelerate representation/QAT research and provide a separate deployment backend. It
does not inherit an Apple claim. CUDA/distributed receipts independently bind:

- operator sources and compiler/kernel versions;
- device, driver, topology, and collective identity;
- packed semantic parity;
- physical, resident, network, and synchronization bytes;
- test-time compute and quality battery identity; and
- exact-resume/distributed-merge state.

The semantic Doctor program can be shared; backend evidence cannot.

---

## 14. Exact greenlight boundary

This design pass does **not** greenlight execution. The current compilers are structurally incapable
of launching: every candidate is `launchable=false`, every executor is unwired, programs materialize
in planned mode, and the heavy-work lease is outside their authority.

There are two separate approvals.

### 14.1 Scientific greenlight

The user must explicitly authorize transition from design/proof-system work to implementation and
measurement. A clear authorization is: **“Greenlight the Doctor-v5 quality campaign.”** That permits
implementing audited adapters and preparing executable cells inside this specification. It does not
authorize deleting unrelated models, publishing claims, spending external money, weakening data
firewalls, or launching every ladder cell indiscriminately.

### 14.2 Per-wave operational admission

Even after scientific greenlight, no cell launches until all gates pass:

1. current contract, campaign, battery, and ladder selftests/compile/validate are green;
2. program, parent, config, tokenizer, template, mechanisms, data, teacher, and evaluator hashes are
   concrete—not `required` placeholders;
3. every selected mechanism has an audited, source-hashed adapter and executable program mode;
4. data firewall, comparator registry, non-inferiority margins, test-time compute, and power plan are
   frozen;
5. exact resume has been replay-tested;
6. source license/access, download manifest, disk projection, scratch, output, and checkpoint budgets
   fit the 1-TB device or an explicitly authorized external source lifecycle;
7. current process inventory, heavy-work lease, CPU/GPU use, thermal state, memory pressure, swap, and
   available storage admit the measured wave;
8. predicted peak resident memory includes OS/control plane, parent window, candidate, optimizer,
   KV/activations, workspace, and safety reserve;
9. checkpoint and source-shard boundaries tolerate unplug/restart; and
10. the launch record names the bounded cells and rollback/stop rules.

Downloads and source deletion are separate lifecycle actions. A quality greenlight does not by
itself authorize wiping models to make space. An OOM-risk, swap, critical pressure, disk-floor,
identity, contamination, or checkpoint failure blocks the wave.

### 14.3 What greenlight will not mean

- It will not mean v5 is dominant.
- It will not make a projected sub-bit artifact real.
- It will not waive equal-byte competitors or sealed evaluation.
- It will not permit augmented evidence in a standalone claim.
- It will not turn speed into the current objective.

Greenlight starts the attempt to produce evidence. Only independent-reproduction evidence can finish
the claim.

---

## 15. Primary-source evidence limits

The frontier justifies the breadth of v5 but not its success claims:

- [ParetoQ](https://arxiv.org/abs/2502.02631) supports a representation-learning transition below
  roughly two bits; it does not prove a universal boundary or large sub-bit fidelity.
- [LittleBit](https://arxiv.org/abs/2506.13771),
  [LittleBit-2](https://arxiv.org/abs/2603.00042), and
  [NanoQuant](https://arxiv.org/abs/2602.06694) support learned/PTQ binary factors, geometry, and
  Hessian-aware initialization. Their lowest body-rate results retain large capability gaps, and
  whole-model lexical overhead matters.
- [BTC-LLM](https://arxiv.org/abs/2506.12040) supports binary pattern grammars/codebooks but does not
  establish high-fidelity <0.5 physical bpw.
- [LC-QAT](https://arxiv.org/abs/2606.10531) supports strong differentiable two-bit codebook QAT at
  limited scale; it is not sub-bit or trillion-scale proof.
- [BWLA](https://arxiv.org/abs/2605.00422) supports transform-plus-low-rank one-bit-class weight
  approximation through large models, but reasoning gaps and all-in rates remain material.
- [CCQ](https://arxiv.org/abs/2507.07145) supplies a 671B-scale control near 2.06 bits and shows the
  value of critical-layer protection; the reported artifact is far above the Studio envelope.
- [QMoE](https://arxiv.org/abs/2310.16795) supplies a historical 1.6T sub-bit MoE precedent below
  160GB; it is not a modern broad-generative-quality or 96-GiB result.
- [MoEQuant](https://arxiv.org/abs/2505.03804),
  [KBVQ-MoE](https://arxiv.org/abs/2602.11184), and
  [AlphaQ](https://arxiv.org/abs/2606.04980) support rare-expert calibration, cross-expert bases, and
  global allocation, mostly at two to four bits and much smaller scale.
- [SPEAR](https://arxiv.org/abs/2606.11244) and
  [MARR](https://arxiv.org/abs/2605.17997) support conditional/module-adaptive correction, not the
  proposition that a small repair can reverse sub-bit computation collapse.
- [Shannon-bound compression](https://arxiv.org/abs/2606.15789) supports lossless entropy coding of
  quantized symbols through 405B; symbol entropy is not capability rate-distortion.
- [Sparse-BitNet](https://arxiv.org/abs/2603.05168) supports native ternary+sparse training at small
  scale; it is not a conversion method for existing frontier checkpoints.

The exact open problem remains: no primary source demonstrates broad modern-model capability at
≤0.5 **physical whole-artifact** bpw, and no source demonstrates a high-quality 671B/1.6T modern model
resident within this 96-GiB machine. Doctor v5 is constructed to test that boundary without
pretending it has already been crossed.

---

## 16. Completion and honest claim

### Static v5 package complete

The static Doctor-v5 package is complete when:

- the four claim scopes and failure routes are machine-enforced;
- program, artifact, observation, exact-resume, and dominance schemas validate;
- the campaign, quality battery, and 1,280-lane ladder compile deterministically;
- the root manifest binds those reports, their validators, and the canonical specifications;
- the hostile Audit-C fixtures are rejected and the audit reports only static integrity;
- all mechanism sources, controls, data firewalls, statistics, and greenlight boundaries are explicit;
  and
- every executor remains fail-closed until separately audited and authorized.

This completion state says that the frozen plan and its local validation boundary are coherent. It
does not prove that no future adversary can find a validator defect, and it is not experimental
evidence.

### Evidence complete

A particular claim becomes evidence complete only when:

- a source-bound executable program runs;
- a real packed artifact meets its all-in physical and resident ceilings;
- full-model quality is replicated across required seeds and calibration draws;
- the sealed battery and same-budget competitor envelope pass;
- the exact dominance decision passes; and
- an independent reproduction reaches the same result.

None of those experimental conditions has yet been completed by Doctor v5. Therefore the canonical
statement is:

> **Doctor v5 is a frozen, adversarially hardened design package for attempting to establish
> capability/quality dominance under exact physical-byte constraints. It authorizes no execution
> and has not measured or proven dominance.**
