# Hawking Training Ladder v5 — Capability-First Condensation

**Status:** canonical pre-green-light research and operator plan. Every v5 lane is `planned`, every
executor is unwired, and every lane has `launch_permitted: false`. This document does not authorize
a download, model load, training step, worker process, source deletion, or change to the active
Studio run.

The machine-readable sources are:

- [`doctor_v5_contract.py`](../../tools/condense/doctor_v5_contract.py), the fail-closed program,
  artifact, observation, and dominance contract;
- [`training_ladder_v5.py`](../../tools/condense/training_ladder_v5.py), the deterministic
  execution-free compiler and validator; and
- [`training_ladder_v5.json`](../../reports/condense/training_ladder_v5.json), the current compiled
  plan.

The current materialization has schema `hawking.training_ladder.v5`, contains 32 models, 10 rates,
four claim scopes, 11 stages, 1,280 research lanes, and 14,080 referenced stage cells. Its canonical
`ladder_sha256` is emitted and validated with the JSON rather than duplicated in this document. The
compiler, not a hand-edited count or stale copied hash, remains authoritative.

## 1. Mission and boundary

The v5 objective is the best verified capability and quality at the declared all-in physical byte
budget. It is not throughput, latency, token rate, FLOPS, or benchmark theater. The selection order
is fixed:

1. non-inferiority to the exact parent in every protected capability domain;
2. the strongest worst-domain quality;
3. dominance over every applicable same-budget competitor; and
4. capability per all-in physical byte.

Speed is explicitly deferred. It cannot break a quality tie, cannot support a v5 quality claim, and
cannot be claimed until a later, separately authorized runtime ladder. Velocity++, speculative
decoding, kernel tuning, CUDA/Metal speed comparisons, and accepted-token goodput therefore do not
belong in v5 candidate selection. Resource limits still protect the Studio; deferring speed does not
permit swap, memory-pressure violations, unsafe concurrency, or uncheckpointed work.

There is no wall-clock campaign cutoff. A week- or month-long experiment is acceptable, but every
mutating stage still requires exact resume, durable progress, and a scientific completion or negative
result.

Most importantly, four different statements remain four different scoreboards:

| Claim scope | What it can honestly claim | Training authority | Inference authority |
|---|---|---|---|
| `codec_fidelity` | Quality of the representation/codec itself | Exact parent for reconstruction only | Packed artifact only; no Doctor repair, hardening, retrieval, tool, or verifier |
| `restorative_training` | Damage restored toward the exact parent | Exact identity parent plus truth oracles | One standalone treated artifact; no stronger or external model |
| `capability_elevation` | Verified beyond-parent capability learned during training | Exact parent, provenance-bound stronger teacher, and truth oracles | One standalone artifact; the stronger teacher is absent |
| `augmented_system` | Capability of a fully billed retrieval/tool/verifier system | Exact parent plus truth oracles for external-system outputs; no stronger training teacher | External mechanisms allowed only when declared, measured, and billed |

Codec fidelity is not Doctor restoration. Restoration is not elevation. Elevation is not retrieval.
An augmented result can never be copied into a standalone-model row.

Teacher authority is explicit and exact-keyed. The parent teacher and any permitted elevation
teacher bind identity, revision, role, provenance, output/cache and split manifests, training-only
lifetime, and externally trusted authorization to the immutable program spec. Restoration and
augmentation reject a stronger-teacher slot; elevation cannot launch without one.

## 2. The complete 32-model universe

All ten rates and all four claim scopes are compiled for every model. A scientifically destructive
small-model sub-bit result remains useful as a negative control; a large-model rate that cannot be
resident remains an out-of-core scientific control. Neither is silently removed.

Parameter figures in the tables below are rounded catalogue estimates imported from the legacy
ladder. They are scheduling hints only. Every lane is blocked from physical-bpw evidence until a
source-bound tensor manifest supplies the exact integer parameter count.

### P0 — within-family scale spine

| Model | Approximate catalogue scale | Source class | Research role |
|---|---:|---|---|
| `qwen2.5-0.5b` | 0.5B | resident | destructive floor and fast mechanism falsification |
| `qwen2.5-1.5b` | 1.5B | resident | first nontrivial recovery control |
| `qwen2.5-3b` | 3.0B | resident | missing middle rung and first v5 pilot wave |
| `qwen2.5-7b` | 7.6B | resident | established small-model quality anchor |
| `qwen2.5-14b` | 14.8B | resident | largest resident-source rung in the spine |
| `qwen2.5-32b` | 32.5B | streamed | first mandatory full streamed treatment proof |
| `qwen2.5-72b` | 72.7B | streamed | large dense scaling and representation-transfer test |

The Qwen spine isolates parameter scale while holding the family approximately fixed. Evidence may
change later-lane priority, but a small-model win or loss never substitutes for a larger-model run.

### P1 — cross-family generality

| Family | Models | Parameters | Source classes |
|---|---|---:|---|
| Llama 3 | `llama3.2-1b`, `llama3.2-3b`, `llama3.1-8b`, `llama3.3-70b` | 1.2B, 3.2B, 8.0B, 70.6B | first three resident; 70B streamed |
| Gemma 2 | `gemma2-2b`, `gemma2-9b`, `gemma2-27b` | 2.6B, 9.2B, 27.2B | first two resident; 27B streamed |
| Mistral | `mistral-7b`, `mistral-nemo`, `mistral-small` | 7.2B, 12.2B, 23.6B | first two resident; 24B streamed |
| Phi 3 | `phi3.5-mini`, `phi3-medium` | 3.8B, 14.0B | resident |

These 12 entries prevent a Qwen-specific artifact, tokenizer behavior, layer geometry, or training
quirk from being presented as a general condensation result.

### P2 — MoE and 20B–235B systems

| Model | Total / active parameters | Source class | Research role |
|---|---:|---|---|
| `qwen3-30b-a3b` | 30.5B / 3.3B | streamed | small-active MoE control |
| `qwen3-235b` | 235B / 22B | streamed | largest source in the bounded streamed class |
| `gpt-oss-20b` | 20.9B / 3.6B | streamed | native low-precision MoE architecture control |
| `gpt-oss-120b` | 116.8B / 5.1B | streamed | next practical 100B+ checkpoint |
| `deepseek-v2-lite` | 15.7B / 2.4B | resident | resident MoE adapter and router control |

For MoE models, total parameters determine installed bytes while active parameters inform expected
active-expert bytes. Both total, expected-active, and worst-active byte ledgers are mandatory.

### P3 — frontier and terminal targets

| Model | Total / active parameters | Source class | Role |
|---|---:|---|---|
| `deepseek-v4-flash` | 284B / 13B | frontier sharded | first frontier architecture adapter |
| `405b` | 405B dense | frontier sharded | large dense control |
| `671b` | 671B / 37B | frontier sharded | established large MoE control |
| `glm-5.2` | 753B / 39B | frontier sharded | cross-architecture large MoE control |
| `kimi-k2-instruct` | 1.0T / 32B | frontier sharded | first trillion-parameter source |
| `kimi-k2.6` | 1.1T / 32B | frontier sharded | terminal Kimi capability target |
| `kimi-k2.7-code` | 1.1T / 32B | frontier sharded | code-heavy trillion-parameter control |
| `deepseek-v4-pro` | 1.6T / 49B | frontier sharded | largest terminal remote-shard target |

No frontier row means “downloaded,” “supported,” or “runnable.” Each requires an immutable source
manifest, architecture adapter, transactional shard map, global merge state, and a complete source
and license gate before L0 can pass.

## 3. Ten physical-rate points

The exact ordered ladder is `4, 3, 2, 1, 0.8, 0.55, 0.5, 0.33, 0.25, 0.1` physical bits per source
parameter. The ceiling is all-in, not payload-only.

| Rate | Role | Required interpretation |
|---:|---|---|
| 4.0 | high-precision anchor | Establish strong scalar, codec-QAT, zero, and same-byte controls |
| 3.0 | strong compression control | Test whether ordinary PTQ remains sufficient |
| 2.0 | representation transition | Run scalar and vector/additive arms plus a fresh representation-reset arm |
| 1.0 | binary boundary | Compare inherited progression against factor/pattern and sensitive-branch reconstruction |
| 0.80 | primary sub-bit point | Structural sub-bit treatment, never LoRA-only by default |
| 0.55 | progressive bridge | Test nested refinement and capability recovery per added byte |
| 0.50 | half-bit frontier | Required terminal-fit lane for large models and negative control for small models |
| 0.33 | one-third-bit frontier | Safer nominal trillion-scale resident target before overhead |
| 0.25 | terminal resident candidate | Primary nominal 1.6T fit target; actual bytes remain authoritative |
| 0.10 | destructive stress control | Never presumed viable and never removed merely because early scales fail |

At 4 and 3 bpw, the branches are zero treatment, scalar PTQ, codec-native QAT, and the best
same-byte public control. At 2 bpw, the ladder adds vector/additive representation and an independent
representation reset. At 1 bpw, it requires inherited progression, representation reset, binary
factor/pattern, and a compact sensitive high-precision branch. Below 1 bpw it additionally requires
the shared-parameter-grammar hypothesis. Every side branch must fit under the same all-in ceiling.

## 4. The L0–L10 stage DAG

The canonical dependency chain is linear for admission:

```text
L0 → L1 → L2 → L3 → L4 → L5 → L6 → L7 → L8 → L9 → L10
```

Failure routes can return to an earlier stage or close a branch as a durable negative result. They
cannot jump forward.

The evidence state is independently monotonic:

```text
planned → feasibility → tensor_oracle → block_oracle → shard_oracle
        → full_model_quality → replicated_quality → sealed_final
        → independent_reproduction
```

A later-stage filename, process exit, or benchmark row cannot skip a proof state. All 1,280 v5
lanes remain at `planned` today.

| Stage | Work | Promotion proof | Principal backroute |
|---|---|---|---|
| L0 — evidence quarantine | Freeze parent/config/tokenizer/chat-template identity, data vaults, claim scope, margins, prompts, scorers, and comparator registry | All identities and firewalls complete; sealed data remains hidden | Invalidate lineage |
| L1 — mechanistic disease atlas | Paired parent/candidate failures, early-signal survival, internal geometry, activation patching, and weight patching | Failure class and confidence recorded; capability absence and evaluator error separated | L0, or force L2 representation reset |
| L2 — equal-byte representation tournament | Zero, scalar, vector/additive, inherited, reset, binary, sensitive-branch, and same-byte controls | Mandatory controls retained; no dense fallback; both inherited and reset arms below 2 bits | L1 or documented negative closure |
| L3 — codec-native reconstruction | Block/shard reconstruction, exact decoder training, packed round trip, gradient-stability ablation, exact resume | Actual all-in bytes pass; training and packed semantics match; resume replay is identical | L2 or reject proxy artifact |
| L4 — identity/elevation restoration | Task loss, teacher distribution, causally weighted internal geometry, teacher tribunal | Scope provenance correct; worst-domain lower bound reported | L2/L3 or rebalance conflicting objectives |
| L5 — active failure foundry | Verified parent-correct/candidate-wrong mining, taxonomy, adversarial but oracle-preserving mutations, replay | Truth oracle present; held-out mutation families protected; forgetting measured | L4 or quarantine teacher data |
| L6 — targeted Doctor synthesis | Causal static repair, capability bank, counterfactual treatment gate, intervention and ablation | Zero treatment still eligible; gate misses and collateral damage measured; every byte billed | L2/L5 or replace unsafe gate with static protection |
| L7 — verified reasoning | Executable/formal verification, mistakes, contradiction detection, backtracking, student rollouts, no-RL control | Invalid traces rejected; knowledge, reasoning, and tools separated | L4/L5 or disable reasoning source |
| L8 — augmentation/scope firewall | Prove zero external dependence for the three standalone scopes; build and bill external plane only for `augmented_system` | Closed-book and augmented scores separated; retrieval index passes firewall | L7 or invalidate claim scope |
| L9 — robustness/alignment/calibration | Metamorphic tests, counterfactuals, instruction following, safety, calibration, selective risk | No protected-domain regression; variation and violation bounds pass | L4/L5/L6 or invalidate candidate |
| L10 — sealed champion audit | Five-seed proof, paired per-item scoring, Holm correction, same-budget comparisons, independent reproduction | Parent non-inferior everywhere and competitors uniformly dominated | L2/L4/L5/L9; retain complete negative |

The codec scope passes through all stage identifiers for a complete audit, but L5 mines failures
without training on them, L6 proves that no Doctor repair is attached, L7 audits reasoning without a
reasoning treatment, L8 proves zero external dependence, and L9 audits without adding a hardener.

## 5. Diagnosis and mandatory backrouting

L1 classifies each exact model/rate/representation identity as:

- `no_material_damage`: preserve the zero-treatment result; additional repair receives no free
  credit;
- `signal_degradation`: computation remains intact but noisy, so static or conditional compensation
  may enter L6 in a treatment-authorized scope;
- `computation_collapse`: an early component or representation has stopped performing the required
  computation, so compensation-only work is forbidden and L2 must run structural reconstruction;
- `mixed_failure`: repair structural collapse first, then re-diagnose any residual signal damage;
- `undetermined`: no promotion; expand or repair the diagnostic evidence.

A parent capability absence or evaluator artifact is recorded separately and cannot be counted as
quantization damage. Activation correlation is not enough: a targeted site needs a verified parent→
candidate activation intervention, a source-oriented weight patch where applicable, and treatment
ablation. A token gate is trained on the counterfactual label “did this treatment help?” and reports
severity-weighted false negatives and easy-token overcorrection. If a gate misses severe cases, the
route becomes static protection or the candidate fails.

## 6. Four branches from one immutable codec artifact

For each model and rate, L2/L3 first produce the codec lineage and its zero-treatment control. The
four claim scopes then fork from that immutable, hash-bound point:

```text
packed codec artifact
  ├─ codec_fidelity: audit only; no Doctor or external mechanism
  ├─ restorative_training: exact-parent treatment up to parent parity
  ├─ capability_elevation: stronger-teacher training, standalone inference
  └─ augmented_system: standalone core first, then billed retrieval/tools/verifiers
```

The BF16 parent must also receive the same treatment data and optimization as a control. This
detects improvements caused by ordinary training rather than recovery of compression damage.
Teacher outputs are split-bound. A stronger teacher is allowed only in `capability_elevation` during
standalone-model training; it cannot remain resident at inference. Unverified teacher disagreement
is quarantined. An answer-correct but logically or executably invalid reasoning trace is rejected.

## 7. Capability and quality proof

Every candidate is scored across all 12 Doctor-v5 domains:

`language_modeling`, `knowledge`, `reasoning`, `mathematics`, `science`, `coding`,
`instruction_following`, `long_context`, `multilingual`, `tool_use`, `calibration`, and
`safety_security`.

Perplexity is a diagnostic, not a substitute. An average cannot hide a rare capability failure. The
pre-registered observation contains per-item outputs and, for every domain, parent and candidate
scores, delta, 95% lower/upper confidence bounds, sample count, and a non-inferiority margin.

### Five independent seeds

L10 requires at least five independent training/calibration seeds and generation-seed clusters.
For deterministic codecs, the independent unit becomes calibration draw, ordering, initialization,
and any stochastic reconstruction state rather than five copies of the same bytes. The seeds cannot
share optimizer state, failure replay state, or an adaptive selection decision. A result that cannot
support the pre-registered interval at five seeds is underpowered and therefore inconclusive, not a
win or a loss.

Independently, every protected-domain metric requires at least five genuine evaluation clusters.
Multiple tokens, samples, prompt variants, or mutations from the same base item remain one cluster;
they cannot inflate `n`.

Every relied-upon parent non-inferiority and competitor superiority result must pass both its
direction-consistent simultaneous confidence-bound rule and its Holm-adjusted preregistered alpha.
A positive lower bound paired with a non-significant adjusted p-value is invalid, not a win.

### Matched test-time compute

The parent, candidate, and same-scope competitors use the exact same prompt and scorer plus a frozen
test-time compute protocol. The declared budget includes:

- maximum input tokens;
- maximum output tokens;
- maximum reasoning tokens;
- samples per item;
- temperature;
- timeout;
- verifier retries;
- retrieval calls;
- tool calls; and
- external-model calls.

The canonical battery additionally pins tokenizer/template/runtime identity; truncation and context
packing; every sampling control; stop and response format; tools, retrieval, verifiers, external
models, persistent state, caches, KV precision, speculative decoding, memory/concurrency/network,
and OOM/timeout behavior. Non-augmented scopes receive zero external authority. A candidate may use
less, but a comparator never receives less authority. More hidden reasoning tokens, samples, retries,
calls, or time invalidate the comparison rather than count as model quality.

### Dominance language

A v5 frontier champion must be parent-non-inferior in every domain, beat the frozen same-budget
competitor set in the macro lower confidence bound, avoid a domain-specific non-inferiority failure,
show at least one predeclared strict domain improvement, and reach independent reproduction.

The stronger phrase “uniformly quality-dominant” requires every applicable competitor-domain lower
bound to be strictly positive. Only that state permits the contract's winning verdict. Anything else
is “frontier champion but not uniform quality dominance” or “unproven.” The plan never promises that
an unbeatable result exists; it defines the evidence required before such language is allowed.

## 8. Sealed data firewall

The ten exact split names are:

1. `calibration`
2. `reconstruction_train`
3. `repair_train`
4. `treatment_search`
5. `selection`
6. `public_validation`
7. `shadow`
8. `frozen_final`
9. `sealed_final`
10. `independent_replication`

`shadow`, `frozen_final`, `sealed_final`, and `independent_replication` are optimizer-inaccessible.
Every concrete manifest is hash-distinct. Exact, near-duplicate, and semantic contamination scans
are mandatory. Mutation families are split before generation; selection prompts remain hidden from
the failure foundry; teacher caches are split-bound; retrieval indexes exclude evaluation material;
and sealed final is revealed only after training and comparator selection are frozen.

Candidate labels are blinded, objective/judge-free scoring is preferred, the candidate and parent
receive paired prompts, the comparator registry is frozen before final evaluation, and sealed final
is consumed once per candidate lineage. Holm familywise correction at alpha ≤ 0.05 and 95%
confidence are mandatory. Independent replication has its own hidden split and cannot reuse the
training operator state. A firewall violation invalidates the entire lineage.

## 9. All-in byte fairness and controls

The byte numerator is the sum of actual, hash-verified artifact files and must equal the complete
component ledger. The denominator is the exact integer tensor-parameter count from the source-bound
parameter manifest—not the rounded `params_b` planning estimate.
The exact v5 components are:

`base`, `pass_through`, `scales`, `codebooks`, `indices`, `corrections`, `routers`, `state`,
`metadata`, `alignment`, `tokenizer`, `retrieval_index`, `auxiliary_models`, and
`persistent_external_state`, plus `decoder_runtime`, `runtime_dependencies`, and `context_state`.

The ledger additionally records decoded residency, peak residency by context, expected active bytes,
worst active bytes, and MoE total/expected/worst expert bytes. Payload bpw may be reported, but all-in
bpw is derived from verified files and that exact integer parameter count and is authoritative. A BF16 shadow,
unbilled tokenizer, hidden retrieval index, external verifier model, or persistent cache is not free.
Non-augmented artifacts must contain zero retrieval-index, auxiliary-model, and persistent-external-
state bytes.

Eight controls are mandatory when applicable: exact full-precision parent, untreated same-rate
artifact, zero correction, scalar PTQ, codec-native QAT, representation reset, progressive inherited
training, and the best known same-physical-byte method. Negative results remain in the ledger.

The frozen registry contains 21 direct implementations across 16 families: GPTQ, AWQ, QuIP# E8,
AQLM, LC-QAT, MatQuant, MatGPTQ, BTC-LLM, DBF, multi-envelope DBF, BWLA, QMoE, SPEAR, MARR, LBLLM,
ScaleBITS, Shannon–ANS, NanoQuant, LittleBit, LittleBit-2, and the previous source-bound Hawking
champion. Applicability is method-, rate-, model-class-, and claim-scope-bound in the compiled plan;
for example QMoE is MoE-only, SPEAR is W4-only, and Shannon–ANS is only a lossless packaging wrapper.
Every applicable named implementation—not one family representative—requires a packed same-parent
reproduction at equal-or-lower physical bytes. A genuinely incompatible method needs a signed
incompatibility receipt and a narrowed claim. Every comparison uses identical data authority,
scope, prompts, scorer, teacher budget, full matched test-time-compute tuple, and augmentation
boundary. A paper number is a hypothesis; only a source-bound reproduction is Hawking evidence.

## 10. Source, resource, and exact-resume classes

| Class | Parameter range | Future execution shape | Future concurrency cap |
|---|---:|---|---:|
| resident | >0–16B | Whole parent/candidate treatment when measured peak fits | 3 |
| streamed | >16–235B | Bounded layer/block/shard windows | 1 |
| frontier out-of-core | >235B | Transactional multi-pass remote/local shards | 1 |

Resident parallelism is a cap, not permission. It requires a measured sum-of-peaks wave fit,
independent atomic checkpoints, normal memory pressure, zero swap, and the heavy lease. Streamed and
frontier work serialize. The ladder itself can never acquire that lease or launch any of them.

Exact resume is required before L2–L9 work. A checkpoint binds program identity, operator cursor and
state, optimizer and scheduler, gradient-accumulation phase, microstep, every backend RNG, sampler
cursor, curriculum state, source shard/byte offset, teacher cache, failure replay, partial output
hashes, best-checkpoint identity, and resume-command identity. Atomic replacement, file and parent-
directory fsync, validation-before-resume, and identity replay are mandatory. A convenient adapter
checkpoint is not exact resume.

## 11. Current Studio handoff

The active v1/v2 Studio chain remains the only owner of present work. v5 does not preempt, mutate,
pause, or reinterpret it.

At the current handoff boundary:

- the existing Studio owner holds the heavy-work lease;
- the v2 frontier observer reports `waiting-current-studio`, `invoke_workers: false`, zero worker
  launches, and zero launchable v2 components;
- the verified 14B processing handoff reports `waiting-studio`;
- the 32B/72B/120B representative-shard queue reports `waiting-14b-processing`; and
- terminal frontier work remains planned/blocked behind architecture, source, disk, and evidence
  gates.

Those are mutable status facts whose authoritative files remain under `reports/condense/`. This plan
does not turn them into v5 evidence. Existing v1/v2 outputs may later become historical controls only
if their identities and protocols satisfy the v5 contract; otherwise they are informative prior work.
No v5 lane is running now.

## 12. Exact post-green-light order

A user green light authorizes the following gated sequence. It does not mean “launch 1,280 jobs.”

1. **Freeze the approved plan.** Record the exact Doctor contract, ladder/report hash, model snapshot,
   claim scopes, rate set, capability domains, comparator registry, margins, test-time compute
   protocol, and data manifests.
2. **Let the current owner finish naturally.** Do not kill or replace the v1/v2 Studio process.
   Snapshot its final logs, artifacts, incomplete checkpoints, and negative results; then verify that
   it has released the heavy lease.
3. **Build the v5 execution boundary.** Implement reviewed in-process adapters, an empty-by-default
   allow-list, artifact validation, data-firewall enforcement, exact-resume checkpoints, and a worker
   that refuses every still-planned or unwired program. No heavy work enters before these tests pass.
4. **Complete L0 without training.** Bind all currently available parents and datasets. Models whose
   source, license, tokenizer, architecture, or shard manifest is missing stay at L0; they do not
   block unrelated resident pilots.
5. **Admit the first resident wave: Qwen 0.5B, 1.5B, and 3B.** The future cap is three only after
   measured peak-wave fit, normal pressure, zero swap, AC/thermal gates, independent checkpoints, and
   heavy-lease admission. Otherwise reduce the wave rather than relax safety.
6. **For that wave, run the ordered rate sweep:** 4 → 3 → 2 → 1 → 0.8 → 0.55 → 0.5 → 0.33 →
   0.25 → 0.1. Run L1 diagnosis and the complete L2 equal-byte tournament at every rate. At and below
   2 bits, run both inherited and representation-reset arms even if one looks poor early.
7. **Establish codec evidence before treatment.** Advance surviving and mandatory-negative arms
   through L3. Fork the four claim scopes only from the same immutable packed codec identity. Run
   `codec_fidelity` without Doctor repair before interpreting restoration, elevation, or augmentation.
8. **Run L4–L9 scope work.** Restoration uses only the exact parent; elevation gets a provenance-bound
   teacher only during training; augmentation begins only at L8 after a standalone core result. Keep
   every scope's data, bytes, test compute, and receipts separate.
9. **Replicate before sealing.** Complete at least five independent seeds/calibration draws for each
   finalist and required control. Underpowered candidates remain inconclusive. Freeze the comparator
   set, then consume sealed final once and send finalists to independent replication.
10. **Promote the rest of the Qwen spine:** 7.6B and 14.8B resident, followed by streamed 32.5B and
    72.7B. Transfer evidence changes priority and initialization only; every scale repeats its own
    controls and sealed proof.
11. **Run resident cross-family waves:** Llama 1.2B/3.2B/8B, Gemma 2.6B/9.2B, Mistral 7.2B/12.2B,
    Phi 3.8B/14B, and DeepSeek-V2-Lite 15.7B. Pack waves only from measured peaks; never assume the
    three-job cap fits all combinations.
12. **Run the streamed 20B–235B set serially:** gpt-oss-20B, Mistral-Small-24B, Gemma-2-27B,
    Qwen3-30B-A3B, Llama-70B, gpt-oss-120B, and Qwen3-235B-A22B, integrating the already scheduled
    Qwen-32B/72B sources without changing their v1/v2 handoff.
13. **Open the frontier only after its adapters pass:** DeepSeek-V4-Flash 284B → Llama 405B →
    DeepSeek-V3 671B → GLM-5.2 753B → Kimi-K2-Instruct 1.0T → Kimi-K2.6 1.1T →
    Kimi-K2.7-Code 1.1T → DeepSeek-V4-Pro 1.6T. Each remains transactional, single-heavy-owner,
    out-of-core work. Source release is never automatic and occurs only under a separately approved,
    artifact-bound lifecycle policy.
14. **Close the capability campaign before speed work.** Retain every negative and inconclusive row,
    publish only contract-permitted claim language, and freeze the v5 quality champions. Only then
    design a separate green-lighted speed/runtime ladder around those exact artifacts.

No stage, model, rate, or scope gains authority merely because the preceding item appears in this
list. Every arrow is conditional on its own validator and evidence gate.

## 13. What “done” means

The v5 ladder is complete when all applicable controls and competitor families have valid evidence
or a concrete incompatibility receipt; every promoted candidate has five-seed replicated full-model
quality, packed artifact and all-in byte proof, sealed evaluation, and independent reproduction; and
all negative/inconclusive experiments remain queryable.

Compilation is not completion. A download is not completion. A reconstruction oracle is not a
deployable artifact. Parent parity on average is not all-domain non-inferiority. A frontier champion
is not uniform dominance. And no v5 speed claim exists at all.
