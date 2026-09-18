# Hawking Long-Haul V2 Scaffold Closeout

## Disposition

- Goal: `GOAL-A52AB48501`
- Disposition: `CLOSED_WITH_CARRY_FORWARD`
- Architecture/source/contract coverage: complete enough for handoff
- Full P-phase release qualification: **not complete**
- New WorkUnit admission on this Goal: disabled
- New provider spend in this closeout: `$0`

The V2 scaffold is handed off as a durable architectural baseline. This closeout does not promote P0–P20 to release-complete status and does not begin the deep pass.

## Source baseline

The verified GitHub `main` source baseline is:

- `b6fac000db664f2e0ee0588664973790b1a4588e`
- parent: `5bcdb20a693d2989a74fd0bd3165ea0c1af00d91`

The closeout change set is kept as a reviewable follow-on commit. Existing local `.hawking` state, model/runtime caches, generated planes, session artifacts, and the pre-existing `receipts/future/MODELLAKE_EVENTS.json` edit are not folded into that commit.

## Accepted architectural owners and infrastructure

The following are accepted as scaffold/source/contract coverage for handoff:

- HAWKING/Hawking Goal, WorkUnit, Auto, frontier/refill, dead-letter, and child-isolation ownership.
- Provider-independent mutation authority, writer-scope protection, deterministic mutation preflight, source anchors, Hawking-side mutation compilation, and compact worker/cognition protocol surfaces.
- HTTP semantic status boundaries, including the distinction between transport success and accepted capability/effect.
- Route diversity, Kimi/DeepSeek/GLM evidence, provider-output classification, cost/funnel telemetry, and model-residency boundaries.
- Tunnel protocol and Sandbox S0 scaffolding.
- Gravity, NR, Nova, DARKMATTER, NX, Apple, NVIDIA, FPGA, autotuning, and Odyssey/ModelLake source/contract evidence owners.
- Fast Python CI topology and isolated Rust test-source lanes.

These are handoff infrastructure and evidence surfaces. They are not a claim that every runtime path is physically qualified.

## Phase boundary

P0–P20 have broad source/architecture/contract coverage sufficient for a deep-pass handoff. Accepted read evidence and partial implementation are retained, but no phase is promoted to full release completion in this Goal.

Explicit carry-forward debt:

- canonical autonomous mutation acceptance through the complete read → anchor → mutation → edit → test → Git → acceptance path;
- complete Gravity → NR vertical proof;
- complete DARKMATTER → NX physical proof;
- native runtime admission;
- Apple, NVIDIA, and FPGA hardware qualification;
- autotuning measurements;
- P17 broad codebase collapse and remaining duplicate/compatibility-owner retirement;
- final release qualification.

## Evidence and tests

- Python collection: `3488/3491` collected, with 3 slow hardware/MLX tests deselected.
- Focused acceptance set: `111 passed`.
- Rust topology verification: passed.
- Rust formatting check: passed locally after formatting the tracked Rust sources required by CI.
- `git diff --check`: passed for the handoff change set.
- The full suite reaches the protected evidence boundary at `receipts/sovereign/G008_metabolism.json`. That receipt is not fabricated, bypassed, or weakened; it is a next-goal dependency.

## Mutation funnel and economics

Compact cognition, source anchors, Hawking-side proposal compilation, local stale/no-op/lease preflight, and the mutation circuit-breaker are present. Canonical autonomous mutation acceptance remains `PENDING_DEEP_PASS`; no release claim is made from provider prose or a rejected proposal.

The local Goal telemetry contains internal owner-cost and funnel measurements, but provider billing is the financial authority and the closeout does not claim that internal spend is reconciled to real billed dollars. Existing telemetry records accepted read evidence and accepted-head history; provider-output-unusable and transport failure classes remain contained dead letters rather than root release blockers.

## Science/runtime qualification

| Area | Current bounded state | Required next proof |
| --- | --- | --- |
| Gravity | source-mapped/read evidence and partial runtime seams | Gravity → NR vertical execution proof |
| NR | representation/contract coverage | end-to-end representation proof |
| Nova | compact-agent/worker contract scaffolding | trained/native specialization qualification |
| DARKMATTER | source/contract evidence | DARKMATTER → NX physical proof |
| NX | contract and admission scaffolding | native runtime admission and execution |
| Apple | backend source/contract evidence | Apple hardware qualification |
| NVIDIA | CUDA/source contract evidence | NVIDIA hardware qualification |
| FPGA | source/contract/preboard evidence | FPGA hardware qualification |
| Autotuning | policy/telemetry scaffolding | measured tuning campaign |
| Odyssey/ModelLake | scheduler/source evidence | end-to-end runtime qualification |

## CI and release boundary

The checked-in workflow has no deployment or release-promotion job. The observed GitHub failure was corrected at the source: the Rust format gate now passes locally, and the `hide` lanes use the actual package name `hawking-process-authority` instead of the nonexistent Cargo package `hide-backend`. No release object was created or promoted by this closeout.

No AI co-author or generated-by attribution is added to the repository or commit metadata.

## Protected resources

Preserve `.hawking/`, model/runtime caches, generated planes, session artifacts, Flash/Pulsar/ModelLake protected lanes, and all protected sovereign evidence. In particular, preserve the negative/unsatisfied state of `receipts/sovereign/G008_metabolism.json` until a canonical contract supersedes it with fresh evidence.

## Next goal

Create a new deep-pass Goal from this handoff. Its first gates should be the mutation canary, the protected G008 evidence contract, and the Gravity/NR plus DARKMATTER/NX vertical proofs. Do not resume or refill `GOAL-A52AB48501`.
