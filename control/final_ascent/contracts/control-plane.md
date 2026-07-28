# Implement the Hawking final-ascent control plane

## Goal

In an isolated worktree at current final-ascent HEAD, implement an idempotent,
evidence-derived status publisher for the autonomous final-ascent campaign.

It must create/update exactly these required root artifacts:

- `HAWKING_FINAL_ASCENT_STATUS.md`
- `HAWKING_FINAL_ASCENT_STATUS.json`
- `HAWKING_FINAL_ASCENT_LEDGER.jsonl`
- `HAWKING_FINAL_ASCENT_DEPENDENCY_DAG.json`
- `HAWKING_FINAL_ASCENT_LANE_OWNERSHIP.json`
- `HAWKING_FINAL_ASCENT_CONTINUATION_GOAL.md`
- `HAWKING_FINAL_ASCENT_NEXT_COMMAND.sh`

Prefer a generator under `tools/campaign/` plus tests. Generated status files must
say they are generated and must not be hand-edited.

## Authoritative inputs

Read all current root status/receipt/gate/process/source/lane-owner artifacts, with
special priority to:

- `HAWKING_RESUME_CHECKPOINT.md`
- `HAWKING_RESUME_NEXT_SESSION.sh`
- `HAWKING_PARALLEL_LANE_OWNERSHIP.json`
- `HAWKING_PARALLEL_STATUS.json` and `.md`
- `HAWKING_CONTINUUM_STATUS.json` and `.md`
- `HAWKING_HEAVY_CONTINUATION_STATUS.json`
- `GLM52_GENERATION_B_CAPABILITY_VERDICT.json`
- `GLM52_BYTE_ATTRIBUTION.json`
- `odyssey/launch/SUBSTRATE_CAPABILITY.json`
- live Git/worktree/process/launchd/lease/heartbeat/resource state

The following continuation files named by the new directive are currently absent
from this HEAD and from `git log --all`; the control plane must record absence
honestly instead of inventing their contents:

- `HAWKING_RAMANUJAN_CONTINUUM_CAMPAIGN.md`
- `HAWKING_EVOLUTION_PARALLEL_CONTINUATION.md`
- `HIDE_YOU_PERSONAL_AI_EXTENSION.md`
- `Hawking_Prometheus_Ramanujan_Canonical_Master_Plan_Revision_3.md`

Live evidence overrides stale status snapshots.

## Required semantics

The sole terminal endpoint is `RAMANUJAN_SANDBOX_READY`, currently not reached.
Keep these fences false:

- `ODYSSEY_LAUNCH_AUTHORIZED`
- `RAMANUJAN_RESEARCH_AUTHORIZED`
- `HIDE_KERNEL_TURN`

At minimum model lanes for:

1. live-state/control-plane;
2. capable GLM basis pilot and substrate;
3. base/accelerated runtime;
4. HIDE YOU/CHAT/IDE;
5. Odyssey T0-T7;
6. Math-Frozen;
7. Fabric/Bridge/adapters/CLI/model-vault;
8. Hawking consolidation;
9. separate Ramanujan migration;
10. local formal-system training;
11. search/cognition/governance;
12. Q0-Q6/offline recovery.

Every lane must declare owner, branch/worktree, resource class, inputs, outputs,
forbidden files, tests, promotion gate, dependencies, PID/lease/heartbeat, and
status. Unknown live fields must be `null`/`UNKNOWN`, not guessed.

The dependency DAG must encode the actual critical path: Odyssey and Math-Frozen
cannot promote without a hash-approved capable Math-Preserve-v2; Ramanujan
training and Q1-Q6 cannot promote without the frozen Director; HIDE kernel turn
cannot promote without the capable real provider. Q0 and bounded preparatory
work may be recorded as already achieved where receipts prove it.

The next-command script is read-only by default: it diagnoses/reconciles and
prints exact safe next commands. Any action mode must be explicit, refuse stale
leases, and preserve MOP.

Transitions and ledger appends must be atomic, idempotent, checkpointed, and
resume-testable. Running the publisher twice on unchanged evidence must not add
duplicate ledger transitions.

## Safety

- Do not kill/start/modify any live process or launch agent.
- Do not modify or inspect MOP-owned files/caches beyond process-name avoidance.
- Do not delete/modify teacher capsules, model bodies, artifacts, or negative
  controls.
- Do not flip any launch/research/kernel fence.
- Do not merge main or push.
- Do not call a stale status claim live without checking its evidence.
- Do not include `.serena` files.

## Verification and report

Add deterministic tests for schema completeness, DAG acyclicity, fence
preservation, idempotent ledger behavior, atomic publication, and status refusal
when capability evidence is absent/refused. Run the relevant tests and the
publisher in the worktree, commit intended files, and report files changed,
tests, measured result, uncertainties, and any evidence gap.
