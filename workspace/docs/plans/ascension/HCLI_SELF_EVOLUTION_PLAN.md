# HCLI Self-Evolution Engine Plan

**Status:** PLAN + SCAFFOLD ONLY — future programme, gated on Proto-Frankenstein offload.  
**Bible:** HAWKING_ASCENSION_BIBLE §23 (Self-evolution engine), with §22 classification.  
**Scaffold:** `lab/hcli/self_evolution.py`  
**Tests:** `lab/tests/test_self_evolution.py`

---

## 1. Purpose

Verified agent trajectories may propose durable self-modifications. The engine
**admits** those proposals only through a multi-stage protected path. It learns
from every accepted *and* rejected experiment. It never rewrites behaviour from
a single noisy result.

This is not online weight training. It is structured admission of:

| Kind | Example |
|------|---------|
| `memory_update` | pinned fact / working-set policy |
| `behavioral_rule` | controller routing or refusal rule |
| `skill_update` | skill body or trigger |
| `tool_wrapper` | bounded wrapper around an existing tool |
| `tool_retirement` | retire a tool with successor/receipt |
| `search_index_update` | reindex / pin corpus slice |
| `new_benchmark` | held-out or CLEAN suite extension |
| `new_anti_pattern` | sealed negative with reopen condition |

---

## 2. Preconditions

| Gate | Requirement |
|------|-------------|
| Proto-Frankenstein | Safely sealed, offloaded, hash-verified, out of active storage envelope |
| Option-C / Agent OS | Logical Option-C roles available (executor may propose; controller admits) |
| Historical traces | ≥ 2 sealed trajectory corpora for replay (see `MIN_REPLAY_TRACES`) |
| Hidden validation | Disjoint train/eval membership surface under controller ownership |
| Negative science | Graveyard laws live (burial ≠ deletion; no free resurrection) |

**Explicit non-goals for this plan revision:** Qwen/Gravity downloads, live model
training, mutation of `lab/operators/frankenstein_*`, writes into `ramanujan/`.

---

## 3. Existing patterns reused

### 3.1 Promotion-gate honesty — `lab/operators/frankenstein_promotion_gate.py`

- Missing evidence → **PENDING**, never fabricated **ACCEPT**.
- Independent challenge required before promotion-class decisions.
- Frozen targets declared before evaluation, not after.

Self-evolution mirrors this: incomplete replay / hidden / compat evidence yields
`PENDING` at `protected_admit`; only full PASS chain may `ACCEPT`.

### 3.2 Graveyard / negative science — `ramanujan/scaffold/core/stores.py` (read-only)

Laws copied into the evolution ledger (new code under `lab/hcli/`, **not** a
write into ramanujan):

1. **Nothing is deleted.** Rejected proposals move to `BURIED` and stay auditable.
2. **Free resurrection is refused.** `revive` requires a *post-burial*
   premise-changing ledger event
   (`new_historical_trace_corpus` | `hidden_validation_redesign` |
   `compatibility_surface_change` | `controller_policy_revision`).
3. **Tribunal separation.** The proposer is never the admitter.

Ramanujan's `Stores.bury` / `Stores.revive` / `GraveyardRefused` are the
reference semantics; the HCLI engine re-implements the laws for proposal objects
so Agent OS does not couple to the math scaffold's claim types.

### 3.3 Sealed negatives — `lab/operators/frontier_controller.py`

Frontier planning already refuses retry of sealed negatives without a specific
reopen condition. Self-evolution anti-patterns and burials are the same idea at
the Agent OS layer: class-level learning only after enough samples
(`MIN_CLASS_SAMPLES_FOR_BIAS = 3`).

---

## 4. Admission pipeline

```text
proposal
  → replay on historical traces   (≥ MIN_REPLAY_TRACES, default 2)
  → hidden validation             (train/eval disjoint; controller-owned)
  → compatibility test            (API / skill / tool surface)
  → protected admission           (controller signs; proposer ≠ admitter)
```

### Stage contracts

| Stage | PASS | FAIL | PENDING |
|-------|------|------|---------|
| Replay | all provided traces PASS | any FAIL → **bury** | &lt;2 traces or missing outcomes |
| Hidden validation | `pass=true` and disjoint | fail or contamination → **bury** | bundle omitted |
| Compat test | `pass=true` | fail → **bury** | report omitted |
| Protected admission | all prior PASS + tribunal OK | N/A (refuses) | incomplete evidence |

### Single-noise fence

- Replay with one trace cannot leave PENDING into ACCEPT.
- Class-level routing bias (`may_bias_routing`) unlocks only after
  `MIN_CLASS_SAMPLES_FOR_BIAS` admitted+buried outcomes of that kind.
- The system **learns from rejects**: buried rows are first-class ledger outcomes.

---

## 5. Interface (scaffold)

```text
lab/hcli/self_evolution.py
  ProposalKind, AdmissionStage
  Proposal, EvolutionLedger, AdmissionPipeline, SelfEvolutionEngine
  GraveyardRefused, EvolutionRefusal
```

Public operations:

| Method | Authority |
|--------|-----------|
| `propose` / `propose_from_trajectory` | executor or controller |
| `run_replay` | replay runner (automated) |
| `run_hidden_validation` | hidden validator (controller-owned surface) |
| `run_compat_test` | compat runner |
| `protected_admit` | **protected controller only** (≠ proposer) |
| `bury` | reviewer or controller |
| `revive` | controller + premise-change seq |
| `learn_summary` | sealed aggregate of accept+reject by kind |

Schemas: `hawking.hcli.self_evolution.v1`, `hawking.hcli.evolution_ledger.v1`.

---

## 6. Phase plan (future, post-scaffold)

### SE.0 — Scaffold (this revision)

- [x] Interface + pure pipeline + graveyard laws  
- [x] Unit tests for happy path, PENDING honesty, free-resurrection refusal, bias fence  
- [x] This plan document  

### SE.1 — Wire ledger persistence

- Append-only JSONL under controller-owned campaign path (not sandbox-writable).  
- Receipt seal on every admit/bury/revive.  
- Do not reuse frankenstein evidence paths.

### SE.2 — Historical trace corpus

- Define trajectory seal format (tool calls, outcomes, receipts).  
- Minimum two independent corpora before any production admission.

### SE.3 — Option-C integration

- Executor may emit proposals only from **verified** trajectories.  
- Reviewer challenges proposal class and hidden-membership.  
- Controller runs `protected_admit` after Option-C decision for mechanism-level
  changes that also touch memory/skills/tools.

### SE.4 — Production bias surface

- Only when `may_bias_routing` is true for a kind, allow soft routing priors.  
- Hard policy still requires fresh admission for each concrete proposal body.

---

## 7. Non-negotiables

- Never fabricate ACCEPT / never promote on PENDING.  
- Never delete buried proposals.  
- Never free-resurrect.  
- Never let proposer admit.  
- Never rewrite global behaviour from one outcome.  
- Never touch live frankenstein operators or ramanujan write paths from this subsystem.

---

## 8. Exit criteria for "engine live"

1. SE.1–SE.3 complete.  
2. At least one buried and one admitted proposal sealed in a real campaign.  
3. Independent audit that graveyard revival without premise change is impossible.  
4. `learn_summary` shows rejects counted alongside admits.
