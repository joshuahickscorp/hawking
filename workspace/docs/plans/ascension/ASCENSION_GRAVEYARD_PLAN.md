# Ascension Negative-Science Graveyard Plan

**Status:** plan + scaffold only  
**Bible:** HAWKING_ASCENSION_BIBLE §32  
**Scaffold:** `lab/operators/ascension_graveyard.py`  
**Existing reference (read-only):** `ramanujan/scaffold/core/stores.py` Graveyard  
**Odyssey kinship:** `ramanujan/scaffold/research/odyssey.py` `ResearchBranch.bury` / `reopen`

---

## 1. Purpose

Every proposal checks the Graveyard first.

Record:

```text
mechanism
model/geometry
measured outcome
failure reason
reopen condition
```

Known classes (Bible §32):

```text
prompt-independent collapse
beats-null misuse
median masking
fewer waits but slower wall time
unmeasured GPU claims
circular parity oracle
synthetic activation mismatch
cold-miss masking
storage accumulation
ignored eviction
capability loss after compression
```

Law:

> Do not repeat a failed mechanism without a materially new premise.

---

## 2. Existing vs new

### 2.1 What already exists (reusable semantics)

`ramanujan.scaffold.core.stores.Stores` implements a production Graveyard as one
of seven stores (Problem, Claim, Proof-State, Counterexample, Prior-Art,
Strategy, **Graveyard**):

| Behaviour | Implementation |
|-----------|----------------|
| Nothing deleted | Claims stay in `claims[]`; `in_graveyard=True`; `graveyard()` vs `live_claims()` is retrieval, not forgetting |
| Bury | `bury(claim_id, reason, actor)` → ledger `objection`; sets `graveyard_reason`, `buried_at_seq`; clears `admitted` |
| Free resurrection refused | `revive` without `premise_change_seq` raises `GraveyardRefused` |
| Premise-change gate | Only `verifier_event` or `literature_query` ledger rows **after** burial, not superseded |
| Double bury refused | `GraveyardRefused` if already buried |
| Auditability | `is_auditable` true for buried claims |

Odyssey `ResearchBranch` adds research-shaped fields:

- `reopen_condition: str`
- `bury(reason)` → status BURIED + `failure_reason`
- `reopen(evidence)` requires non-`MODEL_INFERENCE_ONLY` authority basis

Tests proving this: `ramanujan/scaffold/tests/test_ramanujan.py`,
`test_governance_invariants.py`, `test_odyssey_harness.py`.

### 2.2 Why a separate Ascension scaffold anyway

Task contract: **do not touch ramanujan write paths**. Ascension also needs:

1. **Failure-class catalogue** as first-class enum (Bible §32 list)
2. **Mechanism-keyed pre-check** for proposals (`check_proposal`) before research or kernel work
3. Fields named exactly as §32 (`model_geometry`, `measured_outcome`, …) without forcing Claim/Tribunal ceremony for every kernel negative
4. Linkage to `ResearchItem` ids and credential-broker storage failures (`storage_accumulation`, `ignored_eviction`)

### 2.3 Decision

| Layer | Role |
|-------|------|
| **ramanujan Stores.graveyard** | Formal claim burial for mathematical/scientific claims under Tribunal; **read-only reference** for semantics |
| **AscensionGraveyard (new scaffold)** | Operational negative-science store for Gravity/runtime/kernel mechanisms; adapts bury/revive laws; no ramanujan writes |
| **Future integration** | Optional: promote high-severity Ascension burials into ramanujan claims via explicit human-gated import (not automatic) |

Scaffold semantics intentionally mirror ramanujan:

```text
nothing_deleted = True
free_resurrection_refused = True
premise_change_required_to_revive = True
ramanujan_stores_write_paths_untouched = True
```

---

## 3. Type design (scaffolded)

```text
FailureClass          enum — all §32 classes + OTHER
BurialRecord          frozen — full §32 field set + failure_class + links
AscensionGraveyard    bury / revive / active / by_mechanism /
                      by_failure_class / check_proposal / as_dict
ensure_graveyard_checked   raises on REFUSED
```

Schemas:

- `hawking.ascension.graveyard_burial.v1`
- `hawking.ascension.graveyard.v1`

### `check_proposal` outcomes

| Status | Meaning |
|--------|---------|
| `CLEAR` | No active burial for mechanism; `may_proceed=True` |
| `REFUSED` | Active burial(s), no new premise; `may_proceed=False` |
| `PREMISE_REVIEW_REQUIRED` | New premise claimed; **still** `may_proceed=False` — scaffold never auto-clears; human/tribunal must confirm |

This matches ramanujan's refusal of free resurrection and Odyssey's evidence-gated reopen.

---

## 4. Known failure classes as programme law

| Class | Typical symptom | Example reopen condition |
|-------|-----------------|--------------------------|
| prompt_independent_collapse | "reuse activations across prompts" fails held-out | proof of true prompt-independence + remeasure |
| beats_null_misuse | beats null but not the real baseline | matched baseline + CLEAN bench |
| median_masking | median OK, p99 disaster | report p95/p99 + worst-case |
| fewer_waits_but_slower_wall_time | counter metric gaming | wall-time + accepted TPS primary |
| unmeasured_gpu_claims | FLOPS claimed without counter | hardware counter receipt |
| circular_parity_oracle | student graded by itself | independent CPU authority |
| synthetic_activation_mismatch | Gaussian fixtures ≠ real routes | real teacher-forced activations |
| cold_miss_masking | warm-cache only | cold-start protocol |
| storage_accumulation | full sources + intermediates retained | broker lifecycle seal→evict proof |
| ignored_eviction | VERIFIED but body left resident | source-only reclaim receipt |
| capability_loss_after_compression | BPW win, task fail | capability suite before seal |

`storage_accumulation` and `ignored_eviction` couple directly to the credential
broker lifecycle — a failed eviction is a Graveyard event, not a silent disk leak.

---

## 5. Workflow

```text
proposal
  → AscensionGraveyard.check_proposal(mechanism)
  → if REFUSED: stop; surface reopen_conditions
  → if PREMISE_REVIEW_REQUIRED: human gate; optionally revive with premise evidence
  → if CLEAR: run cheapest distinguishing experiment
  → on failure: bury(mechanism, geometry, outcome, reason, reopen, class)
  → on success: ResearchRegistry.record(... ADMIT_*) with related_graveyard_ids if any
```

Credential-broker / stream programmes should bury under `storage_accumulation`
or `ignored_eviction` when floor or reclaim laws are violated in production runs
(future wiring).

---

## 6. Tests

`lab/tests/test_ascension_graveyard.py`:

- §32 class catalogue completeness
- bury + REFUSED on repeat
- CLEAR for unknown mechanism
- new premise → PREMISE_REVIEW_REQUIRED (not auto-clear)
- free resurrection refused
- revive with premise evidence
- double bury refused
- snapshot declares ramanujan kinship + write paths untouched

---

## 7. Remaining work

1. Seed burials from existing Hawking negative-science ledgers / sealed findings
2. Optional read-only import of ramanujan graveyard claim summaries (no writes)
3. Wire `ensure_graveyard_checked` into research-registry `record` and frontier planners
4. Durable campaign evidence path for sealed graveyard snapshots
5. HCLI retrieval: "prior failures for mechanism X"
6. Couple broker eviction failures to automatic burial candidates (human confirm)

---

## 8. Non-goals

- No writes into `ramanujan/` stores
- No automatic revival
- No deletion of burials
- No replacement of ramanujan Tribunal for formal claims — this is the operational mechanism graveyard
