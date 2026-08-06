# HCLI Verification Authority Plan

**Status:** SCAFFOLD (types + candidate/certify boundary + tests real; live Numeric Parity / held-out ownership wiring gated)  
**Bible:** HAWKING_ASCENSION_BIBLE §22 (Verification authority) + §2 (Human/controller authority — report-only model authority)  
**Code:** `lab/verification_authority.py`  
**Tests:** `lab/tests/test_verification_authority.py`  
**Programme gate:** Future work, gated on Proto-Frankenstein offload — **plan + scaffold only**.

---

## Intent

Make the session’s proven split — **models propose, protected controller certifies** — a standing subsystem rather than a per-campaign habit.

Sandbox models produce **reports**, not **verdicts**. Every finished evaluation is classified as exactly one §22 class. Agent failure must never be silently relabeled as mechanism failure, and incomplete evidence must never become a fabricated `PROMOTED` / `ACCEPT`.

---

## Existing patterns reused

| Pattern | Where | Generalization here |
|---------|--------|---------------------|
| `evaluate_promotion` → ACCEPT / REJECT / **PENDING** | `lab/operators/frankenstein_promotion_gate.py` | `map_promotion_gate_verdict` + `classify_check_aggregate`; PENDING → `INSUFFICIENT_EVIDENCE` (never fabricate ACCEPT) |
| Independent verify / seal | `frankenstein_v0_seal.py` `verify` | Controller certifies; builder lane cannot self-seal promotion |
| Author ≠ admitter | `lab/rules.apply_governance` | `SelfPromotionError` if sandbox certifies or signs |
| Builder ≠ challenger ≠ verifier | `lab/receipts.GateEvidence` | `AuthoritativeVerdict.certified_by` ∈ {protected_controller, human_operator} |
| Forbidden complete labels | `frankenstein_gates.FORBIDDEN_PROTO_COMPLETE_LABELS` | `ForbiddenAuthoritativeClaim` enum (§2) |
| Claim boundary on sealed docs | promotion gate `claim_boundary` | Candidate reports carry `claim_boundary.authoritative=False` |
| Grok orchestration | session practice | Executor emits candidates; Claude/controller reviews; nothing self-promotes |
| Appendix counter authority | `appendix_counter_authority_registry.json` | Roster of allowed certifier/signer identities (future bind) |

**Do not edit** live frankenstein operators while GLM recapture runs; this module is the generalized layer those operators can later call.

---

## Report-only model authority (bible §2)

### Models may emit (`CandidateKind`)

```text
candidate_mechanism
implementation_receipt
parity_evidence
capability_evidence
benchmark_evidence
review_objection
known_limitation
recommended_next_experiment
repetition_fingerprint
ROADBLOCK_CANDIDATE
TG_RUNG_CANDIDATE
```

### Models may **not** authoritatively declare (`ForbiddenAuthoritativeClaim`)

```text
PROMOTED
COMPLETE
PHYSICAL_LIMIT_REACHED
FAMILY_EXHAUSTED
SAFE_TO_DELETE
FINAL_VERDICT
```

### Who holds final authority

```text
protected Hawking controller
human operator
protected held-out tests
formal/exact verification
receipt authority
```

---

## Result classification (bible §22)

Every result is **one** of:

| Class | Meaning |
|-------|---------|
| `PROMOTED_MECHANISM` | Mechanism accepted under frozen targets + independent challenge |
| `REJECTED_MECHANISM` | Mechanism failed on its merits |
| `TOOL_DEFECT` | Harness/tool broke; not a mechanism reject |
| `PLANNING_DEFECT` | Plan/spec was wrong; replan, don’t bury the mechanism |
| `VERIFIER_DEFECT` | Verifier/oracle bug; do not trust scores |
| `ENVIRONMENT_DEFECT` | Host/GPU/lease/storage environment fault |
| `INSUFFICIENT_EVIDENCE` | Honest PENDING — missing corpus/scores/runtime |

This separation is the point of §22: **agent failure ≠ tool failure ≠ environment failure**.

---

## Candidate vs authoritative

```text
Sandbox model
    │  emit_candidate()  →  CandidateReport (authority_level=candidate)
    │                       sealed, claim_boundary forbids self-promotion
    ▼
Protected controller / human
    │  certify()         →  AuthoritativeVerdict (authority_level=authoritative)
    │  sign_receipt()    →  signed envelope (models cannot)
    │  promote_mechanism() → PROMOTED_MECHANISM only with candidate seal chain
    ▼
§22 classification becomes terminal for that campaign step
```

| API | Sandbox model | Protected controller / human |
|-----|---------------|------------------------------|
| `emit_candidate` | Yes | Yes (notes only) |
| `certify` | **No** → `SelfPromotionError` | Yes |
| `sign_receipt` | **No** → `SelfPromotionError` | Yes |
| `promote_mechanism` | **No** | Yes (requires ≥1 candidate seal) |

Ingestion guard: `reject_self_declared_promotion` fails closed on payloads that claim `PROMOTED_MECHANISM` without a protected certifier.

---

## Bridge from frankenstein promotion gate

| Gate verdict | §22 class (via `map_promotion_gate_verdict`) |
|--------------|-----------------------------------------------|
| `ACCEPT` | `PROMOTED_MECHANISM` |
| `REJECT` | `REJECTED_MECHANISM` |
| `PENDING` | `INSUFFICIENT_EVIDENCE` |

`classify_check_aggregate(statuses)` mirrors the gate’s FAIL / all-PASS / else-PENDING rule:

- any `FAIL` → `REJECTED_MECHANISM`
- all `PASS` → `PROMOTED_MECHANISM` (**still requires `certify()` to be authoritative**)
- otherwise → `INSUFFICIENT_EVIDENCE`

Mapping is pure; **certification** remains a separate protected step. A model that computes “all PASS” still only emits a candidate.

---

## Protected controller ownership (bible §22)

The protected controller owns (not scaffolded as live services here — ownership boundary only):

```text
Numeric Parity V2.1
held-out prompts
capability suites
benchmark contracts
receipt signing
promotion
rollback
storage deletion authority
terminal state certification
```

Sandbox policy (`lab/execution_sandbox.py`) denies model mutations of the filesystem surfaces that implement these (oracle, held-out, thresholds, stable artifacts, credentials, self-merge, self-sign).

---

## ROADBLOCK / TG_RUNG candidates

When iterations repeat the same bottleneck:

```text
ROADBLOCK_CANDIDATE
  repeated mechanism class
  unchanged bottleneck
  same failure signature
  number of repetitions
  materially distinct attempts already made
  smallest next representation or architecture change
```

Models emit this as a **candidate**. The human or protected controller decides: continue / new mechanism class / rotate models / freeze family / promote.

`TG_RUNG_CANDIDATE` follows the same rule: candidate only until certified.

---

## Implementation stages

| Stage | Deliverable | Gate |
|-------|-------------|------|
| **V0** | Enums, candidate/certify API, self-promotion rejects, tests | **DONE (this scaffold)** |
| **V1** | Optional thin adapter: `frankenstein_promotion_gate.evaluate_promotion` → candidate + controller certify hook | After frankenstein lane quiet |
| **V2** | Bind certifier roster to appendix / owner key material | Key ceremony |
| **V3** | Numeric Parity V2.1 + held-out prompt stores behind controller-only paths | Evidence programme |
| **V4** | Terminal-state certification + storage deletion authority as explicit APIs | Reclaim safety |
| **V5** | Defect taxonomy telemetry (tool vs planning vs verifier vs environment rates) | Ops dashboards |

---

## Acceptance criteria

1. §22 classification enum is closed and complete (unit-tested).
2. Sandbox model **cannot** `certify`, `sign_receipt`, or smuggle forbidden verdict strings in declaration fields.
3. Sandbox model **can** emit `ROADBLOCK_CANDIDATE` / `TG_RUNG_CANDIDATE` with `authority_level=candidate`.
4. Controller promotion requires a candidate seal chain (no free-floating `PROMOTED_MECHANISM`).
5. PENDING/partial checks map to `INSUFFICIENT_EVIDENCE`, never fabricated promotion.
6. Defect classes exist so tool/environment failures are not auto-buried as mechanism rejects.

---

## Non-goals (this scaffold)

- No live held-out corpus or Numeric Parity runs
- No edits to `lab/operators/frankenstein_*` or frankenstein evidence trees
- No cryptographic owner-key signing yet (seal_sha256 integrity only, same as lab receipts)
- No push/PR, no remote, no daemon

---

## Related

- `workspace/docs/plans/ascension/HCLI_EXECUTION_SANDBOX_PLAN.md` — what models may *do*
- `lab/execution_sandbox.py` — filesystem / action enforcement
- `lab/operators/frankenstein_promotion_gate.py` — reference pattern (read-only for this programme)
- Bible §2, §21, §22
