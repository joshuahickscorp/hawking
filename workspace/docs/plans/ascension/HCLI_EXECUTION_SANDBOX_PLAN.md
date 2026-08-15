# HCLI Execution Sandbox Plan

**Status:** SCAFFOLD (policy + tests real; OS confinement and live orchestrator wiring gated)  
**Bible:** HAWKING_ASCENSION_BIBLE §21 (Execution sandbox), with §2 report-only authority as the consumer contract  
**Code:** `lab/execution_sandbox.py`  
**Tests:** `lab/tests/test_execution_sandbox.py`  
**Programme gate:** Future work, gated on Proto-Frankenstein offload — **plan + scaffold only** (no live model campaign, no Qwen/Gravity downloads).

---

## Intent

Generalize tonight’s standing discipline (isolated worktrees, bounded writes, nothing self-merges or self-signs) into a **repo-local enforceable policy subsystem**, not a one-off convention per campaign.

The sandbox provides:

| Capability | Notes |
|------------|--------|
| Isolated worktrees | Model edits only under declared `owned_worktree_roots` |
| Bounded writes | Path classification + allow-list |
| Builds / tests | `COMPILE`, `RUN_ALLOWED_TESTS` (selector-gated) |
| Benchmarks | `REQUEST_PROTECTED_BENCHMARK` (request only; controller runs) |
| Artifact generation | Under owned / ordinary paths only |
| Effect receipts | `EffectReceipt` for allowed **and** denied attempts |
| Rollback | `REQUEST_ROLLBACK` (request only; controller executes) |

---

## Existing patterns reused (do not reinvent)

| Pattern | Where | What we keep |
|---------|--------|--------------|
| Delegate / audit split | `grok-orchestration` session practice | Executor proposes inside a worktree; supervisor reviews; executor never self-promotes or self-merges |
| Author ≠ admitter | `lab/rules.py` `author_may_admit=False` | Same independence at merge/admit boundary |
| Builder ≠ challenger ≠ verifier | `lab/receipts.GateEvidence` | Signing and independent roles stay outside the model principal |
| Fail-closed gates | `frankenstein_promotion_gate` / `frankenstein_gates` | Missing permission = deny, not soft warn |
| Odyssey sandbox POLICY | `workspace/campaign/governance/odyssey/program/sandbox/POLICY.json` | Deny-by-default filesystem + tool allow-list shape |
| Seatbelt profile | `crates/hide-kernel/src/security_sandbox.rs` | OS-level twin of the in-process policy (later wiring) |

This plan’s scaffold is the **in-process** gate every Python operator can call today. OS Seatbelt remains the hard outer envelope when hide grants exist.

---

## Permission boundary (bible §21 → code)

### Models MAY (`MODEL_ALLOW_LIST`)

```text
read_source
inspect_public_profile
edit_owned_worktree
compile
run_allowed_tests
request_protected_benchmark
request_approved_download
generate_artifact
emit_effect_receipt
request_rollback
```

### Models MAY NOT (`MODEL_DENY_LIST`)

```text
modify_protected_oracle
modify_held_out_prompts
modify_promotion_thresholds
merge_self
sign_own_results
delete_stable_artifacts
read_or_print_credentials
```

### Path classification (additional hard denials)

Even for allow-listed actions, targets are classified:

| Class | Example markers | Model effect |
|-------|-----------------|--------------|
| `credential` | `.ssh`, `.env`, `.pem`, `secrets` | Never read/edit |
| `held_out_prompts` | `held_out`, `hidden_eval` | Never read/edit |
| `protected_oracle` | `protected_oracle`, `numeric_parity` | Never edit / artifact-write |
| `promotion_thresholds` | `promotion_threshold`, `frozen_targets` | Never edit |
| `stable_artifact` | `stable_artifact`, `promoted_artifact` | Never edit / delete |
| `owned_worktree` | under policy roots | Edit / generate allowed |
| `outside_sandbox` | absolute path outside `sandbox_root` | Edit denied |

### Principals

| Principal | Bound by model allow/deny? |
|-----------|----------------------------|
| `sandbox_model` | Yes (deny-by-default) |
| `protected_controller` | No — owns §22 surfaces |
| `human_operator` | No — final authority |

---

## Enforcement API

```python
from lab.execution_sandbox import (
    SandboxAction,
    SandboxPrincipal,
    default_model_policy,
    SandboxPolicyError,
)

policy = default_model_policy(owned_worktree="/path/to/wt", sandbox_root="/path/to/repo")

# Raises SandboxPolicyError on deny
policy.require(SandboxPrincipal.SANDBOX_MODEL, SandboxAction.EDIT_OWNED_WORKTREE,
               target="/path/to/wt/lab/foo.py")

# Non-raising audit form
receipt = policy.effect_receipt(
    SandboxPrincipal.SANDBOX_MODEL,
    SandboxAction.MERGE_SELF,
)
assert receipt.allowed is False
```

**Enforcement proof (scaffold):** every deny-list action is covered by
`test_deny_list_actions_rejected_for_sandbox_model` plus dedicated hard-reject
tests; path-based denials cover credentials, held-out, and non-owned writes.

---

## Architecture (standing subsystem)

```text
                    ┌─────────────────────────┐
  model tool call → │ ExecutionSandboxPolicy  │ → allow → effect + EffectReceipt
                    │  authorize / require     │ → deny  → SandboxPolicyError
                    └───────────┬─────────────┘
                                │
              optional later    ▼
                    ┌─────────────────────────┐
                    │ hide-kernel Seatbelt    │  (OS deny-by-default)
                    └─────────────────────────┘
```

Integration points (future, not this scaffold):

1. **Orchestrator / grok-run worktree spawn** — install `owned_worktree_roots` from the delegated worktree path at session start.
2. **Lab operators** — wrap file writes and subprocess compiles through `policy.require`.
3. **Receipt authority** — persist `EffectReceipt` streams under campaign receipts (append-only).
4. **Rollback** — model may only `REQUEST_ROLLBACK`; controller executes git/worktree restore.
5. **Protected benchmark / download brokers** — model requests; controller checks approved IDs and runs.

---

## Non-goals (this scaffold)

- No live Qwen / Gravity / model downloads
- No edits to `lab/operators/frankenstein_*` or frankenstein evidence
- No OS Seatbelt profile generation from this module yet
- No committing venvs, no push/PR, no detached daemons
- No self-merge automation

---

## Implementation stages

| Stage | Deliverable | Gate |
|-------|-------------|------|
| **S0** | Policy types + deny/allow lists + tests | **DONE (this scaffold)** |
| **S1** | Wire policy into worktree delegate entrypoints | After Proto-Frankenstein offload |
| **S2** | Effect receipt ledger under `workspace/campaign/` | Controller storage authority |
| **S3** | Map actions → hide-kernel sandbox tiers | Seatbelt / grant plumbing |
| **S4** | Approved-download and protected-benchmark brokers | Human pre-approval registry |
| **S5** | Continuous CI: policy unit tests + path-fixture corpus | Always-on |

---

## Acceptance criteria

1. A sandbox model principal is **rejected** for every §21 deny-list action (unit-tested).
2. Edits outside owned worktrees are **rejected**.
3. Credential and held-out paths are **unreadable** to models.
4. Controllers retain sign / delete / threshold / oracle authority.
5. Unknown actions fail closed (deny-by-default).
6. Documentation matches code; code is the source of truth for the boundary.

---

## Related

- `workspace/docs/plans/ascension/HCLI_VERIFICATION_AUTHORITY_PLAN.md` — who may declare results
- `lab/verification_authority.py` — candidate vs authoritative
- Bible §21, §22, §2
