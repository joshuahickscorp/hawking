# HIDE Agent Kernel Options (Bible Book XI)

Run date: 2026-07-19
Grounding: `HIDE_LIVE_ARCHAEOLOGY.md` (§3.5, §5, §4), `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json`, `HIDE_TWO_SURFACE_ARCHITECTURE.md` §2. Packed-crate claims verified read-only against git `5a99d0e2` with file:line.
Status: design decision plus specification. Every mechanism is tagged by the readiness of the primitive it depends on: real-and-wired / real-but-unwired / partial / stub / missing.
Scope: the inner agent loop (single task) and the outer fleet scheduler (multi-agent durable work). Model-role routing is deferred to `HIDE_LOCAL_MODEL_TOPOLOGY.md`; warm-state forks to `HIDE_STATE_CAPSULE_ABI.md`; typed tools to `HIDE_TOOL_SKILL_PLUGIN_MCP_ABI.md`; oracles-as-verification-plane to `HIDE_CONTEXT_OS_SPEC.md` neighbors.

## 1. The one defect this book fixes

The live turn is a single-shot completion, not an agent loop. Two facade facts pin it:

- **S1 (default kernel is a stub):** `AgentKernel::new` installs `StubPlanner` with an empty oracle suite and `runtime/dispatcher/grounding = None`; a fully wired `KernelBuilder` exists but the host never calls it (`5a99d0e2:crates/hide-kernel/src/lib.rs:119`, builder at `:196-296`). [VERIFIED REPO]
- **S2 (the turn is 256 tokens of raw prompt):** `SubmitTurn` sends `messages: Vec::new()`, `max_output_tokens: 256`, no tool loop, no verify (`5a99d0e2:crates/hide-backend/src/host.rs:852-863`). [VERIFIED REPO]

So HIDE ships a real planner, a real deterministic verifier, a real tool runner, and a real fleet fabric, none of which the live turn touches (`HIDE_LIVE_ARCHAEOLOGY.md` §3.5, master reconciliation rows "Planner-Executor-Verifier loop CONFIRMED as built, REFUTED as wired" and "Tool loop not connected to live turns CONFIRMED"). Book XI is a **reconnection and reshape**, not a greenfield build. The reshape question is what loop shape to reconnect into, and the answer is not "preserve the phase machine because it compiles."

## 2. What is already packed (honest inventory)

The `hide-kernel` crate (~5.7k LOC, @ `5a99d0e2`) carries the highest-value idea in the codebase: a plan-as-data DAG where **every step declares its acceptance oracle up front, before acting** (`plan/schema.rs:3-6` "K1: no state advances on faith"; `plan/planner.rs:2`). That discipline is the thing to keep. What surrounds it is a modest phase driver that should not become mandatory.

| Primitive | Readiness | Evidence (@ `5a99d0e2`) | Keep / reshape |
|---|---|---|---|
| Plan-as-data DAG, acceptance declared per step | real-but-unwired | `plan/schema.rs:3`, `plan/dag.rs:5`, `Acceptance::predicate` on every step | **KEEP (flagship)** |
| `RuntimePlanner` (model synthesizes a real DAG) | real-but-unwired | `plan/planner.rs:51-183`, installed only via `KernelBuilder::runtime` `lib.rs:264` | KEEP, make default |
| `StubPlanner` (single canned step) | real-and-wired (the defect) | `lib.rs:119`, `plan/planner.rs:20-49` | **RETIRE from the live path** |
| `ProcessOracle` build / typecheck / test / lint | real-but-unwired | `verify/deterministic.rs:27-140`, registered `lib.rs:280-286` | KEEP (dominant gate) |
| Verification gate (deterministic first) | real-but-unwired | `verify/gate.rs`, `verify/mod.rs` OracleSuite | KEEP |
| `ConsistencyOracle` (self-consistency vote), `LlmJudgeOracle` | real-but-unwired | `verify/probabilistic.rs:6-19,107` | KEEP as tie-break only |
| Phase driver (`Plan/Observe/Verify/Replan/Repair/Paused/Done/Aborted`) | real-but-unwired | `machine/driver.rs:66-93` | **RESHAPE, do not mandate** |
| Governor (interrupts, budget, autonomy gate) | real-but-unwired | `govern.rs:174-263`, driver gate `driver.rs:51` | KEEP |
| `localized_replan` / `supersede` | real-but-unwired | `plan/replan.rs` | KEEP |
| Checkpoint / subagent / skills seams | real-but-unwired | `checkpoint.rs`, `subagent/mod.rs`, `skills/mod.rs` | KEEP |
| Typed tool runner (parse + dispatch) | real-but-unwired | `tools/parse.rs`, `tools/runner.rs`; applier in `hide-tools` | KEEP, wire to serve |
| Fleet fabric (jobs, worktrees, footprint merge) | real-but-unwired (not HTTP-reachable) | `crates/hide-fleet/src/*` | RESHAPE (heavy) |

**Note on the phase driver.** It already gates every transition through the Governor and is only seven or eight phases, not a twelve-stage march (`machine/driver.rs:66-93`). But a mandatory phase enum forces every task, including a one-line typo fix, through the same states, and it encodes cognition as control flow. The archaeology's own verdict is that this is the wrong default ("Rigid FSM possibly wrong for open-ended coding: SUPPORTED", master reconciliation). The frontier direction (dossier §5.8) is a flatter loop with the phase machinery demoted to internal helpers.

## 3. Inner-loop options compared

Three candidate shapes for the single-task inner loop.

| Axis | (A) Rigid Planner-Executor-Verifier FSM | (B) Flat ReAct / act-observe loop | (C) Dynamic hierarchical loop |
|---|---|---|---|
| Control flow | mandatory phase march per task | one repeated step, phases are optional internal artifacts | manager spawns typed sub-loops per subgoal |
| Trivial task cost | pays full FSM traversal | minimal (observe, one action, verify, stop) | pays manager overhead |
| Open-ended coding fit | brittle; real work is not phase-linear | strong; matches Codex-style repeated output/exec/observe (dossier §5.8) | strong but heavier |
| Plan handling | plan is the state machine | plan is a data artifact inside the loop | plan per level |
| Verification | end-of-phase gate | evidence re-enters the loop continuously | per-level gate |
| Failure recovery | replan resets the machine | replan mutates the plan artifact, loop continues | escalate to parent |
| Reuse of packed assets | keep driver as-is (wrong default) | keep DAG + oracles + governor, drop mandatory phases | keep DAG, add a layer |
| Risk | over-structured; the S1/S2 shape dressed up | under-structured if evidence discipline is weak | premature complexity for single-box |
| Precedent | classic BDI/FSM agents | OpenAI Codex loop, mini-SWE-agent, Anthropic migration loop (dossier §5.8, §5.12) | multi-agent frameworks |

**Recommendation: (B), the flat execution-grounded inner loop.** (A) is rejected as the default because preserving a phase FSM merely because `machine/driver.rs` compiles is exactly the "keep it because it exists" trap Book XI must avoid; the dossier explicitly says "do not encode every cognitive phase as a mandatory kernel state" (§5.8). (C) is deferred: hierarchy belongs in the **outer** fleet scheduler (§9), not inside a single task, because a single local box gains little from nested managers and pays coordination cost. The plan-as-data DAG and the deterministic oracles from (A)'s packed crate are **retained as artifacts inside (B)**, which is where their value actually lives.

## 4. The recommended flat inner loop

```text
loop:
  observe        <- bounded evidence from the last typed action + current task state
  decide         <- model chooses the next typed action (or "stop")
  act            <- invoke ONE typed, effect-labelled tool (transactional where it writes)
  evidence       <- receive bounded result; full payload spilled to a content-addressed artifact
  update         <- fold evidence into durable task state (event-log append, not a text blob)
  verify         <- run the step's declared acceptance oracle(s); deterministic gate first
  gate           <- continue | replan | repair | escalate | pause-for-approval | stop
```

Contract, per iteration:

- **One typed action per turn.** The action is schema-versioned and effect-labelled (read / write / network / secret), matching the action-plane contract in `HIDE_TOOL_SKILL_PLUGIN_MCP_ABI.md`. Programmatic fan-out, joins, filtering, and pagination happen inside a single action so the model spends turns on semantic decisions, not plumbing (dossier §4.4).
- **Bounded evidence.** Tool output is capped and the overflow spilled to an artifact handle; failing commands return as data, never as loop-killing errors. This substrate exists: `hide-tools` proc honors `EXEC_NONZERO` so "failing tests and compiler errors are DATA, not an error" (`5a99d0e2:crates/hide-tools/src/proc.rs:4,48`). [VERIFIED REPO]
- **Durable task state, not a transcript.** Each fold appends to the single-writer event log (durable truth plane, `HIDE_TWO_SURFACE_ARCHITECTURE.md` §2); the transcript is a projection. This is what S2's `messages: Vec::new()` throws away today.
- **Verify is a loop step, not a terminal phase.** Oracle receipts re-enter as evidence (dossier §4.5, "Self-judgment is a weak signal, not the accept gate").
- **The Governor gates every iteration first.** Interrupts are polled before any budget or effect check, so Esc / steer / pause is honored at the next boundary (`govern.rs:197-206`), which is the backend that parity `loop.interrupt_and_keep` and `loop.soft_steer` need (both `ui_only` today). Budget is denominated in **irreversible effects, not tokens** (`govern.rs:7`), the correct axis for a no-metering product.

Latency shaping (parity `cost.usage_transparency` inversion, supremacy): the same loop runs at four efforts (Instant / Interactive / Thorough / Background, dossier §4.7). Effort controls files read, tests run, alternative hypotheses, and verification depth, and is a distinct axis from model choice, routed per `HIDE_LOCAL_MODEL_TOPOLOGY.md`.

## 5. Plan-as-data inside the loop (reintegrate, do not mandate a FSM)

The plan is a first-class **artifact the loop reads and mutates**, not the loop's control structure. Reintegrate the packed contract verbatim and change only its status:

- A plan is a DAG of `PlanStep`, each carrying a required `acceptance` field declared **before** the step runs (`plan/schema.rs:3-6`, `:181` `StepKind`). This is the single most valuable idea to REINTEGRATE from the archaeology: the plan commits up front to how each step will be machine-verified.
- Make `RuntimePlanner` the default (it synthesizes a real DAG from the model, `plan/planner.rs:94-183`); **retire `StubPlanner` from the live path** to any non-default fallback, because it is the concrete shape of the S1 defect.
- Plans are revised by `localized_replan` and `supersede` (`plan/replan.rs`), so a failed step edits the artifact and the flat loop continues, rather than resetting a machine.
- Cyclic plans are refused and trigger a replan (`plan/dag.rs`, driver gate `driver.rs:100`). Keep this guard as a pure function the loop calls, not as a phase transition.

What to drop: the requirement that a task **traverse** `Plan -> Observe -> Verify -> Replan -> Done` as states. A trivial task should be able to emit a one-step plan, act, verify, and stop without visiting a phase enum; the `Plan::single_step` helper already models the shape (`plan/schema.rs:28`), it just must not be reachable only through `StubPlanner`.

## 6. Deterministic-first verification

Verification precedence is fixed and non-negotiable: **objective execution evidence dominates model self-judgment.** The packed suite already encodes this two-tier split.

1. **Deterministic gate (must pass first).** `ProcessOracle::{build, typecheck, test, lint}` shell out through the sandboxed dispatcher and treat nonzero exit as structured data (`verify/deterministic.rs:27-140`, registered `lib.rs:280-286`). Patch applies transactionally, project builds/typechecks, targeted tests pass, regression suite does not worsen, lints/security/architecture pass, requested behavior is exercised, diff stays in scope (dossier §4.5). A failing deterministic oracle is authoritative and **cannot be overruled by any model judge** (dossier §5.12 "never let a prompt evaluator overrule a failing deterministic gate").
2. **Probabilistic tie-break (only after the gate is green, only for ambiguous acceptance).** `ConsistencyOracle` (self-consistency vote over K samples, `verify/probabilistic.rs:6-95`) and `LlmJudgeOracle` (`:107`) judge scope and intent for criteria no deterministic oracle can express. They are advisory and are given the exact artifacts and oracle receipts, not prose claims (dossier §5.12).

Supremacy note (gated): the probabilistic judges run as warm local forks at zero marginal cost, so best-of-N judges reduce single-evaluator error (parity `goal.evaluator_loop`, `perm.auto_mode`). This is gated on the fork-exposure build items in `HIDE_STATE_CAPSULE_ABI.md` §8; until they land, the judges are ordinary extra local decodes, which is still free of metered egress but not yet zero-marginal.

## 7. The done signal

The loop stops on exactly one of two conditions, and nothing softer:

- **Objective evidence:** the declared acceptance oracle(s) for every plan step returned a passing `Verdict` through the deterministic gate, and the final diff is within requested scope. This is a receipt, not a feeling.
- **Explicit unresolved boundary:** the loop hit a budget, stall, or repeated-failure limit, or an irreversible effect that requires human approval, and it stops with a **named blocker** (missing oracle, ambiguous acceptance, external dependency, permission gate). Stall detection already exists in the driver (`driver.rs:503` "would only reproduce them, emit run.stalled and replan"). [VERIFIED REPO]

A "done because the model said so" stop is prohibited. This is the backend contract behind parity `goal.evaluator_loop` (objective stop) and the honest-status discipline the package requires. When the boundary is a pending irreversible effect, the stop is a `GovernDecision::Pause` surfaced to the human, not a silent completion (`govern.rs:144`).

## 8. Reproduction-test generation for bugs

For a bug-shaped task, the loop generates a **failing reproduction test first**, then treats that test as the step's acceptance oracle.

- Shape: a bug task synthesizes a `PlanStep` whose `acceptance` is "a new test that reproduces the reported failure exists and currently fails, then passes after the fix." The plan-as-data mechanism to attach a per-step oracle is real (`plan/schema.rs`, §5); the substrate to run the test and read nonzero as data is real (`proc.rs` `EXEC_NONZERO`). [VERIFIED REPO]
- Readiness: **missing as a dedicated primitive.** No packed module authors a red-then-green repro test; this is a build item (dossier Phase 3 item 5, "add test-generation and regression oracles where appropriate"). It is a composition of two real parts (per-step acceptance + failing-test-as-data), not an invention.
- Why it matters: it converts "the agent claims it fixed the bug" into "a test that failed now passes and the regression suite did not worsen," which is the only done signal §7 accepts for a bug.

## 9. Selective review agents

Independent review is spent where expected value is positive, not on every change (dossier §4.6 "adversarial review only where expected value is positive").

| Reviewer | Trigger | Backend | Readiness |
|---|---|---|---|
| Correctness | any change touching load-bearing logic | `LlmJudgeOracle` over the diff + oracle receipts | real-but-unwired (`verify/probabilistic.rs:107`) |
| Security | changes touching effects, secrets, network, config | `hide-security` audit + `perm.rule_engine` taxonomy | packed_unwired (`HIDE_SECURITY_CONSTITUTION.md`) |
| Performance | changes to hot paths / kernels | benchmark oracle over `hawking-bench` | partial (perf harness wired; review agent missing) |
| Adversarial | high-risk or irreversible diffs only | best-of-N judge fork | gated on fork exposure |

Reviewers consume artifacts (diff, test receipt, trace), never hidden autonomy (dossier §5.12 "Review artifacts, not hidden autonomy"). Each reviewer is a subagent with an isolated context and a summary-only return, matching parity `subagents.file_defined`, and routed to an appropriate model role per `HIDE_LOCAL_MODEL_TOPOLOGY.md`.

## 10. The outer durable task DAG (fleet scheduler)

Multi-agent work is an outer **durable job DAG**, distinct from the inner flat loop. This is where hierarchy (option C) legitimately lives. The fabric is packed but heavy and not HTTP-reachable (`HIDE_LIVE_ARCHAEOLOGY.md` §3.5, `hide-fleet` verdict REDESIGN); reshape rather than delete.

Each job carries (dossier §4.6): acceptance criteria, dependency edges, an isolated worktree or write lease, a resource/model/effort policy, current evidence + a state checkpoint, retry/stall policy, an integration owner, and a human-attention state. Packed evidence:

- **Job DAG with fan-out lineage:** `queue.rs:208` (children point at their parent), status lifecycle `Admitted/Running/Merging/...` (`queue.rs:83-111`), crash-recoverable from disk state. [VERIFIED REPO @ `5a99d0e2`]
- **Worktree isolation:** `WorktreeManager` creates `git worktree add -b hide/<run> .hide/wt/<run>` off a shared `.git`, leases ports, prunes on completion; four isolation levels because worktrees alone are insufficient (`isolate.rs:1-24,206`). [VERIFIED REPO]
- **Footprint merge funnel:** `Footprint`/`plan_footprints` group jobs by file set; **non-overlapping footprints merge freely (no conflict by construction), overlapping ones serialize or race through one integration branch** with a conflict ladder (structured -> 3-way -> escalate) (`merge.rs:1-75`). [VERIFIED REPO]

Scheduling policy (dossier §4.6, §7 Phase 4):

- **Parallelize independent read-heavy work** (exploration, doc lookup, test diagnosis, alternative hypotheses); **isolate write-heavy workers**; route everything through **one integration/review path**. Return compact evidence packets, not whole transcripts.
- Retire the heaviness: `hide-fleet` is sized for large fleets, but the single-box need is bounded. Keep the job DAG, worktree leases, footprint merge, and a single integration owner; defer the remote/batch concepts until a measured parallel win exists (dossier §7 Phase 4 exit gate: "parallel work reduces critical-path wall clock on eligible tasks; merge conflicts and duplicate work remain within a defined budget").

Supremacy (gated on `HIDE_STATE_CAPSULE_ABI.md` fork exposure): fan-out is a warm-state fork, not N cold prefills, so best-of-N and agent teams collapse toward the cost of one run instead of Claude Code's ~7x-token / 10x-quota fleet (parity `subagents.fork_worker`, `teams.coordinated`, `session.background_supervisor`). Until the three exposure build items land, fleet workers are independent cold sessions and this is a demo of unwired primitives, stated honestly.

## 11. Autonomous merge constitution

What execution evidence permits which write, and where a human gate is mandatory. Reversibility is the axis, matching the Governor's effect-denominated budget (`govern.rs:7`) and its `Autonomy` levels `FullAuto / SuggestOnly / ReadOnly` with `EffectAuthorization` `Allow / NeedsApproval / Forbidden` (`govern.rs:147-271`). [VERIFIED REPO]

| Action | Reversible? | Auto-permit requires | Human gate |
|---|---|---|---|
| Edit a working-tree file | yes (checkpoint) | deterministic gate green + in scope | no, if `FullAuto` and non-protected path |
| Commit to an **isolated worktree branch** | yes (branch discardable) | gate green + repro/regression not worse | no |
| Push a **worktree branch** to remote | mostly (branch, never `main`) | gate green + branch is `hide/<run>`, never `main`, never force | no, mirrors `session.background_supervisor` (never main, never force-push) |
| Open / update a **draft PR** | yes | gate green + evidence packet attached | no |
| Merge into the integration branch | yes (revert) | non-overlapping footprint, or conflict ladder resolved + gate green | no for non-overlapping; human for escalated conflicts |
| Merge to **`main`** / mark ready | no (shared history) | never automatic | **yes, always** |
| `rm -rf` scope, history rewrite, secret access, external send, purchase, credential entry | no | never | **yes, always; circuit breaker even in bypass** |

Rules, hard:

- **Deterministic evidence is the currency.** No auto-write of any kind occurs without a green deterministic gate (§6). A probabilistic judge can block but never authorize a write past a failing oracle.
- **Irreversible effects always pause** under any non-`FullAuto` autonomy, and destructive/exfiltration/credential actions pause under **every** autonomy including bypass (parity `perm.rule_engine` circuit breaker, `security.sandbox`). These map to the Prohibited/Explicit-permission classes the product must honor.
- **Every autonomous change is reviewable, reproducible, attributable** (dossier §7 Phase 4 exit gate). The evidence packet (diff, oracle receipts, trace) is the unit of review, not a "working" spinner.
- **The trust gate precedes autonomy.** No project config, hook, skill, or MCP definition executes before the folder trust decision (parity `trust.workspace_gate`, `hooks.lifecycle`); autonomy is meaningless until `HIDE_SECURITY_CONSTITUTION.md` containment lands (dossier §7, "Security containment lands before autonomous execution").

## 12. Parity vs supremacy split

| Capability | Parity target (reproduce Claude Code) | Supremacy (gated on) |
|---|---|---|
| Interrupt / steer mid-turn | `loop.interrupt_and_keep`, `loop.soft_steer` via Governor interrupts | zero-latency (no network to cancel) + fork both directions at the interrupt point [state-capsule exposure] |
| Todo / plan artifact | `loop.todo_list` from the plan-as-data DAG | plan survives compaction as a warm capsule [context OS] |
| Objective completion loop | `goal.evaluator_loop` (transcript-only judge) | evaluator as a zero-cost local fork, best-of-N judges [fork exposure] |
| Auto-mode policy gate | `perm.auto_mode` classifier | classifier as a warm fork, egress-off removes the exfiltration class [security + fork] |
| Background fleet | `session.background_supervisor`, `subagents.fork_worker` | no metered quota ceiling; fan-out is a pointer copy [fork exposure] |
| Deterministic verify | `security.sandbox` execution oracles | snapshot workspace to a capsule before a risky run, near-zero rollback [state capsule] |

Every supremacy cell is gated on a named build item and is not claimed as shipping. The parity cells are reconnection work on packed, tested parts.

## 13. Build sequence and readiness ledger

Ordered by the dossier ladder (§7 Phases 1-4); each item names the packed asset it reconnects.

1. **Retire the stub turn.** Replace S2's 256-token single-shot (`host.rs:852-863`) with the flat loop calling `KernelBuilder` (not `AgentKernel::new`), so `RuntimePlanner` + `ProcessOracle` suite + tool runner + Governor are on the live path. [real-but-unwired -> wired]
2. **Wire the deterministic gate to serve** so acceptance oracles actually run the sandboxed build/test (`verify/deterministic.rs` + `hide-tools/proc.rs`). [real-but-unwired]
3. **Demote the phase FSM** to internal helpers; make the flat loop the driver; keep `dag.rs`/`replan.rs`/`govern.rs` as called functions. [reshape]
4. **Reproduction-test generation** for bug tasks (§8). [missing, composition build item]
5. **Selective review agents** behind expected-value triggers (§9), each an isolated subagent. [real-but-unwired + missing perf reviewer]
6. **Outer job DAG on a warm serve** with worktree leases + footprint merge + one integration owner (§10), reshaped down from `hide-fleet`. [real-but-unwired, not HTTP-reachable]
7. **Merge constitution enforcement** through the Governor autonomy levels + the security trust gate (§11). [packed + missing OS enforcement]

Gate before any of this counts: the vertical slice must be reconnected first (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §7, Phase 0/1) and `hawking-eval` reintegrated so completion claims have an active harness (`HIDE_LIVE_ARCHAEOLOGY.md` §6 item 5). No "the loop works" claim is earned until a reproducible end-to-end eval proves it (dossier §8.1).
