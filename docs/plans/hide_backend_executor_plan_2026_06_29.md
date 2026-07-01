# HIDE — The Second Plan: the Executor engine (backend)

Front-end is built and live-wired; this plan makes the **Executor** real. The thesis (from
[hawking_capability_frontier](hawking_capability_frontier_2026_06_28.md) + the
[audit/research report](hide_ux_audit_and_research_2026_06_29.md)): the model is the floor; the
**pipeline we build on top of it is the product** — the fastest, deepest loop that runs and iterates
continuously, on a machine where tokens are local and free, with 4-5M effective context via `.tq`.

Every phase below lights up a surface the front-end *already has*. We are not designing new UI here; we
are filling the seams with real data.

| Backend deliverable | Lights up (already built) |
|---|---|
| `token_batch` stream | Chat streaming + the RAF governor |
| per-step `projection_patch{turn,plan,tool,diff}` | StateTimeline, PlanCard, ToolChipRow, **DiffReview (Tab-to-apply)**, Radiate |
| `fleet` projection backed by state-forks | FleetView (Fork-&-Try-N) |
| `context.compile` / Project Brain | ContextStack |
| `security_gate` | InlineGate capsule |
| local STT | the Composer mic (no time cap, no egress) |

---

## Phase 0 — Live streaming foundation (unblocks what's already built)

These are the three HIGH-severity backend gaps the audit pinned. Smallest change, biggest unlock: the
existing surfaces stop being scaffolds.

- **Emit + persist `token_batch`.** `crates/hide-backend/src/host.rs` `generate_and_publish` /
  `generate_submit_turn` currently broadcast tokens but the host surfaces them as `custom` audit events,
  not `token_batch`, and they aren't appended to the event log. Emit `UiEventKind::token_batch` and
  `event_log.append(...)` in parallel so streams replay on reconnect (pairs with the monotonic reconnect
  cursor we just hardened in `ipc.ts`).
- **Publish `runtime_status` at boot + on transition.** Today it's only inferred via the
  `runtime.roles.list` connector fallback (`store.ts` connectStore). Emit a `runtime_status` event when
  the supervisor reaches ready and on every state change (a 250ms monitor task in `BackendHost`).
- **Per-step events from the kernel loop.** `crates/hide-kernel` runs the agent but emits no progress.
  Wrap `kernel.step()`: on phase change emit `projection_patch{turn}` (phase), per tool-call emit
  `tool_progress`, and on a proposed edit emit `projection_patch{diff}` in the shape `parseDiff` already
  reads (`surfaces/ide/types.ts`) — that feeds straight into the DiffReview + Tab-to-apply gesture.

**Done when:** a real turn streams tokens into the Chat, the StateTimeline fills with live steps, and an
Executor edit shows up as an inline diff you accept with Tab.

## Phase 1 — The continuous executor loop (the headline)

Build the loop that "runs and runs and iterates." On the kernel, not the model.

- **plan -> execute -> verify -> iterate.** After each execute step, run a **verify connector**
  (build / test / lint over the touched files) and feed the result back into the kernel as the next
  observation. Green advances; red becomes the next task. This **reward-from-tests** signal is what makes
  the loop keep going productively instead of declaring done early.
- **Reflexion memory.** Accumulate a bounded list of self-corrections ("tried X, got Y, switched to Z")
  in the run state; emit as a `projection_patch{learnings}`. (New small FE card later; data first.)
- **Pause gates.** The kernel pauses before risky actions (delete > N files, breaking public API,
  network write) and emits `security_gate` — the InlineGate capsule already blocks + collects approval.
- **Background + interrupt.** A run keeps going while the user edits; `cancel_run` / `steer` intents
  already exist on the wire. Surfaced through Radiate's ambient -> detail -> gate tiers.

**Moat:** M2 (free fleets) — this can iterate all night at zero marginal cost.

## Phase 2 — Fork & Try-N for real (state-fork economics)

The FleetView board is built and seeds optimistic branches. Back it with real forks.

- On `custom("fleet_run",{task,n})`, **memcpy the RWKV-7 recurrent state** (constant ~6-16MB) N times and
  run N divergent branches in parallel — no re-prefill, near-free (M1 + M2).
- Stream each branch as its own `fleet` projection entry (state, step/steps, progress) — FleetView
  already renders these.
- **keep-best** -> `custom("resolve_conflict"|"focus_run")`; diff-compare the survivors.

## Phase 3 — Long-context leverage (the 4-5M `.tq` edge)

"Be expensive with tokens" — design for abundance, measured in the other Hawking chat at ~4-5M effective
where a model is nominally ~1M.

- **Whole-repo-on-boot, cached.** On project open, load the repo (up to the practical `.tq` ceiling) into
  a persistent **stable context layer** and prompt-cache it; the ContextStack head shows "whole project
  loaded - cached" (abundance cue, not a budget meter — the meter stays removed per doctrine).
- **Permalayer + actions.** Split context into a stable read-only layer (the repo + Project Brain, ~90%,
  cached) and a volatile action layer (this turn's tool I/O). Sourcegraph's infinite-context pattern.
- **Project Brain (replaces transcript memory).** A structured, evolving store — architecture notes,
  build/test commands, decisions, conventions — persisted across sessions and surfaced in the
  ContextStack. Wire `context.compile` (`connectors.rs`, currently not driven by the loop) to refresh it
  on phase transitions.
- **Honest caveat:** long context != perfect recall (lost-in-the-middle). Keep `code_index.search` as the
  exact-lookup path; long context is for whole-program reasoning, the index for precise retrieval.

**Copy rule:** never "infinite/perfect memory." Say "your whole project, always loaded, instantly
resumed, never billed twice, never truncated."

## Phase 4 — "Just do it" agency

The motivating failure: ask it to push to GitHub and it bounces you to a login. The Executor should carry
the whole task.

- Real tool/connector execution (git, shell, file, http) behind the kernel, with **M4 grammar-guaranteed
  tool calls** (constrained logits = the call is always valid JSON for the schema).
- Side-effectful actions (push, network write, destructive fs) route through the **pause gate** ->
  InlineGate so the user approves once, in-app, instead of context-switching to a browser login.
- Credentials handled by the OS keychain / a credential-request flow — never typed by the agent
  (respects the prohibited-actions boundary).

## Phase 5 — Local voice

- Wire the Composer's `MediaRecorder` capture (already built, no time cap) to **local Whisper** via the
  engine; transcribe on-device, no network egress (privacy doctrine). Stream partial transcripts back as
  the user speaks.

---

## Wire-contract additions
New/used `UiEventKind`s: confirm `token_batch`, `runtime_status`, `tool_progress`, `security_gate` flow
end-to-end; add `projection_patch` projections `learnings` (Phase 1) and `brain` (Phase 3). New custom
intents already present where possible (`fleet_run`, `resolve_conflict`, `focus_run`, `inline_edit`,
`save_file`); add `verify_run` if the verify connector needs an explicit trigger.

## Sequencing & risk
Phase 0 first (it unblocks everything visible and is low-risk plumbing). Phase 1 is the headline and the
most kernel-heavy. Phases 2-5 are independent and can interleave. Throughout: keep the FE honest — a
surface shows real data or says it's a scaffold; never fake a stream.

## Verification
- Phase 0: a live turn streams tokens; StateTimeline fills; an edit accepts via Tab; reconnect replays.
- Phase 1: a failing test drives a second iteration without user input; a risky action gates.
- Phase 2: `try 5` forks real state (measure: no re-prefill, ~constant fork cost) and the board diverges.
- Phase 3: whole-repo loaded + cached; recall measured against `code_index` on a known-answer probe.
- Each phase gated behind the project's parity/CI discipline before merge to `main`.
