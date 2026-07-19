# HIDE Chat Spec

Run date: 2026-07-19
Grounding: `HIDE_TWO_SURFACE_ARCHITECTURE.md` (the shared core), `HIDE_LIVE_ARCHAEOLOGY.md` §3.2 §3.3 §3.4 §3.5 (live truth with file:line), `HIDE_CLAUDE_CODE_BEHAVIORAL_PARITY_SPEC.json` (parity ids), `HIDE_CLAUDE_CODE_UX_GENOME.md` (loved behaviors + spinner anti-genome), `docs/hide-bible/DESIGN_DOCTRINE.md` Part IV "The Chat" + Part V (interaction qualities, visible components, voice, progress signature).
Status: surface specification. Every component is tagged by the readiness of the primitive it depends on. PARITY (reproduce Claude Code) is separated from SUPREMACY (what a local runtime does better); every supremacy claim names the build item it is gated on.

Readiness key (for Hawking mechanisms): **real-and-wired** (reachable on a shipping path) / **real-but-unwired** (built + tested, no live caller) / **partial** / **stub** / **missing**. Parity `hide_status` labels (`ui_only`, `packed_unwired`, `partial`, `wired`, `absent`) are quoted where they add precision.

## 1. What the Chat surface is

HIDE Chat is the terminal-style agent session: one prompt bar, a streamed transcript of the agent's `gather -> act -> verify` loop, and the two steering gestures that make the autonomy feel like a teammate you can redirect, not a black box you launch. It is the surface a power user lives in (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §6).

The invariant it inherits: Chat is not a product, it is one rendering of a single durable session. The live frontend already proves this shape. `Conversation.tsx` is rendered identically by the full-page Chat surface and the docked/floating Executor from one Zustand store, so they are the same conversation with the same context (`app/src/surfaces/chat/Conversation.tsx:1-6`, VERIFIED REPO). Cross-surface transitions (Chat <-> IDE) are specified in `HIDE_TWO_SURFACE_ARCHITECTURE.md` §4 §5 and not repeated here.

Honest current state (HIDE_LIVE_ARCHAEOLOGY.md §3.4): the Chat shell is genuinely built, polished to the doctrine, and **mock-fed**. `ipc.ts` selects Mock (dev default) vs Live; Live targets `/v1/hide/*` on port 8744, a backend the active tree does not build. So every "real" note below means the React surface is real; the loop behind it is the single-shot 256-token `generate()` (S2, `host.rs:848-863`), and the differentiators are backend-deferred. This spec defines the target and states, per component, exactly what is wired today.

## 2. Required interaction qualities (the felt contract)

From the doctrine's Chat and interaction chapters (`DESIGN_DOCTRINE.md:301-371`) and the loved-experience genome (`HIDE_CLAUDE_CODE_UX_GENOME.md` §1, §5). A Claude Code power user must feel these in the first minute, or the polish-parity gate fails (Genome §9: table-stakes roughness loses even while HIDE wins on cost and state).

| Quality | What it means in Chat | Source |
|---|---|---|
| Legible | Read the whole run at a glance; drill in on demand; never a mystery spinner standing in for real work | doctrine "legible + airy" (`:29`); Genome §5 |
| Steerable | Redirect the instant you see it heading wrong, without losing completed work | Genome §1 (highest love, lowest latency tolerance) |
| Calm | One readable column in a wide still room; no chat-app chrome, no avatars, no budget HUD | `DESIGN_DOCTRINE.md:303` |
| Truthful | The live feed shows the agent's actual moves; the progress signature is layered on top, never a substitute | `DESIGN_DOCTRINE.md:344` |
| Undramatic | Terse telemetry voice; specific, blame-free, never "successfully", no emoji, no em/en/middot | `DESIGN_DOCTRINE.md:360-371` |
| Reversible | Any turn is a checkpoint; steer or rewind is one gesture (rewind axis in `HIDE_IDE_SPEC.md`) | Genome §4 |

## 3. Visible components mapped to current FE reality

Every component below exists in the React tree; the column that matters is what backs it. "Backing readiness" is the readiness of the data/loop that feeds the component, not of the component itself.

| Component | FE reality (file:line) | Backing readiness | Parity id |
|---|---|---|---|
| Message column + streaming cusp | `Conversation.tsx`; ~700px column, leading-edge `--light-soft` cusp, no spinner (`DESIGN_DOCTRINE.md:307-308`) | real-but-unwired (mock stream; live turn is 256-tok single-shot) | loop.* |
| SteerBar (redirect / pause / resume / cancel) | `chat/SteerBar.tsx`; phase dot + `redirect_run`/`pause_run`/`resume_run`/`cancel_run` | "ui_only" (no live turn to steer) | loop.interrupt_and_keep, loop.soft_steer |
| PlanCard (steps, 3-state marks, edit, reorder, approve) | `chat/structure.tsx:28`; `approve_plan` custom intent | "ui_only" (no mock/live emits a plan projection) | perm.plan_mode, loop.todo_list |
| ToolChipRow (collapsed tool rows) | `chat/structure.tsx:207` | "partial" (chips real, data mock) | loop.collapsed_tools |
| DiffChipRow ("file edited" -> hunk review) | `chat/structure.tsx:219`; +/- counts, applied/rejected/stale | "ui_only" (no mock emits diff chips) | ide.two_surface_bridge |
| InlineGate (approval card in the stream) | `chat/structure.tsx:278`; `Approve` uses the `.gate` capsule | "packed_unwired" (GateBook in hide-backend, unwired) | perm.plan_mode, security.sandbox |
| StatusBar (phase / model / transport) | `shell/StatusBar.tsx:46-48`; branch + problems **hardcoded** | "partial" | loop.status_line |
| Context Stack (light well) | `surfaces/ContextStack.tsx`; strata, no budget stratum (`:151`), fork-from-current-state (`:73`), `recurrent_state_bytes` metric (`:97`) | "ui_only" / compiler "packed_unwired" | cost.usage_transparency |
| Command palette (`/`) | `ui.tsx` CommandPalette | "partial" (MCP/skill unification absent, backend packed) | palette.unified |
| Integrated terminal (`!` target) | present, **no PTY**: commands only queued as intents | missing (PTY) | palette.unified |

Wire contract note: `wire.ts` defines 11 intents, 7 event kinds, 30 custom names, 26 projections, but its Rust source of truth (`hide-core/src/api.rs`) is packed, so the contract is **unanchored and can drift** (HIDE_LIVE_ARCHAEOLOGY.md §3.4). Restoring `hide-core` as schema authority is a Phase 0/1 item (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §3).

## 4. The streamed gather-act-verify loop

The spine, streamed live with the user inside it (Genome §1, DOCUMENTED). The inner loop is flat and execution-grounded, not a rigid state machine (frontier dossier §4.4, §5.8):

```text
observe -> decide -> invoke typed action -> receive bounded evidence
        -> update task state -> verify -> continue or stop
```

Each stage renders into the transcript as it happens: the current action is the brightest thing on screen (the Context Stack "now" stratum, `ContextStack.tsx:25`, "tool_progress stream = the agent's real moves"), completed tool calls collapse behind it (§8), and verification re-enters the loop as evidence, never as self-congratulation (§14).

Current truth (HIDE_LIVE_ARCHAEOLOGY.md §3.5): the live turn does **not** run this loop. `SubmitTurn` sends the raw prompt with empty history and `max_output_tokens = 256` through a `StubPlanner`; the reserve-then-fill context compiler runs only behind a connector and its output is discarded (S3); `compact_context` is logged, never performed (S4). Making the transcript real is a **reconnection**, not a build: lift `hide-kernel` (RuntimePlanner, not StubPlanner), `hawking-context`, `hawking-index`, `hide-tools`, and replace the single-shot turn with the flat loop (frontier §7 Phase 0/1). Controller-state options are enumerated in `HIDE_AGENT_KERNEL_OPTIONS.md`.

## 5. The two steering gestures and the side query

This gene carries the most love and is the least forgiving of latency (Genome §1). All three are first-class on **every** streaming turn.

### 5.1 Interrupt-and-keep [parity `loop.interrupt_and_keep`, P0]

Two independent gestures on a live turn: a hard interrupt that cancels the in-flight tool call and **keeps all completed work** in context and returns to the prompt, plus a second clear/exit control. Enforcement (parity spec): the interrupt must not discard tool results already returned; it targets only the in-flight call.

- FE today: SteerBar `onCancel -> cancel_run` and `onPause -> pause_run` exist (`SteerBar.tsx`), but there is no live backend turn to interrupt ("ui_only", HIDE_LIVE_ARCHAEOLOGY.md §3.4). Cancellation semantics (cancel in-flight tool only, retain prior evidence) are a kernel item (`hide-kernel` governor/interrupts, packed_unwired, §3.5).
- SUPREMACY (gated on `hide-kernel` reintegration + no network round-trip): the interrupt is genuinely zero-latency because there is no server request to cancel; and the interrupt point can **fork a warm state** to branch both directions from where you stopped (gated on state-fork exposure, `HIDE_STATE_CAPSULE_ABI.md` §8). Parity is table stakes; the fork is the wedge.

### 5.2 Soft steer [parity `loop.soft_steer`, P0]

Typing a correction + Enter is queued and injected at the next turn boundary **without** stopping the running tool (read as soon as the current action completes).

- FE today: the SteerBar input fires a `redirect_run` custom intent (`SteerBar.tsx`; wire `redirect_run` at `wire.ts:85`), placeholder "Redirect this run". No live backend consumes it ("ui_only"). The turn-boundary injection point is a kernel item.
- Distinction the UI must keep visible: soft steer (does not stop the tool) vs interrupt-and-keep (stops the in-flight tool, keeps prior results). Same input field, two outcomes, disambiguated by gesture (Enter vs the interrupt control), matching the doctrine's single persistent steering field (`DESIGN_DOCTRINE.md:309`, `:355` "interruptible, never fire-and-forget").

### 5.3 The side query [parity `loop.side_query`, P2, `/btw`-style]

An ephemeral side-question answered mid-turn from in-context material only (no tools), in a dismissible overlay that never enters history and is promotable to a real session.

- FE today: **absent** (parity `hide_status: absent`). No overlay, no read-only side channel.
- SUPREMACY (gated on state-fork exposure): a warm fork answers on a **second local decode stream at zero marginal cost and zero added latency** to the main turn (parity `hawking_superiority`; `rwkv7.rs:376-378` memcpy fork, real-but-unwired). Claude Code answers `/btw` on the same metered pipeline; HIDE answers it on a sibling that shares the parent state by pointer. This is the cleanest small demonstration of "pass state, not text" and is scoped as an experiment in `HIDE_EXPERIMENT_MENU.md`. Until fork routes land (`HIDE_STATE_CAPSULE_ABI.md` §8), the overlay is a read-only projection over current context with no second stream.

## 6. The unified palette: `/`, `@`, `!` [parity `palette.unified`, P1]

One incremental fuzzy palette on `/` that **merges built-ins, skills, plugin commands, and MCP prompts** into a single list; `@` picks files and MCP resources (`@server:proto://path`); `!` runs a shell command directly into context. Dismissable ghost suggestions may seed from git history, suppressed after the first turn and in plan mode.

| Element | FE reality | Backing readiness | Notes |
|---|---|---|---|
| `/` incremental palette | `ui.tsx` CommandPalette real | "partial" | skill/MCP/plugin unification absent (runtime packed) |
| `@` file + resource picker | `@`-references to files present (`DESIGN_DOCTRINE.md:309`) | partial | MCP-resource half needs the MCP host (`hide-tools` client packed) |
| `!` shell into context | palette affordance can queue it | missing (PTY) | terminal has no PTY; commands only queued as intents (§3) |

Naming (clean-room, Genome §6): short functional labels (clear, compact, context, model, resume, doctor, usage, review) are functional and fine to reuse. The skill/plugin/MCP runtime that populates the palette is specified in `HIDE_TOOL_SKILL_PLUGIN_MCP_ABI.md`; the palette here is the surface, not the registry.

## 7. The agent-authored todo list [parity `loop.todo_list`, P1]

A first-class task-list primitive the model writes to: multi-step checklist, three-state markers, ~5 visible, persisting across compaction, shareable via a named id.

- FE today: `PlanCard` + `PlanStepRow` render exactly this (three-state `STEP_MARK` glyphs, `active` state, inline edit, reorder up/down, `done/total` count, `chat/structure.tsx:28-120`). Status "ui_only": **no mock or live path emits a plan projection** (HIDE_LIVE_ARCHAEOLOGY.md §3.4), so the card never populates in dev.
- Backing: the agent-authored list is the same artifact as the plan the kernel writes (`hide-kernel` plan-as-data DAG, packed_unwired). One list serves two roles: the pre-approval plan (§11) and the running todo. Persistence across compaction requires the compaction path to preserve it (`compact_context` is a checkpoint, not an essay, frontier §3.4; today it is logged-never-performed, S4).
- SUPREMACY: the list survives compaction for free when it is folded into the warm-state checkpoint rather than re-summarized (gated on capsule-persistence, `HIDE_STATE_CAPSULE_ABI.md` §8).

## 8. Collapsed tool rendering + transcript toggle [parity `loop.collapsed_tools`, P1]

Tool and MCP calls collapse to one-liners by default; repeated same-server calls coalesce to a count ("Called index 3 times"); a transcript viewer expands raw I/O plus per-message timestamp and the model used.

- FE today: `ToolChipRow` renders collapsed tool rows (`chat/structure.tsx:207`); data is mock ("partial"). Count-coalescing and the full raw-I/O transcript toggle are **not yet** present and must be added.
- Doctrine fit: collapse-by-default is the "legible + airy" rule made literal (`DESIGN_DOCTRINE.md:29`); the expanded transcript is the drill-in for power users (Genome §5). Rendering stays grayscale; the +/- diff glyphs carry meaning so desaturated color is never alone (`DESIGN_DOCTRINE.md:295-297`).
- Backing: real tool I/O requires the typed-tool runner (`hide-tools`, packed_unwired) and the tool/effect ledger (session core). Per-message model + timestamp come from the event stream (`hide-backend` event bus, packed). Over the current serve boundary a tool-bearing turn **buffers to completion before parsing** (T7, §3.3), so streamed collapsed tools require wiring the tool loop into the batched decode path.

## 9. The status line: a JSON contract, reframed to local telemetry [parity `loop.status_line`, P1]

Claude Code feeds a shell script a stable JSON object and renders its stdout. HIDE keeps the **scriptable JSON contract** (so a user's status-line script is portable) but replaces the metering fields with honest local telemetry. This is the strongest inversion in the whole surface (parity `cost.usage_transparency`: "no meter, no plan limits").

- FE today: `StatusBar.tsx` renders `phase`, `model`, `transport` from real store fields; **branch and problems are hardcoded** (`:46-48`, "partial"). The scriptable-hook layer does not exist yet.

Field contract (target). Fields the doctrine forbids are dropped, not restyled:

| Field | Keep / drop | Reason |
|---|---|---|
| model, cwd, git branch + staged/modified, PR number+state | keep | local facts, no metering |
| context fill (exact resident tokens / state bytes) | keep as **raw counter**, not a percentage meter | doctrine forbids a token budget, a context percentage, or a budget bar on any surface (`DESIGN_DOCTRINE.md:281`); exact local counts framed as headroom, not a countdown |
| tokens/s, energy J/tok, resident GB, state-fork depth | **add** | true local telemetry, the capability-headroom framing (parity `cost.usage_transparency` hawking_superiority) |
| session cost (USD), rate-limit windows, 5-hour/weekly caps | **drop** | no meter, no plan limits (anti-genome, Genome §11); there is no dollar HUD in HIDE |

The distinction the doctrine draws: HIDE's own chrome shows **no** budget/percentage meter anywhere; the scriptable contract exposes exact counters that a user may render however they choose in their own status line. Abundance is expressed by the absence of the meter (`DESIGN_DOCTRINE.md:281`), and context fill is exact from local KV state rather than a dishonest "percent of infinite" (frontier §4.7). Metering-free telemetry design is elaborated in `HIDE_CONTEXT_OS_SPEC.md` (the Context Stack) and `HIDE_SPEED_FRONTIER.md` (the tok/s, J/tok, prefill counters).

## 10. Permission: the inline gate

Consequential actions stop at a lit approval capsule that states plainly what will happen and holds steady (steadiness says it waits for you, breathing says it works, `DESIGN_DOCTRINE.md:322-336`).

- FE today: `InlineGate` renders in the stream with the `.gate` capsule; `Approve` fires, `Dismiss` cancels (`chat/structure.tsx:278`). A bypass/auto-approve toggle exists in the FE (parity `perm.mode_cycle` "partial"). Status "packed_unwired": the enforcing rule engine (`GateBook` in `hide-backend`, `hide-security` allow/ask/deny with deny->ask->allow precedence, protected-path pre-gate, `rm -rf /` circuit breaker) is packed and unwired (parity `perm.rule_engine`).
- Hard rule (parity `trust.workspace_gate`, P0): project config is **data until trusted**. Serve today binds `0.0.0.0` with no auth (G10, `main.rs:133`); no permission mode may resolve before a first-open trust gate. The gate card in the stream is the last mile; the enforcement layer and trust model are specified in `HIDE_PERMISSION_AND_EFFECT_SYSTEM.md` and `HIDE_SECURITY_CONSTITUTION.md`. Security precedes autonomy (frontier §7 Phase 0).
- SUPREMACY (gated on Seatbelt wiring): a denied capability is **physically absent** at the OS/syscall boundary, and with local inference egress is default-fully-off, so the exfiltration class disappears (parity `security.sandbox` hawking_superiority; `hide-security` renders Seatbelt profiles today, egress proxy is a seam).

## 11. Plan card + graded approval [parity `perm.plan_mode`, P0]

Plan mode lets the agent read files and run read-only shell but is **structurally blocked** from editing source until a written plan is approved. Approval is graded, not binary: approve-and-run-autonomously / approve-and-review-each-edit / keep-refining / hand-off-for-deeper-review, and the plan text is editable first.

- FE today: `PlanCard` supports approve, inline step-edit, and reorder (`chat/structure.tsx:28-120`); the `approve_plan` intent exists. Status "ui_only": the graded dialog (the four post-approval modes) and the **executor-level write block** are not present. The write block must be enforced at the tool gate, not by prompt (parity `perm.plan_mode` enforcement; needs `hide-kernel` + the `hide-tools` gate, packed_unwired).
- The plan artifact is the same object as the todo list (§7): one plan-as-data DAG where each step may declare its acceptance oracle up front (the highest-value idea in the packed backend, HIDE_LIVE_ARCHAEOLOGY.md §3.5 `hide-kernel`).
- SUPREMACY (gated on state-fork exposure): best-of-N candidate plans executed speculatively in isolated local forks, so the user picks the plan that **already worked**, not a guess (parity `perm.plan_mode` hawking_superiority; `HIDE_STATE_CAPSULE_ABI.md` §7 fork). Parity is the gate and the write-block; the speculative fork is the wedge.

## 12. The Context Stack light well

The signature surface: a narrow vertical shaft of light along the east wall, present in every chamber, where the agent's work enters the building as light (`DESIGN_DOCTRINE.md:311-320`). In Chat it answers, at a glance, what the model knows, why it was included, what was excluded, what is stale, and what survives compaction (frontier §4.7). It is **not** a "percent of infinite context" meter.

- FE today (`surfaces/ContextStack.tsx`): strata top-to-bottom are current-action (the live feed, `.alive`, breathing), Skills (instant-resume states), retrieved files/symbols, tools called, memory, tests/state. There is **no budget stratum** (`:151`, doctrine-clean). The stack exposes `recurrent_state_bytes` as a state metric (`:97`) and a fork-from-current-state action (`:73` -> `fleet_run` custom intent). Touch affordances (pin, evict, mute, `@`-add) are TE line glyphs (doctrine `:319`). Status "ui_only"; the reserve-then-fill compiler that would populate it truthfully is `hawking-context`, packed_unwired.
- The Context Stack is shared across both surfaces (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §6); its backing compiler and provenance model are specified in `HIDE_CONTEXT_OS_SPEC.md`. This spec covers only its Chat rendering.
- SUPREMACY: "what state survives compaction or provider switching" is answerable exactly because the state is a local object (`HIDE_STATE_CAPSULE_ABI.md`), and `recurrent_state_bytes` is already surfaced from the live engine (the one state fact wired today, HIDE_LIVE_ARCHAEOLOGY.md §3.1).

## 13. Progress grammar mapped to controller state

HIDE has its **own** progress signature and its **own** vocabulary. Two clean-room rules (Genome §5, §11; `DESIGN_DOCTRINE.md:344`):

1. **The signature is light, not a word-spinner.** The ambient "alive" glyph is the event-horizon ring filling or breathing with light, grayscale only, no color, no fast spin, static-lit under `prefers-reduced-motion`. It is layered on top of the truth, never a substitute: the live feed still shows the real move.
2. **The words are HIDE's own, functional, and truthful.** Do not reuse Claude Code's whimsical rotating gerund vocabulary. HIDE's "spinner text" is simply the truthful present participle of the current action, drawn from the fixed progress grammar below and rendered in the terse voice ("Reading guard.rs", "Running 12 tests", `DESIGN_DOCTRINE.md:318`, `:366`). No mystery gerunds, no percentage bar.

The grammar is a closed set of twelve verbs. Each maps to the coarse run-phase projection that exists in the FE store (`RunPhase` at `store.ts:16`, labels at `SteerBar.tsx:89-96`) and to the finer kernel/tool activity that emits it (the target controller, mostly packed). Readiness is of the emitting activity.

| Grammar verb | FE RunPhase projection | Emitting activity (target controller) | Emitting readiness |
|---|---|---|---|
| reading | `executing` ("Running") | typed Read / file open (`hide-tools`) | real-but-unwired (packed) |
| searching | `executing` | index / grep query (`hawking-index` RRF retriever) | real-but-unwired (packed) |
| planning | `planning` ("Planning") | plan-as-data DAG author (`hide-kernel`) | real-but-unwired (packed) |
| editing | `executing` | verifying edit applier, emits a diff (`hide-tools`) | real-but-unwired (packed) |
| running | `executing` | sandboxed `shell.run` w/ watchdog (`hide-tools`) | real-but-unwired (packed); FE terminal has no PTY (§3) |
| verifying | `executing` -> `awaiting` | deterministic-first oracle gate (`hide-kernel`) | real-but-unwired (packed) |
| waiting | `awaiting` ("Awaiting") | governor blocked on approval / input (InlineGate) | partial (gate card real, rule engine packed) |
| blocked | `awaiting` or `failed` | policy denied / missing input / stall policy | partial (permission engine packed) |
| delegating | `executing` | subagent / fleet fan-out (`hide-fleet` + `hawking-orch` role router) | real-but-unwired (packed, not HTTP-reachable) |
| reviewing | `executing` -> `awaiting` | independent review, diff-in-scope check (`hide-kernel`) | real-but-unwired (packed) |
| merging | `executing` | integration / merge funnel (`hide-fleet`) | real-but-unwired (packed) |
| done | `done` ("Done") | verify gate satisfied (§14) | partial (FE `done` label real, evidence packet missing) |

`paused` and `failed` are run-phase states, not grammar verbs: `paused` is the soft-steer/interrupt outcome (§5), `failed` is the terminal of `blocked`. The FE projection is real (the seven `RunPhase` values transition in the store); today the transitions are mock-driven because the live turn has no phases (single-shot 256-tok generate, §4). Wiring the grammar means emitting these twelve labels from the flat kernel loop as it actually moves, which is the Phase 0/1 reconnection.

## 14. Completion: "done" is proof, not celebration

`done` is a verification verdict, not a toast. It renders the objective evidence appropriate to the task, never a congratulation (frontier §4.5; voice rule `DESIGN_DOCTRINE.md:364` "Diff accepted", never "Successfully accepted").

Evidence a `done` turn shows (from the verification plane, frontier §4.5): patch applied transactionally; project builds or typechecks; targeted tests pass; regression suite not worsened; the final diff stayed within requested scope. Self-judgment is a weak signal, not the accept gate (frontier §4.5, §5.12 actor/evaluator separation). High-risk changes carry an independent review verdict (the `reviewing` verb, §13).

- FE today: `RunPhase.done` label is real (`SteerBar.tsx:95`), but the evidence packet behind it is not, because the verification plane (`hide-kernel` oracle gate, `hawking-eval` pass@1 + Wilson CI) is packed and there is no live capability harness (only perf `hawking-bench` is wired, HIDE_LIVE_ARCHAEOLOGY.md §6.5). Reintegrating `hawking-eval` is cheap and unblocks every "done = proof" claim (frontier §7 Phase 0).
- Voice at completion: name the specific thing that changed, drop the trailing period, no superlative, no emoji ("3 files changed. Tests pass.", `DESIGN_DOCTRINE.md:362-371`). Errors are direct and blame-free and never apologize ("Couldn't reach the local engine. It may not be running.").

## 15. Parity vs supremacy summary

| Behavior | Parity obligation (reproduce) | Supremacy (gated on) |
|---|---|---|
| interrupt-and-keep | cancel in-flight tool, keep prior evidence | zero-latency (no network) + fork both directions [state-fork exposure] |
| soft steer | queue correction, inject at boundary | (parity is the win; local, no round-trip) |
| side query | read-only, no-history overlay | answer on a second decode stream, zero marginal cost [state-fork] |
| palette `/ @ !` | one merged palette, resource picker, shell-into-context | local MCP servers stay warm across sessions [MCP host] |
| todo list | 3-state, ~5 visible, survives compaction | folded into warm checkpoint, no re-summarize [capsule persistence] |
| collapsed tools | collapse + count-coalesce + transcript | real live tool I/O, no metered transcript [tool loop wired] |
| status line | scriptable JSON contract | telemetry not a dollar meter; exact context from KV [structural] |
| inline gate + plan | executor-level write block, graded approval | best-of-N plans run speculatively in forks [state-fork] |
| Context Stack | inclusion + provenance, no % meter | exact "what survives compaction" from local state [structural] |
| done | objective evidence, not celebration | local eval harness at zero marginal cost [hawking-eval] |

Every supremacy row is stated conservatively and gated on a named build item; the measured evidence for each is the job of `HIDE_SUPREMACY_THESIS.md` and `HIDE_EXPERIMENT_MENU.md`, not this spec.

## 16. What lands to make Chat real (feed-forward)

The Chat surface is a reconnection, not a build (HIDE_LIVE_ARCHAEOLOGY.md §6, frontier §7). In order:

1. **Restore the wire authority + the `/v1/hide/*` boundary** so the mock-fed store speaks to a real backend (`hide-core` + `hide-serve`, or those routes on `hawking-serve`). This makes the transcript, plan card, tool chips, diff chips, gate, and status bar carry real data at once, because they already share one store (`HIDE_TWO_SURFACE_ARCHITECTURE.md` §7).
2. **Replace the 256-token single-shot turn with the flat kernel loop** (RuntimePlanner, `hawking-context`, `hawking-index`, `hide-tools`), which is what emits the progress grammar (§13) and the gather-act-verify stream (§4).
3. **Wire the todo/plan projection and the graded-approval + executor write block** (§7, §11), then the collapsed-tools transcript and count-coalescing (§8).
4. **Add the scriptable status-line hook with the local-telemetry contract** (§9) and the trust gate before any permission resolves (§10).
5. **Reintegrate `hawking-eval`** so `done` carries proof (§14).

Then the supremacy gestures (side-query second stream, best-of-N plan forks) land on top, each gated on the state-capsule exposure build items in `HIDE_STATE_CAPSULE_ABI.md` §8. The prioritized sequence is in `HIDE_PRIORITIZED_BUILD_LADDER.md`.
