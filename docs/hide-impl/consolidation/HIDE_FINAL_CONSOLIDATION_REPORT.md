# HIDE Final Consolidation Report

Campaign: HIDE consolidation and productivity density, branch `build/hide-impl-2026-07-19`, base
`4fbca8bc`. Nothing is committed. House rule: hyphens and parentheses only, no long dashes.

## 0. Verdict first

The stop condition was NOT met. The campaign required that an adversarial pass find no remaining
high value consolidation in scope. Three adversarial passes ran; all three failed all four lenses.
A fourth remediation round then closed every critical the third pass found, but no fourth
adversarial pass has verified that work. Treat this report as a verified state snapshot plus an
open items list, not a completion claim.

Convergence trend across passes (criticals / total findings): 5/57, then 3/47, then 3/40. Real
improvement, not yet convergence.

## 1. Measured state (my own runs, not agent self reports)

| Gate | Value |
| --- | --- |
| Rust | 648 tests across 15 hide crates, 0 failed (hide-backend 215, protocol 27, sdk 18, serve 9) |
| Frontend tests | 386 passed across 17 files (campaign baseline 101 across 7) |
| Frontend typecheck | clean (`tsc --noEmit`) |
| Frontend production build | succeeds (`vite build`) |
| Frontend lint | NOT RUN. The repository has no eslint config and no lint script. No lint tooling was added. |
| Command catalog | 48 commands, 45 palette visible, 7 carrying a chord, across 10 surfaces |
| Catalog drift | `app/src/generated/command_catalog.json` byte identical to `crates/hide-sdk/goldens/command_catalog.json` |
| Newly authored dashes | 0 (verified on ADDED lines, and against sealed pack `5a99d0e2` for rehydrated files) |
| Git | HEAD still `4fbca8bc`, zero commits, nothing pushed, no authorship attribution |
| Dependencies | `app/pnpm-lock.yaml` and `app/package.json` unchanged. Install command was `pnpm install --frozen-lockfile` (118 packages, 0 downloaded, all from local store) |
| Cargo.lock and root Cargo.toml | MODIFIED, legitimately: nine new crates joined the workspace |
| vendor/strand-quant, vendor/strand-decode-kernel, tools/strand | untouched |
| Model activity | none. No model downloaded, staged, selected, benchmarked or loaded |

## 2. The thirteen required answers

### 1. Which existing controls became deeper

Chat composer submit (state driven: starts a turn when idle, steers a live turn, with an honest
state label instead of a false "Queue turn"), the two New Chat buttons (one shared menu: new thread,
side chat, research fork, fork from checkpoint), StatusBar Problems counter (real diagnostics
projection with a per file popover, and it now triggers the analysis), StatusBar branch label,
StateTimeline (one history control carrying ten verbs: create, inspect, rewind conversation, rewind
code, rewind both, replay, fork, compare, restore, invalidated receipts), ContextStack (checkpoint
create, fork, real memory domain, real context receipt), HunkReview and the Editor diff bar (per
hunk accept, reject, revert with provenance, base hash, related verification, one gated whole diff
revert), Terminal (sandbox confined run, incremental streamed output, persistent process, attach,
stop, artifact capture, state row), PlanCard (real plan projection with acceptance, dependencies,
effects, verification, blocker, write blocking), Explorer and the command palette (one search engine
across files, symbols, references, transcript, threads, tools), CodeActions (selection actions with
stable source refs), Settings (palette path, derived keyboard table), Home and SideBar (goal field,
workspace trust, background jobs, environment, one model chooser).

### 2. Which backend capabilities became reachable

Steering (real InterruptHub Steer, previously log only), transcript search, side chat create and
merge, per hunk diff accept and reject and revert with provenance, static analysis and verification
receipts, the plan domain (approve, edit, reorder, skip with reason, repair), checkpoint create,
inspect, rewind by domain, replay, fork, compare, restore, background promotion and resume in
foreground, durable memory (add, supersede, record outcome, revalidate), goals (set, clear,
evaluate), workspace repo add and trust, environment switch, terminal pty input and resize,
sandboxed process lifecycle.

### 3. How many permanent controls were added

Two genuinely new permanent visible elements (a Terminal state bar and a per hunk detail toggle),
plus three one for one swaps (StatusBar Branch button became the Problems button, StateTimeline
fork from here became the history menu, the Chat attach slot became the composer mode label). This
count is the independently reviewed one, not the sum of per stage self reports. Everything else
added is conditional (rendered only when relevant) or lives inside an existing menu or panel.

### 4. How many permanent controls were removed

Roughly 41 retirements, independently confirmed as real. Source level the frontend NET SHRANK.

### 5. Which mocks and facades were eliminated

Hardcoded Problems 0/0, hardcoded branch label, two voice mic mocks that recorded then discarded,
the ContextStack save skill and three hardcoded load skill rows, the toast only snapshot and the
fleet_run misuse behind fork state, the never rendering DiffChipRow accept and reject, the dead
Attach button, the misleading Artifacts rail item, three empty payload switch_model copies, four
invented model ids, a hardcoded product version, fabricated runtime readiness, the mock file tree
and mock file bodies reaching a live transport, and four controls firing custom names no host arm
handled (save_file, create_pr, switch_profile, focus_run). Seventeen orphan custom names were
retired from the wire contract, and unhandled custom names now return an honest negative ack instead
of being acked as successful.

### 6. Which workflows require fewer interactions

Honestly: few workflows got fewer clicks, and that was never the main win. The composer already
submitted in one gesture. What changed is that the same gesture now produces a real durable effect
instead of a log only no op. Genuine interaction reductions: steering (previously impossible, so the
user had to cancel and restart a turn), fork and checkpoint from one history menu instead of hunting
separate controls, one search surface instead of two implementations, and palette reach for actions
that previously had none (palette went from 6 hand written rows to 45 derived).

### 7. Which workflows became more reversible

Diff review (per hunk revert plus one gated whole diff revert, with invalidated verification
reported), checkpoint and rewind (conversation only, code only, or both, with the code domain now
actually reverting the working tree), fork from checkpoint, and background jobs (pause, steer, stop,
resume in foreground). Effectful steps now pause at a real approval the user can answer.

### 8. Which workflows gained stronger evidence

Diff review (per hunk provenance: originating plan step, agent, turn, blake3 base hash, plus a
sealed review receipt), verification (durable Tier1 receipts, a diagnostics projection, invalidated
receipt reporting), context (a real context receipt from the manifest the host publishes), side
chats (typed foldback with evidence links rather than a transcript dump), and the terminal (captured
output preserved as a durable artifact).

### 9. Which keyboard and command palette paths were added

Palette entries are now DERIVED from the one catalog rather than hand written, going from 6 rows to
45 palette visible commands. Shortcuts are derived from the same catalog with a collision test.
Settings gained a palette row and Mod+, . The five conversation side panels and open a recent
session gained palette paths. Chords that nothing bound were removed from their CommandSpec rather
than left advertised. The palette now displays each row's binding.

### 10. Which donor mechanisms were integrated

Codex (Apache 2.0, commit `678157ac`): durable thread lifecycle (persist, flush, shutdown, discard
plus the init guard drop discard), partial history fork via an ordinal boundary marker (clean room),
side conversation boundary and typed foldback, initialization and capability negotiation.
Grok Build (Apache 2.0, commit `ba76b0a6`): checkpoint store, rewind, replay, hunk tracking (adapted
onto the HIDE event log). OpenCode (MIT, commit `ba4b8e21`): mostly already covered; the comparison
surfaced and fixed two real defects (an inherit all tool list that resolved to empty, and a manifest
that could declare a process effect with no sandbox). LSP was explicitly fenced out of scope.
Provenance is recorded in HIDE_DONOR_PORT_LEDGER.md.

### 11. What remains model dependent

Every model bearing leg, marked DEFERRED_MODEL_REQUIRED (95 markers): live turn generation, semantic
search, probabilistic review roles, live state capsule production, the real browser driver, the ACP
model turn handler, and any capability, quality or parity claim. No such claim is made anywhere in
this campaign.

### 12. What remains unfinished

Known open items, carried honestly:

1. No fourth adversarial pass has verified round 4. Every prior round introduced regressions, so
   assume this one did too until proven otherwise. This is the single most important open item.
2. The ungated per hunk reject is still policy denied on a shipped host: `reject_diff` with a
   `hunk_id` is ApprovalPolicy Auto, so it never enters the approved write scope. Fixing it needs a
   catalog policy change and a golden regeneration.
3. The agent's own edits go through the same dispatcher and are refused under the shipped default
   `workspace_write_default = Ask`, so on a live host the diff store can be empty. This is a policy
   and design decision, not a bug, but the product is unusable until it is decided.
4. The trust chip still reads held after the user approves its gate: no surface consumes the
   `repo_trust_set` event.
5. The record read path (memory list, checkpoint list, goal, job records) is still not surfaced.
6. The dangerous command denylist matches argv[0] only and is evadable; the OS sandbox is what
   actually holds.
7. The permission mode bypass still auto approves the ApprovalPolicy Ask commands, because Ask holds
   share the gate channel with the denylist.
8. Several accessibility items: aria-label on role less spans, a disclosure missing aria-expanded,
   run lifecycle controls nested in a live region, a listbox nested in a menu.
9. Frontend workflow traces were exercised through component, store and command registry tests plus
   deterministic backend traces, NOT through a driven Tauri UI. That limit is stated rather than
   papered over.

### 13. What should be the next implementation campaign

Decide the write policy question (item 3) first, because it gates whether the diff and undo surfaces
can function at all on a shipped host. Then run a fourth adversarial pass, fix what it finds, and
only then consider the campaign closed. After that, the highest value next campaign is a real
frontend workflow harness (driven Tauri or an equivalent) so Traces A through G are proven through
the actual UI rather than through component and backend tests, followed by the record read paths and
the accessibility backlog.

## 3. Process lesson worth keeping

Rounds 1, 2 and 3 each introduced regressions, and the cause was identical every time: a fix was
applied to one call site instead of the shared function every caller routes through. Round 3's own
critical was that it fixed the write policy denial on `save_file` alone. Round 4 was briefed to grep
every caller first, and its fixes went into a permission engine wrapper and one scope around the
single release entry point, into one `effect_failed` helper covering ten side effect blocks, and
into an effect level policy check covering every channel. That is the shape a correct fix takes
here.

Two defects were only found because a reviewer read for structure rather than for symptoms: the diff
surface had no backend producer at all, and the ContextStack surface was imported by nothing. Both
were surfaces the campaign had already "deepened". Verify a surface renders before deepening it.
