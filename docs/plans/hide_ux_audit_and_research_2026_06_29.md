# HIDE — Deep Audit + UX Research (2026-06-29)

Output of a 10-agent pass: 5 read-only code-audit lenses + 5 deep web-research topics. This feeds the
next build waves. Framing: HIDE's edge is **local + free fleets + abundant context (4-5M effective
tokens via the Hawking format) + the Executor** (not a chat). Spend tokens lavishly; iterate continuously.

---

## A. Audit punch-list (ranked)

### High
1. **Hardcoded session IDs** in `Chat.tsx`, `StateTimeline.tsx`, `CodeActions.tsx` — `scrub_to_event` /
   `fork_session` / `submit_turn` all target a fixed mock/`ses_live…` id, not the active run. → Track the
   active `session_id` in the store (from the latest `UiEvent.session_id` / boot handshake) and import it.
2. **FloatingChat not re-clamped on window resize** (`FloatingChat.tsx`) — only clamps during drag; can
   land off-screen after a resize. → `window` resize listener that re-clamps `pos`.
3. **Live-transport reconnect seq hazard** (`ipc.ts`) — reconnect uses the socket's `lastSeq`, not
   `lastAppliedSeq`; an event arriving during the disconnect window can be silently dropped. → Reconnect
   from `lastAppliedSeq`.
4. **Broken Account button, no handler** (`ActivityBar.tsx`) — and per product direction there's no
   account in a local-first app. → Remove it (matches "delete the profile/anonymous thing").
5. **WCAG AA contrast fail** on `--text-3` (#6e6d68) / `--mute` (#5c5b57) over `--void`. → Lighten to
   ~#747169 / ~#66655b (verify ≥4.5:1).
6. **Code-action popover** has no keyboard dismissal/focus (`CodeActions.tsx`). → Esc to close + return
   focus, arrow/enter nav, focus first item.
7. **Perf:** Monaco TS worker dominates the bundle (lazy-load workers on first file open); markdown
   re-parses O(n²) on every streamed token (batch parse ~every 250-500ms); split broad `useStore` reads.
8. **Backend (the live-streaming + executor gap):** token batches aren't persisted to the event log;
   the kernel loop emits no per-step `projection_patch`/`tool_progress`; runtime status isn't published
   at boot. These three are the smallest changes that make **live chat stream + the executor pipeline
   visible**. (Second/backend plan.)

### Med / Low
- Dead code: `shell/SearchView.tsx`, `shell/RunsView.tsx` (unmounted) → delete (functionality folded).
- ARIA/roles + focus management on HunkReview / Explorer tree / ContextStack; reduced-motion not applied
  per-animation (only the global rule); optimistic fleet `seq=0` should be flagged so reconnect ignores it.
- Stale comment in `Editor.tsx` (references a Workstation merge-review that no longer mounts).
- Doctrine fidelity: **clean** — no blue/gold/glow violations; Monaco/xterm token-mirroring intact.

> Note / tension: one audit lens argues the **Agents (Fleet) view is NOT redundant** — it's the
> parallel-run orchestrator, distinct from inline progress. Reconcile with the user's "fold agents into
> the executor": keep Fork-&-Try-N, but reach it **from the Executor**, not a separate rail tab.

---

## B. UX features to add — ranked, tied to HIDE's moats

### 1. The Executor pipeline (the headline: state-of-the-art, continuous, visible progress)
- **Streaming inline diffs + tab-to-apply** — edits stream INTO the editor (not a modal); Tab accepts,
  Esc rejects. (Cursor/Windsurf.) The single biggest "seamless" win.
- **Live diff feed tied to checkpoints + replay** — every change is a checkpoint; scrub/replay the run
  (extends our StateTimeline to diffs). Exploits M1 (state snapshots = instant time-travel).
- **Continuous loop made visible:** plan → execute → **verify (run tests/lint)** → iterate, with
  streaming red/green test badges in the Executor. "Reward-from-tests" is what makes it *keep going*.
- **Reflexion / "Learnings" card** — a bounded list of the agent's self-corrections this run ("tried X,
  got Y, switched to Z"). Makes iteration feel like progress, not flailing.
- **Background agents + interrupt** — launch a long refactor; keep editing; interrupt/steer anytime.
  (M2 free fleets — runs all night for free.)
- **Stratified progress** — ambient radiate dot → expandable phase timeline → "needs you" gate. Calm at
  a glance, deep on demand.
- **Pause gates** — autonomous until a risky action (delete >N files, breaking API change), then the lit
  approval capsule.

### 2. Long-context leverage (the 4-5M-token edge — "be expensive with tokens")
- **Whole-repo-on-boot, cached** — load the project into a persistent context layer on open; "instantly
  warm." (Honest: long context ≠ perfect recall; pair with the index for exact lookups.)
- **Permalayer + Actions (layered context)** — a stable read-only layer (the repo/brain, ~90%) + a
  volatile action layer; cache the stable part. Sourcegraph's pattern for infinite context.
- **"Project Brain" (replaces chat history)** — the Context Stack becomes a structured, evolving memory
  (architecture notes, build/test commands, decisions) that persists across sessions — not a transcript.
- **Cost-aware-but-abundant** — we removed the budget meter on purpose; optionally a quiet "whole repo
  loaded · cached" cue (abundance, not a meter).

### 3. Micro-interactions & feel (the "snap" you liked, compounded)
- **Optimistic UI everywhere** (100ms feedback, reconcile later) + **skeleton, never spinners** (our
  radiate already fits).
- **Weighted/spring motion** for panel/drag/selection; keep Monaco's snappy cursor snap.
- **Peek preview** — Cmd/Shift-click a file/symbol → lightweight glass overlay (not a full open).
- **Command palette with inline shortcuts** + keyboard-first everything; **prefetch on arrow-nav**.

### 4. Voice + local-first
- **Local voice-to-code** (Whisper via the engine; we scaffolded the mic, no time cap, no egress).
- **Local-first as the pitch** — provable no-egress, free, fast; ride the 2026 "Tokenpocalypse" backlash.
- **@-mentions / symbol context** + a **`.hide/rules`** file for per-repo style/context.

---

## C. Directed product changes (from the user, audit-confirmed)
- **Rename "Assistant" → "Executor"** everywhere (it executes; agentic workflows live inside it).
- **Remove the Profile page + the account/anonymous icon** (fully local, your own models, no account).
- **Fold "Agents" into the Executor** — keep Fork-&-Try-N, surface it from the Executor rather than a
  separate sidebar tab (resolves the tension above).
- Build a real **Settings** surface (the gear currently just opens the palette) — model, endpoints,
  keybindings, a11y toggles.

## D. Suggested sequencing
1. **Quick FE pass (now):** Executor rename · remove account button + dead SearchView/RunsView · fix the
   3 high-severity correctness bugs (session id, resize clamp, reconnect cursor) · AA contrast ·
   code-action keyboard.
2. **FE features wave:** streaming inline diffs + tab-to-apply · live diff feed + checkpoint replay ·
   stratified progress + Learnings card · peek preview · optimistic/spring polish · Settings surface.
3. **Backend (the second plan):** the executor pipeline (plan-execute-verify loop, token_batch + per-step
   events, reward-from-tests, background agents) · whole-repo-on-boot + Permalayer caching + Project Brain
   · local Whisper · "just do it" agency (git push itself). This is where the long-context + continuous
   iteration moats become real.
