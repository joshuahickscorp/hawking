# HIDE Claude Code UX Genome

Run date: 2026-07-19 · Research cutoff: 2026-07-19 · Claude Code line studied: v2.1.x (~v2.1.215)
Method: clean-room fleet (`wf_5c04451d-beb`), 11 scouts each independently verified. Evidence labels per claim.
Clean-room boundary: behavioral contracts and workflow patterns only; no proprietary source, assets, or verbatim product copy. HIDE uses its own names, voice, and visual doctrine.

## 0. What "genome" means here

This is the ordered set of *felt behaviors* that make experienced developers keep paying for Claude Code even through cost, rate limits, and a multi-week quality regression. Each gene is: the behavior, why it is loved (the emotional/workflow payoff), its evidence level, and the one-line parity obligation it puts on HIDE. Section 9 ranks them by how much of the "love" they carry, because the Apple-vs-Samsung doctrine says HIDE must reproduce the *coherence*, not a feature checklist.

## 1. The core loop gene - steerable autonomy

Claude Code's loop is `gather context → take action → verify`, streamed live, with the user explicitly *inside* the loop. [DOCUMENTED]

The loved thing is not autonomy; it is **steerable** autonomy - the ability to redirect the instant you see the agent heading wrong, without losing progress:

- **Two distinct steering gestures** [DOCUMENTED]: `Esc` hard-interrupts, cancels the in-flight tool call, and **keeps all completed work** in context; typing a correction + `Enter` **soft-steers without stopping** the running tool (read as soon as the current action completes). `Ctrl+C` is the second interrupt (clear input / exit). [verifier: Ctrl+C omitted by scout, added here]
- **`/btw` ephemeral side-questions** [DOCUMENTED]: ask about the code Claude just read *mid-turn*, answered from in-context material only (no tools), in a dismissible overlay that never enters history and can be promoted to a real session.
- **`/recap` and idle session recap** [DOCUMENTED]: a one-line recap generates in the background when you return after ≥3 min idle.

Parity obligation: HIDE must make interrupt-and-keep and soft-steer first-class on every streaming turn, plus a no-history side-query. **This gene carries the most love and is the least forgiving of latency.**

## 2. The trust-gradient gene - Shift+Tab permission modes

One keystroke moves along a trust gradient with an always-visible mode badge. [DOCUMENTED] Base cycle `default(Manual) → acceptEdits → plan`; optional `bypassPermissions`, `auto` slot in after `plan` when enabled; `dontAsk` is never in the cycle. As of v2.1.200 the display name of `default` became "Manual" (a label change, not a behavior change - verifier corrected the scout's "from Auto" framing). [DOCUMENTED]

Why loved: users self-tune oversight-vs-speed per task without touching config - explore in `plan`, grind in `acceptEdits`, stay cautious in `Manual`. The **always-visible badge** is half the value: you never guess what the agent is about to do.

Parity obligation: a single cycling chord, a permanent glanceable mode indicator in the input footer, and *enforced* semantics (plan blocks edits at the executor level, not by prompt).

## 3. The plan gene - think before doing

Plan mode lets the agent read files and run read-only shell but is **structurally blocked** from editing source until the human approves a written plan. [DOCUMENTED] Approval is graded, not binary: approve→autonomous, approve→review-each-edit, keep refining, or hand off for deeper review; `Ctrl+G` opens the plan text in `$EDITOR` first. [DOCUMENTED]

Why loved: this single gate is credited with preventing the majority of unwanted changes; it separates thinking from doing and gives a reviewable, editable proposal before any file is touched.

Parity obligation: a hard executor-level write block during planning, a structured editable plan artifact, and a graded approval dialog that deterministically sets the post-approval mode.

## 4. The reversibility gene - checkpoints and rewind

Every user prompt auto-creates a code checkpoint (files snapshotted before Write/Edit/NotebookEdit; **100 most recent** kept, 30-day `cleanupPeriodDays`). [DOCUMENTED, verifier-corrected: per-prompt not per-edit] `/rewind` or empty-input `Esc Esc` opens a menu to restore code, conversation, or both, plus "Summarize from/up to here"; checkpoints persist across resume and survive `/clear`. [DOCUMENTED] Separate from git; does **not** cover bash-made file changes or external edits. `/branch` and `--fork-session` make a divergent copy preserving the original. [DOCUMENTED]

Why loved: fearless experimentation - a bad multi-file swing is one gesture to undo, *including the conversation*.

Parity obligation: pre-edit snapshots of working tree + conversation, a bounded ring surviving resume, a three-axis rewind (code / conversation / both), and non-destructive `/clear`.

## 5. The legibility gene - todos, collapsed tools, transcript

- **Ctrl+T task list** [DOCUMENTED]: the agent's own multi-step checklist, three-state markers, ~5 visible, persists across compaction; shareable via `CLAUDE_CODE_TASK_LIST_ID`.
- **Collapsed tool/MCP rendering** with **Ctrl+O transcript viewer** [DOCUMENTED]: default view stays readable; repeated MCP calls coalesce to one line ("Called slack 3 times"); the transcript expands raw I/O, per-message timestamp, and the model used.
- **Scriptable status line** [DOCUMENTED]: a shell script is fed a stable JSON object (model, context used %, session cost+duration, cwd, git branch + staged/modified, PR number+state, rate-limit windows, vim mode, fast_mode) and its stdout is rendered above the footer.
- **Whimsical animated spinner** with rotating gerund words and an inline "(esc to interrupt)" affordance. [OBSERVED/ANECDOTAL - exact vocabulary not in docs]

Why loved: the agent stays legible during minutes-long runs; power users drill in; the colored context bar warns before auto-compaction bites.

Parity obligation: a first-class task-list primitive the model writes to, collapse-by-default tool rendering with a full transcript toggle, and a status-line hook fed a documented JSON contract. **The spinner must use HIDE's own word list and voice - do not reuse Claude Code's vocabulary.**

## 6. The discoverability gene - `/` palette and `@` mentions

Type `/` for an incremental fuzzy-filtered palette that **unifies built-ins, skills, plugin commands, and MCP prompts** into one list; `@` fuzzy-picks files and MCP resources (`@server:proto://path`); `!` runs a shell command directly and feeds output to context. [DOCUMENTED] Grayed-out ghost prompt suggestions seed from git history (Tab/Right accept; suppressed after first turn and in plan mode). [DOCUMENTED]

Parity obligation: one merged incremental palette on `/`, an `@` picker merging files + resources, a `!` shell affordance, and dismissable ghost suggestions. Short functional names (clear, compact, context, model, resume, doctor, usage, review) are functional labels, fine to reuse.

## 7. The memory gene - instructions that persist for free

`CLAUDE.md` at repo root (and `./.claude/CLAUDE.md`) auto-loads every session, git-shared; ancestor files load at launch, subdirectory files lazily when touched; `@path` imports (depth 4, code-spans skipped, first-use approval); `.claude/rules/*.md` with `paths:` globs load only on matching reads. [DOCUMENTED] Claude-authored auto-memory at `~/.claude/projects/<key>/memory/MEMORY.md` (200 lines / 25KB) recalls across sessions. [DOCUMENTED] The `#` quick-memory shortcut is **not in current docs** - treat as [ANECDOTAL], not a parity requirement to name identically. [verifier correction]

Why loved: no per-session re-explaining; a new teammate inherits context via source control; corrections persist without the user writing anything. **This is the primary source of workflow stickiness.**

Parity obligation: read an existing Claude Code project's `CLAUDE.md` tree + auto-memory verbatim (migration), with HIDE's own precedence honored. See `HIDE_CLAUDE_CODE_CONFIGURATION_COMPATIBILITY.md`.

## 8. The durability gene - resume anything, never lose work

Every message/tool call/result streams to `~/.claude/projects/<sanitized-cwd>/<session-id>.jsonl` the instant it happens, so exit/crash/reboot cost nothing. [DOCUMENTED] `--continue`/`-c` (last in dir), `--resume`/`-r` (searchable picker with Space-preview, worktree/branch/all-project scope widening, pasted PR URL), `/resume` in-session, `--from-pr`. [DOCUMENTED] Resume restores conversation, model, permission mode (excl. plan/bypass), active goal, non-expired scheduled tasks; some flags must be re-passed. Background sessions run detached under a supervisor daemon (`roster.json`, `jobs/<id>/state.json`) surviving sleep. [DOCUMENTED] Cross-interface resume is **not** unified - CLI and IDE share history; web is separate. [verifier]

Parity obligation: append-per-turn transcript, one-command resume + smart picker, background supervisor. Hawking superiority: resume the **warm state capsule** (no re-prefill), unlimited local retention (no 30-day clock).

## 9. Love-weighted ranking (which genes carry the love)

Ranked by how much of the retention/love each gene drives (INFERRED from the love/pain facet + community sentiment, cross-checked against what pain users *tolerate* to keep it):

1. **Steerable autonomy** (§1) - "feels like a capable teammate I can redirect." Highest love, lowest latency tolerance.
2. **Repo understanding + end-to-end execution** - reads/edits/runs tests/commits without babysitting; "holds the codebase in its head." [ANECDOTAL cluster, strong]
3. **Reversibility** (§4) - reversibility is what makes people *allow* autonomy.
4. **Plan gate** (§3) - prevents expensive wrong-direction rework.
5. **Legibility** (§5) - todos + collapsed tools keep long runs trustworthy.
6. **Memory/instructions** (§7) - the stickiness moat; low novelty, high lock-in.
7. **Trust-gradient modes** (§2), **discoverability** (§6), **durability** (§8) - table stakes that must be smooth.

The Apple-vs-Samsung reading: HIDE will lose if any *table-stakes* gene (2, 6, 8) is visibly rougher than Claude Code, even while HIDE wins on cost and state. **Polish parity on the boring genes is a gate for the exciting ones.**

## 10. What is model, what is harness (decomposition)

Separating what comes from the Claude model vs the harness matters because HIDE inherits a *local* model but can fully own the harness [INFERRED, aligned with the model/harness decomposition experiment the Bible §15 requires]:

| Loved property | Mostly model | Mostly harness | HIDE can fix without a better model |
|---|---|---|---|
| Repo understanding | model (long-context reasoning) | harness (index + context compiler) | **yes** - a strong index + reserve-then-fill compiler recovers most of it (both packed in HIDE) |
| Takes initiative / judgment | model | - | partly - needs a capable local coder (Qwen3-Coder-Next-class) |
| Steerable interrupt | - | harness | **yes** - pure harness; HIDE can beat it (no network to cancel) |
| Plan quality | model + harness | - | partly |
| Error recovery | model + harness (oracles) | - | **yes** - deterministic verify oracles (packed `hide-kernel`) |
| Pleasant tone / prose | model | harness (response-style) | needs local model tuning + a response-style eval |
| Memory / durability | - | harness | **yes** - entirely harness; HIDE's state capsules exceed it |

Conclusion: **the majority of the loved *workflow* is harness, which HIDE can reproduce and in several places exceed; the loved *judgment* is model, which gates on a capable local coder.** This is why the build ladder prioritizes reconnecting the harness spine before chasing model parity.

## 11. Anti-genome - what HIDE must NOT copy

- Claude Code's specific spinner vocabulary, product copy, and branding (clean-room boundary).
- The **usage meter / weekly-cap anxiety loop** - HIDE has no meter; do not import a dollar HUD. Replace `/usage` with performance telemetry (tokens/s, energy, context fill, best-of-N depth). [see love/pain map]
- Silent server-side harness changes - HIDE ships a version-pinned local runtime; behavior does not change unless the user updates. This is a trust *advantage* to preserve, not a behavior to mimic.
- The `#` shortcut naming (not current) and any behavior the verifiers downgraded to ANECDOTAL - implement the *capability*, not the unverified specific.
