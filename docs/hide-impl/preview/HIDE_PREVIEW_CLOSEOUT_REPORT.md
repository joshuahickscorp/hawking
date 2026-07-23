# HIDE Preview Closeout Report

Branch `build/hide-impl-2026-07-19`, base `4fbca8bc`. Written 2026-07-20. House rule: hyphens and
parentheses only, no long dashes.

## Verdict first

**This is an HONEST BLOCKED CLOSEOUT, not a successful one. Nothing was committed.**

Step 9 of the brief gates the commit on four conditions. Two are met, two are not:

| Gate | Status |
| --- | --- |
| The owner-preview workflow passes | MET. 12 PASS / 0 FAIL / 1 SKIPPED against the live server |
| Authoritative verification is green | MET. 669 Rust tests, 389 frontend tests, clean builds |
| The fourth adversarial pass passes | NOT MET. It FAILED with 5 criticals and 12 majors |
| No critical remains | UNVERIFIED. All 5 criticals were then fixed, but no pass has re-checked them |

The working tree is preserved exactly as it stands: 235 changed or added paths, zero commits, HEAD
still identical to `main`.

## 1. What does HIDE currently look like?

It launches and it is recognisably one product: an Ando-grayscale shell in Geist Mono, a Chat chamber
as the front door and a Code chamber as the IDE, a left rail of sessions, a bottom status bar that
always states the transport, the phase and whether a model is present. See
`HIDE_PREVIEW_GALLERY.md` and `screenshots/`.

The most important visual fact is what it refuses to do. Pointed at a real empty backend with no
model it shows zeros, `No sessions yet`, `Runtime not ready`, `Local engine is down, no model
configured`, and `Problems: no static analysis has run in this session yet`. The same build in mock
mode shows 1,182 sessions and 222.9M tokens. The difference between those two screens is the whole
point of the consolidation campaign.

## 2. Which surfaces are visibly functional?

Verified by driving the running application and the live server:

- Home and the digest, reading real durable counts (0 on a fresh backend, 50 sessions and 2 active days after the workflow).
- The Code chamber: the real repository tree with git status markers, real files opening in Monaco.
- The terminal state row: real env, cwd, `sandbox confined`, process and exit state.
- The status bar: real transport, phase, runtime state, and an honest Problems counter with a real producer.
- The approval surface: every `ApprovalPolicy::Ask` command was HELD at a gate and only took effect after an explicit approval, proven on disk.
- Reload hydration: the open tab, transcript, checkpoint id and write lease survive a browser reload.

Proven over the wire but not persistable as screenshots in this environment: diff and diff-chip
projections (125 frames each, matching the frontend parser exactly), checkpoint create, code rewind
reverting real bytes, fork, sandboxed process start, stream, attach, stop and artifact capture,
transcript search returning typed hits, and review-receipt export.

## 3. Which surfaces remain partial?

- ContextStack is now reachable and mounted, but every publisher of the `context_manifest` projection is inside the model turn, so on a model-free host it is honestly empty rather than populated.
- Plan: the projection has no producer without a served model.
- Fleet and the background-work chip: the `fleet` projection has no producer at all; the surface is dead on a live host.
- `build` and `test` projections are bound by ContextStack with no Rust publisher.
- `run_command` is a wire-reachable workspace write that produces no diff and is not bounded by the lease scopes.

## 4. Did the task-scoped write policy solve the edit deadlock?

Yes, and it was the single highest-value fix in the campaign. Before it, the shipped default
(`workspace_write_default = Ask`) refused every write through the host dispatcher, including the
agent's own edits, so the diff store stayed empty and the product could not do its job.

The lease is granted only through an `ApprovalPolicy::Ask` command (held on every channel), enforced
in exactly one place (`GateReleaseAware::evaluate`), narrow by construction (only `fs.write` with
fully predicted effects inside declared scopes; `shell.exec`, `git.write`, network and `mcp.call` are
different capability kinds and cannot ride it), bound to the granting session with a 30 minute TTL,
containment-checked on the REAL path so a symlink cannot escape, revoked by nine triggers, and
invalidated on restart because it is process memory only.

Proven live: a leased edit lands and produces a real diff; an out-of-scope write is still refused
while the lease is active; the lease revokes on its triggers.

## 5. Did the fourth adversarial pass pass?

No. It returned FAIL on all four areas: 5 criticals, 12 majors, 11 minors.

The criticals were: the agent write path still recorded nothing (the previous round had fixed only
the editor path); `approve_gate` reported success for an unknown, evicted or failed gate;
`open_session` had no frontend consumer; ContextStack could not be opened on a model-free host; and
receipt invalidation was dead on every wire-reachable route.

All five were then fixed, several at genuinely shared roots (recording moved down into the dispatcher
behind a `DispatchObserver` so any dispatch records; `GateBook` now refuses past capacity instead of
silently evicting; the catch-up guard that made a first page load never backfill was removed). But
the pass itself did not pass, and no fifth pass has verified the fixes. Given that every previous
round introduced regressions, that verification is mandatory before any completion claim.

Evidence for this: immediately after the fix round, a manual look at the running app found a
regression the round had introduced. Raw JSON payloads were being dumped into the user-visible notice
area, because the earlier fix for that class had been written as an exclusion list of one
(`if (c.kind !== "search_results")`) and new event traffic walked straight past it. Fixed and
verified live.

## 6. What tests passed?

| Suite | Result |
| --- | --- |
| Rust, 15 hide crates | 669 passed, 0 failed |
| Rust workspace build | clean |
| Frontend typecheck | clean |
| Frontend tests | 389 passed, 17 files |
| Frontend production build | succeeds |
| Command registry parity | catalog byte identical to the Rust golden |
| Live workflow against the running server | 12 PASS, 0 FAIL, 1 SKIPPED |
| Newly authored em or en dashes | 0 |
| Lint | NOT RUN. The repository has no eslint config and no lint script. No lint tooling was added to manufacture a green gate |

## 7. What remains model-dependent?

Every agent behaviour. No model was downloaded, staged, selected or loaded at any point. The agent
turn is the one workflow step that could not be exercised (`DEFERRED_MODEL_REQUIRED`), the kernel
turn is off by default, and the `plan` and `context_manifest` producers live inside the turn. Nothing
in this campaign is evidence about model quality, capability or parity, and no such claim is made.

Note the consequence for the biggest fix: recording now happens in the dispatcher, and a test drives
the exact dispatcher object the kernel is handed, but the step that CHOOSES the tool call needs a
model. So the agent path is proven from the dispatcher down, not end to end.

## 8. What remains non-model work?

- A fifth adversarial pass verifying the five critical fixes (the blocking item).
- `run_command` produces no diff and is not lease-bounded.
- The `fleet`, `build` and `test` projections have no producers.
- Durable replay emits two `tool_progress` frames per dispatch where the live bus emits one, so a reconnected timeline doubles its steps.
- `compare_to_checkpoint` skips the integrity check every other checkpoint verb performs.
- Accessibility: recents are all labelled `session`, Explorer file rows expose an empty accessible name, a listbox is nested in a menu, and a disclosure lacks `aria-expanded`.
- Cosmetics: a transient `event socket error; reconnecting` on first connect, status-bar clipping and toast overlap at narrow widths.

## 9. What known defects are accepted?

- The lease is a single process-global slot, so two concurrent approved tasks would need a map keyed by session.
- `hide-tools`' applier bottoms out in `std::fs::write`, which follows symlinks; containment is checked before it, not inside it.
- The loopback transport has no authentication and gate ids are a predictable counter, so any local process can answer a gate. This is a localhost-trust assumption, and it should be stated to anyone running the preview.
- `shell/onboarding.ts` keeps three exports with no production caller; there is no first-run onboarding surface and the doc comment now says so.
- The dangerous-command denylist matches `argv[0]` only; the OS sandbox is what actually holds.

## 10. Readiness verdicts

| Level | Verdict | Reason |
| --- | --- | --- |
| Owner preview | YES, with a caveat | It runs, it renders real state, and it does not lie about what it does not have. Run it against a scratch repository, not a real one, because of the localhost-trust assumption |
| Internal alpha | NO | Not until a fifth adversarial pass verifies the five critical fixes, and not while `run_command` can write outside the diff and lease spine |
| External alpha | NO | No agent behaviour has ever been exercised; the product's core loop is unproven end to end |
| Merge to main | NO | The branch has zero commits, the fourth pass failed, and the campaign's own gate is unmet |

## 11. The smallest next action

Run a fifth adversarial pass scoped to exactly the five criticals fixed in the final round, plus the
raw-notice regression. If it comes back clean, the commit gate is met and the milestone can be
committed to `build/hide-impl-2026-07-19` and pushed. If it finds another regression, that is the
sixth consecutive round in which a fix introduced one, and the right response is to stop fixing and
freeze the branch as a review artifact instead.

## 12. Process lesson, recorded because it repeated five times

Every round of this campaign fixed one call site instead of the shared path, and every adversarial
pass caught it: `save_file` fixed while the agent dispatcher was left broken; a raw-JSON notice fixed
for one event kind while the fallback stayed; a write policy fixed for one release arm. The fixes
that held were the ones that moved the behaviour DOWN into the single place every caller routes
through: the dispatcher, the permission engine, one `effect_failed` helper covering ten blocks.

And the defect that mattered most was invisible to three static passes and 648 green tests. It took
starting the binary and driving it with a real client to find that no wire-reachable write produced a
diff, because every test proved that property by calling an internal function no client can reach.
