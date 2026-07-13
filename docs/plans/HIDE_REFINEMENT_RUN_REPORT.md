# HIDE refinement run report

Goal-loop scoreboard for driving every audited facet toward 10/10. Unbounded wall-clock; stop-check is convergence, not a clock. Source of truth for scope: docs/plans/hide_refinement_roadmap_2026_07_05.md. House rules apply (no em/en dashes, no commit/push without approval, verify before claiming).

Baseline (2026-07-05, pre-loop): typecheck clean, 53/53 vitest green.

## Scoreboard

Score = current best estimate after the wave lands and verifies. "v" column: how verified (build = tsc+vitest, look = headless render, src = code-confirmed).

| Facet | Base | Now | v | Wave |
|---|---|---|---|---|
| Typography and hierarchy | 4.5 | 8.5 | build+look | A done |
| Color, surfaces, depth | 5.0 | 8.5 | build+look | B done |
| Layout, landing chamber | 3.5 | 7.5 | build+look | C done |
| Layout, IDE chamber | 7.0 | 7.0 | - | - |
| Motion and animation | 6.0 | 7.5 | src | C done |
| Copy and microcopy voice | 4.0 | 6.5 | build+gate | D done (char gate) |
| CSS design-system rigor | 4.5 | 6.5 | src | A-C partial |
| App architecture | 6.5 | 7.0 | build | transport seam started |
| Performance feel | 4.0 | 4.0 | - | F |
| Keyboard and focus | 3.5 | 4.5 | look | E (focus ring done) |
| First-run experience | 2.5 | 3.5 | cargo | sidecar boots now |
| Backend architecture | 4.5 | 4.5 | - | H |
| Rust testing | 7.5 | 7.5 | - | - |
| Ship-readiness | 2.0 | 4.0 | cargo+build | P0 partial |
| Privacy / egress verification | 5.0 | 5.0 | - | H |
| Product differentiation | 3.0 | 3.0 | - | (gated on H3) |

Verified this session: 101 FE tests (53 + 48 voice gate), tsc clean, `vite build` green, `cargo check -p hide-desktop` green, live preview console clean.

## Wave log

### Wave A - typography token substrate (DONE, verified build+look)
Landed: one tokenized type scale in theme.css (--fs-*, --lh-*, --ls-*, --fw-*); swept 98 raw CSS font-sizes + 41 inline tsx fontSizes to tokens (0 raw remain); killed fractional 11.5/12.5px; fixed both faux-bold 700s (ide.css:100, panels.css:83 -> 600/500); rewrote .t-* roles + added .t-metric/.t-ui/.t-small; body line-height; chat prose max-width 68ch; md heading ladder tokenized; hero now clamp(24,2vw,28)/500/-0.02em (verified 25.6px/500 live, was 34/600); xterm 12.5 -> 13.
Digest two-tier: split the 8-metric grid into 4 primary totals (24px/500) + a quieter secondary row (16px/--text-2 for streaks/peak/model), killing the "8 identical numerals, model set as metric" flatness the owner flagged. Verified in DOM (4 substats render), fresh-load console clean, 53/53 tests green.
Remaining for 10: unify the label role across sidebar__head/branch__state/statetl (still carry own tracking); baseline rhythm pass. Deferred to CSS-rigor pass.

### Wave B - color temperature + material (DONE, verified build+look)
Landed: warmed the surface ramp (concrete-1..4 -> #0f0e0d/#161513/#1e1c1a/#262421, verified live) and glass tints to the bone axis; void #070707 kept pure per ruling. Swept every pure-white rgba(255,255,255,x) -> warm bone rgba(244,242,238,x) across theme.css + Monaco + xterm (0 cool whites remain); collapsed hairlines to --line 0.07 / --line-strong 0.13. --depth is now hairline + lit top rim (--rim); added --depth-float halo for the 5 floating glass layers. Focus ring is now bone-on-void (--focus-ring, ~18:1, passes WCAG 1.4.11) replacing the invisible 11%-white hairline. AA fixes: --mute #7d7b72, --text-3 #8a887f. .volume is a lit mass (shadow-as-border, no literal 1px border). Heatmap empty cells recess to --void (were wireframe boxes on same-color card). Synced stale Monaco/xterm text-3 #6E6D68 -> #8A887F. tsc clean, console clean, 53 tests green.
Remaining for 10: unify the 4 border mechanisms to one; deeper per-panel elevation audit in the IDE (glass already elevates chrome, editor stays void by doctrine); fold Monaco/xterm surface alphas into the concrete ramp tokens.

### Wave C - layout, spacing, grid (DONE, verified build+look+prod)
Landed: --col-stage 760px + --stage-pad-x tokens; hero, digest, and composer now share one column edge (verified live, card and composer edges aligned); recovered vertical space (hero margin ma-14->ma-10, scroll top ma-18->ma-14) and added a bottom mask-image fade so the fleet dissolves into the void instead of hard-clipping at the fold; collapsed 6 radii to 3 (composer/bubble alias the canonical set); motion drift purged (chat.css + theme.css hardcoded 0.12s/200ms/320ms -> tokens, HunkReview inline 480ms -> var(--dur-door), dead @keyframes light-travel removed). `vite build` green (23s).
Remaining for 10: heatmap fill-container + 7d form; active-chat panelbar/composer 40px alignment; off-grid 3/5/6px purge; ~400 lines dead/shadowed CSS deletion; @layer cascade.

### Wave D - copy voice + banned-char CI gate (DONE, verified 101 tests)
Landed: new src/voice.test.ts walks the whole tree, strips comments, fails on any em/en dash, middot, ellipsis, or bullet in rendered copy (48-file gate, green). Fixed all 7 real violations: StateTimeline em-dash tooltips reworded; ContextStack/Editor/Explorer middots -> slash/comma/#; Onboarding/Settings ellipses removed; CodeActions truncation marker -> [cut]. Fixed the stale fleet empty-state that instructed a removed "try 3/5/8" toolbar control.
Remaining for 10: one casing regime (sentence-case controls / lowercase values); canonical phrase table ("Describe the work", Explorer vs Navigator); humanize raw JSON/err.message notices; ship docs/hide-bible/VOICE.md.

### P0 ship-blockers (PARTIAL, verified cargo+build)
Landed: transport now defaults to LIVE in a production build (import.meta.env.PROD) so the DMG can never silently ship the mock; dev/preview stays on mock. Sidecar argv fixed: hide-desktop now spawns hide-serve with `--port 8744` + HIDE_SERVE_ADDR env instead of the rejected `--addr` (was exiting the engine at boot); `cargo check -p hide-desktop` green.
Remaining P0: loopback bearer token + CORS lockdown (RCE); gate Bypass server-side + sandbox the exec path; workspace-root passed to sidecar; frontend CI job + build tripwire; global error handlers + sidecar log file.

### UX + branding pass (DONE, verified build+tests+look)
Owner-requested changes, all verified live (100 tests, tsc, vite build green, console clean):
- Removed the first-run onboarding modal entirely (deleted Onboarding.tsx, stripped folderOpened/chooseFolder from App.tsx). The working folder is now added from the composer.
- The composer plus button opens an "Add context" popover (Add folder / Attach files), Claude Code style. Add folder calls the native picker (pickWorkspaceFolder) and sends open_folder; in web/dev without the Tauri dialog plugin it notices "folder picker opens in the desktop app" (the plugin registration is the remaining native piece, Wave G).
- Panel switcher is now icon-only (tighter spacing) and gained two panels: Tools (live tool-progress feed, real store data) and Artifacts (files the run produced, derived from the diff). Terminal / Diff / Preview / Tools / Artifacts, all verified rendering.
- Logo is now just the h everywhere: hero uses LogoH (was LogoMark with the ball/globe); regenerated logo/hide-mark.svg (h centered), logo/hide-mark.png, logo/hide-icon-1024.png, app/public/favicon.png, and the full Tauri icon set via `tauri icon` (icon.icns/.ico + all sizes). The .psd is binary and not programmatically authored; the SVG is now the canonical vector source.
First-run bumped 3.5 -> 4.5 (friction removed, folder-add path exists; native picker plugin still pending).

### Claude Code geometry-fidelity pass (DONE this batch, verified build+tests+look)
Reference: the owner's screenshots of the real Claude Code desktop app. Scope: geometry/shape/layout only, NOT theme/color (grayscale/mono doctrine stays). Mapping: HIDE's Chat view is the analog of Claude Code's Code view (transcript + right panel). Landed, all verified (tsc, 100 tests, vite build green, console clean):
- CC-1 mode switcher: moved Chat/Code into a segmented control at the TOP of the sidebar (was in the window bar); toolbar keeps it only in the Code chamber (which has no rail). Fidelity 5 -> 9.
- CC-3 tool rows: replaced the boxed "Tools" pill section with inline collapsible past-tense-shaped rows in the transcript flow ("Reading the workspace for context ›", chevron disclosure). Fidelity 4 -> 8.5.
- CC-5 composer: moved the permission pill to leftmost (before +), matching Claude Code order. Fidelity 6.5 -> 9.
- CC-7 sidebar: added Artifacts row; dropped the right-aligned age on recents (Claude Code shows none). Fidelity 5.5 -> 8.5, recents 7 -> 9.
- CC-2 title bar: added the project-name + branch tag to the window bar in chat mode (`hawking · main`), matching Claude Code's session identity. Fidelity 4 -> 7 (action-icon cluster still lives in the conversation header, not the title bar).
- CC-4 branch/PR bar: added a right-aligned "Create PR" button to the composer branch row (registered create_pr custom intent). Diffstat honestly omitted (no aggregation exists; would be a fabricated number). Fidelity 4 -> 7.5.
Overall Claude Code geometry fidelity: ~6 -> ~7.7.
Remaining for 10 (each is feature-deep or architectural, not pure geometry): CC-2 relocate the panel action-icon cluster into the title bar (needs lifting panel state from Home to App); CC-6 working-tree multi-file diff list (HIDE's diff is single-file hunk review, a different model; a multi-file list would need real multi-file diff data, not a fake); CC-4 real +N -N diffstat (needs aggregation); a "More" sidebar disclosure.

### Track 1 - make the workspace real (IN PROGRESS, verified cargo+tests)
Owner asked to run all four "make it better" tracks in sequence. Track 1 landed increment, verified (cargo check green, hide-backend 70 tests pass, FE tsc + 100 tests green):
- Registered the Tauri dialog plugin (tauri-plugin-dialog, `.plugin(...)` in main.rs, capabilities/default.json granting dialog:allow-open). `cargo check -p hide-desktop` green. This is the concrete unblock for the native "Add folder" picker (pickWorkspaceFolder's window.__TAURI__.dialog.open now has a backing plugin). True e2e picker is desktop-only verification.
- open_session backend handler: republishes a past session's recorded ui_events on the live bus so the FE (which adopts a session off any event's session_id) switches to it and re-renders the transcript. Real events from the log, not fabricated. Added #[derive(Clone)] to BackendReplayService + spawn_open_session. Was a dropped intent before.
- open_folder: recorded honestly (logged by the router), NOT faked into a misleading workspace switch. The deep re-root (engine serving files/git from the new folder) is owned by the desktop shell relaunching the sidecar with the new root.
Remaining for Track 1: open_folder deep re-root (Tauri shell relaunches sidecar with chosen root); real fs file tree from the workspace root (replace mock); git status/branch/worktree binding to the actual repo. These need the sidecar-relaunch architecture + real-app verification (desktop-only).

### Track 2 - make it safe + shippable (core DONE, verified cargo+tests)
- Closed the drive-by RCE: replaced CORS `Any` in hide-serve with an explicit origin allowlist (tauri://localhost + the Vite dev origin, HIDE_ALLOW_ORIGIN override), locked methods to GET/POST/OPTIONS and headers to content-type. Added a test asserting the app origin is granted and a foreign origin is NOT (7 hide-serve tests green). Locked hawking-serve's default bind 0.0.0.0 -> 127.0.0.1.
- Global error handlers: main.tsx now routes window.onerror + unhandledrejection into the bounded notices strip so uncaught errors never vanish in a bundled .app.
- Frontend CI: added a `frontend` job to ci.yml (pnpm typecheck + test + a live-transport build) with a tripwire that fails if any mock identifier survives the production bundle (verified locally: the live build tree-shakes the mock; dist has zero mock identifiers).
- Bypass safety: App.tsx no longer resumes Bypass across a restart (a persisted bypass would auto-approve gated commands unattended on next launch); such a session boots back to "ask".
Remaining Track 2 (deep backend security): gate Bypass server-side so the backend refuses to auto-run classified-dangerous commands even under bypass; route the FE terminal exec through the sandboxed shell tool.

### Track 3 - make it feel instant (headline DONE, verified build+look)
- Lazy-loaded Monaco out of the chat boot path. Root cause found: Terminal.tsx imported MONO_FONT (a plain string) from monacoTheme.ts, which imports all of Monaco -> the terminal (used in the chat side panel) dragged the ~4.5MB editor into the main chunk. Extracted MONO_FONT to a Monaco-free ideConstants.ts, and made EditorArea a React.lazy import behind a Suspense boundary. Result: main bundle 4.47MB -> 0.65MB (85% cut, verified via vite build); Code lazy-loads Monaco on entry and renders the full IDE; console clean in both chambers.
Remaining Track 3: incremental streaming render (Conversation.tsx re-parses the whole message per token), cold-start splash + font preload, manifestRing cap, pin-aware auto-scroll.

### General UX pass (DONE, verified build+tests+look)
Hands-on usability/feel audit of the live app (beyond Claude Code fidelity), then scaffolded all 6 punch-list items. All verified in the running app (tsc, 101 tests, vite build green, console clean):
- Error surfacing (self-inflicted regression from the Track 2 global handler): the handler was surfacing a benign Monaco DiffEditorWidget disposal race as persistent red status-bar text. Added a BENIGN allowlist (Monaco model-disposal, ResizeObserver loop, cross-origin Script error, Load failed) + dedupe so framework noise never becomes a notice, and made notices auto-expire (errors 8s, info 4s) via a new dismissNotice action. Status bar now clean.
- 7d digest reshape: the heatmap collapsed to a lonely single column in a 60%-empty card; 7d now renders the last seven days as a full-width horizontal strip (verified live, fills the card).
- Fleet order: when a run is live the RUNNING fleet now sits ABOVE the digest (seen first, not buried below a tall card at the fold).
- Fresh-chat anchor: a short conversation now bottom-anchors to the composer (void above, grows up) instead of floating at the top with void below. chat-scroll -> flex column + message-list margin-top:auto.
- Narrow-width nav: the rail hides below 860px (under the 880 min-width); added a toolbar Chat/Code fallback that appears only when the rail is hidden, so chamber switching is never unreachable (verified at 800px: rail hidden, fallback shown).
- Discoverability: panel icons already carry native tooltips + the open panel's header names it; deeper shortcut hints wait for the keymap system (Track 4).
UX scorecard (hands-on): overall ~7/10 -> the sharp weak spot (error feel 3/10) and the empty-state / layout rough edges are now addressed.

### Track 4 - make it keyboard-first (NOT STARTED)
Central keymap registry (fix the typing-eats-diff-hunks bug), Escape stack, focus restoration, native macOS menu + shortcuts.

## Next waves (not yet started)
- E: keyboard registry (kill the typing-eats-hunks bug), Escape stack, shortcut map + native menu, focus restoration. (focus ring already done)
- F: perf (lazy-load Monaco out of chat boot, incremental streaming render, coalesce token batches, cap manifestRing) + FE transport seam sealing + connector typing.
- G: first-run (dialog plugin, open_folder/open_session handlers, first-model step, honest empty digest).
- H: backend (auth + CORS, sandbox exec, real worktree isolation, event-log index, blocking-fs off async) + verifiable egress (NETWORK_LEDGER, loopback default, air-gap toggle, CI egress gate).
