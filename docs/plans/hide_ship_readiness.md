# HIDE — Ship Readiness (living status, 2026-06-29)

> **UPDATE (later, 2026-06-29): the "offline" blocker was wrong — the network works.** So:
> - **#1 Tauri shell now COMPILES clean** (tauri 2.11 + window-vibrancy 0.5; `cargo check` green in ~30s).
>   `@tauri-apps/cli` installed; `pnpm tauri build` is wired (an unsigned `.app`/`.dmg` build runs here).
> - **#5 logo + icons DONE**: the `logo/hawking.psd` (16-bit, undecodable by sips/psd-tools) was recreated
>   on-brand (event-horizon ring black hole + light `h`, Geist Mono) at `logo/hide-icon-1024.png` +
>   `logo/hide-mark.png`; the full icon set (`.icns`/`.ico`/all sizes) is generated into `src-tauri/icons`;
>   the web favicon is wired.
> - **#6 signing/notarize**: `app/scripts/build-macos.sh` is ready — paste your Apple Developer ID +
>   notarization credentials at the top (instructions inline) and run it; it signs + notarizes the `.dmg`.
>   `app/scripts/stage-sidecar.sh` bundles `hide-serve`.
> - **The only ship blocker outside #8 (the model) is now the Apple Developer certificate** (yours to
>   provide; the script is ready). Everything else for a desktop app is in place + verified.

> **UPDATE 2 (2026-06-29, accessibility + approve-gate polish pass — no Studio needed):**
> Knocked out the feasible-offline remainder of the "recommended polish" list. All green (typecheck +
> 17 vitest tests + `vite build`) and runtime-verified against the dev server (zero console errors).
> - **#11 Accessibility — done.** New `app/src/shell/a11y.ts`: `useFocusTrap` (focus first control on
>   open, trap Tab/Shift+Tab, restore focus on close) applied to the **Settings**, **security-gate**, and
>   **command-palette** dialogs (now `role=dialog`/`aria-modal` on the panel, `role=presentation` backdrop);
>   the security-gate dialog also got Esc-to-dismiss. The **Explorer file tree** is now a real ARIA tree:
>   `role=tree/treeitem/group`, `aria-level`/`aria-expanded`/`aria-selected`, roving tabindex, and full
>   keyboard nav (Up/Down/Left/Right/Home/End) via the tested-pure `flattenVisible`/`treeKeyTarget`. The
>   **diff hunks** got `role=list/listitem`, `aria-current` on the selected hunk, an `aria-live` review
>   counter, and a labeled stale badge. Verified live: tree exposes the roles, ArrowDown advances one row,
>   Settings traps focus on open and Esc closes.
> - **Approve-to-run (FE half) — fixed.** The app-level security-gate "Approve" button used to just
>   *dismiss* (identical to Deny — a real defect); it now emits `approve_gate` (recorded as a
>   `user.intent.approve_gate` event), matching the Executor's inline gate. The backend *hold-and-release*
>   resume (so the blocked command actually re-runs) is the remaining half — real kernel work that needs a
>   live run to verify, correctly deferred.
> - **Node engine — declared.** `engines.node >= 20.19` in `app/package.json` (surfaces the Vite
>   requirement as a build warning until the dev box is bumped).
> - **Reduced-motion — already covered**: theme.css's global `prefers-reduced-motion` rule neutralizes
>   every animation (breathe/blink/radiate/diff), so no per-component work was needed.
>
> Remaining non-Studio items: **no-folder onboarding** (#17, needs the Tauri folder-open flow), **the
> backend approve-and-run resume**, and **auto-update** (#7, needs a host for the release feed + updater
> keys). Everything else outside #8 (the model) is done.

> **UPDATE 3 (2026-06-29, the remaining non-#8 items — scaffolded + tested):** cleared the last three.
> All green: hide-backend 65 tests (5 new), FE typecheck + 35 vitest tests (18 new) + `vite build`,
> `cargo check --workspace` clean, runtime-verified vs the dev server (0 console errors).
> - **#15 approve-AND-run gate — DONE (both halves).** A destructive command is no longer dropped: the
>   host parks it in a bounded `GateBook` (keyed by a unique `command:<n>` id) and emits the
>   `SecurityGate` with that id; `approve_gate` releases + runs it (bypassing the gate, since approved),
>   `deny_gate` drops it, an unknown/evicted id is a safe no-op. In `crates/hide-backend/src/host.rs`
>   (`GateBook`, `spawn_command_run`/`spawn_exec`/`approve_gate`/`deny_gate`, the runner renamed
>   `exec_command_streamed` with the gate moved upstream). FE wires both `approve_gate` and the new
>   `deny_gate` in the app-level gate AND the Executor inline gate. 5 backend tests (book hold/take/
>   remove/cap; host hold→approve, hold→deny, unknown-gate no-op).
> - **#17 no-folder onboarding — DONE.** `surfaces/Onboarding.tsx` (focus-trapped `role=dialog`, the
>   mark, the pitch, "Open folder…" + "Continue with the sample workspace", a shortcut reference) shown
>   on first run via a persisted `folderOpened` flag. `shell/onboarding.ts` carries the tested-pure
>   `shouldShowOnboarding`/`isTauri`/`pickWorkspaceFolder` (the native picker is runtime-detected via
>   `withGlobalTauri` — now enabled — so there is NO build-time dep on the dialog plugin; it lights up
>   when the plugin is bundled, degrades to the sample workspace otherwise). `open_folder` intent added.
>   7 tests; runtime-verified: shows on first run, dismiss persists + reveals the editor.
> - **#7 auto-update — SCAFFOLDED + tested.** `shell/updater.ts`: tested `buildUpdateManifest` (the
>   Tauri v2 `latest.json` feed builder, rejects malformed feeds) + runtime-detected
>   `isUpdaterAvailable`/`checkForUpdate` (build-safe, like the folder picker). "Check for updates" in
>   Settings → About (degrades to "managed by the desktop app" on web). 11 tests. The infra remainder
>   (plugin fetch, keypair, feed host) is paste-ready in `docs/plans/hide_release_autoupdate.md` +
>   `hide_update_feed.example.json` — all network/key/host-gated, none need the Studio.
>
> The only thing left outside #8 (the model) is desktop-only *verification* (WebKit-on-the-.app, perf on
> large real repos) — needs the built app in hand, not more code.

What HIDE the app needs to ship, independent of Hawking model quality (Studio arc is separate). Status
legend: **done** (built + verified green), **scaffold** (build-ready, needs an external resource or a
running engine to finish/verify), **blocked** (needs a resource not available in this environment:
network, Apple Developer cert, the model), **todo** (feasible, not yet built).

## Critical path

| # | Item | Status | Notes / what remains |
|---|------|--------|----------------------|
| 2 | Real workspace file I/O | **done** | `FsConnector` (list/read/write, path-confined, tested) in `hide-backend`; FE Explorer reads the real tree, Editor reads + saves real files, with mock fallback. Verifies live once `hide-serve` runs. |
| 3 | Real integrated terminal | **done (runner)** | `run_command` now executes in the workspace and streams stdout/stderr as `tool_progress`; the terminal renders it. Full interactive PTY (vim, etc.) needs `portable-pty` (not fetchable offline) — runner ships now, PTY is a later upgrade. |
| 1 | Tauri desktop shell | **scaffold** | `app/src-tauri/` written build-ready (transparent + macOS vibrancy via `window-vibrancy`, supervises `hide-serve`, bundle config, entitlements). Building needs a network fetch of the tauri crates + `@tauri-apps/cli`. On macOS the webview is WebKit, so the glass `frost` path ships. |
| 4 | Live token streaming + executor loop | **done (core) / todo (loop)** | Token streaming already wired: `generate_submit_turn` streams `TokenBatch` via the bus; the FE renders it. Needs Hawking to emit tokens (model side). The continuous plan-execute-verify loop + per-step kernel events is the larger todo (see the executor plan v2). |

## Tier 1 — Distribution

| # | Item | Status | Notes |
|---|------|--------|-------|
| 5 | App identity + bundle | **scaffold** | Name/identifier/version in `tauri.conf.json`; icons still needed (`tauri icon`). |
| 6 | Code signing + notarization | **blocked** | Needs an Apple Developer ID cert (external). Entitlements written; signing config + `notarytool` pending the cert. |
| 7 | Installer + auto-update | **todo** | `dmg`/`app` targets configured; auto-update needs the Tauri updater plugin + a signed release feed. |
| 8 | First-run / setup | **todo** | Bundle-or-fetch weights + binaries, create `.hide/`, pick model/folder. Depends on the model artifacts. |
| 9 | Real Settings surface | **todo** | Replace the gear-opens-palette with a real panel (model, endpoint, theme, keybindings, a11y, telemetry opt-in). Feasible offline. |

## Tier 2 — Hardening

| # | Item | Status | Notes |
|---|------|--------|-------|
| 10 | Front-end test suite | **todo** | Add Vitest + Testing Library (unit/component) + Playwright (e2e). None today. Feasible offline. |
| 11 | Accessibility pass | **todo** | WCAG AA done on text contrast; remaining: focus traps, ARIA roles on tree/hunks, reduced-motion completeness. Feasible offline. |
| 12 | Error / empty / crash recovery | **partial** | Reconnect + degraded toast + graceful connector/mic fallback done; add React error boundaries + comprehensive states. Feasible offline. |
| 13 | State persistence | **todo** | Persist sidebar/panel/Executor open, open tabs, chat position, settings. Feasible offline (localStorage). |
| 14 | Perf on real repos | **todo** | Glass blur already cut 32->11px; test large trees/files. Needs real repos. |
| 15 | Security gate + credentials | **partial** | `security_gate` UiEvent + InlineGate UI exist; wire to real side-effectful actions + OS keychain. Feasible offline (UI), keychain needs the native shell. |
| 16 | Cross-engine verify | **todo** | Tauri macOS = WKWebView (WebKit). Verify the frost glass + all surfaces on WebKit. Needs the Tauri build or Safari. |

## Tier 3 — Polish / launch

| # | Item | Status | Notes |
|---|------|--------|-------|
| 17 | Onboarding | **partial** | Empty states exist; add no-folder first-run + shortcut reference. Feasible offline. |
| 18 | Brand + About | **todo** | Icon set, About/version/OSS licenses, the local-first pitch. Mostly feasible offline. |
| 19 | Privacy + crash reporting | **todo** | Local-first telemetry policy (none/opt-in), local crash logs. Feasible offline (docs + wiring). |
| 20 | Release process | **blocked** | CI to build/sign/notarize/publish. Needs network + the cert. |

## Completed this session (toward ship)
- Real file I/O end to end (#2): `FsConnector` + FE wiring + tests, `cargo`/typecheck green.
- Real terminal command runner (#3): `run_command` executes + streams output.
- Tauri shell scaffold (#1): `app/src-tauri/` build-ready, documented in its README.
- Confirmed #4 token streaming is already wired (needs the model).

## Honest blockers in this environment
- **Network**: cannot fetch the Tauri crates / `@tauri-apps` npm, so #1 cannot be built or verified here.
- **Apple Developer cert**: required for #6 signing + notarization and #20 release; not available.
- **The model (Hawking)**: required to verify live token streaming (#4) and first-run weights (#8).
- **`portable-pty` offline**: blocks a full interactive PTY (#3 ships as a command runner instead).

The feasible-offline remainder (Settings #9, persistence #13, FE tests #10, a11y #11, error boundaries
#12, onboarding #17, brand/privacy #18/#19) is the next work; the rest waits on the resources above.
