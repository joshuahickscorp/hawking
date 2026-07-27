# HIDE Preview Gallery

First honest owner-viewable preview of HIDE, captured 2026-07-20 from a real running build on branch
`build/hide-impl-2026-07-19`. House rule: hyphens and parentheses only, no long dashes.

## How this was run

| Item | Value |
| --- | --- |
| Backend | `target/debug/hide-serve /tmp/hide-preview-repo --port 8744` (real, not mocked) |
| Frontend | `vite --port 5273` with `VITE_HIDE_TRANSPORT=live` and `VITE_HIDE_BASE=http://127.0.0.1:8744` |
| Workspace under test | `/tmp/hide-preview-repo`, a real git repo (`src/pool.rs`, `src/retry.rs`, `README.md`) |
| Operating system | macOS (darwin), Apple Silicon |
| Theme | the shipped dark grayscale (no theme switch was exercised) |
| Route | single route, `http://localhost:5273/`; HIDE is an SPA whose surfaces are state, not URLs |
| Model | NONE. No model was downloaded, staged, selected or loaded. The app reports this itself |
| Transport shown | `live` in every shot below unless the caption says `mock` |

### Transport honesty, worth stating

Dev defaults to the MOCK transport so the app runs with no backend; a production build defaults to
`live` so a shipped app can never silently ship the mock (`app/src/ipc.ts`). Every capture below was
taken with the override set to `live`, so what you see is real backend state. The first screenshot
taken during this campaign was accidentally in mock mode and showed a fabricated digest (1,182
sessions, 222.9M tokens) plus invented fleet cards; that is recorded here only to make the difference
explicit, and it is why the status bar prints the transport at all times.

## Capture method, and its honest limit

Two methods were used, and each shot says which:

- `headless` means Google Chrome headless captured a real PNG to `screenshots/`. Reproducible, on
  disk, and used for everything that is reachable on first paint.
- `session` means the state was driven and observed in a live browser session (clicking through the
  running app) and is described here rather than stored as a file.

The limit, stated plainly: HIDE surfaces are click-driven application state, not URLs, and no browser
driver was available in this environment (Node 20.17 has no global WebSocket and neither `ws`,
Playwright nor Puppeteer is installed; adding one was out of scope for a closeout). So the
interaction-dependent surfaces below are `session` captures, verified by a human-readable
accessibility tree and screenshots taken during the session, not by files in this directory. The
deterministic proof for those same surfaces is the live workflow receipt
(`HIDE_PREVIEW_WORKFLOW_RECEIPT.json`, 12 PASS / 0 FAIL / 1 SKIPPED against the running server), not
this gallery.

## Files on disk

| File | Viewport | Method | State |
| --- | --- | --- | --- |
| `screenshots/01-home-live-1440x900.png` | 1440x900 | headless | Home, live transport, real backend |
| `screenshots/02-home-live-narrow-900.png` | 900x760 | headless | Home, narrow window |
| `screenshots/03-home-live-narrow-620.png` | 620x860 | headless | Home, very narrow window |

## What each surface actually showed

### 1. Home, empty workspace (headless, on disk)

Title bar reads the REAL workspace name `hide-preview-repo` and branch `main`. On a freshly seeded
backend the digest reads `0 SESSIONS / 0 MESSAGES / 0 ACTIVE DAYS`, streaks `0d`, recents `No
sessions yet`, and the footer line `Nothing left your machine.` After the live workflow ran, the same
surface read `50 SESSIONS / 0 MESSAGES / 2 ACTIVE DAYS` with two lit heatmap cells, all of it real
durable state (881 events in the log at capture time). The composer placeholder is `Runtime not
ready`, a toast says `Local engine is down, no model configured`, and the status bar prints
`phase: idle | no model reported | live transport | Down`. Nothing is invented.

### 2. Home, narrow windows (headless, on disk)

At 900 and 620 CSS pixels the sidebar collapses, the Chat and Code tabs move into the top bar, the
digest reflows, and no control becomes unreachable. Two minor defects are visible and recorded below.

### 3. Code chamber, real repository (session)

Explorer lists the REAL tree from `/tmp/hide-preview-repo` (`src/`, `pool.rs`, `retry.rs`,
`README.md`) with git status markers. The editor shows the honest empty state `Open a file` rather
than auto-opening anything. Opening `src/pool.rs` renders the real file content in Monaco with syntax
highlighting and the breadcrumb `src > pool.rs`, proving live backend to fs connector to editor end
to end.

### 4. Terminal, session-aware process surface (session)

The terminal state row is entirely real and reads
`env hide-preview-repo @ main | cwd /tmp/hide-preview-repo | sandbox confined | process none | exit
not reported | task ses_...`. The header reads `hawking shell . sandbox confined . output streams
live`. During the live workflow a sandboxed process started, streamed three `tool_progress` rows,
attached, stopped and was captured as a durable artifact.

### 5. Status bar and diagnostics (session)

`Problems` reports `no static analysis has run in this session yet` and offers to open the
diagnostics detail. It is NOT a fabricated `0/0`; that mock was removed during the consolidation
campaign and the counter now has a real producer (`run_static_analysis` publishes a `diagnostics`
projection).

### 6. Permission and approval state (session)

The permission control is explicit: `Permission mode, Ask each step. Every gated step waits for your
approval. Select to switch to Bypass`. During the live workflow every `ApprovalPolicy::Ask` command
(repo trust, write lease, rewind, restore) was HELD at a gate and only took effect after an explicit
approval, proven on disk.

### 7. Surfaces NOT captured, and why

Diff review, ContextStack, plan, side chat, checkpoint timeline, background jobs, command palette and
the approval overlay were exercised over the wire and are proven by the workflow receipt, but they
are click-driven states that this environment could not persist as files. They are listed here as
uncaptured rather than illustrated with a fixture, which would defeat the point of the gallery.

## Defects visible in the preview

| Severity | Finding | Status |
| --- | --- | --- |
| major | `localStorage` persisted MOCK fixture tabs and open paths into LIVE sessions, so a live host reopened files that never existed and emitted real `user.intent.open_file` events for them | FIXED, persistence namespaced by transport, verified live |
| minor | The status bar prints `event socket error; reconnecting` for a moment on every connect before the websocket establishes, then recovers | open, cosmetic |
| minor | Recent sessions are all labelled exactly `session`, so twelve entries are indistinguishable, including to a screen reader | open |
| minor | Explorer file rows expose an empty accessible name, so a screen reader cannot identify which file a row is | open |
| minor | At narrow widths the status bar text clips at the right edge, and the toast overlaps the composer row | open |
| minor | `shell/onboarding.ts` documented a first-run surface `surfaces/Onboarding.tsx` that never existed; three of its four exports have no production caller | doc corrected, code left alone (closeout forbids feature work) |

## What the preview proves, and what it does not

It proves HIDE starts, serves a real workspace, renders its primary surfaces from real backend state,
refuses to invent data when the backend has none, and holds every gated effect for approval.

It does not prove any agent behaviour. No model was served, so no agent turn ran. Every model leg
remains `DEFERRED_MODEL_REQUIRED`, and nothing in this gallery should be read as evidence about model
quality, capability or parity.
