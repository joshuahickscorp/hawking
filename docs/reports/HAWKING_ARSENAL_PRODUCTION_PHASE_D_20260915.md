# Hawking Arsenal Production Ascension — Phases D/E checkpoint

Date: 2026-09-15

Production Goal: `GOAL-B33C69F193`

This is a checkpoint for the Production Ascension mission, not a claim that
Phases E–L or the complete production environment are finished.

## Delivered

- Hawking owns a typed browser WorldState at
  `.hawking/browser-world/sessions/<browser_session_id>.json`. It has a schema,
  monotonically increasing revisions, bounded event history, delta replay, and
  screenshot artifacts hashed with `retention_class=DERIVED`.
- `workspace/ops/hawking_browser_helper.mjs` is one private Unix-socket
  Playwright sidecar. It owns only Chromium contexts and page mechanics. It
  does not own Goals, WorkUnits, worker identity, permissions, costs, receipts,
  or durable state.
- The registry owns the only browser tool contracts. Semantic observation,
  target resolution, verification, console/network/download metadata, and
  bounded screenshots are Hawking tools. Browser actions are typed
  `EXTERNAL_WRITE` operations, so they are refused by normal chat and ordinary
  Goal registries.
- A browser-capable Goal can now carry a durable,
  `hawking.goal.browser_authority.v1` capability. Hawking derives one session
  from the Goal id, records the allowed origins, and requires an explicit
  action grant.
- `hawkingd` reconstructs a browser-enabled registry only after it validates
  the durable WorkUnit record, Goal id, worker id, session id, worker attempt,
  contract path, fixed browser session, and navigation origin. Request JSON
  cannot create browser authority or cross into another Goal’s session.
- The Worker Packet prompt now tells an authorized worker its fixed browser
  session, origin fence, action state, and verification obligation. No private
  provider state is made durable.

## Local evidence

`Playwright 1.63.0` and Chromium `153.0.8010.12` are pinned under the existing
`workspace/ops` dependency owner. The browser executable cache is at
`.hawking/browser-runtime/`, a derived/reproducible local artifact rather than
an ambient browser cache.

The direct local qualification opened a disposable HTTP fixture, observed a
button by accessibility label, clicked it, verified `idle -> done`, committed
WorldState deltas, and wrote a SHA-256 screenshot artifact. An active health
probe returned `READY`, `browser_connected=true`, and Chromium
`153.0.8010.12`.

Verification run:

```text
python3 -m pytest -q \
  hawking/tests/test_browser_world.py \
  hawking/tests/test_web_projection.py \
  hawking/tests/test_recovery_projection.py \
  hawking/tests/test_process_authority_resolution.py \
  hawking/tests/test_goal_listing_canonical.py \
  hawking/tests/test_web_live_contract.py \
  hawking/tests/test_serve_openai_surface.py \
  hawking/tests/test_install_shims.py \
  hawking/test_install_stamp.py \
  hawking/tests/test_arsenal_contracts.py \
  hawking/test_tool_registry.py

90 passed
```

`node --check workspace/ops/hawking_browser_helper.mjs` and `git diff --check`
also passed for the Phase D paths.

## Honest state

| Capability | State |
| --- | --- |
| Hawking-owned browser session | PASS |
| Revisioned browser WorldState and replay | PASS |
| Semantic target resolution | PASS |
| Typed external-action fence | PASS |
| Scoped Goal browser authority | PASS |
| Goal/worker/session/origin validation | IMPLEMENTED; deterministic contract coverage passes |
| Remote worker self-repair through the browser | PASS |
| H-Web current production Goal projection after refresh | PASS |
| H-Web live browser activity projection during a fresh worker run | NOT YET QUALIFIED |
| Phase E read-only native desktop foundation | PASS |
| Phase E input lease, screen capture, OCR, and semantic action acceptance | NOT STARTED |
| Phase F existing Auto foundation | RECONCILED; multi-worker execution acceptance remains pending |
| Phases G–L | NOT STARTED |

## Phase E — read-only native desktop foundation

`native/hawking-macos/hawking_macos.swift` is a narrow versioned JSON-lines
helper using Apple-native AppKit, CoreGraphics, and Accessibility APIs. It is
mechanics only: it does not know Goals, WorkUnits, providers, credentials,
budgets, permissions, receipts, or input authority. Its currently reachable
operations are only `health` and `observe`.

`hawking/macos_world.py` owns the durable desktop WorldState at
`.hawking/macos-world/world.json`. The state has monotonic revisions, bounded
delta history, current/freshness semantics, application/window identities,
focused app/element, AX availability, and an explicit `NO_INPUT_CAPABILITY`
input state. The registry exposes only read-side `macos.health`,
`macos.observe`, and `macos.changed`; there is no `macos.click`, type, shell,
or coordinate-input escape.

Live local evidence on this host: the helper compiled with the installed Swift
toolchain and reported `READY`, `helper_version=0.1.0`, trusted Accessibility,
70 observed applications, six visible windows, and monotonic revisions 1–3
with a no-gap delta replay. Screen capture and OCR are truthfully
`NOT_IMPLEMENTED`; no Apple input API has been called.

The Phase E deterministic WorldState/registry coverage, the reconciled Auto
classification coverage, and the relevant Phase D/C2 suite currently pass:

```text
82 passed
```

## Next bounded WorkUnit

The staged acceptance completed in the linked worktree
`.worktrees/browser-world-self-repair-20260915` under the fixed Goal session
`BROWSER-B33C69F193`:

1. `D-BROWSER-SELF-REPAIR-20260915` observed and repaired the fixture plus
   regression test. Its first provider admission was correctly denied while
   the daemon policy was `$0.00` (`$0.00` charged). After the user-authorized
   `$30.00` production policy was recorded, it ran for `$0.0005497548`.
2. Hawking detected that this initial WorkUnit had not actually performed a
   post-repair positive browser verification. The source/test seal and failed
   verification were retained in
   `D-BROWSER-SELF-REPAIR-20260915_BROWSER_COMPLETION_RECTIFICATION.json`; the
   root Goal was returned to `RUNNING`.
3. Completion contracts now support `required_verified_tools`; a successful
   `browser.verify` call with `verified=false` can no longer satisfy a browser
   acceptance gate. Browser-capable Goals also require open, observe, find,
   click, reload, and a later true verification.
4. `D-BROWSER-POST_REPAIR-VERIFY-20260915` then performed the positive
   browser verification, added an explicit fixture marker and matching test,
   ran the focused tests, and sealed for `$0.0006444816`.

Browser WorldState revisions `1–16` preserve the initial failed verification,
the source/test repair, and the later `browser.verify=true` observation. The
fixture’s three deterministic tests now pass. The WorkUnit results are
staging-only; the root Production Ascension Goal remains `RUNNING` for Phase E
instead of being terminally completed by a child WorkUnit.

Remote cost charged by Phase D: `$0.0011942364`.

## Next bounded WorkUnit

`E_MACOS_SEMANTIC_ACTION_LEASE`: add an explicitly authorized, preemptible
local-input lease and a harmless native UI fixture. It must prove semantic
AX observation → target resolution → authorized action → observed verification
without affecting user foreground input. Screen capture/OCR remain separate
read-side tranches and must not be used to bypass a missing AX fact.

## Phase F — existing Auto foundation reconciled

No second scheduler was added. Hawking already owns bounded remote admission,
distinct worker/attempt identities, durable Worker Packets, per-task model
metrics, planning, synthesis, and consolidation in
`hawking/auto_orchestration.py` and the existing Goal/WorkUnit owners.

The production classes now include `browser`, `desktop`, `document`,
`reverse_engineering`, and `physical_engineering`; their specific signals win
over generic review/visual keywords. A live zero-spend `/v1/auto/plan` request
for the next macOS action-lease tranche classified it as `desktop`, selected
DeepSeek V4.1 Flash, and exposed only the currently real read-side native
capabilities (`macos.observe`, `macos.changed`). It created no provider request
or worker. Full multi-worker execution and same-model concurrency remain
unaccepted until a bounded isolated WorkUnit demonstrates them.

Protected lanes touched: none. Flash/Pulsar, ModelLake, Kimi P0, TTS, and
existing receipts were not started, moved, or mutated.
