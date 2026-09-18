# Hawking machine authority — 2026-09-16

This is the durable status for the base Hawking/H-Web permission foundation.
Current source, fresh runtime observations, and deterministic tests are the
authority. No credential value is recorded here.

## Implemented

- `hawking.permissions.PermissionsService` is the single status/require/explain
  broker. It observes fresh OS/logical capability state and writes only a
  diagnostic cache under `.hawking/permissions/state.json`.
- Goal/WorkUnit admission carries `required_capabilities` and a capability
  lease. Missing required authority blocks before provider resolution or
  dispatch as `PERMISSION_BLOCKED`, preserving the exact next WorkUnit.
- `hawking.native_owner` owns the current/candidate/previous release contract
  for `hawking-native-helper` and refuses unsigned or designated-requirement
  changes at promotion. Source-checkout Swift builds are never implicitly
  granted production TCC authority.
- `hawkingd` exposes the local `GET /hawking/web/permissions` projection.
  H-Web exposes the same check through `/permissions` and `/machine`.
- H-Web compact widths use a fixed, bounded right control overlay instead of
  squeezing the chat into a second grid column. Context and Share panels are
  paired beneath their own controls.
- H-Web now supports selectable share TTLs up to the already-supported seven
  days, secondary-click/two-finger/double-click copy for user and assistant
  turns, and a confirmed trash action for deleting one Web chat while returning
  detached Goal handles as preserved.

## Live evidence

- `unset OPENROUTER_API_KEY; h web --no-browser` succeeded using the canonical
  macOS Keychain resolver only. The only credential assertion was
  `OPENROUTER_KEY_PRESENT=PASS`; no key or Authorization value was printed.
- A controlled Hawking restart loaded the new daemon code, then a second
  `h web --no-browser` reused the healthy `hawkingd` without restarting it.
  Current Web release: `web-d3310d676eb6`.
- The launcher observed the three Hawking cloud routes through Hawking model
  search and reported Auto available. Local Kimi P0 remains present but
  withheld because no admitted local runtime exists.
- Fresh broad permission observation: core workspace/network/process/browser
  capabilities are `READY`; no stable signed native helper is installed, so
  Accessibility, Screen Capture, Input Monitoring, Automation, Microphone,
  Speech Recognition, and Full Disk Access are not claimed ready.
- H-Web at the compact CUA viewport measured `738×718` with
  `scrollWidth=738`; the closed and open control-sidebar states both fit the
  viewport, and Context rendered directly below its button. Secondary-click
  copy placed the selected assistant output on the browser clipboard.

## Verification

- Relevant H-Web, permission, capability, native/browser-world, share,
  attachment, memory, projection, and launcher tests: **57 passed**.
- The broader command also reached five pre-existing receipt-gated failures in
  `hawking/test_context_compiler_runtime.py`; all require the absent sovereign
  `receipts/sovereign/G004_context_runtime.json`. No receipt was manufactured
  for this tranche.
- Python compilation: **passed**.
- Swift helper typecheck: **passed**.
- `git diff --check`: **passed**.
- No Flash, Pulsar, ModelLake, or unrelated user work was restarted, reset,
  deleted, or migrated.

## Honest readiness

`MACHINE_READY=YES` means the scoped source-only baseline is ready:
workspace read/write, network, process execution, and browser projection.
`BROAD_MACHINE_READY=NO` remains correct until a stable signed
`hawking-native-helper` is installed and the user grants the specialist macOS
permissions that a Goal actually requires. Hawking must recheck the OS before
each consequential action; the cached observation is not authority.

The next machine-authority step is to build/sign/promote the native helper
with one stable designated requirement, then recheck TCC permissions. No
automatic permission grant or global security weakening is allowed.
