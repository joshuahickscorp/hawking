# Hawking Arsenal C2 live cutover and recovery receipt

Date: 2026-09-15

## Scope

This receipt records C2 control-plane cutover only. It did not launch a new
remote Goal, call a provider, alter Flash/Pulsar, or create a second daemon.
The existing completed disposable Goal `GOAL-945600ECD0` was used as the
checkpoint-safe recovery object.

## Implemented

- `hawking.web.contract.v1` is now a live projection of canonical Session,
  Goal, WorkUnit, worker, capability, artifact, route, budget, and share
  owners. The previous H-Web fields remain only as compatibility fields for
  the deployed minimal page.
- `hawking.web.event.v1` provides durable, per-session, monotonic event IDs,
  sequence, canonical identity, type, timestamp, payload version, and source
  revision. Replay is bounded to 64 events, reports gaps explicitly, and
  requires a projection snapshot after retention overflow. An ACK is a client
  cursor, not browser-owned or daemon-owned domain state.
- Session projections include the latest durable message sequence, so a
  conversation-only write advances the canonical revision and cannot be
  invisible to the event stream.
- H-Web polls that projection and refreshes from Hawking state on a gap. It
  does not manufacture events or persist Goal/worker state in JavaScript.
- Static H-Web assets have a local current/candidate/previous release owner.
  It is one asset selector in `hawkingd`, not a second HTTP writer.
- Recovery status/reconciliation is available through
  `GET/POST /hawking/recovery` and `hawking recovery [--reconcile]`. It reads
  canonical Goals, workunits, leases, and mutation intents; it never replays a
  worker or a mutation. A published-but-unverified mutation remains
  `UNKNOWN_OUTCOME` until deterministic evidence can prove otherwise.
- Canonical Goal enumeration ignores `.contract.json` and `.result.json`
  companions, preventing duplicate Goal objects after refresh/recovery.
- Process inspection now resolves the separately named
  `hawking-process-authority` protocol executable and package. The inference
  binary named `hawking` is never selected for `processes` or
  `processes-server`; the old `hawking-backend` filename and private Rust
  library target remain source-compatibility fallbacks only.

## Live cutover evidence

1. Confirmed no RUNNING worker: only two `CANCELLED/KILLED` and two
   `COMPLETE/RELEASED` workunits existed before restart.
2. Gracefully stopped the prior 8014 owner (PID 24929).
3. Started the current Hawking source at `127.0.0.1:8014` with one daemon
   lease and the explicit native process-authority binary (PID 56301).
4. Attached `GOAL-945600ECD0` to `web-c2-live`; live session returned
   `hawking.web.contract.v1` and event replay emitted five ordered canonical
   session/Goal/WorkUnit/worker/route updates.
5. ACK at sequence 5 replayed zero events with `gap=false` and
   `snapshot_required=false`.
6. Browser opened `http://hawking.localhost:8014/` and visibly retained the
   existing transcript plus completed Goal after reload.
7. Exercised local static release stage/activate and canonical result/diff/
   workers routes. Unit coverage exercises a content-changing blue/green
   transition; the live stage used the same accepted asset bytes.
8. Performed a second controlled restart after the duplicate-list repair, then
   a final restart after the session-cursor projection correction. Recovery
   reports only `CANCELLED` and `COMPLETE` workunits, and `/v1/goal` reports
   exactly four unique Goal IDs.

## Verification

- `python3 -m pytest -q ...`: 87 passed (C2 live projection/recovery,
  process protocol, install, server, Arsenal contract, and registry coverage).
- `cargo test -p hawking-web`: 2 passed.
- `hawking/test_tool_registry.py -k process`: 10 passed, 4 deselected.
- Python compilation and `git diff --check`: passed.

## C2 matrix

| Gate | Result |
| --- | --- |
| Canonical mutation/evidence contract | PASS |
| Scoped source/search foundation | PASS (prior Arsenal tranche; retained) |
| Typed live Web projection | PASS |
| Ordered bounded replay / ACK / gap semantics | PASS |
| Browser refresh/reconnect | PASS |
| Front-end/tab failure recovery | PASS |
| Controlled daemon restart/recovery | PASS |
| Unknown mutation outcome preservation | PASS |
| Process protocol coherence | PASS |
| Web-independent recovery control | PASS |
| Static current/candidate/previous release owner | PASS |
| Duplicate Goal authority | PASS |
| Auto executor | DEFERRED (not required to pass C2) |

`ARSENAL_READY=YES` for the C2 core invariants. This does not claim an admitted
local resident: current runtime remains explicitly `waiting_for_native_runtime`
because there is no admitted local role binding. Remote cognition was not
re-spent during C2.
