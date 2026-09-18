# Hawking Arsenal implementation status — 2026-09-16

This is the reconciled status of the first executable Arsenal tranche in the
current dirty `main` worktree. The Arsenal plan is treated as a specification;
current source, live process state, receipts, and fresh verification remain the
acceptance authority. This report deliberately does not claim the whole plan
is complete.

## Implemented in this tranche

- A0: `hawking.execution.binding.v1` binds request, Goal, WorkUnit, worker
  attempt, capability, canonical root, allowed paths, verification, budget,
  resource policy, and (when present) a Hawking lease. Immutable
  `hawking.execution.candidate_manifest.v1` records are published through the
  existing WorkUnit owner and are restart-readable.
- A1: `hawking.mutation.request.v1` and `hawking.mutation.intent.v1` are the
  typed path for Engine mutations and `filesystem.write`. Paths are normalized
  and admitted against the request scope; base hashes and file identity are
  rechecked; writes use the shared atomic executor; partial/crash states are
  `NOT_STARTED`, `APPLIED`, `VERIFIED`, or `UNKNOWN_OUTCOME`.
- A1 completion proof: accepted typed mutations require grounded changed
  artifacts, applicable passing tests, an inspected diff, and a structured
  result packet. A crash after publish but before verification cannot be
  promoted to `VERIFIED` merely because bytes happen to match.
- B1/B2: `fs.search` and `filesystem.search` expose bounded named corpora,
  searched roots, exclusions, content-derived scope revisions, and explicit
  expansion metadata. The existing Rust `hawking-index` remains the structural
  index owner; bounded read/preview adapters expose `source.owner`,
  `source.outline`, `source.symbol`, `source.references`, `source.ast_query`,
  `source.affected_tests`, and non-writing `source.rewrite_preview`.
- C1: `crates/hawking-web` provides typed Session, Goal, WorkUnit, Worker,
  Capability, Artifact, RouteDecision, Budget, event replay, API negotiation,
  and idempotency DTOs. It is projection/transport staging only; Python
  `hawkingd` and the existing Goal/DAG/event owners remain authoritative.
- Worker/remote boundary: Auto produces bounded cognition plans, distinct
  worker/attempt packets, and durable routing evidence. The existing
  `hawkingd` gateway now uses that same Hawking admission owner for all remote
  requests (global limit plus the initial Kimi premium limit), while stripping
  identity metadata before provider transport. No second scheduler or provider
  runtime was introduced.
- E semantic action lease: WorkUnit
  `E_MACOS_SEMANTIC_ACTION_LEASE-20260915-205B3579` completed the real native
  fixture path through Hawking's `macos.find` → lease admission → semantic
  `AXPress` → post-action observation/verification → release contract. The
  durable receipt records one exact AX target, `before_revision=115`,
  `after_revision=116`, `revision_delta=1`, and `verified=true`; the child then
  completed the Hawking-owned `repo.edit → tests.run → git.status → git.diff`
  continuation. The parent Goal remained `RUNNING`.

## Fresh verification

- Selected relevant Python suite: **304 passed, 25 explicitly deselected** in
  19.64 seconds. The exclusions are the known live-native `processes` probe,
  historical catalog expectations, legacy Grok restart-adoption expectations,
  and tests whose unsafe direct-mutation contract conflicts with the current
  typed completion gate.
- Arsenal focused contracts: **13/13**; the focused mutation/actionable/result
  set is **48/48**.
- `cargo test -p hawking-web`: **2 passed, 0 failed**.
- `cargo test -p hawking-core gravity_repo_search`: **15 passed, 0 failed**.
- Changed Hawking Python modules: `py_compile` passed. `git diff --check`
  passed.
- Read-only live-helper probe: **1 passed, 9 failed, 4 deselected**. The nine
  failures all reach the registered `processes.*` tools but the currently
  running native helper rejects its historical `processes` subcommand with
  `unrecognized subcommand 'processes'`. This is recorded as an existing
  runtime protocol mismatch, not converted into a false pass.
- Final closeout verification: `python3 -m compileall -q hawking`, focused
  semantic/contract/Arsenal tests **36 passed**, Swift helper and fixture
  `swiftc -typecheck` both passed, and `git diff --check` passed.
- Live native fixture evidence remains valid: PID `21505` is alive, the helper
  reports `helper_version=0.2.0`, trusted Accessibility, and
  `semantic_axpress=true`; no screen-capture or OCR capability is claimed.

## Operational boundary

- The bounded capture/OCR child
  `E_MACOS_SEMANTIC_ACTION_LEASE-20260916-3C726AF1` was admitted through the
  same Hawking Goal/WorkUnit owner, but canonically cancelled after a
  prose-only continuation before any mutation. Its recorded spend is
  **$0.00290772**; no `document.extract` source, registry door, or test file
  exists in the tree. This is an explicit deferral, not a partial capability.
- The live `hawkingd` is healthy at PID **35340** on `127.0.0.1:8014` (HTTP
  health and models surfaces return successfully); the Hawking Gravity helper,
  ModelLake watcher, and disposable native fixture remain running. Only
  `hawkingd` was reloaded to pick up the already-started continuation plumbing.
- Flash/Pulsar artifacts, protected temporary build state, historical Claude
  sources, user campaign receipts, and unrelated dirty work were preserved.
- No commit, push, reset, checkout, worktree cleanup, or destructive operation
  was performed.

## Remaining / not claimed

- The typed Web contract is not yet the serving owner (C2); event-store parity,
  replay recovery, browser refresh/reconnect proof, and daemon restart/resume
  qualification remain staging work.
- Auto’s bounded admission is live at the remote gateway, but the full
  multi-WorkUnit CognitionPlan executor, model handoff, synthesis/consolidation,
  overlap detection, and learned routing policy are not yet complete.
- Playwright/browser helper parity, document/OCR,
  reverse-engineering adapters, and the broader HIDE/AgentOS dependency-graph
  migration still require their own owner/parity proofs.
- Capture/OCR is explicitly deferred from this closeout. The next reopenable
  WorkUnit is `E_MACOS_CAPTURE_OCR`; it must be a fresh red-before-green
  `document.extract` owner proof through Hawking, with no direct Codex target
  mutation.
- Legacy Grok adoption remains intentionally disabled by current policy; the
  two historical restart-adoption tests are not acceptance targets.

`ARSENAL_READY=NO`. `E_MACOS_READ_ONLY_FOUNDATION_ACCEPTED` remains the exact
parent phase. The semantic action lease is accepted; capture/OCR is deferred
with its cancellation receipt preserved. Layer II, Ghidra/Rizin, and document
expansion were not started.

The deferred capture/OCR boundary now has a deterministic
`hawking.document_extract.contract.v1` seam. It validates artifact-bound,
workspace-scoped, size-bounded requests and requires a matching source digest
with verified RED-before-GREEN evidence. This makes
`E_MACOS_CAPTURE_OCR` ready to reopen through a fresh Hawking WorkUnit, but
does not claim that capture, OCR, or `document.extract` execution is present.
