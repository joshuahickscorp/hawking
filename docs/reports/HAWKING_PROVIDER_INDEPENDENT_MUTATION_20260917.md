# Hawking provider-independent mutation status

Date: 2026-09-17

## Durable state

- Canonical root: `GOAL-B33C69F193`
- Root status: `RUNNING`
- Root phase: `P1_CONTROL_PLANE_COLLAPSE`
- Root next WorkUnit: `P1_PROVIDER_TOOL_RECEIPT_CONTINUATION_REPAIR`
- Root blocker: none
- Cloud policy: DeepSeek V4.1 Flash only
- Protected lanes: Flash, Pulsar, and ModelLake were not touched

The existing P1 child remains the canonical child:
`E_MACOS_READ_ONLY_FOUNDATION_ACCEPTED-20260917-7BFCD1F9`.
It is checkpointed as `OUTPUT_UNUSABLE` with
`PROVIDER_STRUCTURED_RESULT_INVALID`; its evidence and dead-letter record are
preserved, and it has no live driver. A provider-format failure does not change
the parent Goal to `BLOCKED`.

## Implemented boundary

Provider output is data only. Hawking now accepts and compiles:

- typed `MUTATION_PROPOSED` mappings;
- `HAWKING_PATCH_V1` unified-diff envelopes;
- bounded `HAWKING_PATCH_V1` file/anchor/replacement envelopes;
- `HAWKING_EDIT_V1` file/edit envelopes.

Hawking attaches and validates the canonical Goal, WorkUnit, workspace, writer
lease, allowed paths, expected content identity, and completion obligations.
Only the existing canonical `repo.edit` engine performs the mutation. The
verification chain is then projected as `repo.edit → tests.run → git.status →
git.diff`. Direct provider mutation/verification calls are withheld in proposal
mode and rejected if emitted anyway. Plain prose remains inert.

Provider transport failures and unusable format results are child-level
`PAUSED_PROVIDER`/`OUTPUT_UNUSABLE` conditions with durable provider dead-letter
evidence; they do not become parent Goal blockers.

## Verification

- Python compile: pass
- Focused and integration regression set: **109 passed**
- Live daemon: running Hawking authority on `127.0.0.1:8014` (health reports
  `waiting_for_native_runtime`), current Web release `web-6347550e08ae`,
  Keychain credential path ready
- Live DeepSeek recovery attempts: provider returned an explicit
  `HAWKING_PATCH_V1` label but no parseable executable payload; no mutation was
  applied and no receipt was fabricated
- Latest child remote spend before closeout: `$0.006620292`

Final verification recorded after the last controlled daemon reload:

- Current `hawkingd` PID: `87613`; health: `waiting_for_native_runtime`, authority:
  `remote_goal_write`
- Current Web release: `web-6347550e08ae`; no candidate was promoted
- Focused mutation/remote/serve/DAG/Arsenal set: **109 passed**
- Full `hawking/tests` sweep: **1896 passed, 107 failed, 7 skipped**. The failures
  are outside this selected mutation-boundary/current integration set and span
  unrelated legacy/runtime contracts in the already-dirty worktree; they are
  not represented as mutation-boundary acceptance.
- No live provider mutation, source write, or mutation receipt occurred during
  the final DeepSeek attempts.

## Follow-up reconciliation

The discrete Goal packet had contained stale bootstrap wording that asked the
provider to emit `repo.edit` even though proposal mode withholds provider-owned
mutation tools. That contradiction is removed: fresh packets now request one
complete `HAWKING_PATCH_V1`, `HAWKING_EDIT_V1`, or typed `MUTATION_PROPOSED`
payload and explicitly leave `repo.edit`, `tests.run`, `git.status`, and
`git.diff` to Hawking.

A fresh DeepSeek-only continuation (`r8`) was then run against the same child.
It returned a 24-character prose-only continuation and no executable payload;
the child is therefore durably `OUTPUT_UNUSABLE` with
`MUTATION_PAYLOAD_MISSING`, attempt evidence, and no mutation receipt. The
bounded recovery policy was honored and no further remote retry was started.

The child-liveness repair is now covered by focused tests: stale Goal-owned
drivers become `RECONCILIATION_REQUIRED`, incomplete terminal packets become
`OUTPUT_UNUSABLE`, provider blocking/no-progress and resource ceilings remain
child-owned recoverable states, while legacy bootstrap records retain their
historical `BLOCKED` compatibility behavior. The canonical root remains
`RUNNING` and its phase/next-WorkUnit were not overwritten.

The focused mutation/remote/serve/DAG/Arsenal set now passes **109 tests**;
Python compilation and `git diff --check` pass for the changed boundary.

## Programme continuation after provider failure

The provider-output failure was not retried on the original mutation child.
Two materially different, read-only dependency checks were admitted under the
same root with `deepseek/deepseek-v4.1-flash` only:

- `E_MACOS_READ_ONLY_FOUNDATION_ACCEPTED-20260917-179259D1` — P1 Goal/child
  liveness audit; `$0.002838672`; real bounded read/tool activity, then prose
  continuations until the cap; no usable structured result.
- `E_MACOS_READ_ONLY_FOUNDATION_ACCEPTED-20260917-BDB2BCCC` — independent
  public command-owner reconnaissance; `$0.001242336`; the same bounded
  output deficiency after read/tool activity; no source mutation.

Both children were reconciled through the WorkUnit owner to
`OUTPUT_UNUSABLE`, `PROVIDER_OUTPUT_UNUSABLE`, and durable provider
dead-letter receipts. Neither changed the root status or retried the failed
mutation child. The research continuation-cap path now makes that same
classification automatically for future Goal-owned research children instead
of leaving an output failure as an ambiguous resource pause.

The canonical WorkUnit receipt sum is now `$0.499122964` against the `$55.00`
programme ceiling, leaving `$54.500877036`; the root remains `RUNNING` with
`P1_PROVIDER_TOOL_RECEIPT_CONTINUATION_REPAIR` as the next queued item and no
active child driver. No Flash, Pulsar, or ModelLake lane was touched.

An independent P2 command-surface check also passed without remote spend:
`hawking/cli.py` routes `h web`, `hawking web`, and `h build` to the single
`hawking/web_launcher.py` owner; the older `hawking/web.py` remains only as a
controlled-restart compatibility adapter (`stop_main`), not as public Web
ownership. The launcher regression/live set passed **22 tests**, and
`env -u OPENROUTER_API_KEY h web --no-browser` reused `hawkingd` PID `87613`,
selected Web release `web-6347550e08ae`, and reported the two-model Hawking
cloud roster plus Auto through Hawking search. This is P2 evidence only; the
canonical programme queue remains P1 until its pending mutation acceptance is
earned.

The independent P3 foundation check is likewise green without remote spend:
the artifact/paste, H-NOTES, FTS memory, context compiler, continuity, and
workspace-isolation set passed **43 tests**. This verifies the existing P3
baseline but does not promote the phase or bypass the canonical P1 queue; no
new WorkUnit acceptance is claimed from deterministic tests alone.

The next deterministic programme checks also passed without remote spend:
P4 Auto/ExecutionPlan/H-Web controls passed **21 tests**, and the non-MLX
Gravity/NR/NX/search slice passed **39 tests**. That slice exposed and then
closed two current contracts: security-focused catalog retrieval now keeps the
read-only shell capability in its bounded result, and the native filesystem
search regression explicitly selects the unpartitioned scope that owns Gravity
provenance. The MLX-dependent historical `test_gravity_outlier.py` remains
unrun and untouched. The combined relevant verification set is now **234
passed**; these are deterministic evidence, not remote WorkUnit acceptance.

The independent P6–P9 local slice also passes **112 tests** across
source/Goal compilation, allocation, scheduler resources, GPU-lane deferral,
Nova lineage, physical emission/qualification, graph scoring, and Gravity
adequacy. One stale scheduler expectation was corrected from retired `GROK` to
the current canonical `HAWKING` resource class. The MLX-dependent
`hawking/test_lowrank_nr.py` was excluded at collection and left untouched;
no local model lane was revived.

## Remaining reopen condition

Resume the same child only with a fresh DeepSeek packet that contains an
executable, content-bound `HAWKING_PATCH_V1` or `HAWKING_EDIT_V1` payload. On a
valid payload, Hawking should execute the canonical verification chain and then
return control to the existing root Goal. No provider conversation state is
required for recovery.

## Independent continuation after bounded provider failure

The next materially different, read-only P1 child
`E_MACOS_READ_ONLY_FOUNDATION_ACCEPTED-20260917-94C484FF` was admitted through
the canonical Hawking Auto/WorkUnit path with the pinned DeepSeek V4.1 Flash
policy. It performed six bounded turns and produced real Hawking activity, but
again exhausted the provider continuation cap without a structured completion
packet. No mutation was attempted. The child was preserved as
`OUTPUT_UNUSABLE` with a provider dead letter and its parent active-child
pointer was cleared; it was not retried. This is additional child-level
provider-output evidence, not a root blocker.

The root remains `RUNNING` in `P1_CONTROL_PLANE_COLLAPSE`, with
`P1_PROVIDER_TOOL_RECEIPT_CONTINUATION_REPAIR` still queued. Reconciled
remote spend is **$0.500084** against the **$55.00** authorization, leaving
**$54.499916**. No protected Flash/Pulsar/ModelLake lane was touched.

The subsequent independent recovery-invariant child
`E_MACOS_READ_ONLY_FOUNDATION_ACCEPTED-20260917-E5807E8D` was also admitted
through the canonical path using DeepSeek V4.1 Flash only. It remained live
through five observed turns and then reached the bounded continuation cap on
turn six without a structured packet. Its Hawking evidence and dead letter are
preserved, the parent active-child pointer is clear, and no retry was issued.
This is read-only provider-output evidence and does not establish a general
mutation-architecture blocker. Reconciled spend is now **$0.501225**, leaving
**$54.498775** of the authorized **$55.00**.

## P1 deterministic follow-up

The next local control-plane verification exposed stale tests that still
described retired Qwen/Grok ownership and a fake resident that did not satisfy
the current Hawking backend endpoint contract. Those expectations were aligned
to the current canonical Hawking status/probe surface and RuntimePool fixture.
The adjacent control-plane suite now passes **130 tests**; Python compilation
and `git diff --check` also pass. This is deterministic P1 evidence only and
does not claim the missing provider-backed mutation receipts.

## Deterministic audit closeout — 2026-09-17

The full safe Hawking Python audit was rerun after the control-plane repairs:
**1,941 passed, 17 skipped, 28 subtests passed**. The focused
control-plane/provider/launcher set passed **332 tests**. Python compilation
of the changed source/tests and `git diff --check` both pass. One unrelated
`DeprecationWarning` remains in the terminal perception fixture; it did not
affect the result.

This audit also reconciled the remaining stale expectations without reviving
retired owners. RuntimePool now admits provider-neutral remote endpoints via
the canonical OpenAI-compatible backend while refusing retired MLX/GGUF
local paths; the Qwen38 source/profile audit no longer requires a runnable
native artifact for a semantics-only check; builder schema tests now pass the
explicit write projection needed to expose `tests.run`; focused Python
evidence targets the requested definition rather than a same-name local
assignment; and RED-before-GREEN, current Hawking status, and retired Grok
expectations are aligned with the live contract.

No remote worker was dispatched in this audit and remote spend remains
**$0.501225208313042**, leaving **$54.498774791686955** in the authorized
programme budget. Flash, Pulsar, ModelLake, and sealed receipts were not
touched. The canonical root `GOAL-B33C69F193` remains `RUNNING` in
`P1_CONTROL_PLANE_COLLAPSE`, has no active child, and retains
`P1_PROVIDER_TOOL_RECEIPT_CONTINUATION_REPAIR` as its next WorkUnit. The
provider-backed mutation receipt chain remains unaccepted and is not being
claimed by this deterministic audit.

## Queue liveness audit — 2026-09-17

A fresh local reconciliation found no live WorkUnit-owner process and no
non-terminal child pointer. The canonical root remains `RUNNING` with
`active_workunit_id=null`; its stale failed owner-job record is preserved as
historical evidence rather than treated as an active worker. Current Goal-owned
child states are **9 COMPLETE, 13 OUTPUT_UNUSABLE, 13 CANCELLED, and 3
BLOCKED**. The blocked entries are older permission-classified children and
remain inspectable; the recent provider-output failures are correctly recorded
as child-level `OUTPUT_UNUSABLE` with dead-letter evidence.

No provider retry was issued. The next dependency-valid queue item remains
`P1_PROVIDER_TOOL_RECEIPT_CONTINUATION_REPAIR`; this checkpoint is a liveness
and recovery observation, not a claim that the provider-backed mutation chain
has passed.

## Budget reconciliation — 2026-09-17

The programme budget ledger was reconciled through the canonical Goal owner.
Its prior `spent_usd` value lagged the root by the two latest read-only
provider children; it now matches the authoritative root totals:
**$0.501225208313042 spent** and **$54.498774791686955 remaining**. No receipt
was deleted or rewritten, and this was accounting-only state repair; the root
remains `RUNNING` and the next WorkUnit is unchanged.

The current native workspace was also checked with
`cargo check --workspace --all-targets`: **PASS**. The command produced only
existing dead-code warnings from large research examples. This is compile
evidence for the dirty native workspace, not a claim of target hardware
execution or of acceptance for Flash/Pulsar/ModelLake lanes.

## Frontier compatibility repair — 2026-09-17

The parallel frontier regression found and repaired a stale compatibility edge:
when a parent had an explicit empty `active_workunit_ids` list, dispatch could
retain a legacy scalar active-child hint that no longer belonged to any live
lane. Dispatch now treats the explicit list as authoritative and selects a
current lane as the scalar UI hint. The parent terminal reconciliation and
dispatch regressions now pass **31 tests** across the focused frontier,
WorkUnit, DAG, and remote-contract set; compilation and `git diff --check`
remain clean.

The Rust library test boundary was then exercised with
`cargo test --workspace --lib`: **238 passed, 0 failed** across the workspace
libraries (211 in the main Hawking core library and 27 in the protocol
library). This remains local verification; it does not promote any
unqualified hardware target.

## Parallel Goal frontier — 2026-09-17

The parent Goal now persists a bounded parallel frontier in addition to its
legacy scalar `active_workunit_id` UI hint. Child dispatch records
`active_workunit_ids`, `active_workunit_jobs`, and `runnable_frontier`; terminal
child reconciliation removes only the completed/deferred child and preserves
other active lanes, their primary hint, and the parent's `RUNNING` state. A
focused regression covers two concurrent child lanes and their independent
terminal edges; the scheduler/WorkUnit subset passes **30 tests**.

Three disjoint read-only DeepSeek V4.1 Flash reconnaissance lanes were then
admitted concurrently under `GOAL-B33C69F193`: P4 Auto/MG/ExecutionPlan,
P5/P6 Gravity/NR/NX, and P16 control-plane census. Each produced 19 durable
evidence items before reaching the already-known bounded
`PROVIDER_OUTPUT_UNUSABLE` continuation limit. They were not retried, no
source or protected artifact was mutated, and their dead-letter evidence is
preserved. This adds **$0.003098148** to the canonical ledger: total spend is
**$0.5056020883130419**, leaving **$54.49439791168696** of the authorized
**$55.00**.

The root remains `RUNNING` in `P1_CONTROL_PLANE_COLLAPSE` with no active child
after individual lane reconciliation. `P1_PROVIDER_TOOL_RECEIPT_CONTINUATION_REPAIR`
remains deferred and was not allowed to serialize or monopolize independent
work. The three reconnaissance results are child-level provider-output
evidence, not a root blocker and not an acceptance claim for the deferred
mutation path.

## Dependency-aware frontier projection — 2026-09-17

The parent projection now has one reconciliation seam that derives its
frontier from durable Goal-owned WorkUnit records. `RUNNING`,
`CHECKPOINTING`, `COMPACTING`, and dependency-satisfied `QUEUED` records are
shown as runnable; terminal/provider/resource states are excluded; unresolved
parent dependencies are reported separately as waiting. An explicitly active
sibling is preserved during the short interval before its record is flushed,
while an explicit empty active list cannot resurrect a stale scalar hint.
Terminal child release invokes this projection without starting work. The
frontier regression covers queued, dependent, missing-dependency, active, and
terminal lanes. Full current `hawking/tests` verification passes **1,987
tests**, with 20 skips and 28 subtests; focused frontier/contract coverage is
**37 passed**. The root remains `RUNNING`, with no active lane and no remote
spend added by this repair.

## Post-steer verification — 2026-09-17

The current `hawking/tests` suite completed with **1,983 passed, 20 skipped,
28 subtests passed, and one pre-existing fork warning**. An initial broader
invocation was corrected after collection encountered five legacy MLX-only
root-level fixtures outside the intended `hawking/tests` scope; those files
were not modified and no MLX lane was revived. Python compilation and
`git diff --check` remain clean.

No further remote WorkUnit was launched after the three disjoint lanes. The
root is intentionally left `RUNNING`, with queued/dependency-valid programme
phases still available and no root-level external blocker. The deferred P1
provider-output deficiency remains preserved as child evidence and is not
being retried merely to force a schema-shaped response.

## Distinct P1 mutation attempt — 2026-09-17

The accepted P1 owner-map reconnaissance was used to admit one materially
different, tightly scoped serialization-owner mutation WorkUnit through the
real Hawking admission path. DeepSeek completed one bounded `fs.read`, then
returned `MUTATION_PAYLOAD_MISSING` on the second turn; no `repo.edit`, test,
or source mutation occurred. The WorkUnit is preserved as
`OUTPUT_UNUSABLE` with dead-letter evidence and will not be retried solely to
force provider-native mutation syntax.

This is additional evidence for the provider-output deficiency, not a root
blocker. The root remains `RUNNING`, P1 remains deferred, and independent
programme work remains eligible. The child cost was **$0.001540932**; the
canonical programme ledger now records **$0.5071430203130419 spent** and
**$54.492856979686955 remaining**.

## Serialization-owner baseline verification — 2026-09-17

Inspection of the current dirty source confirms that the owner-map target is
already implemented: `DagStore._unit_to_disk` delegates directly to
`WorkUnit.to_dict`, `DagStore._units_from_document` uses `WorkUnit.from_dict`,
and the remaining disk-extras function is an inert compatibility hook. The
actual DAG/WorkUnit regression boundary passed **17 tests**. No no-op mutation
was attempted; the failed provider child remains `OUTPUT_UNUSABLE`, while
this deterministic baseline is recorded as `BASELINE_SUFFICIENT` for the
serialization seam.

## Five-lane Auto execution — 2026-09-17

At the operator's direction, Hawking Auto admitted five distinct read-only
reconnaissance WorkUnits under `GOAL-B33C69F193`, all pinned by the parent
policy to **DeepSeek V4.1 Flash**. The bounded remote admission ceiling is
four, so four lanes ran concurrently; the fifth hit HTTP 502 before producing
provider evidence and was classified `PAUSED_PROVIDER`. The four live lanes
performed real Hawking read-tool activity and each reached the already-observed
bounded provider-output contract, so they are preserved as `OUTPUT_UNUSABLE`
child evidence rather than retried or promoted to a root blocker:

```text
P3 memory/artifact/source       B1F3FCA1  OUTPUT_UNUSABLE  $0.001206720
P4 Auto/MG/ExecutionPlan        4A5F2BB5  OUTPUT_UNUSABLE  $0.000900552
P5 Gravity/NR/NX                2DD707BC  OUTPUT_UNUSABLE  $0.001384128
P10 native/runtime              C31171D5  OUTPUT_UNUSABLE  $0.001050456
P16 redundancy/collapse         420D13EE  PAUSED_PROVIDER   $0.000000000
```

The peak was **4 concurrent workers**, honoring Hawking's current admission
policy; no uncontrolled fifth process was retained. No source, Goal, Flash,
Pulsar, ModelLake, or sealed receipt was mutated by this batch. The canonical
root remains `RUNNING` with no active child after individual lane
reconciliation. Accounting was reconciled across every Goal-owned WorkUnit:
**$0.512234631113 spent** and **$54.487765368887 remaining** from the
authorized `$55.00` programme cap. These are lane-level provider-output
results, not accepted phase evidence; the next independent action remains
deterministic or materially new work rather than another format retry.

## Provider result normalization — 2026-09-17

The read-only completion seam now has a bounded provider-independent
normalizer. A terminal JSON packet with non-empty `observations` may omit
empty bookkeeping arrays; Hawking supplies those defaults only after the
existing successful-read evidence check. Lowercase terminal status is
normalized, while prose, empty observations, nonterminal status, malformed
arrays, custom missing fields, and mutation claims remain unusable. Focused
contract/frontier verification passes **33 tests**, with Python compilation
and `git diff --check` clean. The full current `hawking/tests` boundary also
passes **1,986 tests**, with 20 skips and 28 subtests. This is a deterministic
contract repair; the five completed lanes above were not replayed.

## H-Web parallel frontier projection — 2026-09-17

The typed H-Web Goal projection now preserves the durable parallel execution
frontier instead of exposing only the scalar active WorkUnit hint. Each Goal
view carries bounded `active_workunit_ids`, `runnable_frontier`, and
`frontier_waiting_dependencies` fields, with no scheduler or provider
authority added to Web. The projection regression and adjacent frontier
tests pass, and the full Hawking suite passes **1,988 tests**, with 20 skips,
28 subtests, and one pre-existing fork warning.

## Event-driven supervision — 2026-09-17

The four fresh DeepSeek completion-repair lanes were allowed to finish once;
all reached the known `OUTPUT_UNUSABLE` child-level result after real Hawking
read evidence. They were not retried and did not block the root. Their
durable identities were `64B9DFBA`, `4484DBF8`, `68D15A42`, and `26734488`.

Supervision now has a deterministic watcher seam in
`hawking/supervisor_watcher.py`. It caches the governing objective by digest,
builds a bounded canonical Goal/WorkUnit state fingerprint, and returns a
`WATCHER` no-action decision when unchanged. Meaningful event names and state
changes wake `LIGHT_REVIEW`; changed fingerprints are written once to the
Hawking-owned `.hawking/supervision/<goal>.deltas.jsonl` ledger. This is a
watcher around the existing Goal frontier, not a second scheduler or receipt
store. Focused watcher/frontier/normalizer verification passes **30 tests**;
Python compilation and `git diff --check` are clean. The root remains
`GOAL-B33C69F193` in `RUNNING`, with canonical accounting reconciled to
**$0.519533547113 spent** and **$54.480466452887 remaining** from the `$55`
programme cap. The current frontier is empty after reconciliation, so the
watcher records the state and waits for a real event rather than issuing
another supervisor call.

The subsequent four-lane source audit also completed without an accepted
structured research packet: `F5B56AE7`, `3345B76B`, `188E5E1C`, and `958A50BB`
each ended `OUTPUT_UNUSABLE` after the known provider-output class, with real
read evidence retained. No format retries were issued. The root frontier was
reconciled again and remains `RUNNING`; total durable accounting is now
**$0.523370487113 spent** and **$54.476629512887 remaining**. Because this
repeats a known failure class, the watcher records it as child evidence and
returns to `WATCHING_FOR_STATE_CHANGE` rather than waking deep analysis.

Inspection of the latest four records confirms the exact repeated behavior:
each worker had admitted Hawking tools and successful `campaign.state` plus
`fs.read` evidence, but then emitted tool-shaped continuation signals rather
than a terminal research result. The unmet contract was therefore
`structured_result:*` plus `terminal_status`, not missing capability
projection. This remains the known provider-output/nonterminal-tool-signal
class; it is preserved as child evidence and does not justify another retry.

The watcher now also batches close child results into one compact packet. It
marks repeated `PROVIDER_OUTPUT_UNUSABLE`/`OUTPUT_UNUSABLE` results as
known-failure-only (`WATCHER`) while a new failure class or accepted child
requests `LIGHT_REVIEW`; this remains bookkeeping and does not mutate the
Goal or start a second scheduler. The watcher and DAG regression boundary is
now **15 tests passing**.

## Auto planning split and Phase G contract coverage — 2026-09-18

Per the updated operator policy, the root cloud pool now admits Kimi K3 for
bounded planning/review and DeepSeek V4.1 Flash for implementation. Kimi
completed a read-only Phase G planning WorkUnit (`2275D01B`) using real
`campaign.state`, `fs.list`, `fs.read`, and `fs.search` evidence; its packet
did not justify further premium escalation. A DeepSeek implementation child
was preserved as `OUTPUT_UNUSABLE` under the known nonterminal-output class.

Deterministic coverage closure added bounds, unknown-digest, malformed-digest,
and RED→GREEN tests for `document.extract`: **8 focused tests pass**. Total
canonical root accounting is **$0.555096881113 spent** and
**$54.444903118887 remaining**. Pulsar/Magnetar remain protected and are not
admitted as model identities.
