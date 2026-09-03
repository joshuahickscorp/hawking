# HAIDER P1 — SELF-HOST PHASE 1

You are HAIDER after mechanically proven Gate Zero.

Gate Zero already proved:

HAIDER
-> owns a local llama-server runtime
-> invokes a model
-> executes deterministic repository tools
-> returns structured observations
-> continues reasoning from observed evidence
-> chooses a grounded scoped edit
-> performs a bounded repository mutation
-> executes deterministic validation
-> writes a machine-generated receipt
-> shuts down its owned runtime
-> returns success

Do not rebuild Gate Zero.

Do not use Aider.

Do not merely write architecture prose.

MODELS THINK.
TOOLS KNOW.
OBSERVE BEFORE CLAIMING.
DISK STATE IS AUTHORITY.
CONTEXT IS A CACHE.
EXECUTION EVIDENCE OUTRANKS MODEL CLAIMS.

======================================================================
PRIMARY OBJECTIVE
======================================================================

Transform the existing single bounded Gate-Zero transaction into the
smallest useful SELF-HOSTING MISSION ENGINE.

The immediate capability target is:

mission
-> observe
-> select one bounded implementation unit
-> edit
-> validate
-> record evidence
-> inspect resulting state
-> decide whether mission is complete
-> if incomplete, continue autonomously with the next bounded unit
-> stop only on completion, explicit failure, or deterministic budget

The human must not need to type "continue" between work units.

======================================================================
MISSION INPUT
======================================================================

Treat the supplied mission as first-class persistent state.

Support and preserve:

haider 1 "mission"
haider 1 --task "mission"
haider 1 --task-file PATH

Record mission provenance.

======================================================================
MULTI-CYCLE ENGINE
======================================================================

Implement the narrowest robust multi-cycle execution loop.

Each cycle must have an explicit bounded structure:

1. observe current repository/evidence state
2. determine smallest useful next work unit
3. obtain exact target evidence
4. make one or more tightly scoped related edits
5. validate deterministically
6. record hashes/results
7. update durable mission state
8. determine completion or next cycle

No unbounded while-true loop.

Provide a deterministic cycle ceiling.

A failure in validation must never silently advance as success.

======================================================================
DURABLE STATE
======================================================================

Use .haider/ as the runtime state plane.

At minimum:

.haider/
  missions/
  receipts/
  logs/

Mission state must survive:

- model context replacement
- process restart
- individual inference failure

Do not make long-term mission continuity depend on one giant conversation.

======================================================================
RECEIPTS
======================================================================

Evolve Gate-Zero receipts toward mission receipts.

Mechanically record at least:

- mission id
- task provenance
- model
- runtime PID
- runtime port
- cycle
- tool observations
- edited paths
- old hashes
- new hashes
- validation command
- validation exit
- elapsed time
- token usage
- final state

Model-generated prose cannot substitute for execution evidence.

======================================================================
CONTEXT / PERFORMANCE
======================================================================

Do not solve the entire final context governor yet.

But establish the architectural separation:

DURABLE MISSION STATE != MODEL CHAT CONTEXT

Persist evidence summaries and receipts.

Reuse already-known deterministic facts.

Do not repeatedly spend inference rediscovering unchanged repository state.

Optimize:

VERIFIED ENGINEERING PROGRESS / WALL-CLOCK SECOND

The current local model is expensive enough that repeated reasoning which
can become deterministic harness behavior should become harness behavior.

======================================================================
FUTURE HAIDER N INVARIANT
======================================================================

Keep all new work compatible with:

haider N = up to N independent inference runtime processes

NOT:

one llama-server process with N logical slots.

Future independent runtimes own distinct:

- PID
- port
- context
- KV/session state
- role
- lifecycle

Do not prematurely implement the entire RuntimePool unless needed for this
phase.

======================================================================
TEST REQUIREMENTS
======================================================================

Preserve every existing green test.

Add deterministic tests for the mission engine.

At minimum test:

- inline mission accepted
- task-file mission accepted
- missing task file rejected
- conflicting task sources rejected
- mission state persists
- cycle count is bounded
- successful validation advances state
- failed validation cannot claim PASS
- receipts link to actual execution evidence
- owned runtime cleanup remains intact

Run all relevant tests.

Inspect actual diffs.

Do not claim PASS unless the executed tests support PASS.

======================================================================
STOP CONDITION
======================================================================

This mission is complete when HAIDER is materially capable of taking a
larger directive and autonomously executing multiple verified work units in
one invocation.

Do not expand into cosmetic CLI polish.

Do not rebuild already-proven components.

Build the smallest high-leverage step toward:

haider 1 --task-file tools/haider/P1_LAUNCH.md

then later:

haider 3

GO.
