# HAIDER SELF-HOST 01 — DURABLE MISSION STATE + INCREMENTAL TESTGATE SEED

This is ONE bounded self-host implementation task.

Do not implement all of P1.
Do not implement RuntimePool.
Do not implement multi-agent scheduling.
Do not implement the full multi-cycle executor.
Do not redesign already-working Gate Zero machinery.

MODELS THINK.
TOOLS KNOW.
OBSERVE BEFORE CLAIMING.
DISK STATE IS AUTHORITY.
CONTEXT IS A CACHE.

======================================================================
VERIFIED BASE
======================================================================

HAIDER already has mechanically proven:

- owned llama-server startup/shutdown
- model inference
- deterministic repository tools
- model -> tool -> observation continuation
- grounded scoped edit
- deterministic validation
- machine-generated Gate Zero receipts
- --task
- --task-file
- installed `haider` shell command

Do not rebuild these.

======================================================================
PRIMARY OBJECTIVE
======================================================================

Add the smallest durable mission-state substrate required for later
multi-cycle autonomous self-hosting.

An explicit mission supplied by:

    haider 1 --task "..."
or
    haider 1 --task-file PATH

must create persistent machine-readable mission state under:

    .haider/missions/

Do NOT implement the future multi-cycle execution loop yet.

======================================================================
MISSION STATE CONTRACT
======================================================================

Persist JSON containing at least:

    mission_id
    task
    source
    status
    created_at
    updated_at
    cycle
    max_cycles

Initial values:

    status = "running"
    cycle = 0

Add CLI:

    --max-cycles N

Default:

    8

Require:

    N >= 1

Do not change the meaning of --max-turns.

Mission IDs must be generated mechanically, not by model prose.

UUID, timestamp+hash, or equivalent deterministic harness-generated identity
is acceptable.

======================================================================
FUNCTIONAL SURFACE
======================================================================

Prefer small testable functions conceptually equivalent to:

    create_mission_state(...)
    write_mission_state(...)
    load_mission_state(...)

Exact names are your choice.

Mission-state writes should be crash-resistant enough for this bootstrap:

    write temporary file
    fsync/close if practical
    atomic replace

Do not require model inference for state serialization.

======================================================================
MISSION PROVENANCE
======================================================================

For --task-file retain the actual resolved task-file path as provenance.

For --task record inline provenance.

Do not lose the exact mission text.

When an explicit mission starts, print:

    [haider] mission: .haider/missions/<mission-id>.json

before model execution.

The no-task Gate Zero fallback may remain as-is and does not need a persistent
mission record yet.

======================================================================
INCREMENTAL VALIDATION / TESTGATE SEED
======================================================================

We are eliminating redundant full-suite execution.

Add the smallest deterministic foundation for incremental validation.

The principle is:

    TEST WHAT CHANGED
    +
    TEST ITS CONTRACT BOUNDARY
    +
    REUSE PREVIOUS PASS WHEN DEPENDENCIES ARE UNCHANGED
    +
    FULL REGRESSION ONLY AT MILESTONES OR SHARED-PRIMITIVE CHANGES

For THIS phase, do not build a sophisticated dependency graph.

Implement only enough structure to support future TestGate evolution.

At minimum, create a simple machine-readable validation record containing:

    executed_tests
    skipped_tests
    reason

It is acceptable for this first version to use explicit named test groups.

Suggested groups:

    mission_ingress
    mission_state
    scoped_edit
    tool_bridge
    runtime

For this specific mission-state change, the expected minimal validation is:

    py_compile tools/haider/haider.py
    test_haider_mission_ingress.py
    test_haider_mission_state.py

Do NOT automatically run:

    test_p0_tool_bridge.py
    test_haider_edit.py

unless this implementation modifies their dependency surfaces.

If shared primitives are changed, escalate appropriately.

======================================================================
HASH-BASED REUSE SEED
======================================================================

If practical within this bounded task, add a tiny helper that can hash files
relevant to a validation group using SHA-256.

Do NOT build the complete dependency engine yet.

The goal is simply to make future validation capable of saying:

    prior PASS remains valid because relevant source hashes are unchanged

rather than rerunning everything blindly.

This helper must be deterministic and model-independent.

======================================================================
TESTS TO ADD
======================================================================

Add ONE focused test file:

    tools/haider/test_haider_mission_state.py

Test at least:

1. mission state creation
2. mission JSON write
3. mission JSON load
4. mission_id exists
5. initial status == "running"
6. initial cycle == 0
7. default max_cycles persisted
8. custom max_cycles persisted
9. max_cycles <= 0 rejected
10. inline task provenance retained
11. task-file provenance retained
12. mission text retained
13. atomic write leaves valid JSON
14. validation-group record can represent executed/skipped tests
15. source hashing is deterministic if hashing helper is implemented

Do not require a live model for these tests.

======================================================================
EDIT SCOPE
======================================================================

Prefer:

    tools/haider/haider.py
    tools/haider/test_haider_mission_state.py

At most add one small helper module under:

    tools/haider/

if separating mission state materially improves clarity.

Do not touch unrelated Hawking code.

Do not modify p0_tool_bridge.py unless absolutely required.

Do not modify scoped-edit behavior unless absolutely required.

======================================================================
VALIDATION POLICY FOR THIS TASK
======================================================================

Run ONLY the directly relevant tests unless your actual diff crosses a shared
boundary.

Required:

    python3 -m py_compile tools/haider/haider.py
    python3 tools/haider/test_haider_mission_ingress.py
    python3 tools/haider/test_haider_mission_state.py

If you add a helper module:

    py_compile that module too

Run broader tests ONLY if:

    - you modify p0_tool_bridge.py
    - you modify RepositoryGuard
    - you modify scoped-edit logic
    - you modify model-client protocol
    - you modify runtime lifecycle
    - a targeted test reveals unexpected collateral behavior

Do not waste wall-clock time rerunning already-proven unrelated suites.

======================================================================
RECEIPT TRUTH
======================================================================

Model prose cannot claim tests passed.

Executed validation evidence must come from subprocess results.

Skipped tests must be explicitly marked skipped/reused, never represented as
executed PASS.

======================================================================
STOP CONDITION
======================================================================

This phase is complete when:

    explicit missions create durable mission state
    --max-cycles exists and validates
    mission state survives load/write
    minimal validation metadata exists
    targeted deterministic tests pass
    existing mission ingress remains functional

Do NOT continue to multi-cycle execution in this invocation.

Do NOT implement RuntimePool.

Do NOT broaden scope.

Make the smallest correct high-leverage change.

EXECUTE.
