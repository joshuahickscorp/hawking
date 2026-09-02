# HAIDER SELF-HOST 02 — BOUNDED PERSISTENT MISSION ENGINE

You are HAIDER modifying HAIDER.

This is a substantial implementation mission.

Do not merely describe the architecture.
Do not rebuild Gate Zero.
Do not implement cosmetic UI.
Do not implement RuntimePool yet.

MODELS THINK.
TOOLS KNOW.
OBSERVE BEFORE CLAIMING.
DISK STATE IS AUTHORITY.
CONTEXT IS A CACHE.

======================================================================
VERIFIED BASE
======================================================================

The harness already provides:

- `haider` shell command
- --task
- --task-file
- --max-cycles
- mission -> mutation grounding
- deterministic fast evidence when explicit missions name source files
- durable mission JSON under .haider/missions/
- owned llama-server lifecycle
- scoped mutations
- .git/** write prohibition
- deterministic validation
- machine receipts
- long model timeout
- targeted tests

Preserve these.

======================================================================
PRIMARY OBJECTIVE
======================================================================

Turn the current ONE-TRANSACTION executor into the smallest real bounded
persistent mission engine.

Current:

    mission
      -> evidence
      -> one edit
      -> validation
      -> exit

Target:

    mission
      -> cycle 1
      -> evidence
      -> edit
      -> validation
      -> persist
      -> reassess
      -> cycle 2
      -> ...
      -> completion OR max_cycles OR explicit failure

No human "continue" should be required between successful cycles.

======================================================================
FILES
======================================================================

Primary implementation target:

    tools/haider/haider.py

Tests may be created/modified under:

    tools/haider/

Prefer adding:

    tools/haider/test_haider_mission_engine.py

Reuse:

    tools/haider/test_haider_mission_state.py

Do not touch unrelated Hawking subsystems.

======================================================================
MISSION STATE
======================================================================

Use the existing mission state.

At minimum maintain:

    mission_id
    task
    source
    status
    created_at
    updated_at
    cycle
    max_cycles

Status should distinguish at least:

    running
    complete
    failed
    max_cycles

Persist state after every meaningful transition.

Cycle advances ONLY after:

    mutation succeeded
    AND
    deterministic validation succeeded

A failed mutation or validation may not advance cycle.

======================================================================
ENGINE CONTRACT
======================================================================

Create small testable orchestration functions rather than stuffing everything
into main().

Conceptually:

    run_mission_cycle(...)
    update_mission_state(...)
    mission_should_continue(...)

Exact names and decomposition are your choice.

A cycle should:

1. load current durable mission state
2. acquire evidence relevant to the SAME mission
3. reuse deterministic fast evidence where available
4. generate one coherent bounded mutation
5. apply through existing scoped-edit validation
6. execute deterministic validation
7. mechanically record results
8. advance cycle only on validated success
9. persist state
10. decide continue / complete / fail

======================================================================
COMPLETION
======================================================================

Do NOT let the model simply assert "mission complete" without evidence.

For this bootstrap, use a narrow completion protocol.

It is acceptable to require a structured model decision grounded in:

    current mission
    current source state
    previous cycle evidence
    validation results

But the harness owns state transitions.

A model completion claim accompanied by failed validation is NOT completion.

If a robust completion evaluator is too large for this transaction, implement
a conservative interim policy that continues until explicit evidence or
max_cycles.

Never silently label incomplete work PASS.

======================================================================
VALIDATION
======================================================================

Move away from always calling only:

    test_p0_tool_bridge.py

The mission engine should be able to run validation appropriate to the changed
HAIDER surface.

For this phase, an explicit deterministic HAIDER-focused suite is acceptable.

Prefer direct tests such as:

    test_haider_mission_state.py
    test_haider_mission_engine.py
    test_haider_edit.py

Do not spend model inference choosing tests if the harness can select them.

======================================================================
RECEIPTS
======================================================================

Add mission-cycle evidence.

For every completed cycle mechanically retain:

    cycle
    edited path
    old hash
    new hash
    validation command
    validation exit code
    elapsed time

The final mission record should identify:

    completed cycles
    status
    reason for stop

Do not fabricate evidence.

======================================================================
PERFORMANCE
======================================================================

The previous observation agent could consume 8-14 model turns merely locating
files.

That is now unacceptable for explicit missions naming concrete files.

Use the deterministic fast-evidence path.

Do not launch an exploratory Session when sufficient named-file evidence is
already available.

Spend model inference on:

    synthesis
    architecture decisions
    ambiguous reasoning

Do NOT spend it on:

    locating explicitly named paths
    reading explicitly named files
    hashing
    test execution
    state serialization

======================================================================
TESTS
======================================================================

Add deterministic tests covering at minimum:

1. cycle starts from persisted state
2. successful validated cycle increments cycle
3. failed validation does not increment
4. failed validation cannot mark complete
5. max_cycles stops execution
6. max_cycles status persists
7. successful completion status persists
8. mission text is authoritative across cycles
9. restart can load existing mission state
10. cycle evidence is retained
11. .git/** remains rejected
12. loop is deterministically bounded

No live model is required for unit tests.

Use mocks/fakes where appropriate.

======================================================================
IMPORTANT
======================================================================

This mission may make a meaningful multi-function change to haider.py.

The previous artificial 1-5 line mutation ceiling no longer applies.

However, remain coherent:

    one transaction
    one primary implementation target
    one architectural objective

Do NOT implement final RuntimePool.

Do NOT implement MemGate yet.

Do NOT implement TUI yet.

======================================================================
STOP CONDITION
======================================================================

SELF-HOST 02 succeeds when the executable contains a mechanically bounded,
persistent multi-cycle mission engine with deterministic tests proving its
state transitions.

EXECUTE.
