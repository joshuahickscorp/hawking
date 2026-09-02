# HAIDER — FIX MISSION-TO-EDIT GROUNDING

ONE ATOMIC IMPLEMENTATION TASK.

The previous self-host run exposed a correctness defect.

A supplied --task-file mission requested durable mission-state implementation.

Observation inspected the relevant HAIDER source.

But the edit phase ignored that implementation objective and instead changed:

    .git/config

That is incorrect.

The outer deterministic validator then proved the required implementation
was absent because:

    tools/haider/test_haider_mission_state.py

did not exist.

Fix THIS defect only.

======================================================================
OBJECTIVE
======================================================================

Make an explicit --task or --task-file mission constrain the edit phase.

The edit generator must receive and obey the explicit mission.

It must not fall back to the old generic Gate-Zero behavior of choosing an
arbitrary small safe repository edit.

======================================================================
REQUIRED INVARIANTS
======================================================================

When mission_text is not None:

    observation is about mission_text
    edit request is about mission_text
    validation is about the resulting implementation

The explicit mission must flow through the entire transaction:

    mission
      -> observation
      -> edit generation
      -> scoped mutation
      -> validation

Do not merely inject the mission into observation and then lose it before
edit generation.

======================================================================
PATH SAFETY
======================================================================

HAIDER must NEVER edit repository metadata under:

    .git/
    .git/**

Reject those paths deterministically before mutation.

This includes:

    .git/config

The model cannot override this rule.

Also reject the repository .git path itself.

Do not rely on prompting for this protection.

Implement it in deterministic path validation.

======================================================================
EDIT CONTRACT
======================================================================

For explicit missions, the edit-generation prompt/request must clearly
contain:

1. the exact mission
2. grounded observations gathered for that mission
3. the requirement to modify only files necessary to implement that mission
4. prohibition on arbitrary cleanup or unrelated safe edits
5. requirement to return no edit rather than an unrelated edit

Conceptually:

    If you cannot produce a grounded edit that advances THIS mission,
    return failure/no-edit.

Never substitute:

    "some safe edit"

for:

    "an edit implementing the supplied mission"

======================================================================
GATE-ZERO COMPATIBILITY
======================================================================

Preserve the no-explicit-task Gate-Zero fallback.

If:

    mission_text is None

the old Gate-Zero behavior may remain available.

If:

    mission_text is not None

mission-directed behavior is mandatory.

======================================================================
TESTS
======================================================================

Add or extend focused deterministic tests proving:

1. explicit mission reaches edit-generation input
2. task-file mission reaches edit-generation input
3. .git/config is rejected as an edit target
4. .git/anything is rejected
5. ordinary repository source path remains allowed
6. explicit mission cannot silently use generic safe-edit instruction
7. no-task fallback remains available

Do not run unrelated full regression suites.

Run only tests touching:

    mission ingress
    mission/edit grounding
    scoped path safety

plus py_compile.

======================================================================
SCOPE
======================================================================

Prefer modifying only:

    tools/haider/haider.py
    tools/haider/test_haider_edit.py

or one new focused test file if cleaner.

Modify p0_tool_bridge.py ONLY if .git path rejection genuinely belongs there.

Do not implement mission persistence yet.

Do not implement multi-cycle execution.

Do not implement RuntimePool.

======================================================================
STOP
======================================================================

Success is:

    explicit mission survives into edit phase
    arbitrary unrelated edits are forbidden
    .git/** mutation is mechanically impossible
    focused tests prove those properties

Nothing more.

EXECUTE.
