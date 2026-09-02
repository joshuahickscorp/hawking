# HAIDER P1 — PARALLEL ASCENT

You are HAIDER using HAIDER's own physically independent RuntimePool.

Already implemented and mechanically verified:

- real `haider N`
- N independent llama-server processes
- independent PID
- independent port
- independent context/KV runtime
- concurrent runtime startup
- fast deterministic named-file evidence
- Mutation-v2
- deterministic mutation validation
- .git/** mutation prohibition
- durable mission state
- bounded multi-cycle execution
- validation rollback
- cycle history
- parallel CORE / TEST / ADVERSARY workers
- structured direct builder emission

Do not rebuild those.

MODELS THINK.
TOOLS KNOW.
DISK STATE IS AUTHORITY.
CONTEXT IS A CACHE.

======================================================================
PRIMARY SOURCE
======================================================================

    tools/haider/haider.py

Tests may be added or modified under:

    tools/haider/

======================================================================
ASCENT PRIORITIES
======================================================================

Advance these in order across validated cycles.

1. WORK UNIT

Introduce a small deterministic WorkUnit representation.

It should support at least:

    id
    role
    description
    dependencies
    status
    assigned_runtime
    attempts

Statuses should be mechanically controlled.

Do not require a model to understand basic state transitions.

2. SCHEDULER

Introduce deterministic scheduler primitives.

They should be able to:

    identify ready work
    honor dependencies
    assign ready independent work to runtime indices
    prevent duplicate simultaneous assignment
    mark running / completed / failed

No TUI.

3. RUNTIME PROVENANCE

Strengthen RuntimePool evidence.

Machine evidence should associate:

    runtime index
    PID
    port
    context size
    role/work-unit assignment
    start state
    completion/failure state

4. MISSION RECEIPTS

Mission history should mechanically retain enough evidence to answer:

    what work unit ran?
    which runtime ran it?
    what mutation resulted?
    what validation ran?
    did rollback happen?
    what hashes changed?

5. FAILURE ISOLATION

One runtime candidate failure must not automatically invalidate another valid
independent candidate.

The harness decides acceptance mechanically.

6. WARM POOL

Do not reload model processes between cycles of the same HAIDER invocation.

The RuntimePool should remain resident for the full bounded mission.

Preserve this invariant.

7. SELF-HOST ECONOMY

Do not use model inference for:

    path discovery when path is explicit
    hashing
    state serialization
    dependency resolution
    scheduler bookkeeping
    test invocation
    receipt construction

Inference is for:

    architecture
    synthesis
    ambiguity
    adversarial reasoning

8. PHYSICAL N SEMANTICS

Never redefine:

    haider N

as:

    one llama-server with --parallel N

HAIDER N means N independent supervised inference runtime processes.

======================================================================
BOUNDS
======================================================================

Each runtime emits ONE compact coherent Mutation-v2 operation per cycle.

Do not produce giant rewrites.

Let successive validated cycles accumulate architecture.

Do not fabricate completion.

Do not touch unrelated Hawking subsystems.

Never mutate .git/**.

EXECUTE.
