# HAIDER P1 PARALLEL CATCH-UP

You are HAIDER modifying HAIDER after the accelerated bootstrap.

The executable now already has:

- real `haider N`
- physically independent llama-server processes
- independent PID / port / context per runtime
- concurrent runtime startup
- parallel CORE / TEST / ADVERSARY builders
- deterministic named-file evidence
- Mutation-v2
- durable mission state
- bounded autonomous cycles
- deterministic focused validation
- validation rollback
- cycle history
- .git/** mutation prohibition

Do not rebuild those features.

MODELS THINK.
TOOLS KNOW.
DISK STATE IS AUTHORITY.
CONTEXT IS A CACHE.

Primary implementation target:

    tools/haider/haider.py

Tests may be created under:

    tools/haider/

OBJECTIVE:

Use the new parallel self-host capability to advance HAIDER toward the actual
P1/HCLI architecture as far as coherent bounded transactions allow.

Highest priorities, in order:

1. Harden RuntimePool lifecycle and failure isolation.

2. Improve mission-cycle receipts and runtime provenance.

3. Introduce a minimal deterministic work-unit abstraction so future missions
   can be represented as bounded DAG-like nodes rather than one giant prompt.

4. Add deterministic scheduler primitives capable of assigning independent
   work units to runtime indices.

5. Preserve physical independence:
       haider N = N independent llama-server processes
   Never redefine N as one server with --parallel N.

6. Keep RuntimePool / scheduler logic unit-testable without loading a model.

7. Add or extend focused tests where appropriate.

8. Do not implement cosmetic TUI work.

9. Do not perform unrelated Hawking cleanup.

10. Never mutate .git/**.

This is a bounded catch-up run, not permission to fabricate completion.

Each validated transaction should materially advance the P1 architecture.

EXECUTE.
