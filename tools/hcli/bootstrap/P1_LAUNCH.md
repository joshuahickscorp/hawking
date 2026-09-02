# P1 MAX — EXECUTION LAUNCH

Governing specification:

    tools/haider/P1_HAIDER_PRODUCTIZATION_MAX.md

System contract:

    tools/haider/HAIDER_SYSTEM_PROMPT.txt

P0 executable:

    tools/haider/p0_tool_bridge.py

P0 deterministic suite:

    tools/haider/test_p0_tool_bridge.py

======================================================================
ESTABLISHED LIVE EVIDENCE
======================================================================

P0 is no longer hypothetical.

Observed locally:

    37 / 37 deterministic tests PASS.

Observed live:

    model
      ->
    git.status
      ->
    actual repository execution
      ->
    structured observation
      ->
    model continuation

The successful final P0-A run required:

    2,199 prompt tokens
      381 completion tokens
    2,580 total tokens

and the actual git.status tool completed in approximately 107 ms.

Current observed branch:

    odyssey-i

Current observed tree:

    dirty

The first-class git.status defect caused by unsupported --no-color was repaired.

The fs.read truncation-semantics defect was repaired.

DO NOT REBUILD OR RE-LITIGATE P0-A.

======================================================================
P1 EXECUTION
======================================================================

Read P1_HAIDER_PRODUCTIZATION_MAX.md.

Do not restate it.

Audit the ACTUAL existing HAIDER implementation before creating architecture.

Use deterministic evidence aggressively.

The immediate critical path is:

    existing HAIDER audit
        ↓
    minimum real `haider 1`
        ↓
    model -> tool -> observation -> edit -> test -> receipt
        ↓
    GATE ZERO
        ↓
    AIDER CEASES TO BE PRIMARY PARENT
        ↓
    HAIDER BUILDS THE REST OF P1

The final `haider N` contract is:

    N independent inference runtimes

not merely:

    one server with N slots.

`haider 3` must eventually supervise up to three distinct llama-server
processes with distinct PIDs, ports, contexts, session state, roles and
lifecycles, all potentially backed by the same model artifact.

The current manually launched :8081 server is BOOTSTRAP INFRASTRUCTURE.

It is not the final RuntimePool.

HAIDER must progressively eliminate:

    manual server startup
    manual context settings
    manual prompt injection
    manual ports
    manual worker management
    Aider itself

======================================================================
EXECUTION ECONOMICS
======================================================================

P0 demonstrated the performance principle directly.

Broken primitive:

    ~22.8K tokens spent finding fallback evidence.

Fixed primitive:

    one git.status call
    ~107 ms
    ~2.6K total model tokens.

Therefore:

    FIX TOOLS BEFORE SPENDING REASONING TOKENS AROUND THEM.

Repeated expensive model work that can become deterministic infrastructure
should be compiled into the harness.

Do not tolerate avoidable reasoning loops.

======================================================================
EVIDENCE
======================================================================

No acceptance gate may be satisfied by model-authored plausible output.

Acceptance evidence must originate from real observations.

If a test cannot be run:

    UNVERIFIED

not:

    PASS.

======================================================================
BEGIN
======================================================================

1. Inspect current Git state.
2. Inspect existing HAIDER implementation.
3. Run the narrowest existing HAIDER tests/build checks.
4. Determine the minimum delta required for Gate Zero.
5. Implement that delta.
6. Prove Gate Zero live.
7. Transfer primary development from Aider to HAIDER.
8. Continue P1 through HAIDER itself.
9. Keep independent ready work saturated.
10. Stop only when P1's actual acceptance gates are mechanically proven.

BEGIN NOW.
