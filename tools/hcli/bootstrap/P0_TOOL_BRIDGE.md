# HAIDER P0 — AUTONOMOUS MULTI-SLOT MODEL → TOOL → RESULT BOOTSTRAP

This is the immediate bootstrap override.

The governing long-term Ultragoal remains:

    docs/ultragoals/HAIDER_HCLI_SUPER_AGENT_OS.md

Do NOT attempt broad speculative Agent OS implementation before this P0 passes.

======================================================================
0. WHY THIS EXISTS
======================================================================

The current verified blocker is architectural:

    Aider owns the chat loop
    and the resident model does not possess deterministic eyes/hands.

The resident can reason.

It cannot yet autonomously:

    inspect Git
    search the repository
    read arbitrary source
    run tests
    execute safe commands
    observe results
    continue reasoning

Repo-map summaries are navigation hints.

They are NOT implementation authority.

MODELS THINK.
TOOLS KNOW.

This bootstrap exists to cross that boundary.

======================================================================
1. DELIVERABLE
======================================================================

Build:

    tools/haider/p0_tool_bridge.py
    tools/haider/test_p0_tool_bridge.py

This program is independent of Aider's conversation loop.

It talks directly to the running local OpenAI-compatible server:

    http://127.0.0.1:8081/v1/chat/completions

Default model:

    qwen3.8-27b-abliterated

It must own:

    model inference
    deterministic tool dispatch
    structured observations
    iterative continuation
    bounded multi-agent execution

This is the first embryo of HCLI-owned cognition.

======================================================================
2. DO NOT BUILD A FAKE WRAPPER
======================================================================

P0 is NOT:

    a shell script that runs git before asking the model
    a canned sequence of repository commands
    a prompt that tells the model to imagine tools
    a mock where tests simulate successful execution
    a user-mediated copy/paste loop

The MODEL must decide when deterministic evidence is needed.

The harness executes it.

The observation returns to the model.

The model continues.

======================================================================
3. STRICT PROTOCOL
======================================================================

Model output MUST be parsed as machine-readable JSON.

A model turn may produce exactly one of:

TOOL:

    {
      "type": "tool",
      "name": "git.status",
      "args": {}
    }

FINAL:

    {
      "type": "final",
      "content": "..."
    }

For worker orchestration:

SPAWN:

    {
      "type": "spawn",
      "role": "scout",
      "task": "...",
      "context": {...}
    }

Malformed output is NOT interpreted heuristically.

Instead:

    reject
    emit deterministic protocol error
    bounded retry

Never convert prose that merely looks command-like into execution.

======================================================================
4. TOOL SET
======================================================================

Implement at minimum:

    git.status
    git.diff
    git.log

    repo.search

    fs.read
    fs.list
    fs.stat

    shell.run_safe

    build.check
    test.run

Every operation returns structured evidence including where applicable:

    tool
    args
    cwd
    repository_root
    exit_code
    stdout
    stderr
    elapsed_ms
    truncated
    timestamp

Output must be bounded.

If truncated, mark it explicitly.

======================================================================
5. REPOSITORY CONTAINMENT
======================================================================

Detect repository root with deterministic Git logic.

All filesystem operations must remain within that root.

Canonicalize paths.

Reject traversal such as:

    ../../
    symlink escape
    absolute path outside project

unless explicitly admitted by future policy.

P0 defaults to current Hawking repository only.

======================================================================
6. SAFE SHELL
======================================================================

Do NOT expose unrestricted shell initially.

Allow deterministic command families required for bootstrap:

    git status
    git diff
    git log
    git show

    rg

    cargo check
    cargo test

    python / python3
        only for specifically admitted local HAIDER bootstrap/test paths

Reject destructive operations including:

    rm -rf
    git reset --hard
    git clean
    git prune
    destructive git gc
    sudo
    killall
    reboot
    shutdown
    arbitrary remote shell installers

Do not rely only on substring filtering.

Represent command admission structurally.

======================================================================
7. MODEL CONTEXT
======================================================================

Do NOT start with tiny artificial context.

The local runtime currently provides a large per-slot context budget.

Discover or configure the actual slot budget.

Protect an output reserve.

Use context efficiently but do not prematurely compact during this bootstrap.

Stable prefix should contain:

    protocol
    tool schemas
    project root
    P0 invariant
    current task

Dynamic context contains:

    observed tool results
    relevant exact source
    worker findings

Never repeatedly inject the full long-term Ultragoal into every child.

======================================================================
8. THREE-LANE EXECUTION
======================================================================

The running llama-server currently exposes THREE logical slots.

P0 should deliberately be capable of using them.

Maximum bootstrap topology:

                    PARENT
                  /        \
              SCOUT      ADVERSARY

All are logical sessions over ONE resident model body.

Do NOT launch three model processes.

======================================================================
9. PARENT ROLE
======================================================================

Parent owns:

    task
    tool authorization
    synthesis
    final answer
    worker admission
    evidence requirements

Parent must continue making progress while independent workers run when useful.

Parent should NOT spawn workers for trivial deterministic tasks.

======================================================================
10. SCOUT ROLE
======================================================================

Scout is read-only.

Purpose:

    locate relevant files
    identify symbols
    inspect implementation
    report evidence
    suggest next deterministic reads

Scout cannot edit.

Scout cannot claim unobserved repository facts.

======================================================================
11. ADVERSARY ROLE
======================================================================

Adversary is read-only.

Purpose:

    attack protocol
    find unsafe command paths
    challenge assumptions
    detect fabricated evidence
    identify likely test failures
    search for conflicting existing architecture

Adversary cannot edit.

======================================================================
12. FORCE-PARALLEL MODE
======================================================================

Provide CLI option:

    --lanes N

where initially:

    N ∈ {1,2,3}

Also provide:

    --force-parallel

When enabled and N >= 3:

    launch SCOUT and ADVERSARY concurrently

using separate HTTP requests to the shared resident.

This exists for testing and benchmarking.

Normal future Agent OS policy may decide whether parallelism has positive EV.

P0 acceptance MUST include one forced 3-lane run.

======================================================================
13. CONCURRENCY IMPLEMENTATION
======================================================================

Use Python standard library where practical.

Acceptable:

    concurrent.futures
    threading
    asyncio

Do not add a giant framework.

Each logical session has independent message history.

Shared immutable prefix may be identical.

Workers return structured packets.

Do not concatenate giant raw transcripts.

======================================================================
14. WORKER RESULT FORMAT
======================================================================

Workers return compact structured findings:

    {
      "role": "...",
      "claims": [...],
      "evidence": [...],
      "files": [...],
      "commands": [...],
      "risks": [...],
      "next_actions": [...]
    }

Parent receives normalized worker outputs.

No worker result becomes truth merely because it sounds convincing.

Important claims require deterministic tool evidence.

======================================================================
15. MODEL → TOOL → RESULT LOOP
======================================================================

Core loop:

    model(messages)
        ↓
    parse strict protocol
        ↓
    if TOOL:
        execute deterministic operation
        append structured observation
        continue

    if SPAWN:
        admit bounded worker
        run session
        harvest result
        append structured result
        continue

    if FINAL:
        return

Bound iterations.

Prevent infinite loops.

======================================================================
16. ANTI-LOOP WATCHDOG
======================================================================

Track:

    model turns
    tool calls
    identical repeated calls
    protocol errors
    elapsed time
    evidence acquired

If the model repeatedly reasons without requesting required evidence:

    inject a compact deterministic reminder:

        UNOBSERVED CLAIMS REQUIRE TOOLS.

If the same failed operation repeats:

    stop or require a different action.

======================================================================
17. LIVE P0-A — SINGLE-LANE TOOL PROOF
======================================================================

Run:

    python3 tools/haider/p0_tool_bridge.py \
      --lanes 1 \
      --task "Use git.status to inspect this repository. Do not guess. Tell me the current branch and whether the tree is dirty."

PASS only if:

    MODEL requests git.status
        ↓
    actual git executes
        ↓
    structured observation returns
        ↓
    MODEL continues
        ↓
    final answer agrees with observation

No user mediation.

======================================================================
18. LIVE P0-B — EXACT FILE READ
======================================================================

Run:

    python3 tools/haider/p0_tool_bridge.py \
      --lanes 1 \
      --task "Use fs.read to read docs/ultragoals/HAIDER_HCLI_SUPER_AGENT_OS.md. Return only the implementation-order phase names."

PASS only if:

    fs.read was actually invoked

and answer derives from observed source.

======================================================================
19. LIVE P0-C — FORCED THREE-LANE PROOF
======================================================================

Run:

    python3 tools/haider/p0_tool_bridge.py \
      --lanes 3 \
      --force-parallel \
      --task "Inspect the existing HAIDER bootstrap implementation. Scout should locate and inspect relevant implementation. Adversary should identify unsafe assumptions or conflicts. Parent must synthesize only observed evidence and identify the first real implementation gate."

PASS only if:

    three model requests overlap in wall time
    shared resident remains ONE process
    SCOUT runs
    ADVERSARY runs
    parent receives both results
    deterministic repository evidence is used
    no guessed implementation claims enter synthesis

Record:

    start/end timestamps
    per-lane latency
    total wall time
    model usage if returned

======================================================================
20. LIVE P0-D — TEST TOOL
======================================================================

After inspecting actual HAIDER source:

    have the parent select the narrowest relevant cargo check/test

The model requests the test.

Harness executes it.

Observed result returns.

Model interprets actual success/failure.

======================================================================
21. UNIT TESTS
======================================================================

Unit tests must cover:

    strict JSON protocol
    malformed output
    root containment
    symlink/path escape
    safe command admission
    destructive command rejection
    output truncation
    timeout
    repeated-call watchdog
    worker result normalization
    lane-count validation

Mock the HTTP model for deterministic unit tests.

Do NOT mock live acceptance.

======================================================================
22. OBSERVABILITY
======================================================================

CLI emits compact events such as:

    ● PARENT
    ● SCOUT
    ● ADVERSARY

    → TOOL git.status
    ← OK 31 ms

    → READ ...
    ← 214 lines

    → TEST cargo test ...
    ← exit=0

    ✓ WORKER scout 3.2s
    ✓ WORKER adversary 4.1s

    ✓ FINAL

Do not dump raw hidden reasoning by default.

Optional:

    --debug

may expose raw model payloads.

======================================================================
23. SERVER / MODEL SETTINGS
======================================================================

Use the existing local server.

Do not launch another Qwen process.

Do not download models.

Do not silently use external APIs.

Environment/config may override:

    HAIDER_API_BASE
    HAIDER_MODEL
    HAIDER_API_KEY

Defaults:

    HAIDER_API_BASE=http://127.0.0.1:8081/v1
    HAIDER_MODEL=qwen3.8-27b-abliterated
    HAIDER_API_KEY=sk-local

======================================================================
24. NO AIDER DEPENDENCY INSIDE BRIDGE
======================================================================

Aider is allowed to WRITE this bootstrap.

The resulting p0_tool_bridge.py itself must not require Aider to operate.

That proves the beginning of detachment:

    Aider writes bridge
        ↓
    bridge runs Qwen autonomously
        ↓
    bridge inspects Hawking
        ↓
    HAIDER absorbs bridge
        ↓
    Aider disappears

======================================================================
25. DO NOT INVENT EXISTING HAIDER STATE
======================================================================

Until the bridge itself can read real files:

DO NOT create guessed:

    mission schemas
    DAG schemas
    CLI architecture
    Rust APIs
    Cargo integration
    persistence formats

After the bridge works:

USE THE BRIDGE TO READ THEM.

======================================================================
26. HANDOFF TO SUPER AGENT OS
======================================================================

Only after P0-A through P0-D pass:

    P0_TOOL_BRIDGE = VERIFIED

Then use the bridge itself to:

    git.status
    git.diff
    repo.search
    fs.read
    build.check
    test.run

Audit the actual overnight HAIDER code.

Then resume:

    docs/ultragoals/HAIDER_HCLI_SUPER_AGENT_OS.md

from the first evidence-backed unmet acceptance gate.

At that point the bridge becomes:

    temporary adapter
        ↓
    HCLI ToolBus
        ↓
    native Agent OS primitive

======================================================================
27. PRIMARY SUCCESS METRIC
======================================================================

Not lines of code.

Not reasoning length.

Not number of files created.

Success is:

    VERIFIED AUTONOMOUS ENGINEERING ACTION / SECOND

======================================================================
28. CORE DOCTRINE
======================================================================

MODELS THINK.
TOOLS KNOW.

OBSERVE BEFORE CLAIMING.

ONE RESIDENT.
MULTIPLE LOGICAL SESSIONS.

CONTEXT IS A CACHE.
DISK STATE IS AUTHORITY.

PARALLELISM MUST BE MEASURABLE.

AIDER IS TEMPORARY.

BUILD THE EYES AND HANDS FIRST.

BEGIN.
