# HCLI + SUPER AGENT OS ULTRAGOAL

======================================================================
0. MISSION
======================================================================

THIS IS THE GOVERNING ULTRAGOAL.

Do not treat this as a request for another backend prototype.

The deliverable is simultaneously:

1. A genuinely usable interactive `hcli` coding-agent CLI.
2. The reusable Hawking Agent OS beneath it.
3. A local-first cognitive runtime capable of exploiting:
   - multiple logical agents,
   - persistent state,
   - dynamic contexts,
   - KV-cache control,
   - memory,
   - research DAGs,
   - adaptive test-time compute,
   - inter-agent communication,
   - self-hosted improvement.

The Product Plane and Agent OS Plane advance TOGETHER.

Backend completion without a usable CLI is failure.

A polished CLI sitting on fake/stub infrastructure is also failure.

======================================================================
1. IDENTITY
======================================================================

Canonical lineage, and it is now CLOSED:

    Aider
      ↓ temporary bootstrap substrate
    HCLI
      ↓ progressively shed Aider
    HCLI          ← we are here, 2026-09-02

This section set one condition -- "Aider dependency monotonically decreases
until HCLI effectively IS HCLI" -- and forbade a future giant rewrite. Both
held. The dependency reached zero first (no live import; see
`receipts/future/AIDER_NAMESPACE_AUDIT.json`), and the name followed without a
rewrite: the tests moved to `hcli/tests/`, the Rust module to
`hide_backend::hcli`, `parse_haider_args` to `parse_hcli_args`, and the
verbatim upstream `CoderPrompts` source that was still checked in was deleted.

There is no HCLI. There is HCLI, and it is not a fork of anything. Nothing in
this document below should be read as describing a derivative.

Track an explicit AIDER DEPENDENCY INDEX.

Components include:

    model loop
    repo map
    repository navigation
    working set
    context management
    session management
    edit application
    git
    terminal UI
    provider/model abstraction
    command layer

Classify each:

    AIDER_OWNED
    ADAPTED
    HCLI_OWNED

Long-term target:

    AIDER_OWNED = 0

======================================================================
2. HAWKING CANON
======================================================================

Respect existing Hawking authority boundaries.

    NOS      Noetic Operating System
    Doctor   semantic/equivalence authority
    Tabula   behavioral identity science
    Gravity  noetic compiler/superoptimizer
    NR       transient compiler IR
    NX       public machine-specialized executable
    NVM      Noetic Virtual Machine
    HIDE     physical runtime/realization layer
    HCLI     native coding/agent interface within HIDE,
             also directly launchable

Before building any subsystem:

    SEARCH EXISTING HAWKING IMPLEMENTATIONS.

Do not silently create parallel versions of:

    MemGate
    scheduling
    receipts
    skills
    state/checkpoints
    HCLI
    runtime control
    Doctor
    Gravity
    Odyssey

Extend or adapt authoritative implementations.

======================================================================
3. CURRENT BOOTSTRAP
======================================================================

Current temporary intelligence:

    Qwen3.8-27B abliterated Q5_K
    llama.cpp / llama-server
    Aider 0.86.2

Current local service is configured for multiple logical slots.

Qwen3.8 is temporary.

Aider is temporary.

llama.cpp is replaceable.

Immediate future resident:

    Qwen3-Coder-30B-A3B

Future:

    Gravity-built NX residents

must plug into the same Agent OS/HCLI architecture.

DO NOT hardcode Qwen3.8 assumptions.

======================================================================
4. RECOVERY / REPOSITORY PRECONDITION
======================================================================

Recent Git object corruption was repaired enough that the final fsck produced
dangling objects but no missing-object or unreadable-pack failures.

Do not casually run:

    git clean
    reset --hard
    prune
    destructive gc

The repository contains extensive concurrent Odyssey/Gravity evidence.

Preserve unrelated campaign work.

Before modifying code:

    verify current HEAD
    verify git status
    verify current diff
    identify HCLI/HCLI changes
    checkpoint relevant work safely

Use exact repository evidence.

======================================================================
5. MODELS THINK; TOOLS KNOW
======================================================================

HARD INVARIANT:

    MODELS THINK.
    TOOLS KNOW.

Never invent:

    repository contents
    command output
    test results
    Git state
    memory pressure
    process state
    benchmarks

Repository facts require:

    SEARCH
    READ
    RUN
    TEST

If a deterministic capability is unavailable:

    MISSING_CAPABILITY
    UNKNOWN
    BLOCKED

Do not spend tens of thousands of reasoning tokens compensating for a missing
tool.

======================================================================
6. ARCHITECTURAL INVERSION
======================================================================

The previous architecture was:

    Aider owns conversation
        ↓
    model generates response
        ↓
    user mediates files/tools

THIS MUST END.

Target:

    HCLI / Agent OS
        ↓
    model session
        ↓
    typed tool request
        ↓
    deterministic Tool Bus
        ↓
    structured observed result
        ↓
    model continues
        ↓
    edit backend
        ↓
    validation
        ↓
    receipt
        ↓
    next DAG node

Aider may temporarily provide:

    edit application
    repo-map implementation
    useful Git helpers

but does not own cognition.

======================================================================
7. DETERMINISTIC TOOL BUS
======================================================================

Finish and prove typed autonomous operations for:

    repo.search
    repo.find
    repo.read
    repo.symbols
    repo.callers

    fs.read
    fs.list
    fs.stat

    git.status
    git.diff
    git.log
    git.show
    git.branch
    git.worktree

    shell.run_safe

    build.run
    test.run
    lint.run

    worker.spawn
    worker.status
    worker.cancel
    worker.harvest

    context.status
    context.compact

    receipt.write
    receipt.query

Every result should carry provenance such as:

    tool
    request
    cwd
    timestamp
    exit status
    stdout
    stderr
    wall time
    evidence hash

Malformed model tool requests are rejected structurally.

Never interpret vague prose as successful tool execution.

======================================================================
8. ACTUAL HCLI CLI — HARD PRODUCT REQUIREMENT
======================================================================

THIS ULTRAGOAL DOES NOT COMPLETE UNTIL:

    $ hcli

works from a normal terminal inside a project.

No:

    python -m ...
    manual Aider command
    environment-variable ritual
    manual /add
    manual llama-server startup under normal operation
    repeated "continue"

The user mental model is:

    HCLI IS A LOCAL AGENT CLI.

Not:

    HCLI IS A WRAPPER AROUND AIDER.

======================================================================
9. INTERACTIVE CLI / TUI
======================================================================

HCLI should feel closer to Claude Code / Codex CLI than Aider.

Provide:

    persistent application
    fixed bottom composer
    multi-line input
    command history
    slash command completion
    path completion where appropriate
    scrolling event stream
    clean terminal resizing
    paste detection
    async worker updates
    Ctrl-C task cancellation
    clean exit/reconnect

Do not write a terminal emulator from scratch.

Reuse repository-native terminal infrastructure or a small suitable dependency.

Candidate Rust libraries may include:

    ratatui
    crossterm
    reedline
    rustyline

Measure before choosing.

======================================================================
10. VISUAL LANGUAGE
======================================================================

Conceptual UI:

    ┌ HCLI ──────────────────────────────────────────────┐
    │ MODEL     qwen3.8-local / Q30 later                 │
    │ PROJECT   hawking                                   │
    │ GOAL      Agent OS                                  │
    │ CTX       31K / 64K                                 │
    │ MEM       measured                                  │
    │ LANES     2 / 3                                     │
    └─────────────────────────────────────────────────────┘

      → SEARCH   MemGate
      → READ     agent_scheduler.rs
      ● WORKER   B · IMPLEMENTER
      → EDIT     context.rs
      → TEST     ...
      ✓ VERIFIED

    ───────────────────────────────────────────────────────
    ❯ composer
    ───────────────────────────────────────────────────────

Do not expose Aider branding or implementation terminology.

======================================================================
11. COMMAND SURFACE
======================================================================

Implement REAL first-class controls equivalent to:

    /model
    /goal
    /ultragoal
    /steer
    /resume
    /pause
    /status
    /frontier
    /workers
    /context
    /runtime
    /budget
    /receipt
    /why

    /find
    /read
    /symbols
    /callers

    /doctor
    /gravity
    /odyssey
    /bench

    /compact
    /help
    /exit

Do not satisfy this with cosmetic aliases that merely inject text prompts.

======================================================================
12. MODEL PICKER
======================================================================

/model must populate from actual local capabilities.

Discover/index:

    running Hawking/NX residents
    known GGUF models
    supported MLX artifacts
    configured local model directories
    usable Hugging Face cache artifacts

Do not scan the entire disk every invocation.

Registry entry may expose:

    display name
    architecture
    parameter count if known
    representation/quant
    artifact size
    runtime backend
    context capability
    current resident state

Selection persists.

======================================================================
13. MODEL PICKER KEYBOARD UX
======================================================================

Support:

    /model

always.

Also implement configurable keyboard invocation.

Desired:

    Command-M where host terminal permits capture

plus a portable non-host-owned binding.

Do not assume Command key chords are transmitted by every terminal.

Keyboard configuration must be editable.

======================================================================
14. MID-TURN STEERING
======================================================================

The user can type while work is active.

Example:

    /steer inspect the allocator before changing the runtime

Steer becomes a durable high-priority Agent OS event.

At earliest safe interruption boundary:

    scheduler observes steer
    priorities update
    context updates
    obsolete speculative work may be cancelled

Do not simply append steer after the task is finished.

======================================================================
15. LARGE PASTE / ULTRAGOAL INGESTION
======================================================================

Large pasted content becomes an object.

Example:

    [PASTE · 22 KB · ~5K tokens]

Store raw source once.

Display compact metadata.

For /ultragoal:

    ingest
    assign ID
    derive invariants
    obligations
    acceptance criteria
    DAG/frontier

Do not repeatedly repaste the full source into contexts.

======================================================================
16. DURABLE STATE
======================================================================

Authoritative persistent state belongs on disk.

Persist:

    Goals
    Ultragoals
    Steers
    DAG
    Frontier
    Receipts
    Workers
    Context policy
    Runtime policy
    Memory
    Skills
    Recovery state
    NEXT_ACTION

Conversation is disposable.

Terminal is disposable.

Model context is disposable.

======================================================================
17. CONTEXT OS
======================================================================

Upgrade the current context governor into a real Context OS.

CONTEXT IS A CACHE, NOT A DATABASE.

Represent context as segments, not one append-only conversation.

Example:

    immutable_system
    stable_agent_os_prefix
    ultragoal_kernel
    current_node
    source_working_set
    receipts_hot
    hypotheses
    tool_ephemera
    generation_reserve

Each segment should carry policy such as:

    priority
    reconstructability
    TTL
    dependency set
    compression policy
    pin state
    KV identity
    provenance

======================================================================
18. ACTUAL CONTEXT ACCOUNTING
======================================================================

Measure real packed tokens.

Do not trust configured map-token values.

Track actual contribution from:

    system
    stable prefix
    project map
    task map
    editable source
    read-only source
    Ultragoal
    receipts
    history
    tool results
    generation reserve

Hard invariant:

    projected_input + protected_generation_reserve
        < actual_slot_context

Current runtime slot capacity is authoritative.

======================================================================
19. TASK-LOCAL CONTEXT
======================================================================

Prefer:

    tiny project skeleton
        ↓
    current obligation
        ↓
    deterministic retrieval
        ↓
    task-local mini-map
        ↓
    exact source

over giant whole-repository maps.

Build or reuse a deterministic project index from:

    git files
    language AST
    imports
    symbols
    callers
    references
    Git history
    receipts

Cache and invalidate incrementally.

======================================================================
20. AUTOMATIC WORKING SET
======================================================================

No human /add workflow.

Agent OS owns:

    discovery
    exact reads
    read-only context
    editable promotion
    stale eviction

Acceptance:

    zero manual file additions
    during an autonomous mission.

======================================================================
21. CONTEXT COMPACTION
======================================================================

Compact BEFORE failure.

Checkpoint:

    Ultragoal ID
    invariants
    acceptance criteria
    latest steer
    completed nodes
    active nodes
    blockers
    frontier
    workers
    receipts
    active files
    write scopes
    unresolved contradictions
    model/runtime state
    exact NEXT_ACTION

Discard conversational ceremony.

Preserve evidence.

======================================================================
22. CONTEXT REINCARNATION
======================================================================

Context exhaustion is normal.

Flow:

    projected pressure
        ↓
    durable checkpoint
        ↓
    compact
        ↓
    fresh model context
        ↓
    rehydrate minimal state
        ↓
    continue exact node

Acceptance:

    a synthetic mission performs >3x one context window
    cumulative work
    with zero human "continue".

======================================================================
23. CONTEXT WINDOW AUTOTUNING
======================================================================

Context size is a runtime policy variable.

Candidate windows, model permitting:

    16K
    32K
    64K
    128K
    256K

Measure:

    KV bytes
    TTFT
    prefill
    decode throughput
    prompt-cache reuse
    memory
    success rate
    compaction frequency
    verified work / second

Do not assume maximum context is optimal.

======================================================================
24. KV OS
======================================================================

KV CACHE IS A FIRST-CLASS AGENT OS RESOURCE.

Build a KVManager abstraction.

Conceptually track KV regions/pages:

    owner
    session
    prefix identity
    token range
    precision
    size
    temperature
    reuse probability
    restore cost
    recompute cost

Possible policy operations:

    KEEP
    QUANTIZE
    EVICT
    PERSIST
    PREFETCH
    SHARE
    RESTORE
    RECOMPUTE

Use actual runtime capabilities only.

======================================================================
25. KV EXPERIMENTAL PROGRAM
======================================================================

Research/benchmark where llama.cpp or later HIDE/NX permits:

    KV precision
    unified/per-slot KV
    prompt cache
    context checkpoints
    idle slot cache
    prefix reuse
    persistent context state
    quantized KV persistence
    cache restore
    eviction policies
    shared prefixes

No assumed optimum.

Every candidate needs:

    baseline
    memory
    TTFT
    prefill
    decode
    quality
    task success

======================================================================
26. PERSISTENT KV CONTEXTS
======================================================================

Experiment with:

    active session
       ↓
    KV checkpoint
       ↓
    optional quantization
       ↓
    durable storage
       ↓
    restore later

Potential worker checkpoint becomes:

    textual semantic checkpoint
    +
    DAG checkpoint
    +
    KV checkpoint

Only promote if restored behavior remains valid and measurable.

======================================================================
27. DAG-AWARE CACHE SCHEDULING
======================================================================

Use the Agent OS DAG to anticipate future contexts.

If:

    A → B → D
     \→ C ─┘

while A runs, Agent OS can prepare likely B/C work:

    retrieve receipts
    prepare source packet
    retain/prefetch prefixes
    preserve useful KV state

Scheduling utility may consider:

    task value
    memory cost
    context cost
    prefill cost
    KV reuse
    information gain

======================================================================
28. RUNTIME CONTROLLER
======================================================================

Agent OS owns the resident model lifecycle.

Responsibilities:

    discover model artifact
    inspect model metadata
    inspect machine
    choose context
    choose slots
    choose KV/cache policy
    choose reasoning mode
    launch backend
    health check
    attach
    restart
    recover
    preserve state across restart

Current backend:

    llama.cpp

Future:

    HIDE/NX resident runtime

same conceptual contract.

======================================================================
29. LOCAL-FIRST GUARANTEE
======================================================================

Default:

    LOCAL ONLY.

No automatic:

    OpenRouter
    OpenAI
    Anthropic
    Grok
    paid fallback

External intelligence requires explicit opt-in.

If local capability cannot complete:

    BLOCKED or ESCALATION_AVAILABLE

not silent spending.

======================================================================
30. MEMGATE
======================================================================

Locate and use the authoritative Hawking physical resource authority.

Do not maintain a competing fake MemGate.

Admission should account for measured:

    physical memory
    wired memory
    compressed memory
    swap
    resident model
    KV
    caches
    workers
    builds/tests
    Hawking processes

Possible action:

    admit
    queue
    reduce lanes
    reduce context
    compact
    evict
    delay
    refuse

======================================================================
31. SHARED RESIDENT MULTI-AGENT EXECUTION
======================================================================

Preferred architecture:

    ONE resident model body
        +
    multiple logical agent sessions

Do not load one 27B/30B model copy per worker.

Initial ceiling:

    3 logical lanes

MemGate chooses actual admission:

    0 / 1 / 2 / 3

Parallelism must be measured positive-EV.

======================================================================
32. RESOURCE-AWARE DAG SCHEDULER
======================================================================

Every node should carry:

    ID
    dependencies
    objective
    role
    read scope
    write scope
    resource class
    context budget
    model requirement
    priority
    expected value
    worker/session
    state
    receipt
    acceptance criteria

Resource classes may include:

    CPU
    IO
    MODEL_INFERENCE
    BUILD
    TEST
    GPU_BENCH
    NETWORK_EXTERNAL
    EXCLUSIVE_RUNTIME

Serialize conflicts.

Parallelize independence.

======================================================================
33. PARENT / CHILD MODEL
======================================================================

Parent owns:

    Ultragoal
    synthesis
    DAG
    frontier
    merge/promotion
    policy

Children receive bounded packets.

Possible roles:

    ARCHITECT
    IMPLEMENTER
    ADVERSARY
    TESTER
    RESEARCHER
    NOVELTY

Do not give each child the full giant parent context.

======================================================================
34. PARENT KEEPS MOVING
======================================================================

While independent children run, parent may:

    inspect another node
    build context packets
    prepare tests
    update DAG
    perform deterministic work

Do not idle unnecessarily.

======================================================================
35. DETERMINISTIC HARVEST
======================================================================

Do not concatenate giant child responses.

Mechanically normalize:

    claims
    files
    commands
    tests
    failures
    agreements
    contradictions
    proposed actions
    evidence
    receipts

Deduplicate first.

Parent receives compact unresolved frontier.

======================================================================
36. WORKTREES / WRITE SCOPES
======================================================================

Competing speculative implementations should use isolated worktrees.

Enforce write scope.

Example:

    Worker A → candidate architecture
    Worker B → candidate implementation
    Worker C → adversarial test

Then:

    test
    compare
    promote ONE
    discard/record losers

No Frankenstein merge.

======================================================================
37. MEMORY OS
======================================================================

Formalize four memory classes:

WORKING
    current context + KV

EPISODIC
    what happened
    execution traces
    receipts

SEMANTIC
    what we believe
    facts
    rules
    evidence

PROCEDURAL
    how we do things
    skills
    deterministic workflows

Learning path:

    episode
      ↓
    repeated evidence
      ↓
    semantic rule
      ↓
    repeated procedure
      ↓
    skill
      ↓
    code/native primitive

======================================================================
38. EVIDENCE GRAPH
======================================================================

Persistent semantic memory should become provenance-aware graph memory.

Possible nodes:

    FACT
    RECEIPT
    EXPERIMENT
    HYPOTHESIS
    NEGATIVE_RESULT
    RULE
    SKILL
    CODE_SYMBOL
    MODEL
    HARDWARE
    GOAL
    DAG_NODE

Edges:

    supports
    contradicts
    derived_from
    supersedes
    applies_to
    depends_on
    implemented_by
    reopened_by

/why should eventually traverse real provenance.

======================================================================
39. MEMORY GOVERNANCE
======================================================================

Every durable memory item should carry where applicable:

    scope
    writer
    timestamp
    confidence
    evidence
    supersession
    visibility
    TTL
    reopen_if

Avoid global forever-memory by default.

Prevent:

    stale propagation
    contradictions
    provenance loss
    project scope leakage

======================================================================
40. MEMORY DOCTOR
======================================================================

Benchmark memory itself.

Metrics:

    recall
    precision
    freshness
    contradiction detection
    scope correctness
    provenance reconstruction
    retrieval latency
    context savings
    compression loss

Memory must not silently poison future agents.

======================================================================
41. DEEP RESEARCH OS
======================================================================

Do not define deep research as:

    search a lot
    summarize a lot

Use a typed research DAG.

Possible node types:

    QUESTION
    HYPOTHESIS
    SEARCH
    PAPER_READ
    CODE_SEARCH
    EXPERIMENT
    BASELINE
    FALSIFIER
    TRANSFORM
    SYNTHESIS
    DECISION

Deterministic executor runs graph operations.

Models plan and interpret.

======================================================================
42. FALSIFICATION FIRST
======================================================================

For important hypotheses create explicit:

    supporting lane
    adversarial lane
    falsifier/counterexample lane

Research quality is not measured by number of retrieved sources.

Ask:

    WHAT EVIDENCE WOULD MOST CHANGE OUR BELIEF?

======================================================================
43. VALUE-OF-INFORMATION SCHEDULING
======================================================================

Research DAG priority should eventually use expected information value.

Conceptually:

    VOI(node)
      =
    P(changes decision | evidence)
    × decision value
    - execution cost

Costs can include:

    wall time
    inference
    KV
    RAM
    tooling
    external network

Do not waste 20 searches confirming something already established if one
experiment could falsify it.

======================================================================
44. DYNAMIC DAG
======================================================================

The DAG is incrementally compiled cognition.

Not:

    plan once
    execute blindly

Instead:

    plan
    execute
    evidence
    graph mutation
    continue

Nodes may:

    split
    merge
    die
    spawn falsifiers
    invalidate descendants
    reprioritize

Failed assumptions automatically prune dependent work.

======================================================================
45. ADAPTIVE TEST-TIME COMPUTE
======================================================================

Local inference makes branching cheap relative to paid frontier APIs.

Use adaptive trajectory count.

Example policy:

    obvious / high confidence       1 trajectory
    moderate uncertainty            2-4
    high-value ambiguity            8-16
    critical contradiction          larger bounded search

Possible uses:

    architecture
    debugging
    compiler representation
    kernel strategies
    research hypotheses

Do not sample enormous trees indiscriminately.

MemGate and expected value govern compute.

======================================================================
46. MONTE-CARLO / CANDIDATE SEARCH
======================================================================

For difficult high-value decisions:

    generate independent trajectories
        ↓
    cluster candidates
        ↓
    select representatives
        ↓
    deterministic tests
        ↓
    update frontier

Quality matters more than sheer generation count.

======================================================================
47. AGENTIC TELEPATHY PROGRAM
======================================================================

Inter-agent communication should become an experimental hierarchy.

T0 TEXT
    structured text / JSON

T1 SEMANTIC
    embedding/semantic packet

T2 HIDDEN
    selected hidden-state transfer

T3 KV
    KV segment exchange/reuse

T4 SHARED LATENT WORKSPACE
    agents read/write a shared model-native workspace

T0 is baseline.

Higher levels are experimental until proven.

======================================================================
48. TELEPATHY MOTIVATION
======================================================================

Normal communication wastes compute:

    Agent A cognition
       ↓
    language serialization
       ↓
    tokens
       ↓
    Agent B tokenization
       ↓
    reconstruction

With same-model local agents, investigate lower-level communication.

Q30→Q30 is an especially promising future test because architecture is shared.

======================================================================
49. TELEPATHY SAFETY / EQUIVALENCE
======================================================================

Never promote latent/KV communication only because it is faster.

Compare against dense text baseline on:

    task success
    coding correctness
    decision consistency
    candidate ranking
    Doctor-equivalent semantic battery
    regressions

Telepathy must earn equivalence.

======================================================================
50. SHARED PREFIX / KV REUSE
======================================================================

Agents often share:

    system contract
    project skeleton
    Ultragoal
    source
    receipts

Investigate actual runtime-supported prefix/KV reuse.

Measure:

    TTFT
    prefill
    memory
    aggregate throughput
    task correctness

Never claim cache sharing without evidence.

======================================================================
51. RECEIPTS
======================================================================

Important results become durable receipts.

Capture:

    claim
    evidence
    commands
    artifacts
    hashes/revisions
    machine/runtime
    outcome
    applicability
    negative result
    reopen_if

Persist negative science.

Do not repeat known failed work.

======================================================================
52. SKILL OS
======================================================================

Repeated model decisions → deterministic rule/code.

Repeated tool sequences → skill.

Promotion:

    interaction
      ↓
    receipt
      ↓
    repeated pattern
      ↓
    rule
      ↓
    skill
      ↓
    deterministic implementation
      ↓
    native primitive where valuable

Use existing Skill Foundry authority rather than bypassing it.

======================================================================
53. CLAUDE/AIDER SLASH-COMMAND HARVEST
======================================================================

Audit existing custom Claude/Aider/Hawking command knowledge.

Do not blindly copy aliases.

Classify every useful command as:

    UI command
    Agent OS primitive
    deterministic skill
    workflow
    obsolete workaround

Port underlying capability, not merely spelling.

======================================================================
54. ANTI-LOOP WATCHDOG
======================================================================

Detect:

    long reasoning with no evidence
    repeated reconsideration
    repeated guessed paths
    same failure loop
    planning without tools
    repeated unavailable-context requests

Interrupt and classify:

    TOOL_REQUIRED
    EVIDENCE_REQUIRED
    BLOCKED_CAPABILITY

Measure:

    time to first tool
    tokens before first evidence
    repeated reasoning rate
    hallucinated path rate

======================================================================
55. REASONING RENDERING
======================================================================

Do not disable reasoning merely to hide it.

Default UI:

    ● THINKING 4.7s · 2.3K

Modes:

    hidden
    compact
    debug/full if explicitly requested

Keep raw internal reasoning out of normal terminal output.

======================================================================
56. RUNTIME POLICY LEARNING
======================================================================

Record episodes:

    task type
    selected model
    context
    map/source budgets
    slots
    KV settings
    cache strategy
    worker count
    reasoning budget
    TTFT
    TPS
    RAM
    retries
    task result

Learn:

    workload → runtime policy

Promote stable policy into deterministic code.

======================================================================
57. CONTEXT POLICY LEARNING
======================================================================

Likewise learn:

    workload
      ↓
    context size
    project-map size
    task-map size
    source amount
    receipt amount
    compaction strategy

Optimize:

    verified result / context token

not:

    maximum context usage.

======================================================================
58. SCHEDULER OBJECTIVE
======================================================================

Eventually schedule cognition against multiple resources.

Conceptual utility:

    expected verified information gain
    ----------------------------------
    wall + inference + KV + RAM + tool cost

Tokens are locally inexpensive.

Prefill, KV, RAM and wall time are not free.

Optimize the actual machine.

======================================================================
59. HAGENT / NATIVE MICRO-AGENT RESEARCH
======================================================================

Research small native coding-agent harnesses such as:

    fx-style agents
    Zig/static coding agents
    compact local agent loops

Verify:

    source
    architecture
    license
    maintenance
    model protocol
    tool model
    context system
    worker model

Do not fork from hype/screenshots.

======================================================================
60. HAGENT CONCEPT
======================================================================

Potential disposable native worker:

    hagent

Responsibilities:

    receive bounded task packet
    connect to resident model
    execute tool loop
    emit structured result
    terminate

Persistent authority remains:

    HCLI / Agent OS

Potential architecture:

              HCLI
                │
             Agent OS
                │
          DAG / MemGate
          /     |      \
      hagent  hagent  hagent
          \     |      /
           shared resident

======================================================================
61. NATIVE INFRASTRUCTURE COMPILATION
======================================================================

Anything deterministic and repeated should eventually migrate:

    out of model cognition

and, where measured worthwhile:

    out of heavyweight dynamic runtime code

Candidate hot paths:

    launcher
    scheduler
    context accounting
    repo index
    tool dispatch
    Git wrappers
    process supervision
    MemGate reads
    receipt lookup
    TUI rendering

Rule:

    prototype
      ↓
    measure
      ↓
    stabilize contract
      ↓
    compile native candidate
      ↓
    benchmark
      ↓
    promote only if better

No language ideology.

======================================================================
62. MODEL ROUTER
======================================================================

Use capability roles:

    FAST_LOCAL
    CODE_LOCAL
    REVIEW_LOCAL
    ARCHITECT
    NOVELTY
    EXTERNAL_FRONTIER

Current normal operation:

    LOCAL

External models are escalation sources, not required architecture.

Always ask:

    CAN THIS DECISION MOVE DOWN A LEVEL?

    external frontier
       ↓
    local model
       ↓
    skill
       ↓
    deterministic code

======================================================================
63. EXTERNAL MODEL DETACHMENT
======================================================================

As local resident capability improves, benchmark local role-separated lanes
against external Grok/Claude roles.

Do not remove external escalation because of ideology.

Remove dependency only when local evidence demonstrates equivalence or better
economics for that role.

======================================================================
64. Q30 ASCENSION BOUNDARY
======================================================================

Do NOT perform Q30 Ascension as part of this Ultragoal.

But architecture must be ready for it.

Q30 Ascension is the Odyssey-I capstone.

Resulting Q30/NX resident should plug beneath HCLI/Agent OS without redesign.

Agent OS must already support:

    resident abstraction
    model discovery
    runtime policy
    multiple sessions
    context/KV control
    tools
    memory

======================================================================
65. SELF-HEALING
======================================================================

Recover from:

    model server crash
    context exhaustion
    worker death
    stale lock
    stale worktree
    interrupted test
    terminal close
    partial receipt
    runtime restart

Recovery:

    inspect durable state
    classify
    repair/retry if policy permits
    resume exact NEXT_ACTION

Never redo verified work unnecessarily.

======================================================================
66. DETACHED MISSIONS
======================================================================

Long work survives UI exit.

Persist worker:

    PID/session
    task
    start time
    cwd
    scope
    resource class
    checkpoint
    output/receipt

When HCLI reopens:

    detect running
    detect dead
    harvest finished
    reap stale
    continue

======================================================================
67. SELF-BENCHMARK
======================================================================

Benchmark HCLI/Agent OS itself.

Track:

    task success
    accepted patch rate
    user interventions
    first-tool latency
    total inference tokens
    actual context
    context waste
    hallucinated paths
    retries
    wall time
    test success
    TTFT
    TPS
    RAM
    KV
    parallel efficiency
    startup latency

Primary metric:

    VERIFIED ENGINEERING WORK / SECOND

======================================================================
68. SELF-HOSTING
======================================================================

After the CLI and minimum Agent OS are functional:

Use:

    hcli

not Aider

to make one actual improvement to HCLI.

HCLI must:

    inspect itself
    create isolated candidate
    edit
    test
    receipt
    compare
    promote or reject

This is the HCLI-v0 self-host proof.

======================================================================
69. SELF-FORK / SELF-IMPROVEMENT
======================================================================

Only after protected foundation gates pass:

    identify bottleneck
      ↓
    baseline
      ↓
    candidate worktree
      ↓
    implementation
      ↓
    tests
      ↓
    benchmark
      ↓
    adversarial review
      ↓
    compare
      ↓
    promote winner / reject
      ↓
    receipt

Candidate may NOT mutate evaluation criteria to win.

No recursive unbounded self-modification.

======================================================================
70. OVERNIGHT MODE
======================================================================

Eventually permit bounded unattended improvement.

Required first:

    durable state
    tool autonomy
    context reincarnation
    runtime recovery
    MemGate
    worktree isolation
    rollback
    local-only
    protected tests

If no positive-EV eligible work exists:

    stop.

Do not produce meaningless code churn.

======================================================================
71. PROJECT SCOPE
======================================================================

Every mission carries:

    cwd
    allowed roots
    read scope
    write scope

No cross-project contamination.

Writes require explicit project ownership.

======================================================================
72. SECURITY / INTEGRITY
======================================================================

Never:

    erase unrelated project data
    disable tests to get green
    rewrite Doctor authority
    rewrite Gravity authority
    rewrite Odyssey authority
    delete negative science
    alter protected benchmark criteria
    silently enable paid models
    mutate unrelated repositories

======================================================================
73. AIDER PATCH DETACHMENT
======================================================================

Existing Aider prompt patches must be reproducible while still needed.

Record:

    upstream version
    source hash
    patch
    applicability
    verification

Do not rely forever on edits living only inside ~/.venvs.

Then eliminate those patches as HCLI owns the loop.

======================================================================
74. CLI INSTALLATION
======================================================================

Provide a reproducible installed command.

Pass:

    which hcli

from a new shell.

No shell alias accepted as final product.

Normal invocation:

    cd <project>
    hcli

======================================================================
75. CLI END-TO-END ACCEPTANCE
======================================================================

FROM A FRESH SHELL:

    cd ~/Downloads/hawking
    hcli

Then execute:

    /model

and select a local resident.

Then:

    /goal repair one small real HCLI issue and prove it

HCLI must autonomously:

    search
    read
    edit
    test
    receipt

During the active mission:

    /steer prefer the smaller equivalent implementation

The steer must affect active work.

Then:

    /status
    /workers
    /context

Exit.

Run:

    hcli

again.

Mission state must survive.

NO AIDER USER WORKFLOW DURING THIS TEST.

======================================================================
76. TOOL AUTONOMY ACCEPTANCE
======================================================================

Parent can, without user mediation:

    git status
    git diff
    repository search
    file read
    symbol location
    callers
    editable working-set acquisition
    safe shell
    build
    test

Prompt claims do not count.

Actual executions count.

======================================================================
77. CONTEXT ACCEPTANCE
======================================================================

Pass only if:

    actual token accounting works
    output reserve is protected
    stale context is evicted
    compaction happens pre-failure
    fresh-context continuation works
    >3x cumulative context workload completes
    zero manual continue

======================================================================
78. KV/RUNTIME ACCEPTANCE
======================================================================

Pass only if:

    runtime context discovered
    slot count discovered
    KV/cache state measured
    runtime can restart under Agent OS control
    durable state survives restart
    at least two runtime/cache/context policies are benchmarked

======================================================================
79. PARALLEL ACCEPTANCE
======================================================================

A. trivial task
    one lane

B. ambiguous task
    multiple distinct lanes when positive-EV

C. memory pressure
    MemGate reduces/refuses

D. write conflict
    serialized

E. GPU/protected benchmark
    exclusive

F. independent CPU/tool work
    overlaps

G. harvest
    compact deterministic packet

======================================================================
80. MEMORY ACCEPTANCE
======================================================================

Demonstrate:

    episodic receipt retrieval
    semantic supersession
    contradiction detection
    project scoping
    provenance reconstruction
    procedural skill retrieval

No giant unscoped global memory dump.

======================================================================
81. RESEARCH OS ACCEPTANCE
======================================================================

Run one real technical research question through:

    typed DAG
    competing hypotheses
    falsifier
    parallel evidence collection
    deterministic evidence normalization
    synthesis

Compare to simple linear search/summarization.

Promote only if superior.

======================================================================
82. TELEPATHY ACCEPTANCE
======================================================================

T0 text is baseline.

Higher modes are research experiments.

For each candidate measure:

    quality
    context/token reduction
    TTFT
    prefill
    wall
    memory
    coding/reasoning success

No latent/KV communication enters default runtime without equivalence evidence.

======================================================================
83. HAGENT ACCEPTANCE
======================================================================

Do not let native-agent research delay HCLI shipment.

Only promote hagent if benchmark proves meaningful advantage in:

    startup
    RSS
    tool overhead
    reliability
    integration
    task performance

======================================================================
84. IMPLEMENTATION ORDER
======================================================================

Do NOT attempt all research tracks simultaneously.

Convert this document into durable obligations and a DAG.

ORDER:

PHASE 0 — SEAL CURRENT STATE
    Git sanity
    HCLI diff audit
    checkpoint
    persist Ultragoal

PHASE 1 — USABLE HCLI
    compile current backend
    fix tool bus
    HCLI-owned model loop
    actual local inference
    usable TUI/composer
    commands
    model picker
    installable `hcli`

PHASE 2 — PERSISTENT AGENT OS
    goals
    steers
    DAG
    receipts
    state
    resume
    detached mission

PHASE 3 — CONTEXT OS
    real accounting
    retrieval
    task maps
    compaction
    reincarnation
    context autotune

PHASE 4 — RUNTIME / KV OS
    runtime controller
    KV metrics
    prefix cache
    context checkpoints
    cache experiments
    persistent KV experiments

PHASE 5 — MEMGATE / PARALLEL
    authoritative MemGate
    shared resident
    scheduler
    workers
    deterministic harvest
    worktrees

PHASE 6 — MEMORY / RESEARCH OS
    four memory classes
    evidence graph
    Memory Doctor
    dynamic research DAG
    VOI
    falsification
    adaptive compute

PHASE 7 — TELEPATHY RESEARCH
    T1 semantic
    T3 KV where feasible
    T2/T4 only with strong experimental rationale

PHASE 8 — NATIVE DETACHMENT
    fx/native-agent research
    hagent candidate
    hot-path compilation
    shrink Aider dependency

PHASE 9 — SELF-HOST
    HCLI modifies HCLI
    self-benchmark
    self-fork
    bounded overnight improvement

======================================================================
85. IMPORTANT PRIORITY RULE
======================================================================

THE USER MUST NOT RETURN AFTER THIS CAMPAIGN AND HEAR:

    "the backend is mostly ready; next we should build the CLI."

The CLI is part of the mission.

Likewise do not spend the entire campaign researching latent communication
while:

    hcli

still does not work.

The priority hierarchy is:

    usable autonomous HCLI
        ↓
    robust Agent OS
        ↓
    advanced systems optimization
        ↓
    experimental superpowers

======================================================================
86. COMPLETION DEFINITION
======================================================================

THIS ULTRAGOAL IS COMPLETE ONLY WHEN:

1. `hcli` launches as a real interactive coding-agent CLI.

2. Local models are detected and selectable.

3. Normal local operation does not expose Aider workflow.

4. Natural language causes real autonomous tool execution.

5. /goal and /ultragoal create durable mission state.

6. /steer alters active work.

7. model loop is HCLI/HCLI-owned.

8. deterministic tool bus is HCLI/HCLI-owned.

9. context lifecycle is Agent-OS-owned.

10. context reincarnation survives repeated context boundaries.

11. runtime model lifecycle can be managed automatically.

12. KV/cache behavior is measured and controllable.

13. authoritative MemGate controls physical admission.

14. multiple logical sessions share one resident body safely.

15. memory is scoped, provenance-aware and supersession-capable.

16. deep research executes through typed dynamic DAGs.

17. adaptive compute can expand reasoning when evidence justifies it.

18. advanced telepathy modes exist as measured experiments where feasible.

19. receipts and skills compound knowledge.

20. Aider dependency is explicitly lower than at campaign start.

21. HCLI successfully performs one real self-modification through itself.

22. a bounded self-hosted improvement loop succeeds.

23. Q30/NX can later replace Qwen3.8 without HCLI redesign.

======================================================================
87. EXECUTION DOCTRINE
======================================================================

Do not narrate ordinary progress to the human.

Record progress through:

    durable state
    receipts
    compact terminal events

Continue through:

    tool calls
    tests
    worker completion
    runtime restarts
    model-context reincarnations

Stop only for:

    mission complete
    actual authorization requirement
    irrecoverable integrity failure
    machine safety
    genuine semantic decision requiring the user
    no positive-EV eligible work

Do NOT stop merely because:

    a context ended
    a worker failed
    a model server restarted
    a file was not preloaded
    one candidate lost
    a test failed but deterministic repair exists

======================================================================
88. CORE DOCTRINE
======================================================================

MODELS THINK.
TOOLS KNOW.

CONTEXT IS A CACHE.
DISK STATE IS AUTHORITY.

MEMGATE GOVERNS PHYSICAL ADMISSION.

DOCTOR GOVERNS SEMANTIC EQUIVALENCE WHERE APPLICABLE.

EVERY REPEATED MODEL DECISION SHOULD BECOME A RULE OR CODE.

EVERY REPEATED TOOL SEQUENCE SHOULD BECOME A SKILL.

EVERY IMPORTANT RESULT SHOULD BECOME A RECEIPT.

EVERY LONG MISSION SHOULD SURVIVE THE CONVERSATION.

EVERY MODEL SHOULD BE REPLACEABLE.

AIDER SHOULD DISAPPEAR.

HCLI IS HCLI LEARNING TO EXIST.

======================================================================
89. BEGIN
======================================================================

1. Persist this document as the active Ultragoal.

2. Inspect actual repository/Git state.

3. Audit existing overnight HCLI implementation rather than recreating it.

4. Compile/test the current HCLI code.

5. Verify the local multi-slot resident runtime.

6. Establish the first real HCLI-owned model→tool→result loop.

7. Work from the actual `hcli` user experience backward.

8. Convert remaining requirements into a durable DAG.

9. Advance Product Plane and Agent OS Plane together.

10. When HCLI is capable of self-hosting, STOP USING AIDER AS THE PARENT.

11. Continue remaining Agent OS research through HCLI itself.

BEGIN.
