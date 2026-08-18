# GENESIS CONTINUITY DIRECTIVE — BUILD THE MIGRATING ORGANISM BEFORE RELAUNCH

Codex:

Do NOT spend this pass optimizing Qwen3.8 BPW or writing its final Metal kernel.

That belongs inside the Genesis sandbox.

Your job in this pass is to build the **system in which Genesis can continuously optimize itself while simultaneously building Hawking.**

The architecture must survive model-generation replacement.

The worker belongs to Hawking.

The task belongs to Hawking.

The context belongs to Hawking.

The model generation is replaceable.

---

# 1. THE CORE ARCHITECTURE

There is:

```text
ONE CURRENT GENESIS GENERATION
ONE RESIDENT IMMUTABLE MODEL BODY
MANY LOGICAL GENESIS AGENTS / WORKERS
ZERO REQUIREMENT THAT A WORKER DIE WHEN GENESIS IS REPLACED
```

Example:

```text
GENESIS G7
│
├── Worker: Gravity / Doctor
├── Worker: Kernel / Metal
├── Worker: Context / KV
├── Worker: AgentOS
├── Worker: HCLI
├── Worker: Memory / World State
└── Worker: Adversarial review
```

These are logical agents.

They are NOT separate Genesis lineages.

They are NOT children merely because they have separate tasks.

They should share the same resident body wherever architecture permits.

Each owns isolated:

```text
KV
context
task state
worktree
receipts
hypotheses
tool state
```

---

# 2. A CHILD IS ONLY A SUCCESSOR ARTIFACT

A CHILD means:

```text
a materially modified Genesis model/runtime/representation candidate
```

Examples:

```text
lower-BPW model
new attention representation
new generator/residual representation
new Gravity artifact
new representation + kernel pair
materially changed execution genome
```

Workers BUILD the child.

Workers TEST the child.

Workers may IMPROVE the child.

The child does not become Genesis until protected promotion.

---

# 3. GENERATION REPLACEMENT MUST NOT DESTROY WORK

When:

```text
GENESIS G7
↓
candidate G8 passes
↓
G8 becomes Genesis
```

every active worker must execute:

```text
CHECKPOINT
↓
record task state
record worktree
record hypotheses
record receipts
record negative science
record dependencies on G7
record NEXT_ACTION
↓
REBIND to G8
↓
invalidate stale G7 assumptions
↓
refresh model/runtime/genome context
↓
resume task
```

Do NOT restart the task from zero.

Do NOT destroy partial research.

Do NOT leave a worker reasoning against the old artifact.

---

# 4. WORKER STATE MUST BE MODEL-INDEPENDENT WHERE POSSIBLE

Task state belongs to AgentOS/Hawking.

Not Qwen3.8's KV cache.

Persist enough state that a worker can be reconstructed with a new model generation.

Separate:

```text
DURABLE TASK STATE
    goal
    subgoal
    repo/worktree
    hypotheses
    findings
    receipts
    negative science
    pending experiments
    tool results
    NEXT_ACTION

EPHEMERAL MODEL STATE
    KV
    current conversational tokens
    transient reasoning
```

On generation replacement:

recompile context from durable state.

Do not require preservation of stale KV to preserve the task.

---

# 5. CONTEXT COMPILER IS NOW CRITICAL INFRASTRUCTURE

Build/reuse a Context Compiler that can produce a compact current context for any worker from:

```text
Genesis System Directive
+
current generation identity
+
task contract
+
current World State
+
relevant receipts
+
relevant negative science
+
worker checkpoint
+
recent research-bus messages
```

This allows:

```text
G7 worker
→ checkpoint
→ G8 promoted
→ context recompiled
→ G8 worker resumes
```

without carrying an enormous transcript.

---

# 6. AGENTOS SHOULD BEGIN NOW, NOT AFTER GENESIS IS "DONE"

Genesis optimization remains Priority Zero.

But once the infrastructure exists, run parallel logical workers for:

```text
A. Gravity / Doctor
B. kernel / execution genome
C. Context/KV
D. AgentOS
E. HCLI
```

Priority:

```text
A + B receive dominant optimization resources until >=100 TPS.
```

C/D/E continue where they do not materially slow the critical path.

At 100 TPS, widen aggressively.

At higher TPS and lower BPW, widen again.

Resource allocation should be dynamic.

---

# 7. 100 TPS IS THE PARALLELISM EXPANSION GATE

Before 100 TPS:

```text
majority resources:
    Gravity
    kernel
    Genesis infrastructure
```

At >=100 valid TPS:

increase concurrent logical work on:

```text
Context/KV OS
AgentOS task graph
Memory OS
HCLI
Model Auto
Cognitive Scheduler
Research Market
World State
Self Model
Skill Foundry
```

Do NOT stop Gravity/kernel optimization.

The organism becomes:

```text
lane 1 — next BPW frontier
lane 2 — next kernel/runtime frontier
lane 3 — Context/KV
lane 4 — AgentOS
lane 5 — HCLI
lane 6 — Memory/World State
...
```

subject to actual RAM/CPU/GPU pressure.

---

# 8. KERNEL AND GRAVITY REMAIN INDEPENDENT BUT COUPLED

Correct doctrine:

```text
Gravity improves representation.
Kernel improves execution of current representation.
```

A kernel improvement can raise TPS without BPW changing.

A Gravity improvement can lower traffic without kernel changing.

Both may proceed concurrently.

BUT:

when Gravity promotes a new representation:

```text
kernel worker must reprofile
```

because the optimal kernel may change.

When kernel changes reconstruction economics:

```text
Gravity worker may reopen representations previously considered expensive.
```

Their work is separate.

Their science communicates.

---

# 9. AGENTOS'S JOB IS TO HIDE TOOL LATENCY

Raw model TPS is not total agent responsiveness.

AgentOS must increasingly overlap:

```text
model thinking
tool execution
builds
tests
file reads
Git
profiling
packing
candidate evaluation
```

Examples:

while Worker A waits on cargo:

```text
Worker B reasons
Worker C profiles
Worker D searches negative science
```

while a protected GPU measurement runs:

```text
CPU-only workers continue
```

while a Gravity candidate packs:

```text
other workers continue unless memory forecast requires preemption
```

Target:

> **minimize idle cognitive and machine time, not merely maximize TPS.**

---

# 10. TOOL-WAIT BACKFILL

Implement/reuse the Context/KV/AgentOS concept already present in Hawking:

```text
tool call begins
↓
worker becomes WAITING
↓
scheduler allocates resources to another ready task
↓
tool returns
↓
worker becomes READY
↓
scheduler resumes it
```

A user should not experience:

```text
"waiting for currency scope"
```

as the entire organism being idle.

One agent may wait.

Hawking continues.

---

# 11. GENERATION-AWARE RESEARCH BUS

All workers communicate through structured state.

Messages:

```text
MEASURED_FACT
HYPOTHESIS
NEGATIVE_SCIENCE
PATCH
COUNTEREXAMPLE
PROFILE_DELTA
REPRESENTATION_CHANGE
KERNEL_CHANGE
NEXT_BOTTLENECK
```

Every message carries:

```text
generation
artifact SHA
runtime SHA
repo HEAD
epistemic state
facet
receipt
```

When G8 replaces G7:

messages tied specifically to G7 become:

```text
STALE_PENDING_REVALIDATION
```

unless structurally transferable.

Do not silently apply G7 measurements to G8.

---

# 12. WORKER MIGRATION ON PROMOTION

Build an explicit promotion event:

```text
GENESIS_PROMOTED(
    old_generation,
    new_generation,
    artifact,
    runtime,
    BPW,
    TOKEN_NS
)
```

All workers subscribe.

On receipt:

```text
1. checkpoint
2. stop new G7-dependent experiments
3. classify current work:
       transferable
       needs rebase
       invalidated
4. attach/rebind to G8
5. compile fresh context
6. resume
```

This should be automatic.

---

# 13. SACRIFICIAL PARENT

Once G8 is protected and workers have rebound:

```text
unload G7
terminate G7
reclaim RAM
```

Keep only:

```text
LAST_KNOWN_GOOD artifact/reconstruction state
```

on disk if required for rollback.

Do not retain superseded active bodies.

---

# 14. IF NEW GENERATION IS SMALLER, EXPAND THE ORGANISM

After every promotion measure:

```text
resident model RAM
marginal worker KV RAM
free RAM
candidate reserve
```

If BPW falls and headroom rises:

increase logical worker capacity.

Example conceptually:

```text
G7:
    model 12 GB
    4 workers

G8:
    model 8 GB
    8 workers

G9:
    model 5 GB
    12 workers
```

Do not hardcode these counts.

Measure and derive.

This is part of the compounding loop.

---

# 15. THE COMPOUNDING LOOP

Engineer specifically for:

```text
LOWER BPW
↓
LESS MODEL RAM
↓
LESS TRAFFIC
↓
HIGHER TPS
↓
MORE FREE RAM
↓
MORE LOGICAL WORKERS
↓
MORE EXPERIMENTS/HOUR
↓
BETTER GRAVITY + KERNEL
↓
LOWER BPW / TOKEN_NS
```

Also:

```text
BETTER AGENTOS
↓
LESS TOOL-WAIT IDLE TIME
↓
MORE VERIFIED WORK/HOUR
↓
FASTER GENESIS IMPROVEMENT
↓
BETTER AGENTOS
```

This feedback structure is intentional.

Measure whether it actually compounds.

---

# 16. TRACK GENERATIONAL ACCELERATION

For each promotion record:

```text
GENERATION
BPW
TOKEN_NS
TPS
resident RAM
marginal worker RAM
workers supported
experiments/hour
verified wins/hour
time since previous promotion
```

We want to discover whether:

```text
T(G0→G1)
>
T(G1→G2)
>
T(G2→G3)
```

If generation time is shrinking while capability survives:

Hawking is empirically accelerating.

If not:

find the bottleneck in the evolutionary process itself.

---

# 17. GENESIS DOES NOT NEED TO WAIT TO BUILD HAWKING

Once >=100 TPS is achieved:

Genesis becomes a major implementation engine for:

```text
AgentOS
HCLI
Context/KV OS
Memory OS
World State
Model Auto
Cognitive Scheduler
Research Market
Self-Evolution Lab
```

Claude/Codex increasingly become:

```text
external architect
protected reviewer
constitutional authority
exception handler
```

rather than routine implementation labor.

---

# 18. CODEx'S CURRENT JOB

Do NOT manually do Genesis's future optimization.

Build the machinery Genesis needs to do it.

Before relaunch, make the following as real as practical:

```text
persistent Genesis system directive
resident model body
logical worker/session abstraction
durable task checkpoint
Context Compiler/rebind
generation promotion event
worker migration
research bus
resource-aware scheduler
preemption/checkpoint
protected test slot
Genesis-only daemon targeting
correct GPU-profile admission
dynamic worker capacity
status/peek that reflects reality
```

Reuse existing Hawking AgentOS/HCLI foundations.

Do not create duplicate subsystems.

---

# 19. RELAUNCH CONDITION

Relaunch Genesis when the organism can demonstrate:

```text
1 resident body
>=2 logical workers
separate tasks
shared verified research
durable checkpoint
generation-change simulation
workers successfully rebind
parent remains safe
candidate test slot available
daemon generates Genesis work
GPU work receives correct profile
```

A real improved child is not required to test migration.

Simulate G7→G8 with compatible artifacts if necessary.

Prove the machinery first.

---

# 20. END STATE

The intended Hawking shape is eventually:

```text
                HAWKING
                   │
           CURRENT GENESIS
                   │
      ┌────────────┼────────────┐
      │            │            │
   Gravity       Kernel      AgentOS
   Doctor        Runtime       HCLI
      │            │         Context/KV
      └──────┬─────┘            │
             │                  │
       SUCCESSOR CHILD          │
             │                  │
      PROTECTED VERIFY          │
             │                  │
          PROMOTION─────────────┘
             │
      ALL WORKERS REBIND
             │
       OLD PARENT DIES
             │
          CONTINUE
```

# TASKS SURVIVE MODELS.

# WORKERS SURVIVE GENERATIONS.

# SCIENCE SURVIVES EVERYTHING.

# LOWER BPW CREATES MORE PARALLELISM.

# HIGHER TPS CREATES MORE COGNITION.

# BETTER AGENTOS HIDES TOOL LATENCY.

# GENESIS BUILDS BETTER GENESIS.

# GENESIS BUILDS HAWKING.
