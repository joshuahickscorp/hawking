# HAWKING ASCENSION: FUNCTIONAL GRAVITY CONTINUATION
## Null-corrected science, end-to-end functional students, GLM decision, decoder completion, cross-model transfer, and Hawking core closure

### Relationship to the existing Ascension plan

This document is the binding continuation and scientific correction to:

```text
/Users/scammermike/Downloads/hawking/HAWKING_ASCENSION_GRAVITY_GAUNTLET_RUNTIME_CLOSURE.md
```

Read both documents.

This continuation supersedes any conflicting statement about:

```text
raw activation cosine
GLM weight-space recovery
the active GLM representation
the next scientific lever
functional-student priority
```

All non-conflicting Ascension requirements remain active, especially:

```text
96 GiB unified-memory policy
approximately 600 GB free-storage policy
one heavy lane plus bounded parallel lanes
`.gravity` container closure
complete tensor coverage
packer closure
decoder and direct runtime closure
complete-token and prefill evidence
FLOP and joule accounting
cross-model partial-first gauntlet
HIDE-facing contracts
HAWKING_CORE_ASCENDED endpoint
```

Live repository state, sealed artifacts, immutable source identity, tests, and current processes override this handoff.

---

# 0. WHY THE PLAN CHANGED

The GLM campaign produced two decisive corrections.

## 0.1 Raw cosine was a broken promotion metric

Reported null behavior:

```text
block_output constant-mean raw cosine:
    approximately 0.898

attention_output constant-mean raw cosine:
    approximately 0.909

post_moe constant-mean raw cosine:
    approximately 0.835
```

The previously reported weight-space scores were below these nulls.

Therefore:

```text
raw activation cosine cannot promote a candidate
raw normalized route-weight correlation cannot promote a candidate
all earlier GLM verdicts depending on those values must be reissued
```

Reported null-corrected expert-path results:

```text
0.3306 BPW:
    centered score approximately 0.006

0.4990 BPW:
    approximately 0.020

0.7531 BPW:
    approximately 0.022

0.8931 BPW:
    approximately 0.067

2.0169 BPW:
    approximately 0.317
```

None beats the required functional floor.

The negative weight-space result is stronger than previously reported.

## 0.2 A functional escape exists

Reported disjoint held-out results against a null near `0.831`:

```text
functional student h1024:
    local exact rate approximately 0.0104 BPW
    centered score approximately 0.724
    beats null

full linear map:
    approximately 0.0623 BPW
    centered score approximately 0.733
    beats null

linear rank 64:
    approximately 0.0013 BPW
    centered score approximately 0.374
    does not pass the frozen gate

weight-space control:
    approximately 0.7531 BPW
    centered score approximately 0.000
```

Controls reported:

```text
shuffled inputs:
    below null

identity:
    approximately 0.078

nonlinear random features:
    only slightly below the full linear map

student width:
    saturates around 1024 in the tested data
```

The honest interpretation is:

> A cheap function of the hidden state predicts the tested GLM MoE output far better than representations of the original expert weights.

This is not yet:

```text
a block result
a next-layer propagation result
a cross-layer result
a full-model rate
a capability result
a runtime result
```

It is the primary open direction.

---

# 1. GLOBAL METRIC RESET

Create:

```text
HAWKING_NULL_CORRECTED_METRIC_CONTRACT.md
HAWKING_NULL_CORRECTED_METRIC_CONTRACT.json
```

This contract applies to every parent and every representation.

## 1.1 Frozen nulls

Every target stage requires nulls fitted only on the fit split:

```text
training-mean predictor
training-affine constant predictor when distinct
identity or residual passthrough
shuffled-input predictor
best permitted trivial structural baseline
```

Do not calculate a null from held-out target statistics.

## 1.2 Primary metrics

For target `y`, prediction `ŷ`, and fit-split mean `μ_fit`, report:

```text
raw cosine:
    diagnostic only

centered cosine:
    cosine(y - μ_fit, ŷ - μ_fit)

null-relative skill:
    1 - SSE(candidate) / SSE(mean-null)

relative L2
normalized RMSE
per-token and per-feature distributions
```

Promotion requires:

```text
positive null-relative skill with positive confidence lower bound
centered-cosine gate
tail gate
domain gate
replication without refit
```

Raw cosine never overrides a failed null-relative gate.

## 1.3 Controls

Every new student or repair must run:

```text
mean-null
shuffled input
identity
full affine/linear control
representation-family ablation
seed replication
new-document replication
```

## 1.4 Reissue prior evidence

For every current GLM pilot artifact:

```text
recompute null-corrected metrics from sealed predictions
invalidate only claims depending on the broken metric
preserve exact payload and accounting evidence
write superseding receipts
```

Create:

```text
GLM52_METRIC_CORRECTION_LEDGER.jsonl
GLM52_CORRECTED_SCIENTIFIC_LAW.json
GLM52_CORRECTED_SCIENTIFIC_LAW.md
```

No hidden rewriting of old evidence.

---

# 2. CURRENT GLM SCIENTIFIC VERDICT

Until new evidence overturns it:

```text
weight-space routed-expert representations:
    CLOSED_NEGATIVE on tested windows

shared expert-weight basis:
    CLOSED_NEGATIVE on tested windows

per-tensor low-rank weight blueprint:
    CLOSED_NEGATIVE and cross-parent dead-lever confirmation

hybrid weight-space family:
    CLOSED_NEGATIVE on tested windows

functional hidden-state-to-MoE-output mapping:
    OPEN_POSITIVE at one layer and one stage
```

Do not restart a full GLM weight-space stream.

Do not spend the full source on:

```text
PQ
low-rank expert weights
shared expert subspace
hybrid weight blueprint
activation-weighted SVD
```

unless a materially new mechanism and null-corrected causal reason reopens one.

The immediate task is to determine whether the functional escape survives:

```text
block integration
next-layer propagation
cross-layer transfer
cross-domain replication
whole-model physical accounting
direct runtime
```

---

# 3. FUNCTIONAL STUDENT CONTRACT

Create a versioned research representation:

```text
glm52.functional.moe.v1
```

Its purpose is to replace the teacher MoE function, not reconstruct teacher expert weights.

## 3.1 Inputs and outputs

At minimum bind:

```text
input hidden state
layer identity
optional architecture state available at inference
predicted shared-plus-routed MoE output
```

Teacher-only fields are forbidden at inference.

## 3.2 Physical components

Bill:

```text
feature-map seed
PRNG/transform algorithm ID and version
feature-map parameters actually stored
readout weights
bias
normalization
selectors or gates
Doctor state
headers
alignment
runtime tables
per-layer metadata
```

## 3.3 Runtime state

Report separately:

```text
artifact bytes
expanded resident bytes
active bytes/token
generated temporary state
operations/token
initialization time
```

A tiny seed with a large expanded runtime matrix is not zero-cost.

The complete Gravity vector counts both artifact and execution consequences.

## 3.4 Determinism

Freeze:

```text
seed
generator algorithm
generator version
dtype
rounding
feature ordering
fit solver
regularization
```

A different implementation must reproduce the same function within tolerance.

---

# 4. FUNCTIONAL STUDENT GAUNTLET

Do not launch a broad grid.

Run the smallest experiment set that distinguishes the real mechanisms.

## FS0, reproduce the positive row

Reproduce the reported layer and split with:

```text
fresh process
fresh seed set
sealed fit/held-out membership
exact payload reconstruction
null-corrected metrics
all controls
```

## FS1, block insertion

Replace the teacher shared-plus-routed MoE output with the student inside the real GLM block.

Measure:

```text
post-MoE state
post-block state
null-relative skill
centered cosine
relative L2
next-layer input
router behavior in the next layer
short logit lens when valid
```

The first decisive gate is:

```text
student block output beats the frozen mean-null and weight-space control
with positive held-out confidence lower bound
```

## FS2, next-layer propagation

Advance at least one complete following layer.

Measure:

```text
attention/state drift
pre-router state
router top-k and margins
weighted MoE output
post-block state
amplification or recovery
```

This is the minimum F2 gate.

## FS3, layer-stratified replication

Fit and test on:

```text
early sparse layer
middle sparse layer
late sparse layer
final sparse layer
```

Include the current layer 38 result as one stratum, not the universal result.

Required comparisons:

```text
per-layer h1024 functional student
per-layer full affine/linear upper control
small structured functional student
weight-space control
null controls
```

## FS4, cross-layer sharing

Only after per-layer results exist, test:

```text
shared functional backbone plus layer-specific readout
shared feature map plus layer-specific readout
layer-conditioned small student
per-layer independent student
```

Do not assume experts or layers share a subspace.

Measure whether sharing saves complete bytes without losing null-corrected skill.

## FS5, corpus and seed replication

Use:

```text
new documents
new prompt construction
English
Chinese
mixed language
code
math
reasoning
tool/agent formatting
low-router-margin tokens
longer contexts
multiple student seeds
```

No refit on replication.

## FS6, complete physical auction

Calculate exact projected and realized complete-model rates for:

```text
per-layer h1024
per-layer affine/linear upper control
best structured student
shared-backbone alternative
optional diagnosis-matched Doctor
```

Do not use organ-local BPW as complete-model BPW.

Required exact complete-model target rungs:

```text
<=0.75
approximately 0.50
approximately 0.333333
lower when earned
```

## FS7, functional full-stream admission

A functional candidate earns the GLM full stream only when:

```text
FS1 block gate passes
FS2 propagation gate passes
early/middle/late replication passes
complete physical BPW is legal
direct runtime codec exists
```

---

# 5. REPRESENTATION DECISION

Seal exactly one:

## FUNCTIONAL_STREAM_ADMITTED

```text
functional student passes FS1 through FS7
freeze one functional primary
freeze one lower-rate challenger when earned
retain a bounded weight-space negative control only on pilot fixtures
restart Generation B source traversal with functional payloads
```

## FUNCTIONAL_PARTIAL_ONLY

```text
functional result replicates locally but complete-model rate or propagation fails
retain the causal law
do not full-stream GLM
move to the next architecture parent
```

## FUNCTIONAL_ESCAPE_REFUTED

```text
student fails null-corrected block, propagation, or cross-layer replication
seal the exact reason
close GLM without a full source run
promote the corrected methodology to the next parent
```

## INSTRUMENT_INVALID

```text
repair data, forward, or metric instrument
invalidate dependent claims
rerun only affected experiments
```

Create:

```text
GLM52_FUNCTIONAL_DECISION.json
GLM52_FUNCTIONAL_DECISION.md
```

---

# 6. FUNCTIONAL GENERATION B STREAM

Run this section only after `FUNCTIONAL_STREAM_ADMITTED`.

The source rotation becomes:

```text
FETCH
-> VERIFY
-> NATURAL TEACHER CAPTURE
-> FIT FUNCTIONAL STUDENT
-> RUN NULLS AND CONTROLS
-> RUN PROPAGATION
-> SERIALIZE `.gravity`
-> COMPLETE TENSOR/FUNCTION DISPOSITION SEAL
-> EVICT BF16
```

## 6.1 Teacher evidence

Capture the minimum inference-target evidence:

```text
student input hidden state
teacher MoE output
post-block state
next-layer state
router margins
data membership
source lineage
```

## 6.2 Source tensor disposition

A functional student may replace an expert organ without storing every original expert weight.

The manifest must state:

```text
REPLACED_BY_FUNCTIONAL_CODEC
```

and bind:

```text
source tensor set
teacher function target
student payload
coverage proof
quality evidence
runtime implementation
```

No source tensor silently disappears.

## 6.3 Candidate count

Full stream:

```text
one functional primary
one lower-rate functional challenger when earned
no full weight-space control
```

## 6.4 Targeted refetch

After complete compact execution:

```text
diagnose failing layers
refetch only affected dependency windows
refit or replace only those functional shards
```

No repeated complete stream.

---

# 7. DECODER AND RUNTIME INTEGRATION

The existing Ascension decoder lane remains mandatory.

The performance methodology must follow matched baselines and complete-workload proof, not the retired `35.9x` claim.

## 7.1 Existing compact-weight paths

Finish and retain when relevant:

```text
safe bounded index unpack
native packed-width execution
2D split-chunk decode-FMA
shared on-chip lookup-linear
kernel selection by geometry
real `.gravity` tensor parity
```

These remain useful for:

```text
protected organs
attention/state tensors
weight-space controls
other model families
```

## 7.2 Functional codec CPU authority

Implement deterministic CPU execution for:

```text
feature generation
affine/linear map
readout
bias/normalization
optional gate/Doctor
```

## 7.3 Functional codec Metal paths

Benchmark materially different execution grammars:

### FRT-A, explicit feature map

```text
resident generated feature matrix
feature projection
activation
readout
```

### FRT-B, procedural feature generation

```text
generate features from seed in-kernel
avoid expanded persistent matrix
fuse feature generation and readout where possible
```

### FRT-C, structured fast transform

Only when scientifically equivalent or retrained:

```text
Hadamard/sign transform
sparse projection
block transform
```

### FRT-D, direct affine/linear control

Use the full linear result as the measured quality upper control and runtime baseline.

## 7.4 Complete functional MoE replacement

The runtime consumes:

```text
hidden state
layer identity
functional payload
```

and returns:

```text
shared-plus-routed MoE output replacement
```

No router or expert execution is required when the frozen student contract replaces the complete MoE function.

If routing remains necessary for quality:

```text
bill and execute it explicitly
```

## 7.5 Block and token integration

Prove:

```text
`.gravity`
-> production adapter
-> attention/state
-> functional MoE replacement
-> residual
-> next layer
-> final norm/head
-> sampling
```

## 7.6 Runtime metrics

Report:

```text
artifact BPW
expanded resident bytes
active bytes/token
operations/token
command buffers/token
block latency
complete-token latency
true batch-1 TPS
prefill TPS
quality
```

## 7.7 Runtime frontier

The functional student may create a substantially lower active-byte and active-expert path than compressed teacher experts.

Update:

```text
PHYSICAL_FRONTIER
TRAJECTORY_FRONTIER
FULL_MODEL_FRONTIER
RUNTIME_FRONTIER
ENERGY_FRONTIER
```

separately.

---

# 8. DIRECT TPS ASCENSION

The current target sequence remains:

```text
first true token
5 TPS
10 TPS
25 TPS
50 TPS
100 TPS
250 TPS
500 TPS
1000 TPS moonshot
```

For the functional path, calculate the new token roofline from:

```text
functional payload reads
expanded runtime state
attention/state path
protected organs
KV/state traffic
operations
sequential stages
command submission
```

Do not reuse the old 5.914-GB/token estimate when the expert function has been replaced.

## 8.1 Immediate performance question

Determine whether the functional replacement changes the binding limit from:

```text
expert weight traffic
```

to:

```text
attention/state
command submission
functional map compute
KV/state
```

## 8.2 Submission collapse

Continue:

```text
attention 3 command buffers -> 1
few command buffers/block
one replayable token graph
no Python in the hot loop
```

## 8.3 Sequential-depth lane

After one-layer functional replacement proves itself, test:

```text
2 teacher blocks -> 1 functional superblock
4 -> 1
8 -> 1
```

This is the route from fast complete tokens toward the 1000-TPS native-model frontier.

---

# 9. FLOP AND JOULE UPDATE

For the functional student report:

```text
teacher dense-equivalent operations
teacher expert bytes
student artifact bytes
student expanded bytes
student operations
avoided routing/expert operations
block latency
token latency
```

Classify:

```text
FEWER_BYTES_SAME_ARITHMETIC
FEWER_FLOPS_AND_BYTES
CONDITIONAL_COMPUTE
NATIVE_FUNCTIONAL_RUNTIME
```

The expected winning class is now:

```text
NATIVE_FUNCTIONAL_RUNTIME
```

but it must be earned.

Energy remains:

```text
measured through an accepted source
or UNAVAILABLE
```

No inferred joules.

---

# 10. CROSS-MODEL METHODOLOGY CHANGE

The next-parent protocol is now function-first.

For every new architecture parent:

```text
1. build correct source and adapter truth
2. establish fit-split nulls
3. run a dense/full affine functional upper control
4. run a tiny functional student
5. run weight-space control
6. compare null-corrected skill and exact physical cost
7. proceed to rate ladder only for the surviving paradigm
```

This above-ceiling and null-first probe must precede broad encoding work.

## 10.1 Retire globally

Do not rerun unchanged:

```text
raw-cosine promotion
mean-dominated activation metrics
weight-space ladder before null controls
activation-weighted SVD dead lever
shared expert-weight basis without architecture evidence
```

## 10.2 Reopen condition

A weight-space method reopens only when:

```text
the new architecture is materially different
the null-corrected oracle predicts recoverability
the runtime grammar changes its physical value
```

## 10.3 Model queue

Continue the Ascension parent queue, but use partial-first function probes.

Do not full-stream the next parent until the function-versus-weight pilot decides the paradigm.

---

# 11. PARALLEL EXECUTION

On the 96-GiB M3 Ultra:

## Heavy lane

Exactly one by default:

```text
GLM functional fit/propagation
complete compact evaluation
Metal block/token benchmark
or promoted parent full stream
```

## CPU/light lanes

Run concurrently:

```text
null correction of old artifacts
student controls
new corpus construction
seed replication
byte/FLOP ledgers
`.gravity` codec work
CPU reference
tests
reports
next-parent admission
```

## Network lane

While GLM functional work runs:

```text
prepare architecture-representative pilot windows for the next parent
```

Do not download multiple complete goliaths under the 600-GB storage policy.

## Contention

A second heavy lane requires paired proof of:

```text
<5% regression
no swap acceleration
no thermal warning
no memory warning
```

---

# 12. HIDE HANDOFF EFFECT

The HIDE contracts must support multiple `.gravity` representation codecs.

The runtime contract must expose:

```text
representation ID
artifact verification
required expanded runtime state
load-time generation
active-byte estimate
runtime selection
quality/research label
```

HIDE may not assume every Hawking model is a conventional quantized transformer.

Add:

```text
FUNCTIONAL_MODEL
```

to the model/runtime capability schema.

Do not refactor HIDE fully until the functional runtime boundary is frozen.

---

# 13. FIRST RETURN GATE

Return only at:

```text
endpoint: FUNCTIONAL_GRAVITY_ACTIVE
```

All must be true:

```text
1. metric contract sealed
2. old GLM artifacts re-scored
3. superseding scientific law sealed
4. FS0 reproduced
5. FS1 block insertion complete
6. FS2 next-layer propagation complete
7. at least early/middle/late layers tested or advancing under a detached controller
8. complete-model byte auction active
9. functional `.gravity` codec and CPU authority exist
10. functional Metal/runtime lane is active
11. next-parent pilot source is admitted or advancing
12. one heavy lane and multiple light lanes are active
13. status ledgers are growing
14. controller is restart-safe and detached
15. green milestones are committed and pushed to campaign branches
```

Required report:

```text
endpoint:
metric correction:
weight-space corrected verdict:
functional candidate:
local exact rate:
null-relative skill:
centered score:
block result:
propagation result:
layers tested:
complete-model projected BPW:
functional codec:
runtime status:
active bytes:
block latency:
true TPS status:
next parent:
controller PID/lease/heartbeat:
commits:
next autonomous action:
```

---

# 14. HAWKING CORE ASCENSION GATE, UPDATED

The existing `HAWKING_CORE_ASCENDED` requirements remain.

Add:

```text
null-corrected metric contract enforced globally
GLM functional escape replicated or refuted
functional `.gravity` representation supported
native functional runtime evaluated
cross-model function-first pilot protocol exercised
raw-cosine claims superseded
```

Hawking may ascend with:

```text
a winning functional architecture
a winning encoding architecture
a hybrid
or an honest empirical floor
```

It may not ascend with a broken promotion metric.

---

# 15. REQUIRED OUTPUTS

Produce:

```text
HAWKING_NULL_CORRECTED_METRIC_CONTRACT.md
HAWKING_NULL_CORRECTED_METRIC_CONTRACT.json
GLM52_METRIC_CORRECTION_LEDGER.jsonl
GLM52_CORRECTED_SCIENTIFIC_LAW.json
GLM52_CORRECTED_SCIENTIFIC_LAW.md
GLM52_FUNCTIONAL_STUDENT_CONTRACT.json
GLM52_FUNCTIONAL_EXPERIMENT_LEDGER.jsonl
GLM52_FUNCTIONAL_BLOCK_RESULT.json
GLM52_FUNCTIONAL_PROPAGATION_RESULT.json
GLM52_FUNCTIONAL_LAYER_TRANSFER.json
GLM52_FUNCTIONAL_BYTE_AUCTION.json
GLM52_FUNCTIONAL_DECISION.json
GLM52_FUNCTIONAL_DECISION.md
GRAVITY_FUNCTIONAL_CODEC_SPEC.md
GLM52_FUNCTIONAL_CPU_PARITY.json
GLM52_FUNCTIONAL_METAL_BENCHMARK.json
GLM52_FUNCTIONAL_RUNTIME_RESULT.json
GLM52_FUNCTIONAL_TOKEN_ROOFLINE.json
HAWKING_GRAVITY_CROSS_MODEL_TRANSFER.md
HAWKING_HIDE_RUNTIME_CONTRACT.json
HAWKING_ASCENSION_STATUS.md
HAWKING_ASCENSION_STATUS.json
HAWKING_ASCENSION_FINAL.md
HAWKING_ASCENSION_FINAL.json
```

---

# FINAL DIRECTIVE

The old weight-space answer is now stronger and more negative.

The new functional answer is smaller, better, and still incomplete.

Do not retreat to the old codec because it has more infrastructure.

Do not celebrate a one-layer student as a model.

Correct every metric.

Insert the student into the real block.

Propagate it forward.

Test early, middle, late, new documents, and new seeds.

Calculate complete-model physical bytes.

Build the native `.gravity` codec.

Build the direct CPU and Metal runtime.

If it survives, restart GLM around the functional representation and finish the source once.

If it fails, close GLM immediately and carry the function-first, null-corrected methodology to the next parent.

Keep completing the decoder, complete token, prefill, FLOP, joule, controller, and HIDE contracts in parallel.

> Hawking Ascension now turns on a sharper question: not how tightly the teacher's expert weights can be encoded, but how little physical function is required to reproduce the teacher's useful causal transition.
