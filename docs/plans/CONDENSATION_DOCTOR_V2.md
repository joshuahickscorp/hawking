# Condensation Doctor v2 — Program Synthesis for Capability Restoration

**Status (2026-07-12): historical scaffold, source-linked, deliberately execution-free, and
scientifically superseded by [`DOCTOR_V5.md`](DOCTOR_V5.md).** The detached v2 observer remains
hash-pinned behind the active Studio owner and is not silently migrated. This document does not
change the active Studio recipe, enqueue a heavy job, claim that a research paper reproduces on
Hawking, or turn a reconstruction oracle into a deployable artifact.

The checked-in scaffold consists of:

- [`healer_abi.py`](../../tools/condense/healer_abi.py), the backend-neutral, fail-closed
  correction-program ABI and identity validator;
- [`doctor_frontier.py`](../../tools/condense/doctor_frontier.py), the deterministic research-space
  compiler and advisory selector; and
- [`doctor_frontier_queue.py`](../../tools/condense/doctor_frontier_queue.py), the detached,
  observation-only campaign checkpoint that waits behind the current Studio owner and never launches
  a worker;
- [`doctor_frontier_worker.py`](../../tools/condense/doctor_frontier_worker.py), a preflight-only
  executable boundary with an intentionally empty adapter allow-list and no launch capability; and
- [`doctor_v2_frontier_campaign.json`](../../reports/condense/doctor_v2_frontier_campaign.json), the
  current materialized plan.

These components perform no model execution. Every generated operator currently carries an explicit
implementation state and an `executor.wired` Boolean. The queue invokes no worker, and the preflight
worker cannot execute a treatment. A planned program is not runnable, and an external result is
inspiration—not Hawking evidence—until independently reproduced through the proof ladder below.

## 1. Mission: heal capability, not weights

The Doctor is not LoRA, QAT, residual quantization, or any one algorithm. It is a mechanism-agnostic
system that answers five questions:

1. **Diagnose:** where, when, and for which capability did condensation damage the model?
2. **Prescribe:** which representation, correction, data, objective, and runtime policy offers the
   best expected capability per physical byte and joule?
3. **Treat:** rewrite the base, add a static correction, train a codec-aware repair, or attach an
   external capability mechanism.
4. **Route:** apply treatment by parameter group, expert, token, domain, failure syndrome, or system
   state instead of treating the whole model uniformly.
5. **Learn:** retain positive and negative experiments, transfer measured effects across scales, and
   choose the next experiment by Pareto value and value of information.

The restoration equation is intentionally broader than a weight delta:

```text
y = K_base(x, packed_base)
    + Σ K_static_i(x, correction_i)
    + Σ route_j(x, state) · K_dynamic_j(x, correction_j)
    + K_external(x, retrieval_or_verifier_state)
```

The Doctor may change the representation, the correction graph, runtime state, or the execution
policy. The only invariant is that all bytes, movement, latency, energy, communication, rejected
work, and failure tails are charged.

The objective is not nominal bits or FLOPS. It is to maximize:

- capability per joule;
- capability per physical and resident byte;
- capability per byte moved;
- capability per active parameter;
- capability per accepted token; and
- capability per unit of wall-clock time at the declared latency SLO.

Wall clock is not a campaign cutoff. A run may last weeks. It still needs progress heartbeats,
atomic checkpoints, safety pauses, and a scientific completion condition.

## 2. The representation transition at two bits

Above roughly two physical bits per weight, it remains reasonable to begin with conventional PTQ
and ask how much last-mile correction is needed. At and below two bits, that assumption must stop
being the default. The experiment becomes **representation reconstruction**:

- scalar rounding no longer has enough states to preserve all important structure;
- a small low-rank adapter cannot be assumed to restore a high-rank representation collapse;
- codebooks, binary factors, structured patterns, learned transforms, QAT/base rewrites, progressive
  slices, expert-specific formats, and conditional correction must receive mandatory coverage; and
- a larger high-precision shadow is not a condensed artifact.

This boundary is a research policy, not a theorem that every model fails at exactly 2.0 bpw. The
point is to prevent LoRA-only recovery from monopolizing the search precisely where a different
representation is most likely required.

External work motivates the breadth but does not establish Hawking results:

- [LittleBit](https://arxiv.org/abs/2506.13771) studies ultra-low-bit latent factorization,
  binarized factors, and multi-scale compensation, including a 0.1-bpw regime.
- [NanoQuant](https://arxiv.org/abs/2602.06694) formulates binary/sub-bit PTQ as low-rank binary
  factorization with ADMM initialization followed by block/model reconstruction.
- [BTC-LLM](https://arxiv.org/abs/2506.12040) combines learned transforms with binary-pattern
  codebooks rather than relying on irregular sparse masks.
- [BWLA](https://arxiv.org/abs/2605.00422) targets binarized weights plus low-bit activations using
  an orthogonal-Kronecker transform and proximal-SVD refinement.
- [LC-QAT](https://arxiv.org/abs/2606.10531) makes a 2-bit vector-quantized representation
  differentiable through a learned affine formulation.
- [MatGPTQ](https://arxiv.org/abs/2602.03537) produces a sliceable, multi-precision parent with
  cross-bit compensation, motivating—but not proving—Hawking's progressive prefix design.
- [ScaleBITS](https://arxiv.org/abs/2602.17698) treats mixed precision as hardware-aligned global
  constrained allocation rather than an irregular per-weight heuristic.
- [MARR](https://arxiv.org/abs/2605.17997) makes residual-reconstruction strength module-specific
  and adapts it with reconstruction feedback.
- [SPEAR](https://arxiv.org/abs/2606.11244) diagnoses token-dependent quantization error and routes
  lightweight error compensators, demonstrating why static correction is not the only recovery
  model worth testing.

These methods sometimes report strong results on different models, hardware, data, and metrics.
Their reported speedups and quality numbers must not be copied into Hawking scorecards. Hawking must
rebuild the relevant mechanism, bill its exact artifacts, and measure it on the same box.

## 3. Doctor as a typed program-synthesis OS

A HealerProgram is a typed directed acyclic graph:

```text
Parent
  → Analyze*
  → BaseTransform*
  → BaseCodec | BaseRewrite
  → StaticCorrection* | GatedCorrection* | Train*
  → StateCodec* | RuntimePolicy* | Retrieval* | Verifier*
  → Package
  → Evaluate*
```

The compiler rejects cycles, missing dependencies, unknown kinds, invalid backend states, and
dynamic operators that do not require mean/p95/worst cost accounting. Operators are selected by
stable mechanism id and version, not by framework name. Apple CPU, Metal, CUDA, distributed, and
future-specialized implementations can lower the same semantic operator, but their receipts never
alias.

### 3.1 Diagnose

Diagnosis builds a multi-resolution damage map rather than one global weight RMSE:

- weight-error spectrum and singular-value decay;
- activation energy, outlier statistics, and Hessian/Fisher sketches;
- parent/condensed hidden-state and logit divergence;
- early/middle/late layer propagation;
- attention, FFN, embedding, router, shared-expert, and routed-expert roles;
- hot/cold expert affinity and routing entropy;
- token entropy and CKA-style error risk;
- capability-failure clusters from paired task traces; and
- causal capability islands tested by activation patching or causal tracing.

The Hawking-original `quant_error_syndrome_model` goes further: it proposes learning a small
predictor for the kind of repair a token/layer needs. It remains unimplemented. Its first job is to
produce an oracle showing predictable syndromes—not to justify a production router.

### 3.2 Prescribe

The prescription compiler chooses a complete program under exact constraints:

```text
minimize  physical_bytes,
          resident_bytes,
          mean/p95/worst_bytes_moved,
          joules_per_accepted_token,
          p95_latency

subject to
          worst-domain capability gate,
          parent-relative task tripwire,
          process-memory envelope,
          zero swap,
          complete tensor ownership,
          native runtime parity
```

Every prescription binds the model revision, config, tokenizer, data sets, teacher, operator source,
seed, hardware/backend, runtime ABI, and byte budget. Changing any one creates a different identity.

### 3.3 Treat

Treatment families include:

- base transforms: identity, RHT, activation scaling, learned rotations, channel reorder, and
  residual-subspace transforms;
- base representations: scalar STRAND/affine controls, mixed precision, additive/vector/lattice
  codebooks, binary factors, binary patterns, structured binary regions, and shared grammars;
- static corrections: zero, bias, residual-SVD, variable rank, low-rank plus sparse exceptions,
  module-adaptive residuals, codec-aware base rewrite, self/large-teacher distillation, and
  capability-targeted adapters;
- state treatment: FP16/INT4/INT2/codebook KV and future architecture-specific persistent state;
- system treatment: nested target/drafter, retrieval repair, output verification, selective
  regeneration, and on-demand weight synthesis.

### 3.4 Route

Routing operates at four different granularities:

- **Parameter routing:** allocate codec, bits, transform, rank, dtype, sparse budget, and objective
  per semantic tensor group, then split only the groups whose marginal capability per byte warrants
  finer row/channel/block treatment. Dense per-weight optimizer state is forbidden at trillion scale;
  individual weights appear only as billed sparse exceptions or compact codes.
- **Expert routing:** protect routers/shared experts, assign hot and cold experts different formats,
  learn cross-expert dictionaries plus expert coefficients, and report both installed bytes and
  active bytes per token.
- **Token routing:** use a syndrome/risk gate to invoke corrections only where predicted information
  gain exceeds cost. Dynamic treatments report installed, mean, p95, and worst traffic and latency.
- **Capability routing:** keep a small bank of domain/capability-specific structured corrections;
  hot adapters remain resident and cold adapters are mmap/prefetch candidates. This is a Hawking
  hypothesis, not an implemented feature.

### 3.5 Learn

The ledger stores treatment effects normalized by:

- capability recovered per serialized byte;
- capability recovered per moved byte;
- recovery per training token;
- recovery per active parameter; and
- system goodput/energy change at the same SLO.

Transfer fingerprints include architecture, tensor role, log parameters/active parameters, depth,
width, head/expert geometry, activation statistics, error spectrum, spectral decay, expert hotness,
routing entropy, and token risk. Small-model evidence changes queue priority only. Every larger scale
still requires its own held-out confirmation.

## 4. Exact ABI and evidence contracts

[`healer_abi.py`](../../tools/condense/healer_abi.py) currently defines six schema identifiers:

| Schema | Purpose | Current validation status |
|---|---|---|
| `hawking.healer_program.v2` | Canonical typed DAG, model/target binding, operator support, cost and evidence policy | Implemented |
| `hawking.healer_artifact.v2` | Packed base, file hashes/bytes, tensor ownership, residency and dynamic costs | Implemented |
| `hawking.healer_cell.v2` | Immutable backend/fidelity/seed/data-specific work unit | Implemented |
| `hawking.healer_checkpoint.v2` | Exact-resume state bound to a cell | Implemented |
| `hawking.healer_observation.v2` | Result envelope for metrics, uncertainty, traces, and verdict | Implemented |
| `hawking.doctor_frontier_campaign.v2` | Search space, candidates, fidelity policy, backend and Velocity++ contracts | Implemented in the compiler |

### 4.1 Program identity

An executable program requires SHA-256 bindings for parent revision, config, and tokenizer. Every
node declares:

- id, kind, phase, mechanism, and mechanism version;
- dependencies and parameters;
- implementation state;
- support state for `apple_cpu`, `metal`, `cuda`, `distributed`, and `future_specialized`;
- a cost contract that makes actual bytes authoritative; and
- an executor envelope with `wired`, source hash, and argv.

Research/unimplemented nodes cannot be wired. A wired node requires executable program mode and a
source hash. The current frontier materializer emits planned programs with every executor unwired.

### 4.2 Artifact contract

A valid v2 artifact must:

- bind the exact program hash;
- bind a packed-base hash;
- enumerate every physical file with SHA-256 and byte length;
- make the physical model byte count equal the exact file-byte sum;
- prove complete tensor ownership;
- declare `dense_parent_fallback: false`;
- record positive resident peak bytes; and
- record monotonic mean ≤ p95 ≤ worst dynamic bytes per token.

A dense reconstructed safetensor may be useful for an oracle, but it is not this artifact.

### 4.3 Cell and checkpoint contract

Each cell binds program, backend, fidelity, seed, calibration/selection/final-eval hashes, worker
source, and resource estimates. Exact resume requires:

- operator state;
- optimizer;
- microstep and gradient-accumulation phase;
- RNG state;
- sampler/data cursor;
- teacher-cache identity;
- source shard/byte offset;
- partial-output hashes; and
- resume-command identity.

A checkpoint must hash each state group, record nonnegative positions, and confirm fsync completion.
An adapter-only `latest` file is recovery material, not an exact-resume checkpoint.

### 4.4 Proof state machine

The exact current ABI order is:

```text
planned
  → reconstruction_oracle
  → packed_artifact
  → native_runtime_parity
  → resident_capability
  → capability_efficiency_promoted
```

Transitions are monotonic and bound to artifact hashes. A process exit code, filename existence, or
logical codec bpw cannot skip a state. A future ABI revision may add more granular logical-byte and
round-trip states, but it must not weaken this order.

## 5. Progressive factor lane: 0.10 → 0.25 → 0.55 → 0.80 bpw

This is a Hawking-original research program, not an implemented codec. It asks whether one nested
representation can behave like a progressively healing model:

```text
S0                    cumulative target 0.10 bpw
S0 + S1               cumulative target 0.25 bpw
S0 + S1 + S2          cumulative target 0.55 bpw
S0 + S1 + S2 + S3     cumulative target 0.80 bpw
```

Each prefix is independently packed, hashed, round-tripped, evaluated, and billed. The factors and
refinement slices are jointly optimized across all four prefix objectives so that a higher prefix
does not merely append an unrelated adapter. No prefix may materialize a full dense shadow.

| Prefix | Scientific role | Candidate representation | Admission rule |
|---|---|---|---|
| 0.10 | Destructive stress control and possible ultra-cheap drafter | LittleBit/NanoQuant-inspired binary latent factors or shared parameter grammar | Never assumed viable; proceed only from exact oracle evidence |
| 0.25 | Primary resident fit target for the 1.6T terminal model | additional factors/pattern codebook/structured exceptions | Full physical bytes including factor scales, indices and metadata must fit |
| 0.55 | Intermediate capability bridge | first large refinement slice, module-adaptive residuals, capability corrections | Must improve held-out capability per added serialized byte |
| 0.80 | High-capability control | expanded factor/codebook capacity plus selective treatment | Must compete with independent 0.8-bpw encodings at equal physical bytes |

Nominal weight-only sizes illustrate why the lane matters, but are not fit claims:

| Model | 0.10 bpw | 0.25 bpw | 0.55 bpw | 0.80 bpw |
|---|---:|---:|---:|---:|
| 120B | 1.5 GB | 3.75 GB | 8.25 GB | 12.0 GB |
| Kimi 1.1T | 13.75 GB | 34.38 GB | 75.63 GB | 110.0 GB |
| V4-Pro 1.6T | 20.0 GB | 50.0 GB | 110.0 GB | 160.0 GB |

Container framing, pass-through tensors, codebooks, factors, corrections, alignment, runtime state,
and working memory increase these values. For 1.6T, 0.55 and 0.80 are out-of-core controls under the
78-GB process envelope; 0.25 is only a resident candidate after actual artifact/runtime proof.

The `shared_parameter_grammar` tests mutual information across layers and experts by learning shared
dictionaries or a compact block generator plus exceptions. `on_demand_weight_synthesis` asks whether
hot compute tiles can be generated from that grammar and cached. Generated dense tiles must be a
bounded runtime cache; they do not disappear from traffic or residency accounting.

## 6. Campaign breadth and scheduling

The current durable campaign contains:

- **10 models:** Qwen2.5 0.5B, 1.5B, 7B, 14B, 32B, 72B; gpt-oss-120B;
  DeepSeek-V4-Flash; Kimi-K2.6; and DeepSeek-V4-Pro;
- **76 model/rate points**;
- **62 operators** across diagnosis, representation, transform, correction, state, runtime,
  packaging, and evaluation;
- **634,580,352 projected Cartesian cells**;
- **8,192 explicit deterministic candidates**, including **484 mandatory controls**; and
- **7 fidelity levels**, F0 through F6.

The explicit candidates are plan identities, not 8,192 running jobs. The compiler avoids scanning
the full Cartesian space by deterministic mixed-radix sampling and retains mandatory controls. It
can materialize more cells after evidence-driven promotion.

The current selector is advisory: it maintains a measured Pareto set and combines fidelity cost,
uncertainty/value-of-information, mandatory-family coverage, low-rate boundary coverage, and
exploration. It is not yet the target learned transfer scheduler; promotion bias must be calibrated
from completed observations rather than assumed from F0--F2 proxies.

### 6.1 Multi-fidelity ladder

| Fidelity | Work | Promotion meaning |
|---|---|---|
| F0 | Weight/statistics and exact-byte feasibility | Program is coherent and worth an oracle |
| F1 | Representative tensor/layer output reconstruction | Mechanism can represent local structure |
| F2 | Representative shard plus activation/logit sketches | Damage does not immediately compound |
| F3 | Full streamed model, multiwindow ≥4, capability tripwire | Candidate has full-model scientific evidence |
| F4 | Three seeds, domain and hard-example ablations | Effect is replicated and robust |
| F5 | Packed round trip and native CPU/Metal parity | Physical Apple artifact executes correctly |
| F6 | Same-box KV/speculation/energy/latency; separate later CUDA receipt | System efficiency claim is eligible |

Fidelity metrics are not assumed unbiased. The campaign must learn how F0–F2 predict F3–F6 and
carry uncertainty through promotion.

### 6.2 Pareto and value of information

The scheduler maintains an uncertainty-aware frontier over:

- worst-domain capability retention;
- physical and resident bytes;
- mean/p95/worst bytes moved;
- joules per accepted token;
- p95 latency; and
- peak memory and communication.

A target acquisition score is:

```text
priority(cell) = P(feasible)
                 · expected_Pareto_hypervolume_gain
                 + expected_information_gain_across_models_and_rates
                 + mandatory_family_coverage_bonus
                 + decision_boundary_uncertainty
```

Estimated resource cost orders work and packs safe waves; it is not a reason to abandon an expensive
family. Every relevant family gets at least a low-fidelity control at each applicable scale. A branch
closes only when replicated evidence shows dominance, a physical/runtime impossibility is recorded,
or an equivalent program has strictly stronger proof.

Campaign completion requires:

1. every applicable family to have valid evidence or a concrete incompatibility receipt;
2. every Pareto candidate to reach replicated full evaluation;
3. every deployable finalist to reach F5 and F6;
4. negative results to remain queryable; and
5. the frontier to remain stable across a complete synthesis round.

## 7. Largest-model execution model

The Doctor must not require the whole parent to be resident. For giant parents, a treatment is a
transactional stream:

```text
fetch verified shard
  → verify revision/path/hash
  → decode one bounded window
  → diagnose/transform/treat
  → pack output shard
  → round-trip and hash
  → fsync artifact + observation + checkpoint
  → release source window only under the source-lifecycle policy
```

Operations requiring global statistics use explicit passes with deterministic merge state. Expert
statistics stratify hot/cold experts. A 1.6T program must bind the architecture adapter, source-shard
map, global-state checkpoint, output-layout contract, and remote network bytes. The terminal queue
remains a readiness supervisor; it must never infer that these workers exist from a planned argv.

## 8. Apple first; CUDA is a separate proof path

### Apple path

Apple is the first product and scientific proof backend:

1. CPU/bfloat16 streamed oracle and training path;
2. physical `.tq`/v2 correction sections;
3. strict tensor ownership and packed round trip;
4. native Metal parity;
5. resident unified-memory execution with zero swap; and
6. same-box bytes moved, latency, energy, KV, and speculative goodput.

Unified memory is helpful for hot/cold corrections and bounded generated-tile caches, but it does
not make movement free. CPU↔GPU coherence, page faults, cache misses, and synchronization must be
measured.

Apple controls include [MLX-LM](https://github.com/ml-explore/mlx-lm),
[T-MAC](https://github.com/microsoft/T-MAC), and Apple research such as
[QuantSpec](https://machinelearning.apple.com/research/quantspec). They are comparison/inspiration
paths; they do not validate Hawking's artifact.

### CUDA path

CUDA is not a hidden flag in an Apple receipt. It gets independent:

- operator lowerings and source hashes;
- kernel/runtime parity;
- device, driver, compiler and topology identity;
- physical byte and communication ledgers;
- latency/energy measurements; and
- distributed merge/evaluation receipts.

CUDA can accelerate codebook/QAT research and later provide deployment, but its results do not enter
the Apple headline row. The shared semantic ABI allows comparison without conflation.

## 9. Velocity++ integration and blockers

“Velocity++” is treated here as Hawking's accepted-token goodput objective, not as a presently
discoverable source module. Doctor-v2 integrates it only after the condensed target has physical and
native proof.

The complete experiment identity binds target artifact and physical bpw, HealerProgram, tokenizer,
backend/kernel, drafter artifact/family, verifier path, KV precision, cache namespace, adaptive
policy, workload, and seed.

Current blockers are explicit:

1. the batched Qwen verifier does not execute the exact `.tq` target used by single-token decode;
2. the verifier cost curve is not yet target/KV/context/backend specific;
3. the runtime router lacks complete draft, verify, synchronization, rollback, cache-miss, and energy
   observations; and
4. speculative KV plus correction-graph cache namespaces are not transactionally proven.

Therefore no live speculative cell is eligible. Admission requires exact TQ single-token versus
TQ-batched parity for B=1–8 and a conservative cost-aware utility lower bound.

Goodput and energy are defined as:

```text
goodput = expected_committed_tokens
          / (draft_time + verify_time + sync_time + rollback_time)

energy = (J_draft + J_verify + J_sync + J_KV_writes
          + J_rejected + J_cache_miss)
         / committed_target_tokens
```

The most original integration is `nested_target_drafter`: use the 0.10–0.25 progressive prefix as
the drafter and enhancement slices as the same hash-bound target, reusing state instead of keeping
two unrelated models resident. This could reduce draft residency and duplicate movement, but it is
unimplemented and depends on transactional batched verification. A routed Doctor correction may
also change target distributions and acceptance; target, drafter, and Doctor policy must therefore
be optimized and measured jointly.

## 10. Horizon evaluation matrix

Bandwidth and latency entries below are hypotheses until measured on the declared hardware. Nominal
payload ratios exclude metadata, pass-through tensors, codebooks, corrections, state, and alignment.

### Immediate implementation

| Proposal | Theoretical complexity | Expected bandwidth / latency | Difficulty | Existing GPU / Apple / future hardware | Quantization + speculation | Distributed + future architectures |
|---|---|---|---|---|---|---|
| Harden HealerProgram/Artifact/Cell/Checkpoint plus observation validator | Compile/validate O(nodes+edges+files) | No direct runtime gain; prevents wasted or invalid month-scale work | Medium | Backend-neutral; stdlib control plane works now | Makes every quant/spec result hash-bound | Content-addressed cells distribute naturally; semantic roles avoid Qwen-only assumptions |
| Parameter/expert exact-byte water-filling | Sensitivity O(TP); allocation multiple-choice knapsack or Lagrangian O(GK log G) | Reduces low-value base/correction traffic; latency improves only if allocations stay kernel-aligned | Medium-high | GPU-portable analysis; CPU Apple planner now, Metal runtime later; natural compiler pass in future hardware | Joint bits/rank/sparse budget; verifier-critical tensors can receive protection | Tensor/expert groups parallelize; supports MoE and future SSM/multimodal roles |
| Diagnosis suite and syndrome oracle | O(P) statistics to O(TP) paired activation traces | No direct gain; should avoid applying correction to easy tokens/layers | Medium | CPU/MPS tracing feasible at small scale; CUDA scalable; future telemetry engines favorable | Identifies base collapse and draft-acceptance damage | Traces shard with deterministic reductions; architecture adapters define semantic probes |
| Static bias, residual-SVD, variable-rank and sparse controls | Offline O(mnr) or O(P log k); runtime narrow GEMV/gather | Correction traffic can be far below dense delta; irregular sparse latency may erase savings | Medium-high | Generic GPU math; Apple packed correction ABI missing; future fused epilogues attractive | Applies after any base; target or drafter correction | Per-tensor work parallel; structured exceptions preferable for future accelerators |
| Compile the 0.10→0.25→0.55→0.80 nested factor oracles | Factor/codebook fitting ranges from O(P) iterations to O(TP epochs) | Nominal weight payload ceilings are 160×/64×/29.1×/20× below BF16; runtime latency unknown | Very high | Research GPU references; Apple decoder absent; bitwise/near-memory hardware promising | Base representation research, not LoRA polish; 0.10–0.25 prefix is a possible drafter | Blocks/experts can fit independently; deterministic shared-grammar merge required |

### Medium-term research

| Proposal | Theoretical complexity | Expected bandwidth / latency | Difficulty | Existing GPU / Apple / future hardware | Quantization + speculation | Distributed + future architectures |
|---|---|---|---|---|---|---|
| LittleBit/NanoQuant/BTC/BWLA representation bakeoff | Iterative factor/transform/codebook reconstruction; typically O(iterations·P) | Sub-1-bit installed payload is possible in the source methods; Hawking latency depends on direct packed kernels | Very high | Mostly research GPU paths; new Apple CPU/Metal round trip; binary/LUT hardware favorable | Mandatory below 1 bit; prefix can draft only if acceptance pays | Block/expert factorization parallel; MoE may share dictionaries |
| LC-QAT plus exact-resume streamed codec-aware QAT/KD | O(TP·epochs), peak one block/shard plus optimizer/checkpoint | Same base bpw after rewrite; quality lever, not inherent speedup | Very high | CUDA references; new CPU/MPS streamed trainer; specialized block-local training later | Tunes real codec, target and drafter; avoids uniform-proxy claims | Shards distribute with global eval; natural expert-local optimization |
| MatGPTQ-style nested physical format | One multi-rate PTQ/QAT optimization plus per-prefix packaging | One installed parent may replace multiple models and reduce duplicate reads; slicing/kernel cost must be measured | Very high | GPU reference; progressive Metal kernel absent; bit-plane hardware natural | Direct foundation for nested target/drafter and adaptive quality | Prefix shards distribute; future architectures can expose native refinement planes |
| MARR/SPEAR-inspired module/token adaptive recovery | Offline diagnosis plus O(gate)+Σp_iC_i runtime | Mean correction traffic falls if routing is selective; p95/worst latency can worsen | Very high | CUDA research; Apple fused gated correction absent; event hardware favorable | Base-agnostic; gating must be part of exact target distribution and spec identity | Expert/tensor parallel synchronization is first-class; MoE/early-exit systems fit naturally |
| Native Apple correction/state runtime and Velocity++ parity | Runtime O(P_active)+correction/state work | Converts density into real bandwidth savings; latency only wins after fused decode/verify | Extreme | Metal is the primary implementation; CUDA receives separate later lowering; future unified cache engines ideal | Required for every deployable quant and spec claim | Distributed verifier/state transactions need explicit communication and rollback bytes |

### Long-term paradigm shifts

| Proposal | Theoretical complexity | Expected bandwidth / latency | Difficulty | Existing GPU / Apple / future hardware | Quantization + speculation | Distributed + future architectures |
|---|---|---|---|---|---|---|
| Shared parameter grammar across layers/experts | Learn/generate shared dictionaries or block generator plus exceptions | Could replace repeated weights with grammar+hot generated tiles; cache misses may dominate tails | Extreme | Prototype-able on GPU; Apple unified-memory tile cache is interesting; near-memory synthesis ideal | Representation is generated, not conventionally quantized; shared prefix may draft | Global grammar learning distributes; expert coefficients/local exceptions merge deterministically |
| On-demand weight synthesis | O(generator) per cache miss, amortized by tile reuse | Trades compute for source bandwidth; bounded dense cache still counts | Extreme | GPU graph/cache research; Apple unified memory useful; future compute-near-memory target | Can synthesize refinement only when target/verifier needs it | Remote tile generation possible but network tails count; applies to novel layer types |
| Capability adapter bank plus active teacher failure mining | Teacher cost only on uncertain/divergent traces; routed adapter cost Σp_iC_i | Small hot adapters can preserve rare skills without global traffic; cold-miss tails must be measured | Very high | Portable training; Apple mmap/prefetch runtime new; future semantic memory favorable | Repairs capability after any base; spec verifier may request a capability treatment | Teacher generation distributes; adapter bank maps to MoE/multimodal capability modules |
| Retrieval/output-verifier repair | ANN O(log N) approximate plus generation/verification | May replace factual parameter traffic but adds index/network and tail latency | High-extreme | Portable; local Apple mmap/ANN feasible; semantic-memory hardware promising | Cannot excuse destroyed reasoning; verifier can selectively regenerate | Retrieval is naturally distributed, but communication and cache-hit distributions are authoritative |
| Native low-bit-trained conditional architectures | Pretraining scale | Largest plausible long-run bandwidth/energy reduction because representation and runtime co-design | Extreme | Requires specialized training stack; Apple/CUDA controls first; future bitwise/near-memory hardware ideal | Removes PTQ mismatch; can train nested exits/drafters and routed corrections jointly | Expert/data parallel training; architectures can make conditional execution native |

## 11. Immediate integration sequence

The campaign validator, observation validator, detached observer, and preflight-only worker boundary
are now scaffolded. The observer pins the campaign hash across restarts, samples the live Studio owner,
heavy-work lease, pressure, swap, power, thermal state, and disk every 120 seconds, and records zero
worker launches. The preflight worker has no executor callable or subprocess path; its production
adapter registry is empty, so every current planned cell is refused.

The next safe engineering order is:

1. keep all v2 treatment executors disabled while the live Studio phase runs;
2. harden the existing single-cell audit bridge by binding the full compiled spec and worker hashes
   into resume identity;
3. implement exact-resume checkpoints before multi-week training is admitted;
4. build the variable-rank/precision correction artifact sections and CPU round trip;
5. register reviewed in-process adapters by immutable source hash, one operator family at a time;
6. admit one F0 zero-correction/equal-byte control only after the current Studio owner releases the
   heavy lease, then expand through the explicit multi-fidelity gates;
7. run streamed 32B/72B/120B cells only under the same heavy-work lease;
8. build native Metal parity before any resident or performance claim;
9. admit Velocity++ only after its explicit correctness/cost gates; and
10. add CUDA as an independent backend and receipt family after the Apple proof path is stable.

## 12. Non-negotiable honesty rules

- An external paper result is a hypothesis, not Hawking evidence.
- A planned candidate, source download, or executable-looking argv is not a launched experiment.
- Nominal bpw is never physical bpw.
- Logical codec accounting is not a packed artifact.
- A dense reconstruction override is an oracle, not a deployable model.
- Average dynamic cost never substitutes for p95 and worst cost.
- A PPL recovery does not substitute for capability evaluation.
- A capability result does not substitute for native parity, latency, energy, or memory evidence.
- Apple and CUDA receipts do not alias.
- A stopped training checkpoint is retryable evidence, not a completed experiment.
- Zero correction is preserved and can win; a treatment is never promoted merely because it ran.
- Negative results remain in the campaign and update the transfer policy.

The Doctor-v2 program is complete only when it produces the smallest source-bound system that
passes the full capability and native-efficiency contract—not when it generates the largest grid,
runs the longest job, or reports the lowest nominal bit rate.
