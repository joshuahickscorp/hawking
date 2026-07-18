# Condensation Doctor v5 — Three Expansion Passes and Adversarial Proof Plan

**Status:** research and proof specification, not a performance result  
**Canonical names:** **Condensation Doctor v5** and **Training Ladder v5**  
**Pass provenance:** v3 representation foundry → mandatory rest/audit → v4 capability restoration
lab → mandatory rest/audit → v5 adversarial proof synthesis  
**Primary objective:** maximum standalone or explicitly billed system capability and quality under an
exact physical-byte budget  
**Deferred objective:** speed. Speed, latency, energy, speculation, and native-kernel claims require a
later runtime ladder and cannot decide this quality-first campaign.

Doctor v5 is designed to discover and then try to prove a quality-dominant condensation system. It
does **not** establish that Hawking is already dominant, unbeatable, near-lossless, sub-bit, or able
to run a particular frontier model. Those words become admissible only after the sealed and
independently reproduced v5 dominance rule passes. Until then the only valid description is
**planned frontier research**.

This document is the expanded research argument behind the executable policy surfaces in
`tools/condense/doctor_v5_contract.py` and `tools/condense/training_ladder_v5.py`. Those surfaces must
follow the four canonical claim lanes in §4; older three-lane terminology is superseded.

---

## 1. The question v5 is allowed to answer

The campaign does not ask whether a compressed checkpoint has low weight MSE or attractive nominal
bits per weight. It asks:

> At a fully accounted physical deployment budget, does this artifact or explicitly declared system
> preserve or improve useful capability better than its exact parent-relative and same-budget
> competitors, without hiding a regression in a weak domain, an external information source, or an
> unbilled file?

The objective order is:

1. satisfy every protected-domain parent-relative non-inferiority gate appropriate to the claim lane;
2. maximize the worst-domain lower confidence bound, not the average score;
3. dominate every applicable same-parent, same-or-lower-physical-byte competitor;
4. maximize capability per physical byte and resident byte; and only later
5. compare capability per joule, moved byte, active parameter, and wall-clock unit at a fixed SLO.

FLOPS are neither an objective nor a proxy for success. Weight MSE, reconstruction error, perplexity,
and teacher KL are diagnostic signals. None may substitute for the capability vector.

### 1.1 Evidence vocabulary

Every mechanism, artifact, and claim must use one of these states:

- **PLANNED:** specified but not executed;
- **FEASIBILITY:** exact-byte or analytic feasibility only;
- **TENSOR/BLOCK/SHARD ORACLE:** local causal evidence, not a model claim;
- **FULL-MODEL QUALITY:** one complete evaluation lineage;
- **REPLICATED QUALITY:** multiple seeds and calibration draws;
- **SEALED FINAL:** preregistered held-out evaluation consumed once; or
- **INDEPENDENT REPRODUCTION:** a separate reproduction reaches the same verdict.

External papers establish priors and competitor obligations. Their numbers are never Hawking
evidence. A projected size is never a packed artifact. A packed artifact is never a resident model.
A resident model is never a quality winner without the complete evaluation contract.

---

## 2. Accepted and rejected assumptions

These are the assumptions carried into the three passes. Each audit may narrow or reject an accepted
hypothesis, but may not silently revive a rejected one.

### 2.1 Accepted as working constraints

| Assumption | Why it is accepted | Consequence for v5 |
|---|---|---|
| Actual model-specific file bytes are authoritative. | Nominal payload routinely omits heads, embeddings, scales, codebooks, indices, corrections, routers, and alignment. | Every candidate has a byte-complete manifest and tensor-ownership proof. |
| Below roughly two bits, compression becomes representation learning. | [ParetoQ](https://arxiv.org/abs/2502.02631) reports a sharp representation transition between two and three bits. | Both progressive inheritance and representation-reset/QAT branches are mandatory below two bits. |
| Different tensors require different representation families. | GQA/KV, lexical, router, shared-expert, and routed-expert tensors have different geometry and failure cost. | The Doctor searches by semantic tensor group and retains uniform controls. |
| Preserved-but-noisy signal and destroyed computation are different diseases. | A correction can denoise a surviving computation but cannot be assumed to recreate a missing algorithm or feature basis. | Diagnosis hard-routes computation collapse to structural reconstruction. |
| Capability is a vector with protected tails. | Perplexity and macro averages can hide failures in reasoning, instruction following, rare experts, or safety. | One protected-domain failure blocks a headline. |
| Any recovered information has a source. | A finite artifact cannot recreate discarded information from nothing. | Teacher data, retrieval, tools, adapters, and persistent state are provenance-bound and billed in the appropriate ledger. |
| Scale transfer is uncertain. | The strongest sub-bit evidence ends well below modern trillion-scale models. | Small-model evidence changes priority only; every scale class needs its own held-out proof. |
| The full parent need not be resident during research. | Giant checkpoints can be transformed transactionally by shard/window. | Frontier work uses immutable source manifests, bounded windows, deterministic global reductions, and exact resume. |

### 2.2 Rejected

| Rejected assumption | Reason for rejection |
|---|---|
| “One bit” or “0.1 bpw” in a paper means a whole deployable artifact at that rate. | Many results quote a decoder-body or selected-linear-matrix target while retaining other tensors at higher precision. |
| A low-rank adapter can necessarily heal sub-bit collapse. | Lost feature rank, routing, or computation may be outside the adapter’s reachable subspace. |
| Weight MSE or average perplexity is a quality claim. | Neither proves instruction, reasoning, code, multilingual, calibration, or safety retention. |
| A larger model at a lower nominal rate automatically beats a smaller, higher-precision model at equal bytes. | This is an empirical competitor comparison, not a scaling law. |
| Sub-bit results at 7B–32B extrapolate to 70B, 671B, or 1.6T. | Training, lexical overhead, expert coverage, and error propagation change with scale and architecture. |
| Residual correction is monotonically helpful. | LittleBit reports cases where residual compensation hurts; [MARR](https://arxiv.org/abs/2605.17997) likewise treats residual strength as a controlled variable. |
| Dynamic per-token precision can be added after PTQ. | [MatGPTQ](https://arxiv.org/abs/2602.03537) reports static configurations often dominating untrained dynamic selection. |
| Lossless entropy estimates are task rate-distortion estimates. | Marginal symbol entropy bounds a declared lossless source model, not retained model capability. |
| MoE expert-average bpw equals total-model bpw. | Attention, routers, shared experts, embeddings, and metadata may remain at different precision. |
| More training or a stronger teacher proves restoration. | It may create a new, elevated model. That evidence belongs to a different claim lane. |
| Retrieval, tools, or a verifier may support a core artifact claim. | They add information and runtime state and therefore belong only to the augmented-system lane. |
| Reaching the v5 code or document version proves dominance. | Version is protocol maturity; proof state comes only from sealed evidence and independent reproduction. |

---

## 3. Hard theoretical and physical limits

### 3.1 Information cannot be restored without side information

An artifact containing (B) bits can distinguish at most (2^B) artifact states. If condensation
discards information needed by a capability, recovery must come from one or more declared sources:

- structural priors in the representation;
- training/calibration examples;
- parent or stronger-teacher outputs;
- a repair module or sparse exception table;
- retrieval, tools, a verifier, or another model; or
- persistent runtime state.

Training-time information is recorded in the provenance ledger. Shipped repair information is
counted in physical artifact bytes. Runtime information is forbidden for the three standalone lanes
and fully billed for `augmented_system`. “The Doctor restored it” is not an information source.

### 3.2 Physical whole-artifact accounting

For a source model with exact nominal parameter count (N_{total}):

\[
b_{physical}=\frac{8\sum_f \operatorname{bytes}(f)}{N_{total}}.
\]

The sum includes every model-specific file required to load the candidate from a clean environment:

- condensed base payload and pass-through tensors;
- embeddings, LM head, norms, position parameters, routers, and shared experts;
- scales, zero points, rotations, envelopes, codebooks, entropy tables, and indices;
- low-rank, sparse, static, conditional, capability-bank, and exception corrections;
- gates and routing metadata;
- tokenizer/model files when candidate-specific;
- shape tables, alignment/padding, manifests, and checksums; and
- any generated table that must persist for future loads.

The receipt must report, separately:

1. **body bpw** for comparison with papers that report only compressed linear layers;
2. **physical whole-artifact bpw** using actual file lengths;
3. **total installed bytes** including optional capability slices;
4. **resident bytes** after load, including expanded tables and allocator overhead;
5. **active bytes per token** for conditional/MoE paths, with mean, p95, and worst case;
6. **runtime state bytes** for KV, activations, workspaces, caches, and generated tiles;
7. **external-system bytes and calls** for `augmented_system`; and
8. **creation peak RSS, scratch bytes, and checkpoint bytes**, which do not change bpw but determine
   whether the experiment is executable and resumable.

For MoE, both total parameters and expected active parameters per token are reported. Installed bpw
always uses total source parameters; active-parameter density is a separate metric.

The Studio has 96 GiB unified memory, but the shared planning envelope reserves approximately 78 GB
for weights so the OS, control plane, KV, activations, and workspaces have room. That 78 GB is a
provisional safety policy, not evidence of fit. Decimal GB and binary GiB must never be mixed.
Resident admission requires measured normal memory pressure, zero swap, and an explicit worst-case
context/workspace receipt.

Illustrative payload-only lower bounds show the research difficulty:

| Nominal parameters | 0.80 bpw | 0.50 bpw | 0.33 bpw | 0.25 bpw | 0.10 bpw |
|---:|---:|---:|---:|---:|---:|
| 120B | 12.0 GB | 7.5 GB | 4.95 GB | 3.75 GB | 1.5 GB |
| 405B | 40.5 GB | 25.3 GB | 16.7 GB | 12.7 GB | 5.1 GB |
| 671B | 67.1 GB | 41.9 GB | 27.7 GB | 21.0 GB | 8.4 GB |
| 1.6T | 160 GB | 100 GB | 66 GB | 50 GB | 20 GB |

These are mathematical payload figures, not projected Hawking artifacts. A 1.6T resident candidate
needs approximately 0.39 bpw or less merely to place payload inside 78 GB, and a meaningfully lower
physical rate once metadata, lexical tensors, corrections, and runtime state are included. No cited
work currently demonstrates modern broad capability at that whole-artifact rate.

### 3.3 Representation floors and optimization limits

A dual-path binary factorization with rank (r), input width (n), output width (m), binary left
and right factors, and 32-bit row/column/latent scales has an approximate storage term

\[
B \approx 2r(m+n)+32(m+n)+32r \quad \text{bits},
\]

before alignment, codebooks, corrections, and untouched tensors. For small or narrow matrices, the
scale floor alone may exceed the target. A nominal global rate may therefore imply zero feasible
rank for GQA/KV, routers, or lexical projections.

ITQ, binary factorization, K-means/codebook fitting, ADMM, alternating SVD, mixed-precision search,
and global representation allocation are nonconvex or combinatorial. They do not provide a global
quality guarantee. Multiple seeds/restarts, zero-treatment controls, and held-out selection are
mandatory. A negative result is retained rather than optimized out of the record.

### 3.4 Capability rate-distortion, not weight rate-distortion

The target is a protected capability vector (C=(C_1,\ldots,C_K)). The search objective is closer to

\[
\max_x\;\min_k \operatorname{LCB}(C_k(x)-C_k(parent))
\]

subject to physical bytes, claim-lane provenance, and non-inferiority margins. Weight MSE and
perplexity remain useful early-fidelity predictors, but the campaign must measure how biased those
predictors are before using them for promotion.

---

## 4. Four canonical claim lanes

Every cell declares exactly one lane before data, teachers, treatments, and evaluators are selected.
Results cannot migrate between lanes after evaluation.

| Lane | Permitted information and treatment | Runtime boundary | Valid claim | Mandatory controls |
|---|---|---|---|---|
| `codec_fidelity` | Exact parent tensors, architecture metadata, declared calibration used to fit the representation; no behavioral restoration curriculum, stronger teacher, retrieval, tool, verifier, or external model. | Standalone packed artifact only. | Fidelity attributable to the representation/codec itself. | Parent, untreated same-rate codec, scalar/equal-byte codec, zero correction. |
| `restorative_training` | Exact parent is the only behavioral teacher; codec-native QAT, parent distillation, repair training, and failure replay may restore parent behavior. | Standalone artifact only; zero external runtime bytes. | Parent capability restored under the physical budget. Improvement over the parent is not called elevation. | Parent, codec-fidelity arm, zero treatment, BF16 parent receiving the same data/optimization, equal-byte public competitor. |
| `capability_elevation` | Provenance-bound stronger teachers and truth oracles may teach verified behavior beyond the parent. Parent identity must remain a separate reference. | Standalone artifact only; all teacher dependence ends at training. | The artifact itself gains capability, with uplift causally separated from compression recovery. | Parent, restorative arm, BF16 model receiving the same elevation treatment, teacher-free and parent-only ablations. |
| `augmented_system` | Retrieval, tools, verifiers, external state, or additional models are allowed with source and contamination controls. | Full system bill: installed/resident bytes, calls, tokens, retries, network and test-time compute. | The declared system gains capability. It cannot support a standalone-model claim. | Closed-book standalone artifact, same augmentation applied to parent and same-budget competitors, zero-augmentation ablation. |

Test-time compute is matched within each comparison: output-token limit, samples per item, verifier
retries, retrieval calls, tool calls, and external-model calls. If a lane cannot match those budgets,
the comparison is invalid rather than adjusted after seeing scores.

---

## 5. Diagnostic router: signal degradation versus computation collapse

The Doctor may not prescribe a repair until it has classified the failure. Thresholds and probes are
preregistered per architecture and rate; the labels are evidence-backed diagnoses, not metaphors.

### 5.1 Required probes

- paired parent/candidate token probabilities and tail mass;
- layerwise hidden-state similarity, CKA/Procrustes alignment, rank, and feature survival;
- attention maps, FFN outputs, router logits, expert choices, and KV-sensitive traces;
- early/middle/late activation replacement and weight/block patching;
- linear and low-rank readout oracles trained only on diagnostic splits;
- causal effect of restoring one tensor, block, expert, or capability island;
- parent-correct/candidate-wrong task traces by capability and severity;
- evaluator sanity checks, chat-template identity, prompt variance, and scorer agreement; and
- confidence that observed absence is model damage rather than data/scorer artifact.

### 5.2 Routing rule

| Diagnosis | Evidence signature | Allowed first treatment | Forbidden inference |
|---|---|---|---|
| `no_material_damage` | Parent-relative differences stay inside preregistered noise and capability margins. | Zero treatment remains champion; package and evaluate. | Adding a repair because budget remains. |
| `signal_degradation` | Parent-relevant geometry and causal features survive; denoising, patching, or a bounded correction reliably recovers outputs. | Static bias/rank/sparse repair, multi-envelope correction, capability bank, or jointly trained conditional compensator. | That local repair will scale without full-model validation. |
| `computation_collapse` | Required feature rank, routing, attention behavior, or algorithmic trace is absent; bounded compensation cannot recover it. | Representation reset, higher-rate island, codec-native QAT/reconstruction, progressive retraining, or close the rate as infeasible. | Compensation-only “healing,” retrieval-backed core claims, or perplexity-only promotion. |
| `mixed_failure` | Some layers/capabilities preserve signal while others collapse. | Structural reset for collapsed components plus independently ablated repair for degraded components. | One global label or one uniform treatment. |
| `undetermined` | Probes disagree or have insufficient power. | Gather discriminating evidence; retain both treatment branches. | Promotion based on the cheaper story. |

Activation patch recovery alone is insufficient: a full-precision patch may inject the missing
computation. The Doctor must test whether the proposed **budgeted** treatment can reach the same
causal subspace. Collapse at one protected capability prevents a compensation-only program even if
mean perplexity improves.

---

## 6. Pass one — v3 Representation Foundry

### 6.1 Purpose

The v3 pass expands the search from “choose bits for existing matrix multiplication” to “choose the
most information-efficient representation for each semantic computation.” It is a foundry of
equal-byte representations, not a tournament whose early winner is presumed final.

### 6.2 Concrete additions

| ID | Addition | Concrete mechanism | Required output and rejection gate |
|---|---|---|---|
| V3.1 | Whole-artifact accountant | Enumerate tensor ownership and every output file; compute body, physical, resident, active, scratch, and external ledgers from actual bytes. | Byte-complete manifest or candidate is invalid. |
| V3.2 | Tensor phenotype atlas | Per semantic group measure spectral-tail decay, effective rank, coherence, kurtosis, bimodality, entropy, Hessian/activation sensitivity, activation covariance, GQA role, router/expert frequency, and lexical role. | Hash-bound atlas plus uncertainty; no global format selected without it. |
| V3.3 | Equal-byte representation tournament | Uniform scalar PTQ, lattice/additive/vector codes, learned binary factors, pattern codebooks, progressive/nested codes, mixed-precision islands, sparse exceptions, and representation-reset controls. | Every applicable family reaches a tensor/block oracle at identical physical projection; missing family requires incompatibility receipt. |
| V3.4 | Entropy-shaped binary latent codec | Joint-ITQ geometry; multi-envelope amplitudes; entropy/codebook-compressed factor signs; optional residual; lossless rANS wrapper. Optimize capability proxy plus **all** metadata bytes. | Must beat plain binary factors and independent entropy coding. If factor entropy stays maximal, reject pattern compression. |
| V3.5 | Adaptive rank/envelope allocator | Allocate sign rank, magnitude-envelope rank, codebook size, group width, exception count, and precision by tensor phenotype under a global physical knapsack. | Uniform and hand-tuned controls retained; report solver regret and restart spread. |
| V3.6 | Progressive semantic bitplanes | Co-train physical prefixes at 4, 3, 2, 1, 0.8, 0.55, 0.5, 0.33, 0.25, and 0.1 bpw where feasible. Optimize every prefix, not arbitrary truncation. | Each prefix has its own packed-byte and capability receipt; a failed prefix is a negative control, not silently skipped. |
| V3.7 | Lexical-boundary foundry | Separate embedding/LM-head treatment: tied semantic/product codes, frequency-aware subspaces, rare-token exceptions, multilingual/morphological groups, and shared latent bases. | Whole-artifact rate must improve without rare-token, number, code-symbol, or multilingual collapse. |
| V3.8 | Hierarchical MoE representation | Shared cross-expert KLT/SVD bases, expert-delta VQ/binary factors, rare-expert minimum coverage, router/shared-expert protection, gate-weighted curvature, critical-layer islands. | Total-model bytes and active bytes both reported; expert-only bpw is inadmissible. |
| V3.9 | Repairability-aware base | Jointly choose condensed representation and a future correction subspace; penalize residual energy orthogonal to every allowed repair basis. | Compare minimum reconstruction, minimum output KL, and maximum repairability objectives at equal bytes. |
| V3.10 | Codec-native reconstruction | Train and reconstruct through exact packing, rounding, codebook lookup, entropy layout, and decoder semantics rather than a float proxy. | Training decoder and packed decoder must agree bit-for-bit or within a preregistered numerical contract. |
| V3.11 | Transactional giant-model path | Immutable shard map; bounded decode window; deterministic global-statistics passes; pack, round-trip, hash, fsync, checkpoint, release. | Exact resume from source shard/byte offset and bit-identical replay before any 70B+ mutating campaign. |
| V3.12 | Multi-fidelity predictor audit | F0 analytic → F1 tensor → F2 block/shard → F3 full model; learn the bias and uncertainty of local proxies. | No F0–F2 metric may prune a family until its false-negative rate against F3 is estimated. |

The composite in V3.4 is Hawking-original synthesis. No cited source validates the combined mechanism
or guarantees that factor signs remain compressible after geometry optimization. It must therefore
enter as a hypothesis beside simpler controls, not as the default.

### 6.3 Representation proposal evaluation

Let (P) be source weights, (T) calibration/training tokens, (r) trial rank, (I) fitting
iterations, (E) experts, (G) tensor groups, (K) candidate encodings, and (N) evaluation items.
Complexities are asymptotic guides; actual memory traffic and Apple execution remain unmeasured.

| Proposal / horizon | Theoretical work | Expected artifact/bandwidth effect | Latency expectation | Difficulty and compatibility | Interactions |
|---|---|---|---|---|---|
| Exact accountant / immediate | (O(F+G)) metadata plus file hashing (O(bytes)). | No compression; prevents false savings. | No serving claim. | Medium; backend-neutral on GPU, Apple, distributed, and future hardware. | Mandatory for every quantized, speculative, routed, or distributed artifact. |
| Phenotype atlas / immediate | (O(P)) statistics; randomized spectral and curvature sketches roughly (O(IPr)+O(TP_{active})). | No direct reduction; avoids spending bytes on insensitive regions. | None claimed. | Medium-high; CPU/Metal feasible at small scale, CUDA/distributed for large. | Architecture adapters extend it to MoE, attention, SSM, and multimodal blocks. |
| Binary factors + ITQ/envelopes / medium | Alternating discrete/low-rank optimization roughly (O(IPr)), nonconvex. | Target payload can enter sub-bit regimes; scale/envelope floors and lexical bytes may dominate. | Unknown until native fused decoder; indirection may lose. | Very high; research GPU practical, Apple decoder/QAT absent, future bitwise/near-memory hardware favorable. | Can form nested drafter/target prefixes; blocks and experts distribute independently with deterministic merges. |
| Pattern grammar + entropy coding / medium | Approximate clustering (O(IMCv)); entropy coding (O(P)). | Factor cost may approach \(\log_2 C/v\) only if measured entropy is low; no guaranteed gain. | Random access and decode overhead may increase latency. | Very high; generic CPU/GPU offline, Apple runtime needs tables; future hardware can fuse lookup/ANS. | Orthogonal to quantization; speculation requires independently addressable target slices. |
| Progressive semantic bitplanes / medium | Multi-prefix QAT (O(JTP_{active})) for (J) prefixes, partially shared. | One artifact can expose multiple physical quality points if prefixes are truly nested. | Speed deferred; low prefix might later draft, but acceptance is unknown. | Very high; CUDA research first, Apple packing/decoder missing, future hierarchical memory favorable. | Must co-train speculation; distributed workers need prefix-consistent checkpoints. |
| Lexical-boundary codec / medium | Product/codebook fitting (O(PT)) or approximate (O(IMCv)), depending on family. | Can remove the whole-artifact floor seen at very low body bpw. | Vocabulary lookup can add indirection; unmeasured. | High; compatible with generic GPUs/Apple, specialized embedding hardware favorable. | Quantization must preserve tied weights; speculation is highly sensitive to token-probability tails. |
| Hierarchical MoE codec / medium | Shared-basis extraction roughly (O(IPr)); calibration (O(TP_{active})); allocation combinatorial. | Shared bases amortize across experts; total savings depend on router/non-expert bill. | Active movement may fall with locality; cold-expert misses may worsen tails. | Very high; CUDA references exist, Apple out-of-core runtime absent, distributed/future expert stores favorable. | Requires rare-expert calibration, router fidelity, and separate total/active accounting. |
| Repairability-aware base / medium | Alternating representation/projector fitting roughly (O(I(Pr+TP_{active}))). | May spend the same base bytes but reduce correction bytes required for capability. | Static correction cost unknown; conditional paths add tails. | Very high; generic training hardware compatible, Apple research path absent. | Quantizer and Doctor must be co-optimized; distributed correction bases require deterministic identity. |
| Global allocator / immediate→medium | Multiple-choice knapsack pseudo-polynomial in byte quanta or Lagrangian search about (O(GK\log G)); no global guarantee for learned objectives. | Improves marginal capability per exact byte if predictors are calibrated. | Kernel-fragmentation risk can erase theoretical movement gains. | Medium-high; planner backend-neutral, deployment constraints backend-specific. | Jointly allocates quantization, correction, speculation-critical islands, and expert formats. |
| Transactional shard transform / immediate | (O(P)) streaming per pass plus explicit global reductions. | No intrinsic size gain; enables otherwise impossible-scale research. | Offline only. | High; POSIX path works in principle, Apple/CUDA compute adapters separate, distributed natural. | Exact resume is mandatory for multi-week QAT and remote source lifecycle. |

Payload ratios in this table are hypotheses until a packed file exists. No v3 proposal earns a speed,
bandwidth, latency, Apple, or distributed-performance claim.

### 6.4 v3 primary-source frontier and evidence ceilings

| Primary source | Useful mechanism | Reported ceiling relevant to v3 | What v3 must not infer |
|---|---|---|---|
| [ParetoQ](https://arxiv.org/abs/2502.02631) | Unified 1–4-bit QAT and representation transition. | Strong conceptual evidence; primarily smaller-model regime. | That its phase boundary guarantees a particular Hawking codec. |
| [LittleBit](https://arxiv.org/abs/2506.13771) | Dual low-rank binary factors, scaling, residual paths. | Experiments through 32B; lowest body rates have large quality loss; 70B is projection rather than demonstrated QAT. | That 0.1 body-bpw is a high-fidelity whole model. |
| [LittleBit-2](https://arxiv.org/abs/2603.00042) | Joint-ITQ and spectral geometry. | Through 27B at 0.1 body-bpw; lexical tensors can raise whole-model rate dramatically. | That “functional” means parent fidelity. |
| [NanoQuant](https://arxiv.org/abs/2602.06694) | Hessian-aware binary-factor PTQ, ADMM initialization, scale-only teacher KL. | PTQ through 70B; 0.55-bpw quality remains far from BF16. | That PTQ alone solves sub-bit capability. |
| [BTC-LLM](https://arxiv.org/abs/2506.12040) | Pattern clustering, invertible transforms, Hamming codebooks. | Through 65B around 0.7 bpw; codebook overhead is material. | That clustered patterns are free or <0.5-bpw proven. |
| [DBF](https://arxiv.org/abs/2505.11076) | Fine-grained dual binary factors. | Full-model evidence centered on 7–8B; larger matrices are error studies. | That matrix reconstruction at 405B is a model result. |
| [Multi-envelope DBF](https://arxiv.org/abs/2512.24545) | Multiple magnitude envelopes rather than only residual sign paths. | Roughly 0.6–8B and 1–1.5 bits, with protected layers and difficult 1-bit cases. | That envelopes remove calibration overfit or collapse. |
| [LC-QAT](https://arxiv.org/abs/2606.10531) | Differentiable lookup-free vector-codebook QAT. | Strong 2-bit evidence mainly through 8B; limited 14B appendix. | That two-bit QAT scales directly to sub-bit or trillion scale. |
| [BWLA](https://arxiv.org/abs/2605.00422) | Bimodal transforms plus low-rank approximation. | Through 70B near 1.15–1.19 actual weight bpw; reasoning gaps remain. | That “one-bit” equals one physical whole-model bit. |
| [MatQuant](https://arxiv.org/abs/2502.06786) | Nested 8/4/2-bit training. | Useful progressive control, not sub-bit evidence. | That arbitrary bit prefixes work without co-training. |
| [Bit-by-Bit](https://arxiv.org/abs/2604.07888) | Progressive QAT and rounding-aware outlier splitting. | Primarily through 8B with 14B reasoning appendix. | That outlier duplication saves physical bytes after metadata. |
| [Shannon-bound model compression](https://arxiv.org/abs/2606.15789) | Tile-level rANS for lossless compression of quantized symbols. | Tested through 405B and multiple input formats. | That marginal entropy predicts capability or random-access runtime. |

### 6.5 v3 exit condition

V3 finishes when the representation search space, physical accountant, diagnostic atlas, equal-byte
controls, and multi-fidelity bias study are complete enough to reject families honestly. It does not
require a sub-bit winner. A documented physical or quality impossibility is a valid v3 result.

---

## 7. Mandatory rest/audit A — representation red team

This is a real stop in hypothesis generation. Freeze source hashes, candidate manifests, negative
results, selection rules, and the v3 assumption register. During the audit interval:

- no new representation is added to rescue a favorite result;
- no candidate is retrained, retuned, or evaluated on the sealed/final data;
- a fresh review reconstructs every physical byte from the artifact manifest;
- advertised and whole-artifact bpw are reconciled;
- equal-byte and smaller/higher-precision controls are checked for omissions;
- local oracle promotion false negatives and false positives are reviewed;
- lexical, GQA, router, shared-expert, rare-expert, and first/last-layer exceptions are exposed;
- all extrapolations beyond a paper’s demonstrated scale are relabeled as hypotheses;
- multiple seeds/restarts and failed representations remain visible; and
- each v3 assumption is marked **accept**, **narrow**, **reject**, or **undetermined** with evidence.

Audit A produces a frozen survivor set and a rejected-family register. V4 may reopen a rejected
family only with a new falsifiable mechanism and a distinct lineage; it may not relabel the same
failed result.

Promotion to v4 requires:

1. actual-byte authority and complete tensor ownership;
2. at least one zero-treatment and one strong equal-byte representation control;
3. explicit diagnosis confidence and unresolved alternatives;
4. no core claim supported by external runtime information; and
5. an audit receipt stating that v3 foundry evidence is local/structural, not capability dominance.

---

## 8. Pass two — v4 Capability Restoration Lab

### 8.1 Purpose

V4 asks what information survived, what computation collapsed, and what smallest causal intervention
restores or elevates verified capability. It expands the Doctor from static residual fitting into a
capability laboratory. The four claim lanes remain isolated throughout.

### 8.2 Concrete additions

| ID | Addition | Concrete mechanism | Required output and rejection gate |
|---|---|---|---|
| V4.1 | Mechanistic disease atlas | Join representation phenotype, layerwise propagation, activation/weight patches, routing drift, feature-rank loss, and parent-correct/candidate-wrong traces. | Per capability/component diagnosis with confidence; `undetermined` retains competing routes. |
| V4.2 | Repairability-aware condensation loop | Generalize [ProjQ](https://arxiv.org/abs/2606.00494): shape quantization residual into static, sparse, codebook, capability-bank, or conditional correction bases during representation fitting. | Equal-byte comparison against independently fitted base+repair; correction bytes fully billed. |
| V4.3 | Counterfactual residual controller | Evaluate no residual, bias, one/multiple sign paths, amplitude envelopes, variable rank, sparse exceptions, protected precision, and conditional repair; learn signed/zero strength. | Zero treatment remains eligible. Intervention-on/off evidence must show causal recovery without collateral regression. |
| V4.4 | Capability-bank Doctor | Small structured corrections specialized by capability and component, with conflict-aware shared and private bases. | Installed and active bytes reported; bank must beat one static equal-byte repair and show no protected-domain forgetting. |
| V4.5 | Jointly trained syndrome gate | Predict token/layer/expert risk only where an oracle proves predictable treatment benefit; train gate and correction together. | Severity-weighted false-negative bound, p95/worst activation rate, static-protection fallback, and gate ablation. |
| V4.6 | Active failure foundry | Mine parent-correct/candidate-wrong cases, verified teacher gaps, causal signatures, and oracle-preserving adversarial mutations; split mutation families before generation. | Truth oracle or quarantined disagreement; generator cannot see selection/sealed sets. |
| V4.7 | Verified reasoning restoration | Use executable/formal verification, mistake and backtracking traces, student rollouts, on-policy selective imitation, parent-good replay, and no-RL controls. | Invalid but answer-correct traces rejected; reasoning, knowledge, and tool dependence separated. |
| V4.8 | Distribution and geometry distillation | Combine token CE/KL, top-k/tail mass, hidden/CKA geometry, attention/router agreement, and causally weighted representation losses. | Objective ablations and gradient-conflict measurement; no proxy promoted merely for lowering training loss. |
| V4.9 | Lexical and long-tail restoration | Rare-token, name, number, code-symbol, multilingual, and tied-head replay with frequency-stratified evaluation. | Tail capability improves on unseen examples; memorized lexical exceptions are billed and contamination-scanned. |
| V4.10 | Rare-expert and routing restoration | Expert-balanced sampling, gate-affinity curvature, router KL/choice agreement, minimum rare-expert examples, expert-delta correction. | Rare-expert lower bounds and worst-expert diagnostics; average router agreement cannot hide cold-expert collapse. |
| V4.11 | Teacher tribunal | Exact parent, stronger teacher, truth oracle, and self-generated traces have separate provenance, authority, and lane permissions. | Any unverified disagreement is quarantined; BF16 same-treatment control identifies training uplift unrelated to condensation. |
| V4.12 | Forgetting and collateral monitor | Parent-good replay, capability gradient-conflict matrix, temporal scorecards, and rollback to best hash-bound checkpoint. | Any protected-domain lower bound below margin blocks treatment, even when target capability improves. |

### 8.3 Restoration proposal evaluation

| Proposal / horizon | Theoretical work | Expected artifact/bandwidth effect | Latency expectation | Difficulty and compatibility | Quantization, speculation, distributed, and future interaction |
|---|---|---|---|---|---|
| Static low-rank/sparse repair / immediate | Offline SVD/fitting roughly (O(Pr))–(O(IPr)); runtime narrow GEMV/gather. | Adds explicit bytes but may dominate spending them in the base. | Unmeasured; sparse indirection can erase movement savings. | Medium-high; generic GPU, CPU, and Metal primitives possible; future fused epilogues favorable. | Codec-agnostic if ABI is exact; speculation requires target/drafter identity; tensors shard naturally. |
| Counterfactual controller / immediate | Number of treatment arms times F1–F3 cost. | Prevents harmful correction and finds marginal capability/byte. | No independent speed claim. | Medium; backend-neutral offline controller. | Quantization family is an input; same protocol applies to MoE/SSM/multimodal components. |
| Repairability-aware joint fitting / medium | Alternating nonconvex (O(I(Pr+TP_{active}))). | Same base rate may need fewer correction bytes; effect empirical. | Depends on chosen repair; conditional tails separately billed. | Very high; CUDA research likely first, Apple fitting path absent, future co-designed codec hardware favorable. | Must co-train nested prefixes and any speculation-sensitive distribution. |
| Capability bank / medium | Multi-task training (O(TP_{active})) plus low-rank bank fitting; gradient conflicts measured. | Installed bytes grow with private bases; sharing may amortize. | Conditional loading can reduce mean movement but increases p95/miss risk. | High; generic accelerator training, Apple mmap/Metal runtime unproven. | Natural per-capability distributed experts; future persistent-state architectures may share bases. |
| Syndrome-gated compensation / medium | Diagnostic oracle plus gate/correction training (O(TP_{active})). | Reduces mean active correction bytes only if gate is accurate. | Gate and cache misses may worsen tails; no claim before measurement. | Very high; [SPEAR](https://arxiv.org/abs/2606.11244) is a four-bit precedent, not sub-bit proof. | Must train with quantizer; speculative acceptance and distributed routing become part of identity. |
| Active failure foundry / immediate→medium | Teacher/student generation (O(N\cdot TTC)), verification task-dependent. | No direct artifact reduction; directs limited repair bytes toward high-value failures. | Offline. | High; backend-neutral orchestration, verifier availability domain-specific. | Supports every representation; data provenance and split firewalls dominate distributed design. |
| Verified reasoning restoration / medium | Supervised/on-policy training (O(TP_{active})) plus verification. | May require more learned correction bytes or base rewrite; no free capability. | Standalone inference unchanged only if all learning is compiled into artifact. | Very high; accelerator training needed, Apple local scale limited, distributed training favorable. | Stronger-teacher/truth data moves result to `capability_elevation`; tools at runtime move it to `augmented_system`. |
| MoE routing/expert repair / medium | (O(TP_{active})) calibration/training plus expert-group fitting. | Shared correction bases may amortize; cold-expert protections add installed bytes. | Locality may help mean traffic; worst expert-transfer tail remains. | Very high; CUDA MoE references exist, Apple/distributed runtime proof absent. | Quantizer, router, expert cache, and any speculative verifier must be jointly identified. |

### 8.4 Capability sources informing v4

- [ProjQ](https://arxiv.org/abs/2606.00494) supplies the important principle that the quantization
  residual should be made correctable rather than merely small. Its evidence reaches roughly 32B
  and two to four bits, not sub-bit restoration.
- [SPEAR](https://arxiv.org/abs/2606.11244) motivates token-dependent error compensation at four
  bits. It does not prove that a gate can recover computation destroyed at sub-bit rates.
- [MARR](https://arxiv.org/abs/2605.17997) motivates module-specific residual strength and the
  possibility that nominal reconstruction adds bias.
- [MoEQuant](https://arxiv.org/abs/2505.03804) motivates expert-balanced calibration and
  gate-affinity-weighted curvature at three to four bits.
- [KBVQ-MoE](https://arxiv.org/abs/2602.11184) motivates shared cross-expert bases plus vector-coded
  expert deltas, with evidence around 47B and roughly 2.08–2.2 physical bits.
- [AlphaQ](https://arxiv.org/abs/2606.04980) motivates global expert bit allocation, while also
  illustrating why expert-average bits cannot be presented as total-model bits.
- [Sparse-BitNet](https://arxiv.org/abs/2603.05168) motivates native joint ternary/sparse training for
  future architectures; its evidence through 3B does not make it an existing-checkpoint conversion
  shortcut.

### 8.5 v4 exit condition

V4 finishes when each surviving representation has a diagnosis-backed treatment tournament, each
claim lane has valid controls and data provenance, and full-model replicated evidence exists for the
scale at which a candidate is being promoted. The exit may be “no repair can restore this rate under
the budget.” V4 does not issue an unbeatable claim.

---

## 9. Mandatory rest/audit B — capability, data, and causal red team

Freeze candidate hashes, objectives, failure generators, data manifests, teacher caches, benchmark
protocols, non-inferiority margins, and comparator set. Stop treatment search. The audit is conducted
from per-item outputs rather than aggregate dashboards.

### 9.1 Causal red team

- Verify that treatment-on repairs the intended failure and treatment-off removes the gain.
- Compare with an equal-byte random/static repair and a BF16 model receiving the same treatment.
- Test whether the base representation, not the Doctor, accounts for the gain.
- Test whether teacher uplift, retrieval, test-time compute, or memorized exceptions leaked into a
  restoration claim.
- Re-run activation/weight patches to distinguish denoising from structural reconstruction.
- Inspect gradient conflict and protected-domain forgetting throughout training, not only at the best
  checkpoint.
- Check conditional-gate false negatives on severe failures and p95/worst routing cost.
- Reclassify failures as mixed/undetermined when probes disagree; do not force a clean narrative.

### 9.2 Benchmark and evaluator red team

- Candidate labels are blinded; parent and competitors receive identical prompts, chat templates,
  stopping rules, seeds where applicable, and test-time compute.
- Prefer executable, exact-match, or formal scorers. An LLM judge is used only when no objective
  scorer exists and receives order randomization, calibration examples, and judge-variance analysis.
- Measure prompt-format sensitivity, answer-extraction failure, refusal versus inability, verbosity
  bias, and invalid-but-answer-correct traces.
- Evaluate natural, adversarial, metamorphic, compositional, and post-training-cutoff examples.
- Keep paired per-item outputs so a macro score cannot hide systematic swaps in capability.
- Require the exact production tokenizer and chat template; raw completion against an instruct
  parent is not a valid capability comparison.

### 9.3 Data firewall

The following split identities are mutually hash-distinct:

1. calibration;
2. reconstruction training;
3. repair training;
4. treatment search;
5. selection;
6. public validation;
7. shadow;
8. frozen final;
9. sealed final; and
10. independent replication.

Exact duplicate, near-duplicate, semantic contamination, benchmark-solution, and mutation-family
leakage scans are mandatory. Teacher output caches are split-bound. Retrieval indexes exclude all
evaluation content. Shadow, frozen, sealed, and independent sets are optimizer-inaccessible. The
sealed set is revealed only after the lineage is frozen and consumed once.

Audit B either promotes a frozen set of causal treatments into v5 or records a complete negative. A
favorite candidate does not survive a data-firewall, causal, or protected-domain failure.

---

## 10. Pass three — v5 Adversarial Proof Synthesis

### 10.1 Purpose

V5 converts the surviving research into falsifiable claims. It does not add an unrestricted new
mechanism after sealed evaluation begins. It freezes lineages, packs physical artifacts, reproduces
competitors, and asks an adversarial intersection-union question: is the candidate non-inferior to
the relevant parent in every protected domain and uniformly better than every applicable
same-budget competitor?

### 10.2 Concrete additions

| ID | Addition | Required proof |
|---|---|---|
| V5.1 | Canonical four-lane program identity | Model revision, config, tokenizer, chat template, rate, lane, data, teacher, operators, seeds, and source hashes form one immutable identity. |
| V5.2 | Byte-complete artifact seal | Actual-file manifest, all tensor ownership, dense-parent-fallback prohibition, round-trip decode, checksum, physical/resident/runtime ledgers. |
| V5.3 | Frozen competitor registry | Every applicable family reproduced on the same parent, no greater physical bytes, same data/teacher authority, same prompt/scorer, and same augmentation scope. |
| V5.4 | Preregistered capability contract | Domain metrics, margins, minimum effect sizes, sample sizes, exclusions, stopping rules, multiple-testing family, and dominance decision frozen before final. |
| V5.5 | Sealed champion evaluation | Paired per-item outputs across every domain, multiple seeds and calibration draws, one-time sealed consumption, complete failures retained. |
| V5.6 | Independent reproduction | Separate environment/reviewer reconstructs the artifact and competitor result from source-bound manifests without candidate-selection access. |
| V5.7 | Claim compiler | Generates only lane-valid statements: codec fidelity, parent restoration, standalone elevation, or augmented-system capability. Unsupported adjectives are rejected. |
| V5.8 | Scale-transfer proof chain | 7/14/32B mechanism evidence precedes 70/120/235B; frontier 284/405/671B and 1T–1.6T remain separate out-of-core cells with their own proof. |
| V5.9 | Negative frontier ledger | Failed rates, representations, repairs, seeds, architectures, and scale transfers remain queryable and affect future priors. |
| V5.10 | Final adversarial audit | Recompute decision from raw per-item outputs and raw file lengths; verify no cross-lane evidence, test-time compute mismatch, hidden exception, or post hoc comparator change. |

### 10.3 Training Ladder v5 alignment

The unified ladder uses physical target ceilings:

`4.0 → 3.0 → 2.0 → 1.0 → 0.8 → 0.55 → 0.50 → 0.33 → 0.25 → 0.10 bpw`.

Every model/rate has four distinct claim-lane cells. A rate is not skipped merely because it is
unlikely; it may be closed cheaply with an analytic impossibility or destructive control. The 0.10
rate is a stress/control point unless capability evidence proves otherwise.

Resource classes are:

- **resident parent research:** roughly ≤16B, subject to measured peak memory;
- **streamed single-host research:** >16B through approximately 235B, with block/shard windows; and
- **frontier out-of-core research:** >235B, with transactional multi-pass source and output shards.

The early parameter tiers should establish mechanism invariants and proxy calibration, not monopolize
the final proof. Parallelism is allowed only for independently checkpointed cells whose measured
combined peak fits the host with normal pressure and zero swap. One giant streamed/out-of-core cell
occupies the heavy-work lease by itself.

Scale promotion requires all of the following:

1. exact source and architecture adapter;
2. whole-artifact feasibility at the destination scale;
3. stable tensor-phenotype and failure-class predictions with uncertainty;
4. no regression in local-to-full-model predictor validity;
5. capability evaluation at the destination scale; and
6. exact resume before mutating work.

No 7B–32B result licenses a 671B or 1.6T quality claim.

---

## 11. Capability, benchmark, and statistical proof contract

### 11.1 Protected capability domains

Every full candidate is evaluated on all twelve domains:

1. language modeling;
2. knowledge;
3. reasoning;
4. mathematics;
5. science;
6. coding;
7. instruction following;
8. long context;
9. multilingual capability;
10. tool use;
11. calibration; and
12. safety/security.

The exact suite is versioned and frozen, but it must contain multiple independent task families per
high-risk domain. Suitable primary benchmark references include
[MMLU-Pro](https://arxiv.org/abs/2406.01574),
[GPQA](https://arxiv.org/abs/2311.12022),
[MATH](https://arxiv.org/abs/2103.03874),
[LiveCodeBench](https://arxiv.org/abs/2403.07974),
[IFEval](https://arxiv.org/abs/2311.07911),
[LongBench](https://arxiv.org/abs/2308.14508),
[RULER](https://arxiv.org/abs/2404.06654),
[MGSM](https://arxiv.org/abs/2210.03057),
[Berkeley Function-Calling Leaderboard](https://arxiv.org/abs/2402.18679),
[HarmBench](https://arxiv.org/abs/2402.04249), and
[XSTest](https://arxiv.org/abs/2308.01263). Selection must account for license, contamination risk,
model release date, and architecture. Named benchmarks are candidates, not permission to train on
their items.

Minimum diagnostic views include:

- held-out multiwindow perplexity and token KL;
- exact answer/compile/test pass rates where objective scoring exists;
- paired win/loss/tie and severity for parent-correct/candidate-wrong cases;
- confidence calibration, Brier/ECE, abstention, and selective-risk curves;
- lexical frequency, language, context length, expert frequency, and task-difficulty strata;
- routing agreement and causal internal geometry for MoE/conditional models;
- robustness to format, paraphrase, distractor, logically invariant, and adversarial mutations; and
- worst-domain, worst-stratum, macro, and physical-capability-density summaries.

### 11.2 Statistical red team

Before frozen-final evaluation, preregister:

- metric direction and aggregation per domain;
- parent-relative non-inferiority margin per domain;
- competitor superiority delta and smallest effect of interest;
- power analysis and minimum item/seed/calibration-draw counts;
- exclusions, parser failures, missing outputs, and retry policy;
- paired bootstrap, randomization, or hierarchical model appropriate to each metric;
- 95% confidence intervals and Holm familywise correction at alpha ≤0.05;
- candidate selection and early-stopping rules using selection data only; and
- the complete comparator registry.

At least five independent training/fitting seeds and multiple calibration draws are required for a
replicated quality claim. Report the distribution and worst valid run, not just the best seed.
Hierarchical or stratified intervals are required when items are clustered by benchmark, language,
expert, or mutation family.

The dominance decision is an intersection-union rule:

1. candidate lower confidence bound is above the negative non-inferiority margin in **every**
   protected domain relative to the appropriate parent/control;
2. for every same-budget competitor, macro capability delta lower bound is positive;
3. no competitor comparison has a protected-domain lower bound below that domain’s margin;
4. at least one preregistered domain is strictly superior by the declared effect; and
5. the same verdict is reached by independent reproduction.

The stronger phrase **uniform quality dominance** requires positive lower bounds in every protected
domain against every required same-budget competitor. If only the weaker frontier-champion rule
passes, v5 must say so. If any condition fails, the emitted claim is **unproven; no dominant or
unbeatable claim permitted**.

---

## 12. Required competitor set

Competitors are frozen before final evaluation and are mandatory when architecture and source access
make them applicable.

### 12.1 Causal and budget controls

- exact full-precision parent;
- untreated candidate at the same representation and rate;
- zero-correction candidate;
- uniform scalar PTQ at the same physical bytes;
- codec-native QAT at the same physical bytes;
- progressive inherited and fresh representation-reset arms below two bits;
- BF16 parent receiving the same restoration/elevation data and optimizer;
- smaller higher-precision model at the same or lower resident/physical bytes;
- Hawking’s prior source-bound champion; and
- same artifact with augmentation disabled for `augmented_system`.

### 12.2 Public mechanism families

- scalar AWQ/GPTQ control, including [AWQ](https://arxiv.org/abs/2306.00978);
- lattice/incoherence control such as [QuIP#](https://arxiv.org/abs/2402.04396);
- additive codebook control such as [AQLM](https://arxiv.org/abs/2401.06118);
- differentiable vector-codebook QAT via [LC-QAT](https://arxiv.org/abs/2606.10531);
- progressive nested quantization via [MatQuant](https://arxiv.org/abs/2502.06786) or
  [MatGPTQ](https://arxiv.org/abs/2602.03537);
- binary factor/pattern controls via LittleBit/LittleBit-2, NanoQuant, BTC-LLM, DBF, and BWLA;
- mixed/critical-layer protection via [CCQ](https://arxiv.org/abs/2507.07145);
- token-gated compensation via SPEAR where the rate and architecture make it applicable;
- MoE controls via MoEQuant, KBVQ-MoE, AlphaQ, and the historical
  [QMoE](https://arxiv.org/abs/2310.16795); and
- the strongest independently reproducible same-parent artifact at no greater actual bytes.

External reported scores are never substituted for reproduction. Fairness requires the same parent,
same or lower actual physical bytes, same data access and teacher authority within a lane, identical
prompt/scorer/test-time compute, same augmentation scope, and a real packed artifact. If source code
or model access prevents reproduction, the headline is narrowed and the gap remains open.

### 12.3 Large-scale reality controls

[CCQ](https://arxiv.org/abs/2507.07145) reports 671B compression at approximately 2.06 bits and 184
GB. A separate [large-model quantization evaluation](https://arxiv.org/abs/2505.02390) reports modern
671B-class behavior around three bits with severe degradation near two bits. QMoE reports an older
1.6T Switch Transformer below 160GB at roughly 0.8 bpw. These are valuable scale controls, not proof
that a modern 671B or 1.6T generative model can retain broad capability inside a 96 GiB Studio.

---

## 13. Final adversarial audit and claim compiler

After sealed evaluation, freeze again. The final audit receives raw artifact files, manifests,
per-item outputs, scorer code, data hashes, seed records, checkpoints, and comparator results. It
must independently recompute:

- physical and resident byte totals;
- source, tokenizer, template, teacher, data, and operator identity;
- claim-lane validity and zero external dependency for standalone lanes;
- paired domain scores and confidence intervals;
- Holm correction and dominance verdict;
- worst-domain and worst-stratum failures;
- test-time compute parity;
- negative-result completeness; and
- independent reproduction status.

Permitted claim templates are deliberately narrow:

- **Codec fidelity:** “At X physical whole-artifact bpw, the packed codec is [measured relation] to
  the exact parent under the frozen suite.”
- **Restorative training:** “At X physical bpw, parent-only Doctor training restores [measured
  relation] without external runtime information.”
- **Capability elevation:** “At X physical bpw, the standalone artifact gains [measured relation]
  using provenance-bound stronger-teacher/truth training; the BF16 same-treatment control is Y.”
- **Augmented system:** “At X artifact bytes plus Y external bytes/calls/test-time compute, the
  declared system gains [measured relation]; the closed-book artifact scores Z.”
- **Dominance:** only the exact statistical verdict—frontier champion or uniform quality
  dominance—and only after independent reproduction.

“Best in the game,” “unbeatable,” “near-lossless,” “sub-bit model,” and “runs on 96GB” are rejected
unless the exact scope, physical denominator, resident receipt, comparator set, capability domains,
and proof state make the statement literally true.

---

## 14. Adversarial Audit C — rejected v5 candidate and hardening pass

The first compiled v5 candidate passed its ordinary self-tests and was still rejected. An
independent hostile-input review constructed re-signed objects that satisfied superficial hashes
while violating the intended scientific contract. This is the third mandatory break audit; its
failures are part of the design history and must never be deleted from the lineage.

| Adversarial break | Why the first candidate was rejected | V5 hardening invariant |
|---|---|---|
| Forge a `PROVEN` observation from a planned, unwired program | State labels and supplied summary statistics were trusted. | Planned/unwired/incomplete programs cannot yield proven evidence; terminal state, evidence state, seeds, draws, raw-result binding, corrected-test receipt, sealed receipt, and independent attestation must be coherent. |
| Claim arbitrary candidate-minus-parent deltas | Deltas were not cross-checked against component scores. | Every reported delta is recomputed or rejected; evidence-grade decisions require a separately verified statistics receipt bound to raw item/cluster outputs. |
| Point the artifact ledger at a nonexistent file | Declared paths, categories, lengths, and hashes were not verified against storage. | Evidence mode stats and hashes every file, reconciles each file with exactly one component, and recomputes all totals and physical bpw. |
| Bill a base byte as metadata | File and component ledgers were not cross-bound. | Per-file category sums must exactly equal the canonical component ledger; no unowned, duplicate, hidden, fallback, decoder, runtime, tokenizer, alignment, or external-state byte exists. |
| Delete benchmark vaults or independent replication and re-sign the manifest | The validator checked a subset of fields. | The entire immutable battery is compared to a freshly compiled canonical policy; removing a suite, vault, matched-compute field, multiplicity rule, sealed receipt, or replication owner fails. |
| Disable the anti-claim guard or shrink a candidate ceiling without changing its identity | Semantic policy and ceiling fields were omitted from candidate identity. | Campaign validation reconstructs canonical policy; candidate identities bind the complete semantic lane, byte ceiling, route, scope, controls, and source/root bindings. |
| Move a mandatory control to an unrelated lane | Only global control count was checked. | Control coverage is validated for every exact model × rate × claim scope × failure route × control-type tuple. |
| Mark an unwired program executable | Executable mode did not require positive authority. | Executable validation requires a verified root manifest, explicit user greenlight receipt, approved adapter registry, every executor wired, real-file artifact/source admission, and a current resource receipt. V5 planning supplies none. |
| Self-assert five seeds, Holm correction, sealing, or reproduction | Booleans and summaries were mistaken for evidence. | Evidence identities enumerate seeds and calibration draws and bind preregistration, raw outputs, scorer, corrected tests, one-time sealed service, and an independent owner. Placeholders can never prove a claim. |
| Present one independent cluster as a proof-bearing domain score | `n=1` was positive and therefore syntactically accepted. | Every protected domain requires at least five independent clusters, separately from the five training seeds and five calibration draws; underpowered rows are rejected. |
| Match only a partial inference budget | Input length, sampling temperature, reasoning tokens, and timeout could differ. | A single field-complete matched-compute contract covers input/output/reasoning tokens, sampling, attempts, timeout/OOM, memory, tools, retrieval, verifiers, external models, caches, and speculation. Missing means invalid. |
| Compare with one arbitrary method or a family label | Direct implementations and rate applicability were collapsed. | Every applicable named implementation receives a same-parent, same-or-lower-all-in-byte reproduction or a signed incompatibility receipt that narrows the claim. A family representative or reported paper table is not evidence. |
| Route mixed/unknown damage through generic representation work | The binary route omitted mixed, undetermined, and no-damage outcomes. | Undetermined blocks treatment; mixed requires both structural reconstruction and residual treatment with causal ablations; no-damage routes to zero treatment; collapse cannot pass through a generic label. |
| Mutate campaign, ladder, battery, audit, or example independently | There was no package root. | One Doctor-v5 root manifest binds core reports, implementation sources, and specifications. Derived programs and audit receipts carry that root hash; the root grants no execution authority. |
| Compute “exact” bpw from rounded `params_b` | Catalogue billions were used as though they were tensor counts. | Rounded counts are scheduling estimates only. Physical bpw requires a source-bound exact integer tensor count and hash-verified whole-artifact bytes. |
| Let the static audit prove dominance with synthetic data | A passing fixture manufactured the appearance of evidence. | Static audit tests that adversarial pseudo-evidence is rejected. It may report package-integrity success only; execution, evidence, and dominance remain false. |

Audit C changes the meaning of a green validator. It is not “the JSON is internally tidy.” It is
“the object is the canonical frozen policy, every relevant identity is bound, and the validator
cannot cross an external trust boundary without a real receipt from the configured verifier.” A
cryptographic digest proves identity, not truth; trusted verifier configuration and independent
custody are separate requirements.

---

## 15. Adversarial Audit D — replay, provenance, and statistical semantics

After Audit C passed, a fresh reviewer attacked the accepted package rather than its original
fixtures. The root was rejected again. Audit D closes five attacks that remained possible even with
role-separated external trust.

| Adversarial break | Why the Audit-C package was rejected | V5 hardening invariant |
|---|---|---|
| Replay valid authorization receipts after changing seeds, parameter count, compute budget, or teacher authority | Receipts named candidate/root/scope but not the complete executable program semantics. | A canonical `program_spec_sha256` covers all semantic program content while excluding only circular receipt wrappers; root, greenlight, parameter, teacher, adapter, and resource receipts bind that exact spec. |
| Replace an exact parameter count with another positive integer | Executable validation checked integer shape, not source provenance. | A trusted parameter-manifest receipt binds exact counted tensors, ownership/classification, parent revision/config, source files/shards, counting implementation, and program spec. Positive integers alone never establish a denominator. |
| Set raw p-values to 0.8 and Holm-adjusted p-values to 0.9 while retaining positive confidence bounds | Dominance consumed confidence bounds but ignored the preregistered adjusted-alpha decision. | Every parent non-inferiority and competitor superiority inference must satisfy its direction-consistent simultaneous interval and Holm-adjusted alpha rule; either failure blocks the verdict. |
| Add an undeclared stronger teacher to restoration or omit the teacher from elevation | Teacher permission was Boolean and unknown fields were ignored. | Exact-key teacher authority binds identity, revision, role, outputs/cache, split, provenance, training-only lifetime, and trusted authorization. Restoration and augmentation forbid stronger training teachers; elevation requires one. |
| Change the package-root hash and re-stamp a standalone program | Root binding was enforced by the binder but not the public program validator. | The program contract itself validates package schema/hash against caller-supplied trusted root identity; a concrete but unexpected root fails in planned and executable modes. |

Audit D also rejects unknown semantic fields rather than allowing an unbound extension to smuggle
teacher or execution authority. External trust membership remains a caller responsibility; the
contract validates that every trusted receipt is scoped to exactly the semantics being evaluated.

---

## 16. Adversarial Audit E — pre-sign unknown-field smuggling

Audit D's statement about unknown fields was itself attacked. A reviewer inserted
`unsafe_override: true` before program-spec signing, regenerated every receipt, and obtained a valid
executable program because several nested objects checked required fields but did not reject extras.
Hashing an undeclared field prevents later mutation; it does not prevent an adapter from assigning
that field dangerous semantics before authorization.

Audit E therefore injects unknown keys independently at the program root, target, diagnostic,
operator, executor, data contract and split rows, training/teacher objects, evaluation and compute
budget, execution receipts, exact-resume state, output contract, artifact files/component/runtime
accounting, observation metrics/comparisons/evidence receipts, and dominance receipt. Every semantic
object has an exact allowed-key set. Campaign provenance is carried only in an explicitly declared,
exact-schema metadata object; there is no generic extension dictionary. An unknown key fails before
receipt trust or program-spec authorization can make it executable.

---

## 17. Exact unresolved gaps at the v5 design boundary

These are open even after completing the three design expansions. They are not disguised roadmap
items or implied successes.

| Gap | Why it remains open | Evidence required to close it |
|---|---|---|
| U1. High-fidelity ≤0.5 physical bpw | No current primary source demonstrates broad modern-model capability at that whole-artifact rate. | Packed full-model artifact, all-domain replicated evaluation, equal-byte competitors. |
| U2. 671B/1.6T resident capability on 96 GiB | Payload arithmetic may fit only at unvalidated rates; lexical/correction/runtime bytes reduce headroom. | Transactional conversion, measured resident receipt with context/workspace, sealed capability proof. |
| U3. Embedding/LM-head floor | At extreme body rates lexical tensors can dominate whole-model bpw. | Lexical codec that wins actual bytes and rare-token/multilingual/code tests. |
| U4. Sub-bit QAT beyond 32B | Published learned-factor evidence is mostly ≤32B; 70B evidence is mainly PTQ or higher-rate. | Exact-resume streamed QAT/reconstruction at 70B+, multiple seeds, full evaluation. |
| U5. Composite factor entropy | Binary factors may be nearly maximum entropy after geometry optimization. | Measured joint/pattern entropy and net bytes after codebook/table/index overhead. |
| U6. Adaptive rank/envelope transfer | Spectral heuristics have not been shown to optimize downstream capability globally. | Calibrated local-to-full predictor and equal-byte allocator ablations across families/scales. |
| U7. Repairability-aware sub-bit base | ProjQ-style evidence is at higher rates and smaller scale. | Joint base/repair experiment below one physical bit with causal and budget controls. |
| U8. Signal/collapse classifier validity | CKA, patches, and readouts can misclassify injected full-precision recovery as reachable repair. | Prospective prediction of which **budgeted** treatment succeeds, with confidence and false-route rate. |
| U9. Conditional Doctor at sub-bit | Existing token-gated compensation does not prove recovery after representation collapse. | Jointly trained gate/treatment, severe-failure FN bound, static equal-byte control, p95/worst cost. |
| U10. Rare-expert and router fidelity | Average calibration under-samples cold experts and hides routing discontinuities. | Coverage-guaranteed expert evaluation, worst-expert bounds, router causal tests, out-of-domain traffic. |
| U11. Quality elevation attribution | Stronger teachers can improve BF16 controls too. | Same-treatment BF16, parent-only, teacher-free, and standalone-runtime ablations. |
| U12. Data contamination | Public benchmarks and teacher corpora may overlap semantically. | Hash/near-duplicate/semantic scans, post-cutoff and sealed suites, independent data construction. |
| U13. Objective conflict | Repairing one capability may degrade another. | Gradient-conflict measurements, protected-domain temporal LCBs, replay and Pareto treatment controls. |
| U14. Codec-native Apple execution | Most frontier methods lack packed Apple decoders and training paths. | CPU/Metal parity, real file loading, no dense fallback, later separate speed/energy ladder. |
| U15. Giant-source lifecycle | Remote shards, licensing, disk reclamation, and resumes can break lineage. | Immutable source manifest, byte-offset resume, transactional fetch/pack/fsync, license/provenance receipt. |
| U16. Static versus progressive prefix semantics | Arbitrary truncation is not a trained model; dynamic token precision often disappoints. | Every prefix co-trained and evaluated, nested identity proven, static controls retained. |
| U17. Benchmark completeness | No finite suite proves universal capability or future distribution robustness. | Multiple task families, metamorphic/post-cutoff tests, explicit scope limits, continuing shadow evaluation. |
| U18. Independent competitor reproduction | Recent 2026 methods may lack stable code or comparable artifacts. | Reproduction or explicit narrowed claim excluding unavailable comparison; never substitute reported table values. |
| U19. Uniform quality dominance | No Hawking v5 sealed comparison exists yet. | All-domain intersection-union pass against frozen competitors plus independent reproduction. |
| U20. Speed/speculation/distributed superiority | This campaign intentionally defers runtime optimization. | Separate native runtime ladder with target/drafter parity, KV/state, bytes moved, p95, energy, and communication. |

The unresolved set is part of the proof. Deleting a gap without the stated evidence invalidates the
lineage.

---

## 18. What “v5 complete” means

There are two different completion states:

### Static-design-complete v5

- all three expansion passes and all five mandatory adversarial audits are specified;
- four claim lanes, physical accounting, diagnostic routing, controls, data firewall, statistical
  rule, competitor registry, and unresolved gaps are explicit;
- the Training Ladder v5 can represent every model/rate/lane/stage cell without authorizing work; and
- no dominance wording is emitted.

This document establishes the static research design boundary for v5; the root-bound static audit
must still pass before the package is handed off for greenlight.

### Evidence-complete v5

- an exact packed artifact and complete physical/resident receipt exist;
- the relevant lane reaches sealed final evaluation;
- parent and every applicable equal-byte competitor satisfy the frozen fairness protocol;
- the intersection-union dominance rule passes; and
- an independent reproduction reaches the same verdict.

Only evidence-complete v5 may claim a frontier champion. Only the stronger all-domain condition may
claim uniform quality dominance. Until those receipts exist, Doctor v5 is an aggressively
over-engineered system for trying to prove dominance—not proof that dominance has already been won.
