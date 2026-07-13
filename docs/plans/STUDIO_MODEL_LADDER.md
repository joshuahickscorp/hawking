# Studio Model Ladder — Condensation, Not Quantization

Operational snapshot for the active v1/v2 run on the **96 GiB M3 Ultra / 1 TB internal SSD**.
Mutable status lives under `reports/`. The canonical next research design is
[`DOCTOR_V5.md`](DOCTOR_V5.md) and the canonical next ladder is
[`TRAINING_LADDER_V5.md`](TRAINING_LADDER_V5.md); nothing in those v5 plans is executable before the
separate green light and admission gates.

## The objective

Hawking is not optimizing a matrix-multiplication benchmark. It is searching for the smallest
**complete executable system** that preserves a model's capability:

> drive the representation to the lowest physical rate, then restore capability by any measured
> correction mechanism whose total bytes, movement, energy, latency, and failure cost are charged.

That is **condensation**. The recovery mechanism may rewrite the packed base, attach correction
operators, route sparse experts, retrieve external state, verify a draft, or conditionally refine an
answer. It does not have to look like conventional quantization. A smaller file is not a win unless
the packed artifact executes natively and passes capability, memory, latency, energy, and integrity
gates.

Primary objectives are capability/joule, capability/byte moved, capability/resident byte,
capability/parameter, and capability/second at the declared SLO. FLOPS remain a diagnostic.

## What is actually scheduled now

The broad `tools/condense/ladder.py` manifest contains 32 research entries. It is not the executable
queue. The detached reality is:

| Tier | Download | Full quality + Doctor | Sub-bit | Native/deployable meaning |
|---|---|---|---|---|
| 0.5B / 1.5B / 7B | present | active parallel Studio wave | follows the full lane | scalar results are formal experiments; VTQ remains an oracle |
| 14B | verified | queued solo after the early wave | queued after its full lane | the barrier before 120B |
| 32B | verified | deferred: estimated 85 GB peak exceeds the 78 GB process budget | representative-shard queue | no full-model Doctor/PPL/deployable floor yet |
| 72B | verified | no resident path | representative-shard queue | reconstruction only; no Doctor/PPL/serve claim |
| gpt-oss-120B | queued after both 14B lanes close | architecture/streamed evaluation missing | representative-shard queue | native MXFP4 parent, not a BF16 Qwen parent |
| DeepSeek-V4-Flash 284B | architecture + disk gated | not implemented | planned | download target is not a processing result |
| Kimi-K2.6 1.1T | terminal architecture + disk gated download | not implemented | planned | largest full repository that can fit the SSD lifecycle after guarded source release |
| DeepSeek-V4-Pro 1.6T | **never full-installed** | remote shard stream-transcode research | planned terminal target | largest parameter target; requires a new streamed architecture/runtime path |

The missing Qwen 3B and cross-family entries are real coverage gaps, not silently scheduled work.
They remain useful controls, but the priority order is now the scale frontier: 120B, 284B, 1.1T,
then the 1.6T remote-stream target.

Executable sources of truth:

- Full resident ladder: `tools/condense/studio_run.py`
- Downloads: `tools/condense/download_queue.py`
- 14B handoff: `tools/condense/processing_queue.py`
- 32B/72B/120B shard experiments: `tools/condense/frontier_stream_queue.py`
- Largest-model readiness schedule: `tools/condense/terminal_frontier_queue.py`
- Active-ladder treatment compiler: `tools/condense/condensation_doctor.py`
- Doctor-v2 typed ABI and frontier compiler: `tools/condense/healer_abi.py` and
  `tools/condense/doctor_frontier.py`
- Doctor-v2 detached observer and fail-closed preflight: `tools/condense/doctor_frontier_queue.py`
  and `tools/condense/doctor_frontier_worker.py`
- Canonical Doctor-v5 contract, campaign, quality battery, ladder, package root, and static audit:
  `tools/condense/doctor_v5_contract.py`, `tools/condense/doctor_v5.py`,
  `tools/condense/quality_battery_v5.py`, `tools/condense/training_ladder_v5.py`,
  `tools/condense/doctor_v5_root.py`, and `tools/condense/doctor_v5_audit.py`

The v2 architecture in [`CONDENSATION_DOCTOR_V2.md`](CONDENSATION_DOCTOR_V2.md) explains the current
detached observer lineage. Its 8,192 explicit cells are identities, not running jobs. V5 supersedes
its scientific breadth with four isolated claim scopes, 191 mechanisms, 21 direct competitors,
32,768 candidates, 1,280 model/rate/scope lanes, a sealed 12-domain quality contract, one package
root, and a hostile-input static audit. Both generations
remain unexecutable; the current v2 observer is left intact rather than silently changing its pinned
campaign.

## The corrected maximum-model answer

There are four different maxima; conflating them caused the earlier confusion.

1. **Largest safe BF16 parent resident for current full-model work: 32B weights.** The weights fit,
   but the present full Doctor pipeline is still correctly deferred because its estimated 85 GB peak
   exceeds the 78 GB interactive process envelope.
2. **Largest next native checkpoint: gpt-oss-120B.** Its selected official `original/*` MXFP4 subset
   is about 65.3 GB. It is the next practical large-model proof once 14B closes.
3. **Largest conventional full repository with a plausible internal-SSD lifecycle: Kimi-K2.6,
   1.1T parameters.** A fresh Hugging Face dry run measured 595.205 GB. It fits only after previous
   parents are guardedly released and only after Kimi architecture support exists.
4. **Largest terminal research target: DeepSeek-V4-Pro, 1.6T parameters.** Its repository measured
   892.763 GB, so it cannot coexist with the 150 GB disk reserve and 32 GB cache reserve. It must be
   transformed one verified remote shard at a time and the source shard discarded only after its
   output and receipt are durable.

Nominal weight-only math, before every required overhead byte:

| Target | 0.50 bpw | 0.33 bpw | 0.25 bpw | 78 GB verdict |
|---|---:|---:|---:|---|
| 1.1T Kimi | 68.75 GB | 45.38 GB | 34.38 GB | 0.50 is overhead-tight; 0.33 is the safer resident target |
| 1.6T V4-Pro | 100.0 GB | 66.0 GB | 50.0 GB | 0.50 is not resident; 0.33 is borderline; 0.25 leaves working room |

For the 1.6T source, the largest observed shard is about 14.1 GB. A future decoder can therefore
hold one source shard, its decoded/block workspace, and a growing 50–66 GB output without ever
installing the 893 GB parent. This is a design target, not a claim that the transformer, evaluator,
or native runtime already supports it.

## Forward execution order

The ordering is durable and gate-driven, not date-driven:

1. Finish the active 0.5B/1.5B/7B full lane without changing its recipe identity.
2. Run 14B full quality, then its sub-bit campaign, solo.
3. Download the 65.3 GB gpt-oss-120B subset automatically after the strict 14B barrier.
4. Run the 32B, 72B, and 120B representative-shard campaigns. These may inform methods but cannot
   substitute for full PPL, Doctor, packed parity, or serving evidence.
5. Build streaming evaluation and the gpt-oss tensor/architecture adapter; promote 120B to an honest
   packed 2-bit target before attempting weaker-rate claims.
6. Build the DeepSeek V4 Flash adapter and process 284B at 2.0, 1.34, 1.0, 0.8, and 0.5 physical bpw.
7. Guardedly release old sources only after a winner is packed, hashed, reloaded, capability-checked,
   and source-bound. Then admit the Kimi 1.1T repository.
8. Process Kimi with MoE-aware streamed experts at 0.8, 0.5, 0.33, and 0.25 physical bpw. A rate is
   resident only when its **actual** artifact and runtime working set fit.
9. Implement remote shard transactionality and target V4-Pro 1.6T at 0.50 (out-of-core control),
   0.33 (borderline resident), and 0.25 (safe resident target).

There is no wall-clock deadline. A campaign may run for weeks. It still has a progress watchdog:
“unlimited wall time” does not mean an undetected dead process.

## The Condensation Doctor

The Doctor becomes a versioned correction graph:

```text
y = K_base(x, packed_base) + Σ K_i(x, correction_i, route_i)
```

- A **base-rewriting healer** earns quality credit only after writing a new packed and hashed base.
- A **sidecar healer** emits ordered bias, low-rank, sparse, diagonal, codebook, or routed correction
  operators. Every file and index byte is billed.
- A dense rehydrated or fused BF16 shadow is never a compressed artifact.
- The exact zero-correction artifact is always preserved. A Doctor regression is a complete negative
  experiment, not permission to promote the trained adapter and not a reason to wedge the queue.

Every custom treatment identity binds parent revision/config/tokenizer, base and quantizer hashes,
codec recipe, calibration/selection/final-eval corpora, teacher, mechanism source/version, exact target
modules and shapes, rank/sparsity/dtype per module, steps, learning rate, objective, seed, optimizer,
sampler, memory class, correction-byte budget, and runtime ABI. Changing any field creates a new
configuration. This prevents custom ranks or learning rates from colliding with ordinary `+dr` resume
rows.

Two rates remain separate:

```text
codec_oracle_bpw     = exact logical baker accounting (research only)
physical_model_bpw  = 8 * actual bytes of base + pass-through tensors + LUTs + scales
                      + corrections + sparse indices + routers + metadata + alignment
                      / source parameter count
```

Dynamic treatments additionally report installed bpw and mean/p95/worst bytes moved per token.

## Harder-than-the-old-MD experiment policy

The old recipe ladder asked whether AWQ, residuals, and LoRA could restore a fixed low-bit base. The
new campaign searches the whole representation/correction/runtime system.

### Breadth

- Base rates: 4, 3, 2, 1, 0.8, 0.5, 0.33, and 0.25 physical bpw.
- Representations: scalar affine, STRAND, exact-byte mixed precision, dense+sparse, residual,
  vector/additive codebooks, binary low-rank factors, and binary pattern codebooks.
- Transforms: none, RHT, activation scaling, and learned rotations.
- Correction operators: zero, output bias, residual-SVD, targeted/variable rank, low-rank+sparse,
  quantized sidecars, codec rewrite, block QAT, self-KD, large-teacher KD, and routed refinements.
- State/runtime: KV16/8/4/2, rotating/prompt caches, CPU packed, Metal packed, cold/warm runs,
  short/long context, no speculation, and adaptive speculation.
- Controls: at least three seeds, domain-matched and multi-domain calibration, independent frozen
  selection/final sets, zero-correction, random allocation, and equal-byte comparisons.

### Allocation

1. Broadly screen every treatment family on small calibration slices and retain negative rows.
2. Use successive halving to spend 4× more tokens/steps on non-dominated candidates. This is not a
   time-saving concession: it transfers month-scale compute away from treatments already proven to
   lose capability per byte.
3. Promote rank or sparse budget only after positive held-out recovery per **serialized** added byte.
4. Run full three-seed, calibration, multiwindow, and capability ablations at the Pareto frontier.
5. Require packed round trip, native parity, resident execution, then same-box efficiency. No oracle
   may skip evidence classes.

## Proposal matrix

All bandwidth/latency entries are hypotheses until measured on this Studio.

| Class | Proposal | Complexity | Expected bandwidth / latency | Difficulty | Existing GPU / Apple / future hardware | Quantization, speculation, distributed, future-architecture interaction |
|---|---|---|---|---|---|---|
| Immediate | MLX uniform 2-bit + mixed 2/6 and T-MAC W2 controls | encode O(P), inference O(P_active) | ~8× less weight payload than BF16 before overhead; latency kernel-dependent | medium | CUDA/GPTQ mature; MLX packed Metal and T-MAC Apple CPU controls; LUT MAC maps forward | honest deployable floor; either can draft; shards distribute independently; MoE needs expert allocation |
| Immediate | Activation-weighted residual-SVD | O(mnr) per tensor; O(r(m+n)) sidecar | >99% less correction traffic than a dense F16 delta is plausible at small rank; adds narrow GEMVs | medium | portable oracle; native Qwen Metal correction ABI missing; easily fused in future | initializes from W-Q(W); repairs target/drafter; tensor-shard parallel; applies to experts |
| Immediate | Exact-byte bit/rank/sparse waterfill | multiple-choice knapsack O(NB) or Lagrangian O(N log N) | removes low-value correction traffic; may improve latency if allocation remains regular | medium | compiler-side today; variable-rank runtime needed; natural hardware allocation pass | jointly selects bits and Doctor budget; can protect verifier-critical tensors; expert-aware |
| Immediate | Bias + row-local sparse error | offline O(P); runtime O(rows+k) | bias is tiny; sparse indices/gathers can erase theoretical savings | medium | generic GPU sparse support; Apple packed/fused apply missing; structured future form preferable | complements any base; compatible with speculation; sparse rows shard; route only high-value errors |
| Immediate | Objective and correction-precision ablations | O(TP·epochs) offline | fixed-byte objective changes have no runtime cost; 8/4/2-bit sidecars cut F16 payload 50/75/87.5% before metadata | medium-high | training portable; slow CPU-bf16; future fused sidecar decoder | top-k+tail KL, feature sketches, capability loss; optimize draft acceptance; teacher caches distribute |
| Immediate | KV/prompt-cache and adaptive speculative controls | attention O(Ld) | 2-bit cache payload ~8× below F16; speed depends on acceptance and decode overhead | high | CUDA methods exist; MLX cache baseline; future cache-native attention | orthogonal to weight floor; rejected work is charged; verifier communication matters; MLA/SSM need adapters |
| Medium | Rotation + Hessian/vector/additive codebooks | block Hessian/sketch + iterative codebook optimization | strongest credible 2-bit quality lane; runtime wins only with fused lookup/rotation | high | CUDA references; Apple MLX/Metal or T-MAC-style port required; excellent near-memory fit | rewrites base at same bytes; speculation-neutral; blocks/codebooks distribute; per-expert codebooks |
| Medium | Exact-resume streamed block QAT/KD | O(TP·epochs), peak one block/shard | quality improvement at unchanged base bytes or billed sidecar; not inherently faster | very high | GPU references; new CPU/MPS block engine needed; future block-local trainer | unlocks 32B+; can tune drafter/target; shard parallel with global eval; natural for MoE experts |
| Medium | Binary factors / pattern codebooks at 1.0→0.25 bpw | iterative block reconstruction | enormous payload reduction; Apple latency unknown until packed decoder | very high | current research is CUDA-centric; CPU then Metal ABI required; maps to bitwise hardware | true sub-bit lane; only useful as drafter if acceptance pays; block/expert parallel; MoE promising |
| Long | Progressive event-driven healer | O(P_active)+Σp_iC_i | mean traffic B0+Σp_iB_i; p95/worst and installed bytes remain billed | very high | prototype on CUDA/Metal; event queues/near-memory ideal | 0.25–0.5 base plus conditional refinement; risk triggers verifier; cold corrections can be distributed |
| Long | Retrieval/output repair | ANN lookup + generation | may replace factual parameter traffic but adds index/cache/network bytes and tail latency | high | portable; local mmap/ANN suits Studio; semantic-memory hardware later | cannot excuse destroyed reasoning; verifier can request evidence; network miss distribution is first-class |
| Long | Native low-bit training + structured sparsity | pretraining scale | largest potential memory/energy win with co-designed runtime | extreme | BitNet/T-MAC are controls, not conversion proof; specialized hardware ideal | changes representation instead of repairing PTQ; early exits/self-spec possible; expert/data parallel |
| Long | 1.6T remote-shard condensation | O(P_total·candidate work), peak O(one shard+block state+output) | network read once per admitted treatment; 0.25 bpw gives 50 GB nominal artifact | extreme | no current end-to-end implementation; Apple unified memory is the target; future near-memory decode | source never installs; speculative module and MoE routing are architecture-bound; shard transactions distribute |

## Evidence and safety gates

Every phase is atomic and resumable at a meaningful boundary:

- Downloads: per-file HF/Xet resume, heartbeat, verification marker, repository revision.
- Stream conversion: source-shard hash, treatment identity, atomic output shard, round-trip check, then
  and only then source-shard release.
- Training: best/zero/latest checkpoints plus optimizer, RNG, sampler, microstep, source, and objective
  state before custom large-model claims.
- Evaluation: independent selection and final sets; multiwindow ≥4; capability tripwire; no full
  override safetensor load for large tiers.
- Runtime: all-tensor ownership, no BF16 fallback, actual peak unified memory, swap=0, bytes moved,
  energy, cold/warm latency, p95, and repeated runs.

The scheduler pauses and checkpoints on non-normal memory pressure, any swap, thermal warning, power
drain, explicit relocation drain, or disk reserve violation. It never deletes arbitrary data. Parent
release is allow-listed and requires a durable winner bound by hash to its source and evidence.

## Research anchors

- 2-bit/vector/codebook frontier: [AQLM](https://arxiv.org/abs/2401.06118),
  [QuIP#](https://arxiv.org/abs/2402.04396), [VPTQ](https://arxiv.org/abs/2409.17066)
- QAT and self-distillation: [BitDistiller](https://arxiv.org/abs/2402.10631),
  [EfficientQAT](https://arxiv.org/abs/2407.11062), [LoftQ](https://arxiv.org/abs/2310.08659)
- Sparse/outlier recovery: [SqueezeLLM](https://arxiv.org/abs/2306.07629),
  [SpQR](https://arxiv.org/abs/2306.03078)
- Sub-bit: [QMoE](https://arxiv.org/abs/2310.16795),
  [STBLLM](https://arxiv.org/abs/2408.01803), [BTC-LLM](https://arxiv.org/abs/2506.12040)
- Apple-native controls: [MLX-LM](https://github.com/ml-explore/mlx-lm),
  [T-MAC](https://github.com/microsoft/T-MAC),
  [Apple QuantSpec](https://machinelearning.apple.com/research/quantspec)
- Official terminal parents: [gpt-oss-120B](https://huggingface.co/openai/gpt-oss-120b),
  [DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark),
  [Kimi K2.6](https://huggingface.co/moonshotai/Kimi-K2.6),
  [DeepSeek V4 Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark)

## Completion condition

The ladder is not complete because a queue exits. It is complete when each promoted frontier has a
source-bound packed artifact, native execution, capability evidence, physical byte ledger, memory and
energy receipts, and a Pareto comparison against the best same-box baseline. Negative results remain
part of the dataset. The terminal success is not “downloaded 1.6T”; it is **the highest-capability
1.6T-derived system that actually fits and runs inside the declared Studio envelope**.
