# STRAND compression map - physical-limit attack plan (2026-06-12)

_Canonical map for the post-32B/70B hardening session. This consolidates the compression map,
the aggressive speed/compression moat plan, and the "attack every inherited standard" framing.
Numbers are measured unless labeled pending/modeled/proposed._

## C2 GATE VERDICT (measured 2026-06-12, Lane A microscope — research/bit-ledger-results.md)

**C2 CLEARS its gate. Measured recoverable bpw (q2, whole 0.5B model):**
- scale + sub_scale entropy coding: **0.106 bpw** (q2) / 0.115 (q3) — map estimate 0.11-0.20 was
  optimistic; real win sits at the bottom (scale_q is ~10.5-bit entropy billed as 32 bits; NOT a
  prediction win — RHT whitened block-to-block structure).
- **outlier POSITIONS: 0.1476 bpw — the single largest recoverable component** (bigger than scale;
  reframes OUTL redesign §3.9 from provisional to a real lead lever).
- outlier values: ~0.02 bpw (modest). init_state: ≤0.012 bpw (use tail-biting, not entropy coding).
- **Total realistic C2+OUTL-position recovery ≈ 0.25 bpw** → q2 2.665 → ~2.42 bpw, no quality cost.
SATURDAY PRIORITY SHIFT: within the hardening session, outlier-position coding (0.15) leads,
scale/sub-scale coding (0.11) second; init-state → tail-biting policy, not a codec.

## 0. The thesis

STRAND is now close enough to the physical limit that ordinary methodology becomes the enemy.
At 2-bit, STRAND's quantizer is roughly **28% rel-RMS** against a **25% Gaussian
rate-distortion lower bound**: about **1.25x the ideal distortion**. That is too close to
the floor for a generic "better trellis" to be the main source of future gains.

The next moat is to attack the defaults that surround the quantizer:

- **bf16 as production object**: keep bf16 as the truth/eval anchor, but stop treating bf16
  tensors as the natural storage, transfer, runtime, or training-shadow representation.
- **fixed-width side-info**: scales, sub-scales, init states, and random-access tables are
  now large enough to be first-class compression targets.
- **uniform bit-rates**: RHT makes rel-RMS flat, so the allocator must optimize held-out PPL
  and output error, not reconstruction error.
- **PTQ as the low-bit methodology**: 2-bit and sub-2-bit are training products.
- **CPU scalar decode as the runtime shape**: the speed moat is exact integer bitslice,
  resident prepared tensors, and batched dispatch.
- **standard container modes**: one artifact should not pay seek-mode overhead when the
  deployment path streams whole tensors.

The north-star product is not "a bf16 replacement" in the naive sense. It is a **custom
production representation below bf16**: deterministic, float-free on decode, trainable at
low bits, and engineered so every bit and every dispatch exists because a gate justified it.

## 1. The measured density and quality ladder

| rung | bpw | 0.5B PPL | 7B / scale signal | 14B | status |
|---|---:|---:|---|---:|---|
| bf16 | 16 | 12.536 | Qwen 7B 6.629; Llama2 7B 5.535 | 5.102 | truth anchor, not product target |
| q4 | ~4.5 | 13.5 class | Qwen 7B 7.81 | - | quality yardstick |
| **mp_light = down_proj@4** | 3.67 | _down-protect 0.5B UNMEASURED_ | Llama2 mp_light 5.855 vs bf16 5.535 (+5.8%); Qwen best class 8.45 | - | **shipping 3-bit moat** |
| omega-star = attn@4/FFN@3 | 3.81 | 15.039 | - | - | DOMINATED (Ω*: on the q3→q4 line, not Pareto) — do not confuse with mp_light |
| q3 uniform | 3.34-3.67 | 16.113 | Qwen 7B 9.42 | - | usable, allocator baseline |
| q2 PTQ | 2.60-2.72 | 79.941 | Qwen 7B 10.538; Llama2 7B 42.407 | 8.925 | size play; family-dependent quality |
| q2 PV-trained | ~2.67 | prior floor **26.77** | 7B selective-PV pending | - | training bridge |
| ~1.5 vec d=2 PTQ | ~2.17 effective | 49k collapse | - | - | dead by PTQ |
| ~1.5 vec d=2 PV | ~2.17 effective | **424** | - | - | training-only frontier |

Notes:

- The 0.5B iso-bpw result is real but narrow: STRAND mp_light 15.039 beats GGUF Q2_K
  15.238 at lower bpw because 896-wide tensors trigger GGUF fallback tax. At 256-aligned
  scale, that specific structural edge shrinks; 7B GGUF is still an open head-to-head.
- Qwen 7B/14B tolerate q2 PTQ much better than Llama2. That is not a contradiction; it is
  the signal that 2-bit quality is model-family and training dependent.
- The live 32B chain is a scale probe for whether Qwen-style tolerance persists upward.

## 2. Three bit-rate regimes

**Regime A - >=3-bit ships now.** Quality is already useful. Binding lever:
**bpw reduction at fixed quality**.

- down-protect mixed precision, not attention-protect.
- side-info entropy coding.
- seek/stream container split.
- speed hardening so density does not come with unusable decode.

**Regime B - 2-bit is contested and training-gated.** PTQ is a size product, not a
universal quality product.

- PV/QAT through the real STRAND encoder is the main route.
- WSD cooldown is the current clean A/B against the prior 26.77 0.5B PV floor.
- selective-PV decides whether a solo 7B training run is a cheap arm or a large expensive arm.
- trained silence may become a compression lever only if PV creates entropy structure.

**Regime C - sub-2-bit is the trailblazer space.** PTQ collapses; vector-trellis plus PV is
the only proven direction.

- the win condition is not "PTQ usable at 1.5-bit"; it is a deterministic runtime for weights
  trained into that rung.
- this is where custom methodology matters most: codebook, objective, schedule, and storage
  must be co-designed.

```
bpw ------------------------------------------------------------>
 1.0     1.5      2.0      2.7       3.3     3.8      4.5     16
  |-- C: PV+vec --|-- B: PV/silence --|-- A: PTQ+alloc+C2 --| bf16 anchor
       frontier          contested             ships now         truth/eval
```

## 3. Attack surfaces: every inherited standard is a constraint

This section is the audit layer. Each standard exists because it was convenient somewhere else.
STRAND should keep it only if it still wins under our gates.

### 3.1 bf16 as the default production object

bf16 is the truth anchor because pretrained models, eval harnesses, and teacher logits live
there. It does not have to be the product representation.

Attack plan:

- **No bf16 reconstruction as product path**: HF safetensor recon is a test/eval adapter.
  The shipped object is `.strand`, with direct runtime decode.
- **No persistent bf16 training shadow unless needed**: PV currently pays huge memory for
  fp32/bf16 shadows and Adam state. Build a "delta-shadow" and "selective-shadow" memory
  ledger so the training object becomes STRAND-native.
- **Teacher is an oracle, not a resident clone**: cache top-k logits or low-rank teacher
  residuals where KD permits, instead of keeping the full bf16 teacher on device.
- **bf16 endpoint only where accuracy demands it**: end-to-end runtime can decode integer Q12,
  accumulate in documented float order, and later test fixed-point/RMSNorm approximations as
  a separate deterministic inference lane.

Gates:

- any bf16-shadow removal must match the current PV loss curve within 2% at the same step.
- any fixed-point activation/norm route must be byte-stable or explicitly documented as a
  deterministic-kernel variant, and must pass held-out PPL before perf is considered.

### 3.2 GGUF/K-quant methodology

GGUF is a format family optimized around its own block geometry and compatibility surface.
STRAND should not copy its assumptions.

Attack plan:

- dimension-aware tiling: never silently fall back to a fatter rung when tensor width is not
  friendly.
- tensor-class and depth-aware rungs, selected by PPL, not by generic quant names.
- explicit iso-bpw harness: every external comparison bills the same weights and runs the same
  eval path.

Gate:

- a GGUF comparison is admissible only if bpw, eval harness, dataset, tokens, and dequant path
  are written next to the result.

### 3.3 Fixed-width side-info

The payload is close to its floor; the metadata is not. Per 256-weight block today:

- 32-bit scale
- 8 x 6-bit sub-scales = 48 bits
- init state up to L bits unless tail-biting removes it
- v2 random-access table: 16 B/block = 0.0625 B/w

The 80-bit scale/sub-scale floor is **0.0391 B/w** before archive table overhead. At 2-bit
payload, that is a large fraction of the real artifact.

Attack plan:

- predictive scale coding: previous-block, previous-row, per-tensor median, and tensor-class
  predictors.
- sub-scale entropy coding: order-0 and context-by-position CDFs for the eight sub-scales.
- init-state coding: exploit tail-biting when it wins; otherwise model init_state deltas or
  low-entropy terminal states.
- archive table split: keep seek tables only in seek-mode; stream-mode delta-codes or omits
  them where dispatch can walk sequentially.

Gate:

- first build a side-info microscope that reports entropy and recoverable B/w by tensor,
  tensor class, rung, and model family.
- implement C2 only if real tensors show at least **0.01 B/w** recoverable from scale/sub-scale
  coding, or at least **0.04 B/w** recoverable including stream-mode table changes.

### 3.4 256-weight blocks and fixed geometry

The 256 block is good for determinism and regular decode, but it is not sacred. At physical
limits, block geometry controls side-info rate, RHT alignment, GPU occupancy, and quality.

Attack plan:

- profile 128/256/512 block variants as separate format profiles, not ad hoc knobs.
- tie block length to tensor width alignment: avoid 0.5B straddle artifacts and preserve
  7B/14B strict rows.
- for stream-mode, allow larger macroblocks that share side-info predictors while keeping
  256-weight decode microblocks for kernel regularity.

Gate:

- no block profile ships unless it passes the full identity matrix and improves either PPL per
  bpw or decode traffic by at least 3%.

### 3.5 RHT as a fixed basis

RHT is one reason the current quantizer is strong; it also kills many naive sensitivity
methods by whitening the tensor. The standard to attack is not "use RHT" but "use one
unexamined basis seed and stop thinking."

Attack plan:

- seed search on small calibration slices: choose among deterministic RHT seeds by output
  error, not rel-RMS.
- family/tensor-class basis policy: allow a tiny seed index if a class consistently benefits.
- structured basis variants: RHT plus sign/permutation families that preserve fast inverse
  and exact determinism.

Gate:

- seed/basis search must improve held-out PPL at fixed bpw. If it only improves rel-RMS, it is
  another proxy win and dies.
- the seed cost must be billed. A 1-byte seed choice is acceptable only if it buys more than
  its side-info cost.

### 3.6 MSE as the objective

Weight-MSE is not the truth. We already saw rel-RMS stay flat while output-RMS can move.

Attack plan:

- PV: optimize the model through the real quantizer.
- live-Fisher requant: only if Phase 0 shows reproducible within-block structure after RHT.
- de-bias: only if real activation means are nonzero and PPL improves.
- allocator: optimize PPL/output error, not reconstruction error.

Kill list already banked:

- generic Hessian/diag-H objectives
- rel-RMS-only layer sensitivity
- natural silence without training
- adaptive effort/Fano where RHT makes symbol entropy flat
- attention-protect as the default mixed-rung story

### 3.7 Runtime methodology

The standard path "decode to a tensor, then GEMM" is a compatibility crutch. It pays Q12
materialization and dispatch overhead that a native runtime does not need.

Attack plan:

- Metal bitslice as the local runtime seed.
- one command buffer per token, or grouped by shader/config, never per tensor.
- B=16 prompt tiling on M3-class GPUs; B=64 regresses from register pressure.
- lean side-info entry so fused kernels do not stream fat metadata after removing Q12 writes.
- CUDA bitslice for the server lane after the pod scale chain finishes.

Gate:

- every runtime path must assert Q12 identity before perf.
- whole-token batching must save at least 20% on multi-tensor synthetic models.
- CUDA must be measured, not inferred from M3 arithmetic.

### 3.8 Evaluation methodology

At this edge, a benchmark can become a product constraint. The eval harness itself must be
treated like code.

Attack plan:

- fixed ledger schema for every result: model, dataset, tokens, ctx, chunks, device, dtype,
  bpw, git, harness key.
- adversarial family set: Qwen, Llama2, Mistral or another hostile architecture, and scale
  points at 0.5B/7B/14B/32B/70B where available.
- every proxy result gets a PPL gate before adoption.
- no cross-harness claims.

Gate:

- a result without provenance can guide intuition, but cannot enter the canon table.

### 3.9 Outlier side-channel methodology

The outlier channel is not just a flag. At 1% with 8-bit residuals it is a second codec
with real bpw cost. It helped q2, but at physical-limit rates every residual bit has to
compete against C2, PV, and allocator alternatives.

Attack plan:

- entropy-code outlier positions by row and local gap, not as a flat sidecar.
- encode residual values with a small signed residual codebook instead of raw 8-bit where
  value entropy permits.
- test row-bucketed residual patches: top-k per row, top-k per macroblock, and down_proj-only
  residuals.
- move from "percentage of weights" to **PPL-per-bpw residual budget**: outliers compete with
  raising selected tensors from q2 to q3/q4.
- preserve deterministic runtime boundary: bulk GPU path first, CPU or ordered GPU sparse
  patch second; no atomics that change floating accumulation order.

Gate:

- adopt only if the residual channel wins PPL at equal bpw against mixed-rung alternatives.
- if q2_out1 helps only because it secretly spends too many bits, fold it into the allocator
  as another priced rung rather than a special exception.

### 3.10 Activation, KV, and non-weight defaults

This map is weight-first, but a product runtime can lose the moat if every non-weight tensor
stays in inherited bf16/fp16 methodology.

Attack plan:

- KV cache quantization as a separate deterministic profile, with attention PPL/latency gates.
- activation tile precision policy: fp32 accumulation where needed, documented lower precision
  or fixed-point approximations where PPL permits.
- norm/rope/sampling determinism profiles: exact mode for reproducibility, fast mode with
  explicit low-bit drift budget if it earns speed.
- keep this product-adjacent until weight runtime is stable; do not let it distract from C2
  and PV.

Gate:

- no activation/KV route ships unless end-to-end PPL and generation sanity pass, not only
  kernel microbenchmarks.

## 4. Ranked lever ledger

| rank | lever | regime | expected win | gate | status |
|---:|---|---|---|---|---|
| 1 | **C2 side-info entropy coding** | A/B/C | recover 0.01-0.04+ B/w with no quality loss | entropy microscope, identity decode | untested, cleanest density win |
| 2 | **seek/stream container split** | A/B/C | stop paying random-access table cost in deployment | identical Q12 in both modes; >=0.04 B/w or >=5% speed | proposed |
| 3 | **Metal bitslice production path** | speed moat | 3.3-3.9x CPU decode; 35-41 Gw/s fused B=1 | whole-token command buffers, prepared resident path | revived, needs hardening |
| 4 | **lean GPU side-info entry** | speed + density | reduce fused metadata traffic; possible 10% B=1 speed | BitsliceEntryLean identity + perf | proposed |
| 5 | **rung allocator, down-protect/depth variants** | A | lower PPL at same bpw | PPL sweep, not rel-RMS | built/screened partly |
| 6 | **PV WSD + selective-PV** | B | 2-bit quality path; 7B solo training | beat 26.77; down_proj >=90% of full PV gain | running/pending |
| 7 | **trained silence** | B/C | entropy-created compression, not natural silence | PPL flat/better and >=0.03 bpw entropy saving | lever built, uncalibrated |
| 8 | **OUTL redesign** | B | current 1% outlier helps but costs bpw | PPL-per-bpw vs q2_out1 and mixed rungs | proposed |
| 9 | **de-bias with real activation means** | A/B | maybe small PPL win for ~0.018 bpw | real mu + PPL A/B | provisional |
| 10 | **basis/seed search** | A/B | tiny free-ish quality if seed matters | PPL at fixed bpw, seed cost billed | proposed |
| 11 | **KV/activation deterministic profiles** | product-adjacent | avoid losing runtime moat outside weights | end-to-end PPL/generation gates | proposed, deferred |
| 12 | **vector-trellis + PV** | C | sub-2-bit frontier | trained PPL, not PTQ | moonshot, direction proven |
| 13 | **rank-convergence/window decode** | speed moonshot | exact intra-chain parallelism | convergence-depth distribution | research gate |

## 5. Approximate improvement budget

_This is planning arithmetic from landed metrics. It is not a claim until the gates land._

### 5.1 Compression budget

Current 0.5B sidecar baselines from `model.safetensors.json`:

| artifact | effective bpw | PPL | note |
|---|---:|---:|---|
| q2_l12_out1 | 2.6653 | 79.9406 | 2-bit + 1% outlier, PTQ |
| q3_l12_out1 | 3.6653 | 16.1128 | 3-bit + 1% outlier |
| mp_light / omega-star | 3.8056 | 15.0391 | mixed 3-bit, attn4/ffn3 in this 0.5B run |
| bf16 anchor | 16.0000 | 12.5358 | truth anchor |

Fixed side-info arithmetic per 256-weight block:

| component | bpw | B/w | action |
|---|---:|---:|---|
| scale + sub-scales | 0.3125 | 0.0391 | C2 entropy coding |
| init state, L=12 when billed | 0.0469 | 0.0059 | tail-biting or init-state model |
| init state, L=7 when billed | 0.0273 | 0.0034 | tail-biting or init-state model |
| v2 random-access table | 0.5000 | 0.0625 | seek/stream split, chunked checkpoints |

Expected density movement:

| lever | likely recovery | encoded-tensor effect | archive/traffic effect | confidence |
|---|---:|---|---|---|
| C2 scale/sub-scale coding | 0.11-0.20 bpw | q2 2.665 -> **2.46-2.56 bpw** | 4-8% smaller encoded tensors | medium, needs entropy microscope |
| init-state model / tail-biting policy | 0.01-0.05 bpw | q2 2.665 -> **2.62-2.65 bpw** alone | small but free if identity holds | low-medium |
| seek/stream table split | up to 0.25-0.50 bpw of archive traffic | no change to encoded-tensor bpw | seek archive q2-ish 3.165 -> **2.67-2.92 bpw** before C2 | high arithmetic, implementation-gated |
| C2 + stream mode combined | 0.36-0.70 bpw archive traffic | q2 payload unchanged | seek archive q2-ish 3.165 -> **2.46-2.81 bpw** | medium |
| OUTL position/value coding | 0.05-0.20 bpw | q2 2.665 -> **2.47-2.62 bpw** if quality holds | also cheaper sparse patch | low-medium, PPL-per-bpw gate |
| de-bias | costs ~0.018 bpw on 0.5B | not a density win | possible PPL win if real activation mean exists | provisional |
| trained silence + C2 | 0.03+ bpw only if entropy drops | q2/sub-2 only | compression win created by training | low until measured |

Interpretation:

- **Near-term realistic encoded-tensor target:** q2_out1 class can probably move from
  **2.665 bpw to about 2.45-2.55 bpw** without touching quality if C2 and OUTL coding work.
- **Near-term realistic archive/deployment target:** if stream-mode can avoid most seek-table
  cost, q2 deployment traffic can plausibly move from roughly **3.17 bpw seek-mode** to
  **2.5-2.8 bpw**.
- **Aggressive target:** with C2, stream-mode, OUTL coding, and trained silence all passing,
  q2-class deployment below **2.4 bpw** is plausible. Below **2.2 bpw** likely requires
  moving down the actual rung or training-created entropy, not metadata cleanup.

### 5.2 Quality budget

Measured PPL moves:

| move | measured effect |
|---|---|
| 0.5B q2 PTQ -> prior q2 PV | 79.94/80 class -> **26.77**, about **66% PPL reduction** |
| 0.5B q3 -> omega-star (attn@4) | 16.113 -> **15.039**, 6.7% for +0.14 bpw — but this is the DOMINATED split; the real mp_light (down@4) 0.5B point is unmeasured and should beat this at lower bpw |
| Qwen 7B bf16 -> q2 PTQ | 6.629 -> **10.538**, +59% |
| Qwen 14B bf16 -> q2 PTQ | 5.102 -> **8.925**, +75% |
| Llama2 7B bf16 -> q2 PTQ | 5.535 -> **42.407**, +666% |
| Llama2 7B bf16 -> mp_light | 5.535 -> **5.855**, +5.8% |
| de-bias modeled output-RMS | **4.19% output-RMS reduction** at assumed nonzero mean; 0 at zero mean |

Planning estimates:

- WSD cooldown on 0.5B PV: **0-5% relative PPL improvement** over 26.77 is the honest range
  until the live run lands.
- selective-PV down_proj gate: if it recovers **>=90%** of full-PV improvement on 0.5B, the
  A100-80 7B arm is justified.
- Qwen 7B q2 PV target: **<8.6 PPL** is useful; **~8.9 PPL** is AQLM-class (AQLM is ~+35% over
  its own bf16; Qwen bf16 6.629 × 1.35 ≈ 8.9). NB the famous AQLM "6.9" is llama2-7b absolute,
  a different model/harness — do not target 6.9 on Qwen.
- Llama2 7B hostile-family target: **<10 PPL** would be the breakthrough; the 0.5B PV law
  alone would map 42.4 -> ~14, so hostile-family success likely needs selective scope,
  schedule, or broader training improvements.

### 5.3 Speed and development-velocity budget

Measured runtime anchors:

| path | measured rate/effect | improvement |
|---|---:|---:|
| CPU rayon decode | ~4.85 Gw/s best ladder; 3.85-4.09 Gw/s in GPU comparison gate | baseline |
| Metal bitslice decode | 12.66-15.89 Gw/s | **2.6-3.3x** vs 4.85, **3.3-3.9x** vs gate CPU |
| Metal fused B=1 | 35-41 Gw/s effective | about **7-8x** CPU-rayon primitive |
| 7B token primitive from Metal fused B=1 | ~5.0-5.9 tok/s arithmetic | before attention/KV/norm/sampling |
| Metal prompt B=16 | 227-255 GMAC/s | 7B prompt primitive **~32-36 tok/s** arithmetic |
| command buffer batching | 24 commits 16.03 ms -> 1 commit 10.58 ms on synthetic token loop | **1.5x**, 34% latency cut |
| GPU encode, k=3 L=7 | 27.8 Mw/s vs 5.14 Mw/s CPU canon | **5.4x** local requant velocity |
| GPU encode, k=2 L=12 | 2.825 Mw/s vs 1.195 Mw/s CPU canon | **2.36x** local requant velocity |

Planning estimates:

- Whole-token/grouped command buffers should be worth **20-35% latency reduction** on real
  multi-projection execution if the synthetic overhead carries over.
- Lean side-info entry is likely a **5-15% fused B=1 speed win**, because the 80 B/block
  entry is 43-53% of remaining fused-kernel traffic but the kernel is not purely bandwidth-bound.
- CUDA bitslice on a 3090-class card is arithmetic-estimated at **25-35 tok/s 7B primitive**;
  this must be measured before any claim.
- Wired GPU encode turns local 0.5B iteration from "minutes per arm" toward **2-5x faster**
  depending on rung, which directly increases how many allocator/PV/C2 experiments we can run.

### 5.4 Where quantization tells us the next push

Run more quantization only when it resolves one of these budgets:

1. **C2 microscope quant pass:** measure entropy of scales/sub-scales/outlier positions on
   q2, q3, mp_light, and live 32B shards.
2. **Allocator quant pass:** q3 baseline, down@4, gate/up@4, attention@4, depth-down@4, all
   with PPL, not rel-RMS.
3. **OUTL quant pass:** q2 with residual budgets at 0.25%, 0.5%, 1%, row-bucketed, and
   entropy-coded estimates.
4. **PV quant pass:** WSD cooldown final, then down_proj selective-PV against full-PV.
5. **Speed quant pass:** same artifacts through seek-mode vs stream-mode and lean side-info
   decode gates.

The goal is not to quantize more for its own sake. The goal is to turn every quant run into a
price curve: PPL per bpw, bpw per metadata component, and tokens/s per byte moved.

## 6. Custom engineering lanes

### Lane A - the bit ledger and entropy microscope

Before changing the format, build the measurement tool that decomposes every artifact:

- payload bits
- scale bits
- sub-scale bits
- init-state bits
- outlier side-channel bits
- v2 offset/table bits
- padding/alignment
- provenance/self-description bits
- prepared resident bytes
- GPU traffic bytes per token

Output: per tensor, per class, per model, per rung. This becomes the budget. If a bit is not
visible in the ledger, it will be wasted.

### Lane B - C2 side-info codec

Design:

- `SDSC` or new section version for side-info codec metadata.
- order-0 static CDFs first, then context-by-subscale-position if entropy justifies it.
- integer rANS/CDF only; no float decode state.
- decode fallback to v1 fixed side-info for old artifacts.
- seek-mode optional checkpoints every N blocks so stream-mode can still partial-seek at a
  coarser granularity.

Overengineered version:

- tensor-class CDFs: attention, up/gate/down, embeddings if ever included.
- rung-aware CDFs: q2 and q3 have different scale statistics.
- predictor selection encoded once per tensor.
- macroblock checksums so corruption is local.

### Lane C - STRAND-native training artifact

The current training stack still thinks in HF/bf16 terms. The custom route is:

- bf16 base as a read-only oracle or cold storage, not the main mutable artifact.
- STRAND payload plus trainable delta-shadow for selected tensors.
- identity-skip requant so frozen tensors never churn.
- KD cache that stores only what the student needs.
- checkpoint layout that saves PV shadows, quant payload, skip manifest, and eval ledger together.

Goal: make "train a low-bit model" mean training a STRAND-native object, not repeatedly
falling back to a bf16 worldview.

### Lane D - allocator as a solver

Do not hand-write rung configs. Build a small solver:

- inputs: candidate tensor rungs, measured per-candidate PPL deltas, bpw cost, family priors,
  depth priors, outlier cost, side-info cost.
- objective: minimize PPL under bpw cap, or minimize bpw under PPL cap.
- constraints: deterministic decode, strict row alignment, allowed kernel profiles.

First solver can be a greedy/lambda sweep. Overengineered solver can be dynamic programming
over tensor groups with cross-family priors.

### Lane E - runtime-native model executor

The `.strand` runtime should not be "convert to safetensors, then call normal inference."
It should become:

- load `.strand`
- prepare resident GPU/CPU tensors
- schedule all projection kernels for a token in grouped command buffers
- apply sparse residual/outlier patch
- expose deterministic kernel variants and documented float-accumulation variants

This is where the product stops being a file format and becomes a runtime system.

### Lane F - sub-2-bit training lab

For Regime C, treat codebook, schedule, silence, and objective as one co-designed experiment:

- vec d=2/d=3
- PV with WSD cooldown
- silence entropy regularizer
- live-Fisher Phase 0 only if it passes
- trained seed/basis variants
- C2 side-info from the beginning

Kill rules stay brutal: if PPL stays unusable after training, the runtime still ships the
format support but the claim remains "frontier research," not product.

### Lane G - residual and side-channel lab

Treat every side-channel as a priced quantization family:

- OUTL position codec
- OUTL value codec
- de-bias row vector
- seed/basis index
- seek checkpoints
- provenance/self-description bytes

Each side-channel gets a PPL-per-bpw curve. The allocator can then decide whether spending
0.02 bpw on de-bias beats 0.02 bpw on better scales, outliers, or a tensor rung lift. This is
how STRAND avoids a pile of individually "small" sidecars that collectively erase the low-bit
win.

## 7. Speed and compression are coupled

Compression changes speed once the Q12 materialization path is gone. The Metal bitslice result
made this explicit:

- decode-only: 12.66-15.89 Gw/s, 3.3-3.9x CPU.
- fused B=1: 35-41 Gw/s effective.
- prompt GEMM: B=16 is the M3-class sweet spot at 227-255 GMAC/s.
- B=64 regresses on GPU from register pressure.
- per-tensor command buffers cost about 0.23-0.25 ms and must be batched away.
- the 80 B/block bitslice entry becomes a meaningful fraction of fused-kernel traffic.

Therefore C2/lean side-info is not "just compression." It is also a speed lever.

Speed session order:

1. prepared resident Metal path.
2. whole-token/grouped command buffer dispatch.
3. B=16 prompt tiling.
4. lean side-info entry.
5. CUDA bitslice decode-only, then fused B=1/B=16.
6. rank-convergence measurement before any exact-scan implementation.

## 8. Post-run hardening order

1. **Bank live scale results.** Finish 32B and any 70B run. Mirror JSONs/logs. Update the
   ledger before code changes.
2. **Run the bit ledger.** Measure exact bit spend and side-info entropy on every landed model.
3. **Decide C2.** Implement only if the entropy microscope clears the B/w gate.
4. **Harden Metal runtime.** Whole-token command buffer, prepared resident tensors, B=16 tiling.
5. **Build lean side-info entry.** Only after the bit ledger says what to remove.
6. **Finish PV verdicts.** WSD cooldown, then 0.5B down_proj selective-PV.
7. **Allocator PPL sweep.** Down-protect, gate/up, attention control, depth variants.
8. **CUDA bitslice.** Port once pod is free from scale quantization.
9. **Moonshots.** Rank convergence, basis search, live-Fisher only after their cheap gates pass.

## 9. What not to do

- Do not chase another scalar trellis tweak unless it composes with prepared/paired and passes
  an idle gate.
- Do not call rel-RMS a product win.
- Do not re-run Hessian/diag-H under a new name without live-Fisher Phase 0.
- Do not quote natural silence as a lever; only trained silence counts.
- Do not optimize for seek-mode if the deployment path streams.
- Do not rent large PV before the 0.5B selective-PV concentration gate.
- Do not claim CUDA from arithmetic estimates.
- Do not let bf16 reconstruction become the product path by inertia.

## 10. The claim we are trying to earn

> STRAND is a deterministic, float-free low-bit model system whose 3-bit rung ships today,
> whose 2-bit rung becomes competitive through train-through-the-real-quantizer PV, and whose
> runtime decodes exact integer payloads close to hardware limits while paying near-minimal
> side-info.

That is the bleeding-edge line. The way to earn it is not to worship any standard method,
including bf16, GGUF, uniform bpw, fixed-width side-info, CPU decode, or PTQ. Keep standards
only where gates prove they still deserve to exist.
