# Hawking — End-to-End Workflow: Studio Maximization → Experiments → Proof → Positioning → GTM (2026-06-27)

> **This is one continuous document, on purpose.** It runs the full arc from the
> science (the redundancy hypothesis and the recovery stack) through the bleeding-edge
> experiment bank, the proof artifacts those experiments manufacture, and finally the
> competitive positioning and go-to-market that the proof artifacts unlock. The
> experiments are not an end in themselves — **they exist to manufacture the competitive
> proof.** Part IV (positioning + GTM) was previously a standalone brief
> (`positioning_competition_2026_06_27.md`); it has been absorbed here, expanded with
> fresh 2025–2026 competitive research, so the whole story reads as a single workflow.
> The "no fake GO" rule holds end to end: nothing is claimed as a moat that isn't either
> measured in this repo or checked against the live landscape and marked.

> The dev box is an M3 Pro **18 GB**. Every measured dead-end in this project traces
> back to that ceiling: 7B doctor jobs die at the 6000 MB swap wall, doctor timeouts
> at 120 min, LoRA recovery plateaus because the *real* lever (full-rank recovery)
> never had RAM. The **Mac Studio (M2 Max, 96 GB unified, 38-core GPU, ~400 GB/s,
> 2 TB SSD)** is a 5.3× RAM jump. This plan spends that jump.
>
> **Operating assumption:** the project owns the WHOLE machine. One project at a time,
> at max capability. **Wall-clock time is an explicit NON-constraint** — it is plugged
> in 24/7. We optimize for maximum quality / thoroughness / proof, "the most
> over-engineering possible." **bf16 throughout** (f32 won't fit at scale; 32B f32 =
> 128 GB > 96 GB; 32B bf16 = 64 GB fits). No CUDA — Metal/MPS only.
>
> This is **additive** to `condense_master_plan_2026_06_22.md`,
> `doctor_maximization_plan.md`, and `studio_era_expansion.md`. Where the older docs
> assume an "M3 Ultra / 512 GB" Studio, this plan re-grounds every budget on the
> **actual 96 GB** target and is explicit about where 96 GB STOPS.

---

## Table of contents — the full arc

**Part I — The science (why condensation can win)**
- [§0. What does NOT change (the measured dead-ends)](#0-what-does-not-change-respect-the-measured-dead-ends)
- [§1. The local model ladder on 96 GB — and where it STOPS](#1-the-local-model-ladder-on-96-gb--and-exactly-where-it-stops)
- [§2. The recovery STACK ordered by leverage](#2-the-recovery-stack-ordered-by-leverage-studio-layers-concretely-specified)
- [§3. The DEVICE decision (stay CPU-bf16)](#3-the-device-decision-for-the-7b32b-doctor-stay-cpu-bf16-with-one-mps-exception)
- [§4. THE central experiment — bit-floor-vs-scale (the redundancy hypothesis)](#4-the-central-experiment--the-bit-floor-vs-scale-curve-confirmrefute-the-redundancy-hypothesis)

**Part II — Proof bar & serving (what shippable looks like)**
- [§5. Native `.tq` serving + the RAM-cliff tps bench](#5-native-tq-serving--the-ram-cliff-tps-bench-the-headline-speed-win)
- [§6. Eval gates — the proof bar (no fake GO)](#6-eval-gates--the-proof-bar-no-fake-go)
- [§7. Sequencing (highest leverage first)](#7-sequencing-highest-leverage-first--wall-clock-is-free-so-do-all-of-it)
- [§8. Where 96 GB stops — the off-box tail](#8-where-96-gb-stops--the-off-box-tail-owner-gated-512-gb--rented-cuda)

**Part III — The bleeding-edge experiment bank (manufacturing the proof)**
- [§9. The bleeding-edge experiment bank (T1/T2/T3)](#9-the-bleeding-edge-experiment-bank--maximize-discovery-on-this-box)
- [§10. THE BRIDGE — experiment → proof artifact → competitive move](#10-the-bridge--experiment--proof-artifact--competitive-move)

**Part IV — Competitive positioning & go-to-market (turning proof into a niche)**
- [§11. The brutal verdict to internalize first](#11-the-brutal-verdict-to-internalize-first)
- [§12. The competitive landscape (2025–2026, expanded research)](#12-the-competitive-landscape-20252026--expanded-with-fresh-research)
- [§13. The BYO-model-condensation slot — does the niche survive scrutiny?](#13-the-byo-model-condensation-slot--does-the-niche-survive-scrutiny)
- [§14. The white space, the wedge, and the verdict for a solo builder](#14-the-white-space-the-wedge-and-the-verdict-for-a-solo-builder)
- [§15. Conditions under which leadership is real vs. not](#15-conditions-under-which-leadership-is-real-vs-not)
- [§16. The non-experimentation roadmap (the GTM core deliverable)](#16-the-non-experimentation-competitive-roadmap-the-gtm-core-deliverable)
- [§17. Market timing — the window, and the risk of being lapped](#17-market-timing--the-window-and-the-risk-of-being-lapped)
- [§18. The honest bottom line](#18-the-honest-bottom-line)
- [§19. Final edge pass — leadership mechanics and the 30-day Studio sprint](#19-final-edge-pass--leadership-mechanics-and-the-30-day-studio-sprint)
- [§20. The Proof System — the methodology that makes Hawking hard to dismiss](#20-the-proof-system--the-methodology-that-makes-hawking-hard-to-dismiss)
- [§21. Pre-Studio Scaffolding (do now on M3 Pro) + The Studio Go-Prompt](#21-pre-studio-scaffolding-do-now-on-m3-pro--the-studio-go-prompt)
- [Sources (web-checked 2026-06-27)](#sources-web-checked-2026-06-27)

> **How to read the arc:** Parts I–II are the proven/settled spine. Part III is the
> ambition layer — the experiments. **§10 is the hinge of the whole document:** it maps
> each experiment to the proof artifact it produces and the competitive move that artifact
> unlocks. Part IV then spends those artifacts in the market. **§19 sharpens the leadership
> mechanics; §20 is the capstone — the *proof system* that makes every claim above
> structurally hard to dismiss** (the epistemic contract, the adversarial layer, the
> reproducibility gradient, and the consolidated artifact schemas). Read top to bottom and
> the through-line is: *we run experiment X → it manufactures proof artifact Y → which lets us
> make competitive move Z that no shipping tool can answer — and §20 is the standard that
> says when Y is allowed to count as proof at all.* The full arc:
> hardware → experiments → proof artifacts → positioning → GTM → final edge pass →
> **methodology ascension.**

---

## 0. What does NOT change (respect the measured dead-ends)

These are settled. More RAM does not reopen them — do not burn Studio cycles retrying:

1. **Low-rank LoRA recovery PLATEAUS (~step 25).** The quantization residual is
   *high-rank* noise; a rank-16/32/64 adapter cannot represent it. RAM doesn't fix the
   rank deficiency — it lets us run the *full-rank* alternatives instead. LoRA-KD stays
   only as a cheap **last-mile** polish on top of a full-rank base, never the main heal.
2. **Uniform-proxy STE-QAT through the STRAND trellis is a catastrophic dead-end.** Any
   QAT at scale must be **codec-aware**: sequential per-column STRAND with GPTQ-style
   Hessian error compensation — **NO STE through the trellis.** (`doctor_strand.py` /
   `doctor_blockwise.py` are the codec-aware scaffolds; the uniform `doctor_qat.py` STE
   path is kept only as the disproven baseline.)
3. **AWQ×residual stacking is a measured NON-WIN** (+3.72% vs plain res3+2 +1.4% at the
   same 6.30 bpw on 0.5B). Plain residual is the quality path; do not stack AWQ under it.
4. **Calibration should MATCH the deployment domain, not maximize diversity** (measured:
   domain-matched prose +14.6% beat diverse +17.7% on a prose eval). Per-deployment calib,
   not a diversity contest.
5. **Judge low-bit quality on BIG models, never on 0.5B.** The 0.5B floors ~3-bit and
   lies pessimistically (no redundancy to spend). Always report **effective** bpw
   (baker AGGREGATE: RHT + outlier + residual-pass overhead), **never nominal.**

Everything below is built to be consistent with all five.

---

## 1. The local model ladder on 96 GB — and exactly where it STOPS

Two distinct memory budgets, because doctoring (training) and serving (inference) hit
the wall at very different sizes. bf16 throughout. Rough bf16 weight sizes: 0.5B≈1 GB,
1.5B≈3 GB, 7B≈14 GB, 14B≈28 GB, 32B≈64 GB, 70B≈140 GB.

### 1a. The DOCTOR (training) budget — caps at ~32B

The doctor's peak resident set during deep distillation is roughly:
**student (bf16) + teacher (bf16, for KD logits/features) + optimizer/activation state.**
With the teacher-first caching trick (`doctor_lora.py` already does this: load teacher,
cache top-k logits to CPU, free teacher, then load student) the teacher need not be
co-resident for *logit* KD. But **feature/attention distillation needs the teacher
resident** (you compare hidden states layer-by-layer at train time).

| model | bf16 student | + resident f16 teacher | + opt/act headroom | verdict on 96 GB |
|---|--:|--:|--:|---|
| 0.5B | 1 GB | 2 GB | small | trivial — the **lab** |
| 1.5B | 3 GB | 6 GB | small | trivial — the **lab** |
| 7B | 14 GB | 28 GB | +~20 GB | **comfortable** (was SWAP-BOUND on 18 GB) |
| 14B | 28 GB | 56 GB | +~25 GB | **fits**, the meaty mid |
| 32B | 64 GB | (logit-KD only: cache+free) | +~25 GB | **the ceiling** — full feature-KD with both resident does NOT fit; **logit-KD-only** (teacher cached then freed) fits; residual/codec-native heal (train-free, single model resident) fits |
| 70B+ | 140 GB | — | — | **STOPS. Does not fit to doctor on 96 GB.** |

**Where the local doctor STOPS:** **~32B is the hard local doctoring ceiling on 96 GB**,
and *only* in the lean configuration — full-rank residual + codec-native error-feedback
(one model resident at a time) and **logit-KD** (teacher cached/freed). **Resident
two-model feature/attention distillation tops out around 14B.** Anything 70B+ — and the
whole scale-curve tail (70B → 405B → 671B) — needs an **M3 Ultra / 512 GB box or rented
CUDA**. That tail is exactly where the redundancy hypothesis predicts the biggest win, so
it is a real (owner-gated) extension, not a nice-to-have. See §4 and §8.

### 1b. INFERENCE (serving) budget — caps at ~70B condensed

Serving holds **one** condensed artifact + KV cache, no teacher, no optimizer. This is
where condensation's *whole point* shows: a model that does NOT fit at Q4_K fits low-bit.

| model | Q4_K (~4.5 bpw) | Hawking TQ2 (~2.3 bpw eff) | fits 96 GB at Q4_K? | fits condensed? |
|---|--:|--:|---|---|
| 7B | ~4 GB | ~2 GB | yes | yes (DENSITY, no cliff — both fit) |
| 14B | ~8 GB | ~4 GB | yes | yes (density) |
| 32B | ~18 GB | ~9.6 GB | yes (tight w/ KV) | yes |
| **70B** | **~40 GB** | **~20 GB** | **borderline / swaps w/ long KV** | **yes — the cliff candidate** |
| 405B | ~230 GB | ~115 GB (2-bit) / ~57 GB (1-bit) | **no** | **no at 96 GB** (1-bit-405B = the 512 GB capstone) |

**Where local serving STOPS:** **~70B condensed is the serving ceiling on 96 GB.** A 70B
at TQ2 (~20 GB) serves comfortably; the 405B 1-bit-or-nothing capstone (~57 GB at 1-bit)
is *just* over the line once you add KV + runtime and is the 512 GB / rented-box story.

### 1c. The recommended local ladder (run all of these on the Studio)

```
0.5B   worst-case stress + the fast LoRA-KD/recipe LAB (floors ~3-bit, lies low)
1.5B   second lab rung — confirms recipe transfer before paying for big bakes
7B     the honest mid substrate — 1-bit floor is judged HERE, not on 0.5B
14B    first real redundancy payoff; full feature-KD still fits both models
32B    the local capstone — doctorable (lean) + servable; the headline density model
─────────────────────────────────────────────────────────────────────────────────
70B    SERVE-only locally (the RAM-cliff tps demo); DOCTOR needs 512 GB / CUDA
405B   neither — the 1-bit capstone, entirely off-box (512 GB / rented CUDA)
```

Local models already on disk: `scratch/qwen-05b`, `scratch/qwen-15b`, `scratch/qwen-7b`.
**14B and 32B are owner-gated downloads** (storage on the 2 TB SSD is ample: a 32B bf16
parent ≈ 64 GB + sharded temps; keep disk discipline, GPU jobs sequential).

---

## 2. The recovery STACK ordered by leverage (Studio layers concretely specified)

The doctor is a **stack of composing layers**, cheapest-first. Layers 0–3 are train-free
(forwards + bakes) and already carry the bulk of recovery — they run anywhere, including
the old 18 GB box. **The Studio's unique job is layers 4–6: the training/compute-heavy
layers the 18 GB box could never hold.** Per-stage bit targets and expected effective-bpw
below are anchored to the measured 0.5B numbers and *extrapolated down* per the redundancy
hypothesis (which §5 then tests, not assumes).

| # | layer | recovers | cost | where | tooling |
|---|---|---|---|---|---|
| 0 | **calibration (domain-matched)** | input to every layer below | forwards | anywhere | `calib_build.py` |
| 1 | **AWQ pre-scale (α=0.5)** | halves the raw gap at 3–4-bit | 1 fwd + bake | anywhere | `awq_bake.py` / `awq_plus.py` |
| 2 | **mixed-precision allocation** | spends bits where output hurts | sensitivity scan | anywhere | `mixed_precision.py` |
| 3 | **full-rank residual (plain)** | the train-free ~1:1 breakthrough | double-bake | anywhere | `residual_bake.py` / `residual_tq.py` |
| 4 | **full-rank / layer-wise QAT** | the high-rank residual LoRA can't reach | **training** | **STUDIO** | `doctor_blockwise.py` |
| 5 | **codec-native GPTQ-Hessian error-feedback** | the ceiling-breaker (below where residual pays) | heavy compute | **STUDIO** | `doctor_strand.py` |
| 6 | **deep distillation (logit/feature/attn)** | richer-than-CE teacher signal | 2 models + train | **STUDIO** | KD path in `doctor_lora.py` (extend) |

### Studio layer 4 — full-rank / layer-wise QAT (the #1 Studio unlock)

**This is the direct fix for the LoRA plateau.** LoRA failed because the residual is
high-rank; the answer is to make the **full weight matrix** the trainable object again,
within RAM that the 18 GB box never had.

- **Mechanism:** layer-wise (block-wise) QAT — hold one transformer block's weights as
  trainable bf16, freeze the rest, fine-tune that block against the teacher's block
  output (teacher-forced), re-bake through STRAND every N steps (the "requant-every-N"
  proxy-transfer design), advance to the next block. Memory peak = one block's weights +
  optimizer + the teacher's cached block I/O, **not** the whole model — so even 32B is
  tractable on 96 GB despite "full-rank" because it is full-rank **per block, sequentially.**
- **CRITICAL discipline:** the re-bake is through the **actual STRAND codec** (codec-aware),
  never a uniform STE proxy (dead-end #2). The gradient sees the real trellis error.
- **Bit targets / expected eff-bpw** (per redundancy hypothesis, to be confirmed §5):
  - 7B: push the floor from ~3-bit (PTQ usable) toward **~2.3–2.5 bpw eff at ≤+2% ppl.**
  - 14B: **~2.0 bpw eff** target at ~1:1.
  - 32B: **~1.6–1.8 bpw eff** target at ~1:1 (the redundancy payoff begins to bite).

### Studio layer 5 — codec-native GPTQ-Hessian error-feedback (the ceiling-breaker)

For the sub-residual tier (where residual's +bpw cost is unaffordable, e.g. the 1-bit
edge), retry the GPTQ-family recovery **inside the codec**:

- **Mechanism:** quantize columns of each weight matrix **sequentially**; after quantizing
  column *j*, compute its quantization error and **compensate it into the not-yet-quantized
  columns** using the (activation) Hessian `H = X Xᵀ` — exactly GPTQ/OBQ error feedback,
  but the quantizer in the loop is **STRAND's trellis**, not round-to-nearest. No STE, no
  uniform proxy. The Hessian comes from the domain-matched calib forward.
- **Bit targets:** the 1-bit / ~1.34-bit attempts on 14B–32B where residual can't pay. This
  is the **only** path the project believes can make sub-2-bit hold at scale; it is the
  unbuilt ceiling-breaker explicitly reserved for the Studio.

### Studio layer 6 — deep distillation (logit + feature + attention)

CE is a weak signal (the doctor's own held-out moved ~3% on CE). The literature's low-bit
recovery method is distillation. The Studio holds **both teacher and student resident**:

- **Logit KD** (already cached/streamed in `doctor_lora.py`): match the f16 teacher's full
  (top-k) output distribution. Resident teacher not required (cache+free) → fits to 32B.
- **Feature KD:** match per-layer hidden states (MSE on residual-stream activations).
  **Needs the teacher resident** → fits to ~14B comfortably, 32B only if quantize-on-the-fly.
- **Attention KD:** match attention maps / KV statistics — the richest signal, most RAM.
- Combine: `L = α·KD_logit + β·KD_feature + γ·KD_attn + δ·CE`, swept on the 0.5B/1.5B lab,
  recipe transferred up. Expected: another **0.2–0.4 bpw** of floor at equal quality on
  top of layer 4, per the literature (QuIP#/AQLM/BitNet class).

**Ordering rationale:** always run 0→3 first (cheap, train-free, large). Then layer 4
(full-rank QAT) is the single biggest *new* lever. Layer 6 (deep KD) polishes layer 4's
base. Layer 5 is the targeted ceiling-breaker only at the sub-residual edge.

---

## 3. The DEVICE decision for the 7B–32B doctor: **stay CPU-bf16** (with one MPS exception)

Current doctor runs **CPU-bf16** (`DOCTOR_DEVICE=cpu`, `DOCTOR_DTYPE=bfloat16`,
`STRAND_NO_GPU=1`). The audit ladder note records *why*: **f16 overflows the 7B CPU
forward → NaN**, so bf16 is mandatory on CPU; and CPU keeps the baker from fighting the
model measurement for Metal memory.

**Recommendation: KEEP the doctor on CPU-bf16 for the 7B–32B training runs.** Reasons,
grounded in the actual constraints:

1. **Wall-clock is free.** The single stated reason to prefer MPS — speed — is explicitly
   a non-constraint here. CPU-bf16 is *reliable and exact*; that is worth more than
   throughput when the whole product rests on the recovery number being real.
2. **MPS numerical correctness is not gated for this training path.** MPS bf16 matmul and
   the autograd path through the codec-aware re-bake are untested for QAT here; a silent
   MPS precision artifact would corrupt the headline recovery result. CPU bf16 is the
   reference. Don't move the *load-bearing* number onto an unverified backend to save time
   we're told not to value.
3. **Unified-memory contention.** On 96 GB the GPU and CPU share the same pool. A long QAT
   on MPS competes with the baker (Metal) for buffers; CPU training + `STRAND_NO_GPU=1`
   bakes keep the two from colliding — exactly the isolation the audit ladder was built for.
4. **The M2 Max GPU advantage is for SERVING, not training.** The 38-core GPU's win is the
   native `.tq` bitslice GEMV at inference (§6). Spend Metal there, where it's gated and
   where the tps cliff lives — not on an ungated training loop.

**The one MPS exception:** the **0.5B/1.5B recipe LAB** (§2, layer 4/6 sweeps). It is
small, fast either way, gated against CPU as a parity reference, and used only to *find*
the recipe (rank, lr, KD weights, calib) — not to produce a shipped number. Run the lab on
MPS for iteration speed; **re-confirm the chosen recipe's final number on CPU-bf16**, and
run all 7B–32B production doctoring on CPU-bf16.

### Swap / RAM budget per model size on 96 GB (the wall that's now gone)

| model | doctor peak (lean: logit-KD + full-rank-per-block) | swap risk on 96 GB |
|---|--:|---|
| 0.5B / 1.5B | < 10 GB | none |
| 7B | ~35–45 GB | **none** (was the 18 GB swap-death case) |
| 14B | ~55–70 GB | low — keep one heavy job at a time |
| 32B | ~80–90 GB | **tight** — logit-KD-only, no co-resident teacher, no parallel baker; this is the budget that defines the local ceiling |

Set `DOCTOR_SWAP_CEIL` / `DOCTOR_SWAP_HARD_CEIL` generously (the 18 GB run pinned them at
12000/18000 MB; on 96 GB raise the soft ceiling to ~60 GB and hard to ~80 GB so the
watchdog only fires on genuine 32B over-commit, not on healthy 7B runs). **The 6000 MB
swap-death and 120-min timeouts that are killing the current 7B frontier simply do not
recur below 32B on this box** — that is the headline operational change.

---

## 4. THE central experiment — the bit-floor-vs-scale curve (confirm/refute the redundancy hypothesis)

> **The user's intuition — "larger models quantize to lower bits more easily" — is the
> project's redundancy hypothesis and is currently UNPROVEN.** "Does the floor descend
> with scale?" is the central scientific claim and the headline result. Everything else in
> this plan is infrastructure for *this measurement*.

### The claim, stated falsifiably

For each model, define its **bit-floor** = the lowest *effective* bpw at which the
condensed+recovered artifact holds **≤ +2% output-space ppl degradation vs its own f16
parent** (the ~1:1 gate), confirmed by the multi-eval capability tripwire (§7).

**Hypothesis H1:** `floor(0.5B) > floor(1.5B) > floor(7B) > floor(14B) > floor(32B)` —
the floor descends monotonically with parameter count.
**Null H0:** the floor is flat (~3-bit) regardless of scale — redundancy buys nothing.

### The experiment (concrete, ordered, measurable)

1. **Fixed protocol across the ladder.** Same recipe family at every rung: domain-matched
   calib (L0) → AWQ α=0.5 (L1) → mixed-precision allocation (L2) → full-rank residual (L3)
   → **full-rank/block-wise codec-aware QAT (L4)** → logit-KD polish (L6). Same eval corpus
   methodology (`MULTIWINDOW>1`, mean of ≥4 held-out 2048-tok windows — a single window is
   overfittable). Report **effective** bpw only.
2. **For each model** in {0.5B, 1.5B, 7B, 14B, 32B}: run `audit_ladder.py` (already
   memory-safe + checkpointed) across the bit budgets, then apply the recovery stack, and
   **binary-search the floor**: find the lowest eff-bpw config whose recovered degradation
   crosses +2%. That eff-bpw is the model's floor datapoint.
3. **Plot the curve:** x = log₁₀(params), y = floor eff-bpw, one point per model, with the
   recovered-vs-PTQ gap drawn as the vertical band (the gap *is* the doctor's contribution).
   Overlay the Q4_K line (4.5 bpw) and the ~1:1 / +2% gate.
4. **Decision rule:**
   - **Monotone descent ⇒ H1 confirmed** → the headline: extrapolate to 70B/405B (off-box,
     §8) and publish "bigger models compress harder," the project's wedge.
   - **Flat ⇒ H0** → the redundancy thesis is wrong at these scales; pivot the pitch to
     pure density-at-fixed-floor, and the 1-bit-405B capstone is abandoned. **Report it
     honestly either way — no fake GO.**
5. **Guardrails (why this won't lie):** judged on 7B+ (0.5B reported but never sets the
   verdict — it floors ~3-bit and lies pessimistically); effective bpw always; multi-eval
   capability tripwire gates every "floor held" claim so we measure capability, not ppl
   theater; CPU-bf16 production numbers (no MPS artifact in the headline).

### What the Studio specifically unlocks for this experiment

The 18 GB box could only place the 0.5B and 7B points (and the 7B point is currently
**SWAP-BOUND** — dying at the 6000 MB ceiling and 120-min timeouts, so even *that* point is
not yet clean). The Studio:
- makes the **7B point clean** (no swap death — the conductor parked at "waiting-for-v3"
  can finally complete v3),
- adds the **14B and 32B points** — the two rungs that, if H1 is true, show the descent
  most clearly (the 0.5B↔7B segment alone is too short to distinguish descent from noise),
- gives the **full-rank QAT recovery** that makes each point a real *floor* rather than a
  PTQ approximation.

**This 5-point curve is the deliverable.** The gap between columns is the redundancy
result; that it is even measurable is the entire reason to use the Studio.

---

## 5. Native `.tq` serving + the RAM-cliff tps bench (the headline SPEED win)

Quality and speed are **two separate streams** — never conflate them. §2–4 is Stream A
(quality: the floor + the doctor). This is **Stream B (serve-tps)**, gated on the native
`.tq` serve path.

### The serve path (per `native_tq_serving_impl.md`)

- **Stage A (correctness):** `HAWKING_QWEN_TQ=<path.tq>` → dequant-on-load to f16 → serve.
  Proves the condensed artifact generates coherent text in Hawking; tps LOSES on a fitting
  model (f16 is 2× Q4_K bytes) — expected.
- **Stage B (the win):** `WeightKind::Tq` carrying encoded bytes → native `strand_bitslice.metal`
  decode → GEMV (staged first: decode to f16 tile → existing f16 GEMV; fuse later). Deploy
  invariant `in_features % 256 == 0`; ragged tensors fall back. This serves the artifact at
  its **true low-bit footprint** — the cliff.
- **Residual two-part serve** (`residual_tq.py`): base bitslice + residual bitslice summed
  on-the-fly (kernel exists, parity 2–4e-6 on Qwen-3B). Needed to serve the *residual*
  recipes, which are the ~1:1 quality winners. Top Stream-B priority alongside Stage B.

### Which model demonstrates the cliff on 96 GB

The cliff requires a model that **does NOT fit at Q4_K but DOES fit condensed.** On the
18 GB box that was 32B (Q4_K 18 GB swaps; TQ2 9.6 GB fits). **On 96 GB the bar moves up —
32B Q4_K (~18 GB) fits fine, so it shows DENSITY, not the cliff.**

**The 96 GB cliff model is ~70B:**
- 70B Q4_K ≈ **40 GB** + a long-context KV cache pushes a 96 GB box into swap/pressure
  (and any concurrent workload tips it over).
- 70B TQ2 ≈ **20 GB** fits comfortably with full KV headroom.
- Serve both, same prompt/length: the Q4_K run degrades to SSD-bandwidth-bound tok/s under
  memory pressure (cf. the measured Mixtral ≈0.1 tok/s out-of-core case in MODELS.md) while
  the condensed run stays fully resident → the **10–100× tps cliff** the master plan claims.

**Therefore the headline speed bench needs a 70B parent** (owner-gated download; serve-only
— no doctoring required for the *speed* demo, single-bake TQ2 suffices). The 32B is the
honest *density* demo on this box (both fit); the 70B is the honest *cliff* demo. Be precise
about which claim each model supports — do not call a density win a speed win.

---

## 6. Eval gates — the proof bar (no fake GO)

Three gates, all required before any "condensed ≠ loss" claim ships:

1. **Output-space ppl** — `ppl(condensed)/ppl(f16_parent) − 1`, real forward passes,
   `MULTIWINDOW≥4` held-out windows (single window is overfittable). Effective bpw always.
   - `~1:1` gate: **≤ +2%**.  `beats-llama` gate: **≤ +8%** at fewer bpw than Q4_K's 4.5.
2. **Multi-eval capability tripwire** (`multi_eval.py`) — small downstream tasks
   (qa/cloze/math/code) so "recovered" means *capability* preserved, not just ppl. A
   floor claim is void if ppl is ~1:1 but a capability task collapses. (Measured precedent:
   AWQ-residual at +3.72% ppl held capability identical to f16 — that's the bar shape.)
3. **The recovery ledger** (`recovery_ledger.py`) — per model/tier, record how much **each
   layer** recovered (AWQ −X%, mixed-prec −Y%, residual −Z%, full-rank-QAT −W%, KD −V%).
   Tells us which layer has headroom left instead of guessing. This is how the stack stays
   empirical as it grows.

### The single proof bar (the whole product reduces to this sentence)

> **A condensed artifact at FEWER effective bpw than Q4_K (4.5) holds ≤ +2% ppl vs its f16
> parent, passes the multi-eval tripwire, and RUNS on Hawking (native `.tq` serve).**

Today: density WON (52% smaller); quality is Pareto (denser at ~+31%, not yet parity —
full-rank QAT is the open lever, now RAM-unblocked); the tps cliff needs Stage B + a 70B.
The Studio's job is to convert the Pareto point into a **GO on that sentence** at 14B/32B,
and to place the 5-point bit-floor-vs-scale curve that says whether the floor descends.

---

## 7. Sequencing (highest leverage first — wall-clock is free, so do all of it)

1. **Clean the 7B floor point.** Re-run the parked 7B frontier on the Studio (no swap
   death) → complete v3 → the conductor's "waiting-for-v3" branch advances. First clean
   mid-substrate datapoint.
2. **Build full-rank/block-wise codec-aware QAT to a real number on 7B** (`doctor_blockwise.py`).
   This is the lever the 18 GB box could never run. Confirm it beats the LoRA plateau on
   held-out (CPU-bf16).
3. **Download + doctor 14B and 32B**; place their floor points with the full stack.
4. **Plot the bit-floor-vs-scale curve** (§4) — the headline science. GO/NO-GO on H1.
5. **Native `.tq` Stage A → Stage B + residual two-part serve** (Stream B). Then the **70B
   serve-only cliff bench** (§5) — the headline speed number.
6. **Deep-KD (layer 6) recipe** on the 0.5B/1.5B lab (MPS), transferred up; codec-native
   GPTQ-Hessian error-feedback (layer 5) at the 1-bit edge on 14B/32B.

---

## 8. Where 96 GB stops — the off-box tail (owner-gated 512 GB / rented CUDA)

Be explicit so nothing here over-claims:

- **Doctoring 70B+** — does not fit on 96 GB (70B bf16 = 140 GB before teacher/optimizer).
  Needs **M3 Ultra / 512 GB** or rented CUDA.
- **The scale-curve tail 70B → 405B → 671B** — the rungs where H1 (if confirmed locally on
  0.5B→32B) predicts the *biggest* win (1-bit-405B). Measuring those points is the capstone
  and is entirely off-box.
- **1-bit-405B serving** (~57 GB at 1-bit) — over the 96 GB line once KV + runtime are added;
  the 512 GB capstone, not a local deliverable.

The local Studio's honest scope: **the 0.5B→32B bit-floor curve, full-rank/codec-native
recovery to a GO at 14B/32B, and the 70B serve-only cliff.** That is a complete, publishable
result on its own — and it is exactly the evidence needed to justify renting the big box for
the tail. If H1 holds locally, the tail is the headline; if it doesn't, we saved the rental.

---

## 9. THE BLEEDING-EDGE EXPERIMENT BANK — maximize discovery on this box

> **Mission:** discover the absolute bleeding edge of condensation. §1–8 is the proven
> spine; this is the **ambition layer** — a numbered bank of frontier experiments, each
> tied to a published SOTA it tries to match or beat, each grounded in Hawking's *existing*
> tooling (STRAND codec, the doctor, `mixed_precision.py`, the residual passes, the bitslice
> serve kernel), each with a **falsifiable null** and an explicit **96 GB feasibility** call.
>
> **Discipline carried in unchanged (no fake GO):** every experiment reports **effective**
> bpw (baker AGGREGATE: RHT + outlier + residual + side-info) and **output-space** quality
> (ppl-delta vs the f16 parent **and** the `multi_eval.py` capability tripwire), **never**
> nominal bpw or weight-space RMSE alone. Every low-bit claim is **judged on a BIG model**
> (7B+, ideally 14B/32B), **never** on 0.5B (it floors ~3-bit and lies pessimistically).
> The 0.5B/1.5B lab is only for *finding* recipes (MPS-fast), re-confirmed on CPU-bf16.
>
> **Respect the measured dead-ends (do NOT re-run them):** low-rank LoRA recovery
> (residual is high-rank); uniform-proxy STE-QAT through the trellis (catastrophic);
> AWQ×residual stacking (a non-win); diversity-maximizing calib (domain-matched wins).
> Several experiments below are deliberately the **codec-aware** way to attack the same
> target those dead-ends failed at — that is allowed and is the point.
>
> **Tiering by leverage:** **T1 = run-first-local** (highest leverage, fits the doctor or
> serve budget on 96 GB, mostly extends code that already exists). **T2 = local-but-heavier**
> (fits, but a real build or the 32B/70B edge). **T3 = frontier/cloud** (needs rented CUDA or
> a 512 GB box — the scale tail). Each card tags **[run-now-local]** or **[needs-bigger]**.

### How to read each card

`ID · name` — `hypothesis (the SOTA it chases)` — `method (the tooling it extends)` —
`metric + baseline` — `NULL (what refutes it)` — `feasibility/RAM on 96 GB` — `payoff (bpw
bought back)`. Payoff is stated as **effective bpw at the ~1:1 / +2% gate**, the only number
that ships.

---

### TIER 1 — run-first-local (highest leverage, fits today, extends existing code)

**T1.1 · Codec-native GPTQ-Hessian error-feedback (the ceiling-breaker, finally run)**
- **Hypothesis (chases GPTQ / OBQ / OBC):** sequential per-column quantization with
  second-order error compensation reaches ~1:1 *below* where the residual pass can afford
  to pay — the sub-residual edge (1.3–1.8 bpw). This is layer 5 of §2, never yet run at scale.
- **Method:** extend `doctor_strand.py`. Quantize columns of each linear sequentially; after
  column *j*, push its error into the not-yet-quantized columns via the activation Hessian
  `H = XXᵀ` (X from the domain-matched calib forward, same hook as `awq_bake` /
  `audit_ladder.capture_sigma`). The quantizer in the loop is **STRAND's trellis**, never
  round-to-nearest, never STE. OBC-style per-column ordering by Hessian diagonal.
- **Metric + baseline:** eff-bpw at +2% ppl on 14B/32B vs **plain full-rank residual (L3)**
  at the same gate. Win = same quality at lower eff-bpw, or lower ppl at equal eff-bpw.
- **NULL:** at matched eff-bpw, codec-native error-feedback does **not** beat plain residual
  on held-out ppl **or** the tripwire (i.e. residual already captures the recoverable error).
  If null holds, error-feedback is shelved as a non-win like AWQ×residual was.
- **Feasibility:** **[run-now-local]** — one model resident, Hessian is per-layer
  (`in×in`, e.g. 32B `down_proj` 27k² f32 ≈ 2.9 GB peak, sequential, freed per layer). Fits
  32B on 96 GB CPU-bf16. The single most-reserved Studio lever; run it **first**.
- **Payoff:** if it pays, **~0.3–0.5 bpw** below the residual floor on 14B/32B — the only
  credible path to sub-2-bit ~1:1 without a +bpw residual pass.

**T1.2 · Learned rotation (QuaRot / SpinQuant) vs Hawking's random-Hadamard (RHT)**
- **Hypothesis (chases QuaRot, SpinQuant):** a *learned* orthogonal rotation before the cut
  flattens incoherence better than the random Hadamard transform the baker already applies,
  buying low-bit headroom for free (rotations are weight-space-exact, fold into adjacent
  layers, cost 0 serve bpw).
- **Method:** the baker already does random-Hadamard (RHT, the ~0.65 bpw/pass overhead noted
  in §0/§6). Add a `rotation_search.py` that (a) reproduces QuaRot's fused Hadamard as the
  parity floor, then (b) optimizes a per-block rotation on Cayley/Stiefel manifold against
  output-space recon on calib (SpinQuant's learned-rotation objective), emits the rotation as
  a baker pre-pass. Compare RHT vs QuaRot-fused vs learned, all feeding the *same* STRAND cut.
- **Metric + baseline:** eff-bpw at +2% on 7B/14B with learned rotation vs the **existing RHT**
  at the same gate. Rotation adds ~0 serve bpw, so any quality gain is pure floor.
- **NULL:** learned rotation does **not** beat random-Hadamard by >0.5% ppl at matched
  eff-bpw on 7B+ (i.e. RHT already captures the incoherence win — plausible, RHT is strong).
- **Feasibility:** **[run-now-local]** — rotation fit is small (per-block orthogonal matrices,
  manifold opt on the GPU lab or CPU). 7B/14B trivially fit; 32B fits.
- **Payoff:** **~0.1–0.3 bpw** if learned beats random; **the RHT-is-already-good null is a
  valuable publishable negative** either way.

**T1.3 · BitNet b1.58-style native ternary as a condensation TARGET**
- **Hypothesis (chases BitNet b1.58 / 1-bit LLMs):** a ternary {−1,0,+1} representation
  (~1.58 bpw nominal) of a *pretrained* parent, recovered by the codec-aware doctor, holds
  ~1:1 on 14B/32B — testing whether the redundancy hypothesis lets us *reach* native-ternary
  density on a model that was **not** trained ternary-native.
- **Method:** encode the parent at the STRAND ternary rung (the trellis at its 1-bit-per-step
  level expressing the {−1,0,+1} alphabet), then heal with T1.1 (codec-native error-feedback)
  + logit-KD (layer 6). This is the *condensation* analogue of BitNet — not pretrain-from-
  scratch, but compress-to-ternary-then-restore. Report **eff** bpw (ternary + scales +
  side-info, ~1.6–1.8 real, not the 1.58 nominal).
- **Metric + baseline:** ppl-delta + tripwire on 14B/32B at the ternary rung vs the f16 parent;
  baseline = BitNet's published near-lossless claim and the project's own residual floor.
- **NULL:** condensed-ternary on 14B/32B **collapses** (>+8%, beats-llama gate failed) — i.e.
  ternary density is only reachable by *native* pretraining, not by condensation. (A clean,
  important negative if it lands.)
- **Feasibility:** **[run-now-local]** for 14B/32B (serve + lean doctor). The honest 1-bit
  floor is judged here, never on 0.5B.
- **Payoff:** the headline density prize — **~1.6–1.8 eff bpw at ~1:1 on 32B** would be
  the wedge no shipping tool has.

**T1.4 · Mixed-precision MoE: per-expert bit allocation + cold-expert drop/merge**
- **Hypothesis (chases expert-redundancy literature):** in an MoE, most experts are cold /
  redundant; per-expert mixed precision (hot experts high-bit, cold experts ternary or
  dropped) buys far more than dense mixed-precision because expert redundancy is the *extreme*
  form of the redundancy hypothesis.
- **Method:** extend `mixed_precision.py` (already does output-space sensitivity → water-fill
  → `--mp-config`) with a **per-expert** axis: route the calib corpus, measure per-expert
  activation frequency + output-sensitivity, allocate bits per expert, and test **drop** (zero
  a cold expert, re-route) and **merge** (average two redundant experts) as discrete moves.
  Serve via the existing MoE code paths (`crates/hawking-serve`, DeepSeek-V2-Lite MLA path).
- **Metric + baseline:** eff-bpw (param-weighted across experts) at +2% on DeepSeek-V2-Lite
  (16B, 2.4B active) and Mixtral-8×7B vs uniform-bit MoE at the same avg bpw.
- **NULL:** cold-expert drop/merge or per-expert allocation does **not** beat uniform MoE
  quant at matched avg bpw (experts are *not* redundant enough to exploit) — refutes the
  extreme-redundancy claim for MoE.
- **Feasibility:** **[run-now-local]** for **DeepSeek-V2-Lite** (16B, fits doctor+serve) and
  **Qwen3-MoE-30B-A3B** (serve + lean doctor). **Mixtral-8×7B (~47B)** is **[run-now-local
  serve / needs-bigger doctor]** (47B bf16 = 94 GB, doctor needs the tail). MoE is where the
  biggest local headline lives because active-param footprint ≪ total.
- **Payoff:** potentially **>1 bpw** param-weighted on a sparsely-activated MoE — the single
  highest-ceiling *local* density experiment.

**T1.5 · Joint prune + quant (SparseGPT / Wanda / OWL) as a co-lever for sub-1-bit**
- **Hypothesis (chases SparseGPT, Wanda, OWL):** pruning and quantization compose —
  `bpw ≈ p·(b + log2(1/p))` (the sub-1-bit arithmetic already in `studio_era_expansion.md`),
  so a 2:4 or 50–90% sparse base + STRAND low-bit reaches amortized **<1 bpw** where the bit
  knob alone floors at 1.
- **Method:** add `prune_bake.py` — Wanda salience (`|W|·‖X‖`, calib-cheap) and a SparseGPT
  pass (Hessian-guided, *reuses T1.1's `H = XXᵀ`*) to pick the mask, OWL's outlier-weighted
  layerwise sparsity ratio, then STRAND-quantize survivors. Store mask + survivors; serve via
  a masked variant of the bitslice GEMV.
- **Metric + baseline:** **amortized eff-bpw** (survivors × their bpw + mask side-info) at +2%
  on 7B/14B/32B vs dense STRAND at the same quality. Tripwire mandatory (sparsity can pass ppl
  but drop a capability).
- **NULL:** joint prune+quant gives **no** amortized-bpw win over dense low-bit at matched
  quality on 7B+ (mask side-info eats the sparsity gain) — bounds the sub-1-bit claim.
- **Feasibility:** **[run-now-local]** for 7B/14B/32B (forward + bake + light Hessian).
- **Payoff:** the **surest sub-1-bit path** per the MDL argument — potentially **0.5 bpw**
  amortized on a big tolerant model, grading a few % off strict 1:1 (the novel research offer).

---

### TIER 2 — local but heavier (a real build, or the 32B/70B edge)

**T2.1 · QTIP-style trellis-coded quant — the DIRECT STRAND head-to-head**
- **Hypothesis (chases QTIP):** QTIP's trellis-coded quantization is the closest published
  cousin of STRAND; a like-for-like bench says whether STRAND's trellis is competitive at the
  2-bit frontier and, if QTIP wins, *what* it does differently (codebook shape, bitshift
  trellis, Viterbi search depth) that STRAND can adopt.
- **Method:** stand up QTIP as an external baseline (its reference encoder) on the *same*
  14B/32B parents, *same* calib, *same* output-space harness (`ppl_bench.py` + `multi_eval.py`).
  Diff against STRAND at matched eff-bpw. Port any QTIP trellis trick that wins back into the
  STRAND codec as a config.
- **Metric + baseline:** eff-bpw at +2% on 14B/32B: STRAND vs QTIP, head-to-head.
- **NULL:** STRAND is **within noise** of QTIP at matched eff-bpw (STRAND is already SOTA-class
  trellis — the desired result), OR QTIP wins and the gap is the to-do list.
- **Feasibility:** **[run-now-local]** to bench (encode is offline, serve f16 for the quality
  number); the **port** of any winning trick is the build cost.
- **Payoff:** either **validates STRAND is frontier** (publishable) or hands a concrete
  **~0.1–0.4 bpw** codec upgrade.

**T2.2 · QuIP# + AQLM lattice/codebook bake-off vs STRAND**
- **Hypothesis (chases QuIP#, AQLM):** incoherence-processing + E8-lattice (QuIP#) and
  additive/vector codebook quant (AQLM) are the other two 2-bit SOTA families; STRAND should
  be benched against both to know its true rank and to harvest their wins (QuIP#'s lattice,
  AQLM's multi-codebook additive structure — note AQLM's additive idea is *residual-shaped*,
  which Hawking already exploits).
- **Method:** external QuIP# and AQLM baselines on the same 14B/32B + harness as T2.1. For
  AQLM specifically, test whether its additive codebooks beat Hawking's plain residual pass
  at matched eff-bpw (a sharper version of the residual question).
- **Metric + baseline:** eff-bpw at +2% on 14B/32B: STRAND vs QuIP# vs AQLM.
- **NULL:** STRAND ties or beats both at matched eff-bpw on big models (best case), OR a
  specific family wins on a specific tensor class (then mixed-codec allocation is the move).
- **Feasibility:** **[run-now-local]** to bench (offline encode + f16 serve for quality).
  AQLM/QuIP# fitting at 32B is heavy but offline and fits.
- **Payoff:** the **competitive map** — tells us exactly where STRAND stands among 2-bit SOTA;
  any harvested trick is **~0.1–0.3 bpw**.

**T2.3 · W4A4 / W2A8 activation + KV-cache quant (KVQuant) — the long-context RAM arm**
- **Hypothesis (chases KVQuant, Atom, QuaRot-A):** weights aren't the only RAM cliff —
  the **KV cache** dominates at long context. Quantizing activations and KV (W4A4, W2A8,
  KV at 2–4-bit) is the long-context arm of the density story; pair it with a needle-in-a-
  haystack (NIAH) eval to prove condensed models *keep* long-context retrieval.
- **Method:** add activation + KV quant to the serve path (per-token/per-channel act scales,
  KVQuant's non-uniform KV codebook with the rotation from T1.2 to tame KV outliers). Eval on
  **NIAH at 8k–32k** on a condensed 14B/32B, reporting retrieval accuracy vs the f16 parent.
- **Metric + baseline:** NIAH accuracy + ppl at long context, W-only vs W+A+KV, vs f16.
- **NULL:** activation/KV quant **collapses** NIAH retrieval (>5% drop) at W2A8 / KV2 on
  14B+ — i.e. the long-context arm needs higher act/KV bits than weights.
- **Feasibility:** **[run-now-local]** to eval and prototype (serve-side, fits 14B/32B + long
  KV in 96 GB — that headroom is *why* 96 GB matters here). Full fused W4A4 GEMV kernel is the
  **build**; staged dequant first (mirror the §5 bitslice staging plan).
- **Payoff:** unlocks **long-context serving in a fraction of the KV RAM** — the arm of the
  cliff §5 doesn't cover; multiplies the effective context the Studio can hold.

**T2.4 · Self-speculative / Medusa / Eagle draft heads on a condensed model (tps ON TOP of density)**
- **Hypothesis (chases Medusa, EAGLE, self-speculation):** graft lightweight draft heads onto
  the *condensed* model so it speculates against itself — stacking a **2–3× tps** decode win
  **on top of** the density/RAM-cliff win, with no quality loss (verification is exact).
- **Method:** train Eagle/Medusa draft heads (small, fit the lab) against the condensed 14B/32B
  as the verifier; serve via the bitslice GEMV with a speculative-decode loop. The draft heads
  are tiny f16 — negligible bpw — and the density win already fits the verifier in RAM.
- **Metric + baseline:** end-to-end tps on condensed-14B/32B with vs without draft heads, at
  identical greedy output (exact-match gate).
- **NULL:** draft acceptance rate on the condensed model is **too low** to net a tps win (low-bit
  weights make the self-draft a poor predictor) — a real risk worth measuring.
- **Feasibility:** **[run-now-local]** — draft-head training fits the lab; serve fits. The
  speculative-decode serve loop is the build.
- **Payoff:** **2–3× tps** multiplicatively on top of the cliff — converts a density win into
  a *speed* win **even on models that already fit** (where §5's cliff doesn't apply).

**T2.5 · Self-generated, domain-matched calibration + bigger-teacher distillation (S4)**
- **Hypothesis (chases self-distillation + the calib-matters finding):** the model writing its
  **own** domain-matched calibration beats any fixed corpus (extends the measured win that
  domain-matched > diverse), and a **bigger teacher** distilling the condensed student
  (true KD, not self-distill) recovers more than same-size logit-KD.
- **Method:** (a) self-calib — sample the f16 parent to generate calib in the deployment domain,
  feed `calib_build.py`; ablate self-calib vs corpus. (b) S4 — use a larger parent (e.g. 32B
  teaching a condensed 14B) for logit+feature KD via the layer-6 path. Both respect dead-end #4
  (match the domain, don't maximize diversity).
- **Metric + baseline:** recovery delta (recovery_ledger) from self-calib vs corpus-calib; and
  big-teacher-KD vs same-size-KD, at matched eff-bpw on 14B.
- **NULL:** self-calib does **not** beat a good domain corpus (the parent can't generate
  better-than-real calib), and big-teacher-KD does **not** beat same-size KD at the +2% gate.
- **Feasibility:** self-calib **[run-now-local]**; **big-teacher KD is [needs-bigger] at the
  32B-teaches-14B config** (two resident models > 96 GB) — runs only as 14B-teaches-7B locally,
  full version on the tail box.
- **Payoff:** **~0.1–0.3 bpw** of floor from richer calib + teacher signal on top of the stack.

---

### TIER 3 — frontier / cloud (the scale tail; needs rented CUDA or 512 GB)

**T3.1 · Fit the BIT-FLOOR-vs-SCALE law and PREDICT the 70B / 405B floor (the headline science)**
- **Hypothesis (the project's redundancy thesis, made quantitative):** the per-model floor
  from §4 is not just monotone — it follows a **power law in params**, `floor(N) ≈ a·N^(−β) + c`,
  so the 5 local points (0.5B→32B) **predict** the 70B/405B floor before we ever rent the box.
  This operationalizes the MDL argument (`floor = MDL(function)/n_weights`, shrinks with N).
- **Method:** take the §4 five-point curve, regress `floor` vs `log N`, fit `β` with a
  confidence band, and **state the falsifiable prediction**: e.g. "if β holds, 70B floors at
  X eff-bpw and 405B at Y < 1 bpw." Then the rented run at 70B is a **pre-registered test**, not
  an exploration.
- **Metric + baseline:** predicted vs measured floor at 70B (the first off-box point); null is
  the curve being **flat** (H0 from §4 — redundancy buys nothing) or **non-power-law** (the
  extrapolation to 405B is unjustified).
- **NULL:** the 70B measured floor falls **outside** the predicted band → the scaling law is
  wrong and the 1-bit-405B headline is not supported by extrapolation. **Report honestly.**
- **Feasibility:** **fit is [run-now-local]** (it's a regression on the 5 local points — do it
  the moment §4 lands); the **70B/405B confirmation points are [needs-bigger]** (rented CUDA /
  512 GB). This is the cheapest possible bridge from local evidence to the capstone claim.
- **Payoff:** a **predictive law** is the strongest possible form of the thesis — it turns
  "bigger compresses harder" from a slope into an equation with a pre-registered 70B test.

**T3.2 · The sub-1-bit-at-scale viability test (1-bit / ~1.34-bit on 70B+, the capstone)**
- **Hypothesis (the project's own frontier):** at 70B→405B the codec-aware doctor (T1.1
  error-feedback + T1.5 sparsity, no affordable residual pass) holds usable quality at **≤1
  eff-bpw** — the 1-bit-405B fit story (~57 GB at 1-bit fits a 512 GB box, over the line on 96).
- **Method:** the §2 layer-5 + T1.5 stack at the ternary/1-bit rung on 70B+, on the tail box.
  Single-bake (no residual) so it stays servable; block-wise full-rank QAT retried **at scale**
  (it failed on 0.5B's trellis — the redundancy hypothesis says big models may behave
  differently; **test, don't assume**, and do not re-run it on small models).
- **Metric + baseline:** ppl + tripwire at ≤1 eff-bpw on 70B/405B vs the f16 parent and the
  T3.1 predicted floor.
- **NULL:** sub-1-bit **collapses** at 70B+ even doctor-assisted → the 1-bit capstone is
  abandoned, the local 0.5B→32B curve stands alone as the result (still publishable).
- **Feasibility:** **[needs-bigger]** entirely — 70B bf16 = 140 GB to doctor, off-box.
- **Payoff:** the capstone — **1-bit-405B that runs** is the single most striking claim the
  whole program can make; gated on T3.1's law holding first.

**T3.3 · The full multi-family scale tail (70B → 405B → 671B, dense + MoE)**
- **Hypothesis:** the descent and the MoE expert-redundancy win both **steepen** at the
  extreme tail (671B MoE has the most redundancy of all).
- **Method:** run the §4 protocol + T1.4 MoE allocation across the rented tail; the MoE arm
  (DeepSeek-V3-class) is where T1.4's per-expert drop/merge should pay the most.
- **Metric/NULL/feasibility:** as §4/§8; **[needs-bigger]** entirely. NULL = the tail floor
  flattens (redundancy saturates) rather than continuing to descend.
- **Payoff:** completes the curve into the regime where the thesis predicts its biggest wins.

---

### The flagship deliverable — the 2-bit-near-lossless 32B QUALITY CARD

The wedge no shipping tool has, assembled from the bank above. One card, one model, every
number honest:

```
PARENT:        Qwen-32B (or equivalent), f16
ARTIFACT:      STRAND condensed, recipe = [AWQ α=0.5 · mixed-prec · plain residual ·
               codec-native error-feedback (T1.1) · logit-KD], CPU-bf16 production
EFFECTIVE BPW: <target ~2.0, reported as baker AGGREGATE — RHT + outlier + residual + side-info>
QUALITY:       ppl-delta vs f16 parent ≤ +2%   (MULTIWINDOW≥4 held-out windows)
CAPABILITY:    multi_eval.py tripwire — qa/cloze/math/code each ≥ f16 − ε  (no ppl theater)
% PARENT KEPT: derived from the tripwire aggregate
RAM FIT:       artifact + KV in 96 GB with headroom (vs Q4_K ~18 GB — DENSITY win on this box)
TPS:           native .tq bitslice serve, optionally ×Medusa/Eagle (T2.4) for the stacked win
```

This card is the §6 proof bar instantiated at the **redundancy-payoff** model size. If T1.1
+ T1.3 land it, **2-bit-near-lossless 32B that runs on a 96 GB box** is the deliverable.

---

### Prioritization — what to run first (leverage × cost on THIS box)

| rank | experiment | tier | local? | why first |
|---|---|---|---|---|
| 1 | **T1.1** codec-native error-feedback | T1 | run-now-local | the reserved ceiling-breaker; below the residual floor; one model resident |
| 2 | **T1.4** MoE per-expert mixed-prec | T1 | run-now-local | highest *local* density ceiling (active params ≪ total); extends `mixed_precision.py` |
| 3 | **T1.3** condensed native-ternary | T1 | run-now-local | the headline ~1.6 bpw density prize, judged on 32B |
| 4 | **T1.5** joint prune + quant | T1 | run-now-local | the surest sub-1-bit path (MDL); reuses T1.1's Hessian |
| 5 | **T3.1** the scaling-LAW fit | T3-fit | run-now-local | turns the §4 curve into a predictive equation the moment §4 lands |
| 6 | T1.2 learned rotation; T2.1 QTIP h2h | T1/T2 | run-now-local | free floor + the codec-competitiveness map |
| 7 | T2.3 KV/act quant + NIAH; T2.4 spec-decode | T2 | run-now-local | the long-context + tps arms |
| 8 | T3.2 / T3.3 sub-1-bit + scale tail | T3 | needs-bigger | the capstone, gated on T3.1's law |

**The single most likely bleeding-edge WIN:** **T1.1 (codec-native GPTQ-Hessian error-feedback)
on 14B/32B** — it is the one reserved lever that attacks the sub-residual edge the *right*
(codec-aware) way, fits 96 GB with one model resident, and, if it pays its expected ~0.3–0.5
bpw, is what makes the **2-bit-near-lossless 32B quality card** real. T1.4 (MoE) is the
highest-*ceiling* local shot; T1.1 is the highest-*probability* one.

---

### BONUS — cross-pollination (downstream consumer)

Hawking could **condense Babel's Stage-B repair LLM** (a 3–8B model): Babel ships a small
repair model that runs alongside its core, so a condensed Stage-B (2–3 bit, ~1:1) directly
buys Babel RAM/tps headroom. It is a clean **downstream consumer** of this bank's 3–8B
recipes — note it, scope it later; not a Hawking experiment in itself.

---

## 10. THE BRIDGE — experiment → proof artifact → competitive move

> **This section is the spine of the whole document.** Everything above (§1–§9) is the
> machinery; everything below (§11–§18) is the market. This is the hinge that makes the two
> halves one continuous workflow. **The experiments are not curiosity — each one exists to
> manufacture a specific PROOF ARTIFACT, and each proof artifact unlocks a specific
> COMPETITIVE MOVE** (a benchmark, a demo, a launch claim) that a competitor in §12 *cannot
> answer*. Read a row left-to-right: *run this → get this number → make this move.*
>
> **Honesty carried across the bridge (no fake GO):** a competitive move is "armed" ONLY
> when its proof artifact has actually cleared the §6 proof bar (effective bpw, ≤+2% ppl
> vs f16 parent, multi-eval tripwire, runs on Hawking). Until then the move is *contingent*,
> not a claim. Several of these artifacts do not exist yet — the table is the plan for
> manufacturing them, with the brutal current state (`STATUS`) called out per row.

### The bridge table

| Experiment (§9) | → Proof artifact it manufactures | → Competitive move it arms | Move targets (from §12) | STATUS (today) |
|---|---|---|---|---|
| **T1.1** codec-native GPTQ-Hessian error-feedback | **Sub-residual floor delta**: 14B/32B at ~0.3–0.5 bpw below the residual floor, ≤+2% ppl, tripwire-passed | "We reach near-lossless **below** where PTQ/residual can pay" — the recovery-depth claim | GGUF/Unsloth (no gradient recovery), EXL3/PonyExl3 (no recovery) | **Unbuilt** — the #1 reserved lever; highest-probability win |
| **T1.3** condensed native-ternary | **The flagship 2-bit-near-lossless 32B quality card** (~1.6–1.8 eff bpw, ≤+2%, tripwire) | The headline launch artifact: "your own 32B at ~2-bit, near-lossless, runs on a Mac" | BitNet (vendor-only, 2B max, no BYO), Gemma-QAT (vendor model only) | **Unbuilt** — the wedge no shipping tool has |
| **T1.4** MoE per-expert mixed-precision | **MoE density card**: DeepSeek-V2-Lite / Qwen3-MoE at >1 bpw param-weighted saving, ≤+2% | "Condense **your** MoE by exploiting cold-expert redundancy" — highest local density ceiling | No shipping tool does per-expert recovery-aware allocation | **Unbuilt** — highest-ceiling local shot |
| **T1.5** joint prune+quant | **Amortized sub-1-bit card** (~0.5 bpw, graded few-% off 1:1, tripwire-bounded) | "The smallest near-usable form of a big tolerant model" — the novel-research offer | HQQ/AWQ/GPTQ floor ~2-bit; none amortize prune+quant to <1 bpw with recovery | **Unbuilt** — surest sub-1-bit path (MDL) |
| **§4 / T3.1** bit-floor-vs-scale curve + scaling law | **The 5-point descent curve** (0.5B→32B) + a **power-law fit** with a pre-registered 70B prediction | "Bigger models compress harder" — the **scientific** wedge, stated as an equation | Nobody has published this curve as a product claim | **Partial** — 0.5B placed; 7B swap-bound; 14B/32B owner-gated |
| **§5 / native `.tq` + 70B** | **The RAM-cliff tps bench**: 70B that swaps at Q4_K but serves resident condensed | "The model that **doesn't fit** at Q4 — runs here" — the money demo (P0) | llama.cpp OOM/swap; PonyExl3 in-core only | **Half-built** — `.tq` decode test-only; needs Stage B + 70B |
| **T2.1/T2.2** QTIP / QuIP# / AQLM bake-off | **The competitive codec map**: STRAND vs the 2-bit SOTA families, head-to-head, same harness | "STRAND is frontier-class" (publishable) OR a concrete codec to-do list | EXL3/QTIP, QuIP#, AQLM (the codec-research incumbents) | **Unbuilt** — bench-then-port |
| **T2.3** KV/act quant + NIAH | **Long-context density card**: condensed 14B/32B keeps NIAH retrieval at fractional KV RAM | "Condense the **KV cliff**, not just weights" — the long-context arm | None of the local engines pair recovery with KV-quant + NIAH proof | **Unbuilt** — staged dequant first |
| **T2.4** self-spec / Medusa / Eagle | **Stacked-speed card**: 2–3× tps on the *condensed* model at exact-match output | "Density **and** speed on models that already fit" — converts density to a speed story | llama.cpp owns iso-quant decode; this stacks on top of density | **Unbuilt** — draft-head + spec-decode loop |
| **§4 + tripwire (the recovery ledger)** | **The reproducible eval harness + per-layer recovery ledger** | "Rerun our number yourself" — the independent-benchmarkability move (P1) | The whole community treats unreproducible wins as lies | **Partial** — harness lineage exists; needs public packaging |

### Reading the bridge as one narrative

The dependency chain is explicit and ordered (it mirrors §7's sequencing):

1. **Place the curve (§4) and clean the 7B point** → manufactures the *descent-curve* artifact → arms the **scientific wedge** ("bigger compresses harder").
2. **Run T1.1, then T1.3** → manufactures the *2-bit-near-lossless 32B quality card* → arms the **headline launch claim** and the **recovery-depth claim** that GGUF/Unsloth/EXL3 structurally cannot answer (none do gradient recovery).
3. **Wire native `.tq` + a 70B (§5)** → manufactures the *RAM-cliff demo* → arms the **P0 money demo** ("the 32B/70B that doesn't fit at Q4 — runs here").
4. **Publish the harness + ledger** → manufactures *independent reproducibility* → arms **P1** (own the eval), without which every claim above is "treated as a lie."
5. **T1.4 / T1.5 / T2.\*** are the *ceiling-expanders* — they widen the wedge (MoE, sub-1-bit, long-context, stacked speed) once the spine artifacts exist.

**The one-line throughline:** *Hawking's experiments manufacture exactly the proof artifacts
that Part IV needs to spend in the market — and the artifacts that matter most (T1.1 → the
32B quality card, and §5 → the RAM-cliff demo) are precisely the ones no competitor's
architecture can produce, because they require **gradient recovery into the deployment codec,
out-of-core, on Apple Silicon** all at once.* That intersection is the wedge (§14); the bridge
is how the lab work becomes the wedge.

> **Brutal honesty checkpoint (do not skip):** every artifact in the bridge marked
> *Unbuilt* / *Half-built* is a **promise, not a proof**. As of 2026-06-27 the only
> *measured* competitive fact Hawking owns is **density (~52% smaller)**; Hawking **loses
> iso-quant** and the **quality win is unproven at scale**. The bridge says what the
> experiments *would* arm — it does not pre-declare a GO. No row graduates from "contingent"
> to "claim" until its artifact clears the §6 proof bar on a 14B/32B model.

---
---

# Part IV — Competitive Positioning & Go-To-Market

> **Absorbed from `positioning_competition_2026_06_27.md` (now deleted) and expanded with
> fresh 2025–2026 web research.** This Part is about **competition and non-research
> execution**, not code. It spends the proof artifacts that Part III manufactures (via the
> §10 bridge). Same "no fake GO" rule: nothing here is asserted as a moat that isn't either
> (a) measured in this repo or (b) checked against the live 2025–2026 landscape and marked.
>
> **Reading note on sourcing.** Competitor facts below were web-checked on **2026-06-27**
> (URLs in [Sources](#sources-web-checked-2026-06-27)). Where a claim relies on model
> knowledge without a fresh source, it is tagged **[UNVERIFIED]** — treat those as
> hypotheses to confirm before any public README. The competitive picture moves monthly.

---

## 11. The brutal verdict to internalize first

Hawking's own docs already deliver the verdict, so this Part does not soften it:

- **Iso-quant inference is a permanent loss** (~0.71× llama.cpp decode). Do not market speed-of-decode.
- **At PTQ, Hawking loses on quality, badly** (TQ2 collapses; TQ3 ~+44% vs Q4_K ~+2%).
- **Density is the only thing currently WON** (~52% smaller).
- **The quality win is UNPROVEN at scale** and gated entirely on STRAND-aware QAT/KD recovery (Part I §2 layers 4–6; Part III T1.1/T1.3).
- **Native low-bit serving is half-built** (the `.tq` decoder is test-only; rehydrate inflates to f16).

Everything below assumes those five facts are true. A positioning that pretends otherwise
gets torn apart on day one of a Show HN thread. **This is why §10 exists:** the only way out
of this verdict is to manufacture the proof artifacts that convert "unproven" into
"independently benchmarkable," and the only artifacts worth building are the ones a
competitor's architecture cannot also produce.

---

## 12. The competitive landscape (2025–2026) — expanded with fresh research

Split the field the way the *market* sees it: **engines** (run a model) and **compressors**
(make the artifact). Hawking tries to be both — the source of both its differentiation and
its over-extension risk.

### 12a. Local inference engines

| Tool | What it is | Apple Silicon | Dynamic per-model condensation + recovery? |
|---|---|---|---|
| **llama.cpp** | De-facto standard. GGUF, Metal backend, OpenAI-compatible `llama-server`, huge ecosystem. | First-class Metal | **No.** `llama-quantize` is static PTQ (K-quants/IQ); no gradient recovery, no per-model search. Supports **mmap out-of-core** (OS pages weights on demand). |
| **MLX / mlx-lm** | Apple's own framework; built for UMA. **By 2026 it has pulled decisively ahead of llama.cpp's Metal backend — ~30–40% faster on M5**, ~4,800 pre-converted models on HF `mlx-community`, Ollama now ships an MLX backend (preview). The only mainstream on-device fine-tune/QAT/LoRA path. | Native (Apple) | **Partially, and the gap is closing.** `mlx-lm` has learned quantization + QAT (4–16 bit), LoRA, and now **AWQ-style and mixed-bit recipes** (e.g. `mlx-optiq` per-layer KL-div bit selection). Not a one-command "condense to floor + recover" product, but the *Apple-native primitives now exist and are maturing fast.* **The single most dangerous competitor for the recovery wedge.** |
| **Ollama** | UX/distribution over llama.cpp; **now MLX-backed on Apple Silicon (preview).** Owns "easy local model" mindshare. | Yes (llama.cpp/MLX) | **No.** Consumes others' quants. |
| **LM Studio** | GUI; runs GGUF and MLX per-model. Distribution/discovery surface. | Yes | **No.** Consumer of artifacts. |
| **MLC-LLM / TVM** | Compiler-based; low TTFT; cross-platform. Identified (with MLX) as production-ready on M2 Ultra. | Yes (Metal) | **No** recovery; static compile-time quant. |
| **vLLM** | Server-grade throughput/batching, datacenter GPUs (now Red Hat/Neural Magic's stack). | Not the Mac story | No. Ignore for Mac positioning. |
| **ExLlamaV2/V3 (EXL2/EXL3)** | turboderp's quant+inference lib. **EXL3 = QTIP-trellis** at 1–8 bpw, SOTA low-bit quality on consumer GPUs. | **NVIDIA-only (official).** | **Quantizes (trellis) but no gradient recovery loop.** Closest *technical* analog to STRAND — and CUDA-only. |
| **PonyExl3** | ⚠️ **Third-party port of EXL3 to Apple Silicon/Metal.** HF→EXL3 converter *and* on-the-fly trellis decode in fused Metal GEMV/GEMM. ~45★, v0.3 (June 2026), CUDA-parity validated. | **Yes — Metal, trellis, native low-bit serving.** | **No recovery/QAT. In-core only.** The single biggest threat to the "native trellis serving on Mac" half of Hawking's claim — *someone already shipped it.* |
| **AirLLM** | ⚠️ **Layer-by-layer out-of-core inference** — processes one transformer layer at a time, the rest on disk/RAM; runs a 70B on modest hardware. Supports Llama-2/3, Mistral, Mixtral, Falcon (early-2026). | Python; runs on Mac | **No** recovery, **no** trellis. Pure out-of-core *serving* at a heavy tps cost. **This is the most direct prior art for the out-of-core angle — see §13d.** |

**Engine takeaway:** "Trellis low-bit decode on Apple Silicon Metal" is **no longer
unclaimed white space** — PonyExl3 occupies it. "Out-of-core *serving*" is **also not
white space** — llama.cpp mmap and AirLLM both do it (badly, but they do it). Hawking
cannot lead on "we run 2-bit trellis models on a Mac" or on "we page a big model from
disk." It can only lead on **how the artifact was made** (gradient recovery into the codec)
and on **out-of-core *condensation*** (producing the artifact, not just serving it). Verify
PonyExl3's and AirLLM's exact capabilities before any "first/only" claim.

### 12b. Quantization / compression methods & tools

| Method/tool | What it does | Recovery (gradient)? | Dynamic per-tensor bits? | Out-of-core? | Apple-native? |
|---|---|---|---|---|---|
| **GGUF + llama-quantize** | Static K-quant/IQ PTQ; fixed-recipe mixed-precision blocks | No | Recipe-fixed | Yes (CPU/mmap) | via llama.cpp |
| **Unsloth Dynamic 2.0** | ⚠️ **Per-layer mixed-precision, model-specific recipe**; lowest PPL/KL among GGUFs; ships on HF, runs in llama.cpp | No (PTQ + imatrix) | **Yes — "dynamic bits by importance," already shipping** | Yes | Runs on Mac via llama.cpp |
| **AWQ** | Activation-aware scaling; great 4-bit, weak ≤3-bit. Now also available **inside MLX** as a recipe. | No | No | GPU calib | **via MLX now** |
| **GPTQ / GPTQModel** | One-shot Hessian PTQ; 3–4 bit sweet spot | No | No | GPU | CPU path exists |
| **HQQ** | Fast data-free PTQ; 2-bit "works" on 70B (≈fp Llama-13B) but not ~1:1 of its parent | No (optional LoRA after) | No | Low-mem-ish | Some MPS |
| **AQLM** | Additive/vector quant; strong 2-bit **with fine-tuning**; heavy to produce | Yes (global FT) | Codebook | GPU, expensive | No |
| **QuIP# / QTIP** | Incoherence + lattice/**trellis** codebooks; SOTA ≤2-bit PTQ | Optional FT | No | GPU | No |
| **bitsandbytes** | On-the-fly 4/8-bit for train/serve | No | No | No | No |
| **SqueezeLLM** | Sensitivity-weighted + dense-and-sparse | No | Sensitivity-based | GPU | No |
| **BitNet b1.58** | **Native 1.58-bit pretraining** (not PTQ); ~fp at 2B/4T. **bitnet.cpp** = official 1-bit inference. ⚠️ **Largest downloadable native-ternary model is still only 2.4B** — must train from scratch (QAT, 4T tokens); FP16 models cannot be converted. 7B/13B/70B exist only as *eval* numbers, not released weights. | N/A (trained low) | N/A | N/A | via bitnet.cpp |
| **ParetoQ (Meta, 2025)** | ⚠️ First unified 1-bit/1.58/2/3/4-bit **QAT** framework; ternary 600M beats prior 3B ternary SOTA; pins the 2-bit "learning transition." Research, not a product. | **Yes (QAT)** | per bit-width | GPU | No |
| **Gemma 3/4 QAT (Google)** | **Vendor ships QAT int4 checkpoints**, ~bf16 quality, 3× less memory, GGUF on HF | Yes (Google's compute) | No | N/A | runs in llama.cpp on Mac |
| **MLX-LM QAT/LoRA** | Apple-native QAT (4–16 bit) + LoRA on-device; AWQ/mixed-bit recipes maturing | **Yes** | group/per-tensor | partial | **Yes** |
| **Pruna AI** | ⚠️ Commercial+OSS **"smash any model" compressor**: 50+ algos (quant, prune, distill, **recovery**, caching), agentic config search, eval. Open-sourced Mar 2025; ~$2.8M rev, image/video-and-cloud focus. **See §13 — the closest BYO-condensation player.** | **Yes (has a "recovery" family)** | yes (config search) | **No (not advertised)** | **No — officially Linux; Mac is algo-dependent** |

### 12c. The compressor takeaway

Three structural facts the table makes concrete, each load-bearing for §13–§14:

1. **Recovery exists, but always GPU/datacenter or vendor-locked.** AQLM, ParetoQ, Gemma-QAT, MLX-QAT and Pruna's "recovery" family all do gradient recovery — but none do it *into a trellis codec*, *out-of-core*, *on Apple Silicon*, as *one verb*. The recovery primitive is commoditizing; the **specific intersection** is not.
2. **Dynamic per-tensor bits is fully shipped (Unsloth, MLX-optiq).** Hawking gets **zero** differentiation from "mixed precision" alone. That axis is closed.
3. **Native sub-2-bit is real but vendor-gated.** BitNet/ParetoQ prove ~2-bit-near-lossless is achievable — but only by *training low from scratch*. **You still cannot bring your own pretrained 32B and get it to ~2-bit near-lossless.** That conversion is the gap (§13b).

---

## 13. The BYO-model-condensation slot — does the niche survive scrutiny?

This is the question the whole positioning rests on, and the one the fresh research was
aimed at: **is there a commercial "condense YOUR model" player who already owns the slot
Hawking wants?** The answer, after checking the obvious candidates, is **the slot survives —
narrowly — but it is narrower and more contested than the original brief implied.** Here is
the evidence, candidate by candidate.

### 13a. Pruna AI — the closest commercial BYO-condenser, but a different corner

Pruna is the strongest "bring your own model, we compress it" commercial player in 2026:
50+ composable algorithms (quantization, pruning, distillation, **an explicit "recovery"
family**, caching), an agentic config-search that *finds* the recipe, built-in eval, and a
real business (open-sourced Mar 2025; ~$2.8M revenue, ~25 people; API + self-hosted + OSS).
On paper it is "smash any model" — exactly the BYO verb.

**But it does not occupy Hawking's corner**, on three checked axes:

- **Hardware/OS:** Pruna is **officially supported on Linux**; macOS/Windows work is
  "algorithm-dependent." It is **not Apple-Silicon-native** and does not advertise Metal.
- **Modality/target:** Pruna's *front page* is **image and video generation** (text-to-image,
  upscaling, avatars) and **cloud/GPU inference cost**, not local-Mac LLM condensation. No LLM
  product is foregrounded; the positioning is "faster/cheaper inference at datacenter scale."
- **Out-of-core:** **not advertised.** Nothing on the site claims compressing a model larger
  than the machine's RAM; the implicit assumption is GPU/cloud where the model fits.

**Verdict:** Pruna validates that "BYO-model compression as a product" is a *real market*
(funded, revenue, agentic recipe search) — which de-risks the *category*. But it sits in the
**cloud/GPU, image/video, fits-in-memory** corner. Hawking's corner — **LLM, Apple Silicon,
out-of-core, gradient-recovery-into-trellis** — is orthogonal. *The BYO-condensation slot is
occupied at the category level and OPEN at Hawking's specific intersection.*

### 13b. The vendor-low-bit players (BitNet, ParetoQ, Gemma-QAT) — prove the prize, can't sell it to *you*

BitNet b1.58 + bitnet.cpp and Meta's ParetoQ prove that **~2-bit / ternary near-lossless is
real**. But both are **train-from-scratch QAT** (BitNet's 2B used 4T tokens; ParetoQ pins a
"learning transition" below 2-bit where representations must be *re-learned*, not converted).
**The largest *downloadable* native-ternary model is still ~2.4B.** Gemma-QAT ships int4 — but
only Google's models. **None of them let you bring your own pretrained 32B and condense it.**
That *conversion* — pretrained-parent → ~2-bit-near-lossless, via recovery rather than
retraining — is precisely Hawking T1.3. The vendor players make the prize credible and
simultaneously leave the BYO door open. **This is the single most important research finding:
the "your model → ~2-bit near-lossless" niche genuinely survives**, because everyone who has
*reached* ~2-bit did it by pretraining low, not by condensing an existing model.

### 13c. The acquired/absorbed optimizers (Deci, Neural Magic, OctoML) — the slot got vacated, into CUDA

A telling structural signal: the model-optimization startups have been **absorbed into the
datacenter/CUDA stacks**, not into Apple-Silicon products:

- **Deci AI → acquired by NVIDIA** (model compression, GPU-focused).
- **Neural Magic → acquired by Red Hat** (Nov 2024; vLLM/CPU-and-GPU enterprise serving).
- **OctoML/OctoAI (Apache TVM) → acquired by NVIDIA**, then the inference API was sunset.
- **TitanML** — the BYO enterprise-inference angle, also datacenter/Kubernetes-shaped.

The pattern is unambiguous: **commercial model-optimization consolidated toward enterprise
GPU/cloud serving.** Nobody acquired a "condense-your-model-for-a-Mac" company because none
exists. The slot Hawking wants wasn't taken — *it was vacated upward into the datacenter.*
That is the good kind of empty for a solo builder (structural neglect, not lack of demand) —
**but the same consolidation means the recovery expertise now lives inside NVIDIA/Red Hat**,
who could turn toward Apple Silicon if the market grew. The empty corner is real *and*
defended only by the incumbents' disinterest, which is not a moat.

### 13d. The out-of-core angle — is it a differentiator or a niche-of-a-niche?

Honest answer, after checking the prior art: **out-of-core *serving* is NOT a differentiator
— it is a solved niche.** llama.cpp's mmap pages weights from disk under memory pressure;
**AirLLM** explicitly runs a 70B layer-by-layer on modest hardware; research systems
(prima.cpp) extend mmap pipeline-parallel across home clusters. All of them exist, and all of
them pay a **brutal tps cost** (the §5 Mixtral ≈0.1 tok/s out-of-core case is exactly this
regime). So "we can serve a model bigger than RAM" is **not** a Hawking claim — others got
there first and it's slow.

**The differentiator is out-of-core *condensation*, not serving:** producing the small
artifact on a machine that can't hold the parent, so that *after* condensation it serves
fully-resident and fast. That is a genuinely under-occupied slot — Pruna assumes the model
fits (cloud/GPU); AirLLM/mmap only *serve* out-of-core, they don't *recover-and-shrink*
out-of-core. **But be honest about the size of the slot:** it is a *niche-of-a-niche*. It
only matters to a user who (a) has a big model, (b) on a small Mac, (c) wants the *condensed*
artifact, (d) and can't or won't rent a GPU for an afternoon to make it. That is a real but
**thin** audience. Out-of-core condensation is best treated as a **supporting differentiator**
(it makes the "small Mac" story coherent) rather than the headline — the headline is
**recovery quality** (the §10 / T1.3 quality card). If recovery doesn't land, out-of-core
alone is not a product.

### 13e. The slot, stated precisely (does it survive? — YES, narrowly)

| Candidate | BYO any model? | Gradient recovery? | Into a trellis codec? | Apple-Silicon-native? | Out-of-core *condense*? | One verb? |
|---|---|---|---|---|---|---|
| **Pruna AI** | ✅ | ✅ (recovery family) | ❌ | ❌ (Linux/GPU) | ❌ | ✅ |
| **MLX-LM QAT** | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Unsloth Dyn 2.0** | ✅ | ❌ | ❌ | (via llama.cpp) | ❌ | partial |
| **EXL3 / PonyExl3** | ✅ | ❌ | ✅ | ❌ / ✅ | ❌ (in-core) | partial |
| **AQLM / QTIP / QuIP#** | ✅ | sometimes | ✅ (lattice/trellis) | ❌ | ❌ | ❌ |
| **BitNet / ParetoQ / Gemma-QAT** | ❌ (vendor/retrain) | ✅ | n/a | ❌ | ❌ | ❌ |
| **AirLLM / llama.cpp mmap** | ✅ (serve only) | ❌ | ❌ | partial | serve-only | ❌ |
| **HAWKING (target)** | ✅ | ✅ (STRAND-aware QAT/KD) | ✅ (STRAND) | ✅ (Metal) | ✅ (the goal) | ✅ (`condense`→`run`) |

**No shipping tool holds the full Hawking row.** The slot survives scrutiny. **But the
survival is conditional and thinner than the original brief:** the recovery primitive is
commoditizing (Pruna, MLX, ParetoQ all have it), trellis-on-Metal serving is taken
(PonyExl3), out-of-core *serving* is taken (AirLLM/mmap), and per-tensor bits is taken
(Unsloth). **What is uniquely open is the *full intersection*** — and that intersection is
*exactly the part of Hawking that is unproven (recovery quality at scale) and half-built
(native serving, out-of-core condense at scale).* **The white space is the unbuilt part.**
That is the honest tension, and it has not improved since the original brief — if anything,
MLX-QAT closing and Pruna's traction make the window tighter (§17).

---

## 14. The white space, the wedge, and the verdict for a solo builder

### Is there a leadable niche for a solo 18-year-old?

**Yes, but a narrow one, and only conditionally — and not the niche the one-line pitch
implies.** Hawking will *not* lead "local LLM engine on Mac" (llama.cpp/MLX/Ollama own it,
well-funded, network-effect-locked) and will *not* lead "fastest 2-bit trellis decode on
Mac" (PonyExl3 already shipped it; EXL3/QTIP own the codec research) and will *not* lead
"run a model bigger than RAM" (AirLLM/mmap got there). Those races are lost or unwinnable for
one person.

The *only* defensible niche is the **artifact-creation** corner: a recovery-based,
out-of-core condenser for Apple Silicon that produces the smallest near-lossless form of
*your own* big model. That corner is empty **because it's hard** (the unbuilt intersection of
§13e), not because nobody wants it — the good kind of empty for a solo builder, *if and only
if* the recovery quality win actually lands and is independently benchmarkable.

### The one-sentence wedge

> **"The only out-of-core, recovery-based condenser for Apple Silicon — bring your own 32B,
> and Hawking gradient-heals it into the smallest near-lossless artifact that runs where a
> Q4 GGUF can't even load."**

This wedge survives contact with every competitor in §12 *only after* T1.1/T1.3 (STRAND-aware
QAT recovery → the 32B quality card) land at scale. Until then, the honest wedge is narrower:

> **"A from-scratch Rust+Metal inference engine and an out-of-core condensation pipeline that
> already wins on density (~52% smaller) — a systems-engineering showcase, with a quality win
> in progress and measured in the open."**

### If the recovery win never lands

Then Hawking is **not a leadable product** and should be reframed honestly as what it
unambiguously *is*: a **portfolio and credibility artifact** of rare depth — a solo,
from-scratch Rust+Metal inference engine + an out-of-core, output-space-rigorous quantization
research harness, with a documented, self-critical engineering process ("no fake GO," measured
NO-GOs, triangulated conclusions). That is a *stronger* hiring signal than most shipped
products, and it requires zero further research risk to bank. **Bank that value now regardless
of whether recovery lands.**

---

## 15. Conditions under which leadership is real vs. not

Leadership is **REAL** only if *all* of these hold (each is currently open):

1. **Recovery quality lands at scale.** Condensed+doctor degradation **< Q4_K's ~+2.1% at
   < 4.5 bpw**, demonstrated at **14B and 32B**, not just 0.5B. (Today: 0.5B, +31%, loses.)
   — *This is the §10 bridge's T1.1 → T1.3 chain.*
2. **Independently benchmarkable.** The eval harness is public, reproducible, standard metrics
   (MMLU / wikitext PPL / KL-div vs parent) so skeptics can rerun it. A win nobody can
   reproduce is treated as a lie. — *§10's "reproducible harness + ledger" artifact.*
3. **Native low-bit serving is wired** so the RAM-cliff *speed* win is demonstrable on a model
   that does **not** fit at Q4 (the 70B-on-96GB demo §5). Rehydrate-to-f16 doesn't count.
4. **Out-of-core condense works at 32B on a small Mac** — the part no competitor has (§13d). If
   condensation itself needs a big machine, the whole "small Mac" story collapses.
5. **PonyExl3 / MLX-QAT / Pruna do not close the recovery+out-of-core gap first.** Monitor all
   three (§17). MLX-QAT is the most dangerous; Pruna the most likely to add an Apple/LLM path.

Leadership is **NOT real** if: the win only shows at ≤1.5B; or it needs a 64GB+ machine to
*produce*; or the benchmark is bespoke/unreproducible; or recovery requires per-model
hand-tuning that doesn't generalize (the overfit lesson, at scale).

---

## 16. The non-experimentation competitive roadmap (the GTM core deliverable)

Prioritized work **outside research**. Scoped for one person. Ordering is deliberate: **the
demo and the eval come before the polish** — in this market credibility is the entire
currency and it is won or lost on a reproducible number. **Each P-item below consumes a §10
proof artifact** — that is the continuous workflow made operational.

### P0 — The one undeniable artifact (do this before anything public)
- **Ship a single reproducible "money demo":** a model that **cannot load at Q4 on the target
  Mac**, condensed by Hawking, **running and coherent**, side-by-side with (a) llama.cpp
  OOM/swap-thrash vs (b) Hawking generating at usable tok/s, **plus a quality card** (PPL/KL
  vs f16 parent). One asciinema/screen-capture. *This consumes the §5 RAM-cliff artifact +
  the §10 quality card.* This *is* the pitch.
- **Gate it on the §6 proof bar.** If the quality number isn't there yet, the demo is
  "density + it runs where nothing else does" — still real, ship that.
- Do **not** publish anything before this artifact exists. A claim without the demo gets shredded.

### P1 — Own the eval (the quality card as the product)
- **Publish the benchmark harness** (`quality_3way.sh` lineage) as a standalone reproducible
  repo/page: parent f16 vs Hawking-condensed vs llama.cpp Q4_K / Unsloth-Dyn-2.0 vs (if
  feasible) EXL3/PonyExl3 / **an MLX-4bit baseline**, on **standard metrics**, exact commands.
  *This consumes §10's reproducible-harness artifact.*
- **Quality cards as a first-class output:** every condensed model emits `{bpw, RAM-fit,
  %-parent-quality, tok/s, PPL-delta, KL}`. Make the card the thing people screenshot.
- **Frame the comparison on YOUR axis** (density-at-fixed-quality, fits-where-others-OOM),
  never iso-quant decode tok/s. Choosing the axis is half of winning the narrative.

### P2 — Packaging & distribution
- **GitHub:** ruthless README leading with the wedge + the demo GIF + the honest scoreboard
  (density WON, quality in-progress). Surface the self-critical docs — they're an asset.
  License **Apache-2.0** (permissive, hireable) — confirm no copyleft deps force otherwise.
- **One-command install:** Homebrew tap or `curl | sh` + a prebuilt binary.
  `brew install hawking` is worth a week of work; friction kills solo projects.
- **HuggingFace presence:** publish 1–2 **condensed artifacts** (a small + a mid model) with
  quality cards as model cards — where the "bring your own model" audience lives, free distribution.
- **Show HN / r/LocalLLaMA launch — only with P0+P1 done.** Title on the wedge, not the engine.
  Lead comment: the honest scoreboard. This crowd rewards candor and *punishes* overclaiming;
  the "no fake GO" voice is perfectly tuned for it.

### P3 — Positioning / messaging discipline
- **Stop competing on decode tok/s in all public copy.** Every doc that mentions 0.71×
  reframes immediately to density + RAM-cliff. One off-message benchmark and the thread becomes
  "but llama.cpp is faster," which you lose.
- **Name the category, don't fight an existing one.** "Condensation" / "model condenser" — own
  a word. Don't position as "another inference engine."
- **Lead with the comparison nobody else can make:** "the 32B that *doesn't fit* at Q4."

### P4 — Community & longevity (low effort, high compounding)
- A `BENCHMARKS.md` updated as recovery improves — visible progress builds trust.
- Respond to every issue/HN comment in the honest voice. For a solo project the author's
  credibility *is* the moat as much as the code.
- Keep the self-critical changelog public — a differentiator, not a liability.

### Deliberately DO NOT build (scope discipline = survival — the kill-list)
- **Not** a faster decode kernel to beat llama.cpp iso-quant. Permanent loss; abandon it.
- **Not** broad model-architecture coverage. **Pick ONE family** (Qwen, verified) and make the demo perfect there.
- **Not** a server/OpenAI-API/agent layer. llama.cpp/Ollama own it; you'd lose years.
- **Not** Windows/Linux/CUDA. Apple Silicon focus *is* the wedge; broadening dilutes it.
- **Not** a GUI/LM-Studio competitor. CLI + great docs is enough for the target user.
- **Not** your own novel trellis codec research *for its own sake* — STRAND exists; the job is
  recovery *into* it (QTIP/EXL3 will out-research you on codebooks).
- **Not** out-of-core *serving* as a headline — AirLLM/mmap own it and it's slow (§13d). Out-of-core
  *condensation* is the supporting differentiator, not the pitch.
- **Not** frontier MoE (671B) chasing — owner-gated, compute-prohibitive, off the critical path.

---

## 17. Market timing — the window, and the risk of being lapped

**The low-bit recovery field is hot, and that cuts both ways.** The fresh research shows
2025–2026 produced a wave of exactly-adjacent work: ParetoQ (unified 1–4-bit QAT, Meta),
Recover-LoRA (data-free 2-bit accuracy recovery), pQuant / D2Quant / "Unifying block-wise PTQ
+ distillation-QAT toward 2-bit" / "Bit-by-Bit progressive QAT" / NVFP4 quantization-aware
distillation. The academic consensus is converging on the same thesis Hawking bets on —
**quantization-aware *distillation* beats plain QAT and recovers near-BF16 at 2-bit.** That is
validation *and* threat.

**Quantify the risk of being lapped:**

- **MLX-QAT is the clock.** Apple's framework went production-mature in 2025 and pulled ahead
  of llama.cpp on Metal in 2026. If Apple (or the MLX community) ships a one-command
  "condense + recover" path **into MLX's own format**, the Apple-native half of Hawking's
  wedge evaporates overnight. This is the **highest-probability lapping event**, and it is
  partly outside anyone's control (it's Apple's roadmap). **Window estimate: 6–18 months
  [UNVERIFIED]** before MLX-QAT + a mixed-bit recipe is "good enough" that the recovery
  delta Hawking needs to show becomes marketing-thin.
- **Pruna is the commercial clock.** Funded, revenue-positive, agentic recipe search, an
  explicit recovery family. If Pruna adds an LLM-on-Apple-Silicon path (it currently does
  not), it has the team and capital to occupy Hawking's corner in a quarter. Lower probability
  (their focus is image/video/cloud) but high impact.
- **PonyExl3 is the codec clock.** Already ships trellis-on-Metal serving. If it adds even a
  light LoRA-recovery step, the "trellis + Metal + some recovery" combination gets close. Its
  *in-core* limitation is the gap Hawking's out-of-core condense exploits — monitor whether it
  closes.
- **The research clock.** The 2-bit-recovery papers are publishing monthly. Hawking's *method*
  (codec-aware GPTQ-Hessian error-feedback + KD, T1.1/§2-layer-5) is **not** novel as research
  — its novelty is **packaging that recovery into a deployable Apple-Silicon condenser with an
  out-of-core path.** Hawking must not try to out-research the field (kill-list §16); it must
  **out-ship** it into the one corner the researchers don't target (BYO, Mac, one verb).

**The window verdict:** the niche is open **now** and the proof artifacts (§10) are buildable
**now** on the Studio — but the moat is *timing and packaging*, not a defensible technical
secret. The honest framing: **Hawking has roughly a 12-month window to manufacture the §10
quality card and the RAM-cliff demo and plant the flag, before MLX-QAT and/or Pruna make
"condense your model on a Mac" a checkbox feature.** [UNVERIFIED — the 12-month figure is a
judgment call, not a sourced forecast.] If the recovery win lands inside that window, the
portfolio value (and possibly product value) is banked; if it slips, fall back to §14's
portfolio framing, which is timeless and needs no window.

---

## 18. The honest bottom line

There **is** a leadable niche, and it is exactly one sentence wide: *out-of-core,
recovery-based condensation of your own model for Apple Silicon.* It is leadable **only** if
the recovery quality win lands at 14B/32B (the §10 / T1.3 quality card) and is independently
reproducible — conditions that are open and non-trivial. Everything adjacent (engines, decode
speed, the trellis codec itself, trellis-on-Metal serving, out-of-core *serving*) is either
lost to incumbents or already shipped by someone else (PonyExl3, EXL3, MLX-QAT, Unsloth,
AirLLM, Pruna).

**The fresh-research verdict on the central question** — *does the BYO-condensation niche
survive?* — is **yes, narrowly, and on borrowed time.** Pruna proves the *category* is a real
business but sits in the cloud/GPU/image corner; the vendor-low-bit players (BitNet, ParetoQ,
Gemma-QAT) prove ~2-bit-near-lossless is real but only by *retraining*, leaving the
*condense-your-existing-model* door open; the optimization startups consolidated upward into
CUDA/datacenter (Deci→NVIDIA, Neural Magic→Red Hat, OctoML→NVIDIA), vacating rather than
occupying the Mac corner; and out-of-core *serving* is solved (AirLLM/mmap) but out-of-core
*condensation* is not. **The full intersection is genuinely empty — and genuinely unbuilt.**

The asymmetry that makes this worth doing as a solo 18-year-old: **even in the failure case,
Hawking is an exceptional portfolio artifact** — and in the success case, it owns a corner
that well-funded labs have structurally ignored because vendors ship *their* QAT models, not a
tool to condense *yours*. Build the P0 demo, own the eval, stay ruthlessly on the
density/RAM-cliff axis, manufacture the §10 quality card inside the §17 window, and refuse to
fight the races you've already lost.

---

## 19. Final edge pass — leadership mechanics and the 30-day Studio sprint

This section is the "make it impossible to ignore" pass. It does **not** add another pile of
experiments. It turns the existing experiment bank into a sharper leadership machine: the exact
claim, the exact anti-claim, the unfair artifact, the proof ladder, the non-code moat, and the
first 30 days after the Studio arrives. The standard here is not "interesting"; the standard is
**a skeptical LocalLLaMA / MLX / quantization person cannot dismiss it in one comment**.

### 19.1 The narrow leadership claim

The strongest claim is not "Hawking is the best inference engine" and not "Hawking has the best
quantizer." The claim is narrower and more dangerous:

> **Hawking is the Apple-Silicon BYO-model condenser that produces a signed density/quality
> receipt for models that do not fit at ordinary 4-bit on the same Mac.**

Every word matters:

- **Apple-Silicon:** the target is the machine researchers and solo builders actually own, not a
  CUDA box they rent for an afternoon.
- **BYO-model:** the user brings an existing model. This avoids vendor-low-bit models, which prove
  2-bit can work but do not help you condense *your* fine-tune.
- **Condenser:** not a general server, not an app shell, not an agent platform. One verb.
- **Signed receipt:** the output is not just weights. It is weights plus reproducible proof:
  memory, bpw, PPL/KL, prompt transcript, baseline commands, exact machine, exact commit.
- **Does not fit at Q4:** the demo must happen at the RAM cliff. If both models fit, the density
  advantage looks optional. If Q4 cannot load and Hawking can, the value becomes physical.

The empty intersection is: **out-of-core recovery-based condensation of an existing LLM into an
Apple-Silicon-native artifact, measured by density-at-fixed-quality at the RAM cliff.**

### 19.2 The anti-claim

These claims are forbidden in public copy unless the evidence later becomes overwhelming:

- **Not fastest decode.** llama.cpp, MLX, EXL3/PonyExl3, and vendor runtimes can win iso-quant
  speed. Hawking's scoreboard is "fits and preserves quality," not raw tok/s.
- **Not universal architecture coverage.** The first public win should support one model family
  cleanly, likely Qwen, and document that boundary.
- **Not lossless.** The claim is near-parent quality under declared metrics, never "no degradation."
- **Not a 405B/671B local system.** 96 GB makes 7B/14B/32B doctoring real and ~70B serving
  plausible; it does not make hyperscale local training real.
- **Not a new trellis-codebook research lab.** STRAND/QTIP/EXL3-style codec research is adjacent.
  Hawking's differentiator is recovery into a deployable Mac condenser.
- **Not a cloud optimizer.** Pruna and CUDA stacks own broader compression menus. Hawking owns the
  narrow local receipt.

### 19.3 The unfair artifact: the Hawking Condensation Receipt

The artifact that can create disproportionate credibility is a **Condensation Receipt**, not a
blog post. It should be emitted for every serious run and be publishable as a model-card attachment.

Minimum receipt fields:

```json
{
  "project": "hawking",
  "receipt_version": "0.1",
  "machine": "Mac Studio M2 Max, 96GB unified, 2TB",
  "model_family": "qwen",
  "source_model": "exact HF id or local sha256",
  "source_precision": "bf16/f16",
  "condensed_artifact": "path or HF id",
  "effective_bpw": 0.0,
  "peak_rss_gb": 0.0,
  "swap_gb": 0.0,
  "tokens_per_second": 0.0,
  "baseline_q4_load_result": "ok|oom|swap-thrash|not-run",
  "ppl_parent": 0.0,
  "ppl_condensed": 0.0,
  "kl_parent_condensed": 0.0,
  "prompt_suite_hash": "sha256",
  "quality_gate": "pass|warn|fail",
  "hawking_commit": "git sha",
  "commands": ["exact reproduce commands"],
  "raw_logs": ["paths or artifact urls"]
}
```

The receipt is unfair because incumbents usually publish either a model, a benchmark table, or a
runtime. They rarely publish a **forensic proof bundle** that says: same Mac, same model family,
Q4 could not load, Hawking did load, here is the quality delta, here are the commands, here is the
raw log. That is hard to hand-wave away.

The public package should contain four visible surfaces:

1. **A 60-second RAM-cliff video:** Q4 baseline fails or thrashes, Hawking artifact loads, coherent
   generation starts.
2. **A one-page quality card:** bpw, RAM, PPL delta, KL, tok/s, exact baselines.
3. **The JSON receipt:** machine-verifiable, not prose.
4. **A failure appendix:** models where Hawking failed, with reasons. This is trust-building, not
   embarrassment.

### 19.4 The proof ladder

The proof path must be a ladder, not a reveal:

| Stage | Private experiment | Metric | Artifact | Public move | Recognition target |
|---|---|---|---|---|---|
| L0 | 0.5B/1.5B sanity ladder | Exact reproduce, PPL/KL sane | Tiny receipts | Internal calibration | No launch |
| L1 | 7B full-rank/QAT/KD recovery | <= +2% PPL, stable KL | First public quality card | Repo README demo | Builder credibility |
| L2 | 14B density sweep | Bpw at quality gate | Bit-floor curve point | Technical writeup | Quantization audience |
| L3 | 32B RAM-cliff run | Q4 OOM or thrash, Hawking runs | Money demo + receipt | Show HN / r/LocalLLaMA | Niche flag planted |
| L4 | Independent rerun by another Mac owner | Receipt reproduced within tolerance | Third-party receipt | README badge/table | Trust moat |
| L5 | Failure atlas | Which layers/models break | Negative result appendix | "No fake GO" update | Durable credibility |

Do not skip L4. Third-party reproduction is the difference between "cool solo project" and
"scoreboard people may start trusting."

### 19.5 Scoreboards to avoid and scoreboards to own

Avoid:

- Iso-quant tok/s.
- Broad model zoo coverage.
- "Best 2-bit quantizer" without recovery and baseline receipts.
- Cloud compression throughput.
- Synthetic prompt cherry-picks.

Own:

- **Density at fixed parent-quality delta.**
- **Fits where Q4 does not fit.**
- **BYO recovery cost on Apple Silicon.**
- **Receipt reproducibility.**
- **Failure transparency by model family and layer class.**

The scoreboard line should look like this:

> "For model X on Mac Y, what is the smallest artifact that passes the parent-quality gate, and
> can another user reproduce the receipt?"

That is the question MLX, llama.cpp, EXL3, and Pruna are not naturally optimized to answer in
public.

### 19.6 The moat beyond code

The real moat is not a secret kernel. It is a bundle of boring things that compound:

- **Receipts:** every run has machine-readable proof.
- **Baseline neutrality:** llama.cpp, MLX, PonyExl3/EXL3 where possible, Unsloth dynamic GGUF, all
  run with exact commands and no rhetorical sandbagging.
- **Versioned prompt and PPL suites:** no private cherry-pick set.
- **A public `FAILURES.md`:** turns dead ends into reliability evidence.
- **A `WATCHLIST.md`:** tracks MLX-QAT, Pruna, PonyExl3, EXL3, BitNet/vendor-QAT, and says what
  would invalidate Hawking's wedge.
- **A one-family-first promise:** Qwen first. Breadth only after the receipt pipeline is trusted.
- **Independent receipt submissions:** a lightweight `receipts/third_party/` folder is enough.
- **Name ownership:** "condensation receipt" is the phrase to repeat until it becomes the category
  handle.

Better-funded incumbents often skip this because they want polished product pages, not
adversarial receipts. A solo builder can make the receipts the product.

### 19.7 First 30 days after the Studio arrives

**Machine-running tasks**

1. Run the L0 sanity ladder on 0.5B/1.5B with the receipt schema stubbed from day one.
2. Re-run the current 7B doctor loop with the old LoRA path only as a baseline, not as the main
   strategy.
3. Start the first full-rank or layer-wise recovery run on 7B with CPU-bf16, logging peak memory,
   swap, wall-clock, and layer failure modes.
4. Run baseline receipts for llama.cpp Q4, MLX 4-bit, and any available dynamic/mixed-bit GGUF on
   the same machine.
5. Expand to 14B only after 7B receipts are clean and reproducible.

**Human judgment tasks**

1. Freeze the public quality gate: PPL delta, KL threshold, prompt-suite policy, and "warn" bands.
2. Pick exactly one model family for public launch.
3. Decide what "Q4 cannot fit" means operationally: hard OOM, swap ceiling, or unusable tok/s.
4. Write the failure taxonomy before the 32B run so failures are classified honestly.
5. Draft the launch copy with the anti-claims included.

**Public artifact tasks**

1. Implement `hawking receipt` or equivalent JSON emitter.
2. Create `QUALITY_CARDS.md`, `FAILURES.md`, and `WATCHLIST.md`.
3. Produce the 7B quality card as a dry run.
4. Produce the 32B RAM-cliff video only if the quality gate is at least "warn."
5. Publish one small condensed artifact first; publish the flagship only after independent rerun
   instructions are clean.

**Stop/continue gates**

- If 7B cannot beat simple mixed-bit PTQ at the quality gate, stop and fix recovery before 14B.
- If 14B shows no density advantage over dynamic GGUF at equal quality, keep Hawking as portfolio
  infrastructure and do not overclaim the niche.
- If 32B passes the quality gate and Q4 fails to fit, launch. That is the money card.
- If third-party receipts disagree wildly, pause launch and make reproducibility the project.

### 19.8 Rare high-leverage ideas, beyond "run more experiments"

1. **Receipt diffing.** Add a tiny tool that compares two condensation receipts and prints what
   changed: bpw, PPL, KL, RAM, prompts, machine. Incumbents publish tables; Hawking publishes
   auditable deltas.
2. **The RAM-cliff simulator.** A script that predicts whether Q4, MLX-4bit, and Hawking should
   fit on 18GB/36GB/64GB/96GB/128GB Macs, then marks predictions as verified or falsified. This
   makes the physical value proposition visible before users run anything.
3. **Baseline bounty without money.** Invite users to submit a baseline that beats Hawking's
   density/quality card on the same machine. Winning baselines go into the README. This converts
   adversaries into benchmark contributors.
4. **The "one bad layer" atlas.** Publish which layers dominate recovery loss. This can become a
   reference artifact for other quantization people even if Hawking's product angle stalls.
5. **Artifact licenses and hashes first.** Every published condensed model gets a source license
   note, source hash, output hash, and "derived artifact" terms. This prevents a future public
   release from getting derailed by provenance questions.
6. **A pessimistic mode.** The CLI should have a flag that runs the strictest baseline and worst
   prompt suite first. Make honesty the default interaction.
7. **"No cherry-pick" prompt cassette.** Freeze a small public prompt suite with seed, tokenizer,
   and expected parent outputs. The demo video uses that suite only.
8. **Recovery cost accounting.** Report joules/kWh or at least wall-clock and estimated energy per
   condensation. Apple-Silicon local optimization can own a "no rented GPU" ecological/economic
   proof line without making it the main claim.
9. **A lapping alarm.** Monthly re-run the watchlist: MLX-QAT, Pruna, PonyExl3. If one closes the
   gap, update the README. This sounds dangerous, but public self-obsolescence monitoring builds
   unusual trust.
10. **Independent Mac matrix.** Recruit 3-5 Mac owners with 18GB, 36GB, 64GB, 96GB, 128GB to run
    the same receipt. This turns Hawking from "works on my box" into a physical memory-scaling
    result.

### 19.9 Final verdict

Hawking can lead a niche **only** if the 14B/32B recovery quality cards land and at least one
third-party Mac owner can reproduce a receipt. The niche is not "LLM inference" and not
"quantization." It is **receipt-backed BYO condensation for Apple Silicon at the RAM cliff**.

If the flagship research fails, Hawking is still a high-grade systems portfolio: a Rust/Metal
engine, an out-of-core doctor, a quantified failure atlas, and a public receipt harness. That is
not a consolation prize; it is still rare evidence of engineering range. But public leadership
requires the money card: **a 32B class model that ordinary Q4 cannot load on the target machine,
while Hawking runs with a defensible quality delta.**

Single next action: **build the Condensation Receipt schema and make every Studio run emit it,
before chasing another recovery idea.** Without the receipt, the experiments produce private
logs. With the receipt, every experiment manufactures public proof.

---

## 20. The Proof System — the methodology that makes Hawking hard to dismiss

> **This is the capstone of the arc, and it is NOT "more experiments."** §1–§9 build the
> machine, §10 bridges lab→market, §11–§18 position, §19 sharpens the leadership mechanics.
> This section ascends the *methodology* itself: the standard, the schemas, the gates, and the
> invalidation rules that make every claim above structurally hard to dismiss. The target
> reader is hostile — a stronger lab, a skeptical MLX/quantization expert, a benchmark
> maintainer, a workshop reviewer, a future employer, or an r/LocalLLaMA commenter looking
> for the one sentence that ends the thread. **The goal is not to sound grand; it is to be
> structurally un-dismissable:** every claim is tagged by reproduction level, every failure is
> an artifact, every receipt has explicit invalidation rules, and no win is asserted that an
> outsider cannot rerun on their own Mac. **All proof artifacts are specified INLINE here**
> (one consolidated doc — §19.3's receipt schema is extended, not scattered). This section
> turns Hawking from "Apple-Silicon BYO model condensation" into a **receipt-backed science of
> local model density, memory cliffs, and recovery quality.**
>
> **Honesty carried in unchanged (no fake GO):** nothing below upgrades a contingent claim to
> a proven one. The five brutal facts from §11 still hold — Hawking loses iso-quant decode, the
> quality win is unproven at scale, native serving is half-built. This section makes the
> *path to proof* rigorous; it does not pretend the proof exists yet.

### 20.1 The epistemic contract (what Hawking promises, refuses, and would retract)

State the contract in plain language so a skeptic knows exactly what is and is not being
claimed. This is the first thing that disarms "you're just overclaiming."

**What Hawking promises to MEASURE (and report regardless of outcome):**
- The **effective bpw** of a condensed artifact (baker AGGREGATE — RHT + outlier + residual +
  side-info), never nominal.
- The **parent-quality delta**: PPL-delta and KL vs the *same model's* f16 parent, on a frozen
  multi-window suite, plus the `multi_eval.py` capability tripwire.
- The **peak memory and swap** of both condensation and serving, on a named machine.
- Whether the **Q4 baseline loads at all** on that machine (ok / oom / swap-thrash / not-run).
- The **exact reproduce commands, commit, and artifact hashes**.

**What Hawking REFUSES to claim:**
- Not "fastest decode" (loses iso-quant, permanently — §11).
- Not "lossless" (only *near-parent under declared metrics*).
- Not a density win on a model where Q4 also fits *unless* it is explicitly labelled a
  **density** demo, not a **cliff** demo (§5's distinction is load-bearing).
- Not a scale-law extrapolation (70B/405B floor) until the 5-point local curve is placed and a
  fit with a confidence band exists (§4 / T3.1).
- Not "first/only" on any axis still tagged **[UNVERIFIED]** against the live landscape.

**Evidence SUFFICIENT for a public claim (all must hold):**
> We do not claim a density win unless the receipt shows, **on the same named machine**:
> parent-quality delta ≤ the stated gate, baseline Q4/MLX-4bit behavior, peak memory + swap,
> exact reproduce commands, and artifact hashes — AND the run is at **R2 or higher** (§20.6).

**Evidence that FORCES a downgrade** (claim → "contingent", not retracted): a single failing
window in the multi-window suite; KL above the warn band while PPL passes (ppl-theater risk); a
peak-memory number that does not reproduce within tolerance on rerun; a baseline that was run
"best effort" rather than tuned (§20-BASELINES).

**Evidence that KILLS the wedge** (forces the §14 portfolio reframe): recovery fails to beat a
*tuned* dynamic GGUF / MLX-4bit at equal quality at 14B **and** 32B; OR the density advantage
vanishes once the Q4 baseline is allowed mmap/out-of-core serving at acceptable tok/s; OR
condensation itself requires a >96 GB machine to produce (the "small Mac" story collapses, §15).

**Publishable EVEN WHEN NEGATIVE:** the bit-floor-vs-scale curve coming out flat (H0); recovery
losing to dynamic GGUF (a clean, sourced negative); the one-bad-layer atlas; memory-prediction
misses; any dead-end with a reproduce command. **A negative with a receipt is a deliverable
(§20.5); a negative without one is noise.**

### 20.2 The adversarial validation layer (assume a hostile expert is reading)

For each load-bearing claim, pre-write the strongest attack and the test that settles it. This
is the section a workshop reviewer respects.

| The claim | Strongest baseline to beat | The likely hostile critique | The test that settles it | If the critique is RIGHT | If it is WRONG |
|---|---|---|---|---|---|
| "We reach near-parent quality below where PTQ pays" (T1.1) | Tuned llama.cpp **Q4_K / IQ-quants**, **Unsloth Dynamic 2.0**, **MLX 4-bit**, EXL3/PonyExl3 where runnable | *"This is just worse quantization with a fancy name — a tuned dynamic GGUF matches it."* | Same machine, same model, same frozen suite: Hawking vs each baseline at matched eff-bpw, receipts side-by-side | **Publish it as a failure** (§20.5), pivot the product to the **receipt harness + RAM-cliff predictor + failure atlas** (§20.12) | **Publish the density-at-quality receipt** — the one number no baseline matches |
| "Bigger models compress harder" (§4) | The flat-floor null (H0) | *"Five noisy points, the 0.5B is rigged pessimistic, this is curve-fitting."* | 5-point curve judged on 7B+ only; 0.5B reported but never sets the verdict; band drawn; pre-registered 70B prediction (T3.1) | Report H0 honestly; the curve still ships as a measured fact | The descent + a power-law fit with a band |
| "The model that doesn't fit at Q4 runs here" (§5) | llama.cpp **mmap out-of-core**, **AirLLM** layer-by-layer | *"mmap/AirLLM already serve bigger-than-RAM — you're not first."* | The cliff is **density at usable tok/s**, not "can it load": show Q4 thrashing to SSD-bound tok/s vs Hawking resident, both timed | Reframe to "fits *resident and fast*", concede serving-out-of-core is solved (§13d) | The tps-cliff receipt with both timings |
| "Out-of-core *condensation* on a small Mac" (§13d) | Pruna (cloud/GPU), MLX-QAT (in-core) | *"Niche-of-a-niche; just rent a GPU for an afternoon."* | Show the 32B artifact produced on 96 GB where the parent + teacher never co-resident | Demote to **supporting** differentiator, lead on recovery quality | A condensation receipt whose peak RSS < parent bf16 size |

**The single most embarrassing failure mode to pre-empt:** a hostile reader fits the *same*
parent with a tuned Unsloth Dynamic 2.0 GGUF or MLX-4bit recipe on their own Mac and matches
Hawking's quality card at equal-or-lower bpw. The defense is **not** rhetoric — it is having
already run those baselines tuned (not "best effort") and published the receipt that shows the
delta, OR having already published the failure. **Run the adversary's experiment before the
adversary does** (this is §20.10).

### 20.3 The public proof grammar (the smallest unit anyone can audit)

The smallest publishable unit is **one `condensation_receipt.json` + one quality card**. Nothing
ships as a "win" without both. The receipt schema below **extends §19.3** (do not maintain two
schemas — this is the canonical one).

```jsonc
// condensation_receipt.json  (schema v0.2 — extends §19.3 v0.1)
{
  "project": "hawking",
  "receipt_version": "0.2",
  "repro_level": "R0|R1|R2|R3|R4|R5",          // §20.6 — REQUIRED, no claim without it
  "claim_type": "density|cliff|scale-point|negative|baseline",  // what this receipt asserts
  "machine": "Mac Studio M2 Max, 96GB unified, 2TB",
  "machine_class": "M2Max-96",                  // for the §20.6 R3 same-class repro
  "os_build": "macOS 26.x (24Gxx)",
  "model_family": "qwen",                        // one-family-first (§19.6)
  "source_model": "HF id",
  "source_sha256": "sha256 of source weights",   // identity, not trust-me
  "source_precision": "bf16|f16",
  "source_license": "apache-2.0|llama-community|...",  // §20.11 provenance
  "derivative_policy": "see LICENSE_DERIVATIVE",       // §20.11
  "condensed_artifact": "path or HF id",
  "artifact_sha256": "sha256 of condensed weights",    // output hash (§19.8 #5)
  "recipe": ["awq-0.5","mixed-prec","residual","t1.1-errfeedback","logit-kd"],
  "effective_bpw": 0.0,                          // baker AGGREGATE, never nominal
  "nominal_bpw": 0.0,                            // reported too, but never headline
  "peak_rss_gb": 0.0,
  "swap_gb": 0.0,
  "condense_peak_rss_gb": 0.0,                   // the out-of-core proof (§13d)
  "wall_clock_s": 0.0,
  "energy_kwh": 0.0,                             // §20.11 cost line (est. ok, label it)
  "tokens_per_second": 0.0,
  "baseline_q4_load_result": "ok|oom|swap-thrash|not-run",
  "baseline_q4_tps": 0.0,                        // if it loads, time it (the cliff number)
  "baseline_mlx4_result": "ok|oom|swap-thrash|not-run|matched|beaten",
  "baseline_best_effort": true,                  // §20-BASELINES honesty flag
  "ppl_parent": 0.0,
  "ppl_condensed": 0.0,
  "ppl_delta_pct": 0.0,                          // (cond/parent − 1)·100
  "kl_parent_condensed": 0.0,
  "multiwindow_n": 4,                            // ≥4 held-out windows (§6)
  "multiwindow_worst_pct": 0.0,                  // the WORST window, not the mean
  "tripwire": {"qa":0.0,"cloze":0.0,"math":0.0,"code":0.0},  // capability, not ppl
  "prompt_suite_hash": "sha256",                 // frozen cassette (§20.11)
  "prompt_suite_version": "v1",
  "quality_gate": "pass|warn|fail|invalid",      // §20-GATES four levels
  "invalidation_reasons": [],                    // populated when gate=invalid
  "hawking_commit": "git sha",
  "commands": ["exact reproduce commands"],
  "raw_logs": ["paths or artifact urls"]
}
```

**What makes a receipt INVALID** (gate = `invalid`, the run does not count, and the reason is
published — see §20-GATES):
1. `effective_bpw` is missing, or only `nominal_bpw` is reported.
2. Quality is from a **single window** (`multiwindow_n < 4`) or the **mean is reported but the
   worst window is hidden**.
3. PPL passes but `kl_parent_condensed` exceeds the warn band (ppl-theater).
4. No `source_sha256` / `artifact_sha256` (the artifact is not identifiable).
5. No `commands` or no `hawking_commit` (not reproducible).
6. A "density win" claim where `baseline_q4_load_result == "ok"` but the receipt is **labelled
   a cliff** (mislabelled claim_type) — or vice versa.
7. The MPS backend produced the **headline** number without a CPU-bf16 confirmation (§3).
8. `baseline_best_effort == true` is used to claim a **win** (best-effort baselines can only
   support a *contingent* or *negative* result, never a public win — §20-BASELINES).

**How an outsider reproduces a receipt:** clone the repo at `hawking_commit`, run the listed
`commands` against the `source_model` (verify `source_sha256`), regenerate the artifact, verify
`artifact_sha256` matches (or, for non-deterministic recovery, verify the *quality numbers*
reproduce within the stated tolerance), and re-emit their own receipt. A reproduction is
**successful** if `effective_bpw`, `ppl_delta_pct`, and `peak_rss_gb` land within tolerance
(default: bpw ±0.05, ppl-delta ±0.3 pp, RSS ±5%).

**How a third party submits a receipt (the external-Mac path):** a lightweight
`receipts/third_party/<machine_class>/<model>__<submitter>.json` drop, via the PR template in
§20.11. The submitter runs the *published commands* on *their* Mac and commits the resulting
receipt + raw log. No code review needed — the receipt is self-verifying against the hashes.

**How FAILED replications are displayed:** never hidden. A failed or disagreeing third-party
receipt lands in `receipts/third_party/` exactly like a passing one, and is surfaced in
`QUALITY_CARDS.md` as a **divergence row** (machine, what disagreed, by how much). A wide
disagreement **pauses any launch** (§19.7 stop-gate) and makes reproducibility the project.

### 20.4 The trust surface (what an evaluator sees FIRST)

A skeptic should hit **proof**, not prose. Order the public surface so the first screen is
machine-verifiable and the claims are visibly bounded:

1. **The 60-second RAM-cliff video** — Q4 baseline fails/thrashes, Hawking artifact loads,
   coherent generation starts (the physical, undeniable frame).
2. **The `condensation_receipt.json`** — machine-readable, hashes first, one click to the raw log.
3. **The one-page quality card** — bpw, RAM, PPL-delta, KL, worst-window, tok/s, **exact
   baselines named with their commands**.
4. **The baseline comparison** — Hawking vs tuned Q4 / MLX-4bit / Unsloth-Dyn-2.0, same machine,
   side by side, with the honest scoreboard (density WON, quality in-progress, iso-quant LOST).
5. **The failure appendix (`FAILURES.md`)** — what broke and why, linked from the front page.

The README leads with the **wedge sentence + the demo GIF + the honest scoreboard**, never a
wall of claims. The first emotion a hostile reader should feel is *"oh, they already ran the
baseline I was about to suggest."*

### 20.5 The negative-result strategy (failures are first-class artifacts)

| Worth publishing (a deliverable) | NOT worth publishing (noise) |
|---|---|
| Recovery loses to dynamic GGUF/MLX-4bit at equal quality (with receipts) | A run that crashed on a misconfigured flag |
| The one-bad-layer bottleneck atlas (which layer class dominates loss) | An OOM from forgetting to set the swap ceiling |
| Memory-prediction misses (predicted fit, actually swapped) | A NaN from an f16 CPU forward (known dead-end #, §3) |
| Bit-floor curve flat (H0) — redundancy buys nothing | A half-finished experiment with no frozen suite |
| Codec-native error-feedback ties plain residual (T1.1 null) | Any result that can't be reproduced |
| Ternary condensation collapses on a model family (T1.3 null) | A cherry-picked good prompt |

**The failure template (one entry in `FAILURES.md`):**
```
## <FAIL-NNN> <short title>
- model / family / size:        qwen-14b
- recipe / config:              <exact recipe + commit>
- what was expected:            <the hypothesis>
- what happened:                <the measured outcome + numbers>
- receipt:                      receipts/failures/<id>.json   (a real receipt, gate=fail)
- reproduce:                    <exact command>
- category / tags:              [recovery-loss | memory-prediction | layer-bottleneck |
                                 codec-null | ternary-collapse | baseline-beat-us]
- severity:                     warn | fail | wedge-threat
- roadmap effect:               <what this changes in §7 sequencing or §16 GTM>
- pivot trigger?:               yes/no  (does this fire a §20.1 "kills the wedge" condition?)
```

**How failures alter the roadmap:** a `recovery-loss` at 14B AND 32B is a **wedge-threat** →
fires the §20.1 kill condition → fall back to the §14 portfolio framing + ship the §20.12
"if-it-fails-it-still-wins" deliverables. A `layer-bottleneck` finding **re-orders §7** (spend
the next bit budget on that layer class). A `memory-prediction` miss **re-calibrates the
RAM-cliff predictor** (§20-RAMCLIFF). **A single failure that fires a kill condition forces the
full pivot**; everything else just re-orders the work.

### 20.6 The reproducibility gradient (R0–R5 — tag every claim by level)

Never pretend all results are equally strong. Every receipt carries a `repro_level`, and public
copy must state it. A claim's *level* is part of the claim.

| Level | Definition | What it proves | Where it may appear |
|---|---|---|---|
| **R0** | Private raw run, no frozen config | The number exists, internally | Lab notebook only — never public |
| **R1** | Exact command + config captured | The author can rerun it | Internal calibration; `warn`-gated cards |
| **R2** | Artifact hash + metrics + frozen suite | The *artifact* is identified and measured | A public quality card may cite it as *contingent* |
| **R3** | One-command local repro on the **same machine class** (M2Max-96) | Anyone with that Mac reruns it | A public **win** may be claimed (minimum bar) |
| **R4** | Third-party repro on **another Mac** (different class) | Not "works on my box" — a memory-scaling fact | The **trust moat**; README badge/table |
| **R5** | The receipt **format itself** is cited/used by someone external | The category handle has taken | "condensation receipt" is becoming a standard |

**Rule:** no public *win* below **R3**. No "first/only/leads" claim below **R4**. The §4 scale
law's local points are R3; its 70B prediction is explicitly **a prediction, not a result**,
until the off-box run makes it R3+ (§8). Tag every sentence in the README and every model card
with its level — an R2 contingent claim stated as an R4 fact is itself an invalidation (§20.3).

### 20.7 The incumbent-resistance test (could this be copied in two weeks?)

For each major public claim, ask: *could an incumbent copy it in two weeks? If yes, why does our
version still matter?* The honest answer is usually **the technique is copyable; the
disincentive to publish it is the barrier.**

| Claim | Copyable in 2 weeks? | Barrier type | Why ours still matters |
|---|---|---|---|
| Codec-aware GPTQ-Hessian recovery (T1.1) | The *method* yes (it's published research, §17) | **Incentive / governance** | Incumbents optimize polished runtime or cloud scale; they are **disincentivized to publish adversarial BYO-local-condensation receipts** that invite "your tool is slower" — we're not |
| The condensation receipt format | Trivially (it's JSON) | **Social / provenance** | A format is worthless without a corpus of honest receipts behind it; the **trust is the moat, not the schema** (R5) |
| RAM-cliff demo | Anyone can record one | **Incentive** | Vendors won't film their product failing to load Q4; we *lead* with the failure |
| Out-of-core condensation | No — genuinely hard (§13d) | **Technical / economic** | The unbuilt intersection; the one barrier that is actually technical |
| The failure atlas + watchlist | Yes, but they won't | **Incentive** | Polished product pages don't ship public self-obsolescence monitoring (§19.6); a solo builder can |

**The sharpened thesis:** incumbents can copy every *technique*, but their **incentives** keep
them from publishing the adversarial, same-machine, fail-included receipts that are Hawking's
actual moat. The defensible barrier is **incentive + provenance (a trusted receipt corpus)**,
not a secret kernel. Treat any claim whose only barrier is "technical and copyable in two weeks"
as a **temporary** lead — bank it fast or don't lead on it.

### 20.8 Category-ownership language (own a word, ban the wrong ones)

**Primary phrase (repeat until it is the category handle):** **"condensation receipt."**
**Secondary phrases:** **"RAM-cliff density"** and **"BYO model condensation."**

**Banned phrases (any speed-race framing loses the thread — §11, §16-P3):** "faster than
llama.cpp", "fastest 2-bit", "fastest decode", "iso-quant tok/s win", "lossless", "best
quantizer", and any "first/only" still tagged [UNVERIFIED].

Three sentences, one per audience:
- **Homepage:** *"Hawking condenses your own model on a Mac and hands you a condensation
  receipt — proof it fits, and runs, where ordinary 4-bit can't even load."*
- **Technical abstract:** *"Hawking performs out-of-core, codec-aware gradient recovery
  (STRAND-trellis QAT/KD) of a pretrained LLM on Apple Silicon, emitting a machine-verifiable
  condensation receipt — effective bpw, parent-quality delta (PPL/KL on a frozen multi-window
  suite + capability tripwire), peak memory, baseline behavior, hashes, and exact commands — at
  reproduction level R0–R5."*
- **Skeptical-expert one-liner:** *"It's not a faster engine and not a new codec — it's a
  reproducible, same-machine receipt that says how small a given model gets at a declared
  quality gate before it stops loading, and whether you can rerun that on your own Mac."*

### 20.9 The 10x artifact (the receipt + the receipt-diff tool)

The single artifact with disproportionate leverage is the **`condensation_receipt.json`
(§20.3) plus a `receipt-diff` tool**. Incumbents publish tables; Hawking publishes auditable
deltas.

**File / folder structure (all inline, one repo, nothing scattered):**
```
receipts/
  schema/condensation_receipt.schema.json     # the v0.2 schema above, as a JSON Schema
  official/<model>/<recipe>__<commit>.json     # Hawking's own runs
  third_party/<machine_class>/<model>__<submitter>.json   # external Macs (R4)
  failures/<FAIL-NNN>.json                      # negative receipts (gate=fail)
tools/
  receipt_diff.py        # prints what changed between two receipts: bpw, ppl, kl, rss, prompts, machine
  receipt_verify.py      # validates a receipt against the schema + the §20.3 invalidation rules
QUALITY_CARDS.md         # human-readable cards, one per official receipt, + divergence rows
FAILURES.md              # the §20.5 template entries
WATCHLIST.md             # §20.11 lapping conditions
BASELINES.md             # §20.11 exact baseline commands + best-effort notes
```

- **Minimum-viable version:** the schema + `receipt_verify.py` + one real `official/` receipt for
  a 7B run. That alone makes every future claim auditable.
- **Gold-standard version:** the full tree above, with R4 third-party receipts, a `receipt_diff`
  rendered into `QUALITY_CARDS.md`, and the RAM-cliff predictor (§20.11) cross-checked against
  real `peak_rss_gb` fields.
- **What makes it trusted:** hashes (source + artifact), the failure receipts sitting next to the
  wins, third-party rows that sometimes *disagree* and are shown anyway.
- **What makes it invalid:** any receipt that fails §20.3's eight rules; a `receipt_diff` that
  compares across different `prompt_suite_hash` without flagging it.
- **How it compounds:** each honest receipt raises the cost of dismissing the next claim; a
  *corpus* of reproduced receipts (R4) is the moat §20.7 says incumbents won't build.

### 20.10 The "embarrass us before launch" checklist (run the adversary's experiment first)

Before any public artifact, every box must be checked on the **same machine**, with results in
the receipt:

- [ ] Can a **tuned MLX 4-bit** match or beat this quality card at ≤ the same eff-bpw? (run it)
- [ ] Can **llama.cpp fit the parent with a different K-quant / IQ-quant** we didn't try?
- [ ] Can **Unsloth Dynamic 2.0** (the per-layer mixed-bit GGUF) match it at equal quality?
- [ ] Did we compare **under the same memory pressure** (same KV length, same machine, same
      resident load) — not a generous Hawking run vs a starved baseline?
- [ ] Are the **prompts frozen** (`prompt_suite_hash` published, seed + tokenizer pinned)?
- [ ] Are **artifact + source hashes** published?
- [ ] Does quality **hold on a less friendly prompt suite** (the pessimistic-mode run, §19.8)?
- [ ] Is the headline number from **CPU-bf16**, with MPS used only for the lab (§3)?
- [ ] Is the claim **tagged R3+** and labelled `density` vs `cliff` correctly?
- [ ] Are all baselines marked **tuned**, not `best_effort`, if they support a *win*?

**Any unchecked box downgrades the claim to contingent.** This checklist is §20.2's adversary,
turned into a gate the project runs on itself first.

### 20.11 Hawking-specific deepening — the consolidated artifact specs (all inline)

These are the Hawking-specific proof artifacts. **Each is specified here inline as a file +
field-list + rules — do NOT scatter them into standalone planning files.**

**`BASELINES.md` — baseline neutrality spec.** The honest baseline set, each with its **exact
command** and a **best-effort note**:
```
- llama.cpp Q4_K_M:    llama-quantize ... ; llama-bench -m ... -p ...   [tuned: tried Q4_K_M, Q4_K_S, IQ4_XS]
- llama.cpp mmap OOC:  llama-cli --no-mmap=false ...                    [best-effort: out-of-core serving baseline for §5]
- MLX 4-bit:           mlx_lm.convert -q --q-bits 4 ... ; mlx_lm.generate ...   [tuned: group sizes 32/64]
- Unsloth Dyn 2.0:     <HF dynamic GGUF id> in llama.cpp                [tuned where a dynamic GGUF exists]
- EXL3/PonyExl3:       <only where runnable on the target Mac>          [best-effort: in-core only; note if N/A]
```
Rule: a baseline marked **best-effort** can support a *contingent* or *negative* claim but
**never a public win** (§20.3 rule 8). No rhetorical sandbagging — if a baseline beats Hawking,
its receipt ships unchanged.

**`WATCHLIST.md` — lapping-condition spec.** One row per threat, each with an explicit
**invalidation trigger** (what, if it ships, kills part of Hawking's wedge):
```
| watched      | what would lap us                                  | check cadence | invalidates |
| MLX-QAT      | one-command "condense+recover into MLX format"      | monthly       | the Apple-native recovery half |
| Pruna        | an LLM-on-Apple-Silicon path                         | monthly       | the BYO-condenser category corner |
| PonyExl3     | adds even light LoRA recovery to trellis-on-Metal   | monthly       | "trellis + Metal + recovery" combo |
| EXL3/QTIP    | an official Metal port                               | monthly       | trellis-on-Mac serving |
| Unsloth      | a recovery (gradient) step on top of dynamic bits   | monthly       | the recovery-depth claim |
| BitNet/QAT   | a downloadable native-ternary model > ~2.4B, BYO    | monthly       | the "BYO → 2-bit" gap (§13b) |
```
This is the §19.8 "lapping alarm" made a spec: public self-obsolescence monitoring builds trust.

**`FAILURES.md` — structured failures.** The §20.5 template; tags + severity (`warn|fail|
wedge-threat`); each entry carries a real `gate=fail` receipt.

**`QUALITY_CARDS.md` — model-density quality cards.** One human-readable card per official
receipt: `{model, eff-bpw, RAM-fit, %-parent-quality (tripwire aggregate), tok/s, PPL-delta,
worst-window, KL, baseline deltas, repro_level}` + **divergence rows** for disagreeing
third-party receipts.

**RAM-cliff prediction table (the `ram_cliff_predict` artifact).** Predict, per Mac memory
tier, whether Q4 / MLX-4bit / Hawking-condensed *should* fit — then mark each prediction
**verified** or **falsified** against a real `peak_rss_gb`. The 96 GB row is anchored to §1/§5;
the others are predictions until a real receipt lands.

| model | Mac tier | Q4_K fit? | MLX-4bit fit? | Hawking TQ2 fit? | status |
|---|---|---|---|---|---|
| 14B | 18 GB | tight | tight | **yes** | predicted |
| 32B | 36 GB | tight+KV | tight | **yes** | predicted |
| 32B | 96 GB | yes (density, not cliff) | yes | yes | **anchored (§5)** |
| 70B | 96 GB | **borderline / swaps w/ long KV** | swaps | **yes (the cliff)** | predicted→verify |
| 405B | 96 GB | no | no | no (512 GB story, §8) | anchored |

A falsified prediction is a publishable negative (§20.5, `memory-prediction` tag) and
**re-calibrates the table**.

**Failure atlas by layer/model-family (the "one bad layer" artifact).** Per family, publish
which **layer class** (attn vs MLP, which projection, which depth band) dominates recovery loss,
derived from the recovery ledger (§6). A reference artifact for the quantization community even
if Hawking's product angle stalls.

**Third-party receipt PR template (inline):**
```
## Third-party condensation receipt
- Mac model + RAM:        <e.g. MacBook Pro M3 Max, 64GB>
- machine_class:          <M3Max-64>
- model + source_sha256:  <verified against the published hash? yes/no>
- hawking_commit used:    <sha>
- command run:            <copied from BASELINES.md / quality card>
- receipt attached:       receipts/third_party/<class>/<model>__<me>.json
- raw log attached:       <path/url>
- did your numbers match within tolerance (bpw ±0.05, ppl ±0.3pp, RSS ±5%)?  yes / no / details
```

**Quality-gate severity levels (`pass|warn|fail|invalid`).** `pass` = clears the §6 bar (≤+2%
ppl, KL in band, tripwire held, R3+). `warn` = within a defined band (e.g. +2–4% ppl) — ships
only as a contingent/portfolio card, never a launch claim. `fail` = misses the gate, ships as a
`FAILURES.md` entry. `invalid` = violates a §20.3 rule (the run does not count; the reason is
published).

**Prompt-suite freeze policy.** One versioned cassette: frozen prompts + seed + tokenizer +
expected parent outputs, identified by `prompt_suite_hash`. The demo video and every public card
use **that suite only**. Changing the suite bumps `prompt_suite_version` and re-runs all cards
(no silent prompt swaps — that is a `receipt_diff` red flag).

**Parent-model license + derivative-artifact policy.** Every published condensed artifact
carries `source_license`, `source_sha256`, `artifact_sha256`, and a `LICENSE_DERIVATIVE` note
stating the parent's terms and that the artifact is a derived, recovered quantization. Provenance
is resolved *before* publication, never after a takedown.

**Energy / wall-clock cost reporting.** Every receipt reports `wall_clock_s` and `energy_kwh`
(estimated is fine, label it). Apple-Silicon local condensation owns a quiet **"no rented GPU"**
economic/ecological proof line (§19.8 #8) — a supporting line, never the headline.

**The three load-bearing definitions (state them once, cite them everywhere):**
- **"Near-parent quality"** = PPL-delta ≤ +2% vs the f16 parent on the frozen multi-window suite
  (worst window, not mean) **AND** the `multi_eval.py` tripwire held within ε **AND** KL in band.
- **"Q4 cannot fit"** = on the named machine, `baseline_q4_load_result` ∈ {oom, swap-thrash} OR
  loads but at tok/s below a declared usability floor (decide the floor before the run — §19.7).
- **"Hawking wins"** = a **density** win (smaller at near-parent quality, both fit) OR a **cliff**
  win (loads + serves where Q4 cannot), with an **R3+ receipt**, a **tuned** baseline, and the
  claim_type labelled correctly. Anything less is *contingent*, not a win.

### 20.12 If it fails, it still wins (real deliverables, not consolation)

If recovery never beats a tuned dynamic GGUF at equal quality (the §20.1 kill condition fires),
Hawking still ships four **real, independently valuable** artifacts — each a deliverable a strong
engineer would respect on its own:

1. **The condensation-receipt harness + `receipt_diff`/`receipt_verify`** (§20.9) — a
   reproducible, hash-backed, baseline-neutral proof format. Useful to *anyone* benchmarking
   local quantization, Hawking's recovery aside.
2. **The RAM-cliff predictor** (§20.11) — a verified-or-falsified memory-fit table across Mac
   tiers. A genuinely useful tool for the whole local-LLM community.
3. **The failure atlas** (§20.11) — which layer classes dominate low-bit recovery loss, by model
   family. A reference contribution to the quantization literature.
4. **The from-scratch Rust/Metal inference engine + out-of-core doctor** — the systems-credibility
   flagship (§14): a solo, from-scratch engine and an out-of-core condensation pipeline that
   already wins on density (~52% smaller).

These are not a fallback narrative — they are **four shippable artifacts that exist regardless of
whether the recovery thesis lands.** The recovery win, if it comes, is the *fifth* and headline
deliverable; the first four are banked the moment the harness emits its first honest receipt.
**That is the asymmetry: in the failure case Hawking is still a receipt-backed science of local
density and memory cliffs; in the success case it also owns the wedge.**

### 20.13 The standard, in one sentence

> **Hawking does not ask to be believed. It asks to be rerun.** Every claim carries a
> reproduction level; every win carries a tuned baseline and a hash; every failure carries a
> receipt; every wedge condition carries an invalidation trigger. The methodology is the moat —
> not because it is grand, but because there is no single sentence a hostile reader can write
> that the receipt corpus has not already answered.

**The first artifact to build:** the **`condensation_receipt.json` schema (§20.3) +
`receipt_verify.py`**, wired so **every Studio run emits a receipt from day one** (§19.7).
Without it the experiments produce private logs; with it, every experiment — win, null, or
failure — manufactures public, rerunnable proof.

---

## 21. Pre-Studio Scaffolding (do now on M3 Pro) + The Studio Go-Prompt

> **Purpose:** the user is about to buy a **Mac Studio M2 Max (96 GB / 2 TB)** and will move
> this project onto it. Everything in this section that does **NOT** need the Studio is meant to
> be done **NOW**, on the current **M3 Pro 18 GB**, so the transition is *"just press go."* The
> boundary rule is mechanical: anything that needs **>18 GB RAM** or **long heavy compute** is
> STUDIO-ONLY and is listed explicitly so it is never started on the small box. Anything that is
> a download-that-fits, a dependency pin, an instrument-build, or a tiny smoke test is DO-NOW.
> **Build the instrument before the run** (§20.9): the receipt harness is DO-NOW, not Studio-day.
>
> **Audit baseline captured 2026-06-27 on the M3 Pro 18 GB** (read-only, nothing installed):
> Rust `cargo 1.94.1 / rustc 1.94.1` (Homebrew), `target/` present (18 GB, debug). Python
> `python3.12` (framework build) with **all key deps already installed**: torch 2.6.0,
> transformers 5.6.2, safetensors 0.7.0, numpy 2.2.6, **mlx 0.31.2 + mlx-lm 0.31.3 (verified
> importing and running)**, huggingface_hub 1.13.0 (+ `hf` CLI), accelerate 1.13.0, datasets
> 4.8.5, jsonschema 4.24.0. On disk: `scratch/qwen-05b` (953 MB), `scratch/qwen-15b` (2.9 GB),
> `scratch/qwen-7b` (14 GB, **HF bf16 safetensors parent** — the real doctor input), three calib
> corpora (`calib_corpus.txt` 392 KB, `calib_corpus_big.txt` 1.9 MB, `calib_multidomain.txt`
> 2.0 MB), `models/qwen32b-gguf` (18 GB, Q4_K_M sharded — **GGUF, not a bf16 parent**), plus 7B/3B/
> 1.5B/0.5B GGUFs and an MLX-4bit 7B. **The live 7B doctor is running and swap-thrashing exactly
> as the plan predicts** (frontier conductor `branch=waiting-for-v3`, run log shows
> `swap=25305MB`, the v3 doctor configs all dying on the 6000 MB ceiling / 120-min timeout /
> leaked-semaphore — the measured 18 GB dead-end, live). **Free disk is the binding constraint:
> only ~55 GB free on the 460 GB volume (87 % full)** — this dominates what can be downloaded now.

### 21.A — Readiness verdict

> **READY to scaffold; NOT ready to download the big parents.** The toolchain and *every* Python
> dependency the plan needs are already present (Rust builds, torch/transformers/mlx/jsonschema
> all import) and the proven 0→3 spine runs today — so the entire instrument layer (receipt
> schema + verifier + folders + stubs + smoke tests) can be built NOW with zero installs. The
> blocker is **disk, not RAM, not deps**: ~55 GB free cannot hold the 14B bf16 (~28 GB) **and** a
> 32B bf16 parent (~64 GB) **and** the 70B serve parent — those wait for the 2 TB SSD. The 7B
> frontier is mid-dead-end (swap-bound, as predicted) and will only complete on the Studio. Net:
> ~90 % of the non-compute scaffolding is doable today; the missing 10 % is purely "buy the disk."

### 21.B — The scaffolding checklist

#### [DO NOW on M3 Pro 18 GB] — fits in 18 GB RAM and on ~55 GB free disk, safe on a sometimes-mobile machine

**B0. Pin the environment (the reproducibility floor).** *Why:* receipts cite a `hawking_commit`
and a machine; a drifting env makes them unrerunnable. *What:* freeze the exact dep set so the
Studio installs byte-identical.
```
python3.12 -m pip freeze > docs/plans/studio_pinned_requirements.txt   # ~1 s, <1 MB, trivial RAM
cargo --version > /dev/null && rustc --version    # already 1.94.1 — record it in the receipt machine block
```
*Safety:* read-only on weights; pure metadata. **Safe mobile.** (Everything imports already, so no
installs are needed — this just *captures* the working set for the Studio.)

**B1. Pre-stage the parents that FIT on ~55 GB free — 7B and 14B bf16 only.** *Why:* §4/§7 need a
**bf16 HF parent** per rung (the 7B one is already here; the 14B is the next floor point and is the
biggest one that fits the current disk). *What:* the exact HF ids the plan needs and their local
status —

| rung | exact HF id | bf16 size | local now? | action |
|---|---|--:|---|---|
| 0.5B | `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB | **yes** (`scratch/qwen-05b`) | none |
| 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | **yes** (`scratch/qwen-15b`) | none |
| 7B | `Qwen/Qwen2.5-7B-Instruct` | ~15 GB | **yes** (`scratch/qwen-7b`, bf16 safetensors) | none |
| 14B | `Qwen/Qwen2.5-14B-Instruct` | ~28 GB | **NO** | **DOWNLOAD NOW (fits)** |
| 32B | `Qwen/Qwen2.5-32B-Instruct` | ~64 GB | **NO** (only Q4_K GGUF present) | **STUDIO** (too big for 55 GB free) |
| 70B (serve) | `Qwen/Qwen2.5-72B-Instruct` | ~140 GB | **NO** | **STUDIO** |

```
# 14B bf16 parent — the next floor point; ~28 GB, fits in 55 GB free with room to spare.
hf download Qwen/Qwen2.5-14B-Instruct --local-dir scratch/qwen-14b
# size ~28 GB · disk after ≈27 GB free · RAM during download trivial (streamed) · ~30–90 min on home wifi
```
*Safety:* **download only, never loaded into RAM on the 18 GB box** — a 14B bf16 forward would swap-
die, so DO NOT doctor or even ppl it here; staging the bytes is safe. **Safe mobile** (resumable;
`hf download` checkpoints). **Do NOT also pull 32B/70B bf16 now — they will not fit on 55 GB.**

**B2. Confirm the MoE + teacher ids the plan references (download only if disk allows; otherwise
record the list).** *Why:* T1.4 (MoE) and T2.5 (big-teacher KD) name specific models; pin the ids
so the Studio pulls them without re-deciding. *What:* the canonical list — `deepseek-ai/DeepSeek-V2-Lite`
(~31 GB bf16, MoE 16B/2.4B-active — **STUDIO**, won't fit now), `Qwen/Qwen3-30B-A3B` (MoE — **STUDIO**),
`mistralai/Mixtral-8x7B-v0.1` (~94 GB — **STUDIO/serve-only**). None fit the current 55 GB; **action
now = write the ids into `BASELINES.md` (B5) as the download manifest, download nothing.**

**B3. THE FIRST ARTIFACT — build the receipt instrument (§20.9) entirely now.** *Why:* §20.13 / the
§20 closing line: *every run emits a receipt from day one* — without it experiments produce private
logs. This is pure code + JSON, **zero RAM, zero downloads**, and is the single highest-leverage
DO-NOW item. *What:* create the §20.9 tree and populate it from the §20.3 v0.2 schema —
```
receipts/
  schema/condensation_receipt.schema.json   # the §20.3 v0.2 schema, as a strict JSON Schema
  official/                                  # R3+ wins (empty skeleton + .gitkeep)
  third_party/                               # external-Mac R4 drops (empty + .gitkeep)
  failures/                                  # gate=fail receipts (empty + .gitkeep)
tools/condense/receipt_verify.py             # validates a receipt vs schema + the §20.3 invalidation rules
BASELINES.md   WATCHLIST.md   FAILURES.md    # the §20.11 stubs
```
- **`condensation_receipt.schema.json`** — encode every §20.3 field; mark `repro_level`,
  `claim_type`, `effective_bpw`, `source_sha256`, `artifact_sha256`, `commands`, `hawking_commit`,
  `multiwindow_n` as **required**; enum the four `quality_gate` levels and the `repro_level` R0–R5.
- **`receipt_verify.py`** — load schema (jsonschema **already installed**), validate, then apply the
  eight §20.3 invalidation rules in code (e.g. fail if only `nominal_bpw` set; if `multiwindow_n<4`;
  if MPS headline without CPU-bf16 confirm; if `baseline_best_effort && claim_type` is a win). Exit
  non-zero + print the `invalidation_reasons` array. *Smoke it now* against one hand-written R0
  fixture and one deliberately-invalid fixture (B6).
- **Stubs:** `BASELINES.md` (the B2 download manifest + the exact tuned-Q4 / MLX-4bit baseline
  commands, each flagged best-effort per §20-BASELINES), `WATCHLIST.md` (the §20.11 rows: MLX-QAT,
  Pruna, PonyExl3/EXL3, BitNet/vendor-QAT, each with its lapping trigger), `FAILURES.md` (seed it
  with the **already-real** entry: *FAIL-001 — 7B LoRA recovery swap-death at 6000 MB on 18 GB*,
  pulled straight from `reports/cron/7b_frontier.jsonl` — a true negative receipt on day one).
*Safety:* code + JSON only, no model touched. **Safe mobile.** Est: a focused session, <5 MB on disk.

**B4. Emit a REAL receipt from the data already on disk (prove the harness end-to-end).** *Why:* a
verifier with no real input is untested; the 0.5B PTQ results already exist
(`scratch/qwen-05b-tq3.safetensors.json`, the frontier 7B `.jsonl`). *What:* write a tiny
`tools/condense/emit_receipt.py` that reads an existing ladder result + the source/artifact hashes
and emits a schema-valid `receipts/official/qwen-05b-tq3.json` tagged **`repro_level: R1`,
`claim_type: baseline`** (0.5B never sets a verdict — §0 rule 5 — so `baseline`, not a win). Run it,
then `receipt_verify.py` it. *Safety:* hashing + JSON; 0.5B forward (≈1 GB) fits 18 GB trivially if
you choose to recompute ppl, else reuse the logged numbers. **Safe mobile.**

**B5. Freeze the prompt-suite + fixtures (the §20.3 `prompt_suite_hash`).** *Why:* every receipt
cites a frozen suite; a moving suite voids cross-run comparison. *What:* assemble the held-out
multi-window eval set + the `multi_eval.py` tripwire tasks (qa/cloze/math/code) into
`prompts/frozen/suite_v1/`, compute its sha256, and record `prompt_suite_version: v1` +
`prompt_suite_hash` as constants the harness reads. *Safety:* text only. **Safe mobile.** <5 MB.

**B6. Tiny validation / dry-runs that fit in 18 GB (0.5B / 1.5B smoke).** *Why:* prove the receipt
emission + a baseline command run green on the small box so the Studio inherits a *working* path,
not a first-run debug. *What (all fit ≤3 GB RAM):*
```
# 0.5B baseline ppl (already memory-safe + checkpointed) → feeds a baseline receipt
python3.12 tools/condense/audit_ladder.py scratch/qwen-05b 0.5B smoke reports/smoke_05b   # ≈1 GB RAM
# 1.5B sanity (still fits 18 GB) — confirms recipe transfer plumbing, not a shipped number
python3.12 tools/condense/audit_ladder.py scratch/qwen-15b 1.5B smoke reports/smoke_15b   # ≈4–6 GB RAM
# verify both emit schema-valid receipts
python3.12 tools/condense/receipt_verify.py receipts/official/*.json
```
*Safety:* 0.5B trivial; 1.5B fits but **do this when plugged in / not mobile** (4–6 GB + the live 7B
doctor already swapping — run it only if you first confirm the 7B doctor is idle, or accept it queues).
**Do NOT smoke the 7B/14B here — they swap-die (that is the whole reason for the Studio).**

**B7. (Optional, disk-permitting) Pre-stage the eval datasets.** The HF `datasets` cache is only 1.9 GB;
pre-pull the wikitext/held-out eval shards `multi_eval.py` uses so the Studio is offline-ready. Tiny,
safe, mobile. Skip if disk gets tight after B1.

#### [STUDIO-ONLY — do NOT run now; needs >18 GB RAM or long heavy compute]

- **Clean the 7B floor point** — re-run the parked frontier (`run_7b_frontier.sh`, `branch=waiting-for-v3`)
  with the raised swap ceilings (§3). On 18 GB it is dying at 25 GB swap **right now**; only 96 GB
  makes the point clean.
- **Full-rank / block-wise codec-aware QAT (layer 4, `doctor_blockwise.py`)** on 7B/14B/32B — the #1
  Studio unlock; training, swap-death on 18 GB.
- **Download + doctor 14B and 32B bf16 parents** — 14B *download* is DO-NOW (B1), but **doctoring** it
  is Studio; the **32B bf16 (~64 GB) download itself is Studio** (won't fit 55 GB free).
- **The 70B/72B serve parent download (~140 GB) + the RAM-cliff tps bench (§5)** — Studio disk + RAM.
- **Codec-native GPTQ-Hessian error-feedback (T1.1, layer 5)** at 14B/32B — heavy compute.
- **Deep distillation logit+feature+attn (layer 6)** with a resident teacher — two models > 18 GB.
- **The full bit-floor-vs-scale curve at 7B→32B (§4)** and **full-rank QAT at scale** — the headline,
  all Studio.
- **MoE per-expert allocation (T1.4)** on DeepSeek-V2-Lite / Qwen3-30B-A3B / Mixtral — all Studio downloads.
- **Native `.tq` Stage B kernel + 70B cliff** — Studio RAM.

### 21.C — THE STUDIO GO-PROMPT (paste this into Claude on the Mac Studio)

```
You are running on the new Mac Studio M2 Max — 96 GB unified memory, 2 TB SSD, Apple Silicon
(Metal/MPS only, NO CUDA). The project is Hawking, at ~/Downloads/hawking (move it here intact;
the M3-Pro scaffolding — receipts/, receipt_verify.py, BASELINES/WATCHLIST/FAILURES.md, the
14B bf16 parent in scratch/qwen-14b, the frozen prompt suite — is already in place). Read the
canonical plan FIRST and treat it as the contract:
  docs/plans/studio_maximization_2026_06_27.md
Read at minimum: §4 (the bit-floor-vs-scale curve — THE deliverable), §7 (sequencing), §2 (the
recovery stack layers 0–6), §20 (the proof system), and §21 (this scaffolding section).

LOCKED CONTEXT — do NOT reopen any of these:
- Hardware is fixed: this 96 GB Studio. No cloud, no CUDA, no 512 GB box, no team. One project
  owns the whole machine, one heavy job at a time. Wall-clock is FREE (plugged in 24/7) — optimize
  for maximum quality/proof, not speed. bf16 throughout. The off-box 70B+ tail (§8) stays off-box.
- Respect the five measured dead-ends in §0: low-rank LoRA plateaus (use full-rank), NO uniform-STE
  through the trellis (codec-aware only), AWQ×residual is a non-win, calib = domain-matched not
  diverse, judge low-bit on 7B+ NEVER on 0.5B. Do not re-run any of them to "check."

BUILD THE INSTRUMENT BEFORE THE RUN. The receipt harness already exists from the M3-Pro phase —
your FIRST act is to confirm it: run `python3.12 tools/condense/receipt_verify.py receipts/official/*.json`
and make sure it passes. From then on, EVERY run — win, null, or failure — emits a
condensation_receipt.json (schema §20.3 v0.2) and is verified. No number is spoken without a receipt.

PROOF DISCIPLINE (non-negotiable, §6 + §20):
- Report EFFECTIVE bpw only (baker AGGREGATE), never nominal.
- Quality = output-space ppl-delta vs the f16 parent with MULTIWINDOW≥4 (report the WORST window)
  AND the multi_eval.py capability tripwire. A floor claim is void if ppl passes but a capability collapses.
- Production headline numbers are CPU-bf16 (DOCTOR_DEVICE=cpu, DOCTOR_DTYPE=bfloat16, STRAND_NO_GPU=1).
  MPS is ONLY for the 0.5B/1.5B recipe lab; re-confirm any chosen recipe on CPU-bf16 before it ships.
- Tag every claim with its repro level R0–R5 (§20.6). No public WIN below R3; no "first/only" below R4.
  The 70B/405B floor is a PREDICTION (T3.1), not a result, until an off-box run makes it real.
- Raise the swap ceilings per §3 (soft ~60 GB, hard ~80 GB) so the watchdog only fires on genuine
  32B over-commit — the 6000 MB death that killed the 18 GB 7B frontier must not recur below 32B.

EXECUTE IN THIS ORDER (§7), stopping at each gate to emit a receipt and decide continue/stop:
1. Clean the 7B floor point: re-run run_7b_frontier.sh (branch waiting-for-v3) to completion — it
   was swap-bound on 18 GB; here it must finish clean. Emit the 7B floor receipt.
2. Build full-rank/block-wise codec-aware QAT (doctor_blockwise.py) to a REAL number on 7B; confirm
   it beats the LoRA plateau on held-out (CPU-bf16). This is the lever the 18 GB box never had.
3. Download the 32B bf16 parent (Qwen/Qwen2.5-32B-Instruct, ~64 GB — now there is disk) and doctor
   14B + 32B; place each floor point with the full L0→L6 stack. (14B parent already staged.)
4. Plot the 5-point bit-floor-vs-scale curve (§4): x=log10(params), y=floor eff-bpw, recovered-vs-PTQ
   band. GO/NO-GO on H1 (monotone descent) vs H0 (flat). REPORT EITHER WAY — no fake GO.
5. Native .tq serve Stage A → Stage B + residual two-part serve; then the 70B serve-only cliff bench
   (§5) — download the 72B parent for the speed demo only (no doctoring).
6. Deep-KD recipe (layer 6) on the 0.5B/1.5B lab (MPS), transferred up; codec-native GPTQ-Hessian
   error-feedback (T1.1, layer 5) at the 1-bit edge on 14B/32B.

STOP/CONTINUE GATES: between every numbered step, write the receipt, check it verifies, and if a
result is a clean negative (e.g. H0 flat, or T1.1 ties plain residual) record it in FAILURES.md as a
first-class artifact and continue — do not retry a clean dead-end. If a third-party/repro disagreement
ever appears, pause and make reproducibility the job (§20.3).

Begin by reading the canonical doc, confirming the receipt harness verifies, then start step 1.
```
```
END GO-PROMPT.
```

---

## Sources (web-checked 2026-06-27)

- MLX vs llama.cpp / Apple Silicon engines (MLX ahead on M5, Ollama-MLX): https://arxiv.org/abs/2511.05502 ; https://ollama.com/blog/mlx ; https://machinelearning.apple.com/research/exploring-llms-mlx-m5 ; https://v-chandra.github.io/on-device-llms/
- MLX quantization / AWQ-in-MLX / mixed-bit (mlx-optiq): https://github.com/ml-explore/mlx-lm ; https://mlx-optiq.pages.dev/ ; https://developer.apple.com/videos/play/wwdc2025/298/
- QuIP# / QTIP / AQLM (trellis, ≤2-bit SOTA): https://www.together.ai/blog/even-better-even-faster-quantized-llms-with-qtip ; https://arxiv.org/pdf/2406.11235
- ExLlamaV3 / EXL3 (QTIP-trellis, CUDA-only): https://github.com/turboderp-org/exllamav3 ; https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md
- PonyExl3 (EXL3→Metal port, converter + on-the-fly trellis decode, in-core): https://github.com/beamivalice/PonyExl3
- Unsloth Dynamic 2.0 (per-layer mixed-precision, model-specific): https://unsloth.ai/blog/dynamic-v2 ; https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
- Gemma 3/4 QAT int4 (vendor-shipped recovery): https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/ ; https://huggingface.co/google/gemma-3-27b-it-qat-q4_0-unquantized
- BitNet b1.58 + bitnet.cpp (native low-bit, 2.4B ceiling on downloadable weights): https://arxiv.org/abs/2504.12285 ; https://github.com/microsoft/BitNet ; https://huggingface.co/microsoft/bitnet-b1.58-2B-4T ; https://en.wikipedia.org/wiki/1.58-bit_large_language_model
- ParetoQ (Meta — unified 1/1.58/2/3/4-bit QAT, the 2-bit learning transition): https://arxiv.org/abs/2502.02631 ; https://pytorch.org/blog/paretoq-scaling-laws-in-extremely-low-bit-llm-quantization/
- Low-bit recovery wave 2025–2026 (QAD beats QAT, Recover-LoRA, progressive PTQ+QAT): https://arxiv.org/pdf/2506.09104 ; https://arxiv.org/html/2606.04238 ; https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf
- MLX-LM QAT/LoRA on-device: https://pypi.org/project/mlx-lm-lora/
- Out-of-core: llama.cpp mmap + AirLLM (layer-by-layer) + prima.cpp: https://github.com/ggml-org/llama.cpp/discussions/4310 ; https://nachoconesa.com/blog/airllm-llms-hardware-modesto ; https://arxiv.org/html/2504.08791v2
- Pruna AI (BYO compressor: 50+ algos incl. recovery, agentic config search; OSS Mar 2025; Linux-supported, image/video/cloud focus): https://www.pruna.ai/ ; https://www.pruna.ai/open-source ; https://docs.pruna.ai/en/stable/setup/install.html ; https://techcrunch.com/2025/03/20/pruna-ai-open-sources-its-ai-model-optimization-framework/
- Model-optimization consolidation (Deci→NVIDIA, Neural Magic→Red Hat, OctoML/OctoAI→NVIDIA): https://techcrunch.com/2024/11/12/red-hat-acquires-ai-optimization-startup-neural-magic/ ; https://www.startuphub.ai/neural-magic-acquired-red-hat-nvidia-and-amd-race-to-acquire-ai-model-optimization-startups/
- HQQ / AWQ / GPTQ comparison (≤2-bit behavior): https://kaitchup.substack.com/p/a-comparison-of-5-quantization-methods ; https://dropbox.github.io/hqq_blog/

**[UNVERIFIED] caveats:** PonyExl3's exact out-of-core/recovery boundaries; MLX-LM QAT's
practical low-bit ceiling and whether it adds a one-command condense path; Pruna's roadmap
toward LLM-on-Apple-Silicon; whether any add a STRAND-equivalent gradient-recovery-into-trellis
loop; and the §17 12-month-window figure (a judgment call, not a sourced forecast). Re-check
all of these directly before any public "first/only" claim. The competitive picture moves monthly.
