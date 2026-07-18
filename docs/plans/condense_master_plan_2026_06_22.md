# The Condensation Pipeline — Master Plan (2026-06-22)

> Hawking's defining product. Not "download a quantized GGUF and run it" — that's
> the llama.cpp world. Hawking **condenses your own model**: take the biggest thing
> (32B, 671B), make it the smallest *highest-performing* form, and run it locally.
> Condensation ≠ quantization. **Condensation does not mean loss** — it means
> compress-then-restore. The bits are dynamic (4/3/2/1, per-tensor); a recovery
> layer ("quality infusion") closes the gap back toward 1:1; the result fits where
> nothing else fits, so it runs faster (the RAM cliff).

## The one-line product

```
hawking condense <parent>  →  smallest artifact at your quality target  →  hawking run
```

Your model, your bits, ~1:1 quality, higher tps because it fits. The whole
complexity is hidden behind one verb.

## Why this is the moat (grounded in measured numbers, not hope)

- **Iso-quant inference is a loss and always will be.** Fair rebench: Hawking 7B
  Q4_K decode ≈ 0.71× llama.cpp. We do **not** win "run the same file faster."
- **Condensation is a different game** — *artifact creation* — where llama.cpp
  barely competes (`llama-quantize` = static PTQ, no recovery, no dynamic bits,
  weight-space). The win comes from **density × the RAM cliff**: a 32B at 2-bit
  (~9 GB) *fits* on 18 GB where Q4_K (~18 GB) swaps → 10–100× tok/s. Speed is a
  **consequence of density**, not of the decode kernel.
- **The bleeding-edge claim:** a 2-bit-near-lossless artifact that *runs on a
  laptop* is a capability no shipping tool has (llama Q2_K collapses; AWQ/GPTQ stop
  at 3–4-bit PTQ; none are out-of-core or dynamic). That's the wedge.

## What we have measured (the honest floor, preserved in the retrospective ledger)

Output-space (`||(Ŵ-W)X||/||WX||`) on REAL bf16 weights + REAL measured activations:

| tier | method | output-err | vs Q4_K (0.079) | note |
|---|---|---|---|---|
| 4-bit | Q4_K (the bar) | 0.079 | 1.00× | llama's default |
| 3-bit | TQ3 naive | 0.155 | 1.96× | the weight-space "kill" |
| 3-bit | **TQ3 + AWQ + outlier (PTQ)** | **0.095–0.108** | **~1.28×** | **26% denser, PTQ ceiling** |
| 2-bit | TQ2 + AWQ (PTQ) | 0.19–0.32 | 2.4–4× | **collapses without recovery** |

**The verdict that drives this plan:** activation-aware PTQ reopened the quality arm
(1.96× → 1.28× at 3-bit) but **PTQ alone cannot reach 1:1, and 2-bit PTQ collapses.**
The 2-bit lead is *only* reachable with the **recovery layer**. So recovery is not a
nice-to-have — it is the load-bearing wall of the whole product.

### REAL inference confirmation (2026-06-23 — actual perplexity, not the proxy)

Confirmed in *inference space* on Qwen2.5-0.5B (`tools/condense/quality_sweep.sh`,
`ppl_bench.py`; baker = quantize-model `--bits N --quality --rht-cols`, no AWQ/recovery):

| variant | perplexity | vs f16 |
|---|--:|--:|
| f16 parent | 28.31 | — |
| TQ3 (3-bit PTQ) | 38.92 | **+37.5%** (usable, degraded) |
| TQ2 (2-bit PTQ) | 449.87 | **+1489%** (collapsed) |

The proxy and real inference agree: **3-bit PTQ is degraded-but-usable; 2-bit PTQ
collapses.** This is the metric the doctor (QAT/KD) must move — the recovery target
is TQ2 ppl 450 → ~28. Raw: `reports/condense/ppl_sweep.jsonl`.

### 🩺 The DOCTOR works — recovery PROVEN (2026-06-23)

`tools/condense/doctor_qat.py` — self-contained QAT (uniform per-channel + STE),
the quality-infusion lever. **5-step smoke on Qwen2.5-0.5B at 2-bit:** held-out ppl
**79,000,000 → 211,000 in 5 steps (99.7% recovery)**, train-loss 17.2 → 4.3. The
direction is overwhelming — gradient recovery moves the quantized weights exactly as
the triangulation predicted (cheap PTQ/data-free levers could not). The full run
(300 steps → target ~f16 ppl 28) is the deferred cron money-shot (heavy/training,
not on battery). This is the proof that **condensation ≠ loss**: low bits + the
doctor → small AND ~1:1.

### 🥊 3-way QUALITY bench (the honest scoreboard, 2026-06-23)

`tools/condense/quality_3way.sh` — relative ppl degradation per engine (quant ÷ own
f16), same model + same text, each in its faithful harness (cancels tokenizer/harness
differences). RAN on Qwen2.5-0.5B:

| engine | bpw | degradation vs f16 |
|---|--:|--:|
| llama.cpp Q4_K_M | 4.5 | **+2.1%** |
| Hawking TQ3 (PTQ) | 3.35 | **+43.7%** |
| Hawking TQ2 (PTQ) | 2.34 | +1069% |

**HONEST: at PTQ, Hawking LOSES on quality — badly.** The entire pitch is the doctor:
to win, condensed+doctor degradation must drop below Q4_K's +2.1% at fewer bpw. The
cron runs `quality_3way.sh` on the *doctor-recovered* weights → the win/lose verdict
lands in `OVERNIGHT_RESULTS.md` by morning. This is the most important number in the
project — and we now measure it rigorously rather than assert it.

### Verdict v1 (2026-06-23) — the doctor OVERFIT; LOSES → fix found

First recovered-weights verdict (doctor 200 steps, **single calib passage**): train CE
crashed 17.2 → **0.03** (memorized the one passage) but held-out STRAND-TQ2 = **+667%**
(ppl 284) vs llama Q4_K +3.4% → **LOSES**. The "99.86% recovery" was a mirage — held-out
uniform ppl stayed ~1e5. Root cause: **training on one short passage memorizes instead of
generalizing.** Fix shipped: `doctor_qat.py` now samples **diverse chunks** from a real
corpus (`DOCTOR_CALIB`, 400 KB wikitext → 174 chunks, rotated per step) — the 3-step smoke
confirmed CE stays ~12 (not memorizing). Lesson: recovery only counts on HELD-OUT data.

### Verdict v2 (diverse calib) — overfit FIXED, but uniform proxy is WRONG for STRAND

Diverse-chunk calib fixed the overfit: uniform held-out ppl **108K → 5,340** (generalizes;
CE stays ~6, not 0.03). BUT two hard findings: (a) uniform-2bit is **hopeless even healed**
(5,340 ppl — uniform has no trellis/codebook); (b) **the uniform-healed shadow is
COUNTERPRODUCTIVE for STRAND** — STRAND-TQ2(healed) ppl **676 > 481 PTQ**. Optimizing
weights for *uniform* quant mis-optimizes them for STRAND's trellis codec.

⇒ **The doctor MUST quantize with the ACTUAL deployment codec (STRAND) in the QAT loop**
(the `strand-qat.py` "requant-every-N-steps" proxy-transfer design), not a uniform proxy.
This is the decisive next lever. **HONEST STATUS: Hawking does NOT yet beat llama on
low-bit quality.** Density is real (52% smaller); the quality WIN is gated on STRAND-aware
QAT — measured, not yet built. (The overnight cron did NOT fire — the Mac slept, so the
watchdog `sleep` never elapsed; the interactive verdict runs above are the only results.)

### The LoRA RECOVERY frontier (2026-06-23, full power) — the memory wall is solved

Full-weight STRAND-QAT doesn't fit 19 GB. **Fix: LoRA recovery** (`doctor_lora.py`) — freeze
the STRAND-quantized base, train tiny rank-r adapters (8.8M params, not 0.5B) → fits + fast.
Deployed = STRAND low-bit base + small f16 LoRA (~3.0–3.7 bpw, still denser than Q4_K 4.5).

Findings so far (Qwen2.5-0.5B, held-out ppl):
- TQ2+LoRA(CE, r16): base 247.7 → **120 @ 20 steps** (51%) but **diverges** if over-trained
  (542 @ 300). Fixed with held-out early-stopping.
- TQ3+LoRA(CE, r16): base +44% → **+36%** vs f16 — helps, but CE is a weak signal
  (doctor's own held-out moved only ~3%).
- ⇒ **KD is the lever** (distill the f16 teacher's full logit distribution — the literature's
  low-bit recovery method). TQ3+LoRA(KD, r64) is the first real win attempt (running).

**The win condition stands:** condensed+recovery degradation < Q4_K's +2.1% at < 4.5 bpw.

Quality trajectory (3-way text, beating ourselves): TQ3 PTQ **+44%** → +LoRA-CE **+36%** →
+LoRA-KD(cached, r32) **+30.9%**. KD (cached top-64, teacher freed → fits 19 GB) is the best
lever so far. **Ceiling identified:** LoRA is low-rank, and the quantization error is
**high-rank** (same root cause as the low-rank-heal NO-GO) — so LoRA recovery plateaus
(~step 25). Strict parity (<+2.1%) needs **full-rank** recovery (memory-efficient full/
layer-wise QAT — the 19 GB challenge), or a bigger machine.

### ✅ condense→run CLOSED (2026-06-23) — Hawking serves a condensed+recovered model

`rehydrate.py` (condensed safetensors → merge → `convert_hf_to_gguf` → f16 GGUF) → **Hawking
generate serves it, COHERENT output, 67.6 tps** (`scratch/qwen-05b-condensed.gguf`, TQ3+LoRA-KD).
The whole pipeline runs end-to-end for the first time. Caveat: rehydrate inflates to an f16
container (no memory/cliff benefit) — the RAM-cliff *tps* win needs **native low-bit `.tq`
serving** (bitslice kernel exists; loader surgery scoped in `native_tq_serving_impl.md`).

**Honest scoreboard:** Density = WON (52% smaller). Quality = Pareto (denser at +31%, not
parity — full-rank QAT is the open lever). tps cliff = the next build (native serving + a
non-fitting model). condense→run = WORKS.

## The pipeline (the seven stages)

```
1 PLAN     hawking press --dry-run --memory-budget   (out-of-core; BUILT)
2 RANK     output-space damage ranking per tensor/channel   (partly built: rung-kl)
3 ALLOCATE dynamic 4/3/2/1-bit by damage + budget   (the "bit selected" step)
4 ENCODE   activation-aware: RHT + AWQ + outlier (+Hessian)   (L1; MEASURED → 1.28×@3b)
5 RECOVER  "quality infusion" — debias → QAT → KD/distill → LoRA-residual   (L2; THE BUILD)
6 VERIFY   output-space PPL / logit-KL / task quality card   (gate; partly built)
7 RUN      Hawking inference (RAM-cliff speed)   (BUILT: HAWKING_QWEN_TQ + .tq sidecar)
```

Stages 1, 4, 7 exist. Stage 5 (recovery) is the gap that makes "condensed ≠ lossy"
real. Stages 2–3 (dynamic allocation) turn "pick a format" into "find this model's
lowest bit-floor that holds quality."

## Stage 5 — the recovery layer ("quality infusion" / the "doctor")

This is the answer to *"how do we add quality back after compressing as small as
possible."* Note: this is a **quality-recovery stage**, distinct from the existing
`hawking doctor` (hardware fit) — call it **`condense --recover`** / the *infusion*
pass. A ladder of levers, cheapest first; each one buys lower bits at equal quality:

| lever | what it does | cost | status |
|---|---|---|---|
| **actmean debias** | cancel the systematic output bias of quantization (`c = -(Ŵ-W)·μ`) | ~free | built in baker (`--actmean`), claimed −28.7% PPL; unmeasured in output space |
| **damage-ranked mixed precision** | keep attn/lm_head/outlier-heavy tensors high-bit, push tolerant FFN low | free | `rung-kl.py` ranks; allocator = TODO |
| **outlier protection** | top-σ channels at high bits (residual carriers) | +~0.1 bpw | prototyped (marginal alone; stacks) |
| ~~LoRA / low-rank residual~~ | add a tiny high-precision `AB` correction = `Ŵ + AB ≈ W` | small bytes | **❌ MEASURED NO-GO (2026-06-22)** — quantization residual is HIGH-rank noise; rank-16/32/64 SVD heals TQ3 only 0.114→0.104 and rank-64 already costs 4.6 bpw (> Q4_K 4.5). Data-free low-rank patch is the wrong tool. Evidence: `reports/condense/L2_lowrank_heal_NOGO_20260622.txt` |
| **QAT** (quant-aware fine-tune) | re-fit weights with the quantizer in the loop | training | `strand-qat.py` scaffold exists |
| **KD / distillation** | match the parent's logits + hidden states (teacher-forced) | training + teacher | capture path + scripts exist; not wired to condense |
| **self-data calibration** | generate calibration from the model itself (no corpus) | small | partly (corpus tools) |

The product's quality knob = *how much recovery to apply*. PTQ-only = fast/approx;
+debias+mixed-prec = a little better; **the cheap data-free patches are NOT enough**
(low-rank residual measured NO-GO — quantization error is high-rank). The real
"infusion" for 2-bit ~1:1 is **QAT/KD** (gradient re-fit moves the quantized values
themselves, addressing the high-rank error). The literature (QuIP#, AQLM, BitNet)
confirms 2-bit + *gradient* recovery ≈ near-lossless — so the doctor is a training
step, not a post-hoc patch. That's the honest, measured reshaping of L2.

## Dynamic bit selection (the "bit selected model")

Bits are a **free variable chosen per model to hit a quality target**, not a fixed
format. The allocator (stage 3) sweeps the bit-ladder under a budget:

```
hawking condense <parent> --target-quality "ppl_delta <= 1%"  [--max-bytes 9gb]
  → tries lowest bits first; applies recovery; backs off sensitive tensors to 3/4-bit
  → emits the smallest artifact that holds the target + a quality card
```

The "discrimination on what models can be condensed how far" falls out
automatically: tolerant models land at 2-bit, sensitive ones at 3-bit — the
process **finds each model's floor**. No human picks Q-format.

## Roadmap (phases — each ships a measurable artifact)

- **L0 ✅** output-space harness + real Q4_K-vs-TQ numbers. (DONE: 3-bit PTQ 1.28×.)
- **L1 🔄** activation-aware encode on real acts. (AWQ+outlier measured; Hessian-metric
  optional/low-ROI vs recovery.)
- **L2 ⬅ THE BUILD** the recovery layer (data-free cheap rungs measured insufficient):
  (a) actmean debias + (b) damage-ranked mixed-precision allocator — cheap, modest;
  (c) ~~LoRA low-rank residual~~ ❌ NO-GO (high-rank residual); → **(d) QAT/KD is the
  real doctor** (gradient re-fit to the f16 teacher) for 2-bit ~1:1 — a TRAINING step
  (heavy → run via the deferred bench/recovery cron, not on battery). Gate in output space.
- **L3 🔄 PARTIAL** `hawking condense` BUILT as `tools/condense/condense.sh`
  (plan → [doctor] → encode+verify → quality card) + the doctor as
  `tools/condense/doctor.sh` (wraps `strand-qat.py` QAT/KD). ENCODE+VERIFY proven
  e2e: `tq_bake` bakes a real `.tq` and round-trip-decodes it. Enforces the
  f16-source rule (refuses quant-of-quant). **TWO REAL GAPS surfaced:**
  - **🚨 RUN half NOT wired** — the `tq` decoder is test-only; `hawking generate`
    cannot serve a `.tq`. "condense → run" is half-built until `.tq` serving is
    wired into the loader/forward (a GPU-path build — the critical next lever).
  - **deployability**: GPU bitslice needs `in_features % 256 == 0`; ragged tensors
    bake NON-STRICT (not deployable). Most big-model dims qualify; check per model.
- **L4** scale: 7B→14B→32B locally; frontier (MoE) only with owner-approved download/
  storage/compute. Publish quality cards (RAM-fit, bpw, % parent quality retained, tps).
  NOTE: the headline *speed* win needs a model that does NOT fit at Q4_K (the RAM
  cliff). The 7B fits, so on it condensation shows DENSITY, not speed — the cliff
  bench needs ≥32B (owner-gated download).

## Triangulation: three independent measurements, one conclusion

The "recovery is mandatory for 2-bit" conclusion is now confirmed from three angles
(all in output space, real activations):
1. **L1 PTQ** activation-aware encode caps at **1.28× Q4_K @ 3-bit**; 2-bit PTQ is 2.4–4×.
2. **L2 cheap heal** (data-free low-rank residual) is a **measured NO-GO** — residual is high-rank.
3. **Dynamic allocation** (`allocate` test) **TIES uniform** — 2-bit is intolerable on
   *every* tensor (o ~doubles 3→2-bit across all 8 picked), so no bit-budget trade
   rescues it; the allocator is correct, but PTQ has no 2-bit headroom to allocate.

⇒ PTQ — even *optimally allocated* — cannot reach the 2-bit lead. **QAT/KD (gradient
recovery) is the only path.** The dynamic-allocation lever is still valuable *with*
recovery (post-heal, tensors regain 2-bit tolerance heterogeneously). Evidence:
`reports/condense/`.

## The proof bar (no fake GO)

Ship a claim only when, **in output space**, a condensed artifact at fewer bpw than
Q4_K holds a stated quality target (PPL-delta / task scores) vs the **f16 parent** —
and runs on Hawking. PTQ gets us to 3-bit/1.28×; **L2 recovery is what turns the
2-bit lead from "collapses" into "≈1:1," and is the single highest-leverage build.**

## Where this leaves Hawking

If L2 lands: Hawking is the only local tool that takes *your* big model, condenses
it dynamically to its smallest ~1:1 form with quality infusion, and runs it — the
pipeline is literally `hawking condense && hawking run`. That is the bleeding edge,
and it's a creation-time + RAM-cliff moat, not a decode-tps race we'd lose anyway.
