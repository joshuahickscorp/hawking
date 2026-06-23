# Doctor → ~1:1 quality + Speed, BEFORE the 32B (2026-06-23)

> "Have our cake and eat it too": condensed (lowest bit) AND ~1:1 quality AND faster.
> The 1.5B PTQ +29.2% (vs llama Q4_K +8.6%) is the **LoRA ceiling**, not the doctor's
> limit. Solve quality to near-zero degradation + nail speed, THEN do the 32B.

## Why the doctor plateaus today (the diagnosis)
- LoRA is **low-rank**; the quantization residual is **high-rank** (proven 3 ways:
  low-rank-heal NO-GO, dynamic-alloc tie, LoRA plateau). So adapters cap out (~+31% @0.5B).
- Full-weight QAT (the real fix) doesn't fit 19 GB naively; KD needs 2 models → swaps ≥1.5B.
- ⇒ The doctor needs FULL-RANK recovery that fits memory. That's the capability gap.

## Doctor capability ladder (ordered by impact → "absolute lowest degradation")
1. **AWQ-scaled base** (biggest cheap win): importance-scale columns by activation magnitude
   (σ^α) before quant, unscale at serve. Output-space 1.96×→**1.28× @3-bit** (measured in the
   harness). The base starts FAR closer → less for the doctor to fix. The baker lacks this;
   add `--awq <calib>` (per-column scale) or pre-scale in Python + fold the inverse into serve.
2. **Block-wise full-rank QAT** (the CEILING-BREAKER, fits 19 GB): for each transformer block,
   optimize its *quantized* weights to match the f16 block's OUTPUT on calib (local KD), ONE
   block at a time → memory = one block (tiny) + FULL-rank (fixes the high-rank error). This is
   the BRECQ/AdaRound/GPTQ-family method that achieves near-lossless 2–3-bit in the literature.
   Pipeline: capture f16 block I/O on calib → per-block optimize quantized W (Adam, STE through
   the STRAND codec via periodic re-bake) → next block. The principled path to ~1:1.
3. **KD with full teacher logits + LR schedule + more/better calib** (richer signal than CE).
4. **Damage-ranked mixed precision**: keep the most output-sensitive tensors at 4-bit, push
   tolerant ones to 2-bit → lowest avg bpw at a quality target (dynamic-bit selection).
5. Higher-rank LoRA only as a fallback (diminishing — low-rank ceiling).

**Target:** condensed+doctor degradation **< llama Q4_K's +8.6%** at **< 4.5 bpw** = the WIN
(denser AND ≥ Q4_K quality). Stretch: < +2% (true ~1:1).

## Speed ladder (the tps, before the 32B)
1. **Native GPU bitslice serving** (`strand_bitslice_gemv_tcb`, PROVEN in RWKV-7): wire into
   Qwen (`TqPreparedGpu` per linear + branch `matmul_q4_dispatch`). Keeps `.tq` compressed in
   RAM (~bpw/8 × params), decodes on-the-fly → real low-bit decode tps. (The CPU `matvec_rht`
   path Q12-inflates to f32 → OOM at scale; GPU is the serve path.)
2. **The RAM cliff**: density → fits → on a non-fitting model (32B), Hawking runs while llama
   Q4_K swaps = 10–100× (the headline; needs #1).
3. Decode-kernel levers (Throughput Bible): predec, f16-scales, 2r geometry — already largely
   shipped on the Q4_K path; port the wins to the TQ GEMV.

## Sequence (the user's directive)
A. **Quality first** — AWQ base (1) + block-wise QAT (2) → drive 1.5B/0.5B degradation toward
   ≤ Q4_K, ideally ~1:1. Bench each (3-way vs llama+MLX) to prove it.
B. **Speed** — wire native GPU TQ serving (#1 speed) → measure low-bit decode tps vs Q4_K.
C. **THEN the 32B** — with quality (~1:1) + speed (GPU serve) in place, condense the 32B →
   the cliff is the capstone, not the experiment.

State + tools in [[condense-32b-native-serving-2026-06-23]] and `tools/condense/`.

## ✅ BREAKTHROUGH (2026-06-23): AWQ + doctor — the levers compound toward ~1:1
Measured, 0.5B 3-bit (the HARDEST case, same 24KB held-out eval, ppl vs f16 36.71):
- TQ3 RHT base (old):        +42.9%
- TQ3 **AWQ** (no training): +18.7%   ← `tools/condense/awq_bake.py`, activation-aware, halves the gap
- TQ3 **AWQ + doctor**:      +14.8%   ← LoRA-KD r128 lr1e-4 on the AWQ base
- llama Q4_K reference:      ~+9-10% @ 4.9 bpw  (Hawking = 3.6 bpw, **20% denser**)

So on the pessimistic floor (0.5B), Hawking is denser at near-comparable quality. The path to
~1:1 / beating llama is REAL and compounding: AWQ (training-free, big) + doctor (LoRA-KD). Key
fixes that unlocked it: AWQ base (was missing from real bakes); doctor stability = LOWER lr for
higher rank (rank256/lr3e-4 DIVERGED; rank128/lr1e-4/top-128 KD is stable). Full-rank STRAND-QAT
DIVERGES (base-add STE drift — abandoned). Next: bigger models (1.5B+) should WIN; tune doctor
steps/alpha further; fold AWQ into the baker (--awq); then 2-bit, ternary, 1-bit same treatment.

## Measured method-comparison (the science, 0.5B 3-bit, ppl vs f16 36.7)
WHAT WORKS (operates WITH the STRAND codec):
- AWQ (activation-aware, training-free, scale-before-bake): +42.9% -> +18.7%  ★ biggest lever
- LoRA-doctor on the FROZEN STRAND base (KD, stable: low lr / high rank): +18.7% -> +14.8%
  => 3.6 bpw at +14.8%, 20% denser than llama Q4_K (~+9-10% @4.9bpw) on the HARDEST case.
WHAT FAILS (uniform-proxy, doesn't transfer to STRAND's trellis):
- Global full-weight STRAND-QAT (base-add periodic re-bake STE): DIVERGES (drift).
- Block-wise QAT (per-layer, uniform fake-quant, local MSE) -> STRAND-bake: +140,000% (CATASTROPHIC).
  Root cause: weights optimized for UNIFORM quant are WRONG for STRAND (confirmed global + block-wise).
CONCLUSION: the doctor = **AWQ + LoRA-KD on the frozen STRAND base**. Path to ~1:1: higher-rank
LoRA (tuned LR), AWQ alpha-sweep, more steps, and bigger models (more redundancy => the gap shrinks).
Uniform-proxy QAT (global/block-wise) and STRAND-in-loop full-rank are dead-ends for this codec.
1-bit: catastrophic on 0.5B (AWQ +2.5M%, +doctor +28K%) -> validates per-model bit-floor discrimination.
