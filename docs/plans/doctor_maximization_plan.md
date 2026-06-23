# Doctor Maximization Plan — Recovery to Outrun Compression's Decay

> Compression is down-packed (STRAND fractional bpw + AWQ + residual). The open frontier is the
> **doctor**: how much quality we instill back after the cut. **The economics: every point the
> doctor recovers is a point we can re-spend on more compression.** So maxing recovery isn't
> polish — it's how we push the bit-floor lower at the same quality. This plan puts as much of
> the doctor as possible on the local box (most of it, because the biggest levers are train-free)
> and reserves for the Studio only what truly needs the RAM/compute.

---

## The doctor as a recovery STACK (ordered by leverage, they compose)
Recovery is not one method — it's layers that stack. Measure each in output-space (ppl **and**
downstream), bpw always *effective*, not nominal.

| # | layer | what it recovers | cost | side |
|--|-------|------------------|------|------|
| 0 | calibration quality | the input to every layer below — multiplies all of them | data + forwards | **LOCAL** |
| 1 | AWQ pre-scale | protects high-activation channels before the cut (halves the raw gap) | 1 forward + bake | **LOCAL** |
| 2 | mixed-precision allocation | spends bits where the *output* hurts most → lowest avg bpw | sensitivity scan + bake | **LOCAL** |
| 3 | residual quant | full-rank, codec-native correction → ~1:1 (the breakthrough) | double-bake, train-free | **LOCAL** |
| 4 | LoRA-KD last-mile | squeezes the residual's remainder | training | LOCAL ≤1.5B / STUDIO 7B+ |
| 5 | codec-native error-feedback recovery | the ceiling-breaker retry (GPTQ-family, codec-aware) | heavy compute | STUDIO |
| 6 | deep distillation (logit/feature/attn) | richer teacher signal than CE | 2 models + training | STUDIO |

**Key truth that makes "max local" work:** layers 0–3 — calibration, AWQ, mixed-precision,
residual — are the bulk of the recovery and are all **train-free** (forwards + bakes). The 19 GB
wall only bites on *training* (layers 4–6). So the local box can carry the whole train-free
recovery stack, even at 7B (CPU-bf16, slow but real).

---

## LOCAL plan — max here (the train-free recovery stack + the lab + the ship bridge)

### L0. Calibration quality — cheap, multiplies everything (strengthen first)
- **Now:** single wikitext-ish corpus.
- **Strengthen:** curated **multi-domain** calib (code · prose · math · dialogue · structured),
  dedup, size-ablated (how many tokens before recovery saturates), domain-matched variants.
- **Incorporate:** importance sampling (weight chunks by rarity/perplexity); per-deployment calib
  (ship a domain-tuned recovery). Every downstream layer (AWQ stats, KD logits) inherits this.

### L1. AWQ — push past global α=0.5
- **Now:** one global α, one forward.
- **Strengthen:** **α-sweep per model AND per-tensor** (layers want different protection); search
  α on held-out, not assumed.
- **Incorporate:** AWQ on the **residual pass** too (activation-aware residual); SmoothQuant-style
  scale migration to neighbors; clip-search. All forward+bake → local.

### L2. Mixed-precision allocation — the floor-search's real engine
- **Insight:** tensors don't decay equally. Rank each by **output-sensitivity** (Δoutput when it
  alone is quantized), then allocate: sensitive → more bits/residual depth; tolerant → less.
  The baker already takes `--mp-config` / `--rung-config`.
- **Build:** a sensitivity scan (output-space Δ per tensor; Hessian-diag or activation-energy as
  cheap proxies) → greedy bit allocation under a bpw budget → bake. This **minimizes average
  effective bpw at a quality target** = directly lowers the per-model floor.
- All measurement is forwards + bakes → **local**, even at 7B.

### L3. Residual — generalize the breakthrough
- **Now:** single residual b1+b2 (≈1:1 at 0.5B 3+2).
- **Strengthen:** bit-allocation search (which split minimizes degr at a total budget);
  **per-tensor residual depth** (tie to L2 sensitivity — sensitive tensors get a deeper residual);
  stack **AWQ × residual** (residual on the AWQ base).
- **Incorporate:** **iterated residual** (b1+b2+b3 → toward exact, diminishing); residual with its
  own AWQ. Train-free → local.
- **★ THE SHIP BRIDGE (highest-value local engineering): the residual `.tq` serve path.** Today
  residual *proves* quality but can't *serve* (single-bake only). Build the additive two-part GPU
  decode — base bitslice + residual bitslice summed on-the-fly (the GPU bitslice kernel already
  exists for single-bake; extend it to sum two passes). **This is what turns the residual quality
  win into a shippable artifact.** Pure engineering on existing kernels → local.

### L4(local half). The small-model LoRA-KD LAB
- Training fits locally ≤1.5B. Use 0.5B/1.5B as the **fast lab** to develop the best LoRA-KD
  last-mile recipe (rank · lr · KD top-k · calib · what residual leaves for it to fix). The tuned
  recipe transfers to the Studio for 7B+. Local iteration speed is the asset here.

### Cross-cutting local: the recovery LEDGER + multi-eval
- **Recovery ledger:** per model/tier, record how much *each layer* recovers (AWQ −X%, mixed-prec
  −Y%, residual −Z%, LoRA −W%). Tells us which layer to push next instead of guessing.
- **Multi-eval:** ppl is a proxy — add small downstream tasks so "recovered" means *capability*
  preserved, before any "near-1:1" claim. Local.

---

## STUDIO plan — reserved (strictly what needs the RAM/compute)

### S1. Training at scale (layers 4–6 for 7B+)
- 7B+ LoRA-KD and full distillation need 2 models + optimizer states in RAM → Studio (bf16:
  7B≈14 GB, teacher+student+states fits 96; 32B≈64 GB fits; **f32 would not** — bf16 throughout).

### S2. Codec-native error-feedback recovery (the ceiling-breaker, retried at scale)
- Uniform-proxy STE-QAT is a *measured* dead-end (catastrophic on the trellis — don't retry).
  The Studio retry is **codec-aware**: sequential per-column STRAND with **GPTQ-style Hessian
  error compensation** (compensate the quantization error of each column into the not-yet-quantized
  ones, within the codec — no STE through the trellis). Compute-heavy; test where the redundancy
  hypothesis says it may finally pay (big models).

### S3. The full ladder → the bit-floor-vs-scale curve
- Run the floor-search across the multi-family ladder to 405B/671B. Output: the **bit-floor
  descends with scale** curve, the sub-1-bit frontier (see studio_era_expansion.md), and the
  1-bit-at-scale viability test (the 405B unlock). Needs the RAM to hold the big bases.

### S4. Big-teacher distillation
- A larger model teaching the condensed one (true KD, not self-distill) — needs both resident.

---

## Discipline / invariants (stay honest as the stack grows)
- **Effective bpw, always** (baker AGGREGATE; residual sums passes). The floor turns on the real number.
- **Output-space + multi-eval**, never weight-space RMSE alone. Recovery = capability, not ppl theater.
- **Two streams, never conflated:** A = quality (residual/train-free proves it), B = serve-tps
  (gated on the `.tq` serve path). A quality win isn't shippable until B exists.
- **Recovery ledger** keeps it empirical — push the layer with the most remaining headroom.
- **Judge low-bit on big models**, never the 0.5B (it floors ~3-bit and lies pessimistically).
- Disk discipline; GPU jobs sequential; bf16 everywhere at scale.

---

## Sequencing — what to strengthen first, locally (highest leverage → lowest)
1. **Residual `.tq` serve path** (L3 bridge) — converts the existing quality win into shippable tps.
2. **Mixed-precision sensitivity allocation** (L2) — the biggest *new* density lever; lowers the floor.
3. **Calibration upgrade** (L0) — cheap, multiplies AWQ + residual + KD.
4. **Per-tensor residual depth + AWQ×residual stack** (L3/L1) — generalize the breakthrough down the ladder.
5. **Small-model LoRA-KD lab** (L4) — develop the last-mile recipe for Studio transfer.
6. **Recovery ledger + multi-eval** (cross-cutting) — make every step above measurable.

> Net: the local box can carry calibration + AWQ + mixed-precision + residual + the serve bridge +
> the LoRA lab — i.e. the entire recovery stack short of *scale-training*. The Studio's only unique
> jobs are training-at-7B+, the codec-native ceiling-breaker, and the full bit-floor-vs-scale curve.
> Maximize the doctor here; let the Studio extend it — not gate it.
