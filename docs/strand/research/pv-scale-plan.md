# PV-at-scale plan — the recipe + the solo-trainable 7B path (REMEDY #3)

_Machine-stamped: Apple M3 Pro, 18GB unified, macOS Darwin 25.5.0. Written 2026-06-11 ~23:30 EDT.
All 7B param counts MEASURED from Qwen2.5-7B's public config (hidden 3584, intermediate 18944,
28 layers, 28 Q / 4 KV heads, head_dim 128). All memory/cost are ARITHMETIC from those counts +
public GPU rental rates — labeled estimate where not measured on-box._

The pitch this decides: **can we trailblaze a deterministic, solo-trained 2-bit 7B that lands
near AQLM's +35% (vs our current PTQ +666% on llama2-7b) — and is it a $30 experiment or a $300
one.** Answer below: it is a **$3–$13 experiment on one A100-80** (GO), and possibly a $20 one on
a 24GB 3090 if we accept the deepest-quarter down_proj scope. The expensive part was never the
GPU — it is the **engineering risk** that selective-PV's frozen-tensor assumption holds.

---

## Part 1 — the deep 0.5B PV run (the recipe 7B inherits)

**Status: SPEC'D + scripted; the live run is BLOCKED on the box.** At write time
`pgrep -f quantize-model` AND `pgrep -f eval-ppl` are BOTH live (pid 11911 wrapper @17min, pid
12375 the Omega* attn4/ffn3 CPU eval @7min). will.md §7 freeze trap forbids any MPS training
while those run (two hard reboots came from exactly this co-tenancy). Per the remedy's own
30-minute rule, the run is spec'd and handed off as `scripts/pv-recipe.sh` rather than forced.

### The recipe (what `pv-recipe.sh` runs)
From base `scratch/qwen-05b`, a strand-PV 2-bit arm, 300 steps, segmented 4×75 (= requant
cadence), every landed lever stacked:

| lever | value | why |
|---|---|---|
| `--quant strand` | real Rust encoder in the loop | train-through-what-you-ship; proxy-transfer DEAD (will.md §4) |
| `--bits 2 --l 12` | 2-bit, 4096 states | the operating point (state count saturates at l=12) |
| `--outlier-channel 1` | pre-RHT top-\|w\| side-channel | the only live PTQ lever (+1%) |
| `--kd` | KL from frozen bf16 teacher | chunked-KD proven through 4 requant segments |
| `--warmup-frac 0.05` | Apple WSD warmup | intel scorecard 2026-06-11 |
| **`--cooldown-frac 0.2`** | **Apple WSD linear decay-to-zero** | **THE NEW VARIABLE — the only schedule change vs the 26.77 run** |
| `--grad-checkpoint` | on | 18GB freeze hardening |
| watermark caps | 0.92 / 0.7 | PROVEN; see deviation note |

**Goal:** beat the prior PV floor of **26.77** (0.5B, 2-bit, plain-cosine, will.md 2026-06-11).
Anchors: bf16 = 12.55, PTQ-only floor = 80.7. Because every lever except the WSD cooldown is held
identical to the 26.77 arm, the final PPL is a **clean A/B isolating the Apple cooldown**. The
Apple finding is that for low-bit QAT a short final cooldown buys more than extra steps — if it is
real here, expect ~26.77 → low-26 / high-25; if the cosine tail already captured it, expect a
wash (the honest null is informative too).

### Deviation from the remedy (flagged, with the reason)
The remedy asked for watermark caps **1.0 / 0.85**. I pinned **0.92 / 0.7** instead, the
will.md-PROVEN values. `1.0` lets MPS map the entire unified pool with no headroom — that is the
exact configuration that froze the box twice on 2026-06-09 (the OS dies before python can OOM).
`0.92` ≈ 12.3GB cap is the documented safe ceiling for the ~10GB ternary+KD driver pool. The
recipe exposes `WATERMARK_HI/LO` env overrides if the owner explicitly wants the looser caps, but
the default protects the machine. (`0.8` was too tight — OOM'd at step 2; 0.92 is the sweet spot.)

### To launch when the box frees
```
until ! pgrep -f quantize-model && ! pgrep -f eval-ppl; do sleep 120; done
nohup ./scripts/pv-recipe.sh >> research/pv-deep/launch.log 2>&1 & disown
```
The script self-guards (refuses to start if either job is live) and segments automatically. Output
trajectory + final HF dir land in `research/pv-deep/`. Feed `pv-deep-hf` to `strand-7b-ppl.sh` for
the canon-protocol PPL if a second confirmation is wanted.

---

## Part 2 — 7B-PV-via-selectivity feasibility (the trailblazer analysis)

Selective-PV (`--pv-tensors` regex in `strand-qat.py`, already wired) trains ONLY a chosen subset
of QuantLinears; the rest freeze at their requant recon and contribute a pure forward (the recon,
no grad). The question: **does the trainable subset + frozen-bf16 base + activations fit a rentable
GPU, and at what $.**

### The memory math (MEASURED param counts, arithmetic footprint)

Qwen2.5-7B has **6.525B** params across the 168 projection tensors. Per-tensor sizes (the lever):

| tensor class | count | params each | share of proj |
|---|---|---|---|
| q_proj / o_proj | 28 each | 12.85M | 12.3% combined |
| k_proj / v_proj | 28 each | 1.84M | small |
| gate_proj / up_proj | 28 each | 67.90M | 58.3% combined |
| **down_proj** | **28** | **67.90M** | **29.1%** |

**down_proj-only = 1.901B trainable params (29.1% of proj, 28 of 168 tensors).**

Footprint model (the QAT harness keeps an **fp32 shadow** per trainable QuantLinear; AdamW m,v fp32;
grad fp32; plus the bf16 delta-base for the delta forward → **18 B/trainable param**. Frozen rest of
the ~7.6B model: bf16 = 2 B/param. Activations: grad-checkpoint, batch 1, ctx 1024 ≈ 2GB rough):

| scope | trainable | train-state | frozen base | act | **total** | 3090 24GB | A100-40 | A100-80 |
|---|---|---|---|---|---|---|---|---|
| down_proj-only (28) | 1.901B | 34.2GB | 11.4GB | 2.0 | **47.6GB** | NO | NO | **FITS** |
| down_proj deepest-half (14) | 0.951B | 17.1GB | 13.3GB | 2.0 | **32.4GB** | NO | **FITS** | FITS |
| down_proj deepest-quarter (7) | 0.475B | 8.6GB | 14.2GB | 1.5 | **24.3GB** | ~edge | **FITS** | FITS |
| FFN-only (84) | 5.703B | — | — | — | 106.5GB | NO | NO | NO |
| full PV (168) | 6.525B | — | — | — | 119.6GB | NO | NO | NO |

**8-bit-AdamW variant** (bitsandbytes int8 optimizer → m,v 2 B/param instead of 8; trainable drops
to 12 B/param). This is the lever that pulls scopes onto smaller cards:

| scope | trainable | **total (8bit-Adam)** | 3090 24GB | A100-40 | A100-80 |
|---|---|---|---|---|---|
| down_proj-only (28) | 1.901B | **36.2GB** | NO | **FITS** | FITS |
| down_proj deepest-half (14) | 0.951B | **26.2GB** | NO | **FITS** | FITS |
| **down_proj deepest-quarter (7)** | 0.475B | **21.5GB** | **FITS** | FITS | FITS |

### The concentration question (CITED open question)
Whether training down_proj-only (or a deeper-layer subset) recovers most of the loss is the
**rung-allocator's open question** — `research/mixed-rung-routing.md` records that under STRAND's
RHT, **rel-RMS sensitivity is flat across tensor classes (0.019pp spread) because the RHT whitens
every tensor to the same i.i.d. Gaussian**, so sensitivity must be measured in OUTPUT (PPL) space,
and that per-tensor PPL sweep is "queued for the pod" (unmeasured). The down_proj-as-the-lever
prior comes from the 3-bit flagship: `mp_light` (down_proj@4, rest@3) is the entire 3-bit quality
lever (9.42→8.45; "attn@4 adds nothing", will.md §3-4). **So the 7B-PV bet rides on an unproven
transfer: that the down_proj concentration which holds for 3-bit mixed-precision ALSO holds for
2-bit PV re-learning.** That transfer is itself the cheap thing to test first on the 0.5B (Part 3).

### GO / NO-GO + $/arm

**GO.** A solo-trained deterministic 2-bit 7B is a **single-GPU, sub-$15 experiment**, not a
multi-GPU one. The decisive facts:
- **down_proj-only fits one A100-80** (47.6GB fp32-Adam, or 36.2GB with 8-bit Adam → also A100-40).
- **down_proj deepest-quarter fits a $0.46/hr 3090** (21.5GB with 8-bit Adam) — IF the
  concentration holds that 7 deep down_projs carry the recovery.
- Full PV (119.6GB) needs 2× A100-80 / an H100-80 with offload — that is the $300 path and we do
  **not** need it for the trailblazer claim.

Cost per 300-step arm (segmented; 0.5B is ~4s/step, 7B fwd+bwd ~15× but selective-PV backprops only
the subset → cheaper bwd; estimate 25–40 s/step → ~2h for 300 steps, ~4h with the 4-segment requant
overhead at ~15min/requant×4 on the rented CPU):

| GPU | down-only 300step (~2h) | +requant cadence (~4h) | full 7B-PV (~8h) |
|---|---|---|---|
| 3090 $0.46/hr | $0.92 | $1.84 | n/a (won't fit) |
| A100-40 $1.1/hr | $2.20 | $4.40 | n/a |
| **A100-80 $1.5/hr** | **$3.00** | **$6.00** | $12.00 |

**Recommended path: A100-80, down_proj-only, ~$6/arm.** It fits with margin (47.6GB of 80GB → room
for the in-loop requant's CPU recon to run alongside), needs no 8-bit-Adam risk, and trains the
class the 3-bit data says is the lever. Two or three arms (cooldown on/off, lr sweep) = **~$15–20
total**. The 8-bit-Adam + 3090 path is the **$2–4 budget variant** but stakes the result on both
the concentration prior AND int8-optimizer stability at 2-bit re-learning — keep it as the cost
floor, not the first arm.

### What this lines up against
Our llama2-7b 2-bit PTQ is **+666% (42.41 vs bf16 5.535)** — squarely in PV territory (the crossover
is ~2× bf16; will.md). AQLM's trained llama2-7b 2-bit ≈ **6.9 PPL (+35%)**. The 0.5B PV law
(78.95→26.77, −66%, the first segment doing −61%) says one segment does most of the work — so the
7B-PV arm against AQLM's anchor is a **few-segment, ~$6, single-A100-80 experiment**, contingent
only on the down_proj concentration transferring from 3-bit-MP to 2-bit-PV.

---

## Part 3 — the cheap gate before paying for 7B (recommendation)

Before any 7B rental, run ONE 0.5B selective-PV A/B (free, on-box, when it frees):
`--pv-tensors 'down_proj'` vs the full-PV 26.77 arm, same 300 steps. If down_proj-only on the 0.5B
recovers ≥ ~90% of the full-PV gain, the concentration prior is confirmed and the 7B down_proj-only
arm is de-risked to a $6 buy. If it does NOT, the 7B path needs FFN-only (106GB → 2×A100-80, the
$300 tier) and the GO downgrades to "expensive — justify against AQLM margin first." This gate is
the cheap-first ladder (will.md §5.7) applied to the rental decision.
