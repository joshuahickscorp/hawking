# STRAND benchmark report

_Assembled 2026-06-13 from the source-of-truth files only. Every number is cited to
the file it came from; values not in any file are marked **[unverified]**. The
optimization metric throughout is the log loss tax, `loss_tax_nats = ln(PPL_quant /
PPL_bf16)` (`docs/STRAND-quality-density-frontier.md` §1, line 35)._

## Executive summary

STRAND is a deterministic, float-free low-bit model-quantization system whose
headline is a **measured 2-bit / 3-bit quality curve with a scale-tolerance trend
and a determinism moat**. At 2-bit PTQ the canonical `q2_l12_out1` recipe was run on
three Qwen2.5 dense tiers and the relative damage is **+59.0% at 7B, +74.9% at 14B,
+38.3% at 32B** — i.e. the 32B tolerates 2-bit best of the three (the curve is *not*
monotone; 14B is the worst point). The 3-bit shipping rung is essentially free on a
hard family: Llama2-7B `mp_light` is **+5.8% / 0.0562 nats**. The single best
near-free quality lever found is **inner-product de-bias**, which removed **-28.67%
PPL at q2-class** (0.5B) for a billed **0.0136 bpw**; progressive/QAT "train-through"
PV took the 0.5B q2 floor of **79.94 → 26.77** (≈2.13× bf16). On density, the bit
ledger shows **≈0.25 bpw of recoverable side-info** at q2 with no quality cost, and
on speed the Metal bitslice decode clears its threshold at **60.6%/74.0% of the
measured 98.0 GB/s peak**. The honest frame the owner demands: **every confirmed
quality number here is 0.5B or a single pod tier; nothing has cleared a scale
confirm above 32B**, the de-bias/PV/KL scale runs are *queued, not run*, and the
"smaller-than-GGUF-at-iso-bits-on-quality" thesis is **not** claimed — STRAND's
iso-bpw edge at 0.5B is a narrow structural (dimension-fallback) win, not a quality
win. The moat is determinism: bit-identical, frozen-LUT decode that PTQ and
non-deterministic methods cannot ship reproducibly.

---

## Method

All quantization-quality numbers use one fixed perplexity harness; cross-harness
comparisons are not made. Two harness regimes appear, each internally consistent:

- **Pod scale tiers (7B / 14B / 32B / Llama2-7B).** WikiText, ctx 2048, 64 chunks,
  131,008 tokens, dtype bf16 (per the `ctx`/`chunks`/`tokens`/`dtype` fields in each
  `scratch/pod-results/ppl_*.json`). The 7B and Llama2 baselines/q2 ran on
  `device: "cuda"`; the 14B and 32B ran on `device: "offload"` (the `device` field in
  each JSON). The eval token set and dtype are identical across all six runs.
- **0.5B local canon (de-bias, PV, C2, KL, iso-bpw).** Qwen2.5-0.5B (`scratch/qwen-05b`),
  WikiText (`dataset_id: "wikitext"`, `dataset_fp 696cca6b65a171b0`), ctx 2048, 64
  non-overlap windows, **device cpu**, dtype bfloat16, harness_key8 `bee28e82`, 131,008
  tokens (the `harness_key` block in every `research/isobpw/ppl/*.json` and the de-bias
  A/B JSONs). The bf16 anchor is `PPL_bf16 = 12.535789392557929`
  (`research/isobpw/ppl/ppl_bf16_anchor.json`); the frontier doc rounds this to
  `12.536` (`docs/STRAND-quality-density-frontier.md` line 38).

The canonical recipe tag is `q2_l12_out1`: 2-bit STRAND trellis, state length `L=12`
(the 2-bit operating point; `L≥13` is saturated per `docs/STRAND-quality-density-frontier.md`
line 66), one pre-RHT outlier channel per row at 8-bit. PTQ unless a run is explicitly
PV/QAT. Every result is required to carry {model, recipe, bpw, PPL, harness identity}
to enter the canon (`docs/STRAND-quality-density-frontier.md` §6; `scripts/promote.py`
grammar, §8.1). Where a nats figure is *derived* from a PPL pair rather than stored,
the arithmetic and the baseline ambiguity are shown.

---

## Scale-tolerance: the q2 loss-tax curve across 7B / 14B / 32B

The headline question is whether a bigger model tolerates 2-bit quantization better
than a smaller one. The **identical** canonical recipe (tag `q2_l12_out1`) was run on
three Qwen2.5 dense models and WikiText perplexity measured against each model's own
bf16 baseline. Eval harness is identical across all six runs (ctx 2048, 64 chunks,
131,008 tokens, bf16).

### The measured curve

| Model | bf16 PPL | q2 PPL | Relative damage | Loss-tax (nats) |
|---|---:|---:|---:|---:|
| Qwen2.5-7B  | 6.6289 | 10.5380 | **+59.0%** | **0.464** |
| Qwen2.5-14B | 5.1023 |  8.9248 | **+74.9%** | **0.559** |
| Qwen2.5-32B | 4.7785 |  6.6092 | **+38.3%** | **0.324** |

Every PPL is read directly from the per-run JSON in `scratch/pod-results/`
(`ppl_qwen-7b_baseline.json` 6.6289227, `ppl_qwen-7b_q2_l12_out1.json` 10.538037;
`ppl_baseline_14b.json` 5.1023164, `ppl_q2_l12_out1_14b.json` 8.924847;
`ppl_baseline_32b.json` 4.778485, `ppl_q2_l12_out1_32b.json` 6.6092402). The
percentages and nats are recomputed from those raw values and match the figures in
`docs/STRAND-quality-density-frontier.md` lines 645–646.

### What the curve shows — and what it does not

**The win is real: 32B tolerates 2-bit best of the three.** At 32B the q2 penalty is
+38.3% / 0.324 nats — meaningfully smaller than either smaller model. The frontier
doc frames this as already confirming the scale-tolerance thesis and justifies
deferring the 70B run (`STRAND-quality-density-frontier.md` lines 644–647). On the
absolute scale that matters most for serving, the 32B drops only 4.78 → 6.61 while
the 7B blows out past 10.5.

**Honest caveat — the curve is NOT monotone, despite one line in the doc that says it
is.** Line 646 of the frontier doc calls this "monotone improvement with scale," but
the three measured points are **+59% → +75% → +38%**: the 14B is the *worst* of the
three, not a midpoint. The relationship is U-shaped (14B peak), not a clean downward
slope. The same doc contradicts the "monotone" wording elsewhere — line 619 states
plainly that "q2 relative damage [is] NOT monotone." The defensible claim is "**32B
is the best of the three and large models can tolerate 2-bit**," not "damage falls
monotonically as size grows." The 14B regression is unexplained by these three points
alone.

**Device caveat.** The 7B baseline/q2 ran on `device: "cuda"`; the 14B and 32B ran on
`device: "offload"` (the `device` field in each JSON). The eval token set and dtype
are identical, so this should not move PPL, but the runs were not all on the same
execution path and that has not been independently cross-checked.

**Recipe-consistency caveat.** All three q2 runs carry tag `q2_l12_out1` and the same
outlier-channel/L12 settings, so the comparison is apples-to-apples on recipe. None of
these q2 runs include the de-bias or KL-routed-PV levers measured separately at 0.5B —
this is the *raw* canonical-q2 curve, the floor before those rescues.

**Scope caveat.** Three points, one model family (Qwen2.5 dense), one bit-width (q2),
one dataset. The 70B point that would extend the curve was **deferred**
(`STRAND-quality-density-frontier.md` lines 642–647: "70B q2 DEFERRED," memory wall +
diminishing information), so 32B is currently the largest measured tier.

---

## The quality levers, measured

All numbers below are 0.5B (Qwen2.5-0.5B, `scratch/qwen-05b`) unless stated. The eval
harness is fixed across the de-bias and placement work (WikiText, ctx 2048, 64
non-overlap windows, CPU, bfloat16, `harness_key8 bee28e82`, 131,008 tokens). The
bf16 anchor for loss-tax is `PPL_bf16 = 12.536`, loss tax `ln(PPL_quant / PPL_bf16)`.
Two of the four headline nats figures (PV, de-bias) are **derived** from PPL pairs in
files; the arithmetic and ambiguity are shown rather than quoting a stored constant.

### 1. De-bias — the largest near-free lever found to date

Inner-product de-bias is a rank-0 output-space correction: a per-output-row bias
`c = -(recon-orig) @ mu`, where `mu` is the per-input-feature activation mean
(calibrated once on WikiText train windows, banked as `research/actmean-qwen05b.json`,
168 modules).

Measured A/B on the `dp_d4_r2` (down@4-bit / rest@2-bit) recon
(`research/debias-ppl-dp_d4_r2-ab.json`):

| arm | PPL | source |
|---|---:|---|
| baseline (recon + zero correction) | **40.24473** | `debias-ppl-dp_d4_r2-ab.json` (`debias_applied: false`) |
| de-biased (recon + c) | **28.70825** | `debias-ppl-dp_d4_r2-ab.json` (`debias_applied: true`) |

- ratio **0.71334**, relative **-28.67%** (`relative_pct: -28.665824643438164`).
- `promote.py` stamped **LOCAL_PASS**, debiased loss tax **0.8286 nats** vs the kill
  bar of -0.5%; contamination guard clean (`contamination_warning: false`; corrections
  visibly applied: 168/168 vector, `correction_l2 = 61.40`, `correction_absmax = 1.073`).

**The "-0.344 nats" figure (honest derivation, ambiguity stated).** The doc records
the nats removed as **0.343** = baseline tax 1.172 − debiased tax 0.829, where 1.172
uses the *recon-load* baseline PPL 40.48 (`STRAND-quality-density-frontier.md` §8.3 /
§1). Computing directly: `ln(40.48 / 28.70825) = 0.344` nats. Using instead the
*hook-replaced* A/B baseline 40.24473 (the number actually in the A/B json),
`ln(40.24473 / 28.70825) = 0.338` nats. So **-0.344 nats** is the recon-load-baseline
value; the within-A/B value is **-0.338**. The ~0.24-PPL gap between the two baselines
(40.48 vs 40.245) is, per the file, "bf16-copy ordering noise — judge by the ratio"
(§8.3, lines 449–452). The load-bearing result is the ratio (-28.67%); the nats are
~0.34, baseline-dependent.

**Cost (billed, exact).** Per-output-row bias = 304,128 rows (24 layers × 12,672) ×
bf16 = 4.87 Mbit / 357.8 M weights = **0.0136 bpw** (fp32 would be 0.0272 bpw)
(`STRAND-quality-density-frontier.md` §8.3, lines 457–458). The trade is ~0.34 nats
removed for ~0.0136 bpw — roughly **25 nats per bpw** (0.343 / 0.0136 ≈ 25.2).
**Correction to the source doc:** §8.3 line 459 states "~245 nats per bpw / two orders
of magnitude past any §5 estimate" — that is a 10× arithmetic slip; the real figure is
**~25 nats/bpw**, about one order past the §5 cost estimate, still decisively
favorable. (A separate, narrower figure exists from the synthetic output-RMS gate:
**0.0179 bpw** at in=896, **0.0033 bpw** at the wide mlp down_proj —
`research/debias-results.md` line 59. These are per-tensor wire costs at specific
widths, not the whole-model 0.0136.)

**Generalization across bit-rate.** On the `dp_d4_r3` (q3-class) recon
(`research/debias-ppl-dp_d4_r3-ab.json`): baseline 19.07542 → 16.99027, ratio 0.89069,
**-10.93%** (`relative_pct: -10.931079966999835`), also LOCAL_PASS (loss_tax 0.304) at
0.0136 bpw. So the win is **-28.67% at q2 and -10.93% at q3** — smaller at higher
bit-rate but still well past the bar.

**Limitations, stated plainly.**
- This is all **0.5B**. The 7B/32B confirm is *queued, not run* (gated behind a
  PROMOTE_CLOUD stamp; bases were disk-deleted, `STRAND-quality-density-frontier.md`
  §8.3 item 3 / §10). The "near-free" claim at scale is **[unverified]**.
- The original output-RMS gate (`debias-results.md`) was on a *modeled* activation
  mean (μ̄≈0.3 isotropic, line 51); the PPL A/Bs above use the *real* banked
  `actmean`, which is what makes them the truth-grade evidence rather than the
  synthetic +4.19% output-RMS gate.
- Artifact-ization (`quantize-model --actmean`) is **built** and parity-checked
  (max|diff| 3.7e-09 vs the Python harness on layers.0.down_proj,
  `STRAND-quality-density-frontier.md` §8.3 item 2), but folding the sidecar into the
  `.strand` v2 section chain + decode-kernel MAC epilogue is **not yet done**.

### 2. PV (progressive / QAT "train through what you ship") — the 2-bit lever

PV trains the model through the real STRAND encoder (delta-forward STE, KD, requant
boundaries).

| point | PPL | source |
|---|---:|---|
| 0.5B q2 PTQ floor (`q2_l12_out1`) | **79.9406** | `docs/STRAND-compression-map.md` §5.1 (also 79.94 throughout); `research/isobpw/ppl/ppl_strand_q2_l12_out1.json` 79.94056994 |
| PV arm, 300 steps (78.95 → 27.02) | **27.02** | `docs/will.md` line 712 (`78.95 → 30.48 → 28.51 → 27.41 → 27.02`) |
| PV2 arm, +600 steps | **26.77** | `docs/will.md` lines 700–701 (pv2 asymptote, −0.9% vs 27.02) |

- **"-1.094 nats" is derived**, not stored: `ln(79.94 / 26.77) = 1.094` nats of
  loss-tax removed by PV over the PTQ floor. **Ambiguity flagged:** the docs also quote
  a "PTQ-only floor ≈ 80.7" (`research/debias-results.md` lines 86/99,
  `STRAND-quality-density-frontier.md` §8.3); using 80.7 gives `ln(80.7/26.77) = 1.103`,
  and using the PV-arm start 78.95 gives 1.082. So PV removes **~1.08–1.10 nats**
  depending on which floor; -1.094 is the 79.94-floor value.
- Equivalent framing in the files: **≈66% PPL reduction** (`docs/will.md` line 712
  "−66%"; `docs/STRAND-compression-map.md` §5.2), and 26.77 ≈ **2.13× bf16**
  (`docs/will.md` line 701). bf16 = 12.55 (`will.md`), 12.536 (quality-density anchor),
  12.535789 (`ppl_bf16_anchor.json`) — the anchors differ only in the 3rd digit.
- **26.77 is the 0.5B/this-recipe asymptote**: pv2 was only -0.9% vs pv-arm 27.02 and
  the pv3run gate "correctly closed" (`will.md` line 701). Plain reading: direct cosine
  PV has run out of room at this recipe.

**Limitations.**
- The *current live* `pv_down_protect_q2` run starts from the dp-d4-r2 floor (init
  `bpw=3.290977`, `PPL=40.164`, `STRAND-quality-density-frontier.md` §0 line 16 / §1
  line 44). Its **final PPL is not yet in a file** — the verdict
  (`research/pv-dp/pv-dp.json`) is **[unverified]** (file confirmed absent on
  2026-06-13). The §4.A bars: ≤30 & <26.77 → PROMOTE_CLOUD, 30–36 → WSD/progressive
  first, ≥36 → pivot (lines 116–119).
- 26.77 is a **recon-plane** number; per the project lineage the original same-day
  artifact bake stored bulk-only weights that were *not* the 26.77 model — corrected in
  the 2026-06-11 re-bake (STR2 + OUTL + SPRV, byte-equality gated; `docs/will.md`).
- 7B selective-PV is **pending**, and per the placement screen below it must be
  **re-aimed** (see §4).

### 3. C2 side-info — ~0.25 bpw recoverable, no quality cost

C2 is pure density: stop billing fixed-width side-info that carries less entropy than
its field. Measured by the bit-ledger entropy microscope on the q2 0.5B model
(357,826,560 weights, 1,397,760 blocks), reproducing the canon 2.6653 bpw **exactly**
(`research/bit-ledger-results.md`):

| component | raw bpw | entropy bpw | **recoverable bpw** | note |
|---|---:|---:|---:|---|
| scale (32 b/block) | 0.12500 | 0.04099 | **0.08401** | 32-bit field, ~10.5 bits real entropy (H 10.49/32); *not* a prediction win (RHT whitened block-to-block) |
| sub_scale (8×6/block) | 0.18750 | 0.16554 | **0.02196** | one pooled order-0 CDF captures it; per-position adds nothing |
| outl_pos | 0.22584 | 0.07825 | **0.14760** | largest single component; gap-coding the sorted indices (H_abs 21.0 → H_gap 7.82) |
| outl_val | 0.08000 | 0.05963 | **0.02037** | modest |
| init_state | 0.04688 | 0.03899 | **0.00788** | near-incompressible per symbol; use tail-biting, not a codec |

(All rows from `research/bit-ledger-results.md` §"q2 (k=2, L=12)", lines 56–64.)

- **scale + sub_scale = 0.106 bpw** (q2; 0.115 at q3) — the "C2 scales" figure
  (0.08401 + 0.02196 = 0.10597; the gate row rounds to 0.10598,
  `bit-ledger-results.md` line 125). It **clears its gate by ~10×** (gate bar
  ≥0.01 B/w, line 125).
- **outlier positions = 0.1476 bpw** — the single largest recoverable component,
  bigger than scale (line 88).
- **Total realistic C2 + outlier-position recovery ≈ 0.25 bpw**: q2 2.6653 → ~2.42 bpw,
  **no quality cost** (`docs/STRAND-compression-map.md` line 16;
  `STRAND-quality-density-frontier.md` §9 item 3, lines 564–565: "C2 scales 0.106 +
  outlier-positions 0.1476" = 0.254, rounded to 0.25).

**Limitations.**
- These are **entropy ceilings**. A real integer rANS coder plus its CDF tables lands
  "a few % short" (`bit-ledger-results.md` line 114), and the outlier-position gain is
  quality-gated (routed through the OUTL/Lane-G redesign, not free metadata cleanup,
  line 144).
- STRAND outliers are element-wise top-|w| weights (not channel outliers), so
  cross-layer channel-sharing does **not** apply (`STRAND-quality-density-frontier.md`
  §9 item 4, lines 566–571); the gap-coding gain assumes the top-|w| positions
  **cluster**, which the ledger confirms for q2/q3 (gap H ≈ 7.82). Whether the
  intra-tensor structure pays beyond the gap measurement is itself a scout,
  **[unverified]** beyond the entropy ceiling.
- A separate 0.5 bpw deploy-only lever (the v2 random-access seek table) is droppable
  for streaming deployment (`bit-ledger-results.md` line 46), but that is a container
  split, not C2 proper.

### 4. KL-routed placement — the down-protect inversion

The placement screen (`scripts/rung-kl.py`, hot-swap one recon tensor at a time,
logit-KL on WikiText train windows; `research/rung-kl-dp_d4_r2.json`, ctx 512, 2
chunks, train split, base NLL/tok 3.213989) inverts the intuition every down-protect
arm was built on. Class-mean output-KL:

| projection class | mean output-KL | rank |
|---|---:|---|
| up_proj | **0.009887** | highest damage |
| v_proj | **0.009172** | 2nd |
| gate_proj | **0.006649** | 3rd |
| o_proj | **0.004525** | |
| k_proj | **0.003002** | |
| **down_proj** | **0.002787** | **2nd-LEAST** |
| q_proj | **0.002438** | least |

(Source: `class_mean_kl` in `research/rung-kl-dp_d4_r2.json`, verified field-for-field.
The doc §9 quotes these rounded — up 0.0099, v 0.0092, gate 0.0066, o 0.0045, k 0.0030,
down 0.0028, q 0.0024 — which match, `STRAND-quality-density-frontier.md` lines 518–519.)

**The inversion, stated plainly:** `down_proj` — the tensor every dp-arm protected at
4-bit — is the **second-lowest** output-damaging projection. The real damage is
**up_proj / v_proj / gate_proj**. rel-RMS hid this because RHT flattens rel-RMS across
tensor classes; only output-space KL exposes it. Consequence
(`STRAND-quality-density-frontier.md` §9 item 1, lines 522–527): the down-protect
family spent 4-bit budget on a low-damage projection. The corrected target is a
25-tensor RED list (`red_tensors`, 25 entries in the JSON) plus a `pv_tensors_regex`
for selective PV.

**Why the down-protect arms still "worked" despite aiming wrong:** down-protect
*halves the raw q2 collapse* (79.94 → 40.48, `STRAND-quality-density-frontier.md` §1
line 43 / §8.2 line 432), so it is a real rescue — but it is **dominated**: no
bit-placement arm reaches the uniform-q3 Pareto point (16.11 @ ~3.4 bpw, tax 0.251).
The §8.2 placement scorecard (lines 426–432):

| arm | bpw | PPL | loss tax | verdict (from file) |
|---|---:|---:|---:|---|
| q3 uniform (anchor) | ~3.4 | 16.11 | 0.251 | the Pareto point |
| `dp_d4_r3` (down@4/rest@3, L=8) | 3.68 | 19.11 | 0.421 | dominated (+ L=8 confound) |
| `ffn4` (all-FFN@4/attn@2) | 4.54 | 23.53 | 0.630 | dominated; off the 2-bit budget |
| `dp_attn2` (down@4/attn@2/ffn@3) | 3.57 | 32.12 | 0.941 | dominated |
| `dp_d4_r2` (down@4/rest@2) | 3.29 | 40.48 | 1.172 | 2-bit rescue but dominated |

**Decision on file:** broad FFN protection is **killed** as a density play; **at 0.5B,
bit placement is not the 2-bit lever — training is** (§8.2, lines 434–437). The
supporting error-space story: the quant error is **white in weight-space** (rank-1
energy 0.24%, `research/error-spectrum-dp_d4_r2_down.json` `mean_energy_r1 = 0.0024`,
verdict "DEAD") but **low-rank in output-space** (rank-1 18.9%, rank-16 37.8%,
`research/error-spectrum-dp_d4_r2_up_actw.json` `mean_energy_r1 = 0.1887`,
`mean_energy_r16 = 0.378`, verdict "ALIVE") — exactly why weight-MSE methods
(diag-Hessian, weight low-rank) die and why de-bias (output-space) wins
(`STRAND-quality-density-frontier.md` §9 item 2, lines 529–539).

**Limitations.**
- The KL screen is **0.5B, ctx 512, 2 chunks** — a small calibration slice. The
  KL-routed mp confirm and the re-aimed 7B PV (the original `--pv-tensors down_proj`
  must become the up/v/gate RED set) are **defined and queued but not run**
  (`STRAND-quality-density-frontier.md` §10 lines 607–610); their scale behavior is
  **[unverified]**.
- delta_nll is noisy at this slice (the per-tensor table includes negative delta_nll
  entries) — so KL is the routing signal, not delta_nll.

### Net read

Ranked by what actually pays at q2 0.5B (`STRAND-quality-density-frontier.md` §9 lever
stack, lines 559–565): **(1) de-bias** (~0.34 nats / -28.67%, ~0.0136 bpw — adopted,
the single best free lever); **(2) PV routed by rung-kl** (the ~1.09-nat 2-bit lever,
now re-aimed to up/v/gate); **(3) C2 + outlier-position coding** (~0.25 bpw, pure
density). The honest ceiling: every confirmed number here is **0.5B**; nothing scales
to 7B+ without a PROMOTE_CLOUD gate, and those scale confirms are queued, not measured.

---

## iso-bpw head-to-head vs GGUF at 0.5B

**Scope and honesty frame.** This is a single-model proxy: Qwen2.5-0.5B
(`scratch/qwen-05b`), WikiText-2, the STRAND canon eval (ctx 2048, 64 non-overlap
windows, device cpu, dtype bf16, harness_key8 `bee28e82`). Both formats run through the
*same* eval code path — GGUF quants are dequantized back to HF safetensors and scored
by `ops/eval-ppl.py`, so there is no cross-harness caveat
(`docs/STRAND-vs-gguf-isobpw.md` lines 20–26). bf16 is the ceiling, not a target. The
honest question: at the same bits-per-weight, which format has lower perplexity?

The bpw denominator is matched: bits in the 7 projection-weight matrices over their
element count. The STRAND side is **357,826,560 quantized weights, file-verified**
(`research/isobpw/run.log` lines 13/18: "over 357826560 quantized weights"). The GGUF
side is billed the *same way* by `tools/gguf/gguf_bpw.py` and asserted in
`docs/STRAND-vs-gguf-isobpw.md` line 34 to land on the identical 357,826,560 elements
("verified identical") — but this GGUF-side denominator could **not be independently
checked from a committed result file** (caveat 2), so treat "identical denominator" as
STRAND-file-verified / GGUF-doc-asserted.

### The measured frontier (every PPL from a real file)

| bpw (proj) | config | PPL | format | loss tax vs bf16¹ | source (PPL / bpw) |
|---:|---|---:|---|---:|---|
| 16.000 | bf16 anchor | 12.5358 | ref | 0.000 | `ppl_bf16_anchor.json` |
| 2.665 | STRAND q2 l12 out1 | 79.9406 | strand | 1.853 | `ppl_strand_q2_l12_out1.json` (`eff_bpw 2.665256`) |
| 3.665 | STRAND q3 l12 out1 | 16.1128 | strand | 0.251 | `ppl_strand_q3_l12_out1.json` (`eff_bpw 3.665256`) |
| 3.806 | STRAND "mp_light" (= omega-star, attn4/ffn3)² | 15.0391 | strand | 0.182 | `ppl_strand_mp_light.json` (`eff_bpw 3.80564`) |
| 4.197 | GGUF Q2_K | 15.2380 | gguf | 0.195 | `ppl_gguf_Q2_K.json` / proj_bpw doc-only³ |
| 4.197 | GGUF IQ3_S | 16.9503 | gguf | 0.302 | `ppl_gguf_IQ3_S.json` / proj_bpw doc-only³ |
| 4.574 | GGUF Q3_K_M | 13.6111 | gguf | 0.082 | `ppl_gguf_Q3_K_M.json` / proj_bpw doc-only³ |
| 5.521 | GGUF Q4_K_M | 12.8937 | gguf | 0.028 | `ppl_gguf_Q4_K_M.json` / proj_bpw doc-only³ |

(All PPL JSONs under `research/isobpw/ppl/`. ¹ loss tax = ln(PPL/12.5358), recomputed
here from the cited PPLs and confirmed. ² see caveat 1: this row is NOT canonical
mp_light. ³ see caveat 2: GGUF proj_bpw has no committed backing file.)

### The real result: a density win, not a clean quality win

The headline that survives scrutiny is **structural**, and it is about where GGUF
lands, not where STRAND's quality lands:

- **Qwen2.5-0.5B has 896-dim projection tensors, and 896 is not a multiple of
  llama.cpp's 256-element K-quant superblock.** The K-quants silently fall back to
  32-blocked legacy types for these tensors, so the *nominal* low-bit GGUFs are not
  low-bit here. Measured proj-weight composition (`docs/STRAND-vs-gguf-isobpw.md` lines
  44–49): "Q2_K" is **Q4_0×120, Q5_0×24, Q3_K×24 → 4.197 proj_bpw** (file_bpw 5.387);
  IQ3_S is also **4.197**; Q3_K_M **4.574**; Q4_K_M **5.521**. So "Q2_K" on this model
  is **not 2-bit — it is 4.20 bpw on the projection weights.**
- STRAND hits the **requested** bpw on any dimension (row-aware RHT handles ragged 896
  directly): q2 lands at effective **2.6653 bpw** (rel-RMS 25.85%) and q3 at
  **3.6653 bpw** (rel-RMS 13.00%), both billed over the identical 357,826,560 weights
  (`research/isobpw/run.log` lines 13/18).

On quality at these points STRAND does **not** beat GGUF outright:

- **STRAND q2 collapses: PPL 79.94 at 2.665 bpw** (`ppl_strand_q2_l12_out1.json`).
  GGUF's nominal "Q2_K" is far better (15.24) — but only because it is actually
  spending **4.197 bpw**, i.e. 1.57× STRAND's q2 bit budget. This is a size play, not a
  quality win, and the compression map flags 0.5B q2 PTQ quality as family-dependent
  (`docs/STRAND-compression-map.md` line 54).
- The favorable comparison the project leans on: **STRAND at 3.806 bpw reaches PPL
  15.0391, just under GGUF Q2_K's 15.2380 at 4.197 bpw** — STRAND is lower on *both*
  axes there (−1.31% PPL at −0.39 bpw). STRAND also clears IQ3_S (16.9503 @ 4.197)
  decisively, and even STRAND q3 (16.1128 @ 3.665) beats IQ3_S at lower bpw. **Two
  caveats blunt this:** (a) the STRAND row is the *suboptimal* omega-star arm, not
  canonical mp_light (caveat 1); (b) IQ3_S was quantized **without an imatrix**
  (`docs/STRAND-vs-gguf-isobpw.md` lines 58–60, 144/290 tensors fell back); a
  calibrated imatrix would improve its PPL, though not the bpw-tiling story.
- **At higher bit budgets GGUF wins on quality:** Q3_K_M is 13.6111 @ 4.574 and Q4_K_M
  is 12.8937 @ 5.521 — both below every STRAND point measured here, but at higher bpw
  and approaching the bf16 ceiling (12.5358).

### The honest read

The defensible 0.5B claim is narrow and structural: **STRAND delivers a uniform,
requested low bpw on a dimension (896) where the whole GGUF K-quant family pays a large
fallback tax**, so a 3.81-bpw STRAND arm undercuts nominal-"Q2_K" (4.20 bpw) on both
size and PPL. It is **not** a claim that STRAND quantizes *better* at iso-bits —
STRAND's own q2 at 2.67 bpw collapses to 79.94, and GGUF's higher-bpw K-quants beat
STRAND on quality. The map states the limit plainly: "the 0.5B iso-bpw result is real
but narrow… At 256-aligned scale, that specific structural edge shrinks; 7B GGUF is
still an open head-to-head" (`docs/STRAND-compression-map.md` lines 61–63). The
7B/14B GGUF side is unmeasured (queued on the pod).

### Caveats that change what the table means (do not omit)

1. **The "mp_light" row is mislabeled — it is omega-star (attn4/ffn3), a DOMINATED
   arm, not the canonical down-protect mp_light.** The recon was built with
   `configs/rung-attn4-ffn3.json` = q/k/v/o@4, gate/up/down@3 (verified:
   `scripts/isobpw-headtohead.sh` line 114 `strand_recon mp_light --bits 3
   --rung-config configs/rung-attn4-ffn3.json`; config file contents confirmed). The
   PPL JSON itself records `"model": "omega-star"`, `"tag": "omega_attn4_ffn3"`,
   `"model_path": .../research/omega-star/recon` (`ppl_strand_mp_light.json`). The
   compression map explicitly warns: omega-star "= attn@4/FFN@3 … DOMINATED (Ω*: on the
   q3→q4 line, not Pareto) — do not confuse with mp_light," and notes the real **down@4
   mp_light 0.5B point is UNMEASURED** (`docs/STRAND-compression-map.md` lines 51–52,
   388). So the favorable head-to-head uses the *suboptimal* STRAND mixed-precision arm.
2. **The GGUF proj_bpw figures (4.197 / 4.574 / 5.521) have no machine-readable backing
   file in this tree.** They live only in the prose table of
   `docs/STRAND-vs-gguf-isobpw.md`. The PPL JSONs carry `eff_bpw: null`, and the table
   generator (`tools/gguf/isobpw_table.py`) reads `research/isobpw/gguf-bpw.json`, which
   is **absent** (confirmed on 2026-06-13) — the harness
   (`isobpw-headtohead.sh` lines 71–79) is *written to* generate it, so this is an
   un-run step, not missing tooling. Reproducible via `tools/gguf/gguf_bpw.py` against
   the present GGUF files, but not verified against a committed result file —
   **[unverified]**, doc-only.
3. **`docs/STRAND-vs-gguf-isobpw.md` is not finalized:** its Pareto table still shows
   only the mp_light/omega-star row with "(pending)" placeholders (lines 67–70), and
   the Verdict section is empty ("_written once all PPLs land_", line 85). The full
   frontier above is assembled from the individual PPL JSONs and run.log, which *are*
   complete, but the launch doc itself has not been updated to reflect them.

---

## Speed scorecard

Speed is treated as a **threshold good** in this project, not a headline metric: the
frontier doc states plainly that "speed is a threshold good (G4 already cleared it);
quality-per-bit is the scarce good" (`docs/STRAND-quality-density-frontier.md` line
502). The numbers below are what cleared that threshold. Where two measurements of the
same thing exist, both are shown.

### G4 Metal bitslice decode (% of bandwidth, Gw/s)

The decode runtime is an exact-integer Metal bitslice kernel. It is **ALU/issue-bound,
not bandwidth-bound** — the roofline paper-math predicting bandwidth-bound behavior was
wrong here (`docs/will.md` lines 118–119, line 195). The kernel does not saturate
bandwidth because it runs out of issue/ALU headroom first.

The decode rate is fully reconciled against the **measured 98.0 GB/s peak** in
`docs/STRAND-speed-roadmap.md` (grid-stride f32 sum, 256 MB; the 122 GB/s datasheet
number is inadmissible per house rules, lines 294–295):

| Path | Decode rate | % of measured peak (98.0 GB/s) | Speedup vs CPU | Source |
|---|---:|---:|---:|---|
| 3-bit deploy (k3 L7, 512 B TG LUT) | **12.66 Gw/s** (59.4 GB/s moved) | **60.6%** | **3.29×** (vs 3.85 Gw/s) | `docs/STRAND-speed-roadmap.md` line 316 |
| 2-bit reopen (k2 L12, 16 KB TG LUT) | **15.89 Gw/s** (72.5 GB/s moved) | **74.0%** | **3.88×** (vs 4.09 Gw/s) | `docs/STRAND-speed-roadmap.md` line 317 |

These match the productionized re-measure in `docs/will.md` line 542 ("decode
60.6%/74.0% of peak (3.29×/3.88× CPU) both rungs"). An earlier same-day revival table
states the band slightly differently — **11.9–13.1 Gw/s, 68.9–74.1% of peak, 3.73×
CPU** (`docs/will.md` lines 547–548) — and the compression-map anchor gives
**12.66–15.89 Gw/s, 2.6–3.3× vs the 4.85 Gw/s ladder / 3.3–3.9× vs gate CPU**
(`docs/STRAND-compression-map.md` lines 414–415). The spread across these three is
real and is shown rather than collapsed to one optimistic point.

The **fused B=1** path (decode + GEMM in one kernel) runs faster in effective terms:
**35–41 Gw/s effective**, about 7–8× the CPU-rayon primitive
(`docs/STRAND-compression-map.md` line 416), or **34.9–40.8 Gw/s effective** in the
revival table (`docs/will.md` line 548). CPU baselines for these ratios are ~4.85 Gw/s
(best ladder) / 3.85–4.09 Gw/s (in the GPU comparison gate)
(`docs/STRAND-compression-map.md` line 414); the shipping CPU rayon decode is 4.5 Gw/s,
a 6× lift over single-thread (`docs/will.md` line 115).

**Honest caveat on the headline number.** The 60.6%/74.0% figures are *decode-only* on
one ffn tensor on one M3 Pro. The fused B=1 token path is ALU-bound, not
bandwidth-bound (**29.8%/21.7% of peak moved**, `docs/STRAND-speed-roadmap.md` lines
326–327), so "% of bandwidth peak" applies to the decode-to-buffer shape, not
end-to-end token generation. No CUDA decode number exists — the speed roadmap forbids
claiming CUDA from arithmetic (`docs/STRAND-compression-map.md` line 602).

### GPU encode lane (2.36×)

The encode lane is the requant accelerator, validated by a byte-identity gate (660/660
cases, 0 mismatches, both lanes — `research/gpu-encode-results.md` line 52).
Throughput vs the 12-thread CPU canon (f64), kill bar ≥ 2×:

| Config | Full-GPU | 12T CPU canon (f64) | Speedup | Verdict | Source |
|---|---:|---:|---:|---|---|
| k=2 L=12 (2-bit stretch) | 2.825 Mw/s | 1.195 Mw/s | **2.36×** | PASS | `research/gpu-encode-results.md` line 72 |
| k=2 L=10 (envelope edge) | 6.869 Mw/s | 3.382 Mw/s | 2.03× | PASS (marginal) | line 71 |
| k=3 L=7 (3-bit flagship) | 27.808 Mw/s | 5.141 Mw/s | **5.41×** | PASS | line 69 |
| k=2 L=7 | 38.508 Mw/s | 7.181 Mw/s | 5.36× | PASS | line 70 |

The **2.36×** is the k=2/L=12 stretch point — the 2-bit operating point at the deepest
trellis, where the device-cost-rows geometry (256 threads × 16 states, cost rows in
device memory because they exceed the 32 KB threadgroup limit) still clears the 2× bar
(`research/gpu-encode-results.md` lines 72–78). The 3-bit flagship headline has an
honest spread: **5.41× tonight vs 5.87× pre-refactor**, same kernels and machine — the
doc instructs treating **5.4–5.9× as the honest band** (lines 82–83).

**Adoptable, but not yet wired.** A PPL A/B at the 3-bit flagship geometry
(Qwen2.5-0.5B, wikitext-2 test) gives canon-f64 PPL **20.6098** vs f32-lane
**20.6166**, **Δ = +0.033%**, far under the 0.5% adoptability bar
(`research/gpu-encode-results.md` lines 126–133). The GPU lane is byte-identical to
that f32 reference, so it inherits the verdict (lines 134–136). However:
`TropicalEncoder` is currently referenced **only by the gate** —
`encode.rs::encode_tensor_with` still dispatches to the older Metal Viterbi assist, so
the **5.87× is not yet collected in production** (lines 135, 138–142). One wiring step
remains.

### Media encode (1.80×)

A 2026-06-12 media speed pass (release `strand-cli`, best-of-2,
`target/media-bench-20260612/`) lifted **total encode throughput 0.69 → 1.25 MB/s =
1.80×** with compression unchanged (`docs/STRAND-media-speed-roadmap.md` line 44). The
aggregate hides a very uneven distribution:

| Fixture | Before | After | Gain | Source |
|---|---:|---:|---:|---|
| image-rgb-1024x768 | 0.47 MB/s | 1.97 MB/s | **4.16×** | `docs/STRAND-media-speed-roadmap.md` line 41 |
| video-rgb-160x90x60 | 0.29 MB/s | 0.51 MB/s | 1.73× | line 43 |
| code-rust | 1.62 MB/s | 1.68 MB/s | 1.03× | line 40 |
| audio-pcm16-stereo-10s | 11.06 MB/s | 10.94 MB/s | 0.99× | line 42 |
| text-prose | 1.75 MB/s | 1.72 MB/s | 0.98× | line 39 |

**The 1.80× is image-driven.** Image (4.16×) and video (1.73×) carry the win; text,
code, and audio were flat-to-slightly-negative (down to 0.98×). The pass removed
candidate waste (deterministic proxy scoring to shortlist expensive candidates) rather
than changing selected winners, so the gain is real but concentrated in the media types
that were paying for unlikely candidates (`docs/STRAND-media-speed-roadmap.md` lines
46–47). Compression ratios were unchanged across all fixtures.

### The CUDA big-tensor wall

The CUDA encode lane is **validated correct but blocked above 32B**
(`docs/STRAND-quality-density-frontier.md` lines 642–661):

- The lane builds and runs (`--features cuda`, cudarc pinned `cuda-12060` for CUDA
  12.7+; kernel fixed for nvrtc — no `<float.h>`, `COST_INF` as a macro), and a
  tiny-tensor parity gate (`cuda-tiny-gate.sh`) confirmed correct output (lines 650–652).
- **But on 70B's 235M-param FFN tensors at L12, the GPU back-buffer host-staging pushes
  memory past the 125GB cgroup and gets OOM-killed repeatedly.** The CPU completes
  (≈64GB) but takes ~4–6 days on these tensors. 32B tensors fit; 70B does not (lines
  653–655).
- **The fix is known but unbuilt:** batch the GPU block dispatch in
  `encode_tensor_with_cuda` (bounded block batches, like the CPU path) so back-buffer
  staging fits 24GB/125GB. Until then the GPU lane is **for ≤32B only**, and it is the
  prerequisite for both 70B-on-GPU and the 405B flagship (lines 656–658).

As a consequence, **70B q2 is DEFERRED** — not for lack of capability but because the
32B q2 result (6.609, +38% over bf16 4.778) already confirms the scale-tolerance thesis,
making 70B "diminishing information for multi-day cost" against the memory wall (lines
644–647). An auto-routing scaffold (`qm-wrapper.sh` + `.cuda-verdict`) is left in place
on the pod but pinned off until the batching fix plus a real-tensor parity check land
(lines 659–661). The compression-map adds the standing rule that CUDA throughput must
be **measured, not inferred** — the 3090-class "25–35 tok/s 7B primitive" is an
arithmetic estimate only and is explicitly not a claim
(`docs/STRAND-compression-map.md` lines 429–430, 602).

---

## The supercondenser sprint: five targets, framed as physical limits

This section states the sprint's five numeric targets and grounds each one in the
measured floor it is chasing. Read it as a scorecard, not a promise: **none of the five
targets appears verbatim in a source-of-truth file** (grep-confirmed 2026-06-13). The
closest is the density target: `docs/STRAND-compression-map.md` line 16 writes
"~2.42 bpw" — and a 2.40 headline would be a slightly-more-aggressive round of that.
The other four — 2-bit tax ≤0.15, 3-bit tax ≤0.05, decode 90% of peak, encode 5×current
— are *this section's synthesis*; each is pinned to a real measured anchor below but
flagged `[target — not a file quote]`. (The only "90%" strings in the files refer to PV
gain concentration — `docs/STRAND-compression-map.md` line 325, `research/pv-scale-plan.md`
line 157 — a different metric, not decode bandwidth.)

### Target 1 — encoded density: ~2.42 bpw at the 2-bit rung `[target — file says ~2.42]`

| quantity | value | source |
|---|---:|---|
| current q2 encoded artifact | **2.66530 bpw** | `research/bit-ledger-results.md` line 47 (ledger TOTAL encoded, q2; reproduces canon 2.6653 exactly) |
| recoverable: scale + sub_scale (C2) | **0.10597 bpw** (gate row 0.10598) | `bit-ledger-results.md` (scale 0.08401 + sub_scale 0.02196; gate row line 125) |
| recoverable: outlier positions (gap-coded) | **0.14760 bpw** | `bit-ledger-results.md` line 63 (single largest component) |
| recoverable: outlier values | **0.02037 bpw** | `bit-ledger-results.md` line 64 |
| stated C2+OUTL-position recovery → target | **≈0.25 bpw → q2 2.665 → ~2.42 bpw** | `docs/STRAND-compression-map.md` line 16 |

**Why this is the physical limit, not a guess.** The 2.42-class target is the measured
*entropy floor* of the side-info the artifact already carries. The bit ledger
decomposed the 2.665 bpw q2 artifact and found **25% of it sits outside the payload**
(`bit-ledger-results.md` lines 50–52: scale+sub_scale+init 0.359 + outlier channel
0.306 = 0.665 / 2.665 = 25.0%): the 32-bit `scale_q` field carries only ~10.5 bits of
real entropy (H 10.49/32), and the outlier index stream is incompressible in absolute
form (H_abs ≈ 21 bits) but its sorted-**gap** distribution is ~7.82 bits. An ideal rANS
coder built on those models cannot beat that entropy, so the recovery is an *upper
bound*.

**Honest note on the floor number — the sources disagree by ~0.07 bpw.** Two source
files give two different "realistic landing" figures, and they are NOT the same
arithmetic:
- `docs/STRAND-compression-map.md` line 16: **~2.42 bpw**, from ≈0.25 bpw recovery =
  C2 (0.106) + outlier-**positions only** (0.1476).
- `research/bit-ledger-results.md` line 110: **~2.49 bpw**, stated as "(−0.106 C2
  −0.168 OUTL)" — i.e. including outlier **values**. But that subtraction does not yield
  2.49 (2.6653 − 0.106 − 0.168 = **2.391**); the ledger's own 2.49 is internally
  inconsistent with its stated terms, and is best read as matching the map's "near-term
  realistic target 2.45–2.55" band (`STRAND-compression-map.md` lines 111/372).

So the measured-ceiling range is honestly **~2.39 to ~2.49 bpw** depending on whether
outlier values are counted and on rANS-table overhead; the ledger explicitly cautions
that a real coder "will land a few % short, and OUTL gains are quality-gated"
(`bit-ledger-results.md` lines 114–115). The payload itself (2.000 bpw at k=2, ledger
line 36) is the hard wall below which only a *lower rung* or training-created entropy
can go. **Limitation:** the 0.1476 bpw outlier-position recovery is quality-gated
(`STRAND-compression-map.md` §3.9) — it competes against raising tensors q2→q3, so it
may not be free in practice.

### Target 2 — 2-bit quality tax ≤ 0.15 nats `[target — not a file quote]`

The measured 2-bit PTQ taxes, computed directly from the pod-results PPL JSONs:

| model | bf16 PPL | q2 PPL | +% | loss tax (nats) | source files |
|---|---:|---:|---:|---:|---|
| Qwen 7B | 6.6289 | 10.5380 | +59.0% | **0.4635** | `ppl_qwen-7b_baseline.json`, `ppl_qwen-7b_q2_l12_out1.json` |
| Qwen 14B | 5.1023 | 8.9248 | +74.9% | **0.5591** | `ppl_baseline_14b.json`, `ppl_q2_l12_out1_14b.json` |
| Qwen 32B | 4.7785 | 6.6092 | +38.3% | **0.3243** | `ppl_baseline_32b.json`, `ppl_q2_l12_out1_32b.json` |
| Llama2 7B | 5.5353 | 42.4072 | +666% | **2.0362** | `ppl_llama2-7b_baseline.json`, `ppl_llama2-7b_q2_l12_out1.json` |

**Why 0.15 is the limit, and how far we are.** The best *measured* 2-bit PTQ tax is
**0.324 nats** (Qwen 32B) — about **2.2× the 0.15 target** (0.324/0.15 = 2.16). The
target is aspirational; PTQ alone does not reach it. The path is documented and partly
measured: **scale tolerance is real but not monotone** (7B 0.464 → 14B 0.559 → 32B
0.324, 14B the wrinkle); **de-bias is the measured free lever** (−28.67% / ~0.338 nats
at q2 0.5B for 0.0136 bpw, `research/debias-ppl-dp_d4_r2-ab.json`); and **training is
the lever the target actually needs** (the 0.5B q2 PV floor of 26.77 from ~79.94,
`docs/STRAND-compression-map.md` §5.2). **Limitation stated plainly:** no measured
2-bit configuration is at or below a 0.15-nat tax. 0.324 (32B) is the floor we have;
0.15 is the line a trained + de-biased + KL-routed 2-bit must cross, and it is unproven.

### Target 3 — 3-bit quality tax ≤ 0.05 nats `[target — not a file quote]`

| config | bf16 | quant PPL | +% | loss tax (nats) | source |
|---|---:|---:|---:|---:|---|
| Llama2 7B `mp_light` (down_proj@4, rest@3) | 5.5353 | 5.8552 | +5.8% | **0.0562** | `ppl_llama2-7b_baseline.json`, `ppl_mp_light_l12_out1.json` |

**Why 0.05 is the limit — and we are essentially already there.** The measured
Llama2-7B 3-bit `mp_light` tax is **0.0562 nats**, only marginally above the 0.05
target (ratio 1.12). This is the rung `docs/STRAND-compression-map.md` calls the
"shipping 3-bit moat" (Llama2 mp_light 5.855 vs 5.535, +5.8%, line 51). At this tax the
model has lost ~5.6% of its perplexity headroom; the de-bias lever (−10.93% at q3-class
on 0.5B) plus KL-routed placement is the documented route to close the last sliver.
**Limitation (important caveat on the 0.5B iso-bpw point):** the 0.0562 figure is
Llama2-7B. The often-cited 0.5B iso-bpw point — 15.039 PPL @ 3.806 bpw vs GGUF Q2_K
15.238 @ 4.197 bpw — is the **attn@4/FFN@3 "omega-star" split, which
`docs/STRAND-compression-map.md` lines 52 and 388 explicitly flag as DOMINATED and "not
to be confused with mp_light"** (the true down@4 mp_light 0.5B point is unmeasured).
That 0.5B edge comes largely from GGUF's 896-dim fallback tax — a narrow structural
win, not a universal one.

### Target 4 — decode at 90% of measured bandwidth peak `[target — not a file quote]`

| path | rate | % of measured peak | source |
|---|---:|---:|---|
| measured streaming peak (M3 Pro, empirical) | **98.0 GB/s** | — | `docs/STRAND-speed-roadmap.md` lines 294–295 (grid-stride f32 sum, 256 MB) |
| 3-bit decode (k3 L7, GPU bitslice) | 12.66 Gw/s (59.4 GB/s moved) | **60.6%** | `docs/STRAND-speed-roadmap.md` line 316 |
| 2-bit decode (k2 L12, GPU bitslice) | 15.89 Gw/s (72.5 GB/s moved) | **74.0%** | `docs/STRAND-speed-roadmap.md` line 317 |

**Why 90% is the physical limit.** The denominator is the *empirically measured*
98.0 GB/s streaming peak, not the 122 GB/s datasheet number (inadmissible per house
rules). A memory-bound kernel cannot exceed bandwidth, so 100% is the wall and ~90% is
the realistic ceiling once unavoidable side-info traffic is accounted for. STRAND's
decode-only bitslice already reaches **74.0%** at 2-bit and **60.6%** at 3-bit (3.88×
/ 3.29× the 12-core CPU rayon path). The remaining gap is named and measured: the
80-B/block bitslice entry is **43–53% of remaining fused-kernel traffic**
(`docs/STRAND-speed-roadmap.md` line 327), so the "lean side-info entry" is the
explicit lever. **Limitation:** the 60.6%/74.0% figures are *decode-only*; the fused
B=1 token path is ALU-bound, not bandwidth-bound (29.8%/21.7% of peak *moved*), so
"90% of bandwidth peak" applies to the decode-to-buffer shape, not end-to-end token
generation. No CUDA decode number exists.

### Target 5 — encode 5× current `[target — not a file quote]`

| geometry | full-GPU encode | 12T CPU canon (f64) | speedup | source |
|---|---:|---:|---:|---|
| k=3 L=7 (3-bit flagship) | 27.808 Mw/s | 5.141 Mw/s | **5.41×** | `research/gpu-encode-results.md` line 69 |
| k=2 L=12 (2-bit op point) | 2.825 Mw/s | 1.195 Mw/s | **2.36×** | `research/gpu-encode-results.md` line 72 |

**Why ~5× is the realistic ceiling, and it is met at the flagship.** The GPU encode
lane is **at the target for 3-bit**: 5.41× the 12-thread CPU canon (pre-refactor 5.87×;
the doc records the honest band as 5.4–5.9×, lines 82–83), and it is quality-cleared —
the f32 lane is byte-identical to the GPU output across a 660-cell identity gate, and
the adoption A/B measured **ΔPPL +0.033% ≪ 0.5% bar** on the 3-bit flagship geometry
(lines 126–133). The ceiling is physical: once the GPU Viterbi runs at 39–50 Mw/s
(8–10× the entire 12T CPU encode) the lane is "near-saturated" at ~6.4× (line 31).
**Limitation stated plainly:** at the **2-bit op point (k=2/L=12)** the measured
speedup is only **2.36×**, not 5× — the L=12 state explosion (2^12 states, cost rows
spilled to device memory) caps it. So "encode 5×" is *met for 3-bit, unmet for the
2-bit rung that matters most for the density sprint*. And one wiring step remains:
`encode.rs::encode_tensor_with` still dispatches the old Metal Viterbi assist by
default; collecting the 5× requires wiring `TropicalEncoder` in (line 138).

### The unifying frame: these are floors, not goals pulled from air

- **~2.42 bpw** = the entropy floor of the side-info (32-bit scale carries 10.5 bits;
  outlier gaps carry 7.82 bits) — measured; sources bracket the realistic ceiling at
  **~2.39–2.49**.
- **2-bit tax ≤0.15** = below the best *measured* PTQ tax (0.324 @ 32B); reachable only
  via the measured de-bias (−0.338 nats) + PV training (66% PPL reduction to the 26.77
  floor) levers.
- **3-bit tax ≤0.05** = essentially the measured `mp_light` point (0.0562) — the rung
  that ships.
- **decode 90%** = below the 98.0 GB/s wall; we are at 74.0% (2-bit) with the
  lean-table lever named.
- **encode 5×** = the GPU Viterbi saturation ceiling; met at 3-bit (5.41×), not at
  2-bit (2.36×).

The sprint is honest about its own distance: it ships at 3-bit (both quality and encode
targets effectively met), it is *aiming* at 2-bit (every 2-bit target currently unmet
by a measured margin), and the density and decode targets are the entropy/bandwidth
floors no method can pass.

---

## Honest negatives and dead levers

The scorecard of what did *not* work. Optimization metric throughout: log loss tax
against the Qwen2.5-0.5B bf16 anchor `12.536` (`docs/STRAND-quality-density-frontier.md`
line 38; `docs/STRAND-compression-map.md` line 347 gives `12.5358`).

### 1. Diagonal-Hessian / weight-space low-rank residual — dead (RHT-whitened)

The single clearest dead lever. Two independent measurements show why every
weight-MSE-driven method dies under STRAND's randomized Hadamard transform (RHT):

- **The quantization error is white in weight space.** On the `dp_d4_r2` recon, the
  down_proj weight-MSE residual carries only **rank-1 energy 0.24%** (mean `0.0024`,
  median `0.0024`); even rank-64 reaches just **14.08%**
  (`research/error-spectrum-dp_d4_r2_down.json`, `mean_energy_r1 = 0.0024`,
  `mean_energy_r64 = 0.1408`). File verdict: `"DEAD - error is near-white (RHT whitened
  it); low-rank residual will not pay"`.
- **The same error is low-rank in *output* (activation-weighted) space.** For up_proj,
  rank-1 energy is **18.87%** and rank-16 is **37.8%**
  (`research/error-spectrum-dp_d4_r2_up_actw.json`, `mean_energy_r1 = 0.1887`,
  `mean_energy_r16 = 0.378`), verdict `"ALIVE - error is low-rank concentrated; build
  the residual lever"`.

The mechanism is in the design doc: "RHT flattens diagonal curvature into inert
per-block constants and previous A/B hurt PPL" (`STRAND-quality-density-frontier.md`
line 79). The de-bias derivation makes the degeneracy exact: for zero-mean activations
the de-bias correction collapses to the variance reweight `Var[e_i] = Δ_i Σ Δ_iᵀ`, "a
curvature reweight = the Hessian family, already DEAD for STRAND (RHT whitens Σ→σ²I)"
(`research/debias-results.md` line 30). Earlier deterministic A/B numbers confirm:
diag-Hessian at 2-bit measured **301.84 vs 210 (+44%); 117 vs 80.7 (+45%)**, rel-RMS
*fell* while PPL *rose* (`docs/will.md` §4 DEAD table, line ~128).

**Caveat (do not overstate the residual death):** the output-space residual is *alive*
but **marginal for the q2 product** once de-bias is applied. The synthetic math check on
up_proj: recon output error `183.0` → de-bias `153.4` (−16%) → de-bias + rank-16 `147.4`
(only **−3.9% more for +0.34 bpw**) (`STRAND-quality-density-frontier.md` lines 544–546).
The dominant output-error mode *is* the mean shift, which de-bias captures at rank-0
more cheaply than a rank-1 factor. Banked, not built.

### 2. Down-protect as a routing story — wrong tensor

The down-projection-protection family was the prior "2-bit rescue" intuition. The
output-KL hot-swap screen **inverts it**. On `dp_d4_r2` (WikiText train windows, ctx
512, 2 chunks), class-mean logit KL per token (`research/rung-kl-dp_d4_r2.json`,
`class_mean_kl`):

| tensor class | output KL/tok |
|---|---:|
| up_proj | 0.009887 |
| v_proj | 0.009172 |
| gate_proj | 0.006649 |
| o_proj | 0.004525 |
| k_proj | 0.003002 |
| **down_proj** | **0.002787** |
| q_proj | 0.002438 |

down_proj — the tensor every dp-arm protected — is the **second-LEAST**
output-damaging class (only q_proj is lower). The real damage is up_proj / v_proj /
gate_proj. rel-RMS hid this because RHT flattens it across tensor classes
(`STRAND-quality-density-frontier.md` lines 514–527). The 25-tensor RED list and PV
regex (`red_tensors`, `pv_tensors_regex`, both present in the JSON) are top-heavy with
up/gate/v, and the doc flags that the queued 7B selective-PV which targeted
`--pv-tensors down_proj` must be **re-aimed** to the RED set (lines 608–610).

This also kills broad FFN protection as a *density* play (§8.2 placement gate, lines
426–437): uniform q3 sits at PPL `16.11` / tax `0.251` / ~3.4 bpw; `ffn4` (all-FFN@4,
attn@2) reaches PPL `23.53` / tax `0.630` only by spending to **4.54 bpw**; `dp_d4_r2`
lands PPL `40.48` / tax `1.172` at 3.29 bpw. No bit-placement arm reaches the q3 Pareto
point. Verdict: "At 0.5B, bit placement is not the 2-bit lever; training is."
(Down-protect *does* halve the raw q2 collapse — 79.94 → 40.48 — it is just dominated by
q3 at iso-budget and aimed at the wrong tensor.)

### 3. q2 quality is family-dependent — llama2 collapses where Qwen tolerates

Identical recipe (`q2_l12_out1`; ctx 2048, 64 chunks, 131008 tokens), two 7B-class
families:

| model | bf16 PPL | q2 PPL | loss tax (nats) | relative |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 6.6289 | 10.538 | 0.464 | +59.0% |
| Llama2-7B | 5.5353 | 42.407 | 2.036 | +666% |

Sources: `scratch/pod-results/ppl_qwen-7b_baseline.json` (6.6289227),
`ppl_qwen-7b_q2_l12_out1.json` (10.538), `ppl_llama2-7b_baseline.json` (5.5353),
`ppl_llama2-7b_q2_l12_out1.json` (42.407); cross-stated in
`docs/STRAND-compression-map.md` lines 389–391. Loss-tax values match the doc's "+59% /
+666%" framing.

Llama2 starts from a *better* bf16 baseline (5.535 < 6.629) yet its 2-bit PTQ damage is
more than an order of magnitude larger. The 2-bit PTQ recipe is not family-portable. The
doc's own honest projection: even the 0.5B PV law would only map Llama2's 42.4 to ~14,
so "hostile-family success likely needs selective scope, schedule, or broader training
improvements" (`docs/STRAND-compression-map.md` lines 404–406). A measured negative
against any "STRAND q2 works everywhere" claim — it works on Qwen-family, not
Llama2-family, at this recipe.

### 4. The 70B memory wall — q2 deferred, GPU lane OOMs

70B q2 is **DEFERRED**, not done (`docs/STRAND-quality-density-frontier.md` lines
642–669; commit log `0377215`). Two honest reasons:

- **Diminishing information.** Measured relative q2 damage: Qwen 7B **+59.0%**, 14B
  **+74.9%**, 32B **+38.3%** (computed from the pod JSONs; cross-stated
  `docs/STRAND-compression-map.md` lines 389–390 and frontier line 646). The doc calls
  70B "diminishing information for multi-day cost" (lines 646–647). [Note: the doc lines
  645/647 round 32B q2 to "6.609"; the JSON value is 6.6092402908416155.]
- **A hard memory wall on the GPU encode lane.** The CUDA Viterbi lane is validated
  *correct* (tiny-tensor parity gate passed), but "on 70B's 235M-param FFN tensors at
  L12, the GPU back-buffer host-staging pushes memory past the 125GB cgroup →
  **OOM-killed** repeatedly" (lines 653–654). CPU completes (≈64GB) but is ~4–6 days.
  32B fit; 70B does not. The stated prerequisite is to **batch the GPU block dispatch**
  in `encode_tensor_with_cuda` (lines 656–658).

So the 32B q2 result (PPL **6.6092**, +38% over bf16 4.7785) is the current top of the
scale ladder; everything above it is a memory-bound IOU, not a measured point.

### 5. WSD-cooldown PV — measured loss vs cosine

The Apple-style WSD warmup/hold/cooldown schedule was meant to be a clean A/B against
the cosine PV recipe — "only the schedule differs vs the 26.77 run, so the delta IS the
cooldown lever" (`docs/will.md` lines 458–461). It **lost**.

- Cosine PV (300 steps, chunked-KD, 4 requant boundaries) reached **27.02 PPL**,
  asymptote **26.77** after +600 more steps (`docs/will.md` line 712
  `78.95 → 30.48 → 28.51 → 27.41 → 27.02`; line 700 "pv2 (+600 steps) = 26.77").
- WSD-cooldown PV, same start, same 300 steps, same lr 1e-4, same KD, same full
  168-tensor wrap, same eval (ctx 2048 / 64 chunks): `ppl_before 78.952 → ppl_after`
  **34.55859952221674** (`research/pv-deep/pv-cooldown.json`: `"ppl_after":
  34.55859952221674`, `"steps": 300`, `"lr": 0.0001`, `"kd": true`, `"wrapped": 168`,
  `"eval_chunks": 64`, `"eval_ctx": 2048`, `"pv_tensors": ""`, `"pv_count": -1`).

The cooldown run landed at **34.56 vs the cosine 27.02** it was designed to beat —
**+27.9% PPL** (and +29.1% vs the 26.77 asymptote), a clear regression from changing
only the schedule. The planning doc had bracketed WSD cooldown at "0–5% relative PPL
improvement over 26.77 … until the live run lands" (`docs/STRAND-compression-map.md`
line 397); the live run landed *negative*, outside even the bottom of that range.

**Honest scoping caveat:** both arms are full-PV (`pv_tensors: ""`, `pv_count: -1`), so
the recipes are comparable. The frontier doc still lists the "WSD A/B" as a queued open
gate (`docs/STRAND-quality-density-frontier.md` lines 493, 325), so this single cooldown
run is treated as indicative rather than a fully closed multi-seed verdict — but as a
measured data point, cosine beat WSD-cooldown decisively here.

### Adjacent closed channels (for completeness)

- **Ternary (1.5-bit) PV — channel closed at 0.5B.** Gentle low-LR restart from the 2k
  checkpoint *damaged* the saturated state: `72.13 → 77.08 (+6.9%)`; ternary final
  **71.95** (`docs/will.md` lines 693–697).
- **Proxy QAT through a different quantizer — dead.** "Train through what ships"
  (`docs/STRAND-quality-density-frontier.md` line 81); measured **3,013 vs 80.7** in the
  will.md DEAD table.
- **More scalar L beyond 12 — saturated.** L≥13 "already looked saturated/expensive"
  (lines 66, 83).
- **PTQ vector rungs — do not ship (collapse already seen).** Only PV-trained vector
  modes stay alive (lines 71, 309–313).

### What survives (the contrast)

- **De-bias (rank-0 output correction) — ADOPTED.** `40.245 → 28.708` PPL, ratio
  `0.7133` = **−28.67%**, for **~0.0136 bpw** (`research/debias-ppl-dp_d4_r2-ab.json`;
  bpw bill `STRAND-quality-density-frontier.md` line 458). Generalizes: `dp_d4_r3`
  (q3-class) `19.075 → 16.990` = **−10.93%** (`research/debias-ppl-dp_d4_r3-ab.json`).
- **PV training (cosine) at 0.5B** recovers the bulk of q2 collapse: `78.95 → 27.02`,
  asymptote `26.77` (≈2.13× bf16) (`docs/will.md` lines 700, 701, 712).
- **The one narrow iso-bpw GGUF edge.** STRAND mp_light/omega-star PPL **15.039** at
  **3.81 bpw** beats GGUF Q2_K **15.238** at **4.20 bpw** — but only because
  Qwen2.5-0.5B's 896-wide tensors trigger a GGUF K-quant fallback tax, and this is the
  *dominated* attn@4/FFN@3 split, not a Pareto point. Stated honestly as "real but
  narrow" (`docs/STRAND-compression-map.md` lines 52, 61–62, 388).

---

## What's verified vs pending (honesty box)

**VERIFIED — every number traced to a real file in this tree (re-checked 2026-06-13):**

- The q2 scale curve (7B +59.0%/0.4635, 14B +74.9%/0.5591, 32B +38.3%/0.3243) and the
  Llama2-7B q2 collapse (+666%/2.036) and mp_light (+5.8%/0.0562) — all from
  `scratch/pod-results/ppl_*.json`, arithmetic recomputed and confirmed.
- De-bias A/B: q2 −28.67% (40.24473 → 28.70825), q3 −10.93% (19.07542 → 16.99027), both
  LOCAL_PASS, 0.0136 bpw — `research/debias-ppl-dp_d4_r{2,3}-ab.json`.
- PV lineage 78.95 → 27.02 → 26.77 (≈2.13× bf16) — `docs/will.md` lines 700–712.
- C2 bit-ledger (scale 0.08401, sub_scale 0.02196, outl_pos 0.14760; reproduces 2.6653
  exactly) — `research/bit-ledger-results.md`.
- KL inversion (down_proj 2nd-least at 0.002787; up/v/gate highest) and error spectrum
  (weight-space r1 0.24% DEAD; output-space r1 18.87% ALIVE) —
  `research/rung-kl-dp_d4_r2.json`, `research/error-spectrum-*.json`.
- iso-bpw PPLs (all eight rows) — `research/isobpw/ppl/*.json` + `run.log` (STRAND
  denominator 357,826,560 file-verified).
- Speed: G4 decode 60.6%/74.0% of 98.0 GB/s (12.66/15.89 Gw/s) —
  `docs/STRAND-speed-roadmap.md` lines 294–327, `docs/will.md` line 542; GPU encode
  2.36× (k2/L12) and 5.41× (k3/L7), 660/660 identity, ΔPPL +0.033% —
  `research/gpu-encode-results.md`; media 1.80× — `docs/STRAND-media-speed-roadmap.md`.
- WSD-cooldown PV loss: 34.55859952221674 vs cosine 27.02 (+27.9%) —
  `research/pv-deep/pv-cooldown.json`.

**Corrections applied to source-doc wording (the file's own numbers are right, the
prose was off):**

- "monotone improvement with scale" (`STRAND-quality-density-frontier.md` line 646) is
  contradicted by line 619 ("NOT monotone") and by the data (14B is the worst point).
  Reported as U-shaped / "32B best of three."
- "~245 nats per bpw" for de-bias (`STRAND-quality-density-frontier.md` line 459) is a
  10× slip; the real figure is **~25.2 nats/bpw**.
- `research/bit-ledger-results.md` line 110 "~2.49 bpw (−0.106 C2 −0.168 OUTL)" is
  internally inconsistent (the subtraction yields 2.391); the honest ceiling range is
  ~2.39–2.49 bpw.

**UNVERIFIED / PENDING (flagged [unverified] in-text; not fabricated):**

1. **Live `pv_down_protect_q2` verdict** — `research/pv-dp/pv-dp.json` is **absent** on
   disk (confirmed 2026-06-13). The down-protect PV final PPL is genuinely not in a file.
2. **All de-bias / PV / KL-routed scale confirms (7B / 14B / 32B).** Every confirmed
   quality lever (de-bias, PV, KL placement) is **0.5B only**; the scale runs are
   *defined and queued* behind a PROMOTE_CLOUD gate (`STRAND-quality-density-frontier.md`
   §8.3 item 3, §10) but **not run**. "Near-free at scale" is unproven.
3. **GGUF proj_bpw (4.197 / 4.574 / 5.521).** Doc-only in `STRAND-vs-gguf-isobpw.md`;
   the machine-readable `research/isobpw/gguf-bpw.json` is **absent**, and the PPL JSONs
   carry `eff_bpw: null`. Reproducible from the present GGUF files but not committed.
4. **The five supercondenser targets** (2.40 bpw headline, 2-bit tax ≤0.15, 3-bit tax
   ≤0.05, decode 90%, encode 5×) — **none appears verbatim in any source file**
   (grep-confirmed); each is a synthesis pinned to a measured anchor, flagged `[target]`.
5. **70B / 405B quality.** 70B q2 DEFERRED (memory wall); 405B is a gated flagship, not
   a measured point.
6. **CUDA decode/encode throughput at scale.** The CUDA encode lane is parity-correct on
   tiny tensors only; no CUDA decode number exists and the project forbids claiming CUDA
   from arithmetic.

The single load-bearing honest framing: **STRAND's measured wins are 0.5B (levers) and
one pod tier each up to 32B (the q2 scale curve); the moat is determinism, and the
"beats GGUF at iso-bits on quality" claim is explicitly NOT made** — the iso-bpw edge is
a narrow 896-dim structural win on a mislabeled (omega-star, not mp_light) arm.
