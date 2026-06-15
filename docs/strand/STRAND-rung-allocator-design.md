# STRAND rung allocator + selective PV (Tier 2a) — design

_Status: DESIGN. Nothing here is measured except the cited canon (every number below is from
will.md or re-derived from it; derivations are marked). The only built artifact is
`scripts/rung-screen.py` (stage-1 screen + allocator math, dry-run verified). The patch plan
in §5 is NOT applied — `scripts/strand-qat.py` is run-frozen while the marathon runs._

---

## 0. The law this design formalizes

Tonight's measurements (will.md §10, 2026-06-10), all Qwen2.5-0.5B, canon 64-ch eval,
bf16 = 12.55:

| rung | PTQ start | PV (300 steps, lr 1e-4) | collapse ratio C = PTQ/bf16 | verdict |
|---|---|---|---|---|
| 3-bit (q3_l12_out1, 3.66 bpw) | 16.10 | 19.64 (peak 20.14) | 1.28 | **DAMAGE +22%** |
| 2-bit (q2_l12_out1, 2.66 bpw) | 78.95 | in flight (chunked-KD re-pass) | 6.29 | locates the crossover |
| ~1.5-bit (vec-d2, 2.165 bpw billed) | 48,978 | **424.46** (4 requant boundaries, still −19%/segment) | 3,903 | **WIN 115×** |

Supporting datum: ternary-3k (will.md 2026-06-10 15:32) — a warm restart of a *converged*
checkpoint at peak lr 1e-4 burned ~600 steps re-converging and gained nothing; the recorded
lesson is "~3e-5-class LR for converged states."

**The law: PV training value is inverse to PTQ starting quality.** Where PTQ collapses,
re-learning is THE tool (ParetoQ's 2↔3-bit transition, will.md §6: below it the good solution
is a different weight-setting, not a rounding). Where PTQ is near-converged, full-LR PV
*destroys* the basin it should be polishing.

**The formalization this document builds:** treat the law per-tensor instead of per-model.
Assign each of the N quantizable tensors (0.5B: 168, 7B: 196) a rung r_t ∈ {2, 3, 4}-bit to
minimize average bpw subject to a quality constraint (the allocator, §3), then PV-train ONLY
the tensors that are collapsed at their assigned rung and freeze the rest (selective PV, §4).

**The per-tensor hypothesis is unproven.** The law above is model-level at uniform rungs.
That per-tensor damage predicts per-tensor PV response is the bet of this design; E4 (§7)
tests it before anything scales. Stated per §5.10: this whole document is *provisional* until
E2–E4 produce numbers.

Why this is worth building (the prize, from canon): at 0.5B the uniform ladder is
q2 = 80.7 @ 2.66 bpw, q3 = 16.11 @ 3.66, q4 = 13.535 @ 4.34. mp_light at 7B proved one
hand-picked class split (down_proj@4, rest@3) buys 9.42 → 8.45 for +0.33 bpw — and that
attn@4 adds *nothing* (q3_mixed_heavy = 8.45 = mp_light, dominated). The allocator is the
systematic version of that discovery, per-tensor, with PV as the rescue for what bits can't fix.

---

## 1. Tensor inventory (derived, cross-checked)

Qwen2.5-0.5B (hidden 896, inter 4864, 24 layers, 14Q/2KV heads):

| class | tensors | params | share |
|---|---|---|---|
| down_proj (896x4864) | 24 | 104.6M | 29.2% |
| gate+up_proj (4864x896) | 48 | 209.2M | 58.5% |
| attn q/k/v/o | 96 | 44.0M | 12.3% |
| **total proj** | **168** | **357.9M** | 100% |

Cross-check: 357.9M = the harness's printed trainable count (will.md §8, "357.9M trainable") —
the derivation is exact, not estimated.

Qwen2.5-7B: 196 proj tensors, 6.525B weights, down_proj 29.1%. Cross-check: predicted
mp_light bpw = 0.291×4.344 + 0.709×3.34 = 3.63 ≈ canon 3.67 (the gap is 4-bit affine-min
side-info; the canon billed number governs).

Rung costs (billed, will.md §3 density + outlier billing verified in `quantize_one`:
`eff_bpw += f * (idx_bits + outlier_bits)`):

| rung | config | bpw (billed) | 0.5B canon PPL |
|---|---|---|---|
| r2 | `--bits 2 --l 12 --outlier-channel 1` | ~2.66 | 80.7 |
| r3 | `--bits 3 --l 12 --outlier-channel 1` | ~3.66 | 16.11 |
| r4 | `--bits 4 --l 12` | ~4.34 | 13.535 |
| (r15, phase 2) | `--bits 3 --vec-dim 2 --learned-codebook ...` | 2.165 billed | 50,134 (PTQ collapse; PV-only rung) |

Per-tensor bpw varies with shape (ragged 896-dim blocks, affine-min at 4-bit) — the allocator
uses the *measured* per-tensor bpw from the screen's sidecar, never the analytic floor
(§5.11, bill everything).

---

## 2. The screening metric — per-tensor PTQ damage

### 2.1 Stage 1: rel-RMS (exists today; within-family proxy ONLY)

One `quantize-model --measure-only --out <p> <rung-flags>` invocation per rung emits a sidecar
`<p>.json` with per-tensor `{name, n, bits, bpw, rel_rms_pct}` for every proj tensor in the
shard — 168 rows per pass, three passes total. (`--only` per tensor gives byte-identical
numbers — tensors are quantized independently: per-tensor FNV-1a RHT seed, per-tensor scales,
per-256-block Viterbi — at 168× the process/shard-read overhead. `rung-screen.py` defaults to
the batched form and keeps `--mode per-tensor` for spot checks.)

What stage 1 MAY be used for (will.md §5.5 discipline):
- anomaly detection (catastrophic per-tensor spikes — the "~47% rel-RMS outlier" class named
  in the `quantize-model.rs` learned-LUT non-regression guard; measured instance: 47% on
  `layers.25.o_proj` when Lloyd diverged, `docs/archive/STRAND-density-roadmap.md`);
- monotonicity sanity (rel-RMS must fall with bits for every tensor; a violation = bug);
- ranking *within* one tensor across rungs of the same family.

What stage 1 MUST NOT be used for: cross-tensor importance or any PPL prediction. The diag-H
death is the standing proof (rel-RMS *fell* 24–26% vs 27% while PPL *rose* +44% — proxy down,
truth up). A 28% rel-RMS on a down_proj and a 28% rel-RMS on a k_proj are not the same damage;
output sensitivity differs per tensor and rel-RMS cannot see it.

### 2.2 Stage 2: the swap screen — per-tensor logit-KL vs bf16 (the new, cheap truth)

**Mechanism.** Because PTQ is per-tensor independent, ONE full-model quant per rung (with
`--out recon.safetensors`) yields every tensor's rung-r recon in a single file. The model
"bf16 everywhere except tensor t at rung r" is assembled by hot-swapping one tensor's weights
in a loaded bf16 model — no further encoder invocations. Per tensor:

1. swap recon[t] into the live bf16 model (≤ 17 MB f32 at 0.5B, ≤ 271 MB at 7B);
2. forward 2 fixed chunks (2 × 2048 tok, WikiText-2 **train** slice — see corpus note below);
3. score vs cached bf16 reference logits;
4. restore the original tensor.

**The measure.** Per token, two numbers from the same forward:

- `dkl_nats` = mean_positions KL(p_bf16(.|ctx) || p_t,r(.|ctx)) — the primary ranking metric.
- `dnll_nats` = mean ΔNLL vs the bf16 model on the actual next tokens — the canon-family
  cross-check.

These estimate the same quantity: ΔNLL ≈ E_data[log p_bf16/p_q], and with p_bf16 standing in
for the data distribution that expectation IS the measured KL. The KL is the
variance-reduced estimator (it integrates over the full vocab instead of the single sampled
token), which is what makes a 2-chunk screen usable: the design is *paired* (same tokens, same
everything except one tensor), so cross-tensor comparisons cancel the text-sampling noise.
Ranking is what the allocator needs; absolute calibration comes from E2c (§7).

**Corpus choice.** Screen on WikiText-2 *train* chunks (the same distribution PV trains on),
validate allocations on the canon test 64-ch eval. Two reasons: (a) never optimize against the
eval set; (b) the diag-H corpse is a cross-corpus calibration (C4 calib → WikiText eval)
overfitting — staying inside WikiText train/test keeps the corpus-shift surface minimal, and
the final arbiter is always the canon test PPL.

**Cost (estimate — verify in E2, measure-don't-model applies to plans too).** bf16 reference
logits cached once (2 × 2047 × 151,936 vocab, fp16 ≈ 1.24 GB on disk). Per tensor: ~2 chunk
forwards on MPS bf16 + chunked-row KL (the `eval_ppl` 512-row trick, same OOM discipline).
Ballpark 5–8 s/tensor → ~15–25 min per rung for 168 tensors, x3 rungs, plus the three recon
quant passes. All `nice -n 19`, never concurrent with QAT (§7 freeze trap).

**Stage-2 harness** is a separate future script (`rung-kl.py`, NOT built in this tier);
`rung-screen.py` emits the stage-1 table it joins into and accepts the join (`--stage2-csv`).

### 2.3 Decision thresholds (provisional — E2c calibrates)

Work in nats of log-PPL. Define the damage budget against a quality target:
`D_budget = ln(PPL_target / PPL_bf16)`. Canon anchors: q4-class = ln(13.535/12.55) = 0.076,
q3-class = ln(16.11/12.55) = 0.250.

First-order additivity assumption: `ln(PPL_mix/PPL_bf16) ≈ α · Σ_t δ_t` with δ_t = per-tensor
`dkl_nats` and α a superadditivity safety factor (start α = 1.5; E2c regresses it on 3 real
configs). Additivity is *expected to fail* for collapsed tensors — that is fine: the screen
only has to separate benign from damaging, and collapsed tensors saturate into the RED class
where their exact δ is irrelevant.

Per-tensor classification at an assigned rung:

| class | rule (per-token nats) | meaning | action |
|---|---|---|---|
| GREEN | δ_t ≤ τ_g = 0.3 · D_budget / N | converged at this rung | freeze in PV; ship PTQ bits |
| AMBER | τ_g < δ_t < τ_r | recoverable by bits | allocator buys rung-ups here first; PV only at 3e-5-class LR |
| RED | δ_t ≥ τ_r = 0.05 | collapsed / out-of-basin | PV re-learning at 1e-4-class LR, or rung-up if budget allows |

τ_g rationale: 168 GREENs may collectively spend ≤ 30% of the budget (q3-class:
τ_g ≈ 4.5e-4 nats/tok). τ_r rationale: one tensor singly costing ≥ +5% PPL is past
"rounding error" and into "wrong basin" territory — the per-tensor analogue of the
C ∈ (1.28, 6.29] model-level crossover bracket. **Both thresholds are provisional**; the 2-bit
re-pass verdict (in flight tonight) narrows the bracket, and E2c fits the constants. The
architecture is parameterized so new verdicts update τ, not the design.

### 2.4 Screen self-validation (the gate before anything uses it)

The screen must reproduce the one allocation fact already proven at 7B: **down_proj is the
whole lever at 3-bit** (mp_light = q3_mixed_heavy = 8.45; attn@4 adds nothing). At rung r3 the
down_proj class must dominate the damage ranking, and attn tensors must screen cheap. If it
does not, the screen is broken — stop, do not allocate with it (E2a kill, §7).

---

## 3. The allocator — water-filling over rungs

### 3.1 Formulation

Given per-tensor damage curves δ_t(r) (stage 2) and billed per-tensor costs
c_t(r) = n_t · bpw_t(r) (stage 1 sidecar, measured):

    minimize   B = Σ_t c_t(r_t) / Σ_t n_t                 (average billed bpw)
    subject to Σ_t δ_t(r_t) ≤ D_budget / α                (quality constraint)
               r_t ∈ {r2, r3, r4}

Lagrangian relaxation: minimize Σ_t [ c_t(r_t) + λ δ_t(r_t) ] — separable, so each tensor
independently picks r_t(λ) = argmin_r [ c_t(r) + λ δ_t(r) ]. Sweeping λ traces the entire
bpw-vs-damage Pareto frontier from ONE screen's data. This is exactly HAWQ-style reverse
water-filling — will.md §4 LIVE queue #7, now with a measured sensitivity instead of a
modeled Hessian (which is structurally dead here: the RHT whitens).

### 3.2 Algorithm (implemented in `rung-screen.py allocate`)

1. Per tensor: damage is monotone non-increasing in bits (stage-1 monotonicity gate enforces
   the proxy version; stage-2 violations at the noise floor are clamped). Take the lower
   convex hull of the (cost, damage) points — dominated rungs drop out.
2. Start every tensor at its cheapest hull point (r2). Maintain a max-heap of hull edges keyed
   by marginal efficiency Δδ_t / Δc_t (damage removed per extra bit).
3. While Σδ > target: pop the steepest edge, upgrade that tensor one hull step, push its next
   edge. (Greedy on convexified curves = optimal for the relaxation; the integrality gap is at
   most one tensor's step — irrelevant at N = 168.)
4. Emit:
   - `mp-<name>.json` — exact-tensor-name MpRules (see §3.3),
   - the residual RED list = tensors with δ_t(assigned) ≥ τ_r even at r4, or tensors the
     budget forced to stay collapsed at r2 → **the PV set** (§4),
   - predicted (B, Σδ) and the λ at the stop point.

Duality: `--target-dnats` (quality-constrained, the primary mode per this task) and
`--budget-bpw` (bit-constrained) walk the same frontier from opposite ends.

### 3.3 Emission details (traps)

- MpRule pattern is a **substring**, first match wins, fallback = `--bits`. Emit exact
  patterns of the form `layers.<i>.<path>.<proj>` *with the surrounding dots*
  ("layers.5.mlp" cannot false-match "layers.51.mlp" because the required ".m" follows the
  "5"; rung-screen emits full dotted paths and verifies pairwise non-collision anyway).
- One rule per non-fallback tensor: 168 rules is nothing (`parse_mp_config` is a linear scan).
- `--l 12` explicit applies to ALL rungs in one invocation (`for_bpw_l(bits, args.l)`) — this
  is the canon trio q2_l12/q3_l12/q4_l12 in a single pass. The q4_l12 lever is FREE
  (13.535 vs 13.948 default-L, will.md −3.0%).
- **Outlier-channel granularity gap:** `--outlier-channel` is global per invocation; an mp
  mix at {r2,r3,r4} gives r4 tensors the +0.32 bpw outlier channel that canon q4_l12 does not
  carry. v1: accept and bill it (the sidecar bills per tensor, so reported bpw stays honest).
  Proper fix (small Rust patch, NOT in this tier): per-rule `outlier_pct` in MpRule. Open
  question §8.3.
- The same mp JSON drives the screen, the in-loop PV requant, and the final deploy quant —
  the dumped shadow names (`model.layers.i...down_proj.weight`) are the HF shard names, so
  one rules file serves all three. No translation layer.

### 3.4 Worked example (analytic, canon densities — illustrative only)

`configs/mp-2bit-down3.json` (down@3, rest@2, +out1) — the PARKED hand allocation:
B = 0.708×2.336 + 0.292×3.34 + 0.32 = **2.95 bpw**. Its PPL was never measured (three
incomplete deaths, will.md §8) — E3 finally banks it as the baseline the per-tensor allocator
must beat at iso-bpw. The allocator's pitch: at the same 2.95 bpw it can split the down_proj
budget unevenly across layers and buy r3/r4 exactly where δ_t says it pays.

---

## 4. Selective PV — train only the collapsed, freeze the converged

### 4.1 The rule

After allocation, the PV set S = the RED list (§3.2). Everything else is FROZEN: no gradient,
no optimizer state, no requant after init. LR policy, grounded:

| class | LR | grounding |
|---|---|---|
| RED (in S) | 1e-4 peak, cosine | pv15bit: 48,978 → 424 at 1e-4 — re-learning wants full LR |
| AMBER (only if explicitly opted in) | 3e-5 peak | ternary-3k: 1e-4 warm restart burned ~600 steps; pv3bit: 1e-4 damaged near-converged 22% |
| GREEN | frozen | pv3bit is the direct measurement: there is no safe full-LR pass over converged tensors |

### 4.2 Why the delta forward makes freezing FREE (zero new forward code)

Strand-mode forward is `wq = base + w` with `base = recon − w_anchor` re-set at each requant
(the delta forward v2, commit 00c8563 — the frozen-STE 59e9 corpse is §4 DEAD). For a frozen
tensor, `w` never moves, so `base + w = recon` **exactly, forever** — a frozen QuantLinear
serves its PTQ recon bit-faithfully with no special-casing. Freezing is purely
`requires_grad_(False)` plus optimizer exclusion (which follows automatically: the param list
is built from `requires_grad`).

### 4.3 Determinism amplification (the moat doing work)

Frozen tensors' shadows are never touched, the in-loop dump casts them bf16 → the *original
bf16 bytes* (exact round-trip: source is bf16), and the encoder is deterministic with a
name-seeded RHT. Therefore the deploy artifact's frozen tensors are **bit-identical to the
screened PTQ artifact** — the screen's per-tensor verdicts remain valid for the shipped bits,
and only the PV set ever needs re-screening. A float-codebook competitor cannot make this
claim; we get it from the determinism contract for free.

### 4.4 PATCH PLAN for `scripts/strand-qat.py` (run-frozen — DO NOT APPLY until the marathon drains)

The script is frozen per will.md (do NOT edit while the orchestrator runs). Hunks below are
anchored to the current file (450 lines, sha at frontier-wave a592c3c) and are the complete
change. Net: ~30 lines, all additive, default behavior byte-identical (empty regex = today).

**P1 — flag.** After the `--train-all` argument (lines 237–240):

```python
    p.add_argument("--pv-tensors", default="",
                   help="selective PV: regex (re.search) over QuantLinear module names; "
                        "matching tensors train, the rest freeze at their requant recon "
                        "(delta forward: base+w is exact recon while w is frozen). "
                        "Empty = train all wrapped (today's behavior).")
```

**P2 — freeze.** Between the existing `--train-all` freeze block (ends line 282) and the
`ntrain` count (line 283):

```python
    npv = -1
    if args.pv_tensors:
        import re
        rx = re.compile(args.pv_tensors)
        npv = 0
        for name, m in model.named_modules():
            if isinstance(m, QuantLinear):
                sel = bool(rx.search(name))
                m.weight.requires_grad_(sel)
                if m.bias is not None:
                    m.bias.requires_grad_(sel)   # bias rides with its weight (v1; see E4d)
                npv += int(sel)
        assert npv > 0, f"--pv-tensors {args.pv_tensors!r} matched 0 QuantLinears"
        print(f"[qat] selective PV: {npv}/{nwrapped} QuantLinears trainable", flush=True)
```

Downstream needs zero changes: `params` (line 324), AdamW, and `clip_grad_norm_` all derive
from `requires_grad`; KD/CE/grad-checkpoint are weight-set-agnostic.

**P3 — selective requant.** In `strand_requant` (line 126): the init requant must stay FULL
(every frozen tensor needs its `base` anchored to the allocator's recon); subsequent requants
dump and reload only the PV set — frozen recons cannot change (§4.2/§4.3).

```python
# dump loop (line 140): replace the isinstance gate with
    sel = (tag != "init") and bool(getattr(args, "pv_tensors", ""))
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear) and (not sel or m.weight.requires_grad):
            sd[name + ".weight"] = m.weight.detach().cpu().to(torch.bfloat16).contiguous()

# reload loop (line 148): key off the recon file's contents, not the module list
    with safe_open(recon, framework="pt") as f:
        have = set(f.keys())
        for name, m in model.named_modules():
            if isinstance(m, QuantLinear) and (name + ".weight") in have:
                r = f.get_tensor(name + ".weight").to(torch.float32).to(args.device)
                m.base.copy_(r - m.weight.data)
                del r
```

**P4 — mixed rungs in the loop: no code change.** `--strand-flags` already passes through to
the subprocess argv (`args.strand_flags.split()`, line 144). The selective-PV launch is:

```
--strand-flags "--mp-config /abs/path/mp-alloc.json --bits 2 --l 12 --outlier-channel 1 --threads 8"
```

Constraints: absolute path (the subprocess inherits cwd), no spaces in the path (`.split()`),
and the fallback `--bits` must equal the allocator's base rung.

**P5 — provenance.** Add `"pv_tensors": args.pv_tensors, "pv_count": npv` to the `--out`
json dump (line 442).

**P6 — deferred (not v1).** Per-class LR via AdamW param groups (RED 1e-4 / AMBER 3e-5 in one
run). v1 runs RED-only at a single LR; an AMBER pass, if E4 motivates it, is a separate
invocation at 3e-5 (`--pv-tensors <amber-regex> --lr 3e-5`), which also keeps the law's
per-rung LR evidence cleanly separated.

**Compatibility notes.** `--init-state` checkpoints are shape-identical either way (full
state_dict, `base` buffers included) — selective and full runs can resume each other.
Segmented arms (`--skip-after` / `--chunk-offset`): the regex MUST be identical across
segments — it belongs in the conductor's arm spec next to `--strand-flags`. `--save-hf` +
final deploy: quantize the saved dir with the SAME mp JSON; frozen tensors reproduce their
screened bits exactly (§4.3).

---

## 5. Cost model — what selective PV buys

Anchors (will.md): step ≈ 4 s (rung 1: 20 steps / 80 s, 0.5B MPS, grad-checkpoint);
full-0.5B requant ≈ 15 min at 12 threads (~900 s; in-loop default is `--threads 8`, so
treat 900 s as the floor); strand-mode MPS demand = 13.3 GB (commit 0873188); KD teacher
≈ +1.4 GB pool (8.6 GB uniform/no-KD vs 10.0 GB ternary+KD); requant cadence default 75.

Scenario: PV set = the down_proj class (24 tensors, 104.6M = 29.2% — the likely shape of a
RED set, pending the screen).

| item | full PV (168) | selective (24 down_proj) | scaling law |
|---|---|---|---|
| trainable params | 357.9M | 104.6M | Σ n_t over S |
| AdamW m+v (fp32, 8 B/param) | 2.86 GB | 0.84 GB | linear in S |
| grads (fp32, 4 B/param) | 1.43 GB | 0.42 GB | linear in S |
| `base` buffers | 1.43 GB | 1.43 GB | ALL tensors forward through recon — no cut |
| steady-state memory saved | — | **≈ 3.0 GB** | vs the 13.3 GB strand-mode demand |
| requant per boundary | ~900 s | ~260 s | ∝ quantized weights (Viterbi is per-weight) |
| segment @ cadence 75 | 300 s train + 900 s requant = 1200 s | 300 + 260 = 560 s | **2.1× PV throughput** |
| KD teacher (fwd + memory) | unchanged | unchanged | teacher is the full model, always |

**The decisive line is the requant.** At cadence 75, full PV spends 75% of wall-clock inside
the Rust encoder. Selective PV either reclaims it (2.1× throughput) or — better — buys
cadence ~25 at the old overhead: 3× more requant boundaries per step is a tighter delta-forward
feedback loop, which is exactly the knob will.md's rung-3 design names ("the outer-loop
cadence is the design knob; AQLM-PV does few rounds").

Step time: freezing skips the dW GEMM (1 of the ~4 GEMM-units/linear under checkpoint
recompute) on 71% of proj params → estimate ~10–15% step cut, diluted by lm_head CE, KD
teacher forward, attention, and norms. Claim NOTHING here until E4 prints its s/step — the
memory and requant rows are the real wins, the step row is a bonus.

Memory consequence on the 18 GB box: −3.0 GB against the 13.3 GB strand-mode demand is the
difference between the teacher-parking dance at every eval and a comfortably resident KD
teacher (or batch 2). It is also 7B-relevant: PV at 7B is currently unthinkable locally;
selective PV on a 29% subset cuts optimizer+grad state from ~30 GB-class to ~9 GB-class
(still pod work, but a different pod class — see will.md 70B go-condition economics).

---

## 6. Interfaces (what exists after this tier)

```
scripts/rung-screen.py screen   --model-dir <hf-dir> [--configs <json>] --out-dir <dir>
                                [--mode batch|per-tensor] [--dry-run] [--resume]
                                [--stage2-csv <csv>]      # join rung-kl.py output when it exists
scripts/rung-screen.py allocate --screen-csv <long.csv> --damage-col dkl_nats
                                (--target-dnats X | --budget-bpw B) --out-dir <dir>
                                [--alpha 1.5] [--tau-red 0.05] [--dry-run]
```

`screen` invokes `quantize-model --measure-only` (batched per rung per shard; per-tensor mode
uses `--only <full tensor name>` — names are unique, `--only` is substring) with
`STRAND_NO_GPU=1` (the Metal encode watchdog SIGKILLs 7B-wide tensors; CPU SIMD beats the
serialized GPU ~8× anyway) and `nice -n 19` (the box is busy). Outputs: long CSV
(tensor × config: n, bits, bpw, rel_rms_pct), wide pivot CSV, aggregate CSV, and the exact
command log. `allocate` runs §3.2 and emits `mp-alloc.json` + `pv-tensors.regex` +
`alloc-summary.json`. The allocator REFUSES to water-fill on `rel_rms_pct`
(within-family-only, §2.1) unless `--force-rel-rms` acknowledges plumbing-test intent.

---

## 7. The experiment ladder (cheapest first, kill criteria attached)

Standing constraints: everything `nice -n 19`; never concurrent with a QAT/PV run on the
18 GB box (freeze trap, will.md §7); 64-ch eval = screening, 146-ch = anchors; every claimed
number re-run serially before it enters will.md (§5.8).

| # | experiment | cost | success gate | kill criterion |
|---|---|---|---|---|
| E0 | `rung-screen.py --dry-run` both modes; command plan eyeballed | minutes, free | plan matches §6 conventions | — (done in this tier) |
| E1 | stage-1 screen: 3 measure-only passes on 0.5B (r2/r3/r4) | ~30–60 min/pass, CPU, niced | per-tensor monotone in bits; aggregates reproduce canon ballpark (q2_l12 ≈ 28%-class weighted rel-RMS) | non-monotone or aggregate off → encoder/script bug, fix before proceeding |
| E2 | stage-2 swap-KL screen: 3 recon writes + `rung-kl.py` (to be built) | ~1–2 h total | (a) down_proj dominates r3 damage (the mp_light fact, §2.4) | (a) fails → screen invalid, STOP Tier 2a |
| E2b | redundancy check | free (same data) | — | Spearman(rel-RMS, dkl) > 0.9 within every rung → stage 2 adds nothing; drop it, allocate on within-CLASS rel-RMS + class-level damage only |
| E2c | additivity calibration: predict q3-uniform, mp-2bit-down3, one alloc config; fit α | 3 × (quant + 64-ch eval) ≈ 3 × ~25 min | predictions within 2× in the ≤ 0.3-nat regime | off > 2× → per-tensor allocation downgraded to class-level (down/gate-up/attn), thresholds re-fit |
| E3 | allocator validation: 2–3 λ-sweep configs at ~2.9–3.3 bpw, real 64-ch PPL | each ≈ quant 15 min + eval | beat mp-2bit-down3 (2.95 bpw, finally measured here) AND the q2↔q3 interpolation line at iso-bpw | cannot beat the hand class-split → per-tensor allocator DEAD; screen survives for PV selection only |
| E4 | selective-PV pilot, 0.5B, 300 steps, alloc config from E3: arms (a) no-PV, (b) full-PV, (c) selective-PV RED@1e-4 | (b) is the priciest arm (~hours, the pv re-pass protocol); (c) ≈ half | (c) ≥ (b) quality at ≤ 0.6× wall-clock AND (c) > (a) | (c) damages vs (a) — the pv3bit failure mode survives per-tensor freezing → per-tensor PV DEAD; PV stays whole-model-at-collapsed-rungs (sub-2-bit) only |
| E4d | (optional rider) train ALL biases at 3e-5 with weights per (c) | +ε | free PPL (the de-biasing-adjacent lever, §4 queue #3) | no movement → drop |
| E5 | 7B transfer on the pod: screen (cuda swap-KL is fast) → allocate → confirm | pod-hours | beat mp_light 8.45 @ 3.67 bpw at iso-bpw, or match at lower bpw | no win at 7B → bank the 0.5B method, do not block the marathon on it |

Decision dependencies: E2 gates E3 gates E4 gates E5. E2b/E2c can only *simplify* the
pipeline, never block it. The 2-bit re-pass verdict (in flight) slots into §2.3's τ_r and
§4.1's LR table whenever it lands — it does not block E1/E2.

---

## 8. Open questions (honest)

1. **Does the inverse law hold per-tensor?** The entire selective-PV bet. Tonight's evidence
   is model-level at uniform rungs. E4 decides; a clean negative is recorded as a §4 DEAD row
   ("per-tensor PV freezing") and whole-model PV at collapsed rungs remains the tool.
2. **Additivity of per-tensor damage.** First-order Σδ is standard but unproven here; the RHT
   decorrelates *weights*, not functional interactions between tensors. E2c bounds it; α is
   the patch. If interactions dominate even at small δ, the allocator falls back to class
   granularity (which mp_light already proved useful).
3. **Outlier channel is invocation-global.** Mixed rungs give r4 tensors an outlier channel
   canon q4_l12 lacks (+0.32 bpw, billed honestly). Proper fix = per-rule `outlier_pct` in
   MpRule — a ~20-line Rust patch + sidecar field, deferred until the allocator proves itself.
4. **Screen variance floor.** 2 chunks may be too few for AMBER-band decisions even paired;
   the fix is 4 chunks at 2× cost (the task's "4-chunk eval" option). E2 prints the
   across-chunk spread per tensor; if the GREEN/AMBER boundary swims, double the chunks.
5. **Is the 2-bit RED set small?** Selective PV only pays if collapse is concentrated. If the
   r2 screen shows diffuse damage (every tensor moderately hurt, none catastrophic), the PV
   set is "all of them" and selectivity buys nothing at 2-bit — it would still pay at the
   1.5-bit rung (where PTQ collapse is total but possibly still tensor-skewed) and for the
   3-bit "protect the converged" direction. The screen answers this before any PV run.
6. **AMBER LR (3e-5) is extrapolated** from model-level evidence (ternary-3k warm-restart,
   pv3bit damage). No per-tensor measurement exists. P6 keeps it out of v1.
7. **vec-d2 (1.5-bit) as an allocator rung** couples allocation to PV (the rung is unusable
   without re-learning: PTQ 50,134). Phase 2: allow r15 only for tensors already in S, billing
   2.165 bpw. Needs E4 positive first.
8. **Bias handling.** STRAND quantizes weights only; biases ride frozen/trained with their
   tensor in v1. Whether bias-only training of FROZEN tensors is free quality (it is
   de-biasing-shaped, queue #3) is E4d's question.
9. **Eval-protocol drift at 7B.** The screen's 2-chunk KL at 7B runs on cuda (pod). The
   bf16 reference-logit cache is the same size class as 0.5B (vocab matches: 2 × 2047 ×
   152,064 × 2 B ≈ 1.25 GB) — fine on a 3090. It becomes 5 GB-class only at the 4-chunk
   fp32 fallback (§8.4); chunked or top-k KL covers that case. Decide when E5 is reached,
   not before.

---

## 9. File map

| path | status | role |
|---|---|---|
| `docs/STRAND-rung-allocator-design.md` | this file | the design + patch plan + ladder |
| `scripts/rung-screen.py` | built (dry-run verified) | stage-1 screen + allocator math |
| `rung-kl.py` | NOT built (E2) | stage-2 swap-KL harness, separate tier |
| `scripts/strand-qat.py` | run-frozen, patch plan §4.4 | selective PV lands here post-marathon |
| `configs/mp-2bit-down3.json` | exists, unmeasured | E3's baseline to beat |
