# Lane A — Bit Ledger / Entropy Microscope: results

_Measured 2026-06-12 on `scratch/qwen-05b/model.safetensors` (Qwen2.5-0.5B, 168
quantizable linear tensors, 357,826,560 quantized weights, 1,397,760 blocks of
256). CPU encode, `STRAND_NO_GPU=1`, deterministic. Tool:
`crates/strand-quant/src/bin/bit-ledger.rs`. Raw CSV + per-class entropy:
`research/bit-ledger-q2.csv`, `research/bit-ledger-q3.csv`._

This is the measurement the compression map (§3.3, §5.1, Lane A) gated C2 on. It
turns "C2 recovers 0.11–0.20 bpw, MEDIUM confidence" into a measured number.

> **Method.** Each tensor is re-encoded through the production path
> (`encode_tensor_with`, RHT on, 1% pre-RHT outlier channel @ 8-bit, exactly
> mirroring `quantize-model::quantize_one`). Raw bits are decomposed per component
> from the real `BlockMeta`/format geometry. Empirical order-0 Shannon entropy is
> computed for each side-info stream, plus simple predictors (prev-block scale
> delta; sub-scale context-by-position and super-scale residual; outlier-position
> gap coding). **Recoverable bpw = raw_bits/weight − entropy_bits/weight**, i.e. the
> ceiling an ideal arithmetic/rANS coder built on that model could reach. C2 will
> not beat the entropy floor, so this is an upper bound on the C2 win.

## Ledger validation

The whole-model encoded totals reproduce the canon sidecar bpw **exactly**, which
confirms the decomposition is faithful:

| rung | ledger TOTAL encoded | map §5.1 canon | match |
|---|---:|---:|---|
| q2_l12_out1 | 2.66530 | 2.6653 | exact |
| q3_l12_out1 | 3.66530 | 3.6653 | exact |

## Raw bit spend (bpw, whole model)

| component | q2 | q3 | notes |
|---|---:|---:|---|
| payload | 2.00000 | 3.00000 | k bits/weight |
| scale (32/block) | 0.12500 | 0.12500 | |
| sub_scale (8×6/block) | 0.18750 | 0.18750 | |
| init_state (L=12/block) | 0.04688 | 0.04688 | tail-biting off at q2/q3 |
| affine_min | 0.00000 | 0.00000 | off below 4-bit |
| rht_seed (64/tensor) | 0.00003 | 0.00003 | |
| header (16/tensor) | 0.00001 | 0.00001 | |
| outl_pos | 0.22584 | 0.22584 | idx_bits × 1% entries |
| outl_val | 0.08000 | 0.08000 | 8 bits × 1% entries |
| outl_hdr | 0.00005 | 0.00005 | |
| **v2 random-access table (deploy-only)** | 0.50000 | 0.50000 | 16 B/block |
| **TOTAL encoded** | **2.66530** | **3.66530** | encoded `.strand` artifact |
| **TOTAL deploy** | **3.16530** | **4.16530** | + v2 seek table |

Note the side-info is a large share of the **encoded** artifact: at q2, scale +
sub_scale + init = 0.359 bpw and the outlier channel another 0.306 bpw — together
**25%** of the 2.665 bpw encoded total sits outside the payload.

## Entropy microscope — recoverable bpw per component (whole model)

### q2 (k=2, L=12)

| component | raw bpw | entropy bpw | **recoverable bpw** | H (bits/sym) | best predictor |
|---|---:|---:|---:|---:|---|
| scale | 0.12500 | 0.04099 | **0.08401** | 10.49 / 32 | prev-block delta (barely; H0 10.55) |
| sub_scale | 0.18750 | 0.16554 | **0.02196** | 5.30 / 6 | ctx-by-position ≈ order-0 |
| init_state | 0.04688 | 0.03899 | **0.00788** | 9.98 / 12 | — |
| outl_pos | 0.22584 | 0.07825 | **0.14760** | 7.82 / 17–23 | gap (H_abs 21.0 → H_gap 7.82) |
| outl_val | 0.08000 | 0.05963 | **0.02037** | 5.96 / 8 | order-0 |

### q3 (k=3, L=12)

| component | raw bpw | entropy bpw | **recoverable bpw** | H (bits/sym) | best predictor |
|---|---:|---:|---:|---:|---|
| scale | 0.12500 | 0.04166 | **0.08334** | 10.67 / 32 | prev-block delta (barely) |
| sub_scale | 0.18750 | 0.15535 | **0.03215** | 4.97 / 6 | ctx-by-position ≈ order-0 |
| init_state | 0.04688 | 0.03511 | **0.01177** | 8.99 / 12 | — |
| outl_pos | 0.22584 | 0.07825 | **0.14760** | 7.82 / 17–23 | gap |
| outl_val | 0.08000 | 0.05963 | **0.02037** | 5.96 / 8 | order-0 |

### Findings on the streams

- **scale** is the biggest C2 lever and it is *not* a prediction win. The `scale_q`
  i32 field is billed at a fixed 32 bits but carries only ~10.5 bits of real
  entropy; the prev-block delta predictor barely moves it (10.55 → 10.49 at q2),
  consistent with RHT having whitened block-to-block structure. The recovery
  (~0.084 bpw) is almost entirely "stop billing 32 bits for a ~10.5-bit symbol."
- **sub_scale** entropy is ~5.3/6 bits at q2, dropping to ~4.97/6 at q3 (deeper
  payload spreads sub-scale codes more). Context-by-position gives essentially no
  gain over pooled order-0 (the 8 positions are statistically interchangeable
  post-RHT), and the super-scale residual is *worse*. So a single static order-0
  CDF captures the whole win; per-position CDFs are not worth their table cost.
- **outlier positions** are the single largest recoverable component (0.1476 bpw).
  The absolute index is ~uniform (H_abs ≈ idx_bits, i.e. incompressible), but the
  sorted-index **gap** distribution is low-entropy (~7.8 bits), because 1%-by-|w|
  outliers cluster — exactly the "encode positions by local gap" hypothesis in map
  §3.9. This is independent of rung (same gt selection at q2 and q3).
- **outlier values** recover ~0.02 bpw (8-bit residual codes carry ~6 bits); modest.
  ffn_down values are most compressible (5.15 bits), attn least (6.98).
- **init_state** is small (≤0.012 bpw) and near-incompressible per symbol
  (H 9.0–10.0 of 12). The real init-state win is *structural* (tail-biting removes
  it entirely), not entropy coding.

## Measured vs map §5.1 estimates — CONFIRM / CORRECT

| lever | map §5.1 estimate | **measured (q2 / q3)** | verdict |
|---|---|---:|---|
| C2 scale + sub_scale coding | 0.11–0.20 bpw | **0.106 / 0.115** | **CORRECT DOWN** — real win sits at/just below the bottom of the estimate, not the middle |
| init-state model | 0.01–0.05 bpw | **0.0079 / 0.0118** | **CORRECT DOWN** — below the range at q2, bottom edge at q3; entropy coding is the wrong tool (use tail-biting) |
| OUTL position + value coding | 0.05–0.20 bpw | **0.168 / 0.168** (pos 0.1476 + val 0.0204) | **CONFIRM (top of range)** — driven almost entirely by gap-coding the positions |
| C2 + stream-mode table drop | (seek-archive lever) | **0.606 / 0.615** B/w of *deploy* traffic | scale+sub 0.106–0.115 + full v2 table 0.500 |

If C2 (scale+sub) and OUTL coding both land at their measured ceilings:

- **q2 encoded** 2.6653 → ~**2.49** bpw (−0.106 C2 −0.168 OUTL). Map's near-term
  realistic target was 2.45–2.55 — **measured supports the conservative end.**
- **q3 encoded** 3.6653 → ~**3.50** bpw.

(These are entropy *ceilings*; a real integer rANS coder plus its CDF tables will
land a few % short, and OUTL gains are quality-gated per map §3.9.)

## GATE VERDICT (map §3.3)

> _"implement C2 only if real tensors show at least **0.01 B/w** recoverable from
> scale/sub-scale coding, or at least **0.04 B/w** recoverable including stream-mode
> table changes."_

| gate clause | q2 | q3 | passes? |
|---|---:|---:|---|
| scale + sub_scale recoverable ≥ 0.01 B/w | **0.10598** | **0.11549** | **YES (10–11×)** |
| incl. stream-mode v2-table drop ≥ 0.04 B/w | **0.60598** | **0.61549** | **YES (15×)** |

### ✅ C2 CLEARS ITS GATE on both rungs, with margin.

- The scale/sub-scale clause passes by ~10× the threshold purely on the encoded
  artifact (**0.106 bpw at q2, 0.115 bpw at q3**), driven by the over-wide 32-bit
  scale field carrying ~10.5 bits of entropy.
- The stream-table clause passes by ~15× because the v2 random-access table is a
  flat 0.5 bpw that stream-mode deployment can drop wholesale.

### Recommended C2 scope (cheapest path to the measured win)

1. **Static order-0 CDF for `scale_q`** — the dominant, prediction-resistant win.
   Skip the prev-block predictor; it adds state for ~0.0007 bpw.
2. **Static order-0 CDF for sub-scale codes** — one pooled CDF; skip per-position
   and super-scale variants (no measured gain).
3. **Gap-code outlier positions** (map §3.9) — the largest single component
   (0.1476 bpw), but route it through Lane G / OUTL since it is quality-gated, not
   pure metadata cleanup.
4. **init-state: do NOT entropy-code** — pursue tail-biting (structural removal)
   instead; entropy recovery is ≤0.012 bpw and the symbol is near-incompressible.
5. **stream/seek container split** (separate lever, map rank 2) captures the 0.5
   bpw v2-table cost for streaming deployments — by far the largest deploy-traffic
   lever, independent of C2.

## Reproduce

```
STRAND_NO_GPU=1 cargo run --release -p strand-quant --bin bit-ledger -- \
  --in scratch/qwen-05b/model.safetensors --bits 2 --l 12 --outlier-channel 1 \
  --csv research/bit-ledger-q2.csv --md research/bit-ledger-q2.md
# and --bits 3 for the q3 rung.
```

Per-tensor and per-class rows (attn / ffn_up_gate / ffn_down) are in the CSVs.
Class spread is small: scale recovery 0.083–0.085 across classes; sub_scale
0.020–0.022 (q2) / 0.030–0.033 (q3); outl_pos 0.125 (attn, narrower tensors →
fewer idx bits) to 0.152 (ffn). No class flips the gate.

---
_machine-stamp: Darwin 25.5.0 (arm64, M-class), Qwen2.5-0.5B, STRAND_NO_GPU=1
CPU encode, q2 run 300.3 s / q3 run 483.7 s, 168 tensors, 357,826,560 weights,
git branch media-waves._
