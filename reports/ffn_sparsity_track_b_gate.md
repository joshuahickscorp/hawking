# Track B (FFN contextual sparsity) — Step 2 GATE: NO-GO at block-256

**Decided:** 2026-05-30T01:58Z
**Halted on:** Step 2 (measure real sparsity, before building any kernel)
**Verdict:** NO-GO at the handoff's 256-channel block granularity. Halt the
track per the handoff rule ("if <50% of blocks are skippable at high recall,
reset expectations"). Do NOT build the sparse FFN kernel (Steps 4–5).

## What the handoff bet on
`memory/HANDOFF_track_B_ffn_sparsity.md`: q3b decode is bandwidth-bound and FFN
is ~72% of bytes/token. If a cheap per-layer predictor says "only blocks
{i,j,k} of the intermediate (256-channel blocks, 43/layer) matter this token,"
the runtime reads ~35% of the FFN → bytes/token 1.9 GB → ~1.0 GB → ceiling
78 → ~150 tps. The handoff flagged the risk up front: SiLU is *soft*-sparse,
not hard-ReLU-zero, so q3b may be less block-sparse than ReLU models, and Step 2
(measure) is the gate before kernels.

## What ran
- Step-1 capture (`DISMANTLE_QWEN_CAPTURE_FFN_PATH`, committed `4800f45`): taps
  each layer's `ffn_norm` output + per-256-block `max|silu*up|` and
  `||silu*up||_2`. 40 diverse prompts × 20 greedy tokens = **800 decode tokens
  × 36 layers** (`_capture/q3b_ffn.bin`, 235 MB).
- `tools/orchestrator/measure_ffn_sparsity.py --bin` (raw f32, no int8 noise).
  Metric = **oracle** skippable fraction: keep top blocks by true activation
  L2, drop the smallest until the relative L2 error of the dropped FFN output
  exceeds (1 − recall). Oracle is the upper bound; a learned predictor is
  strictly worse.

## The number (oracle = best case)
| recall (max FFN-output L2 error) | oracle KEEP frac | skippable | bytes/token cut |
|---|---|---|---|
| 99.9% | 1.000 | **0.0%** | 0.0% |
| 99%   | 0.998 | **0.2%** | 0.1% |
| 98%   | 0.996 | 0.4% | 0.3% |
| 95%   | 0.988 | 1.2% | 0.9% |
| 90% (already quality-destroying) | 0.958 | **4.2%** | 3.0% |

Handoff projected ~65% FFN-byte cut. Reality at block-256 is 0.1–3%, off by
**20–50×**. NO-GO is not borderline.

## Root cause — granularity mismatch, NOT absence of sparsity
This is the important part for the next session. q3b's FFN **is** strongly
neuron-sparse — the handoff's *premise* is correct — but the sparsity is at the
wrong granularity to skip contiguous weight bytes:

- **Participation ratio** (L2/max)² per 256-block = **5.6 effective active
  channels** (min-layer 2.0, max-layer 9.5). I.e. only **~2.2% of channels**
  are effectively active per token — on par with ReLU Deja-Vu/PowerInfer models.
- But those ~5–6 active channels are **scattered** across each block's 256
  channels, so **every** 256-block contains a few active channels and none is
  droppable. Fraction of blocks with L2 < 1% of row-max: 0.0–3.7%. Smallest
  half of blocks still hold 5–28% of the FFN-output energy.

Sparsity lives at **neuron** granularity (fine, scattered); efficient
byte-skipping needs **block** granularity (coarse, contiguous, Q4_K-aligned).
For q3b these are incompatible without reorganizing the weights.

## What attended work would unblock (NOT autonomous — major redesign)
Inspect `_capture/q3b_ffn.bin` + `reports/ffn_sparsity_gate.json`. Options, in
rough order of yield-vs-effort, all needing an attended call:
1. **PowerInfer-style static hot/cold neuron split.** Test whether some neurons
   are active across *most* tokens (keep hot set dense) while a large cold set
   is rarely active (skip/offload). Needs a **neuron-granularity re-capture**
   (per-neuron activation frequency) — the current capture only stored per-block
   reductions, so this can't be measured from existing data. Cheap to add: a
   `DISMANTLE_QWEN_CAPTURE_FFN_NEURON_FREQ` reduction.
2. **Offline co-activation permutation** to cluster co-firing neurons into
   contiguous blocks so block-skipping becomes viable. Risk: co-activation is
   input-dependent, so a single static permutation likely clusters poorly →
   low yield. Measure co-activation stability before building.
3. **Neuron-level gather/scatter GEMV on Q4_K.** The sparsity is there (98%
   inactive) but Q4_K's 256-value super-block layout makes single-column gather
   non-aligned and random-access — historically not bandwidth-favorable on
   Apple Silicon unified memory. Would need a custom sparse Q4_K layout.

## Disposition
- Track B (block-256 predictor, the handoff's design) → **dead lever**, added to
  `reports/dead_levers.md`.
- The path to >50 tps stays on **Track A (megakernel efficiency)** for now; the
  bytes/token lever via sparsity is not viable at this design point and the
  fine-grained variants above are attended R&D, not a kernel build.
- Step-1 capture infra (`--capture-ffn` + `pack_ffn.py` +
  `measure_ffn_sparsity.py`) is kept — it's the measurement tool any future
  neuron-granularity follow-up reuses.
