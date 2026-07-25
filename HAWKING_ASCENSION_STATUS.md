# Hawking Ascension status

```
endpoint:  FUNCTIONAL_GRAVITY_ACTIVE
as of:     2026-07-23
branch:    campaign/glm52-generation-b
```

## Metric

The promotion metric is replaced and enforced globally.
`HAWKING_NULL_CORRECTED_METRIC_CONTRACT.md` / `.json`, implemented in
`tools/condense/hawking_null_metric.py`, whose selftest is the executable form of the
contract: fit-split nulls, centered cosine, signed null-relative skill, a 512-resample
bootstrap lower bound, and a gate that raw cosine cannot override.

Two instrument defects were found and superseded, not rewritten: raw cosine promoting on a
mean-dominated target, and a first correction pass that centred by the held-out teacher
mean. All 22 sealed pilot artifacts were re-scored from their sealed relative errors —
exact, no re-decode — into `GLM52_METRIC_CORRECTION_LEDGER.jsonl`.

## Weight-space verdict, corrected

**Zero of 22 artifacts beat a constant.** Corrected block-output skill runs -1.21 to -5.38,
including -1.77 at 2.0169 BPW, twice the legal ceiling. Every family closed:
product quantization, per-tensor low rank, shared expert basis, by-role hybrid, and the
pilot's own "functional block" weight family. The negative result is stronger than it was
reported, not weaker.

## Functional candidate

`glm52.functional.moe.v1` — pre-router hidden state to MoE output, seeded feature map, only
the readout costs bits.

```
local exact rate      0.010375 BPW    (0.000648 for the rank-64 row)
null-relative skill   0.5318          lower bound 0.5232
centered score        0.7309          against a 0.826 constant-mean raw-cosine null
block result          skill 0.8806    against 0.1968 with the MoE removed
propagation result    skill 0.7589 at one layer, but error amplifies 1.71x
layers tested         5, 38, 74, 77 fitted independently; all pass, skill 0.532 to 0.698
replication           24 of 24 across 6 unfitted splits and 4 seeds, no refit
```

## Why it is held

The GLM-5.2 residual stream is expansive at **every** tested magnitude, and worse at small
ones: 2.41× amplification at a 0.77 percent injected error against 1.42× at the student's
own 15.4 percent. No contractive regime exists. With students in four consecutive layers,
router top-1 agreement falls 0.850 → 0.651 → 0.446, and one functional layer out of 75
already changes the greedy token at 45 percent of positions.

```
decision:  FUNCTIONAL_PARTIAL_ONLY
GLM full stream:  DO NOT STREAM   (1,506,659,919,872 bytes not spent)
```

## Complete-model rate

Exact, over 753,329,940,480 header-derived logical weights. 97.89 percent is functionally
replaceable; the protected 2.11 percent costs **0.337745 BPW at source precision on its
own**, above the one-third rung before any expert byte is stored.

| candidate | organ-local | complete @BF16 | @int8 | @int4 |
|---|---:|---:|---:|---:|
| linear_rank64 | 0.000648 | 0.338379 | 0.169507 | 0.085071 |
| student_h1024 | 0.010375 | 0.347900 | 0.179028 | 0.094592 |
| linear_full | 0.062247 | 0.398677 | 0.229805 | 0.145369 |

Rate was never the obstacle.

## Runtime

`.gravity` codec, deterministic CPU authority, three parity-gated Metal grammars, and a
verified end-to-end chain: shard written and hash-verified → payload read back → CPU
authority on real captured states → Metal parity → MoE substituted in the real block →
residual carried into layer 39 through real attention and a real router → greedy token.

| grammar | resident/layer | measured | vs teacher traffic | parity |
|---|---:|---:|---:|---:|
| **FRT-B procedural** | **12.58 MB** | **2.15 ms** | **48×** | 7.9e-7 |
| FRT-A explicit | 25.17 MB | 2.90 ms | 24× | 2.0e-4 |
| FRT-D direct linear | 75.50 MB | 2.52 ms | 8× | 0.0 |

Procedural in-kernel generation wins on both bytes and time. That is the payoff for
freezing a stateless generator instead of a library RNG.

## The binding limit moved

Measured bandwidth on this M3 Ultra: **417.7 GB/s**, through the same kernel at real
occupancy, not a specification figure.

```
before:  expert weight traffic     51.64 GB/token   63 percent of traffic
after:   attention and protected   29.9  GB/token   the functional payload is 0.96 GB, 3.1 percent
```

Bandwidth-bound ceiling: 5.1 TPS teacher → 13.5 TPS functional at BF16 protected → **49.5
TPS** at int4 protected. Measured kernels reach only 1.4 to 7.2 percent of their own
roofline at 76 command buffers per token, so submission collapse and occupancy are now the
runtime work — not expert compression.

## Lanes

```
heavy      detached controller, pid live, lease + heartbeat + progress ledger
           phase 1: 42 of 42 teacher capsules captured (layers 3,5,11,38,39,40,41,74,75,76,77)
           phase 2: amplification and replication probes on the late and early strata
light      metric correction, controls, byte auction, codec, CPU authority,
           Metal grammars, roofline, HIDE contract, next-parent protocol
storage    129.3 GB free (120.5 GiB), floor 60 GiB, no goliath fetched
           the closed parent's 405.4 GB BF16 body is still resident; eviction is
           legal (every probed layer has a sealed capsule) but was not authorised
```

## Next

1. Finish the phase-2 amplification probes on the early and late strata: is GLM-5.2
   expansive everywhere, or only around layer 38?
2. Runtime: collapse 76 command buffers per token toward one replayable graph, and raise
   occupancy — the kernels are at 1.4 percent of roofline.
3. Next parent: amplification probe **first**, per `HAWKING_GRAVITY_CROSS_MODEL_TRANSFER.md`.
   Three sparse strata and their successors, capture, probe, decide. No full stream before
   the probe reports.

## Honest scope

Every score is on bounded calibration batches — 12,288 fit positions and 4,096 disjoint
score positions from a sealed corpus. The token-level number is a 1024-row logit-lens
probe, not the model's head. Nothing here is capability, full-model quality, or behaviour.
