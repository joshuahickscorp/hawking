# GLM-5.2 functional decision

```
decision:  FUNCTIONAL_PARTIAL_ONLY
GLM full stream:  DO NOT STREAM
source not streamed:  1,506,659,919,872 bytes
next:  carry the corrected methodology to the next architecture parent
```

## What was found

A dense student maps a sparse layer's pre-router hidden state straight to its MoE output.
The feature map is a seeded projection, so only the readout costs bits: **0.010375 bits per
replaced weight**, about a seventieth of the rate every weight-space family was given, and
the only thing in this campaign that beats a constant.

It survives everything except depth.

| gate | result |
|---|---|
| FS0 reproduction, 4 fresh seeds | **PASS** — skill 0.532, lower bound 0.523, seed spread 0.0025 |
| FS1 block insertion | **PASS** — block skill 0.881 against a 0.197 no-MoE control |
| FS2 next-layer propagation | **FAIL** — single-step skill 0.759, but error amplifies 1.71× |
| FS3 early / middle / late / final strata | **PASS** — 0.698, 0.532, 0.592, 0.586 |
| FS4 cross-layer sharing | **FAIL** — every transfer negative; readout is layer-specific |
| FS5 replication, 6 unfitted splits × 4 seeds | **PASS** — 24 of 24, skill 0.481–0.532 |
| FS6 complete-model auction | **PASS** — 0.348 BPW at BF16 protected, 0.179 at int8 |
| FS7 full-stream admission | **FAIL** — blocked by FS2 |

## What kills it

The GLM-5.2 residual stream is an **expansive** operator. Scaling the student's own error
direction down does not find a stable regime — it makes the amplification worse:

| injected relative L2 | amplification per layer | layer-39 router top-1 |
|---:|---:|---:|
| 0.0077 | **2.41×** | 0.987 |
| 0.0154 | **2.03×** | 0.969 |
| 0.0384 | **1.69×** | 0.929 |
| 0.0769 | **1.53×** | 0.904 |
| 0.1537 (the student's own error) | **1.42×** | 0.850 |

There is no magnitude at which this stack contracts. With students in four consecutive
layers, relative error runs 0.154 → 0.285 → 0.361 → 0.534 and router top-1 agreement falls
0.850 → 0.651 → **0.446**. Four layers into a 75-layer stack, the router disagrees with the
teacher more often than it agrees.

The end-to-end integration makes it concrete: **one functional layer out of 75 already
changes the greedy token at 45 percent of positions.**

## Why PARTIAL_ONLY and not REFUTED

The contract reserves `FUNCTIONAL_ESCAPE_REFUTED` for a student that fails a null-corrected
block, propagation, or cross-layer replication gate. This one passes the block gate
decisively, replicates on all four sparse strata, and replicates on six splits it was never
fitted on — including the protected-domain and long-context holdouts — under four
independent seeds with no refit. Its single-step propagation score is positive too.

What fails is composition over depth. That is the definition of `FUNCTIONAL_PARTIAL_ONLY`:
the result replicates locally, and propagation fails.

## Why not stream

Streaming the remaining 1.51 TB would buy 76 per-layer students whose error grows 1.42× to
2.41× per layer. The rate was never the obstacle — the auction says a free student still
cannot reach one-third BPW unless the protected 2.11 percent is also compressed, and every
candidate is comfortably legal once it is. The obstacle is the architecture.

## What is retained

```
the null-corrected metric contract, now global and enforced
the functional-versus-weight-space causal law
glm52.functional.moe.v1, with a generator portable to any language
the deterministic CPU authority and three parity-gated Metal grammars
the amplification probe, promoted to a mandatory next-parent gate
teacher capsules for layers 3, 5, 11, 38, 39, 40, 41, 74, 75, 76, 77
```

The amplification probe is the transferable result. It costs two layer forwards and a
magnitude sweep, it decides the paradigm before any encoding work, and running it on
GLM-5.2 first would have saved the entire weight-space ladder.
