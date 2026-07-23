# GLM-5.2 corrected scientific law

Supersedes every verdict in the Generation B pilot that was reached on raw activation
cosine. Sealed under `HAWKING_NULL_CORRECTED_METRIC_CONTRACT.md`. Evidence:
`GLM52_METRIC_CORRECTION_LEDGER.jsonl`, `GLM52_FUNCTIONAL_EXPERIMENT_LEDGER.jsonl`, and the
`GLM52_FUNCTIONAL_*` artifacts under `reports/condense/glm52_generation_b/`.

## 1. The instrument was wrong twice

Raw activation cosine promoted candidates on a mean-dominated target. Predicting a single
fit-split vector scores 0.826 to 0.896 raw cosine on these stages.

The first correction pass in this campaign centred by the **teacher's own mean** — a
statistic of the held-out target, which the contract forbids because it hands the candidate
information the null did not have. Those centred numbers are superseded. Every number below
uses a mean fitted on a split the score did not come from.

## 2. Weight-space on GLM-5.2 is closed, and harder than reported

All 22 sealed pilot artifacts re-scored from their sealed relative errors. Corrected
null-relative skill on `block_output`, where 0 is "no better than a constant":

| window | rung | complete BPW | corrected block skill |
|---|---|---:|---:|
| W_L38_L38 | DX2 | 2.0169 | **-1.765** |
| W_L00_L00 | G0 | 0.8551 | **-1.214** |
| W_L38_L38 | LR0 | 0.7565 | **-3.988** |
| W_L38_L38 | HY0 | 0.7531 | **-4.393** |
| W_L38_L38 | G0 | 0.7531 | **-4.414** |
| W_L38_L38 | G2 | 0.3306 | **-4.492** |
| W_L74_L74 | GC | 0.8931 | **-5.379** |

Not one artifact beats a constant. The best is worse than the constant by more than 2×
the null's own error. Every family is closed:

```
routed-expert product quantization        CLOSED_NEGATIVE
per-tensor low-rank weight blueprint      CLOSED_NEGATIVE (cross-parent dead lever)
shared expert-weight basis                CLOSED_NEGATIVE
by-role hybrid weight blueprint           CLOSED_NEGATIVE
pilot "functional block" weight family    CLOSED_NEGATIVE (-4.54 at 0.7512 BPW)
```

Even at 2.0169 BPW — twice the legal ceiling — the family is negative. The negative result
is stronger than previously reported, not weaker.

## 3. The functional escape is real and it replicates

A dense student maps the pre-router hidden state to the MoE output. The feature map is a
seeded projection, so only the readout costs bits.

**FS0, reproduction.** Fresh process, four fresh seeds, disjoint score split:

```
student h1024   0.010375 local BPW   skill 0.532 (lower 0.523)  centered 0.731   PASS
linear full     0.062247 local BPW   skill 0.544 (lower 0.535)  centered 0.739   PASS
linear rank256  0.002594 local BPW   skill 0.505 (lower 0.498)  centered 0.711   PASS
linear rank64   0.000648 local BPW   skill 0.445 (lower 0.437)  centered 0.669   PASS
weight-space    0.753    local BPW   skill negative                              FAIL
```

Seed spread across four seeds: 0.0025 in skill. Controls behave: the mean-null is exactly
0, shuffled inputs score -0.591, identity scores -24.2.

The full linear control reaching 0.544 against the student's 0.532 says the map is the win,
not the nonlinearity. The nonlinearity buys compactness: 6× fewer bytes for 2% less skill.

**FS1, block insertion.** `block_output = post_attention_hidden + post_moe` exactly, so the
student's block output is measurable directly:

```
student in the block      skill 0.881 (lower 0.879)   centered 0.938
MoE replaced by constant  skill 0.745
MoE removed entirely      skill 0.197
```

**FS3, layer strata.** Fitted and scored independently on each stratum, no transfer:

```
L05  early   skill 0.698    L38  middle  skill 0.532
L74  late    skill 0.592    L77  final   skill 0.586
```

**FS5, replication without refit.** 4 student seeds × 6 splits never fitted on — including
the protected-domain and long-context holdouts — 24 of 24 pass, skill 0.481 to 0.532.

## 4. Each layer's MoE is its own function

Applying a layer's fitted readout to another layer's inputs collapses:

```
fitted L05 -> scored L38    skill -2.19
fitted L38 -> scored L05    skill -43.7
fitted L77 -> scored L05    skill -113.6
```

No cross-layer transfer at any pairing. `READOUT_IS_LAYER_SPECIFIC`. Sharing saves nothing:
the feature map is already a shared seed, and the readout is the only real byte.

## 5. What kills it: the residual stream is expansive

A perturbation injected at layer 38 and carried through **teacher** layers 39, 40, 41:

```
per-layer relative-L2 amplification   1.706, 1.178, 1.218
geometric mean                        1.348
projected over 75 sparse layers       5.3e9
```

The magnitude sweep is the decisive measurement. Scaling the student's own error direction
down does not find a stable regime — it makes the amplification **worse**:

| injected relative L2 | geometric mean amplification | layer-39 router top-1 agreement |
|---:|---:|---:|
| 0.0077 | **2.407** | 0.987 |
| 0.0154 | **2.032** | 0.969 |
| 0.0384 | **1.691** | 0.929 |
| 0.0769 | **1.527** | 0.904 |
| 0.1537 (the student's own) | **1.418** | 0.850 |

There is no magnitude at which this stack is contractive. A functional student in every
layer compounds it:

```
students in layers 38, 39, 40, 41
relative L2   0.154 -> 0.285 -> 0.361 -> 0.534   (growth 3.47x)
router top-1  0.850 -> 0.651 -> 0.446
top-8 overlap 6.74  -> 6.12  -> 4.47   of 8
```

Four student layers into a 75-layer stack, the router picks a different top expert more
often than not. This is what closes the escape. It is not a rate problem, not a fit
problem, and not a metric problem.

## 6. The rate was never the obstacle

Exact complete-model arithmetic over 753,329,940,480 header-derived logical weights:

```
functionally replaced   737,427,868,672   97.89 percent
protected residue        15,902,071,808    2.11 percent
```

| candidate | organ-local BPW | complete BPW, BF16 protected | int8 protected | int4 protected |
|---|---:|---:|---:|---:|
| linear_rank64 | 0.000648 | 0.338379 | 0.169507 | 0.085071 |
| linear_rank256 | 0.002594 | 0.340284 | 0.171411 | 0.086975 |
| student_h1024 | 0.010375 | 0.347900 | 0.179028 | 0.094592 |
| linear_full | 0.062247 | 0.398677 | 0.229805 | 0.145369 |

The protected 2.11 percent costs **0.337745 BPW at source precision on its own**, which is
above the one-third rung before a single expert byte is stored. An organ-local rate of
0.0104 is not a 0.0104-bit model, and this is the arithmetic that says so.

## 7. Law

```
weight-space routed-expert representations      CLOSED_NEGATIVE
shared expert-weight basis                      CLOSED_NEGATIVE
per-tensor low-rank weight blueprint            CLOSED_NEGATIVE
by-role hybrid weight blueprint                 CLOSED_NEGATIVE
functional hidden-state-to-MoE-output mapping   REPLICATES_LOCALLY, FAILS_UNDER_DEPTH
residual stream, layers 38 to 41                EXPANSIVE_AT_EVERY_TESTED_MAGNITUDE
cross-layer functional sharing                  CLOSED_NEGATIVE
complete-model rate                             LEGAL, AND NOT THE BINDING CONSTRAINT
```

The causal statement that carries forward:

> On GLM-5.2, a cheap function of the hidden state predicts a layer's MoE output far better
> than any representation of that layer's expert weights, and does so at a seventieth of the
> rate. It still does not produce a model, because this architecture's residual stream
> amplifies every perturbation it is given, and amplifies small ones hardest.

The open question this hands the next parent is no longer "how tightly can the weights be
encoded" and no longer "can a function replace an organ". It is: **what is the per-layer
amplification spectrum of the target architecture, and does any architecture have a
contractive regime a functional student can live inside?** That is now a one-day probe, and
it belongs before any encoding work.
