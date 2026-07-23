# Cross-model transfer: the next-parent protocol is now amplification-first

Two campaigns have now spent their source budget answering the wrong question first.
Qwen3-235B ended sealed-negative on weight-space encoding. GLM-5.2 closed five weight-space
families, found a functional escape at a seventieth of the rate, and then lost it to depth.

Both outcomes were decidable in a day, before a byte of encoding work.

## The order changed

The old order was: fetch, verify, encode at a rate ladder, measure fidelity, promote.
The GLM correction made it function-first. This campaign makes it **amplification-first**.

```
1. build correct source and adapter truth
2. capture teacher evidence for three sparse strata and their immediate successors
3. AMPLIFICATION PROBE                    <- new, and first
4. establish fit-split nulls
5. dense/full affine functional upper control
6. tiny functional student
7. weight-space control
8. compare null-corrected skill and exact physical cost
9. rate ladder, only for the surviving paradigm
```

## The amplification probe

Two layer forwards and a magnitude sweep. It costs minutes and it decides the paradigm.

Inject a perturbation at layer *n*, carry it through **teacher** layers *n+1* onward, and
measure how the relative L2 of the deviation evolves. Sweep the magnitude down from the
error a real student would make. Implementation: `glm52_functional_cascade.py threshold`.

Read it like this:

| observation | meaning |
|---|---|
| amplification **< 1** at student magnitudes | the stack absorbs error; a per-layer functional student can compose. Proceed. |
| amplification **> 1** but **< 1** below some magnitude | there is a stability threshold; the student must get under it or the depth must be shortened. Proceed with a budget. |
| amplification **> 1** at every magnitude | no per-layer lossy replacement composes. Do not stream. This is GLM-5.2. |

GLM-5.2 amplifies 2.41× at a 0.77 percent injected error and 1.42× at 15.4 percent — worse
where it should be safest. Running this probe on GLM-5.2 first would have saved the entire
weight-space ladder and the functional ladder both.

Report the router agreement alongside it. On a MoE parent the router is the sharpest
instrument available: layer-39 top-1 agreement fell to 0.850 at the student's own error and
to 0.446 after four student layers, while the L2 numbers still looked survivable.

## What transfers unchanged

```
HAWKING_NULL_CORRECTED_METRIC_CONTRACT      fit-split nulls, signed skill, bootstrap bound
hawking_null_metric.py                      parent-agnostic, no GLM in it
glm52.functional.moe.v1                     rename the id; the codec is architecture-neutral
gen.splitmix64_boxmuller.v1                 stateless, portable, in-kernel generable
the three Metal grammars                    FRT-B wins on bytes and time; expect that again
the complete-model byte auction             organ-local rate is never the model rate
```

## What does not transfer

The GLM verdict itself. `EXPANSIVE_AT_EVERY_TESTED_MAGNITUDE` is a measurement of GLM-5.2's
residual stream between layers 38 and 41, not a law about transformers. Architectures with
different normalisation placement, different residual scaling, or shallower effective depth
may well contract. **Measure it, do not assume it either way.**

## Retired globally

Do not rerun unchanged, on any parent:

```
raw-cosine promotion
any mean-dominated activation metric
centering by a held-out target statistic
a weight-space rate ladder before null controls
activation-weighted SVD (recorded Type-1 kill)
shared expert-weight basis without architecture evidence
a full source stream before the amplification probe
```

## Reopen conditions

A weight-space method reopens only when the architecture is materially different, a
null-corrected oracle predicts recoverability, **and** the amplification probe shows a
regime where the resulting error survives depth. All three, not any one.

## Next parent

The queue is unchanged; the entry gate is not. No parent gets a full stream until the
function-versus-weight pilot and the amplification probe have both reported. Under the
storage policy that means: fetch three sparse strata and their successors, capture, probe,
decide. A goliath body is not downloaded to answer a question three layers can answer.

The GLM-5.2 teacher capsules for layers 3, 5, 11, 38, 39, 40, 41, 74, 75, 76 and 77 stay
resident as the reference fixture: any new probe implementation is checked against them
before it is trusted on a parent nobody has measured yet.
