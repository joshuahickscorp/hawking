# Hawking null-corrected metric contract

Binding on every parent, every representation, and every promotion decision from
2026-07-23 onward. Implemented by `tools/condense/hawking_null_metric.py`, whose selftest
is the executable form of this document.

## Why this exists

The GLM-5.2 pilot promoted candidates on raw activation cosine. The target activations
carry a large constant component, so predicting a single fit-split vector scores:

| stage | constant-mean raw cosine |
|---|---|
| `block_output` | 0.898 |
| `attention_output` | 0.909 |
| `post_moe` (layer 38) | 0.831 |

Every weight-space score the campaign reported was below its own null. "0.87 fidelity" was
worse than storing one vector. The metric, not the science, produced the number.

## What is frozen

### Nulls are fitted on the fit split only

Permitted nulls, all fitted without touching held-out targets:

- training-mean predictor (the reference null for skill)
- training-affine constant, when distinct from the mean
- identity / residual passthrough
- shuffled-input predictor (the candidate evaluated on mismatched inputs)
- the best trivial structural baseline the representation permits

A null calculated from held-out target statistics is void, and any score derived from one
is void with it. `hawking_null_metric.selftest` asserts this is detectable.

### Primary metrics

For target `y`, prediction `ŷ`, fit-split mean `μ_fit`:

```
raw cosine            cos(y, ŷ)                            DIAGNOSTIC ONLY
centered cosine       cos(y - μ_fit, ŷ - μ_fit)            promotion cosine
null-relative skill   1 - SSE(ŷ) / SSE(μ_fit)              signed, 0 = constant, 1 = exact
skill lower bound     5th percentile, 512-resample bootstrap over held-out positions
relative L2           ‖ŷ - y‖ / ‖y‖
normalized RMSE       RMSE / std(y)
per-position skill    p05, median, p95, fraction of positions beating the null
```

### Promotion gate

A candidate is promoted only when **all** hold:

```
skill_lower       > 0.0        positive with a positive confidence lower bound
centered_cosine   >= 0.5
tail gate         per-position fraction beating null is reported and inspected
domain gate       replication on unfitted documents/domains, no refit
seed gate         replication under fresh student seeds, no refit
```

Raw cosine never overrides a failed null-relative gate. There is no threshold on raw
cosine at which a candidate promotes.

### Controls every candidate must run

```
mean-null
shuffled input
identity / passthrough
full affine or linear upper control
representation-family ablation
seed replication
new-document replication
```

The full linear control is mandatory because it separates "the map is the win" from "the
architecture is the win". On GLM-5.2 layer 38 it answered that question against the
nonlinearity.

## Reissue rule

Prior evidence is corrected, never quietly rewritten:

1. recompute null-corrected metrics from sealed predictions where they exist
2. invalidate only the claims that depended on the broken metric
3. preserve exact payload, byte and accounting evidence unchanged
4. write a superseding receipt naming the artifact and the claim it replaces

Correction ledger: `GLM52_METRIC_CORRECTION_LEDGER.jsonl`.
Superseding law: `GLM52_CORRECTED_SCIENTIFIC_LAW.md` / `.json`.

## Local rate is not model rate

A rate measured over the weights one organ replaces is an organ-local rate. Complete-model
BPW bills, separately and exactly:

```
seeds, generator identity and version
generator parameters actually stored
readouts, biases, normalisation
selectors, gates, Doctor state
headers, alignment, runtime tables, per-layer metadata
protected tensors kept at source precision
expanded resident bytes and active bytes per token
```

A tiny seed with a large expanded runtime matrix is not zero cost.
