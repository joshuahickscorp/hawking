# HAWKING resume checkpoint

Written 2026-07-27. The machine was handed to another project; nothing here is running.

## Where the campaign actually stands

`RAMANUJAN_SANDBOX_READY` is **not** reached. It needs a capable Math-Preserve-v2 and there
isn't one.

**Two artifacts have now failed the same way.**

| artifact | BPW | verdict |
|---|---|---|
| Math-Preserve | 0.9774 | collapsed, prompt-independent output |
| Generation B (activation-aware) | 1.0924 | collapsed, argmax 50379 for every prompt |

Generation B was physically perfect: 282/282 shards, coverage COMPLETE at 59,585/59,585
tensors, assembled clean, codec round-trip proven. It still cannot complete `2 + 2 =`.

## Why it failed, and why that matters more than the failure

The allocator's objective was wrong, and it is mine.

`beats_null` was made a hard constraint -- correct, it stops `lm_head` being stored below its
own mean -- and then BPW was minimised subject to it by taking the **lowest rank that cleared
the bar**. The constant-mean cosine null is a **floor test** ("is this better than storing
nothing"). It was used as an **admission criterion**.

    shipped cosine   min 0.0436   median 0.6782   max 0.9320
    100% of 55,398 tensors below 0.90, 78.9% below 0.80
    surplus over null: median +0.0266, MINIMUM +0.0000
    rank 16 chosen for 50,356 of 55,398

A tensor whose null was 0.0435 was admitted at 0.0436.

**The same flaw invalidates the earlier "breakthrough".** That result was 0.755 cosine against
a 0.651 null, replicated 12/12, and was reported as a win because it beat the null. It does
beat the null. 0.755 cosine does not preserve a model. The whole activation-aware programme
was validated against a floor test.

## Re-allocation alone cannot rescue it -- measured

Choosing the **best available** rank for every tensor from the existing measurement:

    cosine  min -0.4639  median 0.7869  max 1.0000
    96.8% of tensors below 0.90
    99.5% OF WEIGHTS have a best-available reconstruction under 0.95

The measurement only holds ranks 16 and 64 (`--ranks 16,64`; the default is
`8,16,32,64,128,256`). And rank 16 -> 64 bought only +0.0167 cosine on gate_proj, so the
curve is flat: more rank is unlikely to close a 0.08 gap.

**Low-rank looks like the wrong family for this model.** The frozen tournament has six arms
and only the low-rank one was ever executed at scale. PQ+residual and additive/multi-codebook
are not low-rank and target exactly this failure.

## What is reusable -- the redo is not from zero

- `Models/GLM-5.2/b4734de4...` 169 GB verified BF16 parent, on disk
- `GLM52Gravity/source_fetch` 80 GB teacher capsules (REAL captured activations). **Do not
  delete.** Gaussian proxies invert the ranking; these are the only valid basis source.
- `GLM52Gravity/activation_aware_pack/MEASUREMENT.json` 80 MB, rank 16/64 curves for all
  59,585 tensors -- 4.3 h of work, still valid as far as it goes
- Codec round-trip proven, assemble proven, capability gate proven (its selfcheck shows a
  correct " 4" passes, so the FAIL is real)

Everything except the packing is intact.

## ANSWERED: the quality floors, measured on Llama-1B

The calibration finished. Three results.

**1. `beats_null` is FALSIFIED as an admission criterion, empirically.** Rank 128 and the
every-layer variant both reach ~100% beats_null while capability is dead. It is a floor test
and nothing more.

**2. Worst-case structural quality predicts collapse better than the median** -- the
hypothesis held. `every_layer` at median cosine 0.961 still collapses, so a good median does
not save you if one layer is bad.

**3. Numeric floors (Llama-1B; transfer the OBJECTIVE, re-measure the numbers on GLM):**

    math + live capability:  min cosine >= 0.70, median >= 0.92, rank >= 256
    strong capability:       min cosine >= 0.74, median >= 0.96, rank >= 512

Generation B swept ranks 16 and 64 and shipped a 0.6782 median. It was never in range.

The corrected objective:

    minimise BPW
    subject to  every tensor: functional_cosine(real X_hold) >= T_tensor
                every layer:  min tensor cosine in that layer >= T_layer
    NEVER rank = cheapest that beats the constant-mean null
    NEVER promote on reconstruction error
    NEVER Gaussian activation proxies

## AND A METHOD BUG IN THE GLM PACKER -- confirmed in source

`glm52_activation_aware_pack.py:454` does `mu = X_fit.mean(axis=0)`, `Xc = X_fit - mu`, then
SVD on the CENTERED activation matrix. The calibration used **uncentered** SVD and notes that
centering **ceilings** functional cosine by discarding the mean activation direction.

This is structurally unfair as scored: the basis is denied the mean direction, and its
reconstruction is then measured against a null that IS the mean. For `gate_proj` that null is
0.9874, i.e. the mean direction dominates the tensor -- and the basis was forbidden from
using it. Some unknown part of Generation B's collapse is this, not the family.

**So test uncentered SVD BEFORE concluding low-rank is dead.** It is a one-line change to the
basis builder and it changes what "rank 64 cannot do this" means.

## Resume commands

```bash
cd /Users/scammermike/Downloads/hawking
git log --oneline -8
cat GLM52_GENERATION_B_CAPABILITY_VERDICT.json   # the refusal and its root cause
cat GLM52_BYTE_ATTRIBUTION.json                  # where the 95.8 GiB went
~/.claude-grok/bin/grok-run status | tail -20    # lane states
```

Re-run the gate against any new artifact:

```bash
.venv/glm52/bin/python tools/condense/glm52_capability_gate.py \
  --artifact "<dir>" --run --out CAPABILITY.json
```

## Fences -- all intact, none may be flipped by an agent

- `ODYSSEY_LAUNCH_AUTHORIZED` false
- `RAMANUJAN_RESEARCH_AUTHORIZED` false
- Generation B and Math-Preserve are both hash-REFUSED in
  `odyssey/launch/SUBSTRATE_CAPABILITY.json`
- Never train on, or distil teacher traces from, either artifact

## Not applied on purpose

`single-admission` (943 lines rewriting the packer, ~2x network by admitting each shard once)
is finished and unapplied. It should land only after the calibration says what the pack should
be doing -- optimising the pipeline before knowing the target is how this failure happened.

## Done and verified this session

Q0 ACHIEVED (capsule re-proves in a `--network=none` pinned container; a false theorem exits 1
with `unsolved goals`). Corpora 16,188 items, split sealed and independently recomputed, zero
overlap. Formal system: retriever and value converged, formalizer/prover/repair did not, and
say so. Governance: 29 adversarial invariant tests. Adapters: 0 of 10 families overclaim.
Write-lease race fixed at the sharing, production unchanged. Capability gate byte-BPE bug
fixed -- it would have failed a correct artifact.
