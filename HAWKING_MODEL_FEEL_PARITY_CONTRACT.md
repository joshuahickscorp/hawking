# Model Feel Parity Contract

## Why reconstruction metrics are not enough

A compact artifact can reproduce its parent's logits to five decimal places on a probe
and still be unusable, and it can fail a cosine test and still be indistinguishable in
conversation. "On par" is a claim about behaviour under use, so it is measured that way.

This contract also fixes a trap the campaign has already walked into once. A sub-bit
artifact is lossy by construction, so *correctness of execution* and *preservation of
capability* are different questions with different oracles:

- **Execution correctness** is graded against what the artifact encodes. The authority is
  a numpy reference reading the same container through the same codec.
- **Capability preservation** is graded against the parent. The authority is the parent
  model's behaviour on frozen prompts.

Conflating them is how a correct runtime gets blamed for a collapsed rate, and how a
collapsed rate hides behind a correct runtime.

## The gate

A candidate rate passes only if all four hold:

1. no material aggregate loss against the parent;
2. no statistically significant demotion in any critical domain;
3. no abnormal loops, shallowness, or truncation;
4. no hidden model substitution at any point in the evaluation.

## Dimensions

Compared blind, parent versus compact, on frozen prompts with paired generation settings:

```text
reasoning persistence          answer depth
coding                         mathematics
instruction following          tool use and structured output
self-correction                uncertainty calibration
creative flexibility           verbosity calibration
refusal and fallback behavior  long-horizon coherence
2K / 8K / 32K context          rare and low-margin cases
```

## Evidence required

```text
frozen prompts, recorded verbatim
paired generation settings (same sampler, same seed, same token budget)
human-blind or independent-judge scoring
task success, not self-reported confidence
logit / NLL evidence
trajectory and route evidence
an explicit no-hidden-fallback assertion, checked rather than stated
```

## Screening gate

Full blind scoring is expensive, so a candidate first faces a cheap screen that a
collapsed rate cannot survive: greedy continuation of a handful of trivially-predictable
prompts. Degenerate repetition, immediate truncation, or a top-5 that is five inflections
of one wrong word is a decisive fail, and no further evaluation is spent on it.

The screen can only fail a candidate. Passing it means the candidate has earned the full
comparison, nothing more.

## Selection rule

The selected General artifact is the **lowest** rate that passes. Preference for a small
headline number never overrides the gate; if no sub-bit rate passes, the ladder brackets
upward (H10, H12, H15) and the lowest passing rate is selected.
