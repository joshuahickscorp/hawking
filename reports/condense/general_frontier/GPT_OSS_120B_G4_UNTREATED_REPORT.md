# GPT-OSS-120B G4 Untreated Control Report

G4 ran the real, bounded-streaming, coherence-validated full-model forward (gptoss_real_forward.py)
over a 6-prompt Harmony holdout, comparing the source-native parent to a uniform untreated RVQ
control near 1 BPW. Real logits, real tokens, real perplexity. 12/12 rows sealed and integrity-verified.

## Parent real perplexity (first-ever in the campaign)

| prompt | domain | parent PPL |
|---|---|---|
| code_py | code | 1.92 |
| reason_syllogism | reasoning | 3.46 |
| math_add | math | 5.14 |
| gen_science | general | 6.68 |
| instr_list | instruction | 12.26 |
| gen_paris | factual | 27.43 |

Domain-ordered (code most predictable, factual-completion least), which corroborates forward correctness.

## Uniform untreated RVQ @ 1 BPW control

| prompt | next-token agreement | sym KL | verdict |
|---|---|---|---|
| code_py | 0.63 | 1.87 | degraded |
| instr_list | 0.55 | 1.40 | degraded |
| gen_paris | 0.20 | 1.84 | degraded |
| reason_syllogism | 0.25 | 4.43 | collapse |
| math_add | 0.22 | 3.56 | collapse |
| gen_science | 0.11 | 3.49 | collapse |

## Conclusion (required)

Uniform untreated RVQ near one BPW is a real-forward NEGATIVE CONTROL. It is not Hawking's strongest
treated candidate. Every prompt fails the capability gate (agreement < 0.95); 3 collapse outright.
Code tolerates it best (most redundant), factual and reasoning collapse hardest. This is the first
REAL-forward (real logits) confirmation of the G0-G3 proxy negative and the Second-Light 0.688 baseline.

The final Doctor correction wave (D0-D6: tensor-class PQ + protected islands + diagnosis-selected
Doctor + global byte allocation) is the decisive test of whether ANY sub-bit allocation preserves
real capability. This control is the equal-byte baseline the treated candidates must beat.
