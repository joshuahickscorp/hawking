# Pre-registration PROM-001
## The iso-memory frontier and the RandomPolicy control

```text
id:            PROM-001
filed:         2026-07-23
status:        FILED (deterministic machinery built; capability arms GATED on S0.8 + L3)
supersedes:    nothing
machinery:     tools/prometheus/prometheus.py  (L1 spine, tested)
profiles:      profiles/prometheus/{uniform,general,math,mathnarrow,random}-v1.json
gate to run:   MODEL LADDER DONE and HAWKING SHIPPABLE (docs/plans/PROMETHEUS_LEG_PLAN.md §1)
```

This document is filed **before** any capability-bearing artifact is generated. The
allocation machinery is deterministic and already runs on real metadata; the capability
measurements (G1–G8) are gated on a served runtime (Gate S0.8) and the evaluator (leg L3).
Filing now, with the hypotheses and analysis plan fixed and hashed, is what prevents the
garden of forking paths (Revision 4 §II.3, §V.1). The seed is recorded here and nowhere
else may it be chosen after seeing a result.

---

## Hypotheses

```text
H1  Non-uniform allocation beats uniform allocation at equal bytes.
H2  Capability-CONDITIONED allocation beats arbitrary non-uniform allocation
    at equal bytes.   (Math vs Random — the load-bearing test.)
H3  Capability ELIMINATION beats capability ALLOCATION at equal bytes.
    (MathNarrow vs Math.)
```

## Arms

At each sampled bitrate, on each substrate, five artifacts from the same machinery:

```text
<Base>-H<rate>-Uniform      profiles/prometheus/uniform-v1.json
<Base>-H<rate>-General      profiles/prometheus/general-v1.json
<Base>-H<rate>-Math         profiles/prometheus/math-v1.json
<Base>-H<rate>-MathNarrow   profiles/prometheus/mathnarrow-v1.json
<Base>-H<rate>-Random       profiles/prometheus/random-v1.json   (CONTROL)
```

The Random arm holds total physical bytes identical to the Math arm and randomizes only
which experts form the protected coalition (seed below). The spine (router, normalization,
indexer) is pinned native in every arm. Byte-match is asserted by the pipeline, not assumed
— `test_prometheus.py::test_math_and_random_are_byte_matched`.

## Substrates and order

```text
1. Qwen3.5-397B-A17B (F2)   build-first; residual stream NOT known-expansive.  RUN FIRST.
2. GLM-5.2 (F5)             flagship; residual stream is expansive (HAWKING_ASCENSION_STATUS).
                           On GLM the honest prior is that H2 resolves NEGATIVE; that is
                           still a publishable result (kill criterion K3).
```

Qwen3.5 first is deliberate: PROMETHEUS_LEG_PLAN.md §3 records that GLM's weight-space
escape was falsified by the null-corrected metric, so GLM is the wrong place to first test
whether conditioning helps.

## Sampling policy

```text
bitrates:   dense   H07 H065 H06 H055 H05 H045 H04   (the cliff region, per the [M] 0.7 result)
            sparse  H09 H15                          (reference)
            iso-mem H08 flagship  vs  H15 build-first at matched bytes (Paper 3)
coalition:  fraction swept {0.02, 0.05, 0.10, 0.20}; membership from cartography (P4, GATED).
seed:       20260723   (RandomPolicy coalition + any stochastic choice; fixed, recorded here)
```

## Metrics

```text
primary:     Math Profile v1 aggregate (G1–G8), per arm, per bitrate.
             G3 (formalization) and G8 (proof-criticism) are the discriminating gates.
secondary:   per-subdomain retention (algebra, number theory, combinatorics, analysis,
             geometry); trajectory divergence depth distribution; tokens/sec;
             full byte decomposition (already emitted by the spine).
```

Any metric not listed here is exploratory and will be labeled so in publication.

## Analysis plan

```text
- Compare arms at MATCHED bytes only. Byte match is enforced by the machinery.
- H1: Uniform vs {General, Math} aggregate gate retention.
- H2: Math vs Random. This is the arm that makes the paper credible. If Math and Random
      are statistically indistinguishable, Claim A is FALSE — publish it (K3).
- H3: MathNarrow vs Math. If MathNarrow wins, elimination beats allocation. If it loses,
      mathematical capability is cross-domain and irreducible. Both outcomes publish.
- The gap in bits between the Uniform cliff and the Math cliff is the headline number.
```

## Stopping rule

```text
Run the full bitrate × arm grid on Qwen3.5 first. Stop and write up when either:
  (a) H2 resolves (Math separates from Random, or provably does not) at >=3 bitrates, or
  (b) capacity kill criterion K6 fires (<8h/week for two quarters).
Do not add arms or bitrates after seeing results without filing PROM-002.
```

## What would falsify

```text
H1 false:  Uniform indistinguishable from General and Math at matched bytes.
H2 false:  Math indistinguishable from Random at matched bytes. (Claim A false.)
H3 false:  MathNarrow indistinguishable from Math at matched bytes.
machinery invalid:  the Random arm fails to reconcile to Math's byte total. The pipeline
                    asserts against this; a violation halts the experiment, it is not a result.
```

---

```text
content hash (sha256 of PROM-001-isomemory-randompolicy.json, canonical, minus the hash field):
see PROM-001-isomemory-randompolicy.json .content_sha256
```
