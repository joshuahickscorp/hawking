# DSV4F residual-composition oracle

Sealed rejection discriminator on the already-exported 43-layer late-hidden dump. Not a coherence, TPS, capability, or tournament claim.

## Claim boundary

- Rejection instrument only: `True`
- Cannot promote coherence: `True`
- Cannot promote full-model coherence: `True`
- Measured on 32 exported sequences (13 unique streams), max_seq_len 3, 96 tokens, site `late_hidden`, shape `[32, 4096]`.
- Geometry: last-position late_hidden, mean-pooled over HC manifold rows onto the exported 4096-d child.
- A fail means: The family cannot compose to end-to-end cosine >= 0.5 under the stated model. That is sufficient to refuse the family as a sub-1.5 static body. It is not a measurement of decode quality.
- A pass would mean: Only that this cheap residual-composition screen does not reject the family. A pass is not evidence of a usable model.

## Verdict

**Family 0.80–0.84: REJECT** under both models.

- Naive product: 0.80 → 6.805647e-05 (FAIL), 0.84 → 5.546376e-04 (FAIL). Necessary organ cosine `0.98400953`.
- Residual-identity (honest bound): 0.80 → 0.363980 (FAIL, margin -0.136020), 0.84 → 0.447659 (FAIL, margin -0.052341). Break-even organ cosine `0.861576`.
- Robustness: 0.80 FAIL is robust to r±10% and to (g-1)±10%. 0.84 FAIL is not robust to r-10% (end-to-end crosses 0.5). A 0.84-only verdict would not be a verdict. The 0.80-0.84 band as a family still rejects: 0.80 stays below the floor and 0.84 does not clear it at measured r.

## Honest bound

The architecture is h' = h + Δ. The naive product c^n treats the whole hidden state as replaced each layer. Residual identity is the honest bound for this discriminator because it uses the measured increment ratios and only corrupts the increment. It is still only a rejection screen: it does not model routing divergence, attention softmax, or decode.

Naive c^n multiplies organ cosine as if each layer replaced the residual. Residual identity only corrupts the increment Δ, so the identity skip dilutes organ error by ~r^2 / (1+r^2) per layer. With measured mean r ≈ 0.34 that dilution is large, which is why 0.84^43 = 5.55e-4 but the identity product is ~0.45. The identity product is the honest bound. The naive product remains a valid harsher necessary screen.

## Per-layer residual gain `||h_{L+1}|| / ||h_L||`

All 32×42 ratios: mean `1.114823`, gmean `1.105135`, min `0.978611`, max `2.151630`. Regime: **expansive** (38 of 1344 per-sequence steps are slightly < 1; no layer-mean gain is < 1).

Collapsed Frobenius gmean `1.117983`, mean `1.127002`. Unique-13 RMS-norm mean gain `1.120974` (this is the prior ~1.121 cascade figure).

| L | L+1 | mean | gmean | min | max | F-gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | 01 | 1.079667 | 1.077290 | 0.978611 | 1.243938 | 1.087206 |
| 01 | 02 | 1.044410 | 1.043193 | 0.989202 | 1.160709 | 1.061720 |
| 02 | 03 | 1.192428 | 1.186981 | 1.023593 | 1.420185 | 1.168158 |
| 03 | 04 | 1.182805 | 1.179059 | 1.040512 | 1.356708 | 1.167681 |
| 04 | 05 | 1.069059 | 1.068530 | 1.017969 | 1.117490 | 1.067678 |
| 05 | 06 | 1.201513 | 1.198149 | 1.066144 | 1.363086 | 1.212793 |
| 06 | 07 | 1.065519 | 1.065054 | 1.031234 | 1.117853 | 1.059724 |
| 07 | 08 | 1.190102 | 1.188208 | 1.090041 | 1.298418 | 1.193407 |
| 08 | 09 | 1.065981 | 1.065569 | 1.035569 | 1.130918 | 1.065896 |
| 09 | 10 | 1.078037 | 1.077759 | 1.026201 | 1.107262 | 1.078023 |
| 10 | 11 | 1.129626 | 1.128703 | 1.041193 | 1.274193 | 1.130104 |
| 11 | 12 | 1.031924 | 1.031649 | 1.011189 | 1.123128 | 1.028546 |
| 12 | 13 | 1.015284 | 1.015119 | 0.987169 | 1.091508 | 1.012302 |
| 13 | 14 | 1.051170 | 1.050959 | 1.018559 | 1.122046 | 1.051266 |
| 14 | 15 | 1.036602 | 1.036463 | 1.016142 | 1.104696 | 1.032145 |
| 15 | 16 | 1.035235 | 1.035065 | 1.006115 | 1.071072 | 1.031148 |
| 16 | 17 | 1.610454 | 1.585243 | 1.154702 | 2.075357 | 1.687877 |
| 17 | 18 | 1.121513 | 1.119985 | 1.030836 | 1.226301 | 1.101064 |
| 18 | 19 | 1.140131 | 1.138770 | 1.026678 | 1.232746 | 1.159984 |
| 19 | 20 | 1.270468 | 1.264439 | 1.114918 | 1.568648 | 1.234039 |
| 20 | 21 | 1.076042 | 1.074662 | 0.998617 | 1.167428 | 1.090183 |
| 21 | 22 | 1.309971 | 1.289848 | 1.038675 | 1.784175 | 1.470372 |
| 22 | 23 | 1.014262 | 1.014007 | 0.983535 | 1.055266 | 1.032959 |
| 23 | 24 | 1.100565 | 1.099038 | 1.028774 | 1.218111 | 1.093604 |
| 24 | 25 | 1.073959 | 1.073721 | 1.047710 | 1.129112 | 1.059728 |
| 25 | 26 | 1.048280 | 1.047377 | 1.011918 | 1.149858 | 1.101145 |
| 26 | 27 | 1.023661 | 1.023367 | 1.003161 | 1.085963 | 1.016723 |
| 27 | 28 | 1.012700 | 1.012693 | 1.006358 | 1.024511 | 1.013865 |
| 28 | 29 | 1.056581 | 1.055999 | 1.013828 | 1.116129 | 1.088727 |
| 29 | 30 | 1.033910 | 1.033795 | 1.012715 | 1.080274 | 1.031724 |
| 30 | 31 | 1.026939 | 1.026602 | 0.988611 | 1.081768 | 1.045407 |
| 31 | 32 | 1.021247 | 1.021127 | 1.000707 | 1.047817 | 1.010921 |
| 32 | 33 | 1.053265 | 1.053206 | 1.039357 | 1.088163 | 1.046195 |
| 33 | 34 | 1.021545 | 1.021418 | 1.003509 | 1.060560 | 1.012419 |
| 34 | 35 | 1.121772 | 1.119713 | 1.059213 | 1.271518 | 1.213775 |
| 35 | 36 | 1.194778 | 1.193831 | 1.109278 | 1.281998 | 1.138871 |
| 36 | 37 | 1.032135 | 1.032015 | 1.007731 | 1.068901 | 1.034755 |
| 37 | 38 | 1.029949 | 1.029800 | 1.001400 | 1.067592 | 1.029259 |
| 38 | 39 | 1.055016 | 1.054378 | 0.988815 | 1.118053 | 1.096238 |
| 39 | 40 | 1.053020 | 1.051234 | 0.981329 | 1.181259 | 1.139041 |
| 40 | 41 | 1.107291 | 1.103467 | 1.028765 | 1.374641 | 1.175811 |
| 41 | 42 | 1.743753 | 1.734240 | 1.315212 | 2.151630 | 1.761612 |

## Per-layer increment ratio `||h_{L+1}-h_L|| / ||h_L||`

All 32×42 ratios: mean `0.344382`, gmean `0.275070`, min `0.044322`, max `1.889764`. Mean alignment cos(h, Δ) `0.103596` (min `-0.390598`, max `0.544284`).

| L | mean r | gmean r | min r | max r | mean α | F-r |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | 0.385253 | 0.361184 | 0.252971 | 0.802240 | -0.003729 | 0.450933 |
| 01 | 0.265142 | 0.246370 | 0.181608 | 0.602258 | 0.007815 | 0.344250 |
| 02 | 0.511558 | 0.494390 | 0.339391 | 0.795639 | 0.130429 | 0.503721 |
| 03 | 0.430810 | 0.417802 | 0.253569 | 0.607839 | 0.225388 | 0.422623 |
| 04 | 0.280539 | 0.276700 | 0.172707 | 0.408734 | 0.109167 | 0.286709 |
| 05 | 0.373187 | 0.360897 | 0.256687 | 0.565346 | 0.387551 | 0.396536 |
| 06 | 0.222481 | 0.217069 | 0.150814 | 0.324323 | 0.185725 | 0.214313 |
| 07 | 0.425264 | 0.419550 | 0.337183 | 0.646433 | 0.267371 | 0.429241 |
| 08 | 0.225727 | 0.216437 | 0.162932 | 0.386885 | 0.179732 | 0.234419 |
| 09 | 0.272210 | 0.268106 | 0.188512 | 0.350689 | 0.157034 | 0.276039 |
| 10 | 0.412010 | 0.405124 | 0.272525 | 0.621265 | 0.120136 | 0.414647 |
| 11 | 0.194466 | 0.184725 | 0.129916 | 0.459152 | 0.057626 | 0.191692 |
| 12 | 0.179384 | 0.173590 | 0.132858 | 0.405026 | -0.017507 | 0.175648 |
| 13 | 0.263464 | 0.260191 | 0.200654 | 0.388582 | 0.063754 | 0.262419 |
| 14 | 0.214187 | 0.208964 | 0.150115 | 0.391911 | 0.060682 | 0.207653 |
| 15 | 0.245413 | 0.238508 | 0.150339 | 0.327772 | 0.013802 | 0.239272 |
| 16 | 1.113894 | 1.053413 | 0.496124 | 1.699780 | 0.131285 | 1.239087 |
| 17 | 0.426711 | 0.417773 | 0.277039 | 0.562135 | 0.074116 | 0.400925 |
| 18 | 0.462188 | 0.452784 | 0.348808 | 0.623550 | 0.083322 | 0.510304 |
| 19 | 0.559472 | 0.533559 | 0.335706 | 0.928559 | 0.251413 | 0.523332 |
| 20 | 0.238240 | 0.220898 | 0.109500 | 0.391011 | 0.167548 | 0.254763 |
| 21 | 0.605053 | 0.524579 | 0.249057 | 1.276361 | 0.231879 | 0.886858 |
| 22 | 0.137496 | 0.124539 | 0.044322 | 0.235345 | 0.002163 | 0.192095 |
| 23 | 0.313312 | 0.290337 | 0.128822 | 0.554978 | 0.159265 | 0.300091 |
| 24 | 0.245681 | 0.238892 | 0.167047 | 0.372791 | 0.186377 | 0.208436 |
| 25 | 0.214356 | 0.195449 | 0.121741 | 0.456166 | 0.094904 | 0.365028 |
| 26 | 0.136259 | 0.118930 | 0.049243 | 0.299941 | 0.080390 | 0.114614 |
| 27 | 0.118664 | 0.115697 | 0.092486 | 0.180164 | 0.050453 | 0.115714 |
| 28 | 0.203420 | 0.187919 | 0.119022 | 0.358746 | 0.164733 | 0.287744 |
| 29 | 0.167013 | 0.159124 | 0.118053 | 0.317862 | 0.122418 | 0.140505 |
| 30 | 0.191933 | 0.185019 | 0.111154 | 0.306655 | 0.036127 | 0.197838 |
| 31 | 0.125058 | 0.111836 | 0.058288 | 0.267925 | 0.084123 | 0.094347 |
| 32 | 0.198557 | 0.194967 | 0.146113 | 0.298987 | 0.176408 | 0.167677 |
| 33 | 0.138254 | 0.126707 | 0.066021 | 0.262594 | 0.072733 | 0.103080 |
| 34 | 0.377302 | 0.366667 | 0.268460 | 0.570509 | 0.135523 | 0.503957 |
| 35 | 0.502442 | 0.491602 | 0.313890 | 0.667060 | 0.170331 | 0.390869 |
| 36 | 0.198270 | 0.192958 | 0.138489 | 0.333302 | 0.060401 | 0.176040 |
| 37 | 0.186207 | 0.181353 | 0.126679 | 0.286713 | 0.067283 | 0.151694 |
| 38 | 0.290307 | 0.281595 | 0.157751 | 0.429385 | 0.031558 | 0.364352 |
| 39 | 0.292364 | 0.278875 | 0.199319 | 0.496589 | 0.003382 | 0.440380 |
| 40 | 0.544694 | 0.520799 | 0.332470 | 1.018835 | -0.088536 | 0.638540 |
| 41 | 1.575808 | 1.551663 | 0.879983 | 1.889764 | -0.143526 | 1.651572 |

## Naive product model `c^n`

n = 43. Necessary organ cosine `0.5^(1/43) = 0.98400953` (Q30 comparable `0.5^(1/48) = 0.98566320`).

| c | c^43 | vs floor 0.5 |
| ---: | ---: | :--- |
| 0.750 | 4.242622e-06 | FAIL |
| 0.800 | 6.805647e-05 | FAIL |
| 0.840 | 5.546376e-04 | FAIL |
| 0.853 | 1.073516e-03 | FAIL |
| 0.900 | 1.077526e-02 | FAIL |
| 0.950 | 1.101831e-01 | FAIL |
| 0.970 | 2.698886e-01 | FAIL |
| 0.990 | 6.491026e-01 | PASS |

## Residual-identity model (honest bound)

Per-layer factor `(1 + c * r^2) / (1 + r^2)` with measured r_L, alpha=0. 42 measured transitions plus one mean-r fill for the unobserved embedding→L0 increment (fill r = 0.344382).

Break-even: n=43 `0.861576`, n=42 `0.858621`, constant-r `0.849182`, measured-alignment `0.855122`, unique-13 `0.861815`.

| c | identity Π (α=0, n=43) | vs floor 0.5 | measured-α Π |
| ---: | ---: | :--- | ---: |
| 0.750 | 0.279987 | FAIL | 0.295855 |
| 0.800 | 0.363980 | FAIL | 0.380663 |
| 0.840 | 0.447659 | FAIL | 0.464202 |
| 0.853 | 0.478541 | FAIL | 0.494826 |
| 0.900 | 0.607756 | PASS | 0.621937 |
| 0.950 | 0.780948 | PASS | 0.790126 |
| 0.970 | 0.862484 | PASS | 0.868582 |
| 0.990 | 0.952011 | PASS | 0.954261 |

## ±10% sensitivity

| knob | scale | break-even c | Π(0.80) | Π(0.84) |
| :--- | ---: | ---: | ---: | ---: |
| r × scale | 0.9 | 0.838552 | 0.421789 | 0.503187 |
| r × scale | 1.0 | 0.861576 | 0.363980 | 0.447659 |
| r × scale | 1.1 | 0.879168 | 0.312974 | 0.397156 |
| 1+(g-1)×scale | 0.9 | 0.854785 | 0.382084 | 0.465189 |
| 1+(g-1)×scale | 1.0 | 0.868440 | 0.344906 | 0.428909 |
| 1+(g-1)×scale | 1.1 | 0.879529 | 0.311944 | 0.396076 |

Literal `g × 0.9` is ill-posed: measured g ≈ 1.11, so a 10% cut in g is a ~100% cut in the expansion (g-1). It is reported in the JSON under `gain_multiplicative_reported_as_ill_posed` and is not used as a verdict knob.

## Prior findings audit

| claim | prior | measured | verdict |
| :--- | ---: | ---: | :--- |
| gain ~1.11×/layer | 1.11 | 1.1051345 | CONFIRMED |
| cascade g ~1.121 | 1.121 | 1.1209744 | CONFIRMED |
| 0.5^(1/43) | 0.98401 | 0.98400953 | CONFIRMED |
| Q30 0.5^(1/48) | 0.98566 | 0.9856632 | CONFIRMED |
| 0.80^43 | 6.81e-05 | 6.8056473e-05 | CONFIRMED |
| 0.84^43 | 0.000555 | 0.00055463759 | CONFIRMED |
| identity break-even ~0.853 | 0.853 | 0.86157647 | REVISED |
| identity Π(0.80) ~0.386 | 0.386 | 0.36398029 | REVISED |
| identity Π(0.84) ~0.470 | 0.47 | 0.44765898 | REVISED |

0.853 is the constant-r residual-identity break-even for r≈0.351. Measured arithmetic-mean r is slightly smaller (~0.344) and that constant-r model breaks even at ~0.849. The contract requires measured per-layer r, not a constant: that bound is ~0.862 because L16 and L41 inject r>1.

## Duplicate-stream caveat

max_seq_len=3 last-position late_hidden collapses same-family prompts that share the first three tokens into identical 4096-d child vectors. Primary stats use all 32 exported rows as specified; unique-13 is a robustness check.

| n | example_ids |
| ---: | :--- |
| 3 | pfv0:d3:Mathlib.Algebra.Algebra.Equiv.symm_trans_apply, pfv0:d3:Mathlib.Algebra.Algebra.Defs.algebraMap.coe_smul', pfv0:d3:Mathlib.Algebra.Algebra.Defs.commute_algebraMap_left |
| 2 | pfv0:d2:Mathlib.Algebra.Algebra.Epi.isEpi_of_surjective_algebraMap:step2, pfv0:d2:Mathlib.Algebra.Algebra.Epi.injective_lift_lsmul:step1 |
| 4 | pfv0:d1:Mathlib.Algebra.Algebra.Hom.comp_apply, pfv0:d1:Mathlib.Algebra.Algebra.NonUnitalHom.restrictScalars_apply, pfv0:d1:Mathlib.Algebra.AddConstMap.Basic.map_const_add, pfv0:d1:Mathlib.Algebra.Algebra.NonUnitalSubalgebra.mem_carrier |
| 3 | pfv0:d4:Mathlib.Algebra.AddConstMap.Basic.map_sub_nat':flip_eq_goal, pfv0:d4:Mathlib.Algebra.Algebra.Basic.End.algebraMap_isUnit_inv_apply_eq_iff':wrong_rw_lemma, pfv0:d4:Mathlib.Algebra.Algebra.Basic.coe_algebraMap_ofSubsemiring:flip_eq_goal |
| 2 | pfv0:d6:mathlib_ce:SeparableNotSecondCountable, pfv0:d6:enum:83:9578a4c57d |
| 2 | pfv0:expert:fix:two_plus_two, pfv0:expert:fix:power |
| 6 | pfv0:thesis:t09_flatten, pfv0:halo:cd04_gcd, pfv0:thesis:t01_add, pfv0:thesis:t15_word_count, pfv0:thesis:r04_reverse_words, pfv0:thesis:r11_count_words |
| 5 | pfv0:d7:checkpoint:extra5, pfv0:d7:checkpoint:extra9, pfv0:d7:search:extra19:7621061d:ea73, pfv0:d7:checkpoint:extra2, pfv0:d7:checkpoint:extra8 |
| 1 | pfv0:halo:lc02_needle |
| 1 | pfv0:halo:lc01_needle |
| 1 | pfv0:halo:rt04_multi |
| 1 | pfv0:halo:gr01_wolves |
| 1 | pfv0:halo:rt03_table |

Seal `d921964eeb6be584a5da7293645aaefa96aaa5c4722d0ed5af6459d73d46b9f1`.
