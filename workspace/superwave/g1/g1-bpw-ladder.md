# G1 intelligent BPW ladder (up from 1.291)

Lane: `70-bpw-ladder`. Arithmetic + catalog parse + receipt read. No GPU, no pack, no generate, no resident touch.

Use only if mixed-sub15-v1 cannot be repaired cheaply. Standing priority remains: evaluate the 1.291 artifact first.

Epistemic tags: **MEASURED** on-disk integer/float. **DERIVED** exact arithmetic on MEASURED integers. **ESTIMATED** catalog slack of a not-yet-emitted HQ38M20. **CITED** prior lane/receipt, not re-run. **EXPECTATION** not a finding. **POLICY** layer order among the 58 unscored layers.

---

## 0. Refusals

- Do not jump to uniform Q3 or uniform Q4. Those are floors/proofs, not rungs.
- Do not use `g1-heterogeneous-allocation.md` 2.0 / 1.5 / 1.2 tables as-is. They interpolate 58 of 64 layers (`g1-heterogeneous-allocation.md:92-102`). Not determined.
- Do not treat mixed-2p0-v1 as the 2.0 rung. It allocated backwards (attention 4.250, `down_proj` 0.1316).
- Do not treat recorded INCOHERENT on sub15 / 2p0 as a floor. Both vehicles are confounded (expand-to-Q4; 2p0 also crushed down and used 2048-col binary/CSR tiles). Standing rule: a confounded failure does not locate a floor.
- Do not propose generator+residual, VQ, or entropy-coding. Dead under the tested constructions. REOPEN_IF those files' reopen conditions fire.
- Do not use affine Q2 as a quality step. Descent: Q2 hold mins sit below rice at higher BPW (`QWEN38_BPW_DESCENT.json` `summary.by_role_codec`).
- Do not use the 1-bit / 2-bit RTN pair as distinct rungs. They are the same operator with `qmax` clamped to 1. The 1-bit floor was never measured.

---

## 1. Accounting

Authority (`crates/hawking-core/src/model/qwen38_pack.rs:673-679`):

```
complete_physical_bpw = 8 * tensor_payload_bytes / source_weight_elements
```

`tensor_payload_bytes` includes codes, scales, JSON/container headers, rice indices, HGRAVS factors. Catalog bytes are extra metadata, billed separately.

Language N **MEASURED** 26,895,998,464 (`g1-bit-budget-accounting.md` §4; G0 `manifest.json`; mixed PACK_REPORTs). Vision excluded.

| class | elements MEASURED | mass DERIVED |
|---|---:|---:|
| mlp.gate / up / down (each) | 5,704,253,440 | 0.212086 |
| MLP | 17,112,760,320 | 0.636257 |
| attention GEMV | 7,237,795,840 | 0.269103 |
| embed + lm_head | 2,542,796,800 | 0.094542 |
| small (f32 in G0) | 2,645,504 | 0.000098 |

Geometry: `qwen38_geometry.rs:20-52`. 64 layers, hidden 5120, intermediate 17408, vocab 248320, GQA iff `(layer+1)%4==0`.

Byte ceiling at target T: `T * N / 8`.

| T | budget bytes DERIVED |
|---:|---:|
| 1.3 | 4,370,599,750.4 |
| 1.4 | 4,706,799,731.2 |
| 1.5 | 5,042,999,712 |
| 1.6 | 5,379,199,692.8 |
| 1.75 | 5,883,499,664 |
| 2.0 | 6,723,999,616 |

---

## 2. Native codecs

`load_mixed` match (`qwen38_hybrid_decode.rs:601-664`):

| catalog codec | magic | today | named change |
|---:|---|---|---|
| 0 | `HGRAVB01` | ACCEPTED → Binary | K-complete bind: `q80_binary_group_matvec_tg256` covers 2048 cols; sibling `q80_binary_group_matvec_simd_bytes` already tiles (`g1-sub15-native-gap.md` §4) |
| 1 | `HGRAVR02` | ACCEPTED → Residual | same 2048-col bind on CSR tg256; sibling `q80_binary_group_csr_matvec_bytes` tiles |
| 2 | `HGRAVS01` | ACCEPTED → Hgravs, locked r160_b3 (`:1116-1134`) | other ranks refused until lock widens |
| 3 | `HGRAVU01` | ACCEPTED → Uniform; bits from JSON | `dispatch_uniform` → `dispatch_factor` (`:1457-1475`). bits=8 → `q80_uniform8_matvec_*`; bits=3 → `q80_hgravs01_factor_matvec_simd3` (`:1396-1406`). simd3 loops `col += 256` (`q80_mixed_decode.metal:869`) — K-complete on 5120/6144/17408 |
| 3 | `HQ30UQ4\0` | ACCEPTED → Q4 map | G0 kernel. Generate-proven |
| 4 | f32v2 | REFUSED (`unknown mixed codec 4`) | accept codec 4 → `f32s` (`g1-sub15-native-gap.md` §2, packer `CODEC_F32 = 4`) |
| other | — | refuse | — |

MLP **name** lock (`:958-1003`): every `gate_proj` must be Binary, every `up_proj` Residual, every `down_proj` Hgravs. Attention has no lock. Embed accepts Uniform or HQ30UQ4.

**Named change A** — drop or widen `assert_mixed_mlp_native` so `MixedGpuWeight::Uniform` is legal on `mlp.*`. Required for any Q3 MLP organ. No new shader.

**Named change B** — catalog emit + codec-4 f32v2 + K-complete binary/CSR bind. Already scoped at 250–500 lines (`g1-sub15-native-gap.md` §0). Required to consume sub15-class Binary/Residual natively.

**Named change C** — exact output-row island: 128 f32 rows + `u32` row-ids, axpy onto write-GEMV outputs. Not a current codec. Uses codec-4 f32v2 once B lands. No new shader family if applied as a row-scatter of already-resident f32.

HQ30UQ2 / HQ30UQ3 (`uniform_qn.rs`, group **128**, diagnostic, unqualified) are **not** in `load_mixed`. Not used.

Ternary `t0.7_g128` beats affine Q2 at the same ~2.25 BPW (CITED descent) but has no native magic. New family. Not used.

---

## 3. Known organ facts (cited; not re-derived)

Base point mixed-sub15-v1 **MEASURED** (`PACK_REPORT.json`):

| organ | codec | bytes | physical BPW |
|---|---|---:|---:|
| mlp.gate | HGRAVB01 g128 | 802,177,344 | 1.1250234267290902 |
| mlp.up | HGRAVR02 rice_q1_rms_2pct | 918,036,000 | 1.2875108157887178 |
| mlp.down | HGRAVS01 r160_b3 | 93,847,197 | 0.13161714918473189 |
| attention GEMV | HGRAVR02 rice_q1_rms_2pct | 1,165,098,376 | 1.2877935788805008 |
| embed | HQ30UQ4 g64 | 675,430,440 | 4.250000251691366 |
| lm_head | HQ30UQ4 g64 | 675,430,440 | 4.250000251691366 |
| small | f32v2 | 10,584,840 | 32.00853977162764 |
| **complete** | | **4,340,604,637** | **1.2910781930062503** |

MLP class physical **MEASURED** 0.8480504639008466 (mixed-2p0 / sub15). Contract: that mix kills tokens. It is below the MLP floor.

Q3 hold, descent 6 layers, holdout 64 of 256 (`QWEN38_BPW_DESCENT.json` `coherence_floor.quality_intact`):

- hold_min across MLP + attn_in = **0.9679** (up L31 Q3 hold 0.9678982165029962)
- gate Q3 hold_min 0.976594, up 0.967898, down 0.972669
- attn_in Q3 hold_min 0.979393 — **not** 0.99
- attn_in Q4 hold_min 0.996128 — holds 0.99
- packed mixed-q3mlp strided weight-cosine min **0.9652814877860332** (L63 down), median 0.96906, n=192 (`mixed-q3mlp-v1/PACK_REPORT.json`)

Q2 is a regression vs rice at higher BPW (CITED `by_role_codec`):

| organ | rice hold_min | Q2 hold_min |
|---|---:|---:|
| attn_in | 0.888151 | 0.835245 |
| down | 0.803614 | 0.799022 |
| gate | 0.876818 | 0.788477 |
| up | 0.812629 | 0.781999 |

Attention at ≤ ~2.07 does not hold 0.99. Exact inversion this lane, tables HQ30UQ4 + small f32 + MLP 0.8480504639008466, complete = 1.5:

```
max b_attn = (1.5*N − E_mlp*0.8480504639008466 − 8*1350860880 − 8*10584840) / 7237795840
           = 2.064157091228481   DERIVED
```

Contract's "about 2.07" is this cell. Q2 attention is 2.25 and still hold_min 0.835. Rice 1.288 hold_min 0.888. Q3 attention 3.25 hold_min 0.979. No measured native point in (1.288, 2.064] holds 0.99.

Q4 is the last generate-proven cheap codec for attention, embed, lm_head (G0 4.252735126866492, 6/6 oracle-32). Embed/lm_head stay HQ30UQ4 on every rung.

Channel 3994: activation-hot 10× in 54/64 layers; top-5 output row on all 128 write tensors (64 down + 48 lin_o + 16 o); L0 lin_o kurtosis 149.36 (`g1-doctor-tensor-map.md:53-69`). Protect what to keep, do not rescale by |X| (activation-weighted scale destroyed L0 out_proj 0.99224 → 0.91865, CITED). Capture is 256 tokens — underdetermined (rows/dim 0.0417 at K=6144, 0.0147 at 17408). Island is a capability bet on a MEASURED energy statistic, not a determined doctor fit.

HGRAVS down quality as hold-cosine: **unmeasured**. Isolated HGRAVS hold was never scored (`g1-heterogeneous-allocation.md:184`).

---

## 4. Spend rule

From §3, incremental bits from 1.291 go here, in order:

1. Exact-protect write-row 3994 (tiny, every rung).
2. Raise `down_proj` HGRAVS01 → HGRAVU01 Q3. This is the crushed organ in the backwards 2.086 pack. Q3 is the first measured hold. Skip Q2.
3. Among remaining MLP, raise measured-hard `up_proj` then `gate_proj` to Q3. Rice up hold_min 0.813 (L31) is worse than binary gate hold_min 0.838.
4. Do **not** raise attention while any down is still HGRAVS01. That is mixed-2p0.
5. Do **not** raise embed/lm_head above Q4. Already generate-proven. Tables at bf16 are KILL 1 (`g1-bit-budget-accounting.md` §9).

Cost to move one full organ 64/64 to Q3, DERIVED from MEASURED class bytes and mixed-q3mlp 36,208,920 B/tensor:

| move | delta bytes | delta BPW |
|---|---:|---:|
| 64 down 0.1316 → Q3 | 2,223,523,683 | 0.6613693664434618 |
| 64 up rice → Q3 | 1,399,334,880 ± rice-header jitter | ~0.4162 |
| 64 gate binary → Q3 | 1,515,193,536 | 0.45066 |
| 64 attn rice → Q3 | ~1.775e9 | ~0.528 |
| 64 attn rice → Q4 | ~2.68e9 | ~0.797 |

Only the 2.0 increment (0.709) can buy all-down Q3. No rung ≤ 2.0 can buy all-attention Q3 without leaving down crushed. Hence no rung on this ladder puts attention at a 0.99-class codec.

Measured-hard downs, binary hold cosine (descent, 6 layers only):

`L63 0.730 < L47 0.780 < L31 0.816 < L15 0.827 < L3 0.847 < L0 0.919`

All six Q3 holds ≥ 0.9727. Upgrade those six first. Remaining 58: **POLICY** descending layer index. Not an interpolator claim.

Island bytes **DERIVED** from geometry (f32v2, 8-byte header each, 128 `u32` ids):

```
64 * (8 + 4*17408) + 64 * (8 + 4*6144) + 128*4 = 6,030,848
island BPW = 0.0017938275860841454
```

Catalog tax **MEASURED** 158,970 B on mixed-2p0 (851 names, same set) = 0.0000472843572512 BPW. A new 851-name HQ38M20 will sit within a few hundred bytes. Cannot flip a rung.

---

## 5. Rungs

Pinned on every rung:

| organ | codec | catalog | consume |
|---|---|---|---|
| embed | HQ30UQ4 g64 | 3 | ACCEPTED, G0 kernel |
| lm_head | HQ30UQ4 g64 | 3 | ACCEPTED, G0 kernel |
| 353 small | f32v2 | 4 | named change B |
| write-row 3994 | f32v2 sidecar | 4 | named change C |

Base of every rung: sub15 gate/up/attn/small/tables bytes, **MEASURED**.

`k` = number of downs moved to HGRAVU01 Q3 (36,208,920 B **MEASURED** mixed-q3mlp, every down). HGRAVS nbytes vary 1,466,360–1,466,365 (**MEASURED** mixed-2p0 catalog).

### 1.3 — island only

k = 0. Cannot buy one down→Q3 (delta 34,742,555–34,742,560 B; slack after island+catalog ≈ 23.8 MB).

| organ | codec | physical BPW |
|---|---|---:|
| gate 64 | HGRAVB01 | 1.1250234267290902 |
| up 64 | HGRAVR02 2% | 1.2875108157887178 |
| down 64 | HGRAVS01 r160_b3 | 0.13161714918473189 |
| attn 304 | HGRAVR02 2% | 1.2877935788805008 |
| embed, lm_head | HQ30UQ4 | 4.250000251691366 |
| small | f32v2 | 32.00853977162764 |
| island 3994 | f32v2 | 0.0017938275860841454 of N |

```
payload = 4,340,604,637 + 6,030,848 = 4,346,635,485
tensor-complete = 8 * 4346635485 / 26895998464 = 1.2928720205923343   DERIVED
artifact-complete (+158970) = 1.2929193049495855                       DERIVED
MLP class = 0.8480504639008466                                        MEASURED (unchanged)
```

EXPECTATION: incoherent. MLP still at the kill mix. Island is the only legal spend of a 30 MB increment.

### 1.4 — 10 downs to Q3

k = 10. Layers: measured `{63,47,31,15,3,0}` + POLICY `{62,61,60,59}`.

```
payload = 4,694,061,059
tensor-complete = 1.3962109836622572
artifact-complete = 1.3962582680195084
down physical = 0.618868394459
MLP class = 1.010467545659
```

54 downs still HGRAVS01. Named change A (Uniform legal on those 10 `down_proj`).

### 1.5 — 20 downs to Q3

k = 20. Layers: six measured + POLICY `{62…49}`.

```
payload = 5,041,486,638
tensor-complete = 1.4995499482193904
artifact-complete = 1.4995972325766416
down physical = 1.106119646746
MLP class = 1.172884629754
slack to 1.5 artifact = 1,353,894 B
```

This is **not** "MLP 0.848 + attn 2.06". That cell is the backwards inversion and is refused. 44 downs remain crushed. Attention stays rice 1.2878.

EXPECTATION: incoherent. Completes the 1.5 budget without putting any organ at a hold that has survived generate.

### 1.6 — 29 downs to Q3

k = 29. Layers: six measured + POLICY `{62…39}` (39–46 plus 48).

```
payload = 5,354,169,659
tensor-complete = 1.5925550162910658
artifact-complete = 1.5926023006483170
down physical = 1.544645773663
MLP class = 1.319060005394
```

### 1.75 — 44 downs to Q3

k = 44. Layers: six measured + POLICY `{62…23}`.

```
payload = 5,875,308,017
tensor-complete = 1.7475634600036241
artifact-complete = 1.7476107443608753
down physical = 2.275522637367
MLP class = 1.562685626628
```

20 downs still HGRAVS01 (layers 1,2,4–14,16–22). Measured-hard six are all Q3.

### 2.0 — all downs Q3, then determined leftover

k = 64. All `down_proj` HGRAVU01 Q3. Core:

```
payload_core = 6,570,159,168
tensor-complete = 1.9542413870357960
artifact-complete = 1.9542886713930472
down physical = 3.2500251321231617   MEASURED mixed-q3mlp class
MLP class = 1.88751979154699
slack to 2.0 = 153,681,286 B
```

Fill slack with **measured-hard** remaining MLP only. Rice-up hold cosine, hardest first: L31 0.813, L47 0.833, L15 0.860, L3 0.860, L0 0.895. L63 up rice hold 0.970 — do not spend. Then one gate: L3 binary hold 0.838 is the worst measured gate.

| add | delta bytes MEASURED−MEASURED | 
|---|---:|
| up L31,47,15,3,0 → Q3 | 109,323,302 |
| gate L3 → Q3 | 23,674,899 |
| remaining slack | 20,683,277 |

Do not spend the last 20.7 MB on an unscored up. That would recreate the interpolator.

```
payload_fill = 6,703,157,369
tensor-complete = 1.9938006400385850
artifact-complete = 1.9938479243958362
gate physical = 1.1582265783758725
up physical  = 1.4408326177036062
down physical = 3.2500251321231617
MLP class = 1.9496947760675467
attn physical = 1.2877935788805008   unchanged rice
```

2.0 codec map:

| organ | n | codec | lock |
|---|---:|---|---|
| down all | 64 | HGRAVU01 Q3 g64 | named change A |
| up {31,47,15,3,0} | 5 | HGRAVU01 Q3 g64 | named change A |
| up other | 59 | HGRAVR02 2% | ACCEPTED |
| gate L3 | 1 | HGRAVU01 Q3 g64 | named change A |
| gate other | 63 | HGRAVB01 | ACCEPTED |
| attn all | 304 | HGRAVR02 2% | ACCEPTED (no lock) |
| embed, lm_head | 2 | HQ30UQ4 | ACCEPTED |
| small | 353 | f32v2 | named change B |
| island 3994 | 128 rows | f32v2 | named change C |

EXPECTATION: still incoherent. Attention remains rice (pack cosine 0.834–0.847, CITED `g1-sub15-native-gap.md:148-157`; descent attn_in rice hold_min 0.888). No measured variant holds 0.99 at that BPW. 2.0 is the first rung that **can** put every down on the Q3 hold. That is necessary, not sufficient.

---

## 6. On-disk packs vs rungs

Search: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b`. GENERATE.json present only on mixed-2p0-v1 (MLX expand vehicle, not native).

| artifact | complete BPW MEASURED | recipe (short) | nearest rung | Δ | native load today | generate |
|---|---:|---|---:|---:|---|---|
| mixed-sub15-v1 | 1.2910781930062503 | 0.848 MLP + rice attn + Q4 tables | 1.3 | −0.008922 | no catalog → eats reconstructed Q4 | INCOHERENT, expand-to-Q4 vehicle |
| **1.3–2.0 designed here** | 1.29287–1.99380 | §5 | — | — | needs A+B+C | none |
| mixed-2p0-v1 | 2.0855934079220506 | 0.848 MLP + Q4 attn/tables | 2.0 | +0.085593 | YES (lock satisfied) | native INCOHERENT (newlines); also MLX GENERATE.json. Backwards alloc. 2048-col tile confound on B/R |
| mixed-q4down-v1 | 2.9590429283570026 | down Q4, gate B, up R, attn Q4 | none (above 2.0) | +0.959 | REFUSED (down is Uniform, lock wants Hgravs) | **never** |
| mixed-floor-q7-v1 | 3.17681583579674 | 0.848 MLP + non-MLP Q7 | none | +1.177 | YES (same 64/64/64/659 split as 2p0) | **never** |
| mixed-floor-q8-v1 | 3.5405591522270545 | 0.848 MLP + non-MLP Q8 | none | +1.541 | YES | **never** |
| mixed-q3mlp-v1 | 3.6138647373176767 | MLP all Q3 + attn/tables Q4 | none | +1.614 | REFUSED (851× Uniform; lock wants B/R/S) | **never** |
| mixed-floor-q8-up10-v1 | 3.6299810080179515 | up rice 10% else like q8 | none | +1.630 | YES | **never** |
| uniform-q4-v1 G0 | 4.252735126866492 | all GEMV HQ30UQ4 | none | +2.253 | YES (uniform path) | COHERENT, 6/6 |

Five native packs in [2.959, 3.630] have never been generated. They are free evidence. None sits on a ladder rung.

mixed-sub15 is the only artifact on the approach to 1.3. Its packed ledger **is** the 1.3 base. Repairing it natively is still target A.

mixed-2p0 is near 2.0 in BPW and opposite in allocation. Do not generate it again to "test 2.0".

---

## 7. Lowest coherent rung

**EXPECTATION, not a finding.** No rung in {1.3, 1.4, 1.5, 1.6, 1.75, 2.0} is expected coherent.

Why, in one chain:

- 1.3–1.75 leave some or all `down_proj` at 0.132. MLP class stays at or near the 0.848 mix the contract calls a token-kill.
- 2.0 can uncrush every down to Q3 (first hold). Gate/up stay mostly binary/rice. Attention stays rice 1.288. Rice/Q2/Q3 attention have no measured 0.99 hold. Q4 attention is the last generate-proven cheap codec and costs +0.797 BPW from this base — more than the 2.0 increment.
- Therefore the first **existing** packed point that is even a coherence candidate is off-ladder: mixed-q3mlp-v1 at 3.6138647373176767 (MLP Q3 hold + attention Q4 proven), or mixed-q4down-v1 at 2.9590429283570026 (down Q4 + attention Q4, gate/up still cheap).

The published bracket "floor ∈ (2.0856, 4.2527]" is a statement about two confounded failures plus G0. It is not used as a location.

### Cheapest experiment that would confirm this

Do **not** pack a new 2.0 artifact first.

1. Named change A only (~40 lines: allow `Uniform` in the three match arms at `:958-1003`).
2. `Qwen38HybridWeights::load` + native greedy generate on **mixed-q3mlp-v1**. All HGRAVU01. No binary/CSR, so no 2048-col confound. bits=3 downs already route to simd3 (`:1402-1403`). Never generated.
3. Same harness, same prompts as `QWEN38_COHERENCE_FLOOR_BRACKETED.json` / G0 6-prompt oracle.

Outcomes:

| result | meaning |
|---|---|
| COHERENT | floor ≤ 3.614. Next cheapest: unlock + generate mixed-q4down-v1 (2.959) after also binding Uniform down. If that is COHERENT, then pack the §5 2.0-fill and generate it (first on-ladder candidate). If q4down is INCOHERENT, gate/up at 1.13/1.29 is still a kill and no ladder rung works. |
| INCOHERENT | floor > 3.614 or Q3-MLP+Q4-attn is the wrong split. Next: generate mixed-floor-q8-v1 (zero code change, lock already satisfied) to test whether 0.848 MLP still kills when attention is Q8. Then G0 remains the only proven coherent point. |

Zero-change generate of floor-q7/q8 is cheaper than step 1+2 but cannot confirm a coherent *ladder* rung: those packs keep the 0.848 MLP mix. They only test the "MLP 0.848 kills" claim.

Do not expand any of these to Q4/float before GEMV.

---

## 8. KILLS / REOPEN_IF

| ID | KILL | REOPEN_IF |
|---|---|---|
| K1 | Uniform Q3 everywhere as a "rung" | someone needs a quality floor, not a G1 target. Descent already named it 3.25 body / ~3.25 complete (`coherence_floor`) |
| K2 | Uniform Q4 everywhere | that is G0, 4.2527. Not a ladder step |
| K3 | Keep MLP at 0.848 and spend the increment on attention | mixed-2p0 backwards alloc. 1.5-with-0.848 forces attn ≤ 2.064157; nothing measured holds 0.99 there |
| K4 | Affine Q2 as the 1.75/2.0 down step | Q2 hold_min ≤ rice hold_min at +0.96 BPW. Ternary would reopen Q2's *budget* slot only if a native ternary codec exists |
| K5 | Hetero 2.0/1.5/1.2 tables | 58/64 layers interpolated. REOPEN_IF a determined per-layer score exists for all 64 (doctor replacement capture, not 256-token fit) |
| K6 | HGRAVS01 on attention | rank-160 fit not measured on attention; lock is r160_b3; down-only number 0.1316 is not transferable (`g1-bit-budget-accounting.md` §5.4) |
| K7 | Raise tables above Q4 before attention is Q4 | 9.45% of N; Q8 tables + MLP 0.8 forbids every incumbent attn codec (`g1-bit-budget-accounting.md` §8.3) |
| K8 | Treat sub15/2p0 INCOHERENT as a floor | both confounded. REOPEN_IF a native non-expand generate of *this* §5 allocation is incoherent |

---

## 9. What this lane did not measure

- Any generate, any Metal load, any TOKEN_NS. GPU forbidden.
- Isolated HGRAVS hold-cosine.
- Whether simd3 on a 5120×17408 Q3 down matches the factor-kernel numeric contract (layout yes; parity unrun).
- Exact catalog bytes of a not-yet-emitted 851+island HQ38M20 (ESTIMATED ± few hundred B).
- Channel 3994 capability effect. Energy statistic only. Capture underdetermined.

---

```
STATUS
IMPLEMENT_READY

CLAIMS
C1. Language N = 26895998464. Complete BPW = 8 * payload / N including headers. pack.rs:673-679. MEASURED elements from g1-bit-budget-accounting.md §4.
C2. Native reader accepts codecs 0/1/2/3 (HGRAVB01/R02/S01/U01 or HQ30UQ4) and refuses 4. MLP is name-locked to Binary/Residual/Hgravs. qwen38_hybrid_decode.rs:601-664, :958-1003.
C3. HGRAVU01 bits=3 is already dispatched via q80_hgravs01_factor_matvec_simd3 (K-complete, col+=256). Named change for Q3 MLP is the role lock, not a new shader. :1396-1475; q80_mixed_decode.metal:845-869.
C4. MLP 0.8480504639008466 is below its token floor (CITED contract + mixed-2p0/sub15 PACK_REPORT). Gate/up/down hold at Q3 with hold_min 0.9679 (CITED QWEN38_BPW_DESCENT.json coherence_floor). Packed q3mlp weight-cosine min 0.9652814877860332 MEASURED.
C5. Complete ≤ 1.5 with that MLP mix and Q4 tables forces attention ≤ 2.064157091228481 DERIVED. No measured native attn codec in that band holds 0.99. Q4 is the last generate-proven cheap attn/embed/lm_head codec.
C6. Affine Q2 is a quality regression vs rice. Do not use it as a rung step. CITED by_role_codec hold_min table §3.
C7. Heterogeneous 2.0/1.5/1.2 tables are not determined (58/64 interpolated). Not used. g1-heterogeneous-allocation.md:92-102.
C8. Rung allocations and tensor-complete BPW:
    1.30 → 1.2928720205923343  island 3994 only
    1.40 → 1.3962109836622572  10 downs Q3
    1.50 → 1.4995499482193904  20 downs Q3
    1.60 → 1.5925550162910658  29 downs Q3
    1.75 → 1.7475634600036241  44 downs Q3
    2.00 → 1.9938006400385850  64 downs Q3 + 5 measured-hard ups Q3 + L3 gate Q3
    DERIVED §5 from MEASURED class bytes + mixed-q3mlp 36208920 + mixed-2p0 per-down HGRAVS nbytes.
C9. No on-disk pack occupies a rung. sub15 is 0.008922 below 1.3. mixed-2p0 is 0.085593 above 2.0 and allocated backwards. Five packs in [2.959043, 3.629981] have no GENERATE.json. MEASURED directory listing + PACK_REPORTs.
C10. EXPECTATION: no ladder rung is coherent. Cheapest confirm: named change A + native generate mixed-q3mlp-v1 (3.6138647373176767, never generated, all Uniform, no 2048-col confound).

EVIDENCE
E1. crates/hawking-core/src/model/qwen38_pack.rs:673-679
E2. crates/hawking-core/src/model/qwen38_geometry.rs:20-52
E3. crates/hawking-core/src/model/qwen38_hybrid_decode.rs:601-664, :938-1003, :1116-1134, :1395-1475
E4. crates/hawking-core/shaders/q80_mixed_decode.metal:845-869 (simd3 tile)
E5. .../mixed-sub15-v1/PACK_REPORT.json complete_physical_bpw 1.2910781930062503 per_tensor_class
E6. .../mixed-2p0-v1/PACK_REPORT.json 2.0855934079220506 mlp_physical_bpw 0.8480504639008466 organ_breakdown; catalog.hq38m20 158970 B, 64 downs nbytes 1466360–1466365
E7. .../mixed-q3mlp-v1/PACK_REPORT.json 3.6138647373176767 mlp_physical_bpw 3.2500251321231617 replaced_strided_weight_cosine min 0.9652814877860332; every down nbytes 36208920; no GENERATE.json
E8. .../mixed-q4down-v1, mixed-floor-q7-v1, mixed-floor-q8-v1, mixed-floor-q8-up10-v1 PACK_REPORT.json BPWs 2.9590429283570026 / 3.17681583579674 / 3.5405591522270545 / 3.6299810080179515; no GENERATE.json
E9. receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json coherence_floor.hold_min_across_mlp_and_attn_in 0.9679; summary.by_role_codec; organs L0/3/15/31/47/63
E10. receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json (cited, confounded)
E11. workspace/superwave/g1/g1-bit-budget-accounting.md §4–§9
E12. workspace/superwave/g1/g1-sub15-native-gap.md §0–§4
E13. workspace/superwave/g1/g1-artifact-inventory.md §5
E14. workspace/superwave/g1/g1-heterogeneous-allocation.md:92-102
E15. workspace/superwave/g1/g1-doctor-tensor-map.md:53-73
E16. workspace/superwave/g1/g1-out-proj-forensics.md:17-35, :119-134
E17. This-lane python: parse HQ38M20; inversion rem_attn_bits/E_attn = 2.064157091228481; payload sums §5

CHANGES
created workspace/superwave/g1/g1-bpw-ladder.md
no tracked file modified

TESTS
test -s workspace/superwave/g1/g1-bpw-ladder.md
wc -l workspace/superwave/g1/g1-bpw-ladder.md
git status --porcelain

RISKS
R1. Island 3994 rests on a 256-token underdetermined capture. Capability effect unmeasured.
R2. simd3 numeric parity on a full 5120×17408 Q3 down is unrun.
R3. Catalog of a new pack ESTIMATED ≈ 158970 B. ±1 KB moves complete BPW by 3e-7.
R4. Named change A without named change B still cannot natively consume remaining Binary/Residual (2048-col bind).
R5. mixed-q3mlp load after A still needs bits=3 bound=3 on simd3; header says bits 3, bound implied 3.

UNRESOLVED
U1. Whether mixed-q3mlp generate is coherent. Cheapest experiment, §7.
U2. Isolated HGRAVS hold-cosine. Not required to accept this ladder.
U3. Determined scores for the 58 unscored layers. POLICY descending used; not claimed as error rank.
U4. G0 TOKEN_NS / TPS not remeasured (GPU lane).

NEXT
N1. Repair/load mixed-sub15 natively (target A). This ladder is unused if that lands coherent.
N2. If A fails: named change A, generate mixed-q3mlp-v1, then follow the table in §7.
N3. Do not pack a new 2.0 artifact until q3mlp / q4down have spoken.
```
