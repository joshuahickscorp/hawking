# G1 doctor recovery — stranded Qwen3.8 sensitivity + the capture hetero allocation actually needs

Lane: `39-doctor-recovery`. One new file. No GPU. No generate. No pack. No resident touch.

STATUS: IMPLEMENT_READY for the capture spec. Current 256-token / 6-layer doctor is **not** an allocation substrate.

Every number is labeled MEASURED (receipt or on-disk file), CITED (wave-1 report, not re-derived), or ESTIMATED (arithmetic on those).

---

## 0. Provenance

| object | value | label |
|---|---|---|
| stranded commit | `7c5f323d9913e6981b19aaa93026db05651bae10` | MEASURED `git rev-parse` |
| subject | `preserve: genesis-doctor-sensitivity-20260816-215243 lane result` | MEASURED `git show --format` |
| HEAD | `0fbf2e2a5e8972e5cb52435c5a2e6cb30b5238f9` | MEASURED |
| `merge-base --is-ancestor 7c5f323d9 HEAD` | exit 1 | MEASURED: **not an ancestor** |
| parent | `78d778a77ecea163a2706d684cbf075b155c65bd` (`Seal Q80…`) | MEASURED; **is** an ancestor of HEAD |
| doctor instrument at HEAD | absent | MEASURED `git cat-file -e HEAD:tools/condense/doctor_qwen38_sensitivity/ops.py` → missing |
| doctor receipts at HEAD | absent | MEASURED same for `receipts/ascent-2026-08-16/QWEN38_DOCTOR_SENSITIVITY_SUMMARY.json` |

The lane landed 11 files / 77213 insertions on a preserve commit that was never merged. Parent is on mainline; the science is not.

Recovered blobs (content hashes from the receipts themselves):

| blob | bytes | `content_sha256` |
|---|---:|---|
| `7c5f323d9:receipts/ascent-2026-08-16/QWEN38_DOCTOR_SENSITIVITY_SUMMARY.json` | 15698 | `bc07351926f9a685ae224f3798e563a56e0f811ed3bf1ac554b8409a0d51c83c` |
| `…/QWEN38_DOCTOR_SENSITIVITY_MAP.json` | 667463 | `55b407462052c825946691498fc77941239ee22e4344edb11f4701f6d2c6422c` |
| `…/QWEN38_DOCTOR_SENSITIVITY_CATALOG.json` | 1208000 | `98056735f3791635156fc3ab7011aac132804f94371073ab925a7e6cfd832462` |

Instrument (same commit): `tools/condense/doctor_qwen38_sensitivity/{__init__,__main__,capture,geometry,ops,sweep,weights}.py` + `tools/condense/tests/test_doctor_qwen38_sensitivity.py` (721 lines).

---

## 1. What was actually measured

### 1.1 Capture admitted (MEASURED)

On-disk, still present, **not** produced by the stranded commit:

```
/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1/capture-result.json
```

| field | value | source |
|---|---|---|
| schema | `hawking.ascension.qwen38_bf16_post_swiglu_activation_capture.v1` | capture-result.json |
| status | `CAPTURED_REAL_BF16_POST_NORM_HIDDEN` | same |
| source.model_dir | `…/qwen38-27b/bf16` | same |
| source.not_synthetic | true | same |
| source.forward | `mlx_lm.qwen3_5_text` | same |
| n_tokens | 256 | same |
| n_layers | 64 | same |
| hidden | 5120 | same |
| fit_kind | `real_routed_activation_capture` | same |
| wall_s | 14.967979082999591 | same |
| sha256_self | `fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512` | hashes hidden payload |
| file sha256 | `01db2f814fba99a1b7dac4668e30e20d69247ee3a4efa83b9ce4665718aedcbe` | `shasum -a 256` of the JSON |
| on-disk size | 320 MB (`du -sh`) = 64 × 256 × 5120 × 4 | MEASURED |

Prompts (token mass 57+60+68+61+10 = 256):

| slice | n | prompt |
|---|---:|---|
| 0..57 | 57 | `The capital of France is` |
| 57..117 | 60 | `Write a short greeting to a colleague.` |
| 117..185 | 68 | `def fibonacci(n):\n    if n < 2:\n        return n` |
| 185..246 | 61 | `In mathematics, the derivative of x squared is` |
| 246..256 | 10 | `Explain gravity in one sentence.` |

Doctor admission (`MAP.admission`): `ok=true`, `x_source=PARENT_BF16_REAL`, `n_prompts=5`. Notes record (1) X is the BF16 parent, not the 4.2527 vehicle; (2) `sha256_self` ≠ JSON bytes, not fatal.

**Q4-vehicle activations: ABSENT.** `ls …/qwen38-27b/` has `activation-capture-v1` only. Binding preserved: the instrument refuses mixed-2p0 / mixed-sub15 / Gaussian X (`capture.py` `CANDIDATE_UNDER_TEST_MARKERS`, `is_synthetic_gaussian`).

Schema name says `post_swiglu`. Stored width is 5120, not 17408. Site is post-norm hidden. CITED wave-1 `g1-doctor-tensor-map.md` §3.3; reconfirmed here from `per_layer["0"].path` → `hidden/L00.f32` and `hidden=5120`.

### 1.2 Sweep coverage (MEASURED from MAP + SUMMARY)

| | |
|---|---|
| swept layers | `{0, 3, 15, 31, 47, 63}` — same 6 as BPW descent |
| unswept layers | 58, named in the catalog, **UNMEASURED not projected** |
| head layers / stride | `{3, 63}` / 6 (plus last q/kv head) |
| bit ladder | `{8, 4, 3, 2, 1}` |
| probe | symmetric group-64 absmax RTN. `not_a_codec: true` |
| wall_s | 205.99686258400106 |
| catalog units (64L) | 2260 = 851 tensor + 64 layer + 1216 head + 64 channel + 64 outlier + 1 expert |
| units swept | 105 |
| GQA layers in catalog | 16: 3,7,11,…,63. Matches `qwen38_geometry.rs` `(layer+1)%4==0` at HEAD |

Counts (`SUMMARY.counts` = `MAP` tallies):

| bucket | n |
|---|---:|
| with a sensitivity number | 69 |
| UNDERDETERMINED | **0** |
| eval_thin | **51** |
| UNMEASURED_BEHAVIOUR | 30 |
| NO_COHERENT_BITS_ON_LADDER | 5 |
| expert | `NOT_APPLICABLE_DENSE_NO_EXPERTS` |
| floors | 8-bit: 38; 4-bit: 15; 3-bit: 10; 1-bit: 6 |

`MAP.units` status counter: `DETERMINED` 69 + `NO_COHERENT_BITS_ON_LADDER` 5 + `UNMEASURED_BEHAVIOUR` 30 + `NOT_APPLICABLE_DENSE_NO_EXPERTS` 1 = 105. All 104 non-expert determinations used `procedure=eval_weight_only`. Zero `fit_from_X`.

Capability gate (instrument, not generate): last-token greedy id through the **real** `lm_head` on all 5 prompts, plus mid-prompt positions. `holds` requires every last token **and** every selected mid to match. Cosine is recorded and is **never** sufficient (`capability_gate.cosine_alone = "NEVER sufficient"`).

Floor rule (`sweep.py` `_lowest_hold`): lowest ladder step such that **that step and every coarser step** hold. A 1-bit hold with a 3-bit miss is not a floor.

### 1.3 Tensor floors that were actually emitted

MLP (sensitivity_bits / status), 6 layers only:

| L | gate | up | down | down outliers (`|W|` top 1% @ 8, body swept) | down channels |
|---:|---:|---:|---:|---:|---:|
| 0 | **1** | **1** | **3** | **1** | 3 |
| 3 | 8 | 4 | 4 | 4 | 3 |
| 15 | 8 | 8 | 8 | 8 | 3 |
| 31 | 8 | 4 | 8 | 8 | **1** |
| 47 | 8 | 8 | 8 | 8 | 8 |
| 63 | 8 | 4 | 8 | 8 | 8 |

Attention tensors:

| unit | bits | status |
|---|---:|---|
| L0 `in_proj_qkv` | — | NO_COHERENT (8-bit cosine 0.99999, last-agree 0.8) |
| L0 `in_proj_z` | 8 | DETERMINED |
| L0 `out_proj` | 8 | DETERMINED (Q4 last-agree 0.4; 8-bit holds) |
| L3 `q` | 8 | DETERMINED |
| L3 `k` | 4 | DETERMINED |
| L3 `v`, `o` | — | NO_COHERENT |
| L15 `q,v,o` | 8 | DETERMINED |
| L15 `k` | 4 | DETERMINED |
| L31 `q,k,v` | 8 | DETERMINED |
| L31 `o` | 4 | DETERMINED |
| L47 all q/k/v/o | 8 | DETERMINED |
| L63 `q` | 4 | DETERMINED |
| L63 `k` | **3** | DETERMINED |
| L63 `v` | 8 | DETERMINED |
| L63 `o` | 4 | DETERMINED |
| `lm_head` | 8 | DETERMINED (last-token holds at 4; one mid flips; all-agree 0.9) |
| embed, all norms, A_log, dt, conv, in_proj_a/b | — | UNMEASURED_BEHAVIOUR (no residual path) |

Layer-wide (all residual-writing tensors quantized together):

| layer | bits | 8-bit cosine | 8-bit last-agree | note |
|---:|---:|---:|---:|---|
| 0 | 8 | 0.99996 | 1.0 | attn side DEGENERATE_NO_RECURRENCE |
| 3 | 8 | 0.99998 | 1.0 | real GQA residual |
| 15 | 8 | 0.99997 | 1.0 | |
| 31 | — | **0.999941** | **0.8** | NO_COHERENT: one last-token flip at 8 bits |
| 47 | — | **0.999941** | **0.8** | same |
| 63 | 4 | 0.99040 @ 4-bit | 1.0 | only layer that holds at Q4 |

Heads actually swept (14 of 1216 catalogued): L3 q/0=8, q/6=**1**, q/12=4, q/18=3, q/23=8; L3 kv/0=3, kv/3=4; L63 q/0=3, q/6=3, q/12=4, q/18=**1**, q/23=3; L63 kv/0=4, kv/3=8. All `rows_per_dim=1.0` (`fit_dim=256` head_dim, 256 rows).

NO_COHERENT five: `L0.in_proj_qkv`, `L3.v_proj`, `L3.o_proj`, `layer/31`, `layer/47`.

### 1.4 The 1-bit / 2-bit probe is the same operator (MEASURED defect)

`ops.py` `apply_probe_to_rows`:

```
qmax = float((1 << (int(bits) - 1)) - 1)
if qmax < 1.0:
    qmax = 1.0
```

bits=1 → `(1<<0)-1 = 0` → clamped to 1. bits=2 → `(1<<1)-1 = 1`. Identical qmax. MAP observations confirm: L0 gate 1-bit and 2-bit are the same float (`output_cosine=0.9822256181097496`, same ids, same `logit_rel_l2=0.15752`). Same collapse on L0 up, L0 down, L0 out_proj, lm_head, layer/0.

**KILLS** any ranking that treats “1-bit floor” as 3 bits removable vs Q4. The probe cannot tell 1 from 2. The six `sensitivity_bits=1` rows are a 2-level (qmax=1) hold, mislabeled.

### 1.5 Capability on L0 MLP is nearly vacuous (MEASURED)

L0 gate/up/down reference greedy ids through lm_head:

```
[220, 220, 220, 220, 220, 220, 220, 198, 220, 198]
```

220 is space, 198 is newline. “Holds at 1 bit” means “still emits space.” lm_head last-tokens are `[1596, 1596, 1596, 1596, 11553]` — four of five last positions share one id. This is why cosine 0.9999 still flips a token on L31/L47, and why a 1-bit L0 hold is not an allocation floor.

---

## 2. What it concluded (the FINDINGS list, verbatim substance)

From `SUMMARY.FINDINGS` (12 items) plus the ranked list. Paraphrase only where the JSON is a full sentence; the numbers are copied.

1. X is PARENT_BF16_REAL, 256 tokens, 5 prompts, non-Gaussian. Q4-vehicle capture ABSENT.
2. Cosine is not capability. Several 4-bit and 8-bit points have output cosine ≥ 0.99 while greedy ids move. (L31 layer: cosine 0.999941, last-agree 0.8.)
3. `sensitivity_bits` is the lowest step at which that step **and every coarser step** hold.
4. MLP sensitivity increases with depth. L0 down holds at 3; L3 at 4; L15/31/47/63 down only at 8 on this probe.
5. L0 gate and up hold at 1 bit **when quantized alone** (down stays full). Combined they are the cheapest large bits: 33,423,360 bytes each vs Q4. One-unit ablation, not a joint MLP pack.
6. L0 down `|W|` top-1% @ 8 + body at 1 HOLDS (33,089,126 bytes billed). Later-layer outlier recipes do not beat the tensor floor.
7. Attention heads are not uniform. L3 q/6 and L63 q/18 hold at 1; L3 q/0 and q/23 only at 8. KV at 3–8.
8. lm_head on L63 post-norm hidden: last-token ids hold at 4, a mid-prompt position flips, so the number is 8. Logit rel-L2 at 4 bits is 0.04591894070798136.
9. Layer-wide Q4 fails on L0/L3/L15. L63 holds at 4. L31 and L47 fail even at 8 with cosine 0.9999 — knife-edge tokens, not a claim that 8 bits is physically insufficient.
10. L0 `in_proj_qkv` never holds on the degenerate `v*silu(z)` path (8-bit cosine 1.000, agreement 0.9). A_log/dt/conv are UNMEASURED_BEHAVIOUR.
11. Expert granularity is NOT_APPLICABLE (dense).
12. Tensor-level evals are eval_thin (`256/5120=0.05`) but they are **weight-only evals, not fits**. Head/channel/outlier meet `rows_per_dim >= 1`. Zero UNDERDETERMINED numbers were emitted.

Cheapest-bits rank 1–3 (the only large-byte rows below Q4): L0 up 1-bit, L0 gate 1-bit, L0 down `|W|` outlier 1-bit. Then two 1-bit heads (983,040 bytes) and a 1-bit channel slice (52,224 bytes).

---

## 3. Which conclusions are still valid at HEAD

Geometry at HEAD (`crates/hawking-core/src/model/qwen38_geometry.rs`) still matches the instrument: 64 layers, hidden 5120, intermediate 17408, vocab 248320, GQA interval 4, o_proj 5120×6144, q 12288×5120, DeltaNet qkv 10240 / z 6144 / o 6144. Vehicle BPW in the receipt is `4.252735126866492`, the surviving G0 number.

### STILL VALID (use)

| conclusion | why it survives | evidence |
|---|---|---|
| BF16 parent sourcing is the correct X | Wave 1: earlier campaign wrecked by calibrating on a broken model. Instrument refuses mixed-2p0 / mixed-sub15 / Gaussian. Preserve this. | `capture.py` admission; contract; on-disk `source.model_dir` ends in `/bf16` |
| Cosine is not capability | Wave 1: `min_q4_cosine=1.0` is a fold of 402 Nones; L31/L47 8-bit cosine 0.999941 flips an id | MAP layer/31, layer/47 observations; `g1-baseline-audit` / contract |
| MLP hardens with depth | Descent hold-L2 and forensics agree: L0 down easy, L63 down hard (binary hold 0.730) | SUMMARY `down_proj_depth`; `QWEN38_BPW_DESCENT.json`; `g1-out-proj-forensics.md` L63 down Q3 0.9728 |
| Q4-vehicle X is still missing | Only `activation-capture-v1` exists | `ls` of the run dir |
| 6/64 layers, 51 eval_thin, 0 underdetermined-as-labeled | Arithmetic unchanged. 0 underdetermined **because procedure was eval_weight_only**, not because 256 rows determine a 5120-dim fit | MAP determinations; `ops.py` `unit_determination` |
| Dense, no experts | Still dense at HEAD | `qwen38_geometry.rs`; catalog `expert: NOT_APPLICABLE` |
| Knife-edge last-token ids | G0 Q4 generate is coherent (CITED). Doctor layer-Q4 “fail” is 5-prompt last+mid ids, not generate | MAP; `QWEN38_COHERENCE_FLOOR_BRACKETED.json` 4.2527 COHERENT |
| `eval_thin` flag + `rows_per_dim < 1` ⇒ do not fit | This **is** NS-014, encoded. The 92-row wreck was Q80, not Qwen3.8 | test `test_undersampled_fit_is_underdetermined`; NS-014 |
| L0 `|W|` island on down is a one-unit eval, not a generate floor | Wave 1: mixed-2p0 crushed MLP to 0.848 with Q4 attention and is INCOHERENT. Doctor said this itself (“one-unit ablation”) | SUMMARY finding 5; mixed-2p0 GENERATE |
| Channel 3994 / L0 lin_o kurtosis 149.36 | Independently remeasured by wave-1 doctor-tensor-map and forensics | `g1-doctor-tensor-map.md` §3.1; forensics `weight.excess_kurtosis 149.3577` |

### VALID AS A MEASUREMENT, NOT AS AN ALLOCATOR INPUT

| conclusion | keep the measurement | do not do this with it |
|---|---|---|
| L0 gate/up “1-bit hold” | The 2-level probe did not flip space/newline ids | Spend 3 bits of a 2.0 recipe on L0 gate/up. Hetero already did (`g1-heterogeneous-allocation.md` L0 MLP all 1). mixed-2p0 is the generate counterexample |
| L0 down `|W|` top-1% @ 8 + body 1 holds | Same vacuous ids | Treat `|W|` outliers as the correction. Forensics: on L0 out_proj, `|W|`∩`|X|` = 0/42; exact `|W|` 42 cols → 0.9533 (no move) |
| Some GQA heads hold at 1 | 14 heads, rpd=1.0, unit residual → lm_head | Allocate per-head bits from a stride-6 sample of 2/16 GQA layers |
| lm_head floor 8 because a mid flips | 4-bit last-agree=1.0, all-agree=0.9, logit_rel=0.0459 | Hetero put lm_head at 3. That **contradicts** this floor if doctor capability is law. Neither is generate |
| L0 `in_proj_qkv` NO_COHERENT | Degenerate `v*silu(z)` path, 8-bit cosine 0.99999 | Call DeltaNet in-proj incompressible. Forensics Q4 output cosine 0.9961, Q3 0.9795 on the same tensor |
| Layer-wide Q4 fails L0/L3/L15 | True on this 10-position id test | Call G0 Q4 incoherent. Generate says otherwise |

### DEAD / SUPERSEDED

| conclusion | status | REOPEN_IF |
|---|---|---|
| Use this ranking as the hetero bit table | KILLS | A native generate of that table vs the Q4 oracle (GPU lane) |
| 1-bit is a distinct rung on this probe | KILLS | Probe with a real 1-bit quantizer (`qmax` not clamped onto the 2-bit operator) |
| `|W|` outlier extraction is the cheap MLP patch | KILLS as a transfer to attention; L0-down-only is untested at generate | Isolated L0-down-outlier pack, native generate |
| Interpolation of 58 layers from these 6 is a measurement | KILLS | 64-layer eval on the capture in §5 |
| act_colscale from this capture is a legal fit | KILLS as a 6144-parameter use | See §4 and §6. `n_fit >= 6144` on mixer-site X, then re-score |

### The wave-1 contradiction, resolved enough to specify a capture

- Density probe, L0 `out_proj`, `HGRAVU01_q4_g64`: output cosine **0.9922374383267348**.
- Same tensor, `HGRAVU01_q4_g64_act_colscale`: **0.9186496062432181**. Note: “column scales from real X_fit RMS; folded at pack time”. `n_x_rows=256`, `x_site=real_derived_v_silu_z_from_fused_in_proj_qkvz`, in-dim 6144.
- Forensics: exact-preserve 42 hottest `|X|` columns, Q3 the rest: **0.97616**. Exact-preserve 42 fattest `|W|` columns: **0.95331**, overlap 0.

These are not two measurements of the same operator. Folding column RMS into W **destroys**. Exacting the hot `|X|` columns **helps**. The 256×6144 scale vector is also `rows_per_dim = 256/6144 = 0.0417` if treated as one joint fit — the Q80 92/2048 = 0.0449 failure mode on a different model.

Doctor never ran act_colscale. It ran `|W|` top-1% on **down_proj**. Do not cite the doctor ranking as a vote in that contradiction.

---

## 4. Why hetero allocation is not yet defensible

`g1-heterogeneous-allocation.md` built the 2.0 / 1.5 / 1.2 tables on:

- the same 256-token BF16 hidden capture
- the same 6 layers `{0,3,15,31,47,63}`
- `fit_n=192`, `hold_n=64`
- output-space scores for gate/up/down/attn_in
- **weight-space only** for `attn_out` (in-dim 6144 ≠ captured 5120)
- linear interp across the other 58 layers
- Gravity **GPT-OSS** class priors, not a Qwen3.8 Jacobian
- no Hessian, no Q4-vehicle X

That is a screen. It is not a determined fit. A 192-row fit of anything whose dimension is 5120 / 6144 / 17408 is underdetermined by NS-014, regardless of how smooth J looks.

Current `rows_per_dim` (MEASURED):

| site / claim | n | dim | rpd | vs Q80 92/2048 = 0.0449 |
|---|---:|---:|---:|---|
| hidden GEMV eval | 256 | 5120 | 0.0500 | same class |
| out_proj eval | 256 | 6144 | 0.0417 | **worse than the wreck** |
| down_proj eval | 256 | 17408 | 0.0147 | 3× worse |
| act_colscale as a 6144-vector | 256 | 6144 | 0.0417 | same class |
| per-column RMS as 1-D stats | 256 | 1 | 256 | determined (statistic only) |
| GQA head (fit_dim=256) | 256 | 256 | 1.0 | determined as a slice |
| HGRAVS01 r160 if anyone fits it | 192 | 160 | 1.20 | barely; descent used 192 |

Eval_thin scores may be published. They must not set a scale, an SVD, an AWQ fold, or a 58-layer interpolation that is then packed.

---

## 5. Capture that would make hetero allocation defensible

Do **not** run this from this lane. GPU / MLX forward / resident-device lock belong to the serialized GPU lane. BF16 weights are ~53.8 GB; peak RSS will exceed this lane’s 20 GB cap and will contend with the resident G0.

### 5.1 Vehicles (two forwards, identical tokens)

| stream | source | why |
|---|---|---|
| A. parent | `…/qwen38-27b/bf16` (`PocketAiHub/Qwen3.8-27B-Abliterated-MLX`, base `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`) | the only legal fit source |
| B. vehicle | `…/qwen38-27b/uniform-q4-v1` **native reader**, no expand-to-float | ranks must be checked against the live genome; absent today |

Refuse: mixed-2p0, mixed-sub15, any candidate under test, synthetic Gaussian (kurtosis in 3±1.25 and std in (0.5, 2)). Admission = current `admit_capture`. `not_synthetic` is not enough; the statistic overrides the flag.

### 5.2 How many tokens

Let `HOLD_FRAC = 0.25` (already the instrument default). Fit rows = `floor(0.75 * N)`. Hold rows = `N - fit`.

| purpose | fit_dim | min N so `n_fit >= fit_dim` | min N so eval_thin is false with no split |
|---|---:|---:|---:|
| hidden GEMV / lm_head X | 5120 | **6827** | 5120 |
| mixer `out_proj` / `o_proj` X | 6144 | **8192** | 6144 |
| down_proj post-SwiGLU X | 17408 | **23211** | 17408 |
| group-64 output-MSE scale (1 scale / group) | 64 | 86 | 64 |
| rank-r ≤ 256 (HGRAVS01-class) | r | ceil(r/0.75) | r |
| per-channel RMS | 1 | 16 (MIN_EVAL_ROWS) | 16 |

**Required N for a determined full-width hetero substrate: 23216** (gives `n_fit = 17412 >= 17408`).

Anything smaller **must** declare which organs are still underdetermined and must not fit those organs. A 2048-token “census” (`n_fit=1536`) determines group-64 scales and rank≤1536, and gives a 64-layer **eval** (still thin vs 5120/6144/17408). It does **not** legalize AWQ / full-dim column-scale / full-rank down_proj.

N=256 remains legal only for: evals flagged `eval_thin`, 1-D channel stats, and slices with `fit_dim <= 256`.

### 5.3 Prompt distribution

Current set is 5 prompts, max length 68, one prompt of 10 tokens, last-token ids collide. Mixer-site X on 68-token sequences never stresses DeltaNet recurrence or GQA softmax.

Target **23216 tokens**, **≥ 64 sequences**, **≥ 32 prompts**, min length 32 (drop 10-token mass), max length 2048.

Token-mass mix (not prompt count):

| mass | class | why |
|---|---:|---|
| 25% | prose / encyclopedia | last-token diversity |
| 20% | code | current set is one fibonacci stub |
| 15% | math / STEM | matches one existing prompt class |
| 15% | instruction / chat, multi-turn templates | stop four-of-five last ids being equal |
| 10% | long sequences 512–2048 | recurrence + GQA; currently max 68 |
| 10% | multilingual / mixed script | vocab tail, lm_head |
| 5% | numbers / punctuation / adversarial short | knife-edge tokens like L31/L47 |

Holdout is a **sequence** holdout, not a random row shuffle that leaks tokens from the same prompt into fit and hold. Instrument today uses `np.random.permutation` of rows (`ops.py` `holdout_split`). That leaks. New capture must split by prompt id.

Same token ids, same slices, on vehicle B.

### 5.4 Activation sites, every layer 0..63

Record at the **in-dim of the organ that will consume them**. Do not reuse post-norm hidden as out_proj X or as down_proj X. That is the hole that forced descent `attn_out` to `quality_space=weight_only`.

| site id | width | layers | consumed by | today |
|---|---:|---|---|---|
| `post_input_norm` | 5120 | 64 | q/k/v, in_proj_qkv/z/a/b, gate, up | YES (this is the 256×5120 dump) |
| `post_attn_norm` | 5120 | 64 | gate, up (MLP input after mixer residual) | NO |
| `post_swiglu` | 17408 | 64 | down_proj. `silu(x@Wg.T)*(x@Wu.T)` stored, not reconstructed later | NO (schema name lied) |
| `mixer_x` | 6144 | 64 | out_proj / o_proj. DeltaNet: **true** recurrent mix, not `v*silu(z)`. GQA: `repeat(v)*sigmoid(q_gate)` after softmax | NO (probes used a derived proxy) |
| `final_norm` | 5120 | 1 | lm_head. Must be confirmed final-norm, not L63 post-norm | NO (`qwen38_L63_post_norm_hidden_NOT_confirmed_final_norm`) |
| `residual_pre_norm` | 5120 | 64 | write-gain / residual-proxy (optional; forensics used post-norm as stand-in) | NO |

Also store, per prompt: token ids, offsets, positions. Per layer: rms, mean_abs, sha256 of the site file.

Do **not** need: vision, embed rows (gather), A_log/dt (already f32, no X-fit).

### 5.5 Disk (ESTIMATED from geometry; 256-token hidden dump MEASURED at 320 MB)

Per token, all required sites: `64*(5120+5120+17408+6144)+5120 = 2,167,808` f32 = 8,671,232 bytes.

| N | f32 | f16 | note |
|---:|---:|---:|---|
| 256 (today, hidden only) | 0.336 GB MEASURED | — | 64×256×5120×4 |
| 256 all sites | 2.22 GB | 1.11 GB | what today would have been |
| 2048 census | 17.76 GB | 8.88 GB | 64-layer eval; fits still thin |
| 8192 (mixer determined) | 71.03 GB | 35.52 GB | down still thin |
| **23216 (down determined)** | **201.31 GB** | **100.66 GB** | legal full-width substrate |
| site-split N (6827 / 8192 / 23216) | 134.38 GB | 67.19 GB | store each site at its own N |

**Write f16, site-split N, stream per layer, do not keep the cube resident.** Peak extra activation RAM: one layer × one microbatch. BF16 weights still ~53.8 GB.

Vehicle B twin: same bytes again if stored. Prefer scoring B live and keeping only the parent cube, **or** store B at census N=2048 (8.88 GB f16) for rank-correlation of floors.

### 5.6 Wall time (ESTIMATED from MEASURED 14.967979 s / 256 tokens hidden-only)

Linear scale: 23216/256 × 14.97 s = **1357 s ≈ 23 min** parent forward+write, if sequences stay ~60 tokens and sites add only write cost.

GQA is quadratic. 10% mass at length 2048 (≈12 sequences) will dominate. Bound: **23 min (short prompts, ESTIMATED) to a few hours (long GQA, ESTIMATED)**. Vehicle B similar. SSD write of 100 GB is minutes, not the limiter.

Box constraint: 96 GB unified. Resident G0 Q4 payload 14.3 GB + BF16 53.8 GB ≈ 68 GB before activations. GPU lane must decide whether to pause the resident. This lane must not.

### 5.7 What the new capture unlocks, and what it still is not

Unlocks:

- 64-layer hold_output_rel_l2 for gate/up/down/**out_proj** (mixer X present) / attn_in. No interp.
- Determined group-64 output-MSE scales (`n_fit >= 64`, actually >= 17412).
- Determined rank-r ≤ 17412 (so r160 is no longer rank-clamped).
- Determined full-dim column stats **and** a legal 6144-vector / 17408-vector fit if someone still wants one.
- Paired Q4-vehicle ranks: if parent-X floors ≠ vehicle-X floors, allocation follows the vehicle (live genome) and the parent remains the codec teacher.

Still not generate. Still not a complete-token claim. Pack + native generate remains the gate (`mixed-2p0` is the warning).

---

## 6. Adequacy test (the thing that must not go undetected again)

This is the law. A pretty cosine does not override it. NS-014, `unit_determination`, and `test_undersampled_fit_is_underdetermined` already state it; hetero ignored the fit half.

### 6.1 Definitions

```
n_rows     = captured rows at this site after prompt-level holdout split
n_fit      = rows eligible to estimate parameters
n_hold     = rows eligible only to score
n_prompts  = number of sequences (not tokens)
fit_dim    = number of free parameters taken from X
procedure  = eval_weight_only | fit_from_X
rpd        = n_fit / fit_dim     if fit_dim > 0 else +inf
```

`fit_dim` is the dimension **being fitted**, not the tensor’s element count:

| action | fit_dim |
|---|---:|
| score `||X W^T − X Ŵ^T||` with Ŵ from weights only (absmax RTN, binary, q3, q4) | in_dim, but procedure = **eval_weight_only** |
| per-column RMS / energy / percentile | **1** (independent 1-D stats) |
| per-group output-MSE scale (one s per group of 64) | **64** if groups share X, else **1** per group with the group’s 64 columns needing `n_fit >= 64` |
| AWQ / fold a length-`in_dim` scale into W | **in_dim** (5120 / 6144 / 17408) |
| HGRAVS01 rank r | **r**, and r must not be `min(budget, n_fit)` |
| full-dim Hessian / Fisher diagonal | **in_dim** |
| interpolate an unswept layer from neighbours | **not a measurement** → UNMEASURED, not DETERMINED |

### 6.2 Gate (must all pass or the number is refused)

Copied from `7c5f323d9:tools/condense/doctor_qwen38_sensitivity/ops.py` `unit_determination` and tightened for allocation:

1. `n_prompts >= 3` else UNDERDETERMINED. (Instrument `MIN_PROMPTS`.)
2. `n_rows >= 16` else UNDERDETERMINED. (`MIN_EVAL_ROWS`.)
3. X admitted: `PARENT_BF16_REAL` or `COHERENT_Q4_VEHICLE`, `not_synthetic`, not Gaussian, not a candidate under test.
4. Holdout is by **prompt**, `n_hold >= max(4, ceil(0.25 * n_rows))`, and hold rows are not in the fit.
5. **If `procedure == fit_from_X`: require `n_fit >= fit_dim`.** Else UNDERDETERMINED, `emit_sensitivity=false`, do not pack a scale/factor/bit from it.
6. If granularity is a slice (`attention_head`, `channel`, `outlier`) claiming a sliced fit: same `n_fit >= fit_dim` on the **sliced** dim.
7. `eval_weight_only` with `n_rows < in_dim` may emit, but **must** set `eval_thin=true`. An `eval_thin` number must not be the sole input to a bit assignment on an unswept layer.
8. Interpolated layers are UNMEASURED. They do not enter J.
9. `rank = min(budget, n_fit)` is a silent starve. Refuse. (NS-014 `why_it_failed`.)

The hardcoded wreck:

```
rows_per_dim(92, 2048) == 0.044921875  → UNDERDETERMINED
```

That is `test_undersampled_fit_is_underdetermined` and NS-014 `prior_q80_run: "median 92 rows against 2048 dims; every score garbage"`. Qwen3.8 today: `256/6144 = 0.04167`, `256/17408 = 0.01471`. **Worse than the wreck**, for any fit of those dims.

### 6.3 What “determined hetero” means in one line

A bit table is DETERMINED only if every organ it assigns was either (a) scored `eval_weight_only` on that organ’s real site with a published `eval_thin` bit, **and** that organ was actually swept (no interp), or (b) fitted with `n_fit >= fit_dim` on that site. The 2.0 table in `g1-heterogeneous-allocation.md` fails (a) on 58 layers and on every `out_proj`, and fails (b) on any X-derived scale.

REOPEN_IF a 23216-token two-vehicle capture passes this gate on all GEMV classes and a native generate of the resulting table is scored against the Q4 oracle.

---

## 7. Evidence excerpts

### 7.1 Ancestry (MEASURED command)

```
$ git rev-parse HEAD
0fbf2e2a5e8972e5cb52435c5a2e6cb30b5238f9

$ git rev-parse 7c5f323d9
7c5f323d9913e6981b19aaa93026db05651bae10

$ git merge-base --is-ancestor 7c5f323d9 HEAD; echo $?
1

$ git merge-base --is-ancestor 7c5f323d9^ HEAD; echo $?
0

$ git log --oneline 7c5f323d9 -2
7c5f323d9 preserve: genesis-doctor-sensitivity-20260816-215243 lane result
78d778a77 Seal Q80, delete its weights, keep the raw-weights recipe

$ git cat-file -e HEAD:tools/condense/doctor_qwen38_sensitivity/ops.py; echo $?
1
$ git cat-file -e HEAD:receipts/ascent-2026-08-16/QWEN38_DOCTOR_SENSITIVITY_SUMMARY.json; echo $?
1
```

### 7.2 SUMMARY head + counts (MEASURED `git show 7c5f323d9:…SUMMARY.json`)

```
"schema": "hawking.doctor.qwen38_sensitivity_summary.v1"
"date": "2026-08-16"
"admission.x_source": "PARENT_BF16_REAL"
"admission.n_prompts": 5
"admission.n_tokens": 256
"admission.fit_kind": "real_routed_activation_capture"
"coverage.swept_layers": [0, 3, 15, 31, 47, 63]
"coverage.n_units_swept": 105
"coverage.n_units_catalog_64L": 2260
"coverage.wall_s": 205.99686258400106
"counts.with_sensitivity_number": 69
"counts.underdetermined": 0
"counts.eval_thin": 51
"counts.unmeasured_behaviour": 30
"counts.no_coherent_bits_on_ladder": 5
"counts.floors": {"8": 38, "1": 6, "3": 10, "4": 15}
"content_sha256": "bc07351926f9a685ae224f3798e563a56e0f811ed3bf1ac554b8409a0d51c83c"
```

### 7.3 Adequacy constants + the 92-row test

`7c5f323d9:tools/condense/doctor_qwen38_sensitivity/geometry.py`:

```
BIT_LADDER: tuple[int, ...] = (8, 4, 3, 2, 1)
PROBE_GROUP = 64
MIN_ROWS_PER_DIM = 1.0
MIN_EVAL_ROWS = 16
MIN_PROMPTS = 3
HOLD_FRAC = 0.25
MIN_HOLDOUT_ROWS = 4
COHERENT_VEHICLE_BPW = 4.252735126866492
```

`7c5f323d9:tools/condense/doctor_qwen38_sensitivity/ops.py` `unit_determination` (docstring + fit branch):

```
A fit from X needs rows_per_dim >= 1 (the 92-row / 2048-dim campaign
was 0.045 and the whole result was invalid). Head/channel units use
the sliced width as fit_dim. Weight-only RTN is an eval, not a fit;
it still needs MIN_EVAL_ROWS and >= 3 prompts, and is flagged
eval_thin when rows_per_dim < 1.
…
if procedure == "fit_from_X" and rpd < MIN_ROWS_PER_DIM:
    return DeterminationReport(status="UNDERDETERMINED", … emit_sensitivity=False)
```

`7c5f323d9:tools/condense/tests/test_doctor_qwen38_sensitivity.py:223-235`:

```
def test_undersampled_fit_is_underdetermined() -> None:
    report = unit_determination(
        n_rows=92, fit_dim=2048, n_prompts=3,
        procedure="fit_from_X", granularity="tensor",
    )
    assert report.status == "UNDERDETERMINED"
    assert report.emit_sensitivity is False
    assert report.rows_per_dim == pytest.approx(92 / 2048)
```

`7c5f323d9:tools/condense/tests/test_doctor_qwen38_sensitivity.py:680-681`:

```
assert rows_per_dim(92, 2048) == pytest.approx(0.044921875)
```

HEAD `receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json` NS-014:

```
"id": "NS-014"
"mechanism": "Fit a rank-r or full-dim codec on fewer captured rows than the fitted dimension, then trust the score"
"class": "REFUTED"
"models": ["q80", "dsv4f"]
"what_was_measured.prior_q80_run": "median 92 rows against 2048 dims; every score garbage"
"why_it_failed": "rows < input_dim is underdetermined for a full-rank score; rows < rank is underdetermined for a rank-r score. rank = min(budget, n_fit_rows) silently starves the codec and the score is not the codec's score."
"retry_when": "never trust a score from an underdetermined fit. Re-score only when n_fit >= the claimed rank (and, for a full-dim claim, n_fit >= dim), with rank not clamped."
```

The 92-row campaign is Q80. Qwen3.8’s 256/6144 and 256/17408 are the same failure class, not yet named on this model because the doctor labeled those rows `eval_weight_only`.

### 7.4 Capture on disk

```
$ shasum -a 256 …/activation-capture-v1/capture-result.json
01db2f814fba99a1b7dac4668e30e20d69247ee3a4efa83b9ce4665718aedcbe

$ python3 -c '…print keys…'
schema: hawking.ascension.qwen38_bf16_post_swiglu_activation_capture.v1
status: CAPTURED_REAL_BF16_POST_NORM_HIDDEN
n_tokens: 256  n_layers: 64  hidden: 5120
wall_s: 14.967979082999591
sha256_self: fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512
per_layer[0].rms: 0.09979002177715302
per_layer[63].rms: 1.166797161102295

$ du -sh …/activation-capture-v1
320M

$ ls …/qwen38-27b | rg -i captur
activation-capture-v1
```

### 7.5 L0 gate 1-bit == 2-bit; vacuous ids; lm_head mid-flip

MAP `tensor/…layers.0.mlp.gate_proj.weight` (eval_thin, rpd=0.05, DETERMINED, sens=1):

```
b=2 cos=0.9822256181097496 holds=True last=1.0 all=1.0 logit_rel=0.15752041956489005
    ids_ref=[220,220,220,220,220,220,220,198,220,198]
    ids_hat=[220,220,220,220,220,220,220,198,220,198]
b=1 cos=0.9822256181097496 holds=True last=1.0 all=1.0 logit_rel=0.15752041956489005
    ids identical
```

MAP `tensor/…lm_head.weight` (sens=8):

```
b=4 cos=0.9989481112047172 holds=False last=1.0 all=0.9 logit_rel=0.04591894070798136
    ids_ref=[1596,1596,1596,1596,11553,55404,888,6607,30246,25]
    ids_hat=[1596,1596,1596,1596,11553,264,888,6607,30246,25]
```

MAP `layer/31` (NO_COHERENT):

```
b=8 cos=0.9999410275955831 holds=False last=0.8 all=0.9 logit_rel=0.011430208231682119
    ids_ref=[96869,192369,72705,191301,133773,17646,95737,101668,14234,151353]
    ids_hat=[96869,192369,72705,191301,95727,17646,95737,101668,14234,151353]
```

### 7.6 Act-colscale destruction (HEAD receipt, not re-run)

`receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_PROBE.json` tensor `qwen38.L0.linear_attn.out_proj`, `n_x_rows=256`, `x_site=real_derived_v_silu_z_from_fused_in_proj_qkvz`:

```
HGRAVU01_q4_g64            output_cosine 0.9922374383267348
HGRAVU01_q4_g64_act_colscale output_cosine 0.9186496062432181
HGRAVU01_q3_g64            output_cosine 0.9531713530055139
HGRAVU01_q3_g64_act_colscale output_cosine 0.8425799898189834
```

Forensics follow-up (CITED `g1-out-proj-forensics.md`): exact 42 `|X|` cols → 0.97616; exact 42 `|W|` cols → 0.95331; overlap 0.

### 7.7 Hetero interpolation (CITED, not re-derived)

`workspace/superwave/g1/g1-heterogeneous-allocation.md:68-102`:

```
Capture: 256 real BF16 post-norm hiddens, fit_n=192, hold_n=64, layers {0,3,15,31,47,63}.
attn_out is weight-space only (out_proj in-dim 6144 ≠ captured 5120).
unmeasured MLP layers → linear interp in layer index among {0,3,15,31,47,63}
```

---

## 8. What this lane did not do

- Did not merge 7c5f323d9. Science recovered by `git show`, not by landing the instrument.
- Did not run capture, generate, Metal, or numpy on the 27B tensors.
- Did not re-derive G0 BPW / TPS / roofs / mixed-2p0 incoherence.
- Did not resolve generate-level floors. Doctor capability ≠ generate.

Cheapest next experiment (GPU lane): the §5 capture at N=23216, f16, site-split, two vehicles; then a 64-layer eval_weight_only ladder with the §6 gate wired to refuse interp and refuse `n_fit < fit_dim`. Do not pack from the 256-token cube.

---

## Completion report

```
STATUS
IMPLEMENT_READY

CLAIMS
1. Commit 7c5f323d9 is not an ancestor of HEAD; parent 78d778a77 is. Instrument + three receipts live only on that commit. MEASURED. Evidence: §7.1.
2. Doctor swept 105 of 2260 catalog units on layers {0,3,15,31,47,63}, 256 BF16 post-norm tokens, 5 prompts, wall 206.0 s. 69 floors, 51 eval_thin, 0 underdetermined, 30 unmeasured, 5 no-coherent. MEASURED. Evidence: SUMMARY counts; MAP admission; §7.2.
3. X is PARENT_BF16_REAL. Q4-vehicle capture is absent. BF16 sourcing is correct and must be preserved. MEASURED. Evidence: capture-result.json source.model_dir; ls of the run dir; admit_capture refusals.
4. Tensor evals are eval_weight_only at rpd 0.05 / 0.0417 / 0.0147. Those are not fits. A fit of dim 6144 or 17408 on 256 rows is underdetermined (rpd 0.0417 / 0.0147), worse than Q80 92/2048 = 0.0449. MEASURED. Evidence: MAP determinations; NS-014; test_undersampled_fit_is_underdetermined.
5. 1-bit and 2-bit rungs are the same RTN operator (qmax clamped to 1). L0 gate 1-bit and 2-bit observations are bitwise-identical floats. KILLS “1-bit floor” as 3 bits removable. MEASURED. Evidence: ops.py qmax; MAP L0 gate observations §7.5.
6. L0 gate/up “hold at 1 bit” is a space/newline id hold on a one-unit ablation. Not a joint-MLP generate floor. mixed-2p0 already killed the joint crush. MEASURED / CITED. Evidence: §7.5 ids; QWEN38_COHERENCE_FLOOR_BRACKETED.json.
7. Still valid at HEAD: cosine≠capability, MLP hardens with depth, dense/no-experts, geometry, G0 BPW 4.252735126866492, 6-layer coverage hole, missing Q4 X, NS-014 adequacy law, channel-3994 / L0 kurtosis 149.36 (remeasured by wave 1). Evidence: §3 table.
8. Dead as allocator input: cheapest-bits ranking, |W| outlier as the correction, head-stride-6 1-bit heads, layer-Q4 “fail” vs G0 generate, degenerate in_proj_qkv NO_COHERENT as a codec floor, 58-layer interp. Evidence: §3; hetero.md:68-102; forensics overlap 0/42.
9. Hetero 2.0/1.5/1.2 tables are not determined. They interpolate 58 layers and score attn_out in weight space because mixer X was never stored. CITED. Evidence: g1-heterogeneous-allocation.md:68-102, C3.
10. Defensible capture: N=23216, prompt-level 25% hold, ≥64 sequences, mass mix in §5.3, sites {post_input_norm, post_attn_norm, post_swiglu 17408, mixer_x 6144, final_norm} × 64, parent BF16 + Q4-vehicle twin, f16 site-split ≈ 67 GB, wall ESTIMATED 23 min–hours, GPU lane only. Adequacy = §6: n_fit >= fit_dim or the number is refused. IMPLEMENT_READY. Evidence: §5–§6 arithmetic; 14.967979 s / 256 tok MEASURED.

EVIDENCE
- git 7c5f323d9 / HEAD / merge-base exit codes §7.1
- 7c5f323d9:receipts/ascent-2026-08-16/QWEN38_DOCTOR_SENSITIVITY_{SUMMARY,MAP,CATALOG}.json
- 7c5f323d9:tools/condense/doctor_qwen38_sensitivity/{capture,geometry,ops,sweep}.py
- 7c5f323d9:tools/condense/tests/test_doctor_qwen38_sensitivity.py:223-235,680-681
- HEAD receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json NS-014
- HEAD receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_PROBE.json qwen38.L0.linear_attn.out_proj
- …/activation-capture-v1/capture-result.json + shasum + du
- crates/hawking-core/src/model/qwen38_geometry.rs:22-52
- workspace/superwave/g1/g1-heterogeneous-allocation.md:68-102
- workspace/superwave/g1/g1-out-proj-forensics.md follow-up 0.97616 / overlap 0
- workspace/superwave/g1/g1-doctor-tensor-map.md §3.1–3.3

CHANGES
workspace/superwave/g1/g1-doctor-recovery.md (this file). No other path touched.

TESTS
see end of lane message

RISKS
- N=23216 f16 is ~67 GB plus a 54 GB BF16 load. Unified memory 96 GB. Resident must be paused or the capture will evict it. GPU lane owns that call.
- True DeltaNet mixer_x requires a recurrent hook the current MLX dump does not have. A derived v*silu(z) proxy is labeled DEGENERATE and must not be silently reused.
- Vehicle-B capture is a native Q4 forward. Expand-to-float is forbidden as a source.
- Long GQA sequences make the 23 min linear estimate a lower bound only.

UNRESOLVED
- Generate floors. Doctor capability ≠ token identity.
- Whether parent-X and Q4-X rank the same organs. No vehicle capture.
- Whether mixer_x hot columns stay disjoint from |W| once the true recurrent mix is stored (forensics REOPEN_IF).
- 1-bit as a real rung: needs a probe that is not qmax-clamped onto 2-bit.

NEXT
GPU lane: §5 capture, §6 gate wired before any scale or bit table. Do not pack from the 256-token cube. Do not merge 7c5f323d9 as code unless a separate lane wants the instrument; the receipts are the science.
```
