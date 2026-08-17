# G1 arch — Qwen3.8 gravity / doctor / pack run ledger

Lane: `24-arch-gravity-runs`. HEAD: `2eee9a004`. No GPU, no pack, no mutate of tracked files.
Artifact root on this machine (not in this sparse worktree, not in git):

`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/`

A path missing from this worktree is not evidence of absence. Every "absent from HEAD" claim below was checked with `git cat-file -e HEAD:<path>` and, where relevant, `git show <preserve-commit>:<path>` plus an on-disk `ls`.

Status vocabulary used in the ledger:

| class | meaning |
|---|---|
| NEVER_PACKED | recipe written; no complete artifact |
| PACKED_COMPLETE | pack wrote a sealed catalog / segments; pack process finished |
| LANE_DIED_ARTIFACT_LIVES | reporting lane died; artifact is on disk and internally consistent |
| EVAL_NATIVE | loaded by `ascension_qwen38_hybrid_greedy` consuming the packed codecs |
| EVAL_EXPAND | loaded by reconstruct-to-Q4 / MLX overwrite (confounded) |
| EVAL_PASSED | generate produced sealed / on-topic English |
| EVAL_FAILED | generate collapsed (cycle, EOS-only, punctuation salad) |
| NEVER_EVALUATED | artifact on disk; no generate receipt |
| PRESERVE_ONLY | exists on a side branch, not ancestor-merged into `2eee9a004` |
| SCREEN_ONLY | organ cosine / RTN probe; not a complete-token generate |

Numbers are labelled MEASURED (bytes on disk / JSON field), RECEIPT (prior GPU or generate run, not re-run here), PROJECTED (byte-ratio arithmetic), or CLAIMED (unverified campaign speech).

---

## 1. Model identity

MEASURED from on-disk BF16 + receipts.

- HuggingFace snapshot: `PocketAiHub/Qwen3.8-27B-Abliterated-MLX` bf16 tree.
- Base: `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Architectures: `Qwen3_5ForConditionalGeneration` / text `qwen3_5_text`. Not `qwen3`, not `qwen3_moe`, not `qwen3_next`, not Q30-compatible.
- Dense. 64 layers. Hidden 5120. Intermediate 17408. Vocab 248320.
- Hybrid mixer: 48 Gated-DeltaNet + 16 GQA, interval 4 (`(layer+1)%4==0`).
- Language tensors packed: 755 (Q4 vehicle) or 851 (mixed catalogs). Vision skipped: 333 tensors.
- Source elements used as BPW denominator: `26895998464`.
- On-disk BF16: 11 shards, 24 files, `54740460836` bytes (~50.981 GiB).

Evidence: `receipts/ascent-2026-08-16/QWEN38_REUSE_MATRIX.json` (`identity`, `geometry`); `receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json`; `qwen38-27b/bf16/validation-summary.json`; `ls` of `qwen38-27b/bf16`.

G0 resident body (RECEIPT, not re-measured): artifact is **uniform-q4-v1**, not a mixed pack.

```
receipts/ascent-2026-08-16/GENESIS_RESIDENT_BODY.json
  "artifact": ".../qwen38-27b/uniform-q4-v1"
  "resident_weight_bytes": 14297675776
```

---

## 2. Capture that every gravity fit used

One capture exists. No Q4-vehicle capture exists.

Path: `qwen38-27b/activation-capture-v1/`
- `capture-result.json` (20360 bytes)
- `hidden/L00.f32` … `L63.f32` — 64 files × `5242880` = `335544320` bytes = `64 * 256 * 5120 * 4`.

| field | value | label |
|---|---|---|
| schema | `hawking.ascension.qwen38_bf16_post_swiglu_activation_capture.v1` | MEASURED |
| status | `CAPTURED_REAL_BF16_POST_NORM_HIDDEN` | MEASURED |
| forward | `mlx_lm.qwen3_5_text` | MEASURED |
| not_synthetic | true | MEASURED (flag + kurtosis later checked by doctor) |
| n_tokens | 256 | MEASURED |
| n_layers | 64 | MEASURED |
| stored width | 5120 (post-norm hidden), **not** 17408 post-SwiGLU | MEASURED |
| prompts | 5 (intended list has 10; cap stopped at 256 tokens) | MEASURED |
| prompt token counts | 57+60+68+61+**10** = 256 | MEASURED |
| fit_n / hold_n (descent) | 192 / 64 | RECEIPT |
| sha256_self field | `fdd937e20500b862452cf4732aa525087e1a3d209c1271e6c021811620687512` | MEASURED (self-declared) |
| sha256 of file bytes | `01db2f814fba99a1b7dac4668e30e20d69247ee3a4efa83b9ce4665718aedcbe` | MEASURED |
| Q4-vehicle capture | ABSENT | MEASURED (find + doctor admission) |

Prompt 5 defect: `"Explain gravity in one sentence."` records `n_tokens: 10` and the ids are the system-prefix slice (`248045, 8678, 198, 24342, 286, 4879, 369, 716, 310, 830`). The packer's capture loop does `ids = ids[:remain]` when the 256-token budget is exhausted (`lab/operators/qwen38_mixed_representation_pack.py` at preserve commit `11be05969`, lines 1134–1147). This is a truncated prefix, not the user sentence.

Schema name says `post_swiglu`. Stored rows are hidden-width. down_proj X for HGRAVS01 is **recomputed** as `silu(X@Wg.T)*(X@Wu.T)` from those hiddens + BF16 gate/up (descent `activation.down_proj_x`; attention-density verdict `activation_honesty.qwen38`).

Excerpt (`activation-capture-v1/capture-result.json`):

```
     1|{
     2|  "schema": "hawking.ascension.qwen38_bf16_post_swiglu_activation_capture.v1",
     3|  "status": "CAPTURED_REAL_BF16_POST_NORM_HIDDEN",
     4|  "source": {
     5|    "model_dir": ".../qwen38-27b/bf16",
     6|    "not_synthetic": true,
     7|    "forward": "mlx_lm.qwen3_5_text",
    10|  "n_tokens": 256,
    11|  "n_layers": 64,
    12|  "hidden": 5120,
```

`sha256_self` vs file bytes mismatch is recorded, not fatal, by the doctor admission (preserve `7c5f323d9`). Mixed packer prefers the file digest (`11be05969` lines 1098–1105) and stamps `01db2f81…` on the 2p0 pack.

---

## 3. Complete-artifact ledger

All roots live under `.../runs/qwen38-27b/`. Sizes from `os.walk` on 2026-08-17.

### 3.1 `bf16` — source, not a pack

- Class: source checkpoint.
- Eval: `bf16-smoke-generate.json` — garbage (`"generated_text": "� 这为标准原 (assistant eng"`), `fallbacks: 0`, `wall_s: 896.69`. This is an MLX smoke, not the Hawking native path. Do not treat as a capability baseline.
- Vendor `validation-summary.json` is an early-refusal screen (128-token ceiling), not a Hawking generate.

### 3.2 `uniform-q4-v1` — G0 vehicle

| | |
|---|---|
| recipe | language-only HQ30UQ4 g64 on GEMVs; f32v2 on small vectors; pack-time fuse of DeltaNet `in_proj` QKVZ/BA. Vision skipped. |
| packer | `crates/hawking-core/src/model/qwen38_pack.rs` (on HEAD). Example `ascension_qwen38_pack`. |
| complete BPW | **4.252735126866492 MEASURED** = `14297694680 * 8 / 26895998464` |
| payload | 14,297,694,680 bytes; 402 q4 + 353 f32 = 755 catalog tensors |
| pack status | `CANDIDATE_QWEN38_LANGUAGE_Q4_FUSED_INPROJ` |
| loaded | YES, native HQ30UQ4 |
| coherence | **EVAL_PASSED** (sealed-3 greedy ids) |
| completed | YES |

`manifest.json` L2–L15:

```
     2|  "complete_physical_bpw": 4.252735126866492,
     3|  "f32_tensors": 353,
     8|  "q4_tensors": 402,
     9|  "schema": "hawking.ascent.qwen38_language_uniform_q4.v1",
    13|  "status": "CANDIDATE_QWEN38_LANGUAGE_Q4_FUSED_INPROJ",
    14|  "tensor_count": 755,
    15|  "tensor_payload_bytes": 14297694680,
```

Native generate (RECEIPT, not re-run):

- `receipts/ascent-2026-08-16/qwen38-native-bringup.json` `generation.verbatim_B_reps_16_new_tokens` = `"<think>\nThe user simply wants me to say \"hi.\" This is a very"`; `fallbacks: 0`; greedy ids start `[248068, 198, 760, 1156, …]`.
- `receipts/ascent-2026-08-16/QWEN38_COHERENCE_SEAL.json` seals 12-token ids for 3 prompts on this artifact.
- `git show 3391c0c26:receipts/ascent-2026-08-16/QWEN38_COHERENCE_VERIFY.json` (not on HEAD `2eee9a004`): sealed-3 PASS; official 6-line file unsealed.
- Genesis tournament (`GENESIS_TOURNAMENT_RESULT.json`): this artifact, prompt "What is the capital of France? Answer in one sentence." → RECEIPT answer `"The capital of France is Paris."`, `tps: 26.6`, `steady_ns_per_token: 37576039` (CLEAN_CANDIDATE paired speed; tournament itself is not a latency claim).

G0 speech in the contract (`BPW ~4.2527`, `TPS ~26.4`, `TOKEN_NS ~37.9e6`) matches this artifact's **file BPW** exactly and the tournament/wall receipts to one digit. This lane did not re-time the GPU.

Pre-RMSNorm-adapter A1 garbage is recorded in the bring-up receipt (`pre_fix_A1_garbage`) and was discarded. The sealed path is post-adapter.

### 3.3 `mixed-2p0-v1` — Q80-recipe dense transfer

| | |
|---|---|
| recipe | gate HGRAVB01 binary_g128; up HGRAVR02 binary+rice_q1_rms@2%; down HGRAVS01 r160_b3 on real post-SwiGLU X; non-MLP HGRAVU01 q4 g64. Language-only. |
| packer | `lab/operators/qwen38_mixed_representation_pack.py` — **PRESERVE_ONLY** commit `11be05969` (`grok/qwen38-mixed-pack-20260816-145209`). `git cat-file -e HEAD:lab/operators/qwen38_mixed_representation_pack.py` → missing. Design: `receipts/ascent-2026-08-16/G016_PACKER_DESIGN.json` (on HEAD). |
| complete BPW | **2.0855934079220506 MEASURED** (payload 7,011,580,330 + billed sides = 7,011,764,637 bytes) |
| mlp / non-mlp BPW | 0.84805 / 4.25014 MEASURED |
| passes 2.0 target | false (ledger field) |
| pack wall | 2865.47 s MEASURED |
| catalog | `catalog.hq38m20` 158970 bytes, sha `6a16f7fd…` |
| segments | 132 files, ~6.53 GiB total dir |
| terminal | `EARNED_COMPLETE_PHYSICAL_MIXED_REPRESENTATION_PACKED` |
| lane | **LANE_DIED_ARTIFACT_LIVES** (`11be05969` message: "preserve work from a lane that died without reporting") |
| eval 1 | EVAL_EXPAND — `GENERATE.json` engine `mlx_lm_weights_overwritten_from_mixed_pack`, 851 tensors replaced, output single token `248046` `<|im_end|>`. Confounded. |
| eval 2 | EVAL_NATIVE — after mixed reader shipped. `QWEN38_NATIVE_MIXED_2P0_GENERATE.json`: 6 prompts, `fallbacks_total: 0`, `dense_w_materialized_total: 0`. Collapse to newlines / `)` / `.`. **EVAL_FAILED**. |
| completed | pack YES; generate YES (failed) |

`PACK_REPORT.json` L10–L13, L41, L59:

```
    10|  "complete_physical_bpw": 2.0855934079220506,
    11|  "mlp_physical_bpw": 0.8480504639008466,
    12|  "nonmlp_physical_bpw": 4.250142713483966,
    13|  "all_required_weight_artifact_bytes": 7011764637,
    41|  "mean_component_cosine": 0.9069688696406788,
    59|    "generation_is_the_gate": true
```

`GENERATE.json` L60–L68 (expand path):

```
    60|  "new_token_ids": [ 248046 ],
    63|  "generated_text": "<|im_end|>",
    65|  "fallbacks": 0,
    67|  "engine": "mlx_lm_weights_overwritten_from_mixed_pack",
    68|  "tensors_replaced": 851
```

Native path (RECEIPT `QWEN38_NATIVE_MIXED_2P0_GENERATE.json`): "Say hi." → 16× token 198 (newline); "What is the capital of France?" → 15× 198 then 8 (`)`). Oracle uniform-q4 on the same binary/prompt emits `<think>`.

HGRAVS01 fit: packer `RANK = 160`, `FACTOR_BITS = 3` (`11be05969` L105–106). Descent records `fit_n: 192`, `hold_n: 64`. `192/160 = 1.20` rows per rank (clears `FLAG_GRAM_RANKDEF` threshold `n_fit_rows < 160`). `192/17408 ≈ 0.011` rows per intermediate dim — the full Gram is thin; the rank-160 fit is only barely determined.

G016 paper said this recipe should land ~1.886 BPW (`G016_BPW_FEASIBILITY.json` scenario A). The packed number is 2.0856 because non-MLP stayed 4.25 and the realized MLP organ BPWs (1.125 / 1.288 / 0.132) are not the Q80-transferred 0.143 / 0.334 / 1.127. Paper 1.886 was PROJECTED, never packed.

### 3.4 `mixed-2p0-materialized` — Q4 generate vehicle of 3.3

| | |
|---|---|
| what it is | hard-linked / reconstructed HQ30UQ4 catalog of the 2p0 mixed weights |
| manifest BPW | **4.252735126866492** — this is the *vehicle* schema (`hawking.ascent.qwen38_language_uniform_q4.v1`), not the mixed payload |
| tensors dir | 1108 files, 14,308,279,520 bytes — same size as uniform-q4-v1 |
| packed/ | only `mlp_rows.json` (67311 B) |
| eval | EVAL_EXPAND: `QWEN38_NO_NATIVE_MIXED_READER.json` records single token 248046, 0 fallbacks. **EVAL_FAILED**, and **confounded** (mixed→float→Q4). |
| completed | YES (materialise step) |

An audit that reads `manifest.json` `complete_physical_bpw` here and reports 4.25 BPW is reading the Q4 vehicle, not the mixed science. The mixed BPW lives in `mixed-2p0-v1/PACK_REPORT.json`.

### 3.5 `mixed-sub15-v1` — attention also rice-compressed

| | |
|---|---|
| recipe | 2p0 MLP copied; attention GEMVs rice_q1_rms@2% from BF16; embed+lm_head HQ30UQ4; small f32 |
| packer | `tools/qwen38_sub15_pack.py` **on HEAD** (`40e073994` is an ancestor). Docstring L13–15: generate vehicle is reconstructed Q4; TPS is PROJECTED. |
| packed BPW | **1.2910781930062503 MEASURED** (`all_required_weight_artifact_bytes`: 4,340,604,637) |
| projected TPS | 79.44 PROJECTED (`1.415 + (38.217-1.415)*(1.291/4.2527)`) |
| packed/ | `attn/` 610 files + `attn_rows.json` + `mlp_rows.json` (~1.16 GiB codes) |
| tensors/+manifest | **Q4 vehicle clone** of uniform-q4-v1 (manifest BPW 4.2527 again) |
| eval | EVAL_EXPAND then recorded as the G006 negative. `QWEN38_SUB15_INCOHERENT.json`: two-token cycle `[220,264,220,…]` = `"  a    a  a  a…"`. fallbacks 0. Control uniform-q4 emits `<think>`. **EVAL_FAILED**. |
| native re-eval after mixed reader | **not found**. No `GENERATE.json` under this dir. No native HQ38M20 catalog (this pack did not write `catalog.hq38m20`). |
| completed | pack YES; generate YES on the Q4 vehicle (failed) |

`tools/qwen38_sub15_pack.py` L1–16 (HEAD):

```
     1|#!/usr/bin/env python3
     2|"""Pack Qwen3.8 under 1.5 BPW and materialize a native Q4 generate catalog.
    13|The generate vehicle is a hard-linked copy of uniform-q4-v1 with overwritten
    14|Q4 files of the *reconstructed* mixed/rice weights. hybrid_greedy only speaks
    15|HQ30UQ4 + f32v2; TPS is projected from packed bytes, not from this vehicle.
```

This is the binding the standing rules reject: low-bpw then expand to Q4 then generic GEMV. The incoherence result is still a real generate, but it cannot isolate representation from the second quantisation. The later native reader closed that confound for **2p0**, not for **sub15**.

### 3.6 Floor packs — native catalogs, never generated

All four write `catalog.hq38m20` + `segments/` and set `reconstruct_to_q4: false`. None has `GENERATE.json`. `git grep` and on-disk `rg` over `receipts/` and `workspace/ops/ascent-lanes/` for these directory names returned empty. Class: **PACKED_COMPLETE + NEVER_EVALUATED**.

| root | recipe | complete BPW MEASURED | payload bytes | pack wall_s | projected ms PROJECTED |
|---|---|---|---|---|---|
| `mixed-floor-q7-v1` | 2p0 MLP copied; non-MLP HGRAVU01 q7 g64 from BF16 | 3.17681583579674 | 10,680,295,260 | 148.72 | 28.91 |
| `mixed-floor-q8-v1` | 2p0 MLP copied; non-MLP HGRAVU01 q8 g64 | 3.5405591522270545 | 11,903,200,220 | 112.16 | 32.05 |
| `mixed-floor-q8-up10-v1` | gate+down copied; up HGRAVR02 rice@10%; non-MLP q8 | 3.6299810080179515 | 12,203,836,482 | 965.61 | 32.83 |
| `mixed-q4down-v1` | 2p0 gate/up copied; down **q4 not r160**; attn q4 | 2.9590429283570026 | 9,948,135,693 | 68.84 | 26.97 (tps 37.08) |
| `mixed-q3mlp-v1` | attn q4 copied; **all three MLP q3 g64** (no HGRAVS01) | 3.6138647373176767 | 12,149,632,429 | 182.70 | 32.66 (tps 30.62) |

`mixed-q4down` and `mixed-q3mlp` are the only complete artifacts that remove r160_b3 down. Their pack reports set `generation_is_the_gate: true` and then nobody ran the gate.

`mixed-q3mlp` `replaced_strided_weight_cosine` min 0.965 / median 0.969 — SCREEN_ONLY, not generate.

`mixed-q4down` cosine min 0.993 / median 0.994 — SCREEN_ONLY.

Cheapest experiment that would close these: one native `ascension_qwen38_hybrid_greedy` pass per root, 3 sealed prompts, GPU lock held by the measurement lane. This lane must not do it.

### 3.7 L00 recon disc — not a model pack

`lab/operators/qwen38_recon_pack.py` + `tools/qwen38_recon_disc/` (on HEAD via `709820607`). Packs L00 organs only for a Metal reconstruction discriminator. Receipt: `QWEN38_RECON_MEASURED.json`, `QWEN38_RECONSTRUCTION_IS_FREE.json`. Class: component microbenchmark. Not a token-level claim. Not a complete BPW.

### 3.8 Organ screen — never a pack

`lab/operators/qwen38_bpw_descent_sweep.py` + receipt `QWEN38_BPW_DESCENT.json` (merged). 6 layers × codec catalog. `claim_boundary.full_model_not_packed: true`, `generation_not_run: true`, `gpu_not_used: true`.

Paper recipes in `candidate_table` (all PROJECTED, **NEVER_PACKED** unless noted):

| codec id | PROJECTED bpw | class |
|---|---|---|
| `ternary_gate_up_hgravs01_twostage_down_ternary_attn_q4_emb` | 1.9897 | NEVER_PACKED |
| `binary_mlp_q3_attn_q4_emb` | 1.9924 | NEVER_PACKED (quality reject on screen) |
| `binary_all_except_emb_q4` | 1.4203 | NEVER_PACKED (graveyard; Q30 warning) |
| `REF_sibling_binary_rice_hgravs01_q4rest` | 2.0853 | this **is** mixed-2p0-v1 (packed) |
| `ternary_t0.7_all_except_emb_q4` | 2.439 | NEVER_PACKED |
| `uniform_q2_all_except_emb_q4` | 2.439 | NEVER_PACKED |
| `q3_gate_up_hgravs01_twostage_down_q3_attn_q4_emb` | 2.6831 | NEVER_PACKED |
| `ternary_mlp_q3_attn_q4_emb` | 2.7082 | NEVER_PACKED |
| `uniform_q3_all` | 3.25 | NEVER_PACKED as whole-model (nearest artifact is mixed-q3mlp = q3 **MLP only**) |
| `incumbent_uniform_q4_all` | 4.25 | uniform-q4-v1 |

Review (`QWEN38_BPW_DESCENT_REVIEW.json`) rescaled 1.99 → ~54 TPS PROJECTED and flagged the 5.9× rice penalty as superseded. Still `generation_not_run: true`.

---

## 4. Evaluation timeline (what was actually loaded)

Order matters. Early mixed negatives went through a reader that could not consume HGRAV*.

1. **No native mixed reader** (`QWEN38_NO_NATIVE_MIXED_READER.json`, on HEAD). `qwen38_hybrid_decode.rs` spoke only HQ30UQ4 + f32v2. Mixed tests had to reconstruct. The receipt itself says the collapse **cannot yet be attributed**.
2. **Native mixed reader shipped** (`QWEN38_NATIVE_MIXED_READER.json`). HQ38M20 catalog. gate→binary matvec, up→CSR leftover, down→HGRAVS01 two-stage, non-MLP HGRAVU01. `silent_reconstruct_to_q4: false`. Uniform-q4 path unchanged (sealed-3 still PASS).
3. **Native 2p0 generate** (`QWEN38_NATIVE_MIXED_2P0_GENERATE.json` + `QWEN38_COHERENCE_FLOOR_BRACKETED.json`). 0 fallbacks, 0 dense-W. Collapse is now attributable to the packed representation (plus whatever numeric error the Q80 occupancy tiles carry). Floor statement: coherence with *this codec set* sits between **2.0856 (fail)** and **4.2527 (pass)**. Nothing between those two BPWs has a native generate.

RECEIPT floor table (`QWEN38_COHERENCE_FLOOR_BRACKETED.json`):

```
4.2527 BPW q4 oracle          COHERENT
2.0856 BPW mixed-2p0-v1       INCOHERENT (native, twice)
1.2910 BPW mixed-sub15-v1     INCOHERENT (expand-to-Q4 only)
```

The 1.2910 row is a weaker attribution than 2.0856.

Interval (2.0856, 4.2527) contains packed-but-unevaluated artifacts at 2.959, 3.177, 3.541, 3.614, 3.630. Those would be the bisect, and they were never run.

---

## 5. Doctor tooling

### 5.1 doctor6 (lab/operators/doctor6) — not a Qwen3.8 instrument

`git grep -i 'qwen38|qwen3.8' -- lab/operators/doctor6 lab/operators/doctor_registry.py lab/operators/doctor_repack.py` → empty.

On-HEAD doctor6 receipts are Q80 only:

- `receipts/QWEN80_DOCTOR6_PRESCRIPTION_V1.json`
- `…_HGRAVS01.json`
- `…_CLAMP145.json`
- `…_MEASURED_BAR.json`

No Qwen3.8 doctor6 prescription, treatment, or verify receipt exists in git or under `Downloads/hawking/receipts`.

### 5.2 `doctor_qwen38_sensitivity` — ran once, then vanished from main

Preserve commit `7c5f323d9` on branch `grok/genesis-doctor-sensitivity-20260816-215243`.
`git merge-base --is-ancestor 7c5f323d9 HEAD` → **false**.
`ls Downloads/hawking/tools/condense/doctor_qwen38_sensitivity` → **missing**.
`ls Downloads/hawking/receipts/ascent-2026-08-16/QWEN38_DOCTOR*` → **missing**.

The run is recoverable only via `git show 7c5f323d9:<path>`. This is the exact failure mode the contract warned about (audit says absent; object is in a preserve commit).

Files in that commit:

```
receipts/ascent-2026-08-16/QWEN38_DOCTOR_SENSITIVITY_SUMMARY.json   488 lines
receipts/ascent-2026-08-16/QWEN38_DOCTOR_SENSITIVITY_MAP.json     24642 lines
receipts/ascent-2026-08-16/QWEN38_DOCTOR_SENSITIVITY_CATALOG.json 48680 lines
tools/condense/doctor_qwen38_sensitivity/{__init__,__main__,capture,geometry,ops,sweep,weights}.py
tools/condense/tests/test_doctor_qwen38_sensitivity.py
```

Instrument (not a codec): `symmetric_group_absmax_rtn` group 64, bit ladder `{8,4,3,2,1}`. Capability gate is greedy last-token ids across ≥3 prompts; cosine is never sufficient.

Admission (SUMMARY):

- `x_source`: `PARENT_BF16_REAL` (the 256-token capture above).
- `n_tokens`: 256. `n_prompts`: 5.
- Q4-vehicle capture **ABSENT** (explicit FINDING).
- `not_synthetic`: true.
- sha mismatch recorded.

Coverage:

- Swept layers: `{0,3,15,31,47,63}` — same 6 as the BPW descent.
- 58 layers named in the catalog and marked UNMEASURED, not projected.
- `n_units_swept`: 105. `n_units_catalog_64L`: 2260.
- `wall_s`: 205.997.

Counts (SUMMARY `counts`):

| | |
|---|---|
| with_sensitivity_number | 69 |
| **underdetermined** | **0** |
| eval_thin | 51 |
| unmeasured_behaviour | 30 |
| no_coherent_bits_on_ladder | 5 |
| expert | NOT_APPLICABLE_DENSE_NO_EXPERTS |
| floors | 8-bit:38, 4-bit:15, 3-bit:10, 1-bit:6 |

`no_coherent_bits_on_ladder`: L0 `linear_attn.in_proj_qkv`, L3 `v_proj`, L3 `o_proj`, whole `layer/31`, whole `layer/47`.

FINDING on determination (SUMMARY, quoted):

> Tensor-level evals are eval_thin (256/5120=0.05) but they are weight-only evals, not fits. Head/channel/outlier meet rows_per_dim >= 1. Zero UNDERDETERMINED numbers were emitted.

down_proj depth (weight-only RTN, eval_thin 256/17408=0.0147): L0 floor 3 bits; L3 floor 4; L15/31/47/63 floor 8.

This is **not** a complete-token generate. The instrument is forbidden to re-pack.

### 5.3 The 92-row / 2048-dim campaign is Q80, not Qwen3.8

The Qwen3.8 doctor encodes the prior failure as a comment, then refuses to repeat it.

`git show 7c5f323d9:tools/condense/doctor_qwen38_sensitivity/ops.py` L82–88:

```
    82|    """Refuse a sensitivity number when the unit is underdetermined.
    84|    A fit from X needs rows_per_dim >= 1 (the 92-row / 2048-dim campaign
    85|    was 0.045 and the whole result was invalid). Head/channel units use
    86|    the sliced width as fit_dim. Weight-only RTN is an eval, not a fit;
    87|    it still needs MIN_EVAL_ROWS and >= 3 prompts, and is flagged
    88|    eval_thin when rows_per_dim < 1.
```

`92/2048 = 0.04492 ≈ 0.045`. That geometry is Q80 expert `gate_proj.weight` shape `[512, 2048]` (`Q80_CAPTURE_COVERAGE.json` `provenance.sample_shape`, `organs[0].w_shape`, `organs[0].x_dim`).

Current Q80 capture census on HEAD (`receipts/ascent-2026-08-16/Q80_CAPTURE_COVERAGE.json`):

- capture: `source-bf16-capture-n192-scale64`, `n_tokens`: 25258.
- gate/up `fitted_dim`: 2048; `underdetermined`: true; `frac_underdetermined_vs_n_fit`: 0.9956; pairs with `n_fit < 2048`: 24468 / 24576.
- n_fit after 25% holdout: **p25 = 93**, **p50 = 194**, p10 = 26. The speech "median of 92" is closest to **p25 = 93**, not p50. I did not find the literal string `median of 92` in HEAD. Do not upgrade 93 to 92.
- down_proj `fitted_dim`: 512, rank target 160; still `underdetermined`: true; post-SwiGLU X **ABSENT** from that capture.
- verdict: `capture_sufficient_for_wellposed_fits: false`.

Qwen3.8 is dense: every row has activations; there is no 2048-wide expert Gram in the doctor. The 256-token capture is **eval_thin** on tensor RTN (0.05 vs hidden, 0.015 vs intermediate) and **determined** on head/channel/outlier slices. HGRAVS01 production fits used 192 tokens vs rank 160 (1.20), which the packer does not flag as `FLAG_GRAM_RANKDEF`.

Runtime was blamed for months on Q80 (host-bound, rice reconstruct). Qwen3.8's measured regime on the Q4 vehicle is the opposite: GPU / bytes (`QWEN38_BANDWIDTH_BOUND.json`, `QWEN38_AT_CEILING_RESOLVED.json`). That is a different model and a different failure.

---

## 6. What prior audits can get wrong (recovered absences)

| thing someone might say is missing | actual state |
|---|---|
| mixed packer not in tree | packed artifact **is** on disk; source is PRESERVE_ONLY `11be05969`, not deleted from the machine |
| doctor Qwen3.8 never ran | it ran; receipts+tool are PRESERVE_ONLY `7c5f323d9`; also absent from the main working tree on disk |
| mixed-2p0 never finished | terminal `EARNED_COMPLETE_PHYSICAL_MIXED_REPRESENTATION_PACKED`; 132 segments; 7.01 GB |
| mixed-2p0 never evaluated | evaluated twice: expand (EOS) and native (newline salad). Both fail |
| sub15 BPW is 4.25 | that is the Q4 **vehicle** manifest; packed BPW is 1.291 in `PACK_REPORT.json` |
| 2p0-materialized is a 2.08 BPW native pack | it is a 4.25 BPW HQ30UQ4 reconstruction of 2p0 |
| floor packs don't exist | five native HQ38M20 artifacts exist and were never generated |
| doctor6 mapped Qwen3.8 | doctor6 never mentioned Qwen3.8; a different instrument did |
| Qwen3.8 fits are the 92/2048 underdetermined set | that set is Q80 experts; Qwen3.8 doctor emitted 0 UNDERDETERMINED |
| G016 1.886 BPW was packed | never packed; realized sibling is 2.0856 |
| 1.99 ternary recipe was packed | NEVER_PACKED, generation_not_run |
| G0 is a mixed artifact | G0 resident + tournament + seal all bind **uniform-q4-v1** |

`QWEN38_COHERENCE_VERIFY.json` is also not on HEAD `2eee9a004` (added in `30bff1f2c` / edited in `3391c0c26`). Sealed-3 PASS is still recoverable from those commits and from `QWEN38_NATIVE_MIXED_READER.json` `oracle_uniform_q4_v1`.

---

## 7. Binding / negative results that stand

KILLS (this codec set, this capture, this reader):

- **HGRAVB01+HGRAVR02+HGRAVS01 MLP + Q4 attention at 2.0856 BPW** is natively incoherent. REOPEN_IF a different down/up/gate family, or a Q4-vehicle (not BF16-parent) activation refit, or a generate of the unevaluated 2.96–3.63 packs shows English on the sealed prompts with 0 fallbacks and 0 dense-W.
- **Rice-on-attention at ~1.29 BPW** is incoherent on the expand-to-Q4 vehicle. REOPEN_IF a native rice-attention kernel is written and the same artifact is read without a second quantisation.
- **Expert Gravity families do not transfer to attention** (`QWEN_ATTENTION_DENSITY_VERDICT.json`, on HEAD). HGRAVB01/R02/S01 miss the 0.99 attention bar everywhere sampled. REOPEN_IF a new attention codec family exists.

Not a kill:

- Q3 MLP (mixed-q3mlp, 3.614 BPW) — packed, cosine ~0.97, **never generated**.
- Q4-down instead of r160 (mixed-q4down, 2.959 BPW) — packed, cosine ~0.994, **never generated**.
- 1.99 ternary + two-stage down + ternary attn — paper only.

Preferred shape vs what shipped: mixed-2p0 **does** consume HGRAV* in the native reader (not expand). That path was measured and died on capability, not on a missing kernel. sub15's only generate is the rejected expand path.

---

## 8. Doctor / capture determination table (Qwen3.8 only)

| question | answer | label | pointer |
|---|---|---|---|
| How many tokens did the capture see? | 256 | MEASURED | `capture-result.json` L10 |
| How many prompts? | 5 (5th truncated to 10 prefix tokens) | MEASURED | same, prompts[] |
| Stored activation rank | (256, 5120) per layer, 64 layers | MEASURED | hidden file sizes |
| Synthetic? | refused by doctor if kurtosis~3; this capture is real | MEASURED | `capture.py` + admission |
| Q4-vehicle X? | ABSENT | MEASURED | doctor FINDING 1 |
| Tensor RTN rows/dim | 256/5120 = 0.05 (eval_thin) | MEASURED | SUMMARY |
| down_proj tensor rows/dim | 256/17408 ≈ 0.0147 (eval_thin) | MEASURED | SUMMARY `down_proj_depth` |
| Head/channel/outlier | rows_per_dim ≥ 1, DETERMINED | RECEIPT | SUMMARY cheapest_bits |
| UNDERDETERMINED units emitted | 0 | RECEIPT | SUMMARY `counts.underdetermined` |
| Layers unswept | 58 / 64 | RECEIPT | SUMMARY `coverage.unswept_layers` |
| HGRAVS01 production n_fit | 192 vs rank 160 (1.20) | RECEIPT | descent `fit_n` + packer `RANK` |
| doctor6 on this model | never | MEASURED | grep empty |
| doctor tool on HEAD / main disk | absent (preserve only) | MEASURED | `git cat-file`, `ls` |

Adequately determined?

- Weight-only RTN tensor scores: **eval_thin**, explicitly not fits. Doctor still emitted floors and flagged `eval_thin: true`. Do not treat L15+ down_proj "needs 8 bits" as a well-posed fit.
- Sliced head/channel/outlier scores: **determined** by the instrument's own rule (`rows_per_dim >= 1`).
- HGRAVS01 r160 on 192 tokens: **marginally determined for the rank**, underdetermined for a 17408-wide Gram. Packer would have set `FLAG_GRAM_RANKDEF` only if `n_fit_rows < 160`.
- 58 layers: **unmeasured**, not projected.
- Comparison to the 92/2048 campaign: that campaign's fits were invalid (`0.045` rows/dim, activation-weighted Gram). Qwen3.8 doctor refused to emit that class of number. The runtime was not the thing that made those Q80 fits invalid.

Cheapest experiment that would make the tensor floors well-posed: a capture with `n_tokens >= 17408` (or ≥5120 if only hidden-width organs are scored) from the **coherent Q4 vehicle**, then re-run the preserve doctor. This lane must not capture.

---

## 9. G0 numbers vs this ledger

| contract claim | what this lane can say |
|---|---|
| complete BPW ~4.2527 | **MEASURED** from `uniform-q4-v1/manifest.json` L2 = `4.252735126866492` |
| TPS ~26.4 | RECEIPT `GENESIS_TOURNAMENT_RESULT.json` paired speed `tps: 26.6`; rung receipt `RUNG_QWEN38_MEASURED.json` `tps: 26.1665`. Not re-timed. |
| TOKEN_NS ~37,900,000 | RECEIPT walls in `QWEN38_COMPLETE_TOKEN_WALL*.json` / `TOKEN_NS_QWEN38.json` / children `complete_token_wall_ns_authority: 35227918`. Not re-timed. |

G1 targets (capability, BPW <1.5, TOKEN_NS ≤1e7, TPS ≥100) are not met by any evaluated coherent artifact. The only coherent complete artifact is 4.2527 BPW.

---

## 10. Evidence index (paths only)

On-disk artifacts (Downloads/hawking):

```
workspace/campaign/records/runs/qwen38-27b/bf16/
workspace/campaign/records/runs/qwen38-27b/bf16-smoke-generate.json
workspace/campaign/records/runs/qwen38-27b/coherence_prompts.txt
workspace/campaign/records/runs/qwen38-27b/activation-capture-v1/capture-result.json
workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1/manifest.json
workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1/{PACK_REPORT,GENERATE,FORMAT,catalog.hq38m20,QWEN38_MIXED_2P0_V1_*}
workspace/campaign/records/runs/qwen38-27b/mixed-2p0-materialized/manifest.json
workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1/{PACK_REPORT,FORMAT,manifest.json,packed/}
workspace/campaign/records/runs/qwen38-27b/mixed-floor-q7-v1/PACK_REPORT.json
workspace/campaign/records/runs/qwen38-27b/mixed-floor-q8-v1/PACK_REPORT.json
workspace/campaign/records/runs/qwen38-27b/mixed-floor-q8-up10-v1/PACK_REPORT.json
workspace/campaign/records/runs/qwen38-27b/mixed-q4down-v1/PACK_REPORT.json
workspace/campaign/records/runs/qwen38-27b/mixed-q3mlp-v1/PACK_REPORT.json
```

On HEAD:

```
receipts/ascent-2026-08-16/G016_BPW_FEASIBILITY.json
receipts/ascent-2026-08-16/G016_PACKER_DESIGN.json
receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json
receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT_REVIEW.json
receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json
receipts/ascent-2026-08-16/QWEN38_COHERENCE_SEAL.json
receipts/ascent-2026-08-16/QWEN38_SUB15_INCOHERENT.json
receipts/ascent-2026-08-16/QWEN38_NO_NATIVE_MIXED_READER.json
receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_READER.json
receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_2P0_GENERATE.json
receipts/ascent-2026-08-16/QWEN38_DENSITY_ROOT_CAUSE.json
receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_VERDICT.json
receipts/ascent-2026-08-16/Q80_CAPTURE_COVERAGE.json
receipts/ascent-2026-08-16/GENESIS_RESIDENT_BODY.json
receipts/ascent-2026-08-16/GENESIS_TOURNAMENT_RESULT.json
receipts/ascent-2026-08-16/qwen38-native-bringup.json
tools/qwen38_sub15_pack.py
crates/hawking-core/src/model/qwen38_pack.rs
```

Preserve only (`git show`):

```
7c5f323d9  receipts/ascent-2026-08-16/QWEN38_DOCTOR_SENSITIVITY_{SUMMARY,MAP,CATALOG}.json
7c5f323d9  tools/condense/doctor_qwen38_sensitivity/*
11be05969  lab/operators/qwen38_mixed_representation_pack.py
3391c0c26  receipts/ascent-2026-08-16/QWEN38_COHERENCE_VERIFY.json
```

Commands used (this lane; no GPU):

```
git ls-tree -r --name-only HEAD | rg -i 'qwen38|doctor|gravity'
git log --all --oneline --grep='qwen38|doctor-sensitivity|mixed-pack|sub15'
git merge-base --is-ancestor 7c5f323d9 HEAD   # exit 1
git merge-base --is-ancestor 11be05969 HEAD   # exit 1
git merge-base --is-ancestor 40e073994 HEAD   # exit 0
git cat-file -e HEAD:lab/operators/qwen38_mixed_representation_pack.py   # missing
git cat-file -e HEAD:tools/condense/doctor_qwen38_sensitivity/__main__.py # missing
git show 7c5f323d9:receipts/ascent-2026-08-16/QWEN38_DOCTOR_SENSITIVITY_SUMMARY.json
ls -la /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b
python3  # sha256 of capture-result.json; walk sizes; json extracts
git grep -i 'qwen38' -- lab/operators/doctor6 lab/operators/doctor_registry.py
```

---

## Completion report

STATUS: SUPPORTED

CLAIMS:

1. The only coherent complete Qwen3.8 artifact is `uniform-q4-v1` at MEASURED complete BPW 4.252735126866492. Evidence: `qwen38-27b/uniform-q4-v1/manifest.json` L2, L15; `QWEN38_COHERENCE_SEAL.json`; `qwen38-native-bringup.json` generation block.
2. G0 resident loads that same artifact, not a mixed pack. Evidence: `GENESIS_RESIDENT_BODY.json` field `artifact`.
3. `mixed-2p0-v1` packed COMPLETE at MEASURED 2.0855934079220506 BPW (recipe HGRAVB01/HGRAVR02/HGRAVS01 + Q4 rest) even though the reporting lane died. Evidence: `mixed-2p0-v1/PACK_REPORT.json` L2, L10; terminal receipt `EARNED_COMPLETE_PHYSICAL_MIXED_REPRESENTATION_PACKED`; `git show --stat 11be05969`.
4. `mixed-2p0-v1` was evaluated natively and FAILED (0 fallbacks, 0 dense-W, newline/punctuation collapse). Evidence: `QWEN38_NATIVE_MIXED_2P0_GENERATE.json`; `QWEN38_COHERENCE_FLOOR_BRACKETED.json`.
5. An earlier 2p0 generate that emitted `<|im_end|>` used `mlx_lm_weights_overwritten_from_mixed_pack` and is confounded. Evidence: `mixed-2p0-v1/GENERATE.json` L67–L68.
6. `mixed-sub15-v1` packed COMPLETE at MEASURED 1.2910781930062503 BPW and FAILED generate on an expand-to-Q4 vehicle. Evidence: `mixed-sub15-v1/PACK_REPORT.json`; `QWEN38_SUB15_INCOHERENT.json`; packer docstring L13–15.
7. Five later native HQ38M20 packs (q7 / q8 / q8-up10 / q4down / q3mlp, BPW 2.959–3.630) exist on disk and have no generate receipt. Evidence: their `PACK_REPORT.json` `status: PACKED`; `find` for `GENERATE.json` returns only `mixed-2p0-v1/GENERATE.json`; `git grep` of those dir names in receipts is empty.
8. Paper recipes at 1.42 / 1.89 / 1.99 BPW were NEVER_PACKED. Evidence: `QWEN38_BPW_DESCENT.json` `claim_boundary.full_model_not_packed`; `G016_BPW_FEASIBILITY.json` scenario A 1.8863; candidate_table rows.
9. Qwen3.8 sensitivity capture is 256 real BF16-parent tokens, 64×(256,5120) hidden, no Q4-vehicle X. Evidence: `activation-capture-v1/capture-result.json` L2–L12; hidden file sizes; doctor SUMMARY admission.
10. Qwen3.8 doctor emitted 0 UNDERDETERMINED, 51 eval_thin, swept 6/64 layers; tool+receipts are PRESERVE_ONLY and absent from HEAD and from the main working tree. Evidence: `git show 7c5f323d9:…/QWEN38_DOCTOR_SENSITIVITY_SUMMARY.json` `counts`; `git merge-base --is-ancestor 7c5f323d9 HEAD` exit 1.
11. The 92-row / 2048-dim underdetermined campaign is Q80 expert Grams, not Qwen3.8. Evidence: doctor `ops.py` L84–85; `Q80_CAPTURE_COVERAGE.json` organs[0] `fitted_dim: 2048`, `n_fit_after_holdout_25pct.p25: 93`, `underdetermined: true`.
12. doctor6 was never pointed at Qwen3.8. Evidence: grep empty; only `QWEN80_DOCTOR6_*` prescriptions on HEAD.

EVIDENCE: excerpts and command results are in §§2–10 and the index above. No claim is carried without a path.

CHANGES: created `workspace/superwave/g1/g1-arch-gravity-runs.md` only.

TESTS: see following shell block in the lane wrap-up.

RISKS:

- Preserve-only doctor receipts can be garbage-collected if that branch is deleted; they are not on `main`.
- Unevaluated floor packs sit in the exact BPW interval the floor-bracket receipt says is untested. Treating the floor as "above 2.0856" without those generates over-claims.
- sub15 native attribution is still open.
- Capture prompt 5 is a 10-token system prefix. Doctor used it as a 5th prompt.

UNRESOLVED:

- Coherence of mixed-q4down (2.959) and mixed-floor-q7 (3.177) — cheapest next generate.
- Whether a Q4-vehicle capture would change HGRAVS01 / doctor floors.
- Literal "median of 92" source document (closest measured: Q80 n_fit p25=93).
- Native generate of sub15 after the mixed reader.

NEXT:

- Measurement lane: native greedy on `mixed-q4down-v1` then `mixed-floor-q7-v1` under the GPU lock.
- Merge or vendor `7c5f323d9` doctor receipts onto main before the preserve branch is reaped.
- Do not resurrect Q80/DSV4F as vehicles; the transferable negative is "92/2048 was a fit-posedness bug, not a Metal bug."
