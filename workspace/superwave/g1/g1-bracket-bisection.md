# G1 bracket bisection — Qwen3.8 coherence floor plan

Date: 2026-08-17. No generate, no GPU, no resident interference.
Write scope: this file only.

The five never-generated HQ38M20 packs do **not** share one composition family.
A 1-D BPW bisection of (2.0856, 4.2527] is the wrong experiment. Three families
sit in that band. HEAD `Qwen38HybridWeights::load_mixed` will generate only
family A (crushed MLP + richer attention). The two packs that can actually
move the floor (Q3-all-MLP, Q4-down) are refused at load by
`assert_mixed_mlp_native`.

Cheapest location: one 20-line assert relaxation, then one native generate of
`mixed-q3mlp-v1`. Not five generates. Not BPW order.

---

## 0. Definitions (used below)

**Complete physical BPW (G0 definition, MEASURED here):**

```
// crates/hawking-core/src/model/qwen38_pack.rs:673-679
complete_physical_bpw = 8 * sum(tensor payload bytes) / source_weight_elements
```

For HQ38M20, payload bytes = sum of catalog `nbytes` (each gravity container,
headers included). Catalog file itself is the analog of `manifest.json` and is
**not** in the G0 numerator. N = 26_895_998_464 (established).

**PACK_REPORT `complete_physical_bpw`** uses `all_required_weight_artifact_bytes`
= payload + `catalog.hq38m20`. Different number. Labeled PACK_REPORT below.

**Dir-bytes / N** includes unreferenced slack. Wrong for q3mlp / q4down.

**Native generate** = `ascension_qwen38_hybrid_greedy` on a `catalog.hq38m20`
root. Binding: log `opening mixed HQ38M20`, `fallbacks=0`,
`DENSE_W_MATERIALIZED: 0`. Expand-to-Q4 / MLX overwrite is not evidence
(standing rule; mixed-2p0 `GENERATE.json` is that confound).

---

## 1. The five packs — exact BPW, composition, admission

Roots (live disk, not in this sparse worktree):

`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/{name}`

All five: magic `HQ38M20\0`, version 1, 851 language tensors, 0 vision, 0
range faults, 0 missing referenced segments. Elements sum to 26_895_998_464
on every pack (MEASURED, catalog parse 2026-08-17).

### 1.1 Headline table

| pack | payload bytes | complete BPW (G0 def) MEASURED | PACK_REPORT BPW | family | HEAD generate |
|---|---:|---:|---:|---|---|
| mixed-q4down-v1 | 9_948_135_693 | **2.9589935339460913** | 2.9590429283570026 | B | **REFUSE** (down not S01) |
| mixed-floor-q7-v1 | 10_680_295_260 | **3.1767685514394888** | 3.17681583579674 | A | **RUN** (no code) |
| mixed-floor-q8-v1 | 11_903_200_220 | **3.5405118678698031** | 3.5405591522270545 | A | **RUN** (no code) |
| mixed-q3mlp-v1 | 12_149_632_429 | **3.6138111608720234** | 3.6138647373176767 | C | **REFUSE** (MLP all U01) |
| mixed-floor-q8-up10-v1 | 12_203_836_482 | **3.6299337236607006** | 3.6299810080179515 | A′ | **RUN** (no code) |

Control (already generated, not one of the five):

| pack | payload bytes | complete BPW MEASURED | PACK_REPORT | generate |
|---|---:|---:|---:|---|
| mixed-2p0-v1 | 7_011_580_330 | 2.0855385872764454 | 2.0855934079220506 | INCOHERENT native, 0 fallbacks |
| uniform-q4-v1 | 14_297_694_680 | 4.252735126866492 | n/a | COHERENT, G0 |

PACK_REPORT BPW = 8 × (payload + catalog_bytes) / N except mixed-2p0, whose
`all_required_weight_artifact_bytes` 7_011_764_637 is 25_337 above payload+catalog
(7_011_739_300). Established lower bracket 2.0856 is the PACK_REPORT figure.

None of the five has `GENERATE.json`. `git grep` of the five directory names
under `receipts/` is empty. Wave-1 reports already said PACKED only.

### 1.2 Per-organ composition (MEASURED, catalog nbytes + header `bits`)

Codec ids: 0=HGRAVB01, 1=HGRAVR02, 2=HGRAVS01 r160_b3, 3=HGRAVU01.
All 851 names classified; 0 OTHER. Split (unfused) ΔNet `in_proj_{qkv,z,a,b}`
— not G0's fused `qkvz`/`ba`. Loader has the split path.

**Family A / A′ / 2p0 — MLP identical except A′ up rice.**

| organ | n | elems | 2p0 / q7 / q8 codec | 2p0 BPW | q7 BPW | q8 BPW | q8-up10 delta |
|---|---:|---:|---|---:|---:|---:|---|
| mlp.gate_proj | 64 | 5_704_253_440 | B01 | 1.1250234267290902 | same | same | same |
| mlp.up_proj | 64 | 5_704_253_440 | R02 rice | 1.2875108157887178 | same | same | **1.7091418182148652** (outlier 0.1) |
| mlp.down_proj | 64 | 5_704_253_440 | S01 r160_b3 | 0.13161714918473189 | same | same | same |
| MLP total | 192 | 17_112_760_320 | | **0.8480504639008466** | same | same | **0.9885941313762291** |
| attn+embed+norm | 659 | 9_783_238_144 | U01 | 4.250142713483966 (bits=4) | **7.250143225379918** (bits=7) | **8.250143382383149** (bits=8) | same as q8 |

2p0 / q7 / q8 gate+up+down bytes 802_177_344 + 918_036_000 + 93_847_197 = 1_814_060_541.
q8-up10 up bytes 1_218_672_262.

**Family B — mixed-q4down-v1.** Attention/embed/norm byte-identical to 2p0
(hardlink 66/66 shared 2p0 segment names). Gate B01, up R02 rice@0.02, **down
U01 q4 g64** (not S01).

| organ | codec | bits | bytes | BPW |
|---|---|---:|---:|---:|
| mlp.gate_proj | B01 | — | 802_177_344 | 1.1250234267290902 |
| mlp.up_proj | R02 rice 0.02 | — | 918_036_000 | 1.2875108157887178 |
| mlp.down_proj | U01 | 4 | 3_030_402_560 | 4.250025132123162 |
| MLP total | | | 4_750_615_904 | **2.2208531248803234** |
| attn+embed+norm | U01 | 4 | 5_197_519_789 | 4.250142713483966 |

64/64 down headers: `schema hawking.gravity.uniform_group.v1`, `bits=4`,
`group_size=64`, `representation uniform_q4_group_scale`, magic `HGRAVU01`.
L0 sample nbytes=47_350_040. Slack 93_847_197 B = leftover unreferenced 2p0
S01 down payloads inside hardlinked segment files. Not in BPW.

**Family C — mixed-q3mlp-v1.** 851×HGRAVU01. Attention/embed/norm hardlinked
from 2p0 (Q4). All 192 MLP re-encoded U01 q3 g64.

| organ | codec | bits | bytes | BPW |
|---|---|---:|---:|---:|
| mlp.gate / up / down each | U01 | 3 | 2_317_370_880 | 3.2500251321231617 |
| MLP total | U01 | 3 | 6_952_112_640 | **3.2500251321231617** |
| attn+embed+norm | U01 | 4 | 5_197_519_789 | 4.250142713483966 |

L0 gate header: `bits=3`, `group_size=64`, `uniform_q3_group_scale`, magic
`HGRAVU01`. Slack 1_814_060_541 B = leftover 2p0 MLP payloads (exact 2p0 MLP
byte sum). Dir-BPW 4.153 is the wrong number.

**Small tensors (all five + 2p0):** 353 vectors, elems ≤ 65_536, codec 3.
`hgravu_is_vector` dequants them to f32 at load (existing 2p0 path). Not a
GEMV expand. `dn.conv1d` is 40_960 elems (≤ 65_536) so it dequants; it is
not a GEMV.

**Rice headers (MEASURED):**

- 2p0 / q4down / q7 / q8 up L0: `outlier_ratio_requested=0.02`, `outlier_count=1782580`, `rice_k=5`, `value_bits=1`, schema `hawking.gravity.binary_outlier_residual.v2`.
- q8-up10 up L0: `outlier_ratio_requested=0.1`, `outlier_count=8912896`, `rice_k=3`, same schema, `value_bits=1`.

**S01 geometry (2p0 / q7 / q8 / q8-up10, 64/64):** rank 160, factor_bits 3,
group 64, `activation_weighted_svd_low_rank_q`. Matches
`QWEN38_MIXED_HGRAVS_{RANK,BITS,GROUP}` (160 / 3 / 64).

### 1.3 Native loader — codec accept vs generate-ready

`load()` prefers `catalog.hq38m20` (`qwen38_hybrid_decode.rs:511-512`).
`load_mixed` match arms (`:601-664`):

| catalog codec | magic | destination | dispatch |
|---|---|---|---|
| 0 | HGRAVB01 | mixed Binary | `q80_binary_group_matvec_tg256` |
| 1 | HGRAVR02 | mixed Residual | `q80_binary_group_csr_matvec_tg256` |
| 2 | HGRAVS01 | mixed Hgravs, **only** r160_b3 | two-stage factor, bits=3 → `q80_hgravs01_factor_matvec_simd3` |
| 3 HGRAVU01 vector | HGRAVU01 | f32 dequant (≤65536, not GEMV) | n/a |
| 3 HGRAVU01 matrix | HGRAVU01 | mixed Uniform | bits=8 `q80_uniform8_matvec_*`; bits=3 `simd3`; else `q80_hgravs01_factor_matvec_simd` |
| 3 HQ30UQ4 | HQ30UQ4 | q4 map | G0 Q4 GEMV |
| other | — | **refuse** | — |

`mixed_gpu_layout` codec 3 accepts bits 2..=8 (`q80_mixed_decode.rs:1143`,
`:1305-1329`). Embed lookup `qwen38_hgravu_embedding_lookup` is bits-generic
(`qwen38_device_activations.metal:325-341`). Q7 attention/embed therefore
has a consume path. `parse_uniform_q8_container` requires bits==8
(`q80_mixed_decode.rs:1055-1058`) but **the Qwen3.8 loader does not call it**.

Then `load_mixed` calls `assert_mixed_mlp_native` (`:667`, body `:958-1004`):

- every `mlp.gate_proj` must be `MixedGpuWeight::Binary` (B01)
- every `mlp.up_proj` must be `Residual` (R02)
- every `mlp.down_proj` must be `Hgravs` (S01 r160_b3)
- missing → "refusing silent dense/Q4 fallback"
- wrong variant → "refusing reconstructed MLP"

This assertion is a **recipe lock for mixed-2p0**, not a codec-family check.
q4down down is Uniform q4 (native HGRAVU01, not reconstructed Q4). q3mlp MLP
is Uniform q3 (native). Both are representation-specific (`dispatch_uniform`).
The assert still refuses them.

**Admission (MEASURED, catalog + header + assert predicate, no Metal):**

| pack | magics | every codec has a match arm | assert_mixed_mlp_native | generate on HEAD |
|---|---|---|---|---|
| mixed-2p0-v1 | 64/64/64/659 B/R/S/U | yes | PASS | already run, INCOHERENT |
| mixed-floor-q7-v1 | same split, U bits=7 | yes | PASS | **no code change** |
| mixed-floor-q8-v1 | same, U bits=8 | yes | PASS | **no code change** |
| mixed-floor-q8-up10-v1 | same, U bits=8, up still R02 | yes | PASS | **no code change** |
| mixed-q4down-v1 | 64 B + 64 R + 723 U | yes | **FAIL** down=Uniform | needs assert relax |
| mixed-q3mlp-v1 | 851 U (192 bits=3, 659 bits=4) | yes | **FAIL** gate/up/down | needs assert relax |

Inventory `g1-artifact-inventory.md:331-338` "structurally loadable YES" for
q4down / q3mlp means catalog parse + magic match-arms. It did not apply
`assert_mixed_mlp_native`. Generate-ready ≠ parse-ready.

**Code change if q3mlp / q4down are to be generated** (not applied this lane):

In `assert_mixed_mlp_native`, accept `MixedGpuWeight::Uniform` on gate / up /
down (in addition to Binary / Residual / Hgravs). Do **not** route them
through HQ30UQ4 or float GEMV. `dispatch_uniform` already handles bits 3 and
4. No new kernel. ~20 lines. Refuse still if a GEMV is missing.

REOPEN_IF someone wants S01 geometry other than r160_b3 (still hard-checked
at `:1117-1134`).

---

## 2. They are not one family

Contract text said the five share a composition family. Catalog bytes say no.

| family | packs | MLP | attention / embed / lm_head | what a verdict is about |
|---|---|---|---|---|
| A | 2p0, q7, q8 | 0.848 (B01 + R02@2% + S01 0.132) | U01 bits 4 / 7 / 8 | does richer-than-Q4 attention rescue the 2p0 MLP? |
| A′ | q8-up10 | 0.989 (same gate/down, rice@10%) | U01 bits 8 | same as A plus slightly richer up |
| B | q4down | 2.221 (B01 + R02@2% + **U01 q4 down**) | U01 bits 4 (= 2p0) | was S01 down the unique killer? |
| C | q3mlp | **3.250 all U01 q3** | U01 bits 4 (= 2p0) | does Q3 MLP + Q4 attention hold? |

Wave 1 already attributed 2p0's native collapse to MLP at 0.848, down at
0.1316, not to Q4 attention (`QWEN38_COHERENCE_FLOOR_BRACKETED.json:17`;
context: "MLP over-compression failure, not an attention failure").

Therefore:

- A q7/q8 **fail** is the expected confirmation of that attribution. It does
  **not** raise the Qwen3.8 floor to 3.18. Family C at 3.614 and family B at
  2.959 remain untested.
- A q7 **pass** would falsify the MLP-only attribution (H4). Huge. Then skip
  q8 (already coherent at Q7).
- Locating a floor on family A does not locate it for B or C at the same BPW.
- Locating a floor on C does not license a 3.61 BPW pack that still crushes
  down to 0.132, or a hetero allocator's 2.0 table (`g1-heterogeneous-allocation.md`).

---

## 3. Ordered evaluation plan (information, not BPW)

### 3.1 Hypotheses and priors

| id | claim | prior | who said it |
|---|---|---|---|
| H1 | MLP 0.848 / down 0.132 is below its own floor, independent of attention bits | **high** | wave 1, native 2p0 INCOHERENT with Q4 attention |
| H2 | Q3-all-MLP + Q4 attention is above the floor (family C) | **medium-high** | doctor: gate/up/down hold Q3 cosine ≥ 0.9679; q3mlp min strided cosine 0.965 |
| H3 | Restoring only down to Q4, keeping B01 gate + rice up, is above the floor (family B) | **medium-low** | 2p0 named down 0.132 as the crush; gate 1.125 + up 1.288 still in the kill recipe |
| H4 | Q7/Q8 attention rescues the 2p0 MLP (family A) | **low** | contradicts H1 |

A BPW-midpoint first run is floor-q7 (3.177 ≈ mid of 2.086–4.253). That tests
H4. Expected information is low: P(pass) small, and a fail does not transfer
to B or C.

### 3.2 Path B — cheapest location (recommended)

Requires the assert relaxation in §1.3. Then **at most two** generates.

**Run 1: `mixed-q3mlp-v1` (family C, 3.6138111608720234).**

Why first (not lowest BPW, not midpoint):

1. Only pack that tests the doctor's explicit Q3-MLP prediction.
2. Only pack that raises *all three* MLP organs above the 0.848 kill recipe
   while keeping generate-proven Q4 attention (same U01 q4 tensors as 2p0,
   hardlinked).
3. Clean codecs: no rice, no r160. A pass is a usable G1 density point, not
   just a bracket tick.
4. `dispatch_uniform` bits=3 is the specialized `simd3` kernel already on
   the 2p0 S01 factor path. No new shader.
5. Fail skips family A (worse MLP) and still leaves family B as a distinct
   question. One run eliminates the largest remaining region of "maybe Q3
   works".

Decision:

| Run 1 | next | skip | what is now known |
|---|---|---|---|
| COHERENT | Run 2 = q4down (hunt lower, different MLP mix) | q7, q8, q8-up10 | floor_C ≤ 3.6138. Bracket for *this family* is (2.0856, 3.6138]. |
| INCOHERENT | Run 2 = q4down (H3 still live) | q7/q8/q8-up10 unless both B and C fail and H4 is reopened | floor_C > 3.6138. Q4-attn + Q3-MLP is below floor. |
| LOAD_REFUSE | stop; assert patch incomplete | everything | not a coherence verdict |

**Run 2: `mixed-q4down-v1` (family B, 2.9589935339460913).**

| Run 2 | stop? | what is now known |
|---|---|---|
| COHERENT | yes for this campaign | floor_B ≤ 2.959. Cheapest coherent mixed point on disk. Family A still untested and low-value. |
| INCOHERENT after C pass | yes | Q3-all-MLP works; B01+rice+Q4-down does not. Floor is family-specific. |
| INCOHERENT after C fail | optional Run 3 = floor-q7 (H4 only) | Q4-attn family has no coherent mixed pack on disk. Floor in (3.6138, 4.2527] *or* needs a new pack (e.g. Q4 MLP). |

Never run q8 or q8-up10 unless q7 passed.

### 3.3 Path A — generate-only, HEAD as-is

If the GPU lane cannot touch Rust:

**Run 1 (only): `mixed-floor-q7-v1`.**

Only remaining information among the three currently loadable packs: H4.
q8 and q8-up10 are the same MLP with more attention bits. If q7 fails
(expected), they add nothing. If q7 passes, q8 is redundant.

| Run 1 | next | what is now known |
|---|---|---|
| INCOHERENT | **stop**. Request assert patch. Do not run q8 / q8-up10 / (unpatched) q3mlp / q4down | family A still dead at Q7. Floor **not** located. Do not write "floor > 3.177" as a Qwen3.8 fact. |
| COHERENT | stop family A; do not run q8 | H1 false. floor_A ∈ (2.0856, 3.1768]. 2p0 failed because Q4 attention could not carry the crushed MLP. |

A Path A fail is almost a zero-new-fact GPU spend. Path B is cheaper *as a
location procedure* because the patch is CPU and unlocks the packs that move
the bracket.

### 3.4 Do not

- Do not generate in BPW order (q4down → q7 → q8 → q3mlp → q8-up10).
- Do not run all five.
- Do not treat a family-A fail as a 1-D floor raise.
- Do not use mixed-2p0 `GENERATE.json` (`engine: mlx_lm_weights_overwritten_from_mixed_pack`) as a control.
- Do not point `genesis-resident` at these roots (would evict G0). Oneshot example only.
- Do not kill or restart the resident. Take `tools/gpu_lane_lock.sh`. If the lock owner is `genesis-resident:parent`, wait or coordinate. Do not RPC the live body to switch artifacts.
- Do not rebuild any of these packs.

---

## 4. Exact GPU commands

Vehicle: `ascension_qwen38_hybrid_greedy` (same binary as the native 2p0 run
and the G0 seal). Rebuild from the HEAD that contains the assert patch if
Path B.

```
CARGO_TARGET_DIR=workspace/ops/build/rust \
  cargo build --release -p hawking-core --example ascension_qwen38_hybrid_greedy
```

Artifacts and tokenizer live under the main repo, not this worktree:

```
ART=/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b
BIN=workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy
TOK=$ART/bf16/tokenizer.json
PROMPTS=$ART/coherence_prompts.txt
```

Chat render (do **not** pass `--raw-prompt`):

```
// qwen38_hybrid_decode.rs:316-317
<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
```

Same render the native 2p0 6-prompt run used
(`QWEN38_NATIVE_MIXED_2P0_GENERATE.json` `rendered` fields).

`--max-seq-len` default is 128. Extended factual runs need more
(France 15+128=143; arithmetic ~40+256). Use 512 on every run.

### 4.1 Phase A — 16-token 6-prompt collapse screen (mandatory first)

Path B Run 1:

```
./tools/gpu_lane_lock.sh qwen38-bracket-c \
  $BIN \
  --artifact-root $ART/mixed-q3mlp-v1 \
  --tokenizer $TOK \
  --prompts-file $PROMPTS \
  --max-new-tokens 16 \
  --max-seq-len 512 \
  --out receipts/ascent-2026-08-16/QWEN38_Q3MLP_GENERATE_16.json
```

Path A Run 1: same command, artifact `mixed-floor-q7-v1`, out
`QWEN38_FLOOR_Q7_GENERATE_16.json`.

Load log must contain `opening mixed HQ38M20` and must **not** contain
`opening Metal + 755 catalog tensors`. If assert still fires, the JSON will
not be written and stderr will say `is not HGRAVB01` / `is not HGRAVS01`.
That is LOAD_REFUSE, not INCOHERENT.

### 4.2 Phase B — only if Phase A is not a collapse

Same binary, one prompt each:

```
./tools/gpu_lane_lock.sh qwen38-bracket-france \
  $BIN --artifact-root $ART/<pack> --tokenizer $TOK \
  --prompt "What is the capital of France?" \
  --max-new-tokens 128 --max-seq-len 512 \
  --out receipts/ascent-2026-08-16/QWEN38_<PACK>_FRANCE_128.json

./tools/gpu_lane_lock.sh qwen38-bracket-arith \
  $BIN --artifact-root $ART/<pack> --tokenizer $TOK \
  --prompt "What is 17 times 19? Reply with the integer product, then one short sentence showing the arithmetic. No other preamble." \
  --max-new-tokens 256 --max-seq-len 512 \
  --out receipts/ascent-2026-08-16/QWEN38_<PACK>_ARITH_256.json
```

Do **not** re-run G0 for the seal. Sealed ids already exist
(`QWEN38_COHERENCE_SEAL.json`, `g1-baseline-remeasure.md` oracle-32).
Id-drift vs those seals is a quality signal, never pass/fail
(`HARVEST_NOTE_G006.json`).

### 4.3 Cost (labeled)

MEASURED on mixed-2p0 native, same harness, 6×16
(`QWEN38_NATIVE_MIXED_2P0_GENERATE.json`): session open 3.391 s; per-prompt
wall 2.77–3.51 s; total generate ~18 s.

q3mlp payload / 2p0 payload = 12.150 / 7.012 ≈ 1.73. ESTIMATED Phase A
~6 s open + ~30 s generate if traffic-bound. Phase B ESTIMATED
128×40 ms + 256×40 ms ≈ 15 s plus prefills, using today's live G0
TOKEN_NS 39.3 ms as a rough scale (G0 is Q4; mixed kernels differ;
this is not a token-level claim).

One pack, both phases: minutes. Five packs: wasted lock time, not more
information.

---

## 5. Prompt set, oracle, tokens, pass criterion

### 5.1 Prompt set

Authority: `$ART/coherence_prompts.txt` (171 B, 6 lines) — the exact set
the native 2p0 run used.

```
Say hi.
Write a function that reverses a string.
What is the capital of France?
Explain what a hash map is in one sentence.
def fibonacci(n):
The three primary colors are
```

Plus, only in Phase B if Phase A is not a collapse:

- France again at 128 new tokens (same string as line 3).
- Arithmetic string in §4.2 (the live G0 capability prompt from
  `g1-baseline-remeasure.md:193`).

### 5.2 Oracle comparison

Two oracles, different jobs.

**Collapse oracle (pass/fail).** Compare *shape* to the native 2p0 failure
and the G0 success, not ids.

G0 / seal, same chat render, `Say hi.` first 12 ids
(`QWEN38_COHERENCE_SEAL.json`):

```
[248068, 198, 760, 1156, 4777, 6587, 728, 310, 1910, 328, 5834, 1149]
= "<think>\nThe user simply wants me to say \"hi.\""
```

G0 live oracle-32, same prompt (`g1-baseline-remeasure.md:176-187`):

```
[248068, 198, 760, 1156, 4777, 6587, 728, 310, 1910, 328, 5834, 1149,
 1061, 369, 264, 1546, 4145, 11, 2050, 1622, 13, 353, 3172, 1066, 1910,
 15131, 303, 264, 11321, 11, 5629, 1560]
= "<think>\nThe user simply wants me to say \"hi.\" This is a very simple,
   direct request. I'll just say hi in a friendly, natural way"
```

France seal 12 (`QWEN38_COHERENCE_SEAL.json`), decoded this lane from
`bf16/tokenizer.json` vocab:

```
[248068, 198, 760, 1156, 369, 9859, 264, 4145, 57879, 3296, 25, 3437]
= "<think>\nThe user is asking a simple factual question: What"
```

**Paris is not in 12 tokens.** G0 is still inside `<think>`. 16 tokens cannot
decide France→Paris even on the coherent oracle.

Native 2p0, 16 new, 0 fallbacks (`QWEN38_NATIVE_MIXED_2P0_GENERATE.json`):

| prompt | new ids (abbrev) | text |
|---|---|---|
| Say hi. | 198 × 16 | 16 newlines |
| reverses a string | 1076, 1076, 8, … | `......)...)...` |
| capital of France | 198 × 15, 8 | 15 newlines + `)` |
| hash map | 198 × 9, 8, 13, 13, 8, 8, 13, 198 | newline / `)` / `.` |
| fibonacci | 578 × 4, 8, 198 × 11 | `))))` + newlines |
| primary colors | 8, 198 × 10, 8 × 4 | `)` + newlines + `))))` |

Decoded: 198=`\n`, 8=`)`, 13=`.`, 1076=`...`, 578=`))`.

**Greedy-id oracle (signal, not gate).** Report prefix match vs the 12-id
seal and vs the 32-id G0 "Say hi." sequence. Expected to drift: these packs
are not HQ30UQ4. `HARVEST_NOTE_G006.json:4-10` forbids judging lower-BPW
artifacts by id-identity to the 4.2527 seal.

### 5.3 How many tokens for a verdict to mean anything

| n new | what it can decide | evidence |
|---|---|---|
| 16 × 6 prompts | **INCOHERENT** (collapse / cycle / salad / immediate EOS) | MEASURED: 2p0 and sub15 collapsed by token 1–2 |
| 16 × 6 | **not** COHERENT | MEASURED: G0 France 12 = still "What"; G0 Say-hi 32 still inside think |
| 32 | G0 capability bar for *id-match on G0 only* | `g1-baseline-remeasure.md` 6/6 oracle-32. Not a mixed-pack bar |
| 128 on France | ESTIMATED minimum for France→Paris given think-then-answer | not measured when G0 first emits "Paris" |
| 256 on 17×19 | MEASURED sufficient for `323` on G0 | `g1-baseline-remeasure.md:191-199` |

Phase A (16×6) is the cheap screen and is the *only* length at which
INCOHERENT is a complete verdict. Phase B is required before writing
COHERENT / locating a floor.

### 5.4 Pass criterion

Binding (else the run is not a floor measurement):

1. Process opened the mixed path (`opening mixed HQ38M20`).
2. `fallbacks_total == 0`.
3. `dense_w_materialized_total == 0` (harness prints this as the literal 0;
   treat a non-zero or a missing-GEMV error as bind-fail).
4. Not an expand-to-Q4 / MLX vehicle.

**INCOHERENT** if binding holds and **any** Phase A prompt:

- is punctuation / whitespace only (2p0 mode: `{198,8,13,1076,578,220}`)
- is a cycle of period ≤ 4 over ≥ 8 tokens (sub15 `220/264`)
- is only `<|im_end|>` (248046) or only `<|endoftext|>` (248044)
- is token salad with no English/code word of length ≥ 3

**PROVISIONAL** (do not locate the floor yet) if binding holds, no collapse
rule fires, and ≥ 5/6 Phase A prompts start with 248068 `<think>` or with
well-formed English/code. Then run Phase B.

**COHERENT** if PROVISIONAL and:

- France 128-new text contains `Paris` (case-insensitive)
- arithmetic 256-new text contains `323`
- both remain well-formed (no late collapse into a cycle)

**LOAD_REFUSE** if assert or missing-GEMV fires. Not a floor datum.

G006 harvest bar (`HARVEST_NOTE_G006.json:6-10`) is the same idea:
well-formed English/code on all 6, France→Paris, 0 fallbacks, 0 dense W,
id-drift reported not gated. Phase A alone cannot satisfy France→Paris.

---

## 6. What a floor found here would and would not license

A COHERENT on **family C** (q3mlp, 3.6138) licenses:

- "Q4 attention + Q3 MLP, this encoding, native kernels, is at or above the
  token floor."
- A G1 density candidate at 3.6138 complete BPW (payload def) with no rice
  and no r160.
- Killing "Q3 MLP cosine is a false friend" *for this mix*.

It does **not** license:

- any pack at 3.61 that still uses S01 down 0.132 or B01/R02 MLP
- the hetero 2.0 / 1.5 tables in `g1-heterogeneous-allocation.md`
- attention below Q4
- embed/lm_head below Q4 (q3mlp leaves them at U01 q4)
- 50 TPS / 100 TPS (PACK_REPORT projected 32.66 ms / 30.6 TPS is PROJECTED
  from a superseded wall formula; not this lane)
- G0 TOKEN_NS claims

A COHERENT on **family B** (q4down, 2.959) licenses:

- "S01 r160 down was sufficient to kill 2p0; Q4 down + leftover B01 gate +
  rice up + Q4 attention is at or above the floor."
- A cheaper coherent point than C.

It does **not** license Q3 MLP, nor family A, nor "2.96 BPW any allocation."

A COHERENT on **family A** (q7) licenses:

- "H1 is false: Q4 attention, not the 0.848 MLP, was the 2p0 limiter."
- floor_A ∈ (2.0856, 3.1768].

It does **not** license Q3 MLP or a 3.18 BPW pack with Q4 attention.

An INCOHERENT on C + B + A (if someone ran all three families) would
license: "no mixed recipe on disk in (2.086, 3.630] is coherent; next
experiment is a **new** pack (Q4-all-MLP + Q4 attention is G0; the missing
cell is Q4 MLP + something cheaper than Q4 attention, or Q3 attention + Q4
MLP)." That pack does not exist.

The open interval (2.0856, 4.2527] as a *single* floor is not a property
these five points can locate. They locate at most three family-conditional
floors.

---

## 7. What the current bracket cannot tell us

Already paid:

- 4.2527 G0 HQ30UQ4: COHERENT (MEASURED, today, 6/6 oracle-32).
- 2.0856 mixed-2p0 family A at Q4 attention: INCOHERENT native (MEASURED).
- 1.291 mixed-sub15: INCOHERENT on an expand-to-Q4 vehicle (confounded).

Cannot tell, even after the plan in §3:

1. The floor of a differently allocated pack at the same complete BPW.
   q3mlp at 3.614 and floor-q8 at 3.541 are not two samples of one curve.
2. Whether 2p0 died of down 0.132 alone, or of the B01+R02+S01 conjunction
   — until q4down is generated (and even then gate/up stay crushed).
3. Isolated organ floors (Q3 gate only, Q3 down only, Q3 lm_head only).
   No such packs exist.
4. Whether Q7/Q8 *attention* is necessary, helpful, or wasted — until a
   family-A generate, and wave 1 says it is the wrong axis.
5. Active-column / write-gain out_proj (unresolved doctor contradiction).
   None of these five implement that.
6. A number that can be compared to 50 TPS. PACK_REPORT projections use
   `ms = 1.415 + (38.217-1.415)*(bpw/4.2527)` or a 1.229 variant. Those
   walls are retired / superseded by today's 39.326 ms live G0. Projections
   from them are not this lane's output.

Cheapest experiment that would produce (2) is Path B Run 2.
Cheapest experiment that would produce (3) is a new overlay pack, not
these five.
Cheapest experiment that would produce (4) is Path A Run 1 (or Path B
optional Run 3).

---

## 8. Evidence (command output and excerpts)

### 8.1 Catalog parse (this lane, 2026-08-17)

Parser: HQ38M20 header + 128-byte records as in
`parse_qwen38_mixed_catalog` (`qwen38_hybrid_decode.rs:96-174`). Payload
headers read via `split_gravity_container` (8-byte magic + u32le + JSON).
No weight bodies decoded. Peak RSS ≪ 20 GB.

```
mixed-2p0-v1
  payload 7011580330 bpw 2.0855385872764454
  accept True
  codecs B01:64 R02:64 S01:64 U01:659  u01_bits {4: 659}
  mlp 0.8480504639008466  non 4.250142713483966

mixed-q4down-v1
  payload 9948135693 bpw 2.9589935339460913
  accept False  DOWN not all S01; unique=[3]
  codecs B01:64 R02:64 U01:723  u01_bits {4: 723}
  mlp 2.2208531248803234  slack 93847197

mixed-floor-q7-v1
  payload 10680295260 bpw 3.1767685514394888
  accept True
  codecs B01:64 R02:64 S01:64 U01:659  u01_bits {7: 659}
  mlp 0.8480504639008466  non 7.250143225379918

mixed-floor-q8-v1
  payload 11903200220 bpw 3.5405118678698031
  accept True
  u01_bits {8: 659}
  mlp 0.8480504639008466  non 8.250143382383149

mixed-q3mlp-v1
  payload 12149632429 bpw 3.6138111608720234
  accept False  GATE/UP/DOWN not B01/R02/S01; unique=[3]
  codecs U01:851  u01_bits {4: 659, 3: 192}
  mlp 3.2500251321231617  slack 1814060541

mixed-floor-q8-up10-v1
  payload 12203836482 bpw 3.6299337236607006
  accept True
  u01_bits {8: 659}
  mlp 0.9885941313762291  (up 1.7091418182148652)
```

All six: n_tensors=851, elements=26895998464, range_fail=0, version=1.

floor-q7 catalog head:

```
magic b'HQ38M20\x00'
version 1  n_tensors 851  n_segments 66  catalog_bytes 158970
```

### 8.2 Never generated

```
mixed-q4down-v1:         FORMAT.md PACK_REPORT.json catalog.hq38m20 segments
mixed-floor-q7-v1:       FORMAT.md PACK_REPORT.json catalog.hq38m20 segments
mixed-floor-q8-v1:       FORMAT.md PACK_REPORT.json catalog.hq38m20 segments
mixed-q3mlp-v1:          FORMAT.md PACK_REPORT.json catalog.hq38m20 segments
mixed-floor-q8-up10-v1:  FORMAT.md PACK_REPORT.json catalog.hq38m20 segments
mixed-2p0-v1:            ... GENERATE.json ...   # MLX expand vehicle, not native

$ git grep -l 'mixed-floor-q7-v1|mixed-q4down-v1|mixed-q3mlp-v1|mixed-floor-q8-v1|mixed-floor-q8-up10-v1' HEAD -- receipts
NO HITS IN receipts/
```

### 8.3 Assert (HEAD)

```
958:        fn assert_mixed_mlp_native(mixed: &HashMap<String, MixedGpuWeight>) -> Result<()> {
967:                            "{gate} is not HGRAVB01; refusing reconstructed MLP"
980:                            "{up} is not HGRAVR02; refusing reconstructed MLP"
993:                            "{down} is not HGRAVS01; refusing reconstructed MLP"
```

Called from `load_mixed` at line 667, after every codec-3 Uniform MLP would
already have been inserted into `mixed`.

### 8.4 PACK_REPORT recipes (CLAIMED composition; confirmed by §8.1)

`mixed-floor-q7-v1/PACK_REPORT.json:6-21` — MLP copied from 2p0; non-MLP
U01 q7; `complete_physical_bpw` 3.17681583579674; `reconstruct_to_q4: false`.

`mixed-q4down-v1/PACK_REPORT.json:9-27` — down U01 q4 not S01;
`generation_is_the_gate: true`; `complete_physical_bpw` 2.9590429283570026.

`mixed-q3mlp-v1/PACK_REPORT.json:9-29` — all three MLP U01 q3;
`generation_is_the_gate: true`; `complete_physical_bpw` 3.6138647373176767.

### 8.5 G006 bar (do not use id-identity)

`receipts/ascent-2026-08-16/HARVEST_NOTE_G006.json:4-10`:

```
"A lower-BPW artifact cannot be id-identical to that seal. Do NOT judge
 the lane against id-identity."
judge_it_against:
  well-formed English or code on all 6 prompts
  France -> Paris
  0 silent fallbacks, 0 dense_w_materialized
  greedy-id drift REPORTED, never pass/fail
```

---

```
STATUS
IMPLEMENT_READY

CLAIMS
1. Five never-generated HQ38M20 packs exist at MEASURED complete BPW
   2.9589935339460913, 3.1767685514394888, 3.5405118678698031,
   3.6138111608720234, 3.6299337236607006 (G0 payload definition).
   Evidence: §8.1 catalog parse; PACK_REPORT tensor_payload_bytes match.
2. They are three families, not one. Family A/A′ keep 2p0's 0.848 (or
   0.989) MLP and only change attention bits. B restores down to Q4. C
   is Q3-all-MLP. Evidence: §1.2 organ tables; PACK_REPORT recipes.
3. HEAD generates A/A′ with no code change. B and C are refused by
   assert_mixed_mlp_native. Evidence: §8.3; §1.3 admission table.
4. No generate receipt exists for any of the five. Evidence: §8.2.
5. Information-optimal first generate is mixed-q3mlp-v1 after accepting
   Uniform on MLP; if that patch is forbidden, first (and only) generate
   is mixed-floor-q7-v1. Do not run five. Evidence: §3.
6. 16 tokens × 6 prompts decides INCOHERENT only. COHERENT requires
   France@128 containing Paris and 17×19@256 containing 323. G0 id-match
   is not the gate. Evidence: §5.2–5.4; HARVEST_NOTE_G006; France seal
   decodes to a think preamble, no Paris.
7. A floor on one family does not license a differently allocated pack
   at the same BPW. Evidence: §2, §6.

EVIDENCE
- §8.1 catalog parse output (this lane; HQ38M20 nbytes + gravity JSON headers)
- crates/hawking-core/src/model/qwen38_hybrid_decode.rs:511-512, 601-667, 958-1004, 1395-1407, 316-317
- crates/hawking-core/src/model/qwen38_pack.rs:673-679
- crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs:1055-1058, 1143, 1305-1329
- crates/hawking-core/shaders/qwen38_device_activations.metal:325-341
- crates/hawking-core/examples/ascension_qwen38_hybrid_greedy.rs:36-40, 620-690
- receipts/ascent-2026-08-16/QWEN38_NATIVE_MIXED_2P0_GENERATE.json
- receipts/ascent-2026-08-16/QWEN38_COHERENCE_FLOOR_BRACKETED.json
- receipts/ascent-2026-08-16/QWEN38_COHERENCE_SEAL.json
- receipts/ascent-2026-08-16/HARVEST_NOTE_G006.json
- workspace/campaign/records/runs/qwen38-27b/{mixed-*-v1}/PACK_REPORT.json
- workspace/superwave/g1/g1-artifact-inventory.md:331-344
- workspace/superwave/g1/g1-baseline-remeasure.md:176-199
- workspace/superwave/g1/g1-doctor-tensor-map.md:218-224

CHANGES
Created workspace/superwave/g1/g1-bracket-bisection.md only.

TESTS
$ test -s workspace/superwave/g1/g1-bracket-bisection.md && echo 'test -s: PASS'
test -s: PASS
$ wc -l workspace/superwave/g1/g1-bracket-bisection.md
     797 workspace/superwave/g1/g1-bracket-bisection.md
$ git status --porcelain
?? workspace/superwave/g1/g1-bracket-bisection.md

RISKS
- Inventory "structurally loadable YES" will be misread as generate-ready
  for q3mlp/q4down. It is not, on HEAD.
- A Path A q7 INCOHERENT will be misread as "floor > 3.18". That is a
  family-A statement only.
- Resident holds the GPU lock and 14 GB. Oneshot is a second Metal
  process. 96 GB is enough on paper; a dirty box can still swap. Do not
  load mixed into the resident.
- PACK_REPORT BPW includes the catalog file. Mixing it with G0's payload
  definition shifts the fourth decimal.

UNRESOLVED
- Whether the GPU lane may land the assert relaxation. This lane cannot.
- When G0 first emits "Paris" (token index). 128 is ESTIMATED.
- Isolated organ floors. No overlay packs on disk.
- Live TOKEN_NS of any mixed pack. Forbidden this lane; not required to
  locate the floor.

NEXT
GPU/code owner: Path B (relax assert, generate q3mlp Phase A then
conditional Phase B). If Rust is frozen, Path A one-shot on floor-q7
Phase A and stop.
```
