# G1 mixed-sub15-v1 native gap

Lane: `30-sub15-native-gap`. No GPU, no generate, no inference.
Every number is `MEASURED` (this process, this tree or this artifact),
`RECEIPT` (quoted field), `SOURCE` (file:line), or `ESTIMATED`.

Artifact:
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1`

Packed complete BPW `1.2910781930062503` is a `RECEIPT` field
(`PACK_REPORT.json:23`) and is independently `MEASURED` as
`8 * 4340604637 / 26895998464` from the same file's
`all_required_weight_artifact_bytes` / `source_weight_elements`.

---

## 0. Verdict

Nothing in this pack needs a new codec family. Every GEMV codec already
has a Metal kernel that consumes the packed bytes (HGRAVR02 after the
existing host CSR expand). The pack is not natively loadable **today**
because it has no `catalog.hq38m20` and no `segments/`, so
`Qwen38HybridWeights::load` takes the uniform-Q4 path and eats the
reconstructed HQ30UQ4 vehicle at 4.2527 BPW. That is the expand-to-Q4
confound. Closing the gap is: emit HQ38M20 over the already-packed
blobs, accept f32v2 as catalog codec 4, and stop dispatching the
Q80-width `*_tg256` tiles on K∈{5120,6144,17408}.

No codec in this pack lacks a plausible direct kernel. Repack into a
higher-BPW family is **not** justified by kernel absence.

`HAWKING_QWEN38_RECON_FUSE=0` is a zero-shader workaround that already
walks every column (`gk_matvec_binary` → `gk_binary_group_serial_row`).
It is the cheapest honest generate experiment once the catalog exists.
Not run here.

---

## 1. What is on disk

`MEASURED` listing:

```
mixed-sub15-v1/
  FORMAT.md
  PACK_REPORT.json
  manifest.json                  schema hawking.ascent.qwen38_language_uniform_q4.v1
  packed/attn/                   304 *.rice + 304 *.json
  packed/attn_rows.json          304 rows
  packed/mlp_rows.json           192 rows
  tensors/                       402 *.hq30uq4 + 353 *.f32v2 + 353 *.f32bin
  catalog.hq38m20                ABSENT
  segments/                      ABSENT
```

`PACK_REPORT.json` recipe (`RECEIPT` 12–19):

| class | packed codec | tensors | elements | bytes | physical BPW |
|---|---|---:|---:|---:|---:|
| mlp.gate_proj | HGRAVB01 binary_g128 from mixed-2p0-v1 | 64 | 5704253440 | 802177344 | 1.1250234267290902 |
| mlp.up_proj | HGRAVR02 rice_q1_rms_2pct from mixed-2p0-v1 | 64 | 5704253440 | 918036000 | 1.2875108157887178 |
| mlp.down_proj | HGRAVS01 r160_b3 real-X from mixed-2p0-v1 | 64 | 5704253440 | 93847197 | 0.13161714918473189 |
| attention_gemv | rice_q1_rms_2pct from BF16 | 304 | 7237795840 | 1165098376 | 1.2877935788805008 |
| embed | HQ30UQ4 g64 oracle | 1 | 1271398400 | 675430440 | 4.250000251691366 |
| lm_head | HQ30UQ4 g64 oracle | 1 | 1271398400 | 675430440 | 4.250000251691366 |
| small | f32 oracle | 353 | 2645504 | 10584840 | 32.00853977162764 |
| **complete** | | **851 language** | **26895998464** | **4340604637** | **1.2910781930062503** |

The generate vehicle the packer wrote is a different object
(`PACK_REPORT.json:77–81`, packer docstring 13–15):

> HQ30UQ4 of reconstructed mixed/rice weights; packed BPW is the ledger above

`manifest.json` `complete_physical_bpw` is `4.252735126866492` — the
reconstructed Q4 catalog, not the packed ledger. `MEASURED` kinds:
402 `q4` + 353 `f32`. Embed and lm_head are hardlinked to
`uniform-q4-v1` (same inode). MLP and attention `.hq30uq4` files are
unique overwrites of decoded mixed/rice.

`QWEN38_SUB15_INCOHERENT.json` ran that Q4 vehicle (`RECEIPT`).
0 fallbacks, 220/264 cycle. That is not a native-codec verdict.

---

## 2. Loader entry

`SOURCE` `qwen38_hybrid_decode.rs:508-513`:

```
if root.join(QWEN38_MIXED_CATALOG_NAME).is_file() {
    return Self::load_mixed(root);
}
```

`QWEN38_MIXED_CATALOG_NAME = "catalog.hq38m20"` (`:34`).
Absent → `load_qwen38_manifest` → kind must be `q4` or `f32` (`:537-571`).
`packed/` is never opened. Rice bytes are invisible.

`parse_qwen38_mixed_catalog` (`:96-174`) joins every segment as
`root.join("segments").join(filename)` (`:124`). Even a catalog that
named `packed/attn/*.rice` would not resolve unless those files also
live under `segments/`.

`load_mixed` codec match (`:601-664`):

| catalog `codec` | magic accepted | destination |
|---:|---|---|
| 0 | `HGRAVB01` | `mixed` Binary |
| 1 | `HGRAVR02` | `mixed` Residual (host `expand_rice_indices` + `rice_q1_row_ptr`) |
| 2 | `HGRAVS01` | `mixed` Hgravs, geometry lock r160_b3 (`:1116-1134`) |
| 3 | `HGRAVU01` matrix | `mixed` Uniform |
| 3 | `HGRAVU01` vector, ≤65536 els, not a GEMV/embed/lm_head name | host dequant → `f32s` (`:612-620,938-956`) |
| 3 | `HQ30UQ4\0` | `q4` |
| other | — | **refuse** (`:659-663`) |

Packer already defines `CODEC_F32 = 4` (`tools/qwen38_sub15_pack.py:86`)
and never writes it. `mixed_gpu_layout` (`q80_mixed_decode.rs:1331`)
also rejects `other`.

MLP role lock `assert_mixed_mlp_native` (`:958-1003`), by **name**, not
by catalog `organ` (parser never reads `organ`):

- every `mlp.gate_proj.weight` must be `MixedGpuWeight::Binary`
- every `mlp.up_proj.weight` must be `MixedGpuWeight::Residual`
- every `mlp.down_proj.weight` must be `MixedGpuWeight::Hgravs`

Missing or wrong kind refuses. There is **no** attention role lock.
`encode_named_matvec` (`:1206-1221`) dispatches whatever kind is in
`mixed` or `q4` for that name. Split DeltaNet names are already wired
(`encode_deltanet_mixed` `:2979-2999`, `encode_split_deltanet_projections`
`:1522-1562`). Embed accepts HGRAVU01 or HQ30UQ4 (`:2858-2908`).

Once `mixed` is non-empty, `encode_embed` / `encode_deltanet` /
`encode_gqa` / `encode_dense_mlp` / `encode_terminal` all take the mixed
branch (`:2523,2555,2606,2708,2834`). An HQ38M20 that omitted mixed MLP
cannot open (lock). An HQ38M20 that catalogued rice attention would
dispatch it.

---

## 3. Per-role census of the packed artifact

### 3.1 Attention rice — `MEASURED` 304 / 304 `HGRAVR02`

`packed/attn_rows.json` plus every `*.rice` header:

| role | n | shape | layers | magic | schema | g | cols % g | cosine vs BF16 |
|---|---:|---|---|---|---|---:|---:|---|
| `in_proj_qkv` | 48 | 10240×5120 | DN | `HGRAVR02` | `hawking.gravity.binary_outlier_residual.v2` | 128 | 0 | 0.8344–0.8474 |
| `in_proj_z` | 48 | 6144×5120 | DN | same | same | 128 | 0 | same band |
| `in_proj_a` | 48 | 48×5120 | DN | same | same | 128 | 0 | same band |
| `in_proj_b` | 48 | 48×5120 | DN | same | same | 128 | 0 | same band |
| `out_proj` | 48 | 5120×6144 | DN | same | same | 128 | 0 | same band |
| `q_proj` | 16 | 12288×5120 | GQA | same | same | 128 | 0 | same band |
| `k_proj` | 16 | 1024×5120 | GQA | same | same | 128 | 0 | same band |
| `v_proj` | 16 | 1024×5120 | GQA | same | same | 128 | 0 | same band |
| `o_proj` | 16 | 5120×6144 | GQA | same | same | 128 | 0 | same band |

All 304: `index_mode=rice`, `value_bits=1`, `value_scale=rms`,
`rice_k=5`, file size == `packed_bytes`, sha256(name) stem matches.
`cols ∈ {5120,6144}`, both divisible by 128 and by 256.
Outlier count sum `MEASURED` 144_756_064 → CSR `u32` indices 579_024_256 B
+ row_ptr 5_393_600 B at load. Under the 20 GB cap.

Cosine 0.834–0.847 is **pack metadata**, not a generate verdict. The
expand-to-Q4 vehicle already encoded these reconstructions; its
incoherence cannot be blamed on a second quant alone, and cannot be
read as a native-codec result either.

### 3.2 MLP — `MEASURED` 192 / 192, byte-identical to mixed-2p0-v1

`packed/mlp_rows.json` vs `mixed-2p0-v1/catalog.hq38m20`: 0 missing,
0 (`codec`,`nbytes`,`elements`) mismatches.

| organ | catalog codec | magic | shape | g / rank | n |
|---|---:|---|---|---|---:|
| 0 gate | 0 | `HGRAVB01` / `hawking.gravity.binary_sign_scale.v1` | 17408×5120 | g=128, groups=696320 | 64 |
| 1 up | 1 | `HGRAVR02` / rice_q1_rms | 17408×5120 | g=128 | 64 |
| 2 down | 2 | `HGRAVS01` / `activation_weighted_svd_low_rank_q` | 5120×17408 | rank=160, bits=3, g=64 | 64 |

Payloads live only in `mixed-2p0-v1/segments/Lxx.hq38seg`.
`materialize_mlp` (`qwen38_sub15_pack.py:402-411`) decodes them and
writes reconstructed HQ30UQ4. It does not copy packed bytes into
mixed-sub15-v1.

### 3.3 Embed / lm_head / small — `MEASURED`

| role | n | on-disk | magic / header | inode |
|---|---:|---|---|---|
| `embed_tokens.weight` | 1 | `tensors/*.hq30uq4` 675_430_440 B | `HQ30UQ4\0` | same as uniform-q4-v1 `314847693` |
| `lm_head.weight` | 1 | same | `HQ30UQ4\0` | same as uniform-q4-v1 `315002072` |
| `input_layernorm` | 64 | `*.f32v2` | u64 numel + f32, already HF δ (L0 first=0.046875) | oracle |
| `post_attention_layernorm` | 64 | f32v2 | same | oracle |
| `model.norm` | 1 | f32v2 | same | oracle |
| `q_norm` / `k_norm` | 16+16 | f32v2 | same | oracle |
| `linear_attn.norm` | 48 | f32v2 | same | oracle |
| `A_log` / `dt_bias` | 48+48 | f32v2 | same | oracle |
| `conv1d.weight` | 48 | f32v2, 40960 els < 65536 | same | oracle |

`read_qwen38_f32_payload` (`qwen38_pack.rs:751-767`) is the reader.
Do **not** run `mlx_residual_norm_to_delta` on these; they are already
δ. (HGRAVU01 small tensors from mixed-2p0 *do* get that subtract.)

---

## 4. Per-codec table (loader × kernel)

Legend: **ACCEPTED** = `load_mixed` match-arm would take it.
**REFUSED** = unknown codec or role lock.
**IGNORED** = bytes exist but current `load()` never sees them.
**KERNEL-OK** = production bind walks all K for this shape.
**KERNEL-BOUND-WRONG** = a consuming kernel exists, the bound tile
drops K>2048. **KERNEL-MISSING** = no plausible in-register consumer.

| packed codec | roles in this artifact | today, no catalog | if HQ38M20 emitted as below | role lock | production kernel | K coverage on this model |
|---|---|---|---|---|---|---|
| 0 `HGRAVB01` | 64× `mlp.gate_proj` 17408×5120 | IGNORED (sibling only) | ACCEPTED | **required** Binary | `q80_binary_group_matvec_tg256` `:701` | **KERNEL-BOUND-WRONG**. `lid*8` covers 2048 cols. K=5120. |
| 1 `HGRAVR02` | 64× `mlp.up_proj` 17408×5120 | IGNORED (sibling only) | ACCEPTED | **required** Residual | `q80_binary_group_csr_matvec_tg256` `:743` | **KERNEL-BOUND-WRONG**. Same 2048-col bind. |
| 1 `HGRAVR02` | 304× attention (table 3.1) | IGNORED (`packed/attn`) | ACCEPTED | none | same CSR tg256 | **KERNEL-BOUND-WRONG**. K∈{5120,6144}. |
| 2 `HGRAVS01` r160_b3 | 64× `mlp.down_proj` 5120×17408 | IGNORED (sibling only) | ACCEPTED | **required** Hgravs + r160_b3 | two × `q80_hgravs01_factor_matvec_simd3` `:845` | **KERNEL-OK**. `col += 256` tiles. Rank lock matches header. |
| 3 `HQ30UQ4` | embed, lm_head | ACCEPTED via uniform path | ACCEPTED as codec 3 | embed must be Uniform or Q4 | `qwen_uniform_q4_embedding_lookup` / `geo_tpr64_tg128` | **KERNEL-OK**. G0 kernels. |
| 3 `HGRAVU01` | **not in this pack** | — | — | — | `q80_hgravs01_factor_matvec_simd` tiles all K (`:521`) | n/a |
| 4 `CODEC_F32` / f32v2 | 353 small | ACCEPTED via uniform path only | **REFUSED** (`unknown mixed codec 4`) | must land in `f32s` or rmsnorm fails | `qwen80_residual_rmsnorm_f32` etc. | **KERNEL-OK** once uploaded as f32 |
| reconstructed HQ30UQ4 of mixed/rice | 400 unique GEMV files | ACCEPTED (this is what load does) | never opened if catalog present | — | geo_tpr64 | KERNEL-OK, **forbidden vehicle** |

No row is KERNEL-MISSING.

Tiling siblings that already consume these exact containers for any K
divisible by 8/256 (this pack's K all are):

| bound (wrong for Q38) | already-compiled sibling (covers K) | file |
|---|---|---|
| `q80_binary_group_matvec_tg256` | `q80_binary_group_matvec_simd_bytes` `:620` (`base += 256`) | `q80_mixed_decode.metal` |
| `q80_binary_group_csr_matvec_tg256` | `q80_binary_group_csr_matvec_bytes` `:796` (same loop) | same |
| same | `gk_matvec_binary` → `gk_binary_group_serial_row` `:116` (all groups) | `gk_family.metal` |
| same | `q80_binary_group_matvec` `:46` (serial, all cols) | `q80_mixed_decode.metal` |

`decode_family.rs:18-20` **names** the 2048-wide tiles as the shipping
occupancy bind. `q80_uniform8_matvec_tg256` `:991-992` documents the
fix those authors already applied to Q8: "Loops 2048-col tiles so
4096-col out_proj is covered." Binary/CSR tg256 never got that loop.

`HAWKING_QWEN38_RECON_FUSE` default ON (`:43-45`). OFF selects the
serial walk. mixed-2p0 native generate (`QWEN38_NATIVE_MIXED_READER.json`)
used the default tiles, `fallbacks_total: 0`. 0 fallbacks does not
mean the tile consumed K=5120. That generate is kernel-confounded for
every HGRAVB01/HGRAVR02 GEMV. Not re-run here.

---

## 5. Ordered change list

Sizes are ESTIMATED added/changed lines. No file in this lane is
modified except this report.

### C1. Emit `catalog.hq38m20` + `segments/` over existing blobs
**Must.** Without this, `load()` never leaves the Q4 vehicle.

- File: `tools/qwen38_sub15_pack.py` (add `write_hq38m20` / new phase
  `catalog`) **or** new `tools/qwen38_sub15_emit_hq38m20.py`.
- Reuse record layout from `lab/operators/q80_mixed_representation_pack.py:631-716`
  (`write_catalog`, 128 B records) with magic `HQ38M20\0`. Inverse already
  exists as `read_mixed_catalog` (`qwen38_sub15_pack.py:251`).
- 851 records, one per language tensor:

  | rows | codec | organ | payload source |
  |---:|---:|---:|---|
  | 64 | 0 | 0 GATE | slice `mixed-2p0-v1/segments/Lxx.hq38seg` at catalog offset |
  | 64 | 1 | 1 UP | same |
  | 64 | 2 | 2 DOWN | same |
  | 304 | 1 | 3 ATTN | hardlink `packed/attn/<sha>.rice` |
  | 1 | 3 | 4 EMB | hardlink oracle `tensors/<sha>.hq30uq4` |
  | 1 | 3 | 5 HEAD | same |
  | 353 | 4 | 6 SMALL | hardlink oracle `tensors/<sha>.f32v2` |

- Segment path contract: `root/segments/<filename>` (`:124`). Hardlink;
  do not copy 4.34 GB.
- Do not re-encode. Do not touch BF16 or `uniform-q4-v1`.
- ESTIMATED 200–300 lines. CPU only.

### C2. `load_mixed` codec 4 = f32v2
**Must**, or C1's small tensors refuse (`:659-663`) and every rmsnorm
dies on `missing f32`.

- File: `crates/hawking-core/src/model/qwen38_hybrid_decode.rs`
- Function: `Qwen38HybridWeights::load_mixed` (`:583`)
- Arm: `4 => { values = read_qwen38_f32_payload(&payload)?; f32s.insert(...) }`
- Do **not** call `mlx_residual_norm_to_delta_named` (already δ).
- Optional: reject payloads that are not `8+4*numel`.
- ESTIMATED 30–50 lines.

### C3. Bind a K-complete HGRAVB01 / HGRAVR02 kernel
**Must** for a native generate to be a codec verdict.

Pick one, in this order:

1. **Host retarget** in `dispatch_binary` (`:1323`) and
   `dispatch_residual` (`:1347`): if `cols > 2048` (or always), dispatch
   `q80_binary_group_matvec_simd_bytes` / `q80_binary_group_csr_matvec_bytes`
   with grid `ceil(rows/8)*256`, TG 256. ESTIMATED 20–40 lines. No new
   shader.
2. **Or** add the 2048-col tile loop that `q80_uniform8_matvec_tg256`
   already has (`:1016`) into the two `*_tg256` kernels. ESTIMATED 15
   lines × 2 in `q80_mixed_decode.metal`.
3. **Or**, generate-only workaround: `HAWKING_QWEN38_RECON_FUSE=0`.
   Zero lines. Serial. Honest. Slow. ESTIMATED TOKEN_NS is not claimed.

Do not invent a new rice kernel. Host CSR expand (`upload_mixed`
`:1071-1114`, `expand_rice_indices` `q80_mixed_decode.rs:485`,
`rice_q1_row_ptr` `:462`) is the existing HGRAVR02 contract.

### C4. Contract tests, no GPU
**Should.**

- Extend `mixed_catalog_contract_tests` (`qwen38_hybrid_decode.rs:3795`):
  codec 4 is a defined arm; unknown 5 still refuses.
- CPU: `mixed_gpu_layout(1, rice_bytes)` on one `in_proj_qkv` and one
  `in_proj_a` (48×5120); `mixed_gpu_layout(0/1/2, …)` on the three L0
  MLP slices. ESTIMATED 80–120 lines.
- Do not call `Qwen38HybridWeights::load` (opens Metal).

### C5. Do not change

- `assert_mixed_mlp_native` — this recipe already satisfies it.
- `encode_deltanet_mixed` / `encode_gqa_mixed` / `encode_embed_mixed` /
  `encode_named_matvec` — split rice names and HQ30UQ4 embed/lm_head
  already dispatch.
- `QWEN38_MIXED_HGRAVS_*` locks — L0 down header is rank 160 / bits 3 /
  group 64.
- New Metal codec. No KERNEL-MISSING row.
- Repack attention to HGRAVU01. That would raise attention from 1.288
  BPW to ~4.25 (bits=4) or drop the 2% residual (bits=1, ~1.125 BPW)
  for no kernel reason.
- Touch the live resident, AgentOS, or any tracked file except this
  report.

### Size of the close

ESTIMATED 250–500 lines across packer + `load_mixed` + dispatch bind +
tests. No new shader family. After C1–C3 the packed 1.291 BPW bytes
are what `step` would address.

---

## 6. Why not repack

The only codec that would justify a higher-BPW repack is one with no
plausible direct kernel. Every codec here has one:

- HGRAVB01 / HGRAVR02: in-register binary ± CSR residual. Bind is
  wrong; the consumer exists.
- HGRAVS01 r160_b3: two-stage factor, already K-complete.
- HQ30UQ4: G0 kernels, already K-complete.
- f32v2: not a GEMV.

Repacking rice attention into HGRAVU01-q4 would abandon the 1.291
ledger and re-enter the 4.25 attention band that mixed-2p0 already
used. That is a different experiment.

---

## 7. Cheapest next measurement (not this lane)

After C1+C2, one `ascension_qwen38_hybrid_greedy` on mixed-sub15-v1
with `HAWKING_QWEN38_RECON_FUSE=0` (or after C3). That is the first
non-expand generate of this artifact. GPU lane owns it.

Do not treat mixed-2p0 native INCOHERENT as evidence about HGRAVB01 /
HGRAVR02 quality on K=5120 until C3 (or fuse-off) is in the vehicle.

---

## 8. Evidence

### 8.1 PACK_REPORT.json (RECEIPT)

```
:2-3   "status": "PACKED"
:12-19  recipe gate HGRAVB01 / up HGRAVR02 / down HGRAVS01 /
        attention rice_q1_rms_2pct / embed+lm_head HQ30UQ4 / small f32
:23     "complete_physical_bpw": 1.2910781930062503
:24     "all_required_weight_artifact_bytes": 4340604637
:25     "source_weight_elements": 26895998464
:77-81  generate_vehicle schema hawking.ascent.qwen38_language_uniform_q4.v1
        note: "HQ30UQ4 of reconstructed mixed/rice weights"
```

### 8.2 Packer (SOURCE, `git show HEAD:tools/qwen38_sub15_pack.py`)

```
:13-15  generate vehicle = hard-linked uniform-q4-v1 with overwritten
        Q4 of reconstructed mixed/rice; hybrid_greedy speaks HQ30UQ4+f32v2
:82-86  CODEC_BINARY=0 RESIDUAL=1 HGRAVS01=2 UNIFORM4=3 F32=4
:160-170 encode_rice = encode_residual_compact(outlier_ratio=0.02,
         group_size=GROUP_BINARY, index_mode=rice, value_bits=1, value_scale=rms)
:383-395 attn_source_names: unfused qkv/z/a/b/out + GQA q/k/v/o
:402-411 materialize_mlp: decode mixed-2p0, overwrite_q4 only
```

### 8.3 Loader / kernels (SOURCE)

```
qwen38_hybrid_decode.rs:34,508-513,583-664,938-1003,1116-1134,
    1206-1241,1323-1374,1522-1562,2522-2525,2606-2607,2858-2908
q80_mixed_decode.rs:38-41 MAGIC_*; :462 rice_q1_row_ptr; :485 expand_rice_indices;
    :1174-1332 mixed_gpu_layout codecs 0-3 only
q80_mixed_decode.metal:699-700 "One 256-thread TG per row; each lane dots 8 columns"
    :722-728 col = lid*8; if col+8<=cols  → 256*8=2048 cols
    :620-648 simd_bytes tiles base+=256
    :796-828 csr_matvec_bytes same
    :991-1016 uniform8 tg256 "Loops 2048-col tiles"
gk_family.metal:116-160 serial walks all groups_per_row
decode_family.rs:18-20 MATVEC_BINARY_TILES / CSR_TILES = the tg256 names
qwen38_pack.rs:751-767 read_qwen38_f32_payload
```

### 8.4 On-disk census (MEASURED, this process)

```
catalog.hq38m20: No such file or directory
segments/: absent
packed/attn: 304 rice + 304 json, all magic b'HGRAVR02'
rice schemas: hawking.gravity.binary_outlier_residual.v2 × 304
index_mode rice × 304, value_bits 1 × 304, value_scale rms × 304, group 128 × 304
cols,cols%128: (5120,0)×240  (6144,0)×64
attn bytes 1165098376 = PACK_REPORT attention_gemv_rice.bytes
mlp vs mixed-2p0 catalog: missing 0 mismatch 0
L0 gate magic HGRAVB01 shape [17408,5120] g=128
L0 up   magic HGRAVR02 shape [17408,5120] rice/1/rms
L0 down magic HGRAVS01 rank=160 factor_bits=3 factor_group_size=64
embed/lm_head HQ30UQ4, inodes 314847693 / 315002072 shared with uniform-q4-v1
f32v2 L0 input_layernorm numel=5120 first=0.046875 (already δ)
manifest schema hawking.ascent.qwen38_language_uniform_q4.v1
  402 q4 + 353 f32, complete_physical_bpw 4.252735126866492
outlier_count_sum 144756064
```

### 8.5 Confounded generate (RECEIPT)

`receipts/ascent-2026-08-16/QWEN38_SUB15_INCOHERENT.json`:
vehicle = reconstructed HQ30UQ4; `fallbacks: 0`; 220/264 cycle.
`speed_caveat_carried`: 79.44 TPS was never measured.

`QWEN38_NATIVE_MIXED_READER.json`: mixed-2p0-v1 native generate
`reconstruct_to_q4: false`, `fallbacks_total: 0`, INCOHERENT.
Used the same default `*_tg256` binds. Not evidence about K=5120
HGRAVB01/HGRAVR02.

### 8.6 Wave-1 inventory (already paid)

`g1-artifact-inventory.md:355-357,372-373`: packed rice not read by
uniform loader; reconstructed Q4 still 4.2527 BPW.
`g1-kernel-inventory.md:396-413,490-498`: codecs 0–3, MLP lock, no
codec 4, no attention-only mixed catalog.

---

## 9. What this lane did not measure

- Any Metal load or generate.
- Whether `*_simd_bytes` is bit-identical to the serial oracle on
  17408×5120 / 48×5120 (cheapest CPU check: `binary_rice_q1_matvec_f32`
  vs a one-row capture; not run).
- Token ns / TPS of a native sub15 graph. The packer's 79.44 TPS is
  PROJECTED from packed bytes against a 4.2527 wall and is retired as
  a speed claim.
- Coherence of the packed representation.

```
STATUS
IMPLEMENT_READY

CLAIMS
1. mixed-sub15-v1 is packed at 1.2910781930062503 BPW and is not what load() reads. Evidence: PACK_REPORT.json:23-25; catalog.hq38m20 absent; qwen38_hybrid_decode.rs:508-513; manifest complete_physical_bpw 4.252735126866492.
2. Every packed GEMV codec is one of {HGRAVB01, HGRAVR02, HGRAVS01 r160_b3, HQ30UQ4}. Evidence: §3 census; L0 MLP headers; 304/304 rice magic HGRAVR02.
3. load_mixed would ACCEPT codecs 0/1/2 and codec-3 HQ30UQ4, REFUSE codec 4 / unknown, and lock MLP to Binary/Residual/Hgravs. Attention HGRAVR02 is not role-locked. Evidence: qwen38_hybrid_decode.rs:601-664,938-1003,1206-1221,2858-2908.
4. Production HGRAVB01/HGRAVR02 tiles cover 2048 columns. Every such GEMV in this pack has K∈{5120,6144}. Tiling siblings already exist. Not KERNEL-MISSING. Evidence: q80_mixed_decode.metal:722-728 vs :620-648 and :796-828; shapes in §3.
5. Closing native load+generate is C1 catalog emit + C2 codec-4 f32v2 + C3 K-complete bind (or fuse-off). ESTIMATED 250–500 lines. No new shader family. Evidence: §5.
6. Repack is not justified by kernel absence. Evidence: §4 table, no KERNEL-MISSING row.

EVIDENCE
PACK_REPORT.json:23-25,77-81
tools/qwen38_sub15_pack.py:13-15,82-86,160-170,251,402-411
qwen38_hybrid_decode.rs:34,508-664,958-1003,1323-1374
q80_mixed_decode.metal:620-648,699-728,743-793,796-828,991-1016
q80_mixed_decode.rs:462,485,1174-1332
gk_family.metal:116-160
This process: §8.4 census (304 rice, 0 mlp mismatch, no catalog)

CHANGES
workspace/superwave/g1/g1-sub15-native-gap.md (this file only)

TESTS
test -s workspace/superwave/g1/g1-sub15-native-gap.md
wc -l workspace/superwave/g1/g1-sub15-native-gap.md
git status --porcelain

RISKS
C1 without C2 refuses 353 small tensors. C1+C2 without C3 (or fuse-off) yields a native generate that silently drops 60% of K on every binary/rice GEMV and will be misread as a codec verdict. Catalog organ field is unused; a wrong organ is silent. Do not mlx-delta f32v2.

UNRESOLVED
Native coherence of this 1.291 BPW recipe. Attention rice cosine 0.834–0.847 (MEASURED pack metadata) plus down_proj 0.1316 BPW (wave-1 floor still open). Cheapest honest generate: C1+C2 then HAWKING_QWEN38_RECON_FUSE=0. GPU lane.

NEXT
Implement C1–C3 on a later write-enabled lane. Serialized GPU lane runs the first native greedy. Do not reopen generator+residual. Do not treat expand-to-Q4 INCOHERENT as the packed-codec result.
```
