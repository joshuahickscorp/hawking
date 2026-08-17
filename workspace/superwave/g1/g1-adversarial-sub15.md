# G1 adversarial — mixed-sub15-v1 native-load claim

Lane: `64-adversarial-sub15`. No GPU, no generate, no inference, no tracked-file edits.
Every number is `MEASURED` (this process), `SOURCE` (file:line), or `RECEIPT`.
Attack target: `g1-sub15-native-gap.md` IMPLEMENT_READY close
("C1 catalog + C2 codec-4 + C3 K-complete bind, ESTIMATED 250–500 lines, no new shader family").

Artifact roots:

```
mixed-sub15-v1 = /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1
mixed-2p0-v1   = /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1
uniform-q4-v1  = /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1
```

---

## 0. Verdict

The no-new-shader half **survives**. Every GEMV shape in this pack is covered by an
already-compiled K-complete sibling. The K-complete item is a bind, not a missing
accumulator. The MLP role lock is already satisfied. The packed *ledger* is
complete at measured BPW `1.2910781930062503`.

The IMPLEMENT_READY close is **wrong as a critical-path bet**. C1 is not "emit
HQ38M20 over existing blobs in this pack": the 1.814 GB of MLP packed payloads
are not in `mixed-sub15-v1`. C2 is not a local match-arm: any non-empty `mixed`
map flips embed / DeltaNet / GQA / MLP / terminal onto the mixed graph, and
copying the codec-3 `mlx_residual_norm_to_delta` path onto already-δ `f32v2`
destroys norms. C1+C2 without C3 (or `HAWKING_QWEN38_RECON_FUSE=0`) is a
silent 2048-col truncate on every binary/rice GEMV. `encode_named_matvec`
prefers `q4` over `mixed`, so a catalog that also names the reconstructed
vehicle GEMVs recreates the expand-to-Q4 confound. 355 files in this pack,
including embed, lm_head, and every norm, share G0 inodes.

A later native generate of this one recipe at this one BPW would still not
locate the Qwen3.8 coherence floor.

`STATUS` of the attacked claim: **FALSIFIED**.

---

## 1. Do all tensors map to an existing tile?

**No shape in this pack falls outside the sibling tiles.** Production binds do.

`MEASURED` unique packed GEMV shapes (304 rice + 192 MLP):

| n | codec | shape | K | K%256 | rows%8 |
|---:|---|---|---:|---:|---:|
| 64 | HGRAVB01 | 17408×5120 | 5120 | 0 | 0 |
| 64 | HGRAVR02 | 17408×5120 | 5120 | 0 | 0 |
| 48 | HGRAVR02 | 10240×5120 | 5120 | 0 | 0 |
| 48 | HGRAVR02 | 6144×5120 | 5120 | 0 | 0 |
| 48 | HGRAVR02 | 48×5120 | 5120 | 0 | 0 |
| 48 | HGRAVR02 | 48×5120 | 5120 | 0 | 0 |
| 48 | HGRAVR02 | 5120×6144 | 6144 | 0 | 0 |
| 16 | HGRAVR02 | 12288×5120 | 5120 | 0 | 0 |
| 16 | HGRAVR02 | 1024×5120 | 5120 | 0 | 0 |
| 16 | HGRAVR02 | 1024×5120 | 5120 | 0 | 0 |
| 16 | HGRAVR02 | 5120×6144 | 6144 | 0 | 0 |
| 64 | HGRAVS01 | 5120×17408 = (5120×160)(160×17408) | 17408 / 160 | 0 / 160%8=0 | 0 |

48×5120 is the only shape that looks exotic versus the Q80 expert
`[512, 2048]` the kernels were named for. It is not exotic to the code:
`q80_binary_group_matvec_simd_bytes` / `q80_binary_group_csr_matvec_bytes`
take `rows`/`cols` as parameters, 8 rows/TG, `row >= rows` return.
`48/8 = 6` TGs. mixed-2p0 already dispatched 48×5120 as HGRAVU01 via the
split path (`catalog.hq38m20` has `in_proj_a` `[48,5120]` codec 3, no
`in_proj_qkvz`). What has not been the vehicle is 48×5120 as **Residual**.

`q80_binary_byte_dot` requires `col % 8 == 0` and "lie inside a single
scale group", documented as true for `group_size=128` (`q80_mixed_decode.metal:295-296`).
All rice/binary here are `g=128`. Group boundaries are 128-aligned, hence
8-aligned. No shape violates it.

Embed / lm_head are HQ30UQ4 `[248320,5120]`, G0 kernels. 353 small tensors
are not GEMVs.

Production bind (`decode_family.rs:18-20`, `dispatch_binary` / `dispatch_residual`
`:1323-1360`) is `*_tg256`: `col = lid * 8` with no tile loop
(`q80_mixed_decode.metal:722-728`, `:768-773`). `256*8 = 2048`. Every
HGRAVB01/HGRAVR02 K in this pack is 5120 or 6144, so the **shipping** tile
covers 2048/5120 = 0.400 or 2048/6144 = 0.333 of K. That is not "some
shape outside siblings". It is the wrong bind on shapes the siblings cover.

HGRAVS is already on `q80_hgravs01_factor_matvec_simd3` (`dispatch_factor`
bits==3, `:1402-1403`). That kernel loops `col += 256` (`:869`).
Left factor cols=160 is the case the comment at `:866-868` already fixed.
Right factor cols=17408, `17408/256 = 68` exactly. Rank lock matches all
64 downs (`MEASURED` §3).

---

## 2. Is K-complete a bind, or a kernel that cannot walk K?

**Bind.** The shipping tile drops K. The siblings and the fuse-off path
accumulate the full K.

| kernel | K walk | SOURCE |
|---|---|---|
| `q80_binary_group_matvec_tg256` | one `lid*8` window, 2048 cols, no loop | `q80_mixed_decode.metal:722-728` |
| `q80_binary_group_csr_matvec_tg256` | same 2048-col binary; CSR tail on lid 0 walks **all** nnz | `:768-790` |
| `q80_binary_group_matvec_simd_bytes` | `for (base = 0; base < cols; base += 256)` | `:641-648` |
| `q80_binary_group_csr_matvec_bytes` | same binary loop + full-row CSR | `:821-838` |
| `q80_binary_group_matvec` / `gk_binary_group_serial_row` | `for group in groups_per_row` | `gk_family.metal:128-159` |
| `q80_sparse_q1_apply_csr` | `for n in [row_ptr[row], row_ptr[row+1])` | `q80_mixed_decode.metal:172-178` |
| `q80_hgravs01_factor_matvec_simd3` | `col += 256` | `:869` |
| `q80_uniform8_matvec_tg256` | `tile += 2048` (the loop binary tg256 never got) | `:991-1016` |

`simd_bytes` / `csr_matvec_bytes` are in the default metallib
(`metal/mod.rs:439` `SHADER_Q80_MIXED_DECODE` inside `all_shader_sources`)
and in the trace-name table (`:1121, :1132`). C3 option 1 is a retarget
to names that already resolve. Argument layout matches `encode_binary_args`
/ `encode_binary_csr_args` (buffers 0–7 / 0–11). Grid is `simd8_grid`
(`ceil(rows/8)*256`, TG 256), which is what those kernels document.

Fuse-off (`HAWKING_QWEN38_RECON_FUSE=0`) already walks full K:
`gk_matvec_binary` + `q80_sparse_q1_apply_csr` (`:1361-1373`). Zero new
lines. Honest. Slow. Not a TOKEN_NS claim.

The kernel does **not** structurally refuse K∈{5120,6144,17408}. A native
generate on the **default** bind will.

---

## 3. Codec-4 f32v2 — second-order dispatch consequences

C2 as written (`g1-sub15-native-gap.md:281-290`) is one `match` arm that
`read_qwen38_f32_payload`s into `f32s`. That arm is necessary if C1 emits
codec 4. It is not local.

1. **Graph flip.** `load_mixed` does not touch the uniform `load()` path.
   But once `mixed` is non-empty, every token-graph entry takes the mixed
   branch: `encode_embed` `:2523`, `encode_dense_mlp` `:2555`,
   `encode_deltanet` `:2606`, `encode_gqa` `:2708`, `encode_terminal` `:2834`.
   All 353 small names must then exist in `f32s` or the first rmsnorm /
   conv1d / A_log / dt_bias fails. `load_mixed` has **no** completeness
   check (`:583-673` only `rows.is_empty()` + `assert_mixed_mlp_native`).
   Uniform `load()` requires 755 (`:515-519`, `QWEN38_EXPECTED_CATALOG_TENSORS`).
   Mixed does not.

2. **Do not subtract 1.** `mlx_residual_norm_to_delta_named` (`:51-63`)
   runs on the codec-3 HGRAVU01 vector arm (`:612-620`). mixed-2p0 small
   tensors are HGRAVU01 (L0 input_layernorm catalog `nbytes=2977`,
   `codec_bpw≈4.65`). sub15 small tensors are already-δ f32v2
   (`MEASURED` L0 input_layernorm first f32 = `0.046875`, numel=5120,
   size=20488=`8+4*5120`). Copying the codec-3 arm onto codec 4 yields
   `δ-1`. rmsnorm then runs on garbage. C2's "do not call mlx-delta" is
   load-bearing, not style.

3. **`mixed_gpu_layout` still refuses 4** (`q80_mixed_decode.rs:1331`).
   Harmless if codec 4 never enters `upload_mixed`. Fatal if C2 is wired
   through `upload_mixed` instead of `read_qwen38_f32_payload`.

4. **`encode_named_matvec` prefers q4** (`:1213-1215`). Cataloguing a
   reconstructed HQ30UQ4 GEMV (the vehicle files already in `tensors/`)
   *and* the packed mixed tensor of the same name puts the name in both
   maps. Dispatch uses Q4. Silent expand-to-Q4. The INCOHERENT receipt
   returns.

5. **conv1d is 3-D** `[10240,4,1]` = 40960 els (2p0 catalog; sub15
   manifest kind `f32`). `read_qwen38_f32_payload` is numel-only
   (`qwen38_pack.rs:751-767`). `hgravu_is_vector` does not treat
   `conv1d.weight` as a GEMV (`:942-950`), so a codec-3 clone of 2p0
   conv1d host-dequants (40960 ≤ 65536, `:202-206`). Either C1 design
   works. Mixing them does not.

6. **Avoiding C2 by cloning 2p0 small as codec 3 is a different
   artifact.** It binds norms to HGRAVU01 payloads in 2p0 segments, not
   to the sub15 f32v2 oracle files the ledger counted. Embed/lm_head
   also differ: 2p0 codec-3 `nbytes=675430686` vs sub15 HQ30UQ4
   `675430440` (`MEASURED`). That is not this pack.

G0 uniform `load()` never enters `load_mixed`. A C2 source change is
dead code on the live G0 path unless someone also drops
`catalog.hq38m20` into `uniform-q4-v1`.

---

## 4. MLP role lock

The lock can be satisfied **without weakening**. Do not touch it.

`assert_mixed_mlp_native` (`:958-1003`) is by name, not `organ`:

- every `mlp.gate_proj.weight` → `MixedGpuWeight::Binary`
- every `mlp.up_proj.weight` → `MixedGpuWeight::Residual`
- every `mlp.down_proj.weight` → `MixedGpuWeight::Hgravs`

`MEASURED` 192/192 `mlp_rows.json` vs mixed-2p0 catalog: 0 missing, 0
`(codec,nbytes,elements)` mismatches. All 64 downs parse as
`HGRAVS01` rank=160 bits=3 group=64 left `[5120,160]` right `[160,17408]`
representation `activation_weighted_svd_low_rank_q`. `upload_mixed`
re-checks that geometry (`:1116-1134`). `n_fit_rows=256` on all 64 downs
(2p0 catalog). That is the doctor underdetermined fit, not a lock miss.

The lock exists to refuse reconstructed-Q4 MLP and attention-only
HQ38M20 catalogs (`g1-kernel-inventory.md:494-498`). Weakening it to
"get the pack to load" would re-open the exact confound this campaign
already paid for. This recipe does not need that.

Attention HGRAVR02 is Residual in the same `mixed` map. No attention
role lock (`:1206-1221` dispatches whatever kind is present). That is
fine.

---

## 5. Catalog emit vs live G0

G0 `load()` of `uniform-q4-v1` takes the mixed path only if
`uniform-q4-v1/catalog.hq38m20` exists (`:508-513`). It does not.
Writing `catalog.hq38m20` + `segments/` under **mixed-sub15-v1** does
not change G0 control flow, dispatch, or the 755-tensor Q4 reader.

What G0 **does** share with this pack is 355 inodes (`MEASURED`):

| name | sub15 inode | G0 inode |
|---|---:|---:|
| `embed_tokens.weight` | 314847693 | 314847693 |
| `lm_head.weight` | 315002072 | 315002072 |
| 353 `*.f32v2` | same as G0 | same as G0 |

400 vehicle Q4 GEMVs (192 MLP + 208 fused/GQA attention) are unique
overwrites. `os.link` of a shared inode into `segments/` is a new
dirent and is safe. `unlink` + rewrite (the packer's `overwrite_q4`,
`:365-371`) of embed / lm_head / any f32v2 **is a live G0 corruption**.
C1 must only create new dirents.

`parse_qwen38_mixed_catalog` joins `root/segments/<filename>` (`:124`).
A catalog that names `packed/attn/*.rice` without also linking those
files under `segments/` will not open. Native-gap already says this.

`load_mixed` does not read `organ`. A wrong organ is silent. Name +
codec + magic are the contract.

---

## 6. Independent pack completeness

### 6.1 Tree (`MEASURED`)

```
mixed-sub15-v1/
  FORMAT.md
  PACK_REPORT.json
  manifest.json          schema hawking.ascent.qwen38_language_uniform_q4.v1
  packed/attn/           304 *.rice + 304 *.json
  packed/attn_rows.json  304
  packed/mlp_rows.json   192
  tensors/               402 *.hq30uq4 + 353 *.f32v2 + 353 *.f32bin
  catalog.hq38m20        ABSENT
  segments/              ABSENT
```

610 files under `packed/` (304+304+2). 1108 files under `tensors/`.
1721 files in the directory, `lstat` sum `15_473_851_850` B.

### 6.2 Expected tensors

Geometry = packer (`tools/qwen38_sub15_pack.py:59-77,374-399`).
GQA iff `layer % 4 == 3`.

| class | expected | present | missing | extra |
|---|---:|---:|---:|---:|
| attention rice names | 304 | 304 | 0 | 0 |
| attention `*.rice` files | 304 | 304 | 0 | 0 |
| MLP rows | 192 | 192 | 0 | 0 |
| MLP vs 2p0 catalog names | 192 | 192 | 0 | 0 |
| manifest q4 (fused vehicle) | 402 | 402 | 0 | 0 |
| manifest f32 | 353 | 353 | 0 | 0 |
| unfused rice names in manifest | 0 | 0 | — | — |

### 6.3 Magics and declared sizes

| bag | check | result |
|---|---|---|
| 304 rice | magic `HGRAVR02`, schema `hawking.gravity.binary_outlier_residual.v2`, `index_mode=rice`, `value_bits=1`, `value_scale=rms`, `g=128` | 304/304 |
| 304 rice | `lstat == json packed_bytes == attn_rows.packed_bytes` | 304/304, sum **1_165_098_376** |
| 304 rice | sha256(name) stem matches filename | 304/304 |
| 304 rice | `outlier_count == 0` | **0** (sum 144_756_064) |
| 64 gate slices | magic `HGRAVB01`, shape `[17408,5120]`, g=128, sha256(slice)==catalog | 64/64 |
| 64 up slices | magic `HGRAVR02`, g=128, sha256, `outlier_count==0` | 64/64, 0 zeros, sum 114_085_120 |
| 64 down slices | magic `HGRAVS01`, r160_b3 as above, sha256 | 64/64 |
| 66 2p0 catalog segments | `lstat == catalog bytes` | 66/66 |
| 2p0 `catalog.hq38m20` | magic `HQ38M20\0`, v1, 851 tensors, 66 segments, `consumed == file_bytes == 158970` | match |
| 402 vehicle Q4 | magic `HQ30UQ4\0`, `lstat == manifest.bytes` | 402/402 |
| 353 f32v2 | `u64 numel == elements`, size `8+4*E` | 353/353 |
| L0 `input_layernorm` | first f32 `0.046875` (already δ) | match |

### 6.4 Final-shard truncation

| object | evidence | truncated? |
|---|---|---|
| last lex rice `ff7ef32f….rice` | lstat 5_063_447 == sidecar `packed_bytes`; name `layers.53.linear_attn.in_proj_z.weight` | no |
| last name-order rice L63 `o_proj` | lstat 5_064_135 == `packed_bytes` | no |
| last MLP slice L50 `down_proj` | offset 88_446_949 + nbytes 1_466_361 = 89_913_310 == `L50.hq38seg` lstat | no (flush with segment end) |
| last 2p0 segment id 65 `99_terminal.hq38seg` | lstat 675_433_663 == catalog | no |
| 2p0 extra 66 `*.hq38seg.records.json` | sidecars, not payloads | n/a |

No truncated final shard.

### 6.5 Complete BPW, recomputed from real bytes

Definition: `8 * sum(payload bytes of the packed ledger) / 26_895_998_464`.
Inputs are `lstat` of 304 rice files, `sha256`-verified slice lengths of
192 MLP payloads in mixed-2p0 segments, `lstat` of embed + lm_head +
353 f32v2. Not `PACK_REPORT.json` fields.

| class | bytes `MEASURED` | elements |
|---|---:|---:|
| mlp.gate HGRAVB01 | 802_177_344 | 5_704_253_440 |
| mlp.up HGRAVR02 | 918_036_000 | 5_704_253_440 |
| mlp.down HGRAVS01 | 93_847_197 | 5_704_253_440 |
| attention rice | 1_165_098_376 | 7_237_795_840 |
| embed HQ30UQ4 | 675_430_440 | 1_271_398_400 |
| lm_head HQ30UQ4 | 675_430_440 | 1_271_398_400 |
| small f32v2 | 10_584_840 | 2_645_504 |
| **sum** | **4_340_604_637** | **26_895_998_464** |

```
8 * 4340604637 / 26895998464 = 1.2910781930062503
```

Equals `PACK_REPORT.json:23-25`. Agreement is from the same arithmetic
on independently measured inputs, not from reading the field.

Other quotients, labeled so they are not that number:

| name | value | note |
|---|---:|---|
| packed ledger BPW | **1.2910781930062503** | above |
| vehicle manifest BPW | 4.252735126866492 | `manifest.json`; 402 q4 + 353 f32 |
| whole-directory BPW | 4.60257368640516 | `8 * 15473851850 / 26895998464`; vehicle + rice + f32bin + json |
| rice CSR expand at load | 1_045_215_040 B | `(144756064+114085120)*4 + row_ptr`; not in packed BPW |

### 6.6 The pack is not a native-load root

`materialize_mlp` decodes mixed-2p0 and writes HQ30UQ4 only
(`tools/qwen38_sub15_pack.py:402-411`). `mixed-sub15-v1/packed/` holds
rice + two json ledgers. There is no `HGRAVB01` / `HGRAVS01` file in
this directory. C1 "over existing blobs" is over **this directory plus
a sibling pack that must remain at a fixed path with intact segment
bytes**. That is a composite artifact. `Qwen38HybridWeights::load(mixed-sub15-v1)`
cannot become native until those 1.814 GB are either hardlinked under
`segments/` or the catalog points at `mixed-2p0-v1/segments` — and the
parser will not do the latter (`:124` is `root.join("segments")`).

---

## 7. What a native generate of this artifact would NOT prove

Even if C1+C2+C3 (or fuse-off) succeed and the greedy is coherent:

- It is **one composition family** (binary gate, rice-q1-rms-2% up and
  attention, HGRAVS r160_b3 down, HQ30UQ4 embed/lm_head, f32 small) at
  **one** complete BPW 1.2910781930062503. Coherence there does not
  locate the floor for a differently allocated pack at the same
  density (protect out_proj / channel 3994, raise down from 0.1316,
  change rice residual, different rank).
- It does not reopen the dead families (generator+residual, VQ as
  tested, entropy-under-this-quantizer). Those kills stand.
- Attention rice cosine vs BF16 is `MEASURED` 0.8344098743646869 …
  0.8474425216212348 (pack metadata, 304/304). That is already a
  reconstruction, not a generate. A coherent answer would not make
  those cosines 0.99.
- All 64 downs were fit with `n_fit_rows=256` against in-dim 17408
  (rows-per-dim 0.0147), worse than the Q80 NS-014 catastrophe 0.0449.
  A coherent or incoherent generate of this down is not a measurement
  of a well-determined HGRAVS operator.
- It does not make the expand-to-Q4 `QWEN38_SUB15_INCOHERENT.json`
  receipt a native-codec verdict, and it does not make mixed-2p0
  native INCOHERENT a K=5120 HGRAVB01/HGRAVR02 verdict (that generate
  used `*_tg256`).
- It is not a speed win. Packer 79.44 TPS is PROJECTED
  (`PACK_REPORT.json:70-76`) and retired. No TOKEN_NS is claimed here.
- A single prompt is not qualification.

The Qwen3.8 coherence floor remains bracketed only as > 2.0856
(confounded mixed-2p0) and ≤ 4.2527 (G0). This pack does not place it.

---

## 8. Failure modes, ranked by probability

Each item is a concrete way the "250–500 lines, no new shader,
IMPLEMENT_READY" close produces a wrong campaign decision. Cheapest
check that confirms or eliminates it.

| p | mode | cheapest check |
|---|---|---|
| 1 | **C1+C2 generate on default fuse silently drops 60–67% of K** on all 432 HGRAVB01/HGRAVR02 GEMVs. Read as a codec verdict. | Confirm vehicle: `HAWKING_QWEN38_RECON_FUSE` value + which name `dispatch_binary`/`dispatch_residual` bind. If still `*_tg256` and K>2048, the generate is not a codec result. Eliminated only by C3 retarget, the 2048-col loop, or fuse-off. |
| 2 | **C1 catalogs vehicle Q4 GEMV names** (fused `in_proj_qkvz` / `in_proj_ba` or reconstructed MLP). `encode_named_matvec` takes q4 first (`:1213-1215`). Expand-to-Q4 confound returns. | After emit: parse `catalog.hq38m20`; assert 0 names ending `in_proj_qkvz.weight` / `in_proj_ba.weight`; assert every `mlp.*.weight` is codec ∈ {0,1,2}; assert 304 attention GEMVs are codec 1 with magic `HGRAVR02`. |
| 3 | **C2 (or a 2p0-clone C1) runs `mlx_residual_norm_to_delta` on already-δ f32v2.** | Unit test: `read_qwen38_f32_payload` of L0 `input_layernorm` equals 0.046875 at `[0]`; the C2 arm must not call `mlx_residual_norm_to_delta_named`. One-line source grep after the implement lane. |
| 4 | **C1 writes through the 355 G0-shared inodes** (embed, lm_head, 353 f32v2). Live organism serves those bytes. | Before/after: `stat -f '%i'` on `uniform-q4-v1/tensors/<embed>` and the dest. Any inode-preserving rewrite is a G0 hit. Allow only new dirents (`os.link` / new files). |
| 5 | **C1 clones mixed-2p0 catalog and leaves HGRAVU01 attention.** Split names match; you load 4.25 BPW attention, not rice. Ledger 1.291 is a lie. | Catalog attention `codec==1` and payload magic `HGRAVR02` for all 304; 0 codec-3 `*proj*.weight` except embed/lm_head. |
| 6 | **C1 emits GEMV-only / omits some of 353 f32.** `load_mixed` returns Ok (MLP lock passes). First `encode_rmsnorm` / conv1d dies. | CPU parse of catalog: 851 names (or a documented 755 fused set, which this rice pack cannot be); 353 small names == uniform manifest f32 names; 2 HQ30UQ4. |
| 7 | **Sibling mixed-2p0-v1 moved or a segment rewritten.** MLP slices are not in this pack. | `test -s mixed-2p0-v1/catalog.hq38m20`; re-hash one L0 gate slice against catalog sha256. |
| 8 | **Rice host expand ~1.045 GB CSR + CPU walk of 258_841_184 outliers** at load. Not a kernel hole; can be misread as "load failed". | After a CPU-only `mixed_gpu_layout` + `expand_rice_indices` on one `in_proj_qkv` and one `in_proj_a`. Full-pack expand is load, not this lane. |
| 9 | Shape outside sibling tiles | **Eliminated** this process. §1. All K%256==0, all rows%8==0, 48×5120 fits 8-row TGs. |
| 10 | Kernel cannot accumulate full K | **Eliminated** this process. §2. Siblings + serial + CSR apply walk all cols / all nnz. |
| 11 | MLP lock must be weakened | **Eliminated** this process. §4. Recipe is already Binary/Residual/Hgravs r160_b3 × 64. |
| 12 | Catalog emit changes G0 `load()` control flow | **Eliminated** if emit stays under mixed-sub15-v1. G0 root has no `catalog.hq38m20`. Residual risk is item 4 (inodes), not the `if catalog.is_file()` branch. |

Items 1–5 are the ways a 250-line close ships a false native verdict.
Items 9–12 are the ways the native-gap analysis was right.

---

## 9. Evidence

### 9.1 Shipping bind drops K (`SOURCE`)

`crates/hawking-core/shaders/q80_mixed_decode.metal:722-728`:

```
    const uint col = lid * 8u;
    float partial = 0.0f;
    if (col + 8u <= cols) {
        partial = q80_binary_byte_dot(
            signs, scales, input, row_base, scale_base, col, group_size);
    }
```

`crates/hawking-core/src/model/qwen38_hybrid_decode.rs:1323-1360` binds
those names when `qwen38_recon_fuse_enabled()` (default ON, `:43-45`).

Sibling that walks K, same file `:641-648`:

```
    for (uint base = 0u; base < cols; base += 256u) {
        const uint col = base + simd_lane * 8u;
        if (col + 8u > cols) {
            continue;
        }
        partial += q80_binary_byte_dot(...);
    }
```

### 9.2 q4-first confound (`SOURCE`)

`qwen38_hybrid_decode.rs:1213-1218`:

```
            if self.weights.q4.contains_key(name) {
                return self.encode_q4_matvec(tcb, name, input, output);
            }
            if self.weights.mixed.contains_key(name) {
                return self.encode_mixed_matvec(tcb, name, input, output);
            }
```

### 9.3 MLP packed bytes never land in this pack (`SOURCE`)

`tools/qwen38_sub15_pack.py:402-411`: decode mixed-2p0, `overwrite_q4` only.

### 9.4 Catalog parser will not look in `packed/` (`SOURCE`)

`qwen38_hybrid_decode.rs:124`: `root.join("segments").join(filename)`.

### 9.5 Codec 4 refused today (`SOURCE`)

`qwen38_hybrid_decode.rs:659-663`; `q80_mixed_decode.rs:1331`.
Contract tests (`:3795-3807`) check magic/record size only. No codec-4 arm.

### 9.6 Confounded INCOHERENT receipt (`RECEIPT`)

`git show HEAD:receipts/ascent-2026-08-16/QWEN38_SUB15_INCOHERENT.json`:
vehicle is reconstructed HQ30UQ4; `fallbacks: 0`; two-token cycle;
`WHAT_THIS_ESTABLISHES` claims "~1.29 BPW … BELOW Qwen3.8's coherence
floor". That establishment is the confound. Oracle control on 4.2527
produced `<think>`. `speed_caveat_carried` retires 79.44 TPS.

### 9.7 This-process census (command output)

Attention rice (304/304, bytes, magics, shapes, outliers, cosine):

```
attn_rows.json count 304
unique names 304
missing expected [] n 0
extra [] n 0
magic HGRAVR02 304 / 304
size mismatch n 0
header errors n 0
outlier_zero []
schema {'hawking.gravity.binary_outlier_residual.v2': 304}
index_mode {'rice': 304} value_bits {1: 304} value_scale {'rms': 304} group_size {128: 304}
cols,cols%128,cols%256 {(5120, 0, 0): 240, (6144, 0, 0): 64}
rows%8 {0: 304} K%256 {0: 304}
rice lstat bytes 1165098376 json packed_bytes sum 1165098376
outlier_sum 144756064
cosine 0.8344098743646869 0.8474425216212348
```

MLP slices vs mixed-2p0 (192/192, sha256, r160_b3, class bytes):

```
2p0 catalog magic b'HQ38M20\x00' version 1 n_tensors 851 n_segments 66
catalog file bytes 158970 consumed 158970 tail 0
segment size mismatches 0
mlp_rows 192 unique 192 missing 0 extra 0
codec/nbytes/elements mismatches 0
slices checked 192 / 192
range overflow []
sha mismatches 0
header problems 0
magics {b'HGRAVB01': 64, b'HGRAVR02': 64, b'HGRAVS01': 64}
down_geom (160, 3, 64, (5120, 160), (160, 17408), 'activation_weighted_svd_low_rank_q') 64
up outlier zero [] sum 114085120
class bytes {'gate': 802177344, 'up': 918036000, 'down': 93847197}
max mlp slice end == L50.hq38seg lstat 89913310
```

Manifest / hardlinks / BPW:

```
manifest schema hawking.ascent.qwen38_language_uniform_q4.v1
n tensors 755 kinds {'q4': 402, 'f32': 353}
missing files [] size mismatches 0
q4 magic ok 402 / 402
f32v2 ok 353 / 353
hardlink same inode as G0 355
hardlink different inode from G0 400
diff inode roles {'fused_attn': 96, 'attn': 112, 'mlp': 192}
embed inode 314847693 shared with G0
lm_head inode 315002072 shared with G0
L0 input_layernorm first=0.046875 numel=5120 size=20488
PACKED LEDGER BYTES (measured) 4340604637
complete BPW measured 1.2910781930062503
sub15 directory files 1721 bytes 15473851850
catalog.hq38m20 False
segments dir False
```

2p0 attention is unfused HGRAVU01 (so split path has run; rice Residual on
those names has not):

```
in_proj_qkvz ABSENT
in_proj_qkv codec 3 shape [10240, 5120]
in_proj_a  codec 3 shape [48, 5120]
embed      codec 3 nbytes 675430686   # not the sub15 HQ30UQ4 675430440
down n_fit_rows Counter({256: 64})
```

---

## 10. What this lane did not measure

- Any Metal load or generate (forbidden; resident holds the device).
- Bit-identity of `*_simd_bytes` vs `gk_binary_group_serial_row` on
  17408×5120 or 48×5120. Cheapest CPU check named in native-gap §9;
  still not run.
- Token ns of a native sub15 graph. Not claimed.

```
STATUS
FALSIFIED

CLAIMS
1. Every packed GEMV shape is covered by an already-compiled K-complete sibling. No new shader family is required. 48×5120 is 6 TGs of the 8-row tile. Evidence: §1 census; q80_mixed_decode.metal:641-648,796-841,869; metal/mod.rs:439.
2. The K-complete item is a bind. Shipping *_tg256 covers 2048 cols and silently drops K∈{5120,6144}. Serial, simd_bytes, csr_bytes, and simd3 walk full K. Evidence: q80_mixed_decode.metal:722-728 vs :641-648,:821-838; gk_family.metal:128-159; dispatch :1323-1373.
3. mixed-sub15-v1 is a complete packed *ledger* at measured BPW 1.2910781930062503 and is not a native-load *root*. 1.814 GB of MLP payloads live only in mixed-2p0-v1/segments. catalog.hq38m20 and segments/ are absent. Evidence: §6; tools/qwen38_sub15_pack.py:402-411; qwen38_hybrid_decode.rs:124,508-513.
4. load_mixed accepts codecs 0/1/2 and codec-3 HQ30UQ4/HGRAVU01, refuses 4, locks MLP to Binary/Residual/Hgravs. This recipe satisfies the lock on all 64 layers without weakening it. Evidence: :601-664,958-1003,1116-1134; 192/192 slice headers.
5. Codec-4 acceptance is not a local arm: non-empty mixed flips all five encode_* entries; load_mixed has no f32 completeness check; mlx-delta on already-δ f32v2 (L0 first=0.046875) is lethal; encode_named_matvec prefers q4. Evidence: :2523,2555,2606,2708,2834,51-63,1213-1215; qwen38_pack.rs:751-767.
6. Catalog emit into mixed-sub15-v1 does not change G0 load() control flow. It does alias 355 G0 inodes (embed, lm_head, 353 f32v2). Evidence: :508-513; MEASURED inodes 314847693 / 315002072 / 355 same.
7. A native generate of this pack, even if coherent, does not locate the Qwen3.8 floor. One family, one BPW, down fit on 256 rows, attention rice cosine 0.834–0.847. Evidence: §7; 2p0 catalog n_fit_rows=256 × 64; rice cosine MEASURED.

EVIDENCE
workspace/superwave/g1/g1-sub15-native-gap.md:18-35,256-342,470-479
tools/qwen38_sub15_pack.py:13-15,82-86,365-371,402-411
crates/hawking-core/src/model/qwen38_hybrid_decode.rs:43-45,51-63,124,508-673,938-1003,1116-1134,1206-1221,1323-1374,2523,2555,2606,2708,2834,2858-2908
crates/hawking-core/src/model/qwen_complete_binary/q80_mixed_decode.rs:462-506,1174-1332
crates/hawking-core/shaders/q80_mixed_decode.metal:295-296,641-648,722-728,768-790,821-838,869,991-1016
crates/hawking-core/shaders/gk_family.metal:116-160
crates/hawking-core/src/metal/mod.rs:439,1121,1132
crates/hawking-core/src/decode_family.rs:18-20
crates/hawking-core/src/model/qwen38_pack.rs:33-34,751-767
PACK_REPORT.json:23-25,77-81
git show HEAD:receipts/ascent-2026-08-16/QWEN38_SUB15_INCOHERENT.json
This process: §9.7 census (304 rice, 192/192 sha256 MLP slices, BPW 1.2910781930062503, 355 G0 inodes, no catalog)

CHANGES
workspace/superwave/g1/g1-adversarial-sub15.md (this file only)

TESTS
test -s workspace/superwave/g1/g1-adversarial-sub15.md
wc -l workspace/superwave/g1/g1-adversarial-sub15.md
git status --porcelain

RISKS
A later implement lane that follows native-gap C1–C3 without the checks in §8 items 1–5 will ship a generate that is K-truncated, expand-to-Q4, or norm-destroyed, and will be read as a floor. C1 that writes embed/lm_head/f32v2 corrupts the live G0 organism. Do not weaken assert_mixed_mlp_native.

UNRESOLVED
Native coherence of this 1.291 recipe. simd_bytes vs serial bit-identity on 48×5120 and 17408×5120 (CPU, not run). Whether the implement lane emits a closed pack or a sibling alias.

NEXT
Do not bet G1's critical path on "just wire mixed-sub15-v1". If a write lane emits the catalog, require the §8 item 1–5 checks before any generate is treated as a codec verdict. Serialized GPU lane owns that generate. Floor location needs more than one allocation at this density.
```
