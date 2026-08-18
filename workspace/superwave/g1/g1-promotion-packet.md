# G1 promotion packet — mixed-q3mlp-v1

Lane: `115-promotion-packet`. Analysis only. No GPU, no generate, no
repack, no shader/hybrid-decode edit, no resident touch, no tracked-file
edit except this file.

Every number is **MEASURED** (this process or a named receipt),
**RECEIPT** (quoted from a prior lane, not re-run), **SOURCE**
(file:line), **DERIVED** (exact arithmetic on MEASURED integers),
**PROJECTED** (not a token wall), or **PLACEHOLDER** (later lane fills
from the paired measurement on the final kernel). A packet that invents
one number is worse than no packet.

`~31 TPS` is **PROJECTED** wherever it appears. It is
`11.472705646e9 B / 639.25e9 B/s` addressing plus APPLIED non-GEMV.
It is not a complete-token wall.

STATUS of this packet: **IMPLEMENT_READY**. The child is reproducible
from the reconstruction recipe. Promotion is not ACCEPT today: the
seated kernel is 3.78× slower than live G0. After lane 110's
geo_tpr64-class HGRAVU01 kernel is measured, fill the PLACEHOLDER
fields and re-run the checklist. Do not reseat from
`make_qwen38_genesis()` until identity binding (§1) lands.

---

## 0. How to use this packet

1. Do not mutate
   `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1`
   or any inode hardlinked into it. Live G0 manifest sha256
   `d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df`
   **MEASURED this process** (unchanged).
2. Do not mutate `mixed-q3mlp-v1`. 66 of 258 segment files are
   hardlinked (nlink=3) with `mixed-2p0-v1` / `mixed-q4down-v1`
   (**MEASURED** this process).
3. Do not edit `q80_mixed_decode.metal` or `qwen38_hybrid_decode.rs`.
   Lane 110 owns both.
4. Do not treat this document's current-kernel TOKEN_NS as the
   promotion wall. Remeasure on the binary that contains the new
   kernel.
5. A `hawking.lineage/` string is not an identity preimage. See §1.

---

## 1. Identity binding — broken today; this is the deliverable

Today's lineage is naming, not provenance. Three independent 1.0s were
assigned, not measured.

### 1.1 What is seated (MEASURED this process)

`labeled_sha(label) = sha256("hawking.lineage/"+label)`
SOURCE `lab/lineage/canon.py:29-30`.

```
labeled_sha("runtime/ascension_qwen38_hybrid_greedy")
  = ecfc1cac8742d51dac35bca3c702520a7409089914b9dd637d7927baae0cfe72
labeled_sha("genome/Qwen38HybridDecodeSession+qwen_uniform_q4_group64")
  = 688d8b87bddc6baa7bd083229f1b1c7c96ea01adb893c42a98ad534c3341cd7e
labeled_sha("bench/complete-token/qwen38/greedy/3prompt/gpu-cb-timestamps")
  = aa6990ca04bcb26dd53a0ccc13b27b2275a2b2e3b4965ba0c6a8274a3e2e38dd
labeled_sha("artifact/qwen38-27b/uniform-q4-v1")
  = 56dd65d465f31741f8d40a86d84de779a939fdd9b9b90ecd3d1cb4f82aa4287a
sha256(uniform-q4-v1/manifest.json)
  = d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
```

`make_qwen38_genesis` SOURCE `lab/lineage/identity.py:276-310`:

| field | bound to | value |
|---|---|---|
| `artifact_sha` | manifest bytes (default `GENESIS_ARTIFACT_MANIFEST_SHA`) | `d650a757…` |
| `runtime_sha` | `labeled_sha("runtime/ascension_qwen38_hybrid_greedy")` | `ecfc1cac…` |
| `kernel_genome_sha` | `labeled_sha("genome/Qwen38HybridDecodeSession+qwen_uniform_q4_group64")` | `688d8b87…` |
| `capability` | `DEFAULT_CAPABILITY_CONTRACT` `{1.0,1.0,1.0}` | assigned |
| `benchmark_fingerprint` | `labeled_sha("bench/…")` | `aa6990ca…` |
| `complete_token_ns` | `GENESIS_COMPLETE_TOKEN_NS` | `35_227_918` |

`file_sha256` already hashes file bytes (`identity.py:51-63`) and is
not used for runtime or kernel.

Git `HEAD:receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json`:

| slot | artifact_sha | runtime_sha | kernel_genome_sha | token_ns |
|---|---|---|---|---:|
| CURRENT | `d650a757…` (manifest bytes) | `ae0bc8de…` (resident file) | `51abdf7b…` (one .metal file) | 37_879_375 |
| LKG | `d650a757…` | `ecfc1cac…` **labeled** | `688d8b87…` **labeled** | 35_227_918 |

CURRENT and LKG of the same `instance_id` disagree on what the sha
fields mean. A rollback restores labeled hashes into a resident that
compares them to `sha256(manifest.json)`.

Live remasure TOKEN_NS **39,326,090** / TPS **25.4284**
(RECEIPT `g1-baseline-remeasure.md:12-13`) is a third number for the
same generation. Lineage stores none of the three as a kinded hash.

Capability on both slots is `{coherence:1.0, complete_token_discipline:1.0,
engineering:1.0}`. Live G0 measured 6/6 oracle-32 + `17*19=323`
(RECEIPT `g1-baseline-remeasure.md:14`). Lineage does not store that
measurement.

`min_q4_cosine: 1.0` on the live G0 manifest is
`fold(1.0, min)` over 402 `None`s. SOURCE
`crates/hawking-core/src/model/qwen38_pack.rs:680-684`; reuse writes
`cosine: None` at `:312` and `:403`. RECEIPT
`g1-capability-gate.md:166-185`. That 1.0 is a fold identity, not a
cosine.

### 1.2 Promotion hashes a label, not bytes

SOURCE `lab/lineage/promotion.py:181-200` `_computed_artifact_sha`:
`sha256(receipt.preimage.encode("utf-8"))`.

SOURCE `lab/lineage/testing.py:35-42` `artifact_preimage_for` returns
`"hawking.lineage/artifact/child-g1"`. Passing
`artifact_identity_exact` proves the label is self-consistent. Swapping
catalog bytes at a path does not move the hash.

`CLAUSE_RUNTIME_GENOME` (`promotion.py:496-514`) compares two asserted
hex strings. No binary or shader bytes are read.

`genesis-resident` hashes only `manifest.json`
(SOURCE `tools/agentos/genesis_body/src/main.rs:398-406`). On mismatch
it logs `"lineage identity is stale; loading measured artifact"` and
loads anyway (`:843-852`). A mixed HQ38M20 root has no `manifest.json`;
`artifact_manifest_sha` cannot bind this candidate.

`tools/genesis_seat.py` `seat()` rebuilds from `make_qwen38_genesis()`.
Re-seating wipes the live bind and restores labeled runtime/kernel
hashes. **KILLS re-seat as a migration step** until §1.3 lands.
RECEIPT `g1-tabula-genome.md:128`.

### 1.3 Required bind (do not implement in this lane)

Hash-kind discriminator required on every sha field that has ever been
labeled (`hawking.lineage.labeled_path.v1` |
`hawking.lineage.content_sha256.v1` |
`hawking.lineage.content_merkle.v1`). A field without `*_kind` is
UNBOUND. RECEIPT `g1-tabula-genome.md:203-211`.

**Artifact (uniform-Q4 / G0):**

```
manifest_sha = sha256(manifest.json bytes)
file_sha[i]  = sha256(bytes of tensors/<artifact> named by tensors[i])
merkle       = sha256( concat_i  name_i || 0x00 || hex(file_sha[i]) || 0x0a )
artifact_content_sha = sha256( bytes.fromhex(manifest_sha) || bytes.fromhex(merkle) )
```

G0 MEASURED (RECEIPT `g1-capability-gate.md:331-337`):
`merkle = c33d59d8811669760eaf6c27a39338f855fce97a48563b2bcab00c2e310c9641`,
`artifact_content_sha = f590664c259cbea8fe90889e06e2f78f09c57f03f34f97b26635e524e5e06b5e`.
Not recomputed this lane.

**Artifact (HQ38M20 / this candidate):**

```
catalog_sha  = sha256(catalog.hq38m20 bytes)
segment_sha[j] = sha256(segments/<file> bytes)
merkle       = sha256( bytes.fromhex(catalog_sha) || concat_j (u16le segment_id || bytes.fromhex(segment_sha[j])) )
artifact_content_sha = sha256( bytes.fromhex(catalog_sha) || bytes.fromhex(merkle) )
```

**MEASURED this process** on `mixed-q3mlp-v1` (258 segments,
13_963_692_970 file bytes streamed, 1 MiB chunks):

```
catalog_sha            = 72ed83a21213605026428daa128231c0a220c8fb997f1ec46ffd760de40fd8fb
merkle_catalog_sid_sha = 511dae054605d1b9e76867b5057502c3249515466541c2073ce5d977fc0ae372
artifact_content_sha   = 89399408f7d296308b21a741b586348d3c514d7525df2f555c35fb5895d94261
```

Also recorded, not the authority: name-sorted
`sha256(concat fname || 0x00 || hex(segsha) || 0x0a)` =
`f660376ed2a6697ff90ecc8913daa0e7c44d5976779f3772e5aef30c4234a6f1`.

A string starting `hawking.lineage/` is FAIL.

**Runtime:**

```
runtime_sha = sha256(bytes of the executable that ran generate)
runtime_sha_kind = hawking.lineage.content_sha256.v1
```

Not `labeled_sha("runtime/…")`. Not the crate name. The oneshot
measurement vehicle and the resident are different files; record both.

**Kernel:**

```
kernel_genome_sha = sha256( concat_i  sha256(shader_i bytes) || 0x00 || dispatch_name_i || 0x0a )
                  over every .metal file that the token actually dispatched,
                  sorted by dispatch-name
kernel_genome_sha_kind = hawking.lineage.content_merkle.v1
```

Hashing only `qwen_uniform_q4.metal` is incomplete for G0
(RECEIPT `g1-artifact-inventory.md:276`) and is the wrong file for
this candidate (`q4=0`).

**Capability:**

No axis float may appear on a `GenesisInstance` unless copied from
`hawking.genesis.qwen38_capability_seal.v1`. Missing seal is PENDING,
never 1.0. Schema RECEIPT `g1-capability-gate.md:49-114`.
`n_scored == 0` ⇒ rate is `null` / `NOT_MEASURABLE`, never 1.0.

### 1.4 Exact files and functions that must change

This lane does not edit them.

| path | function / site | change |
|---|---|---|
| `lab/lineage/identity.py` | `DEFAULT_CAPABILITY_CONTRACT` `:36-40` | delete the 1.0/1.0/1.0 assignment. Construction without a seal is `IdentityError`. |
| `lab/lineage/identity.py` | `make_qwen38_genesis` `:276-310` | `artifact_sha=artifact_content_sha`; `runtime_sha=file_sha256(executable)`; `kernel_genome_sha` from dispatched shader merkle; `capability` from seal; set `*_kind`. |
| `lab/lineage/identity.py` | `file_sha256` `:51-63` | keep; this is the runtime preimage helper. |
| `lab/lineage/canon.py` | `labeled_sha` `:29-30` | keep for tests; banned in production identity. Add `content_sha256`, `catalog_merkle`. |
| `lab/lineage/promotion.py` | `_computed_artifact_sha` `:193-200` | reject `hawking.lineage/` strings; require manifest/catalog bytes + merkle receipt. |
| `lab/lineage/promotion.py` | `CLAUSE_ARTIFACT_IDENTITY` `:429-473` | `child.artifact_sha == artifact_content_sha`. |
| `lab/lineage/promotion.py` | `CLAUSE_RUNTIME_GENOME` `:496-514` | require `genome.runtime_preimage` / `kernel_preimage` (file bytes or list of file sha256s) and recompute. |
| `lab/lineage/promotion.py` | `CLAUSE_CAPABILITY` `:263-290` | compare derived floats from `evidence["capability_seal"]`, not `child.capability` asserted map. |
| `lab/lineage/promotion.py` | `ALL_CLAUSES` `:57-73` | add `dequant_cosine_none_is_fail`, `generation_class_coherent`, `native_vehicle_only`, `tabula_profile_bound`, `gravity_profile_bound`. Split `greedy_token_ids_agree`: exact match only when `artifact_content_sha` equals parent. |
| `lab/lineage/testing.py` | `artifact_preimage_for` `:35-42`, `make_child` `:61-68`, `passing_evidence` | content preimage; no `labeled_sha` child identity. |
| `lab/tests/test_genesis_promotion_gate.py` | happy path | must carry a seal; labeled preimage no longer ACCEPT. |
| `lab/tests/test_genesis_promotion_gate_adversarial.py` | add mixed-2p0 + labeled-sha cases | RECEIPT `g1-capability-gate.md:544-586`. |
| `lab/lineage/capability_seal.py` | **new** | `CapabilitySeal`, `fold_min_or_none`, `evaluate_capability_seal`. |
| `lab/lineage/content_hash.py` | **new** | `merkle_listed` as specified in `g1-tabula-genome.md:182-193`. |
| `crates/hawking-core/src/model/qwen38_pack.rs` | min fold `:680-684` | None → error / `min_q4_cosine: null` + status `UNMEASURED`. Never seed 1.0. |
| `crates/hawking-core/src/model/qwen38_pack.rs` | `try_reuse_q4` `:312`, `pack_q4_named` `:403` | do not write `cosine: None` into a fold that can become 1.0. |
| `tools/agentos/genesis_body/src/main.rs` | `artifact_manifest_sha` `:398-406` | HQ38M20: hash `catalog.hq38m20` + require merkle; a labeled CURRENT sha is a hard error, not a stale-and-load log (`:848-852`). |
| `tools/genesis_seat.py` | `seat` | refuse to seat without a PASS seal; write content hashes. Do not call `make_qwen38_genesis()` as it exists today. |
| `tools/coherence_gate.py` | whole file | keep as id-identity subroutine; do not call it the capability gate. |

Do not change: receipts, live G0 catalog bytes, shaders (lane 110),
Q80/DSV4F vehicles, AgentOS/HIDE/UI.

Phase-0 non-breaking attachment (sidecars + string pointers on
`identity.*_kind` / `identity.*_profile_*`) is specified in
`g1-tabula-genome.md:455-485`. Prefer that if `from_mapping` cannot
move in the same PR.

---

## 2. Model artifact identity

```
root     /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-q3mlp-v1
kind     hq38m20
schema   hawking.ascent.qwen38_mlp_not_r160.v1   (PACK_REPORT)
catalog  catalog.hq38m20   HQ38M20\0 v1
```

| field | value | tag |
|---|---|---|
| catalog_sha256 | `72ed83a21213605026428daa128231c0a220c8fb997f1ec46ffd760de40fd8fb` | MEASURED this process |
| catalog_bytes | 180_124 | MEASURED |
| n_tensors | 851 | MEASURED |
| n_segments | 258 | MEASURED |
| unique segment paths | 258 / 258 under `segments/` | MEASURED |
| absolute filenames | 0 | MEASURED |
| missing / past-EOF / outside-root | 0 / 0 / 0 | MEASURED |
| hardlinked segment files | 66 | MEASURED |
| dir file bytes | 13_963_692_970 | MEASURED |
| slack (file size − merged catalog intervals) | 1_814_060_541 | MEASURED |
| `standalone_root` | YES | MEASURED |
| `artifact_content_sha` | `89399408f7d296308b21a741b586348d3c514d7525df2f555c35fb5895d94261` | MEASURED §1.3 |
| Tabula | `PocketAiHub/Qwen3.8-27B-Abliterated-MLX` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | SOURCE `qwen38_geometry.rs` / `identity.py` |
| tokenizer_sha256 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` | MEASURED this process |
| G0 parent manifest | `d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df` | MEASURED this process |

`FORMAT.md` (on-disk): `HQ38M20 native mixed catalog. MLP down_proj is
HGRAVU01, not HGRAVS01 r160_b3. Packed bytes stay packed.`

Fused G0 names `in_proj_qkvz` / `in_proj_ba`: **0** in this catalog
(MEASURED this process; 48× split `in_proj_qkv` + 48× `in_proj_z` +
48× `in_proj_a` + 48× `in_proj_b`). Loader takes the split path
SOURCE `qwen38_hybrid_decode.rs:3264-3284`.

Do not use dir-bytes/params as BPW. Slack 1.81 GB is leftover 2p0
payloads inside hardlinked Lxx files, unreferenced by this catalog.
RECEIPT `g1-artifact-inventory.md:340`.

---

## 3. Representation genome

| axis | value | tag |
|---|---|---|
| container | HQ38M20 codec 3 everywhere | MEASURED |
| magic | `HGRAVU01` × 851 | MEASURED this process |
| bits | 3 × 192 (all MLP) ; 4 × 659 (rest) | MEASURED this process |
| group | 64 flat | SOURCE packer `GROUP_UNIFORM` + L0 header |
| scale rule | absmax / bound ; bound=`(1<<(bits-1))-1` → 3 or 7 | SOURCE `qwen38_mlp_not_r160_pack.py:encode_uniform_payload` @ `00939c186` |
| reconstruct_to_q4 | false | PACK_REPORT `recipe` / `claim_boundary` |
| fused in_proj | no | MEASURED |
| vision | skipped | inherited from mixed-2p0 / G0 pack |

Load census (RECEIPT `g1-mlp-family-generate.md:141-147`, identical on
16 / France 128 / arith 256 / wall):

```
opening mixed HQ38M20 + 851 catalog tensors (no reconstruct-to-Q4)
census: tensors=851 binary=0 residual=0 hgravs=0
        uniform=498 q4=0 f32=353
        refused=0 expanded_to_q4=0 expanded_to_float_gemv=0
```

498 = 192 bits=3 GEMV + 305 bits=4 GEMV + 1 embed.
353 = HGRAVU01 vectors `elems ≤ 65536` dequant to f32 at load
(`hgravu_is_vector`, SOURCE `qwen38_hybrid_decode.rs:139-157`).
Not a GEMV expand. `q4=0` ⇒ G0 `geo_tpr64` is never bound.

MLP admission: Uniform on gate/up/down is allowed after lane 91
(SOURCE `assert_mixed_mlp_native_kinds`
`qwen38_hybrid_decode.rs:227-275`). RECEIPT
`g1-mlp-rolelock-unlock.md:15-23`. `assert_mixed_mlp_native_catalog`
on this root returned Ok (RECEIPT same file `:118-120`).

---

## 4. Gravity recipe

Same Tabula as G0. Different Gravity.

| class | n | codec | bits | g | physical BPW | consume kernel (current) |
|---|---:|---|---:|---:|---:|---|
| mlp.gate_proj | 64 | HGRAVU01 | 3 | 64 | 3.2500251321231617 | `q80_hgravs01_factor_matvec_simd3` |
| mlp.up_proj | 64 | HGRAVU01 | 3 | 64 | 3.2500251321231617 | simd3 |
| mlp.down_proj | 64 | HGRAVU01 | 3 | 64 | 3.2500251321231617 | simd3 |
| ΔNet in_proj_qkv | 48 | HGRAVU01 | 4 | 64 | 4.250042572021484 | `q80_hgravs01_factor_matvec_simd` |
| ΔNet in_proj_z | 48 | HGRAVU01 | 4 | 64 | 4.250070444742838 | simd |
| ΔNet in_proj_a / b | 48+48 | HGRAVU01 | 4 | 64 | 4.25869140625 | simd |
| ΔNet out_proj | 48 | HGRAVU01 | 4 | 64 | 4.250070444742838 | simd |
| GQA q / k / v / o | 16×4 | HGRAVU01 | 4 | 64 | ~4.250 | simd |
| embed | 1 | HGRAVU01 | 4 | 64 | 4.250001799593266 | `qwen38_hgravu_embedding_lookup` |
| lm_head | 1 | HGRAVU01 | 4 | 64 | 4.250001799593266 | simd |
| small vectors | 353 | HGRAVU01 → f32 at load | 4 | 64 | (header-dominated) | f32 buffer |

All BPW in this table **MEASURED this process** (`8 * sum(catalog nbytes)
/ sum(shape products)`). Attention-bucket (528 rows, includes small
attn tensors) BPW **4.2501555848196455** matches RECEIPT
`g1-mlp-family-generate.md:109`.

Not mixed-2p0: gate=up=down at the same codec; no HGRAVS01; no rice.
RECEIPT `g1-repack-recipe.md:82-89`. This is G1-C already packed.

Post-kernel consume kernels (lane 110, **not yet hashed**):

| bits | proposed name | launch |
|---|---|---|
| 3 | `qwen_uniform_q3_group64_matvec_geo_tpr64_tg128` | TG 128, 64 TPR, col+=512 |
| 4 | `qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128` | same; `q = nibble - 7` (bound 7, not HQ30UQ4's nibble-8) |

PLACEHOLDER: post-kernel dispatch names actually observed in the
measurement binary's TCB dump.

---

## 5. Doctor map (do not reopen search)

Floor is located. Sub-1.5 is DEAD for this model under every tested
mechanism. Campaign brief + RECEIPTS.

| organ | first native codec at bar | min | tag | pointer |
|---|---|---|---|---|
| MLP 192/192, bar 0.95 | Uniform q3 g64 | campaign **0.9663** at L62 down | CAMPAIGN; this lane did not re-score 192 | brief; packed screen min **0.9652814877860332** L63 down PACK_REPORT; 6-layer hold min **0.9679** L31 up `g1-doctor-tensor-map.md:142` |
| MLP q2 down | does not hold 0.95 | bottoms **0.7529**, 1/64 clears | CAMPAIGN | brief |
| MLP binary down | dead | **0.3006** | CAMPAIGN | brief |
| MLP residual down | dead | **0.3642** | CAMPAIGN | brief |
| HGRAVS r160_b3 honest | dead | **0.7302**; 0.9196 was in-sample | CAMPAIGN | brief; mixed-2p0 down 0.1316 PACK_REPORT |
| Attention 0.99 | Q4 g128 MSE | attn BPW **4.125009196169866**, 29/29 | CAMPAIGN | brief |
| Q4 g=64 absmax vs MSE | MSE wins at same BPW/tile | absmax **0.98980** L47 o_proj FAILS 0.99; MSE **0.991123** | CAMPAIGN | brief; `g1-group-partition-geometry.md:334,477` |

Channel 3994 is activation-hot in 54/64 layers and a top-5 output row
on all 128 write tensors. L0 `lin_o` kurtosis 149.36.
RECEIPT `g1-doctor-tensor-map.md:53-69`. Not in this artifact
(no island sidecar). G1-R would add it; do not pack it until the
kernel wall is a win.

Act-colscale on this capture **KILLS**: L0 out 0.99224 → 0.91865.
RECEIPT `g1-doctor-tensor-map.md:73`.

---

## 6. Runtime identity

| role | path | sha256 | tag |
|---|---|---|---|
| live G0 resident | `…/workspace/ops/build/rust/release/genesis-resident` | `ae0bc8defd84a8a1a5cd1c4598224f370c0cfce83a0904e275cbb33df84d32c2` | MEASURED this process |
| candidate measurement oneshot (lane 92) | `…/91-mlp-rolelock-unlock-…/workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy` | `a9d41d09856ff7ffb7e32ff4fd4f7ad49cafc4301afd4f6814f51f8418898fab` | MEASURED this process; matches RECEIPT `g1-mlp-family-generate.md:56` |
| decode.rs at lane 91/92 HEAD | `crates/hawking-core/src/model/qwen38_hybrid_decode.rs` | `5638aab3e77c3b829c43d71c76999238cc60d41f0b8792fe4b6b65c144ae7ac6` | MEASURED this process; matches RECEIPT `:54` |
| post-kernel oneshot | PLACEHOLDER | PLACEHOLDER | hash the binary that ran the promotion wall |
| post-kernel resident | PLACEHOLDER | PLACEHOLDER | do not point the live resident at this root until ACCEPT |

Vehicle: `ascension_qwen38_hybrid_greedy`. `--max-seq-len 512`. No
`--raw-prompt`. Native reader (catalog present). Expand-to-float /
`mlx_lm_weights_overwritten_from_mixed_pack` is FAIL.

`HAWKING_QWEN38_RECON_FUSE` default ON. RECEIPT
`g1-mlp-family-generate.md:522` `recon_fuse=ON`.

---

## 7. Kernel genome

### 7.1 Current token (MEASURED / SOURCE / DERIVED)

`dispatch_factor` SOURCE `qwen38_hybrid_decode.rs:1662-1692`:

```
bits == 3 → q80_hgravs01_factor_matvec_simd3   simd8_grid  TG 256
bits == 4 → q80_hgravs01_factor_matvec_simd    simd8_grid  TG 256
bits == 8 → q80_uniform8_matvec_*
```

`simd8_grid(rows) = (rows.div_ceil(8)*256, 1, 1)`. 8 SG/TG, 1 row/SG,
**32 threads/row**.

| bits | column stride | extract | scale |
|---|---|---|---|
| 3 | 256 (`q80_mixed_decode.metal:869`) | 8-wide 3-byte unpack | 8 scale loads issued / tile (CSE unverified) |
| 4 | 32 (`:521-528`) | 1-wide `gk_uniform_extract_wide` | one `scales[e/64]` per FMA |

G0 production `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`
(TG 128, 4 SG, 2 rows/TG, 64 TPR, col+=512) is **not on this token**.
`q4=0`.

Per-token dispatch census DERIVED
(RECEIPT `g1-q3-uniform-kernel.md:189-215`):

| | G0 | q3mlp current |
|---|---:|---:|
| GEMV | 401 | **497** |
| fuse (`qwen38_fuse_split_{qkvz,ba}_f32`) | 0 | **96** |
| other | 563 | 563 |
| token dispatches | 964 | **1156** |
| production CBs | 1 | 1 |

Cause of the 3.78× slowdown (RECEIPT `g1-q3-uniform-kernel.md:30-36,388-400`):
32 TPR, narrow column pass, per-element scale fetch at bits 4, 1-wide
extract. **Not** bytes (15.7% fewer streamed GEMV bytes). **Not** a
serial reduction. **Not** dispatch count (ceremony MEASURED 1.6 ms).

NO REPACK. Existing HGRAVU01 body feeds geo_tpr64 directly. bits 3 =
24 bytes/group; bits 4 = Q4 nibbles with bound 7.

### 7.2 Shader content hashes (this worktree HEAD, MEASURED)

| file | sha256 | on this token? |
|---|---|---|
| `q80_mixed_decode.metal` | `08166781cfbb683eb54d45c58d1f510d9d3fe9d9e8af855b442d41a4e1a29db8` | YES (simd / simd3) |
| `qwen38_device_activations.metal` | `a95a17344fcca19e557eb48e0fc0b33714bb12624d8647d27a9119704e0ea408` | YES (ΔNet/GQA/fuse) |
| `qwen80_device_activations.metal` | `c3efa790167f2e9259b8eb1bc0a2184600b036f4a20213091ce00ede48d9cc0a` | YES (rms/ba/gated) |
| `gk_family.metal` | `c98d4fc1daf8f1645479d7dea5259afd33fc342ecc1fceb1771142d8bcab5f2b` | YES (extract + swiglu) |
| `mha.metal` | `a2717a7f09f74245d05e228578e5460f30c0bb102f6382dfc78a93ca51138ed5` | YES |
| `qwen_next.metal` | `178b59598cc70d54bfbb36fea3a9f4dbdcdeec627a6a03caf4769ffab2f91bec` | YES |
| `sample.metal` | `918e8250175d30bf11df75ae2f72346a4a93e132851b6e79c36f9aa132d148ff` | YES |
| `qwen_uniform_q4.metal` | `51abdf7be388d62ba080d13a1f97a18ab8b1114c0a6968e9d0f04d109d3efcd1` | **NO** on this candidate |

PLACEHOLDER: `kernel_genome_sha` of the post-110 sources actually
dispatched. Rehash after lane 110 lands. Do not copy the G0 one-file
hash `51abdf7b…` onto this child.

---

## 8. TOKEN_NS receipt

### 8.1 Live G0 (parent) — MEASURED 2026-08-17, DIRTY_ENGINEERING

RECEIPT `g1-baseline-remeasure.md:12-16,129-168`.

```
TOKEN_NS = 39,326,090     median of 6 paired decode-phase means
TPS      = 25.4284        DERIVED 1e9/39326090
BPW      = 4.252735126866492
capability = 6/6 oracle-32 + 17*19=323
fallbacks = 0
```

Historical seated values (do not mix):

| source | ns | TPS | tag |
|---|---:|---:|---|
| `GENESIS_COMPLETE_TOKEN_NS` / LKG | 35_227_918 | 28.3866 | SOURCE identity.py |
| CURRENT lineage / complete-wall | 37_879_375 | 26.3996 | RECEIPT GENESIS_LINEAGE_CURRENT |
| G024 ledger | 35_227_917 | 28.3866 | RECEIPT QWEN38_TOKEN_NS_LEDGER |
| live remasure (use this as parent) | **39_326_090** | **25.4284** | RECEIPT remasure |

### 8.2 Candidate on the CURRENT kernel — MEASURED, DIRTY_ENGINEERING

RECEIPT `g1-mlp-family-generate.md:318-385`. Vehicle: lane-92 oneshot
`a9d41d09…`. Command:

```
tools/gpu_lane_lock.sh qwen38-mlp-family-q3mlp-wall \
  $BIN --artifact-root $ART/mixed-q3mlp-v1 --tokenizer $TOK \
  --prompt "Say hi." --complete-wall --pairs 3 \
  --max-new-tokens 32 --max-seq-len 512
```

```
COMPLETE_WALL_NS_PER_TOKEN   148588917     MEASURED
COMPLETE_WALL_TPS            6.7300        DERIVED 1e9/148588917
GPU_NS_PER_TOKEN             146963124     MEASURED
WALL_MINUS_GPU_NS            1625793       MEASURED
REP_MEDIANS_NS               [148424792, 148588917, 148460333,
                              148480250, 148721583, 148600250]
spread                       148424792 / 148721583  (0.297 ms)
CONTROL_DECODE_WALL          148614061
timing_label                 DIRTY_ENGINEERING
fallbacks                    0
dense_w                      0
census                       uniform=498 f32=353 q4=0 refused=0
```

Authority = median of 6 per-rep medians of per-step complete_wall_ns
on 31 decode steps/rep after one discarded cold. Ratio vs live G0:
148_588_917 / 39_326_090 = **3.778×** DERIVED.

Weight GEMV bucket **137,099,341 ns = 92.27%** of the token is
INFERRED (GPU − APPLIED G0 isolated non-GEMV). RECEIPT
`g1-q3-uniform-kernel.md:318-333`. Ceremony did not explode.

### 8.3 Candidate on the FINAL kernel — PLACEHOLDER

Do not copy §8.2 into the promotion evidence as the child wall.

```
child.complete_token_ns          PLACEHOLDER   paired 3 A/B, GPU timestamps
child.tps                        PLACEHOLDER   1e9/ns
child.gpu_ns                     PLACEHOLDER
child.wall_minus_gpu_ns          PLACEHOLDER
child.rep_medians                PLACEHOLDER
child.timing_label               PLACEHOLDER   CLEAN_CANDIDATE required for ACCEPT
child.weight_gemv_ns             PLACEHOLDER   isolated or addr_probe on HGRAVU kernels
projected_tps_if_geo_matches_g0  ~31           PROJECTED  (11.472705646e9 / 639.25e9
                                               + APPLIED non-GEMV + MEASURED ceremony
                                               ≈ 32.0 ms; g1-q3-uniform-kernel.md:590-599)
```

A 1 ns "win" is not material. Gate:
`MATERIAL_COMPLETE_TOKEN_FRACTION = 0.01` of parent wall
SOURCE `promotion.py:35-36`.

---

## 9. BPW receipt

Definition (G0, SOURCE `qwen38_pack.rs:673-679`):

```
complete_physical_bpw = 8 * tensor_payload_bytes / source_weight_elements
N = 26_895_998_464
```

`tensor_payload_bytes` = sum of catalog `nbytes` (codes + scales +
HGRAVU01 JSON headers). Not dir size. Not catalog file. Not slack.

| quantity | bytes | elems | BPW | tag |
|---|---:|---:|---:|---|
| complete payload | 12_149_632_429 | 26_895_998_464 | **3.6138111608720234** | MEASURED this process |
| MLP (192) | 6_952_112_640 | 17_112_760_320 | **3.2500251321231617** | MEASURED |
| gate = up = down | 2_317_370_880 each | 5_704_253_440 | 3.2500251321231617 | MEASURED |
| attention bucket (528) | 3_846_274_384 | 7_239_780_864 | 4.2501555848196455 | MEASURED |
| embed | 675_430_686 | 1_271_398_400 | 4.250001799593266 | MEASURED |
| lm_head | 675_430_686 | 1_271_398_400 | 4.250001799593266 | MEASURED |
| streamed GEMV (excl embed + 353 vectors) | 11_472_705_646 | — | 11.47 GB/token | MEASURED |
| PACK_REPORT complete (payload+catalog) | 12_149_812_553 | same N | 3.6138647373176767 | PACK_REPORT; **not** the G0 definition |
| G0 complete | 14_297_694_680 | same N | 4.252735126866492 | MEASURED inventory / remasure |

DERIVED vs G0:

- complete bytes: 1 − 3.6138111608720234/4.252735126866492 = **15.02% fewer**
- streamed GEMV vs G0 13_611_663_360
  (RECEIPT `g1-direct-gemv-geometry.md` / `g1-q3-uniform-kernel.md:416-419`):
  1 − 11_472_705_646/13_611_663_360 = **15.71% fewer**

This BPW does not change when the kernel changes. Remeasure only if
the catalog bytes change. They must not.

---

## 10. Capability receipt

**Do not write `{1.0, 1.0, 1.0}`.**

### 10.1 Campaign gate on the CURRENT kernel — MEASURED

RECEIPT `g1-mlp-family-generate.md:13-21,185-315`.

| probe | result | tag |
|---|---|---|
| France @ 128 contains `Paris` | True, 8 times, token 11751 × 8 | MEASURED |
| 17×19 @ 256 contains `323` | True, 3 times; opens `323` then `323 = 17 × 19` | MEASURED |
| fallbacks | 0 | MEASURED |
| dense W / expanded_to_q4 / expanded_to_float_gemv | 0 | MEASURED |
| vehicle | native HQ38M20 | MEASURED |
| 6/6 oracle-32 vs G0 `Say hi.` | **not claimed**; child ids start `[198, 12675, …]` not `[248068, 198, 760, …]` | MEASURED |
| quality | loops after the fact (`Assistant`, `few`, fence ticks) | MEASURED |

16-token France has **no** Paris. 16-token cannot declare COHERENT.
Phase B is the campaign gate. Gate **COHERENT**. Quality is below G0.

mixed-q4down-v1 on the same vehicle: **INCOHERENT** (newline collapse).
MLP 2.221 with Binary gate + rice up + Q4 down does not rescue.
RECEIPT same file `:13-20,443-460`.

### 10.2 Seal fields — PLACEHOLDER after kernel + independent remeasure

Schema `hawking.genesis.qwen38_capability_seal.v1`
(RECEIPT `g1-capability-gate.md:49-114`).

```
verdict                         PLACEHOLDER
artifact.catalog_sha256         72ed83a21213605026428daa128231c0a220c8fb997f1ec46ffd760de40fd8fb
artifact.artifact_content_sha   89399408f7d296308b21a741b586348d3c514d7525df2f555c35fb5895d94261
artifact.complete_physical_bpw  3.6138111608720234
weight_screen.min               PLACEHOLDER   n_none must be 0; never fold 1.0
generation.P1..P6               PLACEHOLDER   re-run on the final binary
generation.oracle32_say_hi      PLACEHOLDER   exact match required only if
                                              artifact_content_sha == G0
derived_capability.coherence    PLACEHOLDER   n_coherent / n_required; not 1.0 default
derived_capability.engineering  PLACEHOLDER   1.0 iff fallbacks==0 and native
                                              and dense_w==0
derived_capability.complete_token_discipline
                                PLACEHOLDER   omit unless same-stopwatch authority
```

`CLAUSE_GREEDY_TOKEN_IDS` exact parent==child is the wrong single bar
for a new pack (RECEIPT `g1-capability-gate.md:380-395`). Use
generation classes + task tokens. Exact 32-id match only when
`artifact_content_sha` equals G0.

Negative control that must stay FAIL: mixed-2p0-v1 native generate,
0 fallbacks, 6/6 INCOHERENT. RECEIPT
`QWEN38_NATIVE_MIXED_2P0_GENERATE.json` /
`g1-capability-gate.md:469-485`.

---

## 11. Fallback receipt

Current kernel (RECEIPT `g1-mlp-family-generate.md:149-154,337`):

```
fallbacks_total              = 0     every prompt, every invocation
dense_w_materialized         = 0
expanded_to_q4               = 0
expanded_to_float_gemv       = 0
refused                      = 0
silent_fallback_ids          = []
```

`fallbacks_total != 0` or `DENSE_W_MATERIALIZED != 0` makes the run
VOID. RECEIPT `g1-mlp-rolelock-unlock.md:293`.

PLACEHOLDER: same four integers on the final-kernel generate and wall.
A new silent fallback vs parent is `CLAUSE_NO_NEW_SILENT_FALLBACK` FAIL.

---

## 12. Machine profile

| field | value | tag |
|---|---|---|
| host | Apple M3 Ultra | MEASURED `sysctl machdep.cpu.brand_string` |
| unified memory | 96 GiB (103_079_215_104 B) | MEASURED `hw.memsize` |
| cores | 28 physical = 28 logical | MEASURED `hw.ncpu` / `hw.physicalcpu` |
| GPU cores | 60 | RECEIPT `g1-baseline-remeasure.md:4` (this process has no GPU-core sysctl) |
| OS | macOS 27.0 (26A5406e) arm64 | MEASURED `sw_vers` / `uname` |
| build dir | `workspace/ops/build/rust` | repo convention |
| GPU lock | `/tmp/hawking-gpu-lane.lock` | SOURCE `tools/gpu_lane_lock.sh` |
| live G0 body | ~13.61 GB resident, must keep running | RECEIPT inventory `weight_bytes=14297675776` |
| G0 addressing regime | 639.25 GB/s | DERIVED 13_611_663_360 / 21_293_102.5 ns RECEIPT `g1-direct-gemv-geometry.md:109` |
| roof | conditioned on the current execution genome | standing rule |

Lane-92 wall machine note (RECEIPT `:319-324`): MEM free 19.01 GB,
anon 40.08 GB, genesis pid 50196 ALIVE, lock FREE at start. Label
DIRTY_ENGINEERING: other CPU/memory lanes live.

PLACEHOLDER: `sysctl` / `system_profiler` snapshot next to the
final-kernel wall receipt.

---

## 13. Negative science delta

First-class. Do not re-propose without a changed premise.

| id | mechanism | verdict | REOPEN_IF |
|---|---|---|---|
| NS-MLP-2p0 | mixed-2p0 (B01 gate / R02 up / S01 down / Q4 attn) | KILLS native generate, 0 fallbacks | a different MLP recipe generates English |
| NS-MLP-q4down | restore only down to Q4, keep B01+R02 | KILLS (lane 92 INCOHERENT) | Uniform on gate+up |
| NS-SUB15 | mixed-sub15 expand-to-Q4 vehicle | KILLS as a codec verdict | native HQ38M20 generate of a *different* recipe |
| NS-SUB15-BYTE | complete < 1.5 under this family | KILLS arithmetic + every tested mechanism | new codec family that holds 0.99 attn and 0.95 MLP |
| NS-GENRES | generator + residual | KILLS zero hits | activation-weighted SVD on a Q4-vehicle capture beats Q4 error at lower BPW |
| NS-VQ | PQ / RVQ / D4 / E8 / Hadamard / trellis | KILLS 0.990 below 4.0 BPW; codebook GEMV 460 µs | new family + generate |
| NS-ENTROPY | entropy coding | KILLS Shannon complete BPW 3.732 under this quantizer | different quantizer |
| NS-F16MEGA | 8-layer f16 megakernel | KILLS 4.4× slower | new occupancy proof |
| NS-USERES | `use_resource` | KILLS 2.62 µs vs 4.5 set_buffer | new Metal API |
| NS-FUSE-VI | fuse-tails-into-vi / encoder-share on ΔNet | KILLS | new measurement |
| NS-COSINE | organ cosine as GO | KILLS | cosine predicts generate class on Qwen3.8 *and* explains mixed-2p0 |
| NS-FOLD1 | `min_q4_cosine` fold 1.0 over None | KILLS | never |
| NS-LABEL | `labeled_sha` as production identity | KILLS | never |
| NS-CAP1 | `DEFAULT_CAPABILITY_CONTRACT` 1.0/1.0/1.0 | KILLS | never |
| NS-EXPAND | low-BPW → expand → generic GEMV | KILLS | complete-token A/B shows net physical win |
| NS-COLSCALE | act-colscale from this capture | KILLS | output-MSE group scale scored on 192/64 split |
| NS-HGRAVS | HGRAVS01 r160_b3 as production down | KILLS honest min 0.7302 | down-only HGRAVS pack generates English, 0 fallbacks |
| NS-RICE | HGRAVR02 rice as production up/attn | KILLS | native generate of a rice-free recipe is a different claim |
| NS-ATTN-CHEAP | attention < Q4 at 0.99 | KILLS | new family, real X, generate |
| NS-SIMD3-TOKEN | current simd3/simd Uniform as G1 velocity | KILLS vs G0 (3.78×) | geo_tpr64-class HGRAVU01 measured complete-token win |

Qwen3.8 is dense: storage BPW = active BPW except the embed table
(one row gathered). Do not import Q80 storage-vs-active ratios.

---

## 14. Reconstruction recipe

The child is reproducible without a new science cycle.

### 14.1 Packer (already executed 2026-08-16, wall 182.70 s)

Source: `lab/operators/qwen38_mlp_not_r160_pack.py` at preserve
`00939c186d79079794650c0ba094773f5c38b784` (not on HEAD; PRESERVE_ONLY).
Argparse SOURCE that commit `:322-330`. Historical command RECEIPT
`g1-recipe-v2.md:308-317`, matches PACK_REPORT `replaced_organs=[0,1,2]`
+ `encoded_tensors=192` + `copied_tensors=659`:

```
ART=/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b

python3 lab/operators/qwen38_mlp_not_r160_pack.py \
  --gate-bits 3 --up-bits 3 --down-bits 3 \
  --tag mixed-q3mlp-v1 \
  --mixed $ART/mixed-2p0-v1 \
  --model-dir $ART/bf16 \
  --root $ART/mixed-q3mlp-v1
```

Does not mutate mixed-2p0 or BF16. Non-replaced segments hardlinked.
MLP re-encoded from BF16 as HGRAVU01 absmax g64. Never materializes
dense W.

Acceptance of a reproduction: HQ38M20 opens; census
`refused=0 expanded_to_q4=0 expanded_to_float_gemv=0`;
`sum(catalog nbytes) == 12149632429`;
`catalog_sha256 == 72ed83a212…`;
`artifact_content_sha == 89399408f7d2…`.
A load is not coherence.

**Do not re-run this command against the live root.** It would
overwrite a COHERENT artifact. If a reproduction is required, write
to a new `--root` and compare hashes.

### 14.2 Runtime bind (no pack change)

```
CARGO_TARGET_DIR=workspace/ops/build/rust \
  cargo build --release -p hawking-core --example ascension_qwen38_hybrid_greedy --offline

tools/gpu_lane_lock.sh g1-q3mlp-remeasure \
  workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy \
  --artifact-root $ART/mixed-q3mlp-v1 \
  --tokenizer $ART/bf16/tokenizer.json \
  --prompts-file $ART/coherence_prompts.txt \
  --max-new-tokens 128 \
  --max-seq-len 512
```

Load log **must** contain `opening mixed HQ38M20` and **must not**
contain `opening Metal + 755 catalog tensors`.

### 14.3 Kernel (lane 110; no repack)

HGRAVU01 layout SOURCE `factor_layout_from_meta`
`q80_mixed_decode.rs:1134-1168`:

```
[HGRAVU01][u32 header_len][JSON][f16 scales[groups]][LSB packed codes]
groups = ceil(elements / 64)
code_bytes = ceil(groups * 64 * bits / 8)
bound = (1<<(bits-1)) - 1
```

`cols % 64 == 0` on every GEMV of this model. bits=3 8-wide tiles are
byte-aligned. bits=4 tiles are nibble-aligned.

`dispatch_factor` grows two arms (lane 110; do not edit here):

```
bits == 3 → qwen_uniform_q3_group64_matvec_geo_tpr64_tg128
            grid = ceil(rows/2)*128, TG 128
bits == 4 → qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128
            same launch; q = nibble - 7
```

Keep simd3/simd as `HAWKING_QWEN38_RECON_FUSE=0` diagnostic. Optional
later pack: fuse `in_proj_qkv+z` and `a+b` (same codec, fewer
launches). Not required for the kernel.

KILL: expand HGRAVU01 → HQ30UQ4 or f32, then G0 geo.

---

## 15. Qualification checklist (protected Hawking)

External invoker only (`lineage_gate` / `protected_controller` /
`human_operator`). Parent and child may not invoke. SOURCE
`promotion.py:73-122`.

Missing evidence is PENDING. Present-but-wrong is REJECT. Do not
trust this packet's numbers — recompute.

### 15.1 Independent re-measurement (do not trust the packet)

| # | action | pass | fail |
|---|---|---|---|
| R1 | `shasum -a 256 $ART/mixed-q3mlp-v1/catalog.hq38m20` | equals `72ed83a2…` | catalog mutated |
| R2 | parse catalog; `8*sum(nbytes)/26895998464` | equals `3.6138111608720234` | different payload |
| R3 | recompute `artifact_content_sha` (§1.3) | equals `89399408f7d2…` | any segment changed |
| R4 | `shasum -a 256 $ART/uniform-q4-v1/manifest.json` | equals `d650a757…` | G0 disturbed; abort |
| R5 | native generate France@128 + 17×19@256 on the **final** binary | `Paris` in text and `323` in text; fallbacks=0; dense_w=0; `opening mixed HQ38M20` | VOID or INCOHERENT |
| R6 | complete-wall `--pairs 3` on the **final** binary, GPU timestamps, lock held | 6 rep medians recorded; timing_label CLEAN_CANDIDATE (or DIRTY named); spread reported | single run / CPU-wait proxy |
| R7 | `sha256` of the executable that ran R5/R6 | written as `runtime_sha`; kind `content_sha256.v1` | labeled_sha or missing file |
| R8 | `sha256` of every dispatched .metal | written as kernel merkle | one-file G0 shader hash |
| R9 | census line | uniform=498 f32=353 q4=0 refused=0 expanded=0 | any expand / refuse |
| R10 | mixed-2p0 negative control (already on disk; do not rerun GPU) | seal FAIL / INCOHERENT | any path that PASS-certifies mixed-2p0 |
| R11 | G0 parent wall remeasure or sealed remasure | parent ns from same stopwatch family as R6 | mix 35.2 M / 37.9 M / 39.3 M without a kind |
| R12 | capability seal | no axis is 1.0 unless n_scored>0 and derived; n_none==0 | fold identity / default contract |

### 15.2 Mechanical promotion clauses (SOURCE `promotion.py`)

Evaluate only after R1–R12. Parent = live G0 remasure, not LKG labeled
hashes.

| clause | expected today (slow kernel) | expected after kernel win |
|---|---|---|
| `capability_ge_parent_contract` | PENDING (no seal; default 1.0 is illegal) | PASS only if derived ≥ parent **measured** |
| `no_new_silent_fallback` | PASS (both []) | PASS iff still [] |
| `complete_token_wall_improves_materially` | **FAIL** 148.6 ms ≱ 1% under 39.3 ms | PASS iff child wall ≤ 0.99 × parent |
| `artifact_identity_exact` | FAIL if preimage is `hawking.lineage/…`; PASS if content merkle | PASS on `89399408…` |
| `representation_bpw_exact` | PASS if child.bpw = 3.613811 (6-dec key 3.613811) | same (bytes did not change) |
| `runtime_and_kernel_genome_exact` | FAIL if labeled; PASS if file hashes match evidence | PASS on final binary + shader merkle |
| `protected_tests_pass` | PENDING until R5/R6 receipts, not `"PASS"` strings | PASS from receipts |
| `state_transfer_test_passes` | PENDING | recompute checksum over payload |
| `rollback_artifact_exists` | PASS if LKG valid **and** LKG hashes are kinded | same |
| `reject_tps_up_capability_down` | n/a (TPS down) | FAIL if TPS up and any axis lost |
| `reject_bpw_improved_token_worse` | **FAIL** today (3.614 < 4.253 and 148.6 ms > 39.3 ms) | PASS iff wall also improves |
| `reject_benchmark_changed` | PASS if fingerprint unchanged | same |
| `generation_strictly_increases` | PASS if child.generation==1 | same |
| `identity_model_matches_parent` | PASS `PocketAiHub/Qwen3.8-27B-Abliterated-MLX` | same |
| `greedy_token_ids_agree` | FAIL if required exact; **do not require exact** for this new pack | generation-class PASS + task tokens |

Today the child is a promotion **candidate**, not ACCEPT. After a
measured kernel win, `complete_token_wall_improves_materially` and
`reject_bpw_improved_token_worse` are the two clauses that flip.

### 15.3 Hard vetoes (no threshold)

- Expand-to-Q4 / MLX overwrite vehicle.
- `hawking.lineage/` identity preimage.
- Capability 1.0 from `DEFAULT_CAPABILITY_CONTRACT`.
- `min_q4_cosine` 1.0 with any None.
- New silent fallback.
- G0 manifest sha changed.
- GPU lock not held / concurrent Metal run.
- Projection reported as measurement.

---

## 16. Fields a later lane fills

Copy this block into the promotion evidence. Replace every
PLACEHOLDER. Do not invent.

```
artifact_content_sha              89399408f7d296308b21a741b586348d3c514d7525df2f555c35fb5895d94261
catalog_sha256                    72ed83a21213605026428daa128231c0a220c8fb997f1ec46ffd760de40fd8fb
complete_physical_bpw             3.6138111608720234
mlp_physical_bpw                  3.2500251321231617
parent_manifest_sha               d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df

runtime_sha                       PLACEHOLDER
runtime_path                      PLACEHOLDER
kernel_genome_sha                 PLACEHOLDER
kernel_dispatch_names             PLACEHOLDER
complete_token_ns                 PLACEHOLDER
complete_token_ns_reps            PLACEHOLDER
tps                               PLACEHOLDER
gpu_ns                            PLACEHOLDER
wall_minus_gpu_ns                 PLACEHOLDER
timing_label                      PLACEHOLDER
timing_authority                  MTLCommandBuffer GPUStartTime/GPUEndTime after wait
projected_tps_geo_tpr64           ~31   PROJECTED  do not write this into complete_token_ns

capability_seal.verdict           PLACEHOLDER
capability_seal.derived           PLACEHOLDER
weight_screen.min                 PLACEHOLDER
weight_screen.n_none              PLACEHOLDER   must be 0
generation.france128_has_paris    PLACEHOLDER
generation.arith256_has_323       PLACEHOLDER
fallbacks_total                   PLACEHOLDER
dense_w_materialized              PLACEHOLDER
census                            PLACEHOLDER   expect uniform=498 f32=353 q4=0
```

---

## 17. Evidence appendix

### 17.1 This-process catalog parse + BPW + merkle

```
catalog_sha256 72ed83a21213605026428daa128231c0a220c8fb997f1ec46ffd760de40fd8fb
catalog_bytes 180124
version 1 n_tensors 851 n_segments 258
payload_bytes 12149632429
complete_physical_bpw 3.6138111608720234
codec_counts {3: 851}
MAGIC HGRAVU01×851
BITS {3: 192, 4: 659}
abs_filenames 0  missing_seg 0  outside 0  past_eof 0
dir_file_bytes 13963692970  slack_bytes 1814060541  hardlinked_files 66
streamed_gemv_bytes_excl_embed_small 11472705646
merkle_catalog_sid_sha 511dae054605d1b9e76867b5057502c3249515466541c2073ce5d977fc0ae372
artifact_content_sha 89399408f7d296308b21a741b586348d3c514d7525df2f555c35fb5895d94261
```

Command: Python HQ38M20 parser matching
`qwen38_hybrid_decode.rs:411-488` + 1 MiB streaming sha256 of 258
segment files. Peak RSS of the hasher is pages, not the 14 GB image.

### 17.2 This-process shasums

```
d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df  uniform-q4-v1/manifest.json
72ed83a21213605026428daa128231c0a220c8fb997f1ec46ffd760de40fd8fb  mixed-q3mlp-v1/catalog.hq38m20
ae0bc8defd84a8a1a5cd1c4598224f370c0cfce83a0904e275cbb33df84d32c2  genesis-resident
a9d41d09856ff7ffb7e32ff4fd4f7ad49cafc4301afd4f6814f51f8418898fab  ascension_qwen38_hybrid_greedy (lane 91/92)
06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523  tokenizer.json
5638aab3e77c3b829c43d71c76999238cc60d41f0b8792fe4b6b65c144ae7ac6  qwen38_hybrid_decode.rs
```

### 17.3 Pointers (not re-derived)

- Identity / labeled vs content: `g1-capability-gate.md` §0, §3; `g1-tabula-genome.md` §1; `lab/lineage/{canon,identity,promotion,testing}.py`
- Artifact inventory / G0 BPW: `g1-artifact-inventory.md` §1–§5
- Candidate generate + wall: `g1-mlp-family-generate.md` (lane 92 worktree)
- Kernel cause + geo design: `g1-q3-uniform-kernel.md` (lane 100 worktree)
- Packer invocation: `g1-recipe-v2.md` §7; packer body `git show 00939c186:lab/operators/qwen38_mlp_not_r160_pack.py`
- Role lock unlock: `g1-mlp-rolelock-unlock.md`
- G0 remasure: `g1-baseline-remeasure.md`
- Doctor: `g1-doctor-tensor-map.md` §3–§5
- Gravity recipes: `g1-repack-recipe.md`
- Negatives: `g1-arch-negative.md`; `receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json`
- Capability suite / gate: `g1-capability-suite.md`; `g1-capability-gate.md`
- Token anatomy (G0 964): `g1-token-anatomy.md`; `g1-kernel-inventory.md`

---

```
STATUS
IMPLEMENT_READY

CLAIMS
C1. Lineage binds runtime/kernel (and promotion artifact preimages) to labeled path-string hashes; capability 1.0 is DEFAULT_CAPABILITY_CONTRACT; min_q4_cosine 1.0 is fold(1.0) over 402 Nones. Evidence: §1.1–1.2; identity.py:36-40,286-292; canon.py:29-30; promotion.py:193-200; testing.py:35-42; qwen38_pack.rs:680-684; this-process labeled_sha recomputes.
C2. Required bind is content merkle of manifest/catalog + listed payload files; runtime = sha256(executable bytes); kernel = merkle of dispatched shader bytes. Files/functions in §1.4. Evidence: §1.3; g1-capability-gate.md:299-376; g1-tabula-genome.md:180-211.
C3. mixed-q3mlp-v1 path, catalog sha 72ed83a2…, artifact_content_sha 89399408…, complete BPW 3.6138111608720234 = 8*12149632429/26895998464, MLP BPW 3.2500251321231617, census 851×HGRAVU01 bits 3×192 / 4×659, streamed GEMV 11.472705646 GB. Evidence: §2, §3, §9, §17.1.
C4. Current token dispatches simd3 (bits 3) and simd (bits 4), not geo_tpr64; TOKEN_NS 148588917 / 6.7300 TPS DIRTY MEASURED; GEMV bucket INFERRED 137099341 ns (92.27%); ceremony MEASURED 1625793 ns. ~31 TPS is PROJECTED. Evidence: §7–§8; g1-mlp-family-generate.md:318-385; g1-q3-uniform-kernel.md:0-45,318-400,590-599.
C5. Campaign gate COHERENT (Paris@128, 323@256), fallbacks 0, dense W 0, native vehicle. Not 6/6 oracle-32. Quality loops. mixed-q4down INCOHERENT. Evidence: g1-mlp-family-generate.md:13-21,185-315,443-460.
C6. Child is reproducible from packer 00939c186 + mixed-2p0 + bf16; no repack for the kernel. Evidence: §14; PACK_REPORT.json; g1-recipe-v2.md:308-317.
C7. Protected Hawking must independently rehash, remeasure wall+generate on the final binary, and refuse labeled/default-1.0 paperwork. Today REJECT on token wall and bpw-up/token-worse. Evidence: §15; promotion.py clauses.

EVIDENCE
- this-process catalog parse / merkle / shasums: §17
- lab/lineage/identity.py:36-40,51-63,276-310
- lab/lineage/canon.py:29-30
- lab/lineage/promotion.py:35-73,181-200,263-290,429-514
- lab/lineage/testing.py:35-42
- crates/hawking-core/src/model/qwen38_pack.rs:673-684,312,403
- crates/hawking-core/src/model/qwen38_hybrid_decode.rs:139-157,227-275,411-488,1662-1692
- tools/agentos/genesis_body/src/main.rs:398-406,843-852
- git show 00939c186:lab/operators/qwen38_mlp_not_r160_pack.py
- workspace/superwave/g1/g1-{capability-gate,tabula-genome,artifact-inventory,mlp-family-generate,q3-uniform-kernel,mlp-rolelock-unlock,baseline-remeasure,doctor-tensor-map,repack-recipe,recipe-v2,arch-negative}.md
- mixed-q3mlp-v1/{PACK_REPORT.json,FORMAT.md,catalog.hq38m20}

CHANGES
created workspace/superwave/g1/g1-promotion-packet.md only

TESTS
test -s / wc -l / git status --porcelain  (see completion report)

RISKS
- TOKEN_NS 148.6 ms is DIRTY_ENGINEERING. Independent CLEAN_CANDIDATE remasure required.
- GEMV 137.1 ms is INFERRED, not an isolated probe on this artifact.
- Campaign 0.9663 / 0.7529 MLP floors were not re-scored this lane; packed screen min 0.96528 and 6-layer hold 0.9679 are the on-disk numbers.
- 66 hardlinked segments: a write to mixed-2p0 Lxx files would silently change this catalog's bytes. Merkle R3 catches it.
- genesis_body still hashes only manifest.json; a mixed child cannot bind through the resident until §1.4 lands.
- Lane 110 may change q80_mixed_decode.metal; §7.2 hashes are this worktree, pre-110.

UNRESOLVED
- Post-kernel TOKEN_NS / TPS / capability seal / runtime+kernel content hashes (PLACEHOLDER).
- Full 192-tensor Q3 hold-cosine census (campaign 0.9663 not independently recomputed).
- G0 fused 96 in_proj cosine still None; G0 seal stays UNMEASURED on those rows until fuse is scored.
- Whether CLEAN_CANDIDATE wall can be taken while the 13.6 GB resident stays up.

NEXT
1. Land §1.4 identity binding (or Phase-0 sidecars) before any seat/promote.
2. Lane 110 lands geo_tpr64 HGRAVU01; do not repack.
3. Measurement lane fills §16 from R5–R9, then invoke the external gate.
4. If wall does not beat parent by 1%, do not promote. Density alone is not ACCEPT.
```
