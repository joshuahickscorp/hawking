# G1 artifact inventory — Genesis G0 physical ground truth

Date of measurement: 2026-08-17. Repo HEAD `2eee9a00493a8631ec7aede5807a3b2292f8370c` (worktree and `/Users/scammermike/Downloads/hawking`).
GPU generate / Metal timing / training / packing: not run (lane prohibition).
Every number is tagged MEASURED, CLAIMED, or PROJECTED.

## 1. Live G0 identity

**The live G0 model is the language-only uniform-Q4 pack**

`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1`

Not a receipt copy: the resident binary, the launch wrapper, the last listen log, the GPU-lane lock, and `sha256(manifest.json)` all name this directory.

### 1.1 Binding evidence

Launch wrapper hard-codes this root (`tools/genesis_forever.sh`):

```
  /usr/bin/env python3 "$RESIDENT_CLIENT" serve \
    --repo "$REPO" \
    --artifact-root "$REPO/workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1" \
    --tokenizer "$REPO/workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json" \
    --lineage "$REPO/receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json" \
    --max-seq-len 8192
```

`launchctl print gui/503/com.hawking.genesis` MEASURED 2026-08-17:

```
state = running
program = /Users/scammermike/Downloads/hawking/tools/genesis_forever.sh
working directory = /Users/scammermike/Downloads/hawking
pid = 74858
runs = 10
```

Last lines of `workspace/ops/genesis-resident.log` (mtime Aug 17 10:42:37, size 13011):

```
genesis-resident: body resident 3.435s weight_bytes=14297675776
genesis-resident: listening /Users/scammermike/Downloads/hawking/workspace/ops/genesis-resident.sock pid=74869
[hawking] HAWKING_TCB_TRACE="(unset)" → mode=Off
```

No later `exit 0`. GPU lock at query time:

```
/tmp/hawking-gpu-lane.lock/pid   = 74869
/tmp/hawking-gpu-lane.lock/owner = genesis-resident:parent
lock mtime                       = Aug 17 10:58:35 2026
```

Unix socket exists, `CONNECT_OK` (AF_UNIX). A health JSON was **not** retrieved this lane: the body is serial and the lock owner is `parent`, so an `{"op":"health"}` RPC would queue behind a live serve. Last on-disk health snapshot `/tmp/genesis-health.json` (mtime Aug 17 09:23) is a **stale** pid 43872; it still names the same artifact + manifest sha.

`ps` is blocked in this sandbox (`operation not permitted`). Live pid 74869 is inferred from listen-log + lock, not from a process table.

### 1.2 Manifest seal

```
$ shasum -a 256 .../uniform-q4-v1/manifest.json
d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
```

This equals lineage `slots.CURRENT.artifact_sha` and `identity.artifact_sha_authority = sha256(manifest.json bytes)` (`receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json`). The resident computes the same hash in `artifact_manifest_sha` (`tools/agentos/genesis_body/src/main.rs:398-406`).

Lineage `slots.CURRENT.identity.resident_pid = "40316"` is **stale** versus listen-log/lock pid 74869.

### 1.3 What G0 is not

- Not the bf16 source tree.
- Not any `catalog.hq38m20` mixed pack. Loader: if `catalog.hq38m20` exists → mixed path; else uniform-Q4 manifest (`qwen38_hybrid_decode.rs:508-514`). G0 has no catalog file.
- Not Q80 / DSV4F artifacts under `workspace/campaign/records/`.

---

## 2. Bytes, parameters, complete BPW (recomputed)

### 2.1 Parameter count — MEASURED three ways, all equal

`26_895_998_464` language-model elements.

| method | elements | evidence |
|---|---:|---|
| product of every manifest `shape` | 26895998464 | python over 755 rows; 0 disagreements with the `elements` field |
| geometry reconstruction (64 layers, fuse in_proj, skip vision) | 26895998464 | 755 names, 0 missing/extra vs manifest |
| bf16 safetensors headers, `language_model.*` only | 26895998464 | 851 unfused source tensors; fusion preserves count |

Geometry used (from `qwen38_geometry.rs` + `bf16/config.json`):

- 64 layers, hidden 5120, intermediate 17408, vocab 248320
- GQA iff `(layer+1)%4==0` → 16 GQA + 48 DeltaNet
- pack-time fuse: `in_proj_{qkv,z}` → `in_proj_qkvz` `[16384,5120]`; `in_proj_{b,a}` → `in_proj_ba` `[96,5120]`
- vision skipped: 333 `vision_tower.*` tensors, **460_730_096** elements (MEASURED from safetensors headers). Not in G0.

Source tensor census (MEASURED, headers only, no weight payload read):

```
weight_map entries 1184
dtype {'BF16': 1184}
language tensors 851 elements 26895998464
vision tensors   333 elements 460730096
other tensors      0
```

851 − 96 fused pairs (48 layers × 2) = 755 catalog tensors.

### 2.2 On-disk bytes — MEASURED by `lstat`

| bag | files | bytes | what |
|---|---:|---:|---|
| manifest-listed tensors (`*.hq30uq4` + `*.f32v2`) | 755 | **14_297_694_680** | every size == manifest `bytes`; 0 missing |
| those files minus per-tensor headers | 755 | **14_297_675_776** | = logged `weight_bytes` / `resident_bytes()` |
| unused `*.f32bin` sidecars | 353 | 10_584_840 | not in manifest, not uploaded |
| `manifest.json` | 1 | 238_879 | sha above |
| whole directory | 1109 | **14_308_518_399** | listed + sidecars + manifest |

Header strip (MEASURED, matches `Qwen38HybridWeights::resident_bytes` at `qwen38_hybrid_decode.rs:676-684`):

- Q4 header = 32 + 4×rank = 40 bytes × 402 = 16_080
- f32v2 header = 8 × 353 = 2_824
- total headers 18_904
- 14_297_694_680 − 18_904 = 14_297_675_776

Codec-formula check: 402/402 Q4 files match `32 + 4*rank + ceil(E/64)*2 + ceil(E/64)*32`. 353/353 f32v2 files match `8 + 4*E`. 0 tensors have `E % 64 != 0`. Every Q4 file begins `HQ30UQ4\0`. Every f32v2 `u64` numel equals the shape product.

### 2.3 Complete BPW — MEASURED, not copied

Definition used here (same arithmetic the packer writes into `complete_physical_bpw`, but the inputs are `lstat` bytes and shape products):

```
complete_physical_bpw
  = 8 * sum(file size of each of the 755 loaded tensors)
  / 26895998464
  = 8 * 14297694680 / 26895998464
  = 4.252735126866492
```

Other MEASURED quotients, labeled so they are not confused with that definition:

| name | formula | value |
|---|---|---:|
| complete physical (loaded tensor files) | 8 × 14297694680 / 26895998464 | **4.252735126866492** |
| resident upload (headers stripped) | 8 × 14297675776 / 26895998464 | 4.252729504022625 |
| whole directory (includes unused f32bin + manifest) | 8 × 14308518399 / 26895998464 | 4.255954555664269 |
| Q4 files only / Q4 elements | 8 × 14287109840 / 26893352960 | 4.250004783338105 |
| Q4 files only / all elements | 8 × 14287109840 / 26895998464 | 4.249586750720005 |
| f32v2 files / f32 elements | 8 × 10584840 / 2645504 | 32.00853977162764 |
| codec nominal (4 + 16/64, no headers) | `UNIFORM_Q4_NOMINAL_BPW` | 4.25 exactly |

The 4.2527… figure is **storage complete BPW of the language body the runtime loads**. Qwen3.8 is dense: every GEMV is read every token, so storage BPW = active BPW for this artifact. That identity is architectural, not a bandwidth measurement.

Lineage `complete_physical_bpw: 4.252735126866492` is the same number. Agreement is expected (same inputs). It is not used as the measurement.

TPS ~26.4 and TOKEN_NS ~37_879_375 remain **CLAIMED** (`QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL.json`, `GENESIS_LINEAGE_CURRENT.json`). This lane did not remeasure them.

---

## 3. Tensor representation / codec

### 3.1 Catalog mix

755 tensors, schema `hawking.ascent.qwen38_language_uniform_q4.v1` (`qwen38_pack.rs:27-34`, `manifest.json:1-15`):

| kind | count | elements | file bytes | ext | codec |
|---|---:|---:|---:|---|---|
| q4 | 402 | 26_893_352_960 | 14_287_109_840 | `.hq30uq4` | HQ30UQ4 group-64 |
| f32 | 353 | 2_645_504 | 10_584_840 | `.f32v2` | u64 numel + f32 LE |

### 3.2 HQ30UQ4 (weight GEMVs)

From `crates/hawking-core/src/model/qwen_complete_binary/uniform_q4.rs:3-18` and `parse_uniform_q4_header`:

- magic `HQ30UQ4\0`, version 1, group_size 64
- header 32 bytes + `u32` dims
- FP16 scale per group of 64 flat elements
- 32 code bytes per group (even nibble low, odd high; `q = nibble - 8`)

Hexdump of embed `7cf1b122….hq30uq4` (size 675_430_440, shape `[248320,5120]`):

```
00000000: 4851 3330 5551 3400 0100 0000 4000 0000  HQ30UQ4.....@...
00000010: 0200 0000 0000 c84b 0000 0000 0000 0000  .......K........
00000020: 00ca 0300 0014 0000                      ....vocab=248320, cols=5120
```

Loader (`qwen38_hybrid_decode.rs:538-559`) parses the header, then uploads **scales and codes only** into two Metal buffers. It does not expand to f32/Q4-generic GEMV. Dispatch name: `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`.

Embed lookup is the same codes/scales via `qwen_uniform_q4_embedding_lookup`.

### 3.3 f32v2 (small vectors)

`read_qwen38_f32_payload` (`qwen38_pack.rs:751-767`): `u64` LE numel + `numel` f32 LE values. Size = `8 + 4*numel`.

RMS norms are stored already converted from MLX `(1+δ)` to HF `δ`. MEASURED on `layers.0.input_layernorm`:

```
f32v2 first value = 0.046875
f32bin first value = 1.046875
difference         = 1.0
```

`f32bin` is the pre-conversion sidecar. Same 8-byte header, different payload. Not in the manifest. Not uploaded.

### 3.4 Per-class file BPW (MEASURED)

```
class                           n kind       elements     file_bytes        bpw
deltanet.A_log                 48 f32            2304           9600  33.333333
deltanet.conv1d                48 f32         1966080        7864704  32.001562
deltanet.dt_bias               48 f32            2304           9600  33.333333
deltanet.in_proj_ba            48 q4         23592960       12535680   4.250651
deltanet.in_proj_qkvz          48 q4       4026531840     2139096960   4.250004
deltanet.norm                  48 f32            6144          24960  32.500000
deltanet.out_proj              48 q4       1509949440      802162560   4.250010
embed                           1 q4       1271398400      675430440   4.250000
gqa.k_norm                     16 f32            4096          16512  32.250000
gqa.k_proj                     16 q4         83886080       44565120   4.250061
gqa.o_proj                     16 q4        503316480      267387520   4.250010
gqa.q_norm                     16 f32            4096          16512  32.250000
gqa.q_proj                     16 q4       1006632960      534774400   4.250005
gqa.v_proj                     16 q4         83886080       44565120   4.250061
lm_head                         1 q4       1271398400      675430440   4.250000
mlp.down_proj                  64 q4       5704253440     3030387200   4.250004
mlp.gate_proj                  64 q4       5704253440     3030387200   4.250004
mlp.up_proj                    64 q4       5704253440     3030387200   4.250004
norm.final                      1 f32            5120          20488  32.012500
norm.input                     64 f32          327680        1311232  32.012500
norm.post_attn                 64 f32          327680        1311232  32.012500
```

f32 BPW > 32 is header overhead on tiny tensors (`A_log`/`dt_bias` are 48 elements + 8-byte header).

Embed and lm_head are **two copies** (same shape, two files, 675_430_440 bytes each). Not tied on disk.

---

## 4. Runtime binary and kernel set

### 4.1 Binaries — MEASURED

| role | path | size | mtime | sha256 |
|---|---|---:|---|---|
| **live resident** | `workspace/ops/build/rust/release/genesis-resident` | 6_735_488 | Aug 17 01:51:23 | `ae0bc8defd84a8a1a5cd1c4598224f370c0cfce83a0904e275cbb33df84d32c2` |
| oneshot / wall | `workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy` | 4_656_528 | Aug 16 22:36:29 | `a01452066e8811bcb0fb327247a74f74d21a9f4a6b3980796be8fcc76d15233f` |
| stale build | `/tmp/genesis-resident-target/release/genesis-resident` | 6_679_600 | Aug 16 23:29 | not the live file |

Live sha matches lineage `identity.resident_executable_sha256`. Source crate: `tools/agentos/genesis_body` (`[[bin]] name = "genesis-resident"`). Load: `Qwen38HybridWeights::load` (`genesis_body/src/main.rs:470-471`). Tokenizer: `.../qwen38-27b/bf16/tokenizer.json` (19_989_325 bytes).

`ascension_qwen38_hybrid_greedy` is the measurement vehicle (`examples/ascension_qwen38_hybrid_greedy.rs:1-16`). Same loader, same artifact. Not the long-lived process.

### 4.2 G0 token-path kernels (used, not merely compiled)

Defaults in `Qwen38HybridDecodeSession::attach` (`qwen38_hybrid_decode.rs:910-912`):

- `matvec_kernel = GeoTpr64Tg128`
- `deltanet_vi_parallel = true`
- `HAWKING_DECODE_FAMILY` default **on** (`decode_family.rs:103-105`) → `gk_swiglu_f32` not `qwen80_silu_mul_f32`

| stage | kernel | shader file | sha256 of file |
|---|---|---|---|
| Q4 GEMV (all large W) | `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` | `qwen_uniform_q4.metal` | `51abdf7be388d62ba080d13a1f97a18ab8b1114c0a6968e9d0f04d109d3efcd1` |
| embed gather | `qwen_uniform_q4_embedding_lookup` | same | same |
| RMSNorm | `qwen80_residual_rmsnorm_f32` | `qwen80_device_activations.metal` | `c3efa790167f2e9259b8eb1bc0a2184600b036f4a20213091ce00ede48d9cc0a` |
| ΔNet rearrange+conv | `qwen38_qkvz_rearrange_conv_l2_f32` | `qwen38_device_activations.metal` | `a95a17344fcca19e557eb48e0fc0b33714bb12624d8647d27a9119704e0ea408` |
| ΔNet β, decay | `qwen80_ba_to_decay_beta_f32` | `qwen80_device_activations.metal` | (above) |
| ΔNet recurrence | `qwen38_gated_delta_decode_vi` | `qwen38_device_activations.metal` | (above) |
| ΔNet gated RMS | `qwen80_deltanet_gated_rmsnorm_f32` | `qwen80_device_activations.metal` | (above) |
| GQA QK RMS+RoPE+cache | `qwen38_gqa_qk_norm_rope_cache_f32` | `qwen38_device_activations.metal` | (above) |
| GQA output gate | `qwen38_attention_apply_sigmoid_gate` | `qwen38_device_activations.metal` | (above) |
| GQA attention | `mha_decode_f32` | `mha.metal` | `a2717a7f09f74245d05e228578e5460f30c0bb102f6382dfc78a93ca51138ed5` |
| SwiGLU | `gk_swiglu_f32` | `gk_family.metal` | `c98d4fc1daf8f1645479d7dea5259afd33fc342ecc1fceb1771142d8bcab5f2b` |
| residual add | `qwen_next_add_residual` | `qwen_next.metal` | `178b59598cc70d54bfbb36fea3a9f4dbdcdeec627a6a03caf4769ffab2f91bec` |
| sample | `sample_argmax_f32` | `sample.metal` | `918e8250175d30bf11df75ae2f72346a4a93e132851b6e79c36f9aa132d148ff` |

Lineage `identity.kernel_source` lists **only** `qwen_uniform_q4.metal`. That is the weight-codec shader. The token graph also binds the activation/GQA/ΔNet/SwiGLU/sample files above. `SHADER_QWEN38_DEVICE_ACTIVATIONS` is appended in `all_shader_sources` at `metal/mod.rs:455`.

`MetalContext` compiles the whole `all_shader_sources()` library (36 `.metal` files, including unused Q80 MoE / DSV4F / RWKV). Compiled ≠ dispatched. G0 dispatch set is the table.

Fallback ΔNet kernel `qwen80_gated_delta_decode_tg` exists but is not the attach default.

### 4.3 Binding: no expand-then-generic-GEMV on G0

`Qwen38HybridWeights::load` for this artifact keeps Q4 packed (`codes` + `scales` buffers) and dispatches `qwen_uniform_q4_*`. That satisfies the representation-specific-kernel preference. Mixed-path reconstruct-to-Q4 is explicitly refused (`qwen38_hybrid_decode.rs:6-7, 652-656`).

---

## 5. Every other Qwen3.8 / gravity artifact on this disk

Search scope MEASURED: `workspace/campaign` (all `catalog.hq38m20`, `*.hq38seg`, `*.hq30uq4` live only under `runs/qwen38-27b`), plus `~/.cache` (no extra Qwen3.8 weight tree), plus `/tmp` (binaries/receipts, no second weight tree). Worktree checkouts of the repo are source, not artifacts.

Root listing (`os.walk` byte sums, local mtimes):

```
DIR  .cache                       files=   50 bytes=           2892 mtime=2026-08-16 02:10:52 -0400
DIR  activation-capture-v1        files=   65 bytes=      335564680 mtime=2026-08-16 15:29:15 -0400
DIR  bf16                         files=   24 bytes=    54740460836 mtime=2026-08-16 02:14:29 -0400
FILE bf16-smoke-generate.json     bytes=            545 mtime=2026-08-16 15:25:05 -0400
FILE coherence_prompts.txt        bytes=            171 mtime=2026-08-16 18:17:37 -0400
DIR  mixed-2p0-materialized       files= 1110 bytes=    14308585711 mtime=2026-08-16 17:19:05 -0400
DIR  mixed-2p0-v1                 files=  138 bytes=     7012099402 mtime=2026-08-16 16:42:01 -0400
DIR  mixed-floor-q7-v1            files=  135 bytes=    10680760838 mtime=2026-08-16 18:20:12 -0400
DIR  mixed-floor-q8-up10-v1       files=  135 bytes=    12204302393 mtime=2026-08-16 18:43:12 -0400
DIR  mixed-floor-q8-v1            files=  135 bytes=    11903664460 mtime=2026-08-16 18:23:54 -0400
DIR  mixed-q3mlp-v1               files=  261 bytes=    13963923935 mtime=2026-08-16 19:38:31 -0400
DIR  mixed-q4down-v1              files=  133 bytes=    10042167641 mtime=2026-08-16 19:29:22 -0400
DIR  mixed-sub15-v1               files= 1721 bytes=    15473851850 mtime=2026-08-16 17:13:56 -0400
DIR  uniform-q4-v1                files= 1109 bytes=    14308518399 mtime=2026-08-16 03:15:06 -0400
```

Loadable means: the **current** `Qwen38HybridWeights::load` would accept the on-disk bytes (catalog/manifest parse + every referenced payload present + codec magic the match-arm handles). It is **not** a generate run. Capability/coherence of non-G0 packs is CLAIMED from receipts, labeled as such.

### 5.1 `bf16/` — source, not G0-loadable

- 11 safetensors, index `total_size` 54_713_457_120; walk 54_740_460_836 (headers/json extras).
- 1184 BF16 tensors: 851 language + 333 vision.
- `config.json`: `Qwen3_5ForConditionalGeneration` / `qwen3_5` / `num_hidden_layers=64` / `hidden_size=5120` / `intermediate_size=17408`.
- Model id (code): `PocketAiHub/Qwen3.8-27B-Abliterated-MLX` (`qwen38_geometry.rs:12-15`).
- Tokenizer used by the resident lives here.
- Loader requires `manifest.json` schema `hawking.ascent.qwen38_language_uniform_q4.v1` or `catalog.hq38m20`. bf16 has neither. **Not loadable by genesis-resident.**
- `bf16-smoke-generate.json` is an MLX-style smoke (wall_s 896.7, garbled text). Not a Hawking-native load.

### 5.2 `activation-capture-v1/` — activations, not weights

335_564_680 bytes, 64 × `Lxx.f32` hidden dumps. Schema `hawking.ascension.qwen38_bf16_post_swiglu_activation_capture.v1`, status `CAPTURED_REAL_BF16_POST_NORM_HIDDEN`, 256 tokens × 64 × 5120. Used to fit mixed down-proj. **Not a model.**

### 5.3 HQ38M20 mixed packs (native mixed loader)

All six parse: magic `HQ38M20\0`, version 1, 851 tensors, 0 missing segments, 0 range faults. Codec 0/1/2/3 are exactly the match-arms in `load_mixed` (`qwen38_hybrid_decode.rs:601-664`).

| artifact | dir bytes | catalog payload bytes | MEASURED BPW (payload×8/params) | codecs on disk | structurally loadable |
|---|---:|---:|---:|---|---|
| mixed-2p0-v1 | 7_012_099_402 | 7_011_580_330 | 2.085538587 | 64×HGRAVB01 + 64×HGRAVR02 + 64×HGRAVS01 + 659×HGRAVU01 | YES |
| mixed-floor-q7-v1 | 10_680_760_838 | 10_680_295_260 | 3.176768551 | same 64/64/64/659 split | YES |
| mixed-floor-q8-v1 | 11_903_664_460 | 11_903_200_220 | 3.540511868 | same | YES |
| mixed-floor-q8-up10-v1 | 12_204_302_393 | 12_203_836_482 | 3.629933724 | same | YES |
| mixed-q4down-v1 | 10_042_167_641 | 9_948_135_693 | 2.958993534 | 64×B01 + 64×R02 + 723×U01 (no S01) | YES |
| mixed-q3mlp-v1 | 13_963_923_935 | 12_149_632_429 | 3.613811161 | 851×HGRAVU01 | YES |

`mixed-q3mlp-v1` segment files contain **1_814_060_541 unused slack bytes** (MEASURED: file size − merged catalog intervals). Do not rebuild; the addressed payload is complete. Dir-BPW using slack is 4.153, which is the wrong denominator for “what the catalog names”.

`mixed-2p0-v1` was opened by the native reader (CLAIMED, `QWEN38_NATIVE_MIXED_READER.json` / `QWEN38_NATIVE_MIXED_2P0_GENERATE.json`): 0 fallbacks, 0 dense-W materialize, output incoherent (newlines / `)`). `GENERATE.json` in the artifact dir is a **different** vehicle (`engine: mlx_lm_weights_overwritten_from_mixed_pack`) — expand-to-float, not the Metal mixed path.

Floor / q3mlp / q4down: no generate was run this lane. Structural loadability ≠ coherence.

Mixed-path kernels (not G0, only if someone loads a catalog): `q80_binary_group_matvec_tg256`, `q80_binary_group_csr_matvec_tg256`, `q80_hgravs01_factor_matvec_simd3`, `q80_hgravs01_factor_matvec_simd`, `q80_uniform8_matvec_*`, `qwen38_hgravu_embedding_lookup`, plus the fuse kernels `qwen38_fuse_split_{qkvz,ba}_f32`. Shader: `q80_mixed_decode.metal` + `qwen38_device_activations.metal`.

### 5.4 Uniform-Q4 siblings (same loader as G0)

Both have schema `hawking.ascent.qwen38_language_uniform_q4.v1`, 755/755 files present, sizes match, no catalog → G0 loader path.

| artifact | listed tensor bytes | inode overlap with G0 | what it actually is |
|---|---:|---|---|
| mixed-2p0-materialized | 14_297_694_680 | 916 shared / 192 unique | G0 attention+embed+lm_head+f32 **hardlinked**; 192 = 64×3 MLP tensors replaced (reconstructed mixed→Q4). packed/mlp_rows.json 67_311 B |
| mixed-sub15-v1 | 14_297_694_680 | 708 shared / 400 unique | shared = 353 f32v2 + 353 f32bin + embed + lm_head. 400 unique `.hq30uq4` still magic `HQ30UQ4\0` (sampled 3). packed/attn = 608 rice files, 1_165_176_979 B — **not read** by the uniform loader |

`mixed-sub15-v1` `PACK_REPORT.json` `generate_vehicle.note`: “HQ30UQ4 of reconstructed mixed/rice weights”. CLAIMED coherence: `QWEN38_SUB15_INCOHERENT.json` degenerate 220/264 cycle. Projected 1.291 BPW / 79 TPS in that receipt is **PROJECTED**, and refers to the rice/mixed ledger, not the reconstructed Q4 tensors the loader would actually eat (those Q4 tensors still sum to 14_297_694_680 B → 4.2527 BPW).

Do not rebuild G0-format Q4 of embed/lm_head/f32: they are already hardlinked into both siblings.

### 5.5 Non-artifacts that look related

- `qwen38-27b/.cache/huggingface/download/...` — 50 lock/metadata files, 2892 B. Not weights.
- `tools/qwen38_recon_disc/` — packer source + `recon.metal`, no weight payload.
- `/tmp/genesis-resident-target` — old resident binary, not live.
- `/tmp/qwen38_*.json`, `/tmp/genesis-health.json` — receipts / stale health. Not models.
- Ascent-lane markdown under `workspace/ops/ascent-lanes/auto-qwen38-*` — notes, not tensors.
- Frankenstein `*.pt` / `*.npz` under `workspace/campaign/evidence/models/frankenstein` — GLM/other campaign, not Qwen3.8 G0.

---

## 6. Implications for later G1 lanes

1. **Do not rebuild** `uniform-q4-v1`. It is the live body, 14.30 GB, 755/755 intact.
2. **Do not rebuild** mixed-2p0 / floor / q3mlp / q4down catalogs. All parse, all segments present.
3. **Do not treat mixed-sub15 packed rice as a native generate vehicle.** The loader will ignore `packed/` and load the reconstructed Q4 at 4.2527 BPW.
4. A sub-1.5 complete BPW candidate must be a new pack (or a mixed catalog the native path consumes directly). Expanding mixed→Q4→`qwen_uniform_q4_*` is the rejected shape unless a complete-token measurement shows a net win.
5. G0 kernel genome is **not** only `qwen_uniform_q4.metal`. Changing ΔNet/GQA/SwiGLU/sample shaders changes the execution genome even if Q4 files stay bit-identical.
6. Lineage `resident_pid` and `/tmp/genesis-health.json` are stale. Probe the socket/lock/log, do not trust those fields as live.

---

## 7. What this lane did not measure

- Token time, TPS, GPU ns, bandwidth. CLAIMED numbers in `QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL.json` were not reproduced.
- Whether pid 74869 still exists in the process table (`ps` denied).
- A live health JSON (would serialize behind the parent serve).
- Actual Metal load of any mixed pack (GPU prohibited). Structural loadability only.
- Vision-tower quality or any multimodal path.

Cheapest experiment for a live health bind: one `{"op":"health"}` RPC when the GPU lock is free. Cheapest experiment for mixed loadability-in-process: `Qwen38HybridWeights::load` on each catalog root with the GPU lock held, no generate.

---

```
STATUS
SUPPORTED

CLAIMS
1. Live G0 weights are workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1. Evidence: genesis_forever.sh artifact-root; genesis-resident.log last listen; GPU lock 74869 genesis-resident:parent; manifest sha d650a757… matches lineage CURRENT.
2. Complete physical BPW of that artifact is 4.252735126866492 = 8*14297694680/26895998464. Evidence: lstat of 755 manifest files; shape-product param count; three-way param agreement with geometry and bf16 headers.
3. Every large GEMV is HQ30UQ4 group-64 consumed as codes+scales by qwen_uniform_q4_group64_matvec_geo_tpr64_tg128; 353 small tensors are f32v2. Evidence: hexdump + parse_uniform_q4_header + load() match arms + 402/402 and 353/353 formula checks.
4. Live runtime is workspace/ops/build/rust/release/genesis-resident sha ae0bc8de… (tools/agentos/genesis_body). Measurement oneshot is ascension_qwen38_hybrid_greedy. Evidence: shasum; launchd; genesis_body Body::load.
5. G0 token-path kernel set is the table in §4.2, shaders as hashed there. Lineage kernel_source naming only qwen_uniform_q4.metal is incomplete. Evidence: attach defaults; dispatch_threads / *_tcb names; kernel-to-file grep.
6. Six HQ38M20 packs and two uniform-Q4 siblings already exist and are structurally loadable; mixed-sub15 rice is not what the uniform loader eats. Evidence: §5 catalog parse + inode overlap.
7. TPS 26.4 and TOKEN_NS 37.879e6 are CLAIMED, not remeasured. Evidence: QWEN38_CURRENT_MAIN_COMPLETE_TOKEN_WALL.json; GPU work forbidden.

EVIDENCE
- shasum -a 256 uniform-q4-v1/manifest.json → d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
- shasum -a 256 genesis-resident → ae0bc8defd84a8a1a5cd1c4598224f370c0cfce83a0904e275cbb33df84d32c2
- shasum -a 256 qwen_uniform_q4.metal → 51abdf7be388d62ba080d13a1f97a18ab8b1114c0a6968e9d0f04d109d3efcd1
- python lstat/shape/codec script: 755 ok, 0 missing, 0 size mismatch, elements 26895998464, payload 14297694680, resident_nohdr 14297675776
- launchctl print …/com.hawking.genesis: state=running pid=74858
- genesis-resident.log tail: listening … pid=74869 weight_bytes=14297675776
- /tmp/hawking-gpu-lane.lock pid=74869 owner=genesis-resident:parent
- qwen38_hybrid_decode.rs:508-580 load(); :232-257 kernel consts; :910-912 attach defaults
- qwen38_pack.rs:27-34, :722-768 manifest + f32v2
- uniform_q4.rs:3-18, :38-153 HQ30UQ4 layout
- genesis_body/src/main.rs:456-485, :398-406
- genesis_forever.sh serve argv
- mixed catalog parse: 6/6 HQ38M20 ok, 0 missing_seg, 0 range_fail

CHANGES
Created workspace/superwave/g1/g1-artifact-inventory.md only.

TESTS
```
$ test -s workspace/superwave/g1/g1-artifact-inventory.md && echo 'test -s: PASS'
test -s: PASS
$ wc -l workspace/superwave/g1/g1-artifact-inventory.md
     448 workspace/superwave/g1/g1-artifact-inventory.md
$ git status --porcelain
?? workspace/superwave/g1/g1-artifact-inventory.md
```

RISKS
- Live health not sampled; a crash-restart between lock write and this sentence would stale pid 74869. Socket+launchd+lock still agreed at measurement time.
- Lineage CURRENT file on disk is dirty in the live repo (git status shows it modified). Artifact path and manifest sha still match.
- mixed-q3mlp on-disk segment slack (1.81 GB) can fool a naive dir-bytes/params BPW.

UNRESOLVED
- Current serve_count / max_seq_len of pid 74869 (needs health when lock is free).
- In-process Metal load of the six mixed catalogs (GPU lane).
- Independent TOKEN_NS / TPS (other lane).

NEXT
Later density lanes should start from the mixed catalogs already on disk, not from a rebuild of uniform-q4-v1. Any candidate must keep a representation-specific kernel; do not promote reconstruct-to-Q4 as G1 without a complete-token measurement.
```
