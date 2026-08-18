# G1 mixed-sub15-v1 native implementation

Lane: `50-sub15-native-impl`. Implements the change list in
`g1-sub15-native-gap.md`. No generate. No GPU benchmark. No repack.

Artifact:
`/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1`

Packed complete BPW `1.2910781930062503` is a `RECEIPT` (`PACK_REPORT.json:23`).

---

## 0. Verdict

C1 + C2 + C3 landed. The packed 1.291 BPW bytes are what `load_mixed` will
address once `catalog.hq38m20` sits in the artifact root.

This sandbox cannot write into the campaign artifact directory
(`PermissionError: Operation not permitted` on both `mkdir segments/` and
`catalog.hq38m20`). The packer therefore emitted the catalog into this tree
with **absolute segment filenames** pointing at the already-packed blobs.
A GPU lane installs it with one `cp` of a 309_294-byte file. That is not a
repack and not a 4.34 GB copy.

The bind is **K-complete**. Fusion stays **on**. Binary / rice GEMVs with
`cols > 2048` dispatch `q80_binary_group_matvec_simd_bytes` /
`q80_binary_group_csr_matvec_bytes` (256-col tiles). This pack's K is only
`{5120, 6144}`, both multiples of 256. A non-multiple refuses rather than
silently dropping a remainder.

No expansion to Q4 or to float exists on the GEMV path. Codec 4 f32v2 is
uploaded as the already-δ oracle vector; `mlx_residual_norm_to_delta` is
not called.

A coherence verdict is now obtainable. It has not been obtained here.

---

## 1. Per-codec load result (CPU census, no Metal)

Command (this process):

```
cargo test -p hawking-core --lib --offline mixed_catalog_contract -- --nocapture
```

Test `sub15_native_catalog_census_if_emitted` walked every catalog row
through `classify_qwen38_mixed_payload` + `mixed_gpu_layout` /
`parse_uniform_q4_header` / `read_qwen38_f32_payload`. Rice indices were
**not** expanded. Metal was **not** opened.

`MEASURED`:

| lane | n | source |
|---|---:|---|
| codec 0 HGRAVB01 Binary | 64 | mixed-2p0 `Lxx.hq38seg` slices (gate) |
| codec 1 HGRAVR02 Residual | 368 | 64 up_proj + 304 attention rice |
| codec 2 HGRAVS01 Hgravs r160_b3 | 64 | mixed-2p0 down_proj slices |
| codec 3 HGRAVU01 Uniform | 0 | not in this pack |
| codec 3 HQ30UQ4 | 2 | embed + lm_head (same inodes as uniform-q4-v1) |
| codec 4 f32v2 | 353 | oracle small tensors, already HF δ |
| **total** | **851** | |
| refused | **0** | |
| expanded_to_q4 | **0** | |
| expanded_to_float_gemv | **0** | |

Packer emit echoed the same census:

```
tensors=851 segments=723 codecs={0: 64, 1: 368, 2: 64, 3: 2, 4: 353}
```

Spot-check magics (packer `peek`, this process):

| name | magic |
|---|---|
| L0 `mlp.gate_proj` | `HGRAVB01` |
| L0 `mlp.up_proj` | `HGRAVR02` |
| L0 `mlp.down_proj` | `HGRAVS01` |
| L0 `linear_attn.in_proj_qkv` | `HGRAVR02` |
| `embed_tokens` | `HQ30UQ4\0` |
| `lm_head` | `HQ30UQ4\0` |

`mixed_gpu_layout` on the three L0 MLP slices and on `in_proj_qkv`
(10240×5120) + `in_proj_a` (48×5120) accepted. No KERNEL-MISSING.

---

## 2. What changed

### C1 — catalog emit (`tools/qwen38_sub15_pack.py`)

`--phase catalog` writes `catalog.hq38m20` over already-packed blobs.

- 192 MLP records copied from mixed-2p0 (codec/organ/offset/nbytes/sha256).
- 304 attention records from `packed/attn/*.rice`.
- embed + lm_head codec 3 HQ30UQ4.
- 353 small tensors codec 4 f32v2.
- Hardlink into `root/segments/` when the OS allows it.
- If hardlink/`mkdir` is denied (this sandbox), segment filenames are
  absolute paths to the existing files. `Path::join` on an absolute
  filename keeps that path; `resolve_mixed_segment_path` makes that
  explicit.

Emitted (this process):

```
workspace/superwave/g1/mixed-sub15-native-catalog/catalog.hq38m20   309294 B
workspace/superwave/g1/mixed-sub15-native-catalog/NATIVE_CATALOG.json
```

Not written (sandbox):

```
.../mixed-sub15-v1/catalog.hq38m20
.../mixed-sub15-v1/segments/
```

### C2 — codec 4 f32v2 (`qwen38_hybrid_decode.rs`)

`classify_qwen38_mixed_payload`:

- `0|1|2` → packed upload (unchanged magics).
- `3` HGRAVU01 vector → host dequant + mlx-delta (mixed-2p0 small tensors).
- `3` HQ30UQ4 → `q4` (embed / lm_head).
- `4` → `read_qwen38_f32_payload`, **no mlx-delta**.
- `5+` → refuse.

`load_mixed` uses this classifier. Unknown codecs still fail the run.

### C3 — K-complete bind (`dispatch_binary` / `dispatch_residual`)

When `HAWKING_QWEN38_RECON_FUSE` is on (default):

| cols | binary kernel | residual kernel | grid |
|---|---|---|---|
| `<= 2048` | `q80_binary_group_matvec_tg256` | `q80_binary_group_csr_matvec_tg256` | `rows*256` |
| `> 2048` and `% 256 == 0` | `q80_binary_group_matvec_simd_bytes` | `q80_binary_group_csr_matvec_bytes` | `ceil(rows/8)*256` |
| `> 2048` and `% 256 != 0` | **refuse** | **refuse** | — |

`HAWKING_QWEN38_RECON_FUSE=0` is unchanged: `gk_matvec_binary` walks every
column (serial, honest, slow). It is **not** required for a codec verdict
anymore.

Load log (GPU lane must see these lines before treating tokens as a
codec result):

```
qwen38-decode opening mixed HQ38M20 + 851 catalog tensors (no reconstruct-to-Q4)
qwen38-decode mixed census: tensors=851 binary=64 residual=368 hgravs=64 uniform=0 q4=2 f32=353 refused=0 expanded_to_q4=0 expanded_to_float_gemv=0
qwen38-decode mixed bind: K-complete; recon_fuse=ON uses q80_binary_group_matvec_simd_bytes / q80_binary_group_csr_matvec_bytes when cols>2048 (256-col tiles; this model K in {5120,6144}); cols<=2048 stay on tg256; recon_fuse=0 walks every column via gk_matvec_binary
```

If the first line is `opening Metal + 755 catalog tensors`, the catalog
was not installed and the run is the expand-to-Q4 vehicle. Discard it.

### Uniform-Q4 path

`Qwen38HybridWeights::load` catalog-absent branch is byte-identical to
HEAD (diff is one trailing newline after the function). G0
`uniform-q4-v1` is a different directory and was not touched.

`q80_mixed_decode.rs` / `q80_mixed_decode.metal` were not modified.
`mixed_gpu_layout` still accepts only codecs 0–3; codec 4 never reaches it.

---

## 3. Diff summary

```
 crates/hawking-core/src/model/qwen38_hybrid_decode.rs | 584 +++++++++++++++++----
 tools/qwen38_sub15_pack.py                         | 425 ++++++++++++++-
 2 files changed, 914 insertions(+), 95 deletions(-)
```

Untracked (this lane's catalog, not a source change):

```
workspace/superwave/g1/mixed-sub15-native-catalog/catalog.hq38m20
workspace/superwave/g1/mixed-sub15-native-catalog/NATIVE_CATALOG.json
workspace/superwave/g1/g1-sub15-native-impl.md
```

No shader family. No public signature change on `load`. No dependency /
CI / lockfile touch. No AgentOS / HCLI / HIDE / packaging touch.

---

## 4. Build

`CARGO_TARGET_DIR=workspace/ops/build/rust`

```
cargo build --release -p hawking-core
...
warning: `hawking-core` (lib) generated 9 warnings
    Finished `release` profile [optimized] target(s) in 3m 00s
```

Also built the GPU-lane binary from the same tree:

```
cargo build --release -p hawking-core --example ascension_qwen38_hybrid_greedy
    Finished `release` profile [optimized] target(s) in 2m 49s
```

Binaries this process produced (mtimes after the source edit):

```
2026-08-17 11:43:51 crates/hawking-core/src/model/qwen38_hybrid_decode.rs
2026-08-17 11:53:56 workspace/ops/build/rust/release/deps/libhawking_core.rlib
2026-08-17 11:54:25 workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy
```

SHA-1 of the greedy binary at that mtime:
`1aa9b4e1c0dcce8e08d8bd3cf89a06812240bbf9`
(rebuilt again after the exact `-p hawking-core` command so the example
matches the crate that just compiled).

The sandbox has no Metal device (`Metal("no Metal-capable GPU")` in lib
tests). Release compile succeeded anyway.

---

## 5. Tests

### Pre-change baseline (`cargo test -p hawking-core --lib --offline`)

```
test result: FAILED. 618 passed; 16 failed; 7 ignored; 0 measured; 0 filtered out; finished in 21.44s
```

### Post-change (`cargo test -p hawking-core --lib --offline`)

```
test result: FAILED. 625 passed; 16 failed; 7 ignored; 0 measured; 0 filtered out; finished in 21.87s
```

+7 passed = the seven new `mixed_catalog_contract_tests`. Failure set
identical (16/16 same names). All 16 are pre-existing: no Metal GPU in
this sandbox, missing receipts/profiles, or unrelated Q80 table ABI.

Failing names (baseline = after):

```
backend::honest_roof::tests::backend_honest_roof_gpu_sweep
kernels::tests::q4_k_llama_b9430_metal_matches_raw_f32_reference
kernels::tests::q4_k_serial_authority_metal_matches_raw_f32_reference
kernels::tests::q5_k_serial_authority_metal_matches_raw_f32_reference
kernels::tests::q5_k_serial_authority_persistent_tcb_matches_raw_f32_reference
kernels::tests::q6_k_llama_b9430_metal_matches_raw_f32_reference
kernels::tests::qwen_binary_sign_scale_component_metal_matches_packed_cpu_oracle
kernels::tests::qwen_uniform_q4_group64_component_metal_matches_exact_packed_cpu_oracle
model::dsv4f_activation_capture::tests::sealed_organ_catalog_matches_schedule_receipt
model::qwen30_complete_runtime::tests::binary_simdgroup_candidate_matches_scalar_and_packed_cpu_oracle
model::qwen30_complete_runtime::tests::direct_binary_vector_decode_and_finite_guard_execute_on_metal
model::qwen30_complete_runtime::tests::direct_packed_matvec_honors_route_major_input_buffer_offset
model::qwen80_complete_runtime::tests::layer3_gqa_bridge_builder_returns_ok
model::qwen80_device_expert_table::tests::device_expert_table_abi_matches_metal_static_asserts
profile::tests::pinned_profiles_still_load_after_field_additions
token_ns::energy::tests::ioreport_gpu_energy_is_readable_without_root
```

New tests, all `ok`:

```
hq38m20_magic_and_record_match_q80_layout
absolute_segment_filename_does_not_join_segments
codec_4_f32v2_is_accepted_without_mlx_delta
unknown_codec_5_still_refuses
k_complete_bind_retargets_wide_columns
catalog_roundtrip_codec_4_census_and_codec_5_refuses
sub15_source_payloads_accept_mixed_gpu_layout
sub15_native_catalog_census_if_emitted
```

`cargo test -p hawking-core --offline` (required command) compiled every
hawking-core test/example target (`Finished test profile ... 494/495` in
51.63s) then ran the lib suite and **fail-fasted** on the same 16
pre-existing lib failures. Exact tail:

```
test result: FAILED. 625 passed; 16 failed; 7 ignored; 0 measured; 0 filtered out; finished in 21.51s
error: test failed, to rerun pass `-p hawking-core --lib`
```

Integration tests were compiled, not executed, because cargo stop after
the first failing target. None of those tests were weakened or skipped
by this change. The sandbox has no Metal device.

---

## 6. Is a coherence verdict obtainable?

**Yes**, after the one-file catalog install. The vehicle is native packed
bytes, K-complete, no expand-to-Q4, no mlx-delta on f32v2.

**No verdict is claimed here.** This lane did not generate.

A generate that runs without the catalog install is the old Q4 vehicle
and must not be filed as a sub-1.5 result.

A generate whose load log does not contain `K-complete` and
`refused=0 expanded_to_q4=0` is not a codec verdict.

---

## 7. Exact command for the GPU lane

Work from this worktree. Use the binary this lane built
(`mtime 2026-08-17 11:54:25`, or rebuild it). Do not reuse a stale
pre-C3 `ascension_qwen38_hybrid_greedy`.

```bash
# 1. Install the catalog (unsandboxed). One 309294-byte file.
#    Segment filenames inside it are absolute paths to the packed blobs
#    on this machine, so segments/ is not required.
cp workspace/superwave/g1/mixed-sub15-native-catalog/catalog.hq38m20 \
  /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1/catalog.hq38m20

# Optional, cleaner: re-emit with hardlinks if the process can mkdir.
# python3 tools/qwen38_sub15_pack.py --phase catalog \
#   --root /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1 \
#   --mixed /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1

# 2. Confirm the switch before generating.
test -s /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1/catalog.hq38m20

# 3. Native greedy. Fusion stays ON. Do not set HAWKING_QWEN38_RECON_FUSE=0
#    unless you are deliberately measuring the serial walk.
#    Hold the GPU lock. Do not disturb the resident Genesis process
#    (it serves uniform-q4-v1, a different directory).
export CARGO_TARGET_DIR=workspace/ops/build/rust
./tools/gpu_lane_lock.sh qwen38-sub15-native \
  workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy \
  --artifact-root /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1 \
  --tokenizer /Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json \
  --prompt "Say hi." \
  --max-new-tokens 16 \
  --out receipts/ascent-2026-08-17/QWEN38_SUB15_NATIVE_GENERATE.json
```

Then the same six prompts used for mixed-2p0 native generate
(`QWEN38_NATIVE_MIXED_2P0_GENERATE.json`):

```
Say hi.
Write a function that reverses a string.
What is the capital of France?
Explain what a hash map is in one sentence.
def fibonacci(n):
The three primary colors are
```

Abort if stderr does not contain all three of:

1. `opening mixed HQ38M20 + 851 catalog tensors`
2. `refused=0 expanded_to_q4=0 expanded_to_float_gemv=0`
3. `mixed bind: K-complete`

`fallbacks_total` must be 0. Tokens under a missing catalog or a
partial-K bind are not a representation result.

---

## 8. What this lane did not do

- No `Qwen38HybridWeights::load` (opens Metal).
- No generate, no TOKEN_NS, no TPS.
- No repack, no BF16 touch, no `uniform-q4-v1` touch.
- No live Genesis restart.
- Did not install `catalog.hq38m20` onto mixed-sub15-v1 (sandbox write
  denied). The file is staged and the install command is above.

```
STATUS
IMPLEMENT_READY

CLAIMS
1. mixed-sub15-v1 is packed at 1.2910781930062503 BPW. Native HQ38M20 catalog is emitted (851 tensors, codecs 64/368/64/0/2/353). Evidence: NATIVE_CATALOG.json; census test; packer peek magics.
2. load_mixed accepts codec 4 f32v2 without mlx-delta and still refuses codec 5. Evidence: classify_qwen38_mixed_payload; codec_4 / unknown_codec_5 tests.
3. Fuse-on HGRAVB01/HGRAVR02 bind is K-complete for K in {5120,6144} via simd_bytes tiles. Non-256 remainder refuses. Evidence: dispatch_binary/dispatch_residual; k_complete_bind_retargets_wide_columns.
4. No expand-to-Q4 and no float GEMV on this path. Evidence: census expanded_*=0; load_mixed F32v2 arm; uniform load() unchanged vs HEAD.
5. A coherence verdict is obtainable after one-file catalog install + the greedy command in §7. Not obtained here.

EVIDENCE
workspace/superwave/g1/g1-sub15-native-gap.md (accepted plan)
tools/qwen38_sub15_pack.py --phase catalog (this process)
workspace/superwave/g1/mixed-sub15-native-catalog/NATIVE_CATALOG.json
cargo test mixed_catalog_contract (8 passed, including full 851-row census)
cargo build --release -p hawking-core → Finished release in 3m 00s
cargo test -p hawking-core --lib: 618→625 passed, same 16 pre-existing fails

CHANGES
crates/hawking-core/src/model/qwen38_hybrid_decode.rs
tools/qwen38_sub15_pack.py
workspace/superwave/g1/mixed-sub15-native-catalog/*
workspace/superwave/g1/g1-sub15-native-impl.md (this file)

TESTS
cargo build --release -p hawking-core
cargo test -p hawking-core --lib --offline
cargo test -p hawking-core --offline
test -s workspace/superwave/g1/g1-sub15-native-impl.md
git status --porcelain

RISKS
Catalog is not yet in mixed-sub15-v1/. A generate against that directory today still takes the Q4 vehicle. Absolute segment paths are machine-local. Host rice-index expand at load is unchanged (not a Q4 expand). This bind also retargets mixed-2p0 native GEMVs with cols>2048.

UNRESOLVED
Native coherence of the 1.291 BPW recipe. Catalog install onto the campaign artifact (one cp, needs write access). GPU generate.

NEXT
GPU lane: §7. Confirm the three load-log lines. Then greedy. Do not reopen generator+residual. Do not treat expand-to-Q4 INCOHERENT as the packed-codec result.
```
