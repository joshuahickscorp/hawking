# G1 MLP role-lock unlock

Lane: `91-mlp-rolelock-unlock`. Unlocks pack-declared HGRAVU01 Uniform on
MLP GEMVs so mixed-q3mlp-v1 and mixed-q4down-v1 admit. No generate. No
GPU load. No GPU benchmark. No repack. Resident Genesis not touched.

Epistemic tags: **MEASURED** = command output or on-disk integer from this
process. **SHOWN** = byte-identity of a function body vs HEAD. **NOT RUN**
= generate / Metal upload.

---

## 0. Verdict

IMPLEMENT_READY. The mixed MLP lock now accepts Uniform (HGRAVU01) on
gate / up / down when the pack declares it. The mixed-2p0 assignment
(gate Binary / up Residual / down Hgravs) still admits. The lock still
refuses:

- Residual on gate (**MEASURED** test `mixed_mlp_unsupported_role_still_refuses`)
- HQ30UQ4 on an MLP GEMV (**MEASURED** test `hq30uq4_on_mlp_is_not_uniform_and_still_refuses`)
- unknown codec 5 (**MEASURED** existing test `unknown_codec_5_still_refuses`, kept)

Both newly admissible artifacts pass the same CPU-side policy
`load_mixed` will run after upload. `Qwen38HybridWeights::load`
catalog-absent arm (the uniform-q4-v1 path) is byte-identical to HEAD.

A coherence verdict is now obtainable. It has not been obtained here.

---

## 1. What the lock was, and what it is

HEAD `assert_mixed_mlp_native` name-locked every layer:

| role | HEAD allowed | now allowed |
|---|---|---|
| `mlp.gate_proj.weight` | Binary only | Binary or Uniform |
| `mlp.up_proj.weight` | Residual only | Residual or Uniform |
| `mlp.down_proj.weight` | Hgravs only | Hgravs or Uniform |

Anything else, including missing (f32 / HQ30UQ4 landed in `q4` or `f32s`,
not `mixed`), still returns `missing … refusing silent dense/Q4 fallback`.
A packed kind on the wrong role still returns `is not HGRAV* or HGRAVU01;
refusing reconstructed MLP`.

Policy lives in `assert_mixed_mlp_native_kinds` (CPU). `load_mixed` maps
`MixedGpuWeight` through it. `assert_mixed_mlp_native_catalog` walks only
MLP rows of an HQ38M20 catalog (64-byte prefixes, no Metal, no full
payload, no rice expand) and applies the same function.

Not admitted (deliberate):

- HQ30UQ4 on MLP (`MixedCatalogLane::Hq30Uq4` → `None` → missing)
- codec 4 f32v2 on MLP
- HgravuVector dequant on MLP
- codec ≥ 5 (classify refuses before the role lock)

Q3 MLP does not need a new shader. `dispatch_uniform` → `dispatch_factor`
already binds `q80_hgravs01_factor_matvec_simd3` when `bits == 3`.
mixed-q4down down is HGRAVU01 **bits=4** and will hit the existing
`q80_hgravs01_factor_matvec_simd` else-branch. That is not a new shader
and is not a generate verdict.

---

## 2. On-disk assignment (**MEASURED**)

Campaign root:

```
/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b
```

Catalog parse + 8-byte magic peek, this process:

| artifact | tensors | MLP | L0 / all-64 assignment | complete_physical_bpw (PACK_REPORT) |
|---|---:|---:|---|---:|
| mixed-q3mlp-v1 | 851 | 192 | gate/up/down codec 3 `HGRAVU01` ×64 | 3.6138647373176767 |
| mixed-q4down-v1 | 851 | 192 | gate codec 0 `HGRAVB01` ×64, up codec 1 `HGRAVR02` ×64, down codec 3 `HGRAVU01` ×64 | 2.9590429283570026 |
| mixed-2p0-v1 | 851 | 192 | gate 0 `HGRAVB01`, up 1 `HGRAVR02`, down 2 `HGRAVS01` | 2.085538587276445 |

HEAD lock dies on q3mlp (gate not Binary) and q4down (down not Hgravs).
That is why those packs have never generated.

---

## 3. Admission result (**MEASURED**, CPU, no Metal)

Command:

```
CARGO_TARGET_DIR=workspace/ops/build/rust CARGO_TERM_COLOR=never \
  cargo test -p hawking-core --lib --offline mixed_catalog_contract \
  -- --nocapture --test-threads=1
```

```
running 14 tests
test ...::absolute_segment_filename_does_not_join_segments ... ok
test ...::catalog_roundtrip_codec_4_census_and_codec_5_refuses ... ok
test ...::codec_4_f32v2_is_accepted_without_mlx_delta ... ok
test ...::hq30uq4_on_mlp_is_not_uniform_and_still_refuses ... ok
test ...::hq38m20_magic_and_record_match_q80_layout ... ok
test ...::k_complete_bind_retargets_wide_columns ... ok
test ...::mixed_2p0_legacy_mlp_assignment_still_admits ... ok
test ...::mixed_mlp_legacy_binary_residual_hgravs_still_admits ... ok
test ...::mixed_mlp_uniform_is_admitted_on_every_role ... ok
test ...::mixed_mlp_unsupported_role_still_refuses ... ok
test ...::mixed_q3mlp_and_q4down_pass_mlp_admission ... ok
test ...::sub15_native_catalog_census_if_emitted ... ok
test ...::sub15_source_payloads_accept_mixed_gpu_layout ... ok
test ...::unknown_codec_5_still_refuses ... ok

test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 640 filtered out; finished in 2.48s
```

`mixed_q3mlp_and_q4down_pass_mlp_admission` called
`assert_mixed_mlp_native_catalog` on both real roots. Both returned `Ok`.
`mixed_2p0_legacy_mlp_assignment_still_admits` also `Ok`.

This is **admission**, not coherence, not a load of weights onto Metal,
not a generate. A GPU lane that sees `is not HGRAVB01` / `is not HGRAVS01`
after this patch has a different bug than the one this lane fixed.

---

## 4. Refusal still observed (**MEASURED**)

A gate never observed refusing is worth nothing. Three refuses fired in
this process:

1. `unknown_codec_5_still_refuses` — classify of codec 5 still contains
   `unknown mixed codec 5`. Untouched assertion.
2. `mixed_mlp_unsupported_role_still_refuses` — Residual on every
   `mlp.gate_proj.weight` returns `is not HGRAVB01`. Kernel for Residual
   exists; the lock still refuses the role mismatch.
3. `hq30uq4_on_mlp_is_not_uniform_and_still_refuses` — codec-3 payload
   with `HQ30UQ4\0` classifies as `Hq30Uq4`, `mixed_mlp_native_kind_from_lane`
   is `None`, lock returns `missing … mlp.down_proj.weight … silent
   dense/Q4 fallback`. Codec 3 is not a rubber stamp.

---

## 5. uniform-q4-v1 load arm is unchanged (**SHOWN**)

`uniform-q4-v1` listing, this process:

```
manifest.json
tensors
```

No `catalog.hq38m20`. `Qwen38HybridWeights::load` therefore takes the
catalog-absent branch (`load_qwen38_manifest` + q4/f32 upload). That
function body is byte-identical to HEAD.

```
HEAD load() 703-776 sha256=59bb24d78dc85e787bb862fc6d84111c4ee8e5bd46ba992e0b3a3d973ac61b52
WORK load() 823-896 sha256=59bb24d78dc85e787bb862fc6d84111c4ee8e5bd46ba992e0b3a3d973ac61b52
load() byte-identical to HEAD: True
catalog-absent arm identical: True
sha256 a62a11a2fdd4bb703a3947a747a0cb7189841b4da447c65f6a737517948c8607
```

Line numbers shifted because 120 lines of policy were inserted above the
device module. The 74-line function text is the same bytes. `git diff`
hunks in this file are only: module doc, new policy, `assert_mixed_mlp_native`
delegation, new tests.

G0 `uniform-q4-v1` artifact was not touched. No shader was touched.

---

## 6. Required commands

rustc 1.94.1 (e408947bf 2026-03-25) (Homebrew). cargo 1.94.1 (Homebrew).
`CARGO_TARGET_DIR=workspace/ops/build/rust`. `--offline`.

### 6.1 `cargo build --release -p hawking-core`

```
CARGO_TARGET_DIR=workspace/ops/build/rust CARGO_TERM_COLOR=never \
  cargo build --release -p hawking-core --offline
```

Tail (**MEASURED**):

```
warning: `hawking-core` (lib) generated 9 warnings (run `cargo fix --lib -p hawking-core` to apply 1 suggestion)
    Finished `release` profile [optimized] target(s) in 3m 13s
```

Exit 0. The 9 warnings are pre-existing (`ended_on_eog`, `groups`,
private `TensorLoc`, dead fields). None in `qwen38_hybrid_decode.rs`.

### 6.2 `cargo test -p hawking-core --lib --offline`

```
CARGO_TARGET_DIR=workspace/ops/build/rust CARGO_TERM_COLOR=never \
  cargo test -p hawking-core --lib --offline -- --test-threads=1
```

```
test result: FAILED. 631 passed; 16 failed; 7 ignored; 0 measured; 0 filtered out; finished in 81.66s
```

Baseline named in the contract: **625 passed, 16 failed**.
This run: **631 passed, 16 failed, 7 ignored**.

`631 - 625 = 6`. The six new tests in `mixed_catalog_contract_tests` all
passed. Failure count is unchanged.

Failed names (all outside this change; Metal unavailable in this sandbox
or sparse-checkout fixtures missing):

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

Typical panic: `Metal("no Metal-capable GPU")` or `No such file or directory`.
None mention MLP, HGRAVU01, or `qwen38_hybrid_decode`.

### 6.3 report exists

`test -s workspace/superwave/g1/g1-mlp-rolelock-unlock.md` — this file.

---

## 7. Exact GPU-lane generate commands (NOT RUN)

Vehicle is `ascension_qwen38_hybrid_greedy` (same binary as native 2p0 and
the G0 seal). Rebuild from **this** HEAD so the assert patch is in the
binary. Take `tools/gpu_lane_lock.sh`. Do **not** point the live resident
at these roots. Do **not** pass `--raw-prompt`. `--max-seq-len 512`.

```
CARGO_TARGET_DIR=workspace/ops/build/rust \
  cargo build --release -p hawking-core --example ascension_qwen38_hybrid_greedy

ART=/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b
BIN=workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy
TOK=$ART/bf16/tokenizer.json
PROMPTS=$ART/coherence_prompts.txt
```

Family C first (Q3 all-MLP, only pack that tests the doctor Q3-MLP hold):

```
./tools/gpu_lane_lock.sh qwen38-bracket-c \
  $BIN \
  --artifact-root $ART/mixed-q3mlp-v1 \
  --tokenizer $TOK \
  --prompts-file $PROMPTS \
  --max-new-tokens 16 \
  --max-seq-len 512 \
  --out receipts/ascent-2026-08-17/QWEN38_Q3MLP_GENERATE_16.json
```

Family B second (down HGRAVU01 q4, gate B01, up R02):

```
./tools/gpu_lane_lock.sh qwen38-bracket-b \
  $BIN \
  --artifact-root $ART/mixed-q4down-v1 \
  --tokenizer $TOK \
  --prompts-file $PROMPTS \
  --max-new-tokens 16 \
  --max-seq-len 512 \
  --out receipts/ascent-2026-08-17/QWEN38_Q4DOWN_GENERATE_16.json
```

Load log **must** contain `opening mixed HQ38M20` and **must not** contain
`opening Metal + 755 catalog tensors`. If the JSON is not written and
stderr says `is not HGRAVB01` / `is not HGRAVS01`, that is LOAD_REFUSE,
not INCOHERENT.

`fallbacks_total != 0` or `DENSE_W_MATERIALIZED != 0` makes the run VOID.

Phase A is the 16-token 6-prompt collapse screen. Phase B (France 128,
arith 256) only if Phase A is not punctuation-only collapse. See
`g1-bracket-bisection.md` §4.

---

## 8. Diff

```
 crates/hawking-core/src/model/qwen38_hybrid_decode.rs | 310 +++++++++++++----
 1 file changed, 262 insertions(+), 48 deletions(-)
```

Hunks: module doc; `MixedMlpNativeKind` + `assert_mixed_mlp_native_kinds`
+ `assert_mixed_mlp_native_catalog`; `assert_mixed_mlp_native` delegates;
six new tests. No shader. No manifest. No lockfile. No live G0 artifact.

---

## 9. What this lane did not do

- No generate. No Metal upload of either pack (would contend with the
  resident 13.6 GB body).
- No repack.
- No change to `dispatch_uniform` / `dispatch_factor`.
- No change to uniform-q4-v1 or any campaign artifact.
- No commit, push, merge, deploy.

A successful admission is not coherence. mixed-floor-q7-v1 already
generated natively at 3.1768 BPW with the same 0.848 MLP and collapsed.
These two packs are the first that vary MLP. Their tokens do not exist yet.
