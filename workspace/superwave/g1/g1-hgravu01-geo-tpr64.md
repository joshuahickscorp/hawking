# G1 HGRAVU01 geo_tpr64 — kernel, parity, generate

Lane: `110-hgravu01-geo-tpr64`. Apple M3 Ultra, 96 GB. Resident Genesis
left running on `uniform-q4-v1`. GPU software lock was held by
`genesis-resident:child_b` (pid 50196) for the whole session; this lane
did not take the lock and did not stop Genesis.

Every number is **MEASURED**, **RECEIPT**, **SOURCE**, **DERIVED**, or
**INFORMAL**. INFORMAL generate GPU times are not a complete-token wall.

## 0. Verdict

The kernel was built and wired for Uniform bits 3 and 4. It consumes the
existing HGRAVU01 packed bytes. No repack.

Numerical parity against the incumbent simd3 / simd kernels is
**bit-identical** on every tested organ except `lm_head`. On `lm_head`
(248320×5120, bits=4) the incumbent is **wrong**: `gk_uniform_extract_wide`
does `bit0 = element * bits` in `uint32`, which wraps at
`element >= 2^30`. For K=5120 that is row **209715**. The new geo kernel
addresses by `(row, group)` and matches the CPU serial oracle on that
tail; simd does not.

Correcting the lm_head tail changes greedy. France 128 on the new kernel
emits `<think>` then `<|im_end|>` (no Paris). The same prompt on the
lane-91 incumbent binary emits the Paris loop. Fallbacks 0, dense W 0
on both.

This is **not** a G1 promotion candidate against G0 at 4.252735126866492
BPW / 39,326,090 ns. The campaign coherence gate fails under the
corrected kernel. Official paired complete-token wall was **not** run:
the contract says stop at parity if incumbent parity fails.

Informal generate GPU (DIRTY, Genesis resident, not a wall): new kernel
median ~46–48 ms/token vs incumbent simd ~163 ms/token on the same
artifact and prompt. Faster and more correct, and it breaks the
substring gate that was measured on the overflowing lm_head.

Live G0 manifest sha256 stayed
`d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df`
before, after France, and after the incumbent control generate.

## 1. What shipped

### 1.1 Kernels (`q80_mixed_decode.metal`)

- `qwen_uniform_q3_group64_matvec_geo_tpr64_tg128`
- `qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128`

G0 mapping: TG 128, 4 simdgroups, 2 rows/TG, 64 threads/row,
`col = lane_in_row * 8`, stride 512, `simd_sum` then 2-way TG add.

bits=3: 24 code bytes/group, 8-wide LSB unpack, `q = code - bound`
(bound=3), one scale per tile.

bits=4: 32 code bytes/group, even nibble low, `q = nibble - bound`
(bound=7). Not the G0 `nibble-8` kernel.

`groups_per_row = cols/64` is derived. No dense W.

### 1.2 Dispatch (`qwen38_hybrid_decode.rs`)

`dispatch_uniform` only. HGRAVS r160 stays on `dispatch_factor` (simd3).

```
bits ∈ {3,4} && group_size==64 && cols%64==0 && recon_fuse ON
  → geo_tpr64, grid = ceil(rows/2)*128, TG 128
else
  → existing simd / simd3 / uniform8 / serial
```

`HAWKING_QWEN38_RECON_FUSE=0` still uses the serial factor kernel.

## 2. Parity (MEASURED, before any official wall)

Artifact: `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-q3mlp-v1`

Input: `x[i] = (i % 17) * 0.125 - 1.0`. Incumbent and geo in one CB.
CPU serial = `uniform_factor_value` left-to-right.

### 2.1 Table

| tensor | bits | shape | incumbent | max_abs vs inc | max_rel | rms | n>\|d\|>1e-2 |
|---|---:|---|---|---:|---:|---:|---:|
| L0 gate | 3 | 17408×5120 | simd3 | 0 | 0 | 0 | 0 |
| L0 up | 3 | 17408×5120 | simd3 | 0 | 0 | 0 | 0 |
| L0 down | 3 | 5120×17408 | simd3 | 0 | 0 | 0 | 0 |
| L0 in_proj_a | 4 | 48×5120 | simd | 0 | 0 | 0 | 0 |
| L0 in_proj_b | 4 | 48×5120 | simd | 0 | 0 | 0 | 0 |
| L0 in_proj_z | 4 | 6144×5120 | simd | 0 | 0 | 0 | 0 |
| L0 in_proj_qkv | 4 | 10240×5120 | simd | 0 | 0 | 0 | 0 |
| L0 out_proj | 4 | 5120×6144 | simd | 0 | 0 | 0 | 0 |
| L3 q_proj | 4 | 12288×5120 | simd | 0 | 0 | 0 | 0 |
| L3 k_proj | 4 | 1024×5120 | simd | 0 | 0 | 0 | 0 |
| L3 v_proj | 4 | 1024×5120 | simd | 0 | 0 | 0 | 0 |
| L3 o_proj | 4 | 5120×6144 | simd | 0 | 0 | 0 | 0 |
| **lm_head** | **4** | **248320×5120** | **simd** | **4.02855492** | **3.47818470** | **0.370043513** | **38328** |

Worst vs incumbent: `language_model.lm_head.weight`, max_abs=4.02855492.

L0 in_proj_a/b vs CPU serial: max_abs=0.

Outputs were live (nonzero). Example L0 gate:
`inc[0..4]=geo[0..4]=[0.10128689, -0.3249321, 0.6938896, -0.17584991]`.

### 2.2 lm_head tail (MEASURED)

```
PARITY_BAD lm_head first=209715 last=248319 even=19148 odd=19180
n=248320  n>1e-4=38602  n>1e-2=38328
inc_nz=248319/248320  geo_nz=248320/248320
```

Row 209714 (last good): cpu = inc = geo = `4.77880716e-1`.

| row | cpu | inc (simd) | geo | d_inc | d_geo |
|---:|---:|---:|---:|---:|---:|
| 209715 | -8.89636278e-1 | -8.10261488e-1 | -8.89636278e-1 | 7.937e-2 | 0 |
| 209716 | -5.90004683e-1 | -6.64217234e-1 | -5.90004683e-1 | 7.421e-2 | 0 |
| 209717 | -1.27641082e0 | -1.07775521e0 | -1.27641082e0 | 1.987e-1 | 0 |

DERIVED overflow: `element * 4` wraps in uint32 at `element >= 2^30 = 1073741824`.
`floor(2^30 / 5120) = 209715`. Matches first bad row.

geo does not use `element * bits`. It is the correct kernel. The
incumbent is the silent corruption on the high 38605 vocab rows
(15.5 % of lm_head).

## 3. Generate / coherence (MEASURED)

Binary (this lane, just built):

```
workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy
sha256 8c05088c12e6cefa2a96763e1a1e3672cd77a235f6f405730aa631f3e1c96f0d
mtime  2026-08-17 13:35:25
age at first France run: 31 s
strings contain both geo kernel names
```

Census on open (both binaries):

```
tensors=851 binary=0 residual=0 hgravs=0 uniform=498 q4=0 f32=353
refused=0 expanded_to_q4=0 expanded_to_float_gemv=0
recon_fuse=ON
```

### 3.1 New kernel — France, max-new=128 (then confirmed at 16)

Prompt (chat-templated, prompt_len=15): `What is the capital of France?`

```
GENERATED_TEXT_VERBATIM: <think>
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
NEW_TOKENS: [248068, 248046]
```

248046 is `QWEN38_EOS_IM_END`. Gate token `paris`: **absent**.
Repeat at max-new=16: same two ids. Deterministic.

Informal GPU (DIRTY, not a wall):

```
128-run median_gpu_ns_per_token = 46346625
16-run  median_gpu_ns_per_token = 47966541
GPU_NS_PER_STEP 128-run:
[76044333, 46346625, 46080999, 46106249, 45774375, 46155583,
 47226166, 46656749, 46750499, 46922291, 46395708, 46404291,
 46275083, 45721249, 46232416, 46271916]
```

16 GPU samples because generate stopped at EOS after the first new token;
the printed vector includes prefill steps.

### 3.2 Incumbent control — same prompt, lane-91 binary

```
/Users/scammermike/.claude-grok/worktrees/91-mlp-rolelock-unlock-20260817-120135/workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy
sha256 a9d41d09856ff7ffb7e32ff4fd4f7ad49cafc4301afd4f6814f51f8418898fab
no geo q3/q4 kernel names
```

```
GENERATED_TEXT_VERBATIM:
Assistant
Assistant
Assistant
Assistant
Assistant
Assistant
Assistant
Assistant
Assistant
Answer: Paris. Answer: Paris. Answer:
Answer: Answer: Answer:
Answer: Paris. Answer: Paris. Answer: What is the capital of
What is the capital of France? Answer: Paris. Answer: Paris. Answer: Paris. Answer: Paris. Answer: What is the

What is the
What is the capital of France? Answer: multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple multiple many many many many many many
What is the capital of France? What is
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
NEW_TOKENS: [69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198, 69267, 198, 15666, 25, 11751, 13, 21134, 25, 11751, 13, 21134, 25, 198, 15666, 25, 21134, 25, 21134, 25, 198, 15666, 25, 11751, 13, 21134, 25, 11751, 13, 21134, 25, 3437, 369, 279, 6511, 314, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 25, 11751, 13, 21134, 25, 11751, 13, 21134, 25, 11751, 13, 21134, 25, 11751, 13, 21134, 25, 3437, 369, 279, 271, 3710, 369, 279, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 25, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 5081, 1599, 1599, 1599, 1599, 1599, 1599, 198, 3710, 369, 279, 6511, 314, 9338, 30, 3437, 369]
MEDIAN_GPU_NS_PER_TOKEN: 162710166
```

`paris` present (×8 in the RECEIPT lane-92 sense). This is the genome the
campaign called COHERENT. It is the overflowing lm_head.

### 3.3 17×19

Not run. France already fails the campaign AND of the two-prompt gate.
Contract: a kernel change can break coherence; it did. Further generate
on the corrected head is not a substitute for the official wall.

## 4. Complete-token wall

**Not measured.** Incumbent parity failed on lm_head. The contract says
stop. No paired 3×A/B spread vs G0 on this binary.

Informal generate GPU above is DIRTY_ENGINEERING (Genesis RSS ~20.3 %,
lock owner `genesis-resident:child_b`). It is not 31 TPS and it is not
the 32 ms PROJECTED addressing figure.

PROJECTED 32 ms / 31 TPS from the prior lane remains a projection.

## 5. G0 manifest

```
before start     d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
after France     d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
after control    d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df
```

Genesis pid 50196 stayed up.

## 6. Promotion

No. mixed-q3mlp-v1 at 3.6138111608720234 BPW is only campaign-COHERENT
on the overflowing simd lm_head. geo_tpr64 is the G0 kernel class and is
faster on an informal generate, but the substring gate fails once lm_head
is computed correctly. A promotion candidate needs both the gate and a
paired complete-token wall. Neither is met.

Matching the incumbent overflow on purpose would be the silent
corruption this repo has already been burned by.

## 7. Build and tests

```
cargo build --release -p hawking-core --example ascension_qwen38_hybrid_greedy --offline
Finished `release` profile [optimized] target(s) in 2m 32s

cargo build --release -p hawking-core --offline
Finished `release` profile [optimized] target(s) in 2m 18s
```

CARGO_TARGET_DIR = this worktree `workspace/ops/build/rust` (not the
Genesis tree).

```
cargo test -p hawking-core --lib --offline
test result: FAILED. 646 passed; 3 failed; 7 ignored; 0 filtered out
```

New tests (both ok):

```
hgravu01_geo_tpr64_bind_is_bits_3_and_4_only
hgravu01_geo_tpr64_matches_incumbent_on_real_tensors
```

The 3 failures are pre-existing / sparse-checkout, not this change:

```
model::dsv4f_activation_capture::tests::sealed_organ_catalog_matches_schedule_receipt
  sealed schedule receipt: NotFound
model::qwen80_device_expert_table::tests::device_expert_table_abi_matches_metal_static_asserts
  left: 6 right: 5
profile::tests::pinned_profiles_still_load_after_field_additions
  qwen3b-instruct-q4k.m3pro18.json: NotFound
```

Baseline cited 631 passed / 16 failed. This tree is a sparse checkout;
file-not-found failures collapse. Pass count rose by our 2 tests plus
environment. None of the 3 remaining failures touch the two edited
source files.

`test -s workspace/superwave/g1/g1-hgravu01-geo-tpr64.md` PASS.

## 8. Next

If a later lane wants the speed and the correct math, the coherence
gate has to be re-established on the **correct** lm_head (new artifact,
or accept that this pack’s “Paris” was an overflow artifact). Do not
reintroduce `element * bits` in uint32. If a later lane wants the old
tokens, keep simd for lm_head only and do not call that promotion.
