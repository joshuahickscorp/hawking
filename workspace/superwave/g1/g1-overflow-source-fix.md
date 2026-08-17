# G1 overflow source fix — extract, parity, mixed-floor-q7-v1 verdict

Lane: `123-overflow-source-fix`. Apple M3 Ultra, 96 GB. Resident Genesis
left running. Software GPU lock was held by `genesis-resident:parent`
(pid 50196) for the whole generate session; this lane did not take the
lock and did not stop Genesis.

Every number is **MEASURED**, **SOURCE**, or **DERIVED**. Informal
generate GPU times with the resident live are **DIRTY** and are not a
complete-token wall.

## 0. Verdict

Source extract `bit0 = element * bits` in `uint32` is gone from
`gk_uniform_extract` / `gk_uniform_extract_wide`. Replacement is a
32-bit decomposition that cannot wrap for bits 1..=8 and any `uint`
element. Sibling `* 3u` sites in `q80_mixed_decode.metal` use the same
helper.

Parity against the CPU `usize` oracle holds above the wrap on embed and
lm_head at 248320×5120, at every HGRAVU01 bit width those tensors use
on disk (4, 7, 8). The new regression test failed on the unfixed kernel
(222 overflow-tail mismatches) and passed on the fixed one.

`mixed-floor-q7-v1` at 3.1767685514394888 is **INCOHERENT** under
correct math. Failure signature: **early stop** — first new token is
`QWEN38_EOS_IM_END` (248046) on France@16, France@128, and 17×19@256.
Fallbacks 0, dense W 0. A G0 control on the same binary emits Paris
and matches `QWEN38_COHERENCE_SEAL` style. No complete-token wall:
the gate failed.

Located generate floor remains G0 at 4.252735126866492.

Live G0 manifest sha256 stayed
`d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df`.

## 1. The wrap (DERIVED, then MEASURED)

`element * bits` in `uint32` wraps at `element >= ceil(2^32 / bits)`.

```
WRAP_TABLE header: bits wrap_el first_row lm_head_elements reaches
WRAP bits=3 wrap_el=1431655765 first_row=279620 elements=1271398400 reaches=false
WRAP bits=4 wrap_el=1073741824 first_row=209715 elements=1271398400 reaches=true
WRAP bits=5 wrap_el=858993459  first_row=167772 elements=1271398400 reaches=true
WRAP bits=6 wrap_el=715827882  first_row=139810 elements=1271398400 reaches=true
WRAP bits=7 wrap_el=613566756  first_row=119837 elements=1271398400 reaches=true
WRAP bits=8 wrap_el=536870912  first_row=104857 elements=1271398400 reaches=true
```

On-disk HGRAVU01 bits on embed/lm_head: 4 (mixed-q3mlp-v1 / 2p0 / q4down),
7 (mixed-floor-q7-v1), 8 (mixed-floor-q8-v1 / q8-up10). bits=3 is used
on smaller MLP tensors only; it does not reach wrap on this model.
bits=5 and 6 are not on disk. mixed-sub15 embed/lm_head are HQ30UQ4,
not HGRAVU01.

Tokenizer-added tokens 248044–248076 all sit above every wrap row, so
the corrupted region is exactly the stop/control block.

## 2. The fix (SOURCE)

`crates/hawking-core/shaders/gk_family.metal` (was lines 184 and 213):

```
// byte0 = (element * bits) >> 3, shift = (element * bits) & 7
// without forming the overflowing product.
byte0 = (element >> 3) * bits + ((element & 7) * bits) >> 3
shift = ((element & 7) * bits) & 7
```

Cost vs the wrapping mul, innermost loop of `gk_uniform_extract_wide`:
**+1 32-bit mul, +1 add, +1 and, +1 shr. No extra 64-bit registers.**
The subsequent `codes[byte0]` load dominates. Serial extract walks
bytes from that (byte, shift) instead of `bit0 + b`.

`q80_mixed_decode.metal` production simd3 and its two probe kernels
now call `gk_packed_lsb_byte(row_base + col, 3u)` instead of
`((row_base + col) * 3u) >> 3u`.

Not chosen: widening geo_tpr64 bind. That only covers bits 3/4 and
would leave 5–8 on the overflowing incumbent. Bind test
`hgravu01_geo_tpr64_bind_is_bits_3_and_4_only` is unchanged.

Sibling sites **not** in permitted scope, still wrapping:

- `crates/hawking-core/shaders/qwen_uniform_qn.metal:20` (lane-N bits 2/3)
- `crates/hawking-core/shaders/q80_codec_activity.metal:12` (private LUT
  lane; not in `all_shader_sources`)

## 3. G0 path unchanged (MEASURED)

`crates/hawking-core/shaders/qwen_uniform_q4.metal` git hash equals HEAD:

```
eb2d08683c30b45814a0c8d70d4058cfcaa8a203
```

`git diff -- crates/hawking-core/shaders/qwen_uniform_q4.metal` empty.
Kernel still uses nibble-8 group addressing (`int(nibble) - 8`), not
`element * bits`. `QWEN38_Q4_MATVEC_KERNEL` still
`qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`. Test
`g0_uniform_q4_geo_tpr64_source_is_unchanged` passed. Release binary
strings contain that kernel name (count 5) and `gk_packed_lsb_byte`.

G0 France@128 on this binary (below) emits Paris. Control holds.

## 4. Parity above threshold (MEASURED)

Kernel under test: incumbent `q80_hgravs01_factor_matvec_simd` (the
extract_wide path). Never geo_tpr64, never uniform8. CPU oracle =
`uniform_factor_value` (`usize` multiply). Input `x[i]=(i%17)*0.125-1`.
Shapes all 248320×5120. Probe: last-good row, first wrap row, stop
tokens 248044–248076, last two rows.

### 4.1 Unfixed kernel — test FAILED in 5.88 s, 222 ABOVE-wrap mismatches

Below wrap matches. Above wrap does not.

| artifact | tensor | bits | row | cpu | gpu (unfixed) | abs_d |
|---|---|---:|---:|---:|---:|---:|
| mixed-q3mlp-v1 | embed | 4 | 209714 | -3.86738777e-2 | -3.86738777e-2 | 0 |
| mixed-q3mlp-v1 | embed | 4 | 209715 | 4.98867273e-1 | 8.32414150e-1 | 3.335e-1 |
| mixed-q3mlp-v1 | lm_head | 4 | 209714 | 4.77880716e-1 | 4.77880716e-1 | 0 |
| mixed-q3mlp-v1 | lm_head | 4 | 209715 | -8.89636278e-1 | -8.10261488e-1 | 7.937e-2 |
| mixed-q3mlp-v1 | lm_head | 4 | 248046 | 1.03333211e0 | 6.78576946e-1 | 3.548e-1 |
| mixed-floor-q7-v1 | embed | 7 | 119836 | -1.02193058e0 | -1.02193034e0 | 2.384e-7 |
| mixed-floor-q7-v1 | embed | 7 | 119837 | -1.58941299e-1 | 1.54443657e0 | 1.703e0 |
| mixed-floor-q7-v1 | lm_head | 7 | 119837 | 1.33218110e-1 | -7.60768652e-1 | 8.940e-1 |
| mixed-floor-q7-v1 | lm_head | 7 | 248046 | 1.09057891e0 | -6.71345592e-1 | 1.762e0 |
| mixed-floor-q7-v1 | lm_head | 7 | 248068 | -1.09376669e-1 | 1.71160173e0 | 1.821e0 |
| mixed-floor-q8-v1 | embed | 8 | 104856 | -6.91564500e-1 | -6.91565037e-1 | 5.364e-7 |
| mixed-floor-q8-v1 | embed | 8 | 104857 | -1.87708870e-1 | -9.60347056e-2 | 9.167e-2 |
| mixed-floor-q8-v1 | lm_head | 8 | 104857 | -3.44940513e-1 | -2.66376495e-1 | 7.856e-2 |
| mixed-floor-q8-v1 | lm_head | 8 | 248046 | 1.09119511e0 | 5.15374243e-1 | 5.758e-1 |

```
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 658 filtered out; finished in 5.88s
failures:
    model::qwen38_hybrid_decode::mixed_catalog_contract_tests::incumbent_extract_matches_cpu_oracle_above_uint32_overflow
```

GPU lock during fail run: `genesis-resident:child_a` pid 50196.

### 4.2 Fixed kernel — test PASSED in 5.64 s

Same probes. abs_d ≤ 2.265e-6 (simd association vs serial CPU).

| artifact | tensor | bits | row | cpu | gpu (fixed) | abs_d |
|---|---|---:|---:|---:|---:|---:|
| mixed-q3mlp-v1 | embed | 4 | 209715 | 4.98867273e-1 | 4.98867273e-1 | 0 |
| mixed-q3mlp-v1 | lm_head | 4 | 209715 | -8.89636278e-1 | -8.89636278e-1 | 0 |
| mixed-q3mlp-v1 | lm_head | 4 | 248046 | 1.03333211e0 | 1.03333211e0 | 0 |
| mixed-floor-q7-v1 | embed | 7 | 119837 | -1.58941299e-1 | -1.58941269e-1 | 2.980e-8 |
| mixed-floor-q7-v1 | lm_head | 7 | 119837 | 1.33218110e-1 | 1.33218110e-1 | 0 |
| mixed-floor-q7-v1 | lm_head | 7 | 248046 | 1.09057891e0 | 1.09058118e0 | 2.265e-6 |
| mixed-floor-q7-v1 | lm_head | 7 | 248068 | -1.09376669e-1 | -1.09376669e-1 | 0 |
| mixed-floor-q8-v1 | embed | 8 | 104857 | -1.87708870e-1 | -1.87708855e-1 | 1.490e-8 |
| mixed-floor-q8-v1 | lm_head | 8 | 104857 | -3.44940513e-1 | -3.44940603e-1 | 8.941e-8 |
| mixed-floor-q8-v1 | lm_head | 8 | 248046 | 1.09119511e0 | 1.09119487e0 | 2.384e-7 |

```
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 658 filtered out; finished in 5.64s
```

GPU lock during pass run: `genesis-resident:child_a` pid 50196.

`hgravu01_geo_tpr64_matches_incumbent_on_real_tensors` also passed after
the lm_head block was inverted from “expect incumbent miss” (that
encoded the bug) to “expect incumbent == geo == CPU”. The test was
not deleted. The bar is stricter.

## 5. Generate — mixed-floor-q7-v1 (MEASURED, DIRTY wall)

Binary (this lane, just built):

```
workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy
sha256 ab7738077a4c30891f97411b414f915c8e38f3664c5007c97c9a2eddc5a1c257
mtime  2026-08-17 14:40
```

Lock at generate start 2026-08-17T14:42:10-0400:

```
owner=genesis-resident:parent
pid=50196
```

Lock at generate end 2026-08-17T14:42:42-0400: same owner/pid. Lane did
not call `gpu_lane_lock.sh` (resident holds it). Token IDs are still
authoritative: fallbacks 0, dense W 0. GPU ns are DIRTY.

Census on open (q7, all three prompts):

```
tensors=851 binary=64 residual=64 hgravs=64 uniform=306 q4=0 f32=353
refused=0 expanded_to_q4=0 expanded_to_float_gemv=0
recon_fuse=ON
```

Machine before generate: pages free 8220, swapins 8848588, swapouts
15803134. After: pages free 976446, swapins unchanged.

### 5.1 France, max-new=16 (probe)

Prompt (chat-templated, prompt_len=15): `What is the capital of France?`

```
GENERATED_TEXT_VERBATIM:
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 15
NEW_TOKENS: [248046]
WALL_NS: 2069478208
RUN_WALL_S q7_france16 9.062
```

248046 is `QWEN38_EOS_IM_END`. Empty decode. 16-token probe is
INCOHERENT and cannot establish COHERENT.

### 5.2 France, max-new=128 (gate)

```
GENERATED_TEXT_VERBATIM:
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 15
NEW_TOKENS: [248046]
WALL_NS: 2032709459
RUN_WALL_S q7_france128 7.035
```

Paris: **absent**. Gate fail.

### 5.3 17 times 19, max-new=256 (gate)

```
GENERATED_TEXT_VERBATIM:
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 18
NEW_TOKENS: [248046]
WALL_NS: 2417347667
RUN_WALL_S q7_arith256 7.454
```

323: **absent**. Gate fail.

### 5.4 Failure signature

**early stop**. First new token is EOS. Not a degenerate cycle, not
punctuation collapse, not fluent-but-wrong. The overflowing kernel’s
16-token probe on this artifact had been `))))))))))))))))` (stop
tokens suppressed). Correct math restores halt, and the model halts
immediately.

### 5.5 G0 control, same binary, France 128

Artifact `uniform-q4-v1`. Manifest sha256
`d650a757c4cffed463ce8c24dfd5052c2cb47c0f6b1eb10349947854fc47b9df`.

```
GENERATED_TEXT_VERBATIM: <think>
The user is asking a simple factual question: What is the capital of France? The answer is Paris.
</think>

The capital of France is **Paris**.
FALLBACKS: 0
DENSE_W_MATERIALIZED: 0
PROMPT_LEN: 15
NEW_TOKENS: [248068, 198, 760, 1156, 369, 9859, 264, 4145, 57879, 3296, 25, 3437, 369, 279, 6511, 314, 9338, 30, 561, 4087, 369, 11751, 13, 198, 248069, 271, 760, 6511, 314, 9338, 369, 2972, 57590, 159034, 248046]
MEDIAN_GPU_NS_PER_TOKEN: Some(46801000)
STEADY_DECODE_WALL_NS_PER_TOKEN: Some(48358474)
WALL_NS: 2515829417
RUN_WALL_S g0_france128 7.753
```

Paris present. Seal-style think then answer. Binary is a valid G0
control. GPU median 46,801,000 ns is DIRTY (resident live), not the
official 39,326,090 ns TOKEN_NS wall.

No paired complete-token wall: q7 did not clear the gate.

## 6. Required commands

### `cargo build --release -p hawking-core --offline`

```
   Compiling hawking-core v0.2.2 (...)
warning: `hawking-core` (lib) generated 9 warnings
    Finished `release` profile [optimized] target(s) in 2m 25s
BUILD_EXIT:0
```

(example build earlier: `Finished release profile [optimized] target(s) in 2m 21s`)

### `cargo test -p hawking-core --lib --offline`

```
test result: FAILED. 649 passed; 3 failed; 7 ignored; 0 measured; 0 filtered out; finished in 82.09s
```

Baseline was 646 passed / 3 failed. +3 new tests passed → 649 / 3.
Same three pre-existing failures (sparse-checkout / ABI, not this lane):

```
model::dsv4f_activation_capture::tests::sealed_organ_catalog_matches_schedule_receipt
model::qwen80_device_expert_table::tests::device_expert_table_abi_matches_metal_static_asserts
profile::tests::pinned_profiles_still_load_after_field_additions
```

`hgravu01_geo_tpr64_matches_incumbent_on_real_tensors` ok.
`incumbent_extract_matches_cpu_oracle_above_uint32_overflow` ok.
`g0_uniform_q4_geo_tpr64_source_is_unchanged` ok.

### `test -s workspace/superwave/g1/g1-overflow-source-fix.md`

this file.

### `git status --porcelain` (after this file exists)

see completion report.

A `backend::honest_roof` test rewrote
`receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.reduced.json`
during the lib suite. Reverted immediately. Receipts not in scope.

## 7. What this does to the floor

Previous measured generate under correct math:

| artifact | BPW | verdict |
|---|---:|---|
| mixed-sub15-v1 | 1.2910781930062503 | INCOHERENT (HQ30UQ4, overflow-immune) |
| mixed-2p0-v1 | 2.0855385872764454 | INCOHERENT, EOS first new token |
| mixed-q4down-v1 | 2.9589935339460913 | INCOHERENT, EOS first new token |
| mixed-floor-q7-v1 | 3.1767685514394888 | **INCOHERENT, EOS first new token** (this lane) |
| mixed-q3mlp-v1 | 3.6138111608720234 | INCOHERENT, think open then EOS |
| uniform-q4-v1 G0 | 4.252735126866492 | COHERENT |

Nothing on disk below G0 is coherent. The previous UNRESOLVED at q7
was the overflowing incumbent hiding an immediate EOS.
