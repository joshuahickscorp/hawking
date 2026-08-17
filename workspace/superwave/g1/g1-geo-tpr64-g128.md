# G1 geo_tpr64 group 128 — HQ30UQ4 parser + sibling kernel

Lane: `127-geo-tpr64-g128`. Apple M3 Ultra, 96 GB. Resident Genesis left running.
This profile cannot open a Metal device (`Device::system_default()` is None;
`MTLCreateSystemDefaultDevice` prints `no device`; torch MPS is False). GPU
dispatch of the new kernel was not executed. Addressing, parse, and bind were.

Every number is **MEASURED**, **SOURCE**, or **DERIVED**.

## 0. Verdict

**Sibling, not a generalized inner loop.** A runtime `group_size` in
`qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` would put a non-constant
divide on the G0 path and change that kernel's compiled body. The group-64
function is byte-identical to HEAD:

```
SOURCE G64_KERNEL_SHA256 e1ddcd18bcd67b50bb77c1ee350443b91a7287a629e09013ee1e8ae2a6fa229e
SOURCE G64_KERNEL_BYTES 1747
SOURCE IDENTICAL_TO_HEAD true
```

HQ30UQ4 parse now admits group sizes **exactly {64, 128}**. Unsupported sizes
still refuse. Sealed G0 admission (`validate_q4_tensor_row` /
representation family) is still group 64 only.

A group-128 HQ30UQ4 GEMV with `cols % 128 == 0` now binds
`qwen_uniform_q4_group128_matvec_geo_tpr64_tg128` (TG 128, 4 simdgroups,
2 rows/TG, 64 threads/row — same launch class as G0). New indexing is
`(row, group)` in `ulong`. It does not form `element * bits`.

An existing HGRAVU01 bits=4 group=128 mixed payload still takes the slow
simd extract path. `q80_mixed_decode.metal` is out of scope and still
hardcodes `group_size == 64`. `qwen38_hgravu01_geo_tpr64_launch(4, 128, …)`
is still None. That is honest, not a silent bind onto the nibble-8 sibling
(`q = nibble - 8` vs HGRAVU01 `q = nibble - bound`).

## 1. Why sibling

| option | G0 kernel body | G0 inner loop | overflow |
|---|---|---|---|
| generalize to runtime group_size | changes | non-constant `/` | easy to get wrong |
| function constant | changes pipeline | maybe folded | bind change |
| **sibling compile-time 128** | **untouched** | **unchanged** | new path uses `ulong` |

G0 is the only coherent artifact. Costing it is worse than no change.

## 2. Bind (MEASURED)

HQ30UQ4 supported set is `{64, 128}` with `cols % group_size == 0`.

```
test hgravu01_geo_tpr64_bind_is_bits_3_and_4_only ... ok
test parse_accepts_group_64_and_128_and_refuses_the_rest ... ok
```

SOURCE bind table:

| path | bits | group | rows×cols | result |
|---|---:|---:|---|---|
| HGRAVU01 | 3 | 64 | 17408×5120 | `qwen_uniform_q3_group64_matvec_geo_tpr64_tg128` |
| HGRAVU01 | 4 | 64 | 48×5120 | `qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128` |
| HGRAVU01 | 8 | 64 | 5120×5120 | None |
| HGRAVU01 | 3 | 64 | 2048×160 | None |
| HGRAVU01 | 3 | 128 | 17408×5120 | None |
| HGRAVU01 | 4 | 128 | 248320×5120 | None |
| HGRAVU01 | 4 | 32 | 5120×5120 | None |
| HQ30UQ4 | — | 64 | 248320×5120 | `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` |
| HQ30UQ4 | — | 128 | 248320×5120 | `qwen_uniform_q4_group128_matvec_geo_tpr64_tg128` |
| HQ30UQ4 | — | 0, 32, 96, 256, 512 | 5120×5120 | None |
| HQ30UQ4 | — | 128 | 5120×160 | None |
| HQ30UQ4 | — | 64 | 5120×160 | None |

Parser refuse message for unsupported sizes is
`uniform Q4 group_size={n} must be 64 or 128` (MEASURED on 0, 32, 96, 256, 512).

## 3. G0 group-64 bit-identity (SOURCE + CPU)

The production group-64 kernel function is byte-identical to HEAD (SHA above).
`QWEN38_Q4_MATVEC_KERNEL` is still
`qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`. G0 dispatch still computes
`groups_per_row = cols.div_ceil(64)` and launches that name. Embed now passes
`weight.group_size`; G0 headers store 64, so the bound value is unchanged.

GPU geo-vs-serial-vs-CPU on a packed 32×256 and on live G0 lm_head
(248320×5120, wrap row 209715) is written
(`hq30uq4_group64_geo_matches_serial_and_cpu`,
`g0_lm_head_geo_matches_cpu_above_uint32_wrap`). Both skipped here:

```
skip: MetalContext::new failed: metal: no Metal-capable GPU
```

Those tests need the `gate` profile.

## 4. Group-128 oracle above the wrap (MEASURED, CPU walk)

Historical wrap: `element * 4` in uint32, first row = 209715 at K=5120.
Sibling does not form that product. CPU stand-in of the sibling's 64-lane
`col = lane*8 + 512k` walk vs left-to-right serial CPU, patterned nibble
`((row+group+local) & 0xf)`, scale f16(1.0), `x[i]=(i%17)*0.125-1`.

MEASURED:

| row | cpu | kernel walk | abs_d | rgb0 | u32 row*40 wraps |
|---:|---:|---:|---:|---:|---|
| 209714 | 3.17500000e1 | 3.17500000e1 | 0 | 8388560 | false |
| **209715** | **-3.08750000e1** | **-3.08750000e1** | **0** | 8388600 | false |
| 209716 | -1.01500000e2 | -1.01500000e2 | 0 | 8388640 | false |
| 248319 | 1.53625000e2 | 1.53625000e2 | 0 | 9932760 | false |

`uint32 rgb*64` wraps at rgb = 2^26 = 67108864 (DERIVED, `wrapping_mul`
gives 0). lm_head g128 max rgb = 9932799. Code offset uses `ulong`.

GPU of the sibling on 16×256 and on a 209720×5120 plane (537 MiB codes)
is written (`hq30uq4_group128_geo_matches_cpu_small_and_above_wrap`) and
skipped in this profile for the same Metal reason.

## 5. What a group-128 artifact would now execute on

| artifact | path |
|---|---|
| HQ30UQ4, group=64, cols%64==0 (G0) | `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128` — unchanged |
| HQ30UQ4, group=128, cols%128==0 | `qwen_uniform_q4_group128_matvec_geo_tpr64_tg128`, grid `ceil(rows/2)*128`, TG 128 |
| HQ30UQ4, any other group | parse refuse |
| HGRAVU01 bits∈{3,4} group=64 | existing HGRAVU geo kernels |
| HGRAVU01 bits=4 group=128 | slow `q80_hgravs01_factor_matvec_simd` (bind None) |

No repack. No generate. No GPU timing.

## 6. Build and tests

`CARGO_TARGET_DIR=workspace/ops/build/rust`

```
cargo build --release -p hawking-core --offline
warning: `hawking-core` (lib) generated 9 warnings
    Finished `release` profile [optimized] target(s) in 2m 00s
BUILD_EXIT:0
```

```
cargo test -p hawking-core --lib --offline
test result: FAILED. 642 passed; 16 failed; 7 ignored; 0 measured; 0 filtered out; finished in 20.93s
```

Cited baseline on gate: 659 passed / 3 failed. This sandbox cannot open
Metal, so 13 Metal/IOReport tests fail that the gate baseline counted as
pass. The same 3 sparse-checkout / ABI failures remain:

```
model::dsv4f_activation_capture::tests::sealed_organ_catalog_matches_schedule_receipt
model::qwen80_device_expert_table::tests::device_expert_table_abi_matches_metal_static_asserts
profile::tests::pinned_profiles_still_load_after_field_additions
```

New tests, all ok here:

```
parse_accepts_group_64_and_128_and_refuses_the_rest
hgravu01_geo_tpr64_bind_is_bits_3_and_4_only   (updated, not deleted)
g0_uniform_q4_geo_tpr64_source_is_unchanged    (tightened)
group128_code_offset_is_u64_because_u32_wraps
group128_kernel_addressing_matches_cpu_at_wrap_row
hq30uq4_group64_geo_matches_serial_and_cpu          skip: no Metal
hq30uq4_group128_geo_matches_cpu_small_and_above_wrap skip: no Metal
g0_lm_head_geo_matches_cpu_above_uint32_wrap          skip: no Metal
```

On gate the last three must be run; they are the GPU bit-identity and
above-wrap tables this profile could not produce.

`test -s workspace/superwave/g1/g1-geo-tpr64-g128.md` — this file.

New kernel name is not in `static_kernel_name` (metal/mod.rs out of scope).
Dispatch uses the real function name. Traces will label it `other`.

## 7. Next

Relaunch this contract under `--profile gate`. Do not change the sibling.
Run the three skipped Metal tests and paste their tables. Then a packer
lane can emit HQ30UQ4 group=128 (MSE scales) onto
`qwen_uniform_q4_group128_matvec_geo_tpr64_tg128`. Do not bind HGRAVU01
g=128 to the nibble-8 sibling.
