# G1 DeltaNet geometry

Lane: `34-deltanet-geometry`. HEAD this worktree. No GPU run (serialized
lane owns measurement). Every number is tagged MEASURED / DERIVED /
PROJECTED / ESTIMATED / UNVERIFIED_CLAIM. A component microbench is not
a token-level claim.

Wave-1 reports consumed before this write: `g1-kernel-inventory.md`,
`g1-token-anatomy.md`, `g1-fusion-persistent.md`, `g1-traffic-anatomy.md`,
`g1-arch-negative.md`, `g1-arch-q80.md`, `g1-resident-harvest.md`.
This morning’s kill `QWEN38_DELTANET_ACTIVATION_TAILS.json` is on
`~/.claude-grok/worktrees/auto-qwen38-deltanet-20260817-092519/`, not
on this sparse checkout’s `receipts/`.

---

## 0. Verdict

The default kernel is `qwen38_gated_delta_decode_vi` launched with
`dispatch_threads` grid `(128, 48, 128)` TG `(128, 1, 1)`. That is
**786,432 threads / 6,144 threadgroups per layer**, not 786,432
threadgroups. Wave-1 inventory and the seated ledger occupancy string
miscounted threads as TGs. Evidence: §1.

Each of those 6,144 TGs owns one `(value_head, value_dim)` column.
128 threads walk the key axis. Then **tid 0 serially adds scratch[0..127]
twice** while 127/128 threads idle behind a barrier. The host oracle
requires that left-to-right sum for greedy bit-identity. Evidence: §2.

Useful arithmetic per token is **264,830,976 FLOP** of gated-delta
(DERIVED from the host-oracle loops) plus the tiny rearrange / ba /
gated-rmsnorm tails. Launched FLOP is the same; the waste is idle
simdgroups, a 128× Q/K reload, a wasted intermediate state store, and
strided `[head][ki][vi]` loads. Evidence: §3–§4.

The seated 223.23× over floor is activation-only bytes (recurrent state
was carved into `kv_state`) divided by the falsified 411.51 GB/s roof.
Effective 1.843 GB/s on 6.88 MB. Sequential stream of the same rec-state
volume is MEASURED 588 GB/s. Evidence: §4.3.

Ranked legal geometries are in §6. Top: tile V=8 + `simd_sum` + one
state store (pure kernel). Next: 128-thread gated_rmsnorm (the named
REOPEN_IF from this morning). Dead: last-TG atomic fusion into vi,
encoder-share as a 1.0–1.5 ms win, 1-TG-per-head as production,
megakernel persistence, expand-to-float. Evidence: §7.

---

## 1. Launch geometry — source, not the wave-1 count

### 1.1 Dispatch

`crates/hawking-core/src/model/qwen38_hybrid_decode.rs:1632-1643`:

```
let (kernel, grid) = if self.deltanet_vi_parallel {
    ("qwen38_gated_delta_decode_vi", (kd, heads, vd))
} else {
    ("qwen80_gated_delta_decode_tg", (kd, heads, 1))
};
tcb.dispatch_threads(kernel, grid, (kd, 1, 1), ...)
```

`kd = vd = 128`, `heads = 48` from `Qwen38DeltaNetLayout::source_exact()`
(`qwen38_geometry.rs:31-36, 246-261`). Default `deltanet_vi_parallel=true`
(`qwen38_hybrid_decode.rs:871-875, 912`).

`TokenCommandBuffer::dispatch_threads_inner` calls Metal
`enc.dispatch_threads(MTLSize grid, MTLSize tg)`
(`metal/mod.rs:4869-4872`), not `dispatch_threadgroups`.

Metal `dispatchThreads:threadsPerThreadgroup:`:

| quantity | value | class |
|---|---:|---|
| `threadsPerGrid` | (128, 48, 128) | SOURCE |
| `threadsPerThreadgroup` | (128, 1, 1) | SOURCE |
| threads / layer | 128 × 48 × 128 = **786,432** | DERIVED |
| threadgroups / layer | (128/128) × (48/1) × (128/1) = **6,144** | DERIVED |
| threads / TG | 128 = 4 simdgroups | DERIVED (`thread_execution_width=32`, MEASURED `matvec-occupancy-230x.json` `pipeline.*.thread_execution_width`) |
| TGs / token | 6,144 × 48 = **294,912** | DERIVED |
| TGs / core if spread | 6,144 / 60 = **102.4** | DERIVED vs ledger `gpu_cores: 60` (`qwen38_token_ns_ledger.rs:183`) |

`group.x` is unused in the shader (`qwen38_device_activations.metal:214-215`
reads `group.y` = head, `group.z` = vi). Under `dispatch_threads` there is
only **one** TG in x, so `group.x` is always 0. This is **not** a 128×
redundant TG launch.

Wave-1 miscount, do not reuse:

- `g1-kernel-inventory.md:212-213`: `Grid (128, 48, 128) TG (128,1,1) = **786_432 TGs / layer**`.
- Seated ledger occupancy string (`qwen38_token_ns_ledger.rs:569`):
  `"gated_delta_vi: grid (128,48,128) = 786432 TGs/layer; 48 layers"`.
- `g1-token-anatomy.md:417` repeats the TG figure.

Those are the same `(128,48,128)` tuple. They counted threads.

Alt (off): `qwen80_gated_delta_decode_tg` grid `(128, 48, 1)` TG `(128,1,1)`
= **48 TGs / layer** (`qwen80_device_activations.metal:204-259`;
encode `hybrid_decode.rs:1638-1641`). Same serial-reduction arithmetic,
128-column loop inside the TG.

Correctness starting point, not on the Q38 path:
`qwen_next_gated_delta_decode_single` — one thread per head, fully
serial 128×128 (`qwen_next.metal:20-66`). Header: “not the final
throughput kernel.”

### 1.2 Mixer prefix this kernel sits in

`QWEN38_DELTANET_MIXER_PREFIX_KERNELS` (`qwen38_64_layer_execution_schedule.rs:17-27`):

| # | kernel | launch (SOURCE) | TGs/layer (DERIVED) |
|---|---|---|---:|
| 1 | `qwen80_residual_rmsnorm_f32` | grid (256,1,1) TG 256 | 1 |
| 2 | geo_tpr64 `in_proj_qkvz` | 16384 rows, 2/TG | 8192 |
| 3 | geo_tpr64 `in_proj_ba` | 96 rows | 48 |
| 4 | `qwen38_qkvz_rearrange_conv_l2_f32` | (256, 16, 1) TG 256 | 16 |
| 5 | `qwen80_ba_to_decay_beta_f32` | (48,1,1) TG 16 | 3 |
| 6 | `qwen38_gated_delta_decode_vi` | (128,48,128) TG 128 | 6144 |
| 7 | `qwen80_deltanet_gated_rmsnorm_f32` | (48,1,1) TG 16 | 3 |
| 8 | geo_tpr64 `out_proj` | 5120 rows | 2560 |
| 9 | `qwen_next_add_residual` | (5120,1,1) TG 256 | 20 |

Encode of 4–7: `qwen38_hybrid_decode.rs:2630-2691`. GEMVs stay
`geo_tpr64_tg128`. Fat GEMVs are not this lane.

---

## 2. What each thread actually does

### 2.1 Host oracle (the required association)

`source_qwen80_recurrent_deltanet` (`qwen80_complete_runtime.rs:7091-7118`):

```
for head:
  for vi:
    for ki:
      state[ki, vi] *= decay[head]
      kv_mem += state[ki, vi] * key[ki]
    delta = (value[vi] - kv_mem) * beta[head]
    for ki:
      state[ki, vi] += key[ki] * delta
  for vi:
    for ki:
      out[vi] += state[ki, vi] * query[ki]
```

State layout `[head][ki][vi]`, all f32. Same comment on
`qwen_next.metal:9-15` (`S <- S*exp(g); delta <- (v - S^T k)*beta;
S <- S + k⊗delta; o <- S^T q`).

Shader comments lock the reduction order:
`qwen80_device_activations.metal:204-206` “Reductions sum key index
0..127 in order to stay close to the host oracle.”
`qwen38_device_activations.metal:195-198` “the vi columns do not share
state, so this is the same serial-reduction arithmetic launched with
128× occupancy.”

### 2.2 VI kernel — one (head, vi) per TG, one ki per thread

`qwen38_device_activations.metal:199-248`. Per thread (`tid = ki`):

1. `s = state[ki, vi] * decay[head]`; **store `s` back** (intermediate).
2. `scratch[tid] = s * key[ki]`; barrier.
3. **if tid==0:** `kv_mem = sum_{i=0..127} scratch[i]` left-to-right;
   write `scratch[0]`. 127 threads idle.
4. barrier; `delta = (value[vi] - scratch[0]) * beta[head]`.
5. `state[ki, vi] += key[ki] * delta`; barrier.
6. `scratch[tid] = state[ki, vi] * query[ki]`; barrier.
7. **if tid==0:** `out[vi] = sum_{i=0..127} scratch[i]`. 127 threads idle.

`ki < key_dim` is always true (TG = key_dim = 128). Dead branch.

TG scratch: 128 × f32 = 512 B (`set_threadgroup_memory_length(0, 128*4)`,
`hybrid_decode.rs:1654`). Used only as a staging buffer for the serial
reduce.

Q80 TG twin (`qwen80_device_activations.metal:231-258`) is the same
body inside `for (vi = 0; vi < 128; ++vi)`. 48 TGs, each looping 128
columns. Key/query can stay in a register across `vi`; state is
reloaded each column.

### 2.3 Rearrange + conv + L2

`qwen38_qkvz_rearrange_conv_l2_f32` (`qwen38_device_activations.metal:32-105`).
Hard-fails unless `key_heads==16 && vpk==3 && dims==128 && conv_kernel==4`.

Per key-head TG (256 threads):

| tid | work |
|---|---|
| 0..127 | causal conv on Q[tid] and K[tid] (3 taps + shift + silu) |
| 0..255 | `for row=tid; row<384; row+=256` causal conv on V[row], copy Z[row]. 128 threads do 2 rows, 128 do 1 |
| 0..255 | 256-wide tree reduce of 128 useful Q² / K² (tid≥128 contribute 0) |
| 0..127 | L2 scale; write Q/K **repeated 3×** onto 48 value-head slots |

Idle: 128/256 threads on Q/K conv and on the L2 write. Half the reduce
tree is zeros.

### 2.4 ba_to_decay

`qwen80_ba_to_decay_beta_f32` (`qwen80_device_activations.metal:155-177`).
48 threads, TG=16, **3 TGs**. Each thread: one value-head
`softplus(a+dt_bias)`, `decay=exp(-exp(A_log)*softplus)`,
`beta=sigmoid(b)`. Token-dependent (reads this token’s BA GEMV).
A_log and dt_bias are static.

### 2.5 gated_rmsnorm

`qwen80_deltanet_gated_rmsnorm_f32` (`qwen80_device_activations.metal:179-202`).
48 threads, TG=16, **3 TGs**. Each thread owns one value head and
**serially** walks 128 dims twice (sum-sq, then `x * inv_rms * w * silu(z)`).
No tree, no simdgroup. 16-wide TG is 0.5 simdgroup.

---

## 3. Useful vs idle vs redundant

### 3.1 Inside one VI TG

| phase | useful threads | idle | note |
|---|---:|---:|---|
| decay load/mul/store | 128 | 0 | also writes an intermediate the next stmt does not need |
| kv contrib | 128 | 0 | |
| barrier + serial reduce | 1 | 127 | 128 dependent adds on tid 0 |
| delta broadcast | 128 | 0 | value/beta/decay already uniform |
| rank-1 update + store | 128 | 0 | |
| query contrib | 128 | 0 | |
| barrier + serial reduce | 1 | 127 | same |

Idle fraction during each reduce: **127/128 = 0.992** (DERIVED).
Four `threadgroup_barrier`s per TG. Three of four simdgroups are parked
for the whole serial loop; even tid0’s simdgroup has 31 idle lanes.

### 3.2 Redundant device traffic (DERIVED from the shader)

Necessary per layer (one load + one store of state, one load of Q/K/V,
one store of out, 48+48 of decay/beta):

| tensor | bytes |
|---|---:|
| rec state R+W | 48 × 128 × 128 × 4 × 2 = 6,291,456 |
| Q, K, V, out | 4 × 48 × 128 × 4 = 98,304 |
| decay, beta | 48 × 4 × 2 = 384 |
| **necessary** | **6,390,144** |

Launched VI actually does:

| extra | how | bytes / layer |
|---|---|---:|
| intermediate `state = decayed` store | metal:227-228 | 3,145,728 |
| key loaded in 128 independent vis TGs | metal:229, 237 | 3,145,728 vs necessary 24,576 (**128×**) |
| query loaded in 128 vis TGs | metal:241 | 3,145,728 vs 24,576 (**128×**) |
| decay/beta re-read per vis TG | metal:222-223 | 49,152 vs 384 (**128×**) |

Q/K amplification is the one that matters: 128 TGs of a head each load
the same 128-vector. Tile V=N cuts that factor to 128/N.

Stride: `index = ki * 128 + vi` (`metal:225`). Adjacent threads in a
simdgroup touch addresses 512 B apart. Sequential stream of the same
resident rec-state is MEASURED 588.49 GB/s
(`TOKEN_NS_QWEN38.json` component `kv_state`). The VI kernel is not
that access.

### 3.3 Occupancy, launch-derived (not a hardware counter)

G024 `unresolved[0]`: “Occupancy is launch-geometry derived (60 cores),
not a hardware counter sample.”

| kernel | TGs/layer | threads/TG | TG/core (60) | work / thread |
|---|---:|---:|---:|---|
| VI current | 6144 | 128 | 102.4 | 1 state elem |
| TG serial (off) | 48 | 128 | **0.80** | 128 vis, serial reduce each |
| gated_rmsnorm | 3 | 16 | **0.05** | 128-long serial RMS+silu |
| ba_to_decay | 3 | 16 | 0.05 | 1 softplus+exp+sigmoid |
| rearrange | 16 | 256 | 0.27 | 1 key-head |

VI is **not** occupancy-starved on TG count. 102 TG/core is more than
enough to fill 60 cores. The starvation is **inside** the TG.

Serial TG **is** occupancy-starved: 48 TGs cannot cover 60 cores.
MEASURED: isolated `gated_delta_48` serial 11,485,249 vs vi 2,141,916
= **5.36×** (`qwen38-layer-dense-q4-swiglu.json`
`gated_delta_vi_parallel` / `isolated.gated_delta_48_serial_vi`).
Production GPU 42,734,499 → 33,449,499. G0 is the vi genome.

Therefore “fewer larger TGs” that collapse back to 48 TGs/layer
**repeats the serial occupancy loss**, even if the serial reduce is
fixed. Floor for a healthy map on this box is ≳60 TGs, preferably
several hundred. See §6.

### 3.4 Session variance (do not blend)

| receipt | isolated gated_delta_48 | class |
|---|---:|---|
| G024 `QWEN38_TOKEN_NS_LEDGER.json` `isolated` | 2,146,166 [2,146,166 / 2,134,125 / 2,229,583] | MEASURED G0 vi |
| layer-dense B | 2,141,916 | MEASURED vi, other session |
| G002 `qwen38_family_p3.json` | 5,321,833 | MEASURED, different genome/state (`g1-token-anatomy.md:379`) |
| this-morning tails baseline | 5,335,249 | MEASURED dirty (`QWEN38_DELTANET_ACTIVATION_TAILS.json` `measured_before`) |

G024 2.146 ms is the number this lane projects against. G002 / tails
5.3 ms is first-class session variance, not a second geometry.

---

## 4. Required arithmetic per token

### 4.1 Gated-delta FLOP (DERIVED, host-oracle loop body)

Per `(head, vi)`: 513 mul + 384 add + 1 sub = **898 FLOP**
(literal `*=`, `+=`, `*` in `qwen80_complete_runtime.rs:7100-7116`).

| scope | FLOP |
|---|---:|
| one layer (48 × 128 × 898) | 5,517,312 |
| one token (× 48 layers) | **264,830,976** |

State elements touched: 48 × 128 × 128 = 786,432 / layer;
37,748,736 / token. Necessary rec R+W: **301,989,888 B / token**
(matches `theoretical_state_bytes` rec × 2,
`qwen38_token_ns_ledger.rs:119-136`, test
`qwen38_geometry.rs:503` `48*128*128`).

This FLOP count does **not** change across legal geometries. The
current kernel does not launch extra useful FLOP; it launches the
same 898 as a 1-useful-thread serial reduce plus idle waits.

### 4.2 Tails (DERIVED, shader loops)

| kernel | per layer | per token |
|---|---|---|
| rearrange conv | 10,240 channels × (3-tap MAC + silu) ≈ 8e4 FLOP | ≈ 3.9e6 |
| L2 Q/K | 2 × (128 sq + 127 add + rsqrt) | × 16 key-heads × 48 |
| ba_to_decay | 48 × (softplus + exp + sigmoid) | × 48 |
| gated_rmsnorm | 48 × (128 sq + 127 add + rsqrt + 128 silu + 128×3 mul) | × 48 |

Tails are orders of magnitude below gated-delta FLOP and still take
MEASURED 1.30 + 0.35 + 0.14 = 1.79 ms isolated (`QWEN38_TOKEN_NS_LEDGER.json`
`isolated`) because they launch 3 / 16 / 3 TGs.

### 4.3 The 223× figure — what it actually is

`TOKEN_NS_QWEN38.json` / `qwen38_token_ns_ledger.rs:561-572` component
`deltanet`:

```
bytes_read  = 6144 * 4 * 48 * 4 = 4,718,592   // 4 activation vectors × 48 layers
bytes_written = 6144*4*48 + 5120*4*48 = 2,162,688
ns_per_token = 3,732,794.93
effective_gb_s = 1.843465854461662
theoretical_lower_bound_ns = 16,722.02   // at HONEST_DECODE_CEILING_GB_S=411.51
measured_over_floor = 223.2262664394026
dispatches = 192   // 48 × (rearrange, ba, vi, rms)
```

Composition (`qwen38_token_ns_ledger.rs:405-409`): isolated
rearrange leftover + ba + gated leftover after rec-stream + gated_rmsnorm
+ dn FMA remainder + 48/64 mixer residual. Rec/conv state bytes live
in `kv_state` (MEASURED 537,665 ns @ 588.49 GB/s, 0.70× that same
411.51 floor — i.e. the sequential stream **beats** the seated roof).

411.51 GB/s is the wave-1-falsified Q80 unique-once control
(`g1-roof-falsification.md`, `g1-token-anatomy.md:353`). Relabel only,
not a new measurement:

| roof (GB/s) | floor ns on 6,881,280 B | over | class |
|---|---:|---:|---|
| 411.51 seated | 16,722 | **223.23×** | MEASURED / seated |
| 639.25 wave-1 effective | 10,765 | 346.8× | PROJECTED re-label |
| 699.57 single-address | 9,836 | 379.5× | PROJECTED re-label |

The 223× claim is “this organ is not DRAM.” That survives the roof
correction. It gets worse, not better.

G024 isolated split used for projections (component, not token):

| family | median GPU ns | ns/launch | TGs/launch |
|---|---:|---:|---:|
| gated_delta_48 | 2,146,166 | 44,712 | 6144 |
| gated_rmsnorm_48 | 1,295,500 | 26,990 | 3 |
| rearrange_48 | 350,999 | 7,312 | 16 |
| ba_to_decay_48 | 139,374 | 2,904 | 3 |
| stream_rec_state | 467,374 | — | 1 sequential |
| stream_conv_state | 19,000 | — | 1 sequential |

Gated leftover after rec-stream, DERIVED: 2,146,166 − 467,374 =
**1,678,792 ns**. That 1.68 ms is the geometry hole (stride + serial
reduce + Q/K amp + extra store + 4 barriers). ALU at 265 MFLOP/token
cannot be 1.68 ms unless occupancy inside the TG is the story.

### 4.4 Per-token setup that does not depend on the token

Static, rebound every dispatch today: `heads`, `kd`, `vd`, `eps`,
A_log, dt_bias, conv1d weight, norm.weight. ICB already interns the
scalars (`g1-fusion-persistent.md` P6, MEASURED −0.66 ms named fixed,
not on HEAD). Not a kernel-geometry lever.

Token-dependent, cannot hoist: BA projection → decay/beta; Q/K/V/Z
activations; rec/conv state. Hard-coded `key_dim != 128` guards are
dead per-thread work (tiny).

Nothing in the VI body is per-token setup. The waste is the body.

---

## 5. Surrounding-kernel fusion — what is still legal

G024 `top_three_attacks[1]` asked for 1.0–1.5 ms by fusing
gated_rmsnorm + ba + rearrange “into the vi kernel or one encoder.”
That experiment ran this morning.

`QWEN38_DELTANET_ACTIVATION_TAILS.json` (`epistemic_state=NEGATIVE`):

| arm | isolated tail-chain GPU | correctness | complete-token |
|---|---|---|---|
| split (default) | 7,382,625 med [7,362,374 / 7,529,624 / 7,382,625] | bit-id | baseline |
| one_encoder | 7,388,708 med, GPU == split | bit-id, seal 3/3 | pair Δ 133 / 245 / 209 µs, inside DIRTY noise |
| fused_vi (last-TG atomic rms + in-kernel ba) | 8,106,916 med [8,106,916 / 8,178,791 / 7,995,166] | **ids drifted** | REJECT |

Cause (receipt `fused_vi.why`): last-arriving TG needs device-scope
visibility of `rec_out[vi]`; `memory_order_device` is not in this
Metal compiler; `memory_order_relaxed` + `threadgroup_barrier(mem_device)`
is insufficient. Also slower than split.

Limiter (receipt `isolated_after.limiter`): “GPU time of 48 underfilled
16-wide rms+silu launches, not host encoder create/end.”

Named REOPEN_IF, explicitly untested: **one TG/head × 128-thread
gated_rmsnorm, still a separate dispatch after vi.**

`g1-fusion-persistent.md` P4b / P5: folding fat GEMVs into one/few
TGs, or a persistent `n_layers` megakernel, is the MEASURED 4.4×
`qwen3b_megakernel_nlayer` kill. Do not fuse qkvz/out GEMVs into
this organ.

Legal fusion remaining:

- RMS as an **epilogue of a TG that already owns all 128 vis of a
  head** (registers or TG mem). That is a geometry change, not
  last-TG atomics.
- ba_to_decay as a 48-scalar prologue of that same TG (reads BA,
  writes two registers). Distinct from the killed last-TG path.
- Rearrange stays 16 key-head TGs; different occupancy map. Fuse
  only if the consumer keeps ≥16 TGs and does not wait on a device
  atomic.

---

## 6. Ranked geometries

Required FLOP/token is 264,830,976 for all rows (DERIVED, §4.1).
“Predicted arithmetic” below is that number plus launched extras
(idle is not extra FLOP). Occupancy is launch-derived TG/core vs 60.
Time predictions are ESTIMATED from G024 isolated 2,146,166 /
1,295,500 and the 467,374 rec-stream floor. They are **not**
token-level claims. Falsifier for every row: isolated 48-layer
family GPUEnd−GPUStart vs G024 vi, then one dirty complete-token
pair, greedy-16 + seal. Serialized GPU lane only.

`maxTotalThreadsPerThreadgroup = 1024` MEASURED
(`matvec-occupancy-230x.json` `pipeline.*.max_total_threads_per_threadgroup`).
Device `maxThreadgroupMemoryLength` is **UNMEASURED** here; Apple
family default is 32 KiB (ESTIMATED). One head of f32 state is
64 KiB and does not fit that default. Tile V=32 is 16 KiB. Cheapest
close: one `MTLDevice.maxThreadgroupMemoryLength` query.

### R1. Tile V=8 + simd_sum + single state store — **pure kernel**

Grid `(128, 48, 16)` TG `(128,1,1)` = **768 TGs / layer**, 12.8 TG/core.
Each thread owns `ki` and loops 8 vis. Holds 8 state floats + key +
query in registers across the tile. One device store of the updated
state. Reduce via `simd_sum` (4 simdgroups) + 4-wide TG combine, not
tid0’s 128-add.

| | value | class |
|---|---|---|
| TGs/core | 12.8 | DERIVED |
| work/thread | 8 state elems (was 1) | DERIVED |
| Q/K amp | 16× (was 128×) | DERIVED |
| extra state store | 0 | DERIVED |
| required FLOP/token | 264,830,976 | DERIVED |
| launched extra FLOP | tree reduce, not 1.57e6 serial adds | DERIVED |
| predicted isolated gated_delta | **0.7–1.2 ms** (was 2.15) | ESTIMATED |
| predicted deltanet-bucket cut | **0.9–1.4 ms** of 3.73 | PROJECTED |
| bit-id vs host left-to-right sum | **NO** (`qwen_uniform_q4.metal:136` “simd_sum. Not bit-identical.”) | SOURCE |
| generate gate | quality / seal, not greedy-id | — |

Why this rank: keeps enough TGs to cover 60 cores (unlike 48-TG),
raises work/thread, kills Q/K amp by 8×, kills the serial reduce,
kills the extra store. Does not share the last-TG fusion kill.
Does not share the megakernel kill.

V=4 (1,536 TGs, 25.6 TG/core) is the conservative sibling if V=8
regfile pressure shows up. V=16 (384 TGs, 6.4 TG/core) is the
aggressive sibling. Do not ship V=128 / 48 TGs as the first try
(§3.3, MEASURED 5.36× loss on that map).

### R2. 128-thread gated_rmsnorm, still a separate dispatch — **pure kernel**

Named REOPEN_IF (`QWEN38_DELTANET_ACTIVATION_TAILS.json`
`next_bottleneck`). Grid `(48*128, 1, 1)` TG `(128,1,1)` = **48 TGs**,
or `(6144,1,1)` TG 128 with one thread per dim and a tree reduce.

Current: 3 TGs × 16 threads, each thread a 128-long serial loop,
MEASURED 1,295,500 ns / 26,990 ns/launch.

| | value | class |
|---|---|---|
| TGs/core | 0.80 (48-TG map) | DERIVED |
| work/thread | 1 dim of a 128-reduce, then 1 apply | DERIVED |
| required FLOP/token | ~1.5e6 (tiny vs delta) | DERIVED |
| predicted isolated rms | **0.6–1.0 ms** (was 1.30) | ESTIMATED |
| predicted bucket cut | **0.3–0.7 ms** | PROJECTED |
| bit-id | tree vs left-to-right; same issue as R1 if tree | SOURCE |
| shares this-morning kill? | **NO** — still a separate dispatch, no last-TG atomic | MEASURED_NEGATIVE was the other mechanism |

48 TGs is occupancy-thin, but the kernel is short; the current 27 µs
is a 128-step serial loop on one thread, not a 11.5 ms rec-state
walk. Parallelizing the 128 is the reopen. If 48 TGs still look like
launch tax, launch 48×2 with 64-wide halves — still not fusion-into-vi.

### R3. simd_sum only, same 6,144-TG VI map — **pure kernel**

Cheapest rewrite. Replace tid0’s two 128-add loops with 4× `simd_sum`
+ 4-wide TG add. Same traffic, same Q/K amp, same extra store.

| | value | class |
|---|---|---|
| TGs/core | 102.4 (unchanged) | DERIVED |
| work/thread | 1 (unchanged) | DERIVED |
| predicted isolated gated_delta | **1.3–1.8 ms** | ESTIMATED |
| predicted bucket cut | **0.3–0.8 ms** | PROJECTED |
| bit-id | NO | SOURCE |

Use as a one-swap A/B to price the serial reduce alone before tiling.
If this does nothing, the hole is stride/traffic and R1/R5 move up;
if this takes most of the 1.68 ms leftover, stop at R3.

### R4. Drop the intermediate `state = decayed` store — **pure kernel**

`metal:227-228` then `237` can keep `decayed` in a register and write
once. Bit-identical. 151,000,000 B fewer writes / token (DERIVED).

At kv_state’s 588 GB/s that is 257 µs (PROJECTED sequential). Under
the current stride the save is larger but UNMEASURED. Do this on any
rewrite. Alone it is not the 1.68 ms.

### R5. State layout `[head][vi][ki]` — **layout change**

Current `index = ki * 128 + vi` de-coalesces a simdgroup (512 B
stride). Transposed `index = vi * 128 + ki` makes 32 consecutive
threads touch 128 B.

This is a representation change of the recurrent-state buffer
(init, slot stride, any host oracle that indexes it, rearrange’s
consumer contract). Not a weight-layout change.

Predicted: state traffic approaches the MEASURED 588 GB/s sequential
stream (467 µs R+W of 302 MB). Combined with R1, isolated gated_delta
**0.5–0.9 ms** ESTIMATED. Do not ship the transpose alone without
R1/R3 — the serial reduce would remain.

`dead_levers.md:36` A10 access-order repack is a **Q4_K GEMV**
de-coalesce kill (−16.8%). It does not transfer: here we are
*undoing* a de-coalesce on an f32 state tensor, not repacking
packed weights.

### R6. Per-head TG, 256–512 threads, RMS+ba epilogue — **kernel rewrite + legal fusion**

One TG owns one head. 256 threads = 128 ki × 2 vis, loop 64; or 512
= 128 ki × 4 vis, loop 32. Output[128] in TG mem (512 B). RMS+silu
and ba→decay/beta run as epilogue in the same dispatch. Deletes 48
rms + 48 ba launches. Rearrange stays separate (different map).

| | value | class |
|---|---|---|
| TGs/core | **0.80** | DERIVED |
| shares serial-TG occupancy fail? | **YES, the map** | MEASURED 5.36× on 48 TGs |
| predicted isolated delta+rms+ba | 1.5–4.0 ms vs split 2.15+1.30+0.14 = 3.59 | ESTIMATED, **wide** |
| predicted token | UNMEASURED; kill if > split | — |
| shares last-TG kill? | NO — one TG, no cross-TG atomic | — |
| shares megakernel kill? | NO — no GEMV inside | — |

Ranked below R1 because the 48-TG map already lost once. Only try
after R1 is measured, and only as an A/B that is allowed to die.

A safer cousin: **192 TGs** (4 vis-tiles per head, 3.2 TG/core) for
delta, still a separate 48-TG rms (R2). That is R1 at V=32 plus R2,
not this row.

### R7. One simdgroup (32 threads) per (head, vi), 4 ki / thread — **pure kernel**

Grid `(32, 48, 128)` TG `(32,1,1)` = 6,144 TGs of 1 simdgroup.
`simd_sum` only, **no threadgroup barrier**. Work/thread = 4.

Same TG count as today, 4× the useful work, no TG-memory reduce.
Predicted isolated gated_delta 1.0–1.6 ms ESTIMATED. Bit-id NO.
Use if R1’s extra registers hurt occupancy; this keeps the current
TG count and shrinks the TG.

---

## 7. Dead levers (do not re-propose)

| # | mechanism | status | evidence | REOPEN_IF |
|---|---|---|---|---|
| D1 | Fuse rms/ba/rearrange **into vi via last-TG atomics** | **KILLS** | `QWEN38_DELTANET_ACTIVATION_TAILS.json` fused_vi 8.11 vs 7.38 ms; ids drifted; no `memory_order_device` | a documented device-scope visibility primitive, plus isolated fused ≤ split |
| D2 | One serial encoder around the four tails as a 1.0–1.5 ms win | **KILLS** as production | same receipt, one_encoder GPU == split; complete-token Δ 133–245 µs inside DIRTY noise | limiter stops being the 16-wide rms GPU time |
| D3 | G024 attack #2 as still-open 1.0–1.5 ms | **superseded** by D1/D2 | G024 `top_three_attacks[1]` is the hypothesis D1/D2 killed this morning | — |
| D4 | 1 TG / head (`qwen80_gated_delta_decode_tg`) as production | **KILLS** vs vi | isolated 11.49 vs 2.14 ms; production GPU 42.73 → 33.45 ms (`qwen38-layer-dense-q4-swiglu.json`) | a 48-TG design that is measured faster than 6,144-TG vi (R6’s gate) |
| D5 | `qwen_next_gated_delta_decode_single` (1 thread / head) as throughput | **KILLS** | shader header `qwen_next.metal:6-7`; 48 threads, fully serial 128×128 | never as a token lever |
| D6 | Persistent TGs / registers **across tokens or layers** | **KILLS** (megakernel) | `g1-fusion-persistent.md` §1.1, P5; `dead_levers.md:25`; registers and TG mem do not survive a dispatch | multi-TG work queue that preserves `geo_tpr64` occupancy and beats 1-CB Q4 on GPU timestamps |
| D7 | Fold qkvz / out GEMVs into the delta TG | **KILLS** | same 4.4× mechanism; G024 `not_a_kernel_win` on addressing | native Q4 in-register **and** TG map ≥ tpr64 **and** complete-token A/B |
| D8 | Fuse tails into the **following** GEMV | **KILLS** | tails receipt `negative_science[2]`; prior −10.68 ms complete-token; harvest §4.4 item 5 “Do not retry GEMV+RMSNorm fusion” | a new measurement that does not regress gate GB/s |
| D9 | Q80 tg256 occupancy-tile launch copied onto this organ | **KILLS as a copy** | `g1-arch-q80.md` Q38 gate tg256 26,541 vs tpr64 15,125 ns | a Q38-shaped probe that beats tpr64 |
| D10 | Expand state or weights to f16 then generic GEMV | **KILLS** | binding rule + 4.4× strike 1 | complete-token net win on packed-in-register |
| D11 | ICB / encoder-collapse as the DeltaNet story | **KILLS as this organ** | D2; ICB is a 0.66 ms ceremony win on another genome (`g1-fusion-persistent.md` P6), not 3.73 ms | — |
| D12 | Generator+residual on this organ | **KILLS** | standing wave-1; `g1-generator-residual.md`; do not resurrect | new evidence the residual quantizes better |

“Keep recurrent state in registers across the **step**” is R1 (tile
inside one dispatch). Across **tokens** is D6.

`dead_levers.md:18` Phase-2.2 trivial-op fusion is a llama
rope/add/memcpy kill. Isolated gated_rmsnorm is 1.30 ms, not
sub-noise. Do not apply that kill to R2. Do apply it to fusing
`qwen_next_add_residual` (118 µs).

---

## 8. What a later GPU lane should run (this lane must not)

Cheapest order, one lock, dirty OK if labelled:

1. R4 (drop intermediate store) on the current VI map. Bit-id.
   Isolated gated_delta_48 vs 2,146,166. Expect ≤0.4 ms. If zero,
   the extra store is hidden behind latency.
2. R3 (`simd_sum`, same grid). Accept bit drift; run seal, not
   greedy-id. Prices the serial reduce.
3. R1 V=8. Same gates. Kill if isolated ≥ current vi.
4. R2 128-thread rms, separate dispatch. The named reopen.
5. Only then R5 transpose, then R6 if R1’s 768-TG map is the new
   incumbent and someone still wants one dispatch.

Do not re-run D1/D2. Do not take the GPU while the resident
organism is generating.

---

## 9. Evidence index

| claim | pointer |
|---|---|
| VI encode grid + TG | `qwen38_hybrid_decode.rs:1632-1655` |
| `dispatch_threads` not `dispatch_threadgroups` | `metal/mod.rs:4869-4872` |
| VI shader body | `qwen38_device_activations.metal:199-248` |
| TG serial twin | `qwen80_device_activations.metal:204-259` |
| 1-thread correctness kernel | `qwen_next.metal:20-66` |
| host oracle | `qwen80_complete_runtime.rs:7091-7118` |
| geometry 16/48/128/3 | `qwen38_geometry.rs:31-36, 503` |
| default vi on | `qwen38_hybrid_decode.rs:871-875, 912` |
| rearrange / ba / rms encode | `qwen38_hybrid_decode.rs:2630-2691` |
| 223.23×, 1.843 GB/s, 6.88 MB | `TOKEN_NS_QWEN38.json` component `deltanet`; formula `qwen38_token_ns_ledger.rs:561-572` |
| isolated family medians | `QWEN38_TOKEN_NS_LEDGER.json` `isolated` |
| serial vs vi 11.49 / 2.14 ms | `qwen38-layer-dense-q4-swiglu.json` `gated_delta_vi_parallel` |
| rec-stream 467,374 ns / 588 GB/s | same ledger `isolated.stream_rec_state` + component `kv_state` |
| 60 GPU cores, launch-derived occupancy | `qwen38_token_ns_ledger.rs:174-186`; G024 `unresolved[0]` |
| simd_sum not bit-id | `qwen_uniform_q4.metal:136` |
| G024 attack #2 (now dead) | `G024_QWEN38_TOKEN_NS.json` `top_three_attacks[1]` |
| this-morning tails kill | `QWEN38_DELTANET_ACTIVATION_TAILS.json` (worktree `auto-qwen38-deltanet-20260817-092519`) |
| megakernel 4.4× | `g1-fusion-persistent.md` §1.1; `dead_levers.md:25` |
| ICB 0.66 ms, not this organ | `g1-fusion-persistent.md` P6 |
| wave-1 786,432-TG miscount | `g1-kernel-inventory.md:212-213`; ledger occupancy string |

---

## Completion report

```
STATUS
IMPLEMENT_READY

CLAIMS
C1. Default launch is dispatch_threads (128,48,128) / TG (128,1,1) = 786,432 threads and 6,144 TGs per layer, not 786,432 TGs. Wave-1 inventory and the seated occupancy string miscounted. EVIDENCE: §1.1; hybrid_decode.rs:1632-1643; metal/mod.rs:4869-4872; g1-kernel-inventory.md:212-213; qwen38_token_ns_ledger.rs:569.
C2. Each VI thread owns one (head, ki, vi) state element; tid 0 then serially reduces 128 floats twice while 127/128 threads idle. Same association as the host oracle. EVIDENCE: §2; qwen38_device_activations.metal:199-248; qwen80_complete_runtime.rs:7091-7118.
C3. Useful gated-delta arithmetic is 264,830,976 FLOP/token (DERIVED). Waste is idle simdgroups, 128× Q/K reload, an extra decayed-state store, and [ki][vi] stride — not extra FLOP and not 128× redundant TGs. EVIDENCE: §3–§4.
C4. Seated 223.23× is 3,732,795 ns on 6,881,280 activation-only bytes at the falsified 411.51 GB/s roof (1.843 GB/s). Sequential rec-state stream of the carved-out bytes is 588 GB/s. EVIDENCE: TOKEN_NS_QWEN38.json deltanet + kv_state; qwen38_token_ns_ledger.rs:405-409, 561-572.
C5. Serial 48-TG kernel is MEASURED 5.36× slower isolated (11.49 vs 2.14 ms). “One TG per head” as production is dead on this box. EVIDENCE: qwen38-layer-dense-q4-swiglu.json gated_delta_vi_parallel.
C6. Fuse-tails-into-vi (last-TG atomic) and encoder-share as a 1.0–1.5 ms win are MEASURED_NEGATIVE this morning. Named reopen is 128-thread gated_rmsnorm, still a separate dispatch. EVIDENCE: QWEN38_DELTANET_ACTIVATION_TAILS.json; g1-resident-harvest.md:374-380.
C7. Ranked legal geometries: R1 tile V=8+simd_sum (pure kernel, 768 TG, 12.8 TG/core, ESTIMATED 0.7–1.2 ms isolated delta); R2 128-thread rms (pure kernel, named reopen); R3 simd_sum-only same map; R4 drop extra store; R5 [vi][ki] layout change; R6 per-head fused rms (occupancy-dangerous); R7 32-wide TG. Megakernel persistence, GEMV fold, last-TG fusion, 1-TG/head production are dead. EVIDENCE: §6–§7.

EVIDENCE
- crates/hawking-core/shaders/qwen38_device_activations.metal:199-248
- crates/hawking-core/shaders/qwen80_device_activations.metal:155-259
- crates/hawking-core/src/model/qwen38_hybrid_decode.rs:1632-1655, 2630-2691, 871-912
- crates/hawking-core/src/metal/mod.rs:4869-4872
- crates/hawking-core/src/model/qwen38_geometry.rs:31-36, 246-261, 503
- crates/hawking-core/src/model/qwen38_token_ns_ledger.rs:119-136, 174-186, 405-409, 561-572
- crates/hawking-core/src/model/qwen80_complete_runtime.rs:7091-7118
- git show HEAD:receipts/ascent-2026-08-16/TOKEN_NS_QWEN38.json component deltanet
- git show HEAD:receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json isolated
- git show HEAD:receipts/ascent-2026-08-16/qwen38-layer-dense-q4-swiglu.json gated_delta_vi_parallel
- /Users/scammermike/.claude-grok/worktrees/auto-qwen38-deltanet-20260817-092519/receipts/ascent-2026-08-16/QWEN38_DELTANET_ACTIVATION_TAILS.json
- workspace/superwave/g1/g1-kernel-inventory.md:200-223
- workspace/superwave/g1/g1-token-anatomy.md:139-186, 363-389
- workspace/superwave/g1/g1-fusion-persistent.md:39-90, 328-375
- workspace/superwave/g1/g1-resident-harvest.md:374-380
- git show HEAD:workspace/docs/guides/dead_levers.md:18,25

CHANGES
Created workspace/superwave/g1/g1-deltanet-geometry.md only.

TESTS
$ test -s workspace/superwave/g1/g1-deltanet-geometry.md && echo PASS
(run at end of this turn)
$ wc -l workspace/superwave/g1/g1-deltanet-geometry.md
(run at end of this turn)
$ git status --porcelain
(run at end of this turn)

RISKS
- simd_sum (R1/R3/R7) breaks left-to-right f32 association; greedy-id will drift. Quality/seal is the gate, not Hi-sequence identity.
- Time predictions are isolated-family ESTIMATES on G024’s 2.15/1.30 ms. G002/tails sessions saw ~5.3 ms gated_delta; a dirty box can hide a real cut.
- 48-TG maps (R2, R6) share the occupancy shape that lost 5.36× on the long kernel. R2 is short enough that it may still win; R6 may not.
- Device maxThreadgroupMemoryLength is UNMEASURED. R6 storing a full head (64 KiB) is illegal if the limit is 32 KiB.
- This lane did not run GPU. A later lane that runs R1–R4 while the resident organism holds the device will confound both.

UNRESOLVED
- Hardware occupancy counters on the VI kernel (G024 unresolved[0], still open).
- Which session’s 2.15 vs 5.3 ms isolated gated_delta is the clean G0 number. Projections use G024.
- Whether a tree reduce that is not left-to-right still seals on the three prompts.
- maxThreadgroupMemoryLength on this MTLDevice.
- Per-layer vi-parallel timestamps (none on G024; layer-dense decomposed is the serial genome).

NEXT
Serialized GPU lane: R4 then R3 then R1 then R2, isolated then one dirty complete-token pair. Do not retry D1/D2. Do not fold GEMVs. Do not persist TGs across tokens.
```
