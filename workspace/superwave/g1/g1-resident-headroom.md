# G1 resident headroom — what a lower-BPW body actually buys

Lane: `67-resident-headroom`. Write scope: this file only.
No GPU, no Metal timing, no inference, no live-process mutation, no health RPC
(lock owner `genesis-resident:parent` the whole window).

Label key: `MEASURED` this process. `RECEIPT` prior receipt, not re-timed.
`SOURCE` file:line. `DERIVED` arithmetic on those. `PROJECTED` scale under a
stated assumption. `ESTIMATED` header-derived residency, not a live Metal load.

A component microbenchmark is not a token. PACK_REPORT `projected_tps=79.44`
is not a token.

---

## 0. Verdict

A 2.0 BPW bootstrap is **generation churn**. The already-packed 1.291 artifact
is the only density step on the table that can free a real body-sized slab,
and only if it is loaded **native** (no expand-to-Q4). Even then it does **not**
buy additional concurrent G0 decode processes, does **not** raise the
concurrent-decode ceiling above 1, and does **not** reach 100 TPS.

What it does buy, if native + coherent + footprint-confirmed:

| convert-into | count | class |
|---|---:|---|
| extra G0 process-pool children (`phys` ~15.66 GB) | **0** | DERIVED: 9.14 GB < 15.66 GB |
| extra G1-native process-pool children (`phys` ~6.52 GB PROJECTED) | **1** on a quiet box; **0 today** (swap already 3.41 GB) | PROJECTED / MEASURED box |
| extra in-process sessions at live `max_seq_len=8192` | **+7** | DERIVED 9.14e9 / 1,232,326,404 |
| extra in-process sessions at seq=128 | **+53** | DERIVED 9.14e9 / 173,703,168 MEASURED marginal |
| extra `memory_lane_cap` grok-run slots | **+1 to +2** | DERIVED from live formula |
| complete-token multiplier vs live 25.4284 TPS | **1.57× to 1.88×** | PROJECTED; not a token |
| 100 TPS | **unreachable by density alone** | DERIVED: remainder floor 55.5–71.8 TPS |

Current child-capacity bind is **not** “the 14.3 GB body is too big for more
sessions.” The live body already has four sessions attached and two of them
have never been served. The binds that actually stop work are (1) GPU
serialization + decode ceiling 1, (2) `memory_lane_cap` on grok-run lanes,
(3) a 79 s propose prefill + 2600-token decode, (4) a promotion queue with
334 entries and 0 promoted.

**Minimum promotion bar** is in §9. Short form: native load, native
coherence, `footprint` weight drop ≥ 8 GB, complete-token ≥ 1.40× live G0
(TOKEN_NS ≤ 28.09 ms). Anything that still expands to Q4, or a 2.0856 BPW
sibling of the same crushed MLP, fails the bar before a restart is paid.

---

## 1. Two child designs (do not collapse)

### 1.1 In-process sessions — the live organism

`SOURCE` `tools/agentos/genesis_body/src/main.rs:3-6,30`:

```
Loads Qwen3.8 weights once, attaches four isolated logical sessions
SESSION_ROLES = parent, child_a, child_b, protected_test
Decode remains serialized because the measured concurrent-decode ceiling is 1.
```

Live argv `MEASURED` `pgrep -lf genesis-resident` pid **74869**:

```
--artifact-root .../qwen38-27b/uniform-q4-v1
--max-seq-len 8192
--max-new-tokens 900
```

Live health `MEASURED` `/tmp/g1-baseline-remeasure/resident_measure_v2.json`
`health_before` (same pid, 11:10 today, `load_count=1`):

| field | value |
|---|---:|
| `resident_weight_bytes` | 14,297,675,776 |
| `workspace_bytes` | 4,929,305,616 |
| `session_count` | 4 |
| `session_workspace_bytes.*` | 1,232,326,404 × 4 |
| `decode_concurrency` | 1 |
| `lineage_children` | 0 |
| `session_serve_counts` | parent=6, protected_test=1, **child_a=0, child_b=0** |
| `load_ns` | 3,434,895,750 |

`health_after` serve_count=14, child_a/child_b still 0. The two worker
sessions are paid for and unused.

Workspace formula `SOURCE` `qwen38_hybrid_decode.rs:331-399` independently
recomputed this lane (activation 1,691,396 + ΔN 156,893,184 + GQA KV
`16*seq*4*256*4*2`). Matches the seq=128 receipt **exactly** and the live
8192 health field **exactly**:

| seq | workspace `total_bytes` | class |
|---:|---:|---|
| 128 | 175,361,796 | RECEIPT `QWEN38_SHARED_SESSIONS.json` + THIS_LANE |
| 2048 | 427,020,036 | THIS_LANE (= genesis_capacity’s 427_000_000 rounded) |
| 4096 | 695,455,492 | THIS_LANE |
| 8192 | 1,232,326,404 | THIS_LANE = live health |

Marginal RSS per attached session at seq=128: **173,703,168 B**, identical
on three deltas. RECEIPT `QWEN38_SHARED_SESSIONS.json`.

### 1.2 Process-pool children — fallback, private Metal copy

`RECEIPT` `GENESIS_CHILDREN_CAPACITY.json` / `GENESIS_POOL.md`.
`SOURCE` `lab/genesis_pool.py:49-53` `MEASURED_PHYS_FOOTPRINT_PER_CHILD =
15_970_043_138`, `MEASURED_SAFE_N = 3`.

| n | seq | phys_footprint each | IOAccelerator dirty | class |
|---:|---:|---:|---:|---|
| 1 | 128 | 15,658,626,568 | 14,476,197,888 | RECEIPT |
| 1 | 8192 | 16,721,080,608 | 15,533,162,496 | RECEIPT |
| 2 | 128 | ~15.663 GB × 2 | 14.476 GB × 2 | RECEIPT |
| 4 | 2048 | 15.96 GB × 4 | 14,727,856,128 × 4 | RECEIPT, **saturated, 751 MB swap, 0.81 GB free** |
| 8 | — | not launched | — | 8 × 15.659 = 125.3 GB vs 96 |

`artifact_pages_shared: false`. mapped-file clean ~4.5 MB, not the artifact.
`do_not_trust: ps RSS` (5.92–12.89 GB vs 15.96 GB phys at N=4).

This lane’s `proc_pidinfo(PROC_PIDTASKINFO)` on live pid 74869 returned
`pti_resident_size=13,320,192` (13.3 MB) against a 14.30 GB Metal body.
`footprint(1)` / `vmmap` Seatbelt-denied. **Do not use `pti_resident_size`
as the live body size.** Use the 11:10 health `resident_weight_bytes` +
workspace, and the 2026-08-16 `footprint` receipts.

Throughput knee RECEIPT (TEXT, no lock, DIRTY):

| n | complete-token | aggregate tok/s | vs N=1 |
|---:|---:|---:|---:|
| 1 | 37.855 ms | 26.4 | 1.00 |
| 2 | 55.3 ms | 36.1 | **1.37** |
| 3 | 79.4 ms | 37.8 | 1.43 |
| 4 | 112.0 ms | 35.7 | 1.35 + swap |

In-process 4-session aggregate was **worse** than one: 9.427 vs 26.653 tok/s.
RECEIPT `QWEN38_SHARED_SESSIONS.json`. Sharing weights saves memory. It does
not amortize DRAM.

---

## 2. Live box (this lane, no `ps`)

`hw.memsize` MEASURED 103,079,215,104 = **96 GiB**.

Three `vm_stat` snapshots this session. The governor reads
`free+inactive+purgeable`. Cap = `memory_lane_cap()` `SOURCE`
`tools/ascent_daemon.py:181-230`:

```
LANE_WORKING_SET_GIB = 6.0
GENERATION_RESERVE_GIB = 14.08   # only if CANDIDATE occupied
NO_SWAP_FLOOR_GIB = 4.0
TARGET_FILL = 0.90
cap = clamp(1..40, budget // 6)
```

Lineage `slots.CANDIDATE` MEASURED `null` → reserve = 0 this tick.

| t | free pgs | inactive | compressor occ. | swap used | available GiB | used GiB | cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 ~11:36 | 244431 | 1931302 | 26.96 GB | 1083 MB / 2 GB | 33.29 | 62.71 | **3** |
| T1 ~11:41 | 8927 | 1328814 | 33.11 GB | 1083 MB / 2 GB | 20.67 | 75.33 | **1** |
| T2 ~11:44 | 26618 | 2333926 | 10.20 GB | **3415 MB / 4 GB** | 36.43 | 59.57 | **4** |

Daemon log `MEASURED` (same morning, oscillating):

```
"hold": "18 OUR lanes live, at the memory-derived cap 3"
"hold": "11 OUR lanes live, at the memory-derived cap 4"
"hold": "9 OUR lanes live, at the memory-derived cap 4"
```

`pgrep -lf 'grok-run delegate'` this lane: **23** lines. Disk MEASURED
`df` 304–309 GiB free. `DISK_FLOOR_GIB=15`. Disk is not the bind.

Swap is engaged (T2 3.41 GB used). Absolute roofs are dirty. The
**formula** is the bind, not any one snapshot.

---

## 3. What 1.291 actually occupies

### 3.1 Packed ledger (the 4.3 GB number)

`MEASURED` `mixed-sub15-v1/PACK_REPORT.json`:

```
all_required_weight_artifact_bytes = 4,340,604,637
source_weight_elements             = 26,895,998,464
complete_physical_bpw              = 1.2910781930062503
                                 = 8 * 4340604637 / 26895998464
```

G0 resident weights RECEIPT+MEASURED 14,297,675,776.
Packed-only delta **9,957,071,139 B** (9.957 GB). This is the contract’s
“~4.3 GB, freeing ~10 GB” identity. It is the **disk ledger of packed
codecs**, not the Metal resident of the current Residual upload path.

On-disk tree MEASURED `os.walk`: **15,473,851,850 B**. `tensors/` still
holds the reconstructed HQ30UQ4 vehicle (14.308 GB) plus `packed/` rice
(1.165 GB). Promoting the directory as-is does not shrink disk and does
not shrink the live body. Current `load()` `SOURCE`
`qwen38_hybrid_decode.rs:508-513` takes the uniform-Q4 path when
`catalog.hq38m20` is absent (it is). That is the expand-to-Q4 confound.

### 3.2 Native Metal resident (ESTIMATED, header-derived)

HGRAVR02 `upload_mixed` `SOURCE` `:1049-1114` expands rice **at load**
into CSR `u32` indices + row_ptr and uploads those plus binary
signs/scales plus residual signs. Rice stream is not a GPU buffer.
CSR is then **read every token**.

Attention rice headers MEASURED this lane (304 `HGRAVR02` files):

```
outlier_count sum = 144,756,064     # matches g1-sub15-native-gap
CSR indices       = 579,024,256
row_ptr           = 5,393,600       # (rows+1) per tensor
sign+scale+resid  = 1,035,909,712
native attn upload= 1,620,327,568   vs packed 1,165,098,376  (+455,229,192)
```

MLP `up_proj` from `mixed-2p0-v1` segment payloads MEASURED (64
`HGRAVR02`, same bytes mixed-sub15 reuses):

```
outlier_count sum = 114,085,120
CSR indices       = 456,340,480
row_ptr           = 4,456,704
native up upload  = 1,277,218,496   vs packed 918,036,000   (+359,182,496)
```

Putting the other organs at their packed payloads (Binary / HGRAVS /
HQ30UQ4 / f32; same peel G0 already applies):

```
native weights ESTIMATED = 5,155,016,325 B   (5.155 GB)
                         = gate 802,177,344
                         + up_native 1,277,218,496
                         + down 93,847,197
                         + attn_native 1,620,327,568
                         + embed 675,430,440
                         + lm_head 675,430,440
                         + small 10,584,840
vs G0 14,297,675,776
saved  9,142,659,451 B   (9.143 GB)
CSR tax vs packed ledger +814,411,688 B
```

Not a live `footprint`. Cheapest experiment that would make it MEASURED:
after C1–C3, one `footprint -p` of a native `genesis-resident` under the
GPU lock. Other lane owns that.

`resident_bytes()` today reports the Q4 catalog. A successful **load** of
mixed-sub15 without C1 is a Q4 load. A successful native load is not
coherence.

### 3.3 mixed-2p0 (the 2.0 “bootstrap”)

`MEASURED` `mixed-2p0-v1/PACK_REPORT.json`:

```
complete_physical_bpw = 2.0855934079220506
all_required_weight_artifact_bytes = 7,011,764,637
mlp_physical_bpw = 0.84805   # same gate/up/down as sub15, including down 0.1316
nonmlp_physical_bpw = 4.25014
```

`GENERATE.json` engine `mlx_lm_weights_overwritten_from_mixed_pack`,
output `<|im_end|>` immediately. Confounded vehicle, not a native-codec
verdict. Same crushed `down_proj`. Attention left at ~Q4. This is not a
safer bootstrap of 1.291; it is a fatter sibling of the same MLP.

Native-2.0 weight ESTIMATED ≈ 7.38 GB (MLP native-up + Q4 attention +
oracle embed/lm/small). Save vs G0 ≈ **6.92 GB**.

---

## 4. Converted headroom

Assume C1–C3 land and native weights are the 5.155 GB ESTIMATE. Host
overhead taken from RECEIPT N=1 seq=8192:

```
host = 16,721,080,608 − 14,297,675,776 − 1,232,326,404
     = 1,191,078,428
```

| object | G0 | G1 native PROJECTED |
|---|---:|---:|
| 1-proc phys seq=128 | 15,658,626,568 RECEIPT | 6,515,967,117 |
| 1-proc phys seq=8192 | 16,721,080,608 RECEIPT | 7,578,421,157 |
| live 4-session body seq=8192 | 20,418,059,820 DERIVED | 11,275,400,369 |
| process-child IOAccel seq=128 | 14,476,197,888 RECEIPT | ~5.33e9 |

### 4.1 Process-pool children

```
9.143 GB saved  <  15.659 GB G0 child     → +0 G0 children
9.143 GB saved  >  6.516 GB G1 child      → +1 G1 child  (quiet box)
today swap 3.41 GB, free pages 0.44 GB    → +0 of either
```

On an empty 96 GB box G0 `safe_n=3` (MEASURED, N=4 swapped). Linear
residency 3 × 15.66 / 6.52 ≈ 7.2 → **safe_n ~6–7 ESTIMATED**, not
measured. Throughput knee stays ~N=2: extra processes still partition
the same DRAM. Density may move the knee slightly. Not claimed.

### 4.2 In-process sessions

```
+7 sessions at seq=8192   (9.143e9 / 1,232,326,404)
+53 at seq=128            (9.143e9 / 173,703,168)
```

Admission already refused **144** seq=128 sessions when free was 38.0 GB
(`QWEN38_ADMISSION_REFUSE_SESSIONS.json`, cost 242,470,660 B/session
including slop, reserve 4 GiB). That refuse is a batch-oversub test, not
the live bind. Live bind: 4 attached, 2 unused, `decode_concurrency=1`.
More sessions are research slots that **step serially**. They are not
workers in the TPS sense.

### 4.3 Grok-run lane cap (this is the memory convert that is real)

`LANE_WORKING_SET_GIB=6`. Freeing 9.143 GB ≈ 8.51 GiB into
`available`:

| snapshot | cap now | cap after 8.51 GiB | Δ |
|---|---:|---:|---:|
| T0 | 3 | 5 | +2 |
| T1 | 1 | 3 | +2 |
| T2 | 4 | 5 | +1 |

So the 1.291 body, if it actually releases ~9 GB, buys **one or two**
additional 6 GiB research lanes under the daemon that is currently
holding with `"N OUR lanes live, at the memory-derived cap C"`. It does
not legalize the 18–23 already-live lanes.

When a CANDIDATE is nominated, the same function subtracts **14.08 GiB**
before computing cap. Dual-seating G0 (20.4 GB PROJECTED) + G1 (11.3 GB
PROJECTED) on today’s used ~60–75 GiB is a swap event. Promotion by
**replace** (restart) is the only dual-seat-safe path on this box.

### 4.4 Expand-to-Q4 path (current loader)

Headroom **0**. Resident stays 14,297,675,776. Restart cost paid for
nothing. This is how mixed-sub15 already “loaded” for the INCOHERENT
receipt.

---

## 5. What currently binds child / worker capacity

Ranked, live, not a spec sheet.

**B1. GPU lock + decode ceiling 1.** MEASURED this window:
`/tmp/hawking-gpu-lane.lock` owner `genesis-resident:parent` pid 74869.
Propose pid 96580 in flight, `--max-new-tokens 2600`.
`decode_concurrency=1` on the live health JSON.
`concurrent_decode_ceiling: 1` RECEIPT `GENESIS_RESIDENT_BODY.json`.
In-process 4-session 9.427 < 26.653 tok/s. Extra sessions do not
parallelize generate.

**B2. `memory_lane_cap` on OUR grok-run lanes.** MEASURED daemon hold
strings. Formula §2. 23 `grok-run delegate` lines live. This is why new
ascent targets are not launching (`generated=0`, `pending=0` on those
ticks), not because Genesis ran out of session structs.

**B3. Organism propose wall.** Daemon `GENESIS_PROPOSE_MAX_NEW_TOKENS=2600`
`SOURCE` `ascent_daemon.py:407-411` (900 was emptying mid-`<think>`).
Live propose prompt names the current bottleneck: capsule **prefill
79.00–79.76 s / 1558 prompt tokens = 95% of that wall**. Plus 2600
decode × 39.326 ms = **102.25 s** DERIVED. G0 propose cycle ~182 s →
~19.8 proposes/h if the GPU were dedicated to proposes. It is not
(lock shared with any TIMING / protected generate).

**B4. Promotion / human drain.** MEASURED
`PROMOTION_QUEUE.json` n=334, `promoted` Counter `{False: 334}`.
disposition: MERGE_READY 134, NO_REPORT_MANUAL_REVIEW 184,
NEEDS_COMPOSITION 15. model: q80=274, qwen38=45, dsv4f=15. Science is
not memory-blocked; it is unconsumed.

**B5. Process-pool Metal copies.** MEASURED N=4 swap. Relevant only if
someone spawns `ascension_qwen38_hybrid_greedy` children. The live
organism does not (`lineage_children=0`).

**Not binds:** disk (304+ GiB). In-process session count (2 unused).
`MAX_CONCURRENT=10` (overridden by `memory_lane_cap`). The 14.3 GB body
size as a session limiter (sessions cost 0.17–1.23 GB).

---

## 6. TPS multipliers — PROJECTED, not complete-token

Live G0 `MEASURED` `g1-baseline-remeasure.md` (do not re-derive):

```
TOKEN_NS = 39,326,090    median of 6 paired decode-phase means, spread 1.83%
TPS      = 25.4284       = 1e9 / 39326090
capability 6/6 oracle-32
```

100 TPS needs 3.9326× that complete-token. Contract fact.

Sealed addressing RECEIPT G024: 21,293,103 ns on 13,611,663,360 B.
Non-addressing remainder:

| remainder source | ns | TPS if addressing → 0 |
|---|---:|---:|
| G024 wall 35,227,917 − addr | 13,934,814 | **71.76** |
| live 39,326,090 − addr | 18,032,987 | **55.45** |

Byte reduction cannot cross those floors. The 1.5 BPW “~45 TPS”
sentence is this remainder plus linear addressing:

```
1.5 / 4.252735126866492 × 21,293,103 = 7,510,379 ns PROJECTED addr
+ 13,934,814 = 21,445,193 ns → 46.63 TPS
```

### 6.1 mixed-sub15 native traffic (not the packed ledger)

CSR indices move every token. Per-token GEMV stream ESTIMATED:

```
4,469,001,045 B = gate + up_native + down + attn_native + lm_head
ratio vs 13,611,663,360 = 0.32832
addr PROJECTED @ sealed 639.25 GB/s = 6,990,982 ns
```

Assumption: same GB/s as `geo_tpr64_tg128` unique-once. **False** if the
bound 2048-col Binary/CSR tiles (`g1-sub15-native-gap` C3) are what
dispatch. Then native can be slower than G0. C3 is a promotion
prerequisite, not a polish.

| remainder held | TOKEN_NS PROJECTED | TPS | × live 25.4284 |
|---|---:|---:|---:|
| G024 13.935 ms | 20,925,796 | 47.79 | **1.88×** |
| live 18.033 ms | 25,023,969 | 39.96 | **1.57×** |

PACK_REPORT `projection` uses `ms = 1.415 + 36.802 × (bpw/4.2527)` →
12.59 ms / **79.44 TPS**. That 1.415 ms is host/sync, not the 14–18 ms
compute remainder. **KILLS that formula as a token claim.**

### 6.2 mixed-2p0 native traffic PROJECTED

```
6,693,752,517 B = gate + up_native + down + Q4 attention GEMV 3,845,079,040 + lm_head
ratio 0.4918 → addr 10.47 ms + 13.93 ms = 24.41 ms → 41.0 TPS = 1.61× live
```

Worse than 1.291, same MLP coherence risk, same native-load gap on
attention (here Q4, so attention would at least hit the known G0 kernel).

### 6.3 Propose-loop multiplier

G0 propose ~182 s (79.4 prefill named + 102.2 decode DERIVED) → 19.8 / h
if exclusive. At 1.88× both legs (same unique-once stream in prefill):
~97 s → 37.2 / h. At 1.57×: ~116 s → 31.1 / h.

That is the organism’s cognition rate, **not** the G1 wave’s experiment
rate. The wave is 23 CPU/analysis lanes gated by `memory_lane_cap` and a
334-deep undrained queue.

---

## 7. How much of the research loop is model-bound

| loop | bound | evidence |
|---|---|---|
| parent `propose` (ascent harvest) | **model + GPU serial** | 79 s prefill named; 2600 decode; one socket; one lock |
| protected capability / complete-token | **GPU serial** | lock; ceiling 1 |
| TEXT process-pool evals | GPU bandwidth then memory | knee N=2; N=4 swap |
| Superwave G1 CPU lanes | **memory_lane_cap + human** | hold strings; 23 live; 334 unpromoted |
| promotion into CURRENT | **human / composition** | 0 promoted, 184 MANUAL_REVIEW, 15 NEEDS_COMPOSITION |
| disk | not bound | 304 GiB free |

Honest split of “experiments per hour”:

- Density-driven TPS helps **only** the generate-shaped slice (propose,
  TEXT eval, TIMING). A 1.6–1.9× token is a 1.6–1.9× on that slice
  minus fixed daemon harvest. It does not multiply the CPU wave.
- The live ascent tick is **not** waiting on 25 TPS. It is waiting on
  `memory_lane_cap` and then on a 1800 s `genesis_proposes` that is
  itself prefill-bound.
- 134 MERGE_READY rows will not move faster because the body is 1.291.

Second-order: a faster token is more cognition per seated second. That
matters after the queue drains and the organism is the experimenter
again. It is not why this wave is stalled.

---

## 8. Restart / rebind cost

| item | value | class |
|---|---|---|
| live cache-hot load | 3.435 s (`load_ns=3434895750`, also resident.log) | MEASURED |
| established box-cold load | 50 s | RECEIPT `GENESIS_RESIDENT_BODY.json` |
| 4× attach at 8192 | included in that load | SOURCE |
| capability re-qualify | 6-rep + arithmetic, minutes, needs lock | MEASURED this morning |
| silent Q4 fallback if C1 absent | resident stays 14.30 GB | SOURCE load() |
| dual-seat while CANDIDATE set | cap subtracts 14.08 GiB; two bodies ~31.7 GB PROJECTED | SOURCE + DERIVED |
| swap already on | 3.41 GB used T2 | MEASURED |

A restart that lands the Q4 vehicle is a 3.4–50 s outage that buys
zero headroom and zero TPS. That is churn.

---

## 9. Promotion bar

Evaluate **mixed-sub15-v1 native**, not a newly invented 2.0.

| gate | minimum | why |
|---|---|---|
| P0 vehicle | native HQ38M20 + C3 K-complete Binary/CSR bind. No expand-to-Q4, no MLX overwrite | two INCOHERENT verdicts were this confound |
| P1 coherence | oracle-32 match + arithmetic emit on the **native** path | a load is not coherence |
| P2 residency | `footprint` / `resident_bytes()` ≤ 6.0 GB weights (ESTIMATE 5.155; 8 GB drop vs 14.30) | otherwise the 10 GB story is false |
| P3 speed | complete-token MEASURED ≥ **1.40×** live 25.4284 (TOKEN_NS ≤ 28,090,064) | pays restart on the propose loop; 1.57–1.88 is the PROJECTED band if C3 holds 639 GB/s |
| P4 not enough | PACK_REPORT 79 TPS, gpu_proxy 56–63, any expand-path wall | not tokens |

Fail any of P0–P2 → do not promote.
Fail P3 after P0–P2 pass → memory-only promote is still legal **after**
a `footprint` receipt, because +1–2 lane-cap slots and a cheaper
generation reserve are real. Do not advertise a TPS win.

**2.0856 BPW mixed-2p0:** fail P0 (no native Q38 attention story beyond
Q4), fail the standing “evaluate 1.291 first”, save only ~6.9 GB, PROJECTED
1.61× < the 1.291 band. Not a bootstrap. Not a restart.

**Anything > 2.0 BPW:** not the target (hierarchy D).

100 TPS is a campaign target, not this artifact’s promotion bar.
Remainder 13.9–18.0 ms already forbids it.

---

## 10. KILLS / REOPEN_IF

| ID | verdict | REOPEN_IF |
|---|---|---|
| K1 | “~10 GB frees N extra G0 process-children” | FALSIFIED (9.14 < 15.66). A genome whose process-child `phys` ≤ 8 GB **and** a quiet box. |
| K2 | “live child capacity is memory-bound at the 14.3 GB body” | FALSIFIED. 4 sessions attached, 2 unused, decode=1. Reopen if child_a/child_b are actually scheduled and then refuse-attach. |
| K3 | “2.0 BPW bootstrap is worth the restart” | FALSIFIED as evolutionary value. Reopen if 1.291 native is incoherent **on the native path** and a ≤2.0 native pack is coherent with `footprint` ≤ 6 GB and ≥1.40× token. |
| K4 | PACK_REPORT 79.44 TPS | FALSIFIED formula (1.415 ms “fixed” vs 14–18 ms remainder). Reopen never; throw the formula out. |
| K5 | “byte cut alone → 100 TPS” | FALSIFIED (remainder 55–72 TPS). Reopen if a MEASURED complete-token remainder < 8 ms on this genome. |
| K6 | “more in-process sessions raise tok/s” | FALSIFIED (9.427 vs 26.653). Reopen only with a weight-stationary GEMM + a single-stream batch source. |
| K7 | `pti_resident_size` / `ps` RSS as body size | FALSIFIED (13 MB vs 14.3 GB). Use `footprint` / health `resident_weight_bytes`. |

Negative that remains first-class: mixed-sub15 is **not loadable native
today**. Headroom is conditional on the other lane’s C1–C3. This lane
did not close that gap.

---

## 11. Evidence

### 11.1 This-lane commands (abridged; full stdout in session logs)

```
$ sysctl hw.memsize hw.ncpu
hw.memsize: 103079215104
hw.ncpu: 28

$ pgrep -lf genesis-resident
74869 .../genesis-resident --artifact-root .../uniform-q4-v1 ... --max-seq-len 8192 --max-new-tokens 900

$ python3 -c 'proc_pidinfo(74869, PROC_PIDTASKINFO)'
pid 74869: rss=13320192 (0.013 GB) vsz=520.880 GB threads=4

$ memory_pressure -Q
System-wide memory free percentage: 65–66%

$ sysctl vm.swapusage
# T0: total=2048.00M used=1082.62M
# T2: total=4096.00M used=3414.69M

$ df -h /System/Volumes/Data
# T2: 304Gi avail, 67% used

$ pgrep -lf 'grok-run delegate' | wc -l
      23
```

Live health excerpt (`/tmp/g1-baseline-remeasure/resident_measure_v2.json`):

```
health_before.pid                    74869
health_before.resident_weight_bytes  14297675776
health_before.workspace_bytes        4929305616
health_before.session_workspace_bytes * 1232326404
health_before.decode_concurrency     1
health_before.lineage_children       0
health_before.session_serve_counts   {parent:6, child_a:0, child_b:0, protected_test:1}
headline_token_ns                    39326090.03225806
headline_tps                         25.428411499331073
```

Workspace recomputation (this lane, constants from
`qwen38_geometry.rs:20-41` + `qwen38_hybrid_decode.rs:331-399`):

```
activation 1691396   deltanet 156893184   (matches RECEIPT)
seq=128  total=175361796
seq=2048 total=427020036
seq=8192 total=1232326404   (= live health per session)
```

Native Residual headers (this lane):

```
attn 304 files  outlier_sum=144756064  native_upload=1620327568
up   64  files  outlier_sum=114085120  native_upload=1277218496
native_weights ESTIMATED=5155016325
saved vs G0=9142659451
```

### 11.2 Receipts / source (not re-timed)

- `receipts/ascent-2026-08-16/GENESIS_CHILDREN_CAPACITY.json`
- `receipts/ascent-2026-08-16/GENESIS_POOL.md`
- `receipts/ascent-2026-08-16/GENESIS_RESIDENT_BODY.json` (`concurrent_decode_ceiling=1`, load footprint 15,385,816,488, IOAccel 14,473,019,392, cold 50 s)
- `receipts/ascent-2026-08-16/QWEN38_SHARED_SESSIONS.json` (marginal 173,703,168; 26.653 vs 9.427 tok/s)
- `receipts/ascent-2026-08-16/QWEN38_ADMISSION_REFUSE_SESSIONS.json` (144 refused)
- `receipts/ascent-2026-08-16/TOKEN_NS_QWEN38.json` / `G024_QWEN38_TOKEN_NS.json` (addr 21,293,103 ns)
- `workspace/superwave/g1/g1-baseline-remeasure.md` (live 39,326,090 / 25.4284)
- `workspace/superwave/g1/g1-sub15-native-gap.md` (C1–C3; 2048-col bind)
- `workspace/superwave/g1/g1-residency-reuse.md` / `g1-resident-harvest.md`
- `tools/ascent_daemon.py:181-230,407-411`
- `tools/agentos/genesis_body/src/main.rs:3-6,30,569-612`
- `lab/genesis_pool.py:49-53`
- `crates/hawking-core/src/model/qwen38_hybrid_decode.rs:331-399,508-513,1049-1114`
- live `PROMOTION_QUEUE.json` n=334 promoted=0
- live `workspace/ops/ascent-daemon.log` hold strings
- live `mixed-sub15-v1/PACK_REPORT.json`, `mixed-2p0-v1/PACK_REPORT.json`

---

## 12. Recommendation

Do **not** restart onto a 2.0 BPW bootstrap. Do **not** restart onto
mixed-sub15 via the Q4 vehicle. Keep G0 seated until native load exists,
then qualify P0–P3 on mixed-sub15-v1. That is the only density
promotion on this box that converts into a measured body-sized slab
(~9.1 GB) and a plausible 1.6–1.9× token. It still leaves the remainder
floor and the GPU-serial / queue-drain binds untouched.

---

# Completion report

```
STATUS
SUPPORTED

CLAIMS
C1. Live G0 body pid 74869, uniform-q4-v1, weight_bytes=14297675776, 4 sessions × 1232326404 B at seq=8192, decode_concurrency=1, child_a=child_b serve 0. MEASURED health_before in /tmp/g1-baseline-remeasure/resident_measure_v2.json; pgrep argv.
C2. Process-pool child phys is 15.659 GB (seq=128) / 16.721 GB (seq=8192), private IOAccel, safe_n=3, N=4 swaps, knee 1.37× at N=2. RECEIPT GENESIS_CHILDREN_CAPACITY.json. ps/pti_rss untrusted (this lane 13.3 MB vs 14.3 GB).
C3. mixed-sub15 packed 4340604637 B = 1.2910781930062503 BPW MEASURED. Native Metal resident ESTIMATED 5155016325 B after CSR expand (attn 144756064 + up 114085120 outliers MEASURED from headers). Save vs G0 9142659451 B, not 10.0 GB packed-delta 9957071139 B.
C4. That 9.14 GB converts to +0 G0 process-children, +1 G1 process-child on a quiet box / +0 today (swap 3.41 GB), +7 seq=8192 sessions or +53 seq=128 sessions, +1..+2 memory_lane_cap slots. DERIVED. Expand-to-Q4 converts to 0.
C5. Current bind is GPU serial + memory_lane_cap + 79 s propose prefill/2600 decode + 334-deep undrained queue, not session-memory. MEASURED lock/pgrep/daemon hold/PROMOTION_QUEUE.
C6. TPS PROJECTED 1.57×–1.88× (39.96–47.79) if native traffic 4.469 GB hits 639 GB/s and remainder holds. 100 TPS impossible from density (remainder floor 55.5–71.8). PACK_REPORT 79.44 FALSIFIED.
C7. 2.0856 BPW mixed-2p0 is generation churn: same crushed MLP, fatter attention, ~6.9 GB save, PROJECTED 1.61×, confounded generate. 1.291 native is the only promotion candidate; bar §9.

EVIDENCE
§11 + files listed in §11.2. Workspace identity: this-lane python == QWEN38_SHARED_SESSIONS.workspace == live health session_workspace_bytes. CSR identity: this-lane rice headers == g1-sub15-native-gap 144756064.

CHANGES
workspace/superwave/g1/g1-resident-headroom.md (new)

TESTS
see runner output in the turn message

RISKS
Native 5.155 GB is ESTIMATED (header→buffer map), not footprint. C3-unfixed CSR tiles can invert the TPS projection. Box is swapping; absolute GiB snapshots move. Propose 79 s is named in the live prompt, not re-timed here.

UNRESOLVED
Live footprint of pid 74869 (Seatbelt). Native mixed-sub15 footprint (needs C1–C3 + GPU lock). Complete-token A/B of any G1 pack (other lane). Whether child_a/child_b will ever be scheduled.

NEXT
Other lane closes C1–C3. Then footprint + native coherence + complete-token A/B against 25.4284. Promote only if §9 holds. Do not invent a 2.0 body.
```
