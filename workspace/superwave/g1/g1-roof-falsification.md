# G1 roof falsification — Qwen3.8 / Genesis

STATUS: FALSIFIED

Lane did not run GPU, Metal, or inference. Every timing number below is
**cited** from a receipt or from live source constants. Label is on every
figure: MEASURED (receipt GPU timestamps), CITED (this lane recomputed from
those fields), PROJECTED (receipt scaled a measured time by a BPW ratio),
ESTIMATED (datasheet ÷ bytes), POLICY (not physics).

A component probe is labeled COMPONENT. A closed complete-token identity is
labeled TOKEN. A load-only `bytes/bandwidth` number is labeled LOAD_ONLY.

---

## 0. G0 numbers — contract claim vs receipts

Contract historical G0 (unverified): complete BPW ~4.2527, TPS ~26.4,
TOKEN_NS ~37,900,000.

| field | contract claim | receipt A | receipt B | receipt C |
|---|---|---|---|---|
| BPW | ~4.2527 | 4.252735126866492 TOKEN | 4.2527 TOKEN | 4.252735126866492 TOKEN |
| TOKEN_NS | ~37,900,000 | 35,227,917 TOKEN | 35,227,918 TOKEN | 38,216,792 TOKEN |
| TPS | ~26.4 | 28.386577 CITED | 28.386577 CITED | 26.1665 MEASURED |

- A: `receipts/ascent-2026-08-16/TOKEN_NS_QWEN38.json` L2–L3 `TOTAL_TOKEN_NS=35227917`, `TOTAL_GPU_BUSY_NS=33912333`. Vehicle `qwen38-27b/uniform-q4-v1`. Label `DIRTY_ENGINEERING`.
- B: `receipts/ascent-2026-08-16/GENESIS_LINEAGE_CURRENT.json` L12–L14 `representation_bpw=4.2527`, `complete_token_ns=35227918`, `tps=28.386576805362157`. Install of `genesis-qwen38-g0`.
- C: `receipts/ascent-2026-08-16/QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json` L34–L39. 6-rep median of per-rep medians, 31 steady decode steps, includes tokenizer+epilogue. `complete_tps=26.1665`.
- ICB successor (not G0 body): `QWEN38_FIXED_OVERHEAD_DELETED.json` L80–L82 `after_headline_ms=36.683916` → 27.2599 TPS. Same artifact.

A and C are different complete-token definitions, both MEASURED, both
`DIRTY_ENGINEERING`. A is encode+submit+wait identity. C adds tokenizer decode
and commit epilogue. Contract 26.4 / 37.9 ms tracks C, not the lineage install.
This lane did not remeasure. Treat G0 as **two receipt authorities**, not one
number.

`UNIFORM_Q4_V1_BPW` is live in
`crates/hawking-core/src/model/qwen38_token_ns_ledger.rs` L30.

---

## 1. Inventory of stated ceilings

| ID | claim | physical op claimed | figure origin | still live? |
|---|---|---|---|---|
| R1 | Honest decode ceiling = 411.51 GB/s | unique-once sequential read-reduce | MEASURED Q80 control, 512 MiB point | YES in Rust ledgers |
| R2 | Qwen3.8 is 97.6% / 98.7% of that roof | GEMV bytes ÷ total GPU ns ÷ 411.51 | CITED mix of wrong bytes, wrong time, wrong ceiling | YES in G024 / ROOF_AND_RUNGS.md |
| R3 | Qwen3.8 roof = 30.21 TPS at 13.622 GB | LOAD_ONLY 13.622e9 / 411.51e9 | ESTIMATED from R1 | YES in RUNG_TARGETS / ROOF_AND_RUNGS.md |
| R4 | Density is nearly the only lever | inference from R2/R3 | ESTIMATED | YES in G024 action; already recanted in CORRECTION |
| R5 | Published peak 819 GB/s is a DRAM floor | datasheet ÷ bytes | SPEC SHEET | YES in PHYSICAL_FLOOR / FS law / TERMINAL_TARGET / ARCH_CENSUS |
| R6 | Measured control band 560–647 GB/s | sequential DRAM-row / reuse probe | MEASURED, reuse-friendly | YES in SUPERWAVE_STATE header |
| R7 | Decode-shape control 320–411 GB/s | 98 CB, 10-of-512 gather, unique once | MEASURED Q80 topology | YES in SUPERWAVE_STATE L74; Q80-only |
| R8 | fs/weight floor = 152.6252 × BPW | (BPW/8) / 819e9 × 1e15 | ESTIMATED from R5 | YES in FS_PER_WEIGHT_LAW / SUPERWAVE_STATE L7 |
| R9 | Sub-100 fs requires BPW < 0.6552 | R8 at efficiency=1 | ESTIMATED | YES in FS_PER_WEIGHT_LAW L40; G013 v1 then v2 retracted the band |
| R10 | 48 × (2–5 µs) dispatch floor = 100–240 µs | 1 Metal dispatch/layer launch latency | ESTIMATED, Q80 48-layer | YES in PHYSICAL_FLOOR.json L33–34 |
| R11 | 1.229 ms fixed host overhead | encode+wait−gpu+submit+tok+epilogue | MEASURED then treated as sticky | SUPERSEDED by ICB |
| R12 | Reconstruction / density costs time | Q4/rice decode vs f32 | COMPONENT, transferred from Q80 serial extract | KILLED at tpr64 |
| R13 | 14.414 ms DRAM floor at 4.0 BPW | 23.611e9 × 4/8 / 819e9 | ESTIMATED, omitted lm_head | KILLED by ACTIVE_BUDGET |
| R14 | 100 TPS physically impossible at 3.0 BPW (ceiling 92.5) | R13 family at 3.0 / 819 | ESTIMATED | YES in ARCH_CENSUS HARD_CONSEQUENCE_1 |
| R15 | 20 ms DSV4F gate vs 24.98 ms floor | 10.280 GB / 411.51 | ESTIMATED from R1 | YES in genesis_tournament.py C7 |
| R16 | Doctor6 / one-bit 1.0 (abs 1.5) BPW ceiling | complete bits / parent weights | POLICY | YES, not a speed roof |
| R17 | Coherence floor ~2.0856; sub-1.5 incoherent | rice_q1 on attention | MEASURED capability | YES, codec-family conditioned |
| R18 | Q4 GEMV genome roof 699.57 addr / 666.68 full / 639.25 sealed | unique-once DRAM of `geo_tpr64_tg128` | MEASURED COMPONENT + CITED sealed TOKEN split | YES in HONEST_ROOF / roof_rungs.py |
| R19 | unique_once at 13.6 GB plateaus ~375.65 GB/s | chunked uint32 unique_once | MEASURED COMPONENT | YES in HONEST_ROOF |
| R20 | 401-organ catalog 530.65 addr / 505.81 full | same kernel, production organ tiling | MEASURED COMPONENT | YES in HONEST_ROOF |
| R21 | THE_SINGLE_SHARED_BLOCKER: packed matvec ~0.4% of 560–647 | Q80 mixed occupancy | MEASURED Q80, mis-applied as shared | YES in TERMINAL_TARGET L75 |
| R22 | No kernel trick escapes the bandwidth floor | batch=1, 1 flop/byte, zero reuse | ESTIMATED law | YES in SUPERWAVE_STATE L38–39 and FS_PER_WEIGHT_LAW L42 |
| R23 | terminal_head 1763 GB/s | lm_head bytes / isolated FMA remainder | MEASURED mismatch | live in TOKEN_NS row; not a real roof |

---

## 2. R1 — 411.51 GB/s "honest decode ceiling"

**Physical op claimed:** unique-bytes-once DRAM traffic of a decode token.

**What was actually measured:** `q80_decode_shape_unique_once` at **512 MiB**,
full occupancy 15360 threads, median 411.51358589633037 GB/s
(`receipts/ascent-2026-08-16/Q80_DECODE_SHAPE_BANDWIDTH.json` L222, L328–335,
L975). Same sweep 1024 MiB = 301.63405407683126 GB/s (L976). Same sweep 256 MiB
= 410.5302328426687 GB/s.

**Regime match:** NO.

- Working set 512 MiB, not Qwen3.8's 13.6 GB.
- Kernel is sequential unique_once reduce, not `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`.
- `nbytes` is `uint32`; a single dispatch cannot name 13.6 GB
  (`HONEST_ROOF_WEIGHT_ADDRESSING.json` L79, L1360–L1361).
- Cherry-pick: 1024 MiB point discarded.

**Same control, later same-box rerun at the real size:** unique_once at
13.611663360 B = **375.6517695934827 GB/s** (HONEST_ROOF L1314, L1345). Did
not reproduce 411.51 (L1347). Plateaus ~375 from 2 GiB through 13.6 GB.

**Still hardcoded as the Qwen3.8 floor:**

```
crates/hawking-core/src/model/qwen38_token_ns_ledger.rs:28
    pub const HONEST_DECODE_CEILING_GB_S: f64 = 411.51;

crates/hawking-core/src/model/qwen38_token_ns_ledger.rs:477
    let floor = bandwidth_floor_ns(bread.saturating_add(bwrite), HONEST_DECODE_CEILING_GB_S);
```

Same constant: `qwen80_mixed_token_ns_ledger.rs:24`,
`tools/ascent/roof_rungs.py` L54–55 (kept as Q80 historical alias
`Q80_HISTORICAL_UNIQUE_ONCE_GB_S`, default of `load_only_floor_ns`),
`tools/genesis_tournament.py` L69, L112, L121.

**The ledger already beats its own floor.**
`TOKEN_NS_QWEN38.json` L42–L55:

```
component: weight_addressing
bytes_read: 13611663360
ns_per_token: 21293102.524500456
effective_gb_s: 639.2522341137478
theoretical_lower_bound_ns: 33077357.439673398
measured_over_floor: 0.6437365065614716
```

CITED: `13611663360 / 411.51 = 33077357.44` ns. Measured 21.293 ms is
**0.6437 of the floor**. A floor you beat is not a floor.
`honest_roof.rs` L1164–L1168 asserts this.

**Genome change:** replace unique_once 512 MiB with Q4 grouped GEMV at 13.6 GB
and the number becomes 639–700 GB/s (R18), or 376 GB/s if you keep unique_once
and only grow the working set (R19).

**Verdict: FALSIFIED as a Qwen3.8 / current-genome roof. Survives only as
"Q80 historical 512 MiB unique_once control."**

---

## 3. R2 — 97.6% / 98.7% of roof

G024 (`G024_QWEN38_TOKEN_NS.json` L28–L30):

```
achieved_gb_s_production: 401.6
honest_decode_ceiling_gb_s: 411.51
pct_of_ceiling: 97.6
```

ROOF_AND_RUNGS.md L13: `406.20` GPU GB/s, `98.71%` occ, `30.21` roof tok/s.

Reproduction (`honest_roof.rs` L235–L252; HONEST_ROOF L75, L1378):

```
13_618_141_856 / 33_912_333 ns / 411.51 = 0.975842643
```

Three errors, all named in the same receipt:

| piece | used | should have been | why |
|---|---|---|---|
| bytes | 13,618,141,856 geometry-active | 13,611,663,360 GEMV codes+scales | norms + embed row are not `weight_addressing` |
| time | 33,912,333 ns production GPU | 21,293,102.5 ns addressing | DeltaNet/GQA/norms/SwiGLU/KV/FMA do not stream those bytes |
| ceiling | 411.51 | this genome's Q4 GEMV roof | R1 |

Correct attribution: `13_611_663_360 / 21_293_102.5 = 639.252234849` GB/s
(HONEST_ROOF L63, L1328). TOKEN-split, not a new GPU run of production.

**Verdict: FALSIFIED.** The 97.6% figure is a denominator artifact.

---

## 4. R3 / R4 — 30.21 TPS roof; "density is the only lever"

R3 is LOAD_ONLY `13.622e9 / 411.51e9 = 33.10 ms = 30.21 TPS`
(`QWEN38_RUNG_TARGETS.json` L12, ROOF_AND_RUNGS.md L13, L39:
"A is unreachable until bytes drop").

CITED load-only TPS at 13,611,663,360 B:

| ceiling | origin | load_only_ns | load_only_tps |
|---|---|---|---|
| 411.51 | R1 unique_once 512 MiB | 33,077,357 | 30.23 |
| 375.65 | R19 unique_once 13.6 GB MEASURED | 36,234,791 | 27.60 |
| 505.81 | R20 catalog full MEASURED | 26,910,625 | 37.16 |
| 530.65 | R20 catalog addr MEASURED | 25,650,709 | 38.99 |
| 639.25 | R18 sealed addressing CITED | 21,293,103 | 46.96 |
| 666.68 | R18 single-GEMV full MEASURED | 20,417,041 | 48.98 |
| 699.57 | R18 single-GEMV addr MEASURED | 19,457,084 | 51.40 |
| 819 | R5 datasheet | 16,619,858 | 60.17 |

Rung A (≥50 TPS) is **load-only reachable** at the single-GEMV addr roof
without a byte drop. The "A unreachable until bytes drop" sentence is
conditioned on R1.

Complete-token is not load-only. Holding the non-addressing remainder fixed
(TOKEN_NS rest 13,934,815 ns; COMPLETE_WALL rest 16,923,690 ns) and replacing
only the addressing bandwidth:

| addressing roof | TOKEN_NS tot / TPS | COMPLETE_WALL tot / TPS |
|---|---|---|
| 411.51 | 47.01 ms / 21.27 | 50.00 ms / 20.00 |
| 639.25 (current sealed) | 35.23 ms / 28.39 | 38.22 ms / 26.17 |
| 699.57 | 33.39 ms / 29.95 | 36.38 ms / 27.49 |
| 819 | 30.55 ms / 32.73 | 33.54 ms / 29.81 |

Closing the catalog→single-GEMV gap does **not** put a complete token on
rung A. R4 ("density is nearly the only lever", G024 L63–L64, RUNG_TARGETS L17)
is the inference CORRECTION_ROOF_IS_CONDITIONED.json L4–L7 already recanted:
Q80 went 1376.3 → 108.3 ms (12.7×) with no BPW change.

G024 action still says G016 density "is the only attack that moves a
millisecond-class EXISTENTIAL bucket." Addressing is 60.44% of the TOKEN_NS
wall (L51–L52). That makes density the largest **byte** lever. It does not
make R1 a physical roof and it does not close the other 40%.

**Verdict: R3 FALSIFIED as a genome-independent roof. R4 FALSIFIED as an
exclusivity claim. Density remains the largest byte lever.**

---

## 5. R5 / R6 / R7 — 819, 560–647, 320–411

**R5 819 GB/s.** SPEC SHEET. `PHYSICAL_FLOOR.json` L8 uses it to emit
`dram_floor_ns`. `FS_PER_WEIGHT_LAW.json` L5 derives 152.6252 from it.
`TERMINAL_TARGET.json` L4, L6 uses it for every model's TPS ceiling, then
L11 says "Use the measured control, not 819, when judging efficiency."
Self-contradictory. `roof_rungs.py` L146–151 FAIL any claim that treats 819
as the decode ceiling.

Never achieved on this box in any receipt cited here. HONEST_ROOF L1351
published_peak=819, kernel roof=699.57 = 85% of datasheet for this access
pattern.

**R6 560–647.** MEASURED sequential/reuse probe. SUPERWAVE_STATE L4.
`roof_rungs.py` REUSE_BAND 535.882–637.496. Judge: reuse band is
cache-resident, **not** a decode ceiling (`test_roof_rungs.py` L128–135).
Regime: reuse. Decode is unique-once. Wrong axis.

**R7 320–411.** MEASURED no-model control in **Q80** 98-CB, 10-of-512 gather
shape (SUPERWAVE_STATE L73–L74, G013 v2). QWEN38_ACTIVE_BUDGET L CORRECTION
already states this gather control does not bound dense sequential Qwen3.8.
`roof_rungs.py` L119: "REUSE vs NO-REUSE, not gather vs sequential" — the
ACTIVE_BUDGET correction used the gather/sequential axis, which the later
instrument rejects.

**Verdict:**
- R5 is a datasheet, not a measured roof for this workload.
- R6 is a reuse probe, not a decode roof.
- R7 is a Q80-topology control. Using it as a Qwen3.8 wall is a regime error.

---

## 6. R8 / R9 — femtosecond-per-weight floor

`FS_PER_WEIGHT_LAW.json` L4–L5:

```
fs_per_weight_floor = 152.6252 * BPW
derivation: (BPW/8) / 819e9 * 1e15
```

CITED: `1/819e9 * 1e15 / 8 = 152.6251526252`. The constant **is** R5.

Same receipt L44: "AMORTIZED THROUGHPUT METRIC - NOT PHYSICAL SINGLE-WEIGHT
LATENCY." `roof_rungs.py` L81–88: a single weight's DRAM round trip is ~100 ns.

G013 v1 used R6 (560–647) and **storage** BPW 0.6462 → "sub-100 fs needs
BPW < 0.448–0.518." G013 v2 (`G013_FS_EFFICIENCY_CLOSURE_V2.json`) KILLS v1:
storage ≠ active, 560–647 is reuse, Q80 active BPW is ~2.5–5.0, best existing
Q80 artifact bottoms at 765–985 fs even with a perfect load-only kernel.

R9 `BPW < 0.6552` is R8 at efficiency=1.0. Unity was not achieved. Conditioned
on R5.

CITED fs/weight constants at efficiency=1 for **this** genome's measured roofs:

| bandwidth | fs per weight per BPW |
|---|---|
| 819 datasheet | 152.625 |
| 699.57 single-addr MEASURED | 178.680 |
| 639.25 sealed CITED | 195.541 |
| 411.51 R1 | 303.761 |

**Verdict: R8 is ESTIMATED from a datasheet. Not a measured floor for this
workload. R9 is R8 at an unachieved efficiency. Both are genome-conditioned.**
The metric is not latency. PHYSICAL_FLOOR.json L12–13 "per-FLOP time is
femtoseconds" is the language `roof_rungs.flag_fs_latency_language` rejects
(`test_roof_rungs.py` L184–188).

---

## 7. R10 — dispatch overhead floor

`PHYSICAL_FLOOR.json` L33–34: 48-layer chain, ≥1 dispatch/layer, 2–5 µs
launch, 100–240 µs, "practical floor ~1 ms/token."

**Regime:** Q80 48-layer mental model. Not Qwen3.8.

Qwen3.8 production genome (`QWEN38_TOKEN_NS_LEDGER.json` dispatches.total=964,
production_command_buffers=1; G024 L measurement.kernel_runtime_genome):
**1 CB / 964 dispatches.**

Q80 decode-shape dispatch control (`Q80_DECODE_SHAPE_BANDWIDTH.json`):

| probe | median | what |
|---|---|---|
| 1 nop CB GPU | 3,334 ns | empty dispatch |
| 1 nop CB host | 192,083 ns | host CB tax |
| 1155 nops, 1 CB GPU | 1,483,875 ns | 1,284.74 ns/dispatch fused |
| 98 CBs × 12 nops host−gpu | 20,079,956 ns | 204,898 ns host tax / serial CB |

CITED Qwen3.8 fused GPU dispatch floor: 1,284.74 × 964 = 1.238 ms.
Serial-CB host tax does not apply (1 CB). PHYSICAL_FLOOR's 100–240 µs is
the wrong topology and the wrong layer count.

ICB (`QWEN38_FIXED_OVERHEAD_DELETED.json`) replaced 964 encoder create/bind
cycles with 1 `executeCommandsInBuffer`. Encode 886,200 → 90,981 ns. Wait−gpu
**rose** 425,900 → 561,994 ns. Net named fixed 1.331 → 0.671 ms. Dispatch
"floor" moved when the command genome moved.

**Verdict: FALSIFIED as a Qwen3.8 complete-token floor. The 2–5 µs × 48
number is an estimate for a genome Qwen3.8 does not run.**

---

## 8. R11 — 1.229 ms fixed overhead

MEASURED TOKEN (`QWEN38_COMPLETE_TOKEN_WALL_AUTHORITY.json` L39):
`wall_minus_gpu_ns=1,229,334`. RUNG_TARGETS L33 holds this constant in
BPW projections.

ICB MEASURED TOKEN (`QWEN38_FIXED_OVERHEAD_DELETED.json` L74, L80–L82):
named fixed 0.670934 ms; complete wall 38.217 → 36.684 ms. Encode fell,
wait rose, net still down.

**Verdict: FALSIFIED as irreducible. Genome change (ICB + scalar slab)
cut it ~2×. Remainder is still a 10 ms-budget tax (6.7%).**

---

## 9. R12 — density assumed to cost speed

`QWEN38_RECONSTRUCTION_IS_FREE.json` L4, L10–L15. COMPONENT. 2 organs, 33
codec variants, REAL BF16 hidden. GPU timestamps.

- f32 control tpr64: gate 15,125 ns, down 7,083 ns
- q4/q3/q2/binary/ternary/additive_q2q2/hadamard/rice at tpr64: 15,124–15,541 ns
- recon excess = 0 on 32/33
- same codecs at tg256: ~26,500 ns — **launch geometry**, not codec

Retires a 5.9× rice penalty that was measured on Q80's serial 1-thread-per-row
extract and transferred to Qwen3.8 without remeasure (L19).

TOKEN_NS `weight_decode_reconstruction` is 1.808 ms, 5.13% of wall — the
addr-vs-decode probe split on isolated class GEMVs, not a 5.9× rice tax.

This is **not** a complete-token "any codec is free" claim. It is a
same-launch-geometry organ probe. Binding still applies: a low-BPW path that
expands to float/Q4 then generic GEMV must win on a complete token.

**Verdict: KILLS "density costs speed" at tpr64 for the tested codecs.
REOPEN_IF launch geometry changes, or the production path expands+generic-GEMVs,
or the codec is not in the 33-variant set.**

---

## 10. R13 / R15 — gate set below the DRAM floor it was supposed to respect

**R13.** `QWEN38_ARCH_CENSUS.json` L156–L157: 4.0 BPW DRAM floor = 14.414 ms
from `23.611e9 × 4 / 8 / 819e9`.

`QWEN38_ACTIVE_BUDGET_MEASURED.json` L18–L20:

```
lane_said_ms: 14.414
true_floor_ms: 16.63
why: floor excluded lm_head. 248,320 vocab must read full lm_head each token.
     1.817 GB shortfall is lm_head plus the embed table.
```

CITED: `23_611_000_000*4/8/819e9 = 14.414530 ms`.
`13_622_264_240/819e9 = 16.632801 ms`. Shortfall 1,816,764,240 B.

The 14.414 ms "floor" sat **below** the datasheet floor of the bytes the
token actually moves.

**R15.** `tools/genesis_tournament.py` L120–L122 C7: DSV4F 20 ms gate vs
10.280 GB / 411.51 = 24.98 ms floor. Gate below the floor it invoked.
(The 411.51 in that sentence is itself R1.)

**Verdict: two documented cases. R13 is the Qwen3.8 one.**

---

## 11. R14 — 100 TPS physically impossible at 3.0 BPW

`QWEN38_ARCH_CENSUS.json` L182 HARD_CONSEQUENCE_1: ceiling 92.5 tok/s at
3.0 BPW. Arithmetic is R13's census weights / 819.

CITED load-only at 3.0 BPW:

| bytes basis | @819 TPS | @699.57 TPS | @639.25 TPS | @411.51 TPS |
|---|---|---|---|---|
| census 8.854e9 (omits lm_head) | 92.50 | 79.01 | 72.20 | 46.48 |
| artifact-scaled 9.610e9 | 85.23 | 72.80 | 66.52 | 42.82 |

100 TPS is load-only impossible at 3.0 BPW on every bandwidth in this table,
including 819. The **92.5** number is still wrong: it uses the incomplete
census and the datasheet. The qualitative "100 TPS needs <3.0 BPW or a
genome that moves fewer bytes" survives as a bytes-budget statement, not as
92.5 being a physical ceiling.

At current 4.2527 BPW / 13.612 GB, 100 TPS needs 1,361 GB/s. Above 819.
**That** load-only claim is intact.

G1 target 1.5 BPW: if bytes scale with BPW, 13.612 × 1.5/4.2527 = 4.800 GB.
Load-only at 639.25 = 7.51 ms (133 TPS); at 699.57 = 6.86 ms (146 TPS); at
819 = 5.86 ms (171 TPS). Complete-token remainder today is 14–17 ms. 1.5 BPW
does **not** imply 100 TPS on the present genome unless that remainder
collapses.

**Verdict: 92.5 is ESTIMATED and uses a short census. "100 TPS at 3.0 BPW
is load-only impossible on this box" is SUPPORTED. "100 TPS at 1.5 BPW is
implied" is NOT supported without an execution-genome attack on the
non-addressing 40%.**

---

## 12. R16 / R17 — policy and capability floors (not speed roofs)

**R16.** `lab/operators/one_bit_ceiling.py` `CEILING = 1/1`.
`lab/operators/doctor6/ceiling.py` TARGET 1.0, ABS_HARD 1.5, mixed-precision
floor 1.34. POLICY. Not a DRAM or dispatch claim. Escape is sealed, not a
physics argument.

**R17.** `QWEN38_DENSITY_ROOT_CAUSE.json`: mixed-2p0-v1 complete physical
BPW 2.0856, MLP 0.848, attention+embed+norms 4.250 (74% of artifact).
`QWEN38_SUB15_INCOHERENT.json`: mixed-sub15 ~1.29 BPW, degenerate token cycle,
0 fallbacks. Capability floor of the **rice_q1-on-attention** genome.
CORRECTION L15: blocks the density front, not execution ascent.

mixed-2p0-v1 was packed, not materialized as a generate catalog (same
receipt). Coherence at 2.0856 is a pack-report figure, not a sealed greedy-id
match against the uniform-q4 seal.

**Verdict: not speed roofs. Codec-family conditioned. Do not treat 2.0856
as a physics BPW floor.**

---

## 13. R18 / R19 / R20 — the genome-conditioned Qwen3.8 roofs that remain

Source: `receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json`.
Timing label `GPU_PROTECTED_CPU_CONTENDED` (L timing_label). Current
TOKEN_NS/TPS **not rerun**. GPU timestamps on completed CBs.

| what | GB/s | kind | payload |
|---|---|---|---|
| single-GEMV addr 13.612 GB | 699.5736545106142 | COMPONENT MEASURED | 13,611,663,360 |
| single-GEMV decode | 683.7970139656385 | COMPONENT MEASURED | same |
| single-GEMV full | 666.6814921907636 | COMPONENT MEASURED | same |
| sealed weight_addressing | 639.2522348492898 | TOKEN split CITED from G024 | same |
| 401-organ catalog addr | 530.6544688491846 | COMPONENT MEASURED | 13.6 GB class |
| 401-organ catalog full | 505.8100047843556 | COMPONENT MEASURED | 13.6 GB class |
| unique_once 13.612 GB | 375.6517695934827 | COMPONENT MEASURED | same bytes, different kernel |
| unique_once 512 MiB this run | 342.7289906725934 | COMPONENT MEASURED | did not reproduce 411.51 |

ALU+decode tax vs addr at 13.6 GB: 4.70%. Sealed / single-addr = 0.9138.
Catalog addr is 24.1% below single-addr. That gap is **dispatch/topology
headroom**, named open by `roof_rungs.py` L62–71 and `test_roof_rungs.py`
L155–166.

`tools/ascent/roof_rungs.py` already routes Qwen3.8 to 699.57 and refuses to
emit a current TPS (`current_token_ns=None`, rung UNMEASURED). Rust ledgers
do not. ROOF_AND_RUNGS.md receipt still prints the 411.51 table.

HONEST_ROOF L1325 `saturated_on_this_genome: true` for the Q4 grouped-GEMV
unique-once DRAM stream — **not** 97.6% of 411.51 (L1368). Provisional:
CPU contended, TOKEN_NS not rerun.

**Verdict: these are the least-wrong roofs on disk. They are still
conditioned on `geo_tpr64_tg128`, cols=5120 groups, 64 threads/row, 128-TG,
unique codes+scales. Change any of those and the roof moves.**

---

## 14. R21 / R22 / R23 — mis-applied shared blockers

**R21.** TERMINAL_TARGET L75: "Measured efficiency is ~0.4% of the 560–647
GB/s control. Until packed matvec occupancy is solved, NONE of these
ceilings are reachable." That is Q80 mixed (`~2.5 GB/s` in L11). Qwen3.8
sealed addressing is 639 GB/s. Applying R21 to Qwen3.8 is a category error.
ROOF_AND_RUNGS physical-limit audit already FAILs this field.

**R22.** "No kernel trick escapes the bandwidth floor" (SUPERWAVE_STATE
L38–39; FS_PER_WEIGHT_LAW L42). True only as "batch=1 unique-once has no
arithmetic reuse." False as "the GB/s number is invariant." Same bytes,
different kernel: 376 (unique_once) vs 667 (Q4 full) vs 700 (Q4 addr).
The floor **is** the genome. Q80 12.7× without BPW (CORRECTION L6) is the
existence proof.

**R23.** TOKEN_NS `terminal_head` L198 `effective_gb_s=1763.66` with
`bytes_read=675,430,400` and `ns=383,535`. Method L202: "lm_head weight
traffic lives in addressing/decode." The 1763 figure divides addressing
bytes by leftover FMA time. Super-unity vs 819. Not a roof. measured_over_floor
0.233 is the same R1-floor artifact.

Other TOKEN_NS over-floor ratios at R1 (411.51), all TOKEN-split MEASURED:

| component | ns | over R1 floor | effective GB/s | actual bound |
|---|---|---|---|---|
| weight_addressing | 21,293,103 | 0.64 | 639.25 | R18 |
| kv_state | 537,665 | 0.70 | 588.49 | sequential f32 state stream |
| terminal_head | 383,535 | 0.23 | 1763.66 | bytes/time mismatch |
| deltanet | 3,732,795 | 223 | 1.84 | launch + 16-wide reductions |
| gqa | 2,443,471 | 313 | 1.31 | rope 24 threads / tiny MHA |
| normalization | 2,367,415 | 123 | 3.35 | 129 RMSNorm launches |
| dense_swiglu | 1,004,198 | 28 | 14.62 | silu + residual + FMA rem |
| synchronization | 384,250 | 3.95e7 | ~0 | wait−gpu; floor is a nop |

DeltaNet / GQA / RMSNorm sit 100–300× above their **byte** floors. Those
are not DRAM roofs. They are launch+occupancy roofs of tiny kernels. G024
L rank 2–3 already names them. They move if fused; they do not move if
BPW drops and the launches remain.

---

## 15. What the roof becomes if the genome changes

| change | R1 411.51 | R18 639–700 | complete-token 35–38 ms |
|---|---|---|---|
| fewer bytes, same kernel (density) | scales | scales | addressing 60% scales; remainder does not automatically |
| same bytes, Q4 GEMV vs unique_once | N/A (wrong kernel) | 376 → 667 already observed | — |
| same bytes, 401-organ catalog → single-GEMV topology | — | 531 → 700 (24%) | remainder unchanged; TOKEN ~30 TPS ceiling if only this moves |
| ICB / persistent scalars | — | — | named fixed 1.33 → 0.67 MEASURED |
| fuse DeltaNet tails + 129 RMSNorms | — | — | G024 ESTIMATE 2.2–3.3 ms; not remeasured this lane |
| consume low-BPW in-register (no expand-to-Q4) | — | roof is the new codec's stream | binding: must win on complete token |
| expand-to-float/Q4 then generic GEMV | — | likely lose the 639–700 Q4 stream | REJECT unless TOKEN measurement shows net win |
| change launch (tpr64 → tg256) | — | recon-free dies (R12) | COMPONENT only so far |

100 TPS (TOKEN_NS ≤ 10,000,000) on this artifact requires **both** fewer
bytes and a smaller remainder. Load-only math at 1.5 BPW leaves 2.5–4.1 ms
for everything that is not addressing. Today that bucket is 14–17 ms.

---

## 16. Historical ceilings already broken (do not re-import)

1. Gate below DRAM floor: R13 14.414 vs 16.63; R15 20 ms vs 24.98 ms.
2. Density costs speed: R12, plus Q80 in-register 867.0 → 36.6 ms at unchanged
   codec (`QWEN38_RECONSTRUCTION_IS_FREE.json` L20).
3. Roof-as-physics: CORRECTION_ROOF_IS_CONDITIONED.json; Q80 12.7×, no BPW.
4. G013 v1 storage-BPW + reuse-band fs law: killed by G013 v2 same day.
5. 33537 µs G015 GPU as a wall proxy: COMPLETE_TOKEN_WALL_AUTHORITY
   `is_33537_gpu_an_honest_proxy_for_wall.answer = NO`. This session GPU
   36.99 ms, wall 38.22 ms.

---

## 17. Still-live wrong roof (the one this lane was told to find)

**R1 + R2 + R3 + R4, still compiled into the Qwen3.8 TOKEN_NS floor.**

Evidence chain:

1. Live constant 411.51 — `qwen38_token_ns_ledger.rs:28`.
2. That constant is the floor for every component row — L477.
3. Sealed TOKEN_NS addressing already reports `measured_over_floor=0.6437`
   at 639.25 GB/s — `TOKEN_NS_QWEN38.json:46,49,55`.
4. The campaign that installed that constant later measured the real
   access pattern at 699.57 / 666.68 / 375.65 — HONEST_ROOF L1329–L1351.
5. G024 / ROOF_AND_RUNGS.md / RUNG_TARGETS still publish 97.6% and 30.21 TPS
   from the dead ceiling.

`tools/ascent_daemon.py` L428, L776 already says the 411.51 / 97.6% story
is REFUTED. The Rust floor function was not updated.

---

## 18. Binding note for G1 density work

Preferred shape: low-BPW representation consumed by a representation-specific
Metal kernel.

R12 says reconstruction is free **at tpr64 on the Q4 GEMV launch**. That is
a COMPONENT result. It does **not** license "pack low, expand to Q4, run
`geo_tpr64`." That path pays the 13.6 GB Q4 stream plus expand traffic.
A candidate that does this must show a complete-token win against
uniform-q4-v1 on the same wall definition as receipt C (or A, named).

---

## 19. What this lane could not answer

GPU measurement is owned by another lane. Not rerun here:

- Current complete-token TOKEN_NS / TPS on HEAD `2eee9a004` (receipts are
  commits `57ee82c`, `8772941`, `9c87c50`).
- Whether 699.57 still holds on a clean box (HONEST_ROOF is
  GPU_PROTECTED_CPU_CONTENDED).
- Whether ICB is the resident G0 genome (lineage install cites 35.228 ms,
  which is pre-ICB TOKEN_NS, not 36.684 ICB complete-wall).
- Energy / pJ/weight (G024 unresolved).
- Hardware occupancy counters (launch-geometry derived only).

Cheapest experiment that would close the remaining hole: one locked
`ascension_qwen38_token_ns` + complete-wall pair on the resident artifact,
same GPU-timestamp authority, paired 3×3, after the GPU lane is free.
Until then G0 is two dirty receipts, not a new measurement.

---

## 20. KILLS / REOPEN_IF

| mechanism | status | REOPEN_IF |
|---|---|---|
| 411.51 as Qwen3.8 decode roof | KILLS | a new unique_once run at 13.6 GB on this kernel exceeds the Q4 GEMV roof and a complete token hits it |
| 97.6% of roof | KILLS | never; the denominator is structurally wrong |
| 30.21 TPS as genome-independent roof | KILLS | only restatable as "load-only at 411.51" |
| density is the only lever | KILLS | — |
| 14.414 ms DRAM floor at 4.0 BPW | KILLS | only if lm_head is not read (it is) |
| reconstruction costs 5.9× | KILLS at tpr64 | launch geometry ≠ tpr64; expand-then-GEMV path; new codec |
| 1.229 ms fixed overhead irreducible | KILLS | ICB disabled |
| 48×(2–5 µs) dispatch floor for Qwen3.8 | KILLS | production returns to 1 CB/layer or 48 serial CBs |
| 100 TPS at 13.6 GB | still dead | bandwidth > 1361 GB/s or bytes drop |
| 100 TPS at 1.5 BPW on current remainder | not implied | remainder must fall from 14–17 ms to ≤2.5–4.1 ms |
| doctor6 1.0/1.5 BPW | POLICY, not a roof | — |
| sub-1.5 coherence | KILLS for rice_q1-on-attention | new attention codec family |

---

## Completion report

### STATUS
FALSIFIED

### CLAIMS
- CLAIM-1 FALSIFIED: 411.51 GB/s is the Qwen3.8 decode roof. Evidence: HONEST_ROOF L1351 `refuted_ceiling_gb_s=411.51`; unique_once at 13.6 GB = 375.65 (L1345); Q4 addr at 13.6 GB = 699.57 (L1329); sealed addressing = 639.25 (TOKEN_NS_QWEN38 L46) beating the ledger floor 33.077 ms (L55).
- CLAIM-2 FALSIFIED: Qwen3.8 is at 97.6% of its roof. Evidence: HONEST_ROOF L1378; honest_roof.rs L235–L252, L1151–L1168.
- CLAIM-3 FALSIFIED: complete-token roof is 30.21 TPS until bytes drop. Evidence: load-only at 699.57 = 51.40 TPS; complete-token even at 819 with today's remainder = 29.81–32.73 TPS. RUNG_TARGETS L12 still prints 30.21 from R1.
- CLAIM-4 FALSIFIED: density is the only remaining lever. Evidence: CORRECTION_ROOF_IS_CONDITIONED.json L4–L7; catalog→single-GEMV 24% gap (HONEST_ROOF L1329–L1330); ICB cut fixed overhead (FIXED_OVERHEAD_DELETED L74); DeltaNet/GQA/RMSNorm 223–313× byte floors (TOKEN_NS_QWEN38 L106, L125).
- CLAIM-5 FALSIFIED (historical): a DRAM floor/gate was set below the bytes it bound. Evidence: ACTIVE_BUDGET L18–L20 (14.414 vs 16.63); genesis_tournament.py L120–L122 (20 ms vs 24.98 ms).
- CLAIM-6 FALSIFIED (historical): density costs speed. Evidence: RECONSTRUCTION_IS_FREE L4, L14–L15 (COMPONENT, tpr64).
- CLAIM-7 SUPPORTED: 100 TPS at 13.612 GB is load-only impossible on this box. Evidence: 13.612e9 / 10e-3 = 1361 GB/s > 819 and > every measured roof in §4.
- CLAIM-8 SUPPORTED: 100 TPS at 1.5 BPW is not implied by the present genome. Evidence: addressing would scale to ~7.51 ms at 639.25; remainder today 14–17 ms; 10 ms budget leaves 2.5–4.1 ms for that remainder.
- CLAIM-9 INCONCLUSIVE (not rerun): live G0 TOKEN_NS/TPS on HEAD 2eee9a004. Evidence: two dirty authorities, 35.228 ms vs 38.217 ms, different definitions; GPU lane not used.

### EVIDENCE
See §§0–19. Commands that produced the CITED arithmetic are the Python
reproductions in this lane (bytes/ns only; no GPU). Source excerpts carry
`path:line`. Receipts were read via `git show HEAD:<path>` (sparse checkout;
receipts are not materialized).

### CHANGES
One new file: `workspace/superwave/g1/g1-roof-falsification.md`.
No tracked file modified. No GPU. No artifact. No resident-process touch.

### TESTS
See lane runner output after this file lands:
`test -s workspace/superwave/g1/g1-roof-falsification.md`
`wc -l workspace/superwave/g1/g1-roof-falsification.md`
`git status --porcelain`

### RISKS
- HONEST_ROOF is GPU_PROTECTED_CPU_CONTENDED; absolute 699.57 is provisional.
- Lineage G0 35.228 ms may not be the ICB genome; do not mix A and C walls.
- R12 is COMPONENT (2 organs). Do not promote to TOKEN.
- roof_rungs.py already moved Qwen3.8 off 411.51; Rust did not. Split-brain.

### UNRESOLVED
- Clean-box paired TOKEN_NS + complete-wall on HEAD.
- Whether resident G0 is pre-ICB or ICB.
- Hardware occupancy counters.
- Energy.
- Whether a representation-specific sub-Q4 attention kernel can take R17
  without expand-to-Q4.

### NEXT
Stop compiling 411.51 as `HONEST_DECODE_CEILING_GB_S` for Qwen3.8.
Treat 639.25 (sealed TOKEN split) / 699.57 (COMPONENT addr) / 505.81
(COMPONENT catalog full) as the current-genome band.
Attack the 14–17 ms non-addressing remainder and the catalog/single
topology gap in parallel with density.
Any <1.5 BPW candidate must be consumed by its own kernel and timed as a
complete token.
