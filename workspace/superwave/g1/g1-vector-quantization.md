# G1 vector / structured-codebook quantization — Qwen3.8 attention

**STATUS: MEASURED_NEGATIVE**

Vector quantization can go below one bit per weight. On real Qwen3.8 attention
GEMV tensors it does not preserve the 0.99 output-cosine bar below ~4.0 BPW.
The only variants that clear that bar are Q4-rate (uniform Q4, Hadamard-Q4,
PQ d=2 K=256, RVQ d=4 K=256 S=2). Every sub-4 BPW codebook, residual, lattice,
and computed-trellis variant fails the bar on L0 `in_proj_qkv` and L3 `q_proj`.
Table-lookup PQ/RVQ already has a direct Metal kernel; a prior same-box
measurement on a similar GEMV rejected it as slower than the source-model
token budget. Register-only structured codes exist and are the only ones a
direct kernel can consume with no codebook gather; they lose to scalar at
the same rate except Hadamard-Q4 (0.125 BPW save, already known).

This is a representation measurement. No GPU timing, no pack, no generate.

---

## 1. Method

| item | value | class |
|---|---|---|
| weights | `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16` (11 bf16 shards, census path) | MEASURED read |
| activations | `.../qwen38-27b/activation-capture-v1/hidden/LXX.f32` 256×5120 f32 post-norm hidden | MEASURED read |
| quality bar | flattened `output_cosine(WX, ŴX) ≥ 0.990` | QUOTED from `receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_VERDICT.json` |
| encode | CPU numpy + sklearn `MiniBatchKMeans` (`~/.grok-vision/bin/python` 3.12.13 / numpy 2.5.1 / sklearn 1.9.0) | MEASURED |
| peak RSS | 5.758 GB during sweep, 6.380 GB during finish | MEASURED `ru_maxrss` |
| wall | sweep 671.39 s + finish 23.95 s | MEASURED |
| results | `/tmp/g1_vq_results.json` sha256 `0090fc39a09ad3bb83fbfacf0ad9edad8d423bce6746f0148329a6ddccd5f487` | MEASURED |
| log | `/tmp/g1_vq_sweep.log` sha256 `ce6abb153812cd4dbde04d394b992fe1d70a8d5387235f2eebaaa3dd10a14c87` | MEASURED |

Tensors (one at a time, bf16→f32):

| tensor | shape | elems | X used |
|---|---|---|---|
| L0 `linear_attn.in_proj_qkv` | 10240×5120 | 52,428,800 | L00.f32 |
| L0 `linear_attn.in_proj_z` | 6144×5120 | 31,457,280 | L00.f32 |
| L0 `linear_attn.out_proj` | 5120×6144 | 31,457,280 | none (mixer-site X not in capture) |
| L3 `self_attn.q_proj` | 12288×5120 | 62,914,560 | L03.f32 |
| L3 `self_attn.k_proj` | 1024×5120 | 5,242,880 | L03.f32 |
| L3 `self_attn.v_proj` | 1024×5120 | 5,242,880 | L03.f32 |
| L3 `self_attn.o_proj` | 5120×6144 | 31,457,280 | none |
| L32 `linear_attn.in_proj_qkv` | 10240×5120 | 52,428,800 | L32.f32 |
| L63 `self_attn.q_proj` | 12288×5120 | 62,914,560 | L63.f32 |

PQ/RVQ/gravity-VQ train on a 262,144-subvector reservoir then assign every
subvector. Uniform / binary / Hadamard / lattice / computed-trellis encode
the whole tensor. Column-wise split (GEMV layout: each row is a sequence of
d-vectors over the K dimension).

Uniform-Qn uses the exact HGRAVU01 rule (`scale = maxabs / ((1<<(bits-1))-1)`,
q clipped to `±bound`, group 64) from
`lab/operators/ascension_dual_gravity_worker.py` `_uniform_codec`.

`lattice_E8_1` in the first sweep is **INVALID** (`b=1` set radius=0, recon
collapsed). Replaced by `lattice_E8_radius1_g64` in the finish pass.

No token-level claim is made from these numbers.

---

## 2. Attention mass (MEASURED from shard headers)

Command: parse `model.safetensors.index.json` + every language_model shard header.

```
language_total 26895998464 26.895998464
mlp              17112760320 17.112760320G 63.6257%
linear_attn      5562051072 5.562051072G 20.6798%
self_attn        1677729792 1.677729792G 6.2378%
lm_head          1271398400 1.271398400G 4.7271%
embed            1271398400 1.271398400G 4.7271%
norm             660480 0.000660480G 0.0025%
attn             7239780864
```

GEMV projections only (the VQ target), from the same headers:

| pattern | n | shape | elems |
|---|---|---|---|
| `linear_attn.in_proj_qkv` | 48 | 10240×5120 | 2,516,582,400 |
| `linear_attn.in_proj_z` | 48 | 6144×5120 | 1,509,949,440 |
| `linear_attn.out_proj` | 48 | 5120×6144 | 1,509,949,440 |
| `linear_attn.in_proj_a/b` | 48+48 | 48×5120 | 23,592,960 |
| `linear_attn.conv1d` | 48 | 10240×4×1 | 1,966,080 |
| `self_attn.q_proj` | 16 | 12288×5120 | 1,006,632,960 |
| `self_attn.k_proj` | 16 | 1024×5120 | 83,886,080 |
| `self_attn.v_proj` | 16 | 1024×5120 | 83,886,080 |
| `self_attn.o_proj` | 16 | 5120×6144 | 503,316,480 |
| **GEMV total** | | | **7,239,761,920** |

Matches `crates/hawking-core/src/model/qwen38_geometry.rs` lines 20–52
(`QWEN38_IN_PROJ_QKV_ROWS=10240`, `QWEN38_Q_PROJ_ROWS=12288`,
`QWEN38_O_PROJ_COLS=6144`, 48 DeltaNet + 16 GQA).

G0 complete BPW 4.2527 is QUOTED (`QWEN38_BPW_DESCENT.json` `baseline.current_bpw`
= 4.252735126866492) and is the uniform-Q4-all vehicle, not independently
re-timed here.

mixed-2p0 already put MLP at 0.848 BPW and left attention+embed+lm_head+norm
at 4.250 BPW (`mixed-2p0-v1/PACK_REPORT.json` lines 10–38). That is why
attention is the remaining density organ.

---

## 3. Calibration against the attention-density probe

Same tensors, same HGRAVU01 rule, same capture.

| tensor | this run Q4 w_cos / out_cos | probe Q4 w_cos / out_cos |
|---|---|---|
| L0 in_proj_qkv | 0.99414 / 0.99615 | 0.99414 / 0.99611 |
| L0 in_proj_z | 0.99401 / 0.99535 | 0.99401 / 0.99541 |
| L0 out_proj | 0.99354 / n/a | 0.99354 / 0.99224 |
| L3 q_proj | 0.99396 / 0.99694 | (probe L3 q present, same family) |
| L63 q_proj | 0.99344 / 0.99829 | 0.99344-class / 0.998-class |
| L63 q Q3 | w_cos **0.966038236180493** | probe **0.9660382361804879** |

Q4/Q3/binary weight cosines match the probe to ≥5 decimals. Pipeline is the
same codec.

Primary bar used below is **flattened** output cosine (the probe's
`output_cosine` field). Mean-row / min-row are secondary: L63 Q3 flattened
0.99098 would pass the probe bar, but min-row here is 0.726 because a few
high-energy rows are badly hit (`/tmp/g1_vq_finish.log`). The probe reported
min-row 0.98852 on the same tensor — different row-reduction, not a different
Ŵ (weight cosine is bit-identical). Flattened is the comparable number.

---

## 4. Results — L0 `in_proj_qkv` (full grid)

Primary mass (2.52 G, 48 replicas). Real X. Source: `/tmp/g1_vq_sweep.log`
lines 9–47.

| variant | BPW (idx+cb+scale)/N | w_cos | out_cos | 0.99 | register-only | table |
|---|---|---|---|---|---|---|
| uniform_q4_g64 | 4.2500 | 0.99414 | 0.99615 | YES | YES | 0 |
| hadamard_q4_g128 | 4.1250 | 0.99320 | 0.99553 | YES | YES (ALU transform) | 0 |
| pq_shared d=2 K=256 | 4.0002 | 0.99326 | 0.99170 | YES | NO | 1 KiB |
| rvq d=4 K=256 S=2 | 4.0006 | 0.99129 | 0.99161 | YES | NO | 4 KiB |
| uniform_q3_g64 | 3.2500 | 0.96939 | 0.97954 | NO | YES | 0 |
| lattice_Z4 b=3 (=scalar q3) | 3.2500 | 0.96939 | 0.97954 | NO | YES | 0 |
| hadamard_q3_g128 | 3.1250 | 0.96461 | 0.97625 | NO | YES | 0 |
| lattice_D4 b=3 | 3.0000 | 0.96044 | 0.97355 | NO | YES | 0 |
| pq_shared d=4 K=1024 | 2.5013 | 0.96848 | 0.97496 | NO | NO | 8 KiB |
| lattice_E8 b=2 | 2.2500 | 0.80431 | 0.85800 | NO | YES | 0 |
| uniform_q2_g64 | 2.2500 | 0.77691 | 0.83655 | NO | YES | 0 |
| lattice_Z4 b=2 (=scalar q2) | 2.2500 | 0.77691 | 0.83655 | NO | YES | 0 |
| computed-trellis d=4 k=8 | 2.2500 | 0.89450 | 0.92288 | NO | YES | 0 |
| hadamard_q2_g128 | 2.1250 | 0.74179 | 0.80723 | NO | YES | 0 |
| pq_shared d=4 K=256 | 2.0003 | 0.93930 | 0.94759 | NO | NO | 2 KiB |
| gravity_VQ D=4 K=256 | 2.0003 | 0.93891 | 0.94899 | NO | NO | 2 KiB |
| rvq d=8 K=256 S=2 | 2.0013 | 0.93183 | 0.94018 | NO | NO | 8 KiB |
| rvq d=8 K=16 S=4 | 2.0002 | 0.91510 | 0.92035 | NO | NO | 1 KiB |
| lattice_D4 b=2 | 2.0000 | 0.71275 | 0.78361 | NO | YES | 0 |
| E8 radius-1 (fixed) | 1.8350 | 0.80431 | 0.85800 | NO | YES | 0 |
| rvq d=16 K=256 S=3 | 1.5037 | 0.88091 | 0.90096 | NO | NO | 24 KiB |
| pq_per_sub d=8 K=256 | 1.4000 | 0.83819 | 0.88982 | NO | NO | 2.5 MiB |
| pq_shared d=8 K=1024 | 1.2525 | 0.85810 | 0.88680 | NO | NO | 16 KiB |
| computed-trellis d=8 k=8 | 1.2500 | 0.79069 | 0.84359 | NO | YES | 0 |
| binary_g128 | 1.1250 | 0.79874 | 0.84346 | NO | YES | 0 |
| pq_shared d=8 K=256 | 1.0006 | 0.80344 | 0.83163 | NO | NO | 4 KiB |
| gravity_VQ D=8 K=256 | 1.0006 | 0.80319 | 0.83184 | NO | NO | 4 KiB |
| OPQ-fold d=8 K=256 | 1.0006 | 0.80344 | 0.83163 | NO | NO | 4 KiB |
| rvq d=16 K=256 S=2 | 1.0025 | 0.79668 | 0.83041 | NO | NO | 16 KiB |
| computed-trellis d=8 k=4 | 0.7500 | 0.57060 | 0.64022 | NO | YES | 0 |
| pq_shared d=16 K=1024 | 0.6300 | 0.69478 | 0.75577 | NO | NO | 32 KiB |
| pq_shared d=16 K=256 | 0.5012 | 0.62877 | 0.68117 | NO | NO | 8 KiB |
| gravity_VQ D=16 K=256 | 0.5012 | 0.63005 | 0.68687 | NO | NO | 8 KiB |
| gravity_VQ D=32 K=256 | 0.2525 | 0.46997 | 0.54278 | NO | NO | 16 KiB |

BPW includes fp16 codebook storage amortized over **this tensor**. Shared
codebook of K×d halves is 2Kd bytes. Per-subspace codebook on d=8 is
S=640 tables = 2.5 MiB → +0.40 BPW and still fails.

**Scalar cannot go below 1 BPW.** The 0.25–0.75 BPW rows are the vector
forms. They exist. Their reconstruction is not attention-legal.

---

## 5. Other tensors (flagship)

Same pattern. Closest sub-Q4 learned VQ is always `pq_shared d=4 K=256`
(~2.00 BPW) or `d=4 K=1024` (~2.50 BPW). Neither clears 0.99 except as
noted.

| tensor | Q4 out | Q3 out | PQ d4K256 out | PQ d8K256 out | D4b2 out | notes |
|---|---|---|---|---|---|---|
| L0 z | 0.99535 | 0.97556 | 0.95452 | 0.83974 | 0.75206 | |
| L3 q | 0.99694 | 0.98366 | 0.96563 | 0.87903 | 0.82042 | PQ d4K1024 = 0.98268 |
| L3 k | 0.99718 | 0.98495 | 0.97165 | 0.89457 | 0.82442 | easiest small GEMV |
| L3 v | 0.99619 | 0.97992 | 0.96487 | 0.87153 | 0.79287 | |
| L32 qkv | 0.99498 | 0.97590 | 0.94614 | 0.83103 | 0.78052 | mid-depth DeltaNet |
| L63 q | 0.99829 | **0.99098** | 0.98021 | 0.92346 | 0.87729 | late GQA is easier; min-row 0.726 |

Weight-space only (no mixer X):

| tensor | Q4 w_cos | Q3 w_cos | PQ d4K256 w_cos | PQ d8K256 w_cos | out_proj is the hard organ |
|---|---|---|---|---|---|
| L0 out | 0.99354 | 0.96674 | 0.91529 | 0.78007 | PQ at 4.0 BPW (d=2 K=256) is only 0.97308 |
| L3 o | 0.99375 | 0.96768 | 0.92345 | 0.79442 | |

out_proj at 2.0 BPW is **worse** than in-proj at 2.0 BPW. A pack that
averages L63 q into a pass and ignores out_proj will be incoherent.

---

## 6. Residual VQ vs product VQ vs gravity S=1

On L0 qkv, at matched rate:

| rate | best single-stage | residual | winner |
|---|---|---|---|
| ~4.0 | PQ d=2 K=256 out 0.99170 | RVQ d=4 K=256 S=2 out 0.99161 | single-stage, barely |
| ~2.0 | PQ/GVQ d=4 K=256 out 0.947–0.949 | RVQ d=8 K=256 S=2 out 0.940; 4×K16 out 0.920 | single-stage |
| ~1.0 | PQ/GVQ d=8 K=256 out 0.832 | RVQ d=16 K=256 S=2 out 0.830 | tie, both fail |
| ~1.5 | — | RVQ d=16 K=256 S=3 out 0.901 | fail |

Additive stages at fixed total bits lose to one wider codebook. This is the
same geometry `gravity_residual_pq_matvec` bills (`stages` sequential
gathers of a D-vector). More stages buy nothing quality-side here and
multiply decode gathers.

OPQ (PCA rotate, fold `C @ R.T` into the stored codebook so decode is
unchanged) is **identical** to unrotated PQ on these tensors. Attention
subvectors are already near-isotropic. Evidence: L0 qkv `pq_opq_8_256`
and `pq_shared_8_256` both 0.80344 / 0.83163.

Gravity production fixture geometry (`tests/fixtures/gravity_pq/manifest.json`
R4: D=32 S=1 K=256, 0.375 BPW on a 512×2048 toy) is 0.2525 BPW on real
L0 qkv and 0.543 output cosine. Sub-bit S=1 VQ is not an attention codec.

---

## 7. Cross-layer codebook

Train one d=8 K=256 codebook on L0 `in_proj_qkv`, assign L32 `in_proj_qkv`.

```
cross out_cos=0.82301 w_cos=0.81351 bpw=1.0006 shared48=1.0000
```

Same-layer L32 `pq_shared_8_256`: w_cos 0.81623 / out_cos 0.83103.

A single codebook serves 48 DeltaNet twins. Amortization of a 4 KiB table
across 48 tensors changes BPW by 0.0006 → 0.0000. Sharing is free and
does not move the quality needle.

---

## 8. Lattice / computed trellis / 1D Viterbi

**D4 / E8 / Z^d** (group-64 fp16 scale, register-only):

- Z^d at b bits **is** uniform-Qb. L0 qkv Z4-b3 == Q3 (0.96939 / 0.97954).
- D4 at the same integer range saves 1 parity bit / 4 weights (3.00 vs 3.25
  BPW) and **loses** cosine (0.97355 vs 0.97954). Coarse lattice gain is
  negative once you clip to a 3-bit box.
- E8 radius-1 (honest log2(3) bits/coord, 1.835 BPW) matches E8-b2
  reconstruction (coords clip to [−1,1]) at 0.804 / 0.858. Binary-class.

**Hadamard lattice** is the repo `HGRAVH01` path (Walsh-Hadamard then
uniform integer, self-inverse). Register-only, extra butterflies.
Q4-rate passes; Q3/Q2 fail. Already in the attention-density verdict
("2.9% save, not the mass").

**Computed vector-trellis** (k bits / d-vector, reconstruction =
group-scale × axis-aligned hashed-Gaussian code, no stored table):
L0 qkv d=4 k=8 is 2.25 BPW / 0.923 out_cos — beats Q2 (0.837) and
Hadamard-Q2 (0.807), still far from 0.99. d=8 k=8 is 1.25 BPW / 0.844,
binary-class.

**1D Viterbi TCQ** on 65,536 L0-qkv weights (256 blocks × 256), Gaussian
computed codebook, no table:

```
TCQ_viterbi_k1_L5_blk256 bpw=1.0820 w_cos=0.80448 w_rel=0.5955
TCQ_viterbi_k2_L6_blk256 bpw=2.0859 w_cos=0.94466 w_rel=0.3283
TCQ_viterbi_k3_L7_blk256 bpw=3.0898 w_cos=0.98330 w_rel=0.1829
```

Viterbi at 3.09 BPW beats scalar Q3 weight cosine (0.983 vs 0.969) on this
sample. It is still scalar: **cannot go below 1 BPW**. It is a weight-space
sample, not an output-cosine on the full tensor. Vector Viterbi (k bits
per d-vector, learned or computed `[2^L × d]` target) was not run — that
is the cheapest leftover quality experiment (see UNRESOLVED).

strand-quant already has the scalar computed path
(`CodebookMode::ComputedAcklam`, `decode.rs` 36–56, `codebook.rs` 131–150)
and a vector path that **gathers** `lut[state*d + j]`
(`decode.rs` 285–353). `vector_lut_from_scalar` (`encode.rs` 1074–1082)
just repeats the scalar codeword d times; it is not a learned vector
codebook.

---

## 9. Decode cost shape (architectural + one cited GPU measurement)

No GPU run in this lane. Numbers below are either counted from the
shipping Metal source or quoted from a prior same-box receipt.

### 9.1 Per-weight work a Metal thread does

| family | sequential traffic / weight | random gathers / weight | ALU / weight | register-only? | shipping kernel |
|---|---|---|---|---|---|
| uniform Qn g64 |  n bits + 16/64 scale | **0** | nibble extract + 1 FMA | YES | `qwen_uniform_q4_group64_matvec` |
| binary g128 | 1 bit + 16/128 scale | **0** | sign + 1 FMA | YES | none for attention |
| Hadamard Qn g128 | n bits + 16/128 | **0** | `g log2 g` butterflies / g + 1 FMA | YES (transform) | none for attention |
| D4 / E8 / Z^d g64 | (db − parity + coset)/d + 16/64 | **0** | int→float + 1 FMA | YES | none |
| computed trellis | k/d bits + 16/64 | **0** | hash + Acklam × d axes + d FMA | YES (ALU-heavy) | strand `ComputedAcklam` is the scalar analogue |
| 1D TCQ Viterbi | k bits + 16/256 | **0** | state shift + 1 quantile + 1 FMA | YES | none |
| PQ / gravity S=1 | log2(K)/d index bits | **1/d** (one `half[d]` gather / d weights) | d FMA / d | NO | `gravity_pq_matvec` |
| RVQ S stages | S·log2(K)/d | **S/d** | S FMA | NO | `gravity_residual_pq_matvec` |
| PQ per-subspace | log2(K)/d | 1/d | d FMA / d | NO | same kernel, working set = S tables |

Q4 decode, `qwen_uniform_q4.metal` 38–67: one sequential nibble stream, one
fp16 scale broadcast per 64 weights, `q = nibble-8`, `sum += q * scale * x`.
Zero table.

PQ decode, `gravity_pq.metal` 399–428:

```
uint flat = (row * p.nchunk + c) * p.subspaces + s;
const device half *entry = cb + pq_index(codes, flat, p.bits) * p.sub;
for (uint j = 0; j < p.sub; ++j) acc = fma(float(entry[j]), xs[j], acc);
```

One 4-byte window extract + one **random** codebook gather of `sub` halves
+ `sub` FMAs, per subvector. Residual path (lines 430–458) does that
`stages` times on a D-vector.

### 9.2 Why a codebook gather is not automatically free

For GEMV the codebook is shared across every row. After warmup a 2–16 KiB
shared table (S=1, K≤256, D≤32) sits in cache. Incremental traffic is the
index stream, not the table. A 2.5 MiB per-subspace table (S=640, d=8,
K=256) is a different working set and is the one that can be
bandwidth-worse than the dense form it replaced.

The residual-PQ reject already measured the S=1 case as too slow **even
with a hot table**. Quote,
`workspace/campaign/evidence/runtime/tg/TG_LLAMA_RESIDUAL_PQ_FFN_RUNTIME_REJECTED.json`:

```
geometry: 14336×4096, D=8, stages=4, card=128, bits=7
median_microseconds_per_gate_projection: 672.625
single_stage_control: D=32, stages=1, card=256 → 460.041 µs
cause: "Additive stages multiply compact decode FMAs and index reads."
reopen_condition: "A materially different execution grammar must demonstrate
a resident real-geometry gate/up pair at <=0.5 ms combined ... Direct
one-codebook-per-chunk lookup is rejected too."
```

That is MEASURED on this box, on a 14336×4096 FFN GEMV, not on Qwen3.8
attention, and not re-run here.

### 9.3 PROJECTED attention-token cost if every attn GEMV paid the 460 µs class

Attention GEMV elements per token (projections only):

```
48*(10240*5120 + 6144*5120 + 5120*6144) + 16*(12288*5120 + 2*1024*5120 + 5120*6144)
= 7,214,202,880
```

`(7,214,202,880 / (14336*4096)) * 460.041 µs = 56.52 ms` attention-only.
**PROJECTED** (linear scale of a different kernel / occupancy). Not a
token measurement. Even as a loose shape it is 5–6× the Q4 sequential
byte-time of the same mass at the QUOTED 406.2 GB/s
(`QWEN38_ACTIVE_BUDGET_MEASURED.json`): `7.214e9 * 4.25 / 8 / 406.2e9 =
9.43 ms`. A representation that needs a gather per chunk can be
**slower than Q4** while storing fewer bytes.

Binding: expand-to-float then generic GEMV is rejected. `gravity_pq_matvec`
already is the representation-specific kernel. It exists. On the one
geometry that was timed, it lost.

---

## 10. Complete-BPW projection (PROJECTED, quality not held)

Language params P = 26,895,998,464.
MLP BPW 0.8480504639008466 QUOTED from mixed-2p0 (not re-packed here).
embed+lm_head+norm held at 4.25.

| attention codec | attn BPW | complete BPW | attn flattened 0.99 on L0 qkv? |
|---|---|---|---|
| Q4 (today's mixed-2p0 non-MLP) | 4.2500 | 2.0855 | YES |
| Hadamard-Q4 | 4.1250 | 2.0518 | YES |
| Q3 | 3.2500 | 1.8163 | NO (0.97954) |
| PQ d=4 K=1024 | 2.5013 | 1.6148 | NO (0.97496) |
| PQ d=4 K=256 | 2.0003 | **1.4799** | NO (0.94759) |
| RVQ d=16 K=256 S=3 | 1.5037 | 1.3462 | NO (0.90096) |
| PQ d=8 K=256 | 1.0006 | 1.2108 | NO (0.83163) |
| gravity D=32 K=256 | 0.2525 | 1.0095 | NO (0.54278) |

The 1.5 complete target is **arithmetically** reachable if attention sits
at ≤2.07 BPW and mixed-2p0's MLP recipe is kept. No variant measured here
holds 0.99 at that rate. A 1.48 complete pack that used PQ d=4 K=256 on
attention would be a quality fail, not a G1 candidate.

G0 4.2527 is uniform-Q4-all (MLP also at 4.25). mixed-2p0's 2.086 is the
relevant "MLP already cheap" baseline for this projection.

---

## 11. Register-only flag

Decodable from the index with **no codebook table fetch**:

- uniform Qn, binary, ternary (not re-run; probe already has it)
- D4 / E8 / Z^d integer lattices
- Hadamard integer lattice (extra butterflies, no gather)
- computed-trellis / ComputedAcklam (ALU: hash + rational; tail is a
  ≤397-entry monotone table, not a per-weight random codebook)
- 1D TCQ Viterbi (state is a shift-register)

**Not** register-only (random gather of a stored codeword):

- PQ shared / per-subspace / OPQ-fold
- gravity S=1 VQ
- residual VQ
- learned vector-trellis LUT (`lut[state*d + j]`, `decode.rs` 338–348)

Those last four are the ones that can be bandwidth-worse than the dense
form they replace, and the ones the residual-PQ receipt already timed.

---

## 12. Verdict

**KILLS** (this organ, this quality bar, this execution genome):

1. Product / residual / gravity VQ as a Q4-quality attention representation
   below ~4.0 BPW. Measured on real Qwen3.8 tensors + real post-norm X.
2. Table-lookup VQ as a token-time win on this box, unless a new kernel
   beats the 460 µs-class measurement. The shipping `gravity_pq_matvec`
   is the preferred shape (direct, no expand-to-float) and was already
   rejected on a similar GEMV.
3. D4 / E8 / computed-trellis as a path under Q3 at Q4 quality. They are
   register-only, which is the right decode shape, and they still fail
   the bar.
4. Sub-bit gravity fixture geometry (D=16/32, K=256) on attention.

**Does not kill:**

- Scalar TCQ Viterbi as a possible *same-rate* quality bump at 2–3 BPW
  (sample only; still ≥1 BPW; no output-cosine, no kernel).
- Hadamard-Q4 (0.125 BPW, already known, register-only).
- Using VQ on a *low-sensitivity* organ. Attention is not that organ.

**REOPEN_IF**

- Activation-weighted or output-aware k-means reaches flattened
  output_cosine ≥ 0.990 at attn BPW ≤ 2.0 on **both** L0 `in_proj_qkv`
  (real X exists) **and** L0 `out_proj` (requires a mixer-site capture
  that this snapshot does not have).
- The GPU lane measures `gravity_pq_matvec` on Qwen3.8 `in_proj_qkv`
  10240×5120 at ≤ the Q4 kernel's measured GPU ns on the same matrix.
- A real vector Viterbi (not independent assign) with d=4 and ≥10 bits
  per subvector is measured on L0 qkv + L0 out.
- A generate test authorises a 0.97 bar. Then `pq_shared d=4 K=1024`
  at 2.50 BPW is the only candidate in this file that is even close
  (L0 qkv 0.975, L3 q 0.983, L0 out weight 0.952).

Cheapest next experiment that would change the verdict: mixer-site X
for out_proj (one capture) + one activation-weighted PQ d=4 K=1024 on
L0 out and L0 qkv. If that still misses 0.99, stop.

---

## 13. Evidence

### 13.1 Sweep invocation and L0 qkv block

```
$ ~/.grok-vision/bin/python -u /tmp/g1_vq_sweep.py
python 3.12.13 numpy 2.5.1 sklearn=True
bf16=/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/bf16 exists=True
act=/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/activation-capture-v1 exists=True

===== language_model.model.layers.0.linear_attn.in_proj_qkv.weight grid=full =====
  loaded (10240, 5120) in 0.08s rss=0.640GB
uniform_4_64                               bpw= 4.2500 w_cos=0.99414 w_rel=0.1088 out_cos= 0.99615 PASS reg=1 tab=       0 rss=3.129GB
uniform_3_64                               bpw= 3.2500 w_cos=0.96939 w_rel=0.2539 out_cos= 0.97954 fail reg=1 tab=       0 rss=3.130GB
uniform_2_64                               bpw= 2.2500 w_cos=0.77691 w_rel=0.7198 out_cos= 0.83655 fail reg=1 tab=       0 rss=3.130GB
binary_128                                 bpw= 1.1250 w_cos=0.79874 w_rel=0.6017 out_cos= 0.84346 fail reg=1 tab=       0 rss=3.132GB
pq_shared_2_256                            bpw= 4.0002 w_cos=0.99326 w_rel=0.1161 out_cos= 0.99170 PASS reg=0 tab=    1024 rss=3.400GB
pq_shared_4_256                            bpw= 2.0003 w_cos=0.93930 w_rel=0.3431 out_cos= 0.94759 fail reg=0 tab=    2048 rss=3.419GB
pq_shared_4_1024                           bpw= 2.5013 w_cos=0.96848 w_rel=0.2492 out_cos= 0.97496 fail reg=0 tab=    8192 rss=3.427GB
pq_shared_8_256                            bpw= 1.0006 w_cos=0.80344 w_rel=0.5954 out_cos= 0.83163 fail reg=0 tab=    4096 rss=3.427GB
pq_shared_16_256                           bpw= 0.5012 w_cos=0.62877 w_rel=0.7776 out_cos= 0.68117 fail reg=0 tab=    8192 rss=3.427GB
gvq_32_256                                 bpw= 0.2525 w_cos=0.46997 w_rel=0.8827 out_cos= 0.54278 fail reg=0 tab=   16384 rss=3.427GB
rvq_8_256_2                                bpw= 2.0013 w_cos=0.93183 w_rel=0.3632 out_cos= 0.94018 fail reg=0 tab=    8192 rss=3.427GB
rvq_4_256_2                                bpw= 4.0006 w_cos=0.99129 w_rel=0.1319 out_cos= 0.99161 PASS reg=0 tab=    4096 rss=3.427GB
lattice_D4_3                               bpw= 3.0000 w_cos=0.96044 w_rel=0.2904 out_cos= 0.97355 fail reg=1 tab=       0 rss=3.579GB
hadamard_4_128                             bpw= 4.1250 w_cos=0.99320 w_rel=0.1173 out_cos= 0.99553 PASS reg=1 tab=       0 rss=4.043GB
```

Full log: `/tmp/g1_vq_sweep.log` (233 lines, sha256
`ce6abb153812cd4dbde04d394b992fe1d70a8d5387235f2eebaaa3dd10a14c87`).
Sweep wall 671.39 s, exit 0 after a leftover KeyError on the cross-layer
print (tensors were finished). Finish pass wrote the JSON.

### 13.2 Finish pass

```
$ ~/.grok-vision/bin/python -u /tmp/g1_vq_finish.py
parsed 9 tensors, variants=189
CROSS-LAYER
  cross out_cos=0.82301 w_cos=0.81351 bpw=1.0006 shared48=1.0000
VITERBI
  TCQ_viterbi_k1_L5_blk256 bpw=1.0820 w_cos=0.80448 w_rel=0.5955 1.21s
  TCQ_viterbi_k2_L6_blk256 bpw=2.0859 w_cos=0.94466 w_rel=0.3283 3.59s
  TCQ_viterbi_k3_L7_blk256 bpw=3.0898 w_cos=0.98330 w_rel=0.1829 11.95s
E8 radius-1 fix
  in_proj_qkv bpw=1.8350 w_cos=0.80431 out_cos=0.857998411215929
  q_proj bpw=1.8350 w_cos=0.79989 out_cos=0.8822610721733494
L63 Q3 min-row recompute
  L63 q Q3 flattened=0.99098 mean_row=0.97684 min_row=0.72638
WROTE /tmp/g1_vq_results.json 67356 bytes rss=6.380GB
```

### 13.3 Geometry authority

`crates/hawking-core/src/model/qwen38_geometry.rs` 20–52:

```
20|pub const QWEN38_LAYERS: usize = 64;
21|pub const QWEN38_DELTANET_LAYERS: usize = 48;
22|pub const QWEN38_GQA_LAYERS: usize = 16;
24|pub const QWEN38_HIDDEN: usize = 5_120;
38|pub const QWEN38_GQA_HEADS: usize = 24;
39|pub const QWEN38_GQA_KV_HEADS: usize = 4;
40|pub const QWEN38_GQA_HEAD_DIM: usize = 256;
43|pub const QWEN38_IN_PROJ_QKV_ROWS: usize = 10_240;
44|pub const QWEN38_IN_PROJ_Z_ROWS: usize = 6_144;
49|pub const QWEN38_Q_PROJ_ROWS: usize = 12_288;
50|pub const QWEN38_KV_PROJ_ROWS: usize = 1_024;
51|pub const QWEN38_O_PROJ_ROWS: usize = 5_120;
52|pub const QWEN38_O_PROJ_COLS: usize = 6_144;
```

### 13.4 Shipping PQ kernel (one gather + d FMAs per subvector)

`crates/hawking-core/shaders/gravity_pq.metal` 399–428, 430–458 (cited in §9).

### 13.5 Shipping Q4 kernel (register-only)

`crates/hawking-core/shaders/qwen_uniform_q4.metal` 38–67 (cited in §9).

### 13.6 Computed codebook / vector LUT

`workspace/vendor/strand-quant/src/decode.rs` 36–56 (ComputedAcklam vs
vector LUT gather).
`workspace/vendor/strand-quant/src/codebook.rs` 131–136 (integer Acklam,
no gather for the central 97.6%).
`workspace/vendor/strand-quant/src/encode.rs` 1074–1082
(`vector_lut_from_scalar` repeats the scalar entry).

### 13.7 Prior residual-PQ reject

`workspace/campaign/evidence/runtime/tg/TG_LLAMA_RESIDUAL_PQ_FFN_RUNTIME_REJECTED.json`
(672.625 µs 4-stage, 460.041 µs 1-stage, reopen ≤0.5 ms combined pair).

### 13.8 Quality bar and mixed-2p0 bytes

`receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_VERDICT.json`:
"Attention GEMVs cannot be cheaply compressed below uniform-Q4 at
Q4-equivalent output quality."
`mixed-2p0-v1/PACK_REPORT.json` lines 10–38:
`complete_physical_bpw` 2.0855934079220506, `mlp_physical_bpw`
0.8480504639008466, `nonmlp_physical_bpw` 4.250142713483966.

### 13.9 Census (unverified DRAM-floor arithmetic, quoted only)

`receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json` `download.path` =
`workspace/campaign/records/runs/qwen38-27b/bf16`. The 819 GB/s and
"100 TPS is physically impossible at 3.0 BPW" lines in that file are
byte/bandwidth arithmetic, **not** used as a floor here.

---

```
STATUS
MEASURED_NEGATIVE

CLAIMS
C1. Real Qwen3.8 attention GEMV tensors total 7,239,761,920 weights (48 DeltaNet + 16 GQA projections). MEASURED from shard headers. Evidence: §2 command output.
C2. Flattened output cosine of HGRAVU01 Q4 on L0 in_proj_qkv is 0.99615 and matches the attention-density probe to 4e-5; Q4/Q3 weight cosines match to ≥5 decimals. MEASURED. Evidence: §3, /tmp/g1_vq_sweep.log:9-11.
C3. Product quantization, residual VQ, gravity S=1 VQ, D4/E8 lattices, Hadamard-Q3/Q2, and computed vector-trellis all fail output_cosine ≥ 0.990 on L0 in_proj_qkv below 4.0 BPW. The only passes are Q4-rate (Q4, Hadamard-Q4, PQ d=2 K=256, RVQ d=4 S=2). MEASURED. Evidence: §4 table, /tmp/g1_vq_sweep.log:9-47.
C4. Vector quantization can go below 1 BPW (PQ d=16 K=256 = 0.5012 BPW including a 8 KiB codebook; gravity D=32 K=256 = 0.2525 BPW). Reconstruction at those rates is 0.54–0.68 output cosine. MEASURED. Evidence: §4.
C5. Residual stages at fixed total bits do not beat single-stage PQ. MEASURED. Evidence: §6.
C6. A codebook trained on L0 in_proj_qkv transfers to L32 (w_cos 0.81351 vs same-layer 0.81623). Sharing across 48 layers is free and does not fix quality. MEASURED. Evidence: /tmp/g1_vq_finish.log CROSS-LAYER.
C7. D4/E8/computed-trellis/Hadamard/binary/uniform/TCQ-Viterbi are register-only (no codebook gather). PQ/RVQ/gravity-VQ/learned vector-trellis are not. MEASURED from Metal/Rust source. Evidence: §9, §11, gravity_pq.metal:399-458, qwen_uniform_q4.metal:38-67, decode.rs:36-56.
C8. Direct codebook-lookup GEMV was measured at 460.041 µs (1-stage D=32 K=256) and 672.625 µs (4-stage D=8 K=128) on a 14336×4096 FFN on this box and rejected. PROJECTED 56.52 ms attention-only if every Qwen3.8 attention GEMV paid the 460 µs class. Evidence: TG_LLAMA_RESIDUAL_PQ_FFN_RUNTIME_REJECTED.json; §9.3.
C9. Complete BPW ≤ 1.5 with mixed-2p0's MLP recipe requires attention ≲ 2.07 BPW. No measured variant holds 0.99 at that rate. PROJECTED arithmetic + MEASURED quality. Evidence: §10.
C10. out_proj is harder than in-proj (L0 out PQ d=4 K=256 w_cos 0.915 vs L0 qkv out_cos 0.948). Mixer-site X was not in the capture; out_proj scores are weight-space only. MEASURED / gap. Evidence: §5.

EVIDENCE
/tmp/g1_vq_sweep.log sha256 ce6abb153812cd4dbde04d394b992fe1d70a8d5387235f2eebaaa3dd10a14c87
/tmp/g1_vq_results.json sha256 0090fc39a09ad3bb83fbfacf0ad9edad8d423bce6746f0148329a6ddccd5f487
/tmp/g1_vq_finish.log
workspace/campaign/records/runs/qwen38-27b/bf16 (read-only)
workspace/campaign/records/runs/qwen38-27b/activation-capture-v1 (read-only)
workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1/PACK_REPORT.json
crates/hawking-core/src/model/qwen38_geometry.rs:20-52
crates/hawking-core/shaders/gravity_pq.metal:399-458
crates/hawking-core/shaders/qwen_uniform_q4.metal:38-67
workspace/vendor/strand-quant/src/decode.rs:36-56
workspace/vendor/strand-quant/src/codebook.rs:131-150
workspace/campaign/evidence/runtime/tg/TG_LLAMA_RESIDUAL_PQ_FFN_RUNTIME_REJECTED.json
receipts/ascent-2026-08-16/QWEN_ATTENTION_DENSITY_VERDICT.json
receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json

CHANGES
workspace/superwave/g1/g1-vector-quantization.md (new, this file)
No tracked file modified. No artifact packed. No GPU run. No live Genesis touch.

TESTS
see final-message TESTS block (test -s, wc -l, git status --porcelain)

RISKS
- MiniBatchKMeans is a local optimum; a better codebook could move 2.0 BPW cosine by a few points, not from 0.95 to 0.99, unless the training objective changes (activation-weighted).
- out_proj functional cosine is missing (no mixer-site X). Weight-space already says it is the hard organ.
- 56.52 ms attention projection uses a different kernel/geometry. It is a shape, not a floor.
- L63 Q3 flattened 0.99098 vs min-row 0.726: do not promote late-GQA ease to the whole stack.

UNRESOLVED
- Vector Viterbi (k bits / d-vector, learned [2^L × d] LUT or computed) on L0 qkv + L0 out.
- Activation-weighted PQ.
- Mixer-site X for out_proj.
- GPU ns of gravity_pq_matvec vs qwen_uniform_q4_group64_matvec on 10240×5120 (other lane).

NEXT
Do not spend a pack or a kernel port on attention PQ/RVQ at <4 BPW unless REOPEN_IF fires. If anything in this family is retried, it is register-only D4/TCQ at 3 BPW (quality still fails today) or the activation-weighted PQ d=4 K=1024 experiment named above — one tensor pair, then stop if it misses.
```
