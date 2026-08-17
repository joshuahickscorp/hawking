# G1 MLP floor locate — Qwen3.8 language body

Lane: `81-mlp-floor-locate`. CPU numpy on real BF16 tensors + real 256-token capture. No GPU, no generate, no pack, no resident touch.

STATUS: **MEASURED_NEGATIVE** for a native MLP floor inside the 0.848–1.6 BPW band at the 0.95-all-64 organ-hold bar. The first native codec that holds 0.95 on every one of the 192 MLP tensors is Uniform Q3 at class physical **3.2500251321231617**. That is not a generate floor.

Every number is tagged **MEASURED** (this-lane command output or on-disk integer), **DERIVED** (exact arithmetic on MEASURED integers), **CITED** (prior receipt/report, not re-run), or **EXPECTATION** (not a finding).

Sweep machine result: `/tmp/g1_mlp_floor.json` (`partial=false`, 192 organs, `wall_s=983.0766312080086`, `rss_max_gb=4.2565155029296875`). Script: `/tmp/g1_mlp_floor.py`.

---

## 0. Verdict

No native-reader codec in {Binary, Residual, Hgravs r160_b3, Uniform q2} holds output cosine 0.95 on all 64 `down_proj`. Uniform Q3 does (min 0.9662531725591231 at L62). Uniform Q4 is the first to hold 0.99 on all 192 MLP tensors.

The 6-layer doctor/descent sample hid a late-layer `down` collapse: L58 binary hold **0.3005754758685758**, L54 **0.3634911395141286**, L59 **0.38321218951309755**. Those layers are not in `{0,3,15,31,47,63}`. A floor measured on early layers is wrong.

mixed-2p0 spent 2.0855934079220506 complete BPW with attention at 4.250 and `down` at 0.1316. The same budget, allocated from this curve (all 64 `down` → Q3, gate Binary, up Residual, tables Q4, small f32), leaves attention at **1.7823662580678705**. That band has no measured 0.99 hold. **EXPECTATION: a correctly allocated 2.0856 would not have been expected to survive.** Not a generate finding.

---

## 1. Method

### 1.1 Native codecs scored

`load_mixed` accepts catalog 0/1/2/3 and refuses 4 (`qwen38_hybrid_decode.rs:601-664`). Uniform bits 2..=8. HGRAVS locked r160_b3. MLP name-locked to Binary/Residual/Hgravs (`:958-1003`) — lock is not a quality result.

| family | operator this lane | physical BPW | source |
|---|---|---:|---|
| Binary | HGRAVB01 sign × f16 mean-abs / 128 | 1.1250234267290902 | MEASURED mixed-2p0 / descent bytes 12534021 |
| Residual | HGRAVR02 rice_q1_rms_2pct | 1.2875108157887178 class | MEASURED mixed-2p0 `up_proj` |
| Hgravs | HGRAVS01 r160_b3 | 0.13161714918473189 class | MEASURED mixed-2p0 `down_proj` 93847197 B |
| Uniform qn | HGRAVU01 absmax / (2^{n-1}-1), g64 | n + 0.25 + 280×8/89128960 | DERIVED; q3 bytes 36208920 MEASURED mixed-q3mlp |

Ternary, Hadamard, additive, HQ30UQ2/3 g128: not in `load_mixed`. Not scored.

### 1.2 Quality spaces

| organ | X | rows/in-dim | space |
|---|---|---:|---|
| gate, up | captured post-norm hidden 256×5120 | 0.0500 | **output**, underdetermined |
| down | reconstructed `silu(X@Wg.T)*(X@Wu.T)` from that hidden + BF16 gate/up | 0.0147 | **output_reconstructed_swiglu**, underdetermined. mixer_x was never captured. |
| all | W vs Ŵ | — | **weight** always |

Hold protocol matches descent: fit 192 / hold 64, `hold_output_cosine = cos(X_hold W^T, X_hold Ŵ^T)`. Weight-only codecs do not fit on X; hold is still a screen, not a generate claim. Capture 256 tokens is the Q80-catastrophe class of underdetermined fit (NS-014 was 0.0449 rows/dim).

HGRAVS packed mixed-2p0 used `n_fit_rows=256` on every down (segment records). Its output cosine on the last 64 tokens is **in-sample**. Honest HGRAVS is `hgravs01_r160_b3_act_thin` (192/64). Exact 5120-gram eigh on L0 gate reproduced the thin operator to 7e-7 (hold 0.83665355 vs 0.83665426).

### 1.3 Calibration vs descent (6 layers)

Descent receipt `QWEN38_BPW_DESCENT.json`. Same operators, same split.

| cell | this lane | descent | Δ |
|---|---:|---:|---:|
| L0 gate binary hold | 0.861852195981 | 0.861852194430 | 1.55e-9 |
| L0 gate Q3 hold | 0.982098354690 | 0.982098354690 | 0 |
| L0 down Q3 hold | 0.992290431331 | 0.992290431331 | 0 |
| L31 up Q3 hold | 0.967898216503 | 0.967898216503 | 0 |
| L63 down binary hold | 0.729727422641 | 0.729727449993 | −2.74e-8 |
| L0 gate binary weight | 0.7983613877 | packed 2p0 `_cosine` 0.7983613876884476 | match |
| L0 down packed HGRAVS weight | 0.21118785 | packed record 0.21118785178932237 | match |

---

## 2. Error curve — 64/64 layers, 192 tensors

Hold output cosine. MEASURED `/tmp/g1_mlp_floor.json`.

### 2.1 Per organ × codec

| organ | codec | n | hold min | at | hold med | hold max | n≥0.95 | n≥0.9679 | n≥0.99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gate | Binary | 64 | 0.795584 | L22 | 0.855190 | 0.954274 | 1 | 0 | 0 |
| gate | Residual 2% | 64 | 0.843710 | L6 | 0.892432 | 0.967528 | 4 | 0 | 0 |
| gate | Uniform q2 | 64 | 0.772672 | L24 | 0.819816 | 0.938719 | 0 | 0 | 0 |
| gate | Uniform q3 | 64 | 0.968335 | L6 | 0.978372 | 0.994048 | 64 | 64 | 5 |
| gate | Uniform q4 | 64 | 0.994268 | L6 | 0.995919 | 0.998893 | 64 | 64 | 64 |
| gate | HGRAVS act-thin | 64 | 0.702464 | L6 | 0.822831 | 0.921576 | 0 | 0 | 0 |
| up | Binary | 64 | 0.741579 | L26 | 0.793355 | 0.960095 | 1 | 0 | 0 |
| up | Residual 2% | 64 | 0.797467 | L26 | 0.838724 | 0.969656 | 1 | 1 | 0 |
| up | Uniform q2 | 64 | 0.781626 | L40 | 0.792314 | 0.957026 | 1 | 0 | 0 |
| up | Uniform q3 | 64 | 0.967673 | L32 | 0.970724 | 0.995720 | 64 | 61 | 1 |
| up | Uniform q4 | 64 | 0.993301 | L32 | 0.994218 | 0.999174 | 64 | 64 | 64 |
| up | HGRAVS act-thin | 64 | 0.692521 | L26 | 0.760421 | 0.929271 | 0 | 0 | 0 |
| down | Binary | 64 | **0.300575** | **L58** | 0.814329 | 0.919477 | 0 | 0 | 0 |
| down | Residual 2% | 64 | **0.364203** | **L58** | 0.861383 | 0.939820 | 0 | 0 | 0 |
| down | Uniform q2 | 64 | 0.752931 | L62 | 0.803396 | 0.950661 | 1 | 0 | 0 |
| down | Uniform q3 | 64 | **0.966253** | **L62** | 0.974289 | 0.992290 | 64 | 62 | 1 |
| down | Uniform q4 | 64 | 0.993738 | L62 | 0.995111 | 0.998556 | 64 | 64 | 64 |
| down | HGRAVS act-thin (honest hold) | 64 | 0.730175 | L9 | 0.811719 | 0.956020 | 1 | 0 | 0 |
| down | HGRAVS packed-2p0 (in-sample) | 64 | 0.919630 | L8 | 0.955374 | 0.989701 | 49 | 4 | 0 |
| down | HGRAVS weight-rsvd | 64 | 0.227399 | L56 | 0.385263 | 0.798581 | 0 | 0 | 0 |

Uniform q5–q8: hold min ≥ 0.998463 on every organ. Not a G1 spend.

Weight-space (do not use as HGRAVS quality): packed-2p0 down weight-cosine min **0.1525058591718298** at L54, median 0.17139, max 0.21119 (L0). mixed-q3mlp strided weight-cosine min **0.9652814877860332** at L63 down (CITED PACK_REPORT; this-lane full-tensor Q3 weight min 0.9652015204, same tensor).

### 2.2 First all-64 hold

| bar | gate | up | down |
|---|---|---|---|
| 0.95 all 64 | Uniform q3 | Uniform q3 | Uniform q3 |
| 0.9679 all 64 | Uniform q3 | Uniform q4 | Uniform q4 |
| 0.99 all 64 | Uniform q4 | Uniform q4 | Uniform q4 |

Q3 organs below the descent 0.9679 citation: L62 down 0.966253, L60 down 0.967650, L32 up 0.967673, L33 up 0.967818, L31 up 0.967898. The 6-layer `hold_min=0.9679` was L31 up and is **not** the 64-layer min.

---

## 3. Depth — down hardens; three layers collapse

Binary / Q2 / Q3 hold on `down_proj`. MEASURED.

| L | binary | q2 | q3 |
|---:|---:|---:|---:|
| 0 | 0.919477 | 0.903649 | 0.992290 |
| 2 | 0.762624 | 0.868355 | 0.983957 |
| 15 | 0.826808 | 0.812581 | 0.975560 |
| 31 | 0.816208 | 0.803714 | 0.974248 |
| 47 | 0.780222 | 0.800948 | 0.972930 |
| 50 | 0.710050 | 0.826373 | 0.975022 |
| **54** | **0.363491** | 0.950661 | 0.987175 |
| 55 | 0.684711 | 0.823385 | 0.973823 |
| **58** | **0.300575** | 0.946791 | 0.987287 |
| **59** | **0.383212** | 0.933480 | 0.983945 |
| 61 | 0.745673 | 0.782271 | 0.968233 |
| 62 | 0.746452 | 0.752931 | **0.966253** |
| 63 | 0.729727 | 0.799022 | 0.972669 |

Binary down < 0.50: {54, 58, 59}. < 0.75: {50, 54, 55, 58, 59, 61, 62, 63}. Descent's six layers never saw 54/58/59. L63 binary 0.730 is not the worst.

L54/L58/L59 are the exception that proves Q2 is not a substitute for Q3: those three have Q2 hold 0.93–0.95 while binary is 0.30–0.38. Everywhere else Q2 ≈ binary. Affine q2 remains dominated as a *class* (min 0.753).

Gate/up harden through mid-depth then recover at L62–L63 (L63 gate binary 0.954, up 0.960) — last-layer exception, do not average.

---

## 4. Where the floor is

**Organ-hold 0.95 floor, native reader, all 64 layers:** Uniform Q3 g64, class physical BPW **3.2500251321231617**. 192/192 hold ≥ 0.966253.

**Organ-hold 0.99 floor:** Uniform Q4 g64, **4.2500251321231617**. 192/192 hold ≥ 0.993301.

**Empty:** every native codec with class BPW ≤ 2.250 fails 0.95 on at least 63/64 downs (Q2 clears 1/64). HGRAVS honest hold clears 1/64. Packed HGRAVS in-sample clears 49/64 at 0.95 and 0/64 at 0.99.

This is a component screen on a 256-token underdetermined capture. It is **not** a generate floor. mixed-q3mlp-v1 at complete **3.6138647373176767** (CITED PACK_REPORT) is the cheapest packed artifact whose MLP side matches the 0.95 hold. It has no GENERATE.json and HEAD `assert_mixed_mlp_native` refuses it.

0.8480504639008466 (mixed-2p0 / sub15 MLP mix) is below every 0.95 hold. That statement does not locate a generate floor: both 1.291 and 2.0856 generates are confounded (expand-to-Q4; 2p0 also crushed down and used 2048-col tiles).

---

## 5. Inversion — attention budget the packer has left

Identity (`qwen38_pack.rs:673-679`):

```
complete_physical_bpw = 8 * tensor_payload_bytes / N
```

MEASURED integers (`g1-bit-budget-accounting.md` §4, reconfirmed):

| symbol | value |
|---|---:|
| N | 26,895,998,464 |
| E_mlp | 17,112,760,320 |
| E_attn | 7,237,795,840 |
| E_tab | 2,542,796,800 |
| E_small | 2,645,504 |
| b_tab | 4.250000251691366 (HQ30UQ4 MEASURED G0) |
| b_small | 32.00853977162764 (f32v2 MEASURED) |
| side_table | 184,307 B (mixed-2p0 catalog+format+manifest MEASURED) |

Pinned: tables Q4, small f32. Solve `b_attn` at equality (G1 `< T` sits strictly under the cell).

### 5.1 Tensor-complete (pack.rs law, no catalog)

| b_mlp | T=1.5 | T=1.75 | T=2.0 |
|---:|---:|---:|---:|
| 1.0 | **1.704893573787** | **2.633905632796** | **3.562917691804** |
| 1.2 | **1.232021426015** | **2.161033485023** | **3.090045544031** |
| 1.4 | **0.759149278242** | **1.688161337250** | **2.617173396259** |
| 1.6 | **0.286277130470** | **1.215289189478** | **2.144301248486** |

All 12 cells `rem_bits > 0`. DERIVED ` (T*N − b_mlp*E_mlp − b_tab*E_tab − b_small*E_small) / E_attn `.

### 5.2 Artifact-complete (+184307 side table)

Catalog tax subtracts **0.000203716163** from every `b_attn` cell. Does not flip a codec band.

| b_mlp | T=1.5 | T=1.75 | T=2.0 |
|---:|---:|---:|---:|
| 1.0 | 1.704689857624 | 2.633701916632 | 3.562713975640 |
| 1.2 | 1.231817709851 | 2.160829768860 | 3.089841827868 |
| 1.4 | 0.758945562079 | 1.687957621087 | 2.616969680095 |
| 1.6 | 0.286073414306 | 1.215085473314 | 2.144097532323 |

### 5.3 What that attention budget buys (native)

| b_attn band | native consume | measured MLP-analog hold |
|---|---|---|
| < 1.125023 | nothing (HGRAVS on attention not transferable; CITED) | — |
| [1.125, 1.288) | Binary | gate min 0.796, not 0.95 |
| [1.288, 2.250) | Residual 2% | rice attn_in hold_min 0.888 CITED descent; not 0.99 |
| [2.250, 3.250) | Uniform q2 | q2 attn_in hold_min 0.835 CITED; not 0.95 |
| [3.250, 4.250) | Uniform q3 | attn_in Q3 hold_min 0.979 CITED; not 0.99 |
| ≥ 4.250 | Uniform q4 / HQ30UQ4 | last generate-proven cheap attn/tables codec (G0) |

Read of §5.1 against §2: **no cell with T≤2.0 and b_mlp≥1.0 leaves attention at a 0.99-class codec** while tables stay Q4. T=2.0 + b_mlp=1.0 is the only cell that reaches Uniform Q3 attention (3.563). Every 1.5 cell is Residual or below. T=1.5 + b_mlp=1.6 leaves attn at 0.286 — no native GEMV codec.

MLP class 3.250025 (the 0.95 hold) + tables Q4 + small f32 is complete **3.61381** tensor / **3.61386** artifact (CITED mixed-q3mlp). That is off every G1 target A–C.

---

## 6. Counterfactual: the same 2.0856, allocated from the curve

mixed-2p0 MEASURED (`PACK_REPORT.json`, `all_required_weight_artifact_bytes=7011764637`):

| class | codec | physical BPW | bytes |
|---|---|---:|---:|
| gate | HGRAVB01 | 1.1250234267290902 | 802,177,344 |
| up | HGRAVR02 2% | 1.2875108157887178 | 918,036,000 |
| down | HGRAVS01 r160_b3 | 0.13161714918473189 | 93,847,197 |
| MLP | mix | **0.8480504639008466** | 1,814,060,541 |
| attn+embed+norm | HGRAVU01 q4 | 4.250142713483966 | 5,197,519,789 |
| complete artifact | | **2.0855934079220506** | 7,011,764,637 |

That is attention left rich and `down` crushed. L54/L58/L59 binary-class hold 0.30–0.38 is what that crush looks like on this capture.

**Same artifact budget, curve-informed discrete assignment** (unpin HGRAVS; put every down on the first all-64 0.95 codec; do not raise tables; do not invent a codec):

| organ | codec | bytes MEASURED/DERIVED |
|---|---|---:|
| down 64 | Uniform q3 | 2,317,370,880 |
| gate 64 | Binary | 802,177,344 |
| up 64 | Residual 2% | 918,036,000 |
| tables | HQ30UQ4 | 1,350,860,880 |
| small | f32v2 | 10,584,840 |
| side table | 2p0 MEASURED | 184,307 |
| **used** | | **5,399,214,251** |
| **attn slack** | | **1,612,550,386** |

```
b_mlp   = 8*(2317370880+802177344+918036000)/17112760320
        = 1.88751979154699          DERIVED
b_attn  = 8*1612550386/7237795840
        = 1.7823662580678705        DERIVED
complete = 2.0855934079220506       MEASURED budget, spent
```

Attention 1.782 is Residual-band. Descent attn_in rice hold_min 0.888, Q2 0.835. No 0.99 hold exists there.

**Would it have survived?** EXPECTATION no. The 2.0856 increment can buy *either* all-down Q3 *or* Q4 attention, not both. The backwards pack bought the wrong one. The curve-informed pack buys the right MLP organ and leaves attention in a band that has never held 0.99. Generate is the gate; this is arithmetic plus the organ screen.

Cheaper curve-informed *partial* uncrush, still at 2.0856, steal only from attention GEMV (2p0 non-MLP 5,197,519,789 minus tables/small):

| k downs → Q3 | steal B | remaining b_attn | complete Δ from 2p0 MLP |
|---:|---:|---:|---:|
| 5 (hold<0.50 plus L50/L55) | 173,712,785 | 4.048 | +0.051669 |
| 8 (binary hold<0.75) | 277,940,456 | 3.933 | +0.082671 |
| 9 | 312,683,013 | 3.894 | +0.093005 |
| 64 | 2,223,523,648 | 1.782 | +0.661369 |

k=8 still cannot keep all attention at Q4 (4.250). It can keep most of it at Q3–Q4. That is the only 2.0856 spend that both uncrushes the collapse set and leaves attention above the Q2 graveyard. It is **not** a generate result. Cheapest missing generate is still mixed-q3mlp-v1 (named change A, all Uniform, no 2048-col confound), not a new 2.0856 pack.

Equal-BPW alternative at the same budget (MLP+attn share the 5,650,134,610 B leftover after tables+small+side): **1.856264661184642** each. That is Residual/q2 interstitial. Q2 does not hold 0.95 on 63/64 downs. Residual does not hold 0.95 on 64/64 downs. Not a candidate.

---

## 7. KILLS / REOPEN_IF

| ID | KILL | REOPEN_IF |
|---|---|---|
| K1 | Native MLP floor in [0.848, 1.6] at 0.95-all-64 | a new native codec, or a capture with rows/dim ≫ 0.05 that moves Q2/Residual/HGRAVS above 0.95 on L54/L58/L59/L62 |
| K2 | HGRAVS r160_b3 as an MLP floor | honest hold on a larger-than-256 capture ≥0.95 on all 64 downs, **and** a native generate of a down-only-HGRAVS pack is English with 0 fallbacks. Packed in-sample 0.92 min is not that |
| K3 | Binary or Residual as a down codec | L54/L58/L59 hold ≥0.95 on a non-underdetermined X |
| K4 | Affine Q2 as the 1.6–2.0 down step | Q2 all-64 hold min is 0.753. Ternary is a different operator and has no Qwen38 mixed magic |
| K5 | 6-layer interpolation of the other 58 | this file. L54/L58/L59 did not exist in the interpolator |
| K6 | Spend a 2.0 increment on attention while down stays HGRAVS | mixed-2p0 backwards alloc, now with a measured collapse on the crushed organ |
| K7 | Treat 2p0/sub15 INCOHERENT as the generate floor | both confounded. REOPEN_IF native non-expand generate of mixed-q3mlp-v1 or of §6 k=8 is incoherent |
| K8 | Act-colscale / activation-weighted *rescale* | already killed (L0 out_proj 0.992→0.918). Protect-vs-rescale stays split. HGRAVS is a fit, not a rescale |

---

## 8. What this lane did not measure

- Any generate, Metal load, TOKEN_NS, TPS. GPU forbidden.
- Attention organs (contract is MLP). Attn bands in §5.3 are CITED descent 6-layer.
- HGRAVS on gate/up as a packed artifact (2p0 never stored those).
- A well-posed down X (17408-wide capture). Replacement capture is another lane.
- simd3 numeric parity on Q3 down.

Cheapest next experiment: named change A (~20 lines, accept `Uniform` in `assert_mixed_mlp_native`) + native greedy generate of **mixed-q3mlp-v1**. That is the first packed point whose MLP side matches the 0.95 hold.

---

```
STATUS
MEASURED_NEGATIVE

CLAIMS
C1. Language N=26895998464, E_mlp=17112760320, E_attn=7237795840, E_tab=2542796800, E_small=2645504. MEASURED. g1-bit-budget-accounting.md §4; pack.rs:673-679.
C2. Native reader accepts Binary/Residual/Hgravs/Uniform bits 2..=8. MLP role-locked. hybrid_decode.rs:601-664, :958-1003.
C3. 192/192 MLP tensors scored this lane, all 64 layers, hold 64 of 256. wall 983.1s, rss_max 4.257G. /tmp/g1_mlp_floor.json.
C4. Calibration vs descent hold: max |Δ|=2.74e-8 (L63 down binary). Same operators.
C5. Binary down hold min=0.3005754758685758 at L58. L54=0.363491, L59=0.383212. Not in the 6-layer sample. MEASURED.
C6. Residual down hold min=0.364203 at L58. MEASURED.
C7. Uniform q2 down hold min=0.752931 at L62; 1/64 ≥0.95. MEASURED.
C8. Uniform q3 is the first native codec with hold≥0.95 on 192/192. Min=0.9662531725591231 (L62 down). 5 tensors sit below the old 0.9679 citation. MEASURED.
C9. Uniform q4 hold min=0.993301 (L32 up). 192/192 ≥0.99. MEASURED.
C10. HGRAVS r160_b3 honest hold min=0.730175 (L9 down). Packed-2p0 in-sample min=0.919630 (L8). Weight min packed=0.152506 (L54). Not a 0.95-all-64 floor. MEASURED.
C11. Inversion table §5.1–5.2. DERIVED from C1 + G0 table/small BPW + 2p0 side_table 184307. No T≤2.0 / b_mlp≥1.0 cell leaves attn at Q4.
C12. Same 2.0855934079220506 budget, all-down Q3 + gate Binary + up Residual + Q4 tables, leaves b_attn=1.7823662580678705. DERIVED §6.
C13. EXPECTATION: that allocation would not survive. Attn 1.78 has no 0.99 hold. Generate not run.
C14. gate/up output space = captured hidden, rows/dim=0.0500. down = reconstructed SwiGLU, rows/dim=0.0147. Both underdetermined. mixer_x absent.

EVIDENCE
E1. crates/hawking-core/src/model/qwen38_pack.rs:673-679
E2. crates/hawking-core/src/model/qwen38_hybrid_decode.rs:601-664, :958-1003
E3. crates/hawking-core/src/model/qwen38_geometry.rs:20-52
E4. workspace/superwave/g1/g1-bit-budget-accounting.md:90-146
E5. /tmp/g1_mlp_floor.py ; /tmp/g1_mlp_floor.json (192 organs, wall_s=983.0766312080086, rss_max_gb=4.2565155029296875)
E6. /tmp/g1_mlp_floor.py stdout: "wrote /tmp/g1_mlp_floor.json wall=983.1s rss_max=4.257G organs=192"
E7. .../mixed-2p0-v1/PACK_REPORT.json complete_physical_bpw 2.0855934079220506 mlp 0.8480504639008466 organ_breakdown
E8. .../mixed-2p0-v1/segments/L*.hq38seg.records.json n_fit_rows=256 rank=160 down _cosine min 0.1525058591718298 (L54)
E9. .../mixed-q3mlp-v1/PACK_REPORT.json 3.6138647373176767; replaced_strided_weight_cosine min 0.9652814877860332 n=192
E10. receipts/ascent-2026-08-16/QWEN38_BPW_DESCENT.json (copy /tmp/QWEN38_BPW_DESCENT.json) layers [0,3,15,31,47,63] coherence_floor hold_min 0.9679
E11. activation-capture-v1/capture-result.json status CAPTURED_REAL_BF16_POST_NORM_HIDDEN n_tokens=256
E12. L0 probe /tmp/g1_mlp_floor_L0.json q3_hold=0.982098 bin_hold=0.861852 packed HGRAVS hold=0.970821 (in-sample)

CHANGES
A1. Created workspace/superwave/g1/g1-mlp-floor-locate.md (this file). No other path written.

TESTS
T1. test -s workspace/superwave/g1/g1-mlp-floor-locate.md
T2. wc -l workspace/superwave/g1/g1-mlp-floor-locate.md
T3. git status --porcelain

RISKS
R1. 256-token capture is underdetermined (0.0500 / 0.0147 rows per in-dim). Hold numbers can move under the replacement capture. The L54/L58/L59 collapse is large enough that a 2× capture is unlikely to turn 0.30 into 0.95, but that is untested.
R2. down X is reconstructed SwiGLU, not a stored intermediate.
R3. Packed HGRAVS output cosine is in-sample (n_fit=256). Honest hold is the thin/eigh operator, not the packed number.
R4. 2.0856 survival is EXPECTATION. Generate is the gate.

UNRESOLVED
U1. Generate of mixed-q3mlp-v1 (cheapest 0.95-MLP + Q4-attn point). Blocked on assert_mixed_mlp_native.
U2. Well-posed down X (17408-wide capture). Other lane.
U3. Whether k=8 collapse-only Q3 + mostly-Q4 attention at 2.0856 is coherent. Costs a pack + generate; not this lane.

NEXT
N1. Do not pack a new 1.0–1.6 MLP artifact. The band has no 0.95-all-64 native codec.
N2. Named change A + native generate mixed-q3mlp-v1.
N3. If that is COHERENT, the generate floor is ≤3.614 and the next cheaper test is mixed-q4down-v1 (2.959) then a k=8 2.0856 uncrush. If INCOHERENT, Q3-MLP+Q4-attn is the wrong split and G0 remains the only proven coherent point.
```
