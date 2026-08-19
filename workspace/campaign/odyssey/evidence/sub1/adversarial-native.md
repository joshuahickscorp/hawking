I'll start by reading the frontier packet in full so the proposals follow its guardrails and output schema.The packet is loaded. Next I’ll pull prior Hawking MoE numbers, bible §29, and patient facts so the proposals stay grounded.I’ll now pull the MoE map, bible §29, and the Q80/GLM receipts that set the active-vs-stored numbers.I’ll read the Odyssey MoE patients, negative science, and the Q80/GLM receipts next so the accounting is exact.I’ll locate GLM 0.167 BPW, bible §29, and native MoE gather kernels so the execution-path claims stay exact.I need Q80/GLM active-mass fractions and native gather kernels so the stored-vs-active split is geometric, not guessed.Lane is **active sub-1 MoE**: expert-gather + extreme per-expert quant + route-conditioned repr. Stored may stay >1; active (touched/token) <1 is the NX win. Selection ratio is **not** a compression stat.

---

## Geometry (do this first — several “sub-1” claims die here)

**O005** `Qwen3-30B-A3B`, 48× MoE, 128 experts, top-8, **0 shared**. Census MEASURED: stored 30,532,122,624 params; expert mass 28,991,029,248 (95.0%); `8/128 = 0.0625`. Honest decode-active (moe_arch_map §3.1, DERIVED; **embed = 1 row, not the table**):

| Touched organ | Params | % of 3.042B |
|---|---:|---:|
| 8/128 experts ×48 | 1,811,939,328 | 59.57% |
| Attn + RMS + QK-norm ×48 | 906,178,560 | 29.79% |
| lm_head + final RMS | 311,166,976 | 10.23% |
| Router ×48 | 12,582,912 | 0.41% |
| Embed **row** | 2,048 | ~0 |
| **Honest active** | **3,041,869,824** | 100% |

Census `active_params=3.353B` / gravity `active_bpw=4.2305` **count the full embed table**. Honest decode drops that 311M. Gravity q3-g32-experts MEASURED: `stored_bpw=4.0253`, `stored_bytes=15,362,682,880`, census-active `1,773,137,920 B/tok`. Honest-active bytes ≈ `1,598 MB/tok` → honest `active_bpw≈4.20` (DERIVED).

**1.0 all-touched budget** = `3.042e9/8 = 380 MB/tok`.  
**q4 attn** ≈ 510 MB (MEASURED gravity `attn=509,632,512`) **already exceeds 380 MB**.  
**Theorem (O005):** all-touched `active_bpw<1` is **geometrically impossible** while attn stays ≥4 complete BPW. Expert-path sub-1 is a different, legal win.

| Metric | q3-g32 (MEASURED/DERIVED) | Fake 16× | Honest expert-path @1.0 | Honest all-touched @1.0 |
|---|---:|---:|---:|---:|
| stored_bpw | 4.0253 | — | still ~4 if only selected change | — |
| expert B/tok | 906 MB | — | 227 MB | — |
| all-touched B/tok | 1.60 GB honest | `0.0625×stored` **illegal** | 227+attn+head | **380 MB** |
| `active_bpw` vs **active** params | 4.20 honest | **0.25 = 16× fiction** | expert-path **1.00** | **1.00** |

`selected/full=0.0625` is a **16× movement lever**, not 1/16 cost (`ODYSSEY_POLICY.moe_policy`, `no_fake_active_density`). mlx `gather_qmm` **computes** selected experts; **full expert body stays resident** (`O005_NX_gather.json`). Q80 graph: **no 512-way device gather** (host fallback, `QWEN80_UNIFORM_Q4_VELOCITY_BASELINE.json`).

**Anchor:** q3-g32-experts battery **10/12 Δ0** vs 4-bit mlx (SPECIMEN, not full Doctor). **q2-g32-experts FAILED**; localize **protect gate/up + router**. Affine complete ≈ `bits + 32/group` (fp16 scale+bias): q3-g32≈4.0, q2-g32≈3.0 (`affine2_kernel_spec.md`). **Affine grouped cannot reach complete sub-1.**

**Routing (O005, 573 tok MEASURED):** entropy 6.16/7.0, **0 cold**, top16=18%, most-pop=1.33%, `P(E_t\|E_{t-1})=0.393`, adjacent overlap **0.408**, cross-layer Jaccard **0.035**. Same 0-cold on O003/O006. **Cold-expert compression is N/A.** Single-expert zero (hot 49 **and** random 77) → **Δhits=0** on 12-item battery: that battery **cannot see one expert**.

**Q80 prior (MEASURED, real acts, organ-cosine bar 0.8604):** binary_g gate **1.1269 BPW** cos 0.8586–0.8932; binary+rice-q1 @2% up **1.2918**; HGRAVS r160-b3 down **1.27** (post-SwiGLU); mixed expert **1.22957**, complete stored **1.43051** @8-bit non-expert. Cross-expert pairwise cos **0.004 / ~0** (`QWEN80_CROSS_EXPERT_STRUCTURE_NEGATIVE.json`). **NS-inter-expert-redundancy / NS-cross-expert-tying stay dead** unless a parent measures pairwise cos ≥0.10 (O005 pairwise = UNKNOWN).

**Kills that stay dead on this lane:** NS-raw-weight-pq-vq-at-one-bit; NS-uniform-subbit-allocation; NS-entropy-coded-PQ-indices; NS-large-expert-cache (0 cross-layer reuse; Jaccard 0.035 agrees); QTIP Metal trellis (Type-1); learned LUT gather (Apple GPU punished); affine-q2 uniform.

---

## P1 — Packed-sign experts + device gather-GEMV

**mechanism:** 1-bit sign payload + **amortized** scale (per-tensor or shared-per-layer, not per-group affine) on **all** routed experts; native **indexed** 1-bit GEMV over top-k only. BitNet-1 / Q80 `binary_g` without the 0.13 group-scale tax.

**complete_byte_accounting:**

| Bin | What | Rate |
|---|---|---|
| payload | packed signs, 1 bit/W | **1.000** nominal |
| scales | **not** g32 fp16+fp16 (+1.0). Per-tensor fp16: `16/(768·2048)=0.000010` on gate/up; per-row: `16/2048=0.0078` gate/up, `16/768=0.0208` down | **0.00001–0.021** |
| biases | none if sign+scale; affine β is the q2-g32 trap | **0** if dropped |
| tables/codebooks | none (lookup-free; LUT is Type-1 dead here) | **0** |
| offsets / correction / tiers | none in P1 | **0** |
| router metadata | router **not** 1-bit (protect; 12.6M params) | counted in non-expert |
| alignment/container | pack to 32-bit words; pad UNKNOWN, typically ≪0.01 | UNKNOWN |
| reconstruction state | fused dequant in FMA; **no** f16 expand buffer | **0** if fused, else +2 B/W **fake** |

Complete expert ≈ **1.00–1.13**. Affine q1-g32 ≈ **2.0 complete** — not this codec.

**stored_bpw / active_bpw** (O005, experts at ~1.01, attn/head stay q4-g64 MEASURED rates):

| | stored_bpw | expert_active_bpw | all_touched_active_bpw | physical B/tok |
|---|---:|---:|---:|---:|
| expert-path | ~1.01×0.950 + 4.5×0.050 ≈ **1.18** | **~1.01** | — | expert **227 MB** |
| + always-on q4 | same stored | 1.01 | **~2.5** (227+510+175+7)/380·1 | **~0.92 GB** |

Expert-path **≈1.0, not <1**. All-touched **not** sub-1.

**expected_reachable_bpw:** expert-path **1.00–1.13** (Q80 binary_g 1.1269 is the measured ceiling of group-scales; amortized scale is the only honest cut). Sub-1 requires dropping group scales **and** Doctor holding — UNKNOWN. Sub-0.2: **no**.

**quality_risk:** **gate_proj** (q2 already failed; Q80 binary gate only 0.8586–0.8932 vs bar 0.8604). Router if 1-bit (do not). Down more tolerant (organ inversion; Q80 HGRAVS down 0.886–0.898 @1.27). Uniform 1-bit will collapse; this is **sign-codec + protect gate or add P2 residual**.

**cheapest_falsifier:** One layer, real routed X (not gaussian). Encode gate/up/down as sign+per-tensor scale. Organ-cosine vs Q80 bar 0.8604 **and** 12-item battery vs q3-g32 10/12. Kill if gate cos <0.86 **or** Δhits≤−2. Do **not** use PQ-proxy (Qwen3.8 trap, output-div ~0.69).

**execution_path:** Native **credible, not present**. LIVE: `dispatch_binary` (dense Qwen3.8), `moe_topk_gate`, `moe_grouped_gemm_q4` (wrong codec). DEAD for this patient: `QwenMoE::load` Unimplemented. **Must not** expand 1.81B signs → f16 then GEMV. Kernel: gather-indexed `x * s * sign(W)` FMA, 8 experts × 3 GEMVs × 48 = 1152 launches/tok unless grouped. FLOPs DERIVED: **1.812e9 MAC** expert-path/tok (1 MAC/expert-param). Kernel µs/tok **UNKNOWN**. mlx 35.4 TPS SPECIMEN @~1.77 GB/tok ≈ 62 GB/s vs 819 GB/s roof → **kernel-bound**; bytes only help if this kernel is cheaper than q3 `geo_tpr64`.

**applicability:** MoE routed experts (O005/O006 8/128; O003 6/64 **except shared**; Q80 10/512; O010 GLM-Air 8/128+shared). Not dense-primary. Not DeltaNet/attn.

**confidence:** **INFERRED** geometry + **MEASURED** Q80 binary_g. Doctor on O005 1-bit = **HYPOTHESIS** (q2 already failed).

**transfer:** Codec: likely O006 (same 8/128). O003/Q80/O010 need **shared-expert exception**. Not two-Qwen universal (`TRANSFER_MATRIX`). BitNet/1-bit PTQ are the prior art; they are **not** a Doctor seal.

---

## P2 — Organ-inverted mixed extreme (Q80 mixed, pushed under 1.0)

**mechanism:** Do **not** uniform-1-bit. Q80 mix: **binary gate**, **binary + sparse rice residual on up**, **activation-weighted low-rank on down**; then cut residual% / rank until expert complete **<1.0**. Route-condition the residual: **only selected experts’ residual bytes move**.

**complete_byte_accounting** (Q80 MEASURED, then O005-scaled HYPOTHESIS):

| Organ | Q80 MEASURED complete | Bytes that must be counted |
|---|---:|---|
| gate binary_g | **1.1269** | signs + group scales (the 0.127) |
| up binary+rice-q1 @2% | **1.2918** | binary base + rice **indices** + q1 magnitudes + outlier bitmap; `bits/outlier=8.24` |
| down HGRAVS r160 b3 | **1.27** | U,V factors + 3-bit codes + scales; **not** “rank/min(m,n) bits” |
| mixed expert | **1.22957** | weighted 3 organs |
| stored complete Q80 @8-bit non-exp | **1.43051** | `0.97032·expert + 0.02968·nonexpert` |
| **O005 target cut** | expert **<1.0** | drop gate group-scales (P1) **or** residual <2% **or** r<160; recount **all** bins |

Active vs stored split (legal): residual/HGRAVS **sidecar is stored for all 128**; DRAM moves sidecar only for top-8. **Density of a selected expert ≈ stored expert density** if every expert has the same sidecar. Active_bpw_expert **does not** drop unless residual is **route- or token-conditional** (P4/P5). Physical B/tok **does** drop 16× vs full sidecar residency.

**stored_bpw / active_bpw:**

| Patient | stored complete | expert_active_bpw | all_touched_active_bpw | expert B/tok |
|---|---:|---:|---:|---:|
| Q80 mixed as-is | **1.43** (MEASURED envelope, **not packed**) | **~1.23** | UNKNOWN (DeltaNet+head) | `10/512·77.3B·1.23/8` ≈ **232 MB** DERIVED |
| O005 transplant HYPOTHESIS | `1.23×0.95 + ~4.5×0.05` ≈ **1.4** | **~1.23** | **~2.6** | **~279 MB** |
| pushed under 1.0 | UNKNOWN | **0.85–1.05** if residual/rank cut holds | still **>2** if attn q4 | **193–238 MB** |

**expected_reachable_bpw:** expert-path **1.05–1.30** now (measured Q80 band); **0.85–1.05** only if gate scale-amortize **and** up residual <2% **and** down rank cut — UNKNOWN jointly. Sub-0.2: **no**.

**quality_risk:** **up_proj residual under-coverage** (Q80 up only 0.864 cos, barely over 0.8604). **gate** next. Down is the slack organ. Packed+generated **not** done on Q80 (`SCREEN_PASSED_NOT_YET_PACKED`). O005 q2 fail = uniform; this mix may still fail on **gate**.

**cheapest_falsifier:** O005 layer 10 (or 0 — NS-layer-zero-is-different-source LIVE): real routed X; Q80-mix encode; organ-cos vs 0.8604. Then **one** fast-Doctor vs q3-g32. Kill mix if any organ <bar **or** Δhits≤−2. Then sweep residual 2%→1%→0.5% (Q80 0.5% up already **failed bar** at 1.37 BPW on down-like organs — don’t assume).

**execution_path:** Three kernels: 1-bit gather-GEMV (P1) + sparse CSR/rice residual **add** on selected up rows + HGRAVS factor GEMV (`q80_hgravs01_factor_matvec*` LIVE on Qwen3.8 dense). Decode cost: 1-bit FMA + `nnz` residual adds + `r` factor MACs. If residual is 2% of `768·2048` at 8-bit: `0.02·1.57e6·1 B` ≈ 31 KB/tensor; ×8 experts ×48 ≈ **12 MB/tok** extra — small vs 279 MB base. **Native path partial**; no O005 MoE graph. Host gather (Q80) is **not** a win.

**applicability:** MoE experts with SwiGLU + measured organ split. O005/O006 first. O003 **protect shared** (zero → 0/12). Q80 already the donor. Dense: organ inversion only, no gather.

**confidence:** **Highest in this lane** for “near-1 expert-path”. Sub-1 is a **cut**, not a transplant. Evidence: Q80 MEASURED mix + O005 q2 localization.

**transfer:** O006 likely. O003 retune (shared + 6/64). Do not claim from two Qwens. Prior art: BitNet (gate) + sparse correction (not AQLM) + ASVD/HGRAVS (down). QuIP#/AQLM/QTIP are 2-bit-class, **fatter** than this mix.

---

## P3 — Activation-aware sub-bit expert (GLM 0.167) + gather

**mechanism:** The **only** Hawking sub-1/5 existence proof: **0.755 cos @ 0.167 BPW on REAL activations**. Put **that** codec on routed experts; evaluate on **routed X**, never gaussian. Gather top-k so physical bytes = 0.167-class × selected mass. **NS-raw-weight-pq-vq-at-one-bit is dead** — this is legal only as **activation-aware / source-changing**, not raw-weight PQ at 1.0/0.5.

**complete_byte_accounting:** Codec internals of the 0.167 artifact = **UNKNOWN** (packet states the rate; this lane did not re-open the shard). Mandatory bins **whatever it was**:

| Bin | Rule |
|---|---|
| payload | indices/codes at the claimed 0.167 |
| codebook/tables | **amortized across experts/layers or they dominate**. A “0.167 payload + 3 BPW codebook” is **3 BPW** (guardrail 1) |
| scales | counted |
| correction streams | counted if present |
| alignment/container | counted |
| reconstruction | if decode **materializes** dense W, add working-set bytes/tok + FLOPs — **fake** if omitted |

If it is PQ `d=48,k=256`: payload 8/48=0.167; codebook `256·48·2=24,576 B`. Shared over 128 experts × `768·2048` of one layer: codebook/payload ≈ 24,576 / (128·768·2048·0.167/8) ≈ **0.005** relative → complete ≈ **0.17** **only if shared**. Per-expert codebooks kill the rate. **Entropy-code the indices: NS-entropy-coded-pq-indices DEAD** (0–0.7%).

**stored_bpw / active_bpw** (O005, **if** complete 0.167 really transfers):

| | stored_bpw | expert_active_bpw | all_touched_active_bpw | expert B/tok |
|---|---:|---:|---:|---:|
| experts @0.167, rest q4 | `0.167×0.95+4.5×0.05` ≈ **0.38** | **0.167** | **~1.9** (38+510+175+7 MB)/380 | **37.8 MB** |
| census-style fake 16× | — | **illegal 0.010** | — | — |

**Sub-0.2 is an expert-path number. All-touched stays ~1.9.** Stored **can** go sub-1 on O005 because experts are 95% of mass — that is a **stored** win riding on this codec, still gated by codebook amortization + Doctor.

**expected_reachable_bpw:** expert-path **0.15–0.40** if GLM-rate complete accounting holds and codebook is shared; **0.7–1.2** if codebooks are per-expert (fake-density trap). Quality at 0.755 cos is **below** Q80 bar 0.8604 and far from generation-safe (GLM-5.2 sub-bit **generation collapsed**: “France”→`combust`, “2+2”→`rus`; dead_levers 0.116–0.157 cos @0.75 BPW **weight-space**).

**quality_risk:** **generation / Doctor**, not cosine. Limiting organ: **gate** (F1 dominant_failure_organ=gate). If it fails, **localize which expert/channel** (guardrail 3) — a gibberish 0.167 artifact is still data.

**cheapest_falsifier:** Replay GLM codec **recipe** on **one** O005 expert, **real** routed activations, complete_bpw **including codebook**. Kill if (a) complete >0.5 once tables counted, (b) organ-cos <0.70, (c) layer-0 vs mid-layer diverge (layer-0 is a different source). Do **not** run gaussian proxy.

**execution_path:** If PQ/AQLM-like: **LUT gather** — Apple GPU **punished** (dead_levers learned codebook). That is a **native-path hole**: 0.167 stored with expand-to-f16 GEMV is a **fake NX**. Need lookup-free decode (QTIP-class) **but QTIP Metal is Type-1 dead**. **Credible native path: UNKNOWN.** This is the highest density lever and the weakest execution story.

**applicability:** Expert tensors with **captured real X**. O005 first (best expert-share of active: 59.6%). O010/O012 GLM family: donor-shaped, still re-measure (weight-space GLM-5.2 path is Type-1 NO-GO). Dense: this is the GLM lesson, not an MoE-only trick — but this lane only gathers experts.

**confidence:** **HYPOTHESIS**. Precedent exists; **codec completeness + native decode + Doctor** are all UNKNOWN. Do not ship 0.167 as reachable-NOW.

**transfer:** Only after complete_bpw + Doctor on two **non-sibling** MoEs (O005 and O003 or O010). O006 is not an independent transfer.

**prior art:** AQLM / PQ (indices+codebook), GLM activation-aware 0.167 (Hawking), **not** QuIP#/QTIP-as-Metal.

---

## P4 — Route-score Matryoshka (T0 always, T1 iff high gate weight)

**mechanism:** Uniform routing kills popularity alloc. **Within** top-8, softmax weights are **not** equal. T0 = P1 1-bit (executable). T1 = residual/HGRAVS repair **loaded only for selected experts with gate score ≥ τ**. Stored holds T0+T1 for everyone; **active density < stored density** without cold experts. This is the clean stored-vs-active split when entropy≈max.

**complete_byte_accounting:**

| Bin | Stored | Active (per tok) |
|---|---|---|
| T0 payload+scale | all 128 experts | top-8 T0 |
| T1 residual/factors | all 128 | only selected **and** score≥τ |
| T1 index/bitmap | occupancy of T1 | same, selected only |
| router | always, protect 8-bit | always |
| τ / tier metadata | tiny per layer | tiny |
| unused T1 | **in stored complete** | **not in DRAM** |

Let `f` = fraction of selected expert-slots that hit T1.  
`stored_expert ≈ T0 + T1`.  
`active_expert ≈ T0 + f·T1`.

Example **arithmetic** (not measured): T0=1.00, T1=0.30, `f=0.25` (top-2 of 8) → stored **1.30**, active **1.075**. To get **active<1**: T0+f·T1<1 → e.g. T0=0.70 (P3-class) + f=0.25·0.40=0.10 → active **0.80**. **f is UNKNOWN** until score-gap is measured.

**stored_bpw / active_bpw:** UNKNOWN until `f`. Bound: `active_bpw ∈ [T0, T0+T1]`. Adjacent-token overlap **0.408** (MEASURED) = ~3.3/8 experts reused next token — T1 cache across **tokens**, not layers (Jaccard 0.035; NS-large-expert-cache).

**expected_reachable_bpw:** active expert **0.7–1.1** if T0~0.7–1.0 and `f≲0.3`; stored **1.1–1.6**. Sub-0.2 active only if T0 itself is P3. If scores inside top-8 are flat, `f→1`, P4=P2.

**quality_risk:** **low-score experts still enter the weighted sum** (`norm_topk_prob=true`). Starving T1 on the tail can still break the residual. Limit: **gate** + tail experts on hard prompts (math already 2 misses at q3).

**cheapest_falsifier:** **Measure first** (not a codec): over ≥1000 held-out tokens (NS-calibration-88-tokens), histogram of **renormalized top-8 scores**. Kill P4 if mean(max)/mean(min) < **2** or mass of top-2/8 < **0.40**. If alive: T0-only Doctor vs T0+T1-on-top-2. Kill if T1-on-top-2 does not beat T0-only.

**execution_path:** Router already returns scores (`moe_topk_gate` + `normalize_topk`). Extra: predicate → bind T1 buffers for ≤8 experts. Prefetch T1 using `P(E_t|E_{t-1})=0.393` is **weak** — not a prefetch win (R-predictable-route-prefetch reopen_if near-uniform transitions). Kernel cost: T0 gather-GEMV + conditional residual. **f** unknown ⇒ decode cost UNKNOWN.

**applicability:** Softmax (O005/O006) and sigmoid-bias (O010/O003) both have **unequal** selected scores. Shared experts: **always T1** (O003 shared zero → 0/12).

**confidence:** **HYPOTHESIS** on `f`; geometry of stored-vs-active is **DERIVED**. Cheapest science in the lane.

**transfer:** The **score-gap measurement** transfers as a method; the τ does not. O003 mildly peaked (top16=31%) may have **larger** `f`-skew than O005.

---

## P5 — Token-conditional residual (activation-selected repair)

**mechanism:** 1-bit T0 on all expert weights. Repair **channels this x actually uses** (post-SiLU sparsity / high-|x| columns of gate/up, high-|silu(g)*u| rows of down). Stored = T0 + full residual map. Active = T0 + **sparse** residual for selected experts **and** active channels. Active_bpw < stored_bpw of the **same** expert. Distinct from P4 (token, not route-score). Distinct from FFN block-256 skip (**DEAD** Type-1, 0.2% @99% recall on **neuron skip for compute**). This is **which residual bytes to move**, not “skip the GEMV”.

**complete_byte_accounting:**

| Bin | Stored | Active |
|---|---|---|
| T0 1-bit + scale | all W | selected W |
| residual values | CSR/bitmap of **all** planned nnz | nnz **hit this token** |
| residual indices | same | same subset |
| hit-map / threshold τ_x | per tensor | evaluated on x (compute, not bytes) |
| if residual is 5% of W at 8-bit | +0.40 BPW stored | `sparsity(x)·0.40` active |

SwiGLU intermediate sparsity on O005 real X = **UNKNOWN**. Q80 down was scored on **post-SwiGLU X** (1296 token-expert pairs) — the capture path exists.

**stored_bpw / active_bpw:** UNKNOWN. Alive only if intermediate sparsity `s` is high. Active expert ≈ `1.0 + 8s` (8-bit repair on fraction `s`). Sub-1 active ⇒ `s<0`. Wait: `1.0 + 8s < 1` ⇒ **s=0** if T0 is 1.0. **So P5 cannot make 1-bit T0 into sub-1**; it can (a) **restore Doctor** at small extra active bytes, or (b) allow **T0<1** (P3) with sparse repair. Honest: P5 is a **quality lever**, not a density lever, unless T0 is already sub-1.

**expected_reachable_bpw:** active expert **1.02–1.20** typical (T0=1 + small repair); **0.2–0.5** only stacked on P3. Do not advertise P5 as sub-1.

**quality_risk:** If SiLU(gate) is **not** sparse, residual is dense → P2. Limit: **down_proj** rows that still fire.

**cheapest_falsifier:** Histogram `|silu(x@Wgate)·(x@Wup)|` on real O005 routed tokens, one layer. Kill if p50 nonzero fraction > **0.5** (then “sparse repair” is a dense add). Cheap, no codec.

**execution_path:** Native: fused 1-bit GEMV + **indexed row/col residual add** (gather of CSR). Extra FLOPs `O(nnz)`. If `s` low, kernel is 1-bit only. If `s` high, this **is** the Q80 rice residual with an activation mask. Decode cost UNKNOWN.

**applicability:** SwiGLU experts (all current MoE patients). Not attn.

**confidence:** **HYPOTHESIS**. Falsifier should run **before** P3/P2 residual design.

**transfer:** Sparsity is patient-specific. Measure O003/O006/Q80 separately.

---

## P6 — Device selected-residency + fused gather-GEMV (physical-byte NX)

**mechanism:** Stop treating mlx `gather_qmm` as NX. **Working set = selected expert packed bytes + always-on organs.** Unselected experts stay in the file/mmap, not in the Metal committed set. Token-dimension cache: adjacent overlap **0.408** ⇒ keep last-token’s ~3 experts hot. **Not** a 64 GB expert cache (NS-large-expert-cache). Codec-agnostic; **multiplies** P1–P5.

**complete_byte_accounting:** This mechanism **does not change stored_bpw**. It changes **DRAM/resident**.

| Bin | Count? |
|---|---|
| packed selected expert payload+scales+sidecar | **yes, every tok** |
| unselected experts | stored only; **0** if not touched (guardrail 5) |
| always-on attn/head/router | **yes** |
| staging buffers / scratch f16 | **yes if exist** — this is where fake NX hides |
| KV / DeltaNet state | not weight-BPW; still resident (Q80 HIGH sensitivity) |
| mmap page waste | UNKNOWN; 16k pages × expert stride |

**stored_bpw / active_bpw:** stored unchanged. `active_bpw` = density of **what the kernel reads**. Physical B/tok at current q3: expert **906 MB** + attn/head **685 MB** = **1.59 GB** honest (DERIVED). At P1: **~0.92 GB**. At P3: **~0.73 GB**. mlx today: **full 14.5 GB expert body resident** even though 906 MB is used.

**expected_reachable_bpw:** Not a BPW mechanism. Physical expert B/tok: **906 MB (now) → 227 MB (P1) → 38 MB (P3)**. All-touched floor still **attn+head**.

**quality_risk:** **None representational** if the bytes are the same codec. Risk is **wrong expert bind** (silent route bug) → Doctor gibberish. Localize: compare gathered ids vs `moe_topk_gate`.

**cheapest_falsifier:** Trace Metal buffer binds on one token: <REDACTED> committed vs `8/128·expert_body`. Kill if committed ≥ 50% of full expert body (still “dense residency”). Second: A/B TPS on a **clean GPU lane** vs mlx gather_qmm — if TPS Δ<10% at 4× less resident, **kernel-bound** (expected; packet: runtime often kernel-bound). That kills “bytes will make TPS” **not** the residency win (memory is the NX).

**execution_path:** Required kernel: **device** grouped-GEMM with expert-index (not Q80 host fallback). `moe_grouped_gemm_q4` + `moe_gather_combine` + `moe_block_fused` are the Q4_K sketch; need codec variants (1-bit / HGRAVS / mixed). Prefetch window=1 token using overlap 0.408. Cross-layer prefetch: **don’t** (Jaccard 0.035). `QwenMoE::load` still Unimplemented — this **is** the native story the representation needs.

**applicability:** All sparse MoE. O005/O006 16×; Q80 51× (`10/512=0.01953`); O003 10.7× **plus shared always resident**; O010 16× + shared + layer-0 dense always.

**confidence:** **MEASURED** that mlx/Q80 do not do this; **INFERRED** that it is the NX. TPS gain **UNKNOWN** and likely small until the GEMV itself is cheap (q3 `geo_tpr64` lesson).

**transfer:** **MOE-UNIVERSAL** as a primitive (`R-sparse-active-expert-gather`, O005 origin, O006 TRANSFERRED_UNCHANGED). O003 NOT_TESTED in the matrix — re-measure shared-expert residency.

---

## P7 — Intra-top-k bit taper + shared-expert firewall

**mechanism:** Among the k selected, allocate bits by **this token’s** gate mass (not corpus frequency — O005 is uniform; NS-calibration-88-tokens). Example policy: rank-1 expert T1 (P2 mix), ranks 2–4 T0+small residual, ranks 5–8 T0-only. **Shared experts (O003/O010/Q80) stay ≥4–8 bit** and are **not** in the taper. Always-on firewall is the O003 lesson (shared zero → 0/12).

**complete_byte_accounting:** Stored: **max** codec per expert (everyone must be able to be rank-1). Active: Σ_i complete_bytes(codec(rank_i)). Same stored-vs-active split as P4 with **k discrete rungs** instead of a τ.

O003 extra tax (MEASURED census): shared **449.8M params**, 0.90 GB bf16, **always**. At q4 ≈ **225 MB/tok** before any routed expert. O003 1.0 all-touched budget ≈ 378 MB ⇒ shared q4 **eats 59%**. Routed-expert sub-1 cannot save O003 all-touched.

**stored_bpw / active_bpw:** stored = hottest codec (wasteful unless experts are **re-encoded** at multiple rungs — then count **all rungs** in stored: T0+T1+T2 Matryoshka). Active = mean rung of selected. If stored only keeps one codec per expert, taper **cannot** reduce stored and only reduces active if you **re-quantize on the fly** (decode cost; usually fake) **or** store multiple rungs (P4).

**Honest version = P4 with 3 rungs.** Do not claim taper without storing T0/T1/T2.

**expected_reachable_bpw:** same band as P4. O003 all-touched sub-1: **not** via experts.

**quality_risk:** Tail experts at T0 on **thinking** traces (O005 `enable_thinking`; battery already misses 2 math items at q3). Shared-expert under-quant on O003/O010/Q80.

**cheapest_falsifier:** Same score histogram as P4. Then: force experts 5–8 of each token to **zero** (stronger than T0). If battery Δhits=0, tail is crushable; if Δhits<0, taper has a floor. Cheap (no new codec). Note: whole-expert-index zero across layers did **not** move the 12-item battery — **must zero the tail of the live route**, not a fixed id.

**execution_path:** Rank by scores already in the gate kernel. Bind per-expert codec id in the grouped GEMV (mixed dispatch like `MixedGpuWeight::{Binary,Residual,Hgravs,Uniform}`). Launch heterogeneity is the cost: 8 different codecs/layer is **worse** than one grouped kernel. Prefer **2 rungs** (T0/T1) not 8.

**applicability:** O005/O006: taper only. O003/O010/Q80: **firewall shared (+ Q80 DeltaNet, O010 layer-0 dense, O006 vision tower)**. Dense: N/A.

**confidence:** **HYPOTHESIS**. The **firewall** part is MEASURED (O003/O005 organ zeros). The taper part waits on P4’s histogram.

**transfer:** Firewall: yes, any MoE with shared/dense-replace. Taper: only if score-skew exists.

---

## Reachable-NOW vs not

| Target | O005 now | Credible path | Blocker |
|---|---|---|---|
| stored sub-1 | **no** (4.03 affine) | P3 codebook-amortized **or** 95% experts at 0.167 | Doctor + complete codebook |
| **expert-path active <1** | **no** (q3 4.03; Q80 mix 1.23) | P1→~1.01; P2 cut; P4 `T0+f·T1`; P3 0.17 | gate Doctor; q2 already failed |
| **expert-path active <0.2** | **no** | **P3 only** | native LUT; GLM gen collapse; fake codebook |
| **all-touched active <1** | **no** | attn ≲2 **and** lm_head ≲4 **and** experts ≲0.3 | **geometry**: q4 attn 510 MB > 380 MB budget |
| physical B/tok cut | mlx **full 14.5 GB resident** | **P6** × any codec | kernel-bound TPS; Q80 host gather |
| cold-expert stored skip | **no** | none | **0 cold** O005/O003/O006 |

**Coupling (not this lane, but the all-touched wall):** lm_head is 10.2% of O005 honest active (311M, **full table every tok**). Vocab-norm screen Type-1 dead. 4-bit head is 175 MB MEASURED (round8 lm_head Δ0). Still not enough if attn stays q4.

---

## Native / decode cost (all proposals)

| Primitive | Status |
|---|---|
| `moe_topk_gate` softmax+renorm | LIVE, **right for O005/O006**, wrong for O010 sigmoid-bias |
| `moe_grouped_gemm_q4` / `moe_block_fused` / `moe_gather_combine` | LIVE Q4_K sketch, **wrong codec** for P1–P3 |
| `dispatch_binary` / HGRAVS factor / uniform `geo_tpr64` | LIVE **dense** Qwen3.8 |
| indexed 1-bit / mixed expert gather | **absent** |
| `QwenMoE::load` | **Unimplemented** (Phase 3) |
| mlx `gather_qmm` | SPECIMEN; **full body resident** |
| Q80 device 512-way gather | **absent** (host fallback) |
| DSV4F expert cache probe | storage only; **0 matvec** |

FLOPs expert-path O005: **1.812e9 MAC/tok** (DERIVED). Reconstruction extra = **0** if fused, **=dequant of selected W** if expand-first (illegal win). Kernel µs **UNKNOWN** on this patient. Do not quote mlx 35.4 TPS as Hawking.

---

## Doctor protocol (all)

Fast 12-item is **too narrow** (G046/G048). q3 held 10/12 because **math already fails**. A “pass” at sub-1 that keeps 10/12 is **not** capability. Required: held-out battery **disjoint** from calibration; **localize** organ/expert/channel on fail; never cosine-only; never proxy activations.

---

## What this lane will not propose

Cross-expert templates / tying / merge (NS + Q80 cos≈0). Uniform affine q1/q2 as frontier. Entropy-coded Lloyd indices. QTIP Metal. Large expert RAM cache. Popularity alloc on O005-class uniform routers. Counting `0.0625×stored_bpw` as active_bpw.

**First measurements (cheap, sequential):** (1) top-8 score histogram ≥1000 tok → kill/keep P4/P7. (2) SwiGLU |act| sparsity → kill/keep P5. (3) O005 pairwise expert cos → keep NS-inter-expert dead. (4) One-layer Q80-mix on real X vs 0.8604 → P2. (5) Sign+per-tensor-scale vs binary_g 1.13 → P1. (6) Buffer-bind census → P6. **P3 last** (codec+kernel unknown, Doctor history bad).
