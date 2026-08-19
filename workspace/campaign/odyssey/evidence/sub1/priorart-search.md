I'll read the campaign packet first, then gather the published extreme-quant results those proposals have to sit on.Packet read. Next I’ll pull Hawking’s existing numbers and the published extreme-quant papers so the proposals sit on complete-BPW, not slogans.I’ll pull Hawking’s measured sub-bit results and the paper PDFs in parallel so the complete-BPW table is sourced, not inferred.AQLM’s complete-BPW formula is in hand. Next I’ll pull QMoE, LittleBit, QTIP, and Hawking’s measured sub-bit receipts so the proposals don’t re-invent those numbers.I have the main paper numbers. Next I’ll lock Hawking’s measured GLM/Qwen results and the remaining sub-1 papers (LittleBit, BTC-LLM, NanoQuant) so the proposals stay on measured complete-BPW.Paper complete-BPW is enough to draft the table. I’ll now read Hawking’s measured GLM/Qwen results and the cost-model files so the search proposals don’t fight existing negatives.I have the SOTA numbers. One last pass on Hawking’s GLM 0.167 receipt, then I’ll write the prior-art table and the structured proposals.I have the published complete-BPW numbers and Hawking’s negatives. Next I’ll write the SOTA table and the search proposals in the packet schema.

---

# Prior-art SOTA + sub-1 search (priorart-search lane)

**Headline:** published PTQ that stays *generation-usable* bottoms out near **~2 complete BPW** on dense LLMs (QTIP / AQLM / QuIP#). Honest **stored sub-1** exists in three families only: **QMoE entropy-of-ternary** (0.81 complete on a 1.6T Switch MoE), **binary-codebook PQ of ±1** (BTC-LLM ~0.77–0.88 complete on Llama-2-7B, PPL still single-digit), and **binary low-rank factors** (LittleBit/NanoQuant: “0.1 BPW” is **not** complete on 7B ≈0.75; on 70B ≈0.23). Hawking’s one real sub-⅕ (GLM **0.755 cos @ 0.167 BPW**) is activation-aware, not a published PTQ codec. **Active** sub-1 on Odyssey MoE is cheaper than stored sub-1 and does not need a new codec.

All quality numbers below are the papers’ PPL / zeroshot, **not Doctor**. Treat them as existence proofs + accounting templates, not Odyssey gates.

---

## 0. Complete-accounting formulas (use these; do not invent)

| Family | Bytes that must enter `complete_bpw` | Nominal vs complete |
|---|---|---|
| Affine grouped (mlx qk-gG) | payload `k·N` + per-group scale+zero `32/G` + container | q2-g32 ≈ 2 + 1.0 = **3.0** if 16+16 scale/zero; q2-g128 ≈ **2.25** |
| AQLM (M codebooks, B-bit, group g) | `16 g M 2^B` codebook + `d_out (d_in/g) M B` codes + `16 d_out` scales. Eq. (10) Egiazarian 2024 | Llama-2-70B `gate` 2×8-bit g=8 → **2.002**. “Avg bits” **excludes** embed/ln/lm_head |
| QuIP# E8P | 2-bit payload + **1 KiB shared** E8P (~**0.01** BPW) + Hadamard signs (`16n` FT / `n` no-FT) | Bielik-Q2-Sharp: **~2.4 effective** on a 3.26 GB 7B-class body |
| QTIP TCQ | k-bit bitstream + (HYB) **2 KiB** LUT or (1MAD/3INST) **0** codebook + RHT signs | Closest 2-bit method to nominal=complete |
| VPTQ residual VQ | indices + **2^16×8×FP16** codebook + residual 256 + perm + norms | Bielik: **2.0 nominal → ~3.58 complete / 5.0 GB**. Fake-density poster child |
| QMoE ternary+dict | ternary payload entropy-coded + **shared 2^16 dict** + row offsets + `{wmin,wmax}` + **uncompressed dense** (attn/embed) | c2048: **158.6 / 3142 GB = 0.807 complete** (full ckpt) |
| BitNet b1.58 | ternary weights + absmean scale + **FP16 embed** + INT8 acts | 3B: 2.22 vs 7.89 GB = **~4.5×** not 10×. Complete **~3.6 BPW** of the 16-bit body |
| STBLLM N:M + residual | `N_param = 2 r_sal + 1(1−r_sal)` then `× N/M` + `N_store = 2 + 1/b_size` | BTC-LLM: naïve 2:4 **mask = 1.25 BPW**, not 0.5. STBLLM “0.55” **excludes embed** |
| BTC-LLM binary codebook | `v·c + ⌈log2 c⌉ · mn / v` + ARB scale/bias. Transform **fused**, 0 store | Llama-2-7B **0.65 GB @ “0.7-bit”** (codebook **1.2%**). Complete **0.65/13.48×16 = 0.77** |
| LittleBit / NanoQuant | `r(n+m)` binary + **16(n+m)** scales (+ residual path if used). LittleBit also `32 r` latent | LittleBit Eq. (A.16): `b = [2r(d_in+d_out)+32(d_in+d_out)+32r]/(d_in d_out)` **per linear**. Embed still FP16 unless counted |
| GGUF IQ1_S / IQ1_M | 1-bit q + superblock scale + **importance matrix** | Stated **1.56 / 1.75 BPW** on quantized tensors. Whole-file higher (embed often Q5/Q6) |

**Hawking negatives that kill whole published families on our parents:**

| ID | Kill | Reopen only if |
|---|---|---|
| NS-raw-weight-pq-vq-at-one-bit | raw-weight PQ/VQ @ ~1 BPW collapsed 6/6 on qwen3-235b real forward | source **changes** (QAT, prune, share, factor) |
| NS-entropy-coded-pq-indices | Lloyd indices 0.0–0.7% extra, not 10–25% | non-Lloyd / biased codebook, H ≤ 0.9 uniform |
| NS-inter-expert-redundancy | pairwise cos ~1e-4 / Q80 0.004 | mean pairwise **≥ 0.10** |
| NS-cross-expert-and-cross-layer-tying | shared template ≈ orthogonal null | row-norm off-diag cos **≥ 0.10** |
| NS-expert-merging | best survivor rel-err ~0.89–1.0 | held-out reconstr **≤ 0.5** |
| NS-uniform-subbit-allocation | uniform + “treated” both collapsed | no organ split (gate≈down) |
| NS-ternary-factorization | ternary **of W** loses to VQ at matched rate (gpt-oss) | new parent where ternary **beats** VQ on real forward |
| NS-calibration-88-tokens | 88-tok route map dead | **≥ 1000** tokens, disjoint holdout |
| Qwen3.8 proxy trap | gaussian/PQ-proxy sub-bit “wins” all collapsed | **real** forward / Doctor only |

NS-ternary-factorization does **not** kill QMoE (entropy of already-ternary, not a competing codec) or LittleBit (binary **factors**, source changed).

---

## 1. Published SOTA — what they actually reach

Quality column = paper metric. **Doctor = UNKNOWN** everywhere.

| Method | Year | On what | Nominal | **Complete BPW** | Quality (paper) | Native path | Sub-1 stored? |
|---|---|---|---|---|---|---|---|
| **QTIP** HYB L=16 V=2 | 2024 | Llama-2-7/13/70, Llama-3.1-405B | 2 / 3 / 4 | **≈2.0–2.05** (2 KiB LUT). Embed excluded | L2-70B 2b Wiki2 **3.70** vs 3.12; L2-7B **5.86** vs 5.12. 188 tok/s 7B | bitshift trellis, 2–4 ALU/weight, **>80% BW** | **No** |
| **AQLM** 1×16 / 2×8 | 2024 | Llama-2-7/13/70, Mixtral-8×7B | 1.97–3.04 “avg bits” | **≈2.00–2.3** on linears (Eq. 10). Embed excluded. 1×16 codebook **does not fit L1** | L2-70B 2.07b Wiki2 **3.94**; Mixtral 1.98b **4.61** vs 3.46 | CUDA 2×8 up to **3×**; 1×16 **1.3×** | **No**. Pareto ~**2.5** (their claim) |
| **QuIP#** E8P | 2024 | Llama-2-7/70, Mixtral | 2 / 3 / 4 | **~2.0 + 0.01 + signs**. Independent re-measure **~2.4** (Bielik) | L2-70B 2b Wiki2 **4.16**; Mixtral 2.01b **4.75** | E8P 1 KiB L1, 176 tok/s 7B | **No** |
| **VPTQ** v8-k65536+res | 2024 | Llama-3.1-70B | 2.0 (+1 res = 3) | **~3.58** (5.0 GB). Codebook tax | Matches QTIP **only after** the extra 1.6 BPW | residual VQ decode | **No** — do not emit as 2-bit |
| **GPTVQ** 2D | 2024 | Llama-1 | 2+Y | UNKNOWN complete (VQ codebook) | Worse than QTIP at 2b | VQ | **No**. Collapses at 0.7–0.9 (BTC table) |
| **BitNet b1.58** | 2024 | 700M–70B **trained from scratch** | 1.58 | **~3.6 complete @ 3B** (2.22/7.89 GB). Asymptotes toward ~2 as embed share → 0 | Matches FP16 **same size/tokens from 3B**. Not a PTQ of an existing patient | BitLinear INT add, bitnet.cpp | **No**. QAT-from-scratch. Transfer to Odyssey PTQ = **UNKNOWN / likely dead** |
| **BiLLM** | 2024 | Llama-1/2/3 | ~1.09–1.11 | ~1.1 + group/residual. Embed excluded | L2-7B PPL **32**; L3-8B **28–56** | double-binary + scales | Borderline. Quality not Doctor-viable |
| **ARB-LLM** | 2025 | Llama-1/2/3 | 1.11 | same class as BiLLM | L2-7B PPL **16.4** | refined binary | **No** |
| **PB-LLM** | 2023 | Llama | 1.7 (10% 8-bit + 90% bin) | **>1.7** | L1-7B PPL **102** | mixed binary/hi | **No** |
| **STBLLM** 4:8 / 6:8 | 2025 ICLR | Llama-1/2/3, OPT, Mistral | **0.55 / 0.70 / 0.80** | Mask/group bits under-counted vs BTC critique. Embed excluded. Honest **~0.7–1.2** UNKNOWN | L1-7B 0.55: PPL **31.72** (FP16 5.68). L1-65B 0.55: **11.07** vs 3.53. **L3-8B 0.55: 254 collapse**. Zeroshot L1-30B 0.55: 51.8 vs 65.4 | N:M sparse binary. 2:4 HW exists; 4:8 software | **Claimed yes, quality no** at 7B. Larger models less dead. **Do not copy 0.55 as complete** |
| **BTC-LLM** | 2026 | Llama-1/2/3, Qwen2.5/3, FBI-LLM | 1.11 / 0.9 / 0.8 / 0.7 | **MEASURED GB:** 0.84 / 0.74 / **0.65** on L2-7B → complete **1.00 / 0.88 / 0.77**. Codebook 9.2%→1.2% | L2-7B 0.8: PPL **6.60** (FP16 5.47). 0.7: **11.02**. L2-13B 0.8: **5.83**. L3-8B 0.8: **9.49**. Qwen3-8B 0.8: **13.12** vs 9.72. FBI-1.3B 0.50: PPL 20.9, acc 39.6 vs 43.5 | **LUT-GEMM**, no dequant, **1.6× vs FP16**. Transform fused | **Yes, best PTQ quality at 0.8 complete-ish** |
| **QMoE** | 2023 | Switch-c2048 **1.6T**, base-128, large-128 | ternary then dict | **0.807 full ckpt**. MoE-only 20.07×. Dict ~20% from Shannon (p0=0.885 → 25.4× limit, they hit 20.1×) | C4 MLM loss 1.73 → ternary **~1.99** (+6.7% rel) on c2048. Not decoder-LM PPL | fused decompress+matvec, **<5% e2e** vs idealized BF16 | **Yes. Gold-standard complete sub-1.** Arch = Switch MLM, **not** Odyssey decoder MoE |
| **LittleBit** (QAT) | 2025 | Llama-2-7/13/70, Phi-4, Qwen | 1.0 … **0.1** “BPW” | **7B “0.1” = 0.63 GB → 0.75 complete**. **13B 0.1 = 0.84 GB → 0.52**. **70B 0.1 = 1.98 GB → 0.23**. Formula counts factors+scales; GB includes embed | L2-7B 0.55: PPL **10.47** vs STBLLM 30.7. 0.1: **15.92**. Zeroshot L2-7B 0.55: 47.3% vs FP16 63 | two binary GEMMs + 3-axis scales. Custom 1-bit GEMV, **5.5× @ sub-0.5**. **Does not expand W** if fused | **Yes if QAT allowed.** 7B “0.1” is **not** 0.1 complete |
| **NanoQuant** (PTQ) | 2026 | Llama-2-70B | sub-1 via rank | **5.35 / 138 GB → 0.62 complete** | “SOTA PTQ sub-1”; **PPL table not independently re-read here → quality UNKNOWN** | binary GEMV/GEMM of factors | **Yes, PTQ.** Prefer over LittleBit for Odyssey (no QAT) |
| **GGUF IQ1_S / IQ1_M** | 2024 | any llama.cpp | 1.56 / 1.75 | stated BPW includes imatrix on those tensors | “smallest possible; substantial quality loss”. Whole-file UNKNOWN | llama.cpp IQ kernels | Borderline. Not sub-1 complete |
| **Hawking GLM** (packet) | — | GLM, **real acts** | — | **0.167** (packet; receipt not re-opened this lane) | **0.755 cos**. Only real sub-⅕ | UNKNOWN this lane | **Yes, activation-aware.** Do not rediscover on a gaussian proxy |
| **Hawking Q80 mixed** | — | Qwen3-80? experts | mixed | **1.43 complete @ 8-bit non-expert** (0.970×1.230 + 0.030×8) | organ-cos screen only; **not packed, not generated, no kernel** | none yet | **No.** Floor so far on that body |
| **Hawking Qwen3.8 proxy** | — | Qwen3.8 | 0.5–0.8 | whatever was counted | **all collapsed**, output-div ~0.69 | n/a | **Artifact.** Proxy trap |

**Information floor (order of magnitude, not a number to hit):**

- i.i.d. Gaussian 2-bit: SQ MSE 0.118, QuIP# 8D 0.089, QTIP 256D 0.069, Shannon DR **0.063** (QTIP T1). **2-bit PTQ is near the Gaussian D(R).**
- Mid/late qwen3-235b layers: non-Gaussianity **0.012–0.018 bits**; NS-post-hoc says post-hoc coding of frozen W is **nearly exhausted**, layer 0 excepted.
- Ternary i.i.d. with p0=0.885: H ≈ 0.63 bits; QMoE realized **0.80**.
- Equiprobable ternary: **1.585**. BitNet lives here **plus** scales/embed.
- Odyssey MoE experts measured **near-orthogonal** → no free sharing. Sub-1 from **structure across experts is DEAD** until cosine ≥ 0.10.

---

## 2. How candgen + cost-model + transfer should search

Deterministic. No per-candidate LLM. Rank `info_gain / cost`. Cost-model is **insufficient data** today (`ODYSSEY_COST_MODEL.json`) — use **ordinal costs**, not fitted seconds.

### 2.1 Candidate-family design (add these; do not expand the 2/3/4 mlx grid first)

| Family id | Mechanism | `complete_bpw` identity | Search dims | Conventionality | Native? |
|---|---|---|---|---|---|
| `F-active-sub1-gather` | Store experts at 2–3 SOTA (QTIP/AQLM/q2-g128) + **gather only top-k** | `stored = 0.95 e + 0.05 ne`; `active = (k/E)·e + ne` | bits∈{2,3}, g∈{32,64,128}, router∈{4,8} | CONVENTIONAL (NX) | mlx q2/q3 **yes**; gather kernel **Hawking** |
| `F-btc-bin-pq` | ARB/activation-aware ±1 then **binary codebook** (BTC) | `(v c + ⌈log2 c⌉ mn/v + 16 n scales)/ (n m)` + embed | v∈{12,16,20}, c such that `⌈log2 c⌉/v ∈ {0.7,0.8,0.9}` | STRUCTURAL | LUT-GEMM **new** |
| `F-qmoe-tern-dict` | Activation-aware **ternary**, then **static dict** iff measured p0 high | payload Ĥ + 2^16×64b dict / N + `{wmin,wmax}` + dense organs | only if **p0 ≥ 0.80** on real-act ternary | STRUCTURAL | QMoE-style warp decode **new** |
| `F-bin-factor-ptq` | NanoQuant `W≈diag(s1) U± V±ᵀ diag(s2)` | `[r(n+m)+16(n+m)]/(n m)` + residual-r if used + embed | r so complete∈{0.3,0.55,0.8,1.0} | STRUCTURAL | fused UV GEMM **new**. **Kill if expands to dense W** |
| `F-base-corr-cond` | q1/tern/binary base + **organ/route-conditioned** residual (Q80 rice+q1) | base + `budget·b_corr` + topology metadata | base∈{1,2}, budget∈{0.02,0.05}, topo∈{row,block,expert} | STRUCTURAL | correction plane **non-native** |
| `F-protect-ne` | Never sub-1 the 3–5% non-expert | Q80: `0.970 e + 0.030 ne` | ne∈{4,6,8}, e from other families | CONVENTIONAL | yes |
| `F-matryoshka-t0` | T0 executable sub-1; T1 repair | count **stored tiers + active hit rate** | {t0, t0t1} | STRUCTURAL | tiers **non-native** |

**Do not emit:** raw PQ/VQ @ 1 bit; entropy-of-Lloyd; shared expert template; expert-merge; uniform sub-bit; STBLLM N:M **without** mask/index bytes; VPTQ 2^16 FP16 codebooks; BitNet-from-scratch as a PTQ candidate; QTIP/QuIP# **as a sub-1 mechanism** (use only as 2-bit quality ceiling).

### 2.2 Search order (cheapest discriminator first)

| Step | Cost (ordinal) | Discriminator | Kill / keep |
|---|---|---|---|
| **S0 Accounting screen** | 0 fwd | Plug census into the formula. Reject if `complete_bpw` cannot hit target even at infinite quality | VPTQ-tax, 2:4-without-mask, “0.1” that is 0.75 after embed |
| **S1 Structure census** | 0–1 capture | pairwise expert cos; Van Loan; **p0 after act-aware ternary**; organ sensitivity; layer-0 vs mid | cos < 0.10 → **no share/delta**. p0 < 0.80 → **no QMoE-dict**. Flat Van Loan @ depth → no Kronecker (layer 0 excepted) |
| **S2 Single-organ real-act** | 1 organ, real X | cosine / rel-err on **gate first** (`dominant_failure_organ`) | proxy/gaussian ranking **forbidden**. If gate dies, localize — do not globally retreat |
| **S3 Fast-Doctor mixed vs uniform** | 1 battery | `mixed-q2q3` vs `q2` vs `q2+corr` at **matched complete** | NS-uniform: if mixed doesn’t beat uniform, sensitivity isn’t a lever **on this patient** |
| **S4 Active vs stored split** | census algebra | compute `active_bpw` at k/E | If `active < 1` at stored 2-bit, **that is the first NX**. Do not spend novelty budget on stored-sub-1 until this is Doctor-scored |
| **S5 Native-path** | kernel / mlx | mlx 2/3/4 grouped first (`require_native` ladder) | 1-bit / LUT / UV / dict pay `non_native_penalty`. Don’t search them until 2-bit+corr is Doctor-dead **or** the target is stored-sub-1 |
| **S6 Full Doctor + decode cost/token** | full | held-out capability + bytes/token + FLOPs/token | Gibberish still valuable if it **names the broken organ** |
| **S7 Transfer** | 0 | write TRANSFER cell | MoE-gather = universal. Uniform-route = measure (O005/O006 yes; do not assume O003). Sharing kill does **not** transfer to a parent with cos≥0.10. Layer-0 codec does **not** transfer to depth |

**Cost-model features to log (even before a fit exists):** `source_passes`, `non_native_penalty`, `stored_GiB`, `moe_active_bytes`, `doctor_fast_count`, `cheap_kills`. Rank families: `F-protect-ne` / `F-active-sub1-gather` (prior 0) → `F-base-corr-cond` → `F-btc-bin-pq` → `F-bin-factor-ptq` → `F-qmoe-tern-dict` (gated) → `F-matryoshka-t0`.

**Transfer rules (do not rediscover):**

- O005 census: experts **28.99B / 30.53B = 94.9%**, active **3.35B (11%)**, 8/128. `active_expert_bpw = stored_e × 0.0625`.
- O003: 6/64 + **2 shared always-on**. Shared is **not** in the 6/64 fraction — count it in **active** like attn.
- Cold-expert compress: **N/A** on O005/O003/O006 (0 cold).
- Q80 identity is the default complete-body mixer until a patient census replaces the 0.97/0.03 split.

---

## 3. Structured proposals

### P1 — Active-sub-1 gather @ 2-bit SOTA
**mechanism:** Store experts at QTIP/AQLM/q2-g128-class (~2 complete on linears); execute only top-k. Stored may stay >1; **active <1** is the NX.

**complete_byte_accounting:**  
`stored = Σ_e (payload_e + scales_e) + attn + router + embed + ln + lm_head + container`  
O005: `complete_stored ≈ 0.949·e + 0.030·attn + 0.010·embed + 0.010·lm_head + 0.0004·router`  
`complete_active ≈ 0.949·(8/128)·e + non-expert`  
q2-g128 experts (nominal 2.25 if 32-bit scale/group): **stored ≈ 2.3**, **active ≈ 0.949·2.25·0.0625 + ~0.25 ≈ 0.13+0.25 ≈ 0.38** (non-expert if 4–8 bit). Numbers are **DERIVED from census + codec formula**, not measured packed.

**stored_bpw / active_bpw:** stored **1.8–2.6** / active **0.3–0.6** (range; exact UNKNOWN until packed).

**expected_reachable_bpw:** active **<1 now** on O005/O006/O003. Stored sub-1: **not this family**.

**quality_risk:** 2-bit SOTA is the published quality ceiling (QTIP L2-70B +0.58 Wiki2). Limiters: **router** (0.03 GB, gates all), **gate_proj** (Q80/F1), decode **kernel-bound** so gather must beat mlx dense-MoE (A3B 29.3 TPS caveat).

**cheapest_falsifier:** one fast-Doctor at `q2-g128-experts` vs incumbent `q3-g32-experts` at counted complete + measured active-bytes/token. If Doctor ≤ −2 and gather TPS does not move, family is NX-dead not codec-dead.

**execution_path:** native mlx affine-grouped **yes**. Gather = Hawking kernel. Reconstruction/token = <REDACTED> if grouped GEMV stays packed. FLOPs = dense-active.

**applicability:** MoE (O003, O005, O006). Dense/hybrid: N/A (no k/E).

**confidence:** **high** that active<1 is algebraically reachable; **medium** Doctor holds at q2 (QTIP/AQLM say yes on Llama, Odyssey patients UNKNOWN).

**transfer:** gather = MoE-universal (TRANSFER `R-sparse-active-expert-gather`). q2 quality = retune per patient. Cite QTIP/AQLM/QuIP# as the 2-bit ceiling, not as sub-1.

---

### P2 — BTC binary-codebook (best published PTQ sub-1 quality)
**mechanism:** Activation-aware binarize (ARB/BiLLM-class) → cluster ±1 v-vectors into a **binary** codebook → store indices + tiny ±1 codebook. Learnable transform **fused into W** (0 extra store). BTC-LLM 2026.

**complete_byte_accounting:**  
`bits = v·c + ⌈log2 c⌉·N/v + 16·n_rows (scale) + 16·n_rows (bias) + embed + ln + container`  
L2-7B measured: **0.65 GB @ 0.7-bit (codebook 8.43 MB = 1.2%)** → **0.77 complete**. 0.8-bit **0.88 complete**.

**stored_bpw / active_bpw:** stored **0.75–1.0** on 7B-class; **lower on 70B** (amortize codebook) UNKNOWN exact. Active = stored on dense; on MoE `× k/E` extra.

**expected_reachable_bpw:** **0.75–0.90 complete** with PPL-class quality; **0.70** is a cliff (L2-7B 6.60 → 11.02). Sub-0.2: **no**.

**quality_risk:** L3-8B and Qwen3 degrade more than Llama-2 (0.8: 9.49 / 13.12). Limiter = **pattern diversity** (need v≥16–20; v=8 dies). Hawking NS-raw-PQ killed **FP** PQ of W, **not** PQ of already-binary patterns — still confirm on real acts.

**cheapest_falsifier:** one organ (gate), real X, v=16 c≈2^13, report complete_bpw + cosine. If cosine < GLM-proxy-fail band or codebook > 5% of body, kill.

**execution_path:** **LUT-GEMM** (act·codewords once, index-accumulate). No expand-to-dense. 1.6× vs FP16 claimed. **Not** in mlx affine-grouped → non-native penalty until Hawking LUT kernel.

**applicability:** all classes. Best first on **dense O004** and **MoE experts** separately (don’t share codebook across orthogonal experts — NS-inter-expert).

**confidence:** **medium-high** accounting (they published GB + codebook %). **medium** Odyssey transfer (PPL ≠ Doctor; Qwen3 already softer).

**transfer:** codebook geometry (v, c) retune; fused transform may transfer. Do **not** share one codebook across experts unless S1 cosine ≥ 0.10.

---

### P3 — QMoE ternary + dictionary (only if p0 is high)
**mechanism:** Activation-aware ternary `{wmin, 0, wmax}` per row; if **measured p0 ≥ 0.80**, static 2^16 dictionary of high-prob ternary-pair sequences (QMoE §4). **Not** Huffman (sequential, GPU-hostile). **Not** entropy of Lloyd PQ (NS-killed).

**complete_byte_accounting:**  
`bits = |packed UINT16 stream| + 2^16 × 64 b dict + row_off + 32 n_rows (minmax) + uncompressed non-expert`  
QMoE c2048: **0.807 complete** at p0=0.886. Shannon at that p0 ≈ 0.63; they left ~20% on the table for decode.

**stored_bpw / active_bpw:** stored **0.7–1.0** **if** p0 holds; else **≥ 1.58 + scales**. Active = stored × k/E on MoE.

**expected_reachable_bpw:** **UNKNOWN on Odyssey**. Switch experts are huge + MLM-robust. Decoder MoE ternary p0 is **UNKNOWN** — this family is **gated on S1**.

**quality_risk:** QMoE quality is MLM loss +6.7% on a **1.6T** model. 7–30B decoder may collapse (BiLLM/STBLLM). Limiter = **p0** and **Doctor on gate**.

**cheapest_falsifier:** one layer, real-act GPTQ-ternary, report **p0 histogram**. If p0 < 0.80, **family dead** (do not build the dict). Cost: 1 organ, no decode kernel.

**execution_path:** QMoE warp-per-row, 16-bit symbols → 14 ternary pairs, fused matvec, <5% vs BF16. Native-credible. mlx: **no**.

**applicability:** MoE experts first (QMoE’s own rationale: only experts need to be sub-1). Dense: only if p0 high.

**confidence:** **high** that the accounting works; **low** that Odyssey ternary is sparse enough. Cite QMoE; do not claim 0.8 until p0 is measured.

**transfer:** dict is static and was tuned for p0=0.885. Rebuild per measured p0. Do not transfer Switch robustness to Qwen/Kimi.

---

### P4 — Binary-factor PTQ (NanoQuant / LittleBit-without-QAT)
**mechanism:** Change the source: `W ≈ diag(s1) U_±1 V_±1ᵀ diag(s2)` (NanoQuant). Rank r sets complete BPW. Optional residual path (LittleBit) **counts twice**. Forward is two skinny binary GEMMs — **never materialize W**.

**complete_byte_accounting:**  
`bits = r(n+m) + 16(n+m) [+ residual clone] + embed + ln`  
LittleBit A.16 also `# Prior-art SOTA + sub-1 search (priorart-search lane)

Guardrails applied: complete bytes only; real activations only; Doctor is the quality authority (published numbers below are PPL/zero-shot, **not** Doctor); native path or honest reconstruct cost; stored ≠ active.

---

## 1. What published methods actually store

Nominal bits in papers are usually **payload of quantized linears only**. Complete BPW below is **derived from their own size formula or published GB**, divided by original param count, ×8. Embeddings/LN/lm_head called out when excluded.

| Method | Class | On what | Nominal | **Complete (honest)** | Quality (published, not Doctor) | Native? | Sub-1? |
|---|---|---|---|---|---|---|---|
| **QTIP** (Tseng+ NeurIPS 2024) | TCQ + RHT, computed/HYB codes | Llama 1/2/3, 7B–405B | 2 / 3 / 4 | ≈ nominal. HYB codebook 2 KiB; 1MAD/3INST codebook ≈ 0. Hadamard signs UNKNOWN (small). Embeddings excluded. | Llama2-70B 2b Wiki2 **3.70** vs 3.12; 7B **5.86** vs 5.12. Best 2-bit PTQ quality. | Yes. Bitshift trellis, 2–4 inst/weight, 188 tok/s 7B (RTX6000 Ada), ~80% peak BW. | **No. Floor ~2.** |
| **QuIP#** (Tseng+ ICML 2024) | RHT + E8P lattice VQ | Llama 2, Mixtral | 2 / 3 / 4 | E8P **1 KiB shared ≈ 0.01 BPW**. Signs: 16n bits w/ FT. Bielik-Q2-Sharp: **~2.4 eff. BPW**, 3.26 GB. | Llama2-70B 2b Wiki2 **4.16** vs 3.12; 7B ~6.2. | Yes. 176–186 tok/s 7B. | **No.** |
| **AQLM** (Egiazarian+ 2024) | Additive MCQ, learned codebooks | Llama 2 7/13/70B, Mixtral 8×7B | 1.97–3.04 “avg bits” | \(\bar b=(16gM2^B+d_{out}(d_{in}/g)B+16d_{out})/(d_{out}d_{in})\). Llama2-70B gate 2×8 g=8 → **2.002**. **Excludes** embed/LN/lm_head. | Llama2-70B 2.07: Wiki2 **3.94** vs 3.12. Mixtral 1.98: **4.61** vs 3.46. 2×8 CUDA up to 3×. | Yes (1×16, 2×8 kernels). 1×16 codebook **does not fit L1**. | **No. Pareto they claim is ~2.5, not <2.** |
| **VPTQ** (Liu+ EMNLP 2024) | Residual VQ, 8-d, 2^16 centroids | Llama 3.1 70B etc. | 2.0 (+ residual → 3) | Bielik: **5.0 GB ≈ 3.58 complete BPW**. Codebook 2^16×8×FP16 + residual 256 + perm + norms. **Fake-density exemplar.** | Quality at 2.0 nominal is really ~3.6-bit quality. | LUT GEMM. Slow if codebook uncached. | **No. Do not emit this family below 3 complete.** |
| **GGUF IQ1_S / IQ1_M** | Importance-matrix 1-bit + superblock scale | llama.cpp zoo | 1.56 / 1.75 | Stated BPW **includes** scale + imatrix on those tensors. Whole-file higher (embed often Q6_K). | “Usable for testing.” PPL heavily degraded. UNKNOWN on Odyssey patients. | Yes (llama.cpp). | **No stored sub-1.** Closest *deployed* 1.5-class. |
| **BitNet b1.58** (Ma+ 2024) | QAT ternary {-1,0,1}, absmean | 700M–70B trained from scratch | log2(3)≈1.58 | **3B: 2.22 vs 7.89 GB = 3.56 complete BPW.** 700M: 0.80 vs 2.08 = **6.15 complete** (embed stays FP16). 70B closer to ~2–3 (UNKNOWN exact GB). | Matches FP16 **same size + same tokens** from 3B. Not a PTQ of an existing FP16 body. | Yes. INT add, bitnet.cpp. | **No. And not PTQ.** |
| **BiLLM / ARB-LLM / PB-LLM** | 1-bit PTQ ± residual / partial 8-bit | Llama 1/2/3, OPT | 1.09–1.11 / 1.7 | Scales + residual bits + group id. Embed excluded. Complete **> claimed**. PB-LLM 10% 8-bit + 90% binary ≈ 1.7 by design. | Llama2-7B BiLLM ~32 PPL; ARB-LLM 16.4; PB-LLM 69. Generation-poor at 7B. | Binary GEMM + extra residual pass. | **No honest sub-1.** |
| **QMoE** (Frantar+ 2023) | GPTQ ternary + dictionary (fixed-to-variable) | SwitchTransformer-c2048 **1.6T**, also base-128 / large-128 | “<1” | **Full ckpt 3142 → 158.6 GB = 0.807 complete BPW.** MoE-only 20.07×. Dict 2^16 shared. p0=0.886; 20.07× vs Shannon 25.40× (~20% leave-on-table for decode speed). | C4 MLM loss 1.73 → ternary **~1.99** (+6.7% rel) on c2048. Not a causal decoder-only. | Yes. Fused decompress+matvec. Isolated kernel **faster** than BF16; e2e **<5%** vs idealized BF16. | **Yes. Best-accounted published stored sub-1.** |
| **STBLLM** (Dong+ ICLR 2025) | N:M prune + binary + residual salient + trisection | Llama 1/2/3, OPT, Mistral | 0.55 / 0.70 / 0.80 | \(N_{param}=2r_{sal}+(1-r_{sal})\), \(N_{store}=2+1/b_{size}\), then ×N/M. **BTC-LLM: 2:4 mask is 1.25 BPW if counted.** Embed excluded. Complete **UNKNOWN, > claimed.** | Llama1-7B 0.55: PPL **31.7** vs 5.68. Llama1-65B 0.55: **11.07** vs 3.53. **Llama3-8B 0.55: 254 (collapse).** Zero-shot 30B 0.55: 51.8% vs 65.4%. | N:M sparse binary. 2:4 HW exists; 4:8 is software. | **Claimed yes. Honest complete dubious. Quality not Doctor-viable at 7B.** |
| **BTC-LLM** (Gu+ 2025/26) | ARB ternary/binary + **binary** codebook + fused transform | Llama 1/2/3, Qwen 2.5/3, FBI-LLM | 1.11 / 0.9 / 0.8 / 0.7 | \(vc+\lceil\log_2 c\rceil\cdot mn/v\). T fused → **0 store**. Llama2-7B **GB-implied complete: 0.9→1.00, 0.8→0.88, 0.7→0.77** (0.65 GB, codebook 8.43 MB = 1.2%). | Llama2-7B 0.8: PPL **6.60** vs 5.47; 0.7: **11.02**. Llama2-13B 0.8: **5.83** vs 4.88. Llama3-8B 0.8: **9.49** vs 6.14. Qwen3-8B 0.8: **13.12** vs 9.72. Zero-shot Llama2-13B 0.8: 61.9% vs 65.0%. | LUT-GEMM, no dequant. 1.6× vs FP16 (H800 MLP, prelim). | **Yes, ~0.77–0.88 complete on 7B linears+codebook.** Best PTQ sub-1 quality in the open literature. |
| **LittleBit** (Lee+ NeurIPS 2025) | QAT: binary low-rank \(W\approx \mathrm{diag}(h)U_{\pm}\mathrm{diag}(\ell)V_{\pm}^\top\mathrm{diag}(g)\) + residual path | Llama/2/3, OPT, Phi-4, Qwen 1.3B–32B | 1.0 … 0.1 | Formula counts **both** paths + FP16 scales. **GB-implied complete: 7B “0.1” 0.63/13.49×16 = 0.75; 13B 0.84/26.06×16 = 0.52; 70B 1.98/138×16 = 0.23.** “0.1 BPW” is **linear-layer factor rate, not complete model.** | Llama2-7B “0.1”: PPL **15.92**; “0.55”: **10.47**. Zero-shot 0.55: 47.3%; 0.3: 45.2% vs FP16 ~63%. Beats STBLLM 0.7. | Two small binary GEMMs + scales. Custom 1-bit GEMV. Up to 5.5× vs FP16 at sub-0.5 (A100, 70B MLP). **Expands in compute, not in stored dense W** if fused. | **Yes at 70B (~0.23 complete). At 7B the “0.1” is ~0.75 complete.** QAT, not PTQ. |
| **NanoQuant** (Chong+ 2026) | PTQ binary low-rank \(W\approx\mathrm{diag}(s_1)U_{\pm}V_{\pm}^\top\mathrm{diag}(s_2)\), ADMM | Llama2-70B, others | sub-1 | \(BPW=[r(n+m)+16(n+m)]/(mn)\). 70B 138→**5.35 GB = 0.62 complete**. | Claims SOTA PTQ sub-1. Exact Wiki2/zero-shot **UNKNOWN** (table not fully recovered). 13 h on 1×H100. | Binary GEMV/GEMM. 70B on 8 GB GPU, up to 20.1 tok/s claimed. | **Yes as PTQ analog of LittleBit.** Quality UNKNOWN until measured. |

**Hawking (do not rediscover)**

| Fact | Number | Note |
|---|---|---|
| GLM act-aware sub-1 | **0.755 cos @ 0.167 BPW** | Packet. REAL activations. Only real sub-1/5 precedent. I did not re-open the receipt this lane. |
| Qwen3.8 gaussian-proxy sub-bit | **all collapsed 0.5–0.8 BPW** | Proxy trap. Output-div ~0.69. |
| Q80 mixed | **1.43 complete** @ 8-bit non-expert | Identity: `0.97032·expert + 0.02968·nonexpert`. Screen only; not packed/Doctor/native. |
| O005 (Qwen3-30B-A3B) | expert **28.99B / 30.53B = 94.95%**; active **3.35B (11.0%)**; 8/128 | 0 cold, entropy ~6.1/7. Stored sub-1 ≠ active sub-1. |
| O003 (Kimi-VL-A3B) | expert 14.39B/16.41B; +2 shared; 6/64 | Shared expert is **not** the sparse path. |
| Cross-expert | pairwise cos **~0.004** (Q80 L10); **1e-4** (gpt-oss) | NS-inter-expert-redundancy, NS-cross-expert-tying **DEAD** unless new parent cos ≥ 0.10. |
| Raw PQ/VQ @ ~1 bit | 6/6 collapse qwen3-235b | NS-raw-weight-pq-vq-at-one-bit. Kills **raw-W** VQ, not binary-pattern VQ. |
| Entropy of Lloyd PQ idx | 0.0–0.7% | NS-entropy-coded-pq-indices. Do not search. |
| Uniform sub-bit | collapsed | NS-uniform-subbit-allocation. Organ inversion: gate/up sensitive, down tolerant. |
| Ternary factorization vs VQ | dominated @ matched rate | NS-ternary-factorization on gpt-oss. Different object than binary **factors as the stored body**. |

**Information floor (not guessed)**

| Source | Floor | Implication |
|---|---|---|
| i.i.d. Gaussian 2-bit DR | MSE 0.063 (QTIP Table 1) | QTIP 256-d TCQ 0.069 — near rate-distortion **at 2 bits**. No published TCQ below 2. |
| Ternary i.i.d. p0=0.885 | Shannon **25.40×** vs QMoE **20.07×** | Entropy coding of **quantized** ternary is live; entropy of Lloyd PQ is dead. |
| Ternary equiprobable | log2(3)≈1.58 | BitNet nominal. If p0≈1/3, you cannot beat ~1.58 without structure or a changed source. |
| Mid/late layer non-Gaussianity | 0.012–0.018 bits (qwen3-235b) | Post-hoc coding of frozen W is nearly exhausted except layer 0. **Change the source** (QAT / factor / prune / share) or stop. |

---

## 2. How candgen + cost-model + transfer should search

`candidate_families.json` today tops out at affine q1–q4 + correction + tiers. That grid **cannot express** the only published stored-sub-1 survivors (QMoE dict, BTC binary codebook, binary-factor). Expand families; do not grow the q1-g32 cube.

### Candidate-family design (add these; do not rediscover killed ones)

| Family id | What it is | Complete-BPW identity | Prior | Skip if |
|---|---|---|---|---|
| `active_sub1_gather` | Stored 2–3 BPW SOTA (QTIP/AQLM/grouped-q2) on experts + native gather | `active = stored_expert·(k/E) + attn + router + embed` | **0 (first)** | not MoE |
| `qmoe_ternary_dict` | Act-aware ternary + **shared** 2^16 dictionary | `payload_entropy + 2^16·entry + row_minmax + row_off` | 2 | measured p0 < 0.80 |
| `btc_binary_codebook` | Binary pattern VQ of ±1, fused T | `vc + ⌈log2 c⌉·mn/v + scales` | 2 | codebook > 5% of body |
| `binary_factor_ptq` | NanoQuant/LittleBit geometry, **PTQ first** | `[r(n+m)+16(n+m)(+residual)]/(nm)` + embed | 3 | no fused UV path (dense expand = fake) |
| `base_corr_organ` | already in file; **force organ split** | Q80 identity + correction_bytes | 1 | uniform map |
| `matryoshka_t0` | already in file | count **stored** unused tiers | 4 | T1 never hits |
| *(killed)* raw PQ/VQ @1, Lloyd entropy, expert template, expert merge, N:M without mask bits, VPTQ-2^16-FP16 | — | — | **never emit** | always |

### Search order (cheapest discriminator first)

```
0. ACCOUNTING SCREEN          0 forwards
   reject if complete_bpw(formula, census) > target even at r→0 / p0→1
   reject VPTQ-style FP16 codebooks; reject 2:4-as-0.5; reject embed-blind "0.1"

1. CENSUS DISCRIMINATORS      0–1 capture, no Doctor
   pairwise expert cos  → kill share/delta if < 0.10          [NS]
   p0 after act-aware ternary on 1 organ → gate dict family
   Van Loan layer0 vs mid  → per-layer codec (R-layer0)
   organ param fractions    → lock Q80-style identity
   k/E, cold count          → active vs stored split

2. SINGLE-ORGAN REAL-ACT      1 short real forward (NOT gaussian)
   gate first (dominant_failure_organ)
   cosine vs FP16 on held-out routed tokens
   if organ cos < Hawking GLM-class (~0.75) at that complete_bpw → localize, don't global-retreat

3. FAST-DOCTOR 2-POINT        cheapest pair that can kill the family
   (see each proposal)
   mixed vs uniform; T0 vs T0+T1; dict vs naked ternary; factor-r vs grouped-q2 @ matched complete

4. NATIVE / RECON COST        kernel or explicit bytes/token + FLOP
   mlx grouped {2,3,4} is cheap; 1-bit / LUT / UV / dict = Hawking kernel
   cost-model: non_native_penalty until a kernel exists
   if reconstruct-to-dense before MAC → account those bytes; usually a kill

5. FULL DOCTOR + transfer     only survivors
   never transfer a kill across arch unless premise still holds
   MoE gather is MoE-universal; uniform-routing is patient-specific; layer0 is layer-specific
```

### Cost-model features to add (ODYSSEY_COST_MODEL is currently `insufficient data`)

| Feature | Why | Unit |
|---|---|---|
| `complete_bpw_formula` | reject before compile | bits |
| `active_bytes_per_tok` | MoE NX | bytes |
| `codebook_frac` | VPTQ/AQLM tax | [0,1] |
| `p0_ternary` | QMoE feasibility | [0,1] |
| `expert_pairwise_cos` | share-family gate | [0,1] |
| `non_native_penalty` | mlx 2/3/4 vs new kernel | {0,1}×passes |
| `source_passes` | already in families | — |
| `proxy=gaussian` | **hard reject as a rank key** | flag |

Rank: `info_gain / (source_passes + 2·non_native + 0.3·stored_GiB)`. Conventional 2-bit gather **before** any stored-sub-1.

### Transfer rules (do not re-test)

| Rule | Transfer | Do not transfer |
|---|---|---|
| Sparse gather (8/128, 6/64) | all MoE | dense/hybrid |
| No cold-expert compress | O005, O006 measured | any MoE until **its** entropy/cold counted (≥1000 tok) |
| Organ inversion | prior F0/F1 | if gate≈down within 10% |
| Raw PQ @ 1 bit dead | decoder-only raw W | BTC (binary-domain VQ), QMoE (entropy of ternary), factors (changed source) |
| Expert orthogonality | Qwen-class MoE | parent with cos ≥ 0.10 |
| Proxy trap | all patients | — |
| QTIP/AQLM 2-bit quality | dense + MoE experts as **quality ceiling at 2**, not a sub-1 mechanism | “just turn k down” |

---

## 3. Proposals (required schema)

### P1 — Active-sub-1 via gather + 2-bit SOTA experts
**mechanism:** Store experts at QTIP/QuIP#/AQLM/grouped-q2 (~2 complete BPW). Move only k/E per token. Stored stays >1; **active <1**. Execution lever, not a compression statistic (bible §29; R-sparse-active-expert-gather).

**complete_byte_accounting:**
| byte | O005 (30.53B) | formula |
|---|---|---|
| expert payload | 28.99B × stored_bpw / 8 | grouped-q2 or QTIP 2 |
| expert scales | 16+16 bit / group | g∈{32,64,128} |
| attn+embed+lm_head+router | 1.54B × {4–8} / 8 | **do not** sub-1 these (Q80 3%) |
| codebook | QTIP HYB 2 KiB **or** AQLM per-layer (count it) | reject if >2% body |
| container/align | pack waste | count |
| **stored complete** | `0.9495·expert_bpw + 0.0505·nonexpert_bpw` | |
| **active complete** | `0.9495·expert_bpw·(8/128) + 0.0505·nonexpert_bpw` | |

**stored_bpw / active_bpw:** stored **~2.1–2.4** (if expert=2, nonexpert=8 → stored 2.30). active **~0.52** (expert 2×0.0625=0.125; nonexpert 8×0.0505=0.40). O003: 6/64=0.094 + 2 shared always-on — shared must sit in the active term.

**expected_reachable_bpw:** active **0.4–0.8** NOW on O005/O006 if q2 experts Doctor-survive. Stored sub-1: **not this proposal**.

**quality_risk + limiter:** Attn/embed at 8-bit already dominate **active** bytes (~0.4 of 0.52). Next limiter = expert q2 Doctor (gate). Router (12.6M) is cheap to keep 8-bit. Kernel-bound decode (A3B tps caveat): fewer bytes help only if gather is real.

**cheapest_falsifier:** One mlx `q2-g64-experts` fast-Doctor vs q3-g32-experts. If q2 `delta_hits ≤ -2`, **localize gate** — do not drop gather. Second shot: measure `active_bytes/tok` vs TPS; if TPS flat, kernel-bound → this family is an NX-memory win only.

**execution_path:** mlx affine-grouped q2 is **native**. QTIP/AQLM kernels are not in Hawking; do not claim their tok/s. Gather must exist or active number is fiction.

**applicability:** MoE (O003/O005/O006). Dense: N/A. Hybrid: only the MoE trunk.

**confidence:** **HIGH** that active<1 is arithmetically reachable. **MED** that q2 experts keep Doctor (QTIP 2b is the published ceiling, not a Hawking measurement).

**transfer:** Gather = MoE-universal. q2 quality = retune per patient. Uniform routing → do **not** add popularity bits.

---

### P2 — QMoE ternary + shared dictionary (conditional)
**mechanism:** Activation-aware ternary (row {wmin,0,wmax}) then QMoE-style **fixed-length codeword → variable ternary run**, one shared 2^16 dict. Sub-1 from **entropy of the quantized source**, not from a fake codebook.

**complete_byte_accounting:**
| byte | count |
|---|---|
| packed UINT16 stream | measured after encode (not 1.58) |
| dict 2^16 × ~8 B | **1 MiB shared** ≈ 0 on 30B |
| per-row minmax | 2×16 bit / row |
| row offsets | 32 bit / row |
| non-expert | 4–8 bit, as P1 |
| **reject** if you store a bitmask (that is +1 BPW and kills sub-1) |

**stored_bpw / active_bpw:** stored ≈ `H(ternary)/ln2 + overhead`. QMoE: **0.80** on experts at p0=0.886. Active = stored × k/E + nonexpert.

**expected_reachable_bpw:** **0.7–1.0 stored on experts IFF p0≥0.85**. If p0≈0.33 (BitNet-like), Shannon ≈1.58 — **family auto-kills at screen**. Hawking ternary-factorization kill is a **different** object (factors vs VQ of W); this is QMoE’s object.

**quality_risk + limiter:** QMoE quality is on Switch **MLM**, not Qwen causal. GPTQ-ternary on decoder-only is UNKNOWN. Limiter = p0 and gate Doctor. Natural sparsity may not appear if act-aware ternary is not wide-grid.

**cheapest_falsifier:** Ternarize **one** O005 gate layer on **real** routed activations. Measure p0 and organ cos. If p0<0.80 **or** cos≪0.75 → **kill family** (one organ, minutes). Do not train a dict first.

**execution_path:** QMoE CUDA dict-decode is real; Hawking Metal port is new. Decode is sequential-per-row, warp-wide. If no kernel: decomp-to-dense is a **fake win** — account BF16 expand/token.

**applicability:** Large MoE experts first (amortize dict; QMoE’s own rates rose with size). Dense 7B-class: embed fraction kills complete (BitNet lesson).

**confidence:** **HIGH** on accounting. **LOW** on quality transfer Switch→Qwen3. **MED** that p0 screen is the right cheap kill.

**transfer:** p0 is per-patient, per-organ. Do not copy 0.886.

---

### P3 — BTC binary codebook + fused transform
**mechanism:** ARB/act-aware binarize, then **binary-domain** product-quant (centroids stay ±1). Learnable T=D±P fused into W (0 store). LUT-GEMM: act·codeword once, index-accumulate. This is **not** NS-raw-weight-pq (source is binary, Hamming, codebook is bits not FP16).

**complete_byte_accounting:** \(vc + \lceil\log_2 c\rceil mn/v + \text{row scales}\). Published Llama2-7B: **0.77 / 0.88 / 1.00 complete** at claimed 0.7/0.8/0.9 (GB). Codebook 1.2–9.2%. T fused.

**stored_bpw / active_bpw:** stored **0.75–1.1** on linears (use GB formula, not the 0.7 label). Active = stored×k/E + nonexpert.

**expected_reachable_bpw:** **0.8–1.0 stored** on Qwen-class linears if transform+codebook transfer. Sub-0.2: **no** — their 0.7 already PPL 11 on 7B / 18.5 on Llama3-8B.

**quality_risk + limiter:** Best published PTQ sub-1 (Llama2-7B 0.8 PPL 6.60; Qwen3-8B 0.8 PPL 13.12). Limiter = vector length vs codebook size (they need v=16–20 to match 1.11-bit). Llama3/Qwen3 degrade more than Llama2. Doctor UNKNOWN. Block-wise transform learn is a calibration cost (they +20 min vs ARB on 7B).

**cheapest_falsifier:** Freeze T=I. Binary-PQ one gate, v=16, c such that \(\lceil\log_2 c\rceil/v \approx 0.8\), **real** acts. If organ cos < grouped-q2 at **matched complete** → transform is load-bearing; second shot is T. If still lose to q2 → family dead vs P1.

**execution_path:** LUT-GEMM is the native story (1.6× claimed). Without it, index→binary→scale is extra ALU; may be compute-bound (BTC’s own note). Hawking has no LUT-GEMM today → non_native.

**applicability:** All classes. Highest prior on dense+MoE experts. Do not use FP16 VQ codebooks (VPTQ tax).

**confidence:** **HIGH** this is the strongest published PTQ sub-1 **quality+accounting** pair. **MED** it transfers to O005 Doctor. **HIGH** it beats STBLLM/N:M.

**transfer:** Geometry (v, c) retune. Qwen3 numbers exist (0.8 lives, 0.7 hurts).

---

### P4 — Binary-factor PTQ (NanoQuant geometry; LittleBit as QAT ceiling)
**mechanism:** Store \(W \approx \mathrm{diag}(s_1)U_{\pm}V_{\pm}^\top\mathrm{diag}(s_2)\) (NanoQuant) or + residual path + latent scale (LittleBit). Sub-1 from **rank**, not bits-per-entry. **Generate in the MAC** (X s2 V then U s1), never materialize dense W.

**complete_byte_accounting:**
| | NanoQuant | LittleBit (both paths) |
|---|---|---|
| binary factors | r(n+m) bits | 2·r(n+m) bits |
| scales | 16(n+m) | 16(2n+2m+2r) |
| BPW | \([r(n+m)+16(n+m)]/(nm)\) | appendix Eq. 16–17 |
| 70B published GB | 5.35 → **0.62 complete** | 1.98 @ “0.1” → **0.23 complete** |
| 7B “0.1” | — | 0.63 GB → **0.75 complete** (embed) |

**stored_bpw / active_bpw:** stored = formula + embed/attn if unfactored. Active: factors of **selected** experts only (same k/E). KV can cache rank-r (LittleBit Table 4: ~d/r).

**expected_reachable_bpw:** PTQ **0.5–1.0** complete on 30B-class if NanoQuant-like ADMM holds. Sub-0.2 stored: only QAT LittleBit-class on **70B+** after embed amortization — **not NOW on 7B-class, not without QAT**. O005 30B sits between: UNKNOWN whether 0.2 complete is reachable without QAT.

**quality_risk + limiter:** LittleBit 0.1 PPL 15.9 / zero-shot ~43% — **not Doctor-grade**. Rank is the cliff (they: 0.3–0.55 “sweet”, 0.1 cliff). NS-ternary-factorization killed **ternary-of-W vs VQ**, not this. Limiter = fused UV kernel + rank needed for gate. QAT is out of Odyssey PTQ budget unless explicitly escalated.

**cheapest_falsifier:** SVD-then-sign one gate at r such that formula = 0.8 complete. Real-act cos. If cos ≪ 0.75 **and** ≪ BTC/q2 at same complete → PTQ factor is dead; QAT-only (LittleBit) is a different, expensive family.

**execution_path:** Native = two binary GEMMs + 3 vector scales. If you write `W_hat = U @ V.T` into RAM, **disqualify**. FLOP: O(r(n+m)) vs O(nm); win iff r ≪ min(n,m)/4 **and** kernel exists.

**applicability:** Large linears (experts, down_proj). Tiny attn heads: scale tax dominates (same as BitNet 700M).

**confidence:** **HIGH** on accounting and “don’t expand”. **MED** PTQ hits 0.6–0.8 complete with usable organ cos. **LOW** that 0.2 is NOW without QAT.

**transfer:** r is per-shape. GQA K/V need higher r (LittleBit A.2.3).

---

### P5 — Organ-conditioned base + correction (Q80 / BiLLM residual, not uniform)
**mechanism:** Extreme base on **tolerant** organs (down); protect **sensitive** (gate/up, router, attn). Sparse residual / rice+q1 outliers / selected hi-prec channels. Q80 already: binary gate, binary+2% rice-q1 up, r160-b3 down → expert 1.23, complete 1.43 @ 8-bit nonexpert.

**complete_byte_accounting:**  
`complete = Σ_organs (payload + scales + residual_stream + locators) / N + nonexpert`.  
Q80 identity is the template. Residual bytes **must** include index/Rice overhead (Q80 up: 8.24 bit/outlier).

**stored_bpw / active_bpw:** stored target **0.8–1.3** (stretch Q80 1.43 downward). Active: same organ mix × k/E on experts. A 0.3 BPW naked base + 0.2 BPW residual can beat 2 BPW naked **only if** residual is sparse **and** counted.

**expected_reachable_bpw:** **0.9–1.3 complete stored** on Qwen-MoE without new kernels (mlx q2 + small correction plane). Sub-0.2 stored: **no**. Sub-1 **active**: yes, stacked on P1.

**quality_risk + limiter:** Gate. NS-uniform-subbit is dead. NS-posthoc-scalar-gain: do not add a global scale on k-means. Correction **topology** (row / block / route-conditioned) is the lever, not +1 bit.

**cheapest_falsifier:** `q2-g32-experts` vs `mixed-q2q3-experts` (gate/up q3, down q2) vs `q2+correction` at **matched complete**. If mixed ≉ better than uniform-q2, organ inversion is **not** a bit lever on this patient (reopen_if in rulebase).

**execution_path:** mlx mixed via `quant_predicate` is native. Correction plane is **not** (`correction_budget` false in native_availability). Scattered correction without a kernel is a bandwidth anti-win.

**applicability:** All. Strongest on MoE (94% experts). Dense: still organ-split attn vs MLP.

**confidence:** **HIGH** this is the correct **next stored step after P1**. **MED** it crosses stored=1.0. **HIGH** it will beat uniform q1.

**transfer:** Organ map retune per patient (measure, don’t copy F1 “gate”).

---

### P6 — Matryoshka T0 executable body
**mechanism:** T0 = P3 or P4 or ternary-dict at sub-1, **runnable**. T1 = residual / selected channels, loaded on Doctor miss or hard organs. Stored = T0+T1+…; active = T0 + hit_rate·T1.

**complete_byte_accounting:** Every unused stored tier **counts in stored**. Active = T0 + Σ hit_i·|T_i|. If T1 never hits, T1 is dead weight — prune (family’s own cheapest_falsifier).

**stored_bpw / active_bpw:** stored **0.9–1.5**; active **0.5–0.9** if T1 hit_rate is low and T0 is already near-Doctor.

**expected_reachable_bpw:** Active sub-1 **plausible**. Stored sub-0.2 **no** (T0 alone isn’t 0.2-viable on these patients).

**quality_risk + limiter:** T0 quality. If T0 is gibberish, T1 must fire always → active≈stored, family collapses to P5. Doctor, not cosine, decides hit.

**cheapest_falsifier:** Decode T0-only vs T0+T1 on fast-Doctor. If T1 never restores a hit **or** never fires → drop T1.

**execution_path:** Needs a decode path that can stop at T0. No Hawking tier kernel today. Expanding T0+T1 to dense every token = <REDACTED> All. Useful as a **search schedule** even if T1 dies: T0 is the cheap screen.

**confidence:** **MED** as a representation. **HIGH** as a search tactic.

**transfer:** Hit-rate is patient/prompt specific. Do not freeze T1 from one battery.

---

### P7 — Search policy (candgen change, not a codec)
**mechanism:** Deterministic emit order and hard filters so the engine spends first dollars on P1→P5→P3/P2-screen→P4, never on killed families.

**complete_byte_accounting:** N/A (policy). Every emitted spec must carry `complete_bpw` **and** `active_bpw` **and** `codebook_frac` **and** `native`.

**stored_bpw / active_bpw:** N/A.

**expected_reachable_bpw:** Does not compress. Changes **time-to-first-honest-sub-1**.

**quality_risk + limiter:** Over-pruning a live family. Mitigate with `reopen_if` already in NEGATIVE_SCIENCE (cos≥0.10, p0, non-Lloyd entropy≤0.9, layer0).

**cheapest_falsifier:** Generate the grid for O005. Assert: (i) no raw-PQ-1bit, (ii) no entropy-Lloyd, (iii) no expert-template unless cos≥0.10, (iv) P1 ranks above P4, (v) every spec has complete≥nominal.

**execution_path:** `odyssey_candgen.generate` + prune Pareto(complete_bpw, DoctorΔ, active_bytes). Cost-model stays “insufficient data” until ≥2 patients — **do not invent ETAs**.

**applicability:** All Odyssey patients.

**confidence:** **HIGH**. This is the only proposal that cannot violate complete-accounting.

**transfer:** Filters are global; priors are class-keyed (moe/dense/hybrid).

---

## 4. What is reachable NOW vs later

| Target | NOW on current patients? | How | Blocker if no |
|---|---|---|---|
| **Active < 1 BPW** (MoE) | **Yes, arithmetically** on O005/O006 | P1: q2 experts + gather + 4–8 nonexpert | mlx q2 Doctor; real gather kernel |
| **Stored < 1 BPW** | **Maybe ~0.8** | P3 BTC or P2 if p0 high; P5 if mixed lands | Doctor, not PPL; native LUT/dict |
| **Stored < 0.2 BPW** | **No, not on 7–30B PTQ** | LittleBit 70B GB-implied 0.23 is QAT + embed amortization | QAT budget; embed still ~0.5 GB @ FP16 on 7B |
| **Hawking GLM 0.167** | Precedent on **GLM real acts**, organ-local | Do not assume it transfers | Qwen3.8 proxy all died |

**Do not run:** raw-W PQ/VQ at 1 bit; Lloyd-index entropy; expert templates on Qwen-class; N:M “0.55 bit” without mask bits; VPTQ 2^16 FP16 codebooks; gaussian-proxy ranking; BitNet-from-scratch as a PTQ of these patients.

**Cite-or-UNKNOWN discipline:** NanoQuant Wiki2/zero-shot, QuIP# exact sign-vector complete on these shapes, GLM 0.167 receipt internals, Odyssey Doctor at any of these codecs — **UNKNOWN**.
