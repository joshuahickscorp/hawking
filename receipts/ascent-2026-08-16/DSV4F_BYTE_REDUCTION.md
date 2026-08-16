# DSV4F per-token active bytes

Authority: `receipts/ascent-2026-08-16/DSV4F_BYTE_REDUCTION.json`.
Instrument: `python3 tools/ascent/roof_rungs.py`.
Artifact: `full-43-layer-stream.gravity` seal `ba9039bfe71328e2e47ced782bd1f931e2d412055382da0ea669092c1d90bfed`.
Honest decode ceiling: **411.51358589633037 GB/s**. A-budget: **8.230271718 GB**.

## Census (GPU-unique stored bytes the BOS graph must read once)

Earlier ledger (attn 4.7 / experts 3.4 / shared 1.1 ≈ 9.2 GB) was the right three organs and the wrong total. Independently recomputed from sealed pair geometry; confirmed against the 69,187-tensor manifest (every routed expert is 4,194,304 packed bytes).

| class | weights | stored bytes | GB | source dtype | BPW | % of 10.280 |
|---|---:|---:|---:|---|---:|---:|
| MLA (5 FP8 pairs) | 4,599,054,336 | 4,599,335,040 | **4.599** | E4M3 + UE8M0 128×128 | 8.0005 | 44.7% |
| routed experts (top-6 of 256) | 6,492,782,592 | 3,449,290,752 | **3.449** | E2M1×2 + UE8M0/32 | 4.2500 | 33.6% |
| shared expert | 1,082,130,432 | 1,082,196,480 | **1.082** | E4M3 + UE8M0 128×128 | 8.0005 | 10.5% |
| lm_head | 529,530,880 | 1,059,061,760 | **1.059** | BF16 | 16.000 | 10.3% |
| router `gate.weight` | 45,088,768 | 90,177,536 | **0.090** | BF16 | 16.000 | 0.9% |
| **GPU unique** | **12,748,587,008** | **10,280,061,568** | **10.280** | | **6.451** | 100% |

Matches `DSV4F_GPU_BODY_DIAGNOSIS.json` byte-for-byte. TOKEN_NS 5.857 GB @ 3.676 BPW is a blend and is not stored traffic.

Not in the 10.280 (and why):

- **indexer/compressor 0.801 GB** — present on 41/43 layers, **not loaded on the BOS graph** (`indexer_compressor_loaded=false`). A pos>0 decode would add it.
- **mHC 0.135 GB F32** — loaded, but on the **host** (`host_source_algorithm_exact_sha`), not GPU unique stored.
- norms + MLA aux (sink, q_norm, kv_norm) < 1 MB.
- embed full table 1.059 GB is storage; the token serves one 8 KB row.
- MTP excluded from the 43-layer path.

`host.memcpy` on the live token is 10,280,249,760 bytes — 188 KB off the unique stored sum.

## DSV4F's own BPW (Q80 rates do not transfer)

Experts are **already 16-level source-native FP4**. MLA/shared are **already source-native FP8**. Q80 mixed 1.44 complete / 0.14–1.13 expert organ rates started from BF16. Applying them here is a unit error.

Shannon of the **native codes**, measured on real chunks:

| organ | samples | entropy | all levels used? |
|---|---|---:|---|
| MLA FP8 | wq_a L0/L21/L42, wq_b/wo_a L0 | **6.54–6.61 bits** | 254/256 (both NaNs unused) |
| shared FP8 | w1 L0, L21 | **6.65–6.85 bits** | 254/256 |
| routed FP4 | w1/w2/w3 L0 e0; w1 L21 e17; w1 L42 e200 | **3.86–3.89 bits** | 16/16 |
| router BF16 | L3 gate.weight | **10.49 bits** | 3659 of 65536 |
| lm_head BF16 | first 8 MiB | **10.50 bits** | 4327 of 65536 |

Lossless floor ≈ those entropies + the existing UE8M0 scale BPW. That is DSV4F's achievable BPW without breaking `c94da765`.

## Candidates

ALU is **0.77%**. Rank by bytes first. Extra reconstruction is affordable here — the opposite of the Q80 trade.

| id | mechanism | GB | floor ms | roof tok/s | highest rung at these bytes | hc_sha |
|---|---|---:|---:|---:|---|---|
| C0 | current native | 10.280 | 24.98 | 40.03 | **none** (roof < 50) | keep `c94da765` |
| C1 | ANS/Huffman of native codes (measured H + 0.5% overhead) | 8.839 | 21.48 | 46.56 | **none** | keep (reconstructs exact codes) |
| **C2** | MLA+shared FP8→native FP4; lm_head BF16→FP8; router BF16→FP8 | **7.042** | **17.11** | **58.44** | **A** | **MUST reseal** |
| C3 | C2 + head/router also FP4 | 6.773 | 16.46 | 60.76 | A | reseal |
| C4 | C2 + top-k 6→4 | 5.892 | 14.32 | 69.84 | A | reseal + routing contract |
| C5 | C3 + routed 16-level→8-level (3.25 BPW) | 5.961 | 14.49 | 69.03 | A | reseal; new expert codec |
| C6 | ~2 BPW on every class (research codec) | 3.187 | 7.74 | 129.12 | **A and B** | reseal |
| C7 | 0.31 BPW (the 0.5 GB “today’s occupancy” fantasy) | 0.494 | 1.20 | 833 | A/B/C | not a contract |

C2 wq_a probe (L0, 1024×4096, captured X `hidden/L00/vocab_bos_v1/000000.f32le`): cosine **0.99653**, rel L2 **0.0837**, weight-row0 max-abs **0.015625**. CPU source-algorithm preview, not a seal.

### Numeric contracts

- **C0 / C1**: keep `SEALED_GRAPH_HC_BF16_SHA256 = c94da765…`. C1 is bit-identical because the decoder emits the same E2M1/E4M3/E8M0 bytes the current kernels already consume.
- **C2**: new sealed HC hash **plus** BOS greedy token-id unchanged, `|logit − 16.78185| ≤ 0.05`, final-HC cosine vs sealed BF16 HC **≥ 0.995** (justified by the 0.9965 wq_a probe; must be re-measured on the full 43-layer HC before the assert moves). Rewrite the e2e assert. Do not drop it.
- **C4**: C2 bar **plus** a fullseq routing contract. Top-k is a source-config change (`num_experts_per_tok`).
- **C5 / C6**: new codec on already-quantized data. Propose HC cosine ≥ 0.98 and fullseq greedy-token match. Not bit-identical.

## Verdict

**≤ 8.230 GB is reachable. It is not reachable while keeping `c94da765`.**

C1 is a proof, not a hope: measured Shannon of the codes that already exist is 8.839 GB / 21.48 ms / 46.56 TPS. That is 0.608 GB and 3.44 TPS short of rung A. You cannot beat Shannon losslessly.

The first honest path is **C2 (7.042 GB, 17.11 ms floor, 58.44 TPS roof)**. It only touches classes that are still 8- or 16-bit. Routed experts stay native 16-level FP4. Attention has never been compressed on any model here; it is 44.7% of the token and it is the lever.

**3–4 GB** is C6 (2 active BPW everywhere) or a deeper sparsity cut than top-4. C4 is still 5.89 GB.

**0.5 GB** (what today’s 6.3% of roof would need) is not reachable on this model.

Occupancy work is still required. At 6.3% of roof C2 is 273 ms. Bytes make 20 ms *possible*; the 16× occupancy gap makes it *real*. A heavier reconstruction codec is the right trade on this ALU, not a Q80-style “don’t add decode work” veto.

Decode-token caveat: pos>0 adds 0.801 GB of indexer/compressor. C2+indexer = 7.843 GB / 19.06 ms — still under 20 ms, still A-reachable.
