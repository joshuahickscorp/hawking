# Kill Ledger — tested + rejected, with evidence (autonomous campaign, 2026-06-21)

Every entry was measured or code-verified this run. "REJECTED" = do not pursue; "CLOSED" = axis is tapped;
"DEPRIORITIZED" = possible but low-EV / high-effort.

| Lever | Verdict | Evidence |
|---|---|---|
| `FFN_DOWN_Q4K` (requant ffn_down → Q4_K) for speed | **REJECTED** | "+29%" was a COLD-START PSO-compile artifact. Warm 5-trial median: default 39.0 vs fdq4k 39.6 = **+1.4% (noise)**. Also not bit-identical. |
| `int4-KV` **per-ROW** scheme | **REJECTED** | Long-ctx 17.69 vs 18.76 = **SLOWER**; **0% argmax-identity** (per-row outlier collapse, "The The The"). Per-row f16 scale rounds non-outlier channels to ~0. (Per-CHANNEL is a *different*, alive scheme — see roadmap.) |
| `Q6_K predec` (pre-decoded scale table for ffn_down) | **REJECTED for speed** | Q6_K scales are `int8` (1 read + 1 cast + 1 mul/block) — trivial, unlike Q4_K's 6-bit *packed* scales (which predec hoisted for +34%). Predec table is additive DRAM (+6.7% f16 / +21.9% f32). FFN_DOWN_Q4K warm-null corroborates ffn_down is off the warm-critical byte path. |
| `Q6_K ffn_down` 1r / 4r row-blocking | **CLOSED** | Warm A/B: **2r (default)=40.48** > 4r=40.04 > 1r=39.91. The chosen default is already optimal. |
| `f16-scales` alone (the predec scale stream) | **opt-in / ~0%** | Direct A/B: off=40.58 on=40.71 (+0.3%, all argmax-identical here). No effect on this binary; stays opt-in (it failed an earlier quality oracle on other inputs). |
| Spec-decode (EH free-market AND trained EAGLE) for speed | **REJECTED for speed** | Per-cycle overhead wall: even **87% accept → 0.91×** on this engine. Verify-cost curve amortizes only at large B with late acceptance. Lossless router kept (`73fc5b4`), default-OFF. |
| Spec-decode batched verify = lossless | **REJECTED (was a latent bug)** | Batched `forward_tokens_verify` B==1 returned the INPUT token + mis-wrote KV → non-lossless at near-ties. Fixed by routing B==1 → greedy; property gate now 20/20 vs no-spec canonical greedy. |
| `Q6_K ql-coalescing` repack (fix stride-8 load) | **DEPRIORITIZED** | The stride-8 `ql` load is real + shared by all row-blocking variants (bit-identical), but fixing needs a Q6_K *repack* (sidecar + new kernel). The bible's A10 layout attempt hit **−16.8%**; low-EV given the ffn_down warm-null. |
| KD self-rollout "≈4× accept lift" | **REJECTED (was noise)** | The self-rollout accept metric was noise; teacher-forced top-1 (control=100%) showed only **+2-4 pts** KD>SFT, all models undertrained (~19% ≪ 60%). Chunked-flag mismatch hypothesis also DISPROVEN (identical re-eval). |

## Decode-kernel micro-opt — bible §3.0 CLOSED for batch=1 decode (re-confirmed this run)
The ~1.6× llama/MLX gap is structural for DECODE (M=1). These were built + measured by prior campaigns; my Q6_K
1r/2r/4r A/B (2r already optimal) re-confirms the track is tapped. (All stay LIVE for PREFILL M>1.)
| Kernel lever | Verdict | Evidence |
|---|---|---|
| simdgroup-matrix (MMA) decode GEMV (A7) | **DEAD (decode)** | MMA is a compute lever; decode is BW-bound and M=1 underfills the tile 7/8. The unwired `gemv_q4_k_m_simdmat_pinned_tcb` was benched at **+1.6%** (below gate). |
| access-order weight layout repack (A10) | **DEAD** | built + measured **−16.8%** (`reports/a10_layout_repack_design.md`); layout must co-design with the kernel. |
| vectorized uint4 unpack (A5) / occupancy tuning (A6) | **DEAD** | loads already simdgroup-coalesced (A5); BW-bound + oversubscribed (A6). |
| `gemv_q4_k_m_v3_llama` (llama-style 2sg×4row) | **within noise** | benched ~0%; kept unwired. |
| Q3_K weight byte-cut (decode) | **DEAD for speed** | clean 33.3 GB/s, ~23% peak — **compute-bound** (3-bit dequant overhead > byte saving). A compression option only (decode-slower), like the trellis. Kernels `gemv_q3_k_*` exist + parity-tested, unwired. |

**Implication:** the MLX-diff agent's simdgroup-matrix hypothesis is largely PRE-CLOSED for decode. The one possibly-untried
structural axis is **split-K / column-split for under-occupied k/v_proj** (SpQt +17%) — but that's a small byte share (modest e2e).

## Open (NOT killed — being actively tested)
- **Per-CHANNEL int4-KV** — built + validated, dead-called → WIRING in progress (`HAWKING_QWEN_INT4_KV_PC`). Gate: real-model PPL.
- **MLX-diff for the ~1.6× gap** — research in progress (structural deltas: split-K / 128-bit loads / register blocking).
- **RWKV-7 SSM moat** (flat long-ctx decode) — bench running.
- **f16-activations in GEMV**, **GQA group-coalesced MHA** — designs, untested.
