# Hawking Ladder V3

Sealed 2026-07-20T19:20:32Z from `/Users/scammermike/HawkingWorktrees/deep-architecture-foundry`.
Machine-readable companions: `HAWKING_LADDER_V3.json`, `HAWKING_PROVIDER_CAPABILITY_MATRIX.json`.

## What this is and how it was made

Metadata-only resolution of every ladder rung and probe against the official Hugging Face API and
each repo's official `config.json` on its pinned revision. No weights were downloaded, no lease was
acquired, no forward was run. Total network transfer under 5 MB.

The live Qwen3-235B campaign (pid 86590, approximately layer 48/94, out of
`/Users/scammermike/hawking-qwen-recovery-20260720`) was not touched.

Three evidence classes are used throughout and are never blurred:

- **VERIFIED** means fetched live from the official repo API or its official `config.json` on this date.
- **DERIVED** means arithmetic over verified numbers, with the arithmetic written out.
- **REPORTED** means it came from the campaign brief or secondary press and was not confirmed against
  an official artifact.

## Verified rungs (8 of 9)

| Rung | Official repo | License | Revision | Arch family | Stage |
|---|---|---|---|---|---|
| F0 | `openai/gpt-oss-120b` | apache-2.0 | `b5c939de` | gpt_oss | A8 (closed) |
| F1 | `Qwen/Qwen3-235B-A22B-Instruct-2507` | apache-2.0 | `ac9c66cc` | qwen3_moe | A7 (live) |
| F2 | `Qwen/Qwen3.5-397B-A17B` | apache-2.0 | `84726181` | qwen3_5_moe | A2 |
| F3 | `MiniMaxAI/MiniMax-M3` | other (minimax-community) | `50942730` | minimax_m3_vl | A2 |
| F4 | `deepseek-ai/DeepSeek-V3.2` | mit | `a7e62ac0` | deepseek_v32 | A2 |
| F5 | `zai-org/GLM-5.2-FP8` | mit | `ba978f7d` | glm_moe_dsa | A2 |
| F6 | `moonshotai/Kimi-K2.6` | other (modified-mit) | `7eb5002f` | kimi_k25 | A2 |
| F7 | `deepseek-ai/DeepSeek-V4-Pro` | mit | `b5968e91` | deepseek_v4 | A2 |

All eight exist, are public, are ungated, and carry config, tokenizer, safetensors index, and real
weight shards. Every id above is in the model author's own namespace. **No rung resolved to a
community repack.**

The F5 fallback `zai-org/GLM-5.2` (BF16, mit, revision `b4734de4`) was also verified and exists.

## Unresolved rung (1 of 9)

**F8 Kimi K3 does not exist publicly.** Four candidate repo ids were probed
(`moonshotai/Kimi-K3`, `-Base`, `-Instruct`, `-Thinking`); all returned HTTP 401. The complete
`moonshotai` author listing was fetched and sorted by last-modified: the newest entries are
`Kimi-K2.7-Code` (2026-06-15), `Kimi-K2.6` (2026-05-19), `Kimi-K2.5` (2026-04-30). There is no K3
repo of any name.

Secondary press dated 2026-07-16 reports a 2.8T model with weights promised on Hugging Face on
2026-07-27 under a modified MIT license. That is REPORTED, not verified. The seal date is
2026-07-20, so the promised date has not arrived.

Status is **PENDING_OFFICIAL_WEIGHTS at stage A0**. No repo id, license, revision, config field,
tensor name, or dimension is recorded for this rung. KDA, Attention Residuals, Stable LatentMoE, and
16-of-896 are recorded only as REPORTED claims with an explicit fabrication guard. The rung advances
only when repo, license, revision, config, tokenizer, index, and weights all verifiably exist.

## Probes

| Probe | Official repo | License | Status | Config readable |
|---|---|---|---|---|
| P1 | `deepseek-ai/DeepSeek-V4-Flash` | mit | public | yes |
| P2 | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` | apache-2.0 | public | yes |
| P3 | `baidu/ERNIE-4.5-300B-A47B-PT` | apache-2.0 | public | yes |
| P4 | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | other (llama4) | **gated: manual** | **no (401)** |
| P5 | `inclusionAI/Ling-1T` | mit | public | yes |

P4 is the only entry on the entire list whose config could not be read. Its metadata and file tree
are public, its file contents are not. Architecture fields are marked UNRESOLVED rather than filled
in from memory. Accepting the license gate is a human decision and was not taken.

## Storage-mode calls

Host envelope at seal: 122 GiB free disk (approximately 131 GB), 96 GB usable RAM, 28 cores, with
roughly 15 cores already committed to the live campaign.

Policy: `FULL_DISK_RESIDENT` at or under 90 GB; `VULTURE_SHARD_SERIAL` above that when the largest
single shard is at or under 22 GB and the total is at or under 1000 GB; `BOUNDED_REMOTE_RANGE` above
1000 GB or when no local materialization is possible.

| Entry | Main-revision bytes | Shards | Max shard | Mode |
|---|---|---|---|---|
| F0 gpt-oss-120b | 130.5 GB | 22 | 10.54 GB | VULTURE_SHARD_SERIAL |
| F1 Qwen3-235B | 470.2 GB | 118 | 4.00 GB | VULTURE_SHARD_SERIAL |
| F2 Qwen3.5-397B | 806.8 GB | 94 | 9.66 GB | VULTURE_SHARD_SERIAL |
| F3 MiniMax-M3 | 854.2 GB | 59 | 16.10 GB | VULTURE_SHARD_SERIAL |
| F4 DeepSeek-V3.2 | 689.5 GB | 163 | 6.64 GB | VULTURE_SHARD_SERIAL |
| F5 GLM-5.2-FP8 | 755.6 GB | 141 | 5.37 GB | VULTURE_SHARD_SERIAL |
| F5 fallback GLM-5.2 BF16 | 1506.7 GB | 282 | 5.37 GB | BOUNDED_REMOTE_RANGE |
| F6 Kimi-K2.6 | 595.2 GB | 64 | 9.81 GB | VULTURE_SHARD_SERIAL |
| F7 DeepSeek-V4-Pro | 864.7 GB | 64 | 13.96 GB | VULTURE_SHARD_SERIAL |
| F8 Kimi K3 | none | none | none | UNDETERMINED |
| P1 DeepSeek-V4-Flash | 159.6 GB | 46 | 3.60 GB | VULTURE_SHARD_SERIAL |
| P2 Qwen3-Coder-480B-FP8 | 482.1 GB | 49 | 10.00 GB | VULTURE_SHARD_SERIAL |
| P3 ERNIE-4.5-300B | 603.0 GB | 124 | 4.98 GB | VULTURE_SHARD_SERIAL |
| P4 Llama-4-Maverick | 803.2 GB | 55 | 21.47 GB | BLOCKED_GATED |
| P5 Ling-1T | 1999.4 GB | 155 | 17.18 GB | BOUNDED_REMOTE_RANGE |

**Nothing on this ladder is FULL_DISK_RESIDENT.** The smallest entry (P1 at 159.6 GB) still exceeds
free disk. Vulture shard-serial is not an optimization for this ladder, it is the only mode that
works, and every large rung happens to shard well: the largest single shard anywhere is 21.47 GB,
and eleven of fifteen entries have a max shard under 17 GB.

Choosing `zai-org/GLM-5.2-FP8` over the BF16 repo for F5 halves traversal bytes and is the single
decision that keeps that rung out of range-read mode.

## Findings worth carrying forward

**F7 really is 1.6T, and the Hugging Face headline is wrong.** HF reports 861.6B parameters for
`DeepSeek-V4-Pro` because it counts packed I8 bytes. The config declares `expert_dtype: fp4`, so
each I8 byte holds two values: 786,381,668,352 x 2 = 1,572,763,336,704 expert values. The E8M0 scale
count is an exact independent check, 1,572,763,336,704 / 32 = 49,148,854,272 against a reported
49,150,268,416, agreeing to 0.003 percent. Real total is approximately 1.599T. The exact 1:32 ratio
also proves block-32 microscaling for the experts, which is a different layout from the `[128,128]`
`weight_block_size` in the fp8 quantization config (that governs the fp8 tensors only). DERIVED.

**P1 is the same layout at one fifth the bytes.** DeepSeek-V4-Flash is `deepseek_v4`, same fp4-in-I8
plus E8M0 scheme, 159.6 GB against 864.7 GB, and its scale check agrees to 0.004 percent. Its real
parameter count is approximately 291B, not the 158B headline. This is the correct place to prove the
`deepseek_v4` adapter and the fp4 dequant path before committing 864.7 GB of traversal to F7.

**F6's INT4 claim is real but narrower than "1T native INT4" suggests.** compressed-tensors,
pack-quantized, 4-bit, group size 32, applied to routed experts only. Attention, shared experts,
dense MLP, `lm_head`, the vision tower, and the mm projector are all in the explicit ignore list and
stay BF16. The parameter split is 1.0147T INT4 against 43.9B BF16, and the byte arithmetic
(507.34 GB + 87.80 GB = 595.14 GB) matches the measured 595.2 GB exactly, so the trillion-parameter
INT4 figure itself checks out.

**F2 is multimodal and the brief did not say so.** `Qwen3.5-397B-A17B` is
`Qwen3_5MoeForConditionalGeneration` with a vision tower, image and video token ids, and both image
and video preprocessor configs. The hybrid recurrent claim is solid and bound to real config: 45
`linear_attention` layers against 15 `full_attention` on a period of 4, with mamba-style
`linear_conv_kernel_dim` and `mamba_ssm_dtype` fields. It is also the only model on the ladder that
requires two different attention operators dispatched per layer.

**F4 is not a 1M-context model.** `DeepSeek-V3.2` declares `max_position_embeddings: 163840`. F5, F7,
and P1 all declare 1048576 and are verified 1M. Any 1M claim for F4 is REPORTED and unsupported.

**F7's architecture names are REPORTED, its fields are VERIFIED.** `compress_ratios`,
`compress_rope_theta`, `num_hash_layers`, `hc_mult`, `hc_sinkhorn_iters`, `hc_eps`, `o_lora_rank`,
`o_groups`, and `sliding_window: 128` all exist in the official config. Mapping the names CSA, HCA,
and mHC onto them came from the brief; no official DeepSeek document establishing that mapping was
fetched. Implement against the field names, not the acronyms. F7 also uses
`scoring_func: sqrtsoftplus`, unique on this ladder, where every other DeepSeek-family rung uses
sigmoid.

**F6 is the only rung that cannot use a Hugging Face tokenizers fast path.** It ships
`tiktoken.model` plus `tokenization_kimi.py` and has no `tokenizer.json` at all. P3 and P4 add
SentencePiece. The ABI currently carries a bare `tokenizer_identity` string, which is not enough to
express three protocols.

**Field naming is not uniform and the adapter tables must not assume it is.** ERNIE uses
`moe_num_experts` / `moe_k` / `moe_num_shared_experts`; Qwen uses `num_experts` /
`num_experts_per_tok`; MiniMax uses `num_local_experts`; DeepSeek and GLM use `n_routed_experts`.
ERNIE is also the only entry with zero shared experts.

**P5 is the worst probe on the list.** Ling-1T declares `max_position_embeddings: 32768`, the
shortest of any entry, while costing 1999.4 GB to traverse, the most of any entry.

## ABI gap

`crates/hawking-seed-c/src/providers/adapters.rs` ships llama, gemma2, phi3, olmoe, mixtral, gpt_oss,
and mamba2. Only `gpt_oss` covers a ladder rung, and that rung is closed. Every rung above F0 is
therefore capped at stage A2 no matter how well its config resolved. Ten architecture families
(`qwen3_moe`, `qwen3_5_moe`, `minimax_m3_vl`, `deepseek_v32`, `glm_moe_dsa`, `kimi_k25`,
`deepseek_v4`, `ernie4_5_moe`, `llama4`, `bailing_moe`) have no entry.

`source_decl.rs` documents MXFP4 and BF16. The ladder additionally requires F8_E4M3, F8_E8M0
(a scale-only dtype the ABI has never declared), I8-packed FP4, I32-packed INT4 group-32, and F32.

Attention families needed that no builtin covers: MLA, lightning-indexer sparse selection,
block-sparse top-k, and gated linear attention. `mamba2` is the nearest builtin to F2's
`linear_attention` layers and is a different operator; it must not be aliased.

Ordered work items and a recommended build order are in
`HAWKING_PROVIDER_CAPABILITY_MATRIX.json` under `derived_abi_work_items` and `recommended_order`.

## Honest boundary

Everything in the tables above with a repo, license, revision, or config field was fetched. Nothing
was written from memory. F8 has no repo and no fields. P4 has a repo and no fields. The 1.6T and
291B parameter figures are arithmetic over fetched numbers, shown in full in the JSON, not quotes
from a model card. The names CSA, HCA, mHC, KDA, Attention Residuals, and Stable LatentMoE are
REPORTED and none of them is bound to a fetched official document.

No synthetic or bounded test in this repository is parent capability evidence, and nothing in this
document claims otherwise.
