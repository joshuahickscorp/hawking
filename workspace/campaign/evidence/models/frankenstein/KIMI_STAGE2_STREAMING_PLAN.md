# Kimi Stage-2 Streaming / Storage Plan

**Status:** PLAN ONLY — no Kimi bytes streamed, no GPU, no Stage-2 training.  
**Companion seals:** `KIMI_STAGE2_READINESS.json`, `KIMI_STRATEGIC_BRIDGE_CONTRACT.json`, `KIMI_CORPUS_PLAN.json`  
**Prior plan:** `STAGE2_KIMI_STREAMING_DISTILL_PLAN.md` / `.json` (still valid; this document freezes the **disk math** with the Proto body kept resident).

---

## 1. Law

1. **Two-stage, not simultaneous.** Stage-2 starts only after `PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED`.
2. **DeepSeek body stays.** Proto = base body + reversible adapters. Stage-2 loads the **same** base and adds Kimi adapters. Do **not** plan around deleting the body.
3. **One donor window at a time.** Kimi full body (~1.56 TB) never fully resident. Same layer-major pattern as GLM.
4. **Metadata admission ≠ stream authorization.** `KIMI_K3_OFFICIAL_SOURCE_ADMITTED_METADATA_ONLY` is identity only.

---

## 2. Source sizes (measured / admitted — not invented)

| Asset | Bytes | GiB (1024³) | Source |
|------|------:|------------:|--------|
| Kimi-K3 weight shards (96) | 1 560 936 091 448 | **~1453.7** | `KIMI_K3_SOURCE_ADMISSION.json` |
| DeepSeek-V4-Flash body (blobs) | 159 617 149 040 | **~148.7 ≈ 149** | `DEEPSEEK_V4_FLASH_SOURCE_ADMISSION.json` `source_bytes_from_blobs` |
| Hard free floor (this programme) | 25 × 1024³ | **25** | teacher-forced executor `MIN_FREE_FLOOR_BYTES` |
| Proto adapters (Stage-1) | small (MB–low-GB class) | **≪ body** | reversible residual bank after V0 seal |
| Control plane (config + index + tokenizer) | ≪ 1 GiB | **≪ 1** | HF control assets only |

Pinned Kimi revision (do not re-admit a different one):

- repo: `moonshotai/Kimi-K3`
- revision: `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`
- admission seal: `597ac05752f84c8fffc175255e64d0c56b976ef3c389a59cebabf322b4fa30b0`

Architecture (HF `config.json` @ same revision, `text_config`):

| Field | Value |
|------|------:|
| hidden_size | 7168 |
| num_hidden_layers | 93 |
| num_experts | 896 |
| num_experts_per_token | 16 |
| num_shared_experts | 2 |
| max_position_embeddings | 1 048 576 |
| attention | hybrid full_attn + KDA (`kimi_linear`) |
| tokenizer | tiktoken (`tiktoken.model`) |

---

## 3. Working-set invariant (Stage-2)

```
resident =
    DeepSeek-V4-Flash body          (~149 GiB)   # STAYS for entire Stage-2
  + Proto Stage-1 adapter set       (small)
  + at most ONE Kimi layer-major window
  + frozen corpus + carry states    (bounded)
  + current output / scratch
  + hard free floor                 (25 GiB must remain free)
```

**Prohibited:**

- GLM donor windows + Kimi windows co-resident
- Full Kimi body resident
- Deleting the DeepSeek body to “make room” for Kimi
- Starting a window when `free − (window + scratch) < 25 GiB`

---

## 4. Where headroom comes from (reclaim script — confirmed)

Script: `workspace/campaign/records/runs/frankenstein/reclaim_storage_keep_proto.py`

**Deletes only (after V0 seal + proto present on Desktop):**

1. `.../records/runs/frankenstein/glm-donor` — GLM donor shard cache  
2. `.../workspace/ops/local/hf-cache` — HF/xet download cache  

**Explicitly does NOT delete:**

- DeepSeek body (comment in script: Stage-2 needs the same base)
- Proto artifact / Desktop deliverable
- Source code, worktrees, git
- Extracted GLM math subspace (small signal)

So Stage-2 headroom is: **GLM donor cache reclaim + hf-cache reclaim + any residual GLM teacher-forced stream root eviction**, **not** the 149 GiB body.

Operator sequence after V0 seals:

1. Confirm `PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED` on Desktop proto receipt  
2. Dry-run reclaim → `--apply`  
3. Measure free disk  
4. Budget first Kimi window against floor + body + adapters  

---

## 5. Bounded layer-major windows (same pattern as GLM)

```
freeze Stage-2 strategic corpus (KIMI_CORPUS_PLAN ladder)
→ for layer L in 0..92:
     floor check
     stream shards for L (and globals only when needed for embed/final)
     hash-verify window
     teacher-forced forward over ALL sequences/microbatches
     capture bounded hidden samples + stats (not full [B,S,H] dumps)
     atomic seal carry after_L
     prefetch L+1
     evict shards only needed by completed layers
→ final norm + bounded short logits
→ emit paired traces (kimi side; student side from Proto forward)
```

Double-buffer accounting: **N−1 seal/evict · N execute · N+1 prefetch**.

Window sizing guidance (planning; exact shard map from Kimi index at stream time):

- Prefer **single-layer** windows first (93 steps), same as GLM 78-layer path  
- If a single layer’s shards exceed safe headroom, split by expert-group / shard subset with fail-closed incomplete-layer reporting  
- Never hold >1 “logical layer window” + its prefetch sibling beyond double-buffer policy  
- Expected peak extra beyond body: **one layer window + prefetch ≈ low tens of GiB class** (exact bytes from index at execution — not guessed here as fact)

Executor generalization ready for this path:

- `lab/operators/frankenstein_teacher_forced_executor.py` + `KIMI_K3_ARCHITECTURE`  
- Backend for Kimi decoder still **PENDING** (prep does not implement Kimi ops)  
- GLM path remains the working reference (`backend="glm52"`)

---

## 6. Disk math (planning budget)

Let:

- `F` = free disk before Stage-2  
- `B` = 149 GiB DeepSeek body (already on volume; not freed by reclaim)  
- `A` = Proto adapter set (small)  
- `W` = max Kimi window + prefetch  
- `S` = carry + corpus + scratch  
- `Floor` = 25 GiB  

**Start gate:**

```
F_after_reclaim - W - S - A_overhead  ≥  Floor
```

Body `B` is already counted in used space; reclaim does not add `B` to free. Do not subtract `B` again from free when it is already resident.

**Minimum free after reclaim (order-of-magnitude):**

```
Floor 25 GiB
+ W   (budget conservatively 40–80 GiB until live index measures real per-layer shards)
+ S   (budget 5–15 GiB for L1 corpus + carries + traces)
────────────────────────────────
≈ 70–120 GiB free after reclaim before first Kimi fetch
```

If free is below that after GLM-donor + hf-cache reclaim: **stop**, free more non-body caches, or reduce window — never the body.

---

## 7. Sequencing vs Stage-1 / GPU lanes

| Lane | Role during Stage-2 prep | Touch? |
|------|--------------------------|--------|
| `dsv4f-fullseq-depth` | owns GPU / DSV4F fullseq | **NO** |
| `v0-correspondence-l0l1` | correspondence measurement | **NO** |
| This prep | architecture + contracts + executor generalization | YES (planning/code only) |
| Stage-2 execute (future) | Kimi stream after V0 | blocked until V0 seal |

---

## 8. Start checklist (execution lane — not this prep)

- [ ] `PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED`  
- [ ] Proto artifact loadable; Stage-1 bridges ablatable  
- [ ] `KIMI_STRATEGIC_BRIDGE` adapters still absent / PRESERVED_UNTOUCHED  
- [ ] Reclaim dry-run + apply (GLM donor + hf-cache only)  
- [ ] Free ≥ floor + planned window + scratch  
- [ ] Kimi control plane at pinned revision verified  
- [ ] Kimi reference backend implemented + smoke on synthetic or first layer  
- [ ] `KIMI_CORPUS_PLAN` materialised to L0/L1 jsonl (local sources)  
- [ ] Student forward available for residual fit at transplant points  

---

## 9. Honesty

- No Kimi weight shards are downloaded by this plan document or the prep lane.  
- MoE counts (896 / 16 / 2) come from HF `config.json` `text_config` at the pinned revision; admission top-level `n_routed_experts` was null due to extractor depth, not model absence.  
- Per-layer shard byte sizes will be measured from `model.safetensors.index.json` at stream time — not invented here.
