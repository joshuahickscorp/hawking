# Qwen3-Coder-30B Gravity First Test — Summary

**Date:** 2026-08-06  
**Source:** `/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct`  
**Revision:** b2cff646eb4bb1d68355c01b18ae02e7cf42d120  
**Method:** symmetric group absmax RTN, group_size=64 (weight-space; not activation-aware)

## Geometry (from real config.json)

| Field | Value |
|-------|-------|
| architecture | Qwen3MoeForCausalLM / qwen3_moe |
| hidden_size | 2048 |
| layers | 48 |
| experts | 128 (top-8) |
| moe_intermediate_size | 768 |
| attn heads / kv | 32 / 4 (head_dim 128) |
| vocab | 151936 |
| dtype on disk | BF16 |
| tensors | 18867 (18432 expert) |
| total weights | 30,532,122,624 |
| expert weight fraction | 94.95% |
| disk size | ~57 GB (index total_size 61,064,245,248 B) |

## Results by bit width

### 8-bit calibration (gs=32/64/128, layers 0+24, 2 experts)
- FFN mean cos ≥ 0.99995, MoE combine near_lossless → metric ladder is sane.

### 4-bit (~4.25 expert BPW; ~4.84 whole-model if rest bf16)
**Bounded functional (layers 0/24/47, experts 0/1/7/63):**
- weight recon mean cos ≈ 0.993, mean rel_l2 ≈ 0.115
- expert FFN mean cos ≈ 0.987, min ≈ 0.981
- mid-layer MoE combine softer (L24 cos≈0.983, rel_l2≈0.18)

**Generation — ALL 48 layers experts quant-dequant (attn/router/norm/embed untouched):**
- mean prefix token agreement vs greedy baseline: **0.990**
- all 3 prompts coherent; 2/3 exact match
- **generation_holds = True**

### 3-bit (~3.25 expert BPW; ~3.89 whole-model if rest bf16)
**Bounded functional:**
- weight recon mean cos ≈ 0.966
- expert FFN mean cos ≈ 0.931, min ≈ 0.906 — degraded on strict bars

**Generation — all experts:**
- mean prefix agreement: **0.448**
- still on-topic / readable, but rewords and changes solution approach
- **generation_holds = False** (not claimed as working floor)

### 2-bit (~2.25 expert BPW; ~2.94 whole-model if rest bf16)
**Bounded functional:**
- weight recon mean cos ≈ 0.769
- expert FFN mean cos ≈ 0.55 — collapse

**Generation — all experts:**
- mean prefix agreement: **0.010**
- Math-Preserve-style: token-ish English, task competence destroyed
  - `"I'll provide code to the point where I can't make a 100% chance of the 100% and 100%"`
  - `"17 * 24 = 17 * 24 = 17 * 24 = 17 *"` (loop)
- **not capable**

## Honest floor

**4-bit is the honest floor** for this first, naive group-RTN pass.

- Lowest BPW with real generation quality still matching the uncompressed model: **~4.25 BPW on experts** (≈ **4.84 BPW whole-model** if non-expert tensors stay bf16).
- 3-bit is **not** reported as working: functional FFN degrades and full-expert generation drifts hard from baseline (agree 0.45).
- 2-bit is a **capability collapse** (Math-Preserve risk realized on generation).

This is **not** a sub-bit claim and **not** a universal 1.5-BPW result. Per the Bible: lowest capable runnable equilibrium for *this* method on *this* model lands at ~4-bit group RTN.

## What was NOT done
- No activation-aware fitting (no Qwen teacher capsules yet; glm52_activation_aware_pack.py ideas reused for ledger/metrics only).
- Attention / router / embed / lm_head not quantized in the generation probe.
- No full coding benchmark / TG gauntlet.
- No production Metal/Rust pack artifact.

## Files
- `lab/operators/qwen30b_gravity_pack.py`
- `workspace/campaign/evidence/models/qwen-30b/gravity-first-test/`
