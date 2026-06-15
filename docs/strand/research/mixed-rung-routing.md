# Mixed-rung routing — layer sensitivity and bpw measurement

_2026-06-11. Produced by Ω* agent (today+next wave). All bpw numbers are MEASURED on M3 Pro.
PPL comparison is partially measured + literature-prior estimate — labeled explicitly below._

---

## Method

**Why rel-RMS is the wrong sensitivity metric:**
The Ω agent (prior wave) measured per-pattern rel-RMS degradation at 4-bit → 3-bit across all 7
tensor patterns. The spread was only **0.019 pp** across all patterns. Cause: the RHT whitens
every tensor's weight spectrum toward i.i.d. Gaussian before encoding, so the trellis quantizer
sees the same distribution regardless of tensor role. Under STRAND's RHT, there is no "easy"
tensor type at 3-bit — they all degrade identically in reconstruction quality.

**Conclusion:** sensitivity must be measured in OUTPUT space (PPL delta), not reconstruction
space (rel-RMS). The routing table below uses a literature-prior for the PPL split since the
full per-pattern PPL sweep takes ~8–15 h on M3 Pro (CPU Viterbi, 168 tensors per encode ×
7 patterns × 4-bit→3-bit). That sweep is queued for the pod.

---

## Routing table (attn@4-bit / FFN@3-bit)

| pattern | bits | weight_frac% | rationale |
|:---|:---:|---:|:---|
| q_proj, k_proj, v_proj, o_proj | 4 | 12.30% | attention projections — route token structure, quality-sensitive (literature) |
| gate_proj, up_proj, down_proj | 3 | 87.69% | FFN projections — linear transform, more noise-tolerant (literature) |

**Rung-config file:** `configs/rung-attn4-ffn3.json`
```json
{"q_proj": 4, "k_proj": 4, "v_proj": 4, "o_proj": 4, "gate_proj": 3, "up_proj": 3, "down_proj": 3}
```

---

## bpw measurement (MEASURED on M3 Pro, Qwen2.5-0.5B, 168 tensors)

| config | bpw | rel-RMS | PPL |
|:---|---:|---:|:---|
| all-4-bit | 4.5001 | 7.47% | **13.535 (MEASURED, stage2.2)** |
| all-3-bit | 3.3399 | 17.43% | **15.57 (MEASURED, stage2.2)** |
| attn@4 / FFN@3 | **3.4861** | 15.31% | ~14.8–15.2 (ESTIMATED, literature-prior, unverified) |

Mixed-rung bpw of **3.4861** sits −1.01 bpw below all-4-bit and +0.15 above all-3-bit.
If the PPL estimate holds, the mixed config buys ~0.4–0.8 PPL points over all-3-bit at nearly
the same wire cost — but this MUST be confirmed by actual PPL measurement on the pod.

---

## Infrastructure built

- `--rung-config path.json` flag added to `crates/strand-quant/src/bin/quantize-model.rs`
  - Priority: mp_config > rung_config > --bits (per-tensor overrides win)
  - Key is a substring match against tensor names; first match wins
- `scripts/mixed-rung-encode.sh` — driver: takes model + rung-config, calls quantize-model,
  reports bpw; accepts `--eval --hf-dir` for PPL measurement when available
- `configs/rung-attn4-ffn3.json` — the attn@4/FFN@3 routing config

---

## Next steps

1. **Pod: full PPL sweep.** Run `scripts/layer-sensitivity.sh` on the pod with actual PPL
   (not rel-RMS proxy). 7 patterns × ~4 min on RTX 3090 = ~30 min. Update the routing table.
2. **Validate mixed-rung PPL.** Run `scripts/mixed-rung-encode.sh --eval --hf-dir $HF_DIR`
   on the pod with the `rung-attn4-ffn3.json` config. Get the real number.
3. **Explore more granular configs.** Embed proj vs. output proj sensitivity, per-layer
   depth sensitivity (early vs. late layers).
