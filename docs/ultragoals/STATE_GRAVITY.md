# STATE GRAVITY (N048 / S026 §50-58, §119)

Runtime state is a first-class representation. At long context and c>1 it
dominates the weights (N016 / `PREFILL_KV.json`: q4 c=4 at 32K, session
state 16.59 GiB exceeds MODEL_BYTES 13.32 GiB). Compression of that state
is designed on **measured** redundancy of OUR GQA KV and DeltaNet
recurrent state, not on paper assumptions.

Receipt: `receipts/headless/STATE_GRAVITY.json`.
Generator: `tools/headless/state_gravity.py`. CPU only; no GPU, no second
27B decode, parent tensors streamed read-only.

Long-context capability is the gate. Bytes booked below are recipe
arithmetic **conditional on measured redundancy**. The capability cost of
every recipe is ABSENT until it is wired into `Qwen38HybridDecodeSession`
and scored on a held-out long-context suite (PREFILL_KV:
`do_not_assume_kv_quant_is_free`).

---

## StateGenome

Hybrid Qwen3.8-27B. Full-attention every 4th layer (16 GQA, 48 DeltaNet).
Production KV is f32 (`mha_decode_f32_tcb`). Sizes lockstep with
`qwen38_workspace_bytes` / PREFILL_KV:

| term | grows with seq | bytes |
|---|---|---|
| GQA KV | yes | 131,072 B/token (16 × 4 kv-heads × 256 × 4 B × 2) |
| DeltaNet rec+conv | no | 156,893,184 B (rec 150,994,944 + conv 5,898,240) |
| activation scratch | no | 1,691,396 B |

At seq=256, DeltaNet is 82% of session state. At 32K it is 3.5%; GQA KV
is the only seq-linear term. Prefix-sharing: GQA KV is shareable on a
byte-identical prefix (not wired). DeltaNet rec_state is a **summary,
not a cache** — diverged suffixes cannot share it (canon law 14).
Speculative/MTP state is N049 (ABSENT here).

---

## Ranked axes

Measured on (1) a 128-token CPU hybrid prefill of the gravity Q4 artifact
on official-tokenizer ids (production-layout KV + rec_state) and (2)
capture_diverse2 hold split (2,212 tokens) with parent k/v/q at
post_attn_norm as a longer-token corroboration, labeled as a neighboring
site.

### 1. H2O (heavy-hitter eviction) — HAS the redundancy

Paper: Zhang et al., arXiv:2306.14048. S026 §54.

Last-query attention on OUR GQA is skewed: runtime T=128, top-20% mass
≈ 0.72, Gini ≈ 0.67, 16/16 layers hold vs a uniform null. Capture hold
prompts (T ≈ 50–496, 54 maps) hold_frac ≈ 0.89; the long code prompts
(T=345/419/496) still show top-20% ≈ 0.67–0.72.

Recipe at 32K×4 (keep recent 16 + 20% heavy-hitters of the prefix):
keep_frac ≈ 0.200, **booked 12.79 GiB** of GQA KV. This is the recipe
evaluated at 32K, not the measured-T keep fraction.

**Risk (the gate):** a needle / tool / schema token is often not a heavy
hitter until queried. Measured T is hundreds, not 32K. DeltaNet
rec_state cannot be token-evicted at all. Capability cost ABSENT.

### 2. DeltaNet rec_state (§56) — HAS the redundancy (low-rank, not shareable)

Final rec_state after the real prefix: mean head rank-99 ≈ 13.4 / 128,
participation ratio ≈ 2.08. Heads are not copies (pairwise cosine ≈
0.008). Adjacent layers are not similar (scale_aware ≈ 0). int4
per-head rel_l2 ≈ 0.49 (not cheap). f16 rel_l2 ≈ 2e-4 (rounding).

The redundancy is **per-head low rank**, not MiniCache-style depth merge
and not KIVI-style quant. Booked recipe: rank-32 f16 factors, **476.6 MiB
at c=4** (2.7% of 32K×4 session state). Cannot move the N016 crossover.
Prefix-sharing remains false. Zeroing this organ lost more function than
zeroing GQA (organ census 0.856 vs 0.607). Capability cost ABSENT.

### 3. KIVI (asymmetric K vs V) — does NOT

Paper: Liu et al., arXiv:2402.02750. S026 §51-52.

K **does** prefer per-channel grouped absmax (16/16 layers, runtime and
capture, int2 and int4). V **does not** prefer per-token (0/16). V also
prefers per-channel; symmetric-per-channel beats the KIVI recipe (int4
rel_l2 0.088 vs 0.100). The paper's asymmetry is absent. Bytes not
booked. A symmetric per-channel KV codec is a different method and is
unmeasured at capability.

### 4. MiniCache (depth merge) — does NOT

Paper: Liu et al., arXiv:2405.14366. S026 §53.

Adjacent GQA KV cosine ≈ 0.007 (K) / 0.003 (V); merge rel_l2 ≈ 0.70;
far-apart pairs are equally dissimilar. Capture corroborates (cosine
≈ 0). OUR full-attention layers are every 4th layer — three DeltaNet
layers sit between every "adjacent" GQA pair. MiniCache's dense
full-attention adjacency is not this architecture. Bytes not booked.

---

## What this does not claim

- It does not claim H2O or rank-32 DeltaNet state is production-safe.
- It does not transfer the GQA *weight* floor (4.125) onto KV-cache
  quant. Those are different operators (NOETIC_GQA_DESIGN,
  ORGAN_FRONTIERS, PREFILL_KV kv_precision).
- It does not treat NOETIC_DELTANET_DESIGN's "0 of 7 static tensors
  duplicate S" as a rec_state result. §56 asked whether S itself is
  redundant; the answer is yes, low-rank, not cross-layer.

Reopen a booked recipe only with a long-context capability suite on the
wired kernel. Reopen MiniCache/KIVI only if a new parent shows adjacent
KV cosine ≳ 0.90 (scale-aware) or V preferring per-token at matched rate.
