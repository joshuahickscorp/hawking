# Hawking Gravity: maximal-fidelity ladder (as close to raw models as the box allows)

2026-07-20. Plan for testing 235B -> 397B -> 685B -> 1T at the highest fidelity this hardware permits,
before implementing. Grounded in measured local storage + live HF download sizes.

## 1. The reality: the GB figure is DISK, not RAM

Two separate limits, and they are always confused:

- **RAM (compute residency): 96 GB unified.** NOTHING on this ladder fits in RAM - not even the 61 GB
  gpt-oss-120b. So every model is tested by STREAMING active weights from disk into RAM per token
  (byte-range read the top-k experts of each layer), exactly as `gptoss_moe_runtime.py` already does.
  RAM is never the thing a model "fits" in here.
- **Disk (source residency): 926 GiB total, ~517 GiB free now (~578 GiB if the 61 GB gpt-oss source is
  released).** This is what the "GiB" download figures are measured against.

So "does model X fit" almost always means DISK, and the answer decides whether we hold the whole raw
model locally (best) or stream it shard-by-shard from HF (workaround for the giants).

## 2. Measured download (disk) sizes vs the ~578 GiB disk budget

| rung | model | download GiB | shards | fits on disk (578 free)? | strategy |
|---|---|---|---|---|---|
| F1 | Qwen3-235B-A22B | 437.9 | 118 | YES (~140 GiB headroom) | FULL disk-resident |
| F4 | Kimi-K2.6 (1T, int4) | 554.3 | 64 | barely (~24 GiB) | full-if-clean, else shard-serial |
| F3 | DeepSeek-V3.2 (685B, fp8) | 642.2 | 163 | NO (over by ~64 GiB) | shard-serial two-pass |
| F2 | Qwen3.5-397B (bf16) | 751.4 | 94 | NO (over by ~173 GiB) | shard-serial two-pass |

Key surprise (why size is not monotonic with params): precision. 397B ships bf16 (751), 685B fp8
(642), 1T int4 (554). The quantized giants are SMALLER on disk than the bf16 397B.

Your read is correct: **235B fits on disk** (barely without releasing gpt-oss, comfortably after), so it
does NOT need the delete-as-you-go workaround. The giants (397B, 685B) genuinely exceed disk and DO
need it. 1T is a coin-flip.

## 3. What "as close to raw as possible" actually means

The workarounds (disk-streaming, shard-serial, sub-bit) do NOT reduce the fidelity of the TEST. Testing
the real weights by streaming them from disk IS testing the raw model - that is exactly what the
gpt-oss campaign did (real mxfp4 weights, real forward, coherent " Paris", whole 36 layers). "Raw
fidelity" is three properties, and we want all three at the largest scale each model allows:

1. **Real weights** (the actual published tensors, not a proxy or a subsample).
2. **Real forward** (true logits / perplexity / next-token, not a synthetic approximation).
3. **Whole-model coverage** (every layer, every expert - not a slice).

The ONLY real limiter is not the download - it is whether we have the **compute engine** (a real forward
+ per-expert packer) for that model's architecture. Without it, streaming shards just downloads bytes;
with it, every streamed expert gets a genuine compression test.

## 4. The method that gives giants full-model fidelity despite not fitting: two-pass

For a model that does not fit on disk, you cannot hold the raw model - but the COMPRESSED artifact is
tiny (sub-bit -> tens of GB). So:

- **Pass 1 (stream + compress, shard-serial):** stream each shard from HF at the immutable revision ->
  decode its raw experts -> run Gravity/Doctor compression on them -> measure per-expert reconstruction
  + functional error on the RAW weights -> append to the small compressed artifact on disk -> release
  the raw shard -> next. After pass 1 you have measured raw-weight compression on the ENTIRE model and a
  complete compressed artifact resident.
- **Pass 2 (real forward on the compressed artifact):** the compressed model is small enough to stream
  from disk cheaply -> run a real end-to-end forward for logits / PPL / capability. The RAW-parent
  end-to-end reference is the one bounded piece (the raw model is gone): we get it by streaming the raw
  active experts per token from HF byte-ranges (slow but real) on a small holdout, or fall back to the
  per-expert error as the raw proxy. This is the only place giants are strictly below a disk-resident
  model, and it is a bounded, honest gap.

For F1 (235B, disk-resident) there is no gap: raw forward and compressed forward both stream from disk;
full raw reference + full compressed capability = maximal.

## 5. The reusable engine (the real build)

A generalized MoE streaming-compression engine, specialized per architecture:

```
architecture adapter        tensor names + geometry + source-precision decode (mxfp4 / bf16 / fp8 / int4)
per-expert streaming loader  byte-range read one expert at a time (bounded RAM), PressureAwareCache
real forward                 chain all layers, stream active experts per token -> real logits
per-expert packer            Gravity families + Doctor (product_quant, protected_islands, doctor_lowrank)
capability metrics           logit KL / cosine / top-k / next-token agreement / PPL vs raw parent
byte ledger                  exact whole-artifact BPW accounting
```

We already have this for gpt-oss (`gptoss_real_forward.py` + `gptoss_moe_runtime.py` + `gravity_forge`
packers). Each new rung is a NEW adapter over the SAME engine:

- F1 Qwen3-MoE: bf16, 94 layers, 128 experts top-8, GQA 64/4, standard MoE. Closest to gpt-oss - the
  engine port is mostly tensor-name + geometry changes.
- F2 Qwen3.5: hybrid 45x linear-attention (Gated DeltaNet) + 15x full-attention, 512 experts top-10 +
  shared, multimodal (text core only). Needs a DeltaNet forward path - the biggest new piece.
- F3 DeepSeek-V3.2: MLA (kv_lora) + fp8 block-quant decode + 256 routed top-8 + 1 shared.
- F4 Kimi-K2.6: MLA + 384 experts top-8 + 1 shared, int4 compressed-tensors decode.

## 6. Implementation order (do this, in this order, before anything else)

1. **Build the F1 Qwen3-MoE compute engine** (real forward + per-expert packer) - the Qwen analog of
   `gptoss_real_forward.py`. This is the concrete next build and turns the full 235B sweep into a real
   test, not a download. It also proves the engine-port pattern for the giants.
2. **Wire it into the overnight supervisor's Qwen phase** as FULL disk-resident coverage for 235B
   (release gpt-oss -> download full 438 GiB -> stream-to-RAM per-expert real forward + Gravity/Doctor
   compression + capability sweep over all 94 layers / 128 experts).
3. **Generalize to shard-serial two-pass** for the giants + write each architecture adapter (Qwen3.5
   DeltaNet, DeepSeek MLA/fp8, Kimi MLA/int4), reusing the same engine + packer + metrics.

The overnight chain, the running Doctor campaign, and the byte-budget RAM protection all keep running
while this is built. Storage discipline stays the one-parent law: one raw source (or one bounded shard
window) at a time; the small compressed artifacts + evidence are always retained.
