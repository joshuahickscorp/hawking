# STRAND Item 4 — a deterministic KV-cache quantization codec

_Authored 2026-06-13. Scope + design only. No code was built or run for this doc
(the local box is hosting a live PV run, PID 28690, ~166 MB free — every claim
here is from reading the source, and every measurement is specified as a **pod
step**, never a local one). This is the design for the surface §14 of
`docs/STRAND-quality-density-frontier.md` banked as "OSCAR-style frozen
KV/activation codec ... a whole new axis for post-project."_

Companion docs read for grounding:
- `docs/STRAND-quality-density-frontier.md` (the moat + RHT/trellis machinery; the
  §17 even-power-of-2 RHT limit is load-bearing here).
- `docs/STRAND-swarm-inference-design.md` (block-independent decode; the
  `decode_block_q12` single-block primitive).
- `crates/strand-quant/src/{decode.rs,rht.rs,trellis.rs,codebook.rs,encode.rs}`
  (the actual integer decode path).

Bottom line up front: **conditional GO for a prototype, NO-GO for shipping it as
a moat product until two genuinely hard problems are settled** — (1) the RHT
bit-identity limit lands exactly on the KV head geometry (head_dim 128 is an odd
power of two), and (2) KV is streaming/asymmetric/outlier-on-specific-channels in
ways weight quant is not. The opportunity is real and under-served; the honest
read is that the *decode* moat transfers cleanly and the *encode-side rotation
determinism* is where this can quietly break.

---

## 1. The opportunity — KV is the unsolved axis, and nobody offers it deterministically

Everyone quantizes weights. Weight quant is a crowded frontier (QTIP, QuIP#,
AQLM, ParetoQ, and STRAND's own q2 lane). The growing, under-served bottleneck is
the **KV cache**, and it has a structural reason to matter more every quarter:

- Weight memory is **fixed** in the model size. KV memory is **linear in context
  length × batch** and is now the dominant runtime allocation for long-context
  and high-concurrency serving. A 7B-class model (Qwen2.5-7B: 28 layers, 4 KV
  heads, head_dim 128) at fp16 spends `2 (K+V) × 28 × 4 × 128 × 2 B = 917 KB per
  token`. At 128k context that is **~117 GB of KV for a single sequence** — far
  past the model weights. The bottleneck STRAND's weight lane optimizes is no
  longer the thing that fills the GPU at long context.
- The state of practice is crude. Apple's own on-device model (the most detailed
  current primary source, arXiv 2507.13575, captured in
  `research/intel/apple-quant-research-raw.json`) ships **8-bit KV** — a flat
  per-tensor scalar grid, no rotation, no codebook, no incoherence processing.
  The common open-source baselines (KIVI, llama.cpp's `--cache-type-k/v q8_0`,
  q4) are block-scalar. **No one ships a trellis/incoherence-class KV codec, and
  critically no one ships a KV codec that is bit-identical across devices.**
- STRAND's whole differentiator is determinism. For weights, determinism is a
  trust/attestation story. For KV, determinism is also an **operational
  correctness** story: disaggregated / prefill-decode-split / multi-node serving
  (the `STRAND-swarm-inference-design.md` direction) requires that a KV block
  written by the prefill node decodes byte-identically on the decode node.
  Non-deterministic KV codecs cannot make that guarantee; a frozen-LUT integer
  KV decode can. **This is the one claim no competitor can make**, and it is the
  reason to do KV in STRAND specifically rather than adopt KIVI.

The product framing, consistent with the §9 black-hole doctrine: minimize
**accuracy loss-tax at fixed KV-bpw**, and sell it as "denser AND deterministic."
The metric is the same shape as the weight lane (`ln(PPL_kv / PPL_fp16-KV)`),
just with the KV cache as the quantized object and weights held at bf16.

---

## 2. How STRAND's float-free integer-LUT decode extends to KV — reusable vs new

The good news is that the **moat primitive is already the right shape for KV.**
The weight decode in `crates/strand-quant/src/decode.rs` is:

```
reconstruct_q(scale_q, q) = (scale_q as i64 * q as i64) >> 16     // SCALE_SHIFT=16
```

where `q = lut[state]` is a frozen symmetric Gaussian-quantile Q12 value
(`codebook.rs`, `QUANTILE_SHIFT=12`) and `state = ((state<<k)|sym) & mask` walks a
trellis (`trellis.rs::next_state`). It is **pure integer, sequential, and
per-element**. KV decode wants exactly that: produce one dequantized scalar at a
time, in order, from a bitstream plus a tiny per-block scale. The element-at-a-
time `out.push(reconstruct_q(es, q) + off)` loop in `decode_tensor_fixed_with_lut`
*is* a streaming decoder already; it just currently runs over a whole weight
tensor.

### Reusable as-is (the moat carries over for free)

| component | file | why it transfers to KV |
|---|---|---|
| Integer Q12 LUT decode | `decode.rs::reconstruct_q`, `eff_scale_q`, `eff_min_q` | Bit-exact, float-free, already the dequant kernel. A KV value decodes the same way a weight does. |
| Frozen Gaussian-quantile codebook | `codebook.rs`, `lut_tables.rs` (FROZEN, golden-hashed) | The decode-side table never changes per tensor; a KV stream reuses the identical LUT. Determinism golden (52M-case byte-stability, §17) covers it. |
| Trellis state machine | `trellis.rs::{next_state, num_states, state_mask}` | The `(state<<k)|sym` recurrence is content-agnostic. K and V streams are just different sources for the same machine. |
| Affine-min (asymmetric) decode | `decode.rs::eff_min_q` | **Directly needed** — KV (esp. post-RoPE K, and V) is asymmetric; the existing affine-min `out = reconstruct_q(es,q) + off` already supports a per-sub-block offset. This is not a new feature, it is an existing weight-lever that KV needs *more* than weights do. |
| Sub-block scaling | `encode.rs::{pack_sub_scales, SUB_BLOCK}`, `decode.rs::eff_scale_q` | 6-bit sub-scales give within-block dynamic range — exactly the per-token magnitude drift KV exhibits along the sequence. |
| Block-independent decode | `provenance.rs::decode_block_q12` (proven in `STRAND-swarm-inference-design.md`) | A KV block decodes from `(start_bit, out_off)` with zero cross-block data dependence (except init_state). This is the property that makes a **ring-buffer / paged** KV layout decodable. |

### New, but small (the streaming wrapper)

| new piece | what it is | rough size |
|---|---|---|
| KV block layout | A `(head, token-group)` tiled section: per (layer, head) a sequence of fixed-token blocks, each with its own scale_q + sub_scales + optional affine-min + init_state. This is structurally `EncodedTensor::blocks` re-keyed by token, not by weight offset. | reuse `BlockMeta` |
| Online per-block encode | At decode-time of the LLM, each new chunk of tokens' K/V for a (layer,head) is quantized as it is produced. The encoder must run **forward-only, no global pass** (see §3). | new, bounded |
| Append/evict semantics | KV grows and (for sliding-window / eviction) shrinks. Blocks are append-only within a layer/head; eviction drops whole blocks. RSLT/SDSC framing is for a static archive — KV needs a *runtime* container, not the `.strand` on-disk format. | new container |
| Per-channel side-channel | Outlier KV channels (see §3) handled like `outlier_wire.rs` but keyed per (head, channel) rather than per top-|w| element. | adapt `outlier_wire` concept |

**The decode kernel is reused verbatim; KV is a new *container and encode
schedule* around the same integer dequant.** That is the whole reason this is a
STRAND-shaped product and not a from-scratch codec.

---

## 3. The hard differences from weight quant — be honest, this is where it bites

This is not "weights but smaller." Four properties of KV break assumptions the
weight lane is built on. These are the genuine engineering risks.

### 3.1 KV is dynamic / streaming — you cannot do a one-shot global RHT

The weight lane's incoherence step (`rht.rs`) runs an offline pass over a whole
static tensor: apply per-element signs, then a blocked FWHT, then quantize. KV
tokens **arrive one at a time during generation**. There is no global tensor to
rotate, and you cannot re-rotate already-cached tokens when token N+1 arrives.

What survives: RHT applied **per head_dim vector, per token, independently**. The
random Hadamard is a fixed (seeded) linear map on a `head_dim`-length vector; it
does not need cross-token information. So `rht_forward_inplace` over a single
token's `[head_dim]` K (or V) vector is well-defined and streaming-safe. The
incoherence benefit (Gaussianizing per-channel outliers into a flatter
distribution so the frozen Gaussian LUT fits) is *per-vector*, so it is preserved.

What is genuinely hard — **and this is the central NO-GO risk**:

> §17 proved the RHT round-trip is bit-exact **iff the effective block size h is an
> even power of two** ({1,4,16,64,256}). For odd-power widths (it names **896→128,
> 384→128, 200→8**) the transform is only **approximate (~1e-6)** because 1/√h is
> not a dyadic f32, so cross-device bit-identity rests on IEEE-754 f32 no-FMA
> *assertion*, not proof.

**head_dim 128 = 2^7 is an odd power of two.** A per-head RHT over a 128-vector
lands exactly on the documented approximate regime. So the moment we add a KV RHT
on the natural geometry (head_dim 128), the **encode-side rotation is no longer
provably bit-identical across devices** — it inherits the same ~1e-6 f32-identity
caveat the weight lane already has for 896-wide tensors. For weights this was
tolerable (encode runs once, on one machine). For KV, **encode runs on the serving
node every token**, possibly a *different* node than the one that later reads the
block (disaggregated serving). That is exactly the case where f32-no-FMA identity
across heterogeneous hardware is least safe.

Three honest options, in preference order:
1. **Decode-only moat, encode-tolerant (recommended first prototype).** Accept
   that the *encode-side RHT* is approximate ~1e-6 (same status the shipping
   weight lane already lives with for 896 tensors). The **decode** (integer Q12
   LUT) stays bit-exact everywhere — so a *given KV bitstream* decodes
   identically on every node. The non-determinism is only in *which* bitstream
   the encoder produces, and only if two different encoders quantize the same
   raw KV on different hardware. Single-node serving is unaffected. Disaggregated
   serving is safe **if the prefill node is the only encoder** and ships the
   bitstream (not raw KV) to decode nodes — which is the natural design anyway.
2. **256-align the head_dim** by padding each head's KV vector 128→256 with zeros
   before RHT (256 is an even power → bit-exact RHT). Costs 2× pre-quant vector
   length (not 2× stored bits — the padding zeros quantize to one symbol and
   entropy-code to ~nothing, but they do cost trellis steps). This **buys back the
   provable cross-device encode identity** at a compute cost. Gate it: is the
   padded-RHT quality ≥ unpadded, and is the 2× encode-step cost acceptable at
   token rate?
3. **Drop RHT for KV entirely**, rely on affine-min + sub-scales + outlier channel
   for dynamic range. Loses the Gaussianization (the frozen Gaussian LUT fits
   worse), but the whole pipeline becomes trivially bit-exact (integer-only, no
   f32 transform). This is the *safest moat* and the *weakest quality*; it is the
   honest fallback if the RHT-on-128 identity cannot be made clean.

The prototype must **measure option 1 vs 3** (and 2 if 1's quality is poor) — do
not assume RHT is worth its determinism cost on KV until the PPL delta is on the
table.

### 3.2 KV is asymmetric (RoPE-rotated K, ReLU-ish V) — symmetric Gaussian LUT is a worse fit

The weight lane's codebook is a **symmetric, antisymmetric-by-construction**
Gaussian quantile table (`codebook.rs`: `lut_is_monotone_and_symmetric`,
`not antisymmetric at s` test). Weights post-RHT are ~zero-mean Gaussian, so
symmetric is right. KV is not:
- **Keys** carry RoPE rotation; per-channel they are not zero-mean, and specific
  frequency channels have structured magnitude.
- **Values** are post-attention-projection activations; they inherit activation
  asymmetry (heavy positive/negative skew per channel).

The existing **affine-min lever (`eff_min_q`) is the answer and it already exists**
— it adds a per-sub-block signed offset, turning the symmetric grid into an
asymmetric one. KV will lean on affine-min *by default* (weights use it
optionally). This is reuse, not new code, but it means the KV bit budget carries
the affine-min side-info (6 bits/sub-block) as a baseline cost, not an option.
Honest consequence: **KV's effective bpw floor is higher than the weight lane's**
for the same L/k, because asymmetry is not optional here.

### 3.3 KV is outlier-heavy on *specific channels/heads* — and the structure is different from weights

STRAND's weight outlier channel (`outlier_wire.rs`, `--outlier-channel`) isolates
the **top-|w| individual weights, element-wise, flattened** (verified in the
frontier doc §9.4: "STRAND outliers are top-|w| *individual weights* ... NOT
channel outliers"). KV outliers are the **opposite structure**: it is
well-established (KIVI, massive-activations literature) that KV outliers
concentrate in **fixed channels** (per head_dim index) and **specific heads/tokens
(attention sinks)**. This is a structural mismatch:

- The weight outlier machinery is element-indexed; KV wants **per-channel**
  isolation (a whole head_dim index kept at high precision across all tokens).
- That is actually *easier* to code than the weight case: a per-(layer,head)
  bitmap of "protected channels" (a few of 128) stored once, not per token, with
  those channels carried at 8-bit (or skipped from quantization). The per-token
  amortization is excellent because the channel set is stable over the sequence.
- **Attention-sink tokens** (first few tokens, and periodic high-norm tokens) may
  warrant whole-token high precision. This is a KV-specific lever with no weight
  analogue.

New work: a per-channel (not per-element) outlier path, and an optional
sink-token full-precision path. The *wire/seal discipline* from `outlier_wire.rs`
(step-over magic sets, digest sealing — see the §15 integration-risk ledger)
transfers, but only if KV ever lands in the sealed `.strand` container; for a
pure runtime KV cache it is a simpler in-memory side array.

### 3.4 KV decode is on the **serving hot path**, not a one-time load

Weight decode happens once at model load (or block-on-demand for swarm). KV
decode happens **every attention step for every cached token**. The
`decode_tensor_fixed_with_lut` loop is sequential per block; for attention we need
the full K/V of all cached tokens each step. Two honest performance facts:
- The Apple-Silicon hardware-profiling warning in the intel file
  (arXiv 2508.08531, captured in `apple-quant-research-raw.json`) flags that
  **trellis/codebook decode is measured slower than block-scalar schemes at very
  low bpw on Apple GPUs**, and becomes compute-bound. KV decode is more
  decode-bound than weights (you re-decode the whole cache every step unless you
  cache the dequantized values, which defeats the memory saving). **This is a real
  risk that the codec is memory-cheaper but throughput-poorer than q8 KV.**
- Mitigation is the same as the swarm lane: the `fold` fast-path in
  `decode_lean_with_lut` (precompute `folded[sb*ns + state]` per sub-block) and
  GPU decode (`metal.rs` gpu_q12 path) already exist. But KV decode batched over
  many tokens × heads is a *new* kernel shape; the existing kernels are tensor-
  shaped. **A KV decode kernel must be written and benchmarked against q8 KV
  throughput before any ship claim.**

---

## 4. Concrete minimal first-prototype scope

Deliberately small, single-model, single-question: **does a STRAND-class KV codec
beat q8/q4 KV on accuracy-at-fixed-KV-bpw, with decode bit-exact?** Everything
heavy is a **pod step** (the local box is off-limits per the safety constraint).

### Model and harness
- **Model:** Qwen2.5-0.5B (the established local hypothesis lab — but **run on the
  pod**, not locally, because of PID 28690). 24 layers, head_dim 64.
  - **Deliberate choice: head_dim 64 = 2^6 is an EVEN power of two → RHT is
    bit-exact (§17).** This sidesteps the 3.1 risk for the *first* prototype, so
    we measure the codec's quality ceiling before we pay the head_dim-128
    determinism tax. The 128-head_dim risk is then a *named follow-up* on a 7B
    model, not a confound in the first number.
  - Weights held at **bf16** (un-quantized). We are isolating KV-quant damage; do
    not stack it with weight-quant damage in the first measurement.
- **Eval:** WikiText-2 perplexity (the frontier-doc canon; `PPL_bf16 ≈ 12.536`
  with bf16 weights), **plus** at least one long-context accuracy probe where KV
  precision actually bites — a needle-in-a-haystack / passkey-retrieval at
  4k–16k, because PPL on short windows under-weights KV damage (KV error
  compounds with distance).

### What to build (prototype, minimal)
1. A KV-quant **eval shim** (Python, like `strand-debias-ppl.py`): hook the
   attention KV path, on each forward quantize K and V through the STRAND encoder
   and dequantize before attention. Parity-gate the dequant against the Rust
   integer decode (the contamination tell from the frontier doc: identical PPL to
   many decimals = the path was not actually used).
2. Three codec arms behind one flag:
   - `kv_strand_noRHT` (option 3.1-#3): affine-min + sub-scales + per-channel
     outlier, **no RHT**. The safe-moat arm.
   - `kv_strand_RHT` (option 3.1-#1): add per-token per-head RHT. On head_dim 64
     this is bit-exact, so it is also the *clean* arm here.
   - (defer the padded-256 arm to the 7B/head_dim-128 follow-up.)
3. Per-channel outlier isolation (a few of 64 channels at 8-bit), gated.

### What to measure (the decision table)
Report each arm with the frontier-doc schema (model, recipe, **KV-bpw with all
side-info billed**, PPL, loss-tax, long-context accuracy, contamination guard):

| arm | KV-bpw target | baseline to beat |
|---|---|---|
| fp16 KV | 16 | (anchor; PPL_bf16) |
| q8 KV (scalar) | 8 | the Apple/llama.cpp baseline |
| q4 KV (scalar) | 4 | the aggressive baseline |
| `kv_strand_noRHT` | ~3–4 | must beat q4 at ≤ its bpw |
| `kv_strand_RHT` | ~3–4 | must beat noRHT to justify RHT's determinism tax |

**Decisive output:** the Pareto frontier of long-context accuracy vs KV-bpw, and
whether STRAND lands a point **below q4 KV-bpw at q8 KV accuracy** — i.e. the same
"denser AND deterministic" claim the weight lane makes, but on the KV axis. Plus
a one-line determinism statement: decode bit-exact (yes everywhere), encode
bit-exact (yes on head_dim 64; **named-open on head_dim 128**).

---

## 5. GO / NO-GO, with the determinism-moat invariants

### The moat invariants that MUST hold (non-negotiable, from §17 + the swarm doc)
1. **Decode is integer-only and bit-identical across devices.** A given KV
   bitstream + the frozen LUT must decode byte-identically on CPU/GPU/any node.
   This is *preserved* — KV reuses `reconstruct_q` / the frozen golden LUT
   verbatim. ✔ holds by construction.
2. **The frozen codebook LUT is the one already golden-hashed.** Do not introduce
   a KV-specific decode table that escapes the 52M-case byte-stability golden. Use
   the existing symmetric Gaussian LUT + affine-min for asymmetry, **not** a new
   asymmetric LUT. ✔ holds if §3.2 is done via affine-min.
3. **Encode-side rotation identity.** This is the one that does **not** hold for
   head_dim 128 (odd power, §3.1). On head_dim 64 it holds; on 128 it is the
   documented ~1e-6 f32-no-FMA regime. ✖ open on the real target geometry.
4. **If KV ever enters the sealed `.strand` container**, it must follow the §15
   integration-risk ledger: new section magic taught to *every* walker, sealed in
   `descriptor_digest` + `verify_archive`, SDSC bumped, RSLT outermost. For a
   pure *runtime* KV cache this does not apply (no on-disk archive) — but a
   "checkpoint the KV cache to disk deterministically" feature would trigger all
   of it. Flag, don't build, until needed.

### Verdict

**GO — to a pod prototype on Qwen2.5-0.5B (head_dim 64), scoped to §4.** The
opportunity is real and genuinely under-served (no deterministic KV codec exists;
Apple ships flat 8-bit; the bottleneck grows with context). The moat's decode
primitive transfers for free, and head_dim 64 sidesteps the rotation-identity
risk so the first number is clean. This is cheap to falsify: one eval shim, three
arms, one model, on the pod.

**NO-GO — to shipping it as a determinism-moat KV product** until two things are
settled, in order:
1. **The head_dim-128 encode identity question** (§3.1 / invariant 3). Either
   prove option 1 (encode-on-one-node) is the right serving model, or measure
   that padded-256 RHT (option 2) is quality-neutral and rate-affordable, or
   accept the no-RHT fallback (option 3) and its quality cost. Until one of these
   is chosen *with numbers*, "deterministic KV codec" is **only true for
   even-power head dims** — which excludes most production 7B+ models (head_dim
   128). Saying otherwise repeats the §12 vanity-number mistake the frontier doc
   was red-teamed for.
2. **The decode-throughput question** (§3.4). KV decode is on the hot path and the
   intel file explicitly flags trellis decode as slower-than-block-scalar at low
   bpw on Apple GPUs. A KV decode kernel must be benchmarked against q8 KV
   throughput. If it is memory-cheaper but materially slower per token, the honest
   product is "deterministic + dense KV for memory-bound long-context, accepting
   a throughput cost," not a free win.

### What is genuinely hard (the honest list, not hand-waved)
- **head_dim 128 is an odd power of two.** The single most load-bearing fact: the
  natural KV geometry of every production model lands on STRAND's *approximate*
  RHT regime, so the rotation step — the thing that makes the Gaussian LUT
  fit — is the thing that costs the cross-device encode guarantee. This is not a
  bug to fix; it is a property of the dyadic-f32 FWHT scale. It forces a real
  architectural choice (encode-on-one-node vs pad-to-256 vs no-RHT), each with a
  cost.
- **Online encode at token rate.** The weight encoder does an expensive Viterbi
  search (`encode.rs`, the `_search` variants) offline. KV must encode forward,
  per chunk, *during generation* — the prototype must use the cheap forward
  encode, and the quality gap between cheap-online-encode and full-search-encode
  is unmeasured and could be large at 2–3 bits.
- **Re-decode cost vs memory saving.** Storing KV compressed only helps if you do
  not have to keep the dequantized copy resident; but attention needs all cached
  K/V each step, so either you re-decode every step (compute cost) or cache
  dequantized (no memory saving). The real win is in the *bandwidth* of streaming
  compressed KV from HBM, which is exactly the regime the Apple profiling paper
  warns is compute-bound at low bpw. This must be measured, not assumed.
- **Asymmetry raises the bpw floor.** Affine-min is mandatory (not optional as for
  weights), so KV's side-info baseline is heavier; the "3–4 bpw" targets above
  already assume that tax.

The principle and the decode moat survive cleanly. The honest risk is that the
*encode-side determinism* — STRAND's actual differentiator — is exactly the part
that the KV geometry (head_dim 128, per-token online rotation, multi-node encode)
stresses hardest. The prototype is designed to surface that as the first real
number rather than discover it after a ship claim.
