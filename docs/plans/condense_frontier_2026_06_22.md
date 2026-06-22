# Condense Frontier (2026-06-22)

This is the post-finalization expansion lane for Hawking's Model Press. The
core product question is no longer only "can Hawking run a compressed model?"
It is:

> Can Hawking quantize and condense models on machines that could not have
> quantized those models before?

That is the condensation mentality: pull a frontier-scale parent through a
narrow local memory aperture, compress it aggressively, verify the artifact, and
leave behind something the user's machine can actually own.

## Trigger: GLM-class open weights

Current external signal, checked on 2026-06-22:

- Z.ai's GLM-5.2 announcement describes a 1M-token context model with MIT
  open-source licensing, public weights on Hugging Face and ModelScope, and
  local serving support through transformers, vLLM, SGLang, xLLM, and
  ktransformers:
  https://huggingface.co/blog/zai-org/glm-52-blog
- The GLM-5 technical paper states the GLM-5 backbone is a 744B total /
  40B-active MoE:
  https://arxiv.org/html/2602.15763v2
- Together's GLM-5.2 model page also lists GLM-5.2 as 744B total /
  40B active and MIT open weights:
  https://www.together.ai/models/glm-52

Do not hard-code GLM-5.2 as the only target. Treat it as the current proof that
open weights are now large enough that "download and quantize locally" has
become a serious product bottleneck. The same lane should apply to future
GLM/Kimi/DeepSeek/Qwen/Llama-class MoE or dense parents.

## Product thesis

Most local runtimes compete at inference time. Hawking can compete one step
earlier: at artifact creation time.

If a user cannot hold the fp16/fp8 parent in RAM or VRAM, then ordinary
post-hoc quantization is blocked before inference even begins. Hawking Press
should become a memory-budgeted condenser:

```text
frontier/open parent -> streamed analysis -> shardwise quant/QAT/KD ->
verified Hawking artifact -> runnable local SKU
```

The winning claim is not "we have a 2-bit format." The winning claim is "we can
produce a verified 4/3/2/1-bit or distilled artifact under a declared local
memory budget, with quality evidence, from a parent that did not fit."

This should be treated as a general Condense pass across the whole program, not
only a GLM-specific experiment. Every compression lane should ask:

- can Hawking lower the peak memory needed to create the artifact versus common
  quantization tools;
- can Hawking reach lower bpw at the same quality, or higher retained quality at
  the same bpw, than ordinary post-hoc quantization;
- can Hawking recover the low-bit quality gap with QAT, KD, activation
  correction, output-damage-ranked bit allocation, and selective protection;
- can Hawking turn a parent model that was previously impractical to quantize
  on the target machine into a reproducible local artifact;
- can Hawking record enough evidence that the result is a product claim, not a
  one-off research run.

The competitive wedge is the combination of **out-of-core creation** and
**compress-then-recover quality**. GGUF/IQ/GPTQ/AWQ/EXL2-style baselines are
important comparison points where supported, but Hawking should not merely clone
their workflow. The target is a press that plans around the user's machine,
streams the parent, chooses the bit ladder from measured output damage, and
optionally distills or QAT-recovers the artifact before publication.

## Condensation doctrine

Every Model Press feature should be judged against these rules:

1. **Budget first:** the user declares RAM, disk, scratch, and optional cloud
   limits before the bake starts. The press must plan to the budget, not crash
   into it.
2. **Stream the parent:** read shard-by-shard and tensor-by-tensor. Never
   require the full parent to be resident unless the user asks for that mode.
3. **Emit early artifacts:** write quantized sections as soon as they verify,
   with resumable manifests and per-section checksums.
4. **Rank damage in output space:** use KL/NLL, heldout loss, or task deltas for
   bit allocation. Weight MSE and rel-RMS are scouts, not final gates.
5. **Treat 4/3/2/1-bit as a ladder:** 4-bit is the compatibility floor, 3-bit
   is the first shipping extreme tier, 2-bit requires recovery, and 1-bit /
   ternary is a retraining or native-low-bit lane.
6. **Recover, do not merely shrink:** combine quantization with QAT, KD,
   activation de-bias, outlier channels, and selective protection when the raw
   PTQ floor is unacceptable.
7. **Verify every claim:** output artifacts must carry base hash, recipe,
   memory budget, calibration data, quality card, and exact commands.
8. **Separate speed from density:** a sub-4-bit tier may be slower to decode but
   still valuable if it makes an impossible model fit. Advertise the correct
   axis.

## Compression ladder

| Tier | Role | Required recovery path | Ship gate |
|---|---|---|---|
| 4-bit | Compatibility and stable local inference | PTQ + AWQ/imatrix calibration | reference-quality delta, warm speed, footprint |
| 3-bit | First extreme public tier | Condense/TQ + per-tensor/rung allocation + light QAT/KD if needed | perplexity/task card vs 4-bit and parent |
| 2-bit | Recovery tier | PV/QAT in the loop, KD, outlier channels, selective protection | beats raw PTQ floor by a large measured margin |
| 1-bit / ternary | Research/native-low-bit tier | BitNet-style retraining or architecture-aware distillation | no launch claim until quality is stable |
| sub-2-bit vector trellis | Moonshot | re-learning only; PTQ floor is expected to fail | research report, not product default |

Naming decision: the public name for this line should be **Condense**. Existing
code still uses `strand` in crate names, scripts, env vars, and wire-format
helpers; treat that as legacy/internal naming until a compatibility-safe
migration lands. See `docs/plans/condense_naming_migration_2026_06_22.md`.

Existing Hawking/Condense assets to reuse:

- `vendor/strand-quant/src/bin/quantize-model.rs` for 1/2/3/4-bit Condense/TQ
  quantization and bpw sidecars.
- `tools/strand/scripts/rung-screen.py` for per-rung screening.
- `tools/strand/scripts/rung-kl.py` for output-space damage ranking.
- `tools/strand/scripts/strand-qat.py` for Condense-in-the-loop QAT/KD.
- `tools/strand/scripts/pv-recipe.sh` for the canonical 2-bit PV recipe.
- `tools/strand/scripts/bake-attested.sh` for verified 2-bit archive bake and
  attestation.
- `tools/strand/scripts/strand-act2-night3.sh` for 3-bit, 2-bit, ternary, and
  sub-2-bit operating points and crash-resume discipline.

## Technical work packages

### C1 - Memory-budgeted press planner

Add a planner that inspects model metadata, shards, tensor shapes, available
RAM, scratch disk, and target bit ladder, then emits a bake plan:

- max resident tensor/window size,
- required scratch bytes,
- thread count and GPU/CPU encode choice,
- shard ordering,
- resumability checkpoints,
- expected output bpw and disk bytes,
- unsupported architecture or license blockers.

Done means `hawking press --dry-run --memory-budget 18gb <model>` tells the
truth before work starts.

**STATUS: ✅ MVP LANDED (2026-06-22).** `hawking press --dry-run --memory-budget <SIZE> --target <BITS> --weights <gguf>`
is implemented (`crates/hawking/src/main.rs` `press_main`). It reads GGUF metadata ONLY (no weights resident, no GPU, no
network) and prints a truthful Press Plan: arch, tensor/param counts, weight bytes + current bpw, largest tensor, **peak
CREATION memory for out-of-core (tensor-at-a-time) vs full-resident-f32**, the Condense ladder (4/3/2/1-bit) output sizes +
ratios, and a budget fit verdict that names the **wedge**. MEASURED truthful on local models:
- Qwen2.5-3B-Q4_K_M: 3.09B params, 434 tensors, largest `token_embd` 311.2M elems (f32 1.16 GiB) → out-of-core peak **1.32
  GiB** vs full-resident **11.50 GiB** (~9×). At `--memory-budget 2gb`: out-of-core **FITS**, full-resident **EXCEEDS** →
  `WEDGE` (the "press a parent that can't fit fully resident" claim, demonstrated).
- RWKV-7-0.4B-SFT: 450.9M params, out-of-core 292 MiB vs full-resident 1.68 GiB (~6×).
- Truth check: planner weight-bytes (1.924 GB) vs file `stat` (1.930 GB) differ only by the ~6 MB GGUF header (excluded).
**safetensors (fp16/bf16 PARENT) support — ✅ LANDED (2026-06-22).** The planner now reads GGUF *or* safetensors metadata
(`read_inventory` auto-detects: GGUF magic vs `.safetensors`; safetensors via the 8-byte LE header length + JSON, no weights
read). MEASURED on local HF parents (no download): `rwkv7-g1-04-hf/model.safetensors` (BF16×795, 450.8M params, ~16 bpw) →
out-of-core 292 MiB vs full-resident 1.68 GiB; at `--memory-budget 1gb` → WEDGE (out-of-core FITS, full-resident EXCEEDS);
4/3/2-bit ladder shows 3.56×/5.33×/8.00× vs the bf16 parent. `mamba2-370m-hf` (F16×434) also planned. Truth check: weight
bytes (901,535,744) vs file `stat` (901,620,328) differ by exactly the 84,584-byte header. GGUF path unchanged (regression
OK; Qwen-3B now also reports its dtype mix `F32×181, Q4_K×216, Q6_K×37`).
Gates: `press_tests` GREEN — parse_size_arg, parse_tier_arg, **safetensors_header_inventory_metadata_only** (synthetic header);
build + clippy clean for this code (a PRE-EXISTING `hawking-core/json_constrain.rs` dead-code warning is unrelated).
Run: `cargo test -p hawking --bins press_tests`.

**REMAINING for C1 (not yet built):** (a) **multi-shard safetensors** — large/frontier HF parents split across
`model-00001-of-000NN.safetensors` + `model.safetensors.index.json`; today the reader handles a single `.safetensors` file
(so a GLM-class parent needs index-aware aggregation). (b) scratch-disk bytes, thread/encode choice, resumability
checkpoints, and license/arch blockers are not yet emitted. Then C2 (out-of-core writer). The bake itself stays owner-gated.

### C2 - Out-of-core tensor pipeline

Make the press stream one tensor or tensor window at a time:

- read safetensors/GGUF shard section,
- transform/calibrate/quantize,
- write verified section,
- drop resident buffers,
- append manifest row,
- resume from the last verified row.

Done means a model larger than RAM can be pressed with bounded peak RSS and a
crash can resume without redoing completed tensors.

### C3 - Damage-ranked bit allocation

Unify scout and quality signals:

- fast scout: rel-RMS, scale entropy, outlier mass, activation means;
- output-space screen: KL/NLL/task deltas on representative windows;
- allocator: reverse water-filling over 4/3/2/1-bit rungs;
- protected sets: attention/lm_head/outlier-heavy tensors stay high-bit unless
  evidence allows downgrade.

Done means the bit plan is not naive "everything 2-bit"; it is a measured
allocation with a budget and a quality bound.

### C4 - Condense-then-recover loop

Integrate QAT/KD into the press:

- teacher logits or top-k targets can stream from a remote/API teacher or a
  local parent when possible;
- compressed student trains with the target quantizer in the loop;
- periodic re-quant verifies the deployable artifact, not a proxy shadow;
- best checkpoint is selected by heldout quality, not last step.

Done means a 2-bit/ternary artifact can recover from the raw PTQ floor with a
quality card that shows the delta.

### C5 - Frontier MoE support

For GLM-class MoE parents, the press must understand:

- sharded expert tensors,
- active-parameter versus total-parameter accounting,
- router/gate tensors that should rarely be aggressively quantized first,
- per-expert streaming and optional expert-level parallelism,
- long-context/KV metadata needed for quality cards.

Done means the planner can dry-run a GLM-class model and name what is
supported, blocked, or requires reference tooling before downloading terabytes.

### C6 - Artifact proof

Every output needs:

- base model id, license, and hash;
- exact source shards and byte ranges;
- quant ladder and per-tensor rung map;
- QAT/KD recipe, data hashes, teacher identity, and eval settings;
- peak RSS and scratch usage measured during the bake;
- quality card and known failure classes;
- compatibility: Hawking runtime path plus at least one reference path when
  possible.

Done means a third party can verify the artifact and understand the trade.

## Phased execution

1. **After finalization gates:** finish P1 runtime, regression gates, and the
   basic P2 bake backend first. Do not let the frontier lane destabilize ship.
2. **Pilot on small:** prove the planner and out-of-core writer on Qwen-0.5B /
   3B where validation is fast.
3. **Scale to 7B/14B:** run memory-budgeted 4/3/2-bit bakes and compare against
   existing GGUF/IQ baselines at iso-bpw.
4. **MoE dry-run:** planner-only pass on a GLM-class model metadata/shard set;
   no full download unless owner approves storage/bandwidth.
5. **Frontier bake:** shardwise press of a massive open-weight parent under a
   declared budget, with quality and license gates.

## Owner approvals

Ask before:

- downloading frontier-scale weights,
- using paid cloud or large remote storage,
- publishing GLM-class derivatives,
- choosing final public claims about a model family,
- advertising 2-bit/1-bit quality before evals prove it.

## Final success statement

Hawking's Condense pipeline succeeds when a user with limited local hardware
can run:

```bash
hawking press zai-org/GLM-5.2 \
  --target tq3 \
  --memory-budget 64gb \
  --scratch-budget 2tb \
  --quality-card \
  --resume
```

and receive a verified, resumable, quality-carded artifact instead of an OOM.
That is the sector opportunity.
