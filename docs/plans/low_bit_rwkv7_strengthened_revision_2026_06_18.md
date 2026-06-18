# Low-bit Strengthening + 1-bit/Ternary RWKV-7 Plan - Strengthened Revision

Date: 2026-06-18

This is the no-expense, no-wall-clock-cap revision of the low-bit RWKV-7 plan. The objective is not merely "try 2-bit" or "wire ternary"; the objective is to build a repeatable low-bit training, export, and serving pipeline that can produce an actually shippable RWKV-7 draft/runtime artifact at 2-bit, 1-bit STRAND, or ternary-trained STRAND-1 density, with hard quality and determinism gates at every layer.

## Executive thesis

The repository already contains most of the hard runtime machinery:

- `tools/strand/scripts/strand-qat.py` already has STE fake quantizers, including ternary.
- `vendor/strand-quant/src/trellis.rs` already supports `k_bits >= 1`, and `TrellisConfig::for_bpw(1.0)` resolves to `k=1, L=5`.
- `crates/dismantle-core/src/tq.rs` already reads STR2/TQ artifacts as the CPU reference.
- `crates/dismantle-core/src/tq_gpu.rs` plus `crates/dismantle-core/src/kernels/mod.rs` already provide a Metal decode-only bitslice identity path.
- `crates/dismantle-core/src/model/rwkv7.rs` already has the right serving seam: `ProjWeight`, currently `Q4k | Q6k | F32`, used by every large RWKV projection.

The draft plan was directionally right, but it missed four load-bearing details:

1. Ternary BitNet training is not the same thing as STRAND `k_bits=1`. Ternary is a 3-state weight alphabet. STRAND-1 is a 1-bit stateful trellis stream with an `L`-bit codebook state. They can reinforce each other, but they are distinct deployment targets.
2. `vendor/strand-quant/src/bin/quantize-model.rs` currently rejects `--bits 1` even though the lower-level trellis supports it. The CLI and tests must be opened to 1-bit before any real STR2 artifact exists.
3. RWKV's current GPU serving path does not call a generic backend `WeightKind`; it uses the local `ProjWeight` abstraction inside `rwkv7.rs`. The runtime integration should extend `ProjWeight`, not only `backend/mod.rs`.
4. The dismantle-side TQ Metal path is currently decode-identity first. For real speed, RWKV needs a fused or prepared bitslice GEMV path ported from `vendor/strand-decode-kernel/src/metal.rs`, not a decode-to-Q12-on-every-token fallback.

The strengthened plan is therefore:

1. Build a real measurement harness.
2. Build RWKV-specific QAT with both fake-quant and real STRAND/PV modes.
3. Enable and test true `--bits 1` STR2 export.
4. Add RWKV TQ serving through `ProjWeight::Tq`.
5. Train a ladder: FFN-only, time-mix-only, full-projection, mixed-rung, then draft-model optimized.
6. Ship the best density/quality point, not a pet theory.

## Non-negotiable outcomes

This project is successful only when all three surfaces are green:

| Surface | Required outcome |
|---|---|
| Quality | Low-bit RWKV PPL and greedy behavior measured against fixed baselines, not vibes |
| Artifact | STR2/TQ artifact contains the intended tensors, bit depths, shapes, provenance, and source hash |
| Runtime | RWKV serves the low-bit artifact through `ProjWeight::Tq` with CPU/GPU parity and useful speed/memory wins |

Target tiers:

| Tier | Meaning | Quality gate |
|---|---|---|
| Bronze | Research-valid | coherent generations, no collapse, PPL <= 2.0x Q4_K_M baseline |
| Silver | Usable draft | PPL <= 1.35x Q4_K_M baseline or <= 20 if baseline eval differs |
| Gold | Shippable model | PPL <= 1.15x Q4_K_M baseline and greedy trajectory mostly stable |
| Platinum | Moat | Gold quality plus measured tps/J/token win over Q4_K_M draft serving |

Do not argue about "good enough" until the baselines exist.

## Ground truth from the current codebase

Important existing files and facts:

| File | Current fact |
|---|---|
| `tools/strand/scripts/strand-qat.py` | Has `quant_uniform`, `quant_ternary`, `QuantLinear`, and external STRAND/PV-style requant machinery, but it is transformer-oriented and imports `AutoModelForCausalLM`. |
| `tools/training/rwkv7_sft_torch.py` | Correct RWKV pure-torch training loop, prompt masking, `lm_loss`, `freeze_to_last_n`, MPS-safe flow. This is the right base for RWKV QAT. |
| `tools/training/rwkv7_torch_model.py` | RWKV projection names are `r_proj`, `k_proj`, `v_proj`, `o_proj`, plus channel-mix `key`, `value`; small LoRA matrices are `w1/w2`, `a1/a2`, `v1/v2`, `g1/g2`. |
| `tools/training/rwkv7_load_weights.py` | Maps HF safetensors into the pure torch model. Reuse this, do not load through `fla` for MPS training. |
| `tools/training/rwkv7_export_hf.py` | Provides the mapping back toward HF/GGUF names. Reuse its naming logic for export. |
| `vendor/strand-quant/src/trellis.rs` | Trellis accepts `k_bits=1`, but only through lower-level config. |
| `vendor/strand-quant/src/bin/quantize-model.rs` | CLI help and assertion currently say bits are `2..=6`; this must be changed for 1-bit. |
| `crates/dismantle-core/src/tq.rs` | CPU reference for STR2/TQ decode and matvec. The internal `StrandTensor` has private `enc`, so GPU-prepared serving needs a small accessor or prepared wrapper. |
| `crates/dismantle-core/src/tq_gpu.rs` | Host-side bitslice entry bake plus decode-only GPU identity. Good base for prepared TQ tensor support. |
| `crates/dismantle-core/src/kernels/mod.rs` | Contains `decode_strand_bitslice`; missing production `matvec`/`gemm` driver for `strand_bitslice_gemv_partials` + `strand_bitslice_reduce_rows`. |
| `crates/dismantle-core/src/model/rwkv7.rs` | RWKV GPU uses `ProjWeight::build()` and `ProjWeight::gemv()` for every large projection. This is the integration seam. |
| `crates/dismantle-core/tests/rwkv7_metal_parity.rs` | Existing CPU/GPU RWKV parity gate. Extend this for TQ. |
| `crates/dismantle-core/tests/tq_trellis_parity.rs` | Existing GPU bitslice decode identity gate. Extend or parallel it for k=1. |

## Correct terminology

Use these terms precisely:

- `STRAND-1`: STR2/TQ artifact with `k_bits=1`; usually `L=5` via `for_bpw(1.0)`, or explicit `L` sweeps such as `L=7`, `L=9`, `L=11`.
- `STRAND-2`: STR2/TQ artifact with `k_bits=2`; default `L=6`, quality `L=8`, or explicit sweeps.
- `binary STE`: training fake quantizer with weight alphabet `{-s, +s}`.
- `ternary STE`: training fake quantizer with weight alphabet `{-s, 0, +s}` using per-row scale.
- `ternary deployment`: literal 3-state packed format and kernel. This does not exist today.
- `ternary-trained STRAND-1`: train with ternary pressure, then export through STRAND `k=1`. This is deployable without a new ternary kernel once `ProjWeight::Tq` exists.

Default deployment target should be STRAND-1/2, not literal ternary, because STRAND already owns the determinism and archive machinery. Literal ternary becomes a separate optional kernel track only if it beats STRAND-1 in quality/speed enough to justify new format surface.

## Architecture-specific quantization map

RWKV-7 large matrices to attack first:

| Torch module | GGUF/TQ name | Shape on 0.4B | Initial policy |
|---|---|---:|---|
| `layers.N.attn.r_proj.weight` | `blk.N.time_mix_receptance.weight` | `n x n` | protect longer; sensitive gate |
| `layers.N.attn.k_proj.weight` | `blk.N.time_mix_key.weight` | `n x n` | quantize after FFN passes |
| `layers.N.attn.v_proj.weight` | `blk.N.time_mix_value.weight` | `n x n` | quantize after FFN passes |
| `layers.N.attn.o_proj.weight` | `blk.N.time_mix_output.weight` | `n x n` | quantize after FFN passes |
| `layers.N.ffn.key.weight` | `blk.N.channel_mix_key.weight` | `n_ff x n` | first low-bit target |
| `layers.N.ffn.value.weight` | `blk.N.channel_mix_value.weight` | `n x n_ff` | first low-bit target, but watch sensitivity after ReLU^2 |
| `lm_head.weight` | `output.weight` | `vocab x n` | protect until late; try STRAND-2 before STRAND-1 |

Keep full precision or high precision by default:

- Embedding table.
- LayerNorm weights and biases.
- Token-shift lerp vectors.
- WKV vectors `k_k`, `k_a`, `r_k`.
- LoRA micro-matrices `w1/w2`, `a1/a2`, `v1/v2`, `g1/g2` until a later ablation proves they tolerate low bit.
- Any tensor whose reduction dimension is not compatible with the bitslice kernels.

## Workstream A - Measurement and baselines

Create `tools/training/rwkv7_eval_ppl.py`.

Requirements:

- Load with `rwkv7_load_weights.load_rwkv7()` for safetensors checkpoints.
- Load GGUF through dismantle/Rust only where required, but keep the training eval Python-native first.
- Use the RWKV World tokenizer from `models/rwkv7-g1-04-hf`.
- Support fixed text corpora:
  - `--corpus wikitext2`
  - `--corpus wikitext103-small`
  - `--corpus artifacts/rwkv7_posttrain/heldout.jsonl`
  - `--text-file <path>`
- Report:
  - token count
  - NLL
  - PPL
  - top-1 agreement vs teacher if `--teacher-logits` is provided
  - argmax trajectory match on committed fixture prompts
- Write JSONL to `artifacts/lowbit_rwkv7/eval_ledger.jsonl`.

Baseline matrix:

| Model/artifact | Metric set |
|---|---|
| HF fp32/bf16 safetensors | PPL, fixture greedy, logit stats |
| shipped RWKV Q4_K_M GGUF | PPL if available, runtime tps/J/token |
| STRAND-3 PTQ from fp32 checkpoint | PPL, relative RMS, artifact bpw |
| STRAND-2 PTQ from fp32 checkpoint | confirm collapse or update prior |
| STRAND-1 PTQ from fp32 checkpoint | expected collapse, but measure it |
| FFN-only STRAND-1/2 | isolate sensitivity |
| time-mix-only STRAND-1/2 | isolate sensitivity |

Add `artifacts/lowbit_rwkv7/baseline.md` with a compact table. This file becomes the gate ledger.

Hard rule: no training result is considered meaningful unless it is compared to the same eval protocol and token count.

## Workstream B - Shared QAT library refactor

Do not make `rwkv7_qat.py` import `strand-qat.py` as a giant CLI. Extract reusable pieces.

Add `tools/training/lowbit_qat.py`:

- `QuantLinear`
- `quant_uniform_symmetric(w, bits)`
- `quant_binary(w)`
- `quant_ternary(w)`
- `wrap_linears(model, suffixes, quant_fn, bits, include_regex=None, exclude_regex=None)`
- `list_wrapped_modules(model)`
- `freeze_by_module_policy(model, policy)`
- digest helpers from `strand-qat.py`

Then update `tools/strand/scripts/strand-qat.py` to import the shared pieces while preserving its current CLI behavior for transformer models.

Important quantizer correction:

- Do not use existing `quant_uniform(w, bits=1)` for binary training. The current formula gives `qmax=0`, so it degenerates toward `{-1, 0}` and is biased.
- Add explicit binary STE:

```python
def quant_binary(w, bits=None):
    s = w.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    q = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    wq = q * s
    return w + (wq - w).detach()
```

Keep ternary STE:

```python
def quant_ternary(w, bits=None):
    s = w.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    q = torch.clamp(torch.round(w / s), -1, 1)
    wq = q * s
    return w + (wq - w).detach()
```

Add dry-run tests:

- `python tools/training/rwkv7_qat.py --dry-run --quant ternary --stage ffn --max-rows 4`
- Assert exactly the expected module names were wrapped.
- Assert gradients flow into wrapped shadow weights.
- Assert excluded LoRA/norm/embed tensors were not wrapped.

## Workstream C - RWKV QAT trainer

Add `tools/training/rwkv7_qat.py`.

Base it on `rwkv7_sft_torch.py`, not on Hugging Face `AutoModelForCausalLM`.

CLI shape:

```text
python tools/training/rwkv7_qat.py \
  --model models/rwkv7-g1-04-hf/model.safetensors \
  --hf-dir models/rwkv7-g1-04-hf \
  --data artifacts/rwkv7_posttrain/sft.jsonl \
  --out artifacts/lowbit_rwkv7/runs/<run_id> \
  --stage ffn|time|all|mixed|lmhead \
  --quant uniform|binary|ternary|strand-pv \
  --bits 1|2|3 \
  --last-n-layers 0|8|12|24 \
  --max-length 1024 \
  --grad-accum 16 \
  --lr 5e-6 \
  --teacher models/rwkv7-g1-04-hf/model.safetensors \
  --kd topk|full|none \
  --kd-temperature 2.0 \
  --ce-weight 0.3 \
  --kd-weight 0.7 \
  --save-every 25 \
  --eval-every 25
```

Stages:

| Stage | Wrapped modules | Purpose |
|---|---|---|
| `ffn` | `ffn.key`, `ffn.value` | Fastest proof that low-bit can survive in RWKV bulk weights |
| `time` | `attn.r_proj/k_proj/v_proj/o_proj` | Time-mix sensitivity isolation |
| `all` | FFN plus time-mix main projections | Real target |
| `mixed` | configurable per-name bit/rung map | Escape hatch for quality |
| `lmhead` | output head only or plus all | Late-stage density experiment |

Training modes:

| Mode | Meaning |
|---|---|
| `uniform --bits 2` | cheap QAT proxy for STRAND-2 resilience |
| `binary --bits 1` | pressure test for 1-bit sign weights |
| `ternary` | BitNet-style 3-state pressure, best candidate for ternary-trained STRAND-1 |
| `strand-pv --bits 1/2` | real deployment quantizer in the forward path, refreshed by Rust encoder |

Distillation:

- First pass: CE-only on prompt-masked instruct data to validate mechanics.
- Second pass: teacher KD.
- For no-expense mode, support full-vocab KL.
- For practical repeated sweeps, add `tools/training/rwkv7_capture_teacher_logits.py` that stores top-k teacher logits per supervised position:
  - `top_k=128` for normal runs.
  - `top_k=512` for final runs.
  - store token id, label, top ids, top logits, teacher entropy.
- KD loss:
  - `loss = ce_weight * CE(labels) + kd_weight * KL(student_topk, teacher_topk)`
  - temperature `2.0` default, sweep `1.0, 2.0, 4.0`.
- Optional hidden-state MSE on final hidden for stability if logit KD is noisy.

Optimizer defaults:

- AdamW, `betas=(0.9, 0.95)`, `weight_decay=0`.
- LR ladder: `1e-5` for FFN-only, `5e-6` for full, `2e-6` for `strand-pv`.
- Grad clip `1.0`.
- Save both latest and best-by-eval.
- Always write `run_config.json`, `module_wrap.json`, `eval_ledger.jsonl`, and `state_dict.pt`.

Escalation ladder:

1. Last 8 layers only.
2. Last 12 layers.
3. Full 24 layers with low LR.
4. Full 24 layers plus KD.
5. Full 24 layers plus real STRAND/PV forward.
6. Mixed-rung protect the worst tensors.

## Workstream D - Real STRAND/PV training

Fake quant is only a scouting tool. The final shippable path must train against the real reconstruction distribution.

Add RWKV support for the existing external STRAND/PV mechanism from `strand-qat.py`:

- Dump only wrapped RWKV projection shadows to safetensors using GGUF/TQ names.
- Run `vendor/strand-quant`'s `quantize-model` periodically.
- Reload reconstruction into `QuantLinear.base` so forward uses `base + weight`.
- Use selective PV: after init, re-encode only trainable tensors.
- Use sharded requant jobs; no single giant encode bottleneck.

PV run shape:

```text
python tools/training/rwkv7_qat.py \
  --quant strand-pv \
  --bits 2 \
  --stage ffn \
  --requant-every 25 \
  --requant-shards 8 \
  --strand-bin target/release/quantize-model \
  --strand-flags "--tail-biting --affine-min auto --threads 8" \
  --eval-every 25
```

For 1-bit:

```text
python tools/training/rwkv7_qat.py \
  --quant strand-pv \
  --bits 1 \
  --stage ffn \
  --strand-flags "--l 7 --tail-biting --affine-min off --threads 8" \
  --requant-every 10
```

Reason for explicit `--l`: `for_bpw(1.0)` gives `L=5`, which may be too weak. Sweep `L=5,7,9,11` and bill the effective bpw honestly.

PV gates:

- Requant does not change frozen tensor digests.
- Reconstruct file has exactly expected tensor names and shapes.
- PPL after a PV refresh is not dramatically worse than the fake-quant eval before refresh.
- If fake quant looks good but PV collapses, the fake quant path is demoted to scouting only.

## Workstream E - Enable true 1-bit STR2 export

Patch `vendor/strand-quant/src/bin/quantize-model.rs`:

- Help text: `--bits <1|2|3|4|5|6>`.
- Assertion: allow `1..=6`.
- Audit any assumptions that `bits >= 2`.
- Ensure `resolve_cfg(1)` gives expected `k=1`, `L=5` unless `--l` is explicit.
- Add a regression test for a tiny safetensors input with `--bits 1 --packed-v2-out`.

Do not patch `vendor/strand-quant/src/trellis.rs` unless a test proves it is wrong. It already clamps `k_bits` to at least 1.

Add tests:

- `vendor/strand-quant/tests/quantize_model_bits1.rs`
  - create a tiny 2D safetensors tensor.
  - run quantize-model or call equivalent library path with `bits=1`.
  - read STR2 header.
  - assert every tensor has `k_bits == 1`.
  - decode and verify shape/total.
- Extend `crates/dismantle-core/tests/tq_trellis_parity.rs`:
  - include `TrellisConfig::for_bpw(1.0)`.
  - include explicit `TrellisConfig::for_bpw_l(1.0, 7)`.
  - include tail-biting and affine-min off/on where legal.

Export script:

Add `tools/training/rwkv7_export_strand.py`:

- Input: QAT `state_dict.pt`.
- Output:
  - `artifacts/lowbit_rwkv7/export/<run_id>/rwkv7-lowbit-proj.safetensors`
  - `artifacts/lowbit_rwkv7/export/<run_id>/rwkv7-lowbit.tq`
  - `manifest.json`
- Map torch names to GGUF names:
  - `layers.{i}.attn.r_proj.weight` -> `blk.{i}.time_mix_receptance.weight`
  - `layers.{i}.attn.k_proj.weight` -> `blk.{i}.time_mix_key.weight`
  - `layers.{i}.attn.v_proj.weight` -> `blk.{i}.time_mix_value.weight`
  - `layers.{i}.attn.o_proj.weight` -> `blk.{i}.time_mix_output.weight`
  - `layers.{i}.ffn.key.weight` -> `blk.{i}.channel_mix_key.weight`
  - `layers.{i}.ffn.value.weight` -> `blk.{i}.channel_mix_value.weight`
  - optionally `lm_head.weight` -> `output.weight`
- Carry untouched full-precision tensors in the base GGUF; the TQ artifact only needs low-bit override tensors.

Quantize command examples:

```text
cargo build -p strand-quant --release --bin quantize-model

target/release/quantize-model \
  --in artifacts/lowbit_rwkv7/export/<run_id>/rwkv7-lowbit-proj.safetensors \
  --out artifacts/lowbit_rwkv7/export/<run_id>/recon.safetensors \
  --bits 1 \
  --l 7 \
  --tail-biting \
  --affine-min off \
  --packed-v2-out artifacts/lowbit_rwkv7/export/<run_id>/rwkv7-lowbit.tq \
  --threads 16
```

For mixed rung:

```json
[
  {"pattern": "channel_mix_key", "bits": 1},
  {"pattern": "channel_mix_value", "bits": 1},
  {"pattern": "time_mix_receptance", "bits": 2},
  {"pattern": "time_mix_key", "bits": 2},
  {"pattern": "time_mix_value", "bits": 2},
  {"pattern": "time_mix_output", "bits": 2},
  {"pattern": "output.weight", "bits": 3}
]
```

Use:

```text
target/release/quantize-model \
  --in .../rwkv7-lowbit-proj.safetensors \
  --out .../recon.safetensors \
  --bits 2 \
  --mp-config artifacts/lowbit_rwkv7/export/<run_id>/rung.json \
  --l 7 \
  --packed-v2-out .../rwkv7-lowbit.tq
```

## Workstream F - RWKV runtime integration

The runtime target is `crates/dismantle-core/src/model/rwkv7.rs`, specifically the `ProjWeight` enum and its `build`, `gemv`, and `gemv_batched` methods.

Add a TQ representation:

```rust
pub enum ProjWeight {
    Q4k { ... },
    Q6k { ... },
    F32 { ... },
    #[cfg(feature = "tq")]
    Tq {
        prepared: TqPreparedGpu,
        rows: usize,
        cols: usize,
    },
}
```

Suggested support structs:

- `crates/dismantle-core/src/tq.rs`
  - expose safe read-only accessors for encoded payload/config, or add `StrandTensor::prepare_for_gpu()`.
- `crates/dismantle-core/src/tq_gpu.rs`
  - add `PreparedTqTensor` / `TqPreparedGpu`:
    - payload bytes
    - baked `BitsliceEntry` table
    - LUT
    - `k_bits`
    - `l_bits`
    - total
    - rows
    - cols
    - `RhtMode`
    - `rht_seed`
    - optional outlier channel status
- `crates/dismantle-core/src/kernels/mod.rs`
  - port and expose host drivers for:
    - `strand_bitslice_gemv_partials`
    - `strand_bitslice_reduce_rows`
    - optional batched path from `strand_bitslice_gemm_partials_b4/b16/b64`
  - source of truth: `vendor/strand-decode-kernel/src/metal.rs`.

Minimum viable runtime:

1. CPU path: decode Q12 once on load and use `crate::tq::matvec_rht` for overridden RWKV projections. This proves loader and quality, but is not the final speed path.
2. Metal decode-on-load path: GPU decode Q12 once into a pinned i32/f16/f32 buffer, then call a normal GEMV. This reduces file size but not fully memory bandwidth per token.
3. Final Metal fused path: bitslice GEMV directly from STR2 payload every token. This is the target for the inference-space advantage.

Because the user explicitly does not care about dev time, build all three in order. Each stage de-risks the next.

TQ artifact discovery:

- Mirror Qwen's `ensure_tq_cache()` behavior.
- Candidate paths:
  - `weights_path.with_extension("tq")`
  - `models/<stem>.tq`
  - explicit env `DISMANTLE_RWKV7_TQ_PATH`
- Guard with env:
  - `DISMANTLE_RWKV7_TQ=1` to enable.
  - absent artifact means inert fallback unless strict env is set.
  - `DISMANTLE_RWKV7_TQ_STRICT=1` means missing/mismatched artifact is an error.

Tensor matching:

- Parse the TQ artifact once.
- Index `StrandTensor` by exact GGUF name.
- In `ProjWeight::build`, before native GGUF Q4/Q6 handling, check whether an override exists for `name`.
- Validate `(st.out_features, st.in_features) == (rows, cols)`.
- Validate `cols % 256 == 0` for fused bitslice GEMV.
- Validate `RhtMode::None` first for MVP. Add `RhtMode::Cols` only after activation transform is explicitly wired and tested.

Runtime gates:

- `cargo test -p dismantle-core --features tq --test tq_trellis_parity -- --nocapture`
- New `cargo test -p dismantle-core --features tq --test rwkv7_tq_loader -- --nocapture`
- New `cargo test -p dismantle-core --features tq --test rwkv7_tq_parity -- --ignored --nocapture --test-threads=1`
- Existing `rwkv7_metal_parity` must remain green without `DISMANTLE_RWKV7_TQ`.
- New parity must compare:
  - CPU f32/Q4 path vs TQ CPU decoded path
  - TQ CPU decoded path vs TQ Metal fused path
  - greedy token trajectory over committed fixtures

## Workstream G - Training ladder

Run the ladder as a set of parallel experiments, not a single serial bet.

### G0 - Baselines

- fp32/bf16 HF PPL.
- Q4_K_M GGUF PPL and speed.
- STRAND-3 PTQ.
- STRAND-2 PTQ.
- STRAND-1 PTQ with `L=5,7,9`.
- Record effective bpw, relative RMS, and PPL.

### G1 - FFN-only low-bit

Purpose: determine whether RWKV bulk FFN can carry the density win.

Runs:

- FFN key/value ternary STE.
- FFN key/value binary STE.
- FFN key/value uniform 2-bit.
- FFN key/value STRAND-PV 2-bit.
- FFN key/value STRAND-PV 1-bit with `L=5,7,9`.

Promote if:

- PPL <= 1.20x Q4_K_M or <= 12 on the canonical eval.
- Greedy fixture argmax drift is acceptable.
- Exported STR2 artifact decodes and serves in CPU TQ path.

### G2 - Time-mix main projections

Purpose: locate the sensitive RWKV recurrent projections.

Runs:

- Quantize each of `r`, `k`, `v`, `o` alone at STRAND-2.
- Quantize pairs: `k/v`, `r/o`, `k/v/o`, all four.
- Repeat the best pair at STRAND-1 with `L` sweep.

Expected sensitivity:

- `r_proj` may be most sensitive because it reads the recurrent state.
- `k_proj` affects WKV state update and l2-normalized key path.
- `v_proj` affects both state update and value-residual carry.
- `o_proj` is likely more forgiving than `r/k/v`, but must be measured.

Promote only the projections that survive.

### G3 - Full projection low-bit

Candidate policies:

| Policy | FFN | time `o` | time `k/v` | time `r` | lm head |
|---|---:|---:|---:|---:|---:|
| Aggressive STRAND-1 | 1 | 1 | 1 | 1 | 2/3 |
| Protected receptance | 1 | 1 | 1/2 | 2 | 3 |
| Balanced STRAND-2 | 2 | 2 | 2 | 2 | 3 |
| FFN moat | 1 | native Q4 | native Q4 | native Q4 | native Q6 |
| Draft-first | 1/2 | 1/2 | 2 | 2 | 3 |

Use `--mp-config` for export and `--stage mixed` for QAT.

### G4 - Real STRAND/PV finalization

Take the best fake-quant policies and re-run them with `--quant strand-pv`.

This is where most proxy winners will die. That is fine. The artifact that survives PV is the artifact that matters.

Final quality gates:

- PPL <= Silver or Gold target.
- No catastrophic prompt family failures in `artifacts/rwkv7_posttrain/heldout.jsonl`.
- Greedy continuation remains coherent.
- Exported `.tq` exact tensor list matches policy manifest.
- Runtime TQ path matches CPU reference within declared tolerance.

### G5 - 191M draft specialization

Once 0.4B mechanics are proven, repeat on the 191M RWKV draft target.

Why:

- The spec decode draft model cares more about acceptance per byte than standalone quality.
- A ternary-trained STRAND-1 191M model may be the true inference moat even if the 0.4B full model lands at mixed STRAND-2.

Metrics:

- draft tps
- resident memory
- accept rate
- accepted tokens/s
- joules/accepted-token

Spec integration only happens after standalone quality is measured. Do not mask a broken draft behind acceptance heuristics.

## Workstream H - Literal ternary deployment, optional

Do this only if ternary STE consistently beats binary/STRAND-PV in quality.

Literal ternary is a separate runtime design:

- Format: pack 5 ternary weights per byte or use 2-bit symbols with one unused state.
- Per-row/per-block scale side info.
- Metal kernel: ternary decode plus GEMV.
- CPU reference: exact ternary matvec.
- Tests: byte-stable pack/unpack, CPU/GPU parity, RWKV parity.

This is not required to get a STRAND-1 advantage. The faster path is ternary-trained STRAND-1.

## File-level implementation plan

| File | Action |
|---|---|
| `tools/training/lowbit_qat.py` | New shared QAT wrappers and quantizers. |
| `tools/strand/scripts/strand-qat.py` | Import shared QAT helpers; preserve existing CLI. |
| `tools/training/rwkv7_qat.py` | New RWKV-specific QAT trainer. |
| `tools/training/rwkv7_capture_teacher_logits.py` | New optional KD cache builder. |
| `tools/training/rwkv7_eval_ppl.py` | New canonical RWKV PPL/eval script. |
| `tools/training/rwkv7_export_strand.py` | New QAT checkpoint to STR2/TQ exporter. |
| `vendor/strand-quant/src/bin/quantize-model.rs` | Allow `--bits 1`; update help and tests. |
| `vendor/strand-quant/tests/quantize_model_bits1.rs` | New 1-bit export regression. |
| `crates/dismantle-core/src/tq.rs` | Add safe accessors/prepared wrapper for GPU serving. |
| `crates/dismantle-core/src/tq_gpu.rs` | Add prepared tensor support for runtime. |
| `crates/dismantle-core/src/kernels/mod.rs` | Add bitslice GEMV/GEMM host drivers from vendor implementation. |
| `crates/dismantle-core/src/model/rwkv7.rs` | Add TQ artifact loader and `ProjWeight::Tq`. |
| `crates/dismantle-core/tests/tq_trellis_parity.rs` | Extend to k=1/L sweeps. |
| `crates/dismantle-core/tests/rwkv7_tq_loader.rs` | New loader/shape/name gate. |
| `crates/dismantle-core/tests/rwkv7_tq_parity.rs` | New CPU/GPU/trajectory gate. |
| `crates/dismantle-core/tests/rwkv7_tq_bench.rs` | New speed and memory gate. |
| `artifacts/lowbit_rwkv7/` | Runtime-generated ledgers and outputs; keep out of git unless summarized. |
| `docs/plans/low_bit_rwkv7_status.md` | Living result ledger, commit only curated summaries. |

## Verification plan

### Python/training gates

```text
python tools/training/rwkv7_parity_torch.py --device cpu
python tools/training/rwkv7_qat.py --dry-run --quant ternary --stage ffn --max-rows 4
python tools/training/rwkv7_qat.py --dry-run --quant binary --stage all --max-rows 4
python tools/training/rwkv7_eval_ppl.py --model models/rwkv7-g1-04-hf/model.safetensors --tokens 4096
```

### STRAND export gates

```text
cargo test -p strand-quant quantize_model_bits1 -- --nocapture
cargo run -p strand-quant --release --bin quantize-model -- \
  --in artifacts/lowbit_rwkv7/export/smoke/rwkv7-lowbit-proj.safetensors \
  --out artifacts/lowbit_rwkv7/export/smoke/recon.safetensors \
  --bits 1 \
  --l 7 \
  --packed-v2-out artifacts/lowbit_rwkv7/export/smoke/rwkv7-lowbit.tq
```

### Rust runtime gates

```text
cargo test -p dismantle-core --features tq --test tq_trellis_parity -- --nocapture
cargo test -p dismantle-core --features tq --test rwkv7_tq_loader -- --nocapture
DISMANTLE_RWKV7_TQ=1 cargo test -p dismantle-core --features tq --test rwkv7_tq_parity -- --ignored --nocapture --test-threads=1
DISMANTLE_RWKV7_TQ=1 cargo test -p dismantle-core --features tq --test rwkv7_tq_bench -- --ignored --nocapture --test-threads=1
```

### Inference gates

Measure every candidate:

- peak RSS
- model resident bytes
- token/s single stream
- token/s multi-stream
- joules/token
- spec decode accept rate if used as draft
- accepted tokens/s
- joules/accepted-token

Use paired runs against Q4_K_M. A candidate that wins size but loses accepted tokens/s is not the final moat.

## Risk register

| Risk | Why it matters | Mitigation |
|---|---|---|
| Fake quant looks good, STRAND export collapses | Proxy quantizers lie | Make STRAND/PV a required final training mode |
| Ternary is mistaken for k=1 deployment | Wrong artifact/kernel assumptions | Keep terminology separate; ship ternary-trained STRAND-1 first |
| `quantize-model --bits 1` hidden assumptions | CLI currently rejects it | Add 1-bit tests before training depends on it |
| RWKV `r/k/v` are too sensitive | Recurrent state can amplify errors | Mixed-rung policies and per-projection sensitivity ladder |
| LM head dominates quality loss | Huge vocab projection can be fragile | Keep native Q6/Q4 first; only lower after body survives |
| Runtime decodes Q12 every token | Would erase speed advantage | Port fused bitslice GEMV and benchmark it |
| TQ artifact name mismatch | Silent wrong tensor override would be disastrous | Exact name map, shape checks, strict mode |
| RHT/OUTL path not wired in fused runtime | Wrong math if artifact uses those modes | MVP uses `RhtMode::None`; enable `Cols` only with explicit tests |
| MPS training instability | RWKV pure torch recurrence is heavy | use grad checkpointing, small max length, deterministic saves, CPU fallback for smoke |
| Overfitting small eval | False quality confidence | use wikitext plus heldout instruct plus fixture trajectory |

## Decision tree

If full STRAND-1 passes Silver:

- Optimize fused runtime.
- Move to 191M draft.
- Run spec integration.

If full STRAND-1 fails but FFN-only STRAND-1 passes:

- Ship FFN-only low-bit body with native time-mix.
- Use mixed-rung for the draft model.
- Continue time-mix sensitivity work.

If STRAND-2 passes Gold:

- Ship STRAND-2 first. It is still a meaningful density win.
- Continue STRAND-1 as the moat track.

If ternary STE beats all fake/binary runs but STRAND-PV loses:

- Try ternary-trained STRAND-1 with larger `L`.
- If still strong in fake quant only, open literal ternary deployment track.

If all low-bit 0.4B runs fail:

- Pivot to 191M draft-only, where acceptance per byte may still win.
- Keep runtime work because `ProjWeight::Tq` is reusable.

## Initial parallel assignment

Lane 1 - Baseline and eval:

- Implement `rwkv7_eval_ppl.py`.
- Produce `baseline.md`.
- Run fp32, Q4_K_M, STRAND-3, STRAND-2, STRAND-1 PTQ.

Lane 2 - QAT trainer:

- Extract `lowbit_qat.py`.
- Implement `rwkv7_qat.py`.
- Dry-run all stage/quant combinations.

Lane 3 - STRAND 1-bit enablement:

- Patch `quantize-model.rs` for `--bits 1`.
- Add 1-bit STR2 regression.
- Extend TQ trellis parity to `k=1`.

Lane 4 - Runtime bridge:

- Add TQ artifact discovery for RWKV.
- Add CPU decoded `ProjWeight::Tq` first.
- Then port fused bitslice GEMV.

Lane 5 - Experiment runner:

- Launch G1 FFN-only runs.
- Launch G2 projection sensitivity runs.
- Maintain `eval_ledger.jsonl`.

## First concrete commands

Build the quantizer:

```text
cargo build -p strand-quant --release --bin quantize-model
```

Run existing RWKV correctness:

```text
python tools/training/rwkv7_parity_torch.py --device cpu
```

Create a quick baseline after `rwkv7_eval_ppl.py` lands:

```text
python tools/training/rwkv7_eval_ppl.py \
  --model models/rwkv7-g1-04-hf/model.safetensors \
  --hf-dir models/rwkv7-g1-04-hf \
  --corpus wikitext2 \
  --tokens 8192 \
  --out artifacts/lowbit_rwkv7/eval_ledger.jsonl
```

First QAT smoke:

```text
python tools/training/rwkv7_qat.py \
  --model models/rwkv7-g1-04-hf/model.safetensors \
  --hf-dir models/rwkv7-g1-04-hf \
  --data tools/training/data/rwkv7_sft_sample.jsonl \
  --out artifacts/lowbit_rwkv7/runs/smoke_ffn_ternary \
  --stage ffn \
  --quant ternary \
  --dry-run \
  --max-rows 8
```

First real training candidate:

```text
python tools/training/rwkv7_qat.py \
  --model models/rwkv7-g1-04-hf/model.safetensors \
  --hf-dir models/rwkv7-g1-04-hf \
  --data artifacts/rwkv7_posttrain/sft.jsonl \
  --out artifacts/lowbit_rwkv7/runs/ffn_ternary_full24_kd \
  --stage ffn \
  --quant ternary \
  --last-n-layers 0 \
  --lr 1e-5 \
  --epochs 1 \
  --kd topk \
  --eval-every 25 \
  --save-every 25
```

First PV candidate:

```text
python tools/training/rwkv7_qat.py \
  --model models/rwkv7-g1-04-hf/model.safetensors \
  --hf-dir models/rwkv7-g1-04-hf \
  --data artifacts/rwkv7_posttrain/sft.jsonl \
  --out artifacts/lowbit_rwkv7/runs/ffn_strand2_pv \
  --stage ffn \
  --quant strand-pv \
  --bits 2 \
  --last-n-layers 0 \
  --lr 2e-6 \
  --requant-every 25 \
  --requant-shards 8 \
  --eval-every 25
```

## Final shippable claim format

Do not claim victory with only a training loss screenshot. The final report must say:

```text
Artifact:
  base model:
  TQ artifact:
  tensors overridden:
  effective bpw:
  source hash:

Quality:
  eval corpus:
  tokens:
  Q4_K_M PPL:
  candidate PPL:
  ratio:
  fixture greedy match:

Runtime:
  mode:
  device:
  resident memory:
  single-stream tok/s:
  multistream tok/s:
  joules/token:
  accepted tokens/s if draft:

Correctness:
  CPU TQ reference:
  Metal TQ parity:
  RWKV greedy parity:
  determinism:
```

That report is the moat asset.
