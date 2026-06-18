# Low-bit RWKV-7 Integration Prompt

**Date:** 2026-06-17  
**Branch:** rwkv7/multiseq-inspect  
**Status:** Scaffold+implementation complete, compile clean on both feature gates

---

## What is wired (do not re-do)

All Rust integration seams are real implementations, not stubs:

| File | What landed |
|---|---|
| `crates/dismantle-core/src/model/rwkv7.rs` | `ProjWeight::Tq` variant; `load_tq_artifact()`; TQ artifact discovery (`DISMANTLE_RWKV7_TQ=1` + `DISMANTLE_RWKV7_TQ_PATH`); post-build `try_replace_with_tq()` swap in all 6 per-layer projections + lm_head |
| `crates/dismantle-core/src/tq_gpu.rs` | `TqPreparedGpu::from_strand_tensor()` — real bake: `bake_bitslice_entries()` + `cfg.codebook()` Q12 LUT, RHT mode map |
| `crates/dismantle-core/src/kernels/mod.rs` | `strand_bitslice_gemv()` + `strand_bitslice_gemm()` — two-pass Metal dispatch (partials → reduce), batch variants b4/b16/b64 |
| `vendor/strand-quant/src/bin/quantize-model.rs` | `--bits 1` now accepted (assertion patched 2..=6 → 1..=6) |
| `tools/training/lowbit_qat.py` | Shared QAT lib: `quant_binary`, `quant_ternary`, `quant_uniform_symmetric`, `QuantLinear`, `wrap_linears`, `RWKV7_ALL_PROJ_SUFFIXES` |
| `tools/training/rwkv7_qat.py` | Full QAT trainer: all stages, all quant modes, KD, STRAND-PV requant loop (real impl with safetensors dump/reload) |
| `tools/training/rwkv7_eval_ppl.py` | Canonical PPL eval, JSONL ledger |
| `tools/training/rwkv7_export_strand.py` | QAT checkpoint → STR2/TQ artifact export |
| `tools/training/rwkv7_capture_teacher_logits.py` | Top-k logit cache for offline KD |

`cargo check -p dismantle-core --features tq` and `cargo check -p dismantle-core` both finish with zero errors.

---

## First: run the dry-run gates

Before using any compute:

```bash
cd /Users/scammermike/Downloads/dismantle

python tools/training/rwkv7_qat.py \
  --dry-run --quant ternary --stage ffn --max-rows 4

python tools/training/rwkv7_qat.py \
  --dry-run --quant binary --stage all --max-rows 4
```

Both should print wrapped module names (only ffn.key / ffn.value or all projections respectively), confirm gradients flow into shadow weights, confirm LoRA/norm/embed are excluded. If either fails, debug before training.

---

## First training run (G1 — FFN-only ternary scout)

```bash
# G1a: last 8 layers only, fastest validation of training mechanics
python tools/training/rwkv7_qat.py \
  --model models/rwkv7-g1-04-hf/model.safetensors \
  --hf-dir models/rwkv7-g1-04-hf \
  --data artifacts/rwkv7_posttrain/sft.jsonl \
  --out artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8 \
  --stage ffn \
  --quant ternary \
  --bits 2 \
  --last-n-layers 8 \
  --lr 5e-6 \
  --save-every 25 \
  --eval-every 25

# Eval the checkpoint
python tools/training/rwkv7_eval_ppl.py \
  --model artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8/best_state_dict.pt \
  --hf-dir models/rwkv7-g1-04-hf \
  --tokens 20000 \
  --out artifacts/lowbit_rwkv7/eval_ledger.jsonl
```

**Promote ladder:**
- PPL ≤ 1.20× Q4_K_M → run G1b (all 24 layers)
- PPL ≤ 1.35× (Silver) → export and gate Rust integration
- PPL > 1.50× → try uniform bits=2 or reduce LR before STRAND-PV

---

## Export trained checkpoint to TQ artifact

```bash
# Build quantize-model first (only needed once)
cargo build --release -p strand-quant --bin quantize-model

# Export
python tools/training/rwkv7_export_strand.py \
  --checkpoint artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8/best_state_dict.pt \
  --hf-dir models/rwkv7-g1-04-hf \
  --out artifacts/lowbit_rwkv7/export/g1_ffn_ternary \
  --bits 2 \
  --strand-bin target/release/quantize-model \
  --strand-flags "--tail-biting --affine-min auto --threads 8"

# Produces:
#   artifacts/lowbit_rwkv7/export/g1_ffn_ternary/rwkv7-lowbit.tq
#   artifacts/lowbit_rwkv7/export/g1_ffn_ternary/manifest.json
```

The TQ artifact contains only the overridden tensors (FFN key/value in G1). All other tensors stay Q4_K_M from the base GGUF — no full-model rebuild needed.

---

## Wire into serving pipeline (first smoke test)

```bash
# Build tq-enabled library
cargo build --release --features tq -p dismantle-core

# Loader shape/name gate (no GPU needed)
cargo test -p dismantle-core --features tq --test rwkv7_tq_loader -- --nocapture

# Trellis parity gate (CPU only, no GPU needed)
cargo test -p dismantle-core --features tq --test tq_trellis_parity -- --nocapture

# Full CPU+GPU parity gate (needs an actual TQ artifact)
DISMANTLE_RWKV7_TQ=1 \
DISMANTLE_RWKV7_TQ_PATH=artifacts/lowbit_rwkv7/export/g1_ffn_ternary/rwkv7-lowbit.tq \
cargo test -p dismantle-core --features tq --test rwkv7_tq_parity \
  -- --ignored --nocapture --test-threads=1

# Speed + memory bench
DISMANTLE_RWKV7_TQ=1 \
DISMANTLE_RWKV7_TQ_PATH=artifacts/lowbit_rwkv7/export/g1_ffn_ternary/rwkv7-lowbit.tq \
cargo test -p dismantle-core --features tq --test rwkv7_tq_bench \
  -- --ignored --nocapture --test-threads=1
```

**Verify:** existing `rwkv7_metal_parity` must stay green WITHOUT `DISMANTLE_RWKV7_TQ`:

```bash
cargo test -p dismantle-core --test rwkv7_metal_parity -- --nocapture
```

---

## Spec decode draft model swap

Once the **191M** RWKV model has a TQ artifact at Silver tier (PPL ≤ 1.35× its own Q4_K_M baseline):

1. Export 191M TQ artifact using the same pipeline (swap `--model` and `--hf-dir`).

2. The draft model in the spec decode path is a separate `Rwkv7` instance. Enable TQ by setting env vars before `Rwkv7::load()`. No changes to spec decode dispatch logic needed — the draft model just serves with lower memory footprint and faster per-token decode:

```rust
// In the draft model loading path (find via `grep -r "draft" crates/ --include="*.rs"`)
#[cfg(feature = "tq")]
{
    std::env::set_var("DISMANTLE_RWKV7_TQ", "1");
    std::env::set_var("DISMANTLE_RWKV7_TQ_PATH", &draft_tq_path);
}
let draft = Rwkv7::load(draft_gguf_path, batch_cfg)?;
#[cfg(feature = "tq")]
std::env::remove_var("DISMANTLE_RWKV7_TQ");
```

3. Measure: draft tps, resident memory delta, accept rate, accepted tokens/s, J/accepted-token. Report using the final shippable claim format in the plan at `docs/plans/low_bit_rwkv7_strengthened_revision_2026_06_18.md`.

---

## Open items requiring real artifacts to complete

| Item | File | Needed |
|---|---|---|
| Fill in fixture asserts in parity test | `crates/dismantle-core/tests/rwkv7_tq_parity.rs` | Live TQ artifact |
| Fill in bench result thresholds | `crates/dismantle-core/tests/rwkv7_tq_bench.rs` | Live TQ artifact + bench run |
| `RhtMode::Cols` wiring | `tq_gpu.rs` / `kernels/mod.rs` | Only needed if artifact has RHT enabled; MVP uses `RhtMode::None` |
| STRAND-PV requant sanity check | `rwkv7_qat.py` | Run once with `--quant strand-pv --bits 2 --requant-every 25` |

---

## Quality tiers reference

| Tier | Gate |
|---|---|
| Bronze | PPL ≤ 2.0× Q4_K_M |
| Silver | PPL ≤ 1.35× Q4_K_M or absolute ≤ 20 |
| Gold | PPL ≤ 1.15× Q4_K_M + greedy mostly stable |
| Platinum | Gold + measured tps/J win over Q4_K_M draft |

---

## Plan + memory references

- Full plan: `docs/plans/low_bit_rwkv7_strengthened_revision_2026_06_18.md`
- Consolidated plan: `/Users/scammermike/.claude/plans/vivid-singing-pie.md`
- Memory entry: `lowbit-rwkv7-plan-2026-06-18` in project memory
