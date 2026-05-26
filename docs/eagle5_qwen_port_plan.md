# Eagle5 spec-decode port to qwen_dense.rs

**Status:** the trained Eagle5 / Eagle6 heads from `colab/qwen3b_reconciliation.ipynb` are silently no-ops on Qwen-3B / Qwen-1.5B today. The `--speculate eagle5` CLI flag and `--eagle5-head <path>` arg parse correctly, populate `EngineConfig::eagle5_head_path`, but Qwen's dense forward loop never calls the Eagle head.

Discovered 2026-05-26 by the end-to-end bench in this session: `--speculate eagle5` at K=2/4/8 produced **identical dec_tps to baseline** with `draft_accepted=0 draft_rejected=0` — the head was inventory in RAM, not running.

## Proof
```
$ grep -c -iE "eagle5|speculate_mode|draft_accept" crates/dismantle-core/src/model/qwen_dense.rs
0
$ grep -c -iE "eagle5|speculate_mode|draft_accept" crates/dismantle-core/src/model/deepseek_v2.rs
54
```

All Eagle5 wire-up lives in `deepseek_v2.rs`. None of it is in `qwen_dense.rs`.

## Source-of-truth: what to port

### Load-time hook (deepseek_v2.rs L715-L750ish)
```rust
let eagle5_head: Option<crate::speculate::eagle5::Eagle5Head> = if speculate_mode
    == SpeculateMode::Eagle5
{
    let head = match config.eagle5_head_path.as_deref() {
        Some(path) => crate::speculate::eagle5::Eagle5Head::load_from_safetensors(path)?,
        None       => crate::speculate::eagle5::Eagle5Head::new_random_deterministic(...),
    };
    Some(head)
} else { None };
```
Goes next to the existing `q4k_predec_cache` / `q4k_fast_buf` lazy-loaders in `QwenDense`. Add a struct field `eagle5_head: Option<Eagle5Head>`.

### Pre-flight gate (deepseek_v2.rs L1049-L1068)
```rust
if self.speculate_mode == SpeculateMode::Eagle5 {
    if sampling.temperature != 0.0 { return Err(...); }
    if sampling.repetition_penalty != 1.0 { return Err(...); }
    if self.eagle5_head.is_none() { return Err("Eagle5 requested but no head loaded"); }
}
```
Goes at the top of `forward_token_greedy_tcb` or `generate`.

### Verify-then-draft dispatch (deepseek_v2.rs L1437-L1586)
This is the substantive 150-line block. Structure:

1. After computing target logits for the prefix, branch on `speculate_mode == Eagle5`
2. Use `self.eagle5_head.as_mut()` to draft K tokens
3. Run the target model on `[prefix + draft_tokens]` in one batched forward (or K sequential forwards)
4. Compare target argmax to draft tokens position-by-position; accept the longest prefix that matches
5. Increment `stats.draft_accepted` / `stats.draft_rejected`
6. The first rejected position becomes the next "target greedy" emit

Key sub-routines we reference but already exist:
- `self.eagle5_head.draft_tokens(ctx_ids, K)` — already implemented in `speculate/eagle5.rs`
- `Self::forward_logits_for_tokens(...)` — the deepseek_v2 path has it; qwen_dense will need a sibling that respects the dense layer structure (no MoE routing).

### Counter aggregation
- `EngineConfig::draft_accepted` / `EngineConfig::draft_rejected` already exist on the engine struct
- Increment from inside the qwen forward loop, same pattern as deepseek_v2.rs L1565-L1566

## Validation gates

1. **Build clean:** `cargo build --release --workspace` zero new errors / warnings.
2. **Existing tests pass:** `cargo test --workspace --lib` — 76 tests still green.
3. **Speculate=off equivalence:** with `--speculate off`, decoded tokens MUST be bit-identical to pre-port behavior (regression test).
4. **Speculate=eagle5 with mock head:** dec_tps roughly the same as baseline (mock head ≈ random; accept rate ≈ 1/vocab); `draft_accepted+draft_rejected > 0` proves the path is engaging.
5. **Speculate=eagle5 with trained head from notebook:** `draft_accepted / (draft_accepted+draft_rejected)` should be in the 30-60% range based on Colab's tau projection. Below 20% suggests draft/target tokenizer mismatch.
6. **End-to-end paired bench:** `tools/bench/eagle5_paired_bench.sh` with trained head should show `dec_tps(K=4) > dec_tps(K=0)` by some positive margin. Even a 10% delta is enough to prove the port works; tuning comes after.

## Files to touch
- `crates/dismantle-core/src/model/qwen_dense.rs` — add fields, hooks, dispatch branch
- `crates/dismantle-core/src/model/mod.rs` — no changes needed (dispatch already routes by arch)
- `crates/dismantle/src/main.rs` — no changes (kernel_profile / eagle5_head_path plumbing already exists)
- New test: `crates/dismantle-core/tests/qwen_eagle5_speculate_smoke.rs` — runs `--speculate eagle5` end-to-end on a small Qwen prompt, asserts `draft_accepted + draft_rejected > 0`

## Effort estimate
- **2 days minimum** for an experienced operator if `deepseek_v2.rs` spec-decode logic ports cleanly
- **4 days realistic** including: gate 1-3 working, then chasing whatever non-trivially differs between dense and MoE (likely: KV cache layout, layer-wise scratch, batched-prefill interaction)
- **NOT a refactor.** Don't try to factor a shared `speculate_mod` trait first — port concretely, then refactor if both paths look identical at the end.

## What's NOT in this plan
- Training improvements. The trained heads from the reconciliation notebook are ready inventory.
- Continuous batching / Track E. Separate effort.
- The qwen_dense AWQ Option B path (already shipped). It's behind env flags and unrelated; spec-decode work shouldn't touch it.

## Predecessor: confirm spec-decode runtime works on something else
Before committing 2-4 days to the port, **run `--speculate eagle5` end-to-end on DeepSeek-V2-Lite** and verify `draft_accepted+draft_rejected > 0` with a non-zero accept rate. If the deepseek path is also dead (broken by recent W4A8 / predec refactors), the port has nothing to copy from and we have a bigger debug job first.

```bash
# Build, then on a clean terminal:
EAGLE5_HEAD=<trained-deepseek-head-if-any.safetensors> \
WEIGHTS=models/deepseek-v2-lite-q4.gguf \
TRIALS=2 TOKENS=64 \
bash tools/bench/eagle5_paired_bench.sh
```
If acceptance is non-zero → port has a working source to copy from. If zero → debug deepseek_v2.rs spec-decode first.
