# Handoff — EAGLE head↔runtime forward parity (the on-device 0%-accept blocker)

> ## ✅✅ RESOLVED 2026-05-30 — Part 1 FIXED (`4384f68`), Part 2 = head generalization
> **Part 1 (wiring) SHIPPED + verified:** the residual/intermediate feed is now active
> in plain `--speculate eagle5` (gated on `eagle5_head.is_some()`/`use_eagle5`, not the
> corpus-dump flag), residuals cosine-1.0 vs the corpus capture path. On-device accept
> **0% → ~10.5%** (K=2), bit-identical no-spec, all tests green. The runtime spec path
> is no longer broken.
> **Part 2 (still open) = HEAD GENERALIZATION, not runtime:** the agent proved the feed
> is already correct, so the remaining gap is the head itself. It scores ~90% depth-0
> **in-sample** on its 619-seq training corpus but only **33% on a fresh held-out
> capture** (13.6% live) — i.e. it OVERFIT the small corpus. The in-scope teacher-forcing
> ceiling is 33%, which the runtime already feeds. So spec is still net-negative on tps
> (~16 vs ~35 no-spec) until the head GENERALIZES. **Fix = retrain on a bigger / more
> diverse capture corpus from the current build** (the 619-seq corpus was too small +
> the capture degenerated into repetition at ~620). That's the cloud/capture track, not
> a runtime change. Everything below is now historical.
>
> ## ✅ ROOT CAUSE CONFIRMED 2026-05-30 — it's NOT the forward, NOT the head
> The single-step forward parity **passes** for the retrained num_blocks=2 head
> (argmax 315=315, L2 identical, top-8 8/8). The 0%-accept is a **wiring bug**:
> the head's residual/intermediate feed is gated behind `DISMANTLE_QWEN_EAGLE5_CAPTURE`
> (`qwen_dense.rs:1727` — `captured_residual = if eagle5_capture_in_use {...} else None`,
> then `res_ref = …unwrap_or(&zeros)` at :1782). In normal spec decode that flag is
> off, so **the head is fed zeros → garbage drafts (`.Content`, `酤`, `Breadcrumb`) →
> 100% reject.** Proven: setting `DISMANTLE_QWEN_EAGLE5_CAPTURE=1` makes the residual
> real (std≈6.1) and accept goes **0 → 6/112**; drafts become code-like.
>
> **The fix has two parts:**
> 1. **Wiring (cheap):** populate the layer-32 residual+intermediate buffers and feed
>    them to the head when `use_eagle5` (spec mode), independent of the corpus-dump
>    flag — gate the buffer *population* + the `:1727`/`:1748` reads on
>    `(eagle5_capture_in_use || use_eagle5)`, and SKIP the expensive corpus quantize+
>    disk-write (`:1965+`). The capture flag is 3× slower (prefill 0.55→1.9 s) because
>    it dumps a corpus; spec decode only needs the buffer, which the GPU already computes.
> 2. **Representation gap:** even with real residuals, accept is only 5.4% vs the
>    PyTorch ~50%+ implied by τ=1.91. So the runtime's residual/intermediate still
>    don't fully match training — check the **intermediate** feed (is it populated or
>    zeros?), and the residual **scale/normalization** vs the dequantized int8
>    (`residual_q`×`residual_scale`) the τ-eval uses. Align them until on-device per-pos
>    accept tracks the PyTorch [0.84,0.54,0.33,0.21].
>
> Bench the fix with `tools/bench/paired_lever.sh` + `eagle5_paired_bench.sh` under the
> §1 gate; net win needs accept high enough to beat the ~31.9 no-spec dec_tps.
> Original parity-diagnostic notes below are now historical context.

Paste as the opening prompt for a fresh session that has (or can make) a **torch
env** — the definitive check needs PyTorch. Runs parallel to the M3 (Stage 2 +
capture) and cloud (training) tracks.

## The problem (measured 2026-05-30)
A retrained EAGLE head — trained on the **M3 Q4_K_M runtime capture corpus** —
scored **80% depth-1 accept / τ=1.33 in the cloud PyTorch τ-eval**, a huge jump
from the prior head's 0%. The distribution fix works. BUT on the **M3 runtime**
it gives **0.000 accept** (every draft rejected), so spec is pure overhead
(27 → 16/12/9 dec_tps at K=2/4/8). The head **loads** fine (`[eagle5] loading
trained head` — not the mock fallback). So good head in PyTorch, useless on-device
⇒ a **head↔runtime forward mismatch**.

## Root-cause diagnosis (done locally)
Compared the two heads' safetensors structure:
- **New head (1.56 GB, 16 tensors):** `block.*` only → **num_blocks=1**.
- **Old head (1.83 GB, 25 tensors):** `block.*` + `extra_blocks.0.*` → **num_blocks=2**.

The runtime (`crates/dismantle-core/src/speculate/eagle5.rs:313-332`) *infers/reads*
`num_blocks`, so it doesn't reject a 1-block head — but the **only structure the
runtime forward is parity-verified against is the 2-block one** (the existing
fixture `tests/fixtures/eagle5_parity_q3b.json` matches the old 2-block head; that
parity test passes). nb02 trained `num_blocks=1`, i.e. an **unverified 1-block
forward path** — the prime suspect for the 0%.

## Fix already applied (likely sufficient)
`colab/02_eagle3_train.py` TRAIN config flipped **`num_blocks` 1 → 2**, so the next
retrain (on the larger corpus now being captured) produces a **2-block head
matching the parity-verified structure**. Expectation: a good-data 2-block head
should reproduce its PyTorch acceptance on-device. **Verify by re-running
`tools/bench/eagle5_paired_bench.sh` with the new head — accept should be ≫ 0.**

## If 2-block STILL gives 0% on-device — the definitive diagnostic (NEEDS TORCH)
Confirm whether the runtime forward reproduces the head's PyTorch logits:
```
# 1) regenerate the parity fixture from the ACTUAL head (PyTorch forward):
python3 tools/eagle5_forward_dump.py --head <head_final.safetensors> \
  --out crates/dismantle-core/tests/fixtures/eagle5_parity_q3b.json --seed 0xea91e5
# 2) run the Rust runtime forward against it:
cargo test -p dismantle-core --test eagle5_forward_parity --ignored -- --nocapture
```
- **Parity PASS** ⇒ runtime forward is correct; the 0% is then in the spec-decode
  **accept/verify loop** or how the runtime feeds `draft_hidden`/the captured
  layer-32 state back in. Debug `crates/dismantle-core/src/speculate/eagle5.rs`
  (the verify/accept path) and how the K-step rollout sources its hidden state vs
  how `eagle5_tau_eval_pytorch.py --chain-hidden` does it (they must match).
- **Parity FAIL** ⇒ the forward itself diverges. Diff `eagle5_forward.rs` against
  the PyTorch `Eagle5Head.forward` (`colab/eagle5_train_pytorch.py`): check the
  `in_proj` (2048×6144 = concat of [embed, hidden, ?]), `residual_gate`,
  `calib_proj`, RMSNorm eps (1e-6), and the per-block attention/MLP order. The
  capture layer is 32 — confirm the runtime taps the same layer the head expects.

## Evidence / pointers
- Bench: `/tmp/eagle_bench_new.log` (0.000 accept, all K). Cloud result:
  `~/Downloads/eagle3_train_result.json` (τ=1.33, depth1=0.80).
- Heads: new `~/Downloads/head_final.safetensors`; old
  `checkpoints/eagle5_final/q3b/head_final.safetensors`.
- Runtime: `crates/dismantle-core/src/speculate/{eagle5,eagle5_forward}.rs`;
  tests `eagle5_forward_parity.rs`, `eagle5_trained_head_load.rs`,
  `qwen_eagle5_speculate.rs`.
- Trainer/eval: `colab/eagle5_{train,tau_eval}_pytorch.py`.
- Prior handoff + context: `plans/eagle_spec_handoff_2026_05_30.md`,
  `plans/bible_colab_audit_2026_05_30.md`; memory `bible_execution_2026_05_30`.

## Done when
`eagle5_paired_bench.sh` with the new head shows accept ≫ 0 and dec_tps **above**
the no-spec baseline, under the §1 gate. That is the first real on-device spec win.
