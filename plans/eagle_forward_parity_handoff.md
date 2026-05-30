# Handoff — EAGLE head↔runtime forward parity (the on-device 0%-accept blocker)

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
