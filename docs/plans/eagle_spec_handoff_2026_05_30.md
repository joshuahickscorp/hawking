# Handoff — EAGLE spec-decode head for Qwen2.5-3B (parallel to Stage 2)

Paste this whole file as the opening prompt for a fresh session. It runs in
parallel with the M3 Stage-2 kernel work (different machine: training is cloud).

---

## Goal
Produce a **working** EAGLE draft head for Qwen2.5-3B so dismantle's
`--speculate eagle5 --eagle5-head <path>` gives a real decode speedup on **code**.
Done when: accepted-length **τ ≥ 2.5** on a held-out code set AND
`tools/bench/eagle5_paired_bench.sh` shows dec_tps **above** the no-spec baseline,
under the §1 gate (`tools/bench/analyze_tcb_trace.py`).

## Why this is open (measured 2026-05-30)
- The existing head `checkpoints/eagle5_final/q3b/head_final.safetensors` (1.83 GB)
  **loads but gives 0.000 acceptance and is 4.5× SLOWER** (34→7.6 dec_tps) on
  Qwen-3B code. So it's trained but useless as-is.
- The free alternative (n-gram/PLD) was ruled out: oracle A τ=1.43 → NO-GO. So a
  **trained head is the only spec path**; it must be fixed.

## Do these IN ORDER — the first two are cheap and may explain the 0%

**Step 1 (cheapest) — head↔runtime parity.** Before retraining anything, confirm
the Rust runtime computes the same logits as the trained head:
```
cargo test -p dismantle-core --test eagle5_forward_parity -- --nocapture
```
Fixture: `crates/dismantle-core/tests/fixtures/eagle5_parity_q3b.json`. Runtime
forward: `crates/dismantle-core/src/speculate/eagle5_forward.rs`. **If parity
FAILS, the 0%-accept is an integration bug, not a data bug — fix the forward and
re-measure before any cloud training.** This could save the whole training run.

**Step 2 — diagnose the capture/serving mismatch (likely root cause).** The head
was trained on captures from `colab/mega_calibrate.py`, which loads a **HF** model
(f16, or bnb-4bit via `--load-4bit`) and captures residuals. But dismantle
**serves Q4_K_M** (ggml k-quant) — a different weight distribution. That mismatch
is the prime suspect for 0% acceptance. Resolve which capture source the trainer
should use:
- `colab/mega_calibrate.py` — HF-model captures (--model, --dataset, --load-4bit,
  --capture-layer 32, --out parquet). Convenient but NOT dismantle's actual dist.
- dismantle runtime capture — the EXACT Q4_K_M residuals dismantle serves, via
  env in `crates/dismantle-core/src/model/qwen_dense.rs`:
  `DISMANTLE_QWEN_EAGLE5_CAPTURE=1`, `DISMANTLE_QWEN_EAGLE5_CAPTURE_LAYER=32`,
  `DISMANTLE_QWEN_CAPTURE_CORPUS_PATH=<dir>`. Run `dismantle generate
  --prompts-file <code corpus> --max-new-tokens 64 --temperature 0` to dump.
- **Hypothesis to test:** train on dismantle-runtime (Q4_K_M) captures, not HF
  captures. Figure out the parquet schema the trainer expects
  (`colab/eagle5_train_pytorch.py` "Input contract" docstring: `tokens`,
  `residual_q` int8, `residual_scale` f32, …). **The runtime-capture packer
  EXISTS:** `tools/orchestrator/pack_corpus.py --in <residuals.bin> --out-dir
  <corpus>` turns a dismantle-runtime capture `.bin` into the `shard_*.parquet`
  the trainer reads. So the Q4_K_M path is: runtime capture (env above) →
  pack_corpus.py → train. (`mega_calibrate.py` is the *other* path — it generates
  HF-model captures itself, the one that produced the 0%-accept head.)

## Pipeline (once steps 1–2 resolved)
1. **M3 — captures** from the source step 2 picks (Q4_K_M runtime preferred).
2. **M3 — frozen base:** `python3 tools/training/build_qwen3b_frozen.py
   --gguf models/qwen2.5-3b-instruct-q4_k_m.gguf --out artifacts/eagle5/qwen3b_frozen.npz`
   (token_embd / lm_head / output_norm; tied-embedding is fine).
3. **CLOUD (A100/L4) — train:** `colab/02_eagle3_train.ipynb` orchestrates
   `colab/eagle5_train_pytorch.py` (Qwen-3B preset: capture-layer 32, num-blocks 1,
   head-heads 16, chained-hidden rollout — see the notebook's TRAIN dict). Upload
   the parquet corpus + frozen npz. ~hours on A100.
4. **CLOUD — τ-eval:** `colab/eagle5_tau_eval_pytorch.py --ckpt … --frozen … --corpus
   … --depth 4`. **GATE: τ ≥ 2.5.**
5. **M3 — verify + bench:** re-run Step-1 parity with the NEW head, then
   `WEIGHTS=models/qwen2.5-3b-instruct-q4_k_m.gguf
   PROFILE=profiles/qwen3b-instruct-q4k.m3pro18.json
   EAGLE5_HEAD=<new head_final.safetensors> PROMPT='def quicksort(arr):'
   bash tools/bench/eagle5_paired_bench.sh`. Must beat 0.000 accept and the no-spec tps.

## Assets that already exist
- Trained-but-broken head: `checkpoints/eagle5_final/q3b/head_final.safetensors`.
- Trainer + eval + corpus builder: `colab/eagle5_train_pytorch.py`,
  `colab/eagle5_tau_eval_pytorch.py`, `colab/mega_calibrate.py`,
  `tools/training/build_qwen3b_frozen.py`, `tools/eagle5_forward_dump.py`.
- Orchestrator notebook: `colab/02_eagle3_train.ipynb` (has the deps + provenance
  gate; **note**: simplify its deps like nb01 if Colab fights numpy — prefer
  Colab's default numpy 2.x, don't pin, don't auto-restart unless forced).
- Runtime: `crates/dismantle-core/src/speculate/eagle5{,_forward}.rs`; tests
  `eagle5_forward_parity.rs`, `eagle5_trained_head_load.rs`, `qwen_eagle5_speculate.rs`.
- Strategy: `plans/throughput_bible_2026_05_30.md` (axis 3); the small-model spec
  caveat is real — EAGLE pays off best AFTER Stage 2 makes the kernels
  bandwidth-bound (idle compute for the draft) and on code (highest acceptance).

## Memory pointers
`eagle5_port_phase_a1_shipped`, `eagle5_train_qwen3b_adapter_notes`,
`eagle5_corrected_pipeline_2026_05_29`, `eagle5_serial_verify_wins_2026_05_29`,
`bible_execution_2026_05_30` (has the 0%-accept measurement + byte-cut result).

## Coordination with Stage 2 (running on the M3 in parallel)
- Training is **cloud** → no resource conflict with M3 kernel work.
- The only M3 contention is capture-gen (GPU) vs Stage-2 perf benches (GPU). Stage 2
  uses paired-delta benches (contamination-robust), so coexistence is fine; or run
  captures when the M3 is idle.
- Do NOT touch the Q4_K kernel files (`shaders/quant.metal`,
  `src/kernels/mod.rs`) — Stage 2 owns those this cycle.
