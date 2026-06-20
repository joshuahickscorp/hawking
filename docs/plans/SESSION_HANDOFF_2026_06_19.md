# Session Handoff — 2026-06-19 (chunked-scan + git cleanup + rename prep)

Paste the "Opening prompt" at the bottom into a new chat to resume.

## One-paragraph state (updated 2026-06-20 ~10:40 EDT)

**G1a is DONE** — RWKV-7 0.4B ternary-QAT (FFN last-8) hit its EMA gate (loss_ema
5.92 ≤ 6.0) at **step 90**, the hawking handoff fired, and it **exported a full HF
model** at `artifacts/lowbit_rwkv7/hawking_arc/ema6p0_step_000090_20260620_041225/`
(candidate ppl 3.449 vs fp32 base 3.356 = **+2.7%**, near-lossless low-bit; samples
coherent with expected 0.4B repetition). Now the **master autonomous chain is running**
(see below): build/parity checks → 7-model draft sweep → hardening → scorecard. The
`dismantle → hawking` rename is still **prepped but NOT executed** (kit ready; wait
until the chain finishes — it uses live training paths).

## Master autonomous chain (launched 2026-06-20 ~10:40 EDT, pid 55871)

One sequential MPS job at a time (18 GB). Detached (`nohup caffeinate`), chunked +
seeded (1337), watermark 0.0. Command was:
`DRAFT_VARIANTS="<7 variants>" DRAFT_EPOCHS=1 SEED=1337 USE_CHUNKED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 nohup caffeinate -dimsu bash tools/training/g1a_v2_expansion_chain.sh 3.4489 pass`

Stages (each soft-fails — one failure never stops the rest):
1. cargo check core/serve/bench/tq (✓ all PASS at launch), json/mamba2 tests, rwkv7
   parity + flatness (skip if no GGUF), TQ synthetic parity, TQ artifact gates,
   Qwen3B llama baseline (soft).
2. **Draft sweep** — 7 RWKV-7 drafts trained from scratch, small→large, chunked,
   EPOCHS=1: `draft_35m_probe 50m 75m 100m 150m 200m 300m`. Per-model eval (ppl +
   accept-rate vs 0.4B) appended to `runs/custom_<v>/eval_log.jsonl`.
3. `rwkv7_spec_hardening.py` (spec physics gate) → `rwkv7_competitive_scorecard.py`.

ETA: ~20–30h (the 300M is the long pole). Runs as long as it needs; not time-boxed.

### Monitor the chain
```bash
tail -f artifacts/lowbit_rwkv7/master_chain.log          # stage-by-stage
tail -f artifacts/lowbit_rwkv7/draft_sweep.log           # which draft model is training
ls -t artifacts/lowbit_rwkv7/runs/custom_*/eval_log.jsonl # per-model results as they land
pgrep -fl "g1a_v2_expansion_chain|rwkv7_train_draft"      # is it alive? (master pid 55871)
```
If it dies mid-sweep, re-launch just the sweep:
`DRAFT_VARIANTS="<remaining>" EPOCHS=1 SEED=1337 USE_CHUNKED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 bash tools/training/launch_draft_sweep.sh`

## G1a output (the finished 0.4B model)
`artifacts/lowbit_rwkv7/hawking_arc/ema6p0_step_000090_20260620_041225/` — `hf/`
(loadable HF export), `state_dict.pt`, `report.md`, `ppl.jsonl`, `samples.jsonl`,
`frontier_queue.sh` (optional recipes to push the 0.4B further: 512_g8 / 768_g8 / 1024_g16).
The autocycle + G1a trainer are STOPPED (by design — gate reached), not crashed.

## Training status

- Step ~78/299 (grad_accum 8 → ~299 steps = 1 epoch over 2393-row SFT corpus).
- Chunked step ~5.8 min median (was 27–47 min sequential) = **~6.1x**. Some long-
  sequence steps spike to ~20 min — normal, not a stall (check `%CPU` oscillates,
  not zero).
- loss_ema ~6.0–6.12 (target the hawking handoff fires at **ema ≤ 6.0**, min_step 60).
- **ETA full G1a: a RANGE, ~Sat Jun 20 → Mon Jun 22.** Clean chunked steps are ~6 min,
  but the corpus has long-sequence clusters that run ~18–22 min/step and inflate the
  live estimate (it read "Mon" at step 78 during one such cluster). The true finish
  depends on the sequence-length distribution over the remaining ~220 steps. Trust
  `rwkv7_progress.py`'s live number but expect it to swing.

### Monitor
```bash
.venv-rwkv/bin/python tools/training/rwkv7_progress.py            # one-line ETA
tail -f artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8/events.jsonl
tail -f artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8/autocycle_step50_ozempic.log
```

### If it crashes (OOM / battery) — resume from latest checkpoint
The autocycle auto-relaunches at each 5-step target; if BOTH it and the trainer are
dead, restart the chunked trainer from the newest `step_NNNNNN`:
```bash
LATEST=$(ls -d artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8/step_* | sort | tail -1)
PYTHONHASHSEED=1337 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
  nohup caffeinate -dimsu .venv-rwkv/bin/python tools/training/rwkv7_qat.py \
  --model models/rwkv7-g1-04-hf/model.safetensors --hf-dir models/rwkv7-g1-04-hf \
  --data artifacts/rwkv7_posttrain/sft.jsonl \
  --out artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8 \
  --stage ffn --quant ternary --last-n-layers 8 \
  --max-length 768 --grad-accum 8 --lr 5e-6 --epochs 1 \
  --save-every 5 --eval-every 0 --device mps --run-id g1a --seed 1337 \
  --pretokenize-workers 4 --mps-empty-cache-every 5 \
  --use-chunked --chunk-size 32 --resume-from "$LATEST" \
  > artifacts/lowbit_rwkv7/runs/g1_ffn_ternary_last8/relaunch_manual.log 2>&1 &
```
Then re-attach autocycle (one shot, with handoff) — see its env block in
`autocycle_step50_ozempic.log` (`OZEMPIC_*` vars), pass the new trainer PID as `$1`.

## The chain (what fires after G1a)

1. autocycle handoff at **ema ≤ 6.0** → `hawking_after_ema.sh` (eval ppl short/long +
   sample prompts + export HF dir + writes `frontier_queue.sh` with next QAT recipes
   `hawking_branch_512_g8`, `hawking_anchor_1024_g16` — both now emit `--use-chunked`).
2. Post-G1a chain (separate): `g1a_v2_expansion_chain.sh` → phase2 (TQ export/build/
   parity) → v2_expansion (cargo checks, parity, Qwen3B llama baseline) → draft sweep
   (`launch_draft_sweep.sh`, 7 RWKV-7 drafts 35M–300M, **now chunked**) → spec hardening
   → competitive scorecard.

## What changed this session (committed to main)

- `77024b9` training arc: chunked-scan threaded through G1a QAT + draft sweep + hawking;
  `rwkv7_qat.py` gained `--use-chunked/--chunk-size`, `--resume-from/--resume-step`,
  batched MPS flush, pretokenize, determinism; `rwkv7_progress.py` tool.
- `f0880a9` vendor/strand + crates: removed dead CUDA backend.
- `22ad5c6` tools: removed retired colab notebooks; strand/bench updates.
- `df96833` docs: hawking plans, serve-matrix reports.
- `e76fc86` docs: rename execution kit + this handoff.
- `773f7f5` test: salvaged chunked-scan parity test (forward+gradient+ref-loop, ALL
  PASS against main) from `rwkv7/chunked-scan`, then deleted that branch as redundant
  (its chunked kernel was byte-identical to main; it was 23 commits behind and would
  have resurrected the deleted CUDA backend).
- **main PUSHED to origin at `773f7f5`** (clean fast-forward, joshuahickscorp/dismantle).
- Branch cleanup: 38 → 13 (deleted 17 `worktree-agent-*` cruft + 8 merged + the stale
  `rwkv7/chunked-scan`). **9 unmerged feature branches LEFT** (all 21–31 behind main,
  likely stale) — decide merge/delete. They are NOT worth pushing (behind main).

### Chunked-scan = the headline win
`use_chunked=True` (parallel-scan WKV-7) is **bit-identical forward / machine-eps
gradient** vs the sequential loop (verified on RWKV7Model), ~6x faster fwd+bwd on MPS,
zero quality change (FFN ternary fake-quant path untouched). It was already supported
in the trainers but the launchers/QAT never passed it.

## Rename status: PREPPED, NOT STARTED

- Strategy: `docs/plans/hawking_total_rename_plan_2026_06_19.md` (the user's plan — says
  "no directory moves yet", phased, repo rename last).
- **Execution kit: `docs/plans/hawking_rename_execution_kit_2026_06_19.md`** — runnable
  commands + gates + hazards (`.dismantle` sidecar, `dismantle_*` metrics, `/v1/dismantle/*`
  routes, env vars are all dual-stack-not-replace). Phase 1 (compat layer) is Rust-only
  and safe to start anytime; Phase 3/4 (dir/crate/tooling moves) WAIT for training to end.

## Open items / next decisions

- [ ] Let G1a + chain finish (~Sat eve), then execute rename Phase 1 → 3 → 4 from the kit.
- [ ] Decide on 9 remaining unmerged feature branches (all 21–31 behind main, stale).
- [x] main pushed to origin (773f7f5). Future commits: `git push origin main`.
- [ ] Re-run draft accept-rate eval vs the real 3B/7B target once downloaded (current
      eval uses 0.4B as teacher proxy).
- [ ] `tools/training/rwkv7_qat.py` etc. are committed but `cargo`/Rust crates were not
      rebuilt this session — verify build before any release.

## Opening prompt (paste into a new chat)

> Resuming the dismantle/hawking project. Read
> `docs/plans/SESSION_HANDOFF_2026_06_19.md` and
> `docs/plans/hawking_rename_execution_kit_2026_06_19.md` first. Current state: RWKV-7
> 0.4B G1a QAT is running on MPS with the chunked-scan kernel (~6x), ETA ~Sat Jun 20
> eve, managed by autocycle (PID may have changed — re-find with
> `pgrep -fl rwkv7_qat.py`). Do NOT rename any folders/crates/git until G1a + the
> post-G1a chain finish (the autocycle relaunches the trainer using live paths). First,
> tell me G1a's current step/ETA via `tools/training/rwkv7_progress.py`. Then we either
> (a) wait, (b) start the Rust-only rename Phase 1 compat layer (safe during training),
> or (c) handle the 10 stale feature branches.
