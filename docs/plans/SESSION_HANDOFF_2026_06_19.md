# Session Handoff — 2026-06-19 (chunked-scan + git cleanup + rename prep)

Paste the "Opening prompt" at the bottom into a new chat to resume.

## One-paragraph state

RWKV-7 0.4B low-bit QAT ("G1a", FFN ternary, last-8 layers) is running on MPS,
now ~6x faster after wiring the **chunked-scan WKV-7** kernel through the trainer.
The working tree was committed clean (4 commits) and 25 dead branches deleted. A
full `dismantle → hawking` rename is **designed and prepped but NOT executed** —
it must wait until training + the post-G1a chain finish, because the live
`autocycle` relaunches the trainer every ~5 steps using current paths.

## Live processes (as of 2026-06-19 ~23:33 EDT)

| What | PID | Notes |
|---|---|---|
| G1a trainer (chunked) | 23689 | `nohup caffeinate` python `rwkv7_qat.py … --use-chunked --chunk-size 32 --resume-from step_000075` |
| autocycle orchestrator | 24703 | targets 80..150 every 5, handoff→hawking armed at ema≤6.0, keep_last 8 |

If PIDs are stale (new session), re-find: `pgrep -fl rwkv7_qat.py` and
`pgrep -fl autocycle_step50_ozempic`.

## Training status

- Step ~78/299 (grad_accum 8 → ~299 steps = 1 epoch over 2393-row SFT corpus).
- Chunked step ~5.8 min median (was 27–47 min sequential) = **~6.1x**. Some long-
  sequence steps spike to ~20 min — normal, not a stall (check `%CPU` oscillates,
  not zero).
- loss_ema ~6.0–6.12 (target the hawking handoff fires at **ema ≤ 6.0**, min_step 60).
- **ETA full G1a ~Sat Jun 20 ~8:50 PM EDT** (~21h from restart), barring crashes.

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
- Branch cleanup: 38 → 14 (deleted 17 `worktree-agent-*` cruft + 8 already-merged
  feature branches). 10 unmerged feature branches LEFT (all 21–31 commits behind main,
  likely stale) — user to decide merge/delete. Nothing pushed.

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
- [ ] Decide on 10 unmerged feature branches (merge selectively or delete — they're stale).
- [ ] Push main to origin? (nothing pushed this session — was deliberate.)
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
