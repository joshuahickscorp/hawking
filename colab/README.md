# Colab notebooks for dismantle calibration

Big-GPU calibration work that doesn't fit on M3 Pro 18 GB.

## Active notebook

### `finish_q3b_reconciliation.ipynb` ⭐ current

Single end-to-end notebook to finish the Qwen reconciliation pipeline.
After a May 26 run lost the q3b long head to over-aggressive cleanup,
the older multi-notebook flow was retired in favour of this one.

Run order (run-all from top):

1. **Setup + pre-flight** — mounts Drive, installs deps, clones repo, and
   hard-asserts every input is present (q3b corpus, frozen, AWQ artifacts,
   q1p5 long head). Halts loudly on any FAIL.
2. **Progress journal** — loads `reconciliation_progress.json`; safe to
   re-run after a disconnect, each stage skips if it already wrote its
   artifact and the sha256 matches.
3. **q3b long retrain** — exact winner spec (`b1_wide`, 16 heads,
   ff_mult=6.0, lr=5e-4, residual_delta=0.020, calib_weight=0.20,
   20 epochs, 8k rows, 192-token windows). On completion, verifies the
   safetensors is loadable, sha256s it, AND triggers an immediate local
   `files.download()` so the head survives any subsequent Drive disaster.
4. **q3b tau eval**
5. **q3b frontier policy search**
6. **Reconciliation summary** — combines q3b + q1p5 frontier winners into
   `reconciliation_frontier_winners.json` + `reconciliation_summary.md`.
7. **Export essentials** — runs `export_reconciliation_essentials.py
   --strict --zip`, then triggers local download of the zip.

Launch:
```
https://colab.research.google.com/github/joshuahickscorp/dismantle/blob/main/colab/finish_q3b_reconciliation.ipynb
```

**Compute:** A100-40GB ≈ 3–4 hr for the retrain + ~20 min for eval/export.
T4 will be ~3× slower; A100/L4 strongly preferred.

## Hard rules learned from the loss

1. **No silent advance.** Every stage hard-asserts its artifact exists,
   is loadable, has expected size, and records its sha256. The trainer's
   `save_safetensors` was patched to write atomically (`.tmp` + rename)
   and raise on missing-deps instead of silently returning.
2. **Local backup after long-train.** Right after the q3b head lands on
   Drive, `google.colab.files.download()` pushes it to the user's local
   machine. Drive-side disasters can't take the head down once this fires.
3. **No inline cleanup cells.** If Drive fills up, stop and triage. Do
   not paste `rm -rf` cells into the notebook — the previous run did this
   to free space mid-training and over-matched several critical paths.

## Why "reconciliation"?

The May 2026 end-to-end paired bench discovered that `--speculate eagle5`
on Qwen-3B/1.5B is a no-op: spec-decode is wired into `deepseek_v2.rs`
only, not `qwen_dense.rs`. The trained heads are inventory waiting on
the Rust port (see `docs/eagle5_qwen_port_plan.md`).

## Supporting scripts (kept; not user-runnable from Colab UI)

- `eagle5_train_pytorch.py` — trainer; `save_safetensors` is now atomic
  and raises on round-trip failure.
- `eagle5_tau_eval_pytorch.py` — tau eval.
- `eagle5_frontier_policy.py` — frontier policy search.
- `mega_calibrate.py` — corpus + activation-stats builder. Not run in the
  current notebook (corpus already on Drive). Kept for rebuilds.
- `build_qwen3b_frozen_hf.py` — frozen baseline dump. Already produced.
- `awq_per_channel_calibrate.py`, `q2k_importance_calibrate.py`,
  `awq_w4a8_validate.py` — calibration helpers. Artifacts already produced.
- `export_reconciliation_essentials.py` — invoked by Cell 7.
