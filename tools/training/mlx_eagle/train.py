"""
train.py — MLX training loop for the EAGLE-3 head.

Wires together model.py (EagleHead) + data.py (ParquetBatchIterator) +
AdamW with cosine LR + position-weighted CE loss into a runnable trainer.

Key design choices (driven by 5k_capture_results.md findings):

  1. Position-weighted loss
     -----------------------
     The loss mask emitted by data.py is already 0 for BOS-warmup
     positions (pos < skip_bos_positions, default 3). Training ignores
     those positions instead of letting them push the head toward
     predicting from low-magnitude hidden states.

  2. Auxiliary MSE loss
     ------------------
     EAGLE paper §3.3 suggests adding 0.1 * MSE(draft_hidden,
     target_hidden) to push the head's intermediate geometry toward
     the target's. Controllable via --aux-weight.

  3. Mixed precision
     ----------------
     Trainable weights live in bf16 (matches M3 Pro native), frozen
     weights stay fp16 (no point converting). AdamW state stays fp32
     for numerical stability — that's the memory bottleneck.

  4. Checkpointing every --save-every steps
     --------------------------------------
     Crashes happen. Saving once per ~5 min of training keeps
     restart cost bounded.

  5. Bounded smoke mode
     ------------------
     --max-steps N caps the run at N steps regardless of epoch.
     Use --max-steps 100 to validate the loop converges before
     committing to a full epoch (~3-6 hr).

Typical first invocation (Monday morning):

    PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
    $PY tools/training/mlx_eagle/train.py \\
      --parquet training_data/c2_hidden/eagle3_v0/shard_000.parquet \\
      --frozen tools/training/mlx_eagle/v2lite_frozen.npz \\
      --max-steps 100 \\
      --batch-size 16 --seq-len 16 \\
      --lr 3e-4 \\
      --ckpt-dir tools/training/mlx_eagle/ckpt \\
      --log tools/training/mlx_eagle/train.log

Watch the log for loss decreasing over the first 100 steps. If yes,
swap --max-steps for --epochs 3 and let it run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys
import time
from typing import Dict, Optional

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

# Imports below are local to this script to keep startup fast on --help.
def _import_local():
    from tools.training.mlx_eagle.data import ParquetBatchIterator
    from tools.training.mlx_eagle.model import (
        EagleHead,
        EagleHeadConfig,
        load_head_from_npz,
    )
    return ParquetBatchIterator, EagleHead, EagleHeadConfig, load_head_from_npz


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def position_weighted_loss(
    logits,  # (B, S, V)
    target_hidden,  # (B, S, H)
    target_next_tokens,  # (B, S)
    loss_mask,  # (B, S)  — 0 where to skip
    draft_hidden=None,  # (B, S, H) or None
    aux_weight: float = 0.0,
):
    """Position-weighted CE + optional MSE auxiliary.

    Returns (total_loss, ce_loss, aux_loss). Each is a scalar mx.array.

    CE is computed per-position then masked-averaged. This matches the
    "ignore BOS-warmup positions" semantics from data.py's loss_mask.

    MSE auxiliary uses the SAME mask so geometry-loss doesn't get
    polluted by BOS-warmup hidden states (which are lower-magnitude by
    construction per 5k_capture_results.md §"Hidden L2 by position
    bucket").
    """
    B, S, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_targets = target_next_tokens.reshape(-1)
    flat_mask = loss_mask.reshape(-1)

    # Per-position CE without reduction.
    ce_per_pos = nn.losses.cross_entropy(
        flat_logits, flat_targets, reduction="none"
    )  # (B*S,)
    masked_sum = mx.sum(ce_per_pos * flat_mask)
    denom = mx.maximum(mx.sum(flat_mask), mx.array(1.0))
    ce = masked_sum / denom

    if aux_weight > 0.0 and draft_hidden is not None:
        # MSE elementwise, then average over hidden_dim, then mask-avg over positions.
        per_pos_mse = mx.mean(
            (draft_hidden - target_hidden) ** 2, axis=-1
        ).reshape(-1)
        mse_masked_sum = mx.sum(per_pos_mse * flat_mask)
        mse = mse_masked_sum / denom
        return ce + aux_weight * mse, ce, mse

    return ce, ce, mx.zeros(())


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def cosine_with_warmup(
    step: int, total_steps: int, base_lr: float, warmup_pct: float = 0.05
) -> float:
    """Cosine decay from base_lr to 0 with linear warmup over warmup_pct of total."""
    warmup_steps = max(1, int(total_steps * warmup_pct))
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(head, opt_state, step: int, ckpt_dir: pathlib.Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # MLX saves via mx.savez / mx.save_safetensors. Use savez for simplicity.
    # Trainable params only (excludes frozen).
    params = head.trainable_parameters()
    # Flatten dict-of-dict to flat key:value for savez.
    flat = {}
    def _flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten(f"{prefix}{k}.", v)
            elif hasattr(v, "shape"):  # mx.array
                flat[f"{prefix}{k}"] = v
    _flatten("p.", params)
    path = ckpt_dir / f"step_{step:06d}.npz"
    mx.savez(str(path), **flat)
    return path


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> int:
    if mx is None:
        print("ERROR: MLX not installed — pip install mlx", file=sys.stderr)
        return 2

    ParquetBatchIterator, EagleHead, EagleHeadConfig, load_head_from_npz = _import_local()

    # Build the head from frozen weights.
    print(f"[train] loading frozen weights from {args.frozen}", file=sys.stderr)
    head = load_head_from_npz(args.frozen)
    cfg = head.cfg
    print(
        f"[train] head built: h={cfg.hidden_dim} V={cfg.vocab_size} "
        f"eps={cfg.rms_eps}",
        file=sys.stderr,
    )

    # Build data iterator.
    paths = args.parquet if isinstance(args.parquet, list) else [args.parquet]
    it = ParquetBatchIterator(
        parquet_paths=paths,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_dim=cfg.hidden_dim,
        skip_bos_positions=args.skip_bos_positions,
        shuffle=True,
        seed=args.seed,
    )
    batches_per_epoch = len(it)
    total_steps = (
        args.max_steps
        if args.max_steps > 0
        else batches_per_epoch * args.epochs
    )
    print(
        f"[train] {batches_per_epoch:,} batches/epoch × {args.epochs} epoch(s) "
        f"= {batches_per_epoch * args.epochs:,} (cap: {total_steps:,})",
        file=sys.stderr,
    )

    # Optimizer.
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    # Loss closure for nn.value_and_grad.
    def loss_fn(head, batch):
        logits, draft_hidden = head(
            batch["prev_tokens"], batch["target_hidden"], return_hidden=True
        )
        total, ce, aux = position_weighted_loss(
            logits,
            batch["target_hidden"],
            batch["target_next_tokens"],
            batch["loss_mask"],
            draft_hidden=draft_hidden if args.aux_weight > 0 else None,
            aux_weight=args.aux_weight,
        )
        return total, (ce, aux)

    loss_and_grad = nn.value_and_grad(head, loss_fn)

    # Log file.
    log_path = pathlib.Path(args.log) if args.log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w")
    else:
        log_f = None

    ckpt_dir = pathlib.Path(args.ckpt_dir) if args.ckpt_dir else None

    step = 0
    epoch = 0
    t_start = time.time()
    last_log_t = t_start
    losses_window = []

    while step < total_steps:
        for batch in it.iter_epoch(epoch=epoch):
            if step >= total_steps:
                break
            # Set per-step LR.
            opt.learning_rate = cosine_with_warmup(
                step, total_steps, args.lr, warmup_pct=args.warmup_pct
            )
            (total, (ce, aux)), grads = loss_and_grad(head, batch)
            opt.update(head, grads)
            mx.eval(head.parameters(), opt.state, total)

            losses_window.append(float(total))
            step += 1

            if step % args.log_every == 0 or step == 1:
                t_now = time.time()
                interval = t_now - last_log_t
                steps_per_sec = (
                    args.log_every / interval if interval > 0 else float("nan")
                )
                avg_loss = sum(losses_window) / len(losses_window)
                line = (
                    f"step={step:6d}/{total_steps} "
                    f"epoch={epoch} "
                    f"loss={float(total):.4f} ce={float(ce):.4f} aux={float(aux):.4f} "
                    f"avg_loss={avg_loss:.4f} "
                    f"lr={opt.learning_rate:.2e} "
                    f"{steps_per_sec:.2f} steps/s"
                )
                print(line)
                if log_f:
                    log_f.write(line + "\n")
                    log_f.flush()
                last_log_t = t_now
                losses_window = losses_window[-100:]

            if (
                ckpt_dir is not None
                and args.save_every > 0
                and step % args.save_every == 0
            ):
                path = save_checkpoint(head, opt.state, step, ckpt_dir)
                msg = f"[ckpt] saved {path}"
                print(msg)
                if log_f:
                    log_f.write(msg + "\n")
                    log_f.flush()

        epoch += 1

    # Final checkpoint.
    if ckpt_dir is not None:
        path = save_checkpoint(head, opt.state, step, ckpt_dir)
        print(f"[ckpt] final {path}")

    total_wall = time.time() - t_start
    summary = {
        "total_steps": step,
        "epochs": epoch,
        "wall_sec": total_wall,
        "steps_per_sec_overall": step / total_wall if total_wall > 0 else 0,
        "final_loss": float(total),
    }
    print(f"[train] done: {json.dumps(summary)}")
    if log_f:
        log_f.write(json.dumps({"summary": summary}) + "\n")
        log_f.close()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    # Data
    p.add_argument(
        "--parquet",
        default="training_data/c2_hidden/eagle3_v0/shard_000.parquet",
        nargs="+",
    )
    p.add_argument("--frozen", default="tools/training/mlx_eagle/v2lite_frozen.npz")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--skip-bos-positions", type=int, default=3,
                   help="Drop loss on positions 0..N-1 (BOS warmup). "
                        "Per 5k_capture_results.md, pos 0-9 have ~17% lower hidden "
                        "L2 norm; N=3 is conservative.")
    # Optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-pct", type=float, default=0.05)
    p.add_argument("--aux-weight", type=float, default=0.1,
                   help="Coefficient for MSE auxiliary on draft hidden vs target hidden. "
                        "0 disables. EAGLE paper §3.3 uses 0.1.")
    # Schedule
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Cap total steps (overrides epochs). 0 = no cap. "
                        "Use --max-steps 100 for smoke tests.")
    # Logging / ckpt
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--ckpt-dir", default="tools/training/mlx_eagle/ckpt")
    p.add_argument("--log", default="tools/training/mlx_eagle/train.log")
    p.add_argument("--seed", type=int, default=20260516)
    args = p.parse_args()

    return train(args)


if __name__ == "__main__":
    sys.exit(main())
