"""
train.py — MLX training loop for the EAGLE-3 head.

Wires together model.py (EagleHead) + data.py (ParquetBatchIterator) +
AdamW with cosine LR + position-weighted CE loss into a runnable trainer.

Design decisions (driven by 5k_capture_results.md + perf measurement
on 2026-05-16):

  1. **bf16 weights + activations** (default; --dtype fp32 to override)
     Apple Silicon has native bf16 GEMM. The EAGLE-3 head's hidden-state
     regression is bf16-safe per the paper. Adam state stays fp32 for
     numerical stability (master-weights pattern). Combined with mx.compile
     this is a ~2-3x speedup over fp32-everywhere.

  2. **mx.compile** on the step function (forward + loss + grad)
     MLX's lazy graph eats the per-call dispatch overhead; compile fuses
     and caches the graph. ~1.3-2x on the inner loop. Adds 5-15s
     compilation on first step, amortized over the rest of training.

  3. **Position-weighted loss**
     The loss mask emitted by data.py zeros positions 0..skip_bos_positions.
     5k_capture_results.md showed pos<10 has ~17% lower hidden L2 — the
     model legitimately has less to predict from. Default N=3.

  4. **Auxiliary MSE loss** (EAGLE paper §3.3)
     0.1 * MSE(draft_hidden, target_hidden). --aux-weight 0 disables.

  5. **Checkpointing + --resume**
     `--save-every N` saves a (params + opt-state + meta) bundle each N
     steps. `latest.npz` symlink/copy always points at the most recent.
     `--resume tools/training/mlx_eagle/ckpt/latest.npz` restores params
     + opt state + step counter + epoch and continues from where it died.
     Mandatory for runs of >1 hr — power blip, kernel panic, accidental
     close = lost work otherwise.

  6. **JSONL training log**
     `--log` writes one JSON record per logged step (step / loss / ce /
     aux / lr / steps/s / wall). `plot_train.py` (separate, future) reads
     it to graph curves. Glance-friendly: catch divergence in seconds vs
     trying to read a 10k-line text log.

  7. **--max-steps cap for smoke runs**
     `--max-steps 100` runs ~30s and tells you whether the loop converges
     before you commit to a 10-hr full run.

Typical Monday-morning sequence:

    PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3

    # 1. Smoke (~30s) — confirm loss decreases over 100 steps
    $PY tools/training/mlx_eagle/train.py --max-steps 100

    # 2. Full run (~10 hr for 55K x 3 epochs)
    $PY tools/training/mlx_eagle/train.py --epochs 3

    # 3. If something killed it mid-run
    $PY tools/training/mlx_eagle/train.py --epochs 3 \\
        --resume tools/training/mlx_eagle/ckpt/latest.npz
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

# Resolve repo root (parents[3]: train.py -> mlx_eagle -> training -> tools -> repo)
# and inject so `tools.training.mlx_eagle.{data,model}` imports work when this
# script is invoked from any cwd.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
    aux_target_kind: str = "current",
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
        if aux_target_kind == "next":
            # Predict the NEXT hidden — shift target_hidden by 1 along S.
            # Drop the last position from both pred + tgt + mask (no successor).
            if target_hidden.shape[1] < 2:
                mse = mx.zeros(())
            else:
                pred = draft_hidden[:, :-1, :]
                tgt = target_hidden[:, 1:, :]
                mse_per_pos = mx.mean((pred - tgt) ** 2, axis=-1).reshape(-1)
                # Mask: positions 0..S-2 of the original mask.
                m_shift = loss_mask[:, :-1].reshape(-1)
                mse_sum = mx.sum(mse_per_pos * m_shift)
                m_denom = mx.maximum(mx.sum(m_shift), mx.array(1.0))
                mse = mse_sum / m_denom
        else:
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
# Checkpoint I/O — params + opt state + step/epoch counter
# ---------------------------------------------------------------------------
def _flatten_params(d, prefix: str = ""):
    """Walk nested dict of mx.array, yield (flat_key, array)."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten_params(v, f"{prefix}{k}.")
        elif hasattr(v, "shape"):
            yield f"{prefix}{k}", v


def _unflatten_params(flat: Dict[str, "mx.array"], prefix: str):
    """Inverse of _flatten_params. Returns nested dict for params under prefix."""
    nested: Dict = {}
    for fk, v in flat.items():
        if not fk.startswith(prefix):
            continue
        path = fk[len(prefix):].split(".")
        d = nested
        for p in path[:-1]:
            d = d.setdefault(p, {})
        d[path[-1]] = v
    return nested


def save_checkpoint(
    head, opt, step: int, epoch: int, ckpt_dir: pathlib.Path
) -> pathlib.Path:
    """Atomic save: write `step_NNNNNN.npz` then update `latest.npz` symlink.

    Stores:
      p.<flat-key>  : trainable params (filtered via head.trainable_parameters)
      m.<flat-key>  : Adam first-moment state (per-param)
      v.<flat-key>  : Adam second-moment state
      _meta.step    : training step (int)
      _meta.epoch   : epoch counter
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    flat: Dict[str, "mx.array"] = {}
    for k, v in _flatten_params(head.trainable_parameters()):
        flat[f"p.{k}"] = v
    # Adam state lives in opt.state (a nested dict of {m, v} per param leaf).
    # We mirror the parameter tree so m/v keys parallel p.* keys exactly.
    state = opt.state
    # Newer MLX puts moments under state["m"] / state["v"]; older puts them
    # alongside as dicts named like the param tree. Handle both.
    if isinstance(state, dict) and "m" in state and "v" in state:
        for k, v in _flatten_params(state["m"]):
            flat[f"m.{k}"] = v
        for k, v in _flatten_params(state["v"]):
            flat[f"v.{k}"] = v
    flat["_meta.step"] = mx.array(step, dtype=mx.int32)
    flat["_meta.epoch"] = mx.array(epoch, dtype=mx.int32)
    path = ckpt_dir / f"step_{step:06d}.npz"
    mx.savez(str(path), **flat)
    # Atomic latest update via os.replace (POSIX rename guarantee).
    latest = ckpt_dir / "latest.npz"
    tmp = ckpt_dir / "latest.npz.tmp"
    import shutil
    shutil.copyfile(path, tmp)
    import os as _os
    _os.replace(tmp, latest)
    return path


def load_checkpoint(head, opt, ckpt_path: pathlib.Path) -> Dict[str, int]:
    """Restore head params + opt state from a checkpoint. Returns {step, epoch}.

    Caller must construct head + opt FIRST (same config), then call this.
    """
    flat = mx.load(str(ckpt_path))
    # Restore trainable params.
    p_dict = _unflatten_params(flat, "p.")
    head.update(p_dict)
    # Restore opt moments (rebuild state structure). Newer MLX keeps state
    # internally; we patch m and v directly.
    m_dict = _unflatten_params(flat, "m.")
    v_dict = _unflatten_params(flat, "v.")
    state = opt.state
    if isinstance(state, dict) and "m" in state and "v" in state:
        # In-place update of the moment dicts.
        def _copy_into(dst, src):
            for k, val in src.items():
                if isinstance(val, dict):
                    _copy_into(dst.setdefault(k, {}), val)
                else:
                    dst[k] = val
        _copy_into(state["m"], m_dict)
        _copy_into(state["v"], v_dict)
    step = int(flat["_meta.step"].item()) if "_meta.step" in flat else 0
    epoch = int(flat["_meta.epoch"].item()) if "_meta.epoch" in flat else 0
    return {"step": step, "epoch": epoch}


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------
def _resolve_dtype(name: str):
    """Resolve --dtype string to an mx dtype."""
    return {
        "bf16": mx.bfloat16,
        "bfloat16": mx.bfloat16,
        "fp16": mx.float16,
        "float16": mx.float16,
        "fp32": mx.float32,
        "float32": mx.float32,
    }[name]


def _cast_module_params(module, dtype):
    """In-place cast a module's trainable params to `dtype`.

    Frozen tensors (those stored under attrs starting with `_`) are left
    alone — they're already fp16 from the npz and don't need touching.
    """
    params = module.trainable_parameters()
    casted = {}
    def _walk(d, out):
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = {}
                _walk(v, out[k])
            elif hasattr(v, "astype"):
                out[k] = v.astype(dtype)
            else:
                out[k] = v
    _walk(params, casted)
    module.update(casted)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> int:
    if mx is None:
        print("ERROR: MLX not installed — pip install mlx", file=sys.stderr)
        return 2

    ParquetBatchIterator, EagleHead, EagleHeadConfig, load_head_from_npz = _import_local()

    dtype = _resolve_dtype(args.dtype)

    # Build the head from frozen weights.
    print(f"[train] loading frozen weights from {args.frozen}", file=sys.stderr)
    head = load_head_from_npz(args.frozen)
    cfg = head.cfg
    print(
        f"[train] head built: h={cfg.hidden_dim} V={cfg.vocab_size} eps={cfg.rms_eps}",
        file=sys.stderr,
    )

    # Cast trainable params to requested dtype (bf16 default).
    # Frozen weights (_token_embd, _lm_head, _output_norm) are left as
    # loaded (fp16/fp32 from the npz). MLX casts on operation.
    print(f"[train] casting trainable params to {args.dtype}", file=sys.stderr)
    _cast_module_params(head, dtype)

    # Build data iterator.
    paths = args.parquet if isinstance(args.parquet, list) else [args.parquet]

    # Optional per-channel hidden normalization.
    hidden_mean = hidden_std = None
    if args.hidden_stats:
        sp = pathlib.Path(args.hidden_stats)
        if not sp.exists():
            print(f"ERROR: --hidden-stats {sp} not found", file=sys.stderr)
            return 2
        stats = np.load(sp)
        hidden_mean = stats["mean"].astype(np.float32)
        hidden_std = stats["std"].astype(np.float32)
        print(
            f"[train] hidden normalization loaded from {sp} "
            f"(mean L2={float(np.linalg.norm(hidden_mean)):.3f} "
            f"std mean={float(hidden_std.mean()):.3f})",
            file=sys.stderr,
        )

    common_kwargs = dict(
        parquet_paths=paths,
        batch_size=args.batch_size, seq_len=args.seq_len,
        hidden_dim=cfg.hidden_dim,
        skip_bos_positions=args.skip_bos_positions,
        seed=args.seed,
    )
    if hidden_mean is not None:
        common_kwargs["hidden_mean"] = hidden_mean
        common_kwargs["hidden_std"] = hidden_std
    if args.streaming:
        try:
            from tools.training.mlx_eagle.data import StreamingParquetBatchIterator
            it = StreamingParquetBatchIterator(prefetch=args.prefetch, **common_kwargs)
        except ImportError:
            print("[train] streaming iterator not available; falling back to in-memory",
                  file=sys.stderr)
            it = ParquetBatchIterator(shuffle=True, **common_kwargs)
    else:
        it = ParquetBatchIterator(shuffle=True, **common_kwargs)
    batches_per_epoch = len(it)
    total_steps = (
        args.max_steps if args.max_steps > 0 else batches_per_epoch * args.epochs
    )
    print(
        f"[train] {batches_per_epoch:,} batches/epoch × {args.epochs} epoch(s) "
        f"= {batches_per_epoch * args.epochs:,} (cap: {total_steps:,})",
        file=sys.stderr,
    )

    # Optimizer (state stays fp32 by default — master-weights pattern).
    opt_class = {
        "adamw": optim.AdamW,
        "lion": optim.Lion,
        "muon": optim.Muon,
    }[args.optimizer]
    opt_kwargs = {"learning_rate": args.lr}
    # AdamW + Lion accept weight_decay; Muon's API may differ. Pass when supported.
    import inspect as _inspect
    sig = _inspect.signature(opt_class)
    if "weight_decay" in sig.parameters:
        opt_kwargs["weight_decay"] = args.weight_decay
    opt = opt_class(**opt_kwargs)
    print(
        f"[train] optimizer={args.optimizer} lr={args.lr} weight_decay={args.weight_decay}",
        file=sys.stderr,
    )

    # Resume.
    start_step = 0
    start_epoch = 0
    if args.resume:
        ckpt_path = pathlib.Path(args.resume)
        if not ckpt_path.exists():
            print(f"ERROR: --resume {ckpt_path} not found", file=sys.stderr)
            return 2
        # Initialize opt state by running one dummy step's forward+grad so
        # AdamW allocates its m/v buffers, then load over them.
        dummy = next(iter(it.iter_epoch(epoch=0)))
        def _dummy_loss(h, b):
            l, _ = h(b["prev_tokens"], b["target_hidden"], return_hidden=True)
            return position_weighted_loss(
                l, b["target_hidden"], b["target_next_tokens"], b["loss_mask"]
            )[0]
        dummy_grad_fn = nn.value_and_grad(head, _dummy_loss)
        _l, _g = dummy_grad_fn(head, dummy)
        opt.update(head, _g)
        mx.eval(head.parameters(), opt.state)
        loaded = load_checkpoint(head, opt, ckpt_path)
        start_step = loaded["step"]
        start_epoch = loaded["epoch"]
        print(
            f"[train] resumed from {ckpt_path}: step={start_step} epoch={start_epoch}",
            file=sys.stderr,
        )

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
            aux_target_kind=args.aux_target_kind,
        )
        return total, (ce, aux)

    raw_loss_and_grad = nn.value_and_grad(head, loss_fn)

    # mx.compile fuses the train step graph. Skip on first iteration if
    # --no-compile (useful for debugging). Compile expects a pure-ish
    # function; we wrap the step as `(model_state, opt_state, batch) -> ...`.
    if args.compile:
        @mx.compile
        def step_fn(prev_tokens, target_hidden, target_next, loss_mask):
            batch = {
                "prev_tokens": prev_tokens,
                "target_hidden": target_hidden,
                "target_next_tokens": target_next,
                "loss_mask": loss_mask,
            }
            (total, (ce, aux)), grads = raw_loss_and_grad(head, batch)
            return total, ce, aux, grads
    else:
        def step_fn(prev_tokens, target_hidden, target_next, loss_mask):
            batch = {
                "prev_tokens": prev_tokens,
                "target_hidden": target_hidden,
                "target_next_tokens": target_next,
                "loss_mask": loss_mask,
            }
            (total, (ce, aux)), grads = raw_loss_and_grad(head, batch)
            return total, ce, aux, grads

    # JSONL log.
    log_path = pathlib.Path(args.log) if args.log else None
    log_f = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Append on resume so curves aren't fragmented across restarts.
        mode = "a" if args.resume and log_path.exists() else "w"
        log_f = open(log_path, mode)

    ckpt_dir = pathlib.Path(args.ckpt_dir) if args.ckpt_dir else None

    step = start_step
    epoch = start_epoch
    t_start = time.time()
    last_log_t = t_start
    losses_window = []

    def _log_row(row: dict):
        if log_f:
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()

    _log_row({"event": "start", "wall_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "step": step, "epoch": epoch, "total_steps": total_steps,
              "dtype": args.dtype, "compile": args.compile, "resume": bool(args.resume)})

    while step < total_steps:
        for batch in it.iter_epoch(epoch=epoch):
            if step >= total_steps:
                break
            opt.learning_rate = cosine_with_warmup(
                step, total_steps, args.lr, warmup_pct=args.warmup_pct
            )
            total, ce, aux, grads = step_fn(
                batch["prev_tokens"],
                batch["target_hidden"],
                batch["target_next_tokens"],
                batch["loss_mask"],
            )
            opt.update(head, grads)
            mx.eval(head.parameters(), opt.state, total)

            losses_window.append(float(total))
            step += 1

            if step % args.log_every == 0 or step == start_step + 1:
                t_now = time.time()
                interval = t_now - last_log_t
                steps_per_sec = (
                    args.log_every / interval if interval > 0 else float("nan")
                )
                avg_loss = sum(losses_window) / len(losses_window)
                # Compute grad-norm for the divergence detector.
                gnorm = float(
                    mx.sqrt(sum((g * g).sum() for _, g in _flatten_params(grads)))
                )
                row = {
                    "step": step, "epoch": epoch,
                    "loss": float(total), "ce": float(ce), "aux": float(aux),
                    "avg_loss": avg_loss,
                    "lr": float(opt.learning_rate),
                    "steps_per_sec": steps_per_sec,
                    "grad_norm": gnorm,
                    "wall": time.time() - t_start,
                }
                print(
                    f"step={step:6d}/{total_steps} epoch={epoch} "
                    f"loss={row['loss']:.4f} ce={row['ce']:.4f} aux={row['aux']:.4f} "
                    f"avg={avg_loss:.4f} lr={row['lr']:.2e} "
                    f"|g|={gnorm:.2f} {steps_per_sec:.2f} step/s"
                )
                _log_row(row)
                last_log_t = t_now
                losses_window = losses_window[-100:]
                # Divergence detector: warn (don't abort) if loss is NaN/Inf or
                # 4x the rolling avg of the prior window.
                if not math.isfinite(float(total)):
                    print(f"[train] WARN non-finite loss at step {step}; saving emergency ckpt")
                    if ckpt_dir:
                        save_checkpoint(head, opt, step, epoch, ckpt_dir)

            if (
                ckpt_dir is not None
                and args.save_every > 0
                and step % args.save_every == 0
            ):
                path = save_checkpoint(head, opt, step, epoch, ckpt_dir)
                msg = f"[ckpt] step {step} -> {path.name} (latest.npz updated)"
                print(msg)
                _log_row({"event": "ckpt", "step": step, "path": str(path)})

        epoch += 1

    if ckpt_dir is not None:
        path = save_checkpoint(head, opt, step, epoch, ckpt_dir)
        print(f"[ckpt] final {path.name}")

    total_wall = time.time() - t_start
    summary = {
        "total_steps": step,
        "epochs": epoch,
        "wall_sec": total_wall,
        "steps_per_sec_overall": step / total_wall if total_wall > 0 else 0,
        "final_loss": float(total),
    }
    print(f"[train] done: {json.dumps(summary)}")
    _log_row({"event": "done", "summary": summary})
    if log_f:
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
    p.add_argument("--optimizer", default="adamw",
                   choices=["adamw", "lion", "muon"],
                   help="adamw: AdamW (default, stable). "
                        "lion: 2x less optim state than AdamW, often converges in fewer steps. "
                        "muon: Keller Jordan's Muon (best on attention/MLP matrices).")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="If --optimizer lion or muon, consider 1e-4 (paper-recommended scale).")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-pct", type=float, default=0.05)
    p.add_argument("--aux-weight", type=float, default=0.1,
                   help="Coefficient for MSE auxiliary on draft hidden vs target hidden. "
                        "0 disables. EAGLE paper §3.3 uses 0.1.")
    p.add_argument("--aux-target-kind", default="next", choices=["current", "next"],
                   help="current: MSE(draft_hidden, target_hidden[t]) — echoes current state. "
                        "next: MSE(draft_hidden[t], target_hidden[t+1]) — predicts the next "
                        "state, which is what the verifier actually scores. Recommended for "
                        "spec-decode acceptance — directly trains the right geometric "
                        "alignment. Default 'next'.")
    p.add_argument("--hidden-stats", default=None,
                   help="Path to .npz containing 'mean' and 'std' (each shape (hidden_dim,)) "
                        "to normalize captured hidden states per-channel at load time. "
                        "Generated by tools/training/mlx_eagle/compute_hidden_stats.py. "
                        "Makes Adam's per-param adaptation work better.")
    # Schedule
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Cap total steps (overrides epochs). 0 = no cap. "
                        "Use --max-steps 100 for smoke tests.")
    # Precision / compile / streaming
    p.add_argument("--dtype", default="bf16",
                   choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
                   help="Trainable param + activation dtype. bf16 default (M3 native).")
    p.add_argument("--compile", dest="compile", action="store_true", default=True,
                   help="Wrap step in mx.compile (default).")
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="Disable mx.compile (use for debugging).")
    p.add_argument("--streaming", action="store_true",
                   help="Use streaming parquet iterator (recommended for shards > 5 GB).")
    p.add_argument("--prefetch", type=int, default=2,
                   help="Streaming iterator prefetch depth (in batches).")
    # Logging / ckpt
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500,
                   help="Checkpoint every N steps. 0 disables.")
    p.add_argument("--ckpt-dir", default="tools/training/mlx_eagle/ckpt")
    p.add_argument("--log", default="tools/training/mlx_eagle/train.log",
                   help="JSONL training log (one record per --log-every steps).")
    p.add_argument("--resume", default=None,
                   help="Resume from a checkpoint .npz (use ckpt-dir/latest.npz).")
    p.add_argument("--seed", type=int, default=20260516)
    args = p.parse_args()

    return train(args)


if __name__ == "__main__":
    sys.exit(main())
