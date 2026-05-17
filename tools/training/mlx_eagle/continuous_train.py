#!/usr/bin/env python3
"""
continuous_train.py — train EAGLE-3 head against the growing capture .bin.

CONTINUOUS-CHECKPOINT TRAINING: as the capture .bin accumulates more
samples (in background, on GPU), this script periodically re-trains the
head on the current dataset and saves a checkpoint per milestone. The
result is an acceptance-vs-data-size curve that:
  - Validates EAGLE-3 paper's scaling claims empirically on V2-Lite
  - Identifies the inflection point where additional data stops helping
  - Could reveal that 50K is enough (saves capture time) or 1M is needed
    (sets realistic expectations)

Runs CPU-only by default (--device cpu) so it doesn't compete with the
GPU capture. ~10-30x slower than GPU training, but the wall-clock that
matters is total elapsed (we want SOMETHING training while capture runs).

Milestone schedule (default): trains at each (5K, 10K, 25K, 50K, 100K,
200K, 350K, 500K) unique sample count. Each training run warm-starts
from the previous milestone's checkpoint (saves ~70% of training compute
per milestone). Configurable via --milestones.

Output layout:
  continuous_ckpt/
    at_005000/                — checkpoint at 5K-sample milestone
      step_000XXX.npz
      latest.npz
    at_010000/                — at 10K
    ...
    curve.jsonl               — append-only log: one row per milestone
    curve.csv                 — same data, CSV for plotting
    plot.png                  — auto-rendered plot (if matplotlib available)

Usage:
  PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
  $PY tools/training/mlx_eagle/continuous_train.py \\
    --shard training_data/c2_hidden/eagle3_v0/shard_000.bin \\
    --device cpu \\
    --milestones 5000 10000 25000 50000 100000 200000 350000 500000 \\
    --max-steps-per-milestone 500 \\
    --root-ckpt-dir training_data/c2_hidden/eagle3_v0/continuous_ckpt
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import struct
import subprocess
import sys
import time
from typing import Optional


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"


def count_unique_samples(shard_path: pathlib.Path, hidden_dim: int = 2048) -> int:
    seen: set = set()
    if not shard_path.exists():
        return 0
    hb_bytes = hidden_dim * 2
    with open(shard_path, "rb") as f:
        hdr = f.read(16)
        if hdr[:4] != b"DCAP":
            return 0
        while True:
            lb = f.read(2)
            if not lb:
                break
            (id_len,) = struct.unpack("<H", lb)
            sid = f.read(id_len).decode()
            f.seek(12 + hb_bytes, 1)
            seen.add(sid)
    return len(seen)


def convert_to_parquet(shard_bin: pathlib.Path, shard_parquet: pathlib.Path) -> bool:
    cmd = [
        PY, "tools/training/capture_hidden.py", "to-parquet",
        "--src", str(shard_bin), "--dst", str(shard_parquet),
        "--compression", "zstd",
    ]
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    return rc == 0


def run_training(
    parquet: pathlib.Path,
    ckpt_dir: pathlib.Path,
    log_path: pathlib.Path,
    args: argparse.Namespace,
    resume_from: Optional[pathlib.Path] = None,
) -> dict:
    """Returns the parsed final loss + wall time of this training run."""
    cmd = [
        PY, "tools/training/mlx_eagle/train.py",
        "--parquet", str(parquet),
        "--frozen", str(REPO_ROOT / "tools/training/mlx_eagle/v2lite_frozen.npz"),
        "--max-steps", str(args.max_steps_per_milestone),
        "--batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len),
        "--log-every", "10",
        "--save-every", str(max(args.max_steps_per_milestone // 5, 50)),
        "--ckpt-dir", str(ckpt_dir),
        "--log", str(log_path),
        "--dtype", args.dtype,
        "--optimizer", args.optimizer,
        "--aux-target-kind", args.aux_target_kind,
        "--device", args.device,
        # CRITICAL: --streaming so train.py uses StreamingParquetBatchIterator
        # (~100 MB peak) instead of in-memory ParquetBatchIterator which
        # loads ALL records (4+ GB for the 1.87M-record shard, OOM'd capture).
        "--streaming",
    ]
    if args.hidden_stats:
        cmd += ["--hidden-stats", str(args.hidden_stats)]
    if resume_from and resume_from.exists():
        cmd += ["--resume", str(resume_from)]
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0
    if rc != 0:
        return {"ok": False, "elapsed": elapsed, "rc": rc}
    # Parse final loss from JSONL log.
    final_loss = None
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if "loss" in row:
                        final_loss = row["loss"]
                    elif "summary" in row:
                        final_loss = row["summary"].get("final_loss", final_loss)
                except Exception:
                    continue
    return {"ok": True, "elapsed": elapsed, "final_loss": final_loss}


def run_eval(
    ckpt_path: pathlib.Path,
    heldout_shard: Optional[pathlib.Path],
    eval_out: pathlib.Path,
    args: argparse.Namespace,
) -> Optional[float]:
    """Returns accept_top1 % or None if held-out is unavailable."""
    if heldout_shard is None or not heldout_shard.exists():
        return None
    cmd = [
        PY, "tools/training/mlx_eagle/eval_acceptance.py", "eval",
        "--ckpt", str(ckpt_path),
        "--shard", str(heldout_shard),
        "--frozen", str(REPO_ROOT / "tools/training/mlx_eagle/v2lite_frozen.npz"),
        "--out", str(eval_out),
        "--batch-size", "128",
        "--max-records", "50000",
    ]
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    if rc != 0:
        return None
    try:
        d = json.load(open(eval_out))
        return d.get("accept_top1")
    except Exception:
        return None


def maybe_plot(curve_csv: pathlib.Path):
    """Render a simple acceptance-vs-data-size plot if matplotlib installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not curve_csv.exists():
        return
    rows = list(csv.DictReader(open(curve_csv)))
    if not rows:
        return
    xs = [int(r["n_samples"]) for r in rows]
    losses = [float(r["final_loss"]) if r.get("final_loss") else None for r in rows]
    accepts = [float(r["accept_top1"]) if r.get("accept_top1") else None for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Unique samples in training set")
    ax1.set_xscale("log")
    ax1.set_ylabel("Final training loss", color="tab:red")
    if any(l is not None for l in losses):
        ax1.plot(xs, losses, "o-", color="tab:red", label="train loss")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Held-out top-1 accept %", color="tab:blue")
    if any(a is not None for a in accepts):
        ax2.plot(xs, [a * 100 if a else None for a in accepts], "s-",
                 color="tab:blue", label="accept top-1")
        ax2.axhline(40, color="gray", linestyle=":", label="ship gate 40%")
        ax2.axhline(70, color="green", linestyle=":", label="paper-target 70%")
    fig.suptitle("EAGLE-3 head: acceptance vs data size (continuous training)")
    fig.tight_layout()
    out = curve_csv.with_suffix(".png")
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--shard", required=True,
                   help="DCAP .bin shard to poll for new records")
    p.add_argument("--root-ckpt-dir", required=True,
                   help="Directory where per-milestone checkpoints are saved")
    p.add_argument("--curve", default=None,
                   help="Path to curve.jsonl (default: <root-ckpt-dir>/curve.jsonl)")
    p.add_argument("--milestones", type=int, nargs="+",
                   default=[5000, 10000, 25000, 50000, 100000, 200000, 350000, 500000])
    p.add_argument("--max-steps-per-milestone", type=int, default=500,
                   help="How many training steps per milestone. Warm-start means later "
                        "milestones converge fast even at modest --max-steps.")
    p.add_argument("--poll-interval", type=int, default=300,
                   help="Seconds between checks of the shard's sample count")
    p.add_argument("--heldout-shard", default=None,
                   help="Optional held-out DCAP shard for per-milestone eval. "
                        "If absent, only training loss is logged (no accept rate).")
    p.add_argument("--hidden-stats", default=None)
    # train.py passthroughs
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--optimizer", default="lion", choices=["adamw", "lion", "muon"])
    p.add_argument("--aux-target-kind", default="next", choices=["next", "current"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=16)
    args = p.parse_args()

    shard_bin = pathlib.Path(args.shard)
    root_dir = pathlib.Path(args.root_ckpt_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    curve_jsonl = pathlib.Path(args.curve) if args.curve else root_dir / "curve.jsonl"
    curve_csv = curve_jsonl.with_suffix(".csv")
    heldout = pathlib.Path(args.heldout_shard) if args.heldout_shard else None
    parquet_cache = root_dir / "shard_at_milestone.parquet"

    # Load milestones already done from curve.jsonl (resume support).
    done_milestones: set = set()
    if curve_jsonl.exists():
        with open(curve_jsonl) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done_milestones.add(int(row["n_samples"]))
                except Exception:
                    continue
        print(f"[cont] {len(done_milestones)} milestone(s) already done: {sorted(done_milestones)}",
              file=sys.stderr)

    print(f"[cont] watching {shard_bin}", file=sys.stderr)
    print(f"[cont] milestones: {sorted(args.milestones)}", file=sys.stderr)
    print(f"[cont] device={args.device} optimizer={args.optimizer} dtype={args.dtype}", file=sys.stderr)
    print(f"[cont] root_ckpt_dir={root_dir}", file=sys.stderr)
    print(f"[cont] poll interval: {args.poll_interval}s", file=sys.stderr)

    while True:
        current = count_unique_samples(shard_bin)
        # Find the highest milestone we've crossed but not yet trained.
        eligible = [m for m in args.milestones if current >= m and m not in done_milestones]
        if not eligible:
            # Nothing to do — wait. If max milestone is done, exit.
            if all(m in done_milestones for m in args.milestones):
                print(f"[cont] all milestones done; exiting", file=sys.stderr)
                break
            print(f"[cont] sleeping; current={current} next_milestone="
                  f"{min(m for m in args.milestones if m not in done_milestones)}",
                  file=sys.stderr)
            time.sleep(args.poll_interval)
            continue
        # Train the smallest unfilled milestone we've crossed.
        milestone = min(eligible)
        print(f"\n[cont] MILESTONE {milestone:,} CROSSED (current={current:,})", file=sys.stderr)

        # Convert .bin to parquet at the current state.
        print(f"[cont] converting shard to parquet…", file=sys.stderr)
        if not convert_to_parquet(shard_bin, parquet_cache):
            print(f"[cont] parquet conversion failed; will retry next cycle", file=sys.stderr)
            time.sleep(args.poll_interval)
            continue

        # Warm-start from previous milestone if exists.
        prev_milestones = sorted([m for m in done_milestones if m < milestone], reverse=True)
        resume_from = None
        for prev in prev_milestones:
            cand = root_dir / f"at_{prev:06d}" / "latest.npz"
            if cand.exists():
                resume_from = cand
                break

        ckpt_dir = root_dir / f"at_{milestone:06d}"
        log_path = root_dir / f"at_{milestone:06d}.train.log"
        t0 = time.time()
        train_result = run_training(parquet_cache, ckpt_dir, log_path, args,
                                    resume_from=resume_from)
        train_wall = time.time() - t0

        if not train_result.get("ok"):
            print(f"[cont] training FAILED at milestone {milestone}: {train_result}",
                  file=sys.stderr)
            time.sleep(args.poll_interval)
            continue

        # Eval if held-out present.
        eval_out = root_dir / f"at_{milestone:06d}.eval.json"
        accept_top1 = run_eval(ckpt_dir / "latest.npz", heldout, eval_out, args)

        # Append to curve.
        row = {
            "n_samples": milestone,
            "actual_samples_in_bin": current,
            "ckpt": str(ckpt_dir / "latest.npz"),
            "warm_start_from": str(resume_from) if resume_from else None,
            "final_loss": train_result.get("final_loss"),
            "accept_top1": accept_top1,
            "train_wall_s": train_wall,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "max_steps": args.max_steps_per_milestone,
            "device": args.device,
            "optimizer": args.optimizer,
        }
        with open(curve_jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        # Rewrite CSV from JSONL (easier than incremental append).
        with open(curve_jsonl) as f, open(curve_csv, "w", newline="") as g:
            rows = [json.loads(line) for line in f if line.strip()]
            if rows:
                w = csv.DictWriter(g, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
        print(f"[cont] MILESTONE {milestone:,} DONE: loss={train_result.get('final_loss')} "
              f"accept_top1={accept_top1} wall={train_wall:.0f}s", file=sys.stderr)
        # Re-render the plot.
        maybe_plot(curve_csv)
        # Emit a notification line (caller can grep for this).
        print(f"CONTINUOUS_MILESTONE_DONE m={milestone} loss={train_result.get('final_loss')} "
              f"accept={accept_top1}")

        done_milestones.add(milestone)

    return 0


if __name__ == "__main__":
    sys.exit(main())
